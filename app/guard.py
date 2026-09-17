"""Validation of LLM-generated SQL before it reaches the database.

This is defence layer one of two (layer two is the read-only connection plus
authorizer in app.db).

A note on the approach: a full SQL parser (sqlglot) would be the 10/10 answer
here. This is the 3/10 answer -- a normaliser plus token checks -- chosen
because it adds no dependency, is small enough to read in one sitting, and
does not have to be airtight on its own, because it is not the only defence.
The one thing it does carefully is strip comments and string literals before
inspecting tokens, since that is where naive keyword matching actually breaks:
a ticket summary containing the word "delete" must not trip the filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.ingest import ENUMS

# Statements that must never reach the database, matched as whole tokens.
FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "grant", "revoke", "attach", "detach", "pragma", "vacuum",
    "reindex", "analyze", "commit", "rollback", "begin", "savepoint", "load_extension",
}

ALLOWED_LEADING = {"select", "with"}

# Only the exposed table may be referenced.
ALLOWED_TABLES = {settings.table_name.lower()}

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SINGLE_QUOTED = re.compile(r"'(?:[^']|'')*'")
_DOUBLE_QUOTED = re.compile(r'"(?:[^"]|"")*"')
_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z_0-9]*)", re.IGNORECASE)
_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)

# Column-value grounding: the safety checks above prove the SQL can't harm
# the database, but say nothing about whether it will silently return the
# wrong (usually empty) answer -- e.g. WHERE status = 'Pending', which is
# valid SQL and executes fine, but 'Pending' isn't one of our three actual
# status values. This is a heuristic regex check, not a real SQL parser
# (same "3/10, not 10/10" tradeoff as the rest of this file), so it only
# catches the common `col = 'x'` / `col IN ('x', 'y')` shapes -- but those
# are exactly the shapes the LLM actually produces for enum filters.
_ENUM_CLAUSE = {
    col: re.compile(rf"\b{col}\b\s*(?:=|IN)\s*(\([^)]*\)|'[^']*')", re.IGNORECASE)
    for col in ENUMS
}
_QUOTED_LITERAL = re.compile(r"'([^']*)'")


class SQLRejected(ValueError):
    """Raised when generated SQL fails validation. The message is fed back to
    the model on retry, so it must read as an instruction, not a stack trace."""


@dataclass
class ValidatedSQL:
    sql: str            # what will actually run (may have LIMIT appended)
    original: str       # what the model produced, for display
    limit_added: bool


def clean(raw: str) -> str:
    """Strip the wrapping an LLM adds despite being told not to."""
    text = raw.strip()
    text = _FENCE.sub("", text).strip()
    # Some models prefix a label even under instruction.
    text = re.sub(r"^(sql|query)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def _strip_literals(sql: str) -> str:
    """Remove comments and quoted strings so token checks see only structure."""
    out = _BLOCK_COMMENT.sub(" ", sql)
    out = _LINE_COMMENT.sub(" ", out)
    out = _SINGLE_QUOTED.sub("''", out)
    out = _DOUBLE_QUOTED.sub('""', out)
    return out


def _enum_violations(sql: str) -> list[tuple[str, str]]:
    """Return (column, bad_value) pairs for enum-column literals that don't
    match any real value in that column, e.g. ("status", "Pending")."""
    violations = []
    for col, clause_re in _ENUM_CLAUSE.items():
        for clause_match in clause_re.finditer(sql):
            for literal in _QUOTED_LITERAL.findall(clause_match.group(1)):
                if literal not in ENUMS[col]:
                    violations.append((col, literal))
    return violations


def validate(raw: str, max_rows: int | None = None) -> ValidatedSQL:
    max_rows = max_rows or settings.max_rows
    sql = clean(raw)

    if not sql:
        raise SQLRejected("No SQL was produced. Return a single SELECT statement.")

    skeleton = _strip_literals(sql)

    # One statement only. After stripping literals, any ';' means a second one.
    if ";" in skeleton:
        raise SQLRejected(
            "Multiple statements detected. Return exactly one SELECT statement."
        )

    tokens = [t.lower() for t in _TOKEN.findall(skeleton)]
    if not tokens:
        raise SQLRejected("No SQL keywords found. Return a single SELECT statement.")

    if tokens[0] not in ALLOWED_LEADING:
        raise SQLRejected(
            f"Query starts with '{tokens[0].upper()}'. "
            "Only SELECT (or WITH ... SELECT) is permitted."
        )

    hit = FORBIDDEN.intersection(tokens)
    if hit:
        raise SQLRejected(
            f"Disallowed keyword(s): {', '.join(sorted(k.upper() for k in hit))}. "
            "This is a read-only system; use SELECT only."
        )

    referenced = {m.lower() for m in _FROM_JOIN.findall(skeleton)}
    # CTE names are legitimate targets of FROM; allow anything defined by WITH.
    cte_names = {
        m.lower()
        for m in re.findall(
            r"(?:with|,)\s+([A-Za-z_][A-Za-z_0-9]*)\s+as\s*\(", skeleton, re.IGNORECASE
        )
    }
    unknown = referenced - ALLOWED_TABLES - cte_names
    if unknown:
        raise SQLRejected(
            f"Unknown table(s): {', '.join(sorted(unknown))}. "
            f"The only available table is {settings.table_name}."
        )

    violations = _enum_violations(sql)
    if violations:
        lines = [
            f"'{val}' is not a valid value for column '{col}'. "
            f"Valid values are: {', '.join(sorted(ENUMS[col]))}."
            for col, val in violations
        ]
        raise SQLRejected(" ".join(lines))

    final, limit_added = sql, False
    if not _HAS_LIMIT.search(skeleton):
        final = f"{sql}\nLIMIT {max_rows}"
        limit_added = True

    return ValidatedSQL(sql=final, original=sql, limit_added=limit_added)
