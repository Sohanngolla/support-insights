"""Prompt construction for the text-to-SQL and answer-phrasing LLM calls.

The schema description is built from app.ingest's EXPECTED_COLUMNS and ENUMS
rather than duplicated as a hand-written string. Two sources of truth for the
same schema is exactly the kind of drift that quietly breaks a demo three
days after it worked -- avoiding it costs nothing here, so it's not on the
"prototype shortcuts" list.
"""

from __future__ import annotations

from app.config import settings
from app.ingest import ENUMS, EXPECTED_COLUMNS

# Sentinel the model outputs instead of SQL when a question isn't about this
# dataset at all. Checked in app.query before the guard even sees the output
# -- there's no SQL to validate for "who is the president of India".
OUT_OF_SCOPE = "OUT_OF_SCOPE"

_COLUMN_TYPES = {
    "ticket_id": "TEXT (primary key)",
    "created_at": "TEXT, format 'YYYY-MM-DD HH:MM:SS'",
    "category": f"TEXT, one of {sorted(ENUMS['category'])}",
    "priority": f"TEXT, one of {sorted(ENUMS['priority'])}",
    "status": f"TEXT, one of {sorted(ENUMS['status'])}",
    "response_time_hrs": "REAL, hours until first response (nullable)",
    "resolution_time_hrs": "REAL, hours until resolution (nullable, NULL if unresolved)",
    "agent_id": "TEXT, e.g. 'AGT-01'",
    "customer_rating": "INTEGER 1-5 (nullable, only set once resolved)",
    "issue_summary": "TEXT, free-text one-line summary",
}


def schema_description() -> str:
    lines = [f"Table: {settings.table_name}", "Columns:"]
    for col in EXPECTED_COLUMNS:
        lines.append(f"  - {col}: {_COLUMN_TYPES[col]}")
    return "\n".join(lines)


FEW_SHOT = [
    (
        "How many tickets are currently open?",
        "SELECT COUNT(*) AS open_tickets FROM {table} WHERE status = 'Open'",
    ),
    (
        "What's the average resolution time by priority?",
        "SELECT priority, ROUND(AVG(resolution_time_hrs), 2) AS avg_hrs "
        "FROM {table} WHERE resolution_time_hrs IS NOT NULL "
        "GROUP BY priority ORDER BY avg_hrs DESC",
    ),
    (
        "Which agent has the most escalated tickets?",
        "SELECT agent_id, COUNT(*) AS n FROM {table} WHERE status = 'Escalated' "
        "GROUP BY agent_id ORDER BY n DESC LIMIT 1",
    ),
    (
        "Show me the 5 lowest-rated resolved tickets",
        "SELECT ticket_id, customer_rating, issue_summary FROM {table} "
        "WHERE status = 'Resolved' AND customer_rating IS NOT NULL "
        "ORDER BY customer_rating ASC LIMIT 5",
    ),
    (
        "What percentage of tickets are Critical priority?",
        "SELECT ROUND(100.0 * SUM(CASE WHEN priority = 'Critical' THEN 1 ELSE 0 END) "
        "/ COUNT(*), 1) AS pct_critical FROM {table}",
    ),
    (
        "Which agents have a worse-than-average resolution time?",
        "SELECT agent_id, ROUND(AVG(resolution_time_hrs), 2) AS avg_hrs FROM {table} "
        "WHERE resolution_time_hrs IS NOT NULL GROUP BY agent_id "
        "HAVING avg_hrs > (SELECT AVG(resolution_time_hrs) FROM {table} "
        "WHERE resolution_time_hrs IS NOT NULL) ORDER BY avg_hrs DESC",
    ),
    (
        "How many billing issues are there?",
        # 'billing issues' maps to the category column, not a text search on
        # issue_summary -- prefer the structured column when the question's
        # noun matches an enum value, since it's exact rather than fuzzy.
        "SELECT COUNT(*) AS n FROM {table} WHERE category = 'Billing'",
    ),
    (
        "How does resolution time compare across categories?",
        "SELECT category, COUNT(*) AS n, ROUND(AVG(resolution_time_hrs), 2) AS avg_hrs, "
        "ROUND(MIN(resolution_time_hrs), 2) AS min_hrs, ROUND(MAX(resolution_time_hrs), 2) AS max_hrs "
        "FROM {table} WHERE resolution_time_hrs IS NOT NULL GROUP BY category",
    ),
]


def sql_system_prompt() -> str:
    table = settings.table_name
    examples = "\n\n".join(
        f"Q: {q}\nSQL: {sql.format(table=table)}" for q, sql in FEW_SHOT
    )
    return f"""You translate a support-ticket question into a single SQLite SELECT statement.

{schema_description()}

This dataset is historical -- it does not extend to today's real-world date.
Do NOT use 'now', CURRENT_DATE, CURRENT_TIMESTAMP, or date('now', ...) for
relative time windows (e.g. "last 30 days", "this week"). Instead compute
relative windows using (SELECT MAX(created_at) FROM {table}) as the
reference point, e.g.:
  WHERE created_at >= (SELECT datetime(MAX(created_at), '-30 days') FROM {table})

Rules:
- Output ONLY the SQL. No markdown fences, no explanation, no "SQL:" prefix.
- Exactly one statement. SELECT or WITH...SELECT only -- never INSERT, UPDATE,
  DELETE, DROP, ALTER, PRAGMA, or any statement that isn't a read.
- Only reference the {table} table. There is no other table.
- Prefer aggregates (COUNT, AVG, GROUP BY) over dumping raw rows when the
  question asks "how many" / "what's the average" / "which X has the most".
- If the question is ambiguous about which rows count (e.g. "slow tickets"),
  make a reasonable choice and let the row data speak for itself.
- If the question asks multiple distinct things at once, you may answer all
  of them in one query (e.g. with UNION ALL and a label column) if that's
  natural, or just answer the primary one -- the answer-phrasing step will
  organize the response either way.

If the question is NOT about this support-ticket dataset -- general
knowledge, world facts, math, other companies, anything that isn't
answerable from the {table} table -- do not write SQL at all. Output
exactly the single token: {OUT_OF_SCOPE}

Examples:
{examples}

Q: Who is the president of India?
SQL: {OUT_OF_SCOPE}

Now produce SQL for the user's question. Output only the SQL, or the
out-of-scope token."""


def sql_repair_prompt(original_sql: str, error: str, question: str) -> str:
    return f"""Your previous answer for the question below was rejected.

Question: {question}
Your SQL: {original_sql}
Rejection reason: {error}

Return corrected SQL only. No markdown fences, no explanation."""


def answer_system_prompt() -> str:
    return """You answer a support-ticket question using ONLY the query result data given
to you. Do not invent numbers that aren't in the data.

Formatting:
- Default to 1-3 plain sentences, leading with the number or fact that
  answers the question.
- If the question actually contains multiple distinct sub-questions, answer
  each one as its own short line starting with "- " (a markdown bullet),
  not blended into one paragraph. Keep each bullet to one sentence.
- Use **bold** only around the single most important number or name per
  sentence/bullet -- not whole phrases, not every number.
- Never produce a markdown table (the data is already shown as a real
  table below your answer) and never use headings.
- If the result set is empty, say plainly that no matching tickets were
  found rather than guessing why."""


def answer_user_prompt(question: str, sql: str, rows: list[dict]) -> str:
    # Cap what we show the model -- it doesn't need 200 rows to phrase a
    # sentence, and it keeps the second call cheap and fast.
    preview = rows[:20]
    truncated_note = f"\n(... {len(rows) - 20} more rows omitted)" if len(rows) > 20 else ""
    return f"""Question: {question}

SQL used: {sql}

Result ({len(rows)} row(s)):
{preview}{truncated_note}

Answer the question in 1-3 sentences using this data."""
