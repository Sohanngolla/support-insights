"""End-to-end NL question -> answer pipeline.

Two LLM calls per question:
  1. question -> SQL (validated by app.guard before it ever touches the db)
  2. question + result rows -> a phrased natural-language answer

If the generated SQL is rejected by the guard, one repair attempt is made
(configurable via settings.sql_retry_attempts) by feeding the rejection
reason back to the model -- this is the "real prompt/output engineering"
piece, not just a bare API call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app import db, prompts
from app.config import settings
from app.guard import SQLRejected, ValidatedSQL, validate
from app.llm import Provider, build_providers, complete_with_fallback

log = logging.getLogger(__name__)

_DECLINE_MESSAGE = (
    "I can only answer questions about this support-ticket dataset -- "
    "things like ticket counts, categories, priorities, agents, ratings, "
    "and response/resolution times. Try asking something like \"how many "
    "tickets are open\" or \"average resolution time by priority\"."
)


class QueryError(RuntimeError):
    """Raised when no valid, executable SQL could be produced."""


@dataclass
class QueryResult:
    question: str
    sql: str                 # the SQL that actually ran (may have LIMIT appended)
    original_sql: str        # what the model produced, before any LIMIT append
    data: list[dict]
    answer: str
    sql_provider: str
    answer_provider: str
    repaired: bool = False    # True if the first SQL attempt was rejected
    warnings: list[str] = field(default_factory=list)
    out_of_scope: bool = False


def _is_out_of_scope(raw: str) -> bool:
    return raw.strip().strip("`").strip().upper().startswith(prompts.OUT_OF_SCOPE)


def _generate_sql(
    question: str, providers: list[Provider]
) -> tuple[ValidatedSQL | None, str, bool]:
    """Returns (validated_sql_or_None, provider_used, was_repaired).

    A None first element means the model declared the question out of
    scope -- there is no SQL to run, validated or otherwise.
    """
    system = prompts.sql_system_prompt()
    raw, provider_used = complete_with_fallback(system, question, providers)

    if _is_out_of_scope(raw):
        return None, provider_used, False

    try:
        return validate(raw), provider_used, False
    except SQLRejected as exc:
        if settings.sql_retry_attempts < 1:
            raise QueryError(f"Generated SQL rejected: {exc}") from exc

        log.info("SQL rejected, attempting repair: %s", exc)
        repair_prompt = prompts.sql_repair_prompt(raw, str(exc), question)
        raw2, provider_used2 = complete_with_fallback(system, repair_prompt, providers)

        if _is_out_of_scope(raw2):
            return None, provider_used2, False

        try:
            return validate(raw2), provider_used2, True
        except SQLRejected as exc2:
            raise QueryError(
                f"Generated SQL rejected twice. First: {exc}. After repair: {exc2}"
            ) from exc2


def answer_question(question: str, providers: list[Provider] | None = None) -> QueryResult:
    """Run the full pipeline for one question. Raises QueryError or LLMError
    if either LLM call fails outright (see app.llm.AllProvidersFailedError)."""
    providers = providers if providers is not None else build_providers()

    validated, sql_provider, repaired = _generate_sql(question, providers)

    if validated is None:
        # Out of scope -- no SQL to run, no second call needed. The decline
        # message is fixed, not LLM-phrased: deterministic, instant, and
        # can't itself be steered off-topic by a crafted question.
        return QueryResult(
            question=question,
            sql="",
            original_sql="",
            data=[],
            answer=_DECLINE_MESSAGE,
            sql_provider=sql_provider,
            answer_provider=sql_provider,
            repaired=False,
            warnings=[],
            out_of_scope=True,
        )

    try:
        rows = db.run_select(validated.sql)
    except db.QueryExecutionError as exc:
        # Valid per our guard, but SQLite itself rejected it (e.g. a column
        # typo the guard has no way to catch, since it doesn't know columns).
        raise QueryError(f"Query passed validation but failed to execute: {exc}") from exc

    answer_system = prompts.answer_system_prompt()
    answer_user = prompts.answer_user_prompt(question, validated.sql, rows)
    answer_text, answer_provider = complete_with_fallback(answer_system, answer_user, providers)

    warnings = []
    if validated.limit_added:
        warnings.append(f"LIMIT {settings.max_rows} was appended automatically")

    return QueryResult(
        question=question,
        sql=validated.sql,
        original_sql=validated.original,
        data=rows,
        answer=answer_text.strip(),
        sql_provider=sql_provider,
        answer_provider=answer_provider,
        repaired=repaired,
        warnings=warnings,
    )
