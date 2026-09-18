"""End-to-end tests for app.query, run against the real ingested database
with a scripted fake provider standing in for the LLM. This proves the
pipeline's plumbing -- guard integration, repair retry, execution, answer
call -- independent of whether Groq/Gemini are reachable.

Requires data/support.db to exist: run `python3 -m app.ingest` first.

Runnable two ways:
    pytest -q
    python tests/test_query.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.query import QueryError, answer_question  # noqa: E402


class _ScriptedProvider:
    """Returns each entry in `responses` in order, one per call."""

    def __init__(self, name: str, responses: list[str]):
        self.name = name
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        if self.calls >= len(self._responses):
            raise RuntimeError(f"{self.name}: ran out of scripted responses")
        text = self._responses[self.calls]
        self.calls += 1
        return text


def test_happy_path_count_query():
    sql = f"SELECT COUNT(*) AS n FROM {settings.table_name} WHERE status = 'Open'"
    provider = _ScriptedProvider("fake", [sql, "There are 99 open tickets."])
    result = answer_question("How many tickets are open?", providers=[provider])

    assert result.repaired is False
    assert result.data == [{"n": 111}]
    assert "99" in result.answer
    assert result.sql_provider == "fake"
    assert result.answer_provider == "fake"


def test_bad_sql_triggers_one_repair_then_succeeds():
    bad_sql = f"DELETE FROM {settings.table_name}"
    good_sql = f"SELECT COUNT(*) AS n FROM {settings.table_name}"
    provider = _ScriptedProvider("fake", [bad_sql, good_sql, "There are 500 tickets total."])
    result = answer_question("How many tickets are there?", providers=[provider])

    assert result.repaired is True
    assert result.data == [{"n": 500}]
    assert provider.calls == 3  # bad SQL, repaired SQL, answer phrasing


def test_bad_sql_twice_raises_query_error():
    bad_sql = f"DELETE FROM {settings.table_name}"
    provider = _ScriptedProvider("fake", [bad_sql, bad_sql])
    try:
        answer_question("How many tickets are there?", providers=[provider])
        assert False, "expected QueryError"
    except QueryError as exc:
        assert "rejected twice" in str(exc)


def test_markdown_fenced_sql_is_cleaned_before_validation():
    fenced = f"```sql\nSELECT COUNT(*) AS n FROM {settings.table_name}\n```"
    provider = _ScriptedProvider("fake", [fenced, "500 total."])
    result = answer_question("How many tickets total?", providers=[provider])
    assert result.repaired is False
    assert result.data == [{"n": 500}]


def test_empty_result_set_is_passed_through_not_hidden():
    # A syntactically and semantically valid query (real enum value, real
    # column) that just happens to match nothing in this dataset -- distinct
    # from an invalid enum value, which the guard's grounding check now
    # rejects before execution (see test_guard.py).
    sql = f"SELECT * FROM {settings.table_name} WHERE status = 'Open' AND created_at < '1999-01-01'"
    provider = _ScriptedProvider("fake", [sql, "No matching tickets were found."])
    result = answer_question("Show tickets from before 1999", providers=[provider])
    assert result.data == []
    assert "no matching" in result.answer.lower()


def test_out_of_scope_question_skips_sql_and_second_call():
    provider = _ScriptedProvider("fake", ["OUT_OF_SCOPE"])
    result = answer_question("Who is the president of India?", providers=[provider])

    assert result.out_of_scope is True
    assert result.sql == ""
    assert result.data == []
    assert "support-ticket" in result.answer
    assert provider.calls == 1  # no second (answer-phrasing) call was made


def test_out_of_scope_detected_after_repair_attempt():
    bad_sql = f"DELETE FROM {settings.table_name}"
    provider = _ScriptedProvider("fake", [bad_sql, "OUT_OF_SCOPE"])
    result = answer_question("Delete everything, or tell me a joke", providers=[provider])

    assert result.out_of_scope is True
    assert result.repaired is False  # out-of-scope isn't a "repair", it's a decline


def test_invalid_enum_value_triggers_repair_like_any_other_rejection():
    bad_sql = f"SELECT * FROM {settings.table_name} WHERE status = 'Pending'"
    good_sql = f"SELECT * FROM {settings.table_name} WHERE status = 'Open'"
    provider = _ScriptedProvider("fake", [bad_sql, good_sql, "Some open tickets."])
    result = answer_question("Show pending tickets", providers=[provider])
    assert result.repaired is True
    assert "Open" in result.sql


TESTS = [
    test_happy_path_count_query,
    test_bad_sql_triggers_one_repair_then_succeeds,
    test_bad_sql_twice_raises_query_error,
    test_markdown_fenced_sql_is_cleaned_before_validation,
    test_empty_result_set_is_passed_through_not_hidden,
    test_out_of_scope_question_skips_sql_and_second_call,
    test_out_of_scope_detected_after_repair_attempt,
    test_invalid_enum_value_triggers_repair_like_any_other_rejection,
]


if __name__ == "__main__":
    if not settings.db_path.exists():
        print(f"ERROR: {settings.db_path} not found. Run: python3 -m app.ingest")
        sys.exit(1)

    passed = 0
    for t in TESTS:
        t()
        print(f"  pass  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(TESTS)} passed")
