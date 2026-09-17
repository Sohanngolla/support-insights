"""Guard tests.

Runnable two ways:
    pytest -q
    python tests/test_guard.py       (no pytest needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guard import SQLRejected, clean, validate  # noqa: E402

TABLE = "support_tickets"


def _rejects(sql: str) -> bool:
    try:
        validate(sql)
    except SQLRejected:
        return True
    return False


# --- things that must be allowed -------------------------------------------

def test_plain_select_passes():
    out = validate(f"SELECT COUNT(*) FROM {TABLE} WHERE status = 'Open'")
    assert out.sql.startswith("SELECT")
    assert out.limit_added is True


def test_cte_passes():
    sql = (
        f"WITH resolved AS (SELECT * FROM {TABLE} WHERE status = 'Resolved') "
        "SELECT agent_id, COUNT(*) FROM resolved GROUP BY agent_id"
    )
    assert validate(sql).sql.startswith("WITH")


def test_existing_limit_is_respected():
    out = validate(f"SELECT * FROM {TABLE} LIMIT 5")
    assert out.limit_added is False
    assert out.sql.count("LIMIT") == 1


def test_literal_containing_keyword_is_not_rejected():
    # The word 'delete' inside a string must not trip the filter.
    sql = f"SELECT * FROM {TABLE} WHERE issue_summary LIKE '%delete my account%'"
    assert validate(sql).sql.startswith("SELECT")


def test_comment_is_stripped_not_executed():
    sql = f"SELECT COUNT(*) FROM {TABLE} -- drop table everything\n"
    assert validate(sql).sql.startswith("SELECT")


def test_markdown_fence_is_removed():
    assert clean("```sql\nSELECT 1\n```") == "SELECT 1"


def test_trailing_semicolon_is_tolerated():
    assert validate(f"SELECT 1 FROM {TABLE};").sql.startswith("SELECT")


# --- things that must be rejected ------------------------------------------

def test_rejects_drop():
    assert _rejects(f"DROP TABLE {TABLE}")


def test_rejects_delete():
    assert _rejects(f"DELETE FROM {TABLE}")


def test_rejects_update():
    assert _rejects(f"UPDATE {TABLE} SET status = 'Resolved'")


def test_rejects_stacked_statement():
    assert _rejects(f"SELECT 1 FROM {TABLE}; DROP TABLE {TABLE}")


def test_rejects_insert_hidden_after_select():
    assert _rejects(f"SELECT * FROM {TABLE} UNION SELECT 1; INSERT INTO x VALUES (1)")


def test_rejects_attach():
    assert _rejects(f"SELECT * FROM {TABLE} WHERE 1=1 ATTACH DATABASE 'x' AS y")


def test_rejects_pragma():
    assert _rejects("PRAGMA table_info(support_tickets)")


def test_rejects_other_table():
    assert _rejects("SELECT * FROM sqlite_master")


def test_rejects_empty():
    assert _rejects("   ")


def test_rejects_prose():
    assert _rejects("I cannot answer that question.")


# --- enum/column grounding ---------------------------------------------

def test_rejects_invalid_status_value():
    ok = False
    try:
        validate(f"SELECT * FROM {TABLE} WHERE status = 'Pending'")
    except SQLRejected as exc:
        ok = "Pending" in str(exc) and "status" in str(exc)
    assert ok


def test_rejects_invalid_value_inside_in_clause():
    ok = False
    try:
        validate(f"SELECT * FROM {TABLE} WHERE status IN ('Open', 'Pending')")
    except SQLRejected as exc:
        ok = "Pending" in str(exc)
    assert ok


def test_accepts_real_enum_values():
    out = validate(f"SELECT * FROM {TABLE} WHERE status IN ('Open', 'Escalated')")
    assert out.sql.startswith("SELECT")


def test_rejects_invalid_priority_and_category_together():
    ok = False
    try:
        validate(
            f"SELECT * FROM {TABLE} WHERE priority = 'Urgent' "
            "AND category = 'Refunds'"
        )
    except SQLRejected as exc:
        ok = "Urgent" in str(exc) and "Refunds" in str(exc)
    assert ok


def test_enum_check_does_not_false_positive_on_unrelated_literal():
    # A literal that happens to be a real column word elsewhere in the
    # dataset (e.g. a status-like word inside issue_summary text) must not
    # trip this check -- it only looks at literals immediately following
    # status/category/priority in a WHERE-style comparison.
    out = validate(
        f"SELECT * FROM {TABLE} WHERE issue_summary LIKE '%pending refund%'"
    )
    assert out.sql.startswith("SELECT")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError:
            failed += 1
            print(f"  FAIL  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
