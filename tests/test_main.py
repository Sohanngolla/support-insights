"""API tests, run against the real SQLite data but with the LLM layer
mocked out -- these test the HTTP wiring and error-contract shaping, not
Groq/Gemini themselves (that's tests/test_llm.py's job).

Requires: pip install fastapi httpx  (httpx is FastAPI's TestClient dependency)

Run:
    pytest -q
    python tests/test_main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.llm import AllProvidersFailedError  # noqa: E402
from app.query import QueryError, QueryResult  # noqa: E402


def _client() -> TestClient:
    from app.main import app

    return TestClient(app)


def _fake_result(**overrides) -> QueryResult:
    base = dict(
        question="How many tickets are open?",
        sql="SELECT COUNT(*) AS n FROM support_tickets WHERE status = 'Open'",
        original_sql="SELECT COUNT(*) AS n FROM support_tickets WHERE status = 'Open'",
        data=[{"n": 99}],
        answer="99 tickets are currently open.",
        sql_provider="groq",
        answer_provider="groq",
        repaired=False,
        warnings=[],
    )
    base.update(overrides)
    return QueryResult(**base)


# --- /query ------------------------------------------------------------

def test_query_happy_path():
    with mock.patch("app.main.answer_question", return_value=_fake_result()):
        resp = _client().post("/query", json={"question": "How many tickets are open?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "99 tickets are currently open."
    assert body["row_count"] == 1
    assert body["sql_provider"] == "groq"
    assert "latency_ms" in body


def test_query_rejects_empty_question():
    resp = _client().post("/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length violation


def test_query_llm_unavailable_maps_to_502():
    with mock.patch(
        "app.main.answer_question",
        side_effect=AllProvidersFailedError("no providers configured"),
    ):
        resp = _client().post("/query", json={"question": "anything"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error_type"] == "llm_unavailable"


def test_query_sql_rejected_maps_to_422():
    with mock.patch(
        "app.main.answer_question",
        side_effect=QueryError("Generated SQL rejected twice. First: ... After repair: ..."),
    ):
        resp = _client().post("/query", json={"question": "anything"})
    assert resp.status_code == 422
    assert resp.json()["error_type"] == "sql_rejected"


def test_query_execution_failure_maps_to_500():
    with mock.patch(
        "app.main.answer_question",
        side_effect=QueryError("Query passed validation but failed to execute: no such column"),
    ):
        resp = _client().post("/query", json={"question": "anything"})
    assert resp.status_code == 500
    assert resp.json()["error_type"] == "sql_execution_failed"


def test_query_empty_result_set_is_not_an_error():
    with mock.patch(
        "app.main.answer_question",
        return_value=_fake_result(data=[], answer="No tickets matched that filter."),
    ):
        resp = _client().post("/query", json={"question": "tickets from the year 1990"})
    assert resp.status_code == 200
    assert resp.json()["row_count"] == 0


# --- UI -----------------------------------------------------------------

def test_root_serves_ui_html():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Support Insights" in resp.text


# --- /anomalies (against the real DB and real anomaly engine) --------------

def test_anomalies_default_returns_all_five_rule_types():
    resp = _client().get("/anomalies")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["counts_by_rule"]) == {
        "stale_high_priority",
        "resolution_time_outlier",
        "response_time_outlier",
        "poor_rating_resolved",
        "agent_performance_outlier",
    }
    assert body["total"] == sum(body["counts_by_rule"].values())


def test_anomalies_type_filter():
    resp = _client().get("/anomalies", params={"type": "poor_rating_resolved"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(f["anomaly_type"] for f in body["findings"]) <= {"poor_rating_resolved"}


def test_anomalies_severity_filter():
    resp = _client().get("/anomalies", params={"severity": "high"})
    assert resp.status_code == 200
    assert all(f["severity"] == "high" for f in resp.json()["findings"])


def test_anomalies_bad_severity_is_400():
    resp = _client().get("/anomalies", params={"severity": "catastrophic"})
    assert resp.status_code == 400


def test_anomalies_bad_type_is_400():
    resp = _client().get("/anomalies", params={"type": "not_a_real_rule"})
    assert resp.status_code == 400


# --- /health and /stats (against the real DB) -------------------------------

def test_health_reports_db_and_providers():
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_reachable"] is True
    assert isinstance(body["total_tickets"], int)
    assert isinstance(body["configured_providers"], list)


def test_stats_matches_db_dataset_stats():
    from app.config import settings
    from app.db import scalar

    resp = _client().get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    # Compare against the live table count rather than a hard-coded number,
    # so this test still passes once the real CSV replaces the sample data.
    assert body["total_tickets"] == scalar(f"SELECT COUNT(*) FROM {settings.table_name}")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
