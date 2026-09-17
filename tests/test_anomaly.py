"""Anomaly engine tests, against a small in-memory DB with known fixtures.

Every expected value here is computed by hand in the comments, not asserted
against "whatever the code currently returns" -- that's the difference
between a test and a tautology.

Run:
    pytest -q
    python tests/test_anomaly.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import anomaly  # noqa: E402
from app.ingest import DDL  # noqa: E402

TABLE = "support_tickets"


def _make_conn(rows: list[dict]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(DDL.format(table=TABLE))
    cols = [
        "ticket_id", "created_at", "category", "priority", "status",
        "response_time_hrs", "resolution_time_hrs", "agent_id",
        "customer_rating", "issue_summary",
    ]
    placeholders = ", ".join("?" for _ in cols)
    for row in rows:
        conn.execute(
            f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ({placeholders})",
            [row.get(c) for c in cols],
        )
    conn.commit()
    return conn


def _row(**kwargs) -> dict:
    base = dict(
        ticket_id="TKT-0000", created_at="2024-01-01 00:00:00", category="Technical",
        priority="Medium", status="Resolved", response_time_hrs=2.0,
        resolution_time_hrs=8.0, agent_id="AGT-01", customer_rating=4,
        issue_summary="x",
    )
    base.update(kwargs)
    return base


def _patched(rows: list[dict], test_fn, reference_now: str | None = None):
    """Run test_fn with app.db.run_select and reference_now backed by an
    in-memory sqlite connection built from `rows`, instead of the real DB file.
    """
    conn = _make_conn(rows)

    def fake_run_select(sql, params=(), row_limit=None):
        cur = conn.execute(sql, params)
        out = cur.fetchall()
        return [dict(r) for r in out]

    def fake_reference_now():
        return reference_now or "2024-01-10 00:00:00"

    with mock.patch("app.anomaly.run_select", fake_run_select), \
         mock.patch("app.anomaly.reference_now", fake_reference_now):
        test_fn()
    conn.close()


# --- stale_high_priority ----------------------------------------------------

def test_stale_critical_ticket_is_flagged():
    rows = [
        _row(ticket_id="TKT-STALE", priority="Critical", status="Open",
             created_at="2024-01-01 00:00:00"),   # 216h before ref -> stale
    ]
    def check():
        found = anomaly.rule_stale_high_priority()
        assert len(found) == 1
        assert found[0].ticket_id == "TKT-STALE"
        assert found[0].severity == "high"          # Critical -> high
    _patched(rows, check, reference_now="2024-01-10 00:00:00")


def test_fresh_high_priority_ticket_is_not_flagged():
    rows = [
        _row(ticket_id="TKT-FRESH", priority="High", status="Open",
             created_at="2024-01-09 12:00:00"),   # 12h before ref -> not stale
    ]
    def check():
        assert anomaly.rule_stale_high_priority() == []
    _patched(rows, check, reference_now="2024-01-10 00:00:00")


def test_resolved_high_priority_ticket_is_never_stale():
    rows = [
        _row(ticket_id="TKT-DONE", priority="Critical", status="Resolved",
             created_at="2024-01-01 00:00:00"),
    ]
    def check():
        assert anomaly.rule_stale_high_priority() == []
    _patched(rows, check, reference_now="2024-01-10 00:00:00")


def test_low_priority_ticket_is_never_flagged_by_this_rule():
    rows = [
        _row(ticket_id="TKT-LOW", priority="Low", status="Open",
             created_at="2024-01-01 00:00:00"),
    ]
    def check():
        assert anomaly.rule_stale_high_priority() == []
    _patched(rows, check, reference_now="2024-01-10 00:00:00")


# --- IQR outlier rules -------------------------------------------------------
# Fixture: 8 Technical tickets with resolution_time_hrs =
#   [4, 5, 6, 7, 8, 9, 10, 40]
# n=8 -> positions for quartiles (linear interpolation, pos = q*(n-1)):
#   Q1 @ pos 1.75 -> between idx1(5) and idx2(6): 5 + 0.75*(6-5) = 5.75
#   Q3 @ pos 5.25 -> between idx5(9) and idx6(10): 9 + 0.25*(10-9) = 9.25
#   IQR = 9.25 - 5.75 = 3.5
#   upper fence = 9.25 + 1.5*3.5 = 14.5
# -> only the 40 exceeds 14.5. Exactly one finding expected.

_IQR_VALUES = [4, 5, 6, 7, 8, 9, 10, 40]


def _iqr_fixture_rows() -> list[dict]:
    return [
        _row(ticket_id=f"TKT-{i:04d}", category="Technical",
             resolution_time_hrs=v, response_time_hrs=1.0, status="Resolved")
        for i, v in enumerate(_IQR_VALUES)
    ]


def test_resolution_outlier_matches_hand_computed_threshold():
    rows = _iqr_fixture_rows()
    def check():
        found = anomaly.rule_resolution_outlier()
        assert len(found) == 1
        f = found[0]
        assert f.value == 40
        assert abs(f.threshold - 14.5) < 0.01
    _patched(rows, check)


def test_resolution_outlier_respects_min_group_size():
    # Same shape as above but only 3 rows -- below min_group_size (8), so the
    # category must be skipped entirely regardless of how extreme the values are.
    rows = [
        _row(ticket_id="TKT-A", category="Billing", resolution_time_hrs=1),
        _row(ticket_id="TKT-B", category="Billing", resolution_time_hrs=2),
        _row(ticket_id="TKT-C", category="Billing", resolution_time_hrs=999),
    ]
    def check():
        assert anomaly.rule_resolution_outlier() == []
    _patched(rows, check)


def test_resolution_outlier_is_computed_per_category_not_globally():
    # Technical's outlier fixture, plus a Billing group whose normal range
    # would make Technical's 9s and 10s look extreme under a *global* IQR.
    # A per-category rule must still flag exactly one row (the Technical 40),
    # not the Technical 9/10 values relative to Billing's tighter spread.
    billing_low = [
        _row(ticket_id=f"TKT-B{i:04d}", category="Billing",
             resolution_time_hrs=v, status="Resolved")
        for i, v in enumerate([1, 1, 1, 2, 2, 2, 1, 2])
    ]
    rows = _iqr_fixture_rows() + billing_low
    def check():
        found = anomaly.rule_resolution_outlier()
        technical_flags = [f for f in found if f.context["category"] == "Technical"]
        billing_flags = [f for f in found if f.context["category"] == "Billing"]
        assert len(technical_flags) == 1
        assert technical_flags[0].ticket_id == "TKT-0007"  # the 40
        assert len(billing_flags) == 0
    _patched(rows, check)


def test_zero_spread_category_produces_no_findings():
    # All identical values -> IQR=0 -> nothing can be "beyond" the fence.
    # This must not divide-by-zero or flag everything.
    rows = [
        _row(ticket_id=f"TKT-{i:04d}", category="General",
             resolution_time_hrs=5.0, status="Resolved")
        for i in range(10)
    ]
    def check():
        assert anomaly.rule_resolution_outlier() == []
    _patched(rows, check)


# --- poor_rating_resolved ----------------------------------------------------

def test_poor_rating_flags_only_1_and_2_on_resolved():
    rows = [
        _row(ticket_id="TKT-BAD1", status="Resolved", customer_rating=1),
        _row(ticket_id="TKT-BAD2", status="Resolved", customer_rating=2),
        _row(ticket_id="TKT-OK", status="Resolved", customer_rating=3),
        _row(ticket_id="TKT-GREAT", status="Resolved", customer_rating=5),
        # Not resolved -- even a hypothetical low rating here must not count.
        _row(ticket_id="TKT-OPEN", status="Open", customer_rating=None),
    ]
    def check():
        found = anomaly.rule_poor_rating_resolved()
        ids = {f.ticket_id for f in found}
        assert ids == {"TKT-BAD1", "TKT-BAD2"}
        sev = {f.ticket_id: f.severity for f in found}
        assert sev["TKT-BAD1"] == "high"    # rating 1
        assert sev["TKT-BAD2"] == "medium"  # rating 2
    _patched(rows, check)


# --- agent_performance_outlier ------------------------------------------

def test_agent_outlier_flags_slow_agent_relative_to_peers():
    # 5 agents averaging 10h, 1 agent averaging 100h -- hand-computed:
    # mean=25, variance=[5*(10-25)^2+(100-25)^2]/6=1125, std=33.54,
    # z=(100-25)/33.54=2.236 -> exceeds threshold 2.0, but under 1.5x (3.0)
    # so severity is "medium", not "high".
    rows = []
    for i, agent_time in enumerate([10, 10, 10, 10, 10, 100]):
        agent = f"AGT-{i:02d}"
        for j in range(5):  # agent_min_tickets = 5
            rows.append(_row(
                ticket_id=f"TKT-{agent}-{j}", agent_id=agent,
                resolution_time_hrs=agent_time, status="Resolved",
            ))
    def check():
        found = anomaly.rule_agent_performance_outlier()
        slow = [f for f in found if f.ticket_id == "AGT-05"]
        assert len(slow) == 1
        assert slow[0].severity == "medium"
        assert round(slow[0].value, 2) == 100.0
        assert round(slow[0].context["z_score"], 2) == 2.24
        # the 5 normal agents must not be flagged
        assert not any(f.ticket_id != "AGT-05" for f in found if f.context["metric"] == "avg resolution time (hrs)")
    _patched(rows, check)


def test_agent_outlier_ignores_agents_below_min_tickets():
    # Same shape as above, but the "outlier" agent only has 2 tickets --
    # below agent_min_tickets=5, so it must be excluded entirely, not
    # flagged and not counted in the peer pool either.
    rows = []
    for i, agent_time in enumerate([10, 10, 10, 10, 10]):
        agent = f"AGT-{i:02d}"
        for j in range(5):
            rows.append(_row(
                ticket_id=f"TKT-{agent}-{j}", agent_id=agent,
                resolution_time_hrs=agent_time, status="Resolved",
            ))
    for j in range(2):  # too few tickets to be eligible
        rows.append(_row(
            ticket_id=f"TKT-AGT-05-{j}", agent_id="AGT-05",
            resolution_time_hrs=500, status="Resolved",
        ))
    def check():
        found = anomaly.rule_agent_performance_outlier()
        assert not any(f.ticket_id == "AGT-05" for f in found)
    _patched(rows, check)


def test_agent_outlier_direction_only_flags_the_bad_side():
    # Rating rule: LOW rating is bad, HIGH rating is not. 5 agents rate
    # 3.0, one rates 5.0 (high, good) -- must NOT be flagged even though
    # its z-score magnitude clears the threshold, because it's on the
    # "good" side. Hand-computed: mean=3.333, std=0.745, z=+2.236.
    rows = []
    for i, rating in enumerate([3, 3, 3, 3, 3, 5]):
        agent = f"AGT-{i:02d}"
        for j in range(5):
            rows.append(_row(
                ticket_id=f"TKT-{agent}-{j}", agent_id=agent,
                customer_rating=rating, status="Resolved",
            ))
    def check():
        found = anomaly.rule_agent_performance_outlier()
        rating_findings = [f for f in found if f.context.get("metric") == "avg customer rating"]
        assert rating_findings == []  # the high-rating agent must not appear
    _patched(rows, check)


def test_agent_outlier_flags_low_rating_agent():
    # Mirror of the above: 5 agents rate 4.0, one rates 1.0 (low, bad).
    # mean=3.5, variance=1.25, std=1.118, z=(1-3.5)/1.118=-2.236 -> flagged.
    rows = []
    for i, rating in enumerate([4, 4, 4, 4, 4, 1]):
        agent = f"AGT-{i:02d}"
        for j in range(5):
            rows.append(_row(
                ticket_id=f"TKT-{agent}-{j}", agent_id=agent,
                customer_rating=rating, status="Resolved",
            ))
    def check():
        found = anomaly.rule_agent_performance_outlier()
        low = [f for f in found if f.ticket_id == "AGT-05" and f.context["metric"] == "avg customer rating"]
        assert len(low) == 1
        assert round(low[0].context["z_score"], 2) == -2.24
    _patched(rows, check)


def test_agent_outlier_needs_at_least_3_eligible_agents():
    # Only 2 agents with enough tickets -- too few to compute a meaningful
    # spread, must return nothing regardless of how different they are.
    rows = []
    for i, agent_time in enumerate([10, 1000]):
        agent = f"AGT-{i:02d}"
        for j in range(5):
            rows.append(_row(
                ticket_id=f"TKT-{agent}-{j}", agent_id=agent,
                resolution_time_hrs=agent_time, status="Resolved",
            ))
    def check():
        found = anomaly.rule_agent_performance_outlier()
        assert found == []
    _patched(rows, check)


# --- detect_all orchestration -------------------------------------------------

def test_detect_all_merges_and_sorts_high_severity_first():
    rows = [
        _row(ticket_id="TKT-MED", status="Resolved", customer_rating=2),
        _row(ticket_id="TKT-HIGH", status="Resolved", customer_rating=1),
    ]
    def check():
        result = anomaly.detect_all(types=["poor_rating_resolved"])
        assert result["total"] == 2
        assert result["findings"][0]["severity"] == "high"
        assert result["counts_by_rule"] == {"poor_rating_resolved": 2}
    _patched(rows, check)


def test_detect_all_rejects_unknown_type():
    try:
        anomaly.detect_all(types=["not_a_real_rule"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


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
