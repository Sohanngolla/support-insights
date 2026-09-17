"""Ground-truth evaluation harness for the NL query pipeline.

Different from tests/test_query.py: that file mocks the LLM to test
*plumbing* (guard integration, retry logic) with zero network calls. This
script makes real calls to your configured provider(s) and checks the
final answer against an independently computed reference value.

Crucially, "ground truth" here is not a frozen number -- each case has a
hand-written reference SQL query that's run live via app.db.run_select
against whatever's actually in data/support.db. That means this harness
keeps working unchanged after you swap in the real support_tickets.csv;
only the LLM's interpretation of the question is being checked, not a
number that would go stale the moment the data changes.

Run:
    python3 scripts/eval_ground_truth.py
    python3 scripts/eval_ground_truth.py --case count_open
    python3 scripts/eval_ground_truth.py --verbose

Requires GROQ_API_KEY / GEMINI_API_KEY configured and data/support.db built
(python3 -m app.ingest). Each case costs 2 LLM calls (SQL + answer
phrasing), so the full run will bump into Groq's rate limit and fall over
to Gemini partway through on a fresh run -- that's expected, not a failure;
watch the provider column if you want to confirm the fallback fired.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import setup_logging, settings  # noqa: E402
from app.db import run_select  # noqa: E402
from app.llm import AllProvidersFailedError, build_providers  # noqa: E402
from app.query import QueryError, answer_question  # noqa: E402

TABLE = settings.table_name


# --- comparators --------------------------------------------------------
# Loose by design: the LLM chooses its own column names and row shape, so
# these check "does the expected value appear anywhere in what came back"
# rather than requiring an exact schema match.

def check_values_present(pipeline_data: list[dict], reference_data: list[dict]) -> tuple[bool, str]:
    """Every value in the single reference row must appear, stringified,
    somewhere among the pipeline's returned cells."""
    if not reference_data:
        return False, "reference query itself returned nothing -- fix the test case"
    ref = reference_data[0]
    all_cells = {str(v) for row in pipeline_data for v in row.values()}
    missing = [f"{k}={v!r}" for k, v in ref.items() if _fuzzy_match(v, all_cells) is False]
    return (not missing), (f"missing: {missing}" if missing else f"matched {dict(ref)}")


def _fuzzy_match(expected, cells: set[str]) -> bool:
    exp_str = str(expected)
    if exp_str in cells:
        return True
    try:
        exp_f = float(expected)
    except (TypeError, ValueError):
        return False
    for c in cells:
        try:
            if abs(float(c) - exp_f) <= max(0.05, abs(exp_f) * 0.01):
                return True
        except ValueError:
            continue
    return False


def check_row_count(pipeline_data: list[dict], reference_data: list[dict]) -> tuple[bool, str]:
    expected = int(next(iter(reference_data[0].values())))
    got = len(pipeline_data)
    if len(pipeline_data) == 1 and len(pipeline_data[0]) == 1:
        # pipeline answered with an aggregate COUNT(*) row instead of raw rows
        try:
            got = int(next(iter(pipeline_data[0].values())))
        except (TypeError, ValueError):
            pass
    return got == expected, f"expected {expected}, pipeline gave {got}"


def check_empty(pipeline_data: list[dict], _reference_data: list[dict]) -> tuple[bool, str]:
    return len(pipeline_data) == 0, f"expected empty, pipeline returned {len(pipeline_data)} row(s)"


@dataclass
class Case:
    id: str
    question: str
    reference_sql: str | None   # None only for the special out_of_scope case
    check: Callable[[list[dict], list[dict]], tuple[bool, str]] | None


CASES: list[Case] = [
    Case("count_open", "How many tickets are currently open?",
         f"SELECT COUNT(*) FROM {TABLE} WHERE status = 'Open'", check_values_present),
    Case("avg_rating", "What's the average customer rating?",
         f"SELECT ROUND(AVG(customer_rating), 1) FROM {TABLE} WHERE customer_rating IS NOT NULL",
         check_values_present),
    Case("distinct_agents", "How many distinct agents are there?",
         f"SELECT COUNT(DISTINCT agent_id) FROM {TABLE}", check_values_present),
    Case("count_critical", "How many tickets are Critical priority?",
         f"SELECT COUNT(*) FROM {TABLE} WHERE priority = 'Critical'", check_values_present),
    Case("pct_critical", "What percentage of tickets are Critical priority?",
         f"SELECT ROUND(100.0 * SUM(CASE WHEN priority='Critical' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM {TABLE}",
         check_values_present),
    Case("count_billing", "How many billing issues are there?",
         f"SELECT COUNT(*) FROM {TABLE} WHERE category = 'Billing'", check_values_present),
    Case("count_escalated", "How many tickets have been escalated?",
         f"SELECT COUNT(*) FROM {TABLE} WHERE status = 'Escalated'", check_values_present),
    Case("max_resolution", "What is the longest resolution time in hours?",
         f"SELECT ROUND(MAX(resolution_time_hrs), 1) FROM {TABLE}", check_values_present),
    Case("min_response", "What is the shortest response time in hours?",
         f"SELECT ROUND(MIN(response_time_hrs), 1) FROM {TABLE}", check_values_present),
    Case("count_poor_rating_resolved", "Which resolved tickets got a rating of 1 or 2?",
         f"SELECT COUNT(*) FROM {TABLE} WHERE status='Resolved' AND customer_rating IN (1,2)",
         check_row_count),
    Case("count_critical_overdue", "Show me all Critical tickets not resolved within 12 hours",
         f"SELECT COUNT(*) FROM {TABLE} WHERE priority='Critical' "
         "AND (resolution_time_hrs IS NULL OR resolution_time_hrs > 12)",
         check_row_count),
    Case("top_agent_resolved", "Which agent has resolved the most tickets?",
         f"SELECT agent_id, COUNT(*) as n FROM {TABLE} WHERE status='Resolved' "
         "GROUP BY agent_id ORDER BY n DESC LIMIT 1",
         check_values_present),
    Case("worst_category_resolution", "Which category has the highest average resolution time?",
         f"SELECT category, ROUND(AVG(resolution_time_hrs), 1) as avg_hrs FROM {TABLE} "
         "WHERE resolution_time_hrs IS NOT NULL GROUP BY category ORDER BY avg_hrs DESC LIMIT 1",
         check_values_present),
    Case("empty_2020", "Show me tickets from January 2020",
         f"SELECT * FROM {TABLE} WHERE created_at LIKE '2020-01%'", check_empty),
    Case("out_of_scope", "Who is the president of India?", None, None),
]


def run_case(case: Case, providers) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        result = answer_question(case.question, providers=providers)
    except (QueryError, AllProvidersFailedError) as exc:
        return False, f"pipeline raised: {exc}", time.time() - t0
    elapsed = time.time() - t0

    if case.id == "out_of_scope":
        ok = result.out_of_scope is True
        return ok, ("declined correctly" if ok else f"NOT declined -- answered: {result.answer!r}"), elapsed

    reference_data = run_select(case.reference_sql)
    ok, detail = case.check(result.data, reference_data)
    provider_note = f"[{result.sql_provider}]"
    return ok, f"{provider_note} {detail}", elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run only the case with this id")
    parser.add_argument("--verbose", action="store_true", help="show detail even for passing cases")
    args = parser.parse_args()

    setup_logging()

    if not settings.db_path.exists():
        print(f"ERROR: {settings.db_path} not found. Run: python3 -m app.ingest")
        sys.exit(1)

    providers = build_providers()
    if not providers:
        print("ERROR: no LLM provider configured. Set GROQ_API_KEY and/or GEMINI_API_KEY in .env.")
        sys.exit(1)

    cases = [c for c in CASES if not args.case or c.id == args.case]
    if not cases:
        print(f"No case with id '{args.case}'. Known ids: {', '.join(c.id for c in CASES)}")
        sys.exit(1)

    passed = 0
    for case in cases:
        ok, detail, elapsed = run_case(case, providers)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {case.id:<28} ({elapsed:4.1f}s)  {case.question}")
        if args.verbose or not ok:
            print(f"           {detail}")
        passed += int(ok)

    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
