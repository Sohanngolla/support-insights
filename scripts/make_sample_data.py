"""Generate a stand-in support_tickets.csv with the exact schema from the brief.

This exists only so the pipeline is runnable before the real dataset is dropped in.
Replace data/support_tickets.csv with the provided file and nothing else changes.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20240115
N_ROWS = 500

CATEGORIES = ["Billing", "Technical", "General"]
CATEGORY_WEIGHTS = [0.30, 0.45, 0.25]

PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS = [0.30, 0.38, 0.22, 0.10]

STATUSES = ["Open", "Resolved", "Escalated"]
STATUS_WEIGHTS = [0.22, 0.68, 0.10]

AGENTS = [f"AGT-{i:02d}" for i in range(1, 13)]

SUMMARIES = {
    "Billing": [
        "Duplicate charge on monthly invoice",
        "Refund not received after cancellation",
        "Unable to update payment method",
        "Invoice shows incorrect tax amount",
        "Subscription renewed despite cancellation",
        "Promo code not applied at checkout",
    ],
    "Technical": [
        "Login fails with 500 error",
        "Data export times out on large accounts",
        "Mobile app crashes on startup",
        "API returns stale results after update",
        "Webhook deliveries silently dropped",
        "Dashboard charts fail to render",
        "SSO redirect loop after password reset",
    ],
    "General": [
        "Question about plan limits",
        "Request for onboarding session",
        "How to add a team member",
        "Feature request: dark mode",
        "Clarification on data retention policy",
    ],
}

# Baseline hours-to-resolution by category; Technical skews slower.
RESOLUTION_BASE = {"Billing": 9.0, "Technical": 21.0, "General": 6.0}
RESPONSE_BASE = {"Billing": 2.2, "Technical": 3.4, "General": 1.6}

# Higher priority gets faster first response but not always faster resolution.
PRIORITY_RESPONSE_FACTOR = {"Low": 1.8, "Medium": 1.2, "High": 0.7, "Critical": 0.35}

WINDOW_START = datetime(2024, 1, 1, 8, 0, 0)
WINDOW_DAYS = 75


def _lognormalish(rng: random.Random, base: float) -> float:
    """Right-skewed positive draw. Support times have a long tail, not a bell."""
    return round(base * rng.lognormvariate(0.0, 0.55), 2)


def build_rows(rng: random.Random) -> list[dict]:
    rows: list[dict] = []

    for i in range(1, N_ROWS + 1):
        category = rng.choices(CATEGORIES, CATEGORY_WEIGHTS)[0]
        priority = rng.choices(PRIORITIES, PRIORITY_WEIGHTS)[0]
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]

        created = WINDOW_START + timedelta(
            days=rng.uniform(0, WINDOW_DAYS),
            hours=rng.uniform(0, 12),
        )

        response = _lognormalish(
            rng, RESPONSE_BASE[category] * PRIORITY_RESPONSE_FACTOR[priority]
        )

        if status == "Resolved":
            resolution = _lognormalish(rng, RESOLUTION_BASE[category])
            resolution = round(max(resolution, response + 0.5), 2)
            rating = rng.choices([1, 2, 3, 4, 5], [0.05, 0.08, 0.17, 0.34, 0.36])[0]
        else:
            resolution = None
            rating = None

        rows.append(
            {
                "ticket_id": f"TKT-{i:04d}",
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "priority": priority,
                "status": status,
                "response_time_hrs": response,
                "resolution_time_hrs": resolution,
                "agent_id": rng.choice(AGENTS),
                "customer_rating": rating,
                "issue_summary": rng.choice(SUMMARIES[category]),
            }
        )

    _plant_anomalies(rng, rows)
    return rows


def _plant_anomalies(rng: random.Random, rows: list[dict]) -> None:
    """Plant a known handful of outliers so the anomaly engine has something to find.

    Without these, a synthetic dataset can look suspiciously well-behaved and the
    anomaly module appears to do nothing.
    """
    resolved = [r for r in rows if r["status"] == "Resolved"]

    # Extreme resolution times.
    for row in rng.sample(resolved, 7):
        row["resolution_time_hrs"] = round(
            RESOLUTION_BASE[row["category"]] * rng.uniform(5.0, 9.0), 2
        )

    # Extreme first-response delays.
    for row in rng.sample(rows, 5):
        row["response_time_hrs"] = round(
            RESPONSE_BASE[row["category"]] * rng.uniform(6.0, 11.0), 2
        )

    # Stale critical tickets sitting near the start of the window.
    for row in rng.sample(rows, 6):
        row["priority"] = "Critical"
        row["status"] = rng.choice(["Open", "Escalated"])
        row["resolution_time_hrs"] = None
        row["customer_rating"] = None
        row["created_at"] = (
            WINDOW_START + timedelta(days=rng.uniform(0, 12))
        ).strftime("%Y-%m-%d %H:%M:%S")

    # A few poor ratings on resolved tickets.
    for row in rng.sample([r for r in rows if r["status"] == "Resolved"], 9):
        row["customer_rating"] = rng.choice([1, 2])


def main() -> None:
    rng = random.Random(SEED)
    rows = build_rows(rng)
    rows.sort(key=lambda r: r["created_at"])

    out = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
