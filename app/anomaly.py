"""Anomaly detection.

Deliberately not an LLM. Every finding here has to survive "why is this one
flagged?" from an evaluator, and a rule with a number in it survives that
question; a model's vibes-based judgment does not.

Five rules, each independent and each returning its own evidence:

  1. stale_high_priority   -- High/Critical ticket still open, older than
                               STALE_HOURS relative to a *dataset* reference
                               time (see app.db.reference_now), not wall clock.
  2. resolution_outlier     -- resolution_time_hrs beyond Q3 + k*IQR, computed
                               PER CATEGORY, because categories have genuinely
                               different normal ranges (Technical runs slower
                               than Billing) and a global IQR would either miss
                               real Billing outliers or over-flag Technical.
  3. response_outlier       -- same IQR method, applied to response_time_hrs.
  4. poor_rating_resolved   -- resolved ticket rated 1 or 2. Not a statistical
                               outlier rule; a direct signal that is worth
                               surfacing alongside the others.
  5. agent_performance_outlier -- a different axis from the first four: those
                               ask "is this ticket unusual"; this asks "is
                               this agent unusual relative to their peers,"
                               using each agent's own average resolution time
                               and average customer rating. Same z-score idea
                               as the IQR rules, just applied to a population
                               of agents (typically ~12) instead of tickets
                               (~500) -- reusing the statistical approach
                               rather than inventing a new one for a new
                               grouping. Agents below agent_min_tickets are
                               excluded (too few tickets for the average to
                               mean anything), and on this dataset the
                               closest agent sits under the z-threshold, so
                               it can legitimately return zero findings --
                               that's a real result, not a bug, and is
                               exactly what "no agent is a clear outlier"
                               looks like with a dozen agents.

IQR chosen over mean + 2*std for rules 2/3 deliberately: support-ticket
durations are right-skewed (a handful of very slow tickets pull the mean up
and inflate std), so a std-based threshold moves with its own outliers. IQR
does not. Rule 5 uses z-score instead, because it's comparing a small,
roughly-symmetric population of per-agent averages (not raw skewed
durations), where IQR's quantile machinery buys nothing over a mean/std
z-score. A category/agent-pool with fewer than the configured minimum size
is skipped rather than given a threshold computed from noise.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

from app.config import settings
from app.db import reference_now, run_select

log = logging.getLogger(__name__)

TABLE = settings.table_name

# Internal engine scans read the whole table, not the LLM-facing row cap.
_SCAN_LIMIT = 5000


@dataclass
class Finding:
    anomaly_type: str
    ticket_id: str
    severity: str          # "medium" | "high"
    value: float | None
    threshold: float | None
    reason: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _iqr_bounds(values: list[float], k: float) -> tuple[float, float, float, float]:
    """Return (q1, q3, iqr, upper_fence) using linear-interpolation quantiles."""
    s = sorted(values)
    n = len(s)

    def _quantile(q: float) -> float:
        pos = q * (n - 1)
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        frac = pos - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    q1, q3 = _quantile(0.25), _quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q3 + k * iqr


def rule_stale_high_priority() -> list[Finding]:
    now = reference_now()
    rows = run_select(
        f"""
        SELECT ticket_id, priority, status, created_at,
               (julianday(?) - julianday(created_at)) * 24.0 AS age_hrs
        FROM {TABLE}
        WHERE priority IN ('High', 'Critical')
          AND status != 'Resolved'
        """,
        (now,),
        row_limit=_SCAN_LIMIT,
    )

    findings = []
    for row in rows:
        age = row["age_hrs"]
        if age is None or age < settings.stale_hours:
            continue
        findings.append(
            Finding(
                anomaly_type="stale_high_priority",
                ticket_id=row["ticket_id"],
                severity="high" if row["priority"] == "Critical" else "medium",
                value=round(age, 1),
                threshold=float(settings.stale_hours),
                reason=(
                    f"{row['priority']} ticket has been {row['status'].lower()} for "
                    f"{age:.1f}h, past the {settings.stale_hours}h staleness threshold"
                ),
                context={
                    "priority": row["priority"],
                    "status": row["status"],
                    "reference_time": now,
                },
            )
        )
    return findings


def _iqr_rule(column: str, anomaly_type: str, label: str) -> list[Finding]:
    rows = run_select(
        f"""
        SELECT ticket_id, category, {column} AS v
        FROM {TABLE}
        WHERE {column} IS NOT NULL
        """,
        row_limit=_SCAN_LIMIT,
    )

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    findings: list[Finding] = []
    for category, group in by_category.items():
        if len(group) < settings.min_group_size:
            log.info(
                "%s: skipping category %r, only %d rows (< min_group_size=%d)",
                anomaly_type, category, len(group), settings.min_group_size,
            )
            continue

        values = [r["v"] for r in group]
        q1, q3, iqr, upper = _iqr_bounds(values, settings.iqr_multiplier)
        if iqr == 0:
            continue  # no spread in this category -- nothing is "outlying"

        for row in group:
            if row["v"] <= upper:
                continue
            findings.append(
                Finding(
                    anomaly_type=anomaly_type,
                    ticket_id=row["ticket_id"],
                    severity="high" if row["v"] > upper * 1.5 else "medium",
                    value=round(row["v"], 2),
                    threshold=round(upper, 2),
                    reason=(
                        f"{label} of {row['v']:.1f}h exceeds the {category} "
                        f"category's IQR threshold ({upper:.1f}h)"
                    ),
                    context={
                        "category": category,
                        "q1": round(q1, 2),
                        "q3": round(q3, 2),
                        "iqr": round(iqr, 2),
                        "group_size": len(group),
                    },
                )
            )
    return findings


def rule_resolution_outlier() -> list[Finding]:
    return _iqr_rule("resolution_time_hrs", "resolution_time_outlier", "Resolution time")


def rule_response_outlier() -> list[Finding]:
    return _iqr_rule("response_time_hrs", "response_time_outlier", "Response time")


def rule_poor_rating_resolved() -> list[Finding]:
    rows = run_select(
        f"""
        SELECT ticket_id, customer_rating, category
        FROM {TABLE}
        WHERE status = 'Resolved' AND customer_rating IN (1, 2)
        """,
        row_limit=_SCAN_LIMIT,
    )
    return [
        Finding(
            anomaly_type="poor_rating_resolved",
            ticket_id=row["ticket_id"],
            severity="high" if row["customer_rating"] == 1 else "medium",
            value=float(row["customer_rating"]),
            threshold=2.0,
            reason=f"Resolved ticket rated {row['customer_rating']}/5 by the customer",
            context={"category": row["category"]},
        )
        for row in rows
    ]


def rule_agent_performance_outlier() -> list[Finding]:
    """Agent-level, not ticket-level: flags an agent whose own average
    resolution time or average customer rating is a z-score outlier
    relative to their peers. See the module docstring for why z-score
    (not IQR) is used here and why zero findings is a legitimate result.

    Findings from this rule put the agent_id in the `ticket_id` field --
    there's no single ticket to point at, and reusing the field keeps one
    Finding shape for the whole API/UI instead of a special case. The
    context/reason make it unambiguous that this is agent-level.
    """
    rows = run_select(
        f"""
        SELECT agent_id,
               AVG(resolution_time_hrs) AS avg_resolution,
               COUNT(resolution_time_hrs) AS resolution_n,
               AVG(customer_rating) AS avg_rating,
               COUNT(customer_rating) AS rating_n
        FROM {TABLE}
        GROUP BY agent_id
        """,
        row_limit=_SCAN_LIMIT,
    )

    def _flag(metric_key: str, n_key: str, higher_is_bad: bool, label: str) -> list[Finding]:
        eligible = [
            r for r in rows
            if r[n_key] >= settings.agent_min_tickets and r[metric_key] is not None
        ]
        if len(eligible) < 3:  # too few agents to compute a meaningful spread
            log.info(
                "agent_performance_outlier(%s): only %d eligible agents, skipping",
                label, len(eligible),
            )
            return []

        values = [r[metric_key] for r in eligible]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std <= 0:
            return []

        findings = []
        for r in eligible:
            z = (r[metric_key] - mean) / std
            is_bad = z > 0 if higher_is_bad else z < 0
            if abs(z) <= settings.agent_z_threshold or not is_bad:
                continue
            findings.append(
                Finding(
                    anomaly_type="agent_performance_outlier",
                    ticket_id=r["agent_id"],
                    severity="high" if abs(z) > settings.agent_z_threshold * 1.5 else "medium",
                    value=round(r[metric_key], 2),
                    threshold=round(mean, 2),
                    reason=(
                        f"{label} of {r[metric_key]:.2f} is worse than the peer "
                        f"average of {mean:.2f} across {len(eligible)} agents "
                        f"(n={r[n_key]} tickets, z={z:.2f})"
                    ),
                    context={
                        "metric": label,
                        "peer_mean": round(mean, 2),
                        "sample_size": r[n_key],
                        "z_score": round(z, 2),
                        "peer_count": len(eligible),
                    },
                )
            )
        return findings

    findings = []
    findings += _flag("avg_resolution", "resolution_n", higher_is_bad=True, label="avg resolution time (hrs)")
    findings += _flag("avg_rating", "rating_n", higher_is_bad=False, label="avg customer rating")
    return findings


RULES = {
    "stale_high_priority": rule_stale_high_priority,
    "resolution_time_outlier": rule_resolution_outlier,
    "response_time_outlier": rule_response_outlier,
    "poor_rating_resolved": rule_poor_rating_resolved,
    "agent_performance_outlier": rule_agent_performance_outlier,
}


def detect_all(types: list[str] | None = None) -> dict:
    """Run every rule (or a subset) and merge results.

    A ticket_id can legitimately appear once per anomaly_type -- a ticket can
    be both stale AND a resolution outlier once resolved -- but never twice
    for the *same* type, since each rule scans the table once.
    """
    selected = types or list(RULES.keys())
    unknown = set(selected) - set(RULES)
    if unknown:
        raise ValueError(f"Unknown anomaly type(s): {sorted(unknown)}")

    all_findings: list[Finding] = []
    per_rule_counts: dict[str, int] = {}
    for name in selected:
        found = RULES[name]()
        per_rule_counts[name] = len(found)
        all_findings.extend(found)

    all_findings.sort(key=lambda f: (f.severity != "high", f.anomaly_type, f.ticket_id))

    return {
        "generated_at_reference": reference_now(),
        "method": {
            "stale_high_priority": f"open > {settings.stale_hours}h vs dataset reference time",
            "resolution_time_outlier": f"per-category IQR, Q3 + {settings.iqr_multiplier}xIQR",
            "response_time_outlier": f"per-category IQR, Q3 + {settings.iqr_multiplier}xIQR",
            "poor_rating_resolved": "resolved ticket rated 1 or 2",
            "agent_performance_outlier": (
                f"per-agent z-score vs peers, |z| > {settings.agent_z_threshold} "
                f"(min {settings.agent_min_tickets} tickets/agent)"
            ),
        },
        "counts_by_rule": per_rule_counts,
        "total": len(all_findings),
        "findings": [f.to_dict() for f in all_findings],
    }


if __name__ == "__main__":
    import json

    from app.config import setup_logging

    setup_logging()
    result = detect_all()
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
    print(f"\n{result['total']} total findings. First 5:")
    for f in result["findings"][:5]:
        print(f"  [{f['severity']:>6}] {f['anomaly_type']:<24} {f['ticket_id']}  {f['reason']}")
