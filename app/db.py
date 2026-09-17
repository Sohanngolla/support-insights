"""Read-only database access.

This is the second of two independent defences against a bad generated query.
The first is app.guard, which inspects the SQL text. This one makes the
*connection itself* incapable of writing, using two mechanisms:

1. SQLite URI mode=ro -- the file is opened read-only at the OS level.
2. A sqlite3 authorizer callback -- vetoes any operation that is not a read,
   and restricts reads to the one table we expose.

Either alone would mostly do. Both together mean a bug in the text-level guard
is not sufficient to cause damage, which is the property you actually want.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)


class QueryExecutionError(RuntimeError):
    """Raised when a validated query still fails to execute."""


# Operations a SELECT legitimately needs. Everything else is denied.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}

_ALLOWED_TABLES = {settings.table_name}


def _authorizer(action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and arg1 not in _ALLOWED_TABLES:
        # Blocks sqlite_master and friends -- no schema exfiltration.
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect(db_path: Path | None = None, read_only: bool = True) -> sqlite3.Connection:
    path = Path(db_path or settings.db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run: python -m app.ingest"
        )

    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row

    # Connection pragmas must be set BEFORE the authorizer is installed --
    # the authorizer denies PRAGMA, including our own.
    conn.execute(f"PRAGMA busy_timeout = {settings.query_timeout_s * 1000}")

    if read_only:
        conn.set_authorizer(_authorizer)
    return conn


def run_select(sql: str, params: tuple = (), row_limit: int | None = None) -> list[dict]:
    """Execute a validated SELECT and return plain dicts.

    row_limit defaults to settings.max_rows, which exists to cap what an
    LLM-generated (and therefore unpredictable) query can pull back for
    display. Internal engine code -- anomaly scans, /stats -- knows exactly
    what it's asking for and passes an explicit higher row_limit so it isn't
    silently capped by a setting meant for a different caller.

    Callers get ordinary Python types, never sqlite3.Row, so nothing downstream
    depends on this module's implementation.
    """
    limit = row_limit if row_limit is not None else settings.max_rows
    conn = connect()
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]
            log.warning("result truncated to %s rows", limit)
        return [dict(row) for row in rows]
    except sqlite3.DatabaseError as exc:
        raise QueryExecutionError(str(exc)) from exc
    finally:
        conn.close()


def scalar(sql: str, params: tuple = ()) -> Any:
    rows = run_select(sql, params)
    if not rows:
        return None
    return next(iter(rows[0].values()))


def dataset_stats() -> dict:
    """Cheap descriptive summary. Proves ingestion worked without an LLM call."""
    table = settings.table_name
    total = scalar(f"SELECT COUNT(*) FROM {table}")
    date_range = run_select(
        f"SELECT MIN(created_at) AS first_ticket, MAX(created_at) AS last_ticket FROM {table}"
    )[0]

    def _breakdown(column: str) -> dict[str, int]:
        # Explicit limit: this is an internal scan (a handful of GROUP BY
        # buckets), not an LLM-facing query, so it shouldn't inherit
        # settings.max_rows just because that happens to be large enough today.
        rows = run_select(
            f"SELECT {column} AS k, COUNT(*) AS n FROM {table} "
            f"GROUP BY {column} ORDER BY n DESC",
            row_limit=1000,
        )
        return {str(r["k"]): r["n"] for r in rows}

    resolved_avg = scalar(
        f"SELECT ROUND(AVG(resolution_time_hrs), 2) FROM {table} "
        f"WHERE resolution_time_hrs IS NOT NULL"
    )
    rating_avg = scalar(
        f"SELECT ROUND(AVG(customer_rating), 2) FROM {table} "
        f"WHERE customer_rating IS NOT NULL"
    )

    return {
        "total_tickets": total,
        "date_range": date_range,
        "by_status": _breakdown("status"),
        "by_category": _breakdown("category"),
        "by_priority": _breakdown("priority"),
        "distinct_agents": scalar(f"SELECT COUNT(DISTINCT agent_id) FROM {table}"),
        "avg_resolution_time_hrs": resolved_avg,
        "avg_customer_rating": rating_avg,
    }


def reference_now() -> str:
    """The 'current time' used by staleness rules.

    The dataset is historical, so wall-clock now would flag every unresolved
    ticket as stale. Default to the newest created_at in the data; allow an
    override via ANOMALY_NOW for demonstrating live behaviour.
    """
    if settings.anomaly_now:
        return settings.anomaly_now
    return scalar(f"SELECT MAX(created_at) FROM {settings.table_name}")
