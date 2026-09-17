"""CSV -> SQLite ingestion.

Design notes worth defending:

* The table is created with an explicit typed schema rather than letting pandas
  infer one. Inference on 500 rows is fine until a column is entirely NULL, at
  which point it silently becomes TEXT and every numeric comparison in generated
  SQL breaks. An explicit DDL removes that failure mode.
* Ingestion returns a validation report instead of raising on dirty data. A
  prototype that refuses to start because one rating is out of range is worse
  than one that loads, flags it, and says so.
* created_at is normalised to 'YYYY-MM-DD HH:MM:SS' so SQLite's date functions
  work. SQLite has no date type; the string format IS the contract.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.config import settings

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]

DDL = """
CREATE TABLE {table} (
    ticket_id           TEXT    PRIMARY KEY,
    created_at          TEXT    NOT NULL,
    category            TEXT,
    priority            TEXT,
    status              TEXT,
    response_time_hrs   REAL,
    resolution_time_hrs REAL,
    agent_id            TEXT,
    customer_rating     INTEGER,
    issue_summary       TEXT
)
"""

INDEXES = [
    "CREATE INDEX idx_status   ON {table}(status)",
    "CREATE INDEX idx_priority ON {table}(priority)",
    "CREATE INDEX idx_category ON {table}(category)",
    "CREATE INDEX idx_created  ON {table}(created_at)",
]

# Allowed values per the brief. Used for reporting only -- we never drop rows
# for failing these, because the real dataset is the authority, not our guess.
ENUMS = {
    "category": {"Billing", "Technical", "General"},
    "priority": {"Low", "Medium", "High", "Critical"},
    "status": {"Open", "Resolved", "Escalated"},
}


@dataclass
class IngestReport:
    rows_loaded: int = 0
    source: str = ""
    date_min: str | None = None
    date_max: str | None = None
    warnings: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    distinct_values: dict[str, list[str]] = field(default_factory=dict)

    def log(self) -> None:
        log.info("Loaded %s rows from %s", self.rows_loaded, self.source)
        log.info("created_at range: %s .. %s", self.date_min, self.date_max)
        for warning in self.warnings:
            log.warning("ingest: %s", warning)


def _normalise_datetime(series: pd.Series, report: IngestReport) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    unparsed = int(parsed.isna().sum())
    if unparsed:
        report.warnings.append(f"{unparsed} created_at values could not be parsed")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def load_dataframe(csv_path: Path, report: IngestReport) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    report.source = str(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Found: {list(df.columns)}"
        )

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        report.warnings.append(f"ignoring unexpected columns: {extra}")

    df = df[EXPECTED_COLUMNS].copy()

    for col in ("ticket_id", "category", "priority", "status", "agent_id"):
        df[col] = df[col].astype("string").str.strip()

    df["created_at"] = _normalise_datetime(df["created_at"], report)

    for col in ("response_time_hrs", "resolution_time_hrs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")
    df["customer_rating"] = df["customer_rating"].astype("Int64")

    _validate(df, report)
    return df


def _validate(df: pd.DataFrame, report: IngestReport) -> None:
    dupes = int(df["ticket_id"].duplicated().sum())
    if dupes:
        report.warnings.append(f"{dupes} duplicate ticket_id values")

    for col, allowed in ENUMS.items():
        seen = set(df[col].dropna().unique())
        unexpected = seen - allowed
        if unexpected:
            report.warnings.append(f"{col} has unexpected values: {sorted(unexpected)}")
        report.distinct_values[col] = sorted(str(v) for v in seen)

    ratings = df["customer_rating"].dropna()
    out_of_range = int(((ratings < 1) | (ratings > 5)).sum())
    if out_of_range:
        report.warnings.append(f"{out_of_range} customer_rating values outside 1-5")

    for col in ("response_time_hrs", "resolution_time_hrs"):
        negatives = int((df[col].dropna() < 0).sum())
        if negatives:
            report.warnings.append(f"{negatives} negative values in {col}")

    # A resolved ticket with no resolution time would quietly break averages.
    resolved_no_time = int(
        ((df["status"] == "Resolved") & df["resolution_time_hrs"].isna()).sum()
    )
    if resolved_no_time:
        report.warnings.append(
            f"{resolved_no_time} Resolved tickets have NULL resolution_time_hrs"
        )

    report.null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    report.rows_loaded = len(df)
    report.date_min = str(df["created_at"].min())
    report.date_max = str(df["created_at"].max())


def build_database(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    force: bool = True,
) -> IngestReport:
    """Rebuild the SQLite database from the CSV. Idempotent by design."""
    csv_path = Path(csv_path or settings.csv_path)
    db_path = Path(db_path or settings.db_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place the provided CSV there."
        )

    report = IngestReport()
    df = load_dataframe(csv_path, report)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if force and db_path.exists():
        db_path.unlink()

    table = settings.table_name
    with sqlite3.connect(db_path) as conn:
        conn.execute(DDL.format(table=table))
        for stmt in INDEXES:
            conn.execute(stmt.format(table=table))
        df.to_sql(table, conn, if_exists="append", index=False)
        conn.commit()

    report.log()
    return report


if __name__ == "__main__":
    from app.config import setup_logging

    setup_logging()
    rep = build_database()
    print(f"\nOK: {rep.rows_loaded} rows -> {settings.db_path}")
    if rep.warnings:
        print("Warnings:")
        for w in rep.warnings:
            print(f"  - {w}")
