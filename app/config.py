"""Single source of truth for configuration.

Everything tunable is read from the environment exactly once, here. No module
reaches for os.environ on its own -- that is how config drift starts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: keeps the app importable without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "%s=%r is not an integer; falling back to %s", name, raw, default
        )
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    # --- data ---
    csv_path: Path = ROOT / "data" / "support_tickets.csv"
    db_path: Path = ROOT / "data" / "support.db"
    table_name: str = "support_tickets"

    # --- llm ---
    # Ordered fallback chain. Groq first (higher requests/min), Gemini second.
    provider_chain: list[str] = field(
        default_factory=lambda: _env_list("LLM_PROVIDER_CHAIN", ["groq", "gemini"])
    )
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 45))
    sql_retry_attempts: int = field(
        default_factory=lambda: _env_int("SQL_RETRY_ATTEMPTS", 1)
    )

    # --- query safety ---
    max_rows: int = field(default_factory=lambda: _env_int("MAX_ROWS", 200))
    query_timeout_s: int = field(default_factory=lambda: _env_int("QUERY_TIMEOUT_S", 10))

    # --- anomaly detection ---
    # "now" for staleness rules. The dataset is historical, so wall-clock time
    # would mark every unresolved ticket as stale. Default to the newest
    # created_at in the data; override with an ISO timestamp when demoing.
    anomaly_now: str | None = field(default_factory=lambda: os.getenv("ANOMALY_NOW"))
    stale_hours: int = field(default_factory=lambda: _env_int("STALE_HOURS", 24))
    iqr_multiplier: float = 1.5
    min_group_size: int = 8  # below this, per-category IQR is not meaningful
    agent_min_tickets: int = 5   # below this, an agent's average is noise, not signal
    agent_z_threshold: float = 2.0  # agent pools are small, so this bar is strict

    # --- service ---
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    api_base_url: str = field(
        default_factory=lambda: os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def configured_providers(self) -> list[str]:
        """Providers in the chain that actually have a key present."""
        keys = {"groq": self.groq_api_key, "gemini": self.gemini_api_key}
        return [p for p in self.provider_chain if keys.get(p)]


settings = Settings()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
