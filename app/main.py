"""FastAPI layer.

Four endpoints: /query, /anomalies, /stats, /health. This module is a thin
wire between HTTP and the pipeline already built and tested in
app.query / app.anomaly -- no business logic lives here, only request
validation, response shaping, and the error contract below.

Error contract (see architecture doc): every failure returns
    {"error_type": ..., "message": ..., "sql": ... | null}
with error_type one of:
    llm_unavailable        -- no provider configured or all providers failed
    sql_rejected            -- generated SQL failed the guard, twice
    sql_execution_failed    -- valid SQL, but SQLite itself rejected it
    empty_result             -- reserved; we treat empty rows as valid data,
                                  not an error (see the endpoint below), so
                                  this type exists for API-consumer symmetry
                                  with the architecture doc rather than being
                                  raised internally today.
A production version would also branch on HTTP status per error_type; we
return 502 for provider failures and 422 for a rejected query, and leave the
rest at 500, which is the honest "we didn't classify this precisely" signal
rather than a guessed 4xx.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import anomaly, db
from app.config import setup_logging, settings
from app.llm import AllProvidersFailedError
from app.query import QueryError, answer_question

log = logging.getLogger(__name__)


# --- request / response models ----------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    question: str
    sql: str
    original_sql: str
    data: list[dict]
    row_count: int
    answer: str
    sql_provider: str
    answer_provider: str
    repaired: bool
    warnings: list[str]
    latency_ms: int
    out_of_scope: bool = False


class ErrorResponse(BaseModel):
    error_type: str
    message: str
    sql: str | None = None


class AnomalyFinding(BaseModel):
    anomaly_type: str
    ticket_id: str
    severity: str
    value: float | None
    threshold: float | None
    reason: str
    context: dict


class AnomalyResponse(BaseModel):
    generated_at_reference: str
    method: dict[str, str]
    counts_by_rule: dict[str, int]
    total: int
    findings: list[AnomalyFinding]


class HealthResponse(BaseModel):
    status: str
    db_reachable: bool
    total_tickets: int | None
    configured_providers: list[str]
    provider_models: dict[str, str]


# --- app setup -----------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    log.info("Support Insights API starting")
    yield


app = FastAPI(
    title="Support Insights API",
    description="NL-queryable support ticket system with rule-based anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Wide-open CORS for local demo/dev use. A real deployment would restrict
# this to the actual Streamlit origin -- flagged here as the deliberately
# easy version of a production concern, not an oversight.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UI ---------------------------------------------------------------
# Single self-contained HTML file, no build step -- served directly rather
# than mounted via StaticFiles, since there's exactly one asset and no
# subpath to serve. Fetches in the page use relative paths (/query,
# /anomalies, /health), so same-origin and no CORS round-trip needed for
# the browser's own requests.
UI_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    return FileResponse(UI_PATH)


# --- error mapping ---------------------------------------------------------

@app.exception_handler(AllProvidersFailedError)
async def handle_llm_unavailable(_, exc: AllProvidersFailedError):
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error_type="llm_unavailable", message=str(exc)).model_dump(),
    )


@app.exception_handler(QueryError)
async def handle_query_error(_, exc: QueryError):
    message = str(exc)
    # QueryError covers two distinct causes from app.query; the message text
    # is the only signal we have to tell them apart without changing that
    # module's exception hierarchy for this stage.
    if "rejected" in message.lower():
        error_type, status_code = "sql_rejected", 422
    else:
        error_type, status_code = "sql_execution_failed", 500
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error_type=error_type, message=message).model_dump(),
    )


@app.exception_handler(db.QueryExecutionError)
async def handle_db_error(_, exc: db.QueryExecutionError):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error_type="sql_execution_failed", message=str(exc)).model_dump(),
    )


# --- endpoints ---------------------------------------------------------

@app.post("/query", response_model=QueryResponse, responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
def query(req: QueryRequest) -> QueryResponse:
    started = time.monotonic()
    result = answer_question(req.question)
    latency_ms = int((time.monotonic() - started) * 1000)

    return QueryResponse(
        question=result.question,
        sql=result.sql,
        original_sql=result.original_sql,
        data=result.data,
        row_count=len(result.data),
        answer=result.answer,
        sql_provider=result.sql_provider,
        answer_provider=result.answer_provider,
        repaired=result.repaired,
        warnings=result.warnings,
        latency_ms=latency_ms,
        out_of_scope=result.out_of_scope,
    )


@app.get("/anomalies", response_model=AnomalyResponse)
def anomalies(
    type: str | None = Query(
        None,
        description="Comma-separated anomaly types to include. Omit for all rules.",
    ),
    severity: str | None = Query(
        None, description="Filter to this severity only: 'high' or 'medium'."
    ),
) -> AnomalyResponse:
    types = [t.strip() for t in type.split(",")] if type else None
    try:
        result = anomaly.detect_all(types=types)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    findings = result["findings"]
    if severity:
        if severity not in {"high", "medium"}:
            raise HTTPException(
                status_code=400, detail="severity must be 'high' or 'medium'"
            )
        findings = [f for f in findings if f["severity"] == severity]

    return AnomalyResponse(
        generated_at_reference=result["generated_at_reference"],
        method=result["method"],
        counts_by_rule=result["counts_by_rule"],
        total=len(findings),
        findings=findings,
    )


@app.get("/stats")
def stats() -> dict:
    return db.dataset_stats()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        total = db.scalar(f"SELECT COUNT(*) FROM {settings.table_name}")
        db_ok = True
    except Exception:  # noqa: BLE001 -- health check must never itself throw
        total, db_ok = None, False

    configured = settings.configured_providers()
    models = {}
    if "groq" in configured:
        models["groq"] = settings.groq_model
    if "gemini" in configured:
        models["gemini"] = settings.gemini_model

    status = "ok" if db_ok and configured else "degraded"
    return HealthResponse(
        status=status,
        db_reachable=db_ok,
        total_tickets=total,
        configured_providers=configured,
        provider_models=models,
    )


# Fail fast and loud at import time if literally nothing is configured --
# better than a server that starts and answers every /query with a 502.
if not settings.configured_providers():
    log.warning(
        "No LLM provider configured (GROQ_API_KEY / GEMINI_API_KEY both "
        "missing from .env). /query will return llm_unavailable until one is set."
    )
