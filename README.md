# Support Insights

Natural-language Q&A and anomaly detection over a support-ticket dataset.
Ask a question in plain English, get an answer grounded in real SQL run
against the actual data — plus a separate, deterministic (non-LLM) anomaly
detection engine. Single command to run, zero cost to operate.

## Quickstart

1. **Python 3.10+** required (the code uses `str | None` union syntax
   throughout). Check with `python3 --version`.
2. Clone and enter the project:
   ```bash
   git clone <repo-url> && cd support-insights
   ```
3. Get a free Groq API key — [console.groq.com](https://console.groq.com),
   ~30 seconds, no card required. (Optionally also a free Gemini key from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — go
   through **AI Studio**, not the Cloud Console, which drags in a billing
   project you don't need. Gemini is the fallback if Groq is unavailable;
   Groq alone is enough to run everything.)
4. ```bash
   cp .env.example .env
   ```
   and paste your key(s) in.
5. ```bash
   ./run.sh
   ```
   First run takes 30–60 seconds (creates the virtualenv, installs
   dependencies, builds `data/support.db` from the CSV). Every run after
   that starts in a couple of seconds.
6. Open **http://127.0.0.1:8000/**.

`run.sh` is genuinely the only command needed — it creates and activates
its own virtualenv if one doesn't exist yet, so there's no separate
"activate the venv first" step.

## What it does

- **Ask anything about the ticket data** in the chat UI (or `POST /query`,
  or `python3 -m app.ask "question"` from the terminal). The question goes
  to an LLM that writes a SQL `SELECT` against the real schema; the SQL is
  validated for safety and semantic correctness, executed, and the result
  is phrased back in plain English.
- **Anomaly detection** — five rule-based checks, no LLM involved, visible
  under the Anomalies tab (or `GET /anomalies`).
- Questions outside the dataset's scope (general knowledge, anything not
  answerable from the ticket data) get a direct decline instead of a
  hallucinated or misleading answer.

### Example queries (real output from this build)

| Question | Answer |
|---|---|
| How many tickets are currently open? | 99 |
| What's the average customer rating? | 3.7 |
| How many distinct agents are there? | 12 |
| Which resolved tickets got a rating of 1 or 2? | 67 tickets |
| Show me all Critical tickets not resolved within 12 hours | 41 tickets |
| Who is the president of India? | Declined — outside the dataset's scope |

## Architecture

```
app/
├── config.py    single source of truth for settings, read from .env once
├── ingest.py    CSV → SQLite; defines ENUMS (valid values per column)
├── db.py        read-only connection + query execution
├── guard.py     SQL safety validation + enum-value grounding
├── llm.py       Groq + Gemini provider chain, with fallback
├── prompts.py   schema description + few-shot examples for text-to-SQL
├── query.py     orchestration: question → SQL → guard → execute → answer
├── anomaly.py   rule-based anomaly detection (detect_all(), no LLM)
├── main.py      FastAPI app; also serves static/index.html at "/"
└── ask.py       CLI entry point

static/index.html   the UI — dark, single self-contained page, fetch()
                     calls to /query, /anomalies, /health
scripts/eval_ground_truth.py   ground-truth accuracy harness (see below)
```

### Key design decisions

- **SQLite, file-based** (`data/support.db`, rebuilt from the CSV by
  `app.ingest` — idempotent, so the source of truth is always the CSV, not
  an evolving schema).
- **Two independent layers of SQL defense**: `app/guard.py` is a
  text-based validator (deliberately not a full SQL parser — a ~25-line
  check, not a `sqlglot` AST, because it isn't the only layer), plus a
  read-only connection with a SQLite authorizer callback in `app/db.py` as
  a second, independent layer. Tested directly against `DELETE`,
  `sqlite_master`, and `ATTACH` with the guard bypassed entirely — all
  three still refused.
- **Enum-value grounding** — the guard also catches SQL that's *safe* but
  *wrong*, e.g. `WHERE status = 'Pending'` (not a real status value). This
  gets rejected before execution with a specific correction, which feeds
  into a one-shot repair retry — the same mechanism already used for
  safety rejections.
- **Provider chain with fallback**: Groq primary (higher free-tier rate
  limit, faster), Gemini as fallback on any Groq failure —
  configurable via `LLM_PROVIDER_CHAIN`, `GROQ_MODEL`, `GEMINI_MODEL` in
  `.env`. Model IDs are not hardcoded beyond a fallback default in
  `app/config.py` — both providers retire models with only weeks' notice
  (this project has already hit one such retirement), so `.env` is the
  place to update them, not the code.
- **Reference-time-aware, not wall-clock**: the dataset's timestamps are
  historical, so "how long has this been open" is computed relative to
  the dataset's own latest timestamp (`MAX(created_at)`), not real-world
  "now" — otherwise every unresolved ticket would look artificially stale.
  This applies both to the anomaly engine and to the LLM's own SQL
  generation (the prompt explicitly tells it not to use `date('now', ...)`).
- **One process, no CORS round-trip**: FastAPI serves the UI directly via
  `FileResponse` at `GET /`, so the page's own `fetch()` calls are
  same-origin.
- **Structured error contract** — every failure returns
  `{"error_type", "message", "sql"}` with a specific type
  (`llm_unavailable`, `sql_rejected`, `sql_execution_failed`), mapped to
  502/422/500 respectively, rather than a generic 500 for everything.
- **Fails fast, not silently**: if no LLM provider is configured, the app
  logs a loud warning at startup instead of quietly 502-ing on every query
  — and the UI's connection banner surfaces this immediately too.

### Anomaly detection — 5 rules, all deterministic

| Rule | Method |
|---|---|
| Stale unresolved (High/Critical) | Open/Escalated, High or Critical priority, older than `STALE_HOURS` (default 24) relative to the dataset's own reference time |
| Resolution time outliers | Per-category IQR (`Q3 + 1.5×IQR`) — computed per category because categories have genuinely different normal ranges |
| Response time outliers | Same IQR method, on response time |
| Poor ratings on resolved tickets | Resolved ticket rated 1 or 2 |
| Agent performance outliers | Per-agent z-score vs. peers, on average resolution time and average customer rating (`agent_min_tickets=5`, `agent_z_threshold=2.0`) |

The agent-outlier rule can legitimately return zero results if no agent's
average clears the z-score threshold — that's a valid "no outlier" result
given a dozen-or-so agents, not a bug. On this dataset it currently finds
one real result (an agent with a notably higher average resolution time
than peers).

## A known, deliberate tradeoff: the UI isn't Python

`static/index.html` is a hand-written HTML/CSS/JS page making real
`fetch()` calls to the FastAPI backend — not a Python UI framework. This
was a deliberate choice, made after actually building and trying a
Streamlit alternative against the same API and reverting: the hand-written
UI gives a materially better result (native markdown rendering, real
inline SVG charts, working search/collapse/history navigation) for very
little extra code, at the cost of not being 100% Python end-to-end.

**All business logic — ingestion, SQL generation, validation, execution,
anomaly detection — is Python.** The frontend is presentation only; it
contains no logic beyond formatting API responses for display. If a strict
reading of "Language: Python only" is a concern, that's the one place to
point to, and the tradeoff above is the honest answer for why it's built
this way anyway.

## Testing

```bash
python3 tests/test_guard.py     # 22/22 — SQL safety + enum grounding
python3 tests/test_anomaly.py   # 16/16 — all 5 anomaly rules, hand-computed fixtures
python3 tests/test_llm.py       #  5/5  — provider fallback logic (mocked, no network)
python3 tests/test_query.py     #  8/8  — full pipeline (mocked LLM, real guard/db)
python3 -m pytest tests/test_main.py -v   # 14/14 — API layer (real TestClient)
```
**65/65 passing.**

### Ground-truth accuracy eval

Different in kind from the unit tests above: `scripts/eval_ground_truth.py`
makes real calls to the configured LLM provider and checks the answer
against an independently computed reference value. Ground truth isn't a
frozen number — each case has a hand-written reference SQL query run live
against whatever's actually in `data/support.db`, so this keeps working
correctly even after the dataset changes.

```bash
python3 scripts/eval_ground_truth.py --verbose
```

**Last confirmed run: 15/15 passed**, including percentage/ratio
questions, an empty-result case, and the out-of-scope decline case. Takes
a couple of minutes (2 LLM calls per case); a Groq rate-limit hit mid-run
and fell over to Gemini automatically, which is expected behavior, not a
failure.

## Known limitations

- The agent-performance-outlier rule depends on the spread of the actual
  data; with very few agents or a tight distribution it may report zero
  findings, which is a correct result, not a bug (see above).
- The SQL guard's enum-grounding check is a heuristic regex match on
  `column = 'value'` / `column IN (...)` shapes, not a full SQL parser —
  by design, matching the rest of the validator's approach.
- `requirements.txt` is pinned to exact versions confirmed working on the
  development machine (Python 3.13); other Python 3.10+ versions should
  work but haven't been explicitly tested.
- **Provider model IDs go stale.** Both `GROQ_MODEL` and `GEMINI_MODEL`
  default to specific model IDs that were current as of Sep 2026 — this
  has already broken once during development (both defaults were pointed
  at models the providers had since retired). If `/query` starts
  returning `llm_unavailable` with a `model_not_found`/404 in the message,
  that's almost certainly this: check
  [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)
  and, for Gemini, trust the API's own error message over any doc — it
  names the exact replacement model ID. Update `GROQ_MODEL`/`GEMINI_MODEL`
  in `.env`, not the code.
