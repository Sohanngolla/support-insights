#!/usr/bin/env bash
# Single-command startup. One process serves everything: the FastAPI
# backend and the UI (static/index.html), served by the same app -- no
# separate frontend server, no CORS round-trip for the browser's own
# fetches.
#
# This now also bootstraps the venv itself, so the whole thing is truly
# one command from a cold terminal (git clone && ./run.sh) -- previously
# it assumed the venv was already active, which meant "source
# .venv/bin/activate && ./run.sh" was really two steps.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "No .venv found -- creating one (first run only)..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.deps_installed" ] || [ requirements.txt -nt ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install --quiet -r requirements.txt
    touch .venv/.deps_installed
fi

if [ ! -f data/support.db ]; then
    echo "No database found -- running ingestion first..."
    python3 -m app.ingest
fi

echo "Starting Support Insights on http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}"
echo "  UI:   http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/"
echo "  Docs: http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/docs"
uvicorn app.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
