"""Manual CLI for the NL query pipeline.

    python3 -m app.ask "How many tickets are open?"

Requires GROQ_API_KEY and/or GEMINI_API_KEY set (in .env or the shell) and
data/support.db already built (python3 -m app.ingest).
"""

from __future__ import annotations

import json
import sys

from app.config import setup_logging
from app.llm import AllProvidersFailedError
from app.query import QueryError, answer_question


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python3 -m app.ask "your question"')
        sys.exit(1)

    setup_logging()
    question = " ".join(sys.argv[1:])

    try:
        result = answer_question(question)
    except (QueryError, AllProvidersFailedError) as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print(f"\nQuestion: {result.question}")
    print(f"SQL ({result.sql_provider}{' , repaired' if result.repaired else ''}):")
    print(f"  {result.sql}")
    print(f"\nAnswer ({result.answer_provider}):\n  {result.answer}")
    print(f"\nRows returned: {len(result.data)}")
    if result.data:
        print(json.dumps(result.data[:5], indent=2, default=str))
        if len(result.data) > 5:
            print(f"  ... {len(result.data) - 5} more")
    if result.warnings:
        print(f"\nWarnings: {result.warnings}")


if __name__ == "__main__":
    main()
