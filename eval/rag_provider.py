"""Promptfoo custom provider that calls the real Mini RAG pipeline.

Promptfoo sends each test question as the prompt. We pass it to
`query.run_rag` (no duplicated RAG logic) and return:
  output   - the generated answer (what gets graded)
  metadata - retrieved context + source metadata (used by
             context-faithfulness via `contextTransform`)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# query.py uses relative paths (chroma_db/, .env), so run from the project root.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import time  # noqa: E402

from query import run_rag  # noqa: E402

# Extra backoff for transient Gemini overload (503/429) so a busy API
# doesn't show up as a failed eval case. query.py's own retry is unchanged.
BACKOFF_SECONDS = [10, 30, 60]


def call_api(prompt, options, context):
    # Generation model for the eval, from promptfooconfig.yaml (providers[].config.rag_model).
    # If unset, query.py's default applies (RAG_MODEL env var, else gemini-3.8-flash).
    model = ((options or {}).get("config") or {}).get("rag_model")

    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            result = run_rag(prompt, model=model)
            break
        except Exception as e:
            msg = str(e)
            # Retry overload / per-minute limits; a per-day quota won't recover in seconds.
            transient = any(c in msg for c in ("503", "429", "UNAVAILABLE")) and "PerDay" not in msg
            if transient and attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])
                continue
            response = {"error": f"RAG pipeline failed: {type(e).__name__}: {e}"}
            if "PerDay" in msg:
                # Tell Promptfoo's scheduler not to retry an exhausted daily quota.
                response["metadata"] = {"rateLimitKind": "quota", "rateLimitRetryable": False}
            return response

    return {
        "output": result["answer"],
        "metadata": {
            "rag_model": result["model"],
            "context": result["context"],
            "sources": [
                f"{m['source']} p{m['page']} c{m['chunk']}"
                for m in result["sources"]
            ],
        },
    }
