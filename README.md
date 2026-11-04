# Mini RAG — ChromaDB + Gemini

A minimal Retrieval-Augmented Generation (RAG) system built with Python.

## Architecture
Documents → Chunking → ChromaDB embeddings → Top-3 retrieval → Gemini → Answer + Sources

## Features
- Ingests local PDF/text documents
- ~500-word chunks with light overlap
- Persistent local ChromaDB vector store
- Local default Chroma embedding function
- Gemini for grounded answer generation
- CLI question-answering loop
- Displays retrieved source pages/chunks
- Refuses to answer when information is not in the knowledge base

## Run

```bash
python ingest.py
python query.py
```

## RAG Evaluation

### Why evaluate?
A RAG system can fail quietly. It can retrieve the wrong chunk, add facts the documents never mention (hallucination), drift off the question, or answer things it should refuse. Spot-checking in the CLI doesn't scale and can't catch regressions. A fixed test suite with automatic scoring turns "it seems to work" into numbers you can track every time you change chunking, retrieval, prompts or models.

### What is measured
The suite uses [Promptfoo](https://promptfoo.dev) with three Gemini models:

| Role | Model | Where it's set |
|---|---|---|
| RAG generation (system under test) | **Model used for this evaluation: `gemini-3.6-flash`**, set explicitly with `rag_model` in `promptfooconfig.yaml` (chosen because the `gemini-3.8-flash` daily quota was used up). The normal CLI default is still `gemini-3.8-flash`. | eval: `rag_model` in `promptfooconfig.yaml`; CLI: `query.py` default or the `RAG_MODEL` env var |
| LLM judge (faithfulness, relevance, refusal rubric) | `gemini-3.5-flash-lite` | `promptfooconfig.yaml` |
| Embeddings for answer relevance | `gemini-embedding-001` | `promptfooconfig.yaml` |

The generation model can be overridden without code changes. Set `RAG_MODEL` for the CLI (e.g. `RAG_MODEL=gemini-3.6-flash python query.py`), or `providers[0].config.rag_model` for the eval. Delete the `rag_model` line in `promptfooconfig.yaml` to evaluate the project default, `gemini-3.8-flash`. Each result records the model that actually answered, in `metadata.rag_model`.

The judge is a different model from the generator. That keeps the generator's small free-tier quota for the RAG calls, and the system isn't grading its own answers.

The Python RAG provider, `eval/rag_provider.py`, is what runs the **real pipeline**. For each test question it calls `query.run_rag()` and passes the answer, the exact retrieved context and the source metadata back to Promptfoo. The judge only grades what the provider returns.

12 test cases (`eval/tests.yaml`), all based on `data/rag_project_knowledge_base.pdf`:
- 3 Fraud Detection, 3 LungScope and 3 StoryTime Machine questions (answerable)
- 3 unsupported questions: two "near-miss" questions about facts absent from the PDF (recall/F1, cloud hosting cost) and one general-knowledge question that tests "no outside knowledge"

| Metric | Applies to | How it works | Pass threshold |
|---|---|---|---|
| **Faithfulness** (`context-faithfulness`) | answerable | The judge breaks the answer into claims and checks each against the *retrieved* context. Low score = hallucination. | ≥ 0.8 |
| **AnswerRelevance** (`answer-relevance`) | answerable | The judge generates the questions the answer would respond to. Their embedding similarity to the real question is the score. Low score = off-topic or evasive. | ≥ 0.7 |
| **KeyFact** (`icontains`) | answerable | Deterministic check that the expected fact appears (e.g. "Isolation Forest", "94%"). Catches a faithful but incomplete answer. | must match |
| **Refusal** (`llm-rubric` + `icontains`) | unsupported | The rubric checks that the system says it lacks information **and** doesn't guess. The string check confirms the refusal phrase. Faithfulness alone can't grade refusals, because a refusal has almost no claims to verify. | both pass |

A test passes only if all its assertions pass.

### How to run
Prerequisites: Node.js >=22.22.0 (required by Promptfoo 0.124.1), the Python venv with `requirements.txt` installed, `GEMINI_API_KEY` in `.env` (Promptfoo loads `.env` automatically) and an ingested database.

```bash
python ingest.py
```

```bash
npx -y promptfoo@0.124.1 eval --no-cache
```

```bash
.venv/bin/python eval/summarize.py
```

**Rate limits.** Tests run one at a time. After each RAG call there's a 25 s pause, and each test is graded straight after its answer. That keeps generation at about 2 requests/min and the judge at about 10 requests/min. A full run makes 12 generation requests plus the judge and embedding calls. The recorded run (`gemini-3.6-flash` generating, `gemini-3.5-flash-lite` judging) took **9m 18s**. Quotas depend on the selected model and your account tier, so check the RPM/RPD for each model you use in Google AI Studio, and raise the 25 s `delay` in `promptfooconfig.yaml` if a model's limit is lower. Promptfoo retries 429s itself, and `eval/rag_provider.py` backs off on temporary 503s and per-minute 429s from generation. To try one case first (1 generation request), add `--filter-first-n 1`.

Promptfoo is run with `npx` at a pinned version, so nothing has to be installed globally. To skip Promptfoo's anonymous telemetry, prefix the command with `PROMPTFOO_DISABLE_TELEMETRY=1`.

### How to view the report
- `eval/results/report.html`: Promptfoo's standalone HTML report (every test, answer, score and grader reason). Open it in a browser.
- `eval/results/summary.md`: totals, average score per metric, refusal results and the failed cases with reasons (from `eval/summarize.py`).
- `npx promptfoo@0.124.1 view`: Promptfoo's interactive web UI for the latest run.

`eval/results/` is git-ignored.
