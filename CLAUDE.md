# CLAUDE.md — working context for Claude Code

## What this project is
A retrospective RAG over the 2025–26 Premier League season. **The point is eval
methodology, not the RAG.** The RAG exists so we have something to evaluate.

The headline deliverable is an **outcome-grounded decision-quality eval**: did the
retrieved context lead to a better captaincy decision (more captain points) than a
baseline, graded against what actually happened that season.

## Status / scope
- **Built this session:** the *retrieval* eval (are the right chunks fetched?), a few
  *code assertions*, and *one LLM-judge* for answer quality.
- **Next session:** the *decision-quality* eval (captain points vs a baseline). This is
  the project's whole point — the retrieval layer is its foundation. Do not treat it as
  a minor follow-up, but do not build it today.

## Today's vertical slice
- ONE decision type: **captaincy**.
- Corpus: ~10–15 short, clean, **date-stamped** gameweek notes in `data/corpus/` (.md).
- Embeddings: local, via sentence-transformers.
- Vector store: **Chroma** (see invariant below — this is a hard choice, not a default).
- Generation + LLM-as-judge: Anthropic API (`ANTHROPIC_API_KEY` in `.env`).

## Non-negotiable: temporal integrity
Any document used to decide gameweek N may contain only info available **before GW N's
deadline**. No hindsight leakage. Concretely:
- Every corpus note carries an explicit, pre-deadline date in its metadata.
- Retrieval for a GW-N question must exclude anything dated on/after GW N's deadline.
- This must be enforced in code and covered by an assertion, not left implicit.

## Why Chroma (not FAISS)
The eval invariant drives the architecture. Temporal integrity requires **metadata
filtering at query time** — return only chunks where `date < GW N deadline`. Chroma
supports `where` predicates on metadata natively; FAISS is a pure similarity index with
no metadata filtering. So the store is Chroma. If you ever reconsider, the temporal
filter is the constraint you must preserve.

## golden.jsonl schema
Each line is one eval case. Required fields:
- `question`     — the captaincy question, e.g. "Who should I captain in GW 12?"
- `gameweek`     — integer N (drives the temporal cutoff).
- `relevant_doc_ids` — list of corpus doc id(s) that are ground truth for the
  RETRIEVAL eval. Without these the retrieval eval has no ground truth.
- `reference_answer` — optional; a reference answer for the LLM-judge.

## Conventions
- Python: type hints always; pydantic for any structured request/response shapes.
- Keep the slice small and runnable end-to-end before adding breadth.
- If the project has tests, run them after changes; if it doesn't yet, flag it.

## Layout
- `src/ingest.py`    chunk + embed + store (date metadata attached per chunk)
- `src/retrieve.py`  question → top-k chunks (temporal `where` filter applied here)
- `src/rag.py`       retrieve + augment + Anthropic answer
- `src/evals/golden.jsonl`       ~8 hand-written cases (schema above)
- `src/evals/retrieval_eval.py`  are the right chunks fetched? (vs `relevant_doc_ids`)
- `src/evals/judge.py`           LLM-as-judge for answer quality
- `data/corpus/`     date-stamped GW notes (.md)
- `notebooks/`       scratch space (not part of the pipeline)
