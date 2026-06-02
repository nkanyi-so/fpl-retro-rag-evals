# fpl-retro-rag-evals

A case study in **eval methodology** for retrieval-augmented decision support.
The RAG is the substrate; the evals are the point.

## Thesis

Most RAG evals stop at "did we retrieve plausible-looking chunks?" — a judge scoring
relevance in a vacuum. This project's differentiator is an **outcome-grounded,
decision-quality eval**: did the retrieved context actually lead to a *better decision*
than a baseline, measured against what really happened?

The testbed is a retrospective RAG over the completed 2025–26 Premier League season,
answering one decision type — **captaincy** ("who should I captain in GW N?"). Because
the season is over, every decision has a ground-truth outcome (the points that captain
actually scored), so we can grade the system on results, not vibes.

### Status (honest)

- ✅ **Retrieval-eval layer — built.** Are the right chunks fetched? Plus code
  assertions for temporal integrity. Aggregate hit@5 0.875, recall@5 0.812.
- ✅ **Generation + answer-quality eval — built.** RAG generation (retrieve →
  ground → answer) plus an LLM-as-judge scoring groundedness and correctness over
  the 8 golden cases: aggregate **groundedness 5.00/5, correctness 4.75/5** (read
  the [limitation](#eval-methodology) below before trusting those numbers).
- 🚧 **Decision-quality eval — in progress (next session).** Did retrieved context
  produce more captain points than a baseline? This is the headline deliverable; the
  retrieval and answer-quality layers are the foundation it stands on.

## Critical invariant — temporal integrity

Any document used to decide gameweek N must contain only information available
*before GW N's deadline*. No hindsight leakage. Every corpus note is date-stamped and
frozen to a pre-deadline date, and retrieval applies a metadata filter (`date < GW
deadline`) at query time. This is enforced in code and covered by an assertion, not
assumed. The invariant is also why the vector store is **Chroma** — see
[Architecture](#architecture).

## This session's slice

- ONE decision type: captaincy.
- Corpus: ~10–15 short, clean, date-stamped gameweek notes (`data/corpus/`, `.md`).
- Embeddings: local, via sentence-transformers.
- Vector store: **Chroma** (native metadata filtering at query time).
- Generation + LLM-as-judge: Anthropic API.

## Architecture

Vector store is **Chroma**, not FAISS. The temporal-integrity invariant requires
metadata filtering at query time (return only chunks whose `date` precedes GW N's
deadline). Chroma supports `where` filters on metadata natively; FAISS is a pure
similarity index with no metadata predicate. The eval invariant drives the
architecture choice.

## The Problem

## Eval Methodology
### Retrieval eval
Are the right chunks fetched? hit@5 / recall@5 against hand-labelled relevant
doc-ids, run with the temporal filter off then on, plus a hard assertion that no
retrieved note post-dates its gameweek deadline. See [Failure Modes](#failure-modes-found).

### Code assertions
Temporal integrity is enforced in code, not assumed — the retrieval eval fails
loudly if any note dated on/after a case's deadline is returned under the filter.

### LLM-as-judge (answer-quality eval)
The RAG generates a captaincy answer (retrieve → ground → answer via Claude Haiku),
then an LLM-as-judge scores it 1–5 on **groundedness** (every claim supported by the
provided notes) and **correctness** (pick + reasoning align with the reference
answer). Over the 8 golden cases: aggregate **groundedness 5.00/5, correctness
4.75/5**, with every judge verdict passing pydantic validation.

> **Limitation — do not trust these numbers yet.** The scores skew high, and the
> setup explains why: the judge is the **same model class (Haiku) grading a Haiku
> answer against a synthetic corpus**. That invites a known LLM-as-judge failure
> mode — leniency and self-consistency bias (a model rates output that looks like
> its own more favourably). Near-perfect groundedness on a corpus written to be
> easy to ground in is weak evidence of real answer quality. **Next step before
> these numbers mean anything: validate the judge against human labels** — score a
> sample by hand, measure judge↔human agreement, and only then report judge scores
> as a quality signal.

### Decision-quality eval (next session)
The headline deliverable: did the retrieved context lead to more captain points than
a baseline, graded against what actually happened in 2025–26?

## Failure Modes Found

The retrieval eval (8 captaincy cases, k=5) surfaced two distinct classes of
failure. They are reported as found — the corpus was **not** adjusted to make any
of these numbers look better, because surfacing real failure modes is the point of
the project.

**1. Hindsight leakage (temporal, the thing this project exists to catch).** With
the temporal filter OFF, the GW1 query ("who should I captain in GW1?") returned a
top-5 made *entirely* of future-dated notes — later-gameweek captaincy previews
that embed close to the question but could not have been known at the GW1 deadline
— and missed the one correct note completely (hit@5 = 0 for that case). Turning the
filter ON restored the correct note and, across the whole set, *raised* the
aggregate scores: hit@5 0.750 → 0.875 and recall@5 0.625 → 0.812. The metrics
improve under filtering precisely because the future notes had been outranking the
legitimate earlier note. This is the temporal-integrity claim, demonstrated rather
than asserted.

**2. Semantic-retrieval misses (embedding weakness, not temporal).** Two cases
failed for ordinary retrieval reasons. GW12 ("captain options after the form
swing") never retrieved `gw12-form-watch`: the question's vocabulary and the note's
wording don't overlap enough for MiniLM to rank it, so captaincy-preview notes
dominated instead (hit@5 = 0, both filter modes). GW19 retrieved only one of its
two relevant notes (recall = 0.50), missing `gw19-fixture-congestion`. These are
limits of the embedding search, independent of the temporal filter, and are left as
documented findings rather than tuned away.

**3. Retrieval miss → honest refusal, not a hallucination (grounding working as
designed).** The GW12 retrieval miss above could have produced a confident wrong
answer — the model knows plenty about the 2025–26 season from pretraining. Instead,
because the system prompt forces grounding in the retrieved notes only, the RAG
answered *"The provided notes do not contain enough information to decide"* and
listed what it would need, rather than inventing a pick from its own knowledge. The
judge scored that refusal 5/5 on both dimensions. This is the intended failure
behaviour: a retrieval gap surfaces as an honest "I don't know," not a fabricated
recommendation — which is exactly what you want from a decision-support system whose
whole premise is temporal integrity.

These numbers are illustrative, not statistical — an 8-question golden set (spanning
12 ground-truth notes) is enough to expose failure modes, not to support
significance claims.
