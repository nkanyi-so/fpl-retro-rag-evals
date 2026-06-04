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
- ✅ **Decision-quality eval — first slice built (with an honest null result).** Over
  a real, temporally-clean corpus (GW1/8/15), scored on *real* captain points with the
  pick taken by **majority vote over K runs** and an explicit **abstention rate**. The
  finding: RAG never picks the wrong player, but abstains ~half the time on even-handed
  notes — so it shows **no reliable marginal value over the crowd baseline yet** (an
  earlier single-run "+10" did not survive majority voting). See
  [Decision-Quality Eval](#decision-quality-eval-first-slice). Still to do: more GWs,
  a manually-sourced corpus for injury/DGW cases, and the remaining baselines.

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

### Decision-quality eval
The headline deliverable — its own section below: did the retrieved context lead to
more captain points than a baseline, graded against what actually happened in
2025–26? See [Decision-Quality Eval (first slice)](#decision-quality-eval-first-slice).

## Decision-Quality Eval (first slice)

The headline eval, and the only one that asks the question that matters for a
decision-support system: not "did we fetch good chunks?" but **did the retrieved
context lead to a better *decision* — more real captain points — than the
baselines?** This is a first, deliberately small slice (n=3 gameweeks). It is
**directional only; it supports no statistical claim.**

### What it measures
For GW1, GW8 and GW15, each strategy picks one captain; we score it on the points
that player *actually* scored that gameweek in 2025–26. The honest headline is the
**divergence-only differential**: RAG is credited only on the gameweeks where its
pick *differs* from the crowd's. Agreeing with the obvious pick earns nothing — that
strips out the trap of RAG "getting credit" for captaining the player everyone
already captains.

Because the RAG pick is **not deterministic even at `temperature=0`** (see below),
the pick for each GW is taken by **majority vote over K runs (default K=5)**, and the
**abstention rate** — how often the model declines to commit — is reported as a
first-class metric, not smoothed away. A GW whose majority vote is "abstain" yields
no scoreable pick.

### Method
- **Real corpus.** GW1/8/15 notes are rebuilt from real, pre-deadline FPL-API facts
  (form-to-date, fixtures, ownership), replacing the original synthetic notes. They
  are temporally clean by construction (data sliced *at* each deadline) and carry
  `source: fpl-derived`; retrieval for this eval is restricted to that source so no
  synthetic note can leak into a graded decision.
- **Outcome oracle.** A committed snapshot of the finished 2025–26 season
  (`data/fpl/`) supplies the answer key — per-gameweek points per player. Outcome is
  the *grader*, applied after the deadline; it is the answer key, not temporal
  leakage. The temporal invariant constrains the corpus (inputs), never the oracle.
- **Strategies / baselines.**
  - *RAG* — retrieve the real notes, Claude picks a captain (returned structured;
    a declined or unresolvable pick is a distinct extraction-error bucket, never a
    0-point football outcome).
  - *Template (most-captained)* — the crowd's actual armband that gameweek, from the
    FPL API's `most_captained`. The "just captain what everyone else captains"
    default, and a genuinely strong one.
  - *Ceiling* — perfect hindsight, the actual top scorer that GW (upper bound).
  - *Floor* — expected points of a random pick from the starter pool (lower bound).

> **Baseline definition (explicit).** The template is defined as **most-captained**
> (the crowd's armband). An earlier definition — the *season-to-date points leader* —
> was considered and rejected: it answers "who has been best", not "what the crowd
> does", and the two diverge at GW1 (points-leader Haaland vs the crowd's Salah).
> `most_captained` is also preferred over `most_selected` (highest-owned overall),
> because the decision under test is captaincy: at GW1 the most-*owned* player was
> Palmer, whom almost nobody captained.

### Results (majority vote, with abstention as a first-class metric)

The decisive measurement is the **abstention rate**, estimated over a larger sample
(15 runs/GW). When RAG *does* commit it is never wrong — the pick direction is rock
stable — but on the two deliberately even-handed notes it declines to commit roughly
half the time, so the majority pick itself flips batch to batch.

| GW | When it commits | Abstention (15 runs) | Majority pick | RAG pts | Template | Tmpl pts | Verdict |
|----|-----------------|---------------------:|---------------|--------:|----------|---------:|---------|
| 1  | Salah (rarely Haaland) | ~33% | Salah *or* abstain (borderline) | 8 | Salah | 8 | tie / abstain |
| 8  | Haaland (always)       | **0%** | Haaland (stable) | 13 | Haaland | 13 | tie (agreed) |
| 15 | Foden (always)         | **~73%** | **abstain** (commits Foden only in lucky batches) | 12 | Haaland | 2 | abstain / (+10 if committed) |

**The headline does not survive majority voting.** GW8 is the only stable GW, and it
agrees with the crowd — zero marginal value. GW1's majority is a coin-flip between
matching the crowd (Salah, a tie) and abstaining. GW15 — the *only* gameweek where
RAG would beat the crowd (+10, Foden over a cooled Haaland) — is exactly where it
abstains most (~73%): so the **divergence-only differential collapses to ≈0 scored
divergent GWs under reliable voting.** The +10 appears only in minority batches where
GW15 happens to commit (e.g. a K=11 run landed Foden ×6 / abstain ×5 → +10; a K=5 run
landed abstain ×5 → 0). The single-run +10 reported earlier was a lucky pass, not a
bankable result.

**Lead with the honesty:** on these three GWs RAG demonstrates **no reliable marginal
value over the crowd**. It never captains the *wrong* player, and on GW15 it clearly
*can* see the better pick (Foden, every time it commits) — but it lacks the confidence
to commit there often enough for a reliability-respecting eval to credit it. The
trustworthy output is the abstention rate, not a points differential.

> **Methodology finding — single-run LLM evals are unreliable even at
> `temperature=0`.** Output drift is not removed by zero temperature. Majority voting
> over K runs plus an explicit **abstention rate** is the mitigation — but its value
> here is *diagnostic*: it did not manufacture a stable pick, it *exposed* that the
> pick on even-handed notes is a near-coin-flip and that the earlier headline rested
> on one lucky pass. An eval that reports a single LLM output as a result is reporting
> noise; the abstention rate is what should be trusted.

### Why this needed a real corpus (the synthetic-corpus finding)
Rebuilding GW1/8/15 against real data exposed that **all three original synthetic
notes contained real factual errors**:
- **GW1** claimed Man City were *home to Burnley* with Haaland the obvious pick — in
  reality City were *away at Wolves*, and *Salah* (home to Bournemouth) was the
  most-owned template.
- **GW8** had Haaland facing a *tougher away trip* and pitched *Mbeumo of Brentford*
  as the differential — Haaland was *home* to Everton, and Mbeumo is a *Man Utd*
  player in 2025–26.
- **GW15** featured *Cole Palmer* as the in-form midfielder — by the real data the
  hottest mid was *Foden*, not Palmer.

This is concrete evidence for the project's core claim: **decision-quality numbers
computed on an invented corpus are meaningless** — a synthetic note that happens to
align (or misalign) with reality grades nothing. It is what motivated the real-corpus
rebuild, and it is why this eval reports only the three real GWs.

### Limitations & next gate
- **n = 3.** Directional only; no significance. There is currently **no reliable
  marginal-value result** — the one potentially-divergent GW (15) is dominated by
  abstention.
- **High abstention on even-handed notes.** The model is right when it commits but
  declines ~half the time on GW1/GW15. Whether to read that as appropriate caution or
  as a prompting weakness (and whether to push it to commit) is an open question for
  the next iteration. Majority voting now exposes this rather than hiding it.
- **GW5 (injury pivot) and GW10 (DGW) are not built.** The API snapshot cannot
  reconstruct pre-deadline *injury/availability* news (those fields are live, not
  point-in-time), and the synthetic DGW calendar does not match the real season's
  doubles. Both need a manually-sourced, date-verified corpus.
- **Two baselines unbuilt:** an xG-based pick, and a no-RAG LLM — the latter
  hindsight-contaminated (the model's training spans part of 2025–26), so it must be
  reported as an upper bound, not a fair peer.

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
