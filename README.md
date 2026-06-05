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
  assertions for temporal integrity. Aggregate (12 cases) hit@5 0.917, recall@5 0.875.
- ✅ **Generation + answer-quality eval — built.** RAG generation (retrieve →
  ground → answer) plus an LLM-as-judge scoring groundedness and correctness over
  the 8 golden cases: aggregate **groundedness 5.00/5, correctness 4.75/5** (read
  the [limitation](#eval-methodology) below before trusting those numbers).
- ✅ **Decision-quality eval — widened to n=8.** Over a real, temporally-clean
  corpus (GW1/5/6/8/9/11/13/15), scored on *real* captain points. The system gives
  **decision support** (top candidates + grounded case + confidence) and a single
  **forced pick** graded by **majority vote over K runs**, with the model's
  **confidence** recorded as data. Result: RAG's divergence-only differential is
  **+19 over 3 divergent GWs (GW1 +5, GW5 +4, GW15 +10) — all at *medium* confidence
  and flagged close**. The widening also enabled a first **confidence-calibration
  check**, whose finding is blunt: **higher confidence did *not* track better
  outcomes** — the high-confidence bucket is all template agreements, two of which
  *busted* (GW9, GW11), and normalized "capture" is essentially flat across buckets
  (0.40 high vs 0.43 medium). See [Decision-Quality Eval](#decision-quality-eval-n8).
  Still to do: a manually-sourced corpus for injury cases, a real double-gameweek
  case (GW26/33/36 are reconstructable), and the remaining baselines.

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
2025–26? See [Decision-Quality Eval (n=8)](#decision-quality-eval-n8).

## Decision-Quality Eval (n=8)

The headline eval, and the only one that asks the question that matters for a
decision-support system: not "did we fetch good chunks?" but **did the retrieved
context lead to a better *decision* — more real captain points — than the
baselines?** This is a small slice (n=8 gameweeks: GW1/5/6/8/9/11/13/15). It is
**directional only; it supports no statistical claim** — but n=8 is enough to add a
first **confidence-calibration** read on top of the points differential.

### What it measures
For each of the eight gameweeks, each strategy picks one captain; we score it on the
points that player *actually* scored that gameweek in 2025–26. The honest headline is the
**divergence-only differential**: RAG is credited only on the gameweeks where its
pick *differs* from the crowd's. Agreeing with the obvious pick earns nothing — that
strips out the trap of RAG "getting credit" for captaining the player everyone
already captains.

The system gives **decision support**, not a bare pick or a refusal (see
[the product layer](#decision-support-the-user-facing-answer)). For grading it always
also emits a single **forced pick** ("if you had to commit, who?") plus a 0–1
**confidence**. Because LLM output is **not deterministic even at `temperature=0`**,
the forced pick for each GW is taken by **majority vote over K runs (default K=5)**,
and the model's **average confidence** is recorded as data — never used to drop a
pick. A win on a low-confidence divergence is weaker evidence than a high-confidence
one, so the headline differential is reported confidence-annotated.

### Method
- **Real corpus, fact-first.** All eight GW notes are built from real, pre-deadline
  FPL-API facts (form-to-date, fixtures/FDR, point-in-time ownership, the crowd's
  `most_captained`) reconstructed by a deterministic builder (`src/fpl_facts.py`), so
  every claim traces to a reproducible source rather than memory — the question and
  reference follow the facts, never the reverse. They carry `source: fpl-derived` and
  retrieval for this eval is restricted to that source so no synthetic note can leak
  into a graded decision. Two integrity constraints shape the corpus: (1) ownership is
  only quoted for the ten players whose per-gameweek `selected` history is in the
  snapshot (the API wipes on rollover), so candidate sets are drawn from them; (2) the
  temporal cutoff uses the **real** bootstrap deadlines (`data/deadlines.json` was
  regenerated from the snapshot), and every note's date and form window were verified
  to precede its real deadline — a fix that re-dated the GW15 notes, which had
  post-dated the real deadline under the earlier synthetic calendar.
- **Outcome oracle.** A committed snapshot of the finished 2025–26 season
  (`data/fpl/`) supplies the answer key — per-gameweek points per player. Outcome is
  the *grader*, applied after the deadline; it is the answer key, not temporal
  leakage. The temporal invariant constrains the corpus (inputs), never the oracle.
- **Strategies / baselines.**
  - *RAG* — retrieve the real notes; Claude returns a structured decision (candidates
    + grounded cases + close-call flag + confidence) and a single forced pick, scored
    by majority vote over K runs.
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

### Decision support: the user-facing answer

On a close call the product does **not** collapse to a bare pick or a refusal. It
returns a structured, JSON-serializable object (so a UI can render it directly): the
top candidates with the grounded case for each, a close-call flag, an explicit
confidence, and the forced pick. Example, GW15 (a genuine form-vs-pedigree call):

```json
{
  "candidates": [
    {"name": "Phil Foden", "element_id": 414,
     "case": "Form midfielder: 37 pts in the last four GWs vs Haaland's 22; same home
              fixture vs promoted Sunderland, ~20% owned — a differential with elite
              fixture backing."},
    {"name": "Erling Haaland", "element_id": 430,
     "case": "The usual premium: season points-leader (120), same fixture, but cooled
              to 22 pts in four GWs and ~70% owned — the template, not the form pick."}
  ],
  "is_close": true,
  "confidence": 0.62,
  "forced_pick_name": "Phil Foden",
  "forced_pick_element_id": 414,
  "rationale": "Form-versus-pedigree, same elite fixture. Foden's recent form outpaces
                a cooled Haaland and offers differential value; confidence is moderate
                because Haaland's pedigree and fixture still carry weight."
}
```

The eval grades only `forced_pick_element_id`; `confidence` and `is_close` are
recorded as data.

### Results (forced pick, majority vote over K=5, confidence-annotated)

Forcing the pick yields a stable, gradeable decision: across all eight GWs the forced
majority was **unanimous (100% vote share at K=5)**, and confidence still flags the
close calls (all three divergences flagged close 100% of runs; the agreements 0%).

| GW | Forced pick | Vote | Avg confidence | RAG pts | Template (most-captained) | Tmpl pts | Verdict |
|----|-------------|-----:|----------------|--------:|---------------------------|---------:|---------|
| 1  | Haaland | 100% | 0.55 (medium, close 100%) | 13 | Salah   | 8  | **+5 (diverged)** |
| 5  | Haaland | 100% | 0.61 (medium, close 100%) | 9  | Salah   | 5  | **+4 (diverged)** |
| 6  | Haaland | 100% | 0.95 (high, close 0%)     | 16 | Haaland | 16 | tie (agreed) |
| 8  | Haaland | 100% | 0.85 (high, close 0%)     | 13 | Haaland | 13 | tie (agreed) |
| 9  | Haaland | 100% | 0.80 (high, close 0%)     | 2  | Haaland | 2  | tie (agreed — **bust**) |
| 11 | Haaland | 100% | 0.88 (high, close 0%)     | 4  | Haaland | 4  | tie (agreed — **bust**) |
| 13 | Haaland | 100% | 0.65 (medium, close 80%)  | 2  | Haaland | 2  | tie (agreed — **bust**) |
| 15 | Foden   | 100% | 0.63 (medium, close 100%) | 12 | Haaland | 2  | **+10 (diverged)** |

**Aggregate:** RAG **71** vs template **52** (ceiling 139, floor 25.3); mean per GW
RAG **8.9** vs template **6.5**; record **3W–0L–5T**; agreement rate **62%**;
**divergence-only differential +19** (GW1 +5, GW5 +4, GW15 +10).

**Lead with the honesty:** the +19 is real but **all three divergence wins are
*medium* confidence and the model flagged each a close call** — directional evidence,
not a confident verdict. The pattern from the first slice holds and sharpens at n=8:
RAG beats the crowd *only where it diverges*, and it diverges *only on calls it is
itself unsure about*. Every high-confidence GW is an agreement with the Haaland
template — and three of those agreements **busted** (GW9 2 pts at conf 0.80, GW11 4
pts at conf 0.88, GW13 2 pts at conf 0.65). The system's confident calls are precisely
the ones where it adds nothing over "just captain Haaland", and being confident did
not protect them from busting.

### Confidence calibration (first check)

The n=8 sample is the first that can ask whether the model's **self-reported
confidence tracks decision quality**. Because the vs-template delta is ~0 on the weeks
RAG agrees with the crowd (a structural confound — the agreements are all
high-confidence), the cross-bucket signal is **normalized capture**: where the forced
pick lands between the random-pick floor (0.0) and the perfect-hindsight ceiling (1.0)
that GW.

| Confidence bucket | GWs | n | beat/match template | mean capture |
|-------------------|-----|--:|--------------------:|-------------:|
| high (≥0.80)      | 6, 8, 9, 11 | 4 | 100% | **0.40** |
| medium (0.50–0.79)| 1, 5, 13, 15 | 4 | 100% | **0.43** |
| low (<0.50)       | — | 0 | — | — |

**Finding: confidence is not calibrated to outcome.** Capture is essentially flat —
in fact *mildly inverted* (medium 0.43 ≥ high 0.40) — so higher-confidence picks did
**not** score better. The high-confidence bucket contains both the clean anchors
(GW6 16 pts) and outright busts (GW9 2, GW11 4); the model's 0.8–0.9 confidence on the
Haaland template carried no extra outcome quality. This is consistent with the
divergence story above: the signal lives in the medium-confidence *divergences*, not
in the high-confidence *agreements*. n=8 makes this a directional read, not a
calibrated probability — but it is concrete motivation for the calibration work the
first slice could only flag as future work.

> **Methodology finding — single-run LLM evals are unreliable even at
> `temperature=0`; the fix is forced pick + majority vote + a confidence signal.**
> Output drift is not removed by zero temperature. An earlier design let the model
> *abstain* on close calls; the abstention rate hovered near 50% on even-handed notes,
> so a single pass (and even a K=5 majority) was a coin-flip and a one-off "+10" was a
> lucky pass. Forcing a single "if you had to commit" pick and majority-voting it over
> K runs yields a **stable, gradeable** decision (here unanimous over 15 runs), while
> the model's **confidence** carries the close-call signal that abstention used to —
> as data, not as a refusal. Report the differential *and* the confidence; a
> low-confidence win is weak evidence.

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
rebuild, and it is why this eval reports only `fpl-derived` GWs (now eight), each note
traceable to the deterministic fact builder.

### Limitations & next gate
- **n = 8.** Wider than the first slice but still directional; no significance. The
  +19 rests on three GWs, all won at *medium* confidence — a signal, not a verdict.
- **Wins are confined to low-confidence divergences.** RAG beats the crowd only on the
  calls it flags as close (conf ~0.55–0.63); every high-confidence GW is a template
  agreement, and three of those *busted*. The open question is now sharper: does RAG
  ever win a divergence it is actually *confident* about? At n=8, it has not faced one.
- **Confidence is not calibrated — now measured, not just asserted.** The first
  calibration check (capture 0.40 high vs 0.43 medium, mildly inverted) shows the 0–1
  self-report does not track outcome quality. Turning this into a calibrated
  probability — and testing whether a *re-weighted* confidence would help — is the next
  step. The current score remains data, never used to drop a pick.
- **Template entrenchment limits the test.** From GW3 on, the crowd's `most_captained`
  is Haaland almost every week, so "divergence" almost always means "not Haaland". The
  eval genuinely measures *when to deviate from the template*, but a richer set of
  divergent shapes (multiple distinct template players) would stress it harder.
- **Injury pivots and double-gameweeks still deferred.** The snapshot cannot
  reconstruct pre-deadline *injury/availability* news (live fields, not point-in-time),
  so injury cases need a manually-sourced corpus. Real double-gameweeks, by contrast,
  *are* reconstructable (the snapshot has doubles at GW26/33/36) — they were deferred
  by choice, not capability, to keep this widening to one decision shape.
- **Two baselines unbuilt:** an xG-based pick, and a no-RAG LLM — the latter
  hindsight-contaminated (the model's training spans part of 2025–26), so it must be
  reported as an upper bound, not a fair peer.

## Failure Modes Found

The retrieval eval (12 captaincy cases, k=5) surfaced two distinct classes of
failure. They are reported as found — the corpus was **not** adjusted to make any
of these numbers look better, because surfacing real failure modes is the point of
the project.

**1. Hindsight leakage (temporal, the thing this project exists to catch).** With
the temporal filter OFF, the GW1 query ("who should I captain in GW1?") returned a
top-5 made *entirely* of future-dated notes — later-gameweek captaincy previews
that embed close to the question but could not have been known at the GW1 deadline
— and missed the one correct note completely (hit@5 = 0 for that case). Turning the
filter ON restored the correct note and, across the whole set, *raised* the
aggregate scores: hit@5 0.833 → 0.917 and recall@5 0.792 → 0.875. The metrics
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

These numbers are illustrative, not statistical — a 12-question golden set is enough
to expose failure modes, not to support significance claims.
