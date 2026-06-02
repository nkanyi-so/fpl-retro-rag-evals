# Corpus

## This vertical slice: synthetic corpus

For the current vertical slice the corpus in `data/corpus/` is **synthetic** — every
note is labelled `source: synthetic` in its frontmatter. It is hand-built to exercise
and validate the retrieval pipeline and the temporal-integrity filter: it contains the
ground-truth answer notes for the golden set, plus deliberate distractors (semantically
near but wrong) and at least one later-gameweek note that an earlier-gameweek question
would semantically match — so retrieval can be wrong and the temporal filter has
something real to exclude.

Temporal integrity: every note is captured as-of a pre-deadline date and frozen — no
content is added after the gameweek it informs (no hindsight). The synthetic calendar is
internally consistent (each note dated a few days before its gameweek's deadline) even
though the dates are not real-world-accurate.

## Upgrade path: real corpus (decision-quality phase)

The decision-quality phase replaces this synthetic set with a **real, temporally-clean
corpus**. The preferred source is the FPL API's per-gameweek history, which is sliceable
by date and therefore temporally clean by construction — you can reconstruct exactly what
was known before any gameweek's deadline. Candidate real sources for richer context:

- Fantasy Premier League official API (structured player/fixture/price data): https://fantasy.premierleague.com/api/bootstrap-static/
- Fantasy Football Scout (analysis & opinion)
- Reddit r/FantasyPL (community context)
- BBC Sport (team news and tactical preview)
- premierinjuries.com (injury reports)
