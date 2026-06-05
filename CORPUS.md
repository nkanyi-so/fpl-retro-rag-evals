# Corpus

The corpus in `data/corpus/` is now **mixed**, by design, and the `source` frontmatter
field tells the two apart.

## `fpl-derived` notes — the decision-quality corpus (real)

The decision-quality eval runs only on `source: fpl-derived` notes. There are eight,
one decision per gameweek (GW1/5/6/8/9/11/13/15), each built **fact-first** from the
frozen FPL snapshot by `src/fpl_facts.py`: form-to-date over a verified-complete
window, fixture + FDR, point-in-time ownership, and the crowd's `most_captained`. The
question and reference follow the reconstructed facts, never the reverse.

Two integrity constraints shape this set:
- **Ownership** is only quoted for the ten players whose per-gameweek `selected`
  history is in the snapshot (the live API wipes on rollover, so it cannot be
  refetched). Candidate sets are drawn from those ten — which are, conveniently, the
  season's captaincy-relevant players.
- **Temporal cutoff** uses the *real* bootstrap deadlines (`data/deadlines.json` is
  regenerated from the snapshot). Every note's date and form window were verified to
  precede its real deadline. Note `source` reflects what is in *this snapshot's*
  reality, which can differ from real-world rosters — follow the data, not memory.

## `synthetic` notes — retrieval-eval distractors

The remaining notes are `source: synthetic`: hand-built distractors that exercise the
retrieval pipeline and the temporal-integrity filter — semantically-near-but-wrong
notes, and later-gameweek notes that an earlier-gameweek question matches (so the
temporal filter has something real to exclude). They are dated a few days before their
gameweek (and re-dated where the corrected real calendar required it) so they stay
pre-deadline, but their content is illustrative, not real. They are excluded from the
decision-quality eval by the `source` filter so no synthetic note can grade a decision.

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
