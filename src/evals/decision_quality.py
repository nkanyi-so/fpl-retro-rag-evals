"""Decision-quality eval: did RAG-informed captaincy beat the naive baselines,
graded against what actually happened in 2025-26?

This is the project's headline eval. The retrieval and answer-quality evals ask
"did we fetch good chunks / write a grounded answer?". This one asks the only
question that matters for a decision-support system: did the retrieved context
lead to a *better decision*, measured in real captain points?

Strategies, each producing one captain pick per gameweek:
  - rag       : retrieve real (fpl-derived) notes -> Claude picks a captain.
  - template  : the most-captained player that GW (the crowd's actual armband,
                from the FPL API). The brutal "just captain what everyone else
                captains" default.
  - ceiling   : perfect hindsight — the actual top scorer that GW (upper bound).
  - floor     : expected points of a random pick from the starter pool (lower bound).

Reliability — MAJORITY VOTING. The RAG pick is not deterministic, even at
temperature=0: on even-handed notes the model's confident-vs-decline boolean drifts
run to run. A single pass is therefore an unreliable measurement. So for the RAG
strategy we run the pick K times (default 5) per GW and take the MAJORITY vote, with
two first-class outputs:
  - the majority pick (a committed player, or ABSTAIN if "decline" is the plurality);
  - the ABSTENTION RATE — how often the model declined to commit. This is treated as
    a real confidence signal, not noise, and reported per GW. Abstain wins ties (if
    the model cannot decisively commit, the decision is "no pick").
The baselines (template/ceiling/floor) are deterministic, so they are not voted.

Honesty guards baked into the metrics:
  - AGREEMENT RATE: how often RAG's majority pick just echoes the template. If RAG
    always picks the obvious player, it is not adding value — it is being measured on
    the crowd's pick.
  - DIVERGENCE-ONLY DIFFERENTIAL (the headline): on the GWs where RAG's majority pick
    DIFFERS from the template, did it outscore the template? This strips out the free
    credit for agreeing with the obvious pick.
  - A GW whose majority vote is ABSTAIN is a distinct bucket, never scored as 0
    football points.

n is tiny (a 3-GW slice). These numbers expose whether the harness works and tell a
directional story; they support NO statistical claim. See the README caveat.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make src/ importable
import fpl_data  # noqa: E402
from rag import answer  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
DECISION_GWS = [1, 8, 15]  # the real, temporally-clean slice built this session
REAL_SOURCE = "fpl-derived"
K_VOTES = 5  # RAG pick runs per GW; majority vote stabilises the nondeterministic pick
ABSTAIN = "ABSTAIN"  # vote label for a declined / unresolved pick
LOW_CONF_ABSTAIN = 0.40  # >= this abstention rate => the majority pick is untrustworthy


class GoldenCase(BaseModel):
    question: str
    gameweek: int
    relevant_doc_ids: list[str]
    reference_answer: str | None = None


def load_cases() -> list[GoldenCase]:
    cases = []
    for line in GOLDEN.read_text().splitlines():
        if line.strip():
            c = GoldenCase(**json.loads(line))
            if c.gameweek in DECISION_GWS:
                cases.append(c)
    return sorted(cases, key=lambda c: c.gameweek)


def _named(pid: int | None) -> str:
    return fpl_data.web_name(pid) if pid is not None else "—"


def _vote_label(label) -> str:
    return "abstain" if label == ABSTAIN else _named(label)


def vote_rag_pick(question: str, gameweek: int, k: int = K_VOTES) -> dict:
    """Run the RAG pick `k` times and majority-vote the result.

    Each run votes for an element_id (confident, resolved pick) or ABSTAIN
    (declined or unresolved). The majority pick is the most-voted player; if
    abstentions equal or outnumber the top player's votes, the GW abstains (the
    model could not decisively commit). Abstention rate is returned as a
    first-class signal, not folded away.
    """
    votes: list = []
    for _ in range(k):
        a = answer(question, gameweek, source=REAL_SOURCE)
        votes.append(a.captain_element_id if (a.captain_confident and a.captain_element_id) else ABSTAIN)

    tally = Counter(votes)
    abstain_count = tally.get(ABSTAIN, 0)
    picks = {label: n for label, n in tally.items() if label != ABSTAIN}
    best_pick = max(picks, key=picks.get) if picks else None
    best_count = picks.get(best_pick, 0) if best_pick is not None else 0

    # abstain wins ties: a non-decisive model means "no pick".
    majority_pid = best_pick if best_count > abstain_count else None
    return {
        "tally": tally,
        "majority_pid": majority_pid,
        "abstain_rate": abstain_count / k,
        "is_strict_majority": best_count > k / 2,
        "k": k,
    }


def run(k: int = K_VOTES) -> None:
    cases = load_cases()
    rows = []

    for case in cases:
        gw = case.gameweek
        vote = vote_rag_pick(case.question, gw, k=k)
        rag_pid = vote["majority_pid"]  # None => majority abstained
        abstained = rag_pid is None
        rag_pts = None if abstained else fpl_data.points(rag_pid, gw)

        tmpl_pid = fpl_data.template_pick(gw)
        tmpl_pts = fpl_data.points(tmpl_pid, gw)

        ceil_pid, ceil_pts = fpl_data.top_scorer(gw)
        floor_pts, floor_n = fpl_data.pool_mean_points(gw)

        diverged = (not abstained) and (rag_pid != tmpl_pid)

        rows.append(
            {
                "case": case,
                "vote": vote,
                "rag_pid": rag_pid,
                "rag_pts": rag_pts,
                "abstained": abstained,
                "tmpl_pid": tmpl_pid,
                "tmpl_pts": tmpl_pts,
                "ceil_pid": ceil_pid,
                "ceil_pts": ceil_pts,
                "floor_pts": floor_pts,
                "floor_n": floor_n,
                "diverged": diverged,
            }
        )

    # ---- per-gameweek detail ----
    print("=" * 74)
    print(f"DECISION-QUALITY EVAL — {len(rows)} gameweeks ({DECISION_GWS}), real corpus, "
          f"majority vote over K={k}")
    print("=" * 74)
    for r in rows:
        c = r["case"]
        tally_str = ", ".join(f"{_vote_label(l)}×{n}" for l, n in r["vote"]["tally"].most_common())
        low_conf = r["vote"]["abstain_rate"] >= LOW_CONF_ABSTAIN
        if r["abstained"]:
            rag_line = f"MAJORITY ABSTAINED  (votes: {tally_str})"
        else:
            flags = []
            if not r["vote"]["is_strict_majority"]:
                flags.append("plurality, not strict majority")
            if low_conf:
                flags.append("LOW CONFIDENCE — high abstention, pick not bankable")
            suffix = f"  [{'; '.join(flags)}]" if flags else ""
            rag_line = f"{_named(r['rag_pid'])} -> {r['rag_pts']} pts  (votes: {tally_str}){suffix}"
        print(f"\nGW{c.gameweek}: {c.question}")
        print(f"  RAG majority  : {rag_line}")
        print(f"  abstention    : {r['vote']['abstain_rate']:.0%} of {k} runs declined to commit")
        print(f"  template pick : {_named(r['tmpl_pid'])}  -> {r['tmpl_pts']} pts")
        print(f"  ceiling (best): {_named(r['ceil_pid'])}  -> {r['ceil_pts']} pts")
        print(f"  random floor  : {r['floor_pts']:.1f} pts (mean over {r['floor_n']} starters)")
        if r["abstained"]:
            verdict = "n/a (majority abstained — no scoreable pick)"
        elif not r["diverged"]:
            verdict = f"AGREES with template (both {_named(r['tmpl_pid'])}) — no marginal credit"
        else:
            delta = r["rag_pts"] - r["tmpl_pts"]
            verdict = f"DIVERGED — RAG {r['rag_pts']} vs template {r['tmpl_pts']}  (delta {delta:+d})"
        print(f"  -> {verdict}")

    # ---- aggregates ----
    scored = [r for r in rows if not r["abstained"]]
    diverged = [r for r in scored if r["diverged"]]
    agreed = [r for r in scored if not r["diverged"]]

    print("\n" + "=" * 74)
    print("AGGREGATE")
    print("=" * 74)
    print(f"  abstention rate (mean): "
          f"{sum(r['vote']['abstain_rate'] for r in rows) / len(rows):.0%} across {len(rows)} GWs")

    if scored:
        rag_total = sum(r["rag_pts"] for r in scored)
        tmpl_total = sum(r["tmpl_pts"] for r in scored)
        ceil_total = sum(r["ceil_pts"] for r in scored)
        floor_total = sum(r["floor_pts"] for r in scored)
        n = len(scored)
        print(f"  scored gameweeks      : {n} (of {len(rows)}; "
              f"{len(rows) - n} majority-abstained)")
        print(f"  total captain points  : RAG {rag_total} | template {tmpl_total} | "
              f"ceiling {ceil_total} | floor {floor_total:.1f}")
        print(f"  mean per GW            : RAG {rag_total/n:.1f} | template {tmpl_total/n:.1f} | "
              f"ceiling {ceil_total/n:.1f} | floor {floor_total/n:.1f}")

        wins = sum(1 for r in scored if r["rag_pts"] > r["tmpl_pts"])
        losses = sum(1 for r in scored if r["rag_pts"] < r["tmpl_pts"])
        ties = sum(1 for r in scored if r["rag_pts"] == r["tmpl_pts"])
        print(f"  RAG vs template record: {wins}W-{losses}L-{ties}T (scored GWs only)")

        print(f"  agreement rate        : {len(agreed)}/{n} scored GWs RAG echoed the template "
              f"({len(agreed)/n:.0%})")
    else:
        print("  no scored gameweeks (every GW majority-abstained)")

    print("-" * 74)
    print("DIVERGENCE-ONLY DIFFERENTIAL (headline — RAG's marginal value)")
    if diverged:
        diff = sum(r["rag_pts"] - r["tmpl_pts"] for r in diverged)
        low_conf_any = False
        for r in diverged:
            lc = r["vote"]["abstain_rate"] >= LOW_CONF_ABSTAIN
            low_conf_any = low_conf_any or lc
            flag = f"  [LOW CONFIDENCE: {r['vote']['abstain_rate']:.0%} abstained]" if lc else ""
            print(f"  GW{r['case'].gameweek}: {_named(r['rag_pid'])} {r['rag_pts']} "
                  f"vs template {_named(r['tmpl_pid'])} {r['tmpl_pts']}  "
                  f"= {r['rag_pts'] - r['tmpl_pts']:+d}{flag}")
        print(f"  net differential over {len(diverged)} divergent GW(s): {diff:+d} pts")
        if low_conf_any:
            print("  NOT BANKABLE: the differential rests on a GW where RAG abstains "
                  "frequently — it materialises only in batches where the model commits.")
    else:
        print("  RAG's majority pick never diverged from the template (or it abstained "
              "where it might have): zero scored divergent GWs, so this slice measures "
              "no marginal RAG value.")

    print("-" * 74)
    print("CAVEAT: n=3. Directional only — no statistical claim. The divergence-only "
          "differential is the honest signal; total points credit RAG for obvious picks.")


if __name__ == "__main__":
    run()
