"""Decision-quality eval: did RAG-informed captaincy beat the naive baselines,
graded against what actually happened in 2025-26?

This is the project's headline eval. The retrieval and answer-quality evals ask
"did we fetch good chunks / write a grounded answer?". This one asks the only
question that matters for a decision-support system: did the retrieved context
lead to a *better decision*, measured in real captain points?

Strategies, each producing one captain pick per gameweek:
  - rag       : retrieve real (fpl-derived) notes -> Claude returns decision support
                AND a single FORCED pick (+confidence). We grade the forced pick.
  - template  : the most-captained player that GW (the crowd's actual armband,
                from the FPL API). The brutal "just captain what everyone else
                captains" default.
  - ceiling   : perfect hindsight — the actual top scorer that GW (upper bound).
  - floor     : expected points of a random pick from the starter pool (lower bound).

Reliability — MAJORITY VOTING + CONFIDENCE. The RAG output is not deterministic even
at temperature=0, so a single pass is an unreliable measurement. The system no longer
abstains: it always emits a forced pick plus a 0–1 confidence. For each GW we run it
K times (default 5) and take the MAJORITY-vote forced pick; the per-GW AVERAGE
CONFIDENCE and the vote share are recorded as data (confidence is never used to drop
a pick). Forcing the pick is what makes even close calls land on a stable captain.
The baselines (template/ceiling/floor) are deterministic, so they are not voted.

Honesty guards baked into the metrics:
  - AGREEMENT RATE: how often RAG's majority pick just echoes the template. If RAG
    always picks the obvious player, it is not adding value — it is being measured on
    the crowd's pick.
  - DIVERGENCE-ONLY DIFFERENTIAL (the headline): on the GWs where RAG's majority pick
    DIFFERS from the template, did it outscore the template? This strips out the free
    credit for agreeing with the obvious pick.
  - CONFIDENCE-ANNOTATED: each divergent result carries the model's average
    confidence. A win on a low-confidence divergence is weaker evidence than a win on
    a high-confidence one.

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
K_VOTES = 5  # RAG runs per GW; majority vote stabilises the nondeterministic pick
LOW_CONFIDENCE = 0.50  # avg confidence below this => a divergence win is weaker evidence


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


def _conf_label(c: float) -> str:
    return "high" if c >= 0.8 else "medium" if c >= LOW_CONFIDENCE else "low"


def vote_rag_pick(question: str, gameweek: int, k: int = K_VOTES) -> dict:
    """Run the forced RAG pick `k` times and majority-vote it.

    The system never abstains: every run yields a forced pick (+confidence). We
    majority-vote the forced pick (ties broken by lowest element_id for
    determinism) and record the per-GW average confidence and vote share as data.
    A forced pick whose name does not resolve to an element_id is dropped from the
    vote (a rare resolution error, not an abstention).
    """
    votes: list[int] = []
    confidences: list[float] = []
    close_flags: list[bool] = []
    for _ in range(k):
        a = answer(question, gameweek, source=REAL_SOURCE)
        d = a.decision
        confidences.append(d.confidence)
        close_flags.append(d.is_close)
        if d.forced_pick_element_id is not None:
            votes.append(d.forced_pick_element_id)

    tally = Counter(votes)
    # majority pick; tie-break on lowest element_id so the result is deterministic.
    best_count = max(tally.values()) if tally else 0
    majority_pid = min((p for p, n in tally.items() if n == best_count), default=None)
    return {
        "tally": tally,
        "majority_pid": majority_pid,
        "vote_share": best_count / k if k else 0.0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "is_close_rate": sum(close_flags) / len(close_flags) if close_flags else 0.0,
        "k": k,
    }


def run(k: int = K_VOTES) -> None:
    cases = load_cases()
    rows = []

    for case in cases:
        gw = case.gameweek
        vote = vote_rag_pick(case.question, gw, k=k)
        rag_pid = vote["majority_pid"]
        unresolved = rag_pid is None  # forced name never resolved (rare)
        rag_pts = None if unresolved else fpl_data.points(rag_pid, gw)

        tmpl_pid = fpl_data.template_pick(gw)
        tmpl_pts = fpl_data.points(tmpl_pid, gw)

        ceil_pid, ceil_pts = fpl_data.top_scorer(gw)
        floor_pts, floor_n = fpl_data.pool_mean_points(gw)

        diverged = (not unresolved) and (rag_pid != tmpl_pid)

        rows.append(
            {
                "case": case,
                "vote": vote,
                "rag_pid": rag_pid,
                "rag_pts": rag_pts,
                "unresolved": unresolved,
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
          f"forced pick, majority vote over K={k}")
    print("=" * 74)
    for r in rows:
        c = r["case"]
        v = r["vote"]
        tally_str = ", ".join(f"{_named(p)}×{n}" for p, n in v["tally"].most_common())
        conf = v["avg_confidence"]
        print(f"\nGW{c.gameweek}: {c.question}")
        if r["unresolved"]:
            print(f"  RAG forced    : UNRESOLVED (forced name never matched a player; votes: {tally_str})")
        else:
            print(f"  RAG forced    : {_named(r['rag_pid'])} -> {r['rag_pts']} pts  "
                  f"(votes: {tally_str}; vote share {v['vote_share']:.0%})")
        print(f"  avg confidence: {conf:.2f} ({_conf_label(conf)})  | "
              f"flagged close in {v['is_close_rate']:.0%} of runs")
        print(f"  template pick : {_named(r['tmpl_pid'])}  -> {r['tmpl_pts']} pts")
        print(f"  ceiling (best): {_named(r['ceil_pid'])}  -> {r['ceil_pts']} pts")
        print(f"  random floor  : {r['floor_pts']:.1f} pts (mean over {r['floor_n']} starters)")
        if r["unresolved"]:
            verdict = "n/a (forced pick unresolved)"
        elif not r["diverged"]:
            verdict = f"AGREES with template (both {_named(r['tmpl_pid'])}) — no marginal credit"
        else:
            delta = r["rag_pts"] - r["tmpl_pts"]
            verdict = (f"DIVERGED — RAG {r['rag_pts']} vs template {r['tmpl_pts']} "
                       f"(delta {delta:+d}), {_conf_label(conf)} confidence")
        print(f"  -> {verdict}")

    # ---- aggregates ----
    scored = [r for r in rows if not r["unresolved"]]
    diverged = [r for r in scored if r["diverged"]]
    agreed = [r for r in scored if not r["diverged"]]

    print("\n" + "=" * 74)
    print("AGGREGATE")
    print("=" * 74)
    print(f"  mean confidence       : "
          f"{sum(r['vote']['avg_confidence'] for r in rows) / len(rows):.2f} across {len(rows)} GWs")

    if scored:
        rag_total = sum(r["rag_pts"] for r in scored)
        tmpl_total = sum(r["tmpl_pts"] for r in scored)
        ceil_total = sum(r["ceil_pts"] for r in scored)
        floor_total = sum(r["floor_pts"] for r in scored)
        n = len(scored)
        print(f"  scored gameweeks      : {n} (of {len(rows)})")
        print(f"  total captain points  : RAG {rag_total} | template {tmpl_total} | "
              f"ceiling {ceil_total} | floor {floor_total:.1f}")
        print(f"  mean per GW            : RAG {rag_total/n:.1f} | template {tmpl_total/n:.1f} | "
              f"ceiling {ceil_total/n:.1f} | floor {floor_total/n:.1f}")

        wins = sum(1 for r in scored if r["rag_pts"] > r["tmpl_pts"])
        losses = sum(1 for r in scored if r["rag_pts"] < r["tmpl_pts"])
        ties = sum(1 for r in scored if r["rag_pts"] == r["tmpl_pts"])
        print(f"  RAG vs template record: {wins}W-{losses}L-{ties}T")

        print(f"  agreement rate        : {len(agreed)}/{n} GWs RAG echoed the template "
              f"({len(agreed)/n:.0%})")
    else:
        print("  no scored gameweeks")

    print("-" * 74)
    print("DIVERGENCE-ONLY DIFFERENTIAL (headline — RAG's marginal value, confidence-weighted)")
    if diverged:
        diff = sum(r["rag_pts"] - r["tmpl_pts"] for r in diverged)
        for r in diverged:
            conf = r["vote"]["avg_confidence"]
            print(f"  GW{r['case'].gameweek}: {_named(r['rag_pid'])} {r['rag_pts']} "
                  f"vs template {_named(r['tmpl_pid'])} {r['tmpl_pts']}  "
                  f"= {r['rag_pts'] - r['tmpl_pts']:+d}  [{_conf_label(conf)} confidence {conf:.2f}]")
        print(f"  net differential over {len(diverged)} divergent GW(s): {diff:+d} pts")
        if any(r["vote"]["avg_confidence"] < LOW_CONFIDENCE for r in diverged):
            print("  NOTE: a divergence won at LOW confidence is weaker evidence — the model "
                  "flagged it a close call. Read the differential alongside the confidence.")
    else:
        print("  RAG's majority pick never diverged from the template — zero divergent GWs, "
              "so this slice measures no marginal RAG value (only agreement).")

    print("-" * 74)
    print("CAVEAT: n=3. Directional only — no statistical claim. The divergence-only "
          "differential is the honest signal; total points credit RAG for obvious picks.")


if __name__ == "__main__":
    run()
