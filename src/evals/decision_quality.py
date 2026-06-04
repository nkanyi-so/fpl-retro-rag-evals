"""Decision-quality eval: did RAG-informed captaincy beat the naive baselines,
graded against what actually happened in 2025-26?

This is the project's headline eval. The retrieval and answer-quality evals ask
"did we fetch good chunks / write a grounded answer?". This one asks the only
question that matters for a decision-support system: did the retrieved context
lead to a *better decision*, measured in real captain points?

Strategies, each producing one captain pick per gameweek:
  - rag       : retrieve real (fpl-derived) notes -> Claude picks a captain.
  - template  : the entrenched premium (season-to-date points leader). The brutal
                "just captain the best player every week" default.
  - ceiling   : perfect hindsight — the actual top scorer that GW (upper bound).
  - floor     : expected points of a random pick from the starter pool (lower bound).

Honesty guards baked into the metrics:
  - AGREEMENT RATE: how often RAG just echoes the template. If RAG always picks the
    obvious player, it is not adding value — it is being measured on Haaland.
  - DIVERGENCE-ONLY DIFFERENTIAL (the headline): on the GWs where RAG's pick
    DIFFERS from the template, did RAG's pick outscore the template's? This strips
    out the free credit for agreeing with the obvious pick.
  - Extraction errors (no confident / unresolved pick) are a distinct bucket, never
    scored as 0 football points.

n is tiny (a 3-GW slice). These numbers expose whether the harness works and tell a
directional story; they support NO statistical claim. See the README caveat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make src/ importable
import fpl_data  # noqa: E402
from rag import answer  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
DECISION_GWS = [1, 8, 15]  # the real, temporally-clean slice built this session
REAL_SOURCE = "fpl-derived"


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


def run() -> None:
    cases = load_cases()
    rows = []

    for case in cases:
        gw = case.gameweek
        rag = answer(case.question, gw, source=REAL_SOURCE)

        rag_pid = rag.captain_element_id  # None => extraction error
        extraction_error = rag_pid is None
        rag_pts = None if extraction_error else fpl_data.points(rag_pid, gw)

        tmpl_pid = fpl_data.template_pick(gw)
        tmpl_pts = fpl_data.points(tmpl_pid, gw)

        ceil_pid, ceil_pts = fpl_data.top_scorer(gw)
        floor_pts, floor_n = fpl_data.pool_mean_points(gw)

        diverged = (not extraction_error) and (rag_pid != tmpl_pid)

        rows.append(
            {
                "case": case,
                "rag": rag,
                "rag_pid": rag_pid,
                "rag_pts": rag_pts,
                "extraction_error": extraction_error,
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
    print(f"DECISION-QUALITY EVAL — {len(rows)} gameweeks ({DECISION_GWS}), real corpus")
    print("=" * 74)
    for r in rows:
        c = r["case"]
        print(f"\nGW{c.gameweek}: {c.question}")
        print(f"  context notes : {r['rag'].context_doc_ids}")
        rag_line = (
            "EXTRACTION ERROR (no confident/resolved pick)"
            if r["extraction_error"]
            else f"{_named(r['rag_pid'])}  -> {r['rag_pts']} pts"
        )
        print(f"  RAG pick      : {rag_line}")
        print(f"    confident={r['rag'].captain_confident}  name={r['rag'].captain_name!r}")
        print(f"  template pick : {_named(r['tmpl_pid'])}  -> {r['tmpl_pts']} pts")
        print(f"  ceiling (best): {_named(r['ceil_pid'])}  -> {r['ceil_pts']} pts")
        print(f"  random floor  : {r['floor_pts']:.1f} pts (mean over {r['floor_n']} starters)")
        if r["extraction_error"]:
            verdict = "n/a (extraction error)"
        elif not r["diverged"]:
            verdict = f"AGREES with template (both {_named(r['tmpl_pid'])}) — no marginal credit"
        else:
            delta = r["rag_pts"] - r["tmpl_pts"]
            verdict = f"DIVERGED — RAG {r['rag_pts']} vs template {r['tmpl_pts']}  (delta {delta:+d})"
        print(f"  -> {verdict}")

    # ---- aggregates ----
    scored = [r for r in rows if not r["extraction_error"]]
    diverged = [r for r in scored if r["diverged"]]
    agreed = [r for r in scored if not r["diverged"]]

    print("\n" + "=" * 74)
    print("AGGREGATE")
    print("=" * 74)

    if scored:
        rag_total = sum(r["rag_pts"] for r in scored)
        tmpl_total = sum(r["tmpl_pts"] for r in scored)
        ceil_total = sum(r["ceil_pts"] for r in scored)
        floor_total = sum(r["floor_pts"] for r in scored)
        n = len(scored)
        print(f"  scored gameweeks      : {n} (of {len(rows)}; "
              f"{len(rows) - n} extraction error)")
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
        print("  no scored gameweeks (all extraction errors)")

    print("-" * 74)
    print("DIVERGENCE-ONLY DIFFERENTIAL (headline — RAG's marginal value)")
    if diverged:
        diff = sum(r["rag_pts"] - r["tmpl_pts"] for r in diverged)
        for r in diverged:
            print(f"  GW{r['case'].gameweek}: {_named(r['rag_pid'])} {r['rag_pts']} "
                  f"vs template {_named(r['tmpl_pid'])} {r['tmpl_pts']}  "
                  f"= {r['rag_pts'] - r['tmpl_pts']:+d}")
        print(f"  net differential over {len(diverged)} divergent GW(s): {diff:+d} pts")
    else:
        print("  RAG never diverged from the template — zero divergent GWs, so this "
              "slice measures no marginal RAG value (it only confirms agreement).")

    print("-" * 74)
    print("CAVEAT: n=3. Directional only — no statistical claim. The divergence-only "
          "differential is the honest signal; total points credit RAG for obvious picks.")


if __name__ == "__main__":
    run()
