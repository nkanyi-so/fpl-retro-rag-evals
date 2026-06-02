"""Retrieval eval: are the right chunks fetched?

Ground truth is `relevant_doc_ids` in golden.jsonl. For each case we retrieve at
k=5 and score hit@k (>=1 relevant id retrieved) and recall@k (fraction of relevant
ids retrieved), aggregated across cases.

The eval runs TWICE — temporal filter OFF, then ON — and prints a side-by-side
comparison so the cost of hindsight leakage is visible. With the filter ON there
is a HARD ASSERTION: no retrieved note may be dated on/after the case's gameweek
deadline. A leak there is a hard failure (with the offending doc_id), not a score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make src/ importable
from retrieve import deadline_int, retrieve  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
TRAP_ID = "gw06-captaincy-preview"  # later-GW note that the GW1 question matches
K = 5


class GoldenCase(BaseModel):
    question: str
    gameweek: int
    relevant_doc_ids: list[str]
    reference_answer: str | None = None


def load_golden(path: Path = GOLDEN) -> list[GoldenCase]:
    cases = []
    for line in path.read_text().splitlines():
        if line.strip():
            cases.append(GoldenCase(**json.loads(line)))
    return cases


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    return len(set(retrieved_ids) & relevant) / len(relevant)


def _fmt(hits: list[dict]) -> str:
    return ", ".join(f"{h['doc_id']}({h['score']:.2f})" for h in hits) or "(none)"


def run() -> None:
    cases = load_golden()
    rows = []
    violations = []  # (gameweek, doc_id, date_int, deadline_int)

    for case in cases:
        off = retrieve(case.question, case.gameweek, k=K, apply_temporal_filter=False)
        on = retrieve(case.question, case.gameweek, k=K, apply_temporal_filter=True)

        # HARD temporal-integrity check on the filtered run.
        dl = deadline_int(case.gameweek)
        for h in on:
            if h["date_int"] >= dl:
                violations.append((case.gameweek, h["doc_id"], h["date_int"], dl))

        rows.append(
            {
                "case": case,
                "off": off,
                "on": on,
                "off_ids": [h["doc_id"] for h in off],
                "on_ids": [h["doc_id"] for h in on],
            }
        )

    def agg(metric, mode):
        return sum(
            metric(r[f"{mode}_ids"], r["case"].relevant_doc_ids) for r in rows
        ) / len(rows)

    # ---- aggregate comparison ----
    print("=" * 72)
    print(f"RETRIEVAL EVAL — {len(rows)} cases, k={K}")
    print("=" * 72)
    print(f"{'mode':<14}{'hit@k':>10}{'recall@k':>12}")
    print("-" * 36)
    for mode, label in (("off", "filter OFF"), ("on", "filter ON")):
        print(f"{label:<14}{agg(hit_at_k, mode):>10.3f}{agg(recall_at_k, mode):>12.3f}")
    print()

    # ---- per-case table ----
    print("PER-CASE retrieved doc_ids (score)")
    print("-" * 72)
    for r in rows:
        c = r["case"]
        print(f"GW{c.gameweek}: {c.question}")
        print(f"  relevant   : {c.relevant_doc_ids}")
        print(
            f"  filter OFF : {_fmt(r['off'])}"
            f"   [hit={hit_at_k(r['off_ids'], c.relevant_doc_ids):.0f}"
            f" recall={recall_at_k(r['off_ids'], c.relevant_doc_ids):.2f}]"
        )
        print(
            f"  filter ON  : {_fmt(r['on'])}"
            f"   [hit={hit_at_k(r['on_ids'], c.relevant_doc_ids):.0f}"
            f" recall={recall_at_k(r['on_ids'], c.relevant_doc_ids):.2f}]"
        )
        print()

    # ---- GW1 temporal-trap spotlight ----
    gw1 = next((r for r in rows if r["case"].gameweek == 1), None)
    print("TEMPORAL-TRAP SPOTLIGHT (GW1)")
    print("-" * 72)
    if gw1 is None:
        print("  no GW1 case in golden set")
    else:
        in_off = TRAP_ID in gw1["off_ids"]
        in_on = TRAP_ID in gw1["on_ids"]
        print(f"  question: {gw1['case'].question}")
        print(f"  {TRAP_ID} dated after GW1 deadline ({deadline_int(1)}).")
        print(f"    retrieved with filter OFF? {'YES' if in_off else 'no'}"
              f"  (leaks in without the filter)" if in_off else
              f"    retrieved with filter OFF? no")
        print(f"    retrieved with filter ON ? {'YES — LEAK!' if in_on else 'no (correctly excluded)'}")
    print()

    # ---- hard assertion ----
    print("TEMPORAL-INTEGRITY ASSERTION (filter ON)")
    print("-" * 72)
    if violations:
        for gw, doc_id, di, dl in violations:
            print(f"  VIOLATION: GW{gw} retrieved {doc_id} dated {di} >= deadline {dl}")
        raise AssertionError(
            f"temporal integrity violated in {len(violations)} case(s); "
            f"first offender: {violations[0][1]}"
        )
    print("  PASS — no retrieved note dated on/after its case deadline.")


if __name__ == "__main__":
    run()
