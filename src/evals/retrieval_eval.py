"""Retrieval eval: are the right chunks fetched?

Ground truth is `relevant_doc_ids` in golden.jsonl. For each case we retrieve
under the gameweek's temporal cutoff and compare the returned doc ids against the
labelled relevant set (precision@k / recall@k / hit rate).

Also the home of the temporal-integrity ASSERTION: no retrieved chunk may be
dated on/after the case's GW deadline. A leak here is a hard failure, not a score.
"""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"


def load_golden(path: Path = GOLDEN) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run() -> None:
    """Run retrieval over every golden case and report recall/precision@k.

    Asserts temporal integrity per case (no chunk dated >= deadline).
    """
    raise NotImplementedError("TODO: retrieve per case, score vs relevant_doc_ids, assert no leak")


if __name__ == "__main__":
    run()
