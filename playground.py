#!/usr/bin/env python
"""
playground.py — manual, interactive exploration of the FPL retro-RAG.

Dev tool only; NOT part of the pipeline. Run it from the repo root.

USAGE
  # Default: retrieve (temporal filter ON) and generate an answer
  python playground.py "Who should I captain in GW8?" --gw 8 --k 5

  # Compare retrieval with the temporal filter OFF vs ON, side by side,
  # so the no-hindsight effect is visible
  python playground.py "Who should I captain in GW8?" --gw 8 --compare

  # Dump everything stored in the Chroma collection (no question needed)
  python playground.py --list

PREREQS
  - Build the store first:  python src/ingest.py
  - ANTHROPIC_API_KEY in .env (only the answer step calls the API; --list and
    plain retrieval do not).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Match the project's import style: put src/ on the path, then import flat.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingest import get_collection  # noqa: E402
from rag import answer  # noqa: E402
from retrieve import deadline_int, retrieve  # noqa: E402

RULE = "=" * 72
THIN = "-" * 72


def _fmt_date(date_int: int) -> str:
    s = str(date_int)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def _collection_map() -> dict:
    """doc_id -> {gameweek, date_int, type, text} for everything in the store."""
    got = get_collection().get(include=["metadatas", "documents"])
    out = {}
    for i, doc_id in enumerate(got["ids"]):
        md = got["metadatas"][i]
        out[doc_id] = {
            "gameweek": md["gameweek"],
            "date_int": md["date_int"],
            "type": md["type"],
            "text": got["documents"][i],
        }
    return out


def _print_retrieved(question, gameweek, k, apply_filter, meta) -> None:
    label = "ON" if apply_filter else "OFF"
    hits = retrieve(question, gameweek, k=k, apply_temporal_filter=apply_filter)
    print(f"RETRIEVED NOTES (temporal filter {label})")
    print("score = cosine similarity (1 - distance); HIGHER means closer.")
    if apply_filter:
        print(f"filter: only notes dated before the GW{gameweek} deadline "
              f"({_fmt_date(deadline_int(gameweek))}).")
    print(THIN)
    if not hits:
        print("  (no notes retrieved)")
    for rank, h in enumerate(hits, 1):
        typ = meta.get(h["doc_id"], {}).get("type", "?")
        snippet = " ".join(h["text"].split())[:200]
        print(f"  {rank}. {h['doc_id']:<28} score={h['score']:.4f}  "
              f"date={_fmt_date(h['date_int'])}  type={typ}")
        print(f"     {snippet}")
    print()


def run_query(question: str, gameweek: int, k: int, compare: bool) -> None:
    meta = _collection_map()

    print(RULE)
    print("THE QUESTION")
    print(RULE)
    print(f"  question: {question}")
    print(f"  gameweek: {gameweek}")
    print()

    if compare:
        _print_retrieved(question, gameweek, k, False, meta)
        _print_retrieved(question, gameweek, k, True, meta)
    else:
        _print_retrieved(question, gameweek, k, True, meta)

    print(RULE)
    print("THE ANSWER")
    print(RULE)
    result = answer(question, gameweek, k=k)
    print(result.answer.strip())
    print()
    print(f"context_doc_ids given to the model: {result.context_doc_ids}")


def run_list() -> None:
    meta = _collection_map()
    print(RULE)
    print(f"CHROMA COLLECTION — {len(meta)} documents")
    print(RULE)
    print(f"  {'doc_id':<28} {'gw':>3}  {'date':<12} type")
    print(THIN)
    for doc_id in sorted(meta, key=lambda d: (meta[d]["gameweek"], d)):
        m = meta[doc_id]
        print(f"  {doc_id:<28} {m['gameweek']:>3}  "
              f"{_fmt_date(m['date_int']):<12} {m['type']}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive RAG playground (dev tool, not part of the pipeline)."
    )
    p.add_argument("question", nargs="?", help="captaincy question, e.g. \"Who should I captain in GW8?\"")
    p.add_argument("--gw", type=int, help="gameweek number for the question")
    p.add_argument("--k", type=int, default=5, help="top-k chunks to retrieve (default 5)")
    p.add_argument("--compare", action="store_true",
                   help="print retrieval with the temporal filter OFF then ON")
    p.add_argument("--list", action="store_true", dest="list_",
                   help="dump every document in the Chroma collection and exit")
    args = p.parse_args()

    if args.list_:
        run_list()
        return
    if not args.question or args.gw is None:
        p.error("provide a question and --gw, or use --list")
    run_query(args.question, args.gw, args.k, args.compare)


if __name__ == "__main__":
    main()
