"""Question -> top-k chunks, with the temporal-integrity filter applied here.

THE invariant of this project lives in this module: a query that decides gameweek
N must never see a chunk dated on/after GW N's deadline. We enforce that with a
Chroma `where` predicate on the `date_int` metadata at query time (which is why
the store is Chroma and not FAISS).

The gameweek -> deadline map is an explicit reference file (`data/deadlines.json`),
consistent with the corpus's synthetic calendar, so the cutoff is data, not magic
numbers buried in code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ingest import ROOT, get_collection, get_model

DEADLINES_PATH = ROOT / "data" / "deadlines.json"


@lru_cache(maxsize=1)
def _deadlines() -> dict[str, str]:
    return json.loads(DEADLINES_PATH.read_text())


def deadline_int(gameweek: int) -> int:
    """Return GW `gameweek`'s deadline as a YYYYMMDD int."""
    deadlines = _deadlines()
    key = str(gameweek)
    if key not in deadlines:
        raise KeyError(f"no deadline for GW{gameweek} in {DEADLINES_PATH}")
    return int(deadlines[key].replace("-", ""))


def retrieve(
    question: str,
    gameweek: int,
    k: int = 5,
    apply_temporal_filter: bool = True,
) -> list[dict]:
    """Return up to `k` chunks for `question`, ranked by similarity.

    When `apply_temporal_filter` is True, only chunks dated strictly before GW
    `gameweek`'s deadline are eligible (Chroma `where`: date_int < deadline). When
    False, no temporal filter is applied (used to show what leaks in without it).

    Each result: {doc_id, score, date_int, text}. `score` is cosine similarity
    (1 - distance), higher is closer.
    """
    model = get_model()
    collection = get_collection()
    query_embedding = model.encode(question, normalize_embeddings=True).tolist()

    where = None
    if apply_temporal_filter:
        where = {"date_int": {"$lt": deadline_int(gameweek)}}

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["distances", "metadatas", "documents"],
    )

    hits: list[dict] = []
    for i, doc_id in enumerate(result["ids"][0]):
        hits.append(
            {
                "doc_id": doc_id,
                "score": round(1.0 - result["distances"][0][i], 4),
                "date_int": result["metadatas"][0][i]["date_int"],
                "text": result["documents"][0][i],
            }
        )
    return hits
