"""Question -> top-k chunks, with the temporal-integrity filter applied here.

THE invariant of this project lives in this module: a query that decides gameweek
N must never see a chunk dated on/after GW N's deadline. We enforce that with a
Chroma `where` predicate on the `date` metadata at query time (which is why the
store is Chroma and not FAISS).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    text: str
    date: str  # ISO date the source note was authored (pre-deadline)
    gameweek: int


def retrieve(question: str, deadline: str, k: int = 5) -> list[Chunk]:
    """Return the top-k chunks for `question`, restricted to chunks whose `date`
    is strictly before `deadline` (the GW's deadline, ISO date/datetime).

    The temporal filter is non-negotiable: pass it to Chroma as a `where` clause,
    do not post-filter after retrieval (that would let leaked chunks displace
    legitimate ones from the top-k).
    """
    raise NotImplementedError("TODO: embed query + Chroma query with where={date < deadline}")
