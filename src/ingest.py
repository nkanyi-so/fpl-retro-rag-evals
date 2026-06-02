"""Chunk + embed + store gameweek notes into Chroma.

Each chunk carries a `date` metadata field (ISO date the note was authored,
pre-deadline). That field is what `retrieve.py` filters on to enforce temporal
integrity, so it MUST be populated at ingest time.
"""

from __future__ import annotations

from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
COLLECTION_NAME = "fpl_gw_notes"


def ingest(corpus_dir: Path = CORPUS_DIR) -> int:
    """Read .md notes, chunk, embed locally, and store in Chroma.

    Returns the number of chunks written. Each stored chunk must include at
    least: `doc_id`, `date` (ISO, pre-deadline), and `gameweek`.
    """
    raise NotImplementedError("TODO: implement chunk + embed + store")


if __name__ == "__main__":
    print(f"ingested {ingest()} chunks")
