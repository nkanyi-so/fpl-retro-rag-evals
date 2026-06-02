"""Chunk + embed + store gameweek notes into a persistent Chroma collection.

One note = one chunk (no splitting). Frontmatter is parsed and validated with a
pydantic model — anything invalid or missing RAISES rather than being skipped, so
a malformed note can never silently drop out of the index.

Each chunk carries a `date_int` (YYYYMMDD as int) in its metadata. That field is
what `retrieve.py` range-filters on to enforce temporal integrity, so it MUST be
populated at ingest time.
"""

from __future__ import annotations

from datetime import date as DateType
from functools import lru_cache
from pathlib import Path

import chromadb
import yaml
from chromadb.config import Settings
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "fpl_gw_notes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

ALLOWED_TYPES = {"form", "injury", "fixture", "tactical"}


class NoteFrontmatter(BaseModel):
    """Validated frontmatter for one corpus note."""

    doc_id: str
    gameweek: int
    date: DateType  # pydantic accepts an ISO string or a yaml date object
    source: str
    team: str
    players: list[str]
    type: str

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in ALLOWED_TYPES:
            raise ValueError(f"type {v!r} not in {sorted(ALLOWED_TYPES)}")
        return v

    @property
    def date_int(self) -> int:
        return int(self.date.strftime("%Y%m%d"))


def parse_note(path: Path) -> tuple[NoteFrontmatter, str]:
    """Parse a single `.md` note into (validated frontmatter, body).

    Raises if the file lacks a `--- ... ---` frontmatter block, if the
    frontmatter fails validation, or if the declared doc_id does not match the
    filename. No silent skipping.
    """
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing frontmatter block")
    _, fm_raw, body = text.split("---", 2)
    data = yaml.safe_load(fm_raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: frontmatter did not parse to a mapping")
    fm = NoteFrontmatter(**data)  # raises pydantic.ValidationError on bad/missing fields
    if fm.doc_id != path.stem:
        raise ValueError(f"{path.name}: doc_id {fm.doc_id!r} != filename {path.stem!r}")
    return fm, body.strip()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the shared embedding model once (reused by retrieve.py)."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def _client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection() -> chromadb.api.models.Collection.Collection:
    """Return the existing persistent collection (read path for retrieve.py)."""
    return _client().get_collection(COLLECTION_NAME)


def ingest(corpus_dir: Path = CORPUS_DIR) -> int:
    """Read every `.md` note, embed the body, and (re)build the Chroma collection.

    Returns the number of chunks written. Idempotent: the collection is dropped
    and rebuilt each run so re-ingesting never duplicates.
    """
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md notes found in {corpus_dir}")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for path in paths:
        fm, body = parse_note(path)
        ids.append(fm.doc_id)
        documents.append(body)
        metadatas.append(
            {
                "doc_id": fm.doc_id,
                "gameweek": fm.gameweek,
                "type": fm.type,
                "date_int": fm.date_int,
            }
        )

    model = get_model()
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(ids)


if __name__ == "__main__":
    n = ingest()
    print(f"ingested {n} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
