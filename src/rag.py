"""Retrieve + augment + answer (Anthropic).

Ties the pipeline together: get temporally-valid chunks for a gameweek, build a
context-grounded prompt, and ask Claude for a captaincy recommendation.

The product path ALWAYS applies the temporal filter — a recommendation for GW N
may only be built from notes available before GW N's deadline. Grounding is forced
in the system prompt: the model must answer only from the retrieved notes and say
so explicitly when they are insufficient, rather than falling back on its own
Premier League knowledge.
"""

from __future__ import annotations

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from ingest import ROOT
from retrieve import retrieve

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY

ANSWER_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are an FPL captaincy assistant. Answer ONLY using the gameweek notes "
    "provided in the user message. Do not use any outside Premier League knowledge, "
    "player form, fixtures, or results you may recall — only what the notes state. "
    "If the notes do not contain enough information to recommend a captain, say so "
    "explicitly (for example: 'The provided notes do not contain enough information "
    "to decide.') instead of guessing. When you do recommend a captain, ground the "
    "recommendation in the specific note(s) you used."
)


class RagAnswer(BaseModel):
    answer: str
    context_doc_ids: list[str]
    context_text: str


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['doc_id']}]\n{c['text']}" for c in chunks)


def answer(question: str, gameweek: int, k: int = 5) -> RagAnswer:
    """Answer a captaincy question for `gameweek`, grounded in retrieved notes.

    Retrieval always runs with the temporal filter on, so the model never sees a
    note dated on/after the gameweek's deadline.
    """
    chunks = retrieve(question, gameweek, k=k, apply_temporal_filter=True)
    context_text = _format_context(chunks)

    user_message = (
        f"Gameweek: {gameweek}\n"
        f"Question: {question}\n\n"
        f"Gameweek notes available before the GW{gameweek} deadline:\n\n"
        f"{context_text if context_text else '(no notes available)'}\n\n"
        "Using only these notes, who should be captained and why?"
    )

    client = Anthropic()
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")

    return RagAnswer(
        answer=text,
        context_doc_ids=[c["doc_id"] for c in chunks],
        context_text=context_text,
    )
