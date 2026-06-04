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

import fpl_data
from ingest import ROOT
from retrieve import retrieve

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY

ANSWER_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are an FPL captaincy assistant. Answer ONLY using the gameweek notes "
    "provided in the user message. Do not use any outside Premier League knowledge, "
    "player form, fixtures, or results you may recall — only what the notes state. "
    "If the notes do not contain enough information to recommend a captain, say so "
    "explicitly instead of guessing.\n\n"
    "Return your response with these fields:\n"
    "- answer: your captaincy reasoning, grounded in the specific note(s) you used.\n"
    "- captain_name: the single player you recommend captaining, written exactly as "
    "the notes name them (e.g. 'Erling Haaland'). If the notes do not let you make a "
    "confident pick, leave this null.\n"
    "- confident: true only if you can confidently ground a single captain pick in "
    "the notes; false if the notes are insufficient and you are declining to pick."
)


class RagOutput(BaseModel):
    """Structured model output: prose + a single captain pick (or a declared
    non-pick). Validated by the Anthropic parse call."""

    answer: str
    captain_name: str | None = None
    confident: bool


class RagAnswer(BaseModel):
    answer: str
    context_doc_ids: list[str]
    context_text: str
    # structured captain pick for the decision-quality eval
    captain_name: str | None
    captain_confident: bool
    captain_element_id: int | None  # resolved from captain_name; None if unresolved


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['doc_id']}]\n{c['text']}" for c in chunks)


def answer(question: str, gameweek: int, k: int = 5, source: str | None = None) -> RagAnswer:
    """Answer a captaincy question for `gameweek`, grounded in retrieved notes.

    Retrieval always runs with the temporal filter on, so the model never sees a
    note dated on/after the gameweek's deadline. When `source` is given, context
    is further restricted to notes of that source (the decision-quality eval
    passes `source='fpl-derived'` to guarantee real, clean context).

    The captain pick is returned structured: `captain_name` as written in the
    notes, `captain_confident` (False => the model declined to pick), and
    `captain_element_id` resolved from the name (None => unresolved). A non-pick
    or an unresolved name is the eval's extraction-error signal, never a 0-point
    football outcome.
    """
    chunks = retrieve(question, gameweek, k=k, apply_temporal_filter=True, source=source)
    context_text = _format_context(chunks)

    user_message = (
        f"Gameweek: {gameweek}\n"
        f"Question: {question}\n\n"
        f"Gameweek notes available before the GW{gameweek} deadline:\n\n"
        f"{context_text if context_text else '(no notes available)'}\n\n"
        "Using only these notes, who should be captained and why?"
    )

    client = Anthropic()
    response = client.messages.parse(
        model=ANSWER_MODEL,
        max_tokens=1024,
        temperature=0,  # reproducible eval runs; the pick must not drift run-to-run
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_format=RagOutput,
    )
    out = response.parsed_output

    element_id = (
        fpl_data.resolve_player(out.captain_name)
        if (out.confident and out.captain_name)
        else None
    )

    return RagAnswer(
        answer=out.answer,
        context_doc_ids=[c["doc_id"] for c in chunks],
        context_text=context_text,
        captain_name=out.captain_name,
        captain_confident=out.confident,
        captain_element_id=element_id,
    )
