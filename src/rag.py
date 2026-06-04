"""Retrieve + augment + answer (Anthropic) — captaincy DECISION SUPPORT.

Ties the pipeline together: get temporally-valid chunks for a gameweek, build a
context-grounded prompt, and ask Claude for a captaincy recommendation.

The product path ALWAYS applies the temporal filter — a recommendation for GW N
may only be built from notes available before GW N's deadline. Grounding is forced
in the system prompt: the model answers only from the retrieved notes.

Two layers come back in one structured `RagDecision` (JSON-serializable, so a UI
can render it without re-parsing):
  1. USER-FACING decision support — the top candidates, the grounded case for each,
     whether it is a close call, and an explicit confidence. Close calls are NOT
     collapsed into a bare pick or a refusal: "it's close, here's the lean and why".
  2. EVAL-FACING forced pick — a single "if forced to commit" captain (name +
     resolved element_id) plus a confidence score. The eval always has something to
     grade; confidence is recorded as data, never used to abstain.
"""

from __future__ import annotations

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import fpl_data
from ingest import ROOT
from retrieve import retrieve

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY

ANSWER_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are an FPL captaincy decision-support assistant. Answer ONLY using the "
    "gameweek notes provided in the user message. Do not use any outside Premier "
    "League knowledge, player form, fixtures, or results you may recall — only what "
    "the notes state.\n\n"
    "Give decision support, not a bare pick and not a refusal. Return these fields:\n"
    "- candidates: the top captain options the notes support (1 to 3), best first. "
    "For each, give `name` exactly as the notes write it (e.g. 'Erling Haaland') and "
    "`case`: the argument for captaining them, grounded in the specific note(s).\n"
    "- is_close: true if it is a genuinely close call between the top candidates.\n"
    "- confidence: a 0.0–1.0 score for the forced pick below. Low (toss-up between "
    "candidates, or thin notes), high (one option clearly best).\n"
    "- forced_pick_name: if you HAD to commit one captain, exactly one player name "
    "(must be one of the candidates). ALWAYS provide this — even on a close call, "
    "never decline.\n"
    "- rationale: a brief overall summary stating the lean and why.\n"
    "If the notes are thin, reflect that in low confidence and say so in the cases — "
    "but still name a forced pick."
)


class _CandidateOut(BaseModel):
    """One candidate as emitted by the model (names only; ids resolved in code)."""

    name: str
    case: str


class RagModelOutput(BaseModel):
    """Raw structured output from the model. Validated by the Anthropic parse call."""

    candidates: list[_CandidateOut] = Field(min_length=1)
    is_close: bool
    confidence: float = Field(ge=0.0, le=1.0)
    forced_pick_name: str
    rationale: str


class Candidate(BaseModel):
    name: str
    element_id: int | None  # resolved from name; None if unresolved
    case: str


class RagDecision(BaseModel):
    """The decision-support answer AND the eval's source of truth. JSON-serializable
    so a future UI can render it directly."""

    candidates: list[Candidate]
    is_close: bool
    confidence: float
    forced_pick_name: str
    forced_pick_element_id: int | None  # resolved; None only if name unresolved
    rationale: str

    @property
    def confidence_label(self) -> str:
        return "high" if self.confidence >= 0.8 else "medium" if self.confidence >= 0.5 else "low"

    def to_prose(self) -> str:
        """Readable rendering of the structured decision (for the CLI / judge)."""
        close = " — close call" if self.is_close else ""
        lines = [
            f"Lean: {self.forced_pick_name} "
            f"(confidence: {self.confidence_label}, {self.confidence:.2f}){close}",
            self.rationale,
            "",
            "Candidates:",
        ]
        for i, c in enumerate(self.candidates, 1):
            lines.append(f"  {i}. {c.name}: {c.case}")
        return "\n".join(lines)


class RagAnswer(BaseModel):
    decision: RagDecision  # structured source of truth (the product answer)
    answer: str  # human-readable rendering of `decision` (for the judge / CLI)
    context_doc_ids: list[str]
    context_text: str


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['doc_id']}]\n{c['text']}" for c in chunks)


def answer(question: str, gameweek: int, k: int = 5, source: str | None = None) -> RagAnswer:
    """Answer a captaincy question for `gameweek`, grounded in retrieved notes.

    Retrieval always runs with the temporal filter on, so the model never sees a
    note dated on/after the gameweek's deadline. When `source` is given, context
    is further restricted to notes of that source (the decision-quality eval
    passes `source='fpl-derived'` to guarantee real, clean context).

    Returns a `RagAnswer` whose `decision` is the structured decision-support
    object: candidates with grounded cases, a close-call flag, a confidence score,
    and an always-present forced pick (`forced_pick_element_id` resolved from the
    name). The eval grades the forced pick and records the confidence as data.
    """
    chunks = retrieve(question, gameweek, k=k, apply_temporal_filter=True, source=source)
    context_text = _format_context(chunks)

    user_message = (
        f"Gameweek: {gameweek}\n"
        f"Question: {question}\n\n"
        f"Gameweek notes available before the GW{gameweek} deadline:\n\n"
        f"{context_text if context_text else '(no notes available)'}\n\n"
        "Using only these notes, give your captaincy decision support."
    )

    client = Anthropic()
    response = client.messages.parse(
        model=ANSWER_MODEL,
        max_tokens=1024,
        temperature=0,  # reduce (does not eliminate) run-to-run drift
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_format=RagModelOutput,
    )
    out = response.parsed_output

    decision = RagDecision(
        candidates=[
            Candidate(name=c.name, element_id=fpl_data.resolve_player(c.name), case=c.case)
            for c in out.candidates
        ],
        is_close=out.is_close,
        confidence=out.confidence,
        forced_pick_name=out.forced_pick_name,
        forced_pick_element_id=fpl_data.resolve_player(out.forced_pick_name),
        rationale=out.rationale,
    )

    return RagAnswer(
        decision=decision,
        answer=decision.to_prose(),
        context_doc_ids=[c["doc_id"] for c in chunks],
        context_text=context_text,
    )
