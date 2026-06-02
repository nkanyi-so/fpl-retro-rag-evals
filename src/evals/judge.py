"""LLM-as-judge for answer quality.

Grades the RAG's captaincy answer against the optional `reference_answer` in
golden.jsonl (and the retrieved context). This is answer-quality scoring, NOT the
outcome-grounded decision-quality eval — that lands next session.
"""

from __future__ import annotations

from dataclasses import dataclass

JUDGE_MODEL = "claude-opus-4-8"


@dataclass
class Verdict:
    score: float  # 0..1
    rationale: str


def judge(question: str, answer: str, reference_answer: str | None) -> Verdict:
    """Ask Claude to score `answer` for the captaincy `question`.

    If `reference_answer` is provided, grade against it; otherwise grade on
    groundedness/usefulness alone.
    """
    raise NotImplementedError("TODO: call Anthropic judge prompt, parse score + rationale")
