"""Retrieve + augment + answer (Anthropic).

Ties the pipeline together: get temporally-valid chunks for a gameweek, build a
context-grounded prompt, and ask Claude for a captaincy recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieve import Chunk, retrieve

ANSWER_MODEL = "claude-opus-4-8"


@dataclass
class RagAnswer:
    answer: str
    context: list[Chunk]


def answer_captaincy(question: str, deadline: str, k: int = 5) -> RagAnswer:
    """Retrieve temporally-valid context for the gameweek and generate an answer.

    `deadline` is the GW's deadline (ISO); it is threaded straight into
    `retrieve` so the temporal-integrity filter is honored end to end.
    """
    context = retrieve(question, deadline=deadline, k=k)
    raise NotImplementedError("TODO: build grounded prompt + call Anthropic with `context`")
