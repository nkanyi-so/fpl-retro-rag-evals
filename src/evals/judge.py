"""LLM-as-judge for answer quality.

For each golden case: run the RAG (`rag.answer`), then ask the judge to score the
system's answer on two 1-5 dimensions against the retrieved context and the case's
reference answer. The judge's output is validated with a pydantic model — a
malformed or out-of-range verdict RAISES and is counted as a validation failure,
never silently accepted.

This is answer-quality scoring, NOT the outcome-grounded decision-quality eval —
that lands next session.
"""

from __future__ import annotations

import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make src/ importable
from rag import answer  # noqa: E402
from retrieval_eval import load_golden  # noqa: E402  (same directory)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

JUDGE_MODEL = "claude-haiku-4-5"


class JudgeVerdict(BaseModel):
    """Validated judge output. Out-of-range scores raise ValidationError."""

    groundedness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    reasoning: str


JUDGE_SYSTEM = (
    "You are a strict evaluator of an FPL captaincy assistant's answer. Score two "
    "dimensions on an integer 1-5 scale using these anchors exactly:\n"
    "- groundedness: 5 = every claim in the answer is supported by the provided "
    "notes; 1 = the answer invents facts not in the notes, or ignores them.\n"
    "- correctness: 5 = the captain pick and reasoning align with the reference "
    "answer; 1 = wrong pick, or no clear pick.\n"
    "Judge only against the provided notes and the reference answer — not your own "
    "Premier League knowledge. Give brief reasoning for the two scores."
)


def judge_case(
    question: str,
    context_text: str,
    system_answer: str,
    reference_answer: str | None,
) -> JudgeVerdict:
    """Score one answer. Raises ValidationError if the verdict is malformed."""
    user_message = (
        f"Question: {question}\n\n"
        f"Retrieved notes given to the assistant:\n{context_text or '(none)'}\n\n"
        f"Assistant's answer:\n{system_answer}\n\n"
        f"Reference answer:\n{reference_answer or '(none provided)'}\n\n"
        "Score groundedness and correctness (1-5 each) and explain briefly."
    )
    client = Anthropic()
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
        output_format=JudgeVerdict,
    )
    return response.parsed_output


def run() -> None:
    cases = load_golden()
    scored = []
    failures = []  # (gameweek, kind, message)

    for case in cases:
        rag = answer(case.question, case.gameweek)
        try:
            verdict = judge_case(
                case.question, rag.context_text, rag.answer, case.reference_answer
            )
        except ValidationError as e:
            failures.append((case.gameweek, "validation", str(e)))
            continue
        except Exception as e:  # API / parse errors — surface, don't hide
            failures.append((case.gameweek, type(e).__name__, str(e)))
            continue
        scored.append({"case": case, "rag": rag, "verdict": verdict})

    print("=" * 72)
    print(f"LLM-AS-JUDGE — {len(cases)} cases, model={JUDGE_MODEL}")
    print("=" * 72)

    for s in scored:
        c, rag, v = s["case"], s["rag"], s["verdict"]
        print(f"\nGW{c.gameweek}: {c.question}")
        print(f"  retrieved : {rag.context_doc_ids}")
        print(f"  answer    : {rag.answer.strip()[:300]}")
        print(f"  groundedness={v.groundedness}/5  correctness={v.correctness}/5")
        print(f"  reasoning : {v.reasoning.strip()}")

    print("\n" + "-" * 72)
    if scored:
        avg_g = sum(s["verdict"].groundedness for s in scored) / len(scored)
        avg_c = sum(s["verdict"].correctness for s in scored) / len(scored)
        print(f"AGGREGATE ({len(scored)} scored): "
              f"groundedness={avg_g:.2f}/5  correctness={avg_c:.2f}/5")
    else:
        print("AGGREGATE: no cases scored.")

    print("-" * 72)
    if failures:
        print(f"JUDGE-OUTPUT FAILURES ({len(failures)}):")
        for gw, kind, msg in failures:
            print(f"  GW{gw} [{kind}]: {msg.splitlines()[0]}")
    else:
        print("JUDGE-OUTPUT FAILURES: none — every verdict validated.")


if __name__ == "__main__":
    run()
