"""The evaluation branch of the flow.

Four metrics, each isolating a different failure mode:

  faithfulness       - is the answer supported by the retrieved context?
                       (catches hallucination)
  answer_relevance   - does the answer address the question asked?
                       (catches evasion / topic drift)
  context_precision  - how much of what we retrieved was actually useful?
                       (catches noisy retrieval / bad chunking)
  context_recall     - did we retrieve everything the reference answer needs?
                       (catches missed retrieval — the silent killer)

Plus answer_correctness against your reference answer, when you provide one.

Implemented as LLM-as-judge with claim decomposition rather than a single
"score this 1-10" call, because holistic scores from an LLM are close to noise.
Decomposing into atomic claims and checking each one is far more stable.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import CONFIG
from .rag import get_llm

# ---------------------------------------------------------------------------
# Judge plumbing
# ---------------------------------------------------------------------------


def _judge(prompt: str) -> dict[str, Any]:
    """Call the judge model and parse JSON out of it, defensively."""
    llm = get_llm(model=CONFIG.judge_model, temperature=0.0)
    raw = llm.invoke(prompt).content
    text = raw if isinstance(raw, str) else str(raw)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _ratio(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return round(sum(1 for i in items if i.get(key)) / len(items), 3)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are auditing an AI answer for hallucination.

Step 1: break the ANSWER into atomic factual claims. Ignore hedges, citation \
markers like [S1], and meta-statements such as "the excerpts do not say".
Step 2: for each claim, decide whether it is directly supported by the CONTEXT. \
A claim is supported only if the context states it or entails it without \
outside knowledge. Paraphrase is fine; inference beyond the text is not.

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY JSON:
{{"claims": [{{"claim": "...", "supported": true, "reason": "..."}}]}}"""


def faithfulness(answer: str, contexts: list[str]) -> dict:
    if not answer.strip():
        return {"score": 0.0, "claims": []}
    out = _judge(
        FAITHFULNESS_PROMPT.format(context="\n\n---\n\n".join(contexts), answer=answer)
    )
    claims = out.get("claims", [])
    return {
        "score": _ratio(claims, "supported"),
        "claims": claims,
        "unsupported": [c["claim"] for c in claims if not c.get("supported")],
    }


RELEVANCE_PROMPT = """Judge whether the ANSWER actually addresses the QUESTION.

Score 1.0  - fully answers what was asked
Score 0.5  - partially answers, or answers a narrower/adjacent question
Score 0.0  - does not answer it

An honest "the paper does not state this" counts as 1.0 IF the question truly \
cannot be answered from the context; otherwise it is 0.0.

QUESTION: {question}
CONTEXT AVAILABLE: {context}
ANSWER: {answer}

Return ONLY JSON: {{"score": 0.0, "reason": "..."}}"""


def answer_relevance(question: str, answer: str, contexts: list[str]) -> dict:
    out = _judge(
        RELEVANCE_PROMPT.format(
            question=question, context="\n\n---\n\n".join(contexts), answer=answer
        )
    )
    return {"score": float(out.get("score", 0.0)), "reason": out.get("reason", "")}


PRECISION_PROMPT = """For each retrieved CHUNK, decide whether it contains \
information useful for answering the QUESTION. Judge each chunk independently.

QUESTION: {question}

CHUNKS:
{chunks}

Return ONLY JSON:
{{"verdicts": [{{"index": 1, "useful": true, "reason": "..."}}]}}"""


def context_precision(question: str, contexts: list[str]) -> dict:
    if not contexts:
        return {"score": 0.0, "verdicts": []}
    chunks = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    out = _judge(PRECISION_PROMPT.format(question=question, chunks=chunks))
    verdicts = out.get("verdicts", [])
    return {"score": _ratio(verdicts, "useful"), "verdicts": verdicts}


RECALL_PROMPT = """Break the REFERENCE ANSWER into atomic claims, then check \
whether each one can be found in the RETRIEVED CONTEXT.

This measures retrieval, not generation: you are asking "did we fetch the right \
material?", not "did the model write a good answer?".

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}

Return ONLY JSON:
{{"claims": [{{"claim": "...", "present": true}}]}}"""


def context_recall(reference_answer: str, contexts: list[str]) -> dict:
    if not reference_answer.strip():
        return {"score": None, "claims": []}
    out = _judge(
        RECALL_PROMPT.format(
            reference=reference_answer, context="\n\n---\n\n".join(contexts)
        )
    )
    claims = out.get("claims", [])
    return {
        "score": _ratio(claims, "present"),
        "claims": claims,
        "missing": [c["claim"] for c in claims if not c.get("present")],
    }


CORRECTNESS_PROMPT = """Compare the GENERATED ANSWER to the REFERENCE ANSWER.

Score 1.0  - same substance; wording may differ
Score 0.5  - partially correct, or correct but incomplete
Score 0.0  - contradicts the reference, or misses the point

Numbers and metric names must match exactly to score above 0.5.

QUESTION: {question}
REFERENCE: {reference}
GENERATED: {generated}

Return ONLY JSON: {{"score": 0.0, "reason": "..."}}"""


def answer_correctness(question: str, generated: str, reference: str) -> dict:
    if not reference.strip():
        return {"score": None, "reason": "no reference answer provided"}
    out = _judge(
        CORRECTNESS_PROMPT.format(
            question=question, reference=reference, generated=generated
        )
    )
    return {"score": float(out.get("score", 0.0)), "reason": out.get("reason", "")}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    question: str
    reference_answer: str = ""
    reference_context: str = ""  # optional: a snippet that must be retrieved
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    question: str
    generated_answer: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float | None
    answer_correctness: float | None
    retrieved_pages: list[int]
    unsupported_claims: list[str]
    missing_claims: list[str]
    tags: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_case(case: EvalCase, paper_id: str, title: str = "the paper") -> CaseResult:
    from .rag import answer_question

    result = answer_question(case.question, paper_id, title)

    faith = faithfulness(result.answer, result.contexts)
    rel = answer_relevance(case.question, result.answer, result.contexts)
    prec = context_precision(case.question, result.contexts)
    rec = context_recall(case.reference_answer, result.contexts)
    corr = answer_correctness(case.question, result.answer, case.reference_answer)

    return CaseResult(
        question=case.question,
        generated_answer=result.answer,
        faithfulness=faith["score"],
        answer_relevance=rel["score"],
        context_precision=prec["score"],
        context_recall=rec["score"],
        answer_correctness=corr["score"],
        retrieved_pages=[s.page for s in result.sources],
        unsupported_claims=faith.get("unsupported", []),
        missing_claims=rec.get("missing", []),
        tags=case.tags,
    )


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    def mean(key: str) -> float | None:
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n_cases": len(results),
        "faithfulness": mean("faithfulness"),
        "answer_relevance": mean("answer_relevance"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "answer_correctness": mean("answer_correctness"),
    }
