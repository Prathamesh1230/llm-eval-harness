"""LLM-as-judge using Groq, independent from the Gemini generator.

The generator and the judge are deliberately different model families.
A model grading its own output scores higher than an independent grader —
that's documented self-preference bias. Splitting them is the fix, and
it's the strongest design decision in this project.
"""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from evalharness.schemas import JudgeVerdict, PipelineOutput, TestCase

load_dotenv()

RUBRIC = """You are a strict evaluator of a retrieval-augmented QA system.

Score the ANSWER on three dimensions, 1-5. Use the full range.

FAITHFULNESS — is every claim supported by the CONTEXT?
  5 = every claim traceable to context
  4 = all key claims supported, minor unsupported phrasing
  3 = one notable claim not in context
  2 = several unsupported claims
  1 = largely fabricated

RELEVANCE — does it answer the QUESTION that was asked?
  5 = directly and fully addresses the question
  4 = addresses it with minor digression
  3 = partially on-topic
  2 = mostly answers a different question
  1 = unrelated

COMPLETENESS — compared to EXPECTED_ANSWER, how much is covered?
  5 = covers all key points
  4 = misses one minor point
  3 = misses a significant point
  2 = covers only a fraction
  1 = covers essentially nothing

Then assign exactly one failure_mode from:
  none, retrieval_miss, hallucination, incomplete, irrelevant, refusal
Use "retrieval_miss" ONLY when the CONTEXT itself lacks the needed information.

Respond with ONLY a JSON object, no markdown fences, no preamble:
{"faithfulness": int, "relevance": int, "completeness": int,
 "failure_mode": string, "reasoning": "one or two sentences"}
"""


def build_prompt(case: TestCase, output: PipelineOutput) -> str:
    context = "\n\n".join(
        f"[{c.doc_id}] {c.text}"
        for c in sorted(output.retrieved, key=lambda x: x.rank)
    )
    return (
        f"{RUBRIC}\n\n"
        f"QUESTION:\n{case.question}\n\n"
        f"CONTEXT:\n{context or '(no context retrieved)'}\n\n"
        f"EXPECTED_ANSWER:\n{case.expected_answer}\n\n"
        f"ANSWER:\n{output.answer}\n"
    )


def _extract_json(raw: str) -> dict:
    """Judges sometimes wrap JSON in fences despite instructions."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Judge returned no parsable JSON: {raw[:200]}")
        return json.loads(match.group(0))


class GroqJudge:
    """Independent judge running on Groq. Separate model family from the generator."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def judge(self, case: TestCase, output: PipelineOutput) -> JudgeVerdict:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an evaluation assistant. Respond only with valid JSON."},
                {"role": "user", "content": build_prompt(case, output)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        return JudgeVerdict(**_extract_json(raw))


def judge_agreement(a: list[JudgeVerdict], b: list[JudgeVerdict]) -> dict[str, float]:
    """Exact and within-one agreement between two judges.

    Report this in your README. It is the honest answer to
    "how do you know your judge is right?" — and almost nobody checks.
    """
    if not a or len(a) != len(b):
        return {}
    out = {}
    for dim in ("faithfulness", "relevance", "completeness"):
        xs = [getattr(v, dim) for v in a]
        ys = [getattr(v, dim) for v in b]
        n = len(xs)
        out[f"{dim}_exact"] = sum(x == y for x, y in zip(xs, ys)) / n
        out[f"{dim}_within_1"] = sum(abs(x - y) <= 1 for x, y in zip(xs, ys)) / n
    return out


if __name__ == "__main__":
    from evalharness.pipeline.rag import RagPipeline
    from evalharness.runner import load_golden

    cases = load_golden()
    pipe = RagPipeline()
    judge = GroqJudge()

    case = cases[0]
    output = pipe.answer(case.question, test_case_id=case.id)
    verdict = judge.judge(case, output)

    print(f"\nQ: {case.question}")
    print(f"A: {output.answer}")
    print(f"\nJudge model: {judge.model_name}")
    print(f"faithfulness  {verdict.faithfulness}/5")
    print(f"relevance     {verdict.relevance}/5")
    print(f"completeness  {verdict.completeness}/5")
    print(f"failure_mode  {verdict.failure_mode.value}")
    print(f"reasoning     {verdict.reasoning}")