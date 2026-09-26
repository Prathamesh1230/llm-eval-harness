"""Orchestrates one full eval run.

Loads the golden set, runs every case through the target, scores it, and
aggregates into a RunSummary. The judge is optional — retrieval-only runs
are fast and free, which makes them the default during development.

The target is anything implementing the Target protocol: the local RAG
pipeline, an HTTP adapter pointed at someone else's chatbot, or a team's
own code. The runner never knows which.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Protocol

import yaml

from evalharness.adapters.base import Target
from evalharness.metrics.retrieval import (
    aggregate_retrieval,
    latency_percentiles,
    score_retrieval,
)
from evalharness.schemas import CaseResult, JudgeVerdict, PipelineOutput, RunSummary, TestCase


class Judge(Protocol):
    """Anything with this shape can be the judge. Keeps the runner decoupled."""

    def judge(self, case: TestCase, output: PipelineOutput) -> JudgeVerdict: ...


def load_golden(path: str | Path = "data/golden.yaml") -> list[TestCase]:
    """Parse the golden dataset into validated TestCase objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path.resolve()}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = [TestCase(**item) for item in raw]

    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test case ids in golden dataset")

    return cases


def run_eval(
    pipeline: Target,
    cases: list[TestCase],
    config_name: str = "baseline",
    judge: Optional[Judge] = None,
    top_k: int = 5,
    verbose: bool = True,
    on_case_done: Optional[Callable[[int, str], None]] = None,
) -> tuple[RunSummary, list[CaseResult]]:
    """Run every case through the target and score it.

    on_case_done is called after each case with (index, question) so a UI
    can show live progress. Ignored by the CLI, used by the API.

    Returns the aggregate summary AND the per-case results — the summary
    is what CI compares, the per-case results are what you debug with.
    """
    run_id = uuid.uuid4().hex[:8]
    started = datetime.now()
    results: list[CaseResult] = []

    for i, case in enumerate(cases, start=1):
        if verbose:
            print(f"[{i}/{len(cases)}] {case.id}: {case.question[:60]}")

        try:
            output = pipeline.answer(case.question, test_case_id=case.id)
        except Exception as e:
            # One bad case shouldn't kill a 50-case run.
            print(f"    ERROR: {e}")
            if on_case_done:
                on_case_done(i, case.question)
            continue

        retrieval = score_retrieval(output.retrieved, case.relevant_doc_ids, k=top_k)

        verdict = None
        if judge is not None:
            try:
                verdict = judge.judge(case, output)
            except Exception as e:
                print(f"    JUDGE ERROR: {e}")

        result = CaseResult(
            test_case_id=case.id,
            question=case.question,
            answer=output.answer,
            retrieval=retrieval,
            verdict=verdict,
            latency_ms=output.latency_ms,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        )
        results.append(result)

        if verbose:
            mark = "PASS" if result.passed else "FAIL"
            extra = ""
            if verdict:
                extra = f" | faith={verdict.faithfulness} rel={verdict.relevance}"
            print(f"    {mark}  rr={retrieval.reciprocal_rank:.2f}{extra}")

        if on_case_done:
            on_case_done(i, case.question)

    summary = summarize(results, run_id, config_name, started)
    return summary, results


def summarize(
    results: list[CaseResult],
    run_id: str,
    config_name: str,
    started: datetime,
) -> RunSummary:
    """Aggregate per-case results into the run-level summary."""
    retrieval = aggregate_retrieval(results)
    p50, p95 = latency_percentiles(results)

    judged = [r for r in results if r.verdict is not None]
    n_judged = len(judged) or 1  # avoid div-by-zero when running without a judge

    return RunSummary(
        run_id=run_id,
        config_name=config_name,
        started_at=started,
        n_cases=len(results),
        hit_rate=retrieval["hit_rate"],
        mrr=retrieval["mrr"],
        precision_at_k=retrieval["precision_at_k"],
        avg_faithfulness=sum(r.verdict.faithfulness for r in judged) / n_judged,
        avg_relevance=sum(r.verdict.relevance for r in judged) / n_judged,
        avg_completeness=sum(r.verdict.completeness for r in judged) / n_judged,
        pass_rate=sum(r.passed for r in results) / len(results) if results else 0.0,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_tokens=sum(r.total_tokens for r in results),
    )


def print_summary(s: RunSummary) -> None:
    print(f"\n{'=' * 52}")
    print(f"RUN {s.run_id}  |  config: {s.config_name}  |  {s.n_cases} cases")
    print(f"{'=' * 52}")
    print(f"  hit_rate         {s.hit_rate:.3f}")
    print(f"  mrr              {s.mrr:.3f}")
    print(f"  precision@k      {s.precision_at_k:.3f}")
    print(f"  pass_rate        {s.pass_rate:.3f}")
    if s.avg_faithfulness:
        print(f"  faithfulness     {s.avg_faithfulness:.2f} / 5")
        print(f"  relevance        {s.avg_relevance:.2f} / 5")
        print(f"  completeness     {s.avg_completeness:.2f} / 5")
    print(f"  latency p50/p95  {s.p50_latency_ms:.0f}ms / {s.p95_latency_ms:.0f}ms")
    print(f"  total tokens     {s.total_tokens}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    from evalharness.pipeline.rag import RagPipeline

    cases = load_golden()
    print(f"Loaded {len(cases)} test cases\n")

    pipeline = RagPipeline()
    summary, results = run_eval(pipeline, cases, config_name="baseline")
    print_summary(summary)