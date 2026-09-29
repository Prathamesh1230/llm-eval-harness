"""Measures how much we can trust the judge, and how noisy runs are.

Two things the harness claims but hadn't verified:

  1. Judge agreement — do two judges from different model families give
     similar scores to the same answers? If they disagree wildly, the
     scores mostly reflect judge personality rather than answer quality.

  2. Run variance — does the same config produce the same numbers when
     run repeatedly? If not, small "improvements" are indistinguishable
     from noise, and regression detection is meaningless.

Run:  python -m experiments.judge_reliability
"""

import argparse
import statistics
import time
from collections import Counter

from evalharness.judge.llm_judge import GroqJudge, judge_agreement
from evalharness.pipeline.rag import RagPipeline
from evalharness.runner import load_golden, run_eval

PRIMARY_JUDGE = "openai/gpt-oss-120b"
SECOND_JUDGE = "openai/gpt-oss-20b"

DIMS = ("faithfulness", "relevance", "completeness")


def with_retry(fn, *args, attempts: int = 4, **kwargs):
    """Retry on transient API failures with exponential backoff.

    Providers return 503 when they're overloaded. One busy moment
    shouldn't discard an entire experiment's worth of API calls.
    """
    delay = 2.0
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            transient = any(
                code in str(e)
                for code in ("503", "429", "UNAVAILABLE", "rate limit")
            )
            if not transient or attempt == attempts - 1:
                raise
            print(f"  retrying in {delay:.0f}s ({type(e).__name__})      ")
            time.sleep(delay)
            delay *= 2


def bar(label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")


# ------------------------------------------------------- judge agreement

def measure_agreement() -> None:
    bar("Judge agreement across model families")
    print(f"primary: {PRIMARY_JUDGE}")
    print(f"second:  {SECOND_JUDGE}\n")

    cases = load_golden()
    pipeline = RagPipeline(collection_name="agreement_test")

    # Generate answers ONCE. Both judges score the identical outputs,
    # otherwise we'd be measuring generator variance too.
    outputs = []
    for i, case in enumerate(cases, 1):
        print(f"  generating {i}/{len(cases)}", end="\r")
        outputs.append(
            with_retry(pipeline.answer, case.question, test_case_id=case.id)
        )
    print(" " * 40, end="\r")

    judge_a = GroqJudge(model_name=PRIMARY_JUDGE)
    judge_b = GroqJudge(model_name=SECOND_JUDGE)

    verdicts_a, verdicts_b, kept_cases = [], [], []
    for case, out in zip(cases, outputs):
        try:
            va = with_retry(judge_a.judge, case, out)
            vb = with_retry(judge_b.judge, case, out)
        except Exception as e:
            print(f"  skipped {case.id}: {e}")
            continue
        verdicts_a.append(va)
        verdicts_b.append(vb)
        kept_cases.append(case)

    if not verdicts_a:
        print("No verdicts collected.")
        return

    agree = judge_agreement(verdicts_a, verdicts_b)

    print(f"\nScored {len(verdicts_a)} answers with both judges.\n")
    print(f"{'dimension':<16}{'exact':>10}{'within 1':>12}")
    print("-" * 38)
    for dim in DIMS:
        print(f"{dim:<16}{agree[f'{dim}_exact']:>9.0%}{agree[f'{dim}_within_1']:>12.0%}")

    overall_exact  = statistics.mean(agree[f"{d}_exact"]    for d in DIMS)
    overall_within = statistics.mean(agree[f"{d}_within_1"] for d in DIMS)
    print("-" * 38)
    print(f"{'mean':<16}{overall_exact:>9.0%}{overall_within:>12.0%}")

    # Mean score per judge shows systematic leniency, which exact
    # agreement alone would hide.
    print("\nMean score per judge (detects a systematically softer judge):")
    for dim in DIMS:
        a = statistics.mean(getattr(v, dim) for v in verdicts_a)
        b = statistics.mean(getattr(v, dim) for v in verdicts_b)
        print(f"  {dim:<14} primary {a:.2f}   second {b:.2f}   diff {b - a:+.2f}")

    # Where they disagree is more interesting than the headline number.
    disagreements = [
        (c.id, d, getattr(a, d), getattr(b, d))
        for c, a, b in zip(kept_cases, verdicts_a, verdicts_b)
        for d in DIMS
        if getattr(a, d) != getattr(b, d)
    ]
    if disagreements:
        print(f"\n{len(disagreements)} scores differed:")
        for cid, dim, a, b in disagreements[:10]:
            print(f"  {cid:<10} {dim:<14} {a} vs {b}")
        if len(disagreements) > 10:
            print(f"  ... and {len(disagreements) - 10} more")
    else:
        print("\nBoth judges agreed on every score.")

    modes_a = Counter(v.failure_mode.value for v in verdicts_a)
    modes_b = Counter(v.failure_mode.value for v in verdicts_b)
    print(f"\nFailure modes  primary: {dict(modes_a)}")
    print(f"               second:  {dict(modes_b)}")


# --------------------------------------------------------- run variance

def measure_variance(n_runs: int = 5) -> None:
    bar(f"Run-to-run variance ({n_runs} identical runs)")
    print("Same config, same questions, same judge. Any spread here is\n"
          "noise, and a 'regression' smaller than it means nothing.\n")

    cases = load_golden()
    judge = GroqJudge(model_name=PRIMARY_JUDGE)
    metrics = {
        "hit_rate": [], "mrr": [], "precision_at_k": [],
        "avg_faithfulness": [], "avg_relevance": [], "pass_rate": [],
    }

    for i in range(n_runs):
        print(f"  run {i + 1}/{n_runs}", end="\r")
        pipeline = RagPipeline(collection_name=f"variance_{i}")
        summary, _ = run_eval(
            pipeline, cases, config_name=f"variance_{i}",
            judge=judge, verbose=False,
        )
        for key in metrics:
            metrics[key].append(getattr(summary, key))
    print(" " * 30, end="\r")

    print(f"{'metric':<20}{'mean':>8}{'min':>8}{'max':>8}{'spread':>9}")
    print("-" * 53)
    for key, values in metrics.items():
        spread = max(values) - min(values)
        print(
            f"{key:<20}{statistics.mean(values):>8.3f}"
            f"{min(values):>8.3f}{max(values):>8.3f}{spread:>9.3f}"
        )

    worst = max(max(v) - min(v) for v in metrics.values())
    print(f"\nLargest spread across {n_runs} identical runs: {worst:.3f}")
    if worst == 0:
        print("Fully deterministic — every run produced identical scores.")
    else:
        print(f"Treat changes smaller than {worst:.3f} as noise, not signal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-agreement", action="store_true")
    ap.add_argument("--skip-variance",  action="store_true")
    ap.add_argument("--runs", type=int, default=5,
                    help="repeat runs for variance check")
    args = ap.parse_args()

    if not args.skip_agreement:
        measure_agreement()
    if not args.skip_variance:
        measure_variance(args.runs)

    print("\nDone.")