"""Command-line interface.

CI needs a command, not a notebook. Every operation the harness supports
is exposed here so GitHub Actions, a cron job, or a human can all drive it
the same way.

The target under test is chosen by config or flag:
  - local  : the built-in RAG pipeline (default)
  - http   : any chatbot API, e.g.  --url https://example.com/chat
"""

from pathlib import Path
from typing import Optional

import typer
import yaml

from evalharness.adapters.base import Target
from evalharness.adapters.http_adapter import HttpAdapter
from evalharness.adapters.security import UnsafeTargetError
from evalharness.judge.llm_judge import GroqJudge
from evalharness.pipeline.rag import RagPipeline
from evalharness.runner import load_golden, print_summary, run_eval
from evalharness.storage.db import get_connection, save_run
from evalharness.storage.queries import (
    compare_runs,
    failed_cases,
    list_runs,
    regression_check,
)

app = typer.Typer(help="LLM evaluation harness for RAG pipelines.")


def load_config(path: Optional[str]) -> dict:
    """Load a YAML config, or return sensible defaults."""
    defaults = {
        "name": "baseline",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 5,
        "gen_model": "gemini-3.1-flash-lite",
        "judge_model": "openai/gpt-oss-120b",
        "target": {"type": "local"},
    }
    if not path:
        return defaults

    cfg_path = Path(path)
    if not cfg_path.exists():
        typer.secho(f"Config not found: {cfg_path}", fg="red")
        raise typer.Exit(1)

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {**defaults, **loaded}


def build_target(cfg: dict, url: Optional[str], allow_private: bool) -> tuple[Target, str]:
    """Turn the config (plus any CLI overrides) into a system under test.

    Returns the target and a short label describing it for the console.
    """
    target_cfg = dict(cfg.get("target") or {"type": "local"})

    # --url on the command line overrides whatever the config says.
    if url:
        target_cfg = {"type": "http", "url": url}

    kind = target_cfg.pop("type", "local")

    if kind == "local":
        pipeline = RagPipeline(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            top_k=cfg["top_k"],
            gen_model=cfg["gen_model"],
            collection_name=cfg["name"],
        )
        label = f"local pipeline (chunk={cfg['chunk_size']}, k={cfg['top_k']})"
        return pipeline, label

    if kind == "http":
        if "url" not in target_cfg:
            typer.secho("HTTP target needs a 'url'.", fg="red")
            raise typer.Exit(1)
        if allow_private:
            target_cfg["allow_private"] = True
        try:
            adapter = HttpAdapter(**target_cfg)
        except UnsafeTargetError as e:
            typer.secho(f"Blocked unsafe target: {e}", fg="red")
            typer.secho("Testing a local or internal chatbot? Add --allow-private.", fg="yellow")
            raise typer.Exit(1)
        except TypeError as e:
            typer.secho(f"Invalid HTTP target settings: {e}", fg="red")
            raise typer.Exit(1)
        return adapter, f"HTTP API ({adapter.url})"

    typer.secho(f"Unknown target type '{kind}'. Use 'local' or 'http'.", fg="red")
    raise typer.Exit(1)


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to a YAML config"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Evaluate a chatbot API at this URL"),
    allow_private: bool = typer.Option(
        False, "--allow-private", help="Allow localhost/internal URLs (self-hosted use only)"
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judging (fast, free)"),
    golden: str = typer.Option("data/golden.yaml", "--golden", "-g"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist the run to SQLite"),
):
    """Run one full evaluation."""
    cfg = load_config(config)
    target, label = build_target(cfg, url, allow_private)

    typer.secho(f"Config: {cfg['name']}", fg="cyan")
    typer.secho(f"Target: {label}", fg="cyan")

    cases = load_golden(golden)
    typer.echo(f"Loaded {len(cases)} test cases")

    judge = None if no_judge else GroqJudge(model_name=cfg["judge_model"])

    try:
        summary, results = run_eval(
            target, cases, config_name=cfg["name"], judge=judge, top_k=cfg["top_k"]
        )
    finally:
        if hasattr(target, "close"):
            target.close()

    print_summary(summary)

    if save:
        conn = get_connection()
        save_run(conn, summary, results)
        typer.secho(f"run_id: {summary.run_id}", fg="green", bold=True)


@app.command()
def runs(limit: int = typer.Option(20, "--limit", "-n")):
    """List recent runs."""
    conn = get_connection()
    rows = list_runs(conn, limit)
    if not rows:
        typer.echo("No runs recorded yet.")
        return

    typer.echo(f"\n{'run_id':<10} {'config':<14} {'hit':<7} {'mrr':<7} {'faith':<7} {'pass':<7} started")
    typer.echo("-" * 78)
    for r in rows:
        typer.echo(
            f"{r['run_id']:<10} {r['config_name']:<14} "
            f"{r['hit_rate']:<7.3f} {r['mrr']:<7.3f} "
            f"{r['avg_faithfulness']:<7.2f} {r['pass_rate']:<7.3f} "
            f"{r['started_at'][:19]}"
        )


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline run_id"),
    current: str = typer.Argument(..., help="Run to compare against baseline"),
):
    """Compare two runs metric by metric."""
    conn = get_connection()
    rows = compare_runs(conn, baseline, current)
    if not rows:
        typer.secho("One or both runs not found.", fg="red")
        raise typer.Exit(1)

    row = rows[0]
    typer.echo(f"\n{baseline}  ->  {current}\n")
    pairs = [
        ("hit_rate", "a_hit_rate", "b_hit_rate", "delta_hit_rate"),
        ("mrr", "a_mrr", "b_mrr", "delta_mrr"),
        ("faithfulness", "a_faith", "b_faith", "delta_faith"),
        ("relevance", "a_rel", "b_rel", "delta_rel"),
        ("pass_rate", "a_pass", "b_pass", "delta_pass"),
    ]
    for label, a, b, d in pairs:
        delta = row[d]
        arrow = "+" if delta > 0 else ""
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        typer.secho(
            f"  {label:<14} {row[a]:.3f}  ->  {row[b]:.3f}   ({arrow}{delta:.3f})",
            fg=color,
        )


@app.command()
def failures(run_id: str = typer.Argument(..., help="Run to inspect")):
    """Show the cases that failed in a run."""
    conn = get_connection()
    rows = failed_cases(conn, run_id)
    if not rows:
        typer.secho("No failures in this run.", fg="green")
        return

    typer.secho(f"\n{len(rows)} failed case(s):\n", fg="yellow")
    for r in rows:
        typer.echo(f"  [{r['test_case_id']}] {r['question']}")
        typer.echo(f"      answer: {r['answer'][:90]}")
        typer.echo(f"      scores: faith={r['faithfulness']} rel={r['relevance']} comp={r['completeness']}")
        typer.echo(f"      mode:   {r['failure_mode']}")
        typer.echo(f"      why:    {r['reasoning']}\n")


@app.command()
def check(
    baseline: str = typer.Argument(..., help="Baseline run_id"),
    current: str = typer.Argument(..., help="Current run_id"),
    threshold: float = typer.Option(0.05, "--threshold", "-t"),
):
    """Fail (exit 1) if current regressed beyond threshold. This is what CI calls."""
    conn = get_connection()
    result = regression_check(conn, baseline, current, threshold)

    if result.get("error"):
        typer.secho(result["error"], fg="red")
        raise typer.Exit(1)

    if result["passed"]:
        typer.secho(f"PASS — no metric regressed more than {threshold:.0%}", fg="green", bold=True)
        raise typer.Exit(0)

    typer.secho(f"FAIL — regression beyond {threshold:.0%}:", fg="red", bold=True)
    for metric, delta in result["regressions"].items():
        typer.secho(f"  {metric}: {delta:.3f}", fg="red")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()