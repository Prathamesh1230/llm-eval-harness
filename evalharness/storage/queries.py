"""SQL queries for comparing runs and drilling into results.

These are the queries the dashboard and CI both use. Writing them as
plain SQL rather than ORM calls is deliberate — it keeps the SQL
visible and interviewable, and ties to your NexGen SQL experience.
"""

import sqlite3


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Most recent runs first."""
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_case_results(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM case_results WHERE run_id = ? ORDER BY test_case_id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def compare_runs(conn: sqlite3.Connection, run_a: str, run_b: str) -> list[dict]:
    """Side-by-side comparison of two runs, per metric.

    Returns one row per metric with the value from each run and the delta.
    Positive delta means run_b is better (higher).
    """
    rows = conn.execute(
        """
        SELECT
            a.run_id   AS run_a,
            b.run_id   AS run_b,
            a.hit_rate          AS a_hit_rate,
            b.hit_rate          AS b_hit_rate,
            b.hit_rate - a.hit_rate AS delta_hit_rate,
            a.mrr               AS a_mrr,
            b.mrr               AS b_mrr,
            b.mrr - a.mrr       AS delta_mrr,
            a.avg_faithfulness  AS a_faith,
            b.avg_faithfulness  AS b_faith,
            b.avg_faithfulness - a.avg_faithfulness AS delta_faith,
            a.avg_relevance     AS a_rel,
            b.avg_relevance     AS b_rel,
            b.avg_relevance - a.avg_relevance AS delta_rel,
            a.pass_rate         AS a_pass,
            b.pass_rate         AS b_pass,
            b.pass_rate - a.pass_rate AS delta_pass
        FROM runs a, runs b
        WHERE a.run_id = ? AND b.run_id = ?
        """,
        (run_a, run_b),
    ).fetchall()
    return [dict(r) for r in rows]


def failed_cases(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """All cases that failed in a run — the debugging starting point."""
    rows = conn.execute(
        """
        SELECT test_case_id, question, answer,
               faithfulness, relevance, completeness,
               failure_mode, reasoning
        FROM case_results
        WHERE run_id = ? AND passed = 0
        ORDER BY faithfulness ASC
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def regression_check(conn: sqlite3.Connection, baseline_id: str, current_id: str, threshold: float = 0.05) -> dict:
    """Check if current run regressed beyond threshold vs baseline.

    Returns a dict with passed=True/False and per-metric deltas.
    This is what GitHub Actions calls.
    """
    comp = compare_runs(conn, baseline_id, current_id)
    if not comp:
        return {"passed": False, "error": "One or both runs not found"}

    row = comp[0]
    regressions = {}
    for metric in ("hit_rate", "mrr", "faith", "rel", "pass"):
        delta = row[f"delta_{metric}"]
        if delta < -threshold:
            regressions[metric] = delta

    return {
        "passed": len(regressions) == 0,
        "regressions": regressions,
        "deltas": {k: v for k, v in row.items() if k.startswith("delta_")},
    }