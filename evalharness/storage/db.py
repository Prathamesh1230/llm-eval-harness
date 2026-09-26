"""SQLite schema and connection management.

Two tables: runs (one row per eval run) and case_results (one row per
question per run). Everything is append-only — we never update or delete,
because eval history is the whole point.
"""

import sqlite3
from pathlib import Path

from evalharness.schemas import CaseResult, RunSummary

DEFAULT_DB = "eval_runs.db"


def get_connection(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT PRIMARY KEY,
            config_name     TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            n_cases         INTEGER NOT NULL,
            hit_rate        REAL NOT NULL,
            mrr             REAL NOT NULL,
            precision_at_k  REAL NOT NULL,
            avg_faithfulness REAL DEFAULT 0,
            avg_relevance   REAL DEFAULT 0,
            avg_completeness REAL DEFAULT 0,
            pass_rate       REAL DEFAULT 0,
            p50_latency_ms  REAL DEFAULT 0,
            p95_latency_ms  REAL DEFAULT 0,
            total_tokens    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS case_results (
            run_id          TEXT NOT NULL,
            test_case_id    TEXT NOT NULL,
            question        TEXT NOT NULL,
            answer          TEXT NOT NULL,
            hit             INTEGER NOT NULL,
            reciprocal_rank REAL NOT NULL,
            precision_at_k  REAL NOT NULL,
            faithfulness    INTEGER,
            relevance       INTEGER,
            completeness    INTEGER,
            failure_mode    TEXT,
            reasoning       TEXT,
            latency_ms      REAL NOT NULL,
            total_tokens    INTEGER DEFAULT 0,
            passed          INTEGER NOT NULL,
            PRIMARY KEY (run_id, test_case_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
    """)


def save_run(conn: sqlite3.Connection, summary: RunSummary, results: list[CaseResult]) -> None:
    """Persist one complete eval run."""
    conn.execute(
        """INSERT INTO runs VALUES (
            :run_id, :config_name, :started_at, :n_cases,
            :hit_rate, :mrr, :precision_at_k,
            :avg_faithfulness, :avg_relevance, :avg_completeness,
            :pass_rate, :p50_latency_ms, :p95_latency_ms, :total_tokens
        )""",
        {
            "run_id": summary.run_id,
            "config_name": summary.config_name,
            "started_at": summary.started_at.isoformat(),
            "n_cases": summary.n_cases,
            "hit_rate": summary.hit_rate,
            "mrr": summary.mrr,
            "precision_at_k": summary.precision_at_k,
            "avg_faithfulness": summary.avg_faithfulness,
            "avg_relevance": summary.avg_relevance,
            "avg_completeness": summary.avg_completeness,
            "pass_rate": summary.pass_rate,
            "p50_latency_ms": summary.p50_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "total_tokens": summary.total_tokens,
        },
    )

    for r in results:
        v = r.verdict
        conn.execute(
            """INSERT INTO case_results VALUES (
                :run_id, :test_case_id, :question, :answer,
                :hit, :reciprocal_rank, :precision_at_k,
                :faithfulness, :relevance, :completeness,
                :failure_mode, :reasoning,
                :latency_ms, :total_tokens, :passed
            )""",
            {
                "run_id": summary.run_id,
                "test_case_id": r.test_case_id,
                "question": r.question,
                "answer": r.answer,
                "hit": int(r.retrieval.hit),
                "reciprocal_rank": r.retrieval.reciprocal_rank,
                "precision_at_k": r.retrieval.precision_at_k,
                "faithfulness": v.faithfulness if v else None,
                "relevance": v.relevance if v else None,
                "completeness": v.completeness if v else None,
                "failure_mode": v.failure_mode.value if v else None,
                "reasoning": v.reasoning if v else None,
                "latency_ms": r.latency_ms,
                "total_tokens": r.total_tokens,
                "passed": int(r.passed),
            },
        )

    conn.commit()
    print(f"Saved run {summary.run_id} ({summary.n_cases} cases) to {conn.execute('PRAGMA database_list').fetchone()[2]}")