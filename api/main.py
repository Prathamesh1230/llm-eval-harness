"""FastAPI layer over the eval harness.

Serves the dashboard as static files and exposes run history, comparisons,
and per-case results as JSON. Also provides a trigger endpoint that kicks
off an eval run in the background, and a demo chatbot endpoint the harness
can be aimed at.

Workspace endpoints (upload, questions, multi-config evaluation) live in
api/workspace_routes.py and are mounted as a router.
"""

import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.workspace_routes import router as workspace_router
from evalharness.judge.llm_judge import GroqJudge
from evalharness.pipeline.rag import RagPipeline
from evalharness.runner import load_golden, run_eval
from evalharness.storage.db import get_connection, save_run
from evalharness.storage.queries import (
    compare_runs,
    failed_cases,
    get_case_results,
    get_run,
    list_runs,
)
from evalharness.workspace import manager

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ------------------------------------------------------------- demo chatbot state

# The demo chatbot gets its own Chroma collection. Eval runs delete and
# rebuild their collection on startup, so sharing a name would wipe the
# demo's index mid-use.
DEMO_COLLECTION = "demo"

_demo_pipeline: Optional[RagPipeline] = None
_demo_lock = threading.Lock()


def _get_demo_pipeline() -> RagPipeline:
    """Build the demo pipeline once. The lock stops two requests from
    indexing at the same time."""
    global _demo_pipeline
    with _demo_lock:
        if _demo_pipeline is None:
            _demo_pipeline = RagPipeline(collection_name=DEMO_COLLECTION)
        return _demo_pipeline


def _warm_demo() -> None:
    try:
        _get_demo_pipeline()
        print("Demo chatbot ready.")
    except Exception as e:
        # Don't crash the whole server if keys are missing; the demo
        # endpoint will report the error when called.
        print(f"Demo warm-up failed: {e}")


def _cleanup_workspaces() -> None:
    try:
        removed = manager.cleanup_expired()
        if removed:
            print(f"Cleaned up {removed} expired workspace(s).")
    except Exception as e:
        print(f"Workspace cleanup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Index the demo docs in the background so the server starts instantly
    # and the first visitor doesn't wait for embeddings.
    threading.Thread(target=_warm_demo, daemon=True).start()
    threading.Thread(target=_cleanup_workspaces, daemon=True).start()
    yield


app = FastAPI(
    title="LLM Eval Harness",
    description="Regression testing for RAG pipelines",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workspace_router)


# ------------------------------------------------------------- models

class EvalRequest(BaseModel):
    config_name: str = "baseline"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    use_judge: bool = True


class EvalStatus(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    run_id: Optional[str] = None
    message: Optional[str] = None


class ChatRequest(BaseModel):
    question: str


# In-memory job tracker. Fine for a single-user tool; a real deployment
# would use Redis or a jobs table.
_jobs: dict[str, EvalStatus] = {}


# ------------------------------------------------------------- routes

@app.get("/")
def dashboard():
    """Serve the dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api")
def api_info():
    return {
        "name": "LLM Eval Harness API",
        "docs": "/docs",
        "endpoints": [
            "/runs",
            "/runs/{run_id}",
            "/runs/{run_id}/cases",
            "/runs/{run_id}/failures",
            "/compare/{baseline_id}/{current_id}",
            "/evaluate",
            "/demo/chat",
            "/workspace",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "demo_ready": _demo_pipeline is not None}


@app.get("/runs")
def get_runs(limit: int = 50):
    """All runs, most recent first."""
    return list_runs(get_connection(), limit)


@app.get("/runs/{run_id}")
def get_single_run(run_id: str):
    run = get_run(get_connection(), run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


@app.get("/runs/{run_id}/cases")
def get_cases(run_id: str):
    """Per-question results for one run — powers the drill-down view."""
    conn = get_connection()
    if not get_run(conn, run_id):
        raise HTTPException(404, f"Run {run_id} not found")
    return get_case_results(conn, run_id)


@app.get("/runs/{run_id}/failures")
def get_failures(run_id: str):
    """Only the cases that failed, worst faithfulness first."""
    conn = get_connection()
    if not get_run(conn, run_id):
        raise HTTPException(404, f"Run {run_id} not found")
    return failed_cases(conn, run_id)


@app.get("/compare/{baseline_id}/{current_id}")
def compare(baseline_id: str, current_id: str):
    """Metric-by-metric delta between two runs."""
    rows = compare_runs(get_connection(), baseline_id, current_id)
    if not rows:
        raise HTTPException(404, "One or both runs not found")
    return rows[0]


def _run_eval_job(job_id: str, req: EvalRequest) -> None:
    """Background worker. Runs the full eval and persists it."""
    _jobs[job_id] = EvalStatus(status="running")
    try:
        pipeline = RagPipeline(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            top_k=req.top_k,
            collection_name=req.config_name,
        )
        judge = GroqJudge() if req.use_judge else None

        summary, results = run_eval(
            pipeline,
            load_golden(),
            config_name=req.config_name,
            judge=judge,
            top_k=req.top_k,
            verbose=False,
        )
        save_run(get_connection(), summary, results)
        _jobs[job_id] = EvalStatus(status="done", run_id=summary.run_id)

    except Exception as e:
        _jobs[job_id] = EvalStatus(status="error", message=str(e))


@app.post("/evaluate", response_model=EvalStatus)
def trigger_eval(req: EvalRequest, background: BackgroundTasks):
    """Kick off an eval run on the built-in dataset. Poll /jobs/{id}."""
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = EvalStatus(status="queued")
    background.add_task(_run_eval_job, job_id, req)
    return EvalStatus(status="queued", run_id=job_id)


@app.get("/jobs/{job_id}", response_model=EvalStatus)
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


# ------------------------------------------------------------- demo chatbot

@app.post("/demo/chat")
def demo_chat(req: ChatRequest):
    """A real chatbot endpoint, shaped like any company's chatbot API.
    Visitors can aim the HTTP adapter here to see the harness work."""
    try:
        pipe = _get_demo_pipeline()
    except Exception as e:
        raise HTTPException(503, f"Demo chatbot unavailable: {e}")

    output = pipe.answer(req.question)
    return {
        "answer": output.answer,
        "sources": [
            {"doc_id": c.doc_id, "text": c.text, "score": c.score}
            for c in output.retrieved
        ],
    }


# Mounted last so it doesn't shadow the routes above.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")