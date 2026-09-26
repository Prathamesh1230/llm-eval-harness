"""Workspace endpoints: the public-facing half of the API.

A visitor creates a workspace, uploads documents, writes test questions,
then runs an evaluation across one or more configurations. Everything is
scoped to their workspace ID so concurrent users never collide.

Notes on the tricky parts:
  - Chroma collection names get a random suffix per run. Collections are
    global to the Chroma database, so two visitors using the config name
    "baseline" would otherwise delete each other's index mid-run.
  - A semaphore caps concurrent evaluations. A small server running five
    at once would run out of memory.
  - User-supplied API keys are passed through and never persisted.
"""

import secrets
import threading
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from evalharness import keys as keymod
from evalharness.judge.llm_judge import GroqJudge
from evalharness.pipeline.rag import RagPipeline
from evalharness.progress import ProgressTracker
from evalharness.runner import run_eval
from evalharness.schemas import TestCase
from evalharness.storage.db import get_connection, save_run
from evalharness.workspace import manager
from evalharness.workspace.limits import (
    MAX_CONCURRENT_EVALS,
    LimitExceeded,
    check_configs,
    check_questions,
    summary as limits_summary,
)

router = APIRouter(prefix="/workspace", tags=["workspace"])

# Caps how many evaluations run at once across all visitors.
_eval_slots = threading.Semaphore(MAX_CONCURRENT_EVALS)

# job_id -> ProgressTracker, for the frontend to poll.
_jobs: dict[str, ProgressTracker] = {}
_jobs_lock = threading.Lock()


# ------------------------------------------------------------ models

class QuestionIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=2000)
    relevant_doc_ids: list[str] = Field(default_factory=list, max_length=10)


class ConfigIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class EvaluateIn(BaseModel):
    configs: list[ConfigIn] = Field(min_length=1)
    use_judge: bool = True
    gemini_key: Optional[str] = None
    groq_key: Optional[str] = None


class JobStarted(BaseModel):
    job_id: str
    status: Literal["queued"]


# ------------------------------------------------------------ helpers

def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind a proxy the real IP is in X-Forwarded-For; the first entry is
    the client. Falls back to the socket address locally.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _require_workspace(ws_id: str):
    ws = manager.get(ws_id)
    if ws is None:
        raise HTTPException(404, "Workspace not found or expired")
    return ws


def _to_test_cases(questions: list[dict]) -> list[TestCase]:
    return [TestCase(**q) for q in questions]


# ------------------------------------------------------------ routes

@router.get("/limits")
def get_limits():
    """Current limits, so the UI can show them before a user hits one."""
    return {
        **limits_summary(),
        "free_tier_available": keymod.server_keys_available(),
    }


@router.post("")
def create_workspace(request: Request):
    """Create an isolated workspace for this visitor."""
    try:
        ws = manager.create(client_key=_client_key(request))
    except LimitExceeded as e:
        raise HTTPException(429, str(e))
    return ws.to_dict()


@router.get("/{ws_id}")
def get_workspace(ws_id: str):
    return _require_workspace(ws_id).to_dict()


@router.post("/{ws_id}/files")
async def upload_files(ws_id: str, files: list[UploadFile]):
    """Upload documents into the workspace."""
    ws = _require_workspace(ws_id)

    uploads: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        uploads.append((f.filename or "unnamed", data))

    try:
        saved = manager.add_files(ws, uploads)
    except LimitExceeded as e:
        raise HTTPException(400, str(e))

    return {"saved": saved, "workspace": ws.to_dict()}


@router.delete("/{ws_id}/files/{filename}")
def delete_file(ws_id: str, filename: str):
    ws = _require_workspace(ws_id)
    try:
        removed = manager.remove_file(ws, filename)
    except LimitExceeded as e:
        raise HTTPException(400, str(e))
    if not removed:
        raise HTTPException(404, "File not found")
    return ws.to_dict()


@router.put("/{ws_id}/questions")
def set_questions(ws_id: str, questions: list[QuestionIn]):
    """Replace the workspace's test set."""
    ws = _require_workspace(ws_id)

    try:
        check_questions(len(questions))
    except LimitExceeded as e:
        raise HTTPException(400, str(e))

    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "Question ids must be unique")

    manager.set_questions(ws, [q.model_dump() for q in questions])
    return ws.to_dict()


def _run_job(job_id: str, ws_id: str, body: EvaluateIn, api_keys: keymod.ApiKeys) -> None:
    """Background worker: run every requested config and save each result."""
    tracker = _jobs[job_id]

    acquired = _eval_slots.acquire(timeout=300)
    if not acquired:
        tracker.fail("Server busy. Please try again shortly.")
        return

    try:
        ws = manager.get(ws_id)
        if ws is None:
            tracker.fail("Workspace expired while queued")
            return

        cases = _to_test_cases(ws.questions)
        run_ids: list[str] = []

        for i, cfg in enumerate(body.configs):
            tracker.start_config(cfg.name, i, len(cases))

            # Unique per run so concurrent visitors never share a collection.
            collection = f"{ws.collection_prefix}_{cfg.name}_{secrets.token_hex(3)}"

            pipeline = RagPipeline(
                docs_dir=str(ws.docs_dir),
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                top_k=cfg.top_k,
                collection_name=collection,
                api_key=api_keys.gemini,
            )

            judge = GroqJudge(api_key=api_keys.groq) if body.use_judge else None

            summary, results = run_eval(
                pipeline,
                cases,
                config_name=cfg.name,
                judge=judge,
                top_k=cfg.top_k,
                verbose=False,
                on_case_done=tracker.case_done,
            )

            save_run(get_connection(), summary, results)
            run_ids.append(summary.run_id)

        tracker.finish(run_ids[-1] if run_ids else "")
        tracker.update(message=f"Completed {len(run_ids)} configuration(s)")
        # Stash all run ids so the results page can show every config.
        tracker.update(config_name=",".join(run_ids))

    except Exception as e:
        tracker.fail(f"{type(e).__name__}: {e}")
    finally:
        _eval_slots.release()


@router.post("/{ws_id}/evaluate", response_model=JobStarted)
def evaluate(ws_id: str, body: EvaluateIn):
    """Start an evaluation. Returns immediately; poll /progress/{job_id}."""
    ws = _require_workspace(ws_id)

    if not ws.files():
        raise HTTPException(400, "Upload at least one document first")
    if not ws.questions:
        raise HTTPException(400, "Add at least one test question first")

    try:
        check_configs(len(body.configs))
    except LimitExceeded as e:
        raise HTTPException(400, str(e))

    names = [c.name for c in body.configs]
    if len(names) != len(set(names)):
        raise HTTPException(400, "Configuration names must be unique")

    try:
        api_keys = keymod.resolve(body.gemini_key, body.groq_key)
    except keymod.KeyError_ as e:
        raise HTTPException(400, str(e))

    # Free runs are only consumed when using the server's keys.
    if not api_keys.is_user_supplied:
        try:
            manager.record_run(ws)
        except LimitExceeded as e:
            raise HTTPException(429, str(e))

    job_id = secrets.token_urlsafe(8)
    tracker = ProgressTracker(
        total=len(ws.questions),
        config_total=len(body.configs),
    )
    with _jobs_lock:
        _jobs[job_id] = tracker

    threading.Thread(
        target=_run_job, args=(job_id, ws_id, body, api_keys), daemon=True
    ).start()

    return JobStarted(job_id=job_id, status="queued")


@router.get("/{ws_id}/progress/{job_id}")
def get_progress(ws_id: str, job_id: str):
    """Poll for live evaluation status."""
    with _jobs_lock:
        tracker = _jobs.get(job_id)
    if tracker is None:
        raise HTTPException(404, "Job not found")
    return tracker.snapshot()