"""Per-visitor workspaces.

On a public deployment, visitors upload their own documents and write their
own test questions. Each gets an isolated workspace so nobody can see, break,
or overwrite anyone else's work.

Isolation covers three things:
  - files      each workspace has its own directory under WORKSPACE_ROOT
  - vectors    each gets its own Chroma collection name
  - test set   questions live in the workspace, not the shared golden.yaml

Security notes:
  - IDs come from secrets.token_urlsafe so they can't be guessed
  - uploaded filenames are sanitized, and every write path is verified to
    resolve inside the workspace (blocks '../../.env' style attacks)
"""

import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from evalharness.workspace.limits import (
    MAX_RUNS_PER_DAY,
    MAX_WORKSPACES_PER_DAY,
    WORKSPACE_TTL_HOURS,
    LimitExceeded,
    check_file,
    check_upload_batch,
)

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "workspaces"))
META_FILE = "workspace.json"
DOCS_DIR = "docs"
GOLDEN_FILE = "golden.json"

_lock = threading.Lock()


# ------------------------------------------------------------ filenames

def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to something safe to write.

    Strips any directory component, then allows only letters, digits,
    dot, dash, underscore and space. '../../.env' becomes '.env', which
    then fails the extension check in limits.check_file.
    """
    base = Path(name).name                       # drop directories
    base = re.sub(r"[^A-Za-z0-9._\- ]", "_", base).strip()
    base = base.lstrip(".") or "file"            # no hidden files
    return base[:100]                            # keep paths short


# ------------------------------------------------------------ model

@dataclass
class Workspace:
    id: str
    created_at: datetime
    root: Path
    runs_used: int = 0
    questions: list[dict] = field(default_factory=list)

    @property
    def docs_dir(self) -> Path:
        return self.root / DOCS_DIR

    @property
    def collection_prefix(self) -> str:
        """Chroma collection names are per-workspace, so two visitors
        indexing at once never touch the same vectors."""
        return f"ws_{self.id}"

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(hours=WORKSPACE_TTL_HOURS)

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def files(self) -> list[dict]:
        if not self.docs_dir.exists():
            return []
        return [
            {"name": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(self.docs_dir.iterdir())
            if p.is_file()
        ]

    def resolve_in_docs(self, filename: str) -> Path:
        """Return a path inside docs_dir, or raise if it would escape."""
        candidate = (self.docs_dir / filename).resolve()
        docs_root = self.docs_dir.resolve()
        if not candidate.is_relative_to(docs_root):
            raise LimitExceeded("Invalid filename")
        return candidate

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "runs_used": self.runs_used,
            "runs_remaining": max(0, MAX_RUNS_PER_DAY - self.runs_used),
            "files": self.files(),
            "n_questions": len(self.questions),
        }


# ------------------------------------------------------------ persistence

def _meta_path(ws_id: str) -> Path:
    return WORKSPACE_ROOT / ws_id / META_FILE


def _save(ws: Workspace) -> None:
    ws.root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": ws.id,
        "created_at": ws.created_at.isoformat(),
        "runs_used": ws.runs_used,
    }
    (ws.root / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ws.root / GOLDEN_FILE).write_text(json.dumps(ws.questions, indent=2), encoding="utf-8")


def _load(ws_id: str) -> Optional[Workspace]:
    meta_file = _meta_path(ws_id)
    if not meta_file.exists():
        return None

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        root = WORKSPACE_ROOT / ws_id
        golden_file = root / GOLDEN_FILE
        questions = json.loads(golden_file.read_text(encoding="utf-8")) if golden_file.exists() else []

        return Workspace(
            id=meta["id"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            root=root,
            runs_used=meta.get("runs_used", 0),
            questions=questions,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# ------------------------------------------------------------ rate limiting

_recent_creations: dict[str, list[float]] = {}


def _check_creation_rate(client_key: str) -> None:
    """Stop one visitor from creating unlimited workspaces.

    Keyed on client IP. Imperfect (shared networks, proxies) but enough
    to stop casual abuse without adding accounts.
    """
    now = time.time()
    day_ago = now - 86400

    with _lock:
        timestamps = [t for t in _recent_creations.get(client_key, []) if t > day_ago]
        if len(timestamps) >= MAX_WORKSPACES_PER_DAY:
            raise LimitExceeded(
                f"You've created {MAX_WORKSPACES_PER_DAY} workspaces today. Try again tomorrow."
            )
        timestamps.append(now)
        _recent_creations[client_key] = timestamps


# ------------------------------------------------------------ public api

def create(client_key: str = "anonymous") -> Workspace:
    """Create a fresh, isolated workspace."""
    _check_creation_rate(client_key)

    ws_id = secrets.token_urlsafe(12)          # unguessable, ~16 chars
    ws = Workspace(
        id=ws_id,
        created_at=datetime.now(timezone.utc),
        root=WORKSPACE_ROOT / ws_id,
    )
    ws.docs_dir.mkdir(parents=True, exist_ok=True)
    _save(ws)
    return ws


def get(ws_id: str) -> Optional[Workspace]:
    """Load a workspace, or None if it doesn't exist or has expired."""
    if not ws_id or not re.fullmatch(r"[A-Za-z0-9_\-]{8,64}", ws_id):
        return None

    ws = _load(ws_id)
    if ws is None:
        return None
    if ws.is_expired:
        delete(ws_id)
        return None
    return ws


def add_files(ws: Workspace, uploads: list[tuple[str, bytes]]) -> list[str]:
    """Write uploaded files into the workspace. Returns the saved names."""
    existing = ws.files()
    total_bytes = sum(len(data) for _, data in uploads)
    check_upload_batch(len(existing), len(uploads), total_bytes)

    saved: list[str] = []
    for raw_name, data in uploads:
        name = safe_filename(raw_name)
        check_file(name, len(data))
        path = ws.resolve_in_docs(name)
        path.write_bytes(data)
        saved.append(name)

    return saved


def remove_file(ws: Workspace, filename: str) -> bool:
    path = ws.resolve_in_docs(safe_filename(filename))
    if path.is_file():
        path.unlink()
        return True
    return False


def set_questions(ws: Workspace, questions: list[dict]) -> None:
    ws.questions = questions
    _save(ws)


def record_run(ws: Workspace) -> None:
    """Count a run against the workspace's daily allowance."""
    if ws.runs_used >= MAX_RUNS_PER_DAY:
        raise LimitExceeded(
            f"This workspace has used its {MAX_RUNS_PER_DAY} runs. Create a new one or use your own API keys."
        )
    ws.runs_used += 1
    _save(ws)


def delete(ws_id: str) -> bool:
    root = WORKSPACE_ROOT / ws_id
    if root.exists() and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)
        return True
    return False


def cleanup_expired() -> int:
    """Delete workspaces past their TTL. Called periodically by the API."""
    if not WORKSPACE_ROOT.exists():
        return 0

    removed = 0
    for path in WORKSPACE_ROOT.iterdir():
        if not path.is_dir():
            continue
        ws = _load(path.name)
        if ws is None or ws.is_expired:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


if __name__ == "__main__":
    import tempfile

    passed = failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  OK    {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    # Run against a temp directory so the real workspaces folder is untouched.
    with tempfile.TemporaryDirectory() as tmp:
        WORKSPACE_ROOT = Path(tmp)

        print("Filename sanitizing:")
        check("strips directories", safe_filename("../../.env") == "env")
        check("strips path separators", safe_filename("a/b/c.pdf") == "c.pdf")
        check("keeps normal names", safe_filename("HR Policy.pdf") == "HR Policy.pdf")
        check("replaces odd chars", safe_filename("re;port$.txt") == "re_port_.txt")
        check("handles empty", safe_filename("...") == "file")

        print("\nWorkspace lifecycle:")
        ws = create(client_key="test-1")
        check("id is unguessable length", len(ws.id) >= 12)
        check("docs dir created", ws.docs_dir.is_dir())
        check("collection is namespaced", ws.collection_prefix.startswith("ws_"))

        loaded = get(ws.id)
        check("loads by id", loaded is not None and loaded.id == ws.id)
        check("rejects bad id format", get("../etc/passwd") is None)
        check("missing id returns None", get("doesnotexist123") is None)

        print("\nFiles:")
        add_files(ws, [("policy.pdf", b"x" * 500), ("notes.txt", b"y" * 300)])
        check("two files saved", len(ws.files()) == 2)
        check("file lands inside workspace", (ws.docs_dir / "policy.pdf").exists())

        try:
            add_files(ws, [("evil.exe", b"z" * 100)])
            check("blocks bad extension", False)
        except LimitExceeded:
            check("blocks bad extension", True)

        try:
            ws.resolve_in_docs("../../escape.txt")
            check("blocks path traversal", False)
        except LimitExceeded:
            check("blocks path traversal", True)

        check("removes a file", remove_file(ws, "notes.txt") and len(ws.files()) == 1)

        print("\nQuestions and runs:")
        set_questions(ws, [{"id": "q1", "question": "test?", "expected_answer": "yes"}])
        check("questions persist", len(get(ws.id).questions) == 1)

        for _ in range(MAX_RUNS_PER_DAY):
            record_run(ws)
        try:
            record_run(ws)
            check("blocks run limit", False)
        except LimitExceeded:
            check("blocks run limit", True)

        print("\nIsolation and cleanup:")
        ws2 = create(client_key="test-2")
        check("ids differ", ws.id != ws2.id)
        check("collections differ", ws.collection_prefix != ws2.collection_prefix)
        check("files are separate", len(ws2.files()) == 0)

        check("delete removes workspace", delete(ws2.id) and get(ws2.id) is None)

        # Force expiry by backdating creation.
        ws.created_at = datetime.now(timezone.utc) - timedelta(hours=WORKSPACE_TTL_HOURS + 1)
        _save(ws)
        check("expired workspace is gone", get(ws.id) is None)

        print("\nCreation rate limit:")
        try:
            for _ in range(MAX_WORKSPACES_PER_DAY + 2):
                create(client_key="spammer")
            check("blocks workspace spam", False)
        except LimitExceeded:
            check("blocks workspace spam", True)

    print(f"\n{passed} passed, {failed} failed")