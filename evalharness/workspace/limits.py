"""Resource limits for the public deployment.

Every value can be raised with an environment variable, so a company
self-hosting the harness isn't bound by limits that exist to protect a
free-tier server and a personal API budget.

These are the difference between a demo and a service that survives
contact with the public internet.
"""

import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Uploads
MAX_FILES = _int_env("MAX_FILES", 5)
MAX_FILE_MB = _int_env("MAX_FILE_MB", 5)
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TOTAL_MB = _int_env("MAX_TOTAL_MB", 15)
MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Test set
MAX_QUESTIONS = _int_env("MAX_QUESTIONS", 20)
MAX_QUESTION_CHARS = _int_env("MAX_QUESTION_CHARS", 500)
MAX_ANSWER_CHARS = _int_env("MAX_ANSWER_CHARS", 2000)

# Evaluation
MAX_CONFIGS_PER_RUN = _int_env("MAX_CONFIGS_PER_RUN", 3)
MAX_CONCURRENT_EVALS = _int_env("MAX_CONCURRENT_EVALS", 2)

# Rate limiting, per visitor
MAX_RUNS_PER_DAY = _int_env("MAX_RUNS_PER_DAY", 3)
MAX_WORKSPACES_PER_DAY = _int_env("MAX_WORKSPACES_PER_DAY", 5)

# Cleanup
WORKSPACE_TTL_HOURS = _int_env("WORKSPACE_TTL_HOURS", 24)


class LimitExceeded(ValueError):
    """Raised when a request would exceed a configured limit.

    Carries a message meant to be shown directly to the user, so the API
    can pass it through without leaking internals.
    """


def check_file(filename: str, size_bytes: int) -> None:
    """Validate one uploaded file before it's written to disk."""
    from pathlib import PurePosixPath

    ext = PurePosixPath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise LimitExceeded(f"'{ext or 'no extension'}' is not supported. Allowed: {allowed}")

    if size_bytes > MAX_FILE_BYTES:
        mb = size_bytes / 1024 / 1024
        raise LimitExceeded(f"{filename} is {mb:.1f} MB; the limit is {MAX_FILE_MB} MB per file")

    if size_bytes == 0:
        raise LimitExceeded(f"{filename} is empty")


def check_upload_batch(n_existing: int, n_new: int, total_bytes: int) -> None:
    """Validate an upload against the workspace's current contents."""
    if n_existing + n_new > MAX_FILES:
        raise LimitExceeded(
            f"That would be {n_existing + n_new} files; the limit is {MAX_FILES} per workspace"
        )
    if total_bytes > MAX_TOTAL_BYTES:
        mb = total_bytes / 1024 / 1024
        raise LimitExceeded(f"Total upload is {mb:.1f} MB; the limit is {MAX_TOTAL_MB} MB")


def check_questions(n: int) -> None:
    if n == 0:
        raise LimitExceeded("Add at least one test question")
    if n > MAX_QUESTIONS:
        raise LimitExceeded(f"{n} questions; the limit is {MAX_QUESTIONS} per evaluation")


def check_configs(n: int) -> None:
    if n == 0:
        raise LimitExceeded("Pick at least one configuration to test")
    if n > MAX_CONFIGS_PER_RUN:
        raise LimitExceeded(f"{n} configurations; the limit is {MAX_CONFIGS_PER_RUN} per run")


def summary() -> dict:
    """Current limits, for the frontend to display so users aren't surprised."""
    return {
        "max_files": MAX_FILES,
        "max_file_mb": MAX_FILE_MB,
        "max_total_mb": MAX_TOTAL_MB,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_questions": MAX_QUESTIONS,
        "max_configs_per_run": MAX_CONFIGS_PER_RUN,
        "max_runs_per_day": MAX_RUNS_PER_DAY,
        "workspace_ttl_hours": WORKSPACE_TTL_HOURS,
    }


if __name__ == "__main__":
    passed = failed = 0

    def expect_block(label: str, fn) -> None:
        global passed, failed
        try:
            fn()
            failed += 1
            print(f"  FAIL  {label} (should have been blocked)")
        except LimitExceeded as e:
            passed += 1
            print(f"  OK    {label}  ({e})")

    def expect_allow(label: str, fn) -> None:
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  OK    {label}")
        except LimitExceeded as e:
            failed += 1
            print(f"  FAIL  {label} (blocked: {e})")

    print("File checks:")
    expect_allow("valid pdf", lambda: check_file("policy.pdf", 1000))
    expect_block("wrong extension", lambda: check_file("virus.exe", 1000))
    expect_block("no extension", lambda: check_file("README", 1000))
    expect_block("too large", lambda: check_file("big.pdf", MAX_FILE_BYTES + 1))
    expect_block("empty file", lambda: check_file("empty.txt", 0))

    print("\nBatch checks:")
    expect_allow("within limits", lambda: check_upload_batch(2, 2, 1000))
    expect_block("too many files", lambda: check_upload_batch(4, 3, 1000))
    expect_block("total too large", lambda: check_upload_batch(0, 1, MAX_TOTAL_BYTES + 1))

    print("\nQuestion and config checks:")
    expect_allow("5 questions", lambda: check_questions(5))
    expect_block("zero questions", lambda: check_questions(0))
    expect_block("too many questions", lambda: check_questions(MAX_QUESTIONS + 1))
    expect_allow("2 configs", lambda: check_configs(2))
    expect_block("too many configs", lambda: check_configs(MAX_CONFIGS_PER_RUN + 1))

    print(f"\n{passed} passed, {failed} failed")