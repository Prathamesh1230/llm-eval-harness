"""Progress reporting for long-running evaluations.

A full eval makes two API calls per question, so a 20-question run takes
minutes. Without progress, a user watching a web UI can't tell the
difference between "working" and "crashed".

The runner takes an optional callback and calls it after each case. The CLI
ignores it; the API uses it to update a job the frontend polls.
"""

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Optional

Phase = Literal["starting", "indexing", "evaluating", "done", "error"]


@dataclass
class Progress:
    """A snapshot of one eval run, safe to serialise straight to JSON."""

    phase: Phase = "starting"
    config_name: str = ""
    current: int = 0
    total: int = 0
    current_question: str = ""
    message: str = ""
    started_at: float = field(default_factory=time.time)

    # Set when the run finishes
    run_id: Optional[str] = None
    error: Optional[str] = None

    # For multi-config runs: which config we're on
    config_index: int = 0
    config_total: int = 1

    @property
    def percent(self) -> float:
        """Overall completion across every config in the run."""
        if self.phase == "done":
            return 100.0
        if self.total == 0 or self.config_total == 0:
            return 0.0
        per_config = 100.0 / self.config_total
        within = (self.current / self.total) * per_config
        return min(99.0, self.config_index * per_config + within)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def eta_s(self) -> Optional[float]:
        """Rough estimate from the average so far. None until 2 cases are done."""
        done = self.config_index * self.total + self.current
        if done < 2:
            return None
        total_cases = self.config_total * self.total
        return (self.elapsed_s / done) * (total_cases - done)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["percent"] = round(self.percent, 1)
        d["elapsed_s"] = round(self.elapsed_s, 1)
        eta = self.eta_s()
        d["eta_s"] = round(eta, 1) if eta is not None else None
        return d


class ProgressTracker:
    """Thread-safe holder for a Progress snapshot.

    The eval runs in a background thread while HTTP requests read the
    progress, so every update and read takes the lock.
    """

    def __init__(self, total: int = 0, config_total: int = 1):
        self._lock = threading.Lock()
        self._p = Progress(total=total, config_total=config_total)

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._p, key, value)

    def start_config(self, name: str, index: int, n_cases: int) -> None:
        with self._lock:
            self._p.config_name = name
            self._p.config_index = index
            self._p.total = n_cases
            self._p.current = 0
            self._p.phase = "indexing"
            self._p.message = f"Indexing documents for {name}"

    def case_done(self, index: int, question: str) -> None:
        with self._lock:
            self._p.phase = "evaluating"
            self._p.current = index
            self._p.current_question = question[:120]
            self._p.message = f"Scored {index} of {self._p.total}"

    def finish(self, run_id: str) -> None:
        with self._lock:
            self._p.phase = "done"
            self._p.run_id = run_id
            self._p.current = self._p.total
            self._p.message = "Complete"

    def fail(self, error: str) -> None:
        with self._lock:
            self._p.phase = "error"
            self._p.error = str(error)[:500]
            self._p.message = "Failed"

    def snapshot(self) -> dict:
        with self._lock:
            return self._p.to_dict()


# The callback shape the runner expects: (case_index, question) -> None
ProgressCallback = Callable[[int, str], None]


if __name__ == "__main__":
    passed = failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  OK    {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    print("Single config:")
    t = ProgressTracker(total=10, config_total=1)
    check("starts at 0%", t.snapshot()["percent"] == 0.0)

    t.start_config("baseline", 0, 10)
    check("phase is indexing", t.snapshot()["phase"] == "indexing")

    t.case_done(5, "How many days of leave?")
    snap = t.snapshot()
    check("halfway is 50%", snap["percent"] == 50.0)
    check("phase is evaluating", snap["phase"] == "evaluating")
    check("carries the question", "leave" in snap["current_question"])

    t.finish("abc123")
    snap = t.snapshot()
    check("done is 100%", snap["percent"] == 100.0)
    check("carries run_id", snap["run_id"] == "abc123")

    print("\nMulti config:")
    t = ProgressTracker(total=10, config_total=3)
    t.start_config("chunk_400", 0, 10)
    t.case_done(10, "q")
    check("first config done is ~33%", 33.0 <= t.snapshot()["percent"] <= 34.0)

    t.start_config("chunk_800", 1, 10)
    t.case_done(5, "q")
    check("second config half is ~50%", 49.0 <= t.snapshot()["percent"] <= 51.0)

    check("never reports 100 before finish", t.snapshot()["percent"] < 100.0)

    print("\nErrors and ETA:")
    t.fail("Gemini API key invalid")
    snap = t.snapshot()
    check("error phase set", snap["phase"] == "error")
    check("error message kept", "Gemini" in snap["error"])

    t2 = ProgressTracker(total=10, config_total=1)
    t2.start_config("x", 0, 10)
    check("no ETA before 2 cases", t2.snapshot()["eta_s"] is None)
    time.sleep(0.05)
    t2.case_done(2, "q")
    check("ETA appears after 2 cases", t2.snapshot()["eta_s"] is not None)

    print(f"\n{passed} passed, {failed} failed")