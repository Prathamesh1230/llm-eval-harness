"""The adapter contract.

Anything the harness evaluates must implement this one method. The runner,
metrics, judge, storage, and dashboard only ever see this interface, which
is why the harness can test a local pipeline, a remote API, or a team's own
Python code without any of the scoring code changing.
"""

from typing import Protocol, runtime_checkable

from evalharness.schemas import PipelineOutput


@runtime_checkable
class Target(Protocol):
    """A system under test.

    answer() takes a question and returns what the system produced:
    the answer text, the chunks it retrieved (if any), latency, and tokens.
    Systems that don't expose retrieved chunks return an empty list, and
    the harness scores what it can.
    """

    def answer(self, question: str, test_case_id: str = "adhoc") -> PipelineOutput:
        ...