"""Search policy contract, independent of editing, checks and persistence."""

from typing import Protocol

from onnx_reduce.search.types import CandidateFeedback, CoverageSummary, ScheduledAttempt
from onnx_reduce.types import ModelAdapter


class Scheduler(Protocol):
    name: str
    stop_reason: str

    def next_attempt(self, model: ModelAdapter) -> ScheduledAttempt | None: ...

    def observe(
        self,
        attempt: ScheduledAttempt,
        feedback: CandidateFeedback,
        accepted_model: ModelAdapter | None = None,
    ) -> None: ...

    def coverage(self) -> CoverageSummary: ...

    def to_dict(self) -> dict: ...
