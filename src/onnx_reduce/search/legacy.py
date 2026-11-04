"""Original pass order and restart-on-accept search policy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from onnx_reduce.edits import Candidate
from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.performance import measured, timed_candidates
from onnx_reduce.search.state import SearchState
from onnx_reduce.search.types import ScheduledAttempt
from onnx_reduce.types import ModelAdapter


class LegacyScheduler:
    """Preserve pass order, lazy iteration and restart-on-accept exactly."""

    name = "legacy"
    stop_reason = "fixed_point"

    def __init__(self, passes: Sequence[ReductionPass], model: ModelAdapter):
        self._state = SearchState(passes, model)
        self._pass_index = 0
        self._iterator: Iterator[Candidate] | None = None

    @measured("scheduler")
    def next_attempt(self, model: ModelAdapter) -> ScheduledAttempt | None:
        while self._pass_index < len(self._state.passes):
            if self._iterator is None:
                self._iterator = iter(timed_candidates(self._state.passes[self._pass_index], model))
            try:
                candidate = next(self._iterator)
            except StopIteration:
                self._iterator = None
                self._pass_index += 1
                continue
            return self._state.attempt(candidate, self._pass_index)
        self._state.complete()
        return None

    @measured("scheduler")
    def observe(self, attempt, feedback, accepted_model=None) -> None:
        self._state.record(attempt, feedback)
        if accepted_model is not None:
            self._state.accept(accepted_model)
            self._pass_index = 0
            self._iterator = None

    def coverage(self):
        return self._state.coverage()

    def to_dict(self):
        return self._state.to_dict(self.name, self.coverage())
