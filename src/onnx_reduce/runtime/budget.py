"""Soft search budgets; mandatory initial and final checks are outside the call quota."""

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    started: float
    time_limit: float | None
    call_limit: int | None

    def allows_time(self) -> bool:
        # Generation, editing, validation and an already-started check can overrun.
        return self.time_limit is None or time.monotonic() - self.started < self.time_limit

    def allows_call(self, candidate_calls: int) -> bool:
        return self.call_limit is None or candidate_calls < self.call_limit
