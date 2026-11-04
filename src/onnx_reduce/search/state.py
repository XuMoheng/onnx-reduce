"""Shared version, coverage and statistics bookkeeping; no candidate selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict

from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.search.types import (
    PASS_SCOPE,
    CandidateFeedback,
    CoverageSummary,
    ScheduledAttempt,
)
from onnx_reduce.types import ModelAdapter


class SearchState:
    """Run-local bookkeeping composed by each independent search strategy."""

    scope = PASS_SCOPE

    def __init__(self, passes: Sequence[ReductionPass], model: ModelAdapter):
        self.passes = tuple(passes)
        self.model_version = 0
        self.base_digest = model.canonical_hash()
        self._completed = False
        self._inconclusive = 0
        self._stats: dict[str, dict] = {}
        self._sources: dict[str, Counter] = {}
        self._last_visit: dict[int, int] = {}
        self.phase_revisits = 0
        self.scale_reuses = 0
        self.deduplicated = 0

    def attempt(self, candidate, index, **metadata) -> ScheduledAttempt:
        if index in self._last_visit and self._last_visit[index] != self.model_version:
            self.phase_revisits += 1
        self._last_visit[index] = self.model_version
        return ScheduledAttempt(
            candidate,
            self.passes[index],
            index,
            self.model_version,
            self.base_digest,
            **metadata,
        )

    def record(self, attempt: ScheduledAttempt, feedback: CandidateFeedback) -> None:
        # Power-of-two buckets bound historical statistics even for arbitrary
        # successful scales. Exact scales remain available on each attempt.
        scale = 1 << (attempt.scale.bit_length() - 1) if attempt.scale else None
        key = f"{attempt.reduction_pass.name}/{attempt.direction}/{scale}"
        bucket = self._stats.setdefault(
            key,
            {
                "attempts": 0,
                "failure_check_calls": 0,
                "accepted": 0,
                "statuses": Counter(),
                "cache_hits": Counter(),
                "accepted_cost_delta": dict.fromkeys(asdict(feedback.base_cost), 0),
                "preparation_seconds": None,
                "evaluation_seconds": None,
            },
        )
        bucket["attempts"] += 1
        bucket["failure_check_calls"] += feedback.failure_check_calls
        bucket["statuses"][feedback.status] += 1
        if feedback.cache_source:
            bucket["cache_hits"][feedback.cache_source] += 1
        if feedback.status == "accepted":
            bucket["accepted"] += 1
            for name, before in asdict(feedback.base_cost).items():
                bucket["accepted_cost_delta"][name] += before - getattr(
                    feedback.candidate_cost, name
                )
        for name in ("preparation_seconds", "evaluation_seconds"):
            value = getattr(feedback, name)
            if value is not None:
                bucket[name] = (bucket[name] or 0.0) + value
        source = self._sources.setdefault(attempt.source, Counter())
        source["attempts"] += 1
        source["failure_check_calls"] += feedback.failure_check_calls
        source["accepted"] += feedback.status == "accepted"
        source["cache_hits"] += feedback.cache_source is not None
        for name in ("preparation_seconds", "evaluation_seconds"):
            value = getattr(feedback, name)
            if value is not None:
                source[name] += value
        if attempt.source == "coverage" and feedback.status == "inconclusive":
            self._inconclusive += 1

    def accept(self, model: ModelAdapter) -> None:
        self.model_version += 1
        self.base_digest = model.canonical_hash()
        self._completed = False
        self._inconclusive = 0

    def complete(self) -> None:
        self._completed = True

    def coverage(self) -> CoverageSummary:
        return CoverageSummary(
            self.base_digest,
            self.model_version,
            self.scope,
            self._completed,
            self._inconclusive,
        )

    def to_dict(self, strategy, coverage, *, phase_quantum=None) -> dict:
        return {
            "strategy": strategy,
            "phase_quantum": phase_quantum,
            "model_version": self.model_version,
            "phase_revisits": self.phase_revisits,
            "scale_reuses": self.scale_reuses,
            "deduplicated": self.deduplicated,
            "coverage": asdict(coverage),
            "sources": {
                name: {
                    **values,
                    "preparation_seconds": values.get("preparation_seconds"),
                    "evaluation_seconds": values.get("evaluation_seconds"),
                }
                for name, values in sorted(self._sources.items())
            },
            "buckets": {
                key: {
                    **values,
                    "statuses": dict(values["statuses"]),
                    "cache_hits": dict(values["cache_hits"]),
                    "acceptance_rate": values["accepted"] / values["attempts"],
                }
                for key, values in sorted(self._stats.items())
            },
        }
