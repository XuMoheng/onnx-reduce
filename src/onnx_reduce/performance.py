"""Opt-in, run-scoped measurements; never participate in reduction decisions."""

from __future__ import annotations

import resource
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from time import perf_counter

_ACTIVE: ContextVar[PerformanceStats | None] = ContextVar("reduction_performance", default=None)


@dataclass
class PerformanceStats:
    stages: dict[str, dict] = field(default_factory=dict)
    counters: Counter = field(default_factory=Counter)
    samples: dict[str, dict] = field(default_factory=dict)
    progress: list[dict] = field(default_factory=list)
    progress_compactions: int = 0
    _stack: list = field(default_factory=list)
    _started: float = field(default_factory=lambda: perf_counter())
    _elapsed: float | None = None
    _peak_rss: int | None = None

    def record_progress(self, cost, *, stage, reason, checks, force=False):
        elapsed = perf_counter() - self._started
        if not force and self.progress and elapsed - self.progress[-1]["elapsed_seconds"] < 5:
            return None
        point = {
            "elapsed_seconds": elapsed,
            "stage": stage,
            "reason": reason,
            "failure_check_calls": checks,
            "cost": asdict(cost),
        }
        if len(self.progress) >= 2048:
            self.progress = self.progress[:1] + self.progress[2::2]
            self.progress_compactions += 1
        self.progress.append(point)
        return point

    def finish(self) -> None:
        self._elapsed = perf_counter() - self._started
        self._peak_rss = self.peak_rss()

    @staticmethod
    def peak_rss() -> int:
        # macOS reports bytes; Linux reports KiB. This is the process lifetime
        # high-water mark, including native ONNX/IR allocations, excluding children.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "progress": list(self.progress),
            "progress_compactions": self.progress_compactions,
            "elapsed_seconds": self._elapsed
            if self._elapsed is not None
            else perf_counter() - self._started,
            "stages": {name: dict(values) for name, values in sorted(self.stages.items())},
            "counters": dict(sorted(self.counters.items())),
            "samples": {name: dict(values) for name, values in sorted(self.samples.items())},
            "peak_rss_bytes": self._peak_rss if self._peak_rss is not None else self.peak_rss(),
            "peak_rss_scope": "reducer process lifetime; excludes child processes",
        }


def current_performance() -> PerformanceStats | None:
    return _ACTIVE.get()


@contextmanager
def collect_performance(enabled: bool) -> Iterator[PerformanceStats | None]:
    stats = PerformanceStats() if enabled else None
    token = _ACTIVE.set(stats)
    try:
        yield stats
    finally:
        if stats is not None:
            stats.finish()
        _ACTIVE.reset(token)


@contextmanager
def measure(name: str) -> Iterator[None]:
    stats = _ACTIVE.get()
    if stats is None:
        yield
        return
    frame = [perf_counter(), 0.0]
    stats._stack.append(frame)
    try:
        yield
    finally:
        elapsed = perf_counter() - frame[0]
        stats._stack.pop()
        if stats._stack:
            stats._stack[-1][1] += elapsed
        stage = stats.stages.setdefault(
            name, {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0}
        )
        stage["calls"] += 1
        stage["inclusive_seconds"] += elapsed
        stage["exclusive_seconds"] += max(0.0, elapsed - frame[1])


def measured(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _ACTIVE.get() is None:
                return function(*args, **kwargs)
            with measure(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def count(name: str, amount: int = 1) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        stats.counters[name] += amount


def sample(name: str, value: int) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        values = stats.samples.setdefault(name, {"count": 0, "sum": 0, "max": 0})
        values["count"] += 1
        values["sum"] += value
        values["max"] = max(values["max"], value)


def timed_candidates(reduction_pass, model):
    if _ACTIVE.get() is None:
        yield from reduction_pass.generate(model)
        return
    # Time actual lazy iteration, including setup and exhausted generators.
    with measure("candidate_generation"), measure(f"generate/{reduction_pass.name}"):
        candidates = iter(reduction_pass.generate(model))
    while True:
        with measure("candidate_generation"), measure(f"generate/{reduction_pass.name}"):
            try:
                candidate = next(candidates)
            except StopIteration:
                return
        yield candidate
