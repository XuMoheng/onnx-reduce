"""Reduction configuration and option compatibility checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    objective: str = "nodes"
    combination_depth: int = 2
    combination_limit: int = 256
    validation_mode: str = "strict"
    max_candidate_check_calls: int | None = None
    time_budget_seconds: float | None = None
    cache_enabled: bool = True
    collect_performance: bool = False
    target_mode: str = "baseline"
    signature_regex: str | None = None
    scheduler: str = "legacy"
    goal: str | None = None
    finalize_from: str | None = None
    model_edit_mode: str = "ir"
    search_policy: str = "restart"
    history_order: str = "off"
    direction_order: str = "fixed"
    checkpoint_reuse: bool = True
    validation_cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.objective not in {"nodes", "bytes", "tensors"}:
            raise ValueError("unknown objective")
        if not 1 <= self.combination_depth <= 3 or self.combination_limit < 0:
            raise ValueError("combination depth must be 1..3 and limit nonnegative")
        for name, options in (
            ("model_edit_mode", {"ir", "auto"}),
            ("search_policy", {"restart", "cursor"}),
            ("history_order", {"off", "defer"}),
            ("direction_order", {"fixed", "adaptive"}),
        ):
            if getattr(self, name) not in options:
                raise ValueError(f"unknown {name}: {getattr(self, name)}")
        if self.search_policy != "restart" and self.scheduler != "staged":
            raise ValueError("search_policy=cursor requires scheduler=staged")
        if (self.history_order != "off" or self.direction_order != "fixed") and (
            self.scheduler != "staged" or self.search_policy != "cursor"
        ):
            raise ValueError("history/direction ordering requires staged search_policy=cursor")
        if self.scheduler not in {"legacy", "stateful", "staged"}:
            raise ValueError(f"unknown scheduler: {self.scheduler}")
        if self.goal not in {None, "fast", "complete"}:
            raise ValueError(f"unknown goal: {self.goal}")
        if (self.goal is not None or self.finalize_from is not None) and self.scheduler != "staged":
            raise ValueError("goal and finalize_from require scheduler=staged")
        if self.finalize_from is not None and self.goal == "fast":
            raise ValueError("finalize_from requires goal=complete")
        if self.validation_mode not in {"strict", "structural", "permissive"}:
            raise ValueError(f"unknown validation mode: {self.validation_mode}")
        if self.target_mode not in {"baseline", "script"}:
            raise ValueError(f"unknown target mode: {self.target_mode}")
        if self.max_candidate_check_calls is not None and (
            type(self.max_candidate_check_calls) is not int or self.max_candidate_check_calls < 0
        ):
            raise ValueError("max_candidate_check_calls must be a non-negative integer")
        if self.time_budget_seconds is not None and (
            not math.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0
        ):
            raise ValueError("time_budget_seconds must be finite and greater than zero")
        if self.signature_regex is not None:
            re.compile(self.signature_regex)
