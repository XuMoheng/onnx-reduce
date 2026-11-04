"""Configuration by responsibility, with a compatible flat EngineConfig facade."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class SearchConfig:
    scheduler: str = "legacy"
    combination_depth: int = 2
    combination_limit: int = 256
    goal: str | None = None
    finalize_from: str | None = None

    def __post_init__(self):
        if self.scheduler not in {"legacy", "stateful", "staged"}:
            raise ValueError(f"unknown scheduler: {self.scheduler}")
        if not 1 <= self.combination_depth <= 3 or self.combination_limit < 0:
            raise ValueError("combination depth must be 1..3 and limit nonnegative")
        if self.goal not in {None, "fast", "complete"}:
            raise ValueError(f"unknown goal: {self.goal}")
        if (self.goal is not None or self.finalize_from is not None) and self.scheduler != "staged":
            raise ValueError("goal and finalize_from require scheduler=staged")
        if self.finalize_from is not None and self.goal == "fast":
            raise ValueError("finalize_from requires goal=complete")


@dataclass(frozen=True)
class ExecutionConfig:
    objective: str = "nodes"
    max_candidate_check_calls: int | None = None
    time_budget_seconds: float | None = None
    cache_enabled: bool = True
    collect_performance: bool = False

    def __post_init__(self):
        if self.objective not in {"nodes", "bytes", "tensors"}:
            raise ValueError("unknown objective")
        if self.max_candidate_check_calls is not None and (
            type(self.max_candidate_check_calls) is not int or self.max_candidate_check_calls < 0
        ):
            raise ValueError("max_candidate_check_calls must be a non-negative integer")
        if self.time_budget_seconds is not None and (
            not math.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0
        ):
            raise ValueError("time_budget_seconds must be finite and greater than zero")


@dataclass(frozen=True)
class CheckConfig:
    validation_mode: str = "strict"
    target_mode: str = "baseline"
    signature_regex: str | None = None

    def __post_init__(self):
        if self.validation_mode not in {"strict", "structural", "permissive"}:
            raise ValueError(f"unknown validation mode: {self.validation_mode}")
        if self.target_mode not in {"baseline", "script"}:
            raise ValueError(f"unknown target mode: {self.target_mode}")
        if self.signature_regex is not None:
            re.compile(self.signature_regex)


@dataclass(frozen=True)
class ExperimentConfig:
    model_edit_mode: str = "ir"
    search_policy: str = "restart"
    history_order: str = "off"
    direction_order: str = "fixed"
    checkpoint_reuse: bool = True
    validation_cache_enabled: bool = True

    def __post_init__(self):
        for name, options in (
            ("model_edit_mode", {"ir", "auto"}),
            ("search_policy", {"restart", "cursor"}),
            ("history_order", {"off", "defer"}),
            ("direction_order", {"fixed", "adaptive"}),
        ):
            if getattr(self, name) not in options:
                raise ValueError(f"unknown {name}: {getattr(self, name)}")


_GROUPS = {
    "search": SearchConfig,
    "execution": ExecutionConfig,
    "checking": CheckConfig,
    "experiments": ExperimentConfig,
}
_OWNERS = {item.name: group for group, cls in _GROUPS.items() for item in fields(cls)}


@dataclass(frozen=True, init=False)
class EngineConfig:
    """Compose narrow configs; accept legacy keywords and dataclasses.replace.

    Runtime services receive only their relevant groups. to_dict() preserves
    the existing flat task/replay manifest, independently of internal layout.
    """

    search: SearchConfig
    execution: ExecutionConfig
    checking: CheckConfig
    experiments: ExperimentConfig

    def __init__(self, *, search=None, execution=None, checking=None, experiments=None, **legacy):
        supplied = {
            "search": search,
            "execution": execution,
            "checking": checking,
            "experiments": experiments,
        }
        unknown = legacy.keys() - _OWNERS.keys()
        if unknown:
            raise TypeError(f"unknown engine options: {', '.join(sorted(unknown))}")
        for name, cls in _GROUPS.items():
            value = supplied[name]
            if isinstance(value, dict):
                value = cls(**value)
            elif value is None:
                value = cls()
            elif not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__}")
            overrides = {key: value for key, value in legacy.items() if _OWNERS[key] == name}
            object.__setattr__(self, name, replace(value, **overrides) if overrides else value)
        if self.experiments.search_policy != "restart" and self.search.scheduler != "staged":
            raise ValueError("search_policy=cursor requires scheduler=staged")
        if (
            self.experiments.history_order != "off" or self.experiments.direction_order != "fixed"
        ) and (self.search.scheduler != "staged" or self.experiments.search_policy != "cursor"):
            raise ValueError("history/direction ordering requires staged search_policy=cursor")

    def __getattr__(self, name):
        # Compatibility for existing callers; no duplicated mutable configuration.
        if name in _OWNERS:
            return getattr(object.__getattribute__(self, _OWNERS[name]), name)
        raise AttributeError(name)

    def to_dict(self):
        return {
            key: value for name in _GROUPS for key, value in asdict(getattr(self, name)).items()
        }
