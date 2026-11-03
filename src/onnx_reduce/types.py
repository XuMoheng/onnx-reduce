"""Pure data structures shared by the engine, passes, and ONNX adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from onnx_reduce.edits import Candidate


@dataclass(frozen=True, order=True)
class ModelCost:
    """A deterministic lexicographic cost used to accept reductions."""

    node_count: int
    byte_size: int
    output_count: int = 0
    input_count: int = 0
    initializer_count: int = 0


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None
    validation_mode: str | None = None
    custom_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelArtifact:
    model_path: Path
    resources: tuple[Path, ...]
    digest: str
    total_bytes: int


class ModelAdapter(Protocol):
    """The engine-facing interface to an opaque candidate model."""

    def copy(self) -> ModelAdapter: ...

    def apply_candidate(self, candidate: Candidate) -> ModelAdapter: ...

    def validate(self, mode: str) -> ValidationResult: ...

    def cost(self) -> ModelCost: ...

    def canonical_hash(self) -> str: ...

    def save(self, path: str | Path) -> None: ...

    def materialize(self, path: str | Path) -> ModelArtifact: ...


@runtime_checkable
class NodeSearchModel(Protocol):
    """Optional main-graph identity view; IDs survive edits within one lineage."""

    @property
    def node_ids(self) -> tuple[int, ...]: ...

    @property
    def id_to_index(self) -> Mapping[int, int]: ...


class EditApplicationError(RuntimeError):
    """An edit could not be applied to the current model."""
