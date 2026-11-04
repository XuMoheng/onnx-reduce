"""Pure data structures shared by the engine, passes, and ONNX adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Protocol, runtime_checkable

from onnx_reduce.edits import Candidate


@total_ordering
@dataclass(frozen=True)
class ModelCost:
    """A deterministic lexicographic cost used to accept reductions."""

    node_count: int
    byte_size: int
    output_count: int = 0
    input_count: int = 0
    initializer_count: int = 0
    tensor_elements: int = 0

    def __lt__(self, other):
        if not isinstance(other, ModelCost):
            return NotImplemented
        return self.key() < other.key()

    def key(self, objective="nodes"):
        if objective not in {"nodes", "bytes", "tensors"}:
            raise ValueError("unknown objective")
        if objective == "bytes":
            return (
                self.byte_size,
                self.node_count,
                self.tensor_elements,
                self.output_count,
                self.input_count,
                self.initializer_count,
            )
        if objective == "tensors":
            return (
                self.tensor_elements,
                self.node_count,
                self.byte_size,
                self.output_count,
                self.input_count,
                self.initializer_count,
            )
        return (
            self.node_count,
            self.tensor_elements,
            self.byte_size,
            self.output_count,
            self.input_count,
            self.initializer_count,
        )


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


@dataclass(frozen=True)
class RegionValue:
    producer: int | None
    consumers: tuple[int, ...]
    graph_output: bool
    type_complete: bool


@dataclass(frozen=True)
class RegionGraph:
    """Main-graph links, including outer captures, using stable node IDs."""

    node_ids: tuple[int, ...]
    modules: tuple[tuple[int, tuple[str, ...]], ...]
    values: tuple[RegionValue, ...]


@runtime_checkable
class RegionSearchModel(NodeSearchModel, Protocol):
    def region_graph(self) -> RegionGraph: ...


@runtime_checkable
class ArtifactValidationModel(Protocol):
    """Optional read-back validation of the delivered artifact."""

    def validate_artifact(self, artifact: ModelArtifact, mode: str) -> ValidationResult: ...


class EditApplicationError(RuntimeError):
    """An edit could not be applied to the current model."""
