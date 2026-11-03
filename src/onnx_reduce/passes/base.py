"""Generic reduction-pass interfaces kept independent of ONNX."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from onnx_reduce.edits import Candidate
from onnx_reduce.types import NodeSearchModel


@runtime_checkable
class ReductionPass(Protocol):
    """A deterministic source of candidates cheaper than its input model."""

    name: str

    def generate(self, model: Any) -> Iterator[Candidate]: ...


@runtime_checkable
class NodeSearchPass(ReductionPass, Protocol):
    """Explicit opt-in to contiguous, rotatable main-graph partition search."""

    def node_regions(
        self, model: NodeSearchModel, size: int, *, start: int = 0
    ) -> Iterator[tuple[int, ...]]: ...


@runtime_checkable
class StagedPass(ReductionPass, Protocol):
    """Optional coarse/fine/cleanup candidates; generic passes remain fine-only."""

    def generate_stage(self, model: Any, stage: str) -> Iterator[Candidate]: ...
