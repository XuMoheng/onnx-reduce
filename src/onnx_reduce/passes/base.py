"""Generic reduction-pass interfaces kept independent of ONNX."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from onnx_reduce.edits import Candidate


@runtime_checkable
class ReductionPass(Protocol):
    """A deterministic source of candidates cheaper than its input model."""

    name: str

    def generate(self, model: Any) -> Iterator[Candidate]: ...
