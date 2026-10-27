"""Serializable edit operations used to describe reduction candidates.

Edits are deliberately closed and small.  The engine only stores and replays
them; :mod:`onnx_reduce.model` knows how to apply them to an ONNX model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeleteNodes:
    """Delete nodes at the given indices in the candidate base's topology."""

    indices: tuple[int, ...]


@dataclass(frozen=True)
class DropOutputs:
    """Remove named graph outputs without deleting their producers."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class DropInputs:
    names: tuple[str, ...]


@dataclass(frozen=True)
class DropInitializers:
    names: tuple[str, ...]


Edit = DeleteNodes | DropOutputs | DropInputs | DropInitializers


@dataclass(frozen=True)
class Candidate:
    """An ordered, replayable sequence of edits relative to a model base."""

    edits: tuple[Edit, ...]
    base_digest: str | None = None
    pass_name: str | None = None
    sequence_number: int | None = None

    def describe(self) -> str:
        if not self.edits:
            return "empty candidate"
        return ", ".join(type(edit).__name__ for edit in self.edits)

    def to_dict(self) -> dict:
        return {
            "base_digest": self.base_digest,
            "pass_name": self.pass_name,
            "sequence_number": self.sequence_number,
            "edits": [{"op": type(edit).__name__, **asdict(edit)} for edit in self.edits],
        }
