"""Graph-output shrinking without implicit cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DropOutputs
from onnx_reduce.passes.base import RemovalGroup
from onnx_reduce.passes.groups import deletion_groups, is_coarse


class DropOutputsPass:
    name = "drop_outputs"
    requires_node_ids = False

    def removal_groups(self, model):
        yield RemovalGroup(
            "outputs", tuple(reversed(model.output_names)), DropOutputs, keep_one=True
        )

    def generate(self, model: Any) -> Iterator[Candidate]:
        # Preserve the old reverse-output preference within each batch.
        for names in deletion_groups(tuple(reversed(model.output_names)), keep_one=True):
            yield Candidate((DropOutputs(names),))

    def generate_stage(self, model: Any, stage: str) -> Iterator[Candidate]:
        if stage == "cleanup":
            return
        for candidate in self.generate(model):
            coarse = is_coarse(len(candidate.edits[0].names), len(model.output_names))
            if coarse == (stage == "coarse"):
                yield candidate
