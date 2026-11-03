"""Graph-output shrinking without implicit cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DropOutputs
from onnx_reduce.passes.groups import deletion_groups


class DropOutputsPass:
    name = "drop_outputs"

    def generate(self, model: Any) -> Iterator[Candidate]:
        # Preserve the old reverse-output preference within each batch.
        for names in deletion_groups(tuple(reversed(model.output_names)), keep_one=True):
            yield Candidate((DropOutputs(names),))
