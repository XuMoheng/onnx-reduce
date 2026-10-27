"""Graph-output shrinking without implicit cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DropOutputs


class DropOutputsPass:
    name = "drop_outputs"

    def generate(self, model: Any) -> Iterator[Candidate]:
        names = model.output_names
        # An ONNX M1 candidate always keeps at least one output.  Dropping in
        # reverse order makes the candidate sequence deterministic.
        if len(names) > 1:
            for name in reversed(names):
                yield Candidate((DropOutputs((name,)),))
