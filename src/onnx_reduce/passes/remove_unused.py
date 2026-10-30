"""Safe cleanup of nodes that do not contribute to any graph output."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs


class RemoveUnusedPass:
    name = "remove_unused"
    # Dead nodes can still be the sole carrier of a compiler bug.  Unlike an
    # optimizer, the reducer must never remove them without running the failure check,
    # so this cleanup is gated by failure checks.

    def generate(self, model: Any) -> Iterator[Candidate]:
        # Each cleanup is independently gated by failure checks.
        dead_indices = model.dead_node_indices()
        if dead_indices:
            yield Candidate((DeleteNodes(tuple(dead_indices)),))
        for name in model.unused_input_names():
            yield Candidate((DropInputs((name,)),))
        for name in model.unused_initializer_names():
            yield Candidate((DropInitializers((name,)),))
