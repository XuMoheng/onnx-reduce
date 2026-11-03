"""Safe cleanup of nodes that do not contribute to any graph output."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs
from onnx_reduce.passes.groups import deletion_groups, is_coarse


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
        for names in deletion_groups(model.unused_input_names()):
            yield Candidate((DropInputs(names),))
        for names in deletion_groups(model.unused_initializer_names()):
            yield Candidate((DropInitializers(names),))

    def generate_stage(self, model: Any, stage: str) -> Iterator[Candidate]:
        # After node edits, reclaim promoted inputs/weights before retrying dead
        # nodes. The scheduler bounds this detour across acceptances too.
        for names, edit_type in (
            (model.unused_input_names(), DropInputs),
            (model.unused_initializer_names(), DropInitializers),
        ):
            for group in deletion_groups(names):
                if stage == "cleanup":
                    yield Candidate((edit_type(group),))
                    break
                if is_coarse(len(group), len(names)) == (stage == "coarse"):
                    yield Candidate((edit_type(group),))
        if stage != "cleanup":
            dead = model.dead_node_indices()
            if dead and is_coarse(len(dead), model.node_count) == (stage == "coarse"):
                yield Candidate((DeleteNodes(dead),))
