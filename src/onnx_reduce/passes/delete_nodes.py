"""Graph-aware ddmin-style node deletion."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.passes.groups import is_coarse
from onnx_reduce.types import NodeSearchModel


def _granularities(node_count: int) -> tuple[int, ...]:
    if node_count <= 0:
        return ()
    if node_count == 1:
        return (1,)
    sizes = []
    size = node_count // 2
    while size >= 1:
        sizes.append(size)
        size //= 2
    return tuple(dict.fromkeys(sizes))


class DeleteNodesPass:
    name = "delete_nodes"

    def node_regions(
        self, model: NodeSearchModel, size: int, *, start: int = 0
    ) -> Iterator[tuple[int, ...]]:
        """Yield current partitions, rotated to the block containing the resume anchor."""

        ids = model.node_ids
        count = (len(ids) + size - 1) // size
        for offset in range(count):
            begin = ((start + offset) % count) * size
            yield ids[begin : begin + size]

    def generate(self, model: Any) -> Iterator[Candidate]:
        count = model.node_count
        for granularity in _granularities(count):
            for start in range(0, count, granularity):
                indices = tuple(range(start, min(start + granularity, count)))
                if indices:
                    yield Candidate((DeleteNodes(indices),))

    def generate_stage(self, model: Any, stage: str) -> Iterator[Candidate]:
        if stage == "cleanup":
            return
        for candidate in self.generate(model):
            if is_coarse(len(candidate.edits[0].indices), model.node_count) == (stage == "coarse"):
                yield candidate
