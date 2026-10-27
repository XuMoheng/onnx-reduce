"""Graph-aware ddmin-style node deletion."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from onnx_reduce.edits import Candidate, DeleteNodes


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

    def generate(self, model: Any) -> Iterator[Candidate]:
        count = model.node_count
        for granularity in _granularities(count):
            for start in range(0, count, granularity):
                indices = tuple(range(start, min(start + granularity, count)))
                if indices:
                    yield Candidate((DeleteNodes(indices),))
