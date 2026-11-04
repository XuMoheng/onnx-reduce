"""Built-in reduction passes in deterministic priority order."""

from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.passes.combinations import CombinationsPass
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.drop_outputs import DropOutputsPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass
from onnx_reduce.passes.rewrite import (
    BypassNodesPass,
    ReduceSubgraphsPass,
    ShrinkTensorsPass,
    SimplifyAttributesPass,
)


def default_passes(*, combination_depth=2, combination_limit=256) -> tuple[ReductionPass, ...]:
    """Try structural deletion before connection and attribute rewrites."""

    return (
        DropOutputsPass(),
        RemoveUnusedPass(),
        DeleteNodesPass(),
        BypassNodesPass(),
        SimplifyAttributesPass(),
        ShrinkTensorsPass(),
        ReduceSubgraphsPass(),
        CombinationsPass(combination_depth, combination_limit),
    )


__all__ = [
    "BypassNodesPass",
    "DeleteNodesPass",
    "DropOutputsPass",
    "ReductionPass",
    "RemoveUnusedPass",
    "SimplifyAttributesPass",
    "default_passes",
]
