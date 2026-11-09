"""Built-in reduction passes in deterministic priority order."""

from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.passes.combinations import CombinationsPass
from onnx_reduce.passes.compact import CompactFragmentsPass
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.drop_outputs import DropOutputsPass
from onnx_reduce.passes.extract import ExtractNodesPass, ProbeFragmentsPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass
from onnx_reduce.passes.rewrite import (
    BypassNodesPass,
    ReduceSubgraphsPass,
    ShrinkTensorsPass,
    SimplifyAttributesPass,
)


def default_passes(*, combination_depth=2, combination_limit=256) -> tuple[ReductionPass, ...]:
    """Cut live witnesses first, then shrink before repeated deletion sweeps."""

    return (
        CompactFragmentsPass(),
        ProbeFragmentsPass(),
        DropOutputsPass(),
        RemoveUnusedPass(),
        # Retain the early probes for large graphs. On a small witness, finish
        # dimension reductions before restart-on-accept replays deletion scans.
        ShrinkTensorsPass(),
        DeleteNodesPass(),
        BypassNodesPass(),
        SimplifyAttributesPass(),
        ReduceSubgraphsPass(),
        CombinationsPass(combination_depth, combination_limit),
        ExtractNodesPass(),
    )


__all__ = [
    "BypassNodesPass",
    "CompactFragmentsPass",
    "DeleteNodesPass",
    "DropOutputsPass",
    "ExtractNodesPass",
    "ReductionPass",
    "RemoveUnusedPass",
    "SimplifyAttributesPass",
    "default_passes",
]
