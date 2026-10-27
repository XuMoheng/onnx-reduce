"""Built-in M1 reduction passes in deterministic priority order."""

from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.drop_outputs import DropOutputsPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass


def default_passes() -> tuple[ReductionPass, ...]:
    """Return the pass order fixed by the M1 design document."""

    return (
        DropOutputsPass(),
        RemoveUnusedPass(),
        DeleteNodesPass(),
    )


__all__ = [
    "DeleteNodesPass",
    "DropOutputsPass",
    "ReductionPass",
    "RemoveUnusedPass",
    "default_passes",
]
