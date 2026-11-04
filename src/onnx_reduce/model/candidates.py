"""ONNX-specific candidate generation over read-only snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel


def rewrite_candidates(model: OnnxModel, kind):
    """Offer ONNX-specific rewrites without exporting a copy for each pass."""
    from onnx_reduce.model.operations.rewrites import rewrite_candidates

    if kind == "tensors":
        from onnx_reduce.model.operations.local_tensors import local_tensor_candidates
        from onnx_reduce.model.operations.tensors import tensor_candidates

        yield from local_tensor_candidates(model._snapshot)
        yield from tensor_candidates(model._snapshot)
    elif kind == "subgraphs":
        from onnx_reduce.model.operations.graphs import subgraph_candidates

        yield from subgraph_candidates(model._snapshot)
    else:
        yield from rewrite_candidates(model._snapshot, kind)
