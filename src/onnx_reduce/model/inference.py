"""Bounded, run-local shape and type inference summaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import onnx
from onnxscript import ir

from onnx_reduce.performance import count, measured
from onnx_reduce.preparation import current_preparation, metadata_key

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel


def _infer_value_types(model: OnnxModel, values: Iterable[ir.Value]) -> dict[str, ir.Value]:
    """Infer types and shapes for main-graph promotions; return an empty mapping on failure."""

    missing = [value for value in values if value.name]
    if not missing:
        return {}
    caches = current_preparation()
    key = metadata_key("shape", model.canonical_hash()) if caches is not None else None
    summary = caches.shapes.get(key) if caches is not None else None
    if summary is None:
        summary = model._infer_type_summary()
        if caches is not None:
            caches.shapes.put(key, summary)
    # Empty bytes are a cached failure, distinct from a cache miss (None).
    graph = onnx.GraphProto.FromString(summary)
    wanted = {value.name for value in missing}
    return {
        value.name: ir.serde.deserialize_value_info_proto(value, None)
        for value in graph.value_info
        if value.name in wanted
    }


@measured("shape_inference")
def _infer_type_summary(model: OnnxModel) -> bytes:
    """Retain only main-graph names and types, without weights or IR references."""

    try:
        inferred_proto = onnx.shape_inference.infer_shapes(model._payload)
        # Match IR import semantics, including main-graph outputs whose types
        # come from graph.output instead of intermediate value_info.
        inferred_model = ir.from_proto(inferred_proto)
        if not isinstance(inferred_model, ir.Model):
            return b""
        graph = inferred_model.graph
        available = {value.name: value for value in graph.inputs}
        for node in graph:
            available.update({value.name: value for value in node.outputs if value.name})
        summary = onnx.GraphProto()
        for value in available.values():
            info = summary.value_info.add()
            info.name = value.name
            info.type.CopyFrom(ir.serde.serialize_value(value).type)
        return summary.SerializeToString(deterministic=True)
    except Exception:
        count("shape_inference_failures")
        # An untyped promotion will be rejected by the configured checker.
        return b""
