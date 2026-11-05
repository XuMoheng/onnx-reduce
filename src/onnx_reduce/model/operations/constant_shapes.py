"""Evaluate small, constant-only shape expressions at extraction boundaries.

No data inputs, overridable defaults, local functions or arbitrary operators
are executed. Limits apply before decoding tensors and before allocating a
ConstantOfShape result; vector elementwise operations cannot grow the rank.
"""

import math

import numpy as np
import onnx
from onnx import numpy_helper

_MAX_ELEMENTS = 1024
_MAX_NODES = 128
_MAX_DEPTH = 64
_TYPES = {onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.BOOL}
_OPS = {"Identity", "Add", "Sub", "Mul", "Equal", "Where", "ConstantOfShape"}


def constant_shapes(model, names):
    """Return known int64 shape vectors; unsupported expressions remain unknown."""
    graph = model._snapshot.graph
    view = model._extraction_view
    inputs = {value.name for value in graph.input}
    functions = {(f.domain, f.name) for f in model._snapshot.functions}
    memo = {}
    remaining = _MAX_NODES

    def small_tensor(tensor):
        if (
            tensor.data_type not in _TYPES
            or len(tensor.dims) > 1
            or math.prod(tensor.dims) > _MAX_ELEMENTS
            or tensor.data_location == onnx.TensorProto.EXTERNAL
        ):
            return None
        return numpy_helper.to_array(tensor)

    def evaluate(name, active):
        nonlocal remaining
        if name in memo:
            return memo[name]
        if name in inputs or name in active or len(active) >= _MAX_DEPTH or remaining <= 0:
            return None
        remaining -= 1
        # Record failed evaluations too, including malformed constant data.
        memo[name] = None
        try:
            if name in view.weights:
                link = view.links.get(name)
                if link is not None and link.producer is not None:
                    producer = graph.node[link.producer]
                    if (producer.domain, producer.op_type) in functions:
                        return None
                result = small_tensor(view.weights[name])
            else:
                link = view.links.get(name)
                if link is None or link.producer is None:
                    return None
                node = graph.node[link.producer]
                if (
                    node.domain != ""
                    or node.op_type not in _OPS
                    or (node.domain, node.op_type) in functions
                    or len(node.output) != 1
                ):
                    return None
                values = [evaluate(value, active | {name}) for value in node.input]
                if not values or any(value is None for value in values):
                    return None
                if node.op_type == "ConstantOfShape":
                    shape = values[0]
                    if (
                        shape.dtype != np.int64
                        or shape.ndim != 1
                        or shape.size > 1
                        or any(d < 0 or d > _MAX_ELEMENTS for d in shape)
                    ):
                        return None
                    for attr in node.attribute:
                        if attr.name == "value" and small_tensor(attr.t) is None:
                            return None
                # A one-node GraphProto preserves the source opset (NodeProto
                # evaluation alone uses the latest opset in ONNX's evaluator).
                from onnx.reference import ReferenceEvaluator

                fragment = onnx.helper.make_graph(
                    [node],
                    "constant-shape",
                    [],
                    [onnx.helper.make_empty_tensor_value_info(name)],
                )
                result = ReferenceEvaluator(
                    fragment,
                    opsets={item.domain: item.version for item in model._snapshot.opset_import},
                ).run(None, dict(zip(node.input, values, strict=True)))[0]
            if (
                isinstance(result, np.ndarray)
                and result.dtype.kind in "ib"
                and result.ndim <= 1
                and result.size <= _MAX_ELEMENTS
            ):
                memo[name] = result
        except Exception:
            # Evaluation is an optional candidate opportunity, never a reason
            # to abort reduction or invent a value for a dynamic expression.
            pass
        return memo[name]

    tensors = {}
    for name in names:
        result = evaluate(name, set())
        if result is not None and result.dtype == np.int64 and result.ndim == 1:
            tensors[name] = numpy_helper.from_array(result, name)
    return tensors
