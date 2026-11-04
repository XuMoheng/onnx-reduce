"""Local axis constraints for boundary-fed MatMul/Gemm, Conv and Reshape.

Only named graph inputs and dense initializers can be resized as sources.
No producer graph is silently changed; the engine validates all downstream uses.
"""

import math

import numpy as np
from onnx import numpy_helper

from onnx_reduce.edits import Candidate, ResizeOperator
from onnx_reduce.graphs import graph_at, walk_graphs
from onnx_reduce.types import EditApplicationError


def extents(graph, name):
    tensor = next((t for t in graph.initializer if t.name == name), None)
    if tensor is not None:
        return tuple(tensor.dims)
    value = next((v for v in graph.input if v.name == name), None)
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims = value.type.tensor_type.shape.dim
    if not value.type.tensor_type.HasField("shape") or any(
        not d.HasField("dim_value") for d in dims
    ):
        return None
    return tuple(d.dim_value for d in dims)


def terminal(graph, output):
    return not any(output in node.input for node in graph.node)


def plan(graph, index, kind):
    """Return coupled axes and their original common extent, or decline."""
    if not 0 <= index < len(graph.node):
        return None
    node = graph.node[index]
    if node.domain not in {"", "ai.onnx"} or len(node.output) != 1:
        return None
    attrs = {a.name: a.i for a in node.attribute}
    if kind == "contract" and node.op_type in {"MatMul", "Gemm"} and len(node.input) >= 2:
        a, b = (extents(graph, name) for name in node.input[:2])
        if a is None or b is None or len(a) != 2 or len(b) != 2:
            return None
        ax = 0 if node.op_type == "Gemm" and attrs.get("transA", 0) else 1
        bx = 1 if node.op_type == "Gemm" and attrs.get("transB", 0) else 0
        if a[ax] != b[bx]:
            return None
        return ((node.input[0], ax), (node.input[1], bx)), a[ax], 1
    if node.op_type == "Conv" and len(node.input) >= 2 and kind in {"conv_input", "conv_output"}:
        data, weight = (extents(graph, name) for name in node.input[:2])
        group = attrs.get("group", 1)
        if weight is None or len(weight) < 3 or group < 1:
            return None
        if kind == "conv_input":
            if data is None or len(data) != len(weight) or data[1] != weight[1] * group:
                return None
            return ((node.input[0], 1), (node.input[1], 1)), data[1], group
        if not terminal(graph, node.output[0]) or weight[0] % group:
            return None
        axes = [(node.input[1], 0), (node.output[0], 1)]
        if len(node.input) > 2 and node.input[2]:
            if extents(graph, node.input[2]) != (weight[0],):
                return None
            axes.append((node.input[2], 0))
        return tuple(axes), weight[0], group
    if kind == "reshape" and node.op_type == "Reshape" and len(node.input) >= 2:
        shape = next((t for t in graph.initializer if t.name == node.input[1]), None)
        data = extents(graph, node.input[0])
        if shape is None or data is None or not data or not terminal(graph, node.output[0]):
            return None
        target = numpy_helper.to_array(shape)
        if (
            target.ndim != 1
            or target.dtype.kind not in "iu"
            or not len(target)
            or any(d <= 0 for d in target)
        ):
            return None
        if math.prod(data) != math.prod(int(d) for d in target):
            return None
        return ((node.input[0], 0),), data[0], 1
    return None


def local_tensor_candidates(proto):
    for path, graph in walk_graphs(proto):
        for index, _ in enumerate(graph.node):
            for kind in ("contract", "conv_input", "conv_output", "reshape"):
                match = plan(graph, index, kind)
                if match is None:
                    continue
                _, old, unit = match
                targets = {unit, (old // 2 // unit) * unit, old - unit}
                for new in sorted(targets):
                    if 0 < new < old and new % unit == 0:
                        yield Candidate((ResizeOperator(index, kind, new, path),))


def resize_axis(graph, name, axis, old, new):
    tensor = next((t for t in graph.initializer if t.name == name), None)
    if tensor is not None:
        if axis >= len(tensor.dims) or tensor.dims[axis] != old:
            raise EditApplicationError("local tensor extent changed")
        slices = [slice(None)] * len(tensor.dims)
        slices[axis] = slice(0, new)
        replacement = numpy_helper.from_array(
            np.array(numpy_helper.to_array(tensor)[tuple(slices)], copy=True), name
        )
        replacement.doc_string = tensor.doc_string
        replacement.metadata_props.extend(tensor.metadata_props)
        tensor.CopyFrom(replacement)
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.name != name:
            continue
        dims = value.type.tensor_type.shape.dim
        if axis < len(dims):
            dims[axis].dim_value = new


def resize_operator(proto, edit):
    graph = graph_at(proto, edit.graph_path)
    match = plan(graph, edit.index, edit.kind)
    if match is None:
        raise EditApplicationError("operator has no supported local resize contract")
    axes, old, unit = match
    if not 0 < edit.new < old or edit.new % unit:
        raise EditApplicationError("local extent must decrease and respect group")
    node = graph.node[edit.index]
    try:
        if edit.kind == "reshape":
            data = extents(graph, node.input[0])
            shape = next(t for t in graph.initializer if t.name == node.input[1])
            target = numpy_helper.to_array(shape).copy()
            total = math.prod(data[1:]) * edit.new
            for axis in range(len(target)):
                others = math.prod(int(d) for i, d in enumerate(target) if i != axis)
                if total % others == 0 and 0 < total // others < target[axis]:
                    target[axis] = total // others
                    resize_axis(graph, node.input[0], 0, old, edit.new)
                    shape.CopyFrom(numpy_helper.from_array(target, shape.name))
                    for value in (*graph.output, *graph.value_info):
                        if value.name == node.output[0]:
                            dims = value.type.tensor_type.shape.dim
                            if len(dims) == len(target):
                                for dim, extent in zip(dims, target, strict=True):
                                    dim.dim_value = int(extent)
                    return
            raise EditApplicationError("reshape target cannot represent the new element count")
        changed = set()
        for name, axis in axes:
            if (name, axis) in changed:
                continue
            changed.add((name, axis))
            divisor = unit if edit.kind == "conv_input" and name == node.input[1] else 1
            resize_axis(graph, name, axis, old // divisor, edit.new // divisor)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise EditApplicationError(f"cannot resize local tensor: {exc}") from exc
