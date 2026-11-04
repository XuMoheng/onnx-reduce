"""Coordinated, concrete dimension reductions; correctness is checked by engine."""

import math

import numpy as np
import onnx
from onnx import numpy_helper

from onnx_reduce.model.operations.graphs import walk_graphs
from onnx_reduce.types import EditApplicationError


def tensor_entries(graph):
    for tensor in graph.initializer:
        yield tensor, tensor.name
    for node in graph.node:
        for attr in node.attribute:
            value_name = (
                node.output[0]
                if node.op_type == "Constant" and node.domain in {"", "ai.onnx"} and node.output
                else None
            )
            if attr.type == onnx.AttributeProto.TENSOR:
                yield attr.t, value_name
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    yield tensor, None


def tensors(graph):
    for tensor, _ in tensor_entries(graph):
        yield tensor


def tensor_elements(proto):
    """Stored dense elements plus known concrete graph-input extents.

    Symbolic/unknown dimensions contribute one; sparse payloads remain opaque.
    Scalars count as one and zero extents as zero.
    """
    total = 0
    for _, graph in walk_graphs(proto):
        total += sum(math.prod(t.dims) for t in tensors(graph))
        initializers = {t.name for t in graph.initializer}
        for value in graph.input:
            if value.name in initializers or not value.type.HasField("tensor_type"):
                continue
            shape = value.type.tensor_type.shape
            total += math.prod(d.dim_value if d.HasField("dim_value") else 1 for d in shape.dim)
    return total


def shrink_dimension(proto, old, new):
    if not 0 < new < old:
        raise EditApplicationError("dimension must strictly decrease and remain positive")
    try:
        for _, graph in walk_graphs(proto):
            # Shape-vector constants need their values adjusted along with data
            # tensors. Only explicit shape inputs of these standard ops qualify.
            shape_names = {
                n.input[1]
                for n in graph.node
                if n.domain in {"", "ai.onnx"}
                and n.op_type in {"Reshape", "Expand"}
                and len(n.input) > 1
            }
            shape_names.update(
                n.input[0]
                for n in graph.node
                if n.domain in {"", "ai.onnx"} and n.op_type == "ConstantOfShape" and n.input
            )
            for tensor, value_name in tensor_entries(graph):
                is_shape = value_name in shape_names
                if old not in tensor.dims and not is_shape:
                    continue
                data = numpy_helper.to_array(tensor)
                if is_shape and data.dtype.kind in "iu" and data.ndim == 1:
                    data = np.where(data == old, new, data)
                elif old in tensor.dims:
                    data = data[
                        tuple(slice(0, new) if d == old else slice(None) for d in data.shape)
                    ]
                else:
                    continue
                replacement = numpy_helper.from_array(np.array(data, copy=True), tensor.name)
                replacement.doc_string = tensor.doc_string
                replacement.metadata_props.extend(tensor.metadata_props)
                tensor.CopyFrom(replacement)
            for value in (*graph.input, *graph.output, *graph.value_info):
                for dim in value.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value == old:
                        dim.dim_value = new
    except (ValueError, TypeError, RuntimeError) as exc:
        raise EditApplicationError(f"cannot resize tensor: {exc}") from exc
