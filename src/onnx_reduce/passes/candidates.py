"""ONNX candidate rules: read model queries and produce ordered edit descriptions."""

from dataclasses import replace

import onnx

from onnx_reduce.edits import (
    BypassNode,
    Candidate,
    DeleteSubgraphNodes,
    DropNodeAttribute,
    ResizeOperator,
    ShrinkDimension,
)
from onnx_reduce.model.operations.graphs import walk_graphs
from onnx_reduce.model.operations.local_tensors import plan
from onnx_reduce.model.operations.rewrites import optional_attributes
from onnx_reduce.model.operations.tensors import tensors
from onnx_reduce.passes.groups import deletion_groups


def supports_rewrites(model):
    return hasattr(model, "candidate_snapshot") or hasattr(model, "rewrite_candidates")


def rewrite_candidates(model, kind):
    """Select rules in the pass layer; adapters only expose a read-only query view."""
    if not hasattr(model, "candidate_snapshot"):
        # Older third-party adapters may supply their own candidate extension.
        if hasattr(model, "rewrite_candidates"):
            yield from model.rewrite_candidates(kind)
        return
    proto = model.candidate_snapshot()
    if kind == "tensors":
        yield from local_tensor_candidates(proto)
        yield from tensor_candidates(proto)
    elif kind == "subgraphs":
        yield from subgraph_candidates(proto)
    else:
        yield from node_candidates(proto, kind)


def node_candidates(proto, kind):
    for index, node in enumerate(proto.graph.node):
        if kind == "bypass" and len(node.output) == 1 and node.output[0]:
            for input_index, name in enumerate(node.input):
                if name and name != node.output[0]:
                    yield Candidate((BypassNode(index, input_index),))
        elif kind == "attributes":
            for name in optional_attributes(proto, node):
                yield Candidate((DropNodeAttribute(index, name),))


def subgraph_candidates(proto):
    for path, graph in walk_graphs(proto):
        if not path:
            continue
        # Reuse the main-graph candidate rules with this graph's local indices.
        view = onnx.ModelProto()
        view.graph.CopyFrom(graph)
        view.opset_import.extend(proto.opset_import)
        view.functions.extend(proto.functions)
        for kind in ("bypass", "attributes"):
            for candidate in node_candidates(view, kind):
                yield Candidate(tuple(replace(e, graph_path=path) for e in candidate.edits))
        for indices in deletion_groups(tuple(range(len(graph.node)))):
            yield Candidate((DeleteSubgraphNodes(path, indices),))


def tensor_candidates(proto):
    dimensions = set()
    for _, graph in walk_graphs(proto):
        for tensor in tensors(graph):
            dimensions.update(d for d in tensor.dims if d > 1)
        for value in graph.input:
            dimensions.update(
                d.dim_value
                for d in value.type.tensor_type.shape.dim
                if d.HasField("dim_value") and d.dim_value > 1
            )
    for old in sorted(dimensions, reverse=True):
        for new in sorted({1, old // 2, old - 1}):
            if 0 < new < old:
                yield Candidate((ShrinkDimension(old, new),))


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
