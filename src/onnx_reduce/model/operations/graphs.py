"""Explicit attribute-graph addresses and interface-preserving subgraph edits."""

import onnx

from onnx_reduce.edits import CleanupGraph
from onnx_reduce.types import EditApplicationError


def walk_graphs(proto):
    def visit(graph, path):
        yield path, graph
        for index, node in enumerate(graph.node):
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from visit(attr.g, (*path, (index, attr.name, -1)))
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child_index, graph in enumerate(attr.graphs):
                        yield from visit(graph, (*path, (index, attr.name, child_index)))

    yield from visit(proto.graph, ())


def graph_at(proto, path):
    graph = proto.graph
    try:
        for index, name, child_index in path:
            if not 0 <= index < len(graph.node):
                raise ValueError("node out of range")
            attr = next(a for a in graph.node[index].attribute if a.name == name)
            if child_index == -1 and attr.type == onnx.AttributeProto.GRAPH:
                graph = attr.g
            elif attr.type == onnx.AttributeProto.GRAPHS and 0 <= child_index < len(attr.graphs):
                graph = attr.graphs[child_index]
            else:
                raise ValueError("attribute is not the requested graph")
    except (ValueError, StopIteration) as exc:
        raise EditApplicationError("invalid subgraph path") from exc
    return graph


def references(graph):
    """Conservative use set, including nested outer captures."""
    result = {v.name for v in graph.output}
    for node in graph.node:
        result.update(name for name in node.input if name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                result.update(references(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    result.update(references(child))
    return result


def apply_graph_edit(proto, edit):
    if isinstance(edit, CleanupGraph):
        # Keep body inputs/outputs: their positional slots are part of the
        # Loop/Scan interface, even when unused inside the body.
        for path, graph in walk_graphs(proto):
            used = references(graph)
            kept = [t for t in graph.initializer if t.name in used]
            del graph.initializer[:]
            graph.initializer.extend(kept)
            if not path:
                kept = [v for v in graph.input if v.name in used]
                del graph.input[:]
                graph.input.extend(kept)
        return
    graph = graph_at(proto, edit.graph_path)
    if (
        not edit.graph_path
        or not edit.indices
        or any(i < 0 or i >= len(graph.node) for i in edit.indices)
    ):
        raise EditApplicationError("invalid subgraph deletion")
    for index in sorted(set(edit.indices), reverse=True):
        del graph.node[index]
    defined = {v.name for v in graph.input} | {t.name for t in graph.initializer}
    defined.update(name for n in graph.node for name in n.output)
    kept = [v for v in graph.value_info if v.name in defined]
    del graph.value_info[:]
    graph.value_info.extend(kept)
