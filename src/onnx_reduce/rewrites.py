"""Main-graph rewrites that preserve connections instead of promoting inputs.

These are reducer transformations, not semantics-preserving optimizations.
The engine must check validity, cost and the target failure for every candidate.
"""

import onnx

from onnx_reduce.edits import BypassNode, Candidate, DropNodeAttribute
from onnx_reduce.graphs import graph_at
from onnx_reduce.types import EditApplicationError


def optional_attributes(proto, node):
    # Do not guess the contract of custom/local functions or unknown opsets.
    domain = "" if node.domain == "ai.onnx" else node.domain
    if domain not in {"", "ai.onnx.ml", "ai.onnx.training"}:
        return ()
    versions = {
        ("" if item.domain == "ai.onnx" else item.domain): item.version
        for item in proto.opset_import
    }
    if domain not in versions:
        return ()
    if any(f.domain == node.domain and f.name == node.op_type for f in proto.functions):
        return ()
    try:
        schema = onnx.defs.get_schema(node.op_type, versions[domain], domain)
    except onnx.defs.SchemaError:
        return ()
    return tuple(
        attr.name
        for attr in node.attribute
        if attr.name in schema.attributes
        and not schema.attributes[attr.name].required
        and not attr.ref_attr_name
        and attr.type not in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
    )


def rewrite_candidates(proto, kind):
    for index, node in enumerate(proto.graph.node):
        if kind == "bypass" and len(node.output) == 1 and node.output[0]:
            for input_index, name in enumerate(node.input):
                if name and name != node.output[0]:
                    yield Candidate((BypassNode(index, input_index),))
        elif kind == "attributes":
            for name in optional_attributes(proto, node):
                yield Candidate((DropNodeAttribute(index, name),))


def _replace_captures(graph, old, new):
    # A nested local definition shadows the outer value throughout its scope.
    local = {v.name for v in graph.input}
    local.update(v.name for v in graph.initializer)
    local.update(v.values.name for v in graph.sparse_initializer)
    local.update(name for node in graph.node for name in node.output if name)
    if old in local:
        return
    # Rebinding a capture to a shadowed source would silently change its scope.
    if new in local:
        raise EditApplicationError("bypass source is shadowed inside a subgraph")
    _replace_uses(graph, old, new)


def _replace_uses(graph, old, new):
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _replace_captures(attr.g, old, new)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    _replace_captures(child, old, new)
    for output in graph.output:
        if output.name == old:
            output.name = new
    # The removed value's inferred type may differ from its replacement.
    retained = [v for v in graph.value_info if v.name != old]
    del graph.value_info[:]
    graph.value_info.extend(retained)


def apply_rewrite(proto, edit):
    graph = graph_at(proto, edit.graph_path)
    if not 0 <= edit.index < len(graph.node):
        raise EditApplicationError("rewrite node index out of range")
    node = graph.node[edit.index]
    if isinstance(edit, DropNodeAttribute):
        if edit.name not in optional_attributes(proto, node):
            raise EditApplicationError("attribute is not schema-optional")
        retained = [attr for attr in node.attribute if attr.name != edit.name]
        del node.attribute[:]
        node.attribute.extend(retained)
        return
    if len(node.output) != 1 or not node.output[0]:
        raise EditApplicationError("bypass requires one named output")
    if not 0 <= edit.input_index < len(node.input) or not node.input[edit.input_index]:
        raise EditApplicationError("bypass requires a named input")
    old, new = node.output[0], node.input[edit.input_index]
    if old == new:
        raise EditApplicationError("bypass input and output must differ")
    outputs = {value.name for value in graph.output}
    if old in outputs and new in outputs:
        raise EditApplicationError("bypass would duplicate a graph output")
    del graph.node[edit.index]
    _replace_uses(graph, old, new)
