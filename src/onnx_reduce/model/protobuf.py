"""Conservative protobuf edits and queries for normalized, flat ONNX graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel

import json

import onnx
from onnxscript import ir

from onnx_reduce.performance import count, measured
from onnx_reduce.types import EditApplicationError, RegionGraph, RegionValue


def _fast_capability(model: OnnxModel):
    proto = model._snapshot
    graph = proto.graph
    if proto.functions or graph.sparse_initializer or graph.quantization_annotation:
        return False
    definitions = {v.name for v in graph.input} | {v.name for v in graph.initializer}
    if not all(definitions):
        return False
    for node in graph.node:
        if any(
            a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for a in node.attribute
        ):
            return False
        if not all((*node.input, *node.output)):
            return False
        if any(name not in definitions for name in node.input):
            return False
        for name in node.output:
            if name in definitions:
                return False
            definitions.add(name)
    infos = (*graph.input, *graph.output, *graph.value_info)
    return all(
        not info.HasField("type") or info.type.WhichOneof("value") in (None, "tensor_type")
        for info in infos
    )


def _proto_infos(model: OnnxModel):
    graph = model._snapshot.graph
    infos = {
        v.name: onnx.helper.make_tensor_value_info(v.name, v.data_type, list(v.dims))
        for v in graph.initializer
    }
    infos.update({v.name: v for v in (*graph.input, *graph.value_info, *graph.output)})
    return infos


@measured("fast_edit")
def _try_fast_delete_nodes(model: OnnxModel, indices):
    if not model._fast_capability:
        count("fast_edit_fallbacks")
        return None
    graph = model._snapshot.graph
    selected = []
    seen = set()
    for index in indices:
        if index < 0 or index >= len(graph.node):
            raise EditApplicationError(f"node index {index} is out of range")
        if index not in seen:
            selected.append(index)
            seen.add(index)
    referenced = {name for i, node in enumerate(graph.node) if i not in seen for name in node.input}
    outputs = {v.name for v in graph.output}
    removed = {name for i in selected for name in graph.node[i].output}
    promoted = [
        name
        for i in selected
        for name in graph.node[i].output
        if name in referenced or name in outputs
    ]
    # Promotion deliberately follows the IR's new Value construction. In
    # particular, copying the annotation verbatim would retain metadata that
    # the IR path drops. Unknown annotation fields conservatively fall back.
    if any(
        model._proto_infos.get(name) is not None and model._proto_infos[name].metadata_props
        for name in promoted
    ):
        count("fast_edit_fallbacks.metadata")
        return None
    missing = []
    values = {}
    for name in promoted:
        source = model._proto_infos.get(name)
        value = (
            ir.serde.deserialize_value_info_proto(source, None)
            if source is not None
            else ir.Value(name=name)
        )
        values[name] = value
        if value.type is None or value.shape is None:
            missing.append(value)
    inferred = model._infer_value_types(missing)
    proto = onnx.ModelProto()
    proto.CopyFrom(model._snapshot)
    result_graph = proto.graph
    for name in promoted:
        value = values[name]
        typed = inferred.get(name, value)
        replacement = ir.Value(
            name=name, type=typed.type, shape=typed.shape, doc_string=value.doc_string
        )
        info = ir.serde.serialize_value(replacement)
        result_graph.input.append(info)
        if name in outputs:
            next(v for v in result_graph.output if v.name == name).CopyFrom(info)
    kept = [v for v in result_graph.value_info if v.name not in removed]
    del result_graph.value_info[:]
    result_graph.value_info.extend(kept)
    for index in sorted(seen, reverse=True):
        del result_graph.node[index]
    ids = tuple(node_id for i, node_id in enumerate(model.node_ids) if i not in seen)
    count("fast_edit_hits")
    return type(model)._from_snapshot(proto, ids)


def _proto_links(model: OnnxModel):
    from onnx_reduce.model import TensorLinks

    graph = model._snapshot.graph
    links = {}
    producers = {name: i for i, node in enumerate(graph.node) for name in node.output}
    consumers = {}
    names = dict.fromkeys(v.name for v in graph.input)
    inputs = set(names)
    weights = {v.name for v in graph.initializer}
    names.update(dict.fromkeys(weights))
    for i, node in enumerate(graph.node):
        names.update(dict.fromkeys((*node.output, *node.input)))
        for name in node.input:
            consumers.setdefault(name, set()).add(i)
    outputs = {v.name for v in graph.output}
    names.update(dict.fromkeys(v.name for v in graph.output))
    for name in names:
        links[name] = TensorLinks(
            producers.get(name),
            tuple(sorted(consumers.get(name, ()))),
            name in inputs,
            name in outputs,
            name in weights,
        )
    return links


def _proto_dead(model: OnnxModel):
    graph = model._snapshot.graph
    producers = {name: i for i, node in enumerate(graph.node) for name in node.output}
    pending = [v.name for v in graph.output]
    live = set()
    while pending:
        index = producers.get(pending.pop())
        if index is not None and index not in live:
            live.add(index)
            pending.extend(graph.node[index].input)
    return tuple(i for i in range(len(graph.node)) if i not in live)


def _proto_unused(model: OnnxModel, kind):
    graph = model._snapshot.graph
    used = {name for node in graph.node for name in node.input}
    used.update(v.name for v in graph.output)
    return tuple(v.name for v in getattr(graph, kind) if v.name not in used)


def _proto_region_graph(model: OnnxModel):
    graph = model._snapshot.graph
    links = model._proto_links()
    # Match the IR summary's insertion order (outputs then dependencies).
    names = {}
    modules = []
    for node_id, node in zip(model.node_ids, graph.node, strict=True):
        props = {p.key: p.value for p in node.metadata_props}
        try:
            path = json.loads(props.get("onnx_reduce.module", ""))
        except (ValueError, TypeError):
            path = None
        if (
            isinstance(path, list)
            and 0 < len(path) <= 64
            and all(isinstance(p, str) and 0 < len(p) <= 512 for p in path)
        ):
            modules.append((node_id, tuple(path)))
        names.update(dict.fromkeys((*node.output, *node.input)))
    names.update(dict.fromkeys(v.name for v in graph.output))

    def complete(name):
        info = model._proto_infos.get(name)
        return (
            info is not None
            and info.type.HasField("tensor_type")
            and info.type.tensor_type.elem_type != 0
            and info.type.tensor_type.HasField("shape")
        )

    return RegionGraph(
        model.node_ids,
        tuple(modules),
        tuple(
            RegionValue(
                model.node_ids[link.producer] if link.producer is not None else None,
                tuple(model.node_ids[i] for i in link.consumers),
                link.is_graph_output,
                complete(name),
            )
            for name in names
            for link in (links[name],)
        ),
    )
