"""Extract a live fragment without cloning discarded nodes and tensor payloads."""

import onnx
from onnxscript import ir

from onnx_reduce.types import EditApplicationError


def _copy_except(message, excluded):
    result = type(message)()
    for field, value in message.ListFields():
        if field.name in excluded:
            continue
        if field.is_repeated:
            getattr(result, field.name).extend(value)
        elif field.message_type is not None:
            getattr(result, field.name).CopyFrom(value)
        else:
            setattr(result, field.name, value)
    return result


def _interface_mapping(current, original, *, allow_unused=False):
    if len(current) > len(original) or (not allow_unused and len(current) != len(original)):
        raise EditApplicationError("fragment cannot preserve the original interface arity")
    remaining = {value.name: value for value in original}
    mapping = {}
    # Bind retained names first; positional matching would swap a retained
    # boolean condition with a new float capture on control-flow fragments.
    for value in current:
        if value.name in remaining and value.type == remaining[value.name].type:
            mapping[value.name] = value.name
            del remaining[value.name]
    for value in current:
        if value.name in mapping:
            continue
        match = next((name for name, old in remaining.items() if old.type == value.type), None)
        if match is None:
            raise EditApplicationError(
                "fragment cannot preserve original interface types and shapes"
            )
        mapping[value.name] = match
        del remaining[match]
    return mapping


def _rename_scope(graph, mapping, *, root=False):
    defined = {v.name for v in graph.input}
    defined.update(t.name for t in graph.initializer)
    defined.update(t.values.name for t in graph.sparse_initializer)
    defined.update(name for node in graph.node for name in node.output if name)
    if root:
        if any(new in defined and new not in mapping for new in mapping.values()):
            raise EditApplicationError("interface alias collides with a retained definition")
    else:
        from onnx_reduce.model.operations.graphs import references

        used = references(graph)
        mapping = {old: new for old, new in mapping.items() if old not in defined and old in used}
        if any(new in defined for new in mapping.values()):
            raise EditApplicationError("interface alias would shadow a subgraph capture")
    for node in graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = mapping.get(name, name)
        if root:
            for i, name in enumerate(node.output):
                node.output[i] = mapping.get(name, name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _rename_scope(attr.g, mapping)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    _rename_scope(child, mapping)
    for info in (*graph.input, *graph.output, *graph.value_info):
        info.name = mapping.get(info.name, info.name)
    for annotation in graph.quantization_annotation:
        annotation.tensor_name = mapping.get(annotation.tensor_name, annotation.tensor_name)
        for entry in annotation.quant_parameter_tensor_names:
            entry.value = mapping.get(entry.value, entry.value)


def _preserve_interface(graph, original):
    inputs = _interface_mapping(graph.input, original.input, allow_unused=True)
    outputs = _interface_mapping(graph.output, original.output)
    mapping = dict(inputs)
    for old, new in outputs.items():
        if old in mapping and mapping[old] != new:
            raise EditApplicationError("conflicting input/output aliases")
        mapping[old] = new
    if len(set(mapping.values())) != len(mapping):
        raise EditApplicationError("interface aliases merge different values")
    mapping = {old: new for old, new in mapping.items() if old != new}
    _rename_scope(graph, mapping, root=True)
    original_inputs = {value.name for value in original.input}
    if any(name in original_inputs for node in graph.node for name in node.output if name):
        raise EditApplicationError("interface output would redefine an original graph input")
    # Keep the exact original ordering, types and annotations, including unused
    # inputs that a fixed-interface test may still supply.
    del graph.input[:]
    graph.input.extend(original.input)
    del graph.output[:]
    graph.output.extend(original.output)
    defaults = {t.name: t for t in original.initializer}
    retained = {t.name for t in graph.initializer}
    for value in graph.input:
        if value.name in defaults and value.name not in retained:
            graph.initializer.append(defaults[value.name])
    sparse_defaults = {t.values.name: t for t in original.sparse_initializer}
    sparse_retained = {t.values.name for t in graph.sparse_initializer}
    for value in graph.input:
        if value.name in sparse_defaults and value.name not in sparse_retained:
            graph.sparse_initializer.append(sparse_defaults[value.name])


def extract_nodes(model, edit):
    """Expose live fragment outputs and retain only its inputs and weights.

    Dropping the old graph outputs is part of the same checked edit. Otherwise
    a retained fault carrier can become dead code when its tail is deleted.
    """
    source = model._snapshot
    indices = tuple(sorted(set(edit.indices)))
    if not indices or any(i < 0 or i >= len(source.graph.node) for i in indices):
        raise EditApplicationError("extraction requires nonempty, in-range node indices")
    view = model._extraction_view
    selected = set(indices)
    nodes = [source.graph.node[i] for i in indices]
    produced = {name for node in nodes for name in node.output if name}
    needed = dict.fromkeys(name for i in indices for name in view.dependencies[i])
    external = {name for name in needed if name not in produced}
    outputs = [
        name
        for node in nodes
        for name in node.output
        if name
        and (
            view.links[name].is_graph_output
            or not view.links[name].consumers
            or any(i not in selected for i in view.links[name].consumers)
        )
    ]
    if edit.outputs is not None:
        if (
            not edit.outputs
            or len(set(edit.outputs)) != len(edit.outputs)
            or any(name not in produced for name in edit.outputs)
        ):
            raise EditApplicationError("fragment outputs must be unique retained node outputs")
        outputs = list(edit.outputs)
    if not outputs:
        raise EditApplicationError("extracted fragment has no named live outputs")
    graph = _copy_except(
        source.graph,
        {
            "node",
            "input",
            "output",
            "value_info",
            "initializer",
            "sparse_initializer",
            "quantization_annotation",
        },
    )
    graph.node.extend(nodes)
    for name in needed:
        if name not in external:
            continue
        if name in view.weights:
            tensor = graph.initializer.add()
            tensor.CopyFrom(view.weights[name])
            tensor.name = name
        elif name in view.sparse_weights:
            graph.sparse_initializer.append(view.sparse_weights[name])
    input_names = [
        name
        for name in needed
        if name in external and name not in view.weights and name not in view.sparse_weights
    ]
    boundary = tuple(dict.fromkeys((*input_names, *outputs)))
    infos = {}
    missing = []
    for name in boundary:
        info = view.infos.get(name)
        value = (
            ir.serde.deserialize_value_info_proto(info, None)
            if info is not None
            else ir.Value(name=name)
        )
        infos[name] = value
        if value.type is None or (isinstance(value.type, ir.TensorType) and value.shape is None):
            missing.append(value)
    inferred = model._infer_value_types(missing)
    for names, destination in ((input_names, graph.input), (outputs, graph.output)):
        for name in names:
            info = ir.serde.serialize_value(inferred.get(name, infos[name]))
            destination.append(info)
    defined = produced | external
    graph.value_info.extend(
        v for v in source.graph.value_info if v.name in defined and v.name not in boundary
    )
    # Keep annotations only when their referenced resources remain in scope.
    graph.quantization_annotation.extend(
        annotation
        for annotation in source.graph.quantization_annotation
        if annotation.tensor_name in defined
        and all(item.value in defined for item in annotation.quant_parameter_tensor_names)
    )
    if edit.preserve_interface:
        _preserve_interface(graph, source.graph)
    proto = _copy_except(source, {"graph"})
    proto.graph.CopyFrom(graph)
    return proto, tuple(model.node_ids[i] for i in indices)
