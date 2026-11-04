"""Read-only IR graph queries and connection summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from onnx_reduce.types import (
    RegionGraph,
    RegionValue,
)

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel


@dataclass(frozen=True)
class TensorLinks:
    """Summarize a main-graph value's links; the IR owns the actual connections."""

    # ==========================================
    # Attributes: populated by tensor_index(), with no graph edits
    # ==========================================
    producer: int | None  # Main-graph producer index, or None if there is no producer
    consumers: tuple[int, ...]  # Main-graph consumer indices
    is_graph_input: bool
    is_graph_output: bool
    is_initializer: bool


def tensor_index(model: OnnxModel) -> dict[str, TensorLinks]:
    """Summarize main-graph value links and roles for inspection and tests."""

    if model._ir_model is None:
        return model._proto_links()
    graph = model._model.graph
    links: dict[str, TensorLinks] = {}

    def record(name: str) -> TensorLinks:
        return links.setdefault(
            name,
            TensorLinks(
                producer=None,
                consumers=(),
                is_graph_input=False,
                is_graph_output=False,
                is_initializer=False,
            ),
        )

    for value in graph.inputs:
        if value.name:
            links[value.name] = replace(record(value.name), is_graph_input=True)
    for value in graph.initializers.values():
        if value.name:
            links[value.name] = replace(record(value.name), is_initializer=True)
    for index, node in enumerate(graph):
        for value in node.outputs:
            if value is not None and value.name:
                links[value.name] = replace(record(value.name), producer=index)
        for value in model._node_dependencies(node):
            # Captured values belong to their defining scope, even when a
            # nested graph contains a local value with the same name.
            if value.graph is not graph or not value.name:
                continue
            item = record(value.name)
            links[value.name] = replace(
                item,
                consumers=tuple(sorted({*item.consumers, index})),
            )
    for value in graph.outputs:
        if value.name:
            links[value.name] = replace(record(value.name), is_graph_output=True)
    return links


def _region_graph(model: OnnxModel) -> RegionGraph:
    if model._ir_model is None:
        return model._proto_region_graph()
    graph = model._model.graph
    links = {}
    consumers = {}
    modules = []
    for node in graph:
        node_id = model._node_ids[node]
        # Explicit reducer metadata contract, never guessed from node names.
        raw = node.metadata_props.get("onnx_reduce.module", "")
        try:
            path = json.loads(raw)
        except (ValueError, TypeError):
            path = None
        if (
            isinstance(path, list)
            and 0 < len(path) <= 64
            and all(isinstance(part, str) and 0 < len(part) <= 512 for part in path)
        ):
            modules.append((node_id, tuple(path)))
        for value in node.outputs:
            if value is not None:
                links[id(value)] = value
        for value in model._node_dependencies(node):
            if value.graph is graph:
                links[id(value)] = value
                consumers.setdefault(id(value), set()).add(node_id)
    for value in graph.outputs:
        links[id(value)] = value
    return RegionGraph(
        model.node_ids,
        tuple(modules),
        tuple(
            RegionValue(
                model._node_ids.get(value.producer()),
                tuple(sorted(consumers.get(key, ()), key=model.id_to_index.__getitem__)),
                value.is_graph_output(),
                value.type is not None and value.shape is not None,
            )
            for key, value in links.items()
        ),
    )


def region_graph(model: OnnxModel) -> RegionGraph:
    """Read-only graph summary; outer captures count as main-graph dependencies."""
    return model._region_graph


def dead_node_indices(model: OnnxModel) -> tuple[int, ...]:
    """Return main-graph node indices outside the backward closure of all outputs."""

    if model._ir_model is None:
        return model._proto_dead()
    # Use a stack to trace producers backward from every graph output.
    needed_values = list(model._model.graph.outputs)
    # Track object identities to distinguish same-named values in different scopes.
    seen_values: set[int] = set()
    needed_nodes: set[int] = set()
    while needed_values:
        value = needed_values.pop()
        if id(value) in seen_values:
            continue
        seen_values.add(id(value))
        producer = value.producer()
        if producer is None or id(producer) in needed_nodes:
            continue
        needed_nodes.add(id(producer))
        needed_values.extend(model._node_dependencies(producer))
    return tuple(
        index for index, node in enumerate(model._model.graph) if id(node) not in needed_nodes
    )


def unused_input_names(model: OnnxModel) -> tuple[str, ...]:
    """Find graph inputs with no consumers that are not themselves graph outputs."""

    if model._ir_model is None:
        return model._proto_unused("input")
    return tuple(
        value.name
        for value in model._model.graph.inputs
        if not value.uses() and not value.is_graph_output()
    )


def unused_initializer_names(model: OnnxModel) -> tuple[str, ...]:
    """Find initializers with no consumers that are not themselves graph outputs."""

    if model._ir_model is None:
        return model._proto_unused("initializer")
    return tuple(
        name
        for name, value in model._model.graph.initializers.items()
        if not value.uses() and not value.is_graph_output()
    )
