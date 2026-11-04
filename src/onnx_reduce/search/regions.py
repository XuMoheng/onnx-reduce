"""Deterministic regions. Boundary quality orders attempts, never excludes them."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.passes.delete_nodes import _granularities
from onnx_reduce.performance import measured
from onnx_reduce.types import NodeSearchModel, RegionSearchModel


@dataclass(frozen=True)
class NodeRegion:
    ids: tuple[int, ...]

    def candidate(self, model: NodeSearchModel, direction: str, digest: str) -> Candidate | None:
        """Bind stable IDs and complements only against the current topology."""
        live = model.id_to_index
        indices = {live[node_id] for node_id in self.ids if node_id in live}
        if not indices:
            return None
        if direction == "keep":
            indices = set(range(len(model.node_ids))) - indices
        if not indices:
            return None
        return Candidate((DeleteNodes(tuple(sorted(indices))),), base_digest=digest)


@dataclass(frozen=True)
class Region:
    node_ids: tuple[int, ...]
    source: str
    parent: tuple[str, ...] = ()
    boundary_count: int = 0
    incomplete_boundary_count: int = 0


def _structure_regions(ids, values):
    """Partition into maximal unbranched chains, including captured dependencies."""
    successors = {node_id: set() for node_id in ids}
    predecessors = {node_id: set() for node_id in ids}
    for value in values:
        if value.producer is not None:
            for consumer in value.consumers:
                successors[value.producer].add(consumer)
                predecessors[consumer].add(value.producer)
    seen = set()
    for node_id in ids:
        if node_id in seen:
            continue
        chain = []
        current = node_id
        while current not in seen:
            chain.append(current)
            seen.add(current)
            if len(successors[current]) != 1:
                break
            following = next(iter(successors[current]))
            if len(predecessors[following]) != 1:
                break
            current = following
        if len(chain) > 1:
            yield tuple(chain)


@measured("region_generation")
def graph_regions(model: NodeSearchModel, *, minimum_size: int = 1) -> tuple[Region, ...]:
    """One immutable snapshot per base. Optional graph information has a safe fallback.

    Module groups and chains are followed by every standard topology partition.
    Singletons and poorly typed boundaries are retained. This is also the exact
    region set included in the staged finalizer's declared candidate scope.
    """
    ids = model.node_ids
    graph = model.region_graph() if isinstance(model, RegionSearchModel) else None
    values = graph.values if graph is not None else ()
    incidence = defaultdict(set)
    consumed_values = defaultdict(list)
    for index, value in enumerate(values):
        if value.producer is not None:
            incidence[value.producer].add(index)
        for consumer in value.consumers:
            incidence[consumer].add(index)
            consumed_values[consumer].append(index)

    def describe(selected, source, parent=()):
        selected = tuple(sorted(selected, key=model.id_to_index.__getitem__))
        members = set(selected)
        adjacent = set()
        selected_consumers = Counter()
        for node_id in selected:
            adjacent.update(incidence[node_id])
            selected_consumers.update(consumed_values[node_id])
        boundary = []
        for index in adjacent:
            value = values[index]
            inside_producer = value.producer in members
            inside_consumer = selected_consumers[index] > 0
            outside_consumer = selected_consumers[index] < len(value.consumers)
            if (inside_consumer and not inside_producer) or (
                inside_producer and (outside_consumer or value.graph_output)
            ):
                boundary.append(value)
        return Region(
            selected,
            source,
            parent,
            len(boundary),
            sum(not value.type_complete for value in boundary),
        )

    def ordering(region):
        return (
            region.incomplete_boundary_count,
            region.boundary_count,
            -len(region.node_ids),
            tuple(model.id_to_index[node_id] for node_id in region.node_ids),
            region.parent,
        )

    result = []
    if graph is not None:
        modules = defaultdict(list)
        for node_id, path in graph.modules:
            for depth in range(1, len(path) + 1):
                modules[path[:depth]].append(node_id)
        result.extend(
            sorted(
                (
                    describe(nodes, "module", path[:-1])
                    for path, nodes in modules.items()
                    if len(nodes) >= minimum_size
                ),
                key=ordering,
            )
        )
        result.extend(
            sorted(
                (
                    describe(nodes, "structure")
                    for nodes in _structure_regions(ids, values)
                    if len(nodes) >= minimum_size
                ),
                key=ordering,
            )
        )
    for size in _granularities(len(ids)):
        result.extend(
            sorted(
                (
                    describe(ids[start : start + size], "topology")
                    for start in range(0, len(ids), size)
                    if len(ids[start : start + size]) >= minimum_size
                ),
                key=ordering,
            )
        )
    return tuple(result)


def region_choices(model, regions: tuple[Region, ...]) -> Iterator[tuple[Candidate, str, Region]]:
    digest = model.canonical_hash()
    for region in regions:
        for direction in ("delete", "keep"):
            candidate = NodeRegion(region.node_ids).candidate(model, direction, digest)
            if candidate is not None:
                yield candidate, direction, region
