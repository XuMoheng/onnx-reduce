"""Bounded atomic paths, balanced across depths, topology and edit families."""

from collections import deque
from itertools import combinations, islice

from onnx_reduce.edits import Candidate, CleanupGraph, DeleteNodes
from onnx_reduce.passes.candidates import rewrite_candidates, supports_rewrites


def interleave(streams):
    """Deterministic round robin, including streams that happen to be empty."""
    pending = deque(iter(stream) for stream in streams)
    while pending:
        stream = pending.popleft()
        item = next(stream, None)
        if item is not None:
            yield item
            pending.append(stream)


def spread_indices(count):
    """Visit both ends before progressively filling gaps throughout the graph."""
    if not count:
        return
    yield 0
    if count == 1:
        return
    yield count - 1
    pending = deque([(1, count - 2)])
    while pending:
        lo, hi = pending.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        yield mid
        pending.extend(((lo, mid - 1), (mid + 1, hi)))


def balanced_groups(count, size, limit, related=()):
    """Bounded topology hints mixed with dispersed, reproducible exploration.

    No finite prefix covers all combinations. Each depth gets its own stream;
    an O(limit * size) exploration avoids scanning C(n, k) rejected prefixes.
    """
    if count < size or not limit:
        return

    def explore():
        anchors = tuple(spread_indices(count))
        distances = tuple(dict.fromkeys((1, 2, max(1, count // 2), max(1, count // 3))))
        for distance in distances:
            for anchor in anchors:
                group = tuple(sorted({(anchor - distance * i) % count for i in range(size)}))
                if len(group) == size:
                    yield group
        # Small graphs can afford a complete fallback; large graphs remain bounded.
        yield from islice(combinations(anchors, size), limit)

    hints = (tuple(sorted(group)) for group in related if len(group) == size)
    seen = set()
    for group in islice(interleave((hints, explore())), 2 * limit):
        if group not in seen:
            seen.add(group)
            yield group
            if len(seen) == limit:
                return


def topology_groups(model, depth, limit):
    if not hasattr(model, "region_graph"):
        return ()
    graph = model.region_graph()
    positions = model.id_to_index
    groups = []
    for value in (graph.values[i] for i in spread_indices(len(graph.values))):
        ids = tuple(
            dict.fromkeys((() if value.producer is None else (value.producer,)) + value.consumers)
        )
        indices = tuple(positions[item] for item in ids if item in positions)
        for size in range(2, depth + 1):
            groups.extend(islice(combinations(indices, size), 2))
        if len(groups) >= limit:
            break
    # Spread hints too: an early high-degree region must not consume the pool.
    return tuple(groups[i] for i in islice(spread_indices(len(groups)), limit))


def dispersed_pool(stream, limit):
    items = list(stream)
    return tuple(items[i] for i in islice(spread_indices(len(items)), limit))


def balanced_product(left, right, limit):
    """Diagonals diversify both partners instead of fixing the first left item."""
    if not left or not right:
        return
    order = tuple(spread_indices(len(right)))
    for offset in range(min(len(right), limit)):
        for index, item in enumerate(left):
            yield item, right[order[(index + offset) % len(right)]]


class CombinationsPass:
    name = "combinations"

    def __init__(self, depth=2, limit=256):
        self.depth = depth
        self.limit = limit

    def generate(self, model):
        if self.depth < 2 or not self.limit or not supports_rewrites(model):
            return
        count = model.node_count
        related = topology_groups(model, self.depth, self.limit)
        groups = interleave(
            balanced_groups(count, size, self.limit, related) for size in range(2, self.depth + 1)
        )
        yield from (
            Candidate((DeleteNodes(indices), CleanupGraph()))
            for indices in islice(groups, self.limit)
        )
        edits = dispersed_pool(
            (c.edits[0] for c in rewrite_candidates(model, "bypass")), self.limit
        )

        def paths(size):
            for indices in balanced_groups(len(edits), size, self.limit):
                group = tuple(edits[i] for i in indices)
                if len({e.index for e in group}) == size:
                    yield Candidate((*sorted(group, key=lambda e: -e.index), CleanupGraph()))

        yield from islice(interleave(paths(size) for size in range(2, self.depth + 1)), self.limit)
        pools = [
            dispersed_pool((c.edits[0] for c in rewrite_candidates(model, kind)), self.limit)
            for kind in ("attributes", "tensors")
        ]
        yield from islice(
            interleave(
                (
                    Candidate((partner, bypass, CleanupGraph()))
                    for partner, bypass in balanced_product(pool, edits, self.limit)
                )
                for pool in pools
            ),
            self.limit,
        )
        nodes = tuple(spread_indices(count))
        yield from islice(
            interleave(
                (
                    Candidate((partner, DeleteNodes((index,)), CleanupGraph()))
                    for partner, index in balanced_product(pool, nodes, self.limit)
                )
                for pool in pools
            ),
            self.limit,
        )
