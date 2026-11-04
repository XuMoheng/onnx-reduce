"""Bounded atomic paths; intermediate edits need not lower cost or reproduce.

Only the final model is checked and accepted. Enumeration bounds are explicit,
so fixed point claims cover this finite set, never all possible combinations.
"""

from itertools import combinations, islice, product

from onnx_reduce.edits import BypassNode, Candidate, CleanupGraph, DeleteNodes


class CombinationsPass:
    name = "combinations"

    def __init__(self, depth=2, limit=256):
        self.depth = depth
        self.limit = limit

    def generate(self, model):
        if self.depth < 2 or not self.limit or not hasattr(model, "rewrite_candidates"):
            return
        count = model.node_count
        # Include noncontiguous deletions even when every singleton fails.
        deletions = (
            Candidate((DeleteNodes(indices), CleanupGraph()))
            for size in range(2, self.depth + 1)
            for indices in combinations(range(count), size)
        )
        yield from islice(deletions, self.limit)
        # Descending indices keep all original addresses valid when nodes vanish.
        edits = list(islice((c.edits[0] for c in model.rewrite_candidates("bypass")), self.limit))

        def paths():
            for size in range(2, self.depth + 1):
                for group in islice(combinations(edits, size), self.limit):
                    if len({e.index for e in group}) == size:
                        yield Candidate((*sorted(group, key=lambda e: -e.index), CleanupGraph()))

        yield from islice(paths(), self.limit)
        # Attribute simplification or shape shrinking may only preserve the
        # failure when paired with bypass. Rebind after removal by editing first.
        partners = list(
            islice(
                (
                    c.edits[0]
                    for kind in ("attributes", "tensors")
                    for c in model.rewrite_candidates(kind)
                ),
                self.limit,
            )
        )
        yield from islice(
            (
                Candidate((partner, bypass, CleanupGraph()))
                for partner, bypass in product(partners, edits)
                if isinstance(bypass, BypassNode)
            ),
            self.limit,
        )
        # Shrinking and deleting are one transaction when a standalone resize
        # creates a temporary incompatible operator contract.
        yield from islice(
            (
                Candidate((partner, DeleteNodes((index,)), CleanupGraph()))
                for partner, index in product(partners, range(count))
            ),
            self.limit,
        )
