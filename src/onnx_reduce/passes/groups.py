"""Deterministic coarse-to-fine selections, including every singleton."""

from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def deletion_groups(items: tuple[T, ...], *, keep_one: bool = False) -> Iterator[tuple[T, ...]]:
    """Try the largest legal selection, then balanced partitions down to singletons.

    Generators need no feedback: rejection continues the partition scan, while
    acceptance makes the engine restart on the new base. No monotonicity is assumed.
    """

    if not items or (keep_one and len(items) == 1):
        return
    if keep_one:
        # Prefer retaining the first output, but also try removing it. Otherwise
        # even the singleton fallback would miss a required reduction.
        yield items[:-1]
        yield items[1:]
    else:
        yield items
    partitions = [items]
    while any(len(group) > 1 for group in partitions):
        next_partitions = []
        for group in partitions:
            if len(group) <= 1:
                continue
            middle = len(group) // 2
            for part in (group[:middle], group[middle:]):
                next_partitions.append(part)
                # The leading output batch above can equal a partition for n=2/3.
                if keep_one and part in (items[:-1], items[1:]):
                    continue
                yield part
        partitions = next_partitions
