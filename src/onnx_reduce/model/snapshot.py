"""Invalidate all representation-derived properties after an intermediate edit."""

from functools import cached_property, lru_cache


@lru_cache(maxsize=32)
def _derived_names(model_type):
    return tuple(
        name
        for base in model_type.__mro__
        for name, descriptor in vars(base).items()
        if isinstance(descriptor, cached_property)
    )


def invalidate_derived(model):
    for name in _derived_names(type(model)):
        model.__dict__.pop(name, None)
