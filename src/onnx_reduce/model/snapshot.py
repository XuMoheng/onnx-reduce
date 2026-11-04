"""Owned snapshots and centralized invalidation after intermediate edits."""

import hashlib
from dataclasses import dataclass
from functools import cached_property, lru_cache

import onnx

from onnx_reduce.performance import measured


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


@dataclass(frozen=True)
class ModelSnapshot:
    """One owned proto and its exact deterministic bytes and identity.

    The proto is an internal read-only view; public model export copies it.
    Snapshot owners must never mutate it after publication.
    """

    proto: onnx.ModelProto

    @cached_property
    @measured("serialization")
    def payload(self) -> bytes:
        return self.proto.SerializeToString(deterministic=True)

    @cached_property
    @measured("hash")
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()
