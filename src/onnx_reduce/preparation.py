"""Versioned, bounded metadata caches scoped to a single reduction run."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from onnx_reduce.performance import count, sample

CACHE_VERSION = 1


def metadata_key(*parts) -> str:
    payload = json.dumps([CACHE_VERSION, *parts], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class MetadataCache:
    """LRU of encoded metadata, never model objects, tensors or mutable IR values.

    The byte limit covers encoded keys and values; the entry limit also bounds
    Python container overhead. Oversized entries are used without retention.
    """

    def __init__(self, name, *, enabled=True, max_entries=4096, max_bytes=8 * 1024 * 1024):
        self.name = name
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.byte_size = 0
        self._entries: OrderedDict[str, bytes] = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def discard(self, key):
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.byte_size -= len(key.encode()) + len(previous)

    def get(self, key: str) -> bytes | None:
        if not self.enabled:
            count(f"{self.name}_cache_disabled")
            return None
        value = self._entries.get(key)
        count(f"{self.name}_cache_{'misses' if value is None else 'hits'}")
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: bytes) -> None:
        if not self.enabled:
            return
        size = len(key.encode()) + len(value)
        if self.max_entries < 1 or size > self.max_bytes:
            count(f"{self.name}_cache_oversized")
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.byte_size -= len(key.encode()) + len(previous)
        while self._entries and (
            len(self._entries) >= self.max_entries or self.byte_size + size > self.max_bytes
        ):
            old_key, old_value = self._entries.popitem(last=False)
            self.byte_size -= len(old_key.encode()) + len(old_value)
            count(f"{self.name}_cache_evictions")
        self._entries[key] = value
        self.byte_size += size
        sample(f"{self.name}_cache_bytes", self.byte_size)
        sample(f"{self.name}_cache_entries", len(self))


@dataclass
class PreparationCaches:
    enabled: bool
    validation_mode: str
    validation_cache_enabled: bool = True
    edits: MetadataCache = field(init=False)
    shapes: MetadataCache = field(init=False)
    validations: MetadataCache = field(init=False)
    domains: MetadataCache = field(init=False)

    def __post_init__(self):
        self.edits = MetadataCache("edit", enabled=self.enabled)
        self.shapes = MetadataCache("shape", enabled=self.enabled)
        self.validations = MetadataCache(
            "validation", enabled=self.enabled and self.validation_cache_enabled
        )
        self.domains = MetadataCache(
            "domain", enabled=self.enabled and self.validation_cache_enabled
        )


_ACTIVE: ContextVar[PreparationCaches | None] = ContextVar("preparation_caches", default=None)


def current_preparation() -> PreparationCaches | None:
    return _ACTIVE.get()


def edit_key(candidate, validation_mode):
    return metadata_key(
        "edit", candidate.base_digest, candidate.to_dict()["edits"], validation_mode
    )


@contextmanager
def preparation_caches(enabled=True, validation_mode="strict", validation_cache_enabled=True):
    caches = PreparationCaches(enabled, validation_mode, validation_cache_enabled)
    token = _ACTIVE.set(caches)
    try:
        yield caches
    finally:
        _ACTIVE.reset(token)
