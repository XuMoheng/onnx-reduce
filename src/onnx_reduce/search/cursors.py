"""Version-rebound fine-search cursors and bounded heuristic ordering."""

from __future__ import annotations

import json
from collections import deque

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.preparation import MetadataCache


class RemovalCursor:
    """Resume after the last offered chunk, retaining no executable old edit."""

    scale_quantum = 8

    def __init__(self, group):
        self.items = group.items
        self.size = max(1, len(self.items) // 4 - 1)
        self.anchor = self.items[0] if self.items else None
        self.done = not self.items
        self.without_accept = 0

    def state(self):
        return self.items, self.size, self.anchor, self.done, self.without_accept

    def restore(self, state):
        self.items, self.size, self.anchor, self.done, self.without_accept = state

    def accepted(self):
        self.without_accept = 0

    def next_candidate(self, model, group):
        live = set(group.items)
        previous = set(self.items)
        new = [item for item in group.items if item not in previous]
        if self.anchor is not None and self.anchor not in live:
            start = self.items.index(self.anchor)
            self.anchor = next((item for item in self.items[start:] if item in live), None)
        if new and (self.done or self.anchor is None):
            self.anchor = new[0]
            self.done = False
        self.items = group.items
        self.size = min(self.size, max(1, len(self.items) // 4 - 1))
        while self.items and not self.done:
            if self.size > 1 and self.without_accept >= self.scale_quantum:
                self.size = max(1, self.size // 2)
                self.anchor = self.items[0]
                self.without_accept = 0
            if self.anchor is None:
                if self.size == 1:
                    self.done = True
                    break
                self.size = max(1, self.size // 2)
                self.without_accept = 0
                self.anchor = self.items[0]
            start = self.items.index(self.anchor)
            selected = self.items[start : start + self.size]
            following = start + len(selected)
            self.anchor = self.items[following] if following < len(self.items) else None
            if group.keep_one and len(selected) == len(self.items):
                selected = selected[:-1]
            if not selected:
                continue
            indices = (
                tuple(model.id_to_index[item] for item in selected) if group.nodes else selected
            )
            self.without_accept += 1
            return Candidate((group.edit_type(indices),), base_digest=model.canonical_hash())
        return None


def history_key(attempt, model):
    edits = attempt.candidate.edits
    if len(edits) != 1:
        return None
    edit = edits[0]
    if isinstance(edit, DeleteNodes):
        if not hasattr(model, "node_ids") or any(
            i < 0 or i >= len(model.node_ids) for i in edit.indices
        ):
            return None
        items = tuple(dict.fromkeys(model.node_ids[i] for i in edit.indices))
    elif hasattr(edit, "names"):
        items = edit.names
    else:
        return None
    # Retain order: promotions can change bytes if deletion order changes.
    return json.dumps([attempt.pass_index, type(edit).__name__, items])


class RejectionOrder:
    """Old failures only delay work. They never become candidate verdicts."""

    quantum = 8

    def __init__(self):
        self.cache = MetadataCache("history")
        self.deferred = 0
        self.retried = 0
        self.pending = deque()

    def observe(self, attempt, model, feedback):
        key = history_key(attempt, model)
        if key is None:
            return
        if feedback.status in {"accepted", "inconclusive"}:
            self.cache.discard(key)
        elif feedback.status in {"invalid", "edit_failed", "not_cheaper", "not_interesting"}:
            self.cache.put(
                key, json.dumps({"base": attempt.base_digest, "status": feedback.status}).encode()
            )

    def order(self, attempts, model):
        pending = self.pending = deque()
        ordinary = 0
        looked = 0
        for attempt in attempts:
            key = history_key(attempt, model)
            record = self.cache.get(key) if key is not None else None
            if record is not None and json.loads(record)["base"] != attempt.base_digest:
                pending.append(attempt)
                self.deferred += 1
                looked += 1
            else:
                yield attempt
                ordinary += 1
                looked = 0
            # Bounded lookahead and a fair share even when all remaining work
            # is deferred. The generator is discarded after any acceptance.
            if pending and (
                ordinary >= self.quantum or looked >= self.quantum or len(pending) >= self.quantum
            ):
                self.retried += 1
                yield pending.popleft()
                ordinary = looked = 0
        while pending:
            self.retried += 1
            yield pending.popleft()
