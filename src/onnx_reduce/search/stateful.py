"""Adaptive partition search followed by a deterministic coverage sweep."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from onnx_reduce.passes.base import NodeSearchPass, ReductionPass
from onnx_reduce.performance import measure, measured, timed_candidates
from onnx_reduce.preparation import edit_key
from onnx_reduce.search.coverage import iter_coverage
from onnx_reduce.search.regions import NodeRegion
from onnx_reduce.search.state import SearchState
from onnx_reduce.search.types import NODE_SCOPE, ScheduledAttempt
from onnx_reduce.types import ModelAdapter, NodeSearchModel


class _NodeSearch:
    """Lazy partition cursor; no executable candidate survives a model change."""

    def __init__(self, reduction_pass: NodeSearchPass, model: NodeSearchModel):
        self.reduction_pass = reduction_pass
        self.size = max(1, len(model.node_ids) // 2)
        self.preferred_direction = "delete"
        self.ids: tuple[int, ...] = ()
        self.anchor: int | None = None
        self.deduplicated = 0
        self.max_seen = 4096
        self.rebase(model)

    def rebase(self, model: NodeSearchModel, attempt: ScheduledAttempt | None = None) -> None:
        if attempt is not None and attempt.scale is not None:
            self.size = attempt.scale
            self.preferred_direction = attempt.direction
        if self.ids and self.anchor in self.ids:
            start = self.ids.index(self.anchor)
            self.anchor = next(
                (
                    self.ids[(start + offset) % len(self.ids)]
                    for offset in range(len(self.ids))
                    if self.ids[(start + offset) % len(self.ids)] in model.id_to_index
                ),
                None,
            )
        self.ids = model.node_ids
        self.size = max(1, min(self.size, len(self.ids)))
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._region = NodeRegion(())
        self._directions = iter(())
        self._exhausted = not self.ids
        self._start_scale(model)

    def _start_scale(self, model: NodeSearchModel) -> None:
        start = model.id_to_index.get(self.anchor, 0) // self.size
        self._regions = iter(self.reduction_pass.node_regions(model, self.size, start=start))

    def next_choice(self, model, digest):
        while not self._exhausted:
            try:
                direction = next(self._directions)
            except StopIteration:
                try:
                    self._region = NodeRegion(next(self._regions))
                except StopIteration:
                    if self.size == 1:
                        self._exhausted = True
                        return None
                    self.size = max(1, self.size // 2)
                    self.anchor = None
                    self._start_scale(model)
                    continue
                last = model.id_to_index[self._region.ids[-1]]
                self.anchor = self.ids[(last + 1) % len(self.ids)]
                other = "keep" if self.preferred_direction == "delete" else "delete"
                self._directions = iter((self.preferred_direction, other))
                continue
            candidate = self._region.candidate(model, direction, digest)
            if candidate is None:
                continue
            key = edit_key(candidate, "scheduler")
            if key in self._seen:
                self.deduplicated += 1
                continue
            if self.max_seen:
                self._seen[key] = None
                if len(self._seen) > self.max_seen:
                    self._seen.popitem(last=False)
            return candidate, direction, self.size, self._region.ids
        return None


class StatefulScheduler:
    """Continue useful phases/scales, then verify a final deterministic sweep."""

    name = "stateful"
    stop_reason = "fixed_point"
    phase_quantum = 8

    def __init__(self, passes: Sequence[ReductionPass], model: ModelAdapter):
        self._state = SearchState(passes, model)
        node_passes = {
            index: reduction_pass
            for index, reduction_pass in enumerate(self._state.passes)
            if isinstance(reduction_pass, NodeSearchPass)
        }
        if node_passes and not isinstance(model, NodeSearchModel):
            raise ValueError("stateful node search requires node_ids and id_to_index on the model")
        self._nodes = {index: _NodeSearch(item, model) for index, item in node_passes.items()}
        if self._nodes:
            self._state.scope = NODE_SCOPE
        self._mode = "heuristic"
        self._phase = 0
        self._used = 0
        self._done = [False] * len(self._state.passes)
        self._generators = [None] * len(self._state.passes)
        self._coverage_iterator = None

    def _advance(self) -> None:
        self._phase = (self._phase + 1) % len(self._state.passes)
        self._used = 0

    def _cover(self, model):
        for attempt in iter_coverage(
            self._state.passes, model, self._state.model_version, self._state.base_digest
        ):
            # Preserve phase-revisit accounting as well as the historical order.
            yield self._state.attempt(
                attempt.candidate,
                attempt.pass_index,
                direction=attempt.direction,
                scale=attempt.scale,
                region_ids=attempt.region_ids,
            )

    @measured("scheduler")
    def next_attempt(self, model: ModelAdapter) -> ScheduledAttempt | None:
        while self._mode == "heuristic":
            if all(self._done):
                self._mode = "coverage"
                self._coverage_iterator = iter(self._cover(model))
                break
            index = self._phase
            if self._done[index] or self._used >= self.phase_quantum:
                self._advance()
                continue
            if index in self._nodes:
                cursor = self._nodes[index]
                before = cursor.deduplicated
                with measure("candidate_generation"):
                    choice = cursor.next_choice(model, self._state.base_digest)
                self._state.deduplicated += cursor.deduplicated - before
                if choice is None:
                    self._done[index] = True
                    self._advance()
                    continue
                candidate, direction, size, ids = choice
                attempt = self._state.attempt(
                    candidate,
                    index,
                    source="heuristic",
                    direction=direction,
                    scale=size,
                    region_ids=ids,
                )
            else:
                if self._generators[index] is None:
                    self._generators[index] = iter(
                        timed_candidates(self._state.passes[index], model)
                    )
                try:
                    candidate = next(self._generators[index])
                except StopIteration:
                    self._done[index] = True
                    self._advance()
                    continue
                attempt = self._state.attempt(candidate, index, source="heuristic")
            self._used += 1
            return attempt
        try:
            return next(self._coverage_iterator)
        except StopIteration:
            self._state.complete()
            return None

    @measured("scheduler")
    def observe(self, attempt, feedback, accepted_model=None) -> None:
        self._state.record(attempt, feedback)
        if accepted_model is None:
            return
        self._state.accept(accepted_model)
        for index, cursor in self._nodes.items():
            hint = attempt if attempt.pass_index == index and attempt.scale is not None else None
            cursor.rebase(accepted_model, hint)
            if hint is not None and accepted_model.node_ids:
                self._state.scale_reuses += 1
        self._generators = [None] * len(self._state.passes)
        self._done = [False] * len(self._state.passes)
        self._coverage_iterator = None
        if self._mode == "coverage":
            self._phase = attempt.pass_index
            self._used = 1
        self._mode = "heuristic"

    def coverage(self):
        return self._state.coverage()

    def to_dict(self):
        return self._state.to_dict(self.name, self.coverage(), phase_quantum=self.phase_quantum)
