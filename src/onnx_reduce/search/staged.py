"""Fair coarse, region, fine and deep search followed by independent finalization."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace

from onnx_reduce.edits import DeleteNodes
from onnx_reduce.passes.base import CursorPass, NodeSearchPass, StagedPass
from onnx_reduce.performance import measure, measured, timed_candidates
from onnx_reduce.search.coverage import REGION_SCOPE, Finalizer
from onnx_reduce.search.cursors import RejectionOrder, RemovalCursor
from onnx_reduce.search.regions import graph_regions, region_choices
from onnx_reduce.search.state import SearchState
from onnx_reduce.search.types import PASS_SCOPE
from onnx_reduce.types import NodeSearchModel


class StagedScheduler:
    name = "staged"
    stages = ("coarse", "regions", "fine", "deep")
    pass_quantum = 8
    deep_no_progress_limit = 256
    no_progress_limit = 64
    cleanup_quantum = 2

    def __init__(
        self,
        passes,
        model,
        *,
        goal="complete",
        finalize_only=False,
        search_policy="restart",
        history_order="off",
        direction_order="fixed",
    ):
        self._state = SearchState(passes, model)
        self.goal = goal
        self.search_policy = search_policy
        self.direction_order = direction_order
        self.history_order = history_order
        self._history = RejectionOrder() if history_order == "defer" else None
        if (
            search_policy == "cursor"
            and any(
                isinstance(item, CursorPass) and item.requires_node_ids
                for item in self._state.passes
            )
            and not isinstance(model, NodeSearchModel)
        ):
            raise ValueError("cursor node editing requires node_ids and id_to_index on the model")
        self._fine_cursors = {}
        self._coarse_start = 0
        self._region_anchor = None
        self._direction_stats = {"delete": [0, 0], "keep": [0, 0]}
        self._current_model = model
        self._node_passes = [i for i, item in enumerate(passes) if isinstance(item, NodeSearchPass)]
        if self._node_passes and not isinstance(model, NodeSearchModel):
            raise ValueError("staged node search requires node_ids and id_to_index on the model")
        self._state.scope = REGION_SCOPE if self._node_passes else PASS_SCOPE
        self._stage_index = len(self.stages) if finalize_only else 0
        self._turn_pass = 0
        self._turn_used = 0
        self._turn_budget = None
        self._pass_history = {}
        self._pass_failures = {}
        self._pass_states = {}
        self._iterator = None
        self._cleanup_iterator = None
        self._cleanup_used = 0
        self._cleanup_pending = False
        self._without_accept = 0
        self._stage_ends = {}
        self._stage_cost_delta = {}
        self._finalizer = None
        self.stop_reason = "search_exhausted"
        self._finalize_only = finalize_only

    @property
    def stage(self):
        return (
            self.stages[self._stage_index]
            if self._stage_index < len(self.stages)
            else "finalization"
        )

    def _one_pass(self, model, stage, index):
        reduction_pass = self._state.passes[index]
        if (
            stage == "deep"
            and isinstance(reduction_pass, NodeSearchPass)
            and self._region_anchor in model.id_to_index
        ):
            from onnx_reduce.search.regions import NodeRegion

            anchor = model.id_to_index[self._region_anchor]
            for width in (1, 2, 4, 8):
                ids = model.node_ids[max(0, anchor - width // 2) : anchor + max(1, width // 2)]
                for direction in ("delete", "keep"):
                    candidate = NodeRegion(ids).candidate(model, direction, self._state.base_digest)
                    if candidate is not None:
                        yield self._state.attempt(
                            candidate,
                            index,
                            source=stage,
                            stage=stage,
                            direction=direction,
                            region_ids=ids,
                            scale=len(ids),
                        )
        if (
            stage == "fine"
            and self.search_policy == "cursor"
            and isinstance(reduction_pass, CursorPass)
        ):
            for group in reduction_pass.removal_groups(model):
                key = (index, group.key)
                cursor = self._fine_cursors.setdefault(key, RemovalCursor(group))
                while True:
                    state = cursor.state()
                    candidate = cursor.next_candidate(model, group)
                    if candidate is None:
                        break
                    yield self._state.attempt(
                        candidate,
                        index,
                        source=stage,
                        stage=stage,
                        cursor_key=key,
                        cursor_state=state,
                    )
        elif isinstance(reduction_pass, StagedPass) and stage != "deep":
            with measure("candidate_generation"), measure(f"generate/{reduction_pass.name}"):
                iterator = iter(reduction_pass.generate_stage(model, stage))
            while True:
                with measure("candidate_generation"), measure(f"generate/{reduction_pass.name}"):
                    candidate = next(iterator, None)
                if candidate is None:
                    break
                yield self._state.attempt(candidate, index, source=stage, stage=stage)
        elif stage in {"fine", "deep"}:
            for candidate in timed_candidates(reduction_pass, model):
                yield self._state.attempt(candidate, index, source=stage, stage=stage)

    def _quantum(self, index):
        history = self._pass_history.get(index, ())
        # Deterministic success feedback, never wall-clock timings. Every pass
        # retains an eight-attempt turn; productive passes get at most sixteen.
        return self.pass_quantum * (
            2 if len(history) >= 8 and sum(history) * 4 >= len(history) else 1
        )

    def _pass_candidates(self, model, stage):
        indices = list(range(len(self._state.passes)))
        start = self._turn_pass if stage != "cleanup" else 0
        indices = indices[start:] + indices[:start]
        queue = deque((index, iter(self._one_pass(model, stage, index))) for index in indices)
        for index in indices:
            self._pass_states.setdefault(f"{stage}/{self._state.passes[index].name}", "not_visited")
        limit = (
            self.deep_no_progress_limit
            if stage == "deep" and self.no_progress_limit
            else self.no_progress_limit
        )
        while queue:
            index, stream = queue.popleft()
            key = (stage, index)
            if self._pass_failures.get(key, 0) >= limit:
                self._pass_states[f"{stage}/{self._state.passes[index].name}"] = "plateau"
                continue
            state_key = f"{stage}/{self._state.passes[index].name}"
            yielded = False
            quantum = self._quantum(index)
            if index == self._turn_pass and stage != "cleanup":
                quantum = max(1, (self._turn_budget or quantum) - self._turn_used)
            for _ in range(quantum):
                if self._pass_failures.get(key, 0) >= limit:
                    break
                attempt = next(stream, None)
                if attempt is None:
                    self._pass_states[state_key] = (
                        "exhausted"
                        if yielded or self._pass_states.get(state_key) == "visited"
                        else "no_candidates"
                    )
                    break
                yielded = True
                self._pass_states[state_key] = "visited"
                yield attempt
            else:
                queue.append((index, stream))
                continue
            if self._pass_failures.get(key, 0) >= limit:
                self._pass_states[state_key] = "plateau"

    def _generate(self, model):
        attempts = self._generate_raw(model)
        if self._history is not None:
            yield from self._history.order(attempts, model)
        else:
            yield from attempts

    def _generate_raw(self, model):
        stage = self.stage
        if stage == "regions":
            if self._node_passes:
                index = self._node_passes[0]
                # Singleton fallback is in fine search and in the full finalizer.
                regions = graph_regions(model, minimum_size=2)
                if self.search_policy == "cursor" and self._region_anchor is not None:
                    regions = tuple(
                        sorted(regions, key=lambda r: self._region_anchor not in r.node_ids)
                    )
                choices = region_choices(model, regions)
                if self.direction_order == "adaptive":
                    from onnx_reduce.search.regions import NodeRegion

                    delete, keep = (self._direction_stats[d] for d in ("delete", "keep"))
                    preferred = (
                        "keep"
                        if (keep[0] + 1) * (delete[1] + 1) > (delete[0] + 1) * (keep[1] + 1)
                        else "delete"
                    )
                    directions = (preferred, "delete" if preferred == "keep" else "keep")
                    choices = (
                        (candidate, direction, region)
                        for region in regions
                        for direction in directions
                        if (
                            candidate := NodeRegion(region.node_ids).candidate(
                                model, direction, self._state.base_digest
                            )
                        )
                        is not None
                    )
                for candidate, direction, region in choices:
                    yield self._state.attempt(
                        candidate,
                        index,
                        source=stage,
                        stage=stage,
                        direction=direction,
                        scale=len(region.node_ids),
                        region_ids=region.node_ids,
                        region_source=region.source,
                    )
        else:
            yield from self._pass_candidates(model, stage)

    @measured("scheduler")
    def next_attempt(self, model):
        self._current_model = model
        while self._stage_index < (3 if self.goal == "fast" else len(self.stages)):
            if self._cleanup_pending:
                if self._cleanup_iterator is None:
                    self._cleanup_iterator = iter(self._pass_candidates(model, "cleanup"))
                if self._cleanup_used < self.cleanup_quantum:
                    attempt = next(self._cleanup_iterator, None)
                    if attempt is not None:
                        self._cleanup_used += 1
                        return attempt
                self._cleanup_pending = False
                self._cleanup_iterator = None
            if self.stage == "regions" and self._without_accept >= self.no_progress_limit:
                attempt = None
                reason = "plateau"
            else:
                if self._iterator is None:
                    self._iterator = iter(self._generate(model))
                attempt = next(self._iterator, None)
                reason = "exhausted"
            if attempt is not None:
                return attempt
            if any(
                state == "plateau"
                for key, state in self._pass_states.items()
                if key.startswith(self.stage + "/")
            ):
                reason = "plateau"
            self._stage_ends[self.stage] = reason
            self._stage_index += 1
            if self._history is not None:
                self._history.pending.clear()
            self._direction_stats = {"delete": [0, 0], "keep": [0, 0]}
            if self._stage_index >= 3:
                self._fine_cursors.clear()
            self._iterator = None
            self._without_accept = 0
        if self.goal == "fast":
            self.stop_reason = (
                "plateau" if "plateau" in self._stage_ends.values() else "search_exhausted"
            )
            return None
        if self._finalizer is None:
            self._finalizer = Finalizer(
                self._state.passes, model, model_version=self._state.model_version
            )
        attempt = self._finalizer.next_attempt(model)
        if attempt is None:
            self.stop_reason = (
                "coverage_inconclusive" if self._finalizer.inconclusive else "fixed_point"
            )
            return None
        # Keep the usual visit statistics while retaining finalizer-owned cursors.
        return self._state.attempt(
            attempt.candidate,
            attempt.pass_index,
            direction=attempt.direction,
            scale=attempt.scale,
            region_ids=attempt.region_ids,
            stage="finalization",
            region_source=attempt.region_source,
        )

    @measured("scheduler")
    def observe(self, attempt, feedback, accepted_model=None):
        self._state.record(attempt, feedback)
        if self._finalizer is None:
            if attempt.stage != "cleanup":
                if self._turn_pass != attempt.pass_index:
                    self._turn_used = 0
                    self._turn_budget = None
                if self._turn_budget is None:
                    self._turn_budget = self._quantum(attempt.pass_index)
                self._pass_history.setdefault(attempt.pass_index, deque(maxlen=16)).append(
                    feedback.status == "accepted"
                )
                self._turn_pass = attempt.pass_index
                self._turn_used += 1
                if self._turn_used >= self._turn_budget:
                    self._turn_pass = (attempt.pass_index + 1) % max(1, len(self._state.passes))
                    self._turn_used = 0
                    self._turn_budget = None
                key = (attempt.stage, attempt.pass_index)
                self._pass_failures[key] = self._pass_failures.get(key, 0) + 1
            if self._history is not None:
                self._history.observe(attempt, self._current_model, feedback)
            if attempt.direction in self._direction_stats:
                stats = self._direction_stats[attempt.direction]
                stats[0] += feedback.status == "accepted"
                stats[1] += feedback.failure_check_calls
        if self._finalizer is not None:
            self._finalizer.observe(feedback, accepted_model)
        if accepted_model is None:
            if attempt.stage != "cleanup":
                self._without_accept += 1
            return
        if self._history is not None:
            # Undo lookahead for unexecuted fine attempts before discarding this
            # version's generator; only immutable identities/cursor state survive.
            restored = set()
            for pending in self._history.pending:
                if pending.cursor_key is not None and pending.cursor_key not in restored:
                    self._fine_cursors[pending.cursor_key].restore(pending.cursor_state)
                    restored.add(pending.cursor_key)
            self._history.pending.clear()
        self._state.accept(accepted_model)
        self._pass_failures.clear()
        self._pass_states.clear()
        for cursor in self._fine_cursors.values():
            cursor.accepted()
        if attempt.stage == "coarse" and self._state.passes:
            self._coarse_start = (attempt.pass_index + 1) % len(self._state.passes)
        affected_ids = attempt.region_ids
        if not affected_ids and isinstance(self._current_model, NodeSearchModel):
            affected_ids = tuple(
                self._current_model.node_ids[i]
                for edit in attempt.candidate.edits
                for i in getattr(edit, "indices", ())
                if 0 <= i < len(self._current_model.node_ids)
            )
        if affected_ids:
            old_ids = self._current_model.node_ids
            old_positions = self._current_model.id_to_index
            end = max(old_positions[item] for item in affected_ids)
            neighborhood = old_ids[end + 1 :] + old_ids[: end + 1]
            self._region_anchor = next(
                (item for item in neighborhood if item in accepted_model.id_to_index), None
            )
        deltas = self._stage_cost_delta.setdefault(
            attempt.stage, dict.fromkeys(asdict(feedback.base_cost), 0)
        )
        for name, before in asdict(feedback.base_cost).items():
            deltas[name] += before - getattr(feedback.candidate_cost, name)
        self._without_accept = 0
        self._iterator = None
        self._cleanup_iterator = None
        if self._finalizer is not None:
            return
        if attempt.stage != "cleanup" and any(
            isinstance(edit, DeleteNodes) for edit in attempt.candidate.edits
        ):
            self._cleanup_pending = True
            self._cleanup_used = 0

    def coverage(self):
        if self._finalizer is not None:
            return self._finalizer.coverage()
        return replace(self._state.coverage(), completed=False)

    def to_dict(self):
        return {
            **self._state.to_dict(self.name, self.coverage()),
            "goal": self.goal,
            "search_policy": self.search_policy,
            "cursor_scale_quantum": RemovalCursor.scale_quantum
            if self.search_policy == "cursor"
            else None,
            "direction_order": self.direction_order,
            "history_order": self.history_order,
            "history": {
                "deferred": self._history.deferred,
                "retried": self._history.retried,
                "entries": len(self._history.cache),
                "bytes": self._history.cache.byte_size,
            }
            if self._history
            else None,
            "stage": self.stage if self.goal == "complete" or self._stage_index < 3 else "done",
            "pass_quantum": self.pass_quantum,
            "max_pass_quantum": self.pass_quantum * 2,
            "deep_no_progress_limit": self.deep_no_progress_limit,
            "pass_states": dict(self._pass_states),
            "pass_state_model_version": self._state.model_version,
            "stage_ends": dict(self._stage_ends),
            "stage_accepted_cost_delta": {
                key: dict(value) for key, value in self._stage_cost_delta.items()
            },
            "no_progress_limit": self.no_progress_limit,
            "cleanup_quantum": self.cleanup_quantum,
            "finalize_only": self._finalize_only,
            "target_origin": getattr(self, "target_origin", None),
            "scope_version": Finalizer.scope_version,
            "region_sources": ["module", "structure", "topology"] if self._node_passes else [],
            "finalization": {
                "started": self._finalizer is not None,
                "restarts": self._finalizer.restarts if self._finalizer else 0,
                "attempts": self._finalizer.attempts if self._finalizer else 0,
                "deduplicated": self._finalizer.deduplicated if self._finalizer else 0,
            },
        }
