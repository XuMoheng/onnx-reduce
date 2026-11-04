"""Fair coarse, region, fine and deep search followed by independent finalization."""

from __future__ import annotations

from dataclasses import replace

from onnx_reduce.edits import DeleteNodes
from onnx_reduce.passes.base import CursorPass, NodeSearchPass, StagedPass
from onnx_reduce.performance import measure, measured, timed_candidates
from onnx_reduce.search.coverage import REGION_SCOPE, Finalizer
from onnx_reduce.search.cursors import RejectionOrder, RemovalCursor
from onnx_reduce.search.progress import StageProgress
from onnx_reduce.search.regions import graph_regions, region_choices
from onnx_reduce.search.rotation import PassRotation
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
        self._progress = StageProgress(self.stages, len(self.stages) if finalize_only else 0)
        self._rotation = PassRotation()
        self._iterator = None
        self._cleanup_iterator = None
        self._cleanup_used = 0
        self._cleanup_pending = False
        self._finalizer = None
        self.stop_reason = "search_exhausted"
        self._finalize_only = finalize_only

    @property
    def stage(self):
        return self._progress.stage

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

    def _pass_candidates(self, model, stage):
        limit = (
            self.deep_no_progress_limit
            if stage == "deep" and self.no_progress_limit
            else self.no_progress_limit
        )
        yield from self._rotation.candidates(
            model,
            stage,
            passes=self._state.passes,
            generate=self._one_pass,
            quantum=self.pass_quantum,
            limit=limit,
        )

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
        while self._progress.index < (3 if self.goal == "fast" else len(self.stages)):
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
            if self.stage == "regions" and self._progress.without_accept >= self.no_progress_limit:
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
                for key, state in self._rotation.pass_states.items()
                if key.startswith(self.stage + "/")
            ):
                reason = "plateau"
            self._progress.advance(reason)
            if self._history is not None:
                self._history.pending.clear()
            self._direction_stats = {"delete": [0, 0], "keep": [0, 0]}
            if self._progress.index >= 3:
                self._fine_cursors.clear()
            self._iterator = None
        if self.goal == "fast":
            self.stop_reason = (
                "plateau" if "plateau" in self._progress.ends.values() else "search_exhausted"
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
            self._rotation.observe(attempt, feedback, len(self._state.passes), self.pass_quantum)
            if self._history is not None:
                self._history.observe(attempt, self._current_model, feedback)
            if attempt.direction in self._direction_stats:
                stats = self._direction_stats[attempt.direction]
                stats[0] += feedback.status == "accepted"
                stats[1] += feedback.failure_check_calls
        if self._finalizer is not None:
            self._finalizer.observe(feedback, accepted_model)
        if accepted_model is None:
            self._progress.rejected(attempt.stage)
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
        self._rotation.accepted()
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
        self._progress.accepted(attempt.stage, feedback)
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
            "stage": self.stage if self.goal == "complete" or self._progress.index < 3 else "done",
            "pass_quantum": self.pass_quantum,
            "max_pass_quantum": self.pass_quantum * 2,
            "deep_no_progress_limit": self.deep_no_progress_limit,
            "pass_states": dict(self._rotation.pass_states),
            "pass_state_model_version": self._state.model_version,
            "stage_ends": dict(self._progress.ends),
            "stage_accepted_cost_delta": {
                key: dict(value) for key, value in self._progress.cost_delta.items()
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
