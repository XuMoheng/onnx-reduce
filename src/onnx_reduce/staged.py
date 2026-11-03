"""Coarse, region and fine search followed by an independent optional finalizer."""

from __future__ import annotations

from dataclasses import asdict, replace

from onnx_reduce.cursors import RejectionOrder, RemovalCursor
from onnx_reduce.edits import DeleteNodes
from onnx_reduce.finalizer import REGION_SCOPE, Finalizer
from onnx_reduce.passes.base import CursorPass, NodeSearchPass, StagedPass
from onnx_reduce.performance import measure, measured, timed_candidates
from onnx_reduce.regions import graph_regions, region_choices
from onnx_reduce.scheduler import LegacyScheduler
from onnx_reduce.search_types import PASS_SCOPE
from onnx_reduce.types import NodeSearchModel


class StagedScheduler(LegacyScheduler):
    name = "staged"
    stages = ("coarse", "regions", "fine")
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
        super().__init__(passes, model)
        self.goal = goal
        self.search_policy = search_policy
        self.direction_order = direction_order
        self.history_order = history_order
        self._history = RejectionOrder() if history_order == "defer" else None
        if (
            search_policy == "cursor"
            and any(isinstance(item, CursorPass) and item.requires_node_ids for item in self.passes)
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
        self.scope = REGION_SCOPE if self._node_passes else PASS_SCOPE
        self._stage_index = 3 if finalize_only else 0
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
        return self.stages[self._stage_index] if self._stage_index < 3 else "finalization"

    def _pass_candidates(self, model, stage):
        indices = list(range(len(self.passes)))
        if stage == "coarse" and self.search_policy == "cursor":
            indices = indices[self._coarse_start :] + indices[: self._coarse_start]
        for index in indices:
            reduction_pass = self.passes[index]
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
                        yield self._attempt(
                            candidate,
                            index,
                            source=stage,
                            stage=stage,
                            cursor_key=key,
                            cursor_state=state,
                        )
            elif isinstance(reduction_pass, StagedPass):
                with measure("candidate_generation"):
                    iterator = iter(reduction_pass.generate_stage(model, stage))
                while True:
                    with measure("candidate_generation"):
                        candidate = next(iterator, None)
                    if candidate is None:
                        break
                    yield self._attempt(candidate, index, source=stage, stage=stage)
            elif stage == "fine":
                for candidate in timed_candidates(reduction_pass, model):
                    yield self._attempt(candidate, index, source=stage, stage=stage)

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
                    from onnx_reduce.regions import NodeRegion

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
                                model, direction, self.base_digest
                            )
                        )
                        is not None
                    )
                for candidate, direction, region in choices:
                    yield self._attempt(
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
        while self._stage_index < 3:
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
            if self._without_accept >= self.no_progress_limit:
                attempt = None
                reason = "plateau"
            else:
                if self._iterator is None:
                    self._iterator = iter(self._generate(model))
                attempt = next(self._iterator, None)
                reason = "exhausted"
            if attempt is not None:
                return attempt
            self._stage_ends[self.stage] = reason
            self._stage_index += 1
            if self._history is not None:
                self._history.pending.clear()
            self._direction_stats = {"delete": [0, 0], "keep": [0, 0]}
            if self._stage_index == 3:
                self._fine_cursors.clear()
            self._iterator = None
            self._without_accept = 0
        if self.goal == "fast":
            self.stop_reason = (
                "plateau" if "plateau" in self._stage_ends.values() else "search_exhausted"
            )
            return None
        if self._finalizer is None:
            self._finalizer = Finalizer(self.passes, model, model_version=self.model_version)
        attempt = self._finalizer.next_attempt(model)
        if attempt is None:
            self.stop_reason = (
                "coverage_inconclusive" if self._finalizer.inconclusive else "fixed_point"
            )
            return None
        # Keep the usual visit statistics while retaining finalizer-owned cursors.
        return self._attempt(
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
        self._record(attempt, feedback)
        if self._finalizer is None:
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
        self._accept(accepted_model)
        for cursor in self._fine_cursors.values():
            cursor.accepted()
        if attempt.stage == "coarse" and self.passes:
            self._coarse_start = (attempt.pass_index + 1) % len(self.passes)
        if attempt.region_ids:
            old_ids = self._current_model.node_ids
            old_positions = self._current_model.id_to_index
            end = max(old_positions[item] for item in attempt.region_ids)
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
        return replace(super().coverage(), completed=False)

    def to_dict(self):
        return {
            **super().to_dict(),
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
