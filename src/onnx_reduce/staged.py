"""Coarse, region and fine search followed by an independent optional finalizer."""

from __future__ import annotations

from dataclasses import asdict, replace

from onnx_reduce.edits import DeleteNodes
from onnx_reduce.finalizer import REGION_SCOPE, Finalizer
from onnx_reduce.passes.base import NodeSearchPass, StagedPass
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

    def __init__(self, passes, model, *, goal="complete", finalize_only=False):
        super().__init__(passes, model)
        self.goal = goal
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
        for index, reduction_pass in enumerate(self.passes):
            if isinstance(reduction_pass, StagedPass):
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
        stage = self.stage
        if stage == "regions":
            if self._node_passes:
                index = self._node_passes[0]
                # Singleton fallback is in fine search and in the full finalizer.
                for candidate, direction, region in region_choices(
                    model, graph_regions(model, minimum_size=2)
                ):
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
        if self._finalizer is not None:
            self._finalizer.observe(feedback, accepted_model)
        if accepted_model is None:
            if attempt.stage != "cleanup":
                self._without_accept += 1
            return
        self._accept(accepted_model)
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
