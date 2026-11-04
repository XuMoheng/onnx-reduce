"""Independent, version-bound fixed-point scanning without heuristic restarts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace

from onnx_reduce.edits import DeleteNodes
from onnx_reduce.passes.base import NodeSearchPass, ReductionPass
from onnx_reduce.passes.delete_nodes import _granularities
from onnx_reduce.performance import measured, timed_candidates
from onnx_reduce.preparation import edit_key
from onnx_reduce.search.regions import NodeRegion, graph_regions, region_choices
from onnx_reduce.search.types import NODE_SCOPE, PASS_SCOPE, CoverageSummary, ScheduledAttempt
from onnx_reduce.types import ModelAdapter, NodeSearchModel

REGION_SCOPE = NODE_SCOPE + " and module, chain and topology region delete/keep candidates (v1)"


def iter_coverage(passes, model, version, digest, *, include_regions=False):
    """Preserve existing stateful order; append the declared region set when enabled."""
    node_passes = [index for index, item in enumerate(passes) if isinstance(item, NodeSearchPass)]
    for index, reduction_pass in enumerate(passes):
        for candidate in timed_candidates(reduction_pass, model):
            metadata = {}
            if index in node_passes and len(candidate.edits) == 1:
                edit = candidate.edits[0]
                if (
                    isinstance(edit, DeleteNodes)
                    and edit.indices
                    and all(0 <= i < len(model.node_ids) for i in edit.indices)
                ):
                    metadata = {
                        "direction": "delete",
                        "scale": len(edit.indices),
                        "region_ids": tuple(model.node_ids[i] for i in edit.indices),
                    }
            yield ScheduledAttempt(candidate, reduction_pass, index, version, digest, **metadata)
    for index in node_passes:
        reduction_pass = passes[index]
        for size in _granularities(len(model.node_ids)):
            for ids in reduction_pass.node_regions(model, size):
                candidate = NodeRegion(ids).candidate(model, "keep", digest)
                if candidate is not None:
                    yield ScheduledAttempt(
                        candidate,
                        reduction_pass,
                        index,
                        version,
                        digest,
                        direction="keep",
                        scale=size,
                        region_ids=ids,
                    )
    if include_regions and node_passes:
        # Region semantics do not depend on which node pass opts in. One owner
        # is enough; custom node passes' own partitions are scanned above too.
        index = node_passes[0]
        for candidate, direction, region in region_choices(model, graph_regions(model)):
            yield ScheduledAttempt(
                candidate,
                passes[index],
                index,
                version,
                digest,
                direction=direction,
                scale=len(region.node_ids),
                region_ids=region.node_ids,
                region_source=region.source,
            )


class Finalizer:
    """Scan all configured candidates on each accepted base, never return to search."""

    scope_version = 1
    max_seen = 4096

    def __init__(self, passes: Sequence[ReductionPass], model: ModelAdapter, *, model_version=0):
        self.passes = tuple(passes)
        self._has_nodes = any(isinstance(item, NodeSearchPass) for item in passes)
        if self._has_nodes and not isinstance(model, NodeSearchModel):
            raise ValueError("region finalization requires node_ids and id_to_index on the model")
        self.scope = REGION_SCOPE if self._has_nodes else PASS_SCOPE
        self.model_version = model_version
        self.restarts = 0
        self.attempts = 0
        self.deduplicated = 0
        self._reset(model)

    def _reset(self, model):
        self.base_digest = model.canonical_hash()
        self.completed = False
        self.inconclusive = 0
        self._seen = OrderedDict()
        self._iterator = iter(
            iter_coverage(
                self.passes,
                model,
                self.model_version,
                self.base_digest,
                include_regions=True,
            )
        )

    @measured("finalization")
    def next_attempt(self, model):
        if model.canonical_hash() != self.base_digest:
            raise ValueError("finalizer base changed without accepted feedback")
        for attempt in self._iterator:
            candidate = attempt.candidate
            key = edit_key(
                replace(
                    candidate,
                    base_digest=(
                        self.base_digest if candidate.base_digest is None else candidate.base_digest
                    ),
                ),
                "finalization",
            )
            if key in self._seen:
                self.deduplicated += 1
                continue
            if self.max_seen:
                self._seen[key] = None
                if len(self._seen) > self.max_seen:
                    self._seen.popitem(last=False)
            return attempt
        self.completed = True
        return None

    @measured("finalization")
    def observe(self, feedback, accepted_model=None):
        self.attempts += 1
        if accepted_model is not None:
            self.model_version += 1
            self.restarts += 1
            self._reset(accepted_model)
        elif feedback.status == "inconclusive":
            self.inconclusive += 1

    def coverage(self):
        return CoverageSummary(
            self.base_digest,
            self.model_version,
            self.scope,
            self.completed,
            self.inconclusive,
        )
