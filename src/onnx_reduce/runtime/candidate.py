"""Execute one candidate through preparation, validation and failure matching."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from onnx_reduce.config import ExecutionConfig
from onnx_reduce.edits import Candidate
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.evaluator import Evaluation, EvaluationBudgetExceeded, Evaluator
from onnx_reduce.performance import current_performance, sample
from onnx_reduce.preparation import PreparationCaches, edit_key
from onnx_reduce.runtime.budget import SearchBudget
from onnx_reduce.runtime.result import AttemptResult
from onnx_reduce.search.types import CandidateFeedback
from onnx_reduce.types import EditApplicationError, ModelAdapter, ModelCost, ValidationResult


class CandidateExecutor:
    """One run's execution services, independent of the selected search policy."""

    def __init__(
        self,
        *,
        config: ExecutionConfig,
        validation_mode: str,
        evaluator: Evaluator,
        run_dir: Path,
        caches: PreparationCaches,
        budget: SearchBudget,
    ):
        self.config = config
        self.validation_mode = validation_mode
        self.evaluator = evaluator
        self.run_dir = run_dir
        self.caches = caches
        self.budget = budget
        self.initial_calls = evaluator.calls

    def execute(
        self,
        base_model: ModelAdapter,
        candidate: Candidate,
    ) -> AttemptResult:
        evaluator, run_dir = self.evaluator, self.run_dir
        clock = time.perf_counter if current_performance() is not None else None
        attempt_started = clock() if clock else None
        evaluation_seconds = 0.0
        materialization_seconds = 0.0
        base_cost = base_model.cost()
        validation = None
        result = None
        evidence = {}
        record_status = None

        def finish(status, cost=None, accepted_model=None, **details):
            preparation_seconds = (
                max(0.0, clock() - attempt_started - evaluation_seconds) if clock else None
            )
            return AttemptResult(
                CandidateFeedback(
                    status,
                    base_cost,
                    cost,
                    preparation_seconds=preparation_seconds,
                    evaluation_seconds=evaluation_seconds if clock else None,
                    **details,
                ),
                accepted_model,
                record_status=record_status or status,
                details=evidence,
                validation=validation,
                evaluation=result,
            )

        if current_performance() is not None:
            sample("candidate_edit_count", len(candidate.edits))
            sample(
                "candidate_selected_items",
                sum(
                    len(edit.indices)
                    if hasattr(edit, "indices")
                    else len(edit.names)
                    if hasattr(edit, "names")
                    else 1
                    for edit in candidate.edits
                ),
            )
        caches = self.caches
        cached = caches.edits.get(edit_key(candidate, caches.validation_mode))
        if cached is not None:
            entry = json.loads(cached)
            status, details = entry["status"], entry["details"]
            if "cost" in details:
                sample("candidate_nodes", details["cost"]["node_count"])
                sample("candidate_bytes", details["cost"]["byte_size"])
            if "validation" in details:
                validation = ValidationResult(**details["validation"])
            if status == "not_interesting":
                result = evaluator.reuse(
                    Path(entry["model"]),
                    details["digest"],
                    Evaluation(
                        FailureCheckResult(status),
                        target_identity_mismatch=details["target_identity_mismatch"],
                    ),
                    metadata=candidate.to_dict(),
                )
                details["cache_hit"] = True
            evidence = {**details, "edit_cache_hit": True}
            return finish(
                status,
                ModelCost(**details["cost"]) if "cost" in details else None,
                cache_source="edit",
                target_identity_mismatch=details.get("target_identity_mismatch", False),
            )
        try:
            candidate_model = base_model.apply_candidate(candidate)
        except EditApplicationError as exc:
            self._cache_rejection(candidate, "edit_failed", reason=str(exc))
            evidence = {"reason": str(exc)}
            return finish("edit_failed")
        candidate_cost = candidate_model.cost()
        sample("candidate_nodes", candidate_cost.node_count)
        sample("candidate_bytes", candidate_cost.byte_size)
        if candidate_cost.key(self.config.objective) >= base_cost.key(self.config.objective):
            details = {"cost": asdict(candidate_cost)}
            self._cache_rejection(candidate, "not_cheaper", **details)
            evidence = details
            return finish("not_cheaper", candidate_cost)
        validation = candidate_model.validate(self.validation_mode)
        if not validation.valid:
            details = {"validation": asdict(validation), "cost": asdict(candidate_cost)}
            self._cache_rejection(candidate, "invalid", **details)
            evidence = details
            return finish("invalid", candidate_cost)
        digest = candidate_model.canonical_hash()
        candidate_path = run_dir / "candidates" / f"{candidate.sequence_number:08d}" / "model.onnx"
        if not self.budget.allows_time():
            evidence = {"reason": "time"}
            return finish("budget_exhausted")

        def prepare():
            nonlocal materialization_seconds
            materialization_started = clock() if clock else None
            candidate_model.materialize(candidate_path)
            if clock:
                materialization_seconds += clock() - materialization_started
            if not self.budget.allows_time():
                raise EvaluationBudgetExceeded("search time budget exhausted")

        # Cached decisions consume no calls, even when the quota is exactly
        # spent. The evaluator checks its cache before enforcing this limit.
        try:
            evaluation_started = clock() if clock else None
            result = evaluator.evaluate(
                candidate_path,
                digest,
                metadata=candidate.to_dict(),
                allow_execution=self.budget.allows_call(evaluator.calls - self.initial_calls),
                prepare=prepare,
            )
            if clock:
                evaluation_seconds = max(
                    0.0, clock() - evaluation_started - materialization_seconds
                )
        except EvaluationBudgetExceeded:
            evidence = {"reason": "calls" if self.budget.allows_time() else "time"}
            return finish("budget_exhausted")
        details = {
            "digest": digest,
            "cost": asdict(candidate_cost),
            "cache_hit": result.cache_hit,
            "target_identity_mismatch": result.target_identity_mismatch,
        }
        self._cache_rejection(
            candidate,
            result.outcome.value,
            model_path=str(result.model_path),
            **details,
        )
        evidence = details
        record_status = result.outcome.value
        if result.outcome is not FailureCheckResult.INTERESTING:
            return finish(
                result.outcome.value,
                candidate_cost,
                cache_source="content" if result.cache_hit else None,
                failure_check_calls=int(not result.cache_hit),
                target_identity_mismatch=result.target_identity_mismatch,
            )
        # All passes, including cleanup, are gated by failure checks.
        return finish(
            "accepted",
            candidate_cost,
            candidate_model,
            cache_source="content" if result.cache_hit else None,
            failure_check_calls=int(not result.cache_hit),
        )

    def _cache_rejection(self, candidate, status, *, model_path=None, **details):
        # Only deterministic rejections belong in the preparation cache.
        # Evidence writers must not change execution or cache state.
        if status in {"edit_failed", "not_cheaper", "invalid", "not_interesting"}:
            self.caches.edits.put(
                edit_key(candidate, self.caches.validation_mode),
                json.dumps(
                    {"status": status, "details": details, "model": model_path}, sort_keys=True
                ).encode(),
            )
