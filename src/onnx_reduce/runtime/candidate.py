"""Execute one candidate through preparation, validation and failure matching."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from onnx_reduce.config import EngineConfig
from onnx_reduce.edits import Candidate
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.evaluator import Evaluation, EvaluationBudgetExceeded, Evaluator
from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.performance import current_performance, sample
from onnx_reduce.preparation import PreparationCaches, edit_key
from onnx_reduce.report import ReductionReport
from onnx_reduce.runtime.budget import SearchBudget
from onnx_reduce.runtime.reporting import (
    record_acceptance,
    record_candidate,
    record_evaluation,
    record_validation,
)
from onnx_reduce.search.types import AttemptResult, CandidateFeedback
from onnx_reduce.types import EditApplicationError, ModelAdapter, ModelCost, ValidationResult


class CandidateExecutor:
    """One run's execution services, independent of the selected search policy."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        evaluator: Evaluator,
        run_dir: Path,
        report: ReductionReport,
        caches: PreparationCaches,
        budget: SearchBudget,
        progress: Callable[[str], None],
    ):
        self.config = config
        self.evaluator = evaluator
        self.run_dir = run_dir
        self.report = report
        self.caches = caches
        self.budget = budget
        self.progress = progress

    def execute(
        self,
        base_model: ModelAdapter,
        candidate: Candidate,
        reduction_pass: ReductionPass,
    ) -> AttemptResult:
        report, evaluator, run_dir = self.report, self.evaluator, self.run_dir
        clock = time.perf_counter if current_performance() is not None else None
        attempt_started = clock() if clock else None
        evaluation_seconds = 0.0
        materialization_seconds = 0.0
        base_cost = base_model.cost()

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
            )

        stats = report.pass_stats[reduction_pass.name]
        stats.generated += 1
        report.candidates_generated += 1
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
            if status in {"edit_failed", "invalid"}:
                report.candidates_invalid += 1
                stats.invalid += 1
                if "validation" in details:
                    record_validation(ValidationResult(**details["validation"]), report)
            elif status == "not_cheaper":
                report.candidates_not_cheaper += 1
                stats.not_cheaper += 1
            else:
                result = evaluator.reuse(
                    Path(entry["model"]),
                    details["digest"],
                    Evaluation(
                        FailureCheckResult(status),
                        target_identity_mismatch=details["target_identity_mismatch"],
                    ),
                    metadata=candidate.to_dict(),
                )
                record_evaluation(result, report)
                details["cache_hit"] = True
            record_candidate(run_dir, candidate, status, **details, edit_cache_hit=True)
            return finish(
                status,
                ModelCost(**details["cost"]) if "cost" in details else None,
                cache_source="edit",
                target_identity_mismatch=details.get("target_identity_mismatch", False),
            )
        try:
            candidate_model = base_model.apply_candidate(candidate)
        except EditApplicationError as exc:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._cache_rejection(candidate, "edit_failed", reason=str(exc))
            record_candidate(run_dir, candidate, "edit_failed", reason=str(exc))
            return finish("edit_failed")
        candidate_cost = candidate_model.cost()
        sample("candidate_nodes", candidate_cost.node_count)
        sample("candidate_bytes", candidate_cost.byte_size)
        if candidate_cost.key(self.config.objective) >= base_cost.key(self.config.objective):
            report.candidates_not_cheaper += 1
            stats.not_cheaper += 1
            details = {"cost": asdict(candidate_cost)}
            self._cache_rejection(candidate, "not_cheaper", **details)
            record_candidate(run_dir, candidate, "not_cheaper", **details)
            return finish("not_cheaper", candidate_cost)
        validation = candidate_model.validate(self.config.validation_mode)
        record_validation(validation, report)
        if not validation.valid:
            report.candidates_invalid += 1
            stats.invalid += 1
            details = {"validation": asdict(validation), "cost": asdict(candidate_cost)}
            self._cache_rejection(candidate, "invalid", **details)
            record_candidate(run_dir, candidate, "invalid", **details)
            return finish("invalid", candidate_cost)
        digest = candidate_model.canonical_hash()
        candidate_path = run_dir / "candidates" / f"{candidate.sequence_number:08d}" / "model.onnx"
        if not self.budget.allows_time():
            record_candidate(run_dir, candidate, "budget_exhausted", reason="time")
            raise EvaluationBudgetExceeded("search time budget exhausted")

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
                allow_execution=self.budget.allows_call(report.failure_check_calls - 1),
                prepare=prepare,
            )
            if clock:
                evaluation_seconds = max(
                    0.0, clock() - evaluation_started - materialization_seconds
                )
        except EvaluationBudgetExceeded:
            record_candidate(
                run_dir,
                candidate,
                "budget_exhausted",
                reason="calls" if self.budget.allows_time() else "time",
            )
            raise
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
        record_candidate(run_dir, candidate, result.outcome.value, **details)
        record_evaluation(result, report)
        if not result.cache_hit:
            stats.failure_check_calls += 1
        if result.outcome is not FailureCheckResult.INTERESTING:
            return finish(
                result.outcome.value,
                candidate_cost,
                cache_source="content" if result.cache_hit else None,
                failure_check_calls=int(not result.cache_hit),
                target_identity_mismatch=result.target_identity_mismatch,
            )
        # All passes, including cleanup, are gated by failure checks.
        report.candidates_accepted += 1
        stats.accepted += 1
        self.progress(
            f"accepted {reduction_pass.name}: nodes {base_cost.node_count}->{candidate_cost.node_count}, "
            f"bytes {base_cost.byte_size}->{candidate_cost.byte_size}"
        )
        record_acceptance(run_dir, candidate, digest, candidate_cost)
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
