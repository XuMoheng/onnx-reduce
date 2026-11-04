"""Deterministic fixed-point reduction engine."""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from onnx_reduce.edits import Candidate
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.evaluator import (
    Evaluation,
    EvaluationBudgetExceeded,
    Evaluator,
)
from onnx_reduce.failure_check.runner import FailureCheckRunnable
from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.performance import (
    collect_performance,
    count,
    current_performance,
    measured,
    sample,
)
from onnx_reduce.preparation import current_preparation, edit_key, preparation_caches
from onnx_reduce.report import PassStats, ReductionReport
from onnx_reduce.resume import restore_target
from onnx_reduce.scheduler import (
    AttemptResult,
    CandidateFeedback,
    LegacyScheduler,
    ScheduledAttempt,
    StatefulScheduler,
)
from onnx_reduce.staged import StagedScheduler
from onnx_reduce.storage import write_json, write_text
from onnx_reduce.types import (
    ArtifactValidationModel,
    EditApplicationError,
    ModelAdapter,
    ModelCost,
    ValidationResult,
)

ProgressCallback = Callable[[str], None]


class ReductionError(RuntimeError):
    """The reduction could not start or encountered a tool-level failure."""


class InitialNotInteresting(ReductionError):
    def __init__(self, result: FailureCheckResult):
        self.result = result
        if result is FailureCheckResult.INCONCLUSIVE:
            super().__init__(
                "the original model check is inconclusive; inspect evaluations.jsonl "
                "and checks/ for timeout, inconclusive exit code or truncated signature"
            )
            return
        super().__init__(
            "the original model is not interesting; the test direction may be reversed "
            "(exit code 0 must mean the failure is still present); "
            "also check the signature regex and retained check logs"
        )


@dataclass(frozen=True)
class EngineConfig:
    objective: str = "nodes"
    combination_depth: int = 2
    combination_limit: int = 256
    validation_mode: str = "strict"
    max_candidate_check_calls: int | None = None
    time_budget_seconds: float | None = None
    cache_enabled: bool = True
    collect_performance: bool = False
    target_mode: str = "baseline"
    signature_regex: str | None = None
    scheduler: str = "legacy"
    goal: str | None = None
    finalize_from: str | None = None
    model_edit_mode: str = "ir"
    search_policy: str = "restart"
    history_order: str = "off"
    direction_order: str = "fixed"
    checkpoint_reuse: bool = True
    validation_cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.objective not in {"nodes", "bytes", "tensors"}:
            raise ValueError("unknown objective")
        if not 1 <= self.combination_depth <= 3 or self.combination_limit < 0:
            raise ValueError("combination depth must be 1..3 and limit nonnegative")
        for name, options in (
            ("model_edit_mode", {"ir", "auto"}),
            ("search_policy", {"restart", "cursor"}),
            ("history_order", {"off", "defer"}),
            ("direction_order", {"fixed", "adaptive"}),
        ):
            if getattr(self, name) not in options:
                raise ValueError(f"unknown {name}: {getattr(self, name)}")
        if self.search_policy != "restart" and self.scheduler != "staged":
            raise ValueError("search_policy=cursor requires scheduler=staged")
        if (self.history_order != "off" or self.direction_order != "fixed") and (
            self.scheduler != "staged" or self.search_policy != "cursor"
        ):
            raise ValueError("history/direction ordering requires staged search_policy=cursor")
        if self.scheduler not in {"legacy", "stateful", "staged"}:
            raise ValueError(f"unknown scheduler: {self.scheduler}")
        if self.goal not in {None, "fast", "complete"}:
            raise ValueError(f"unknown goal: {self.goal}")
        if (self.goal is not None or self.finalize_from is not None) and self.scheduler != "staged":
            raise ValueError("goal and finalize_from require scheduler=staged")
        if self.finalize_from is not None and self.goal == "fast":
            raise ValueError("finalize_from requires goal=complete")
        if self.validation_mode not in {"strict", "structural", "permissive"}:
            raise ValueError(f"unknown validation mode: {self.validation_mode}")
        if self.target_mode not in {"baseline", "script"}:
            raise ValueError(f"unknown target mode: {self.target_mode}")
        if self.max_candidate_check_calls is not None and (
            type(self.max_candidate_check_calls) is not int or self.max_candidate_check_calls < 0
        ):
            raise ValueError("max_candidate_check_calls must be a non-negative integer")
        if self.time_budget_seconds is not None and (
            not math.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0
        ):
            raise ValueError("time_budget_seconds must be finite and greater than zero")
        if self.signature_regex is not None:
            re.compile(self.signature_regex)


class ReductionEngine:
    """Evaluate candidates independently and commit only verified reductions."""

    def __init__(
        self,
        passes: Sequence[ReductionPass] | None = None,
        *,
        config: EngineConfig | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.config = config or EngineConfig()
        if passes is None:
            from onnx_reduce.passes import default_passes

            passes = default_passes(
                combination_depth=self.config.combination_depth,
                combination_limit=self.config.combination_limit,
            )
        self.passes = tuple(passes)
        if len({item.name for item in self.passes}) != len(self.passes):
            raise ValueError("pass names must be unique")
        self.progress = progress or (lambda message: None)

    def run(
        self,
        model: ModelAdapter,
        failure_check_runner: FailureCheckRunnable,
        workdir: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> tuple[ModelAdapter, ReductionReport]:
        with (
            collect_performance(self.config.collect_performance) as performance,
            preparation_caches(
                self.config.cache_enabled,
                self.config.validation_mode,
                self.config.model_edit_mode,
                self.config.validation_cache_enabled,
            ),
        ):
            reduced, report = self._run(
                model, failure_check_runner, workdir, output_path=output_path
            )
        if performance is not None:
            # Freeze metrics before serializing their final snapshot.
            write_json(Path(report.work_directory) / "report.json", report.to_dict())
        return reduced, report

    def _run(
        self,
        model: ModelAdapter,
        failure_check_runner: FailureCheckRunnable,
        workdir: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> tuple[ModelAdapter, ReductionReport]:
        started = time.monotonic()
        work_root = Path(workdir).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
        evaluator = Evaluator(
            failure_check_runner,
            run_dir,
            cache_enabled=self.config.cache_enabled,
            target_mode=self.config.target_mode,
            signature_regex=self.config.signature_regex,
        )
        failure_check_config = getattr(failure_check_runner, "failure_check_config", None)
        failure_check_config_data = (
            {
                **asdict(failure_check_config),
                "inconclusive_exit_codes": sorted(failure_check_config.inconclusive_exit_codes),
            }
            if failure_check_config is not None
            else None
        )
        write_json(
            run_dir / "task.json",
            {
                "engine": asdict(self.config),
                "failure_check_config": failure_check_config_data,
                "cwd": str(Path.cwd()),
            },
        )
        validation = model.validate(self.config.validation_mode)
        if not validation.valid:
            raise ReductionError(f"input model is invalid: {validation.reason}")
        if self.config.scheduler == "staged":
            scheduler = StagedScheduler(
                self.passes,
                model,
                goal=self.config.goal or "complete",
                finalize_only=self.config.finalize_from is not None,
                search_policy=self.config.search_policy,
                history_order=self.config.history_order,
                direction_order=self.config.direction_order,
            )
        else:
            scheduler_type = (
                LegacyScheduler if self.config.scheduler == "legacy" else StatefulScheduler
            )
            scheduler = scheduler_type(self.passes, model)
        if self.config.finalize_from is not None:
            scheduler.target_origin = restore_target(
                self.config.finalize_from,
                model_digest=model.canonical_hash(),
                config=self.config,
                failure_check_config=failure_check_config_data,
                matcher=evaluator.matcher,
            )
        report = ReductionReport(
            stop_reason="searching",
            performance=current_performance(),
            original=model.cost(),
            final=model.cost(),
            requested_validation_mode=self.config.validation_mode,
            work_directory=str(run_dir),
            pass_stats={item.name: PassStats() for item in self.passes},
        )
        self._record_validation(validation, report)
        self.progress("failure check contract: exit code 0 means the target failure is present")
        initial_artifact = model.materialize(run_dir / "initial" / "model.onnx")
        result = evaluator.evaluate(
            initial_artifact.model_path,
            initial_artifact.digest,
            initial=self.config.finalize_from is None,
            mandatory=True,
        )
        self._record_evaluation(result, report)
        if result.outcome is not FailureCheckResult.INTERESTING:
            raise InitialNotInteresting(result.outcome)
        write_json(
            run_dir / "target.json",
            {
                "mode": self.config.target_mode,
                "signature_regex": self.config.signature_regex,
                "baseline_identity": evaluator.matcher.baseline_identity,
            },
        )
        report.scheduler = scheduler.to_dict()
        current_model = model
        checkpoint_artifacts = {} if self.config.checkpoint_reuse else None
        report.optimizations = {
            name: getattr(self.config, name)
            for name in (
                "model_edit_mode",
                "search_policy",
                "history_order",
                "direction_order",
                "checkpoint_reuse",
                "validation_cache_enabled",
            )
        }
        self._checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)
        self.progress(f"initial model is interesting ({report.original.node_count} nodes)")
        candidate_sequence_number = 0
        while True:
            attempt = scheduler.next_attempt(current_model)
            if attempt is None:
                report.stop_reason = getattr(scheduler, "stop_reason", "fixed_point")
                break
            if self.config.max_candidate_check_calls == 0 or not self._time_budget_allows(started):
                report.stop_reason = "budget_exhausted"
                break
            base_digest = current_model.canonical_hash()
            candidate = attempt.candidate
            if attempt.base_digest != base_digest or candidate.base_digest not in (
                None,
                base_digest,
            ):
                raise ReductionError("pass produced a candidate for a different base")
            candidate = replace(
                candidate,
                base_digest=base_digest,
                pass_name=attempt.reduction_pass.name,
                sequence_number=candidate_sequence_number,
            )
            candidate_sequence_number += 1
            try:
                outcome = self._try_candidate(
                    current_model,
                    candidate,
                    attempt.reduction_pass,
                    evaluator,
                    run_dir,
                    report,
                    started,
                )
            except EvaluationBudgetExceeded:
                self._record_schedule(run_dir, attempt, candidate, None)
                report.stop_reason = "budget_exhausted"
                break
            self._record_schedule(run_dir, attempt, candidate, outcome.feedback)
            scheduler.observe(attempt, outcome.feedback, outcome.accepted_model)
            if outcome.accepted_model is not None:
                current_model = outcome.accepted_model
                report.scheduler = scheduler.to_dict()
                self._checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)

        coverage = scheduler.coverage()
        if report.stop_reason == "fixed_point" and (
            not coverage.completed or coverage.base_digest != current_model.canonical_hash()
        ):
            raise ReductionError("scheduler did not cover the current model")
        report.scheduler = scheduler.to_dict()

        # Recheck precisely the artifact delivered to the caller. Mandatory
        # evaluations never reuse the verdict cache.
        final_artifact = current_model.materialize(output_path or run_dir / "final" / "model.onnx")
        artifact_validation = None
        if isinstance(current_model, ArtifactValidationModel):
            artifact_validation = current_model.validate_artifact(
                final_artifact, self.config.validation_mode
            )
            self._record_validation(artifact_validation, report)
            if not artifact_validation.valid:
                raise ReductionError(f"output artifact is invalid: {artifact_validation.reason}")
        result = evaluator.evaluate(
            final_artifact.model_path, final_artifact.digest, mandatory=True
        )
        self._record_evaluation(result, report)
        report.artifact_recheck = {
            "model": str(final_artifact.model_path),
            "digest": final_artifact.digest,
            "read_back_validated": artifact_validation is not None,
            "outcome": result.outcome.value,
            "cache_hit": result.cache_hit,
        }
        if result.outcome is not FailureCheckResult.INTERESTING:
            report.stop_reason = "unstable_check"
            self.progress("final output did not reproduce the target failure")
        else:
            self.progress(
                f"final independent recheck is interesting ({current_model.cost().node_count} nodes)"
            )
        report.minimality = {
            "status": (
                "pass_fixed_point"
                if report.stop_reason == "fixed_point" and not coverage.inconclusive_candidate_count
                else "unproven"
            ),
            "scope": coverage.scope,
            "passes": [item.name for item in self.passes],
            "sweep_inconclusive_candidate_count": coverage.inconclusive_candidate_count,
            "global_minimum": False,
            "objective": self.config.objective,
            "node_count_scope": "recursive attribute graphs; excludes function definitions",
            "combination_depth": self.config.combination_depth,
            "combination_limit": self.config.combination_limit,
        }
        self._checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)
        write_json(run_dir / "report.json", report.to_dict())
        replay = {
            "model": str(final_artifact.model_path),
            "digest": final_artifact.digest,
            "target_mode": self.config.target_mode,
            "signature_regex": self.config.signature_regex,
            "baseline_identity": evaluator.matcher.baseline_identity,
            "cwd": str(Path.cwd()),
            "failure_check_config": failure_check_config_data,
            "stop_reason": report.stop_reason,
        }
        write_json(run_dir / "replay.json", replay)
        return current_model, report

    def _time_budget_allows(self, started: float) -> bool:
        # Soft deadline: synchronous generation, editing and validation cannot
        # be interrupted. Recheck before starting another failure check evaluation.
        return (
            self.config.time_budget_seconds is None
            or time.monotonic() - started < self.config.time_budget_seconds
        )

    def _try_candidate(
        self,
        base_model: ModelAdapter,
        candidate: Candidate,
        reduction_pass: ReductionPass,
        evaluator: Evaluator,
        run_dir: Path,
        report: ReductionReport,
        started: float,
    ) -> AttemptResult:
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
        caches = current_preparation()
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
                    self._record_validation(ValidationResult(**details["validation"]), report)
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
                self._record_evaluation(result, report)
                details["cache_hit"] = True
            self._record_candidate(run_dir, candidate, status, **details, edit_cache_hit=True)
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
            self._record_candidate(run_dir, candidate, "edit_failed", reason=str(exc))
            return finish("edit_failed")
        candidate_cost = candidate_model.cost()
        sample("candidate_nodes", candidate_cost.node_count)
        sample("candidate_bytes", candidate_cost.byte_size)
        if candidate_cost.key(self.config.objective) >= base_cost.key(self.config.objective):
            report.candidates_not_cheaper += 1
            stats.not_cheaper += 1
            self._record_candidate(run_dir, candidate, "not_cheaper", cost=asdict(candidate_cost))
            return finish("not_cheaper", candidate_cost)
        validation = candidate_model.validate(self.config.validation_mode)
        self._record_validation(validation, report)
        if not validation.valid:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._record_candidate(
                run_dir,
                candidate,
                "invalid",
                validation=asdict(validation),
                cost=asdict(candidate_cost),
            )
            return finish("invalid", candidate_cost)
        digest = candidate_model.canonical_hash()
        candidate_path = run_dir / "candidates" / f"{candidate.sequence_number:08d}" / "model.onnx"
        if not self._time_budget_allows(started):
            self._record_candidate(run_dir, candidate, "budget_exhausted", reason="time")
            raise EvaluationBudgetExceeded("search time budget exhausted")

        def prepare():
            nonlocal materialization_seconds
            materialization_started = clock() if clock else None
            candidate_model.materialize(candidate_path)
            if clock:
                materialization_seconds += clock() - materialization_started
            if not self._time_budget_allows(started):
                raise EvaluationBudgetExceeded("search time budget exhausted")

        # Cached decisions consume no calls, even when the quota is exactly
        # spent. The evaluator checks its cache before enforcing this limit.
        try:
            evaluation_started = clock() if clock else None
            result = evaluator.evaluate(
                candidate_path,
                digest,
                metadata=candidate.to_dict(),
                allow_execution=(
                    self.config.max_candidate_check_calls is None
                    or report.failure_check_calls - 1 < self.config.max_candidate_check_calls
                ),
                prepare=prepare,
            )
            if clock:
                evaluation_seconds = max(
                    0.0, clock() - evaluation_started - materialization_seconds
                )
        except EvaluationBudgetExceeded:
            self._record_candidate(
                run_dir,
                candidate,
                "budget_exhausted",
                reason="calls" if self._time_budget_allows(started) else "time",
            )
            raise
        self._record_candidate(
            run_dir,
            candidate,
            result.outcome.value,
            digest=digest,
            cost=asdict(candidate_cost),
            cache_hit=result.cache_hit,
            _model_path=str(result.model_path),
            target_identity_mismatch=result.target_identity_mismatch,
        )
        self._record_evaluation(result, report)
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
        write_text(
            run_dir / "accepted.jsonl",
            json.dumps(
                {
                    "candidate": candidate.to_dict(),
                    "digest": digest,
                    "cost": asdict(candidate_cost),
                },
                sort_keys=True,
            )
            + "\n",
            append=True,
        )
        return finish(
            "accepted",
            candidate_cost,
            candidate_model,
            cache_source="content" if result.cache_hit else None,
            failure_check_calls=int(not result.cache_hit),
        )

    @staticmethod
    def _record_schedule(
        run_dir: Path,
        attempt: ScheduledAttempt,
        candidate: Candidate,
        feedback: CandidateFeedback | None,
    ) -> None:
        # Keep scheduling evidence separate so replayable candidate JSON and
        # instrumentation-equivalence traces retain their existing format.
        write_text(
            run_dir / "scheduler.jsonl",
            json.dumps(
                {
                    "sequence_number": candidate.sequence_number,
                    "base_digest": attempt.base_digest,
                    "model_version": attempt.model_version,
                    "pass_name": attempt.reduction_pass.name,
                    "source": attempt.source,
                    "direction": attempt.direction,
                    "scale": attempt.scale,
                    "region_anchor": attempt.region_ids[0] if attempt.region_ids else None,
                    "region_size": len(attempt.region_ids),
                    **({"stage": attempt.stage} if attempt.stage is not None else {}),
                    **(
                        {"region_source": attempt.region_source}
                        if attempt.region_source is not None
                        else {}
                    ),
                    "status": feedback.status if feedback else "budget_exhausted",
                    "cache_source": feedback.cache_source if feedback else None,
                    "failure_check_calls": feedback.failure_check_calls if feedback else 0,
                    "target_identity_mismatch": (
                        feedback.target_identity_mismatch if feedback else False
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            append=True,
        )

    @staticmethod
    def _record_candidate(
        run_dir: Path, candidate: Candidate, status: str, *, _model_path=None, **details
    ) -> None:
        """Keep rejection evidence even when a candidate never reaches the failure check runner."""
        count(f"candidate_status.{status}")
        caches = current_preparation()
        if (
            caches is not None
            and not details.get("edit_cache_hit")
            and status
            in {
                "edit_failed",
                "not_cheaper",
                "invalid",
                "not_interesting",
            }
        ):
            caches.edits.put(
                edit_key(candidate, caches.validation_mode),
                json.dumps(
                    {"status": status, "details": details, "model": _model_path}, sort_keys=True
                ).encode(),
            )
        write_text(
            run_dir / "candidates.jsonl",
            json.dumps(
                {"candidate": candidate.to_dict(), "status": status, **details},
                sort_keys=True,
            )
            + "\n",
            append=True,
        )

    @staticmethod
    def _record_evaluation(result: Evaluation, report: ReductionReport) -> None:
        if result.cache_hit:
            report.cache_hits += 1
            return
        report.failure_check_calls += 1
        if result.target_identity_mismatch:
            report.target_identity_mismatches += 1
        if result.outcome is FailureCheckResult.INTERESTING:
            report.interesting += 1
        elif result.outcome is FailureCheckResult.NOT_INTERESTING:
            report.not_interesting += 1
        else:
            report.inconclusive += 1

    @staticmethod
    def _record_validation(validation: ValidationResult, report: ReductionReport) -> None:
        if (
            validation.validation_mode
            and validation.validation_mode not in report.effective_validation_modes
        ):
            report.effective_validation_modes.append(validation.validation_mode)
        report.custom_domains = sorted(set(report.custom_domains) | set(validation.custom_domains))

    @staticmethod
    @measured("checkpoint")
    def _checkpoint(
        model: ModelAdapter,
        report: ReductionReport,
        run_dir: Path,
        started: float,
        known_artifacts: dict | None = None,
    ) -> None:
        # Model is self-contained and atomically replaced before the manifest.
        from onnx_reduce.storage import checkpoint_artifact

        artifact = checkpoint_artifact(
            model, run_dir / "checkpoint" / f"{model.canonical_hash()}.onnx", known_artifacts
        )

        report.final = model.cost()
        report.final_hash = artifact.digest
        report.duration_seconds = time.monotonic() - started
        write_json(
            run_dir / "checkpoint.json",
            {
                "model": str(artifact.model_path),
                "digest": artifact.digest,
                "report": report.to_dict(),
            },
        )
