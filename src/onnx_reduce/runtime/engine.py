"""Orchestrate initialization, search, accepted checkpoints and final verification."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from inspect import getattr_static
from pathlib import Path

from onnx_reduce.config import EngineConfig as EngineConfig
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.evaluator import Evaluator
from onnx_reduce.failure_check.runner import FailureCheckRunnable
from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.performance import collect_performance, current_performance
from onnx_reduce.preparation import PreparationCaches, preparation_caches
from onnx_reduce.report import PassStats, ReductionReport
from onnx_reduce.runtime.artifacts import checkpoint
from onnx_reduce.runtime.budget import SearchBudget
from onnx_reduce.runtime.candidate import CandidateExecutor
from onnx_reduce.runtime.reporting import (
    record_attempt,
    record_evaluation,
    record_progress,
    record_schedule,
    record_validation,
)
from onnx_reduce.runtime.resume import restore_target
from onnx_reduce.search.base import Scheduler
from onnx_reduce.search.legacy import LegacyScheduler
from onnx_reduce.search.staged import StagedScheduler
from onnx_reduce.search.stateful import StatefulScheduler
from onnx_reduce.storage import write_json
from onnx_reduce.types import (
    ArtifactValidationModel,
    ConfigurableEditingModel,
    EditOptions,
    ModelAdapter,
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


class ReductionEngine:
    """Coordinate search; candidate execution and evidence live in runtime services."""

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
                combination_depth=self.config.search.combination_depth,
                combination_limit=self.config.search.combination_limit,
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
        # Python 3.11's protocol check follows __getattr__. A forwarded editor
        # can return the inner model and discard the adapter's capabilities.
        # Require an explicit implementation, as protocol checks do on 3.12+.
        if getattr_static(model, "with_edit_options", None) is not None and isinstance(
            model, ConfigurableEditingModel
        ):
            model = model.with_edit_options(EditOptions(self.config.experiments.model_edit_mode))
        with (
            collect_performance(self.config.execution.collect_performance) as performance,
            preparation_caches(
                self.config.execution.cache_enabled,
                self.config.checking.validation_mode,
                self.config.experiments.validation_cache_enabled,
            ) as caches,
        ):
            reduced, report = self._run(
                model, failure_check_runner, workdir, caches=caches, output_path=output_path
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
        caches: PreparationCaches,
        output_path: str | Path | None = None,
    ) -> tuple[ModelAdapter, ReductionReport]:
        started = time.monotonic()
        budget = SearchBudget(
            started,
            self.config.execution.time_budget_seconds,
            self.config.execution.max_candidate_check_calls,
        )
        scheduler: Scheduler
        work_root = Path(workdir).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
        evaluator = Evaluator(
            failure_check_runner,
            run_dir,
            cache_enabled=self.config.execution.cache_enabled,
            target_mode=self.config.checking.target_mode,
            signature_regex=self.config.checking.signature_regex,
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
                "engine": self.config.to_dict(),
                "failure_check_config": failure_check_config_data,
                "cwd": str(Path.cwd()),
            },
        )
        validation = model.validate(self.config.checking.validation_mode)
        if not validation.valid:
            raise ReductionError(f"input model is invalid: {validation.reason}")
        if self.config.search.scheduler == "staged":
            scheduler = StagedScheduler(
                self.passes,
                model,
                goal=self.config.search.goal or "complete",
                finalize_only=self.config.search.finalize_from is not None,
                search_policy=self.config.experiments.search_policy,
                history_order=self.config.experiments.history_order,
                direction_order=self.config.experiments.direction_order,
            )
        else:
            scheduler_type = (
                LegacyScheduler if self.config.search.scheduler == "legacy" else StatefulScheduler
            )
            scheduler = scheduler_type(self.passes, model)
        if self.config.search.finalize_from is not None:
            scheduler.target_origin = restore_target(
                self.config.search.finalize_from,
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
            requested_validation_mode=self.config.checking.validation_mode,
            work_directory=str(run_dir),
            pass_stats={item.name: PassStats() for item in self.passes},
        )
        record_validation(validation, report)
        self.progress("failure check contract: exit code 0 means the target failure is present")
        initial_artifact = model.materialize(run_dir / "initial" / "model.onnx")
        result = evaluator.evaluate(
            initial_artifact.model_path,
            initial_artifact.digest,
            initial=self.config.search.finalize_from is None,
            mandatory=True,
        )
        record_evaluation(result, report)
        if result.outcome is not FailureCheckResult.INTERESTING:
            raise InitialNotInteresting(result.outcome)
        write_json(
            run_dir / "target.json",
            {
                "mode": self.config.checking.target_mode,
                "signature_regex": self.config.checking.signature_regex,
                "baseline_identity": evaluator.matcher.baseline_identity,
            },
        )
        report.scheduler = scheduler.to_dict()
        current_model = model
        checkpoint_artifacts = {} if self.config.experiments.checkpoint_reuse else None
        report.optimizations = {
            name: getattr(self.config.experiments, name)
            for name in (
                "model_edit_mode",
                "search_policy",
                "history_order",
                "direction_order",
                "checkpoint_reuse",
                "validation_cache_enabled",
            )
        }
        checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)
        self.progress(f"initial model is interesting ({report.original.node_count} nodes)")
        record_progress(current_model, report, run_dir, "initial", "initial", force=True)
        executor = CandidateExecutor(
            config=self.config.execution,
            validation_mode=self.config.checking.validation_mode,
            evaluator=evaluator,
            run_dir=run_dir,
            caches=caches,
            budget=budget,
        )
        candidate_sequence_number = 0
        while True:
            attempt = scheduler.next_attempt(current_model)
            if attempt is None:
                report.stop_reason = scheduler.stop_reason
                break
            if self.config.execution.max_candidate_check_calls == 0 or not budget.allows_time():
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
            outcome = executor.execute(current_model, candidate)
            record_attempt(
                run_dir, candidate, attempt.reduction_pass.name, outcome, report, self.progress
            )
            if outcome.feedback.status == "budget_exhausted":
                record_schedule(run_dir, attempt, candidate, None)
                report.stop_reason = "budget_exhausted"
                break
            record_schedule(run_dir, attempt, candidate, outcome.feedback)
            scheduler.observe(attempt, outcome.feedback, outcome.accepted_model)
            record_progress(
                outcome.accepted_model or current_model,
                report,
                run_dir,
                attempt.stage or attempt.source,
                outcome.feedback.status,
                force=outcome.accepted_model is not None,
            )
            if outcome.accepted_model is not None:
                current_model = outcome.accepted_model
                report.scheduler = scheduler.to_dict()
                checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)

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
                final_artifact, self.config.checking.validation_mode
            )
            record_validation(artifact_validation, report)
            if not artifact_validation.valid:
                raise ReductionError(f"output artifact is invalid: {artifact_validation.reason}")
        result = evaluator.evaluate(
            final_artifact.model_path, final_artifact.digest, mandatory=True
        )
        record_evaluation(result, report)
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
        record_progress(current_model, report, run_dir, "final", report.stop_reason, force=True)
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
            "objective": self.config.execution.objective,
            "node_count_scope": "recursive attribute graphs; excludes function definitions",
            "combination_depth": self.config.search.combination_depth,
            "combination_limit": self.config.search.combination_limit,
        }
        checkpoint(current_model, report, run_dir, started, checkpoint_artifacts)
        write_json(run_dir / "report.json", report.to_dict())
        replay = {
            "model": str(final_artifact.model_path),
            "digest": final_artifact.digest,
            "target_mode": self.config.checking.target_mode,
            "signature_regex": self.config.checking.signature_regex,
            "baseline_identity": evaluator.matcher.baseline_identity,
            "cwd": str(Path.cwd()),
            "failure_check_config": failure_check_config_data,
            "stop_reason": report.stop_reason,
        }
        write_json(run_dir / "replay.json", replay)
        return current_model, report
