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
from onnx_reduce.report import PassStats, ReductionReport
from onnx_reduce.storage import write_json
from onnx_reduce.types import EditApplicationError, ModelAdapter, ValidationResult

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
    validation_mode: str = "strict"
    max_candidate_check_calls: int | None = None
    time_budget_seconds: float | None = None
    cache_enabled: bool = True
    target_mode: str = "baseline"
    signature_regex: str | None = None

    def __post_init__(self) -> None:
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
        if passes is None:
            from onnx_reduce.passes import default_passes

            passes = default_passes()
        self.passes = tuple(passes)
        if len({item.name for item in self.passes}) != len(self.passes):
            raise ValueError("pass names must be unique")
        self.config = config or EngineConfig()
        self.progress = progress or (lambda message: None)

    def run(
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
        report = ReductionReport(
            stop_reason="searching",
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
            initial_artifact.model_path, initial_artifact.digest, initial=True, mandatory=True
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
        current_model = model
        self._checkpoint(current_model, report, run_dir, started)
        self.progress(f"initial model is interesting ({report.original.node_count} nodes)")
        candidate_sequence_number = 0
        sweep_inconclusive_candidate_count = 0
        while True:
            accepted_model = None
            budget_exhausted = False
            # Counts only the final incumbent's completed sweep. A prior sweep's
            # inconclusive neighbors say nothing about this incumbent.
            sweep_inconclusive_candidate_count = 0
            base_digest = current_model.canonical_hash()
            for reduction_pass in self.passes:
                for candidate in reduction_pass.generate(current_model):
                    if self.config.max_candidate_check_calls == 0 or not self._time_budget_allows(
                        started
                    ):
                        budget_exhausted = True
                        break
                    if candidate.base_digest not in (None, base_digest):
                        raise ReductionError("pass produced a candidate for a different base")
                    candidate = replace(
                        candidate,
                        base_digest=base_digest,
                        pass_name=reduction_pass.name,
                        sequence_number=candidate_sequence_number,
                    )
                    candidate_sequence_number += 1
                    inconclusive_count_before = report.inconclusive
                    try:
                        accepted_model = self._try_candidate(
                            current_model,
                            candidate,
                            reduction_pass,
                            evaluator,
                            run_dir,
                            report,
                            started,
                        )
                    except EvaluationBudgetExceeded:
                        budget_exhausted = True
                        break
                    sweep_inconclusive_candidate_count += (
                        report.inconclusive - inconclusive_count_before
                    )
                    if accepted_model is not None:
                        break
                if accepted_model is not None or budget_exhausted:
                    break
            if accepted_model is not None:
                current_model = accepted_model
                self._checkpoint(current_model, report, run_dir, started)
                continue
            report.stop_reason = "budget_exhausted" if budget_exhausted else "fixed_point"
            break

        # Recheck precisely the artifact delivered to the caller. Mandatory
        # evaluations never reuse the verdict cache.
        final_artifact = current_model.materialize(output_path or run_dir / "final" / "model.onnx")
        result = evaluator.evaluate(
            final_artifact.model_path, final_artifact.digest, mandatory=True
        )
        self._record_evaluation(result, report)
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
                if report.stop_reason == "fixed_point" and not sweep_inconclusive_candidate_count
                else "unproven"
            ),
            "scope": "valid, strictly cheaper candidates generated by configured passes",
            "passes": [item.name for item in self.passes],
            "sweep_inconclusive_candidate_count": sweep_inconclusive_candidate_count,
            "global_minimum": False,
        }
        self._checkpoint(current_model, report, run_dir, started)
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
    ) -> ModelAdapter | None:
        stats = report.pass_stats[reduction_pass.name]
        stats.generated += 1
        report.candidates_generated += 1
        try:
            candidate_model = base_model.apply_candidate(candidate)
        except EditApplicationError as exc:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._record_candidate(run_dir, candidate, "edit_failed", reason=str(exc))
            return None
        base_cost = base_model.cost()
        candidate_cost = candidate_model.cost()
        if candidate_cost >= base_cost:
            report.candidates_not_cheaper += 1
            stats.not_cheaper += 1
            self._record_candidate(run_dir, candidate, "not_cheaper", cost=asdict(candidate_cost))
            return None
        validation = candidate_model.validate(self.config.validation_mode)
        self._record_validation(validation, report)
        if not validation.valid:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._record_candidate(run_dir, candidate, "invalid", validation=asdict(validation))
            return None
        artifact = candidate_model.materialize(
            run_dir / "candidates" / f"{candidate.sequence_number:08d}" / "model.onnx"
        )
        if not self._time_budget_allows(started):
            self._record_candidate(run_dir, candidate, "budget_exhausted", reason="time")
            raise EvaluationBudgetExceeded("search time budget exhausted")
        # Cached decisions consume no calls, even when the quota is exactly
        # spent. The evaluator checks its cache before enforcing this limit.
        try:
            result = evaluator.evaluate(
                artifact.model_path,
                artifact.digest,
                metadata=candidate.to_dict(),
                allow_execution=(
                    self.config.max_candidate_check_calls is None
                    or report.failure_check_calls - 1 < self.config.max_candidate_check_calls
                ),
            )
        except EvaluationBudgetExceeded:
            self._record_candidate(run_dir, candidate, "budget_exhausted", reason="calls")
            raise
        self._record_candidate(
            run_dir,
            candidate,
            result.outcome.value,
            digest=artifact.digest,
            cost=asdict(candidate_cost),
            cache_hit=result.cache_hit,
        )
        self._record_evaluation(result, report)
        if not result.cache_hit:
            stats.failure_check_calls += 1
        if result.outcome is not FailureCheckResult.INTERESTING:
            return None
        # All passes, including cleanup, are gated by failure checks.
        report.candidates_accepted += 1
        stats.accepted += 1
        self.progress(
            f"accepted {reduction_pass.name}: nodes {base_cost.node_count}->{candidate_cost.node_count}, "
            f"bytes {base_cost.byte_size}->{candidate_cost.byte_size}"
        )
        with (run_dir / "accepted.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "candidate": candidate.to_dict(),
                        "digest": artifact.digest,
                        "cost": asdict(candidate_cost),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return candidate_model

    @staticmethod
    def _record_candidate(run_dir: Path, candidate: Candidate, status: str, **details) -> None:
        """Keep rejection evidence even when a candidate never reaches the failure check runner."""
        with (run_dir / "candidates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"candidate": candidate.to_dict(), "status": status, **details},
                    sort_keys=True,
                )
                + "\n"
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
    def _checkpoint(
        model: ModelAdapter,
        report: ReductionReport,
        run_dir: Path,
        started: float,
    ) -> None:
        # Model is self-contained and atomically replaced before the manifest.
        artifact = model.materialize(run_dir / "checkpoint" / f"{model.canonical_hash()}.onnx")
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
