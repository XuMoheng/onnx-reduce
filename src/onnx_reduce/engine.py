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
from onnx_reduce.oracle.contract import OracleResult
from onnx_reduce.oracle.evaluator import Evaluation, EvaluationBudgetExceeded, Evaluator
from onnx_reduce.oracle.runner import OracleRunnable
from onnx_reduce.passes.base import ReductionPass
from onnx_reduce.report import PassStats, ReductionReport
from onnx_reduce.storage import write_json
from onnx_reduce.types import EditApplicationError, ModelAdapter, ValidationResult

ProgressCallback = Callable[[str], None]


class ReductionError(RuntimeError):
    """The reduction could not start or encountered a tool-level failure."""


class InitialNotInteresting(ReductionError):
    def __init__(self, result: OracleResult):
        self.result = result
        if result is OracleResult.INCONCLUSIVE:
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
    checker_mode: str = "strict"
    max_tests: int | None = None
    time_budget_seconds: float | None = None
    cache_enabled: bool = True
    target_mode: str = "baseline"
    signature_regex: str | None = None

    def __post_init__(self) -> None:
        if self.checker_mode not in {"strict", "structural", "permissive"}:
            raise ValueError(f"unknown checker mode: {self.checker_mode}")
        if self.target_mode not in {"baseline", "script"}:
            raise ValueError(f"unknown target mode: {self.target_mode}")
        if self.max_tests is not None and (type(self.max_tests) is not int or self.max_tests < 0):
            raise ValueError("max_tests must be a non-negative integer")
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
        oracle: OracleRunnable,
        workdir: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> tuple[ModelAdapter, ReductionReport]:
        started = time.monotonic()
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        evaluator = Evaluator(
            oracle,
            work,
            cache_enabled=self.config.cache_enabled,
            target_mode=self.config.target_mode,
            signature_regex=self.config.signature_regex,
        )
        spec = getattr(oracle, "spec", None)
        oracle_config = (
            {**asdict(spec), "inconclusive_exit_codes": sorted(spec.inconclusive_exit_codes)}
            if spec is not None
            else None
        )
        write_json(
            work / "task.json",
            {
                "engine": asdict(self.config),
                "oracle": oracle_config,
                "cwd": str(Path.cwd()),
            },
        )
        validation = model.validate(self.config.checker_mode)
        if not validation.valid:
            raise ReductionError(f"input model is invalid: {validation.reason}")
        report = ReductionReport(
            stop_reason="searching",
            original=model.cost(),
            final=model.cost(),
            checker_mode=self.config.checker_mode,
            work_directory=str(work),
            pass_stats={item.name: PassStats() for item in self.passes},
        )
        self._record_validation(validation, report)
        self.progress("oracle contract: exit code 0 means the target failure is present")
        initial = model.materialize(work / "initial" / "model.onnx")
        result = evaluator.evaluate(
            initial.model_path, initial.digest, initial=True, mandatory=True
        )
        self._record_evaluation(result, report)
        if result.outcome is not OracleResult.INTERESTING:
            raise InitialNotInteresting(result.outcome)
        write_json(
            work / "target.json",
            {
                "mode": self.config.target_mode,
                "signature_regex": self.config.signature_regex,
                "baseline_identity": evaluator.matcher.baseline,
            },
        )
        current = model
        self._checkpoint(current, report, work, started)
        self.progress(f"initial model is interesting ({report.original.node_count} nodes)")
        sequence = 0
        unresolved = 0
        while True:
            accepted = None
            exhausted = False
            # Counts only the final incumbent's completed sweep. A prior sweep's
            # inconclusive neighbors say nothing about this incumbent.
            unresolved = 0
            base_digest = current.canonical_hash()
            for reduction_pass in self.passes:
                for proposal in reduction_pass.generate(current):
                    if self.config.max_tests == 0 or not self._time_budget_allows(started):
                        exhausted = True
                        break
                    if proposal.base_digest not in (None, base_digest):
                        raise ReductionError("pass produced a candidate for a different base")
                    candidate = replace(
                        proposal,
                        base_digest=base_digest,
                        pass_name=reduction_pass.name,
                        sequence_number=sequence,
                    )
                    sequence += 1
                    before = report.inconclusive
                    try:
                        accepted = self._try_candidate(
                            current,
                            candidate,
                            reduction_pass,
                            evaluator,
                            work,
                            report,
                            started,
                        )
                    except EvaluationBudgetExceeded:
                        exhausted = True
                        break
                    unresolved += report.inconclusive - before
                    if accepted is not None:
                        break
                if accepted is not None or exhausted:
                    break
            if accepted is not None:
                current = accepted
                self._checkpoint(current, report, work, started)
                continue
            report.stop_reason = "budget_exhausted" if exhausted else "fixed_point"
            break

        # Recheck precisely the artifact delivered to the caller. Mandatory
        # evaluations never reuse the verdict cache.
        final = current.materialize(output_path or work / "final" / "model.onnx")
        result = evaluator.evaluate(final.model_path, final.digest, mandatory=True)
        self._record_evaluation(result, report)
        if result.outcome is not OracleResult.INTERESTING:
            report.stop_reason = "unstable_check"
            self.progress("final output did not reproduce the target failure")
        else:
            self.progress(
                f"final independent recheck is interesting ({current.cost().node_count} nodes)"
            )
        report.minimality = {
            "status": (
                "pass_fixed_point"
                if report.stop_reason == "fixed_point" and not unresolved
                else "unproven"
            ),
            "scope": "valid, strictly cheaper candidates generated by configured passes",
            "passes": [item.name for item in self.passes],
            "unresolved_candidates": unresolved,
            "global_minimum": False,
        }
        self._checkpoint(current, report, work, started)
        write_json(work / "report.json", report.to_dict())
        replay = {
            "model": str(final.model_path),
            "digest": final.digest,
            "target_mode": self.config.target_mode,
            "signature_regex": self.config.signature_regex,
            "baseline_identity": evaluator.matcher.baseline,
            "cwd": str(Path.cwd()),
            "oracle": oracle_config,
            "stop_reason": report.stop_reason,
        }
        write_json(work / "replay.json", replay)
        return current, report

    def _time_budget_allows(self, started: float) -> bool:
        # Soft deadline: synchronous generation, editing and validation cannot
        # be interrupted. Recheck before starting another oracle evaluation.
        return (
            self.config.time_budget_seconds is None
            or time.monotonic() - started < self.config.time_budget_seconds
        )

    def _try_candidate(
        self,
        base: ModelAdapter,
        candidate: Candidate,
        reduction_pass: ReductionPass,
        evaluator: Evaluator,
        work: Path,
        report: ReductionReport,
        started: float,
    ) -> ModelAdapter | None:
        stats = report.pass_stats[reduction_pass.name]
        stats.generated += 1
        report.candidates_generated += 1
        try:
            candidate_model = base.apply_candidate(candidate)
        except EditApplicationError as exc:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._record_candidate(work, candidate, "edit_failed", reason=str(exc))
            return None
        base_cost = base.cost()
        candidate_cost = candidate_model.cost()
        if candidate_cost >= base_cost:
            report.candidates_not_cheaper += 1
            stats.not_cheaper += 1
            self._record_candidate(work, candidate, "not_cheaper", cost=asdict(candidate_cost))
            return None
        validation = candidate_model.validate(self.config.checker_mode)
        self._record_validation(validation, report)
        if not validation.valid:
            report.candidates_invalid += 1
            stats.invalid += 1
            self._record_candidate(work, candidate, "invalid", validation=asdict(validation))
            return None
        artifact = candidate_model.materialize(
            work / "candidates" / f"{candidate.sequence_number:08d}" / "model.onnx"
        )
        if not self._time_budget_allows(started):
            self._record_candidate(work, candidate, "budget_exhausted", reason="time")
            raise EvaluationBudgetExceeded("search time budget exhausted")
        # Cached decisions consume no calls, even when the quota is exactly
        # spent. The evaluator checks its cache before enforcing this limit.
        try:
            result = evaluator.evaluate(
                artifact.model_path,
                artifact.digest,
                metadata=candidate.to_dict(),
                allow_execution=(
                    self.config.max_tests is None or report.oracle_calls - 1 < self.config.max_tests
                ),
            )
        except EvaluationBudgetExceeded:
            self._record_candidate(work, candidate, "budget_exhausted", reason="calls")
            raise
        self._record_candidate(
            work,
            candidate,
            result.outcome.value,
            digest=artifact.digest,
            cost=asdict(candidate_cost),
            cache_hit=result.cache_hit,
        )
        self._record_evaluation(result, report)
        if not result.cache_hit:
            stats.oracle_calls += 1
        if result.outcome is not OracleResult.INTERESTING:
            return None
        # All passes, including cleanup, are oracle-gated.
        report.candidates_accepted += 1
        stats.accepted += 1
        self.progress(
            f"accepted {reduction_pass.name}: nodes {base_cost.node_count}->{candidate_cost.node_count}, "
            f"bytes {base_cost.byte_size}->{candidate_cost.byte_size}"
        )
        with (work / "accepted.jsonl").open("a", encoding="utf-8") as stream:
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
    def _record_candidate(work: Path, candidate: Candidate, status: str, **details) -> None:
        """Keep rejection evidence even when a candidate never reaches the oracle."""
        with (work / "candidates.jsonl").open("a", encoding="utf-8") as stream:
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
        report.oracle_calls += 1
        if result.signature_mismatch:
            report.signature_mismatches += 1
        if result.outcome is OracleResult.INTERESTING:
            report.interesting += 1
        elif result.outcome is OracleResult.NOT_INTERESTING:
            report.not_interesting += 1
        else:
            report.inconclusive += 1

    @staticmethod
    def _record_validation(validation: ValidationResult, report: ReductionReport) -> None:
        if (
            validation.checker_mode
            and validation.checker_mode not in report.effective_checker_modes
        ):
            report.effective_checker_modes.append(validation.checker_mode)
        report.custom_domains = sorted(set(report.custom_domains) | set(validation.custom_domains))

    @staticmethod
    def _checkpoint(
        model: ModelAdapter,
        report: ReductionReport,
        work: Path,
        started: float,
    ) -> None:
        # Model is self-contained and atomically replaced before the manifest.
        artifact = model.materialize(work / "checkpoint" / f"{model.canonical_hash()}.onnx")
        report.final = model.cost()
        report.final_hash = artifact.digest
        report.duration_seconds = time.monotonic() - started
        write_json(
            work / "checkpoint.json",
            {
                "model": str(artifact.model_path),
                "digest": artifact.digest,
                "report": report.to_dict(),
            },
        )
