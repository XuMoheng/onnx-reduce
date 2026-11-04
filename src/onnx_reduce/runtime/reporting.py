"""Run evidence and report accounting; no cache writes or search decisions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from onnx_reduce.edits import Candidate
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.evaluator import Evaluation
from onnx_reduce.performance import count, current_performance
from onnx_reduce.report import ReductionReport
from onnx_reduce.search.types import CandidateFeedback, ScheduledAttempt
from onnx_reduce.storage import write_text
from onnx_reduce.types import ModelCost, ValidationResult


def record_progress(model, report, run_dir, stage, reason, *, force=False):
    profile = current_performance()
    if profile is not None:
        point = profile.record_progress(
            model.cost(),
            stage=stage,
            reason=reason,
            checks=report.failure_check_calls,
            force=force,
        )
        if point is not None:
            write_text(
                run_dir / "progress.jsonl",
                json.dumps(point, sort_keys=True) + "\n",
                append=True,
            )


def record_schedule(
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


def record_candidate(run_dir: Path, candidate: Candidate, status: str, **details) -> None:
    """Keep rejection evidence even when a candidate never reaches the failure check runner."""
    count(f"candidate_status.{status}")
    write_text(
        run_dir / "candidates.jsonl",
        json.dumps(
            {"candidate": candidate.to_dict(), "status": status, **details},
            sort_keys=True,
        )
        + "\n",
        append=True,
    )


def record_evaluation(result: Evaluation, report: ReductionReport) -> None:
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


def record_validation(validation: ValidationResult, report: ReductionReport) -> None:
    if (
        validation.validation_mode
        and validation.validation_mode not in report.effective_validation_modes
    ):
        report.effective_validation_modes.append(validation.validation_mode)
    report.custom_domains = sorted(set(report.custom_domains) | set(validation.custom_domains))


def record_acceptance(run_dir: Path, candidate: Candidate, digest: str, cost: ModelCost) -> None:
    write_text(
        run_dir / "accepted.jsonl",
        json.dumps(
            {"candidate": candidate.to_dict(), "digest": digest, "cost": asdict(cost)},
            sort_keys=True,
        )
        + "\n",
        append=True,
    )


def record_attempt(run_dir, candidate, pass_name, outcome, report, progress):
    """Consume execution facts; reporting never decides budgets, edits or caches."""
    feedback = outcome.feedback
    stats = report.pass_stats[pass_name]
    stats.generated += 1
    report.candidates_generated += 1
    if outcome.validation is not None:
        record_validation(outcome.validation, report)
    if outcome.evaluation is not None:
        record_evaluation(outcome.evaluation, report)
    stats.failure_check_calls += feedback.failure_check_calls
    if feedback.status in {"edit_failed", "invalid"}:
        report.candidates_invalid += 1
        stats.invalid += 1
    elif feedback.status == "not_cheaper":
        report.candidates_not_cheaper += 1
        stats.not_cheaper += 1
    elif feedback.status == "accepted":
        report.candidates_accepted += 1
        stats.accepted += 1
    record_candidate(run_dir, candidate, outcome.record_status, **outcome.details)
    if outcome.accepted_model is not None:
        before, after = feedback.base_cost, feedback.candidate_cost
        progress(
            f"accepted {pass_name}: nodes {before.node_count}->{after.node_count}, "
            f"bytes {before.byte_size}->{after.byte_size}"
        )
        record_acceptance(run_dir, candidate, outcome.details["digest"], after)
