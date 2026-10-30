"""Per-reduction target matching, exact-content cache, and evidence records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.runner import FailureCheckObservation, FailureCheckRunnable


@dataclass(frozen=True)
class Evaluation:
    outcome: FailureCheckResult
    cache_hit: bool = False
    target_identity_mismatch: bool = False


class EvaluationBudgetExceeded(RuntimeError):
    """The next evaluation needs a process call but has no search quota."""


class TargetMatcher:
    """Match baseline termination and optionally a regex-derived signature.

    Capture groups form the signature when present, otherwise the full match.
    Regex matching is deliberately not implicit bug-identity inference.
    """

    def __init__(self, mode: str = "baseline", signature_regex: str | None = None):
        if mode not in {"baseline", "script"}:
            raise ValueError(f"unknown target mode: {mode}")
        self.mode = mode
        self.regex = re.compile(signature_regex) if signature_regex else None
        self.baseline_identity: tuple | None = None

    def match(self, observation: FailureCheckObservation, *, initial: bool = False) -> Evaluation:
        if observation.outcome is not FailureCheckResult.INTERESTING:
            return Evaluation(observation.outcome)
        signature = None
        if self.regex:
            match = self.regex.search(observation.stdout + "\n" + observation.stderr)
            if match is None:
                if observation.output_truncated:
                    return Evaluation(FailureCheckResult.INCONCLUSIVE)
                return Evaluation(FailureCheckResult.NOT_INTERESTING, target_identity_mismatch=True)
            signature = match.groups() if match.groups() else match.group(0)
        identity = (observation.termination if self.mode == "baseline" else None, signature)
        if initial:
            self.baseline_identity = identity
        if self.baseline_identity != identity:
            return Evaluation(FailureCheckResult.NOT_INTERESTING, target_identity_mismatch=True)
        return Evaluation(FailureCheckResult.INTERESTING)


class Evaluator:
    """Cache is scoped to one run and one failure check and target configuration.

    No cross-run cache is reused. Initial and final checks always bypass cache.
    """

    def __init__(
        self,
        failure_check_runner: FailureCheckRunnable,
        workdir: Path,
        *,
        cache_enabled: bool = True,
        target_mode: str = "baseline",
        signature_regex: str | None = None,
    ):
        self.failure_check_runner = failure_check_runner
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.matcher = TargetMatcher(target_mode, signature_regex)
        self.cache: dict[str, Evaluation] = {}
        self.calls = 0
        self.cache_hits = 0

    def evaluate(
        self,
        path: Path,
        digest: str,
        *,
        initial: bool = False,
        mandatory: bool = False,
        metadata: dict | None = None,
        allow_execution: bool = True,
    ) -> Evaluation:
        if not mandatory and self.cache_enabled and digest in self.cache:
            cached = self.cache[digest]
            result = Evaluation(cached.outcome, True, cached.target_identity_mismatch)
            self.cache_hits += 1
            self._record(path, digest, result, None, metadata)
            return result
        if not mandatory and not allow_execution:
            raise EvaluationBudgetExceeded("candidate failure check call budget exhausted")
        self.calls += 1
        observe = getattr(self.failure_check_runner, "observe", None)
        observation = (
            observe(path)
            if observe is not None
            else FailureCheckObservation(self.failure_check_runner.run(path))
        )
        result = self.matcher.match(observation, initial=initial)
        logs = self.workdir / "checks" / f"{self.calls:08d}"
        logs.mkdir(parents=True)
        logs.joinpath("stdout.log").write_text(observation.stdout, encoding="utf-8")
        logs.joinpath("stderr.log").write_text(observation.stderr, encoding="utf-8")
        self._record(path, digest, result, observation, metadata)
        if self.cache_enabled and result.outcome is not FailureCheckResult.INCONCLUSIVE:
            self.cache[digest] = result
        return result

    def _record(self, path, digest, result, observation, metadata):
        record = {
            "model": str(path),
            "digest": digest,
            "outcome": result.outcome.value,
            "cache_hit": result.cache_hit,
            "target_identity_mismatch": result.target_identity_mismatch,
            "candidate": metadata,
        }
        if observation is not None:
            record["observation"] = {
                "outcome": observation.outcome.value,
                "returncode": observation.returncode,
                "termination": observation.termination,
                "timed_out": observation.timed_out,
                "duration_seconds": observation.duration_seconds,
                "output_truncated": observation.output_truncated,
                "logs": str(self.workdir / "checks" / f"{self.calls:08d}"),
            }
        with (self.workdir / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
