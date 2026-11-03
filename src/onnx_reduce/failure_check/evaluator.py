"""Per-reduction target matching, exact-content cache, and evidence records."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.runner import FailureCheckObservation, FailureCheckRunnable
from onnx_reduce.performance import measure
from onnx_reduce.preparation import MetadataCache, metadata_key
from onnx_reduce.storage import write_text


@dataclass(frozen=True)
class Evaluation:
    outcome: FailureCheckResult
    cache_hit: bool = False
    target_identity_mismatch: bool = False
    model_path: Path | None = None


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
        self.cache = MetadataCache("content", enabled=cache_enabled)
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
        prepare: Callable[[], None] | None = None,
    ) -> Evaluation:
        key = metadata_key("content", digest)
        if not mandatory:
            cached = self.cache.get(key)
            if cached is not None:
                entry = json.loads(cached)
                return self.reuse(
                    Path(entry["model"]),
                    digest,
                    Evaluation(
                        FailureCheckResult(entry["outcome"]),
                        target_identity_mismatch=entry["target_identity_mismatch"],
                    ),
                    metadata=metadata,
                    source="content",
                )
        if not mandatory and not allow_execution:
            raise EvaluationBudgetExceeded("candidate failure check call budget exhausted")
        # Preparation is lazy: hits and exhausted call quotas need no model file.
        if prepare is not None:
            prepare()
        self.calls += 1
        observe = getattr(self.failure_check_runner, "observe", None)
        with measure("external_check"):
            observation = (
                observe(path)
                if observe is not None
                else FailureCheckObservation(self.failure_check_runner.run(path))
            )
        result = replace(self.matcher.match(observation, initial=initial), model_path=path)
        logs = self.workdir / "checks" / f"{self.calls:08d}"
        logs.mkdir(parents=True)
        write_text(logs / "stdout.log", observation.stdout)
        write_text(logs / "stderr.log", observation.stderr)
        self._record(path, digest, result, observation, metadata)
        if self.cache_enabled and result.outcome is not FailureCheckResult.INCONCLUSIVE:
            self.cache.put(
                key,
                json.dumps(
                    {
                        "model": str(path),
                        "outcome": result.outcome.value,
                        "target_identity_mismatch": result.target_identity_mismatch,
                    }
                ).encode(),
            )
        return result

    def reuse(self, path, digest, result, *, metadata=None, source="edit") -> Evaluation:
        """Record a cached verdict against its existing artifact and this attempt's edits."""

        result = replace(result, cache_hit=True, model_path=path)
        self.cache_hits += 1
        self._record(path, digest, result, None, metadata, cache_source=source)
        return result

    def _record(self, path, digest, result, observation, metadata, *, cache_source=None):
        record = {
            "model": str(path),
            "digest": digest,
            "outcome": result.outcome.value,
            "cache_hit": result.cache_hit,
            "target_identity_mismatch": result.target_identity_mismatch,
            "candidate": metadata,
        }
        if cache_source is not None:
            record["cache_source"] = cache_source
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
        write_text(
            self.workdir / "evaluations.jsonl",
            json.dumps(record, sort_keys=True) + "\n",
            append=True,
        )
