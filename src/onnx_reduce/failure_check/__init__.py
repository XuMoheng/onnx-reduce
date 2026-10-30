"""Failure check configuration, execution, and target matching."""

from onnx_reduce.failure_check.contract import FailureCheckConfig, FailureCheckResult
from onnx_reduce.failure_check.evaluator import Evaluator, TargetMatcher
from onnx_reduce.failure_check.runner import (
    FailureCheckError,
    FailureCheckObservation,
    FailureCheckRunnable,
    FailureCheckRunner,
)

__all__ = [
    "Evaluator",
    "FailureCheckConfig",
    "FailureCheckError",
    "FailureCheckObservation",
    "FailureCheckResult",
    "FailureCheckRunnable",
    "FailureCheckRunner",
    "TargetMatcher",
]
