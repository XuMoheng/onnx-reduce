"""Interestingness oracle contract and process runner."""

from onnx_reduce.oracle.contract import OracleResult, OracleSpec
from onnx_reduce.oracle.evaluator import Evaluator, TargetMatcher
from onnx_reduce.oracle.runner import OracleError, OracleObservation, OracleRunner

__all__ = [
    "Evaluator",
    "OracleError",
    "OracleObservation",
    "OracleResult",
    "OracleRunner",
    "OracleSpec",
    "TargetMatcher",
]
