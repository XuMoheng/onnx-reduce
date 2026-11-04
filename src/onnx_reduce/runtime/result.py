"""Completed execution facts consumed by search and run reporting."""

from dataclasses import dataclass, field

from onnx_reduce.failure_check.evaluator import Evaluation
from onnx_reduce.search.types import CandidateFeedback
from onnx_reduce.types import ModelAdapter, ValidationResult


@dataclass(frozen=True)
class AttemptResult:
    feedback: CandidateFeedback
    accepted_model: ModelAdapter | None = None
    record_status: str = ""
    details: dict = field(default_factory=dict)
    validation: ValidationResult | None = None
    evaluation: Evaluation | None = None
