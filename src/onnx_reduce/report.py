"""Reduction statistics and the public JSON report schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from onnx_reduce.types import ModelCost


@dataclass
class PassStats:
    generated: int = 0
    invalid: int = 0
    oracle_calls: int = 0
    accepted: int = 0
    not_cheaper: int = 0


@dataclass
class ReductionReport:
    stop_reason: str
    original: ModelCost
    final: ModelCost
    oracle_calls: int = 0
    interesting: int = 0
    not_interesting: int = 0
    inconclusive: int = 0
    candidates_generated: int = 0
    candidates_invalid: int = 0
    candidates_accepted: int = 0
    candidates_not_cheaper: int = 0
    duration_seconds: float = 0.0
    final_hash: str = ""
    cache_hits: int = 0
    signature_mismatches: int = 0
    work_directory: str = ""
    effective_checker_modes: list[str] = field(default_factory=list)
    custom_domains: list[str] = field(default_factory=list)
    minimality: dict[str, Any] = field(default_factory=dict)
    checker_mode: str = "strict"
    pass_stats: dict[str, PassStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def cost_dict(cost: ModelCost) -> dict[str, int]:
            return {
                "node_count": cost.node_count,
                "byte_size": cost.byte_size,
                "input_count": cost.input_count,
                "output_count": cost.output_count,
                "initializer_count": cost.initializer_count,
            }

        valid_rate = None
        checked = self.candidates_generated - self.candidates_not_cheaper
        if checked:
            valid_rate = round((checked - self.candidates_invalid) / checked, 6)
        node_rate = None
        byte_rate = None
        if self.original.node_count:
            node_rate = round(1 - self.final.node_count / self.original.node_count, 6)
        if self.original.byte_size:
            byte_rate = round(1 - self.final.byte_size / self.original.byte_size, 6)
        return {
            "version": 2,
            "work_directory": self.work_directory,
            "minimality": self.minimality,
            "effective_checker_modes": self.effective_checker_modes,
            "custom_domains": self.custom_domains,
            "stop_reason": self.stop_reason,
            "original": cost_dict(self.original),
            "final": cost_dict(self.final),
            "node_reduction_rate": node_rate,
            "byte_reduction_rate": byte_rate,
            "oracle": {
                "calls": self.oracle_calls,
                "cache_hits": self.cache_hits,
                "signature_mismatches": self.signature_mismatches,
                "interesting": self.interesting,
                "not_interesting": self.not_interesting,
                "inconclusive": self.inconclusive,
            },
            "candidates": {
                "generated": self.candidates_generated,
                "invalid": self.candidates_invalid,
                "accepted": self.candidates_accepted,
                "not_cheaper": self.candidates_not_cheaper,
                "valid_rate": valid_rate,
            },
            "passes": {name: stats.__dict__.copy() for name, stats in self.pass_stats.items()},
            "checker_mode": self.checker_mode,
            "final_hash": self.final_hash,
            "duration_seconds": round(self.duration_seconds, 6),
        }
