"""Canonical interfaces and consistent serialized field names."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from onnx_reduce.cli import build_parser
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check import FailureCheckConfig, FailureCheckResult
from onnx_reduce.failure_check.contract import make_failure_check_config
from onnx_reduce.model import OnnxModel
from onnx_reduce.types import ValidationResult
from tests.helpers import chain_model


class ConfiguredCheck:
    failure_check_config = FailureCheckConfig(
        (sys.executable,),
        inconclusive_exit_codes=frozenset({125, 124}),
    )

    def run(self, model_path):
        return FailureCheckResult.INTERESTING


def test_configuration_round_trips_through_dataclass_fields():
    config = EngineConfig(validation_mode="structural", max_candidate_check_calls=2)
    assert EngineConfig(**asdict(config)) == config
    assert replace(config, max_candidate_check_calls=3).max_candidate_check_calls == 3
    validation = ValidationResult(True, validation_mode="structural")
    assert asdict(validation) == {
        "valid": True,
        "reason": None,
        "validation_mode": "structural",
        "custom_domains": (),
    }
    check_config = make_failure_check_config(
        'python "check script.py"',
        timeout_seconds=3,
        crash_is_interesting=True,
        timeout_is_interesting=False,
    )
    assert check_config.command == ("python", "check script.py")


def test_run_files_use_consistent_configuration_and_report_fields(tmp_path):
    config = EngineConfig(validation_mode="strict", max_candidate_check_calls=0)
    _, report = ReductionEngine(config=config).run(
        OnnxModel.from_proto(chain_model()),
        failure_check_runner=ConfiguredCheck(),
        workdir=tmp_path,
    )
    run_dir = Path(report.work_directory)
    task = json.loads((run_dir / "task.json").read_text())
    assert set(task) == {"engine", "failure_check_config", "cwd"}
    assert task["engine"] == asdict(config)
    assert task["failure_check_config"] == {
        "command": [sys.executable],
        "timeout_seconds": 60.0,
        "crash_is_interesting": True,
        "timeout_is_interesting": False,
        "inconclusive_exit_codes": [124, 125],
    }
    replay = json.loads((run_dir / "replay.json").read_text())
    assert set(replay) == {
        "model",
        "digest",
        "target_mode",
        "signature_regex",
        "baseline_identity",
        "cwd",
        "failure_check_config",
        "stop_reason",
    }
    assert replay["failure_check_config"] == task["failure_check_config"]
    payload = json.loads((run_dir / "report.json").read_text())
    assert set(payload) == {
        "version",
        "work_directory",
        "minimality",
        "effective_validation_modes",
        "custom_domains",
        "stop_reason",
        "original",
        "final",
        "node_reduction_rate",
        "byte_reduction_rate",
        "failure_checks",
        "candidates",
        "passes",
        "requested_validation_mode",
        "final_hash",
        "duration_seconds",
    }
    assert payload["version"] == 3
    assert payload["requested_validation_mode"] == "strict"
    assert payload["effective_validation_modes"] == ["strict"]
    assert payload["failure_checks"] == {
        "calls": 2,
        "cache_hits": 0,
        "target_identity_mismatches": 0,
        "interesting": 2,
        "not_interesting": 0,
        "inconclusive": 0,
    }
    assert set(payload["minimality"]) == {
        "status",
        "scope",
        "passes",
        "sweep_inconclusive_candidate_count",
        "global_minimum",
    }
    for stats in payload["passes"].values():
        assert set(stats) == {
            "generated",
            "invalid",
            "failure_check_calls",
            "accepted",
            "not_cheaper",
        }
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert checkpoint["report"] == payload
    records = [
        json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    for record in records:
        assert set(record) == {
            "model",
            "digest",
            "outcome",
            "cache_hit",
            "target_identity_mismatch",
            "candidate",
            "observation",
        }


def test_cli_uses_explicit_validation_and_candidate_call_options():
    args = build_parser().parse_args(
        [
            "input.onnx",
            "--test",
            "check",
            "-o",
            "output.onnx",
            "--validation-mode",
            "structural",
            "--max-candidate-check-calls",
            "4",
        ]
    )
    assert args.validation_mode == "structural"
    assert args.max_candidate_check_calls == 4
