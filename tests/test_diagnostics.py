"""User-facing evidence for import, candidate rejection and interruption."""

import json
import shlex
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_reduce.cli import main
from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.engine import EngineConfig, InitialNotInteresting, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.failure_check.runner import FailureCheckRunner
from onnx_reduce.model import OnnxModel
from onnx_reduce.types import ValidationResult
from tests.helpers import chain_model, external_data_model


@pytest.mark.parametrize("location", ["missing.bin", "../weights.bin", "/tmp/weights.bin"])
def test_external_data_load_errors_are_diagnostic(tmp_path, capsys, location):
    proto = external_data_model()
    tensor = proto.graph.initializer[0]
    onnx.external_data_helper.set_external_data(tensor, location=location)
    tensor.ClearField("raw_data")
    source = tmp_path / "external.onnx"
    source.write_bytes(proto.SerializeToString())
    output = tmp_path / "out.onnx"
    assert main([str(source), "--test", "must-not-run", "-o", str(output)]) == 2
    message = capsys.readouterr().err
    assert "cannot load ONNX model" in message
    assert str(source) in message
    assert not output.exists()
    assert not (tmp_path / "out.onnx.work").exists()


def test_unsupported_ir_type_is_a_diagnostic_cli_error(tmp_path, capsys):
    map_type = helper.make_map_type_proto(
        TensorProto.INT64,
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1]),
    )
    value = helper.make_value_info("x", map_type)
    source = tmp_path / "map.onnx"
    onnx.save(helper.make_model(helper.make_graph([], "map", [value], [value])), source)
    assert main([str(source), "--test", "must-not-run", "-o", str(tmp_path / "out")]) == 2
    message = capsys.readouterr().err
    assert str(source) in message
    assert "cannot import ONNX model" in message
    assert "Map types are not supported" in message
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "outcome", [FailureCheckResult.INCONCLUSIVE, FailureCheckResult.NOT_INTERESTING]
)
def test_initial_failure_distinguishes_unknown_from_rejected(tmp_path, outcome):
    class FailureCheck:
        def run(self, path):
            return outcome

    with pytest.raises(InitialNotInteresting) as caught:
        ReductionEngine().run(OnnxModel.from_proto(chain_model()), FailureCheck(), tmp_path)
    assert caught.value.result is outcome
    message = str(caught.value)
    if outcome is FailureCheckResult.INCONCLUSIVE:
        assert "inconclusive" in message
        assert "direction" not in message
    else:
        assert "direction" in message
    record = json.loads(next(tmp_path.glob("run-*/evaluations.jsonl")).read_text())
    assert record["outcome"] == outcome.value


def test_candidate_rejections_retain_reasons_without_failure_check_calls(tmp_path, monkeypatch):
    class Proposals:
        name = "diagnostics"

        def generate(self, model):
            yield Candidate((DeleteNodes((999,)),))
            yield Candidate(())
            yield Candidate((DeleteNodes((0,)),))

    class FailureCheck:
        def run(self, path):
            return FailureCheckResult.INTERESTING

    validate = OnnxModel.validate

    def candidate_invalid(model, mode="strict"):
        if model.node_count == 5:
            return ValidationResult(False, "simulated missing type", mode)
        return validate(model, mode)

    monkeypatch.setattr(OnnxModel, "validate", candidate_invalid)
    _, report = ReductionEngine(passes=[Proposals()]).run(
        OnnxModel.from_proto(chain_model()),
        FailureCheck(),
        tmp_path,
    )
    records = [
        json.loads(line)
        for line in (Path(report.work_directory) / "candidates.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in records] == ["edit_failed", "not_cheaper", "invalid"]
    assert "out of range" in records[0]["reason"]
    assert records[2]["validation"]["reason"] == "simulated missing type"
    assert [r["candidate"]["sequence_number"] for r in records] == [0, 1, 2]
    assert report.failure_check_calls == 2
    assert report.candidates_invalid == 2
    assert report.candidates_not_cheaper == 1
    assert report.to_dict()["candidates"]["valid_rate"] == 0


def test_evaluated_candidates_have_correlated_diagnostics(tmp_path):
    class Reject:
        def run(self, path):
            return (
                FailureCheckResult.NOT_INTERESTING
                if "candidates" in Path(path).parts
                else FailureCheckResult.INTERESTING
            )

    _, report = ReductionEngine().run(OnnxModel.from_proto(chain_model()), Reject(), tmp_path)
    work = Path(report.work_directory)
    candidates = [json.loads(line) for line in (work / "candidates.jsonl").read_text().splitlines()]
    evaluations = [
        json.loads(line) for line in (work / "evaluations.jsonl").read_text().splitlines()
    ]
    evaluated = [item for item in evaluations if item["candidate"]]
    assert len(candidates) == len(evaluated)
    for candidate, evaluation in zip(candidates, evaluated, strict=True):
        assert candidate["candidate"] == evaluation["candidate"]
        assert candidate["digest"] == evaluation["digest"]
        assert candidate["status"] == evaluation["outcome"]
        assert candidate["cache_hit"] == evaluation["cache_hit"]


def test_cli_interrupt_preserves_checkpoint_and_returns_130(tmp_path, capsys, monkeypatch):
    source = tmp_path / "source.onnx"
    output = tmp_path / "output.onnx"
    onnx.save(chain_model(), source)
    calls = 0

    def observe(self, path):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        from onnx_reduce.failure_check.runner import FailureCheckObservation

        return FailureCheckObservation(FailureCheckResult.INTERESTING, 0)

    monkeypatch.setattr(FailureCheckRunner, "observe", observe)
    assert main([str(source), "--test", "unused", "-o", str(output)]) == 130
    assert "checkpoint.json" in capsys.readouterr().err
    checkpoint = next((tmp_path / "output.onnx.work").glob("run-*/checkpoint.json"))
    state = json.loads(checkpoint.read_text())
    model = OnnxModel.load(state["model"])
    assert model.node_count < 6
    assert model.canonical_hash() == state["digest"]
    assert not output.exists()


@pytest.mark.parametrize("body", ["raise SystemExit(125)", "import time; time.sleep(5)"])
def test_cli_initial_inconclusive_retains_process_evidence(tmp_path, capsys, body):
    source = tmp_path / "source.onnx"
    output = tmp_path / "output.onnx"
    onnx.save(chain_model(), source)
    command = shlex.join([sys.executable, "-c", body])
    assert main([str(source), "--test", command, "--timeout", "0.2", "-o", str(output)]) == 2
    assert "check is inconclusive" in capsys.readouterr().err
    record_path = next((tmp_path / "output.onnx.work").glob("run-*/evaluations.jsonl"))
    record = json.loads(record_path.read_text())
    observation = record["observation"]
    assert observation["timed_out"] == ("sleep" in body)
    if "sleep" not in body:
        assert observation["returncode"] == 125
    assert not output.exists()


def test_candidate_call_budget_is_recorded(tmp_path):
    class Reject:
        def run(self, path):
            return (
                FailureCheckResult.NOT_INTERESTING
                if "candidates" in Path(path).parts
                else FailureCheckResult.INTERESTING
            )

    _, report = ReductionEngine(config=EngineConfig(max_candidate_check_calls=1)).run(
        OnnxModel.from_proto(chain_model()),
        Reject(),
        tmp_path,
    )
    records = [
        json.loads(line)
        for line in (Path(report.work_directory) / "candidates.jsonl").read_text().splitlines()
    ]
    assert records[-1]["status"] == "budget_exhausted"
    assert records[-1]["reason"] == "calls"
    assert report.failure_check_calls == 3
    assert report.minimality["status"] == "unproven"
