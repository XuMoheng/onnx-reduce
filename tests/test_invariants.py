"""Regressions for graph ownership, artifact identity and search limits."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_reduce.cli import main
from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckConfig, FailureCheckResult
from onnx_reduce.model import OnnxModel
from onnx_reduce.storage import atomic_write
from tests.helpers import chain_model, external_data_model, tensor_value


def captured_model(*, nested=False):
    branch = helper.make_graph(
        [helper.make_node("Add", ["x", "weight"], ["b"])],
        "branch",
        [],
        [tensor_value("b")],
    )
    if nested:
        branch = helper.make_graph(
            [helper.make_node("If", ["cond"], ["b"], then_branch=branch, else_branch=branch)],
            "outer",
            [],
            [tensor_value("b")],
        )
    return OnnxModel.from_proto(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)],
                "capture",
                [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
                [tensor_value("y")],
                [helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )


@pytest.mark.parametrize("nested", [False, True])
def test_deleted_subgraphs_release_captures_in_same_candidate(nested):
    model = captured_model(nested=nested)
    original_hash = model.canonical_hash()
    assert model.validate().valid
    assert model.unused_input_names() == ()
    assert model.tensor_index()["x"].consumers == (0,)
    assert model.tensor_index()["weight"].consumers == (0,)

    deleted = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert deleted.unused_input_names() == ("x", "cond")
    assert deleted.unused_initializer_names() == ("weight",)
    reduced = model.apply_candidate(
        Candidate(
            (
                DeleteNodes((0,)),
                DropInputs(("x", "cond")),
                DropInitializers(("weight",)),
            )
        )
    )
    assert reduced.validate().valid
    assert reduced.input_names == ("y",)
    assert reduced.cost().initializer_count == 0
    assert model.canonical_hash() == original_hash
    assert model.unused_input_names() == ()


def test_retained_subgraphs_keep_shared_captures():
    proto = captured_model().to_proto()
    second = proto.graph.node.add()
    second.CopyFrom(proto.graph.node[0])
    second.output[0] = "z"
    proto.graph.output.append(tensor_value("z"))
    model = OnnxModel.from_proto(proto)
    reduced = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert reduced.validate().valid
    assert reduced.unused_input_names() == ()
    assert reduced.unused_initializer_names() == ()
    assert reduced.tensor_index()["x"].consumers == (0,)


def test_artifact_snapshot_is_shared_by_cost_hash_and_materialization(tmp_path, monkeypatch):
    model = OnnxModel.from_proto(chain_model())
    serialize_calls = 0
    original = model.to_proto

    def to_proto():
        nonlocal serialize_calls
        serialize_calls += 1
        return original()

    monkeypatch.setattr(model, "to_proto", to_proto)
    cost = model.cost()
    digest = model.canonical_hash()
    model.save(tmp_path / "saved.onnx")
    artifact = model.materialize(tmp_path / "artifact.onnx")
    payload = artifact.model_path.read_bytes()
    assert serialize_calls == 1
    assert hashlib.sha256(payload).hexdigest() == digest == artifact.digest
    assert len(payload) == cost.byte_size == artifact.total_bytes
    assert payload == (tmp_path / "saved.onnx").read_bytes()
    onnx.checker.check_model(onnx.load(artifact.model_path), full_check=True)
    reduced = model.apply_candidate(Candidate((DeleteNodes((0,)),), base_digest=digest))
    assert reduced.canonical_hash() != digest
    assert model.canonical_hash() == digest


def test_proto_tensor_mutations_cannot_invalidate_an_owned_snapshot(tmp_path):
    original = external_data_model()
    model = OnnxModel.from_proto(original)
    digest = model.canonical_hash()
    expected = original.graph.initializer[0].raw_data
    original.graph.initializer[0].raw_data = bytes(len(expected))
    exported = model.to_proto()
    assert exported.graph.initializer[0].raw_data == expected
    exported.graph.initializer[0].raw_data = bytes(len(expected))
    assert model.copy().canonical_hash() == digest
    artifact = model.materialize(tmp_path / "isolated.onnx")
    assert onnx.load(artifact.model_path).graph.initializer[0].raw_data == expected


class RejectCandidates:
    def run(self, path):
        return (
            FailureCheckResult.NOT_INTERESTING
            if "candidates" in Path(path).parts
            else FailureCheckResult.INTERESTING
        )


class Duplicates:
    name = "duplicates"

    def generate(self, model):
        yield Candidate((DeleteNodes((0,)),))
        yield Candidate((DeleteNodes((0,)),))


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_cache_still_works_at_exact_call_limit(tmp_path, cache_enabled):
    _, report = ReductionEngine(
        passes=[Duplicates()],
        config=EngineConfig(max_candidate_check_calls=1, cache_enabled=cache_enabled),
    ).run(OnnxModel.from_proto(chain_model()), RejectCandidates(), tmp_path)
    assert report.failure_check_calls == 3  # Initial, one candidate, mandatory final.
    assert report.cache_hits == int(cache_enabled)
    assert report.stop_reason == ("fixed_point" if cache_enabled else "budget_exhausted")
    assert report.minimality["status"] == ("pass_fixed_point" if cache_enabled else "unproven")


def test_expired_time_budget_after_preparation_does_not_start_failure_check(tmp_path, monkeypatch):
    # A fake clock makes the deadline deterministic without slow sleeps.
    clock = [0.0]
    monkeypatch.setattr("onnx_reduce.engine.time.monotonic", lambda: clock[0])
    materialize = OnnxModel.materialize

    def slow_materialize(model, path):
        result = materialize(model, path)
        if "candidates" in Path(path).parts:
            clock[0] = 2.0
        return result

    monkeypatch.setattr(OnnxModel, "materialize", slow_materialize)
    _, report = ReductionEngine(
        passes=[Duplicates()],
        config=EngineConfig(time_budget_seconds=1),
    ).run(OnnxModel.from_proto(chain_model()), RejectCandidates(), tmp_path)
    assert report.failure_check_calls == 2
    assert report.stop_reason == "budget_exhausted"


def test_non_decreasing_candidates_are_not_counted_as_validated(tmp_path):
    class NoChange:
        name = "no_change"

        def generate(self, model):
            yield Candidate(())

    _, report = ReductionEngine(passes=[NoChange()]).run(
        OnnxModel.from_proto(chain_model()),
        RejectCandidates(),
        tmp_path,
    )
    counts = report.to_dict()["candidates"]
    assert counts["not_cheaper"] == 1
    assert counts["valid_rate"] is None
    assert report.failure_check_calls == 2


@pytest.mark.parametrize(
    "config",
    [
        {"max_candidate_check_calls": -1},
        {"max_candidate_check_calls": 1.5},
        {"max_candidate_check_calls": True},
        {"time_budget_seconds": 0},
        {"time_budget_seconds": float("nan")},
        {"time_budget_seconds": float("inf")},
        {"validation_mode": "typo"},
        {"target_mode": "typo"},
    ],
)
def test_library_engine_configuration_is_validated(config):
    with pytest.raises(ValueError):
        EngineConfig(**config)


@pytest.mark.parametrize(
    "config",
    [
        {"command": ()},
        {"command": ("",)},
        {"command": "echo"},
        {"command": ("echo\0",)},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"inconclusive_exit_codes": {0}},
        {"inconclusive_exit_codes": {-1}},
        {"inconclusive_exit_codes": {256}},
    ],
)
def test_library_failure_check_configuration_is_validated(config):
    with pytest.raises(ValueError):
        FailureCheckConfig(**({"command": (sys.executable,)} | config))


@pytest.mark.parametrize("alias", ["input", "output", "symlink", "hardlink"])
def test_cli_rejects_report_aliasing_a_model_before_execution(tmp_path, alias, capsys):
    source = tmp_path / "source.onnx"
    output = tmp_path / "output.onnx"
    onnx.save(chain_model(), source)
    original = source.read_bytes()
    report = source if alias == "input" else output
    if alias in {"symlink", "hardlink"}:
        report = tmp_path / "alias.json"
        if alias == "symlink":
            report.symlink_to(source)
        else:
            report.hardlink_to(source)
    assert (
        main([str(source), "--test", "must-not-run", "-o", str(output), "--report", str(report)])
        == 2
    )
    assert "must not overwrite" in capsys.readouterr().err
    assert source.read_bytes() == original
    assert not output.exists()
    assert not (tmp_path / "output.onnx.work").exists()


def test_invalid_model_is_a_cli_error(tmp_path, capsys):
    source = tmp_path / "invalid.onnx"
    source.write_bytes(b"not an ONNX protobuf")
    assert main([str(source), "--test", "must-not-run", "-o", str(tmp_path / "out")]) == 2
    assert "cannot decode ONNX model" in capsys.readouterr().err


def test_atomic_write_preserves_existing_tmp_neighbor_and_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old")
    neighbor = tmp_path / "model.onnx.tmp"
    neighbor.write_bytes(b"unrelated")

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    with monkeypatch.context() as patch:
        patch.setattr("onnx_reduce.storage.os.replace", fail_replace)
        with pytest.raises(OSError, match="simulated"):
            atomic_write(path, b"new")
    assert path.read_bytes() == b"old"
    assert neighbor.read_bytes() == b"unrelated"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx", "model.onnx.tmp"]
    atomic_write(path, b"new")
    assert path.read_bytes() == b"new"
