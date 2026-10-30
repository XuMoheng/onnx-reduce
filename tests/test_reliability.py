"""Regressions for target identity, artifact isolation and search coverage."""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_reduce.cli import main
from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult, make_failure_check_config
from onnx_reduce.failure_check.evaluator import Evaluator
from onnx_reduce.failure_check.runner import FailureCheckObservation, FailureCheckRunner
from onnx_reduce.model import OnnxModel
from tests.helpers import chain_model, external_data_model, tensor_value


class Always:
    def run(self, path):
        return FailureCheckResult.INTERESTING


def two_outputs():
    return OnnxModel.from_proto(
        helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Relu", ["x"], ["first"]),
                    helper.make_node("Identity", ["x"], ["second"]),
                ],
                "outputs",
                [tensor_value("x")],
                [tensor_value("first"), tensor_value("second")],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )


def test_last_node_can_be_removed(tmp_path):
    reduced, report = ReductionEngine().run(
        OnnxModel.from_proto(chain_model(1, 0)),
        Always(),
        tmp_path,
    )
    assert reduced.node_count == 0
    assert reduced.validate().valid
    assert report.stop_reason == "fixed_point"
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.minimality["global_minimum"] is False


def test_first_output_is_tried_and_dead_bug_is_preserved(tmp_path):
    class SecondOutputFailureCheck:
        def run(self, path):
            m = OnnxModel.load(path)
            return (
                FailureCheckResult.INTERESTING
                if "second" in m.output_names and "Relu" in m.op_types
                else FailureCheckResult.NOT_INTERESTING
            )

    reduced, _ = ReductionEngine().run(two_outputs(), SecondOutputFailureCheck(), tmp_path)
    assert reduced.output_names == ("second",)
    assert "Relu" in reduced.op_types  # The now-dead node still carries the bug.


def test_external_weights_are_portable_and_part_of_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = external_data_model()
    expected_data = original.graph.initializer[0].raw_data
    onnx.save_model(
        original,
        source / "m.onnx",
        save_as_external_data=True,
        location="weights.bin",
        size_threshold=0,
    )
    model = OnnxModel.load(source / "m.onnx")
    artifact = model.materialize(tmp_path / "candidate" / "m.onnx")
    shutil.rmtree(source)
    reloaded = onnx.load(artifact.model_path)
    assert reloaded.graph.initializer[0].raw_data == expected_data
    assert artifact.total_bytes > 8192
    assert model.cost().byte_size == artifact.total_bytes
    assert OnnxModel.load(artifact.model_path).canonical_hash() == artifact.digest
    other = external_data_model()
    other.graph.initializer[0].raw_data = bytes(8192)
    assert OnnxModel.from_proto(other).canonical_hash() != artifact.digest


def test_stale_candidate_is_rejected():
    model = OnnxModel.from_proto(chain_model())
    from onnx_reduce.types import EditApplicationError

    with pytest.raises(EditApplicationError, match="base digest"):
        model.apply_candidate(Candidate((DeleteNodes((0,)),), base_digest="stale"))


def test_capture_keeps_outer_producer_alive():
    branch = helper.make_graph(
        [helper.make_node("Identity", ["captured"], ["b"])],
        "branch",
        [],
        [tensor_value("b")],
    )
    model = OnnxModel.from_proto(
        helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Identity", ["x"], ["captured"]),
                    helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch),
                ],
                "capture",
                [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
                [tensor_value("y")],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )
    assert model.validate().valid
    assert model.dead_node_indices() == ()
    reduced = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert reduced.validate().valid
    assert "captured" in reduced.input_names


def test_inconclusive_neighbors_do_not_prove_minimality(tmp_path):
    class Uncertain:
        calls = 0

        def run(self, path):
            self.calls += 1
            return (
                FailureCheckResult.INTERESTING
                if "initial" in str(path) or "final" in str(path)
                else FailureCheckResult.INCONCLUSIVE
            )

    _, report = ReductionEngine().run(
        OnnxModel.from_proto(chain_model(1, 0)), Uncertain(), tmp_path
    )
    assert report.stop_reason == "fixed_point"
    assert report.minimality["status"] == "unproven"
    assert report.minimality["sweep_inconclusive_candidate_count"] > 0


def test_duplicate_candidates_are_cached_and_have_unique_paths(tmp_path):
    class Duplicates:
        name = "duplicates"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            yield Candidate((DeleteNodes((0,)),))

    class Reject:
        def __init__(self):
            self.paths = []

        def run(self, path):
            self.paths.append(path)
            return (
                FailureCheckResult.NOT_INTERESTING
                if "candidates" in Path(path).parts
                else FailureCheckResult.INTERESTING
            )

    failure_check_runner = Reject()
    _, report = ReductionEngine(passes=[Duplicates()]).run(
        OnnxModel.from_proto(chain_model()),
        failure_check_runner,
        tmp_path,
    )
    assert report.failure_check_calls == 3
    assert report.cache_hits == 1
    records = [
        json.loads(line)
        for line in (Path(report.work_directory) / "evaluations.jsonl").read_text().splitlines()
    ]
    candidates = [r for r in records if r["candidate"]]
    assert len({r["model"] for r in candidates}) == 2
    assert candidates[0]["candidate"]["base_digest"]
    assert candidates[0]["candidate"]["edits"][0]["op"] == "DeleteNodes"
    assert candidates[1]["cache_hit"]


def test_no_free_pass_bypasses_failure_check(tmp_path):
    class Unsafe:
        name = "unsafe"
        is_free = True

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))

    class Reject:
        def run(self, path):
            return (
                FailureCheckResult.NOT_INTERESTING
                if "candidates" in Path(path).parts
                else FailureCheckResult.INTERESTING
            )

    model = OnnxModel.from_proto(chain_model())
    reduced, report = ReductionEngine(passes=[Unsafe()]).run(model, Reject(), tmp_path)
    assert reduced.canonical_hash() == model.canonical_hash()
    assert report.failure_check_calls == 3


def test_signature_rejects_another_bug_and_does_not_cache_inconclusive(tmp_path):
    class Observations:
        def observe(self, path):
            if path.name == "initial":
                return FailureCheckObservation(
                    FailureCheckResult.INTERESTING, 0, stderr="assertion A"
                )
            return FailureCheckObservation(FailureCheckResult.INTERESTING, 0, stderr="assertion B")

    evaluator = Evaluator(Observations(), tmp_path, signature_regex=r"assertion (\w+)")
    assert (
        evaluator.evaluate(tmp_path / "initial", "a", initial=True, mandatory=True).outcome
        is FailureCheckResult.INTERESTING
    )
    result = evaluator.evaluate(tmp_path / "candidate", "b")
    assert result.outcome is FailureCheckResult.NOT_INTERESTING
    assert result.target_identity_mismatch
    assert evaluator.evaluate(tmp_path / "candidate", "b").cache_hit

    class Unknown:
        def run(self, path):
            return FailureCheckResult.INCONCLUSIVE

    uncertain = Evaluator(Unknown(), tmp_path / "uncertain")
    for _ in range(2):
        assert not uncertain.evaluate(tmp_path / "x", "same").cache_hit
    assert uncertain.calls == 2


def test_baseline_signal_matching():
    from onnx_reduce.failure_check.evaluator import TargetMatcher

    matcher = TargetMatcher()
    assert (
        matcher.match(
            FailureCheckObservation(FailureCheckResult.INTERESTING, -6), initial=True
        ).outcome
        is FailureCheckResult.INTERESTING
    )
    result = matcher.match(FailureCheckObservation(FailureCheckResult.INTERESTING, -11))
    assert result.outcome is FailureCheckResult.NOT_INTERESTING
    assert result.target_identity_mismatch


def test_checkpoint_survives_failure_check_failure(tmp_path):
    class Failing:
        calls = 0

        def run(self, path):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("backend unavailable")
            return FailureCheckResult.INTERESTING

    with pytest.raises(RuntimeError, match="backend unavailable"):
        ReductionEngine().run(OnnxModel.from_proto(chain_model()), Failing(), tmp_path)
    checkpoint = next(tmp_path.glob("run-*/checkpoint.json"))
    state = json.loads(checkpoint.read_text())
    model = OnnxModel.load(state["model"])
    assert model.node_count < 6
    assert model.canonical_hash() == state["digest"]
    assert (checkpoint.parent / "task.json").exists()


def test_cli_final_output_recheck_fails_with_nonzero_exit(tmp_path):
    source = tmp_path / "source.onnx"
    output = tmp_path / "final.onnx"
    report_path = tmp_path / "report.json"
    onnx.save(chain_model(1, 0), source)
    script = tmp_path / "check with spaces.py"
    script.write_text(
        "import sys\nraise SystemExit(1 if sys.argv[-1].endswith('final.onnx') else 0)\n"
    )
    command = shlex.join([sys.executable, str(script)])
    assert (
        main(
            [
                str(source),
                "--test",
                command,
                "-o",
                str(output),
                "--max-candidate-check-calls",
                "0",
                "--report",
                str(report_path),
            ]
        )
        == 1
    )
    report = json.loads(report_path.read_text())
    assert report["stop_reason"] == "unstable_check"
    assert report["minimality"]["status"] == "unproven"
    assert output.exists()
    assert (Path(report["work_directory"]) / "replay.json").exists()


def test_runner_observes_bounded_output_and_quoted_argv(tmp_path):
    script = tmp_path / "script with spaces.py"
    script.write_text("import sys\nprint(sys.argv[1])\nprint('x' * 10000, file=sys.stderr)\n")
    failure_check_config = make_failure_check_config(
        shlex.join([sys.executable, str(script), "argument with spaces"]),
        timeout_seconds=5,
        crash_is_interesting=True,
        timeout_is_interesting=False,
    )
    observation = FailureCheckRunner(failure_check_config, log_limit=100).observe(
        tmp_path / "model.onnx"
    )
    assert observation.outcome is FailureCheckResult.INTERESTING
    assert observation.stdout.strip() == "argument with spaces"
    assert len(observation.stderr) == 100
    assert observation.output_truncated


def test_effective_checker_policy_is_reported(tmp_path):
    from tests.helpers import custom_domain_model

    _, report = ReductionEngine(config=EngineConfig(max_candidate_check_calls=0)).run(
        OnnxModel.from_proto(custom_domain_model()),
        Always(),
        tmp_path,
    )
    assert report.requested_validation_mode == "strict"
    assert report.effective_validation_modes == ["structural"]
    assert report.custom_domains == ["com.example"]


def test_cli_regex_preserves_target_instead_of_another_bug(tmp_path):
    source = tmp_path / "source.onnx"
    output = tmp_path / "reduced.onnx"
    onnx.save(chain_model(4, 1), source)
    script = tmp_path / "check.py"
    script.write_text(
        "import sys, onnx\n"
        "m = onnx.load(sys.argv[-1])\n"
        "print('assertion ' + ('A' if any(n.op_type == 'Relu' for n in m.graph.node) else 'B'))\n"
    )
    assert (
        main(
            [
                str(source),
                "--test",
                shlex.join([sys.executable, str(script)]),
                "--signature-regex",
                r"assertion (\w+)",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert [n.op_type for n in onnx.load(output).graph.node] == ["Relu"]
    report = json.loads(
        next((tmp_path / "reduced.onnx.work").glob("run-*/report.json")).read_text()
    )
    assert report["failure_checks"]["target_identity_mismatches"] > 0


def test_cli_external_weights_end_to_end(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    onnx.save_model(
        external_data_model(),
        source / "m.onnx",
        save_as_external_data=True,
        location="weights.bin",
        size_threshold=0,
    )
    output = tmp_path / "out" / "final.onnx"
    script = tmp_path / "check.py"
    script.write_text(
        "import sys, onnx\n"
        "m = onnx.load(sys.argv[-1])\n"
        "raise SystemExit(0 if len(m.graph.initializer) else 1)\n"
    )
    assert (
        main(
            [
                str(source / "m.onnx"),
                "--test",
                shlex.join([sys.executable, str(script)]),
                "-o",
                str(output),
                "--max-candidate-check-calls",
                "0",
            ]
        )
        == 0
    )
    shutil.rmtree(source)
    assert len(onnx.load(output).graph.initializer[0].raw_data) == 8192


def test_cache_can_be_disabled_and_mandatory_checks_always_run(tmp_path):
    class Reject:
        calls = 0

        def run(self, path):
            self.calls += 1
            return (
                FailureCheckResult.NOT_INTERESTING
                if self.calls in (2, 3)
                else FailureCheckResult.INTERESTING
            )

    class Duplicates:
        name = "duplicate"

        def generate(self, model):
            for _ in range(2):
                yield Candidate((DeleteNodes((0,)),))

    failure_check_runner = Reject()
    _, report = ReductionEngine(
        passes=[Duplicates()], config=EngineConfig(cache_enabled=False)
    ).run(
        OnnxModel.from_proto(chain_model()),
        failure_check_runner,
        tmp_path,
    )
    assert report.failure_check_calls == 4
    assert report.cache_hits == 0


def test_truncated_missing_signature_remains_inconclusive(tmp_path):
    from onnx_reduce.failure_check.evaluator import TargetMatcher

    matcher = TargetMatcher(signature_regex=r"assertion (\w+)")
    matcher.match(
        FailureCheckObservation(FailureCheckResult.INTERESTING, 0, stdout="assertion A"),
        initial=True,
    )
    result = matcher.match(
        FailureCheckObservation(FailureCheckResult.INTERESTING, 0, output_truncated=True)
    )
    assert result.outcome is FailureCheckResult.INCONCLUSIVE
