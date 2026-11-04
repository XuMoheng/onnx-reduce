"""Reduction quality beyond deletion with promoted graph inputs."""

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_reduce.edits import BypassNode, Candidate, DropNodeAttribute
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.drop_outputs import DropOutputsPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass
from onnx_reduce.types import EditApplicationError
from tests.helpers import tensor_value


def make_model(nodes, *, inputs=None, outputs=None):
    return OnnxModel.from_proto(
        helper.make_model(
            helper.make_graph(
                nodes, "rewrite", inputs or [tensor_value("x")], outputs or [tensor_value("y")]
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
    )


class ReluWithNegCheck:
    def run(self, path):
        proto = onnx.load(path)
        # Preserve the original connection OR a direct Relu(x), but never a
        # new promoted input. This models a failure tied to the input contract.
        names = {n.op_type for n in proto.graph.node}
        interesting = [v.name for v in proto.graph.input] == ["x"] and "Relu" in names
        return FailureCheckResult.INTERESTING if interesting else FailureCheckResult.NOT_INTERESTING


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
def test_bypass_escapes_deletion_fixed_point(tmp_path, scheduler):
    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["middle"]),
            helper.make_node("Relu", ["middle"], ["y"]),
        ]
    )
    old_passes = (DropOutputsPass(), RemoveUnusedPass(), DeleteNodesPass())
    config = EngineConfig(scheduler=scheduler)
    old, old_report = ReductionEngine(passes=old_passes, config=config).run(
        model, ReluWithNegCheck(), tmp_path / "old"
    )
    assert old_report.stop_reason == "fixed_point"
    assert old.node_count == 2
    reduced, report = ReductionEngine(config=config).run(
        model, ReluWithNegCheck(), tmp_path / "new"
    )
    assert reduced.op_types == ("Relu",)
    assert reduced.input_names == ("x",)
    assert reduced.to_proto().graph.node[0].input == ["x"]
    assert reduced.validate("strict").valid
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.pass_stats["bypass_nodes"].accepted > 0
    assert model.node_count == 2


def test_bypass_preserves_stable_ids_and_supports_followup_edit():
    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["mid"]),
            helper.make_node("LeakyRelu", ["mid"], ["y"], alpha=0.2),
        ]
    )
    reduced = model.apply_candidate(Candidate((BypassNode(0, 0), DropNodeAttribute(0, "alpha"))))
    assert reduced.node_ids == (1,)
    assert reduced.to_proto().graph.node[0].input == ["x"]
    assert not reduced.to_proto().graph.node[0].attribute
    assert reduced.copy().node_ids == (1,)
    assert reduced.validate("strict").valid
    assert model.to_proto().graph.node[0].op_type == "Neg"


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
def test_optional_attributes_are_failure_gated(tmp_path, scheduler):
    class Check:
        def run(self, path):
            proto = onnx.load(path)
            keep = any(n.op_type == "LeakyRelu" for n in proto.graph.node)
            return FailureCheckResult.INTERESTING if keep else FailureCheckResult.NOT_INTERESTING

    model = make_model([helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.2)])
    reduced, report = ReductionEngine(config=EngineConfig(scheduler=scheduler)).run(
        model, Check(), tmp_path
    )
    assert not reduced.to_proto().graph.node[0].attribute
    assert reduced.cost() < model.cost()
    assert report.pass_stats["simplify_attributes"].accepted == 1
    assert report.minimality["status"] == "pass_fixed_point"


def test_required_and_custom_attributes_are_not_offered():
    model = make_model([helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)])
    assert not list(model.rewrite_candidates("attributes"))
    with pytest.raises(EditApplicationError, match="schema-optional"):
        model.apply_candidate(Candidate((DropNodeAttribute(0, "to"),)))
    custom = model.to_proto()
    custom.graph.node[0].domain = "custom"
    custom.opset_import.append(helper.make_opsetid("custom", 1))
    custom = OnnxModel.from_proto(custom)
    assert not list(custom.rewrite_candidates("attributes"))


def test_bypass_output_and_duplicate_guard():
    model = make_model([helper.make_node("Neg", ["x"], ["y"])])
    reduced = model.apply_candidate(Candidate((BypassNode(0, 0),)))
    assert reduced.output_names == ("x",)
    assert reduced.validate("strict").valid
    proto = model.to_proto()
    proto.graph.output.append(tensor_value("x"))
    with pytest.raises(EditApplicationError, match="duplicate"):
        OnnxModel.from_proto(proto).apply_candidate(Candidate((BypassNode(0, 0),)))


def test_bypass_updates_outer_captures():
    branch = helper.make_graph(
        [helper.make_node("Identity", ["mid"], ["out"])], "branch", [], [tensor_value("out")]
    )
    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["mid"]),
            helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch),
        ],
        inputs=[tensor_value("x"), tensor_value("cond", TensorProto.BOOL, ())],
    )
    reduced = model.apply_candidate(Candidate((BypassNode(0, 0),)))
    assert reduced.validate("strict").valid
    for attr in reduced.to_proto().graph.node[0].attribute:
        assert attr.g.node[0].input == ["x"]


def test_bypass_does_not_capture_shadowed_source():
    branch = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["x"],
                value=helper.make_tensor("const", TensorProto.FLOAT, [1], [0.0]),
            ),
            helper.make_node("Add", ["mid", "x"], ["out"]),
        ],
        "branch",
        [],
        [tensor_value("out")],
    )
    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["mid"]),
            helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch),
        ],
        inputs=[tensor_value("x"), tensor_value("cond", TensorProto.BOOL, ())],
    )
    with pytest.raises(EditApplicationError, match="shadowed"):
        model.apply_candidate(Candidate((BypassNode(0, 0),)))


def test_attribute_needed_by_failure_is_retained(tmp_path):
    class Check:
        def run(self, path):
            proto = onnx.load(path)
            keep = any(
                n.op_type == "LeakyRelu" and any(a.name == "alpha" for a in n.attribute)
                for n in proto.graph.node
            )
            return FailureCheckResult.INTERESTING if keep else FailureCheckResult.NOT_INTERESTING

    model = make_model([helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.2)])
    reduced, report = ReductionEngine().run(model, Check(), tmp_path)
    assert reduced.to_proto().graph.node[0].attribute[0].name == "alpha"
    assert report.pass_stats["simplify_attributes"].accepted == 0


def test_bypass_then_delete_does_not_reuse_stale_snapshot():
    from onnx_reduce.edits import DeleteNodes

    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["mid"]),
            helper.make_node("Relu", ["mid"], ["y"]),
        ]
    )
    reduced = model.apply_candidate(Candidate((BypassNode(0, 0), DeleteNodes((0,)))))
    assert reduced.node_count == 0
    assert not reduced.to_proto().graph.node
    assert reduced.validate("strict").valid


def test_type_incompatible_bypass_fails_validation():
    model = make_model(
        [
            helper.make_node("Cast", ["x"], ["mid"], to=TensorProto.FLOAT),
            helper.make_node("Relu", ["mid"], ["y"]),
        ],
        inputs=[tensor_value("x", TensorProto.INT64)],
    )
    reduced = model.apply_candidate(Candidate((BypassNode(0, 0),)))
    assert not reduced.validate("strict").valid


def test_independent_finalizer_includes_rewrites():
    from onnx_reduce.passes import default_passes
    from onnx_reduce.search.coverage import Finalizer

    model = make_model([helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.2)])
    finalizer = Finalizer(default_passes(), model)
    seen = set()
    while (attempt := finalizer.next_attempt(model)) is not None:
        seen.update(type(edit) for edit in attempt.candidate.edits)
    assert {BypassNode, DropNodeAttribute} <= seen
    assert finalizer.coverage().completed


def test_bypass_can_choose_second_input_and_skips_multi_output_nodes():
    model = make_model(
        [helper.make_node("Add", ["x", "z"], ["y"])], inputs=[tensor_value("x"), tensor_value("z")]
    )
    edits = [c.edits[0] for c in model.rewrite_candidates("bypass")]
    assert edits == [BypassNode(0, 0), BypassNode(0, 1)]
    reduced = model.apply_candidate(Candidate((edits[1],)))
    assert reduced.output_names == ("z",)
    assert reduced.validate("strict").valid
    model = make_model([helper.make_node("Dropout", ["x"], ["y", "mask"])])
    assert not list(model.rewrite_candidates("bypass"))
