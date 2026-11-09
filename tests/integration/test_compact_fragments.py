"""Connected multi-op witnesses must survive bypasses and boundary extraction."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from onnx_reduce.edits import BypassNode
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult as F
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.compact import CompactFragmentsPass
from onnx_reduce.preparation import preparation_caches
from tests.helpers import chain_model, control_flow_model


def branched_model(prefix=""):
    def node(op, inputs, output, **attrs):
        return h.make_node(op, [prefix + name for name in inputs], [prefix + output], **attrs)

    def value(name, shape):
        return h.make_tensor_value_info(prefix + name, T.FLOAT, shape)

    def tensor(data, name):
        return nh.from_array(np.asarray(data), prefix + name)

    nodes = [
        node("Transpose", ["x"], "t", perm=[0, 2, 1, 3]),
        node("MatMul", ["t", "w"], "m"),
        node("Add", ["m", "bias"], "a"),
        node("Constant", [], "shape", value=tensor(np.array([1, 2, 3, 4, 5]), "")),
        node("Reshape", ["a", "shape"], "r"),
        node("Unsqueeze", ["mask"], "u", axes=[-1]),
        node("Mul", ["r", "u"], "mul"),
        node("ReduceSum", ["mul"], "sum", axes=[3], keepdims=0),
        node("ReduceSum", ["mask"], "count", axes=[-1], keepdims=1),
        node("Constant", [], "epsilon", value=tensor(np.array(1e-4, dtype="float32"), "")),
        node("Add", ["count", "epsilon"], "denominator"),
        node("Div", ["sum", "denominator"], "div"),
        node("Unsqueeze", ["div"], "expanded", axes=[2]),
        node("MatMul", ["expanded", "w2"], "last"),
        node("Add", ["last", "bias2"], "y"),
    ]
    proto = h.make_model(
        h.make_graph(
            nodes,
            "branched",
            [value("x", [1, 3, 2, 5]), value("mask", [1, 2, 3, 4])],
            [value("y", [1, 2, 1, 3, 2])],
            [
                tensor(np.arange(100, dtype="float32").reshape(5, 20) / 100, "w"),
                tensor(np.zeros(20, dtype="float32"), "bias"),
                tensor(np.ones((5, 2), dtype="float32"), "w2"),
                tensor(np.zeros(2, dtype="float32"), "bias2"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 12)],
    )
    return OnnxModel.from_proto(onnx.shape_inference.infer_shapes(proto))


class ConnectedChainCheck:
    def run(self, path):
        graph = onnx.load(path).graph
        producers = {name: node for node in graph.node for name in node.output}
        pending = [value.name for value in graph.output]
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited or name not in producers:
                continue
            visited.add(name)
            node = producers[name]
            pending.extend(node.input)
            if node.op_type != "ReduceSum":
                continue
            current = node
            for expected in ("Reshape", "MatMul", "Transpose"):
                current = producers.get(current.input[0])
                while current is not None and current.op_type in {"Add", "Mul"}:
                    current = producers.get(current.input[0])
                if current is None or current.op_type != expected:
                    break
            else:
                return F.INTERESTING
        return F.NOT_INTERESTING


@pytest.mark.parametrize("prefix", ["", "renamed/"])
def test_compound_fragment_preserves_connected_chain_and_computes_correctly(prefix):
    source = branched_model(prefix)
    original = source.canonical_hash()
    candidates = list(CompactFragmentsPass().generate(source))
    assert 0 < len(candidates) <= CompactFragmentsPass.limit
    reduced = source.apply_candidate(candidates[0])
    assert reduced.validate("strict").valid
    assert reduced.op_types == ("Transpose", "MatMul", "Reshape", "ReduceSum")
    assert reduced.node_ids == (0, 1, 4, 7)
    assert reduced.input_names == (prefix + "x",)
    assert reduced.output_names == (prefix + "sum",)
    assert source.canonical_hash() == original
    proto = reduced.to_proto()
    assert {tensor.name for tensor in proto.graph.initializer} == {prefix + "w", prefix + "shape"}
    x = np.arange(30, dtype="float32").reshape(1, 3, 2, 5)
    weight = np.arange(100, dtype="float32").reshape(5, 20) / 100
    expected = (x.transpose(0, 2, 1, 3) @ weight).reshape(1, 2, 3, 4, 5).sum(axis=3)
    actual = ReferenceEvaluator(proto).run(None, {prefix + "x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
@pytest.mark.parametrize("mode", ["ir", "auto"])
def test_default_search_reaches_four_nodes_in_one_candidate_check(tmp_path, scheduler, mode):
    reduced, report = ReductionEngine(
        config=EngineConfig(scheduler=scheduler, model_edit_mode=mode, max_candidate_check_calls=1)
    ).run(branched_model(), ConnectedChainCheck(), tmp_path)
    assert reduced.op_types == ("Transpose", "MatMul", "Reshape", "ReduceSum")
    assert report.pass_stats["compact_fragments"].accepted == 1
    assert report.failure_check_calls == 3  # Initial, compound candidate, independent recheck.
    assert report.artifact_recheck["outcome"] == "interesting"


def test_compaction_cannot_replace_the_original_failure_without_a_positive_check(tmp_path):
    class RequiresAdd:
        def run(self, path):
            return F.INTERESTING if "Add" in OnnxModel.load(path).op_types else F.NOT_INTERESTING

    source = branched_model()
    reduced, report = ReductionEngine([CompactFragmentsPass()]).run(source, RequiresAdd(), tmp_path)
    assert reduced.canonical_hash() == source.canonical_hash()
    assert report.candidates_accepted == 0
    assert report.not_interesting > 0


def test_cast_is_not_bypassed_on_shape_equality_alone():
    proto = h.make_model(
        h.make_graph(
            [
                h.make_node("Neg", ["x"], ["neg"]),
                h.make_node("Cast", ["neg"], ["cast"], to=T.INT64),
                h.make_node("Reshape", ["cast", "shape"], ["r"]),
                h.make_node("ReduceSum", ["r"], ["y"], axes=[1], keepdims=0),
            ],
            "cast",
            [h.make_tensor_value_info("x", T.FLOAT, [2])],
            [h.make_tensor_value_info("y", T.INT64, [1])],
            [nh.from_array(np.array([1, 2], dtype="int64"), "shape")],
        ),
        opset_imports=[h.make_opsetid("", 12)],
    )
    model = OnnxModel.from_proto(onnx.shape_inference.infer_shapes(proto))
    candidate = next(CompactFragmentsPass().generate(model))
    reduced = model.apply_candidate(candidate)
    assert reduced.op_types == ("Cast", "Reshape", "ReduceSum")
    assert reduced.validate().valid


def test_unknown_shapes_control_flow_and_plain_chains_do_not_add_probes():
    proto = h.make_model(
        h.make_graph(
            [
                h.make_node("MatMul", ["x", "w"], ["m"]),
                h.make_node("Neg", ["m"], ["n"]),
                h.make_node("ReduceSum", ["n"], ["y"], axes=[1], keepdims=0),
            ],
            "dynamic",
            [
                h.make_tensor_value_info("x", T.FLOAT, ["n", "k"]),
                h.make_tensor_value_info("w", T.FLOAT, ["k", "m"]),
            ],
            [h.make_tensor_value_info("y", T.FLOAT, ["n"])],
        ),
        opset_imports=[h.make_opsetid("", 12)],
    )
    for model in (
        OnnxModel.from_proto(proto),
        OnnxModel.from_proto(control_flow_model()),
        OnnxModel.from_proto(chain_model(1500, 1140)),
    ):
        assert not list(CompactFragmentsPass().generate(model))


def test_export_without_intermediate_annotations_reuses_inference(monkeypatch):
    proto = branched_model().to_proto()
    del proto.graph.value_info[:]
    model = OnnxModel.from_proto(proto)
    digest = model.canonical_hash()
    infer = model._infer_type_summary
    calls = []

    def counted_inference():
        calls.append(1)
        return infer()

    monkeypatch.setattr(model, "_infer_type_summary", counted_inference)
    with preparation_caches():
        first = next(CompactFragmentsPass().generate(model))
        assert first == next(CompactFragmentsPass().generate(model))
    assert len(calls) == 1
    reduced = model.apply_candidate(first)
    assert reduced.op_types == ("Transpose", "MatMul", "Reshape", "ReduceSum")
    assert reduced.validate().valid
    assert model.canonical_hash() == digest


def test_large_unannotated_model_does_not_force_inference(monkeypatch):
    model = branched_model()
    proto = model.to_proto()
    del proto.graph.value_info[:]
    model = OnnxModel.from_proto(proto)

    def unexpected_inference(names):
        raise AssertionError("inference exceeded the configured model size limit")

    monkeypatch.setattr(model, "infer_value_info", unexpected_inference)
    limited = CompactFragmentsPass()
    limited.max_inference_bytes = 1
    assert not list(limited.generate(model))


def test_work_and_bypass_limits_decline_incomplete_fragments():
    model = branched_model()
    limited = CompactFragmentsPass()
    limited.max_visits = 1
    assert not list(limited.generate(model))
    limited.max_visits = CompactFragmentsPass.max_visits
    limited.max_bypasses = 1
    for candidate in limited.generate(model):
        assert sum(isinstance(edit, BypassNode) for edit in candidate.edits) <= 1
        assert model.apply_candidate(candidate).validate().valid
    limited.limit = 0
    assert not list(limited.generate(model))


def test_many_independent_branches_cannot_exhaust_the_candidate_budget():
    nodes, outputs = [], []
    for index in range(40):
        nodes.extend(
            [
                h.make_node("Neg", ["x"], [f"neg{index}"]),
                h.make_node("Reshape", [f"neg{index}", "shape"], [f"r{index}"]),
                h.make_node("ReduceSum", [f"r{index}"], [f"y{index}"], axes=[1], keepdims=0),
            ]
        )
        outputs.append(h.make_tensor_value_info(f"y{index}", T.FLOAT, [1, 2]))
    proto = h.make_model(
        h.make_graph(
            nodes,
            "wide",
            [h.make_tensor_value_info("x", T.FLOAT, [1, 2])],
            outputs,
            [nh.from_array(np.array([1, 1, 2], dtype="int64"), "shape")],
        ),
        opset_imports=[h.make_opsetid("", 12)],
    )
    model = OnnxModel.from_proto(onnx.shape_inference.infer_shapes(proto))
    candidates = list(CompactFragmentsPass().generate(model))
    assert len(candidates) == CompactFragmentsPass.limit
    for candidate in candidates:
        reduced = model.apply_candidate(candidate)
        assert reduced.op_types == ("Reshape", "ReduceSum")
        assert reduced.validate().valid
