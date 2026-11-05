"""Computed shape values must survive extraction without guessing runtime data."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_reduce.edits import Candidate, ExtractNodes
from onnx_reduce.model import OnnxModel
from onnx_reduce.model.operations.constant_shapes import constant_shapes
from onnx_reduce.types import EditApplicationError, EditOptions
from tests.helpers import dynamic_expand_model, tensor_value


@pytest.mark.parametrize("mode", ["ir", "auto"])
def test_constant_shape_extraction_keeps_values_and_infers_output(mode):
    source = OnnxModel.from_proto(dynamic_expand_model()).with_edit_options(EditOptions(mode))
    before = source.canonical_hash()
    ordinary = source.apply_candidate(Candidate((ExtractNodes((8,)),)))
    assert ordinary.validate().valid
    assert ordinary.input_names == ("expanded_input", "expand_shape")
    fragment = source.apply_candidate(Candidate((ExtractNodes((8,), materialize_shapes=True),)))
    assert fragment.validate().valid
    assert fragment.op_types == ("Expand",)
    assert fragment.node_ids == (8,)
    assert fragment.input_names == ("expanded_input",)
    graph = fragment.to_proto().graph
    assert len(graph.initializer) == 1
    np.testing.assert_array_equal(
        numpy_helper.to_array(graph.initializer[0]), [1, 6, 3, 2, 2, 12, 32, 32]
    )
    assert [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim] == [
        1,
        6,
        3,
        2,
        2,
        12,
        32,
        32,
    ]
    assert fragment.edit_options.mode == mode
    assert source.canonical_hash() == before
    assert not source.to_proto().graph.value_info


@pytest.mark.parametrize("default", [False, True])
def test_overridable_graph_inputs_are_never_evaluated(default):
    proto = dynamic_expand_model()
    del proto.graph.node[1]  # target becomes a runtime input, optionally with a default.
    proto.graph.input.append(tensor_value("target", TensorProto.INT64, [8]))
    if default:
        proto.graph.initializer.append(
            helper.make_tensor("target", TensorProto.INT64, [8], [1, 6, 3, 2, 2, 12, 32, 32])
        )
    model = OnnxModel.from_proto(proto)
    assert constant_shapes(model, ["expand_shape"]) == {}
    with pytest.raises(EditApplicationError, match="no bounded constant shape"):
        model.apply_candidate(Candidate((ExtractNodes((7,), materialize_shapes=True),)))
    assert model.apply_candidate(Candidate((ExtractNodes((7,)),))).validate().valid


def test_large_constant_of_shape_is_declined_before_allocation(monkeypatch):
    from onnx.reference import ReferenceEvaluator

    proto = dynamic_expand_model()
    proto.graph.node[2].attribute[0].t.CopyFrom(
        helper.make_tensor("rank", TensorProto.INT64, [1], [10**9])
    )

    def unexpected(*args, **kwargs):
        pytest.fail("oversized ConstantOfShape must not run")

    monkeypatch.setattr(ReferenceEvaluator, "run", unexpected)
    assert constant_shapes(OnnxModel.from_proto(proto), ["ones"]) == {}


def test_unsupported_and_local_operators_are_not_executed(monkeypatch):
    from onnx.reference import ReferenceEvaluator

    proto = dynamic_expand_model()
    proto.graph.node[7].domain = "custom"
    proto.opset_import.append(helper.make_opsetid("custom", 1))

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported expressions must not run")

    monkeypatch.setattr(ReferenceEvaluator, "run", unexpected)
    assert constant_shapes(OnnxModel.from_proto(proto), ["expand_shape"]) == {}
    proto.graph.node[7].domain = ""
    proto.functions.append(
        helper.make_function(
            "",
            "Where",
            ["c", "x", "y"],
            ["z"],
            [helper.make_node("Identity", ["x"], ["z"])],
            proto.opset_import,
        )
    )
    assert constant_shapes(OnnxModel.from_proto(proto), ["expand_shape"]) == {}


def test_deep_expressions_are_bounded_without_recursion_errors():
    proto = dynamic_expand_model()
    current = "target"
    for i in range(1500):
        name = f"deep{i}"
        proto.graph.node.append(helper.make_node("Identity", [current], [name]))
        current = name
    assert constant_shapes(OnnxModel.from_proto(proto), [current]) == {}


def test_evaluation_failure_leaves_the_ordinary_candidate_available(monkeypatch):
    from onnx.reference import ReferenceEvaluator

    def fail(*args, **kwargs):
        raise RuntimeError("reference evaluator unavailable")

    monkeypatch.setattr(ReferenceEvaluator, "run", fail)
    model = OnnxModel.from_proto(dynamic_expand_model())
    assert constant_shapes(model, ["expand_shape"]) == {}
    assert model.apply_candidate(Candidate((ExtractNodes((8,)),))).validate().valid


def test_result_matches_onnx_reference_execution():
    from onnx.reference import ReferenceEvaluator

    proto = dynamic_expand_model(tail=0)
    # Only execute the shape computation; never allocate the large data path.
    del proto.graph.node[8:]
    del proto.graph.node[0]
    del proto.graph.input[:]
    del proto.graph.output[:]
    proto.graph.output.append(tensor_value("expand_shape", TensorProto.INT64, [8]))
    expected = ReferenceEvaluator(proto).run(None, {})[0]
    actual = constant_shapes(OnnxModel.from_proto(proto), ["expand_shape"])
    np.testing.assert_array_equal(numpy_helper.to_array(actual["expand_shape"]), expected)


def test_reshape_uses_the_same_constant_boundary_rule():
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["data"]),
                helper.make_node("Add", ["a", "b"], ["shape"]),
                helper.make_node("Reshape", ["data", "shape"], ["y"]),
            ],
            "reshape",
            [tensor_value("x", shape=[6])],
            [tensor_value("y", shape=[2, 3])],
            initializer=[
                helper.make_tensor("a", TensorProto.INT64, [2], [1, 2]),
                helper.make_tensor("b", TensorProto.INT64, [2], [1, 1]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 12)],
    )
    fragment = OnnxModel.from_proto(proto).apply_candidate(
        Candidate((ExtractNodes((2,), materialize_shapes=True),))
    )
    assert fragment.op_types == ("Reshape",)
    assert fragment.input_names == ("data",)
    assert fragment.validate().valid
    np.testing.assert_array_equal(
        numpy_helper.to_array(fragment.to_proto().graph.initializer[0]), [2, 3]
    )
