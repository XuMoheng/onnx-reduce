import numpy as np
import pytest
from onnx import helper as h
from onnx import numpy_helper

from benchmarks.quality_cases import model, value
from onnx_reduce.edits import Candidate, ResizeOperator
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.candidates import local_tensor_candidates
from onnx_reduce.types import EditApplicationError


def weight(name, shape):
    return numpy_helper.from_array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), name)


@pytest.mark.parametrize("op,trans_a,trans_b", [("MatMul", 0, 0), ("Gemm", 0, 1), ("Gemm", 1, 0)])
def test_contract_axis_does_not_resize_equal_unrelated_dimensions(op, trans_a, trans_b):
    attrs = {} if op == "MatMul" else {"transA": trans_a, "transB": trans_b}
    proto = model(
        [h.make_node(op, ["x", "w"], ["y"], **attrs)],
        [value("x"), value("unrelated")],
        [value("y")],
        [weight("w", (4, 4))],
    )
    original = OnnxModel.from_proto(proto)
    resized = original.apply_candidate(Candidate((ResizeOperator(0, "contract", 1),)))
    result = resized.to_proto()
    assert (
        result.graph.input[1].type.tensor_type.shape == proto.graph.input[1].type.tensor_type.shape
    )
    assert list(result.graph.initializer[0].dims) == ([4, 1] if trans_b else [1, 4])
    assert [d.dim_value for d in result.graph.output[0].type.tensor_type.shape.dim] == [4, 4]
    assert resized.validate("strict").valid
    assert original.cost().tensor_elements > resized.cost().tensor_elements


@pytest.mark.parametrize(
    "kind,new,expected", [("conv_input", 2, (4, 1, 3, 3)), ("conv_output", 2, (2, 2, 3, 3))]
)
def test_grouped_conv_respects_channels_bias_and_group(kind, new, expected):
    proto = model(
        [h.make_node("Conv", ["x", "w", "b"], ["y"], group=2)],
        [value("x", (1, 4, 5, 5))],
        [value("y", (1, 4, 3, 3))],
        [weight("w", (4, 2, 3, 3)), weight("b", (4,))],
    )
    source = OnnxModel.from_proto(proto)
    resized = source.apply_candidate(Candidate((ResizeOperator(0, kind, new),)))
    assert tuple(resized.to_proto().graph.initializer[0].dims) == expected
    assert resized.validate("strict").valid
    if kind == "conv_output":
        assert list(resized.to_proto().graph.initializer[1].dims) == [2]
    with pytest.raises(EditApplicationError, match="group"):
        source.apply_candidate(Candidate((ResizeOperator(0, kind, 1),)))


def test_local_reshape_preserves_element_count_and_other_inputs():
    shape = numpy_helper.from_array(np.array([4, 4], dtype=np.int64), "shape")
    proto = model(
        [h.make_node("Reshape", ["x", "shape"], ["y"])],
        [value("x"), value("other")],
        [value("y")],
        [shape],
    )
    source = OnnxModel.from_proto(proto)
    resized = source.apply_candidate(Candidate((ResizeOperator(0, "reshape", 1),)))
    result = resized.to_proto()
    assert resized.validate("strict").valid
    np.testing.assert_array_equal(numpy_helper.to_array(result.graph.initializer[0]), [1, 4])
    assert [d.dim_value for d in result.graph.input[1].type.tensor_type.shape.dim] == [4, 4]


def test_declines_intermediate_producer_instead_of_modifying_unrelated_graph():
    proto = model(
        [h.make_node("Relu", ["x"], ["mid"]), h.make_node("MatMul", ["mid", "w"], ["y"])],
        [value("x")],
        [value("y")],
        [weight("w", (4, 4))],
    )
    assert list(local_tensor_candidates(proto)) == []
