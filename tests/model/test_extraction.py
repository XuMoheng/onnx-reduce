"""Live boundary reconstruction must preserve types, constants and lexical scope."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_reduce.edits import Candidate, DeleteNodes, ExtractNodes
from onnx_reduce.model import OnnxModel
from onnx_reduce.preparation import preparation_caches
from onnx_reduce.types import EditApplicationError, EditOptions
from tests.helpers import chain_model, tensor_value


def extract(model, indices):
    return model.apply_candidate(Candidate((ExtractNodes(indices),)))


@pytest.mark.parametrize("mode", ["ir", "auto"])
def test_fragment_is_live_typed_and_owned_with_stable_ids(mode):
    source = OnnxModel.from_proto(chain_model(5, 2)).with_edit_options(EditOptions(mode))
    before = source.canonical_hash()
    with preparation_caches():
        fragment = extract(source, (2,))
    assert fragment.node_ids == (2,)
    assert fragment.input_names == ("v1",)
    assert fragment.output_names == ("v2",)
    assert fragment.op_types == ("Relu",)
    assert fragment.dead_node_indices() == ()
    assert fragment.validate().valid
    assert fragment.edit_options.mode == mode
    assert source.canonical_hash() == before
    fragment.to_proto().graph.ClearField("node")
    assert fragment.node_count == 1


def test_repeated_and_noncontiguous_selection_keeps_current_topology():
    source = OnnxModel.from_proto(chain_model(5, 2))
    fragment = extract(source, (3, 1, 1))
    assert fragment.node_ids == (1, 3)
    assert fragment.input_names == ("v0", "v2")
    assert fragment.output_names == ("v1", "v3")
    assert fragment.validate().valid


def test_compound_extraction_uses_intermediate_indices_and_fresh_queries():
    source = OnnxModel.from_proto(chain_model(5, 2))
    fragment = source.apply_candidate(Candidate((ExtractNodes((1, 2, 3)), ExtractNodes((1,)))))
    assert fragment.node_ids == (2,)
    assert fragment.input_names == ("v1",)
    assert fragment.output_names == ("v2",)
    assert fragment.validate().valid
    follow = fragment.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert follow.node_ids == ()
    assert follow.validate().valid


@pytest.mark.parametrize("indices", [(), (-1,), (1000,)])
def test_invalid_selection_is_not_a_tool_error(indices):
    with pytest.raises(EditApplicationError, match="in-range"):
        extract(OnnxModel.from_proto(chain_model()), indices)


def test_stale_extraction_is_rejected():
    with pytest.raises(EditApplicationError, match="base digest"):
        OnnxModel.from_proto(chain_model()).apply_candidate(
            Candidate((ExtractNodes((0,)),), base_digest="stale")
        )


def test_weight_and_constant_data_are_retained_without_unused_resources():
    weights = numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(2, 2), "w")
    constant = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), "original_name")
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Constant", [], ["c"], value=constant),
                helper.make_node("MatMul", ["c", "w"], ["z"]),
                helper.make_node("Identity", ["z"], ["y"]),
            ],
            "weights",
            [tensor_value("unused", shape=[2, 2])],
            [tensor_value("y", shape=[2, 2])],
            initializer=[
                weights,
                numpy_helper.from_array(np.zeros(1000, dtype=np.float32), "unused_weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    source = OnnxModel.from_proto(proto)
    fragment = extract(source, (1,))
    assert fragment.validate().valid
    assert fragment.input_names == ()
    assert fragment.output_names == ("z",)
    retained = {t.name: numpy_helper.to_array(t) for t in fragment.to_proto().graph.initializer}
    assert set(retained) == {"c", "w"}
    np.testing.assert_array_equal(retained["w"], numpy_helper.to_array(weights))
    np.testing.assert_array_equal(retained["c"], numpy_helper.to_array(constant))
    assert len(source.to_proto().graph.initializer) == 2


def test_multi_output_and_optional_input_positions_are_preserved():
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["v"]),
                helper.make_node("Dropout", ["v", "", ""], ["y", "mask"]),
            ],
            "optional",
            [tensor_value("x")],
            [tensor_value("y")],
            value_info=[tensor_value("v"), tensor_value("mask", TensorProto.BOOL)],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    fragment = extract(OnnxModel.from_proto(proto), (1,))
    assert fragment.validate().valid
    assert tuple(fragment.to_proto().graph.node[0].input) == ("v", "", "")
    assert fragment.output_names == ("y", "mask")
    assert fragment.input_names == ("v",)


def test_subgraph_captures_keep_outer_weights_and_inputs():
    body = helper.make_graph(
        [helper.make_node("Add", ["mid", "w"], ["inner"])], "body", [], [tensor_value("inner")]
    )
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["mid"]),
                helper.make_node("If", ["cond"], ["z"], then_branch=body, else_branch=body),
                helper.make_node("Identity", ["z"], ["y"]),
            ],
            "captures",
            [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
            [tensor_value("y")],
            initializer=[helper.make_tensor("w", TensorProto.FLOAT, [1], [2.0])],
            value_info=[tensor_value("mid"), tensor_value("z")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    fragment = extract(OnnxModel.from_proto(proto), (1,))
    assert fragment.validate().valid
    assert set(fragment.input_names) == {"cond", "mid"}
    assert [t.name for t in fragment.to_proto().graph.initializer] == ["w"]
    assert fragment.output_names == ("z",)
    assert fragment.cost().node_count == 3


def test_inference_failure_does_not_invent_boundary_types(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("inference unavailable")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail)
    source = OnnxModel.from_proto(chain_model(5, 2))
    with preparation_caches():
        fragment = extract(source, (2,))
    assert not fragment.validate("strict").valid
    assert fragment.validate("permissive").valid


def test_interface_projection_preserves_names_order_types_and_unused_inputs():
    proto = chain_model(5, 2)
    proto.graph.input.append(tensor_value("unused"))
    source = OnnxModel.from_proto(proto)
    fragment = source.apply_candidate(Candidate((ExtractNodes((2,), preserve_interface=True),)))
    result = fragment.to_proto()
    assert tuple(result.graph.input) == tuple(source.to_proto().graph.input)
    assert tuple(result.graph.output) == tuple(source.to_proto().graph.output)
    assert tuple(result.graph.node[0].input) == ("x",)
    assert tuple(result.graph.node[0].output) == ("y",)
    assert fragment.validate().valid
    assert fragment.node_ids == (2,)


@pytest.mark.parametrize("outputs", [(), ("missing",), ("v2", "v2")])
def test_projection_rejects_invalid_output_selection(outputs):
    source = OnnxModel.from_proto(chain_model(5, 2))
    with pytest.raises(EditApplicationError, match="unique retained"):
        source.apply_candidate(Candidate((ExtractNodes((2,), outputs=outputs),)))


def test_interface_projection_does_not_guess_incompatible_shapes():
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Split", ["x"], ["a", "b"], axis=0),
                helper.make_node("Relu", ["a"], ["z"]),
                helper.make_node("Concat", ["z", "b"], ["y"], axis=0),
            ],
            "different-boundaries",
            [tensor_value("x", shape=[2])],
            [tensor_value("y", shape=[2])],
            value_info=[tensor_value("a"), tensor_value("b"), tensor_value("z")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    source = OnnxModel.from_proto(proto)
    assert extract(source, (1,)).validate().valid
    with pytest.raises(EditApplicationError, match="types and shapes"):
        source.apply_candidate(Candidate((ExtractNodes((1,), preserve_interface=True),)))


def captured_interface_model():
    children = [helper.make_node("Neg", ["mid"], ["inner"])]
    # A local result named y is safe: the parent If result is not captured.
    children.append(helper.make_node("Relu", ["inner"], ["y"]))
    body = helper.make_graph(children, "scope", [], [tensor_value("y")])
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["mid"]),
                helper.make_node("If", ["cond"], ["z"], then_branch=body, else_branch=body),
                helper.make_node("Identity", ["z"], ["y"]),
            ],
            "captured",
            [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
            [tensor_value("y")],
            value_info=[tensor_value("mid"), tensor_value("z")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def test_interface_aliases_rebind_free_captures_without_renaming_local_definitions():
    source = OnnxModel.from_proto(captured_interface_model())
    fragment = source.apply_candidate(Candidate((ExtractNodes((1,), preserve_interface=True),)))
    result = fragment.to_proto()
    assert fragment.input_names == ("x", "cond")
    assert fragment.output_names == ("y",)
    assert fragment.validate().valid
    node = result.graph.node[0]
    assert list(node.input) == ["cond"]
    for attribute in node.attribute:
        assert attribute.g.node[0].input[0] == "x"
        assert attribute.g.output[0].name == "y"
    assert source.to_proto().graph.node[1].attribute[0].g.node[0].input[0] == "mid"


def test_interface_aliases_decline_shadowed_capture_targets():
    body = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["y"],
                value=helper.make_tensor(
                    "local",
                    TensorProto.FLOAT,
                    [1],
                    [1.0],
                ),
            ),
            helper.make_node("Add", ["z", "y"], ["inner"]),
        ],
        "shadow",
        [],
        [tensor_value("inner")],
    )
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["mid"]),
                helper.make_node("Neg", ["mid"], ["z"]),
                helper.make_node("If", ["cond"], ["t"], then_branch=body, else_branch=body),
                helper.make_node("Identity", ["z"], ["y"]),
                helper.make_node("Identity", ["t"], ["q"]),
            ],
            "capture-shadow",
            [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
            [tensor_value("y"), tensor_value("q")],
            value_info=[tensor_value("mid"), tensor_value("z"), tensor_value("t")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    source = OnnxModel.from_proto(proto)
    assert source.validate().valid
    assert extract(source, (1, 2)).validate().valid
    with pytest.raises(EditApplicationError, match="shadow"):
        source.apply_candidate(Candidate((ExtractNodes((1, 2), preserve_interface=True),)))


def test_interface_cannot_alias_node_output_to_unused_input_passthrough():
    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["dead"],
                    value=helper.make_tensor(
                        "data",
                        TensorProto.FLOAT,
                        [1],
                        [1.0],
                    ),
                )
            ],
            "passthrough",
            [tensor_value("x")],
            [tensor_value("x")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    with pytest.raises(EditApplicationError, match="redefine"):
        OnnxModel.from_proto(proto).apply_candidate(
            Candidate((ExtractNodes((0,), preserve_interface=True),))
        )
