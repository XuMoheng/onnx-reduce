from __future__ import annotations

from onnx import TensorProto, helper

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropOutputs
from onnx_reduce.model import OnnxModel
from tests.helpers import chain_model, custom_domain_model


def load_chain() -> OnnxModel:
    return OnnxModel.from_proto(chain_model())


def test_tensor_index_has_producer_and_consumers():
    model = load_chain()
    links = model.tensor_index()
    assert links["v0"].producer == 0
    assert links["v0"].consumers == (1,)
    assert links["x"].is_graph_input
    assert links["y"].is_graph_output


def test_delete_nodes_promotes_dangling_output_and_passes_checker():
    model = load_chain()
    reduced = model.apply_candidate(Candidate((DeleteNodes((2,)),)))
    validation = reduced.validate("strict")
    assert validation.valid, validation.reason
    assert reduced.op_types == ("Identity",) * 5
    assert "v2" in reduced.input_names


def test_delete_all_nodes_makes_output_a_graph_input():
    model = load_chain()
    reduced = model.apply_candidate(Candidate((DeleteNodes((0, 1, 2, 3, 4, 5)),)))
    validation = reduced.validate("strict")
    assert validation.valid, validation.reason
    assert reduced.node_count == 0
    assert reduced.input_names[-1] == "y"
    assert reduced.output_names == ("y",)


def test_drop_output_preserves_producers_until_separate_cleanup():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    first = helper.make_tensor_value_info("first", TensorProto.FLOAT, [1])
    second = helper.make_tensor_value_info("second", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Identity", ["r"], ["first"]),
        helper.make_node("Identity", ["x"], ["second"]),
    ]
    graph = helper.make_graph(nodes, "two_outputs", [x], [first, second])
    model = OnnxModel.from_proto(helper.make_model(graph))
    reduced = model.apply_candidate(Candidate((DropOutputs(("second",)),)))
    assert reduced.validate("strict").valid
    assert reduced.output_names == ("first",)
    assert reduced.op_types == ("Relu", "Identity", "Identity")


def test_remove_unused_identifies_dead_branch_only():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    dead_weight = helper.make_tensor("dead_weight", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Identity", ["r"], ["y"]),
        helper.make_node("Add", ["x", "dead_weight"], ["unused"]),
    ]
    graph = helper.make_graph(nodes, "dead", [x], [y], [dead_weight])
    model = OnnxModel.from_proto(helper.make_model(graph))
    assert model.dead_node_indices() == (2,)
    reduced = model.apply_candidate(Candidate((DeleteNodes((2,)),)))
    assert reduced.validate("strict").valid
    assert reduced.op_types == ("Relu", "Identity")
    assert "dead_weight" in reduced.unused_initializer_names()
    cleaned = reduced.apply_candidate(Candidate((DropInitializers(("dead_weight",)),)))
    assert "dead_weight" not in str(cleaned.to_proto())


def test_custom_domain_downgrades_strict_to_structural():
    model = OnnxModel.from_proto(custom_domain_model())
    result = model.validate("strict")
    assert result.valid, result.reason
    assert result.validation_mode == "structural"
    assert result.custom_domains == ("com.example",)


def test_canonical_hash_is_deterministic_but_sensitive():
    model = load_chain()
    assert model.canonical_hash() == model.copy().canonical_hash()
    reduced = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert reduced.canonical_hash() != model.canonical_hash()
