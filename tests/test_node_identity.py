"""Node identity survives clones, intermediate edits and actual reordering."""

from __future__ import annotations

import pytest
from onnx import helper
from onnxscript import ir

from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs, DropOutputs
from onnx_reduce.model import OnnxModel
from onnx_reduce.scheduler import NodeRegion
from tests.helpers import chain_model, control_flow_model, tensor_value


def test_clone_and_composite_edits_preserve_lineage_and_parent_bytes():
    model = OnnxModel.from_proto(chain_model(7, 3))
    payload = model.to_proto().SerializeToString(deterministic=True)
    digest, cost = model.canonical_hash(), model.cost()
    clone = model.copy()
    assert clone.node_ids == model.node_ids == tuple(range(7))
    assert all(a is not b for a, b in zip(model._model.graph, clone._model.graph, strict=True))
    edits = (DeleteNodes((0,)), DeleteNodes((0, 2)), DeleteNodes((1,)))
    reduced = model.apply_candidate(Candidate(edits, base_digest=digest))
    sequential = model
    for edit in edits:
        sequential = sequential.apply_candidate(Candidate((edit,)))
    assert reduced.node_ids == sequential.node_ids == (2, 5, 6)
    assert dict(reduced.id_to_index) == {2: 0, 5: 1, 6: 2}
    assert len(reduced._node_ids) == reduced.node_count
    assert reduced.canonical_hash() == sequential.canonical_hash()
    assert reduced.validate().valid
    assert model.node_ids == tuple(range(7))
    assert model.canonical_hash() == digest
    assert model.cost() == cost
    assert model.to_proto().SerializeToString(deterministic=True) == payload
    assert clone.canonical_hash() == digest
    with pytest.raises(TypeError):
        reduced.id_to_index[2] = 10


@pytest.mark.parametrize("name", ["", "same"])
def test_unnamed_and_duplicate_nodes_have_distinct_ids_after_sort(name, monkeypatch):
    # Independent nodes admit multiple valid topologies. Exercise a sorter
    # choosing another one, rather than relying on today's tie-breaking rule.
    original_sort = ir.Graph.sort

    def reorder(graph):
        original_sort(graph)
        graph.extend(reversed(list(graph)))

    monkeypatch.setattr(ir.Graph, "sort", reorder)
    nodes = [
        helper.make_node("Identity", ["x"], ["a"], name=name),
        helper.make_node("Identity", ["x"], ["y"], name=name),
        helper.make_node("Identity", ["x"], ["dead"], name=name),
    ]
    model = OnnxModel.from_proto(
        helper.make_model(
            helper.make_graph(
                nodes,
                "independent",
                [tensor_value("x")],
                [tensor_value("y")],
                value_info=[tensor_value("a")],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )
    reduced = model.apply_candidate(Candidate((DeleteNodes((2,)),)))
    assert reduced.node_ids == (1, 0)
    assert dict(reduced.id_to_index) == {1: 0, 0: 1}
    assert reduced.validate().valid
    selected = reduced.apply_candidate(
        NodeRegion((0,)).candidate(reduced, "delete", reduced.canonical_hash())
    )
    assert selected.node_ids == (1,)
    assert selected.validate().valid
    assert model.node_ids == (0, 1, 2)


def test_region_rebinds_survivors_and_current_complement():
    model = OnnxModel.from_proto(chain_model(6, 3))
    region = NodeRegion((1, 3, 4))
    reduced = model.apply_candidate(Candidate((DeleteNodes((0, 3)),)))
    assert reduced.node_ids == (1, 2, 4, 5)
    delete = region.candidate(reduced, "delete", reduced.canonical_hash())
    keep = region.candidate(reduced, "keep", reduced.canonical_hash())
    assert delete.edits == (DeleteNodes((0, 2)),)
    assert keep.edits == (DeleteNodes((1, 3)),)
    assert reduced.apply_candidate(keep).node_ids == (1, 4)
    assert NodeRegion((0, 3)).candidate(reduced, "delete", reduced.canonical_hash()) is None
    assert NodeRegion(reduced.node_ids).candidate(reduced, "keep", reduced.canonical_hash()) is None


def test_resource_edits_do_not_rename_nodes_or_change_serialized_identity():
    proto = chain_model(3, 1)
    proto.graph.input.append(tensor_value("unused"))
    proto.graph.output.append(tensor_value("v0"))
    model = OnnxModel.from_proto(proto)
    reduced = model.apply_candidate(Candidate((DropInputs(("unused",)), DropOutputs(("v0",)))))
    assert reduced.node_ids == model.node_ids
    reloaded = OnnxModel.from_proto(reduced.to_proto())
    assert reloaded.canonical_hash() == reduced.canonical_hash()
    assert reloaded.cost() == reduced.cost()
    assert not reduced.to_proto().metadata_props
    assert [node.name for node in reduced.to_proto().graph.node] == ["n0", "n1", "n2"]


def test_subgraph_nodes_do_not_enter_main_graph_identity():
    model = OnnxModel.from_proto(control_flow_model())
    assert model.node_ids == (0,)
    assert model.copy().node_ids == (0,)
    reduced = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert reduced.node_ids == ()
    assert dict(reduced.id_to_index) == {}
    assert len(reduced._node_ids) == 0
    assert reduced.validate().valid
