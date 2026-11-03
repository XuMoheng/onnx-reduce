"""Read-only regions, reliable annotations, capture links and topology fallback."""

import json

import pytest
from onnx import TensorProto, helper

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.finalizer import iter_coverage
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.performance import collect_performance
from onnx_reduce.regions import NodeRegion, graph_regions
from tests.helpers import chain_model, tensor_value


def annotated_chain():
    proto = chain_model(8, 4)
    for index, node in enumerate(proto.graph.node):
        prop = node.metadata_props.add()
        prop.key = "onnx_reduce.module"
        # Noncontiguous module membership must survive sorting and deletion.
        prop.value = json.dumps(["encoder", "even" if index % 2 == 0 else "odd"])
    return proto


def test_explicit_module_hierarchy_and_noncontiguous_membership():
    model = OnnxModel.from_proto(annotated_chain())
    digest = model.canonical_hash()
    before = model.to_proto().SerializeToString(deterministic=True)
    regions = graph_regions(model)
    modules = [region for region in regions if region.source == "module"]
    assert {region.node_ids for region in modules} == {
        tuple(range(8)),
        (0, 2, 4, 6),
        (1, 3, 5, 7),
    }
    assert modules[0].source == "module"
    assert any(region.parent == ("encoder",) for region in modules)
    assert model.canonical_hash() == digest
    assert model.to_proto().SerializeToString(deterministic=True) == before


def test_malformed_annotation_and_node_names_do_not_invent_module_groups():
    proto = chain_model(6, 3)
    for index, node in enumerate(proto.graph.node):
        node.name = f"encoder/block{index}/op"
        prop = node.metadata_props.add()
        prop.key = "onnx_reduce.module"
        prop.value = ["broken", '"string"', "[]", "[null]", '[""]', "[1]"][index]
    regions = graph_regions(OnnxModel.from_proto(proto))
    assert not any(region.source == "module" for region in regions)
    assert any(region.source == "topology" for region in regions)


def test_annotations_and_graph_summary_refresh_on_each_accepted_base():
    model = OnnxModel.from_proto(annotated_chain())
    original = model.region_graph()
    changed = model.apply_candidate(Candidate((DeleteNodes((0, 3, 6)),)))
    assert model.region_graph() is original
    assert changed.region_graph().node_ids == (1, 2, 4, 5, 7)
    assert {node_id for node_id, _ in changed.region_graph().modules} == {1, 2, 4, 5, 7}
    assert {region.node_ids for region in graph_regions(changed) if region.source == "module"} == {
        (1, 2, 4, 5, 7),
        (2, 4),
        (1, 5, 7),
    }
    assert changed.copy().region_graph() == changed.region_graph()


def test_optional_graph_capability_falls_back_to_all_standard_topology_regions():
    class IdsOnly:
        node_ids = (20, 3, 80, 7, 19)

        def __init__(self):
            self.id_to_index = {node_id: index for index, node_id in enumerate(self.node_ids)}

    regions = graph_regions(IdsOnly())
    assert all(region.source == "topology" for region in regions)
    assert {region.node_ids for region in regions} == {
        (20, 3),
        (80, 7),
        (19,),
        (20,),
        (3,),
        (80,),
        (7,),
    }


def test_chain_regions_split_at_forks_and_joins():
    nodes = [
        helper.make_node("Identity", ["x"], ["root"]),
        helper.make_node("Identity", ["root"], ["a"]),
        helper.make_node("Identity", ["a"], ["aa"]),
        helper.make_node("Identity", ["root"], ["b"]),
        helper.make_node("Identity", ["b"], ["bb"]),
        helper.make_node("Add", ["aa", "bb"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "fork", [tensor_value("x")], [tensor_value("y")])
    model = OnnxModel.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    )
    regions = [region for region in graph_regions(model) if region.source == "structure"]
    assert {region.node_ids for region in regions} == {(1, 2), (3, 4)}
    assert all(region.boundary_count == 2 for region in regions)


def capture_model():
    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["outer"], ["then_y"])],
        "then",
        [],
        [tensor_value("then_y")],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["outer"], ["else_y"])],
        "else",
        [],
        [tensor_value("else_y")],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["outer"]),
            helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph),
        ],
        "capture",
        [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, ())],
        [tensor_value("y")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_outer_capture_is_a_main_graph_dependency_and_boundary():
    model = OnnxModel.from_proto(capture_model())
    summary = model.region_graph()
    assert any(value.producer == 0 and value.consumers == (1,) for value in summary.values)
    assert any(
        region.source == "structure" and region.node_ids == (0, 1)
        for region in graph_regions(model)
    )
    producer = next(
        region
        for region in graph_regions(model)
        if region.source == "topology" and region.node_ids == (0,)
    )
    assert producer.boundary_count == 2
    assert producer.incomplete_boundary_count >= 1


def test_missing_types_do_not_filter_regions_or_trigger_shape_inference():
    model = OnnxModel.from_proto(chain_model(8, 4))
    with collect_performance(True) as stats:
        regions = graph_regions(model)
    assert stats.stages.get("shape_inference", {}).get("calls", 0) == 0
    assert {region.node_ids for region in regions if len(region.node_ids) == 1} == {
        (node_id,) for node_id in model.node_ids
    }
    assert any(region.incomplete_boundary_count > 0 for region in regions)


def test_final_scope_includes_module_candidates_even_with_incomplete_boundaries():
    model = OnnxModel.from_proto(annotated_chain())
    attempts = list(
        iter_coverage([DeleteNodesPass()], model, 0, model.canonical_hash(), include_regions=True)
    )
    assert any(
        attempt.region_source == "module"
        and attempt.region_ids == (0, 2, 4, 6)
        and attempt.direction == "delete"
        for attempt in attempts
    )
    assert any(
        attempt.region_source == "module"
        and attempt.region_ids == (0, 2, 4, 6)
        and attempt.direction == "keep"
        for attempt in attempts
    )


def test_regions_bind_to_current_indices_after_deletion():
    model = OnnxModel.from_proto(annotated_chain())
    changed = model.apply_candidate(Candidate((DeleteNodes((0, 3)),)))
    candidate = NodeRegion((0, 2, 4, 6)).candidate(changed, "delete", changed.canonical_hash())
    assert candidate.edits[0].indices == (1, 2, 4)
    kept = NodeRegion((0, 2, 4, 6)).candidate(changed, "keep", changed.canonical_hash())
    assert kept.edits[0].indices == (0, 3, 5)
    assert kept.base_digest == changed.canonical_hash()


@pytest.mark.parametrize("size", [0, 1])
def test_zero_and_one_node_regions(size):
    if size:
        model = OnnxModel.from_proto(chain_model(1, 0))
    else:
        graph = helper.make_graph([], "empty", [tensor_value("x")], [tensor_value("x")])
        model = OnnxModel.from_proto(helper.make_model(graph))
    regions = graph_regions(model)
    assert len(regions) == size
    assert model.validate().valid
