"""Batch acceptance, rejected groups and non-monotonic cleanup conditions."""

import pytest
from onnx import TensorProto, helper

from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes import DropOutputsPass, RemoveUnusedPass
from onnx_reduce.passes.groups import deletion_groups
from tests.helpers import tensor_value


def resource_model(kind, size):
    inputs = [tensor_value("x")]
    outputs = [tensor_value("y")]
    weights = []
    names = [f"resource_{i}" for i in range(size)]
    if kind == "inputs":
        inputs.extend(tensor_value(name) for name in names)
    elif kind == "initializers":
        weights = [helper.make_tensor(name, TensorProto.FLOAT, [1], [1]) for name in names]
    else:
        outputs.extend(tensor_value(name) for name in names)
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    if kind == "outputs":
        nodes.extend(helper.make_node("Identity", ["x"], [name]) for name in names)
    graph = helper.make_graph(nodes, "cleanup", inputs, outputs, weights)
    return OnnxModel.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    )


def resources(model, kind):
    if kind == "inputs":
        return set(model.unused_input_names())
    if kind == "initializers":
        return set(model.unused_initializer_names())
    return set(model.output_names)


class ResourceCheck:
    def __init__(self, kind, predicate, rejection=FailureCheckResult.NOT_INTERESTING):
        self.kind = kind
        self.predicate = predicate
        self.rejection = rejection

    def run(self, path):
        names = resources(OnnxModel.load(path), self.kind)
        return FailureCheckResult.INTERESTING if self.predicate(names) else self.rejection


@pytest.mark.parametrize("size", range(1, 18))
@pytest.mark.parametrize("keep_one", [False, True])
def test_groups_cover_every_singleton_without_duplicates(size, keep_one):
    items = tuple(range(size))
    groups = list(deletion_groups(items, keep_one=keep_one))
    assert len(groups) == len(set(groups))
    if keep_one and size == 1:
        assert groups == []
        return
    assert {g[0] for g in groups if len(g) == 1} == set(items)
    assert all(0 < len(g) <= size - int(keep_one) for g in groups)


@pytest.mark.parametrize("kind", ["inputs", "initializers", "outputs"])
def test_all_resources_are_removed_in_one_batch(tmp_path, kind):
    passes = [DropOutputsPass()] if kind == "outputs" else [RemoveUnusedPass()]
    reduced, report = ReductionEngine(passes).run(
        resource_model(kind, 512), ResourceCheck(kind, lambda names: True), tmp_path
    )
    assert resources(reduced, kind) == ({"y"} if kind == "outputs" else set())
    assert report.candidates_accepted == 1
    assert report.failure_check_calls == 3
    assert reduced.validate().valid


@pytest.mark.parametrize("kind", ["inputs", "initializers", "outputs"])
def test_rejected_groups_fall_back_to_smaller_selections(tmp_path, kind):
    passes = [DropOutputsPass()] if kind == "outputs" else [RemoveUnusedPass()]
    reduced, report = ReductionEngine(passes).run(
        resource_model(kind, 8),
        ResourceCheck(kind, lambda names: "resource_3" in names),
        tmp_path,
    )
    assert resources(reduced, kind) == {"resource_3"}
    assert report.stop_reason == "fixed_point"
    assert reduced.validate().valid


@pytest.mark.parametrize("kind", ["inputs", "initializers", "outputs"])
def test_combination_can_succeed_when_single_deletion_cannot(tmp_path, kind):
    # Only deleting both resources preserves this synthetic non-monotonic bug.
    passes = [DropOutputsPass()] if kind == "outputs" else [RemoveUnusedPass()]
    reduced, report = ReductionEngine(passes).run(
        resource_model(kind, 2),
        ResourceCheck(kind, lambda names: ("resource_0" in names) == ("resource_1" in names)),
        tmp_path,
    )
    assert resources(reduced, kind) == ({"y"} if kind == "outputs" else set())
    assert report.candidates_accepted == 1


@pytest.mark.parametrize("kind", ["inputs", "initializers", "outputs"])
def test_inconclusive_batches_still_try_singletons_and_prevent_minimality(tmp_path, kind):
    model = resource_model(kind, 4)
    passes = [DropOutputsPass()] if kind == "outputs" else [RemoveUnusedPass()]
    original = resources(model, kind)
    reduced, report = ReductionEngine(passes, config=EngineConfig(collect_performance=True)).run(
        model,
        ResourceCheck(kind, lambda names: names == original, FailureCheckResult.INCONCLUSIVE),
        tmp_path,
    )
    assert reduced.canonical_hash() == model.canonical_hash()
    assert report.candidates_generated >= len(original)
    assert report.minimality["status"] == "unproven"
    assert report.performance.samples["candidate_selected_items"]["max"] > 1
