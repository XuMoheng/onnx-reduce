"""A fault carrier must stay reachable after aggressive graph isolation."""

import numpy as np
import onnx
import pytest

from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult as F
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.extract import ExtractNodesPass, ProbeFragmentsPass
from tests.helpers import chain_model, dynamic_expand_model


class LiveReluCheck:
    """Simulate a backend that prunes unreachable code before compiling it."""

    def run(self, path):
        graph = onnx.load(path).graph
        producers = {name: node for node in graph.node for name in node.output if name}
        pending = [v.name for v in graph.output]
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            node = producers.get(name)
            if node is not None:
                if node.op_type == "Relu":
                    return F.INTERESTING
                pending.extend(node.input)
        return F.NOT_INTERESTING


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
@pytest.mark.parametrize("mode", ["ir", "auto"])
@pytest.mark.parametrize("cache", [True, False])
def test_large_live_chain_reaches_one_node_with_bounded_checks(tmp_path, scheduler, mode, cache):
    source = OnnxModel.from_proto(chain_model(1500, 1140))
    reduced, report = ReductionEngine(
        config=EngineConfig(
            scheduler=scheduler,
            model_edit_mode=mode,
            cache_enabled=cache,
            max_candidate_check_calls=32,
        )
    ).run(source, LiveReluCheck(), tmp_path)
    assert reduced.op_types == ("Relu",)
    assert reduced.node_ids == (1140,)
    assert reduced.dead_node_indices() == ()
    assert reduced.validate().valid
    assert report.final.node_count == 1
    assert report.failure_check_calls < 16
    assert report.stop_reason == "fixed_point"
    assert report.artifact_recheck["outcome"] == "interesting"
    assert not report.artifact_recheck["cache_hit"]
    assert source.node_count == 1500


def test_full_extraction_covers_every_singleton_and_early_probes_are_bounded():
    model = OnnxModel.from_proto(chain_model(80, 40))
    full = list(ExtractNodesPass().generate(model))
    edits = [candidate.edits[0].indices for candidate in full]
    assert len(full) == len({candidate.edits for candidate in full})
    assert {(i,) for i in range(80)} <= set(edits)
    early = list(ProbeFragmentsPass().generate(model))
    assert len(early) <= ProbeFragmentsPass.limit
    assert early[0].edits[0].indices == (40,)
    assert early == full[: len(early)]


def test_many_operator_families_do_not_delay_partition_probes():
    proto = chain_model(80, 40)
    for index, node in enumerate(proto.graph.node):
        node.op_type = f"Custom{index}"
        node.domain = "test"
    proto.opset_import.append(onnx.helper.make_opsetid("test", 1))
    model = OnnxModel.from_proto(proto)
    early = list(ProbeFragmentsPass().generate(model))
    assert early[0].edits[0].indices == (0,)
    assert any(c.edits[0].indices == tuple(range(40)) for c in early[:4])


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
def test_single_node_witness_can_preserve_required_outer_interface(tmp_path, scheduler):
    class FixedInterface(LiveReluCheck):
        def run(self, path):
            graph = onnx.load(path).graph
            if [v.name for v in graph.input] != ["x"] or [v.name for v in graph.output] != ["y"]:
                return F.NOT_INTERESTING
            return super().run(path)

    source = OnnxModel.from_proto(chain_model(1500, 1140))
    reduced, report = ReductionEngine(
        config=EngineConfig(
            scheduler=scheduler,
            max_candidate_check_calls=32,
        )
    ).run(source, FixedInterface(), tmp_path)
    assert reduced.op_types == ("Relu",)
    assert reduced.input_names == ("x",)
    assert reduced.output_names == ("y",)
    assert report.failure_check_calls <= 16
    assert report.artifact_recheck["outcome"] == "interesting"


def test_multi_output_witness_can_project_only_the_relevant_output(tmp_path):
    from onnx import TensorProto, helper

    from tests.helpers import tensor_value

    class OneOutputSplit:
        def run(self, path):
            graph = onnx.load(path).graph
            split = next((node for node in graph.node if node.op_type == "Split"), None)
            keep = (
                split is not None
                and len(graph.output) == 1
                and (
                    graph.output[0].name == split.output[1]
                    or any(n.input and n.input[0] == split.output[1] for n in graph.node)
                )
            )
            return F.INTERESTING if keep else F.NOT_INTERESTING

    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["mid"]),
                helper.make_node("Split", ["mid"], ["a", "b"], axis=0),
                helper.make_node("Identity", ["b"], ["y"]),
            ],
            "split-witness",
            [tensor_value("x", TensorProto.FLOAT, [2])],
            [tensor_value("y")],
            value_info=[tensor_value("mid", shape=[2]), tensor_value("a"), tensor_value("b")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    reduced, report = ReductionEngine(
        [ProbeFragmentsPass(), ExtractNodesPass()],
        config=EngineConfig(max_candidate_check_calls=24),
    ).run(
        OnnxModel.from_proto(proto),
        OneOutputSplit(),
        tmp_path,
    )
    assert reduced.op_types == ("Split",)
    assert reduced.output_names == ("b",)
    assert report.artifact_recheck["outcome"] == "interesting"


def test_odd_partition_tail_precedes_deferred_interface_variants():
    proto = chain_model(5, 0)
    for node in proto.graph.node:
        node.op_type = "Relu"
    attempts = list(ExtractNodesPass().generate(OnnxModel.from_proto(proto)))
    tail = next(
        i
        for i, c in enumerate(attempts)
        if c.edits[0].indices == (4,) and not c.edits[0].preserve_interface
    )
    first_alias = next(i for i, c in enumerate(attempts) if c.edits[0].preserve_interface)
    assert tail < first_alias


class StaticExpandCheck:
    """Simulate a backend needing a live Expand with concrete tiling dimensions.

    This is a synthetic contract, not an emulation of Typhoon's tiling limits.
    The shape oracle is independent of the reducer's materialization code.
    """

    def run(self, path):
        graph = onnx.shape_inference.infer_shapes(onnx.load(path)).graph
        known = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
        for value in graph.input:
            known.pop(value.name, None)
        for node in graph.node:
            if node.op_type == "Constant":
                known[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
            elif node.op_type in {"ConstantOfShape", "Mul", "Equal", "Where"} and all(
                name in known for name in node.input
            ):
                args = [known[name] for name in node.input]
                if node.op_type == "ConstantOfShape":
                    result = np.ones(tuple(args[0]), dtype=np.int64)
                elif node.op_type == "Mul":
                    result = args[0] * args[1]
                elif node.op_type == "Equal":
                    result = args[0] == args[1]
                else:
                    result = np.where(*args)
                known[node.output[0]] = result
        producers = {name: node for node in graph.node for name in node.output if name}
        pending = [v.name for v in graph.output]
        live = set()
        while pending:
            name = pending.pop()
            if name in live:
                continue
            live.add(name)
            if name in producers:
                pending.extend(producers[name].input)
        infos = {v.name: v for v in (*graph.input, *graph.value_info, *graph.output)}
        for node in graph.node:
            if node.op_type != "Expand" or node.output[0] not in live:
                continue
            shape = known.get(node.input[1])
            data = infos.get(node.input[0])
            if (
                shape is not None
                and data is not None
                and shape.tolist() == [1, 6, 3, 2, 2, 12, 32, 32]
                and [d.dim_value for d in data.type.tensor_type.shape.dim]
                == [1, 6, 1, 2, 2, 12, 32, 32]
            ):
                return F.INTERESTING
        return F.NOT_INTERESTING


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
@pytest.mark.parametrize("mode", ["ir", "auto"])
@pytest.mark.parametrize("cache", [True, False])
def test_1573_nodes_with_computed_expand_shape_reach_one_node(tmp_path, scheduler, mode, cache):
    source = OnnxModel.from_proto(dynamic_expand_model(prefix=252, tail=1312))
    assert source.node_count == 1573
    assert not source.to_proto().graph.value_info
    assert source.op_types[260] == "Expand"
    reduced, report = ReductionEngine(
        config=EngineConfig(
            scheduler=scheduler,
            model_edit_mode=mode,
            cache_enabled=cache,
            # Uncached final coverage executes repeated candidates again.
            max_candidate_check_calls=64 if cache else 128,
        )
    ).run(source, StaticExpandCheck(), tmp_path)
    assert reduced.op_types == ("Expand",)
    assert reduced.node_ids == (260,)
    assert reduced.input_names == ("expanded_input",)
    assert report.stop_reason == "fixed_point"
    assert report.artifact_recheck["outcome"] == "interesting"
    assert not report.artifact_recheck["cache_hit"]
    assert source.node_count == 1573
