from __future__ import annotations

import pytest

from onnx_reduce.engine import EngineConfig, InitialNotInteresting, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.model import OnnxModel
from tests.helpers import chain_model


class ReluFailureCheck:
    def run(self, model_path) -> FailureCheckResult:
        model = OnnxModel.load(model_path)
        if "Relu" in model.op_types:
            return FailureCheckResult.INTERESTING
        return FailureCheckResult.NOT_INTERESTING


class NeverInteresting:
    def run(self, model_path) -> FailureCheckResult:
        return FailureCheckResult.NOT_INTERESTING


def reduce_with_runner(model, failure_check_runner, workdir, **config):
    return ReductionEngine(config=EngineConfig(**config)).run(model, failure_check_runner, workdir)


def test_engine_reduces_chain_to_trigger_node(tmp_path):
    model = OnnxModel.from_proto(chain_model(node_count=8, trigger_index=3))
    reduced, report = reduce_with_runner(model, ReluFailureCheck(), tmp_path)
    assert reduced.op_types == ("Relu",)
    assert reduced.validate("strict").valid
    assert report.stop_reason == "fixed_point"
    assert report.original.node_count == 8
    assert report.final.node_count == 1
    assert report.final_hash == reduced.canonical_hash()
    assert report.failure_check_calls > 2
    assert report.candidates_generated > report.candidates_accepted


def test_initial_model_must_be_interesting(tmp_path):
    model = OnnxModel.from_proto(chain_model())
    with pytest.raises(InitialNotInteresting):
        reduce_with_runner(model, NeverInteresting(), tmp_path)


def test_max_candidate_check_calls_budget_is_reported(tmp_path):
    model = OnnxModel.from_proto(chain_model(node_count=8, trigger_index=3))
    reduced, report = reduce_with_runner(
        model, ReluFailureCheck(), tmp_path, max_candidate_check_calls=1
    )
    assert report.stop_reason == "budget_exhausted"
    assert report.final.node_count == 1
    assert report.failure_check_calls == 3
    assert "Relu" in reduced.op_types


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
def test_large_extent_chain_is_cut_before_spending_the_budget_on_dimensions(tmp_path, scheduler):
    proto = chain_model(node_count=512, trigger_index=341)
    for value in (*proto.graph.input, *proto.graph.output):
        value.type.tensor_type.shape.dim[0].dim_value = 256
    reduced, report = reduce_with_runner(
        OnnxModel.from_proto(proto),
        ReluFailureCheck(),
        tmp_path,
        scheduler=scheduler,
        max_candidate_check_calls=1,
    )
    assert reduced.op_types == ("Relu",)
    assert report.final.tensor_elements == 256
    assert report.failure_check_calls == 3
    assert report.artifact_recheck["outcome"] == "interesting"


def test_report_serializes_all_contract_counters(tmp_path):
    model = OnnxModel.from_proto(chain_model(node_count=5, trigger_index=2))
    _, report = reduce_with_runner(model, ReluFailureCheck(), tmp_path)
    payload = report.to_dict()
    assert payload["stop_reason"] == "fixed_point"
    assert payload["failure_checks"]["calls"] == report.failure_check_calls
    assert payload["candidates"]["generated"] == report.candidates_generated
    assert set(payload["passes"]) == {
        "compact_fragments",
        "probe_fragments",
        "extract_nodes",
        "drop_outputs",
        "remove_unused",
        "delete_nodes",
        "bypass_nodes",
        "simplify_attributes",
        "shrink_tensors",
        "reduce_subgraphs",
        "combinations",
    }
