"""Differential tests use the original IR editing path as the oracle."""

import json
from pathlib import Path

import onnx
import pytest
from onnx import helper

from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs, DropOutputs
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.model import OnnxModel
from onnx_reduce.performance import collect_performance
from onnx_reduce.preparation import preparation_caches
from onnx_reduce.storage import checkpoint_artifact
from onnx_reduce.types import EditApplicationError, EditOptions
from tests.helpers import (
    chain_model,
    control_flow_model,
    custom_domain_model,
    external_data_model,
    qdq_model,
    tensor_value,
)
from tests.runtime.test_engine import ReluFailureCheck


def configured_model(proto, mode):
    return OnnxModel.from_proto(proto).with_edit_options(EditOptions(mode))


def typed_chain():
    p = chain_model(7, 3)
    p.graph.value_info.extend(tensor_value(f"v{i}") for i in range(6))
    return p


def metadata_chain():
    p = typed_chain()
    p.graph.value_info[0].metadata_props.add(key="boundary_tag", value="kept?")
    p.graph.node[2].metadata_props.add(key="onnx_reduce.module", value='["a", "b"]')
    return p


@pytest.mark.parametrize(
    "fixture",
    [typed_chain, chain_model, metadata_chain, qdq_model, custom_domain_model, control_flow_model],
)
@pytest.mark.parametrize("indices", [(), (0,), (1,), (2, 0), (0, 0, 1), (0, 1, 2, 3, 4, 5, 6)])
def test_fast_and_ir_paths_match_bytes_queries_and_followup(fixture, indices):
    p = fixture()
    indices = tuple(i for i in indices if i < len(p.graph.node))
    models = []
    for mode in ("ir", "auto"):
        with preparation_caches():
            base = configured_model(p, mode)
            before = base.canonical_hash()
            model = base.apply_candidate(Candidate((DeleteNodes(indices),)))
            state = (
                model._payload,
                model.cost(),
                model.node_ids,
                dict(model.id_to_index),
                model.input_names,
                model.output_names,
                model.op_types,
                model.tensor_index(),
                model.dead_node_indices(),
                model.unused_input_names(),
                model.unused_initializer_names(),
                model.region_graph(),
                tuple(model.validate(m) for m in ("strict", "structural", "permissive")),
            )
            follow = (
                model.apply_candidate(Candidate((DeleteNodes((0,)),)))
                if model.node_count
                else model
            )
            assert before == base.canonical_hash()
            models.append((state, follow._payload))
    assert models[0] == models[1]


def test_snapshot_stays_lazy_and_public_proto_is_independent():
    with preparation_caches(), collect_performance(True) as stats:
        parent = configured_model(typed_chain(), "auto")
        model = parent.apply_candidate(Candidate((DeleteNodes((2,)),)))
        assert model._ir_model is None
        model.cost(), model.region_graph(), model.tensor_index(), model.dead_node_indices()
        model.unused_input_names(), model.unused_initializer_names(), model.validate()
        assert model._ir_model is None
        assert stats.counters["lazy_ir_imports"] == 0
        p = model.to_proto()
        p.graph.ClearField("node")
        assert model.node_count == 6
        assert model.copy().node_ids == model.node_ids
        assert stats.counters["lazy_ir_imports"] == 1


@pytest.mark.parametrize("kind", ["metadata", "optional", "unordered", "sequence", "subgraph"])
def test_unsupported_capabilities_really_fall_back(kind):
    p = metadata_chain() if kind == "metadata" else typed_chain()
    if kind == "optional":
        p.graph.node[0].input.append("")
    elif kind == "unordered":
        nodes = list(p.graph.node)
        del p.graph.node[:]
        p.graph.node.extend(reversed(nodes))
    elif kind == "sequence":
        p.graph.input[0].type.CopyFrom(
            helper.make_sequence_type_proto(
                helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1])
            )
        )
    elif kind == "subgraph":
        p = control_flow_model()
    if kind == "unordered":
        # The existing IR cloner itself rejects forward references. Auto must
        # decline, then preserve the same failure rather than silently reorder.
        for mode in ("ir", "auto"):
            with preparation_caches(), pytest.raises(RuntimeError):
                configured_model(p, mode).apply_candidate(Candidate((DeleteNodes((0,)),)))
        assert not configured_model(p, mode)._fast_capability
        return
    with preparation_caches():
        result = configured_model(p, "auto").apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert result._ir_model is not None


def test_mixed_edits_use_current_state_and_do_not_change_parent():
    p = typed_chain()
    p.graph.input.append(tensor_value("unused"))
    p.graph.output.append(tensor_value("v1"))
    results = []
    for mode in ("ir", "auto"):
        with preparation_caches():
            base = configured_model(p, mode)
            changed = base.apply_candidate(
                Candidate(
                    (
                        DropOutputs(("v1",)),
                        DeleteNodes((0,)),
                        DropInputs(("unused",)),
                        DeleteNodes((0,)),
                    )
                )
            )
            results.append((changed._payload, changed.node_ids))
            assert base.node_count == 7
    assert results[0] == results[1]


def test_external_weights_and_symbolic_shape_are_byte_identical(tmp_path):
    source = tmp_path / "external.onnx"
    onnx.save_model(
        external_data_model(),
        source,
        save_as_external_data=True,
        location="weights.bin",
        size_threshold=0,
    )
    for p in (OnnxModel.load(source).to_proto(), typed_chain()):
        if len(p.graph.node) == 7:
            for info in (*p.graph.input, *p.graph.output, *p.graph.value_info):
                info.type.tensor_type.shape.dim[0].dim_param = "batch"
        results = []
        for mode in ("ir", "auto"):
            with preparation_caches():
                model = configured_model(p, mode).apply_candidate(Candidate((DeleteNodes((0,)),)))
                results.append(
                    (model._payload, model.cost(), model.validate(), model.region_graph())
                )
        assert results[0] == results[1]


def test_inference_failure_keeps_partial_type_and_same_rejection(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("inference unavailable")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail)
    results = []
    for mode in ("ir", "auto"):
        with preparation_caches():
            model = configured_model(chain_model(), mode).apply_candidate(
                Candidate((DeleteNodes((0,)),))
            )
            results.append((model._payload, model.validate()))
    assert results[0] == results[1]
    assert not results[0][1].valid


def test_multi_output_boundaries_and_initializer_input_aliases():
    branching = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["x"], ["r"]),
                helper.make_node("Identity", ["r"], ["a"]),
                helper.make_node("Identity", ["r"], ["b"]),
                helper.make_node("Add", ["a", "b"], ["y"]),
            ],
            "branches",
            [tensor_value("x")],
            [tensor_value("y"), tensor_value("b")],
            value_info=[tensor_value("r"), tensor_value("a")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    alias = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "w"], ["a"]),
                helper.make_node("Relu", ["a"], ["y"]),
            ],
            "alias",
            [tensor_value("x"), tensor_value("w")],
            [tensor_value("y")],
            initializer=[helper.make_tensor("w", onnx.TensorProto.FLOAT, [1], [1.0])],
            value_info=[tensor_value("a")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    for p, indices in ((branching, (2, 0)), (alias, (0,))):
        results = []
        for mode in ("ir", "auto"):
            with preparation_caches():
                model = configured_model(p, mode).apply_candidate(
                    Candidate((DeleteNodes(indices),))
                )
                results.append(
                    (
                        model._payload,
                        model.tensor_index(),
                        model.region_graph(),
                        model.unused_input_names(),
                        model.unused_initializer_names(),
                    )
                )
        assert results[0] == results[1]


@pytest.mark.parametrize("indices", [(-1,), (999,)])
def test_fast_errors_match_ir_including_stale_base(indices):
    for mode in ("ir", "auto"):
        with preparation_caches():
            model = configured_model(typed_chain(), mode)
            with pytest.raises(
                EditApplicationError, match=f"node index {indices[0]} is out of range"
            ):
                model.apply_candidate(Candidate((DeleteNodes(indices),)))
            with pytest.raises(EditApplicationError, match="base digest"):
                model.apply_candidate(Candidate((DeleteNodes((0,)),), base_digest="stale"))


@pytest.mark.parametrize("enabled", [True, False])
def test_validation_cache_and_independent_readback(tmp_path, enabled):
    with preparation_caches(enabled), collect_performance(True) as stats:
        model = OnnxModel.from_proto(typed_chain())
        assert model.validate().valid
        assert model.validate().valid
        assert stats.counters["checker_calls"] == (1 if enabled else 2)
        artifact = model.materialize(tmp_path / "model.onnx")
        assert model.validate_artifact(artifact, "strict").valid
        assert stats.counters["checker_calls"] == (2 if enabled else 3)
        artifact.model_path.write_bytes(b"corrupt")
        assert not model.validate_artifact(artifact, "strict").valid


def test_checkpoint_reuse_verifies_private_file_and_repairs_damage(tmp_path):
    model = OnnxModel.from_proto(typed_chain())
    known = {}
    destination = tmp_path / "checkpoint.onnx"
    with collect_performance(True) as stats:
        first = checkpoint_artifact(model, destination, known)
        inode = destination.stat().st_ino
        assert checkpoint_artifact(model, destination, known) == first
        assert destination.stat().st_ino == inode
        assert stats.counters["checkpoint_reuses"] == 1
        destination.write_bytes(b"corrupt")
        checkpoint_artifact(model, destination, known)
        assert destination.read_bytes() == model._payload
        destination.unlink()
        checkpoint_artifact(model, destination, known)
        assert destination.read_bytes() == model._payload
        assert stats.counters["checkpoint_reuse_fallbacks"] == 2


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
@pytest.mark.parametrize("budget", [0, 4, None])
def test_edit_mode_preserves_engine_decisions_and_fixed_point(tmp_path, scheduler, budget):
    results = []
    for mode in ("ir", "auto"):
        _, report = ReductionEngine(
            config=EngineConfig(
                scheduler=scheduler,
                model_edit_mode=mode,
                max_candidate_check_calls=budget,
            )
        ).run(OnnxModel.from_proto(typed_chain()), ReluFailureCheck(), tmp_path / mode)
        path = Path(report.work_directory) / "candidates.jsonl"
        results.append(
            (
                path.read_bytes() if path.exists() else b"",
                report.final_hash,
                report.final,
                report.failure_check_calls,
                report.stop_reason,
                report.minimality,
            )
        )
        assert (
            json.loads((Path(report.work_directory) / "task.json").read_text())["engine"][
                "model_edit_mode"
            ]
            == mode
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_edit_policy_survives_cache_context_and_followup_edits(enabled):
    parent = OnnxModel.from_proto(typed_chain())
    auto = parent.with_edit_options(EditOptions("auto"))
    candidate = Candidate((DeleteNodes((2,)),))
    with preparation_caches(enabled):
        changed = auto.apply_candidate(candidate)
    assert changed._ir_model is None
    assert changed.edit_options == EditOptions("auto")
    # Leaving the cache scope must not silently change the editor on the result.
    followup = changed.apply_candidate(candidate)
    assert followup._ir_model is None
    assert parent.edit_options == EditOptions("ir")
    assert parent.apply_candidate(candidate)._ir_model is not None
    assert parent.node_count == 7
