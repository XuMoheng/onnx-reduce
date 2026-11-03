"""Preparation reuse preserves decisions, ownership and cache boundaries."""

import json
from pathlib import Path

import onnx
import pytest

from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs
from onnx_reduce.engine import EngineConfig, InitialNotInteresting, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.model import OnnxModel
from onnx_reduce.preparation import (
    MetadataCache,
    current_preparation,
    edit_key,
    metadata_key,
    preparation_caches,
)
from tests.helpers import chain_model
from tests.test_engine import NeverInteresting, ReluFailureCheck
from tests.test_invariants import RejectCandidates


def test_lru_has_byte_and_entry_bounds_and_replacement_accounting():
    cache = MetadataCache("test", max_entries=2, max_bytes=8)
    cache.put("a", b"123")
    cache.put("b", b"456")
    assert cache.byte_size == 8
    assert cache.get("a") == b"123"  # b is now the least recently used.
    cache.put("c", b"78")
    assert cache.get("b") is None
    assert cache.byte_size == 7
    cache.put("a", b"1")
    assert cache.byte_size == 5
    cache.put("huge", b"123456789")
    assert len(cache) == 2
    assert cache.byte_size <= cache.max_bytes
    assert cache.get("huge") is None
    empty = MetadataCache("empty", max_entries=0)
    empty.put("a", b"")
    assert len(empty) == 0


def test_empty_failure_summary_is_a_hit_and_disabled_cache_retains_nothing():
    cache = MetadataCache("test")
    cache.put("failure", b"")
    assert cache.get("failure") == b""
    disabled = MetadataCache("test", enabled=False)
    disabled.put("a", b"data")
    assert disabled.get("a") is None
    assert disabled.byte_size == 0


def test_byte_limit_evicts_even_below_entry_limit():
    cache = MetadataCache("test", max_entries=100, max_bytes=8)
    cache.put("a", b"123")
    cache.put("b", b"456")
    cache.put("c", b"7")
    assert cache.get("a") is None
    assert cache.get("b") == b"456"
    assert cache.get("c") == b"7"
    assert len(cache) == 2
    assert cache.byte_size == 6


def test_edit_keys_include_base_and_order_but_exclude_attempt_metadata():
    first = Candidate((DeleteNodes((0,)), DropInputs(("x",))), base_digest="base")
    duplicate = Candidate(first.edits, base_digest="base", pass_name="another", sequence_number=99)
    assert edit_key(first, "strict") == edit_key(duplicate, "strict")
    assert edit_key(first, "strict") != edit_key(first, "permissive")
    assert edit_key(first, "strict") != edit_key(Candidate(first.edits, "new"), "strict")
    assert edit_key(first, "strict") != edit_key(Candidate(first.edits[::-1], "base"), "strict")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_shape_summary_reused_on_same_base_including_failure(monkeypatch, enabled, fail):
    model = OnnxModel.from_proto(chain_model(6, 3))
    infer = onnx.shape_inference.infer_shapes
    calls = []

    def inference(*args, **kwargs):
        calls.append(1)
        if fail:
            raise RuntimeError("synthetic failure")
        return infer(*args, **kwargs)

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", inference)
    with preparation_caches(enabled) as caches:
        candidates = [model.apply_candidate(Candidate((DeleteNodes((i,)),))) for i in range(3)]
        assert len(calls) == (1 if enabled else 3)
        assert all(m.validate().valid == (not fail) for m in candidates)
        if enabled:
            payload = caches.shapes.get(metadata_key("shape", model.canonical_hash()))
            summary = onnx.GraphProto.FromString(payload)
            assert not summary.node and not summary.initializer
    assert current_preparation() is None


def test_composite_edits_infer_from_intermediate_graph_without_stale_snapshots():
    model = OnnxModel.from_proto(chain_model(6, 3))
    edits = (DeleteNodes((0,)), DeleteNodes((0,)), DeleteNodes((1,)))
    with preparation_caches() as caches:
        composite = model.apply_candidate(Candidate(edits))
        sequential = model
        for edit in edits:
            sequential = sequential.apply_candidate(Candidate((edit,)))
        assert composite.canonical_hash() == sequential.canonical_hash()
        assert composite.validate().valid
        assert composite.node_count == len(composite.to_proto().graph.node) == 3
        assert len(caches.shapes) == 3
    assert model.node_count == 6


class RepeatedRejections:
    name = "repeated"

    def generate(self, model):
        for _ in range(2):
            yield Candidate((DropInputs(("unknown",)),))
        for _ in range(2):
            yield Candidate(())
        yield Candidate((DeleteNodes((2,)),))
        yield Candidate((DeleteNodes((2,)),))
        # A different edit key producing the same content exercises the content cache.
        yield Candidate((DeleteNodes((2, 2)),))


def decisions(report):
    records = (Path(report.work_directory) / "candidates.jsonl").read_text().splitlines()
    return [
        (row["candidate"]["edits"], row["status"], row.get("digest"), row.get("cost"))
        for row in map(json.loads, records)
    ]


def test_edit_and_content_hits_skip_preparation_and_writes_without_changing_decisions(tmp_path):
    reports = []
    for enabled in (False, True):
        _, report = ReductionEngine(
            [RepeatedRejections()],
            config=EngineConfig(cache_enabled=enabled, collect_performance=True),
        ).run(OnnxModel.from_proto(chain_model(5, 2)), ReluFailureCheck(), tmp_path / str(enabled))
        reports.append(report)
        files = list((Path(report.work_directory) / "candidates").glob("*/model.onnx"))
        assert len(files) == (1 if enabled else 3)
        assert report.candidates_generated == 7
        assert report.candidates_invalid == 2
        assert report.candidates_not_cheaper == 2
    off, on = reports
    assert decisions(off) == decisions(on)
    assert off.final_hash == on.final_hash
    assert on.performance.stages["clone"]["calls"] == 4
    assert off.performance.stages["clone"]["calls"] == 7
    assert on.performance.counters["edit_cache_hits"] == 3
    assert on.performance.counters["content_cache_hits"] == 1
    assert on.cache_hits == 2  # Preparation failures are not verdict-cache hits.

    def model_bytes(report):
        return sum(
            path.stat().st_size
            for path in (Path(report.work_directory) / "candidates").glob("*/model.onnx")
        )

    # On tiny models, extra profiling/log metadata can outweigh saved payloads.
    assert model_bytes(on) * 3 == model_bytes(off)
    evaluations = [
        json.loads(row)
        for row in (Path(on.work_directory) / "evaluations.jsonl").read_text().splitlines()
    ]
    assert {row.get("cache_source") for row in evaluations} == {None, "edit", "content"}
    assert all(Path(row["model"]).is_file() for row in evaluations)


def test_inconclusive_verdicts_are_retried_while_shape_preparation_is_shared(tmp_path):
    class Uncertain:
        def run(self, path):
            return (
                FailureCheckResult.INCONCLUSIVE
                if "candidates" in Path(path).parts
                else FailureCheckResult.INTERESTING
            )

    class Repeated:
        name = "repeated"

        def generate(self, model):
            for _ in range(3):
                yield Candidate((DeleteNodes((0,)),))

    _, report = ReductionEngine([Repeated()], config=EngineConfig(collect_performance=True)).run(
        OnnxModel.from_proto(chain_model()), Uncertain(), tmp_path
    )
    assert report.inconclusive == 3
    assert report.failure_check_calls == 5
    assert report.cache_hits == 0
    assert report.performance.stages["shape_inference"]["calls"] == 1
    assert report.minimality["status"] == "unproven"


def test_old_base_rejection_cannot_exclude_edit_on_new_base(tmp_path):
    class Shrink:
        name = "shrink"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            yield Candidate((DeleteNodes((3,)),))

    class NonMonotonic:
        def run(self, path):
            nodes = onnx.load(path).graph.node
            return (
                FailureCheckResult.INTERESTING
                if len(nodes) <= 2 or any(node.name == "n0" for node in nodes)
                else FailureCheckResult.NOT_INTERESTING
            )

    reduced, report = ReductionEngine([Shrink()]).run(
        OnnxModel.from_proto(chain_model(4, 2)), NonMonotonic(), tmp_path
    )
    assert reduced.node_count == 0
    rows = decisions(report)
    assert rows[0][1] == "not_interesting"
    assert rows[2][1] == "interesting"
    assert rows[0][0] == rows[2][0]


def test_run_scopes_reset_after_success_and_exception_even_with_same_model(tmp_path):
    engine = ReductionEngine([RepeatedRejections()], config=EngineConfig(collect_performance=True))
    model = OnnxModel.from_proto(chain_model(5, 2))
    with pytest.raises(InitialNotInteresting):
        engine.run(model, NeverInteresting(), tmp_path)
    assert current_preparation() is None
    for _ in range(2):
        _, report = engine.run(model, RejectCandidates(), tmp_path)
        assert report.failure_check_calls == 3
        assert report.performance.stages["shape_inference"]["calls"] == 1
        assert current_preparation() is None


def test_eviction_only_recomputes_rejected_candidates(tmp_path, monkeypatch):
    from onnx_reduce import preparation

    original = preparation.MetadataCache
    monkeypatch.setattr(
        preparation,
        "MetadataCache",
        lambda *args, **kwargs: original(*args, **kwargs, max_entries=0),
    )
    _, report = ReductionEngine(
        [RepeatedRejections()], config=EngineConfig(collect_performance=True)
    ).run(OnnxModel.from_proto(chain_model(5, 2)), ReluFailureCheck(), tmp_path)
    # No edit or shape retention, but the independent content cache still works.
    assert report.performance.stages["clone"]["calls"] == 7
    assert report.performance.stages["shape_inference"]["calls"] == 3
    assert report.failure_check_calls == 3


def test_invalid_validation_is_reused_without_repeating_clone_or_checker(tmp_path, monkeypatch):
    def fail_inference(*args, **kwargs):
        raise RuntimeError("missing shape")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail_inference)

    class Invalid:
        name = "invalid"

        def generate(self, model):
            for _ in range(2):
                yield Candidate((DeleteNodes((0,)),))

    _, report = ReductionEngine([Invalid()], config=EngineConfig(collect_performance=True)).run(
        OnnxModel.from_proto(chain_model()), ReluFailureCheck(), tmp_path
    )
    assert report.candidates_invalid == 2
    assert report.failure_check_calls == 2
    assert report.performance.stages["clone"]["calls"] == 1
    # Initial model, unique invalid candidate, and independently loaded final artifact.
    assert report.performance.stages["validation"]["calls"] == 3
    assert report.performance.counters["edit_cache_hits"] == 1


def test_content_hit_at_exact_quota_does_not_materialize_another_file(tmp_path):
    class SameContent:
        name = "same_content"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            yield Candidate((DeleteNodes((0, 0)),))

    _, report = ReductionEngine(
        [SameContent()],
        config=EngineConfig(max_candidate_check_calls=1, collect_performance=True),
    ).run(OnnxModel.from_proto(chain_model()), RejectCandidates(), tmp_path)
    assert report.stop_reason == "fixed_point"
    assert report.failure_check_calls == 3
    assert report.performance.counters["content_cache_hits"] == 1
    assert report.performance.counters["edit_cache_hits"] == 0
    assert len(list((Path(report.work_directory) / "candidates").glob("*/model.onnx"))) == 1


def test_inferred_main_graph_types_do_not_use_same_named_nested_values():
    from onnx import TensorProto, helper

    from tests.helpers import tensor_value

    branch = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["v0"],
                value=helper.make_tensor("weight", TensorProto.FLOAT, [2], [1, 2]),
            )
        ],
        "branch",
        [],
        [tensor_value("v0", shape=[2])],
    )
    graph = helper.make_graph(
        [
            helper.make_node("If", ["cond"], ["z"], then_branch=branch, else_branch=branch),
            helper.make_node("Identity", ["x"], ["v0"]),
            helper.make_node("Identity", ["v0"], ["y"]),
        ],
        "scopes",
        [tensor_value("x"), tensor_value("cond", TensorProto.BOOL, [])],
        [tensor_value("y"), tensor_value("z", shape=[2])],
    )
    model = OnnxModel.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    )
    assert model.validate().valid
    with preparation_caches():
        for _ in range(2):
            # Remove the If too: promoting v0 to a graph input while retaining
            # the nested local v0 would itself violate ONNX's SSA constraint.
            candidate = model.apply_candidate(Candidate((DeleteNodes((0, 1)),)))
            promoted = next(
                value for value in candidate.to_proto().graph.input if value.name == "v0"
            )
            assert promoted.type.tensor_type.shape.dim[0].dim_value == 1
            assert candidate.validate().valid
