"""Search scheduling preserves acceptance, budgets and final-version coverage."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
from onnx import helper

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs
from onnx_reduce.engine import EngineConfig, ReductionEngine, ReductionError
from onnx_reduce.failure_check.contract import FailureCheckResult as F
from onnx_reduce.failure_check.runner import FailureCheckError
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.search.stateful import StatefulScheduler
from onnx_reduce.search.types import CandidateFeedback
from tests.helpers import chain_model, tensor_value
from tests.integration.test_reliability import Always, two_outputs
from tests.runtime.test_engine import ReluFailureCheck


def rows(report, filename="scheduler.jsonl"):
    path = Path(report.work_directory) / filename
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class CountCheck:
    def __init__(self, interesting):
        self.interesting = interesting

    def run(self, path):
        return (
            F.INTERESTING
            if self.interesting(len(onnx.load(path).graph.node))
            else F.NOT_INTERESTING
        )


def run(model, runner, workdir, *, passes=None, **config):
    return ReductionEngine(passes, config=EngineConfig(scheduler="stateful", **config)).run(
        model, runner, workdir
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_cross_region_triggers_survive_failed_keep_attempts(tmp_path, cache_enabled):
    class CrossRegion:
        def run(self, path):
            names = {node.name for node in onnx.load(path).graph.node}
            return F.INTERESTING if {"n1", "n14"} <= names else F.NOT_INTERESTING

    reduced, report = run(
        OnnxModel.from_proto(chain_model(17, 1)),
        CrossRegion(),
        tmp_path,
        cache_enabled=cache_enabled,
    )
    assert {node.name for node in reduced.to_proto().graph.node} == {"n1", "n14"}
    assert reduced.validate().valid
    assert report.minimality["status"] == "pass_fixed_point"
    assert any(
        row["direction"] == "keep" and row["status"] == "not_interesting" for row in rows(report)
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_threshold_failure_refines_partitions_to_exact_threshold(tmp_path, cache_enabled):
    reduced, report = run(
        OnnxModel.from_proto(chain_model(17, 8)),
        CountCheck(lambda n: n >= 5),
        tmp_path,
        cache_enabled=cache_enabled,
    )
    assert reduced.node_count == 5
    assert reduced.validate().valid
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.scheduler["scale_reuses"] > 0
    assert report.scheduler["coverage"]["base_digest"] == report.final_hash
    assert report.scheduler["model_version"] == report.candidates_accepted


def test_fragment_extraction_finds_non_monotonic_reduction_for_both_strategies(tmp_path):
    results = {}
    for strategy in ("legacy", "stateful"):
        reduced, report = ReductionEngine(config=EngineConfig(scheduler=strategy)).run(
            OnnxModel.from_proto(chain_model(8, 4)),
            CountCheck(lambda n: n in {8, 5, 2}),
            tmp_path / strategy,
        )
        results[strategy] = reduced.node_count
        assert report.minimality["status"] == "pass_fixed_point"
    assert results == {"legacy": 2, "stateful": 2}


def test_cover_acceptance_invalidates_sweep_and_covers_new_model(tmp_path):
    reduced, report = run(
        OnnxModel.from_proto(chain_model(16, 8)),
        CountCheck(lambda n: n in {16, 12, 6}),
        tmp_path,
        passes=[DeleteNodesPass()],
    )
    assert reduced.node_count == 6
    assert report.scheduler["sources"]["coverage"]["accepted"] >= 1
    coverage = report.scheduler["coverage"]
    assert coverage["completed"]
    assert coverage["model_version"] == report.candidates_accepted
    assert coverage["base_digest"] == reduced.canonical_hash()
    attempts = rows(report)
    cover_accept = next(
        i
        for i, row in enumerate(attempts)
        if row["source"] == "coverage" and row["status"] == "accepted"
    )
    assert any(row["source"] == "heuristic" for row in attempts[cover_accept + 1 :])
    assert attempts[-1]["source"] == "coverage"


def test_earlier_stage_failure_is_revisited_after_other_stage_accepts(tmp_path):
    class DropA:
        name = "a"

        def generate(self, model):
            if 0 in model.id_to_index:
                yield Candidate((DeleteNodes((model.id_to_index[0],)),))

    class DropB:
        name = "b"

        def generate(self, model):
            if 1 in model.id_to_index:
                yield Candidate((DeleteNodes((model.id_to_index[1],)),))

    class NonMonotonic:
        def run(self, path):
            names = {node.name for node in onnx.load(path).graph.node}
            return (
                F.INTERESTING
                if "n2" in names and ("n0" in names or "n1" not in names)
                else F.NOT_INTERESTING
            )

    reduced, report = run(
        OnnxModel.from_proto(chain_model(3, 2)), NonMonotonic(), tmp_path, passes=[DropA(), DropB()]
    )
    assert reduced.node_ids == (2,)
    a_attempts = [row for row in rows(report) if row["pass_name"] == "a"]
    assert [row["status"] for row in a_attempts] == ["not_interesting", "accepted"]
    assert a_attempts[0]["base_digest"] != a_attempts[1]["base_digest"]
    assert report.minimality["status"] == "pass_fixed_point"


def test_dead_bug_and_output_identity_remain_failure_check_gated(tmp_path):
    class DeadBug:
        def run(self, path):
            model = OnnxModel.load(path)
            return (
                F.INTERESTING
                if "second" in model.output_names and "Relu" in model.op_types
                else F.NOT_INTERESTING
            )

    reduced, report = run(two_outputs(), DeadBug(), tmp_path)
    assert reduced.output_names == ("second",)
    assert reduced.op_types == ("Relu",)
    assert reduced.dead_node_indices() == (0,)
    assert report.minimality["status"] == "pass_fixed_point"


@pytest.mark.parametrize("node_count", [0, 1])
def test_zero_and_one_node_models_keep_valid_output(tmp_path, node_count):
    proto = (
        chain_model(1, 0)
        if node_count
        else helper.make_model(
            helper.make_graph([], "empty", [tensor_value("x")], [tensor_value("x")]),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )
    reduced, report = run(OnnxModel.from_proto(proto), Always(), tmp_path)
    assert reduced.node_ids == ()
    assert len(reduced.output_names) == 1
    assert reduced.validate().valid
    assert report.minimality["status"] == "pass_fixed_point"


def test_phase_quantum_counts_continuous_acceptances(tmp_path):
    class Resources:
        name = "resources"

        def generate(self, model):
            names = model.unused_initializer_names()
            if names:
                yield Candidate((DropInitializers((names[0],)),))

    class Marker:
        name = "marker"

        def generate(self, model):
            yield Candidate(())

    proto = chain_model(1, 0)
    proto.graph.initializer.extend(
        helper.make_tensor(f"unused{i}", onnx.TensorProto.FLOAT, [1], [0]) for i in range(20)
    )
    reduced, report = run(
        OnnxModel.from_proto(proto), Always(), tmp_path, passes=[Resources(), Marker()]
    )
    attempts = rows(report)
    assert [row["status"] for row in attempts[:8]] == ["accepted"] * 8
    assert attempts[8]["pass_name"] == "marker"
    assert attempts[8]["model_version"] == 8
    assert reduced.cost().initializer_count == 0


def test_phase_quantum_counts_preparation_rejections(tmp_path):
    class Long:
        name = "long"

        def generate(self, model):
            for _ in range(20):
                yield Candidate((DropInputs(("missing",)),))

    class Marker:
        name = "marker"

        def generate(self, model):
            yield Candidate(())

    _, report = run(
        OnnxModel.from_proto(chain_model()), Always(), tmp_path, passes=[Long(), Marker()]
    )
    assert rows(report)[8]["pass_name"] == "marker"
    assert report.failure_check_calls == 2
    assert report.scheduler["coverage"]["completed"]


@pytest.mark.parametrize("budget", [0, 1, 4])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_check_budgets_and_final_independent_recheck(tmp_path, budget, cache_enabled):
    output = tmp_path / "actual-output.onnx"
    reduced, report = ReductionEngine(
        config=EngineConfig(
            scheduler="stateful", max_candidate_check_calls=budget, cache_enabled=cache_enabled
        )
    ).run(
        OnnxModel.from_proto(chain_model(12, 6)),
        CountCheck(lambda n: n == 12),
        tmp_path / "work",
        output_path=output,
    )
    assert report.failure_check_calls <= budget + 2
    assert report.stop_reason == "budget_exhausted"
    assert report.minimality["status"] == "unproven"
    assert not report.scheduler["coverage"]["completed"]
    checks = rows(report, "evaluations.jsonl")
    assert Path(checks[-1]["model"]) == output
    assert not checks[-1]["cache_hit"]
    assert checks[-1]["outcome"] == "interesting"
    assert OnnxModel.load(output).canonical_hash() == reduced.canonical_hash()


class Duplicates:
    name = "duplicates"

    def generate(self, model):
        yield Candidate((DeleteNodes((0,)),))
        yield Candidate((DeleteNodes((0,)),))


class RejectCandidates:
    def run(self, path):
        return F.NOT_INTERESTING if "candidates" in Path(path).parts else F.INTERESTING


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_exact_budget_still_allows_cache_hits_through_coverage(tmp_path, cache_enabled):
    _, report = run(
        OnnxModel.from_proto(chain_model()),
        RejectCandidates(),
        tmp_path,
        passes=[Duplicates()],
        cache_enabled=cache_enabled,
        max_candidate_check_calls=1,
    )
    assert report.failure_check_calls == 3
    assert report.cache_hits == (3 if cache_enabled else 0)
    assert report.stop_reason == ("fixed_point" if cache_enabled else "budget_exhausted")
    assert report.scheduler["coverage"]["completed"] == cache_enabled


def test_inconclusive_is_retried_in_coverage_and_does_not_prove_fixed_point(tmp_path):
    class Uncertain:
        def run(self, path):
            return F.INCONCLUSIVE if "candidates" in Path(path).parts else F.INTERESTING

    _, report = run(
        OnnxModel.from_proto(chain_model()), Uncertain(), tmp_path, passes=[Duplicates()]
    )
    assert report.failure_check_calls == 6
    assert report.cache_hits == 0
    assert report.inconclusive == 4
    assert report.stop_reason == "fixed_point"
    assert report.minimality["status"] == "unproven"
    assert report.minimality["sweep_inconclusive_candidate_count"] == 2


def test_coverage_runs_deterministic_checks_again_when_cache_disabled(tmp_path):
    _, report = run(
        OnnxModel.from_proto(chain_model(1, 0)),
        ReluFailureCheck(),
        tmp_path,
        passes=[DeleteNodesPass()],
        cache_enabled=False,
    )
    assert report.failure_check_calls == 4
    assert report.cache_hits == 0
    assert report.scheduler["sources"]["heuristic"]["failure_check_calls"] == 1
    assert report.scheduler["sources"]["coverage"]["failure_check_calls"] == 1
    assert report.minimality["status"] == "pass_fixed_point"


def test_failed_final_output_recheck_overrides_completed_coverage(tmp_path):
    class Unstable(ReluFailureCheck):
        def run(self, path):
            return (
                F.NOT_INTERESTING if Path(path).name == "actual-output.onnx" else super().run(path)
            )

    _, report = ReductionEngine(config=EngineConfig(scheduler="stateful")).run(
        OnnxModel.from_proto(chain_model(4, 2)),
        Unstable(),
        tmp_path / "work",
        output_path=tmp_path / "actual-output.onnx",
    )
    assert report.scheduler["coverage"]["completed"]
    assert report.stop_reason == "unstable_check"
    assert report.minimality["status"] == "unproven"


def test_stateful_rejects_stale_generic_candidate_without_fallback(tmp_path):
    class Stale:
        name = "stale"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),), base_digest="old")

    with pytest.raises(ReductionError, match="different base"):
        run(OnnxModel.from_proto(chain_model()), Always(), tmp_path, passes=[Stale()])


def test_old_inconclusive_does_not_taint_completed_new_version(tmp_path):
    class First:
        name = "first"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))

    class Middle:
        name = "middle"

        def generate(self, model):
            if model.node_count == 3:
                yield Candidate((DeleteNodes((1,)),))

    class Check:
        def run(self, path):
            names = {node.name for node in onnx.load(path).graph.node}
            if names == {"n1", "n2"}:
                return F.INCONCLUSIVE
            return F.INTERESTING if "n0" in names and "n2" in names else F.NOT_INTERESTING

    _, report = run(
        OnnxModel.from_proto(chain_model(3, 2)), Check(), tmp_path, passes=[First(), Middle()]
    )
    assert report.inconclusive == 1
    assert report.minimality["sweep_inconclusive_candidate_count"] == 0
    assert report.minimality["status"] == "pass_fixed_point"


class OpaqueModel:
    """An adapter with no optional node-search capability."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name in {"node_ids", "id_to_index"}:
            raise AttributeError(name)
        return getattr(self.model, name)

    def apply_candidate(self, candidate):
        return OpaqueModel(self.model.apply_candidate(candidate))


def test_node_capability_error_precedes_initial_backend_call(tmp_path):
    class NotCalled:
        def run(self, path):
            pytest.fail("capability preflight must precede the backend")

    with pytest.raises(ValueError, match="requires node_ids"):
        run(OpaqueModel(OnnxModel.from_proto(chain_model())), NotCalled(), tmp_path)


def test_same_named_custom_pass_stays_generic_without_node_capability(tmp_path):
    source = OpaqueModel(OnnxModel.from_proto(chain_model()))

    class Custom:
        name = "delete_nodes"

        def generate(self, model):
            assert model is source  # Configuring editing must not unwrap adapters.
            yield Candidate(())

    _, report = run(source, Always(), tmp_path, passes=[Custom()])
    assert report.candidates_generated == 2
    assert {row["direction"] for row in rows(report)} == {"pass"}
    assert "keep-region" not in report.minimality["scope"]


def test_generic_generators_are_discarded_on_version_change(tmp_path):
    class Guarded:
        name = "guarded"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            if model.node_count > 1:
                pytest.fail("an accepted base must not resume its old generator")

    reduced, report = run(
        OnnxModel.from_proto(chain_model(4, 3)), ReluFailureCheck(), tmp_path, passes=[Guarded()]
    )
    assert reduced.node_count == 1
    assert report.minimality["status"] == "pass_fixed_point"


def test_engine_reuse_resets_state_after_runner_error(tmp_path):
    class Broken:
        calls = 0

        def run(self, path):
            self.calls += 1
            if self.calls > 1:
                raise FailureCheckError("backend unavailable")
            return F.INTERESTING

    engine = ReductionEngine(config=EngineConfig(scheduler="stateful"))
    model = OnnxModel.from_proto(chain_model(8, 4))
    with pytest.raises(FailureCheckError, match="backend unavailable"):
        engine.run(model, Broken(), tmp_path / "failed")
    reports = [engine.run(model, ReluFailureCheck(), tmp_path / str(i))[1] for i in range(2)]
    assert reports[0].final_hash == reports[1].final_hash
    assert rows(reports[0]) == rows(reports[1])
    assert reports[0].scheduler == reports[1].scheduler


@pytest.mark.parametrize("max_seen", [0, 1, 4096])
def test_dedup_eviction_does_not_prevent_scan_termination(max_seen):
    model = OnnxModel.from_proto(chain_model(17, 8))
    scheduler = StatefulScheduler([DeleteNodesPass()], model)
    scheduler._nodes[0].max_seen = max_seen
    attempts = []
    while (attempt := scheduler.next_attempt(model)) is not None:
        attempts.append(attempt)
        assert len(attempts) < 200
        assert len(scheduler._nodes[0]._seen) <= max_seen
        assert attempt.candidate.edits[0].indices
        scheduler.observe(attempt, CandidateFeedback("not_interesting", model.cost()))
    assert scheduler.coverage().completed
    assert any(attempt.source == "coverage" for attempt in attempts)


def test_feedback_distinguishes_preparation_and_cache_branches(tmp_path, monkeypatch):
    class Rejections:
        name = "rejections"

        def generate(self, model):
            for edit in (DropInputs(("missing",)), None, DeleteNodes((0,)), DeleteNodes((2,))):
                for _ in range(2):
                    yield Candidate(()) if edit is None else Candidate((edit,))
            yield Candidate((DeleteNodes((2, 2)),))

    monkeypatch.setattr(OnnxModel, "_infer_type_summary", lambda self: b"")
    engine = ReductionEngine([Rejections()])
    _, report = engine.run(OnnxModel.from_proto(chain_model(3, 1)), RejectCandidates(), tmp_path)
    feedback = rows(report)
    assert [item["status"] for item in feedback] == [
        "edit_failed",
        "edit_failed",
        "not_cheaper",
        "not_cheaper",
        "invalid",
        "invalid",
        "not_interesting",
        "not_interesting",
        "not_interesting",
    ]
    assert [item["cache_source"] for item in feedback] == [
        None,
        "edit",
        None,
        "edit",
        None,
        "edit",
        None,
        "edit",
        "content",
    ]
    assert sum(item["failure_check_calls"] for item in feedback) == 1
    assert report.failure_check_calls == 3
