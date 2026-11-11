"""Staged goals, independent finalization and target continuity across runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import onnx
import pytest

from onnx_reduce.cli import main
from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs
from onnx_reduce.engine import EngineConfig, InitialNotInteresting, ReductionEngine, ReductionError
from onnx_reduce.failure_check.contract import FailureCheckConfig
from onnx_reduce.failure_check.contract import FailureCheckResult as F
from onnx_reduce.failure_check.runner import FailureCheckObservation
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes import default_passes
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass
from onnx_reduce.search.coverage import Finalizer
from onnx_reduce.search.staged import StagedScheduler
from onnx_reduce.search.types import CandidateFeedback
from tests.helpers import chain_model, tensor_value
from tests.runtime.test_engine import ReluFailureCheck
from tests.search.test_scheduler import CountCheck, rows


def run(model, runner, workdir, *, passes=None, **config):
    return ReductionEngine(passes, config=EngineConfig(scheduler="staged", **config)).run(
        model, runner, workdir
    )


@pytest.mark.parametrize("cache", [False, True])
def test_complete_covers_final_version_and_reads_back_actual_output(tmp_path, cache):
    reduced, report = run(
        OnnxModel.from_proto(chain_model(17, 8)),
        ReluFailureCheck(),
        tmp_path,
        cache_enabled=cache,
    )
    assert reduced.node_count == 1
    assert report.stop_reason == "fixed_point"
    assert report.minimality["status"] == "pass_fixed_point"
    coverage = report.scheduler["coverage"]
    assert coverage["completed"]
    assert coverage["base_digest"] == report.final_hash == reduced.canonical_hash()
    assert coverage["model_version"] == report.candidates_accepted
    assert report.artifact_recheck["read_back_validated"]
    assert not report.artifact_recheck["cache_hit"]
    assert rows(report, "evaluations.jsonl")[-1]["model"] == report.artifact_recheck["model"]


def test_fast_never_claims_fixed_point_even_after_search_exhaustion(tmp_path):
    _, report = run(
        OnnxModel.from_proto(chain_model(8, 4)),
        ReluFailureCheck(),
        tmp_path,
        goal="fast",
    )
    assert report.stop_reason == "search_exhausted"
    assert report.minimality["status"] == "unproven"
    assert not report.scheduler["coverage"]["completed"]
    assert not report.scheduler["finalization"]["started"]
    assert all(row["source"] != "coverage" for row in rows(report))
    assert report.artifact_recheck["outcome"] == "interesting"


def test_plateau_is_deterministic_and_complete_still_runs_full_finalizer(tmp_path, monkeypatch):
    monkeypatch.setattr(StagedScheduler, "no_progress_limit", 1)
    for goal in ("fast", "complete"):
        _, report = run(
            OnnxModel.from_proto(chain_model(8, 4)),
            CountCheck(lambda n: n == 8),
            tmp_path / goal,
            goal=goal,
        )
        assert "plateau" in report.scheduler["stage_ends"].values()
        assert report.stop_reason == ("plateau" if goal == "fast" else "fixed_point")
        assert report.scheduler["coverage"]["completed"] == (goal == "complete")


def test_finalizer_acceptance_restarts_only_finalizer_and_invalidates_all_old_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(StagedScheduler, "no_progress_limit", 0)

    # A custom pass returns candidates only in full scanning, not heuristic stages.
    class FinishOnly(DeleteNodesPass):
        def generate_stage(self, model, stage):
            return iter(())

    reduced, report = run(
        OnnxModel.from_proto(chain_model(16, 8)),
        CountCheck(lambda n: n in {16, 12, 6}),
        tmp_path,
        passes=[FinishOnly()],
    )
    assert reduced.node_count == 6
    assert report.scheduler["finalization"]["restarts"] >= 1
    attempts = rows(report)
    first = next(
        i
        for i, row in enumerate(attempts)
        if row["source"] == "coverage" and row["status"] == "accepted"
    )
    assert all(row["stage"] == "finalization" for row in attempts[first:])
    assert report.scheduler["coverage"]["base_digest"] == reduced.canonical_hash()
    assert report.minimality["status"] == "pass_fixed_point"


def test_independent_finalizer_rejects_unobserved_base_change():
    model = OnnxModel.from_proto(chain_model())
    finalizer = Finalizer(default_passes(), model)
    changed = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    with pytest.raises(ValueError, match="without accepted feedback"):
        finalizer.next_attempt(changed)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 4])
@pytest.mark.parametrize("goal", ["fast", "complete"])
def test_global_call_budget_includes_search_and_finalization(tmp_path, cache, budget, goal):
    _, report = run(
        OnnxModel.from_proto(chain_model(32, 16)),
        CountCheck(lambda n: n == 32),
        tmp_path,
        cache_enabled=cache,
        max_candidate_check_calls=budget,
        goal=goal,
    )
    assert report.failure_check_calls <= budget + 2
    assert report.stop_reason == "budget_exhausted"
    assert report.minimality["status"] == "unproven"
    assert report.artifact_recheck["outcome"] == "interesting"


def test_inconclusive_scan_does_not_report_fixed_point(tmp_path):
    class Uncertain:
        def run(self, path):
            return F.INCONCLUSIVE if "candidates" in Path(path).parts else F.INTERESTING

    _, report = run(OnnxModel.from_proto(chain_model(4, 2)), Uncertain(), tmp_path)
    assert report.stop_reason == "coverage_inconclusive"
    assert report.scheduler["coverage"]["completed"]
    assert report.minimality["status"] == "unproven"
    assert report.minimality["sweep_inconclusive_candidate_count"] > 0


def test_old_inconclusive_does_not_taint_new_finalizer_version():
    model = OnnxModel.from_proto(chain_model())
    finalizer = Finalizer(default_passes(), model)
    finalizer.next_attempt(model)
    finalizer.observe(CandidateFeedback("inconclusive", model.cost()))
    changed = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    finalizer.observe(CandidateFeedback("accepted", model.cost(), changed.cost()), changed)
    assert finalizer.coverage().inconclusive_candidate_count == 0
    assert not finalizer.coverage().completed
    assert finalizer.coverage().base_digest == changed.canonical_hash()


def test_cleanups_run_after_node_acceptance_with_bounded_quota(tmp_path):
    _, report = run(
        OnnxModel.from_proto(chain_model(32, 16)),
        ReluFailureCheck(),
        tmp_path,
        passes=[RemoveUnusedPass(), DeleteNodesPass()],
        max_candidate_check_calls=4,
        goal="fast",
    )
    attempts = rows(report)
    assert any(row["stage"] == "cleanup" and row["status"] == "accepted" for row in attempts)
    count = 0
    for row in attempts:
        count = count + 1 if row["stage"] == "cleanup" else 0
        assert count <= StagedScheduler.cleanup_quantum


def test_generic_passes_keep_their_full_candidate_definition(tmp_path):
    class Generic:
        name = "delete_nodes"  # Name alone must not opt into region search.

        def generate(self, model):
            for index in range(model.node_count):
                yield Candidate((DeleteNodes((index,)),))

    reduced, report = run(
        OnnxModel.from_proto(chain_model(6, 2)),
        ReluFailureCheck(),
        tmp_path,
        passes=[Generic()],
    )
    assert reduced.node_count == 1
    assert report.scheduler["region_sources"] == []
    assert "region" not in report.minimality["scope"]


def test_earlier_generic_candidate_is_revisited_on_new_final_base(tmp_path):
    class Resources:
        name = "resources"

        def generate(self, model):
            for name in ("a", "b"):
                if name in model.unused_input_names():
                    yield Candidate((DropInputs((name,)),))

    proto = chain_model(1, 0)
    proto.graph.input.extend([tensor_value("a"), tensor_value("b")])

    class Check:
        def run(self, path):
            names = {value.name for value in onnx.load(path).graph.input}
            return F.INTERESTING if "a" in names or "b" not in names else F.NOT_INTERESTING

    reduced, report = run(OnnxModel.from_proto(proto), Check(), tmp_path, passes=[Resources()])
    assert reduced.input_names == ("x",)
    assert report.minimality["status"] == "pass_fixed_point"


@pytest.mark.parametrize("goal", ["fast", "complete"])
def test_failed_output_recheck_overrides_any_search_success(tmp_path, goal):
    class Unstable:
        def run(self, path):
            return F.NOT_INTERESTING if "final" in Path(path).parts else F.INTERESTING

    _, report = run(OnnxModel.from_proto(chain_model(3, 1)), Unstable(), tmp_path, goal=goal)
    assert report.stop_reason == "unstable_check"
    assert report.minimality["status"] == "unproven"


def test_read_back_detects_bad_materialization(tmp_path, monkeypatch):
    materialize = OnnxModel.materialize

    def corrupt(model, path):
        artifact = materialize(model, path)
        if "final" in Path(path).parts:
            artifact.model_path.write_bytes(b"bad bytes")
        return artifact

    monkeypatch.setattr(OnnxModel, "materialize", corrupt)
    with pytest.raises(ReductionError, match="digest mismatch"):
        run(
            OnnxModel.from_proto(chain_model()),
            ReluFailureCheck(),
            tmp_path,
            max_candidate_check_calls=0,
        )
    assert list(tmp_path.glob("run-*/checkpoint.json"))


class SignatureCheck:
    failure_check_config = FailureCheckConfig(("configured-check",))

    def __init__(self, signature="A"):
        self.signature = signature
        self.calls = 0

    def observe(self, path):
        self.calls += 1
        return FailureCheckObservation(F.INTERESTING, returncode=1, stdout=f"BUG:{self.signature}")


def original_run(tmp_path):
    model = OnnxModel.from_proto(chain_model(4, 2))
    _, report = run(
        model,
        SignatureCheck(),
        tmp_path / "fast",
        goal="fast",
        max_candidate_check_calls=0,
        signature_regex=r"BUG:(\w+)",
    )
    return OnnxModel.load(report.artifact_recheck["model"]), report


def test_finalize_from_restores_identity_and_starts_fresh_scan(tmp_path):
    model, first = original_run(tmp_path)
    _, report = run(
        model,
        SignatureCheck(),
        tmp_path / "finish",
        finalize_from=first.work_directory,
        signature_regex=r"BUG:(\w+)",
    )
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.scheduler["target_origin"] == first.work_directory
    assert report.scheduler["finalize_only"]
    assert all(row["stage"] == "finalization" for row in rows(report))
    assert rows(report, "evaluations.jsonl")[0]["cache_hit"] is False
    assert json.loads((Path(report.work_directory) / "target.json").read_text())[
        "baseline_identity"
    ] == [
        "exit:1",
        ["A"],
    ]


def test_finalize_from_does_not_rebaseline_to_a_different_bug(tmp_path):
    model, first = original_run(tmp_path)
    check = SignatureCheck("B")
    with pytest.raises(InitialNotInteresting):
        run(
            model,
            check,
            tmp_path / "finish",
            finalize_from=first.work_directory,
            signature_regex=r"BUG:(\w+)",
        )
    assert check.calls == 1


@pytest.mark.parametrize("manifest_name", ["task.json", "replay.json"])
@pytest.mark.parametrize("contract", [None, "zero-interesting-v1"])
def test_finalize_from_rejects_older_exit_code_contract_before_backend(
    tmp_path, manifest_name, contract
):
    model, first = original_run(tmp_path)
    manifest_path = Path(first.work_directory) / manifest_name
    manifest = json.loads(manifest_path.read_text())
    if contract is None:
        manifest.pop("failure_check_contract")
    else:
        manifest["failure_check_contract"] = contract
    manifest_path.write_text(json.dumps(manifest))
    check = SignatureCheck()
    with pytest.raises(ValueError, match="exit-code contract differs; start a new run"):
        run(
            model,
            check,
            tmp_path / "finish",
            finalize_from=first.work_directory,
            signature_regex=r"BUG:(\w+)",
        )
    assert check.calls == 0


@pytest.mark.parametrize("mismatch", ["model", "command", "timeout", "regex", "cwd", "manifest"])
def test_finalize_from_rejects_mismatched_configuration_before_backend(
    tmp_path, mismatch, monkeypatch
):
    model, first = original_run(tmp_path)
    check = SignatureCheck()
    config = {"signature_regex": r"BUG:(\w+)"}
    if mismatch == "model":
        model = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    elif mismatch == "command":
        check.failure_check_config = replace(check.failure_check_config, command=("other-check",))
    elif mismatch == "timeout":
        check.failure_check_config = replace(check.failure_check_config, timeout_seconds=1)
    elif mismatch == "regex":
        config["signature_regex"] = "OTHER"
    elif mismatch == "cwd":
        monkeypatch.chdir(tmp_path)
    else:
        (Path(first.work_directory) / "target.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot finalize from"):
        run(model, check, tmp_path / "finish", finalize_from=first.work_directory, **config)
    assert check.calls == 0


@pytest.mark.parametrize(
    "config",
    [
        {"goal": "fast"},
        {"finalize_from": "run"},
        {"scheduler": "staged", "goal": "unknown"},
        {"scheduler": "staged", "goal": "fast", "finalize_from": "run"},
    ],
)
def test_invalid_goal_combinations(config):
    with pytest.raises(ValueError):
        EngineConfig(**config)


def test_cli_fast_then_independent_complete(tmp_path):
    from tests.integration.test_cli import write_failure_check

    source = tmp_path / "input.onnx"
    fast = tmp_path / "fast.onnx"
    final = tmp_path / "final.onnx"
    first_report = tmp_path / "fast.json"
    final_report = tmp_path / "final.json"
    onnx.save(chain_model(4, 2), source)
    check = write_failure_check(
        tmp_path,
        "check.py",
        "print('BUG:A'); raise SystemExit(1 if has_trigger else 0)",
    )
    common = ["--test", str(check), "--scheduler", "staged", "--signature-regex", r"BUG:(\w+)"]
    assert (
        main(
            [
                str(source),
                "-o",
                str(fast),
                *common,
                "--goal",
                "fast",
                "--max-candidate-check-calls",
                "0",
                "--report",
                str(first_report),
            ]
        )
        == 0
    )
    first = json.loads(first_report.read_text())
    assert (
        main(
            [
                str(fast),
                "-o",
                str(final),
                *common,
                "--finalize-from",
                first["work_directory"],
                "--report",
                str(final_report),
            ]
        )
        == 0
    )
    report = json.loads(final_report.read_text())
    assert report["final"]["node_count"] == 1
    assert report["minimality"]["status"] == "pass_fixed_point"
    assert hashlib.sha256(final.read_bytes()).hexdigest() == report["final_hash"]


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize(
    "condition", ["cross-region", "threshold", "non-monotonic", "dead-trigger"]
)
def test_fault_shapes_remain_preserved_in_complete_goal(tmp_path, cache, condition):
    from types import SimpleNamespace

    from benchmarks.reduction import SyntheticCheck, make_model

    args = SimpleNamespace(size=16, condition=condition, threshold=4, interesting_sizes=(16, 9, 4))
    proto = make_model("chain", 16, condition=condition)
    reduced, report = run(
        OnnxModel.from_proto(proto), SyntheticCheck(args), tmp_path, cache_enabled=cache
    )
    assert reduced.validate().valid
    assert report.minimality["status"] == "pass_fixed_point"
    if condition == "cross-region":
        assert {node.name for node in reduced.to_proto().graph.node} == {"n4", "n12"}
    elif condition in {"threshold", "non-monotonic"}:
        assert reduced.node_count == 4
    else:
        assert {node.name for node in reduced.to_proto().graph.node} == {"dead_trigger"}


@pytest.mark.parametrize("max_seen", [0, 1, 4096])
def test_finalizer_dedup_eviction_cannot_skip_distinct_edits_or_prevent_termination(max_seen):
    from onnx_reduce.preparation import edit_key
    from onnx_reduce.search.coverage import iter_coverage

    model = OnnxModel.from_proto(chain_model(8, 4))
    passes = default_passes()
    finalizer = Finalizer(passes, model)
    finalizer.max_seen = max_seen
    expected = {
        edit_key(replace(attempt.candidate, base_digest=model.canonical_hash()), "test")
        for attempt in iter_coverage(passes, model, 0, model.canonical_hash(), include_regions=True)
    }
    visited = []
    for _ in range(1000):
        attempt = finalizer.next_attempt(model)
        if attempt is None:
            break
        visited.append(
            edit_key(replace(attempt.candidate, base_digest=model.canonical_hash()), "test")
        )
        finalizer.observe(CandidateFeedback("not_interesting", model.cost()))
    else:
        pytest.fail("finalizer did not terminate")
    assert set(visited) == expected
    assert finalizer.coverage().completed
    if max_seen == 4096:
        assert len(visited) == len(set(visited))
        assert finalizer.deduplicated > 0


def test_inconclusive_from_heuristic_is_retried_by_independent_finalizer(tmp_path):
    class OneCandidate:
        name = "one"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))

    class Sometimes:
        def __init__(self):
            self.seen = set()

        def run(self, path):
            if "candidates" not in Path(path).parts:
                return F.INTERESTING
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest not in self.seen:
                self.seen.add(digest)
                return F.INCONCLUSIVE
            return F.NOT_INTERESTING

    _, report = run(
        OnnxModel.from_proto(chain_model(4, 2)), Sometimes(), tmp_path, passes=[OneCandidate()]
    )
    assert report.inconclusive > 0
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.scheduler["coverage"]["inconclusive_candidate_count"] == 0


def test_module_keep_candidate_reduces_noncontiguous_coupled_nodes(tmp_path):
    from tests.search.test_regions import annotated_chain

    class CoupledModule:
        def run(self, path):
            names = {node.name for node in onnx.load(path).graph.node}
            return (
                F.INTERESTING
                if names
                in (
                    {f"n{i}" for i in range(8)},
                    {"n0", "n2", "n4", "n6"},
                )
                else F.NOT_INTERESTING
            )

    reduced, report = run(OnnxModel.from_proto(annotated_chain()), CoupledModule(), tmp_path)
    assert {node.name for node in reduced.to_proto().graph.node} == {"n0", "n2", "n4", "n6"}
    assert any(
        row.get("region_source") == "module"
        and row["direction"] == "keep"
        and row["status"] == "accepted"
        for row in rows(report)
    )
    assert report.minimality["status"] == "pass_fixed_point"
