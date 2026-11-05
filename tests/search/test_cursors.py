"""Continuation, heuristic ordering, and fixed-point evidence stay separate."""

import json
from dataclasses import replace
from pathlib import Path

import onnx
import pytest

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult as F
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.base import RemovalGroup
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.search.cursors import RejectionOrder, RemovalCursor, history_key
from onnx_reduce.search.staged import StagedScheduler
from onnx_reduce.search.types import CandidateFeedback, ScheduledAttempt
from tests.helpers import chain_model
from tests.runtime.test_engine import ReluFailureCheck


def test_cursor_rebinds_stable_ids_and_visits_newly_freed_position():
    model = OnnxModel.from_proto(chain_model(8, 4))
    group = RemovalGroup("nodes", model.node_ids, DeleteNodes, nodes=True)
    cursor = RemovalCursor(group)
    candidate = cursor.next_candidate(model, group)
    assert candidate.edits[0].indices == (0,)
    changed = model.apply_candidate(candidate)
    next_group = replace(group, items=changed.node_ids)
    following = cursor.next_candidate(changed, next_group)
    assert following.edits[0].indices == (0,)
    assert following.base_digest == changed.canonical_hash()


def test_resource_cursor_discovers_new_items_and_keeps_one_output():
    model = OnnxModel.from_proto(chain_model())
    group = RemovalGroup("weights", ("a", "b"), DropInitializers)
    cursor = RemovalCursor(group)
    assert cursor.next_candidate(model, group).edits[0].names == ("a",)
    assert cursor.next_candidate(model, group).edits[0].names == ("b",)
    assert cursor.next_candidate(model, group) is None
    group = replace(group, items=("c",))
    assert cursor.next_candidate(model, group).edits[0].names == ("c",)
    assert (
        RemovalCursor(replace(group, keep_one=True)).next_candidate(
            model, replace(group, keep_one=True)
        )
        is None
    )


def attempt(model, index):
    return ScheduledAttempt(
        Candidate((DeleteNodes((index,)),)), DeleteNodesPass(), 0, 0, model.canonical_hash()
    )


def test_history_is_stable_order_sensitive_bounded_and_never_a_verdict():
    model = OnnxModel.from_proto(chain_model(12, 6))
    changed = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
    assert history_key(attempt(model, 2), model) == history_key(attempt(changed, 1), changed)
    one = replace(attempt(model, 0), candidate=Candidate((DeleteNodes((0, 1)),)))
    two = replace(one, candidate=Candidate((DeleteNodes((1, 0)),)))
    assert history_key(one, model) != history_key(two, model)
    history = RejectionOrder()
    history.cache.max_entries = 3
    for index in range(10):
        history.observe(
            attempt(model, index), model, CandidateFeedback("not_interesting", model.cost())
        )
    assert len(history.cache) == 3
    pending = list(history.order([attempt(changed, i) for i in range(10)], changed))
    assert {a.candidate.edits for a in pending} == {
        attempt(changed, i).candidate.edits for i in range(10)
    }
    assert history.deferred > 0
    assert history.retried == history.deferred
    first = attempt(changed, 1)
    history.observe(first, changed, CandidateFeedback("inconclusive", changed.cost()))
    assert history.cache.get(history_key(first, changed)) is None


def test_all_deferred_work_has_bounded_lookahead_and_is_drained():
    model = OnnxModel.from_proto(chain_model(30, 15))
    history = RejectionOrder()
    old = [attempt(model, i) for i in range(29)]
    for item in old:
        history.observe(item, model, CandidateFeedback("invalid", model.cost()))
    now = [replace(item, base_digest="next") for item in old]
    ordered = history.order(iter(now), model)
    seen = []
    for item in ordered:
        seen.append(item)
        assert len(history.pending) < history.quantum
    assert len(seen) == len(now)
    assert {a.candidate.edits for a in seen} == {a.candidate.edits for a in now}


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("goal", ["fast", "complete"])
@pytest.mark.parametrize("budget", [0, 4, None])
def test_cursor_budget_goals_and_actual_output_recheck(tmp_path, cache, goal, budget):
    _, report = ReductionEngine(
        config=EngineConfig(
            scheduler="staged",
            search_policy="cursor",
            model_edit_mode="auto",
            goal=goal,
            history_order="defer",
            direction_order="adaptive",
            cache_enabled=cache,
            max_candidate_check_calls=budget,
        )
    ).run(OnnxModel.from_proto(chain_model(12, 6)), ReluFailureCheck(), tmp_path)
    if budget is not None:
        assert report.failure_check_calls <= budget + 2
    assert report.artifact_recheck["read_back_validated"]
    assert not report.artifact_recheck["cache_hit"]
    if goal == "fast" or report.stop_reason != "fixed_point":
        assert report.minimality["status"] == "unproven"
    else:
        assert report.minimality["status"] == "pass_fixed_point"
        assert report.scheduler["coverage"]["base_digest"] == report.final_hash


def test_old_rejection_can_be_accepted_after_another_edit(tmp_path):
    class Check:
        def run(self, path):
            names = {n.name for n in onnx.load(path).graph.node}
            return (
                F.INTERESTING
                if "n2" in names and ("n0" in names or "n3" not in names)
                else F.NOT_INTERESTING
            )

    outcomes = []
    for order in ("off", "defer"):
        model, report = ReductionEngine(
            config=EngineConfig(
                scheduler="staged",
                search_policy="cursor",
                history_order=order,
            )
        ).run(OnnxModel.from_proto(chain_model(4, 2)), Check(), tmp_path / order)
        assert model.op_types == ("Relu",)
        assert report.minimality["status"] == "pass_fixed_point"
        outcomes.append(model.cost())
    assert outcomes[0] == outcomes[1]


def test_cursor_instrumentation_does_not_change_decisions(tmp_path):
    traces = []
    for enabled in (False, True):
        _, report = ReductionEngine(
            config=EngineConfig(
                scheduler="staged",
                search_policy="cursor",
                model_edit_mode="auto",
                history_order="defer",
                direction_order="adaptive",
                collect_performance=enabled,
            )
        ).run(
            OnnxModel.from_proto(chain_model(24, 12)), ReluFailureCheck(), tmp_path / str(enabled)
        )
        run_dir = Path(report.work_directory)
        traces.append(
            (
                (run_dir / "candidates.jsonl").read_bytes(),
                (run_dir / "scheduler.jsonl").read_bytes(),
                report.final_hash,
                report.failure_check_calls,
                report.minimality,
            )
        )
    assert traces[0] == traces[1]


def test_finalizer_acceptance_never_reenters_cursor_or_history(tmp_path, monkeypatch):
    monkeypatch.setattr(StagedScheduler, "no_progress_limit", 0)
    _, report = ReductionEngine(
        config=EngineConfig(
            scheduler="staged",
            search_policy="cursor",
            history_order="defer",
        )
    ).run(OnnxModel.from_proto(chain_model(8, 4)), ReluFailureCheck(), tmp_path)
    rows = [
        json.loads(row)
        for row in (Path(report.work_directory) / "scheduler.jsonl").read_text().splitlines()
    ]
    assert report.scheduler["finalization"]["restarts"] > 0
    assert all(row["stage"] == "finalization" for row in rows)
    assert report.scheduler["history"]["entries"] == 0
    assert report.minimality["status"] == "pass_fixed_point"


def test_independent_finalization_can_change_optimization_options(tmp_path):
    from tests.search.test_staged import SignatureCheck

    _, first = ReductionEngine(
        config=EngineConfig(
            scheduler="staged",
            search_policy="cursor",
            goal="fast",
            model_edit_mode="auto",
            history_order="defer",
            signature_regex=r"BUG:(\w+)",
            max_candidate_check_calls=2,
        )
    ).run(OnnxModel.from_proto(chain_model(8, 4)), SignatureCheck(), tmp_path / "fast")
    _, final = ReductionEngine(
        config=EngineConfig(
            scheduler="staged",
            goal="complete",
            model_edit_mode="ir",
            finalize_from=first.work_directory,
            signature_regex=r"BUG:(\w+)",
        )
    ).run(OnnxModel.load(first.artifact_recheck["model"]), SignatureCheck(), tmp_path / "finish")
    assert final.minimality["status"] == "pass_fixed_point"
    assert final.scheduler["finalize_only"]
    assert final.scheduler["history"] is None


def test_ordinary_custom_pass_keeps_its_protocol_with_an_opaque_adapter(tmp_path):
    from tests.search.test_scheduler import OpaqueModel

    class Custom:
        name = "delete_nodes"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            if model.node_count > 1:
                pytest.fail("accepted opaque bases must regenerate generic passes")

    model, report = ReductionEngine(
        [Custom()],
        config=EngineConfig(
            scheduler="staged",
            search_policy="cursor",
            history_order="defer",
            model_edit_mode="auto",
        ),
    ).run(OpaqueModel(OnnxModel.from_proto(chain_model(8, 7))), ReluFailureCheck(), tmp_path)
    assert model.node_count == 1
    assert report.minimality["status"] == "pass_fixed_point"
    assert "keep-region" not in report.minimality["scope"]


def test_cursor_node_requirement_is_checked_before_backend(tmp_path):
    from onnx_reduce.passes.remove_unused import RemoveUnusedPass
    from tests.search.test_scheduler import OpaqueModel

    class NotCalled:
        def run(self, path):
            pytest.fail("capability check must run before the backend")

    with pytest.raises(ValueError, match="cursor node editing requires"):
        ReductionEngine(
            [RemoveUnusedPass()],
            config=EngineConfig(
                scheduler="staged",
                search_policy="cursor",
            ),
        ).run(OpaqueModel(OnnxModel.from_proto(chain_model())), NotCalled(), tmp_path)


def test_deferred_fine_lookahead_is_rewound_after_acceptance():
    model = OnnxModel.from_proto(chain_model(12, 6))
    scheduler = StagedScheduler(
        [DeleteNodesPass()], model, search_policy="cursor", history_order="defer"
    )
    scheduler._progress.index = 2
    # Reject the first physical edit on an earlier base, forcing lookahead.
    first = replace(
        attempt(model, 0), base_digest="old", candidate=Candidate((DeleteNodes((0, 1)),))
    )
    scheduler._history.observe(first, model, CandidateFeedback("not_interesting", model.cost()))
    offered = scheduler.next_attempt(model)
    assert offered.candidate.edits[0].indices == (2, 3)
    changed = model.apply_candidate(offered.candidate)
    scheduler.observe(offered, CandidateFeedback("accepted", model.cost(), changed.cost()), changed)
    following = scheduler.next_attempt(changed)
    assert following.candidate.edits[0].indices == (0,)


@pytest.mark.parametrize(
    "config",
    [
        {"scheduler": "legacy", "search_policy": "cursor"},
        {"scheduler": "staged", "history_order": "defer"},
        {"scheduler": "stateful", "direction_order": "adaptive"},
    ],
)
def test_incompatible_options_fail_before_backend_calls(config):
    with pytest.raises(ValueError):
        EngineConfig(**config)
