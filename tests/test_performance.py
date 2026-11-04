from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from onnx_reduce import performance
from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs
from onnx_reduce.engine import EngineConfig, InitialNotInteresting, ReductionEngine
from onnx_reduce.model import OnnxModel
from onnx_reduce.performance import collect_performance, measure
from onnx_reduce.storage import atomic_write, write_text
from tests.helpers import chain_model
from tests.test_engine import NeverInteresting, ReluFailureCheck


def test_nested_timers_count_exclusive_time_and_exceptions(monkeypatch):
    ticks = iter([0, 1, 2, 5, 8, 10])
    monkeypatch.setattr(performance, "perf_counter", lambda: next(ticks))
    with collect_performance(True) as stats:
        with measure("outer"):
            with pytest.raises(ValueError), measure("inner"):
                raise ValueError("measured failures still count")
    assert stats.stages["outer"] == {
        "calls": 1,
        "inclusive_seconds": 7,
        "exclusive_seconds": 4,
    }
    assert stats.stages["inner"]["inclusive_seconds"] == 3
    assert stats.to_dict()["elapsed_seconds"] == 10
    assert performance.current_performance() is None


def test_byte_counter_includes_repeated_atomic_and_utf8_append_writes(tmp_path):
    with collect_performance(True) as stats:
        atomic_write(tmp_path / "model", b"abc")
        atomic_write(tmp_path / "model", b"d")
        write_text(tmp_path / "events", "中文\n", append=True)
    assert stats.counters["written_bytes"] == 11
    assert stats.counters["write_calls"] == 3


@pytest.mark.parametrize("budget", [None, 0, 4])
@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
def test_instrumentation_preserves_ordered_decisions_and_final_artifact(
    tmp_path, budget, scheduler
):
    reports = []
    traces = []
    for enabled in (False, True):
        model = OnnxModel.from_proto(chain_model(12, 6))
        _, report = ReductionEngine(
            config=EngineConfig(
                collect_performance=enabled,
                max_candidate_check_calls=budget,
                scheduler=scheduler,
            )
        ).run(model, ReluFailureCheck(), tmp_path / str(enabled))
        reports.append(report.to_dict())
        trace = Path(report.work_directory) / "candidates.jsonl"
        traces.append(trace.read_bytes() if trace.exists() else b"")
    assert traces[0] == traces[1]
    schedules = [Path(report["work_directory"]) / "scheduler.jsonl" for report in reports]
    schedule_traces = [path.read_bytes() if path.exists() else b"" for path in schedules]
    assert schedule_traces[0] == schedule_traces[1]
    assert reports[0]["scheduler"]["coverage"] == reports[1]["scheduler"]["coverage"]
    for key in (
        "final_hash",
        "final",
        "candidates",
        "passes",
        "failure_checks",
        "stop_reason",
        "minimality",
    ):
        assert reports[0][key] == reports[1][key]
    assert "performance" not in reports[0]
    profile = reports[1]["performance"]
    assert profile["peak_rss_bytes"] > 0
    assert profile["stages"]["external_check"]["calls"] == reports[1]["failure_checks"]["calls"]
    assert profile["stages"]["checkpoint"]["calls"] == reports[1]["candidates"]["accepted"] + 2
    assert (
        sum(s["exclusive_seconds"] for s in profile["stages"].values())
        <= profile["elapsed_seconds"]
    )
    assert (
        json.loads((Path(reports[1]["work_directory"]) / "report.json").read_text()) == reports[1]
    )
    if budget is None:
        assert profile["stages"]["shape_inference"]["calls"] > 0


def test_cache_and_rejections_are_measured_without_extra_checks(tmp_path):
    class RepeatedRejectedCandidates:
        name = "repeated"

        def generate(self, model):
            yield Candidate((DropInputs(("unknown",)),))
            yield Candidate(())
            yield Candidate((DeleteNodes((2,)),))
            yield Candidate((DeleteNodes((2,)),))

    _, report = ReductionEngine(
        [RepeatedRejectedCandidates()],
        config=EngineConfig(collect_performance=True),
    ).run(OnnxModel.from_proto(chain_model(5, 2)), ReluFailureCheck(), tmp_path)
    counters = report.performance.counters
    assert counters["candidate_status.edit_failed"] == 1
    assert counters["candidate_status.not_cheaper"] == 1
    assert counters["edit_cache_hits"] == 1
    assert counters["content_cache_hits"] == 0
    assert report.failure_check_calls == 3  # initial, one candidate, final
    assert report.performance.samples["candidate_selected_items"]["count"] == 4


def test_failed_run_does_not_leak_measurements_into_next_run(tmp_path):
    engine = ReductionEngine(config=EngineConfig(collect_performance=True))
    with pytest.raises(InitialNotInteresting):
        engine.run(OnnxModel.from_proto(chain_model()), NeverInteresting(), tmp_path)
    assert performance.current_performance() is None
    _, report = engine.run(OnnxModel.from_proto(chain_model()), ReluFailureCheck(), tmp_path)
    assert report.performance.stages["external_check"]["calls"] == report.failure_check_calls


def test_cli_performance_flag(tmp_path):
    from onnx_reduce.cli import main

    model = tmp_path / "original.onnx"
    OnnxModel.from_proto(chain_model()).save(model)
    report = tmp_path / "report.json"
    import shlex
    import sys

    assert (
        main(
            [
                str(model),
                "--test",
                f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(0)'",
                "-o",
                str(tmp_path / "reduced.onnx"),
                "--max-candidate-check-calls",
                "0",
                "--performance",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text())["performance"]["stages"]["external_check"]["calls"] == 2


def test_shape_inference_failure_is_counted_and_preserves_decisions(tmp_path, monkeypatch):
    import onnx

    def fail_inference(*args, **kwargs):
        raise RuntimeError("synthetic shape inference failure")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail_inference)
    traces = []
    for enabled in (False, True):
        _, report = ReductionEngine(config=EngineConfig(collect_performance=enabled)).run(
            OnnxModel.from_proto(chain_model(5, 2)),
            ReluFailureCheck(),
            tmp_path / str(enabled),
        )
        traces.append((Path(report.work_directory) / "candidates.jsonl").read_bytes())
        if enabled:
            assert report.performance.counters["shape_inference_failures"] > 0
            assert report.performance.counters["candidate_status.invalid"] > 0
    assert traces[0] == traces[1]


def test_progress_tracks_initial_accepted_and_final_without_changing_outcome(tmp_path):
    reduced, report = ReductionEngine(
        config=EngineConfig(scheduler="staged", collect_performance=True)
    ).run(OnnxModel.from_proto(chain_model(12, 6)), ReluFailureCheck(), tmp_path)
    progress = report.to_dict()["performance"]["progress"]
    assert progress[0]["reason"] == "initial"
    assert progress[-1]["stage"] == "final"
    assert progress[-1]["cost"]["node_count"] == reduced.cost().node_count
    assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in pairwise(progress))
    assert len(progress) == report.candidates_accepted + 2
    logged = [
        json.loads(line)
        for line in (Path(report.work_directory) / "progress.jsonl").read_text().splitlines()
    ]
    assert logged == progress
    assert report.minimality["status"] == "pass_fixed_point"


def test_progress_compaction_preserves_endpoints_and_bounds_memory():
    from onnx_reduce.types import ModelCost

    stats = performance.PerformanceStats()
    for i in range(3000):
        stats.record_progress(
            ModelCost(3000 - i, 1), stage="fine", reason="accepted", checks=i, force=True
        )
    assert len(stats.progress) <= 2048
    assert stats.progress[0]["cost"]["node_count"] == 3000
    assert stats.progress[-1]["cost"]["node_count"] == 1
    assert stats.progress_compactions > 0
