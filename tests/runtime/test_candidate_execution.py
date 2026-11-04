"""Candidate execution owns cache decisions independently of evidence writers."""

import json
from pathlib import Path

import pytest

from onnx_reduce.config import EngineConfig, ExecutionConfig
from onnx_reduce.edits import Candidate, DeleteNodes, DropInputs
from onnx_reduce.engine import ReductionEngine
from onnx_reduce.failure_check.evaluator import Evaluator
from onnx_reduce.model import OnnxModel
from onnx_reduce.preparation import PreparationCaches, preparation_caches
from onnx_reduce.runtime.budget import SearchBudget
from onnx_reduce.runtime.candidate import CandidateExecutor
from onnx_reduce.runtime.reporting import record_candidate
from tests.helpers import chain_model
from tests.integration.test_invariants import RejectCandidates


@pytest.mark.parametrize("write_evidence", [True, False])
def test_rejection_cache_does_not_depend_on_candidate_logging(
    tmp_path, monkeypatch, write_evidence
):
    class RepeatedRejections:
        name = "repeated_rejections"

        def generate(self, model):
            for edit in (DropInputs(("missing",)), DeleteNodes((2,))):
                yield Candidate((edit,))
                yield Candidate((edit,))

    if not write_evidence:
        monkeypatch.setattr("onnx_reduce.runtime.reporting.record_candidate", lambda *a, **k: None)

    _, report = ReductionEngine([RepeatedRejections()]).run(
        OnnxModel.from_proto(chain_model(3, 1)), RejectCandidates(), tmp_path
    )
    records = [
        json.loads(line)
        for line in (Path(report.work_directory) / "scheduler.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in records] == [
        "edit_failed",
        "edit_failed",
        "not_interesting",
        "not_interesting",
    ]
    assert [r["cache_source"] for r in records] == [None, "edit", None, "edit"]
    assert report.failure_check_calls == 3  # Initial, one candidate, final recheck.
    assert report.cache_hits == 1


def test_executor_uses_supplied_cache_instead_of_ambient_cache(tmp_path):
    class BadInputPass:
        name = "bad_input"

    model = OnnxModel.from_proto(chain_model())
    supplied = PreparationCaches(True, "strict")
    executor = CandidateExecutor(
        config=ExecutionConfig(),
        validation_mode="strict",
        evaluator=Evaluator(RejectCandidates(), tmp_path),
        run_dir=tmp_path,
        caches=supplied,
        budget=SearchBudget(0, None, None),
    )
    candidate = Candidate((DropInputs(("missing",)),), base_digest=model.canonical_hash())
    with preparation_caches() as ambient:
        first = executor.execute(model, candidate)
        second = executor.execute(model, candidate)
        assert len(ambient.edits) == 0
    assert len(supplied.edits) == 1
    assert first.feedback.status == second.feedback.status == "edit_failed"
    assert first.feedback.cache_source is None
    assert second.feedback.cache_source == "edit"


def test_writing_candidate_evidence_does_not_populate_execution_cache(tmp_path):
    candidate = Candidate((DropInputs(("missing",)),), base_digest="base")
    with preparation_caches() as caches:
        record_candidate(tmp_path, candidate, "edit_failed", reason="missing input")
        assert len(caches.edits) == 0
    row = json.loads((tmp_path / "candidates.jsonl").read_text())
    assert row["candidate"] == json.loads(json.dumps(candidate.to_dict()))
    assert row["reason"] == "missing input"


def test_reporting_is_not_required_for_candidate_call_quota(tmp_path, monkeypatch):
    class DistinctCandidates:
        name = "distinct"

        def generate(self, model):
            yield Candidate((DeleteNodes((0,)),))
            yield Candidate((DeleteNodes((1,)),))

    class CountChecks(RejectCandidates):
        calls = 0

        def run(self, path):
            self.calls += 1
            return super().run(path)

    monkeypatch.setattr("onnx_reduce.runtime.engine.record_attempt", lambda *a, **k: None)
    runner = CountChecks()
    _, report = ReductionEngine(
        [DistinctCandidates()], config=EngineConfig(max_candidate_check_calls=1)
    ).run(OnnxModel.from_proto(chain_model(3, 1)), runner, tmp_path)
    assert runner.calls == 3  # Initial, exactly one candidate, mandatory final.
    assert report.stop_reason == "budget_exhausted"
    assert report.candidates_generated == 0  # The observer was disabled.
