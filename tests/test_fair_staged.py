"""Later strategies get real execution before heuristic plateau/finalization."""

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.model import OnnxModel
from onnx_reduce.search_types import CandidateFeedback
from onnx_reduce.staged import StagedScheduler
from tests.helpers import chain_model


class Rejecting:
    name = "rejecting"

    def generate(self, model):
        for _ in range(1000):
            yield Candidate(())


class Later:
    name = "later"

    def generate(self, model):
        yield Candidate((DeleteNodes((0,)),))


def test_later_pass_is_reached_before_first_pass_plateau():
    model = OnnxModel.from_proto(chain_model(80, 40))
    scheduler = StagedScheduler([Rejecting(), Later()], model)
    attempts = []
    for _ in range(9):
        attempt = scheduler.next_attempt(model)
        attempts.append(attempt)
        scheduler.observe(attempt, CandidateFeedback("not_interesting", model.cost()))
    assert attempts[-1].reduction_pass.name == "later"
    assert all(a.stage == "fine" for a in attempts)


def test_acceptance_does_not_reset_turn_and_starve_other_pass():
    model = OnnxModel.from_proto(chain_model(80, 40))
    scheduler = StagedScheduler([Rejecting(), Later()], model)
    for _ in range(8):
        attempt = scheduler.next_attempt(model)
        changed = model.apply_candidate(Candidate((DeleteNodes((0,)),)))
        scheduler.observe(
            attempt, CandidateFeedback("accepted", model.cost(), changed.cost()), changed
        )
        model = changed
    assert scheduler.next_attempt(model).reduction_pass.name == "later"


def test_deep_search_precedes_proof_and_still_requires_current_coverage():
    model = OnnxModel.from_proto(chain_model(3, 1))
    scheduler = StagedScheduler([Later()], model)
    stages = []
    while (attempt := scheduler.next_attempt(model)) is not None:
        stages.append(attempt.stage)
        scheduler.observe(attempt, CandidateFeedback("not_interesting", model.cost()))
    assert stages == ["fine", "deep", "finalization"]
    assert scheduler.coverage().completed
    assert scheduler.to_dict()["pass_states"]["coarse/later"] == "no_candidates"
