from itertools import islice

from onnx_reduce.edits import BypassNode, Candidate, DropNodeAttribute, ShrinkDimension
from onnx_reduce.passes.combinations import CombinationsPass, balanced_groups, spread_indices


class LargeGraph:
    node_count = 350

    def rewrite_candidates(self, kind):
        if kind == "bypass":
            for i in range(self.node_count):
                yield Candidate((BypassNode(i, 0),))
        elif kind == "attributes":
            for i in range(self.node_count):
                yield Candidate((DropNodeAttribute(i, "alpha"),))
        elif kind == "tensors":
            yield Candidate((ShrinkDimension(64, 1),))


def test_depth_three_and_tail_are_not_starved_by_pairs():
    candidates = list(CombinationsPass(3, 32).generate(LargeGraph()))
    groups = [c.edits[0].indices for c in candidates if hasattr(c.edits[0], "indices")]
    assert any(len(g) == 2 for g in groups)
    assert any(len(g) == 3 for g in groups)
    assert any(max(g) == 349 for g in groups)
    assert len(candidates) <= 4 * 32
    assert [c.to_dict() for c in candidates] == [
        c.to_dict() for c in CombinationsPass(3, 32).generate(LargeGraph())
    ]


def test_tensor_partners_survive_many_attributes():
    candidates = list(CombinationsPass(3, 16).generate(LargeGraph()))
    assert any(isinstance(c.edits[0], ShrinkDimension) for c in candidates)
    assert any(isinstance(c.edits[0], DropNodeAttribute) for c in candidates)
    assert any(isinstance(c.edits[0], BypassNode) and c.edits[0].index == 349 for c in candidates)


def test_tail_triple_and_unique_bounded_sampling():
    groups = list(balanced_groups(350, 3, 1024))
    assert (345, 347, 349) in groups
    assert len(groups) == len(set(groups)) <= 1024
    assert list(islice(spread_indices(350), 2)) == [0, 349]
    assert sorted(spread_indices(350)) == list(range(350))


def test_three_tail_deletions_reproduce_when_all_smaller_groups_fail(tmp_path):
    import onnx

    from onnx_reduce.engine import EngineConfig, ReductionEngine
    from onnx_reduce.failure_check.contract import FailureCheckResult
    from onnx_reduce.model import OnnxModel
    from tests.helpers import chain_model

    proto = chain_model(80, 40)
    for i, node in enumerate(proto.graph.node):
        node.name = f"protected_{i}"
    targets = {"protected_75", "protected_77", "protected_79"}
    required = {n.name for n in proto.graph.node} - targets

    class Check:
        def run(self, path):
            present = {n.name for n in onnx.load(path).graph.node}
            keep = required <= present and len(present & targets) in {0, 3}
            return FailureCheckResult.INTERESTING if keep else FailureCheckResult.NOT_INTERESTING

    reduced, _ = ReductionEngine(
        passes=[CombinationsPass(3, 256)], config=EngineConfig(max_candidate_check_calls=300)
    ).run(OnnxModel.from_proto(proto), Check(), tmp_path)
    assert reduced.node_count == 77
    assert not targets & {n.name for n in reduced.to_proto().graph.node}
    assert reduced.validate("strict").valid
