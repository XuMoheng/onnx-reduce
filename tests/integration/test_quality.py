"""Shape/weight coordination, body interfaces, objectives and atomic search."""

import numpy as np
import onnx
import pytest
from onnx import helper as h
from onnx import numpy_helper

from benchmarks.quality_cases import SimulatedFailure, model, quality_cases, value
from onnx_reduce.edits import Candidate, ShrinkDimension
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.model import OnnxModel
from onnx_reduce.model.operations.graphs import walk_graphs
from onnx_reduce.passes import default_passes
from onnx_reduce.passes.combinations import CombinationsPass
from onnx_reduce.search.coverage import Finalizer
from onnx_reduce.types import EditApplicationError, ModelCost


@pytest.mark.parametrize("scheduler", ["legacy", "stateful", "staged"])
@pytest.mark.parametrize(
    "name,proto,ops", list(quality_cases()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_architectural_quality(tmp_path, scheduler, name, proto, ops):
    source = OnnxModel.from_proto(proto)
    assert source.validate("strict").valid
    runner = SimulatedFailure(ops, paired=name == "paired_bypass")
    reduced, report = ReductionEngine(
        config=EngineConfig(scheduler=scheduler, combination_limit=16)
    ).run(source, runner, tmp_path)
    assert reduced.validate("strict").valid
    assert report.minimality["status"] == "pass_fixed_point"
    assert report.final.key() < report.original.key()
    assert report.final.tensor_elements < report.original.tensor_elements
    assert report.artifact_recheck["outcome"] == "interesting"
    if name in {"if", "loop", "scan"}:
        assert report.final.node_count < report.original.node_count
        old_graphs = list(walk_graphs(source.to_proto()))
        new_graphs = list(walk_graphs(reduced.to_proto()))
        assert len(old_graphs) == len(new_graphs)
        for (path, before), (_, after) in zip(old_graphs, new_graphs, strict=True):
            if path:
                assert len(before.input) == len(after.input)
                assert len(before.output) == len(after.output)
    if name == "paired_bypass":
        assert reduced.op_types == ("Relu", "Sigmoid")
        assert report.pass_stats["combinations"].accepted > 0


def test_resize_weights_and_shape_vectors_preserves_original():
    cases = {name: proto for name, proto, _ in quality_cases()}
    source = OnnxModel.from_proto(cases["matmul"])
    resized = source.apply_candidate(Candidate((ShrinkDimension(4, 2),)))
    weights = resized.to_proto().graph.initializer[0]
    assert list(weights.dims) == [2, 2]
    np.testing.assert_array_equal(numpy_helper.to_array(weights), [[0, 1], [4, 5]])
    assert list(source.to_proto().graph.initializer[0].dims) == [4, 4]
    source = OnnxModel.from_proto(cases["reshape"])
    resized = source.apply_candidate(Candidate((ShrinkDimension(4, 2),)))
    np.testing.assert_array_equal(
        numpy_helper.to_array(resized.to_proto().graph.initializer[0]), [2, 2]
    )
    assert resized.validate("strict").valid


def test_unnamed_constant_shape_tensor_does_not_resize_other_constants():
    proto = model(
        [
            h.make_node(
                "Constant",
                [],
                ["shape"],
                value=numpy_helper.from_array(np.array([4, 4], dtype=np.int64)),
            ),
            h.make_node(
                "Constant",
                [],
                ["unused"],
                value=numpy_helper.from_array(np.ones((4, 4), dtype=np.float32)),
            ),
            h.make_node("Reshape", ["x", "shape"], ["y"]),
        ],
        [value("x")],
        [value("y")],
    )
    resized = OnnxModel.from_proto(proto).apply_candidate(Candidate((ShrinkDimension(4, 2),)))
    assert resized.validate("strict").valid
    nodes = resized.to_proto().graph.node
    np.testing.assert_array_equal(numpy_helper.to_array(nodes[0].attribute[0].t), [2, 2])
    assert list(nodes[1].attribute[0].t.dims) == [2, 2]


def test_zero_symbolic_and_scalar_dimensions():
    proto = model(
        [h.make_node("Identity", ["x"], ["y"])],
        [value("x", ("batch", 0, 4))],
        [value("y", ("batch", 0, 4))],
    )
    source = OnnxModel.from_proto(proto)
    resized = source.apply_candidate(Candidate((ShrinkDimension(4, 1),)))
    dims = resized.to_proto().graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "batch" and dims[1].dim_value == 0
    assert resized.validate("strict").valid
    with pytest.raises(EditApplicationError):
        source.apply_candidate(Candidate((ShrinkDimension(4, 0),)))


def test_objective_keys_and_recursive_cost():
    a, b = ModelCost(1, 100, tensor_elements=10), ModelCost(2, 50, tensor_elements=5)
    assert a.key("nodes") < b.key("nodes")
    assert b.key("bytes") < a.key("bytes")
    assert b.key("tensors") < a.key("tensors")
    proto = next(p for name, p, _ in quality_cases() if name == "if")
    m = OnnxModel.from_proto(proto)
    assert m.node_count == 1
    assert m.cost().node_count == 5


def test_combination_bounds_and_noncontiguous_pairs():
    proto = next(p for name, p, _ in quality_cases() if name == "paired_bypass")
    m = OnnxModel.from_proto(proto)
    candidates = list(CombinationsPass(3, 2).generate(m))
    assert len(candidates) <= 8  # Four families, two candidates each.
    assert list(CombinationsPass(1, 256).generate(m)) == []
    assert list(CombinationsPass(3, 0).generate(m)) == []
    full = list(CombinationsPass(2, 256).generate(m))
    assert any(getattr(c.edits[0], "indices", ()) == (0, 2) for c in full)


def test_finalizer_covers_new_passes():
    proto = next(p for name, p, _ in quality_cases() if name == "if")
    m = OnnxModel.from_proto(proto)
    finalizer = Finalizer(default_passes(combination_limit=2), m)
    names = set()
    while (attempt := finalizer.next_attempt(m)) is not None:
        names.add(attempt.reduction_pass.name)
    assert {"shrink_tensors", "reduce_subgraphs", "combinations"} <= names


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "unknown"},
        {"combination_depth": 0},
        {"combination_depth": 4},
        {"combination_limit": -1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


@pytest.mark.parametrize("objective", ["nodes", "bytes", "tensors"])
def test_engine_honors_objective(tmp_path, objective):
    proto = next(p for name, p, _ in quality_cases() if name == "reshape")
    source = OnnxModel.from_proto(proto)
    _, report = ReductionEngine(config=EngineConfig(objective=objective)).run(
        source, SimulatedFailure({"Reshape"}), tmp_path
    )
    assert report.final.key(objective) <= report.original.key(objective)
    assert report.minimality["objective"] == objective
    if objective == "bytes":
        assert report.final.byte_size <= report.original.byte_size
    else:
        assert report.final.tensor_elements < report.original.tensor_elements


def test_resize_that_loses_failure_is_rejected(tmp_path):
    class Check:
        def run(self, path):
            from onnx_reduce.failure_check.contract import FailureCheckResult

            proto = onnx.load(path)
            keep = any(n.op_type == "MatMul" for n in proto.graph.node) and any(
                list(t.dims) == [4, 4] for t in proto.graph.initializer
            )
            return FailureCheckResult.INTERESTING if keep else FailureCheckResult.NOT_INTERESTING

    proto = next(p for name, p, _ in quality_cases() if name == "matmul")
    _, report = ReductionEngine().run(OnnxModel.from_proto(proto), Check(), tmp_path)
    assert report.pass_stats["shrink_tensors"].accepted == 0
    assert report.final.tensor_elements == report.original.tensor_elements


def test_real_manifest_benchmark_entrypoint(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.onnx"
    onnx.save(
        model([h.make_node("Relu", ["x"], ["y"])], [value("x", (1,))], [value("y", (1,))]), source
    )
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "demo",
                    "model": "source.onnx",
                    "test": [sys.executable, str(root / "examples/relu/check.py")],
                    "signature_regex": "assertion: ([A-Za-z0-9_]+)",
                }
            ]
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/quality.py"),
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "results"),
            "--max-calls",
            "10",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["condition_source"] == "user_manifest"
    assert all(
        r["artifact_recheck"]["outcome"] == "interesting"
        for r in summary["cases"][0]["runs"].values()
    )


def test_strategy_compare_runs_both_sources_with_equal_budgets(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.onnx"
    onnx.save(model([h.make_node("Relu", ["x"], ["y"])], [value("x")], [value("y")]), source)
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "demo",
                    "model": "source.onnx",
                    "test": [sys.executable, str(root / "examples/relu/check.py")],
                }
            ]
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/strategy_compare.py"),
            "--manifest",
            str(manifest),
            "--baseline-source",
            str(root),
            "--budgets",
            "0.01",
            "--out",
            str(tmp_path / "compare"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "compare/summary.json").read_text())
    assert {run["source"] for run in summary["runs"]} == {"baseline", "current"}
    assert all(run["budget_seconds"] == 0.01 for run in summary["runs"])
    assert all(
        run["result"]["artifact_recheck"]["outcome"] == "interesting" for run in summary["runs"]
    )
