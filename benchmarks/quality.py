"""Compare original deletion passes and extended reduction on identical checks.

Run with --manifest to evaluate actual compiler failures. Built-in fixtures use
simulated conditions, and must never be described as real compiler regressions.
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import onnx

from benchmarks.quality_cases import SimulatedFailure, quality_cases
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult, make_failure_check_config
from onnx_reduce.failure_check.runner import FailureCheckRunner
from onnx_reduce.model import OnnxModel
from onnx_reduce.passes.delete_nodes import DeleteNodesPass
from onnx_reduce.passes.drop_outputs import DropOutputsPass
from onnx_reduce.passes.remove_unused import RemoveUnusedPass


def run_case(name, source, runner_factory, directory, config, signature=None):
    runs = {}
    for label in ("original_passes", "extended"):
        passes = (
            (DropOutputsPass(), RemoveUnusedPass(), DeleteNodesPass())
            if label == "original_passes"
            else None
        )
        runner = runner_factory()
        _, report = ReductionEngine(
            passes=passes, config=EngineConfig(**config, signature_regex=signature)
        ).run(source, runner, directory / label)
        if report.artifact_recheck["outcome"] != "interesting":
            raise RuntimeError(f"{name}/{label}: final failure check failed")
        runs[label] = report.to_dict()
    return {"name": name, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--manifest", type=Path, help="JSON list of real {name,model,test,signature_regex?}"
    )
    sources.add_argument(
        "--architecture-zoo",
        type=Path,
        help="directory containing model_zoo_real.py and conditions.py",
    )
    parser.add_argument("--combination-limit", type=int, default=32)
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--time-budget", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--objective", choices=("nodes", "bytes", "tensors"), default="nodes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "objective": args.objective,
        "max_candidate_check_calls": args.max_calls,
        "time_budget_seconds": args.time_budget,
        "combination_limit": args.combination_limit,
    }
    results = []
    if args.architecture_zoo:
        sys.path.insert(0, str(args.architecture_zoo.resolve()))
        from conditions import check_condition
        from model_zoo_real import CASES_REAL, build_real_case

        class ZooFailure:
            def __init__(self, spec):
                self.spec = spec

            def run(self, path):
                return (
                    FailureCheckResult.INTERESTING
                    if check_condition(self.spec, path)
                    else FailureCheckResult.NOT_INTERESTING
                )

        for name, case in CASES_REAL.items():
            proto = build_real_case(name)
            if case.external_data:
                model_path = args.out / name / "source" / "model.onnx"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                onnx.save_model(
                    proto,
                    model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.bin",
                    size_threshold=1024,
                )
                source = OnnxModel.load(model_path)
            else:
                source = OnnxModel.from_proto(proto)
            results.append(
                run_case(
                    name,
                    source,
                    lambda spec=case.condition: ZooFailure(spec),
                    args.out / name,
                    config,
                )
            )
    elif args.manifest:
        cases = json.loads(args.manifest.read_text())
        for index, case in enumerate(cases):
            path = Path(case["model"])
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            source = OnnxModel.load(path)
            check = make_failure_check_config(
                case["test"],
                timeout_seconds=args.timeout,
                crash_is_interesting=False,
                timeout_is_interesting=False,
            )
            results.append(
                run_case(
                    case["name"],
                    source,
                    lambda check=check: FailureCheckRunner(check),
                    args.out / f"case-{index:03}",
                    config,
                    case.get("signature_regex"),
                )
            )
    else:
        for name, proto, ops in quality_cases():
            results.append(
                run_case(
                    name,
                    OnnxModel.from_proto(proto),
                    lambda ops=ops, name=name: SimulatedFailure(
                        ops, paired=name == "paired_bypass"
                    ),
                    args.out / name,
                    config,
                )
            )
    payload = {
        "condition_source": (
            "user_manifest"
            if args.manifest
            else "architecture_simulated"
            if args.architecture_zoo
            else "simulated"
        ),
        "baseline": "original three passes under the same engine, metrics and budgets",
        "config": config,
        "cases": results,
    }
    output = args.out / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
