"""Compare two source trees on identical failure cases and wall-clock budgets.

Each run uses a fresh process and the same interpreter, dependencies, checker,
working directory, staged complete settings and combination limits.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    sys.path.insert(0, str(Path(request["source"]) / "src"))
    import onnx_reduce

    if not Path(onnx_reduce.__file__).resolve().is_relative_to(Path(request["source"]).resolve()):
        raise RuntimeError("worker imported the wrong source tree")
    from onnx_reduce.engine import EngineConfig, ReductionEngine
    from onnx_reduce.failure_check.contract import make_failure_check_config
    from onnx_reduce.failure_check.runner import FailureCheckRunner
    from onnx_reduce.model import OnnxModel

    check = make_failure_check_config(
        request["test"],
        timeout_seconds=request["timeout"],
        crash_is_interesting=False,
        timeout_is_interesting=False,
    )
    model = OnnxModel.load(request["model"])
    engine = ReductionEngine(
        config=EngineConfig(
            scheduler="staged",
            goal="complete",
            objective=request["objective"],
            combination_depth=3,
            combination_limit=request["combination_limit"],
            time_budget_seconds=request["budget"],
            collect_performance=True,
            signature_regex=request.get("signature_regex"),
        )
    )
    _, report = engine.run(model, FailureCheckRunner(check), request["run_dir"])
    data = report.to_dict()
    if data["artifact_recheck"]["outcome"] != "interesting":
        raise RuntimeError("final output no longer reproduces the target")
    Path(request["result"]).write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--current-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budgets", default="60,120,300")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--combination-limit", type=int, default=1024)
    parser.add_argument("--objective", choices=("nodes", "bytes", "tensors"), default="nodes")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.manifest or not args.baseline_source or not args.out:
        parser.error("--manifest, --baseline-source and --out are required")
    budgets = [float(b) for b in args.budgets.split(",")]
    if not budgets or any(b <= 0 for b in budgets):
        parser.error("budgets must be positive")
    sources = {"baseline": args.baseline_source.resolve(), "current": args.current_source.resolve()}
    for source in sources.values():
        if not (source / "src/onnx_reduce").is_dir():
            parser.error(f"source tree missing: {source}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.manifest.read_text())
    results = []
    for index, case in enumerate(cases):
        model = Path(case["model"])
        if not model.is_absolute():
            model = args.manifest.resolve().parent / model
        for budget_index, budget in enumerate(budgets):
            # Alternate execution order to reduce systematic warm-cache bias.
            order = list(sources) if (index + budget_index) % 2 == 0 else list(reversed(sources))
            for label in order:
                directory = out / f"case-{index:03d}" / str(budget) / label
                directory.mkdir(parents=True, exist_ok=True)
                request = {
                    "source": str(sources[label]),
                    "test": case["test"],
                    "model": str(model.resolve()),
                    "timeout": args.timeout,
                    "objective": args.objective,
                    "combination_limit": args.combination_limit,
                    "budget": budget,
                    "run_dir": str(directory / "work"),
                    "result": str(directory / "result.json"),
                    "signature_regex": case.get("signature_regex"),
                }
                request_path = directory / "request.json"
                request_path.write_text(json.dumps(request, indent=2) + "\n")
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                    capture_output=True,
                    text=True,
                )
                (directory / "stdout.log").write_text(completed.stdout)
                (directory / "stderr.log").write_text(completed.stderr)
                if completed.returncode:
                    raise RuntimeError(f"{case['name']}/{budget}/{label} failed; see {directory}")
                report = json.loads(Path(request["result"]).read_text())
                results.append(
                    {
                        "case": case["name"],
                        "budget_seconds": budget,
                        "source": label,
                        "result": report,
                    }
                )
                (out / "summary.json").write_text(
                    json.dumps(
                        {"sources": {k: str(v) for k, v in sources.items()}, "runs": results},
                        indent=2,
                    )
                    + "\n"
                )
    print(out / "summary.json")


if __name__ == "__main__":
    main()
