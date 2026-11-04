"""Compare source revisions on deterministic decisions and synthetic quality.

Uses each revision's own unchanged fixtures and a fresh process per run. Calls
and unbounded sweeps require identical evidence; time runs report median quality
and preparation cost. No result is evidence of a real backend failure.
"""

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path


def trace(path):
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def worker(request):
    source = Path(request["source"]).resolve()
    sys.path[:0] = [str(source / "src"), str(source)]
    import onnx_reduce

    assert Path(onnx_reduce.__file__).resolve().is_relative_to(source)
    from benchmarks.quality_cases import SimulatedFailure, quality_cases
    from benchmarks.reduction import NodeCountCheck, ReluCheck, RepeatedDeletionPass, make_model
    from onnx_reduce.engine import EngineConfig, ReductionEngine
    from onnx_reduce.model import OnnxModel

    name = request["case"]
    passes = None
    if name in {"chain", "gated", "initializers", "preparation"}:
        size = request["size"]
        proto = make_model(name, size)
        runner = NodeCountCheck(size) if name == "preparation" else ReluCheck(name, size)
        if name == "preparation":
            passes = [RepeatedDeletionPass()]
    else:
        _, proto, ops = next(item for item in quality_cases() if item[0] == name)
        runner = SimulatedFailure(ops, paired=name == "paired_bypass")
    _, report = ReductionEngine(passes, config=EngineConfig(**request["config"])).run(
        OnnxModel.from_proto(proto), runner, request["out"]
    )
    data = report.to_dict()
    assert data["artifact_recheck"]["outcome"] == "interesting"
    directory = Path(report.work_directory)
    # Timing and absolute artifact paths are intentionally outside this identity.
    decisions = {
        "candidates": trace(directory / "candidates.jsonl"),
        "accepted": trace(directory / "accepted.jsonl"),
        "schedule": trace(directory / "scheduler.jsonl"),
        **{
            key: data[key]
            for key in (
                "final",
                "final_hash",
                "stop_reason",
                "minimality",
                "passes",
                "failure_checks",
            )
        },
        "coverage": data["scheduler"]["coverage"],
    }
    stages = data["performance"]["stages"]
    preparation = sum(
        value["exclusive_seconds"] for key, value in stages.items() if key != "external_check"
    )
    Path(request["result"]).write_text(
        json.dumps(
            {
                "decisions": decisions,
                "elapsed": data["performance"]["elapsed_seconds"],
                "preparation": preparation,
                "candidates": data["candidates"]["generated"],
                "report": str(directory / "report.json"),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--current-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--time-budget", type=float, default=1)
    parser.add_argument("--cases", help="optional comma-separated fixture filter")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--experimental-only",
        action="store_true",
        help="compare staged cursor/order/cache-control variants",
    )
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    if not args.baseline_source or not args.out:
        parser.error("--baseline-source and --out are required")
    if args.repeat < 1 or args.time_budget <= 0:
        parser.error("repeat and time-budget must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {"baseline": args.baseline_source.resolve(), "current": args.current_source.resolve()}
    results = []
    # Calls plus complete sweeps include all six architectural fixtures, cached
    # rejections, lazy edits and all three schedulers. Zero quotas are in pytest.
    scenarios = [
        (name, "calls")
        for name in (
            "matmul",
            "reshape",
            "if",
            "loop",
            "scan",
            "paired_bypass",
            "chain",
            "gated",
            "initializers",
            "preparation",
        )
    ] + [("paired_bypass", "complete")]
    scenarios += [(name, "time") for name in ("chain", "gated", "preparation")]
    if args.cases:
        selected = set(args.cases.split(","))
        if selected - {name for name, _ in scenarios}:
            parser.error("unknown cases")
        scenarios = [(case, budget) for case, budget in scenarios if case in selected]
    for case, budget in scenarios:
        combinations = (
            (("staged", "ir", True), ("staged", "auto", False))
            if args.experimental_only
            else itertools.product(("legacy", "stateful", "staged"), ("ir", "auto"), (True, False))
        )
        for scheduler, edit_mode, cache in combinations:
            if args.experimental_only and budget == "time":
                continue
            # Time measurements focus on standard caching; cache-off behavior
            # has exact-decision comparisons in the calls/complete matrix.
            if budget == "time" and not cache:
                continue
            config = {
                "scheduler": scheduler,
                "model_edit_mode": edit_mode,
                "cache_enabled": cache,
                "combination_limit": 16,
                "collect_performance": True,
                "max_candidate_check_calls": 16 if budget == "calls" else None,
                "time_budget_seconds": args.time_budget if budget == "time" else None,
            }
            if args.experimental_only:
                config.update(
                    search_policy="cursor",
                    history_order="defer",
                    direction_order="adaptive",
                    checkpoint_reuse=False,
                    validation_cache_enabled=False,
                )
            samples = []
            for repeat in range(args.repeat if budget == "time" else 1):
                pair = {}
                order = (
                    list(sources) if (len(results) + repeat) % 2 == 0 else list(reversed(sources))
                )
                for label in order:
                    directory = (
                        args.out
                        / f"{case}-{budget}-{scheduler}-{edit_mode}-{cache}-{repeat}"
                        / label
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    request = {
                        "source": str(sources[label]),
                        "case": case,
                        "size": 256 if budget == "time" else 32,
                        "config": config,
                        "out": str(directory / "work"),
                        "result": str(directory / "result.json"),
                    }
                    request_path = directory / "request.json"
                    request_path.write_text(json.dumps(request))
                    with (directory / "worker.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                str(request_path),
                            ],
                            check=True,
                            stdout=log,
                            stderr=log,
                        )
                    pair[label] = json.loads(Path(request["result"]).read_text())
                if (
                    budget != "time"
                    and pair["baseline"]["decisions"] != pair["current"]["decisions"]
                ):
                    raise AssertionError(f"decision regression: {directory.parent}")
                samples.append(pair)
            results.append({"case": case, "budget": budget, "config": config, "samples": samples})
            print(f"{len(results)} {case}/{budget}/{scheduler}/{edit_mode}/{cache}", flush=True)
    summary = {
        "condition_source": "simulated",
        "sources": {key: str(value) for key, value in sources.items()},
        "time_budget": args.time_budget,
        "repeat": args.repeat,
        "equal_decision_pairs": sum(
            len(row["samples"]) for row in results if row["budget"] != "time"
        ),
        "time_pairs": sum(len(row["samples"]) for row in results if row["budget"] == "time"),
        "time_summary": [
            {
                "case": row["case"],
                "scheduler": row["config"]["scheduler"],
                "edit_mode": row["config"]["model_edit_mode"],
                **{
                    label: {
                        "median_final_nodes": statistics.median(
                            s[label]["decisions"]["final"]["node_count"] for s in row["samples"]
                        ),
                        "median_elapsed": statistics.median(
                            s[label]["elapsed"] for s in row["samples"]
                        ),
                        "median_preparation": statistics.median(
                            s[label]["preparation"] for s in row["samples"]
                        ),
                        "median_candidates": statistics.median(
                            s[label]["candidates"] for s in row["samples"]
                        ),
                    }
                    for label in sources
                },
            }
            for row in results
            if row["budget"] == "time"
        ],
        "runs": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
