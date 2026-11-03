"""Reproducible reduction benchmarks, each measured in a fresh Python process."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Allow a source checkout without installation, also in spawned workers.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import onnx
import onnxscript
from onnx import TensorProto, helper

from onnx_reduce.edits import Candidate, DeleteNodes
from onnx_reduce.engine import EngineConfig, ReductionEngine
from onnx_reduce.failure_check.contract import FailureCheckResult, make_failure_check_config
from onnx_reduce.failure_check.runner import FailureCheckRunner
from onnx_reduce.model import OnnxModel
from onnx_reduce.performance import PerformanceStats

CASES = ("initializers", "chain", "branches", "outputs", "missing-shape")


def value(name, size=1):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [size])


def make_model(case: str, size: int) -> onnx.ModelProto:
    nodes = []
    outputs = [value("y")]
    infos = []
    weights = []
    if case == "initializers":
        nodes = [helper.make_node("Relu", ["x"], ["y"])]
        weights = [
            helper.make_tensor(f"unused_{i:04d}", TensorProto.FLOAT, [1], [1.0])
            for i in range(size)
        ]
    elif case in {"chain", "missing-shape", "preparation"}:
        previous = "x"
        for i in range(size):
            name = "y" if i == size - 1 else f"v{i}"
            nodes.append(
                helper.make_node("Relu" if i == size // 2 else "Identity", [previous], [name])
            )
            if name != "y" and case == "chain":
                infos.append(value(name))
            previous = name
    else:
        nodes.append(helper.make_node("Relu", ["x"], ["trigger"]))
        infos.append(value("trigger"))
        branch_outputs = []
        for i in range(size):
            name = f"branch_{i}"
            nodes.append(helper.make_node("Identity", ["trigger"], [name]))
            branch_outputs.append(value(name))
        if case == "outputs":
            outputs = branch_outputs
        else:
            nodes.append(
                helper.make_node("Concat", [v.name for v in branch_outputs], ["y"], axis=0)
            )
            infos.extend(branch_outputs)
            outputs = [value("y", size)]
    graph = helper.make_graph(
        nodes, case, [value("x")], outputs, initializer=weights, value_info=infos
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class ReluCheck:
    """Deterministic synthetic condition; deliberately excludes process startup cost."""

    def run(self, path):
        model = onnx.load(path)
        return (
            FailureCheckResult.INTERESTING
            if any(node.op_type == "Relu" for node in model.graph.node)
            else FailureCheckResult.NOT_INTERESTING
        )


class NodeCountCheck:
    """Synthetic threshold failure: every node must remain for the bug to reproduce."""

    def __init__(self, size):
        self.size = size

    def run(self, path):
        return (
            FailureCheckResult.INTERESTING
            if len(onnx.load(path).graph.node) == self.size
            else FailureCheckResult.NOT_INTERESTING
        )


class RepeatedDeletionPass:
    name = "repeated_deletion"

    def generate(self, model):
        for index in range(model.node_count):
            yield Candidate((DeleteNodes((index,)),))
            # Different edit, identical content, then an identical edit.
            yield Candidate((DeleteNodes((index, index)),))
            yield Candidate((DeleteNodes((index,)),))


def worker(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if args.model:
        model = OnnxModel.load(args.model)
        runner = FailureCheckRunner(
            make_failure_check_config(
                args.test,
                timeout_seconds=args.timeout,
                crash_is_interesting=True,
                timeout_is_interesting=False,
            )
        )
    else:
        proto = make_model(args.case, args.size)
        onnx.checker.check_model(proto, full_check=True)
        model = OnnxModel.from_proto(proto)
        runner = NodeCountCheck(args.size) if args.case == "preparation" else ReluCheck()
    config = EngineConfig(
        cache_enabled=not args.no_cache,
        collect_performance=not args.no_performance,
        max_candidate_check_calls=args.candidate_budget if args.budget == "calls" else None,
        time_budget_seconds=args.time_budget if args.budget == "time" else None,
        signature_regex=args.signature_regex,
    )
    started = time.perf_counter()
    passes = [RepeatedDeletionPass()] if args.case == "preparation" else None
    _, report = ReductionEngine(passes, config=config).run(
        model,
        runner,
        destination / "work",
        output_path=destination / "reduced.onnx",
    )
    elapsed = time.perf_counter() - started
    # Hash the ordered decisions, including edits, base identities and rejection
    # reasons, to compare both successful and unsuccessful candidate attempts.
    trace_path = Path(report.work_directory) / "candidates.jsonl"
    trace = trace_path.read_bytes() if trace_path.exists() else b""
    return {
        "case": args.case,
        "size": args.size,
        "budget": args.budget,
        "instrumented": not args.no_performance,
        "wall_seconds": elapsed,
        "peak_rss_bytes": PerformanceStats.peak_rss(),
        "decision_trace_sha256": hashlib.sha256(trace).hexdigest(),
        "report": report.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=(*CASES, "preparation", "backend", "all"), default="initializers"
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--budget", choices=("calls", "time", "both"), default="both")
    parser.add_argument("--candidate-budget", type=int, default=600)
    parser.add_argument("--time-budget", type=float, default=5.0)
    parser.add_argument("--compare-instrumentation", action="store_true")
    parser.add_argument(
        "--no-cache", action="store_true", help="disable preparation/verdict caches"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "performance")
    parser.add_argument(
        "--model", type=Path, help="real backend reproducer, used with --case backend"
    )
    parser.add_argument("--test", help="real backend interestingness command")
    parser.add_argument("--signature-regex")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-performance", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size < 1 or args.repeat < 1 or args.candidate_budget < 0 or args.time_budget <= 0:
        parser.error(
            "size/repeat/time budget must be positive; candidate budget must be nonnegative"
        )
    if (args.case == "backend") != bool(args.model and args.test) or bool(args.model) != bool(
        args.test
    ):
        parser.error(
            "--case backend requires both --model and --test; synthetic cases accept neither"
        )
    if args.worker:
        print(json.dumps(worker(args)))
        return
    cases = CASES if args.case == "all" else (args.case,)
    budgets = ("calls", "time") if args.budget == "both" else (args.budget,)
    records = []
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    batch = args.output / time.strftime("%Y%m%d-%H%M%S")
    batch.mkdir()  # Never mix artifacts with an earlier batch.
    for case in cases:
        for budget in budgets:
            group = []
            for repeat in range(args.repeat):
                pair = []
                # Alternate execution order to reduce systematic warm-cache bias.
                modes = [True, False] if args.compare_instrumentation else [True]
                if repeat % 2:
                    modes.reverse()
                for enabled in modes:
                    directory = batch / f"{case}-{budget}-{repeat}-{'on' if enabled else 'off'}"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--case",
                        case,
                        "--size",
                        str(args.size),
                        "--budget",
                        budget,
                        "--candidate-budget",
                        str(args.candidate_budget),
                        "--time-budget",
                        str(args.time_budget),
                        "--output",
                        str(directory),
                    ]
                    if not enabled:
                        command.append("--no-performance")
                    if args.no_cache:
                        command.append("--no-cache")
                    if args.model:
                        command += [
                            "--model",
                            str(args.model.resolve()),
                            "--test",
                            args.test,
                            "--timeout",
                            str(args.timeout),
                        ]
                    if args.signature_regex:
                        command += ["--signature-regex", args.signature_regex]
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(
                            f"benchmark worker failed ({directory}):\n{result.stderr}"
                        )
                    record = json.loads(result.stdout)
                    (directory / "measurement.json").write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n"
                    )
                    records.append(record)
                    group.append(record)
                    pair.append(record)
                    print(
                        f"{directory.name}: {record['wall_seconds']:.3f}s, "
                        f"final={record['report']['final']}, checks={record['report']['failure_checks']['calls']}",
                        flush=True,
                    )
                if budget == "calls" and len(pair) == 2:
                    if pair[0]["decision_trace_sha256"] != pair[1]["decision_trace_sha256"]:
                        raise RuntimeError(
                            f"instrumentation changed candidate decisions: {case}, repetition {repeat}"
                        )
                    for key in (
                        "final",
                        "final_hash",
                        "candidates",
                        "failure_checks",
                        "stop_reason",
                    ):
                        if pair[0]["report"][key] != pair[1]["report"][key]:
                            raise RuntimeError(
                                f"instrumentation changed {key}: {case}, repetition {repeat}"
                            )
            on = [r["wall_seconds"] for r in group if r["instrumented"]]
            off = [r["wall_seconds"] for r in group if not r["instrumented"]]
            summary = {
                "case": case,
                "budget": budget,
                "instrumented_median_seconds": statistics.median(on),
            }
            if off:
                summary.update(
                    uninstrumented_median_seconds=statistics.median(off),
                    overhead_ratio=statistics.median(on) / statistics.median(off) - 1,
                )
            summaries.append(summary)
    payload = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "onnx": onnx.__version__,
            "onnxscript": onnxscript.__version__,
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(
                    [*ROOT.glob("src/onnx_reduce/**/*.py"), Path(__file__).resolve()]
                )
            },
        },
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summaries": summaries,
        "runs": records,
    }
    result_path = batch / "baseline.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Baseline: {result_path}")


if __name__ == "__main__":
    main()
