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

CASES = (
    "initializers",
    "chain",
    "branches",
    "outputs",
    "missing-shape",
    "gated",
    "one-stubborn-init",
)


def value(name, size=1):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [size])


def make_model(
    case: str, size: int, *, trigger_position: str = "middle", condition: str = "relu"
) -> onnx.ModelProto:
    nodes = []
    outputs = [value("y")]
    infos = []
    weights = []
    if case in {"initializers", "one-stubborn-init"}:
        nodes = [helper.make_node("Relu", ["x"], ["y"])]
        weights = [
            helper.make_tensor(f"unused_{i:04d}", TensorProto.FLOAT, [1], [1.0])
            for i in range(size)
        ]
    elif case in {"chain", "missing-shape", "preparation", "gated"}:
        trigger = {"front": 0, "middle": size // 2, "back": size - 1}[trigger_position]
        previous = "x"
        for i in range(size):
            name = "y" if i == size - 1 else f"v{i}"
            nodes.append(
                helper.make_node(
                    "Relu" if i == trigger and condition != "dead-trigger" else "Identity",
                    [previous],
                    [name],
                    **(
                        {"name": f"mustkeep_{i}" if i % 3 == 0 else f"n{i}"}
                        if case == "gated"
                        else {"name": f"n{i}"}
                        if condition != "relu"
                        else {}
                    ),
                )
            )
            if (name != "y" and case == "chain") or case == "gated":
                infos.append(value(name))
            previous = name
        if condition == "dead-trigger":
            nodes.append(helper.make_node("Relu", ["x"], ["dead_value"], name="dead_trigger"))
            infos.append(value("dead_value"))
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

    def __init__(self, case="chain", size=0):
        self.case = case
        self.stubborn = f"unused_{size // 2:04d}"
        self.anchors = {f"mustkeep_{i}" for i in range(0, size, 3)}

    def run(self, path):
        model = onnx.load(path)
        interesting = any(node.op_type == "Relu" for node in model.graph.node)
        if self.case == "gated":
            interesting &= self.anchors <= {node.name for node in model.graph.node}
        elif self.case == "one-stubborn-init":
            interesting &= self.stubborn in {v.name for v in model.graph.initializer}
        return FailureCheckResult.INTERESTING if interesting else FailureCheckResult.NOT_INTERESTING


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


class SyntheticCheck:
    """Named adversarial conditions; no numerical inference or real backend."""

    def __init__(self, args):
        self.args = args

    def run(self, path):
        nodes = onnx.load(path).graph.node
        names = {node.name for node in nodes}
        size = self.args.size
        condition = self.args.condition
        if condition == "cross-region":
            interesting = {f"n{size // 4}", f"n{3 * size // 4}"} <= names
        elif condition == "threshold":
            interesting = len(nodes) >= (self.args.threshold or max(1, size // 4))
        elif condition == "non-monotonic":
            allowed = self.args.interesting_sizes or (size, size // 2 + 1, max(1, size // 4))
            interesting = len(nodes) in allowed
        elif condition == "old-rejection":
            interesting = f"n{size // 2}" in names and (
                "n0" in names or f"n{size - 1}" not in names
            )
        else:
            interesting = "dead_trigger" in names
        return FailureCheckResult.INTERESTING if interesting else FailureCheckResult.NOT_INTERESTING


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
        proto = make_model(
            args.case, args.size, trigger_position=args.trigger_position, condition=args.condition
        )
        if args.module_width:
            for index, node in enumerate(proto.graph.node):
                prop = node.metadata_props.add()
                prop.key = "onnx_reduce.module"
                prop.value = json.dumps(["model", f"block{index // args.module_width}"])
        onnx.checker.check_model(proto, full_check=True)
        model = OnnxModel.from_proto(proto)
        runner = (
            NodeCountCheck(args.size)
            if args.case == "preparation"
            else SyntheticCheck(args)
            if args.condition != "relu"
            else ReluCheck(args.case, args.size)
        )
    config = EngineConfig(
        cache_enabled=not args.no_cache,
        model_edit_mode=args.model_edit_mode,
        search_policy=args.search_policy,
        history_order=args.history_order,
        direction_order=args.direction_order,
        checkpoint_reuse=not args.no_checkpoint_reuse,
        validation_cache_enabled=not args.no_validation_cache,
        collect_performance=not args.no_performance,
        max_candidate_check_calls=args.candidate_budget if args.budget == "calls" else None,
        time_budget_seconds=args.time_budget if args.budget == "time" else None,
        signature_regex=args.signature_regex,
        scheduler="staged" if args.scheduler.startswith("staged") else args.scheduler,
        goal=("fast" if args.scheduler == "staged-fast" else "complete")
        if args.scheduler.startswith("staged")
        else None,
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
    schedule_path = Path(report.work_directory) / "scheduler.jsonl"
    return {
        "case": args.case,
        "size": args.size,
        "budget": args.budget,
        "instrumented": not args.no_performance,
        "scheduler": args.scheduler,
        "condition": args.condition,
        "module_width": args.module_width,
        "trigger_position": args.trigger_position,
        "wall_seconds": elapsed,
        "peak_rss_bytes": PerformanceStats.peak_rss(),
        "decision_trace_sha256": hashlib.sha256(trace).hexdigest(),
        "scheduler_trace_sha256": hashlib.sha256(
            schedule_path.read_bytes() if schedule_path.exists() else b""
        ).hexdigest(),
        "report": report.to_dict(),
    }


def run_worker(args, case, budget, repeat, enabled, scheduler, batch):
    directory = batch / f"{case}-{budget}-{repeat}-{scheduler}-{'on' if enabled else 'off'}"
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
        "--scheduler",
        scheduler,
        "--condition",
        args.condition,
        "--trigger-position",
        args.trigger_position,
    ]
    command += ["--module-width", str(args.module_width)]
    for option in ("model-edit-mode", "search-policy", "history-order", "direction-order"):
        command += ["--" + option, getattr(args, option.replace("-", "_"))]
    if args.no_checkpoint_reuse:
        command.append("--no-checkpoint-reuse")
    if args.no_validation_cache:
        command.append("--no-validation-cache")
    if not enabled:
        command.append("--no-performance")
    if args.no_cache:
        command.append("--no-cache")
    if args.threshold is not None:
        command += ["--threshold", str(args.threshold)]
    if args.interesting_sizes:
        command += ["--interesting-sizes", ",".join(map(str, args.interesting_sizes))]
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
        raise RuntimeError(f"benchmark worker failed ({directory}):\n{result.stderr}")
    record = {**json.loads(result.stdout), "repeat": repeat}
    (directory / "measurement.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"{directory.name}: {record['wall_seconds']:.3f}s, "
        f"final={record['report']['final']}, checks={record['report']['failure_checks']['calls']}",
        flush=True,
    )
    return record


def assert_same_decisions(first, second):
    for key in ("decision_trace_sha256", "scheduler_trace_sha256"):
        if first[key] != second[key]:
            raise RuntimeError(
                f"fixed-call-budget decisions changed: {first['case']}, {first['scheduler']}, {key}"
            )
    for key in (
        "final",
        "final_hash",
        "candidates",
        "passes",
        "failure_checks",
        "stop_reason",
        "minimality",
    ):
        if first["report"][key] != second["report"][key]:
            raise RuntimeError(
                f"fixed-call-budget result changed: {first['case']}, {first['scheduler']}, {key}"
            )


def scheduler_comparisons(group):
    comparisons = []
    fields = ("node_count", "byte_size", "output_count", "input_count", "initializer_count")
    for legacy in (r for r in group if r["scheduler"] == "legacy"):
        for other in group:
            if (
                other["scheduler"] == "legacy"
                or other["repeat"] != legacy["repeat"]
                or other["instrumented"] != legacy["instrumented"]
            ):
                continue
            before, after = (
                tuple(r["report"]["final"][name] for name in fields) for r in (legacy, other)
            )
            comparisons.append(
                {
                    "case": legacy["case"],
                    "budget": legacy["budget"],
                    "repeat": legacy["repeat"],
                    "instrumented": legacy["instrumented"],
                    "strategy": other["scheduler"],
                    "quality": "improved"
                    if after < before
                    else "regressed"
                    if after > before
                    else "equal",
                    "legacy_cost": before,
                    "candidate_cost": after,
                    "legacy_run": legacy["report"]["work_directory"],
                    "candidate_run": other["report"]["work_directory"],
                }
            )
    return comparisons


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
        "--scheduler",
        choices=("legacy", "stateful", "staged-fast", "staged-complete"),
        default="legacy",
    )
    parser.add_argument(
        "--compare-staged",
        action="store_true",
        help="compare legacy/stateful/staged-fast/staged-complete",
    )
    parser.add_argument(
        "--module-width",
        type=int,
        default=0,
        help="explicit module annotations every N synthetic nodes (0 disables)",
    )
    parser.add_argument("--compare-schedulers", action="store_true")
    parser.add_argument(
        "--condition",
        choices=(
            "relu",
            "cross-region",
            "threshold",
            "non-monotonic",
            "old-rejection",
            "dead-trigger",
        ),
        default="relu",
    )
    parser.add_argument("--trigger-position", choices=("front", "middle", "back"), default="middle")
    parser.add_argument("--threshold", type=int)
    parser.add_argument(
        "--interesting-sizes", type=lambda text: tuple(int(n) for n in text.split(","))
    )
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
    parser.add_argument("--model-edit-mode", choices=("ir", "auto"), default="ir")
    parser.add_argument("--search-policy", choices=("restart", "cursor"), default="restart")
    parser.add_argument("--history-order", choices=("off", "defer"), default="off")
    parser.add_argument("--direction-order", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--no-checkpoint-reuse", action="store_true")
    parser.add_argument("--no-validation-cache", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-performance", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.module_width < 0 or (args.module_width and args.model):
        parser.error("--module-width must be nonnegative and applies only to synthetic models")
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
    if args.condition != "relu" and args.case not in {"chain", "missing-shape"}:
        parser.error("non-relu conditions require --case chain or missing-shape")
    if args.condition in {"cross-region", "old-rejection"} and args.size < 3:
        parser.error("cross-region and old-rejection require --size >= 3")
    if args.trigger_position != "middle" and args.case not in {
        "chain",
        "missing-shape",
        "preparation",
    }:
        parser.error("trigger-position applies to chain, missing-shape and preparation")
    if args.threshold is not None and (
        args.condition != "threshold" or not 1 <= args.threshold <= args.size
    ):
        parser.error("--threshold requires threshold condition and a value in [1, size]")
    if args.interesting_sizes is not None and (
        args.condition != "non-monotonic"
        or args.size not in args.interesting_sizes
        or min(args.interesting_sizes) < 0
    ):
        parser.error(
            "--interesting-sizes requires non-monotonic condition, nonnegative sizes and the original size"
        )
    if args.worker:
        print(json.dumps(worker(args)))
        return
    cases = CASES if args.case == "all" else (args.case,)
    budgets = ("calls", "time") if args.budget == "both" else (args.budget,)
    records = []
    summaries = []
    comparisons = []
    args.output.mkdir(parents=True, exist_ok=True)
    batch = args.output / time.strftime("%Y%m%d-%H%M%S")
    batch.mkdir()  # Never mix artifacts with an earlier batch.
    for case in cases:
        for budget in budgets:
            group = []
            for repeat in range(args.repeat):
                # Alternate execution order to reduce systematic warm-cache bias.
                strategies = (
                    ["legacy", "stateful", "staged-fast", "staged-complete"]
                    if args.compare_staged
                    else ["legacy", "stateful"]
                    if args.compare_schedulers
                    else [args.scheduler]
                )
                if repeat % 2:
                    strategies.reverse()
                for strategy in strategies:
                    modes = [True, False] if args.compare_instrumentation else [True]
                    if repeat % 2:
                        modes.reverse()
                    pair = []
                    for enabled in modes:
                        record = run_worker(args, case, budget, repeat, enabled, strategy, batch)
                        if budget == "calls":
                            prior = next((r for r in group if r["scheduler"] == strategy), None)
                            if prior is not None:
                                assert_same_decisions(prior, record)
                        records.append(record)
                        group.append(record)
                        pair.append(record)
                    if budget == "calls" and len(pair) == 2:
                        assert_same_decisions(pair[0], pair[1])
            for strategy in sorted({r["scheduler"] for r in group}):
                on = [
                    r["wall_seconds"]
                    for r in group
                    if r["scheduler"] == strategy and r["instrumented"]
                ]
                off = [
                    r["wall_seconds"]
                    for r in group
                    if r["scheduler"] == strategy and not r["instrumented"]
                ]
                summary = {
                    "case": case,
                    "budget": budget,
                    "scheduler": strategy,
                    "instrumented_median_seconds": statistics.median(on),
                }
                if off:
                    summary.update(
                        uninstrumented_median_seconds=statistics.median(off),
                        overhead_ratio=statistics.median(on) / statistics.median(off) - 1,
                    )
                summaries.append(summary)
            comparisons.extend(scheduler_comparisons(group))
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
        "comparisons": comparisons,
        "runs": records,
    }
    result_path = batch / "baseline.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Baseline: {result_path}")


if __name__ == "__main__":
    main()
