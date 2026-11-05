"""Compare reduction of live fault carriers in independent source processes.

Conditions simulate a backend that discards unreachable code. They are not
real compiler regressions. Use strategy_compare.py with a real failure manifest
for an internal model; this benchmark requires no private model data.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def worker(request):
    source = Path(request["source"]).resolve()
    sys.path[:0] = [str(source / "src"), str(source)]
    import onnx
    from onnx import TensorProto, helper

    import onnx_reduce
    from onnx_reduce.engine import EngineConfig, ReductionEngine
    from onnx_reduce.failure_check.contract import FailureCheckResult as F
    from onnx_reduce.model import OnnxModel

    assert Path(onnx_reduce.__file__).resolve().is_relative_to(source)
    size, trigger = request["size"], request["trigger"]
    case = request["case"]

    def value(name, extent=1):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [extent])

    projected = case == "project_output"
    nodes, infos = [], []
    for i in range(size):
        previous = "x" if not i else f"v{i - 1}"
        if projected and i == trigger:
            nodes.append(
                helper.make_node("Split", [previous], ["other", f"v{i}"], axis=0, name=f"n{i}")
            )
            infos.append(value("other"))
        else:
            nodes.append(
                helper.make_node(
                    "Relu"
                    if case == "live_name" or (not projected and i == trigger)
                    else "Identity",
                    [previous],
                    [f"v{i}"],
                    name=f"n{i}",
                )
            )
        if i < size - 1:
            infos.append(value(f"v{i}", 2 if projected and i < trigger else 1))
    proto = helper.make_model(
        helper.make_graph(
            nodes,
            "live-fragment",
            [value("x", 2 if projected else 1)],
            [value(f"v{size - 1}")],
            value_info=infos,
        ),
        opset_imports=[helper.make_opsetid("", 13 if projected else 17)],
    )

    class Check:
        def run(self, path):
            graph = onnx.load(path).graph
            producers = {name: node for node in graph.node for name in node.output if name}
            pending = [v.name for v in graph.output]
            seen, live_names, live_ops = set(), set(), set()
            second_split_output = False
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                node = producers.get(name)
                if node is not None:
                    if node.op_type == "Split" and name == node.output[1]:
                        second_split_output = True
                    live_names.add(node.name)
                    live_ops.add(node.op_type)
                    pending.extend(node.input)
            if case == "locked":
                keep = len(live_names) == size
            elif case == "live_name":
                keep = f"n{trigger}" in live_names
            elif case == "project_output":
                keep = second_split_output and len(graph.output) == 1
            else:
                keep = "Relu" in live_ops
                if case == "fixed_io":
                    keep = (
                        keep
                        and [v.name for v in graph.input] == ["x"]
                        and [v.name for v in graph.output] == [f"v{size - 1}"]
                    )
            return F.INTERESTING if keep else F.NOT_INTERESTING

    _, report = ReductionEngine(config=EngineConfig(**request["config"])).run(
        OnnxModel.from_proto(proto), Check(), Path(request["out"]) / "work"
    )
    data = report.to_dict()
    assert data["artifact_recheck"]["outcome"] == "interesting"
    Path(request["result"]).write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--current-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--trigger", type=int, default=1140)
    parser.add_argument("--candidate-budget", type=int, default=128)
    parser.add_argument("--time-budget", type=float, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--schedulers", default="legacy,stateful,staged")
    parser.add_argument("--cases", default="live_op,live_name,locked,fixed_io,project_output")
    parser.add_argument("--model-edit-mode", choices=("ir", "auto"), default="ir")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    if not args.baseline_source or not args.out:
        parser.error("--baseline-source and --out are required")
    if not 0 <= args.trigger < args.size or args.repeat < 1:
        parser.error("require 0 <= trigger < size and repeat >= 1")
    cases = args.cases.split(",")
    schedulers = args.schedulers.split(",")
    if set(cases) - {"live_op", "live_name", "locked", "fixed_io", "project_output"} or set(
        schedulers
    ) - {
        "legacy",
        "stateful",
        "staged",
    }:
        parser.error("unknown case or scheduler")
    sources = {"baseline": args.baseline_source.resolve(), "current": args.current_source.resolve()}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        for scheduler in schedulers:
            for budget in ("calls", "time"):
                config = {
                    "scheduler": scheduler,
                    "model_edit_mode": args.model_edit_mode,
                    "combination_limit": 16,
                    "collect_performance": True,
                    "max_candidate_check_calls": args.candidate_budget
                    if budget == "calls"
                    else None,
                    "time_budget_seconds": args.time_budget if budget == "time" else None,
                }
                for repeat in range(args.repeat):
                    pair = {}
                    order = list(sources) if repeat % 2 == 0 else list(reversed(sources))
                    for label in order:
                        directory = out / f"{case}-{scheduler}-{budget}-{repeat}" / label
                        directory.mkdir(parents=True, exist_ok=True)
                        request = {
                            "source": str(sources[label]),
                            "case": case,
                            "size": args.size,
                            "trigger": args.trigger,
                            "config": config,
                            "out": str(directory),
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
                                stdout=log,
                                stderr=log,
                                check=True,
                            )
                        pair[label] = json.loads((directory / "result.json").read_text())
                    results.append(
                        {
                            "case": case,
                            "scheduler": scheduler,
                            "budget": budget,
                            "repeat": repeat,
                            "runs": pair,
                        }
                    )
                    (out / "summary.json").write_text(
                        json.dumps(
                            {
                                "condition_source": "simulated live graph traversal",
                                "python": sys.version,
                                "platform": platform.platform(),
                                "sources": {k: str(v) for k, v in sources.items()},
                                "size": args.size,
                                "trigger": args.trigger,
                                "runs": results,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    print(
                        f"{len(results)} {case}/{scheduler}/{budget}: "
                        f"{pair['baseline']['final']['node_count']} -> {pair['current']['final']['node_count']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
