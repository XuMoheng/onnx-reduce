"""Serial, fresh-process ablations for the fifth and sixth optimization batches.

The baseline checkout must contain the original src/ and reduction.py with the
same gated/one-stubborn fixtures. No baseline code is imported into this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("node_count", "byte_size", "output_count", "input_count", "initializer_count")


def source_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted([*root.glob("src/onnx_reduce/**/*.py"), root / "benchmarks/reduction.py"])
    }


def profiles(suite):
    if suite == "preparation":
        return [
            ("before", True, []),
            ("snapshot-ir", False, ["--no-validation-cache", "--no-checkpoint-reuse"]),
            (
                "fast",
                False,
                ["--model-edit-mode", "auto", "--no-validation-cache", "--no-checkpoint-reuse"],
            ),
            ("cached", False, ["--model-edit-mode", "auto", "--no-checkpoint-reuse"]),
            ("stored", False, ["--model-edit-mode", "auto"]),
        ]
    return [
        ("restart-complete", False, ["--model-edit-mode", "auto"]),
        ("restart-fast", False, ["--model-edit-mode", "auto", "--scheduler", "staged-fast"]),
        ("cursor-complete", False, ["--model-edit-mode", "auto", "--search-policy", "cursor"]),
        (
            "cursor-fast",
            False,
            [
                "--model-edit-mode",
                "auto",
                "--search-policy",
                "cursor",
                "--scheduler",
                "staged-fast",
            ],
        ),
        (
            "adaptive",
            False,
            [
                "--model-edit-mode",
                "auto",
                "--search-policy",
                "cursor",
                "--direction-order",
                "adaptive",
            ],
        ),
        (
            "history",
            False,
            [
                "--model-edit-mode",
                "auto",
                "--search-policy",
                "cursor",
                "--direction-order",
                "adaptive",
                "--history-order",
                "defer",
            ],
        ),
    ]


def scenarios(suite):
    if suite == "preparation":
        cases = [
            (f"{case}-512", case, 512, "legacy", [])
            for case in ("chain", "branches", "outputs", "initializers", "missing-shape")
        ]
        cases += [
            ("chain-1000", "chain", 1000, "legacy", []),
            ("missing-shape-1000", "missing-shape", 1000, "legacy", []),
            ("preparation", "preparation", 128, "legacy", []),
            ("no-cache", "preparation", 32, "legacy", ["--no-cache"]),
        ]
        cases += [
            (f"{strategy}-{case}", case, 128, strategy, [])
            for strategy in ("stateful", "staged-complete")
            for case in ("chain", "branches", "outputs", "initializers", "missing-shape")
        ]
        cases += [
            ("gated-90", "gated", 90, "staged-complete", ["--candidate-budget", "5000"]),
            ("initializers-2000", "initializers", 2000, "legacy", []),
            ("stubborn-2000", "one-stubborn-init", 2000, "legacy", []),
        ]
        return cases
    return [
        ("tight", "chain", 32, "staged-complete", ["--candidate-budget", "4"]),
        (
            "threshold",
            "chain",
            128,
            "staged-complete",
            ["--condition", "threshold", "--threshold", "32", "--candidate-budget", "100"],
        ),
        (
            "scale",
            "chain",
            16,
            "staged-complete",
            ["--condition", "non-monotonic", "--interesting-sizes", "16,12,6"],
        ),
        (
            "cross",
            "chain",
            128,
            "staged-complete",
            ["--condition", "cross-region", "--candidate-budget", "100"],
        ),
        (
            "non-monotonic",
            "chain",
            128,
            "staged-complete",
            ["--condition", "non-monotonic", "--candidate-budget", "100"],
        ),
        ("old-rejection", "chain", 32, "staged-complete", ["--condition", "old-rejection"]),
        ("dead", "chain", 32, "staged-complete", ["--condition", "dead-trigger"]),
        ("modules", "chain", 128, "staged-complete", ["--module-width", "16"]),
        ("gated-30", "gated", 30, "staged-complete", ["--candidate-budget", "5000"]),
        ("gated-90", "gated", 90, "staged-complete", ["--candidate-budget", "5000"]),
        ("gated-300", "gated", 300, "staged-complete", ["--candidate-budget", "1000"]),
        ("stubborn", "one-stubborn-init", 512, "staged-complete", []),
        (
            "no-cache-threshold",
            "chain",
            32,
            "staged-complete",
            ["--no-cache", "--condition", "threshold", "--threshold", "8"],
        ),
        (
            "time-gated",
            "gated",
            90,
            "staged-complete",
            ["--budget", "time", "--time-budget", "0.2"],
        ),
        (
            "time-chain",
            "chain",
            512,
            "staged-complete",
            ["--budget", "time", "--time-budget", "0.2"],
        ),
    ]


def decisions(record):
    report = record["report"]
    return {
        k: report[k]
        for k in (
            "final",
            "final_hash",
            "candidates",
            "passes",
            "failure_checks",
            "stop_reason",
            "minimality",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("preparation", "search"), required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "output/performance")
    parser.add_argument("--only", help="comma-separated scenario names")
    parser.add_argument("--compare-instrumentation", action="store_true")
    parser.add_argument("--variants", help="comma-separated variant labels")
    parser.add_argument("--full-gated", action="store_true", help="raise gated-300 quota to 50000")
    args = parser.parse_args()
    before = args.baseline_root.resolve()
    batch = args.output.resolve() / f"batch56-{args.suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    batch.mkdir(parents=True)
    records = []
    selected = set(args.only.split(",")) if args.only else None
    hashes = {"before": source_hashes(before), "current": source_hashes(ROOT)}
    for name, case, size, strategy, options in scenarios(args.suite):
        if selected is not None and name not in selected:
            continue
        if args.full_gated and name == "gated-300":
            name = "gated-300-full"
            options = [*options, "--candidate-budget", "50000"]
        for repeat in range(args.repeat):
            variants = profiles(args.suite)
            if args.variants:
                variants = [item for item in variants if item[0] in args.variants.split(",")]
            if not variants:
                raise ValueError("no selected variants")
            if repeat % 2:
                variants.reverse()
            for label, old, flags in variants:
                modes = [True, False] if args.compare_instrumentation else [True]
                if repeat % 2:
                    modes.reverse()
                for enabled in modes:
                    directory = batch / f"{name}-{label}-{repeat}-{'on' if enabled else 'off'}"
                    root = before if old else ROOT
                    command = [
                        sys.executable,
                        str(root / "benchmarks/reduction.py"),
                        "--worker",
                        "--case",
                        case,
                        "--size",
                        str(size),
                        "--budget",
                        "calls",
                        "--candidate-budget",
                        "600",
                        "--scheduler",
                        strategy,
                        "--output",
                        str(directory),
                        *options,
                        *flags,
                    ]
                    if not enabled:
                        command.append("--no-performance")
                    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
                    if result.returncode:
                        raise RuntimeError(f"{command}:\n{result.stderr}")
                    record = json.loads(result.stdout)
                    record.update(scenario=name, variant=label, repeat=repeat, command=command)
                    report = record["report"]
                    artifact = Path(report["artifact_recheck"]["model"])
                    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["final_hash"]
                    assert report["artifact_recheck"]["read_back_validated"]
                    assert report["artifact_recheck"]["outcome"] == "interesting"
                    assert not report["artifact_recheck"]["cache_hit"]
                    directory.mkdir(exist_ok=True)
                    (directory / "measurement.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    with (batch / "runs.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
            print(f"{name} repeat {repeat + 1}/{args.repeat}", flush=True)
    summaries = []
    for name in dict.fromkeys(r["scenario"] for r in records):
        group = [r for r in records if r["scenario"] == name]
        for label in dict.fromkeys(r["variant"] for r in group):
            rows = [r for r in group if r["variant"] == label and r["instrumented"]]
            summaries.append(
                {
                    "scenario": name,
                    "variant": label,
                    "median_seconds": statistics.median(r["wall_seconds"] for r in rows),
                    "range_seconds": [
                        min(r["wall_seconds"] for r in rows),
                        max(r["wall_seconds"] for r in rows),
                    ],
                    "median_rss": statistics.median(r["peak_rss_bytes"] for r in rows),
                    "costs": [r["report"]["final"] for r in rows],
                    "calls": [r["report"]["failure_checks"]["calls"] for r in rows],
                    "minimality": [r["report"]["minimality"]["status"] for r in rows],
                }
            )
        for row in group:
            if row["budget"] == "time":
                continue
            equal = [r for r in group if r["variant"] == row["variant"]]
            assert all(decisions(r) == decisions(row) for r in equal), (name, row["variant"])
            assert all(r["decision_trace_sha256"] == row["decision_trace_sha256"] for r in equal)
            assert all(r["scheduler_trace_sha256"] == row["scheduler_trace_sha256"] for r in equal)
            if args.suite == "preparation":
                control = next(
                    r
                    for r in group
                    if r["variant"] == "before"
                    and r["repeat"] == row["repeat"]
                    and r["instrumented"] == row["instrumented"]
                )
                assert decisions(control) == decisions(row), (
                    name,
                    row["variant"],
                    "before mismatch",
                )
                assert control["decision_trace_sha256"] == row["decision_trace_sha256"]
    assert hashes == {"before": source_hashes(before), "current": source_hashes(ROOT)}, (
        "source changed during measurement"
    )
    (batch / "summary.json").write_text(
        json.dumps(
            {
                "suite": args.suite,
                "environment": {
                    "python": sys.version,
                    "executable": sys.executable,
                    "platform": platform.platform(),
                },
                "command": sys.argv,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "count": len(records),
                "source_sha256": hashes,
                "summaries": summaries,
                "runs": [
                    str(Path(r["report"]["work_directory"]).parents[0] / "measurement.json")
                    for r in records
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Evidence: {batch / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
