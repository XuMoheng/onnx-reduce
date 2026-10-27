"""Command-line entry point for onnx-reduce."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from onnx_reduce import __version__
from onnx_reduce.engine import EngineConfig, ReductionEngine, ReductionError
from onnx_reduce.model import OnnxModel
from onnx_reduce.oracle.contract import make_spec
from onnx_reduce.oracle.runner import OracleError, OracleRunner
from onnx_reduce.storage import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-reduce",
        description=(
            "Reduce an ONNX model while preserving a user-defined failure. "
            "The candidate model path is appended to the test command's arguments; "
            "exit code 0 means the failure is still present."
        ),
    )
    parser.add_argument("model", type=Path, help="ONNX model to reduce")
    parser.add_argument("--test", required=True, help="interestingness test command")
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="path for the reduced model"
    )
    parser.add_argument("--report", type=Path, help="write a JSON summary to this path")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-test timeout in seconds")
    parser.add_argument(
        "--checker",
        choices=("strict", "structural", "permissive"),
        default="strict",
        help="candidate validation mode (custom domains automatically use structural checks)",
    )
    parser.add_argument("--max-tests", type=int, help="maximum candidate oracle calls")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        help="soft wall-clock search budget; final recheck always runs",
    )
    parser.add_argument(
        "--work-dir", type=Path, help="retain run logs/checkpoints here (default: OUTPUT.work)"
    )
    parser.add_argument("--no-cache", action="store_true", help="disable per-run verdict cache")
    parser.add_argument(
        "--target-mode",
        choices=("baseline", "script"),
        default="baseline",
        help="match baseline termination, or trust script interestingness only",
    )
    parser.add_argument(
        "--signature-regex",
        help="match baseline signature in stdout/stderr (capture groups if present)",
    )
    parser.add_argument(
        "--inconclusive-code",
        action="append",
        type=int,
        default=[],
        help="exit code judged inconclusive (repeatable; default: 125)",
    )
    parser.add_argument(
        "--no-crash-interesting",
        action="store_true",
        help="judge signal-terminated tests not interesting instead of interesting",
    )
    parser.add_argument(
        "--timeout-interesting",
        action="store_true",
        help="judge timeouts interesting (for hang bugs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("onnx-reduce")

    try:
        config = EngineConfig(
            checker_mode=args.checker,
            max_tests=args.max_tests,
            time_budget_seconds=args.time_budget_seconds,
            cache_enabled=not args.no_cache,
            target_mode=args.target_mode,
            signature_regex=args.signature_regex,
        )
        if args.report:
            for model_path in (args.model, args.output):
                if args.report.resolve() == model_path.resolve() or (
                    args.report.exists()
                    and model_path.exists()
                    and args.report.samefile(model_path)
                ):
                    raise ValueError("--report must not overwrite the input or output model")
        codes = tuple(args.inconclusive_code or (125,))
        spec = make_spec(
            args.test,
            timeout_seconds=args.timeout,
            crash_is_interesting=not args.no_crash_interesting,
            timeout_is_interesting=args.timeout_interesting,
            inconclusive_exit_codes=codes,
        )
        runner = OracleRunner(spec)
        model = OnnxModel.load(args.model)
        engine = ReductionEngine(
            config=config,
            progress=logger.info,
        )
        workdir = args.work_dir or args.output.with_name(args.output.name + ".work")
        logger.info("retaining logs and checkpoints under %s", workdir)
        _, report = engine.run(model, runner, workdir, output_path=args.output)
        payload = report.to_dict()
        if args.report:
            write_json(args.report, payload)
        logger.info(
            "wrote %s: %d/%d nodes, stop_reason=%s",
            args.output,
            payload["final"]["node_count"],
            payload["original"]["node_count"],
            payload["stop_reason"],
        )
        return 1 if report.stop_reason == "unstable_check" else 0
    except KeyboardInterrupt:
        print(
            "onnx-reduce: interrupted; inspect the run directory for checkpoint.json "
            "(available after the initial successful check); final output may be unverified",
            file=sys.stderr,
        )
        return 130
    except (OracleError, ReductionError, ValueError, OSError, re.error) as exc:
        print(f"onnx-reduce: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
