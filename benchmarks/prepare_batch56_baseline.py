"""Freeze the original perf code and transplant only the shared benchmark fixtures."""

from __future__ import annotations

import argparse
import ast
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "3f7cb3a64d84c6ce1324a94da24c9f3cd2c86fd1"


def definitions(source):
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in {"CASES", "make_model", "ReluCheck"}:
            result[name] = (
                node.lineno - 1,
                node.end_lineno,
                lines[node.lineno - 1 : node.end_lineno],
            )
    assert len(result) == 3
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; existing paths are refused"
    )
    args = parser.parse_args()
    archive = subprocess.check_output(["git", "archive", BASE, "src", "benchmarks"], cwd=ROOT)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
    target = destination / "benchmarks/reduction.py"
    old = target.read_text()
    current = definitions((ROOT / "benchmarks/reduction.py").read_text())
    replacements = definitions(old)
    lines = old.splitlines(keepends=True)
    for name, (start, end, _) in sorted(replacements.items(), key=lambda pair: -pair[1][0]):
        lines[start:end] = current[name][2]
    unified = "".join(lines).replace("else ReluCheck()", "else ReluCheck(args.case, args.size)")
    target.write_text(unified)
    print(destination)


if __name__ == "__main__":
    main()
