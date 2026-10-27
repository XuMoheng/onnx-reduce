"""Verify an installed wheel, isolated from the checkout with ``python -I``."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx

import onnx_reduce


def main() -> None:
    package = Path(onnx_reduce.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    version = importlib.metadata.version("onnx-reduce")
    assert onnx_reduce.__version__ == version
    cli = Path(sys.executable).with_name("onnx-reduce")
    examples = Path(__file__).resolve().parents[1] / "examples" / "relu"
    with tempfile.TemporaryDirectory(prefix="onnx-reduce-installed-") as directory:
        work = Path(directory)

        def run(*argv: str | Path) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(arg) for arg in argv],
                cwd=work,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

        assert run(cli, "--version").stdout.strip() == f"onnx-reduce {version}"
        run(sys.executable, "-I", "-m", "onnx_reduce", "--help")
        source = work / "original.onnx"
        output = work / "reduced.onnx"
        report_path = work / "report.json"
        run(sys.executable, "-I", examples / "make_model.py", source)
        run(
            cli,
            source,
            "--test",
            shlex.join([sys.executable, "-I", str(examples / "check.py")]),
            "--signature-regex",
            r"assertion: (\w+)",
            "-o",
            output,
            "--report",
            report_path,
        )
        model = onnx.load(output)
        onnx.checker.check_model(model, full_check=True)
        assert [node.op_type for node in model.graph.node] == ["Relu"]
        report = json.loads(report_path.read_text())
        assert report["original"]["node_count"] == 4
        assert report["final"]["node_count"] == 1
        assert report["stop_reason"] == "fixed_point"
        assert report["minimality"]["status"] == "pass_fixed_point"
        assert report["final_hash"] == hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"Installed onnx-reduce {version}: 4 -> 1 nodes, output checked, digest matches")


if __name__ == "__main__":
    main()
