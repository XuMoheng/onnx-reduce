from __future__ import annotations

import json
import sys
import textwrap

import onnx

from onnx_reduce.cli import main
from tests.helpers import chain_model


def write_oracle(tmp_path, name: str, body: str):
    path = tmp_path / name
    prefix = (
        f"#!{sys.executable}\n"
        "import os\n"
        "import signal\n"
        "import sys\n"
        "import onnx\n"
        "model = onnx.load(sys.argv[1])\n"
        "has_trigger = any(node.op_type == 'Relu' for node in model.graph.node)\n"
    )
    path.write_text(prefix + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


def test_cli_end_to_end_error_reducer(tmp_path):
    source = tmp_path / "model.onnx"
    output = tmp_path / "reduced.onnx"
    report_path = tmp_path / "report.json"
    onnx.save_model(chain_model(node_count=9, trigger_index=4), source)
    oracle = write_oracle(
        tmp_path,
        "error-oracle.py",
        "raise SystemExit(0 if has_trigger else 1)",
    )
    code = main(
        [
            str(source),
            "--test",
            str(oracle),
            "--output",
            str(output),
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    reduced = onnx.load(output)
    assert [node.op_type for node in reduced.graph.node] == ["Relu"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "fixed_point"
    assert payload["final"]["node_count"] == 1
    assert payload["original"]["node_count"] == 9
    assert payload["oracle"]["calls"] > 2


def test_cli_accepts_crashing_compiler_oracle(tmp_path):
    source = tmp_path / "model.onnx"
    output = tmp_path / "crash-reduced.onnx"
    onnx.save_model(chain_model(node_count=7, trigger_index=2), source)
    oracle = write_oracle(
        tmp_path,
        "crash-oracle.py",
        """\
        if has_trigger:
            os.kill(os.getpid(), signal.SIGSEGV)
        raise SystemExit(1)
        """,
    )
    code = main([str(source), "--test", str(oracle), "--output", str(output)])
    assert code == 0
    reduced = onnx.load(output)
    assert [node.op_type for node in reduced.graph.node] == ["Relu"]


def test_cli_rejects_reversed_interestingness_direction(tmp_path, capsys):
    source = tmp_path / "model.onnx"
    output = tmp_path / "not-written.onnx"
    onnx.save_model(chain_model(), source)
    oracle = write_oracle(
        tmp_path,
        "reversed.py",
        "raise SystemExit(0 if not has_trigger else 1)",
    )
    code = main([str(source), "--test", str(oracle), "--output", str(output)])
    assert code == 2
    assert "direction may be reversed" in capsys.readouterr().err
    assert not output.exists()


def test_cli_output_is_bit_deterministic(tmp_path):
    source = tmp_path / "model.onnx"
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    onnx.save_model(chain_model(node_count=10, trigger_index=5), source)
    oracle = write_oracle(
        tmp_path,
        "deterministic.py",
        "raise SystemExit(0 if has_trigger else 1)",
    )
    for output in (first, second):
        assert main([str(source), "--test", str(oracle), "--output", str(output)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_package_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import onnx_reduce

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert onnx_reduce.__version__ == pyproject["project"]["version"]
