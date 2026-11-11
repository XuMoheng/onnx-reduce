from __future__ import annotations

import json
import sys
import textwrap

import onnx
import pytest

from onnx_reduce.cli import main
from tests.helpers import chain_model


def write_failure_check(tmp_path, name: str, body: str):
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


@pytest.mark.parametrize("failure_code", [1, 7])
def test_cli_end_to_end_error_reducer(tmp_path, failure_code):
    source = tmp_path / "model.onnx"
    output = tmp_path / "reduced.onnx"
    report_path = tmp_path / "report.json"
    onnx.save_model(chain_model(node_count=9, trigger_index=4), source)
    failure_check_script = write_failure_check(
        tmp_path,
        "error-failure_check_script.py",
        f"raise SystemExit({failure_code} if has_trigger else 0)",
    )
    code = main(
        [
            str(source),
            "--test",
            str(failure_check_script),
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
    assert payload["failure_checks"]["calls"] > 2


def test_cli_accepts_crashing_compiler_failure_check(tmp_path):
    source = tmp_path / "model.onnx"
    output = tmp_path / "crash-reduced.onnx"
    onnx.save_model(chain_model(node_count=7, trigger_index=2), source)
    failure_check_script = write_failure_check(
        tmp_path,
        "crash-failure_check_script.py",
        """\
        if has_trigger:
            os.kill(os.getpid(), signal.SIGSEGV)
        raise SystemExit(0)
        """,
    )
    code = main([str(source), "--test", str(failure_check_script), "--output", str(output)])
    assert code == 0
    reduced = onnx.load(output)
    assert [node.op_type for node in reduced.graph.node] == ["Relu"]


@pytest.mark.parametrize("target_mode", ["baseline", "script"])
def test_cli_rejects_passing_original_even_with_matching_signature(tmp_path, capsys, target_mode):
    source = tmp_path / "model.onnx"
    output = tmp_path / "not-written.onnx"
    onnx.save_model(chain_model(), source)
    failure_check_script = write_failure_check(
        tmp_path,
        "passing.py",
        "print('assertion A'); raise SystemExit(0 if has_trigger else 1)",
    )
    code = main(
        [
            str(source),
            "--test",
            str(failure_check_script),
            "--output",
            str(output),
            "--target-mode",
            target_mode,
            "--signature-regex",
            r"assertion (\w+)",
        ]
    )
    assert code == 2
    assert "exit code 0 means pass" in capsys.readouterr().err
    assert not output.exists()


def test_cli_output_is_bit_deterministic(tmp_path):
    source = tmp_path / "model.onnx"
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    onnx.save_model(chain_model(node_count=10, trigger_index=5), source)
    failure_check_script = write_failure_check(
        tmp_path,
        "deterministic.py",
        "raise SystemExit(1 if has_trigger else 0)",
    )
    for output in (first, second):
        assert (
            main([str(source), "--test", str(failure_check_script), "--output", str(output)]) == 0
        )
    assert first.read_bytes() == second.read_bytes()


def test_package_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import onnx_reduce

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert onnx_reduce.__version__ == pyproject["project"]["version"]


def test_cli_exposes_quality_objective_and_search_bounds(tmp_path):
    from onnx import helper

    from benchmarks.quality_cases import model, value

    source, output = tmp_path / "source.onnx", tmp_path / "small.onnx"
    report = tmp_path / "quality.json"
    onnx.save_model(
        model([helper.make_node("Relu", ["x"], ["y"])], [value("x")], [value("y")]), source
    )
    check = write_failure_check(tmp_path, "check.py", "raise SystemExit(1 if has_trigger else 0)")
    assert (
        main(
            [
                str(source),
                "--test",
                str(check),
                "-o",
                str(output),
                "--report",
                str(report),
                "--objective",
                "tensors",
                "--combination-depth",
                "3",
                "--combination-limit",
                "4",
                "--scheduler",
                "staged",
                "--goal",
                "complete",
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text())
    assert payload["version"] == 7
    assert payload["minimality"]["objective"] == "tensors"
    assert payload["minimality"]["combination_depth"] == 3
    assert payload["minimality"]["combination_limit"] == 4
    assert payload["final"]["tensor_elements"] == 1
    assert payload["minimality"]["status"] == "pass_fixed_point"
