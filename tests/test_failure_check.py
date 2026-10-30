from __future__ import annotations

import os
import signal
import sys
import textwrap
import time

import pytest

from onnx_reduce.failure_check.contract import FailureCheckResult, make_failure_check_config
from onnx_reduce.failure_check.runner import FailureCheckError, FailureCheckRunner


def write_script(tmp_path, body: str):
    path = tmp_path / "check.py"
    path.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            {body}
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def runner(body: str, tmp_path, **kwargs) -> FailureCheckRunner:
    defaults = {
        "timeout_seconds": 5.0,
        "crash_is_interesting": True,
        "timeout_is_interesting": False,
    }
    defaults.update(kwargs)
    return FailureCheckRunner(
        make_failure_check_config(str(write_script(tmp_path, body)), **defaults)
    )


def test_exit_zero_is_interesting_and_nonzero_is_not(tmp_path):
    assert runner("raise SystemExit(0)", tmp_path).run("x") is FailureCheckResult.INTERESTING
    assert runner("raise SystemExit(7)", tmp_path).run("x") is FailureCheckResult.NOT_INTERESTING


def test_signal_is_interesting_by_default_and_can_be_disabled(tmp_path):
    crash = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    assert runner(crash, tmp_path).run("x") is FailureCheckResult.INTERESTING
    disabled = runner(
        crash,
        tmp_path,
        crash_is_interesting=False,
    )
    assert disabled.run("x") is FailureCheckResult.NOT_INTERESTING


def test_timeout_is_inconclusive_unless_configured_interesting(tmp_path):
    hang = "import time; time.sleep(10)"
    assert runner(hang, tmp_path, timeout_seconds=0.1).run("x") is FailureCheckResult.INCONCLUSIVE
    interesting = runner(
        hang,
        tmp_path,
        timeout_seconds=0.1,
        timeout_is_interesting=True,
    )
    assert interesting.run("x") is FailureCheckResult.INTERESTING


def test_configured_exit_code_is_inconclusive(tmp_path):
    test = runner("raise SystemExit(125)", tmp_path)
    assert test.run("x") is FailureCheckResult.INCONCLUSIVE


def test_missing_command_is_a_tool_error(tmp_path):
    failure_check_config = make_failure_check_config(
        str(tmp_path / "does-not-exist"),
        timeout_seconds=1.0,
        crash_is_interesting=True,
        timeout_is_interesting=False,
    )
    with pytest.raises(FailureCheckError):
        FailureCheckRunner(failure_check_config).run("x")


@pytest.mark.parametrize("hang", [False, True])
def test_escaped_descendant_cannot_hold_log_pipes_open(tmp_path, hang):
    pid_path = tmp_path / "child.pid"
    script = tmp_path / "escape.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'], "
        "start_new_session=True)\n"
        "pathlib.Path(sys.argv[-1]).write_text(str(child.pid))\n"
        "print('assertion A', flush=True)\n" + ("time.sleep(5)\n" if hang else ""),
        encoding="utf-8",
    )
    failure_check_config = make_failure_check_config(
        [sys.executable, str(script)],
        timeout_seconds=0.5,
        crash_is_interesting=True,
        timeout_is_interesting=False,
    )
    started = time.monotonic()
    try:
        observation = FailureCheckRunner(failure_check_config).observe(pid_path)
        assert time.monotonic() - started < 2
        assert observation.timed_out is hang
        assert observation.outcome is (
            FailureCheckResult.INCONCLUSIVE if hang else FailureCheckResult.INTERESTING
        )
        assert observation.stdout == "assertion A\n"
        assert observation.output_truncated  # EOF was never observed.
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_large_output_on_both_streams_is_drained_and_bounded(tmp_path):
    failure_check_config = make_failure_check_config(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'x' * 2097152); os.write(2, b'y' * 2097152)",
        ],
        timeout_seconds=5,
        crash_is_interesting=True,
        timeout_is_interesting=False,
    )
    observation = FailureCheckRunner(failure_check_config, log_limit=1024).observe(
        tmp_path / "model.onnx"
    )
    assert observation.outcome is FailureCheckResult.INTERESTING
    assert not observation.timed_out
    assert observation.stdout == "x" * 1024
    assert observation.stderr == "y" * 1024
    assert observation.output_truncated
