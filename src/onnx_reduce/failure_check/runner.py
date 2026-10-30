"""Execute checks and retain bounded observations independently of matching."""

from __future__ import annotations

import os
import selectors
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from onnx_reduce.failure_check.contract import (
    FailureCheckConfig,
    FailureCheckResult,
    classify_return_code,
)


class FailureCheckRunnable(Protocol):
    def run(self, model_path: str | os.PathLike[str]) -> FailureCheckResult: ...


class FailureCheckError(RuntimeError):
    """The interestingness command could not be executed."""


@dataclass(frozen=True)
class FailureCheckObservation:
    outcome: FailureCheckResult
    returncode: int | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False

    @property
    def termination(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.returncode is None:
            return "script_contract"
        if self.returncode < 0:
            return f"signal:{-self.returncode}"
        return f"exit:{self.returncode}"


class FailureCheckRunner:
    """Append the model path to command argv; preserve the caller's cwd.

    run() retains the original enum API. observe() exposes execution evidence.
    Output is drained continuously, retaining at most log_limit bytes per stream.
    """

    def __init__(self, failure_check_config: FailureCheckConfig, *, log_limit: int = 1024 * 1024):
        if log_limit <= 0:
            raise ValueError("log_limit must be positive")
        self.failure_check_config = failure_check_config
        self.log_limit = log_limit

    @property
    def display_command(self) -> str:
        return shlex.join(self.failure_check_config.command)

    def run(self, model_path: str | os.PathLike[str]) -> FailureCheckResult:
        return self.observe(model_path).outcome

    def observe(self, model_path: str | os.PathLike[str]) -> FailureCheckObservation:
        started = time.monotonic()
        argv = (*self.failure_check_config.command, os.fspath(model_path))
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise FailureCheckError(
                f"cannot execute interestingness test {self.display_command!r}: {exc}"
            ) from exc

        stdout, stderr, timed_out, truncated = self._collect_output(process, started)

        if timed_out:
            outcome = (
                FailureCheckResult.INTERESTING
                if self.failure_check_config.timeout_is_interesting
                else FailureCheckResult.INCONCLUSIVE
            )
        else:
            outcome = classify_return_code(process.returncode, self.failure_check_config)
        return FailureCheckObservation(
            outcome=outcome,
            returncode=process.returncode,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
            stdout=stdout,
            stderr=stderr,
            output_truncated=truncated,
        )

    def _collect_output(
        self,
        process: subprocess.Popen[bytes],
        started: float,
    ) -> tuple[str, str, bool, bool]:
        """Drain both pipes without letting inherited descriptors block cleanup."""
        buffers = [bytearray(), bytearray()]
        truncated = False
        timed_out = False
        with selectors.DefaultSelector() as selector:

            def drain(timeout: float) -> None:
                nonlocal truncated
                for key, _ in selector.select(timeout):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = buffers[key.data]
                    remaining = self.log_limit - len(buffer)
                    buffer.extend(chunk[:remaining])
                    truncated |= len(chunk) > remaining

            try:
                for index, stream in enumerate((process.stdout, process.stderr)):
                    selector.register(stream, selectors.EVENT_READ, index)
                deadline = started + self.failure_check_config.timeout_seconds
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    drain(min(remaining, 0.05))
            finally:
                # Kill same-group descendants on normal exit and interruption too.
                _kill_process_group(process)
                process.wait()
                # A descendant may have called setsid() and kept the pipes open.
                # Retain buffered output, but bound this final drain even if that
                # descendant keeps writing. Missing signatures stay inconclusive.
                drain_deadline = time.monotonic() + 0.1
                try:
                    while selector.get_map() and time.monotonic() < drain_deadline:
                        drain(0.01)
                    truncated |= bool(selector.get_map())
                finally:
                    process.stdout.close()
                    process.stderr.close()
        return (
            buffers[0].decode("utf-8", errors="replace"),
            buffers[1].decode("utf-8", errors="replace"),
            timed_out,
            truncated,
        )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
