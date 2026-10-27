"""The interestingness contract.

The direction is intentionally the same as C-Vise: exit code zero means "the
failure is still present".  Compiler crashes are first-class interesting
outcomes because internal compiler errors are the primary M1 use case.
"""

from __future__ import annotations

import math
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class OracleResult(Enum):
    INTERESTING = "interesting"
    NOT_INTERESTING = "not_interesting"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class OracleSpec:
    command: tuple[str, ...]
    timeout_seconds: float = 60.0
    crash_is_interesting: bool = True
    timeout_is_interesting: bool = False
    inconclusive_exit_codes: frozenset[int] = field(default_factory=lambda: frozenset({125}))

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            raise ValueError("command must be an argv sequence; use make_spec for shell quoting")
        command = tuple(self.command)
        if (
            not command
            or not command[0]
            or any(not isinstance(part, str) or "\0" in part for part in command)
        ):
            raise ValueError(
                "the interestingness test command must contain a valid executable and argv"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        codes = frozenset(self.inconclusive_exit_codes)
        if any(type(code) is not int or not 1 <= code <= 255 for code in codes):
            raise ValueError("inconclusive exit codes must be integers from 1 to 255")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "inconclusive_exit_codes", codes)


def make_spec(
    command: str | Iterable[str],
    *,
    timeout_seconds: float,
    crash_is_interesting: bool,
    timeout_is_interesting: bool,
    inconclusive_exit_codes: Iterable[int] = (125,),
) -> OracleSpec:
    """Build an oracle specification from CLI arguments."""

    if isinstance(command, str):
        parts: tuple[str, ...] = tuple(shlex.split(command))
    else:
        parts = tuple(command)
    return OracleSpec(
        command=parts,
        timeout_seconds=timeout_seconds,
        crash_is_interesting=crash_is_interesting,
        timeout_is_interesting=timeout_is_interesting,
        inconclusive_exit_codes=frozenset(inconclusive_exit_codes),
    )


def classify_return_code(code: int, spec: OracleSpec) -> OracleResult:
    """Map a child process return code to the public outcome vocabulary."""

    if code == 0:
        return OracleResult.INTERESTING
    if code < 0:
        if spec.crash_is_interesting:
            return OracleResult.INTERESTING
        return OracleResult.NOT_INTERESTING
    if code in spec.inconclusive_exit_codes:
        return OracleResult.INCONCLUSIVE
    return OracleResult.NOT_INTERESTING
