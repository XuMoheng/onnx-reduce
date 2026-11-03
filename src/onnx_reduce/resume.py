"""Restore target identity, never scheduler cursors or verdict caches, across runs."""

from __future__ import annotations

import json
from pathlib import Path


def restore_target(run_directory, *, model_digest, config, failure_check_config, matcher):
    directory = Path(run_directory).resolve()
    try:
        task = json.loads((directory / "task.json").read_text())
        target = json.loads((directory / "target.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        normalized_check = json.loads(json.dumps(failure_check_config))
        if failure_check_config is None or task["failure_check_config"] != normalized_check:
            raise ValueError("failure check configuration differs or cannot be verified")
        if task["cwd"] != str(Path.cwd()):
            raise ValueError("failure check working directory differs")
        for key in ("target_mode", "signature_regex", "validation_mode"):
            if task["engine"][key] != getattr(config, key):
                raise ValueError(f"{key} differs from the original run")
        if (
            target["mode"] != config.target_mode
            or target["signature_regex"] != config.signature_regex
            or target["baseline_identity"] != replay["baseline_identity"]
        ):
            raise ValueError("target identity manifest is inconsistent")
        if (
            replay["digest"] != model_digest
            or report["final_hash"] != model_digest
            or report["stop_reason"] == "unstable_check"
            or replay["stop_reason"] != report["stop_reason"]
        ):
            raise ValueError("input does not match the verified output of the original run")
        identity = target["baseline_identity"]
        if not isinstance(identity, list) or len(identity) != 2:
            raise ValueError("invalid baseline identity")
        termination, signature = identity
        if (config.target_mode == "baseline" and not isinstance(termination, str)) or (
            config.target_mode == "script" and termination is not None
        ):
            raise ValueError("invalid baseline termination")
        if isinstance(signature, list):
            if not all(item is None or isinstance(item, str) for item in signature):
                raise ValueError("invalid baseline signature")
            signature = tuple(signature)
        elif signature is not None and not isinstance(signature, str):
            raise ValueError("invalid baseline signature")
        matcher.baseline_identity = (termination, signature)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot finalize from {directory}: {exc}") from exc
    return str(directory)
