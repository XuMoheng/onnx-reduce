"""Persist accepted models before atomically replacing their checkpoint manifest."""

from __future__ import annotations

import time
from pathlib import Path

from onnx_reduce.performance import measured
from onnx_reduce.report import ReductionReport
from onnx_reduce.storage import checkpoint_artifact, write_json
from onnx_reduce.types import ModelAdapter


@measured("checkpoint")
def checkpoint(
    model: ModelAdapter,
    report: ReductionReport,
    run_dir: Path,
    started: float,
    known_artifacts: dict | None = None,
) -> None:
    # Model is self-contained and atomically replaced before the manifest.
    artifact = checkpoint_artifact(
        model, run_dir / "checkpoint" / f"{model.canonical_hash()}.onnx", known_artifacts
    )

    report.final = model.cost()
    report.final_hash = artifact.digest
    report.duration_seconds = time.monotonic() - started
    write_json(
        run_dir / "checkpoint.json",
        {
            "model": str(artifact.model_path),
            "digest": artifact.digest,
            "report": report.to_dict(),
        },
    )
