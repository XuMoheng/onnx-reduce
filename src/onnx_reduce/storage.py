"""Atomic artifact writes shared by the adapter, engine and CLI."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from onnx_reduce.performance import count, measured


@measured("write")
def atomic_write(path: str | Path, payload: bytes) -> None:
    """Replace a file using a unique temporary in the same directory.

    Readers see either complete version. This is process-interruption safety,
    not a guarantee of durability across power loss (no fsync).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            written = stream.write(payload)
            count("written_bytes", written)
            count("write_calls")
            stream.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


@measured("write")
def write_text(path: Path, text: str, *, append: bool = False) -> None:
    """Write UTF-8 evidence and count bytes submitted to the filesystem."""
    with path.open("a" if append else "w", encoding="utf-8") as stream:
        stream.write(text)
    count("written_bytes", len(text.encode("utf-8")))
    count("write_calls")
