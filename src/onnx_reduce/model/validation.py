"""Model costs, domain policy, checker validation and independent artifact rechecks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import onnx

from onnx_reduce.performance import count, measured
from onnx_reduce.preparation import current_preparation, metadata_key, preparation_caches
from onnx_reduce.types import (
    ModelArtifact,
    ModelCost,
    ValidationResult,
)

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel

BUILTIN_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml", "ai.onnx.training"})


def custom_domains(model: OnnxModel) -> tuple[str, ...]:
    """Collect non-builtin operator domains from the main graph and subgraphs."""

    caches = current_preparation()
    key = metadata_key("domains", model.canonical_hash()) if caches else None
    cached = caches.domains.get(key) if caches else None
    if cached is not None:
        return tuple(json.loads(cached))
    count("domain_scans")
    if model._ir_model is None:
        domains = {
            n.domain
            for n in model._snapshot.graph.node
            if n.domain and n.domain not in BUILTIN_DOMAINS
        }
    else:
        domains = {
            n.domain
            for g in model._model.graphs()
            for n in g
            if n.domain and n.domain not in BUILTIN_DOMAINS
        }
    result = tuple(sorted(domains))
    if caches:
        caches.domains.put(key, json.dumps(result).encode())
    return result


@measured("validation")
def validate(model: OnnxModel, mode: str = "strict", *, use_cache: bool = True) -> ValidationResult:
    """Check the model and report validity, the failure reason, and the effective mode."""

    if mode not in {"strict", "structural", "permissive"}:
        raise ValueError(f"unknown validation mode: {mode}")
    graph = model._snapshot.graph
    domains = model.custom_domains()
    if not graph.output:
        return ValidationResult(False, "graph has no outputs", mode, domains)
    output_names = [v.name for v in graph.output]
    if any(not name for name in output_names):
        return ValidationResult(False, "graph output has an empty name", mode, domains)
    if len(set(output_names)) != len(output_names):
        return ValidationResult(False, "graph output names are not unique", mode, domains)
    defined = {v.name for v in graph.input} | {v.name for v in graph.initializer}
    defined.update(name for n in graph.node for name in n.output)
    for name in output_names:
        if name not in defined:
            return ValidationResult(False, f"graph output {name!r} has no producer", mode, domains)

    # Downgrade strict to structural when custom domains are present.
    # Permissive mode performs only the basic output checks above.
    effective_mode = "structural" if mode == "strict" and domains else mode
    if effective_mode == "permissive":
        return ValidationResult(True, None, effective_mode, domains)
    caches = current_preparation() if use_cache else None
    key = (
        metadata_key("validation", model.canonical_hash(), mode, effective_mode) if caches else None
    )
    cached = caches.validations.get(key) if caches else None
    if cached is not None:
        data = json.loads(cached)
        data["custom_domains"] = tuple(data["custom_domains"])
        return ValidationResult(**data)
    count("checker_calls")
    try:
        onnx.checker.check_model(
            model._payload,
            full_check=True,
            check_custom_domain=effective_mode == "strict",
        )
    except Exception as exc:  # Checker failures describe an invalid candidate.
        result = ValidationResult(False, f"onnx checker: {exc}", effective_mode, domains)
    else:
        result = ValidationResult(True, None, effective_mode, domains)
    if caches:
        caches.validations.put(key, json.dumps(asdict(result)).encode())
    return result


@measured("artifact_validation")
def validate_artifact(model: OnnxModel, artifact: ModelArtifact, mode: str) -> ValidationResult:
    """Check actual bytes, then reload and validate independently of the source IR."""
    with artifact.model_path.open("rb") as stream:
        actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_digest != artifact.digest or actual_digest != model.canonical_hash():
        return ValidationResult(False, "output artifact digest mismatch", mode)
    from onnx_reduce.model.adapter import OnnxModel

    loaded = OnnxModel.load(artifact.model_path)
    with preparation_caches(False, mode):
        return loaded.validate(mode)


def cost(model: OnnxModel) -> ModelCost:
    """Collect recursive structural and tensor costs; ModelCost.key selects priority."""

    from onnx_reduce.model.operations.graphs import walk_graphs
    from onnx_reduce.model.operations.tensors import tensor_elements

    graphs = [graph for _, graph in walk_graphs(model._snapshot)]
    return ModelCost(
        node_count=sum(len(g.node) for g in graphs),
        byte_size=len(model._payload),
        output_count=sum(len(g.output) for g in graphs),
        input_count=sum(len(g.input) for g in graphs),
        initializer_count=sum(len(g.initializer) for g in graphs),
        tensor_elements=tensor_elements(model._snapshot),
    )
