"""ONNX model adapter implemented on top of ``onnxscript.ir``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType

import onnx
from google.protobuf.message import DecodeError
from onnxscript import ir

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs, DropOutputs
from onnx_reduce.performance import count, measured
from onnx_reduce.preparation import current_preparation, metadata_key
from onnx_reduce.storage import atomic_write
from onnx_reduce.types import (
    EditApplicationError,
    ModelArtifact,
    ModelCost,
    ValidationResult,
)

BUILTIN_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml", "ai.onnx.training"})


@dataclass(frozen=True)
class TensorLinks:
    """Summarize a main-graph value's links; the IR owns the actual connections."""

    # ==========================================
    # Attributes: populated by tensor_index(), with no graph edits
    # ==========================================
    producer: int | None  # Main-graph producer index, or None if there is no producer
    consumers: tuple[int, ...]  # Main-graph consumer indices
    is_graph_input: bool
    is_graph_output: bool
    is_initializer: bool


class OnnxModel:
    """An immutable public view of an owned ONNX IR model.

    Edits mutate only a fresh clone, before its serialized snapshot is read.
    Callers passing an IR model to the constructor transfer ownership and must
    not subsequently mutate it or its tensors. to_proto() returns a fresh copy.
    """

    # ==========================================
    # Initialization, Model Conversion, and Copying
    # ==========================================

    def __init__(self, model: ir.Model, *, _node_ids: tuple[int, ...] | None = None):
        """Take ownership of the IR model; callers must stop mutating it and its tensors."""

        self._model = model  # Owned IR; edits are applied only to fresh clones
        ids = range(model.graph.num_nodes()) if _node_ids is None else _node_ids
        self._node_ids = dict(zip(model.graph, ids, strict=True))
        self._refresh_node_indices()

    def _refresh_node_indices(self) -> None:
        # Keep only live objects. Identity metadata never enters the ONNX payload.
        self._node_ids = {node: self._node_ids[node] for node in self._model.graph}
        self._ordered_node_ids = tuple(self._node_ids.values())
        self._id_to_index = MappingProxyType(
            {node_id: index for index, node_id in enumerate(self._ordered_node_ids)}
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> OnnxModel:
        # M1 uses self-contained tensors. This intentionally loads weights into
        # memory; lazy/COW storage can replace it without changing the engine.
        """Load an ONNX file and its external weights, then convert the model to IR."""

        try:
            return cls._from_owned_proto(onnx.load_model(path, load_external_data=True))
        except DecodeError as exc:
            raise ValueError(f"cannot decode ONNX model {os.fspath(path)!r}: {exc}") from exc
        except onnx.checker.ValidationError as exc:
            # Loading external data also invokes ONNX path/bounds checks.
            raise ValueError(f"cannot load ONNX model {os.fspath(path)!r}: {exc}") from exc
        except ir.serde.SerdeError as exc:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            raise ValueError(
                f"cannot import ONNX model {os.fspath(path)!r} into IR: {cause}"
            ) from exc

    @classmethod
    def from_proto(cls, proto: onnx.ModelProto) -> OnnxModel:
        # IR tensor wrappers retain references to TensorProto. Isolate caller
        # mutations before caching any serialized representation. load() already
        # owns its protobuf and can avoid this extra copy of large weights.
        """Copy and import a ModelProto to isolate subsequent mutations by the caller."""

        owned = onnx.ModelProto()
        owned.CopyFrom(proto)
        return cls._from_owned_proto(owned)

    @classmethod
    def _from_owned_proto(cls, proto: onnx.ModelProto) -> OnnxModel:
        """Convert an already-owned ModelProto to IR without another copy."""

        converted = ir.from_proto(proto)
        if not isinstance(converted, ir.Model):
            raise TypeError("proto did not convert to an onnxscript IR Model")
        return cls(converted)

    @measured("clone")
    def copy(self) -> OnnxModel:
        """Clone the IR for candidate edits while preserving the original model."""

        return OnnxModel(self._model.clone(), _node_ids=self.node_ids)

    @measured("export")
    def to_proto(self) -> onnx.ModelProto:
        """Export a ModelProto that can be modified independently."""

        converted = ir.to_proto(self._model)
        if not isinstance(converted, onnx.ModelProto):
            raise TypeError("onnxscript IR did not convert to a ModelProto")
        return converted

    # ==========================================
    # Read-only Properties
    # ==========================================

    @property
    def graph_name(self) -> str:
        """Return the main-graph name, or an empty string if unnamed."""

        return self._model.graph.name or ""

    @property
    def node_count(self) -> int:
        """Return the main-graph node count, excluding nodes inside attribute subgraphs."""

        return self._model.graph.num_nodes()

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Return stable lineage IDs in the current main-graph order."""

        return self._ordered_node_ids

    @property
    def id_to_index(self):
        """Expose an immutable index mapping for this model version."""

        return self._id_to_index

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return main-graph output names in their current order."""

        return tuple(value.name or "" for value in self._model.graph.outputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return main-graph input names in their current order."""

        return tuple(value.name or "" for value in self._model.graph.inputs)

    @property
    def op_types(self) -> tuple[str, ...]:
        """Return operator types in main-graph node order."""

        return tuple(node.op_type for node in self._model.graph)

    # ==========================================
    # Graph Queries: Candidate Generation and Connection Inspection
    # ==========================================

    def tensor_index(self) -> dict[str, TensorLinks]:
        """Summarize main-graph value links and roles for inspection and tests."""

        graph = self._model.graph
        links: dict[str, TensorLinks] = {}

        def record(name: str) -> TensorLinks:
            return links.setdefault(
                name,
                TensorLinks(
                    producer=None,
                    consumers=(),
                    is_graph_input=False,
                    is_graph_output=False,
                    is_initializer=False,
                ),
            )

        for value in graph.inputs:
            if value.name:
                links[value.name] = replace(record(value.name), is_graph_input=True)
        for value in graph.initializers.values():
            if value.name:
                links[value.name] = replace(record(value.name), is_initializer=True)
        for index, node in enumerate(graph):
            for value in node.outputs:
                if value is not None and value.name:
                    links[value.name] = replace(record(value.name), producer=index)
            for value in self._node_dependencies(node):
                # Captured values belong to their defining scope, even when a
                # nested graph contains a local value with the same name.
                if value.graph is not graph or not value.name:
                    continue
                item = record(value.name)
                links[value.name] = replace(
                    item,
                    consumers=tuple(sorted({*item.consumers, index})),
                )
        for value in graph.outputs:
            if value.name:
                links[value.name] = replace(record(value.name), is_graph_output=True)
        return links

    def dead_node_indices(self) -> tuple[int, ...]:
        """Return main-graph node indices outside the backward closure of all outputs."""

        # Use a stack to trace producers backward from every graph output.
        needed_values = list(self._model.graph.outputs)
        # Track object identities to distinguish same-named values in different scopes.
        seen_values: set[int] = set()
        needed_nodes: set[int] = set()
        while needed_values:
            value = needed_values.pop()
            if id(value) in seen_values:
                continue
            seen_values.add(id(value))
            producer = value.producer()
            if producer is None or id(producer) in needed_nodes:
                continue
            needed_nodes.add(id(producer))
            needed_values.extend(self._node_dependencies(producer))
        return tuple(
            index for index, node in enumerate(self._model.graph) if id(node) not in needed_nodes
        )

    def unused_input_names(self) -> tuple[str, ...]:
        """Find graph inputs with no consumers that are not themselves graph outputs."""

        return tuple(
            value.name
            for value in self._model.graph.inputs
            if not value.uses() and not value.is_graph_output()
        )

    def unused_initializer_names(self) -> tuple[str, ...]:
        """Find initializers with no consumers that are not themselves graph outputs."""

        return tuple(
            name
            for name, value in self._model.graph.initializers.items()
            if not value.uses() and not value.is_graph_output()
        )

    def custom_domains(self) -> tuple[str, ...]:
        """Collect non-builtin operator domains from the main graph and subgraphs."""

        domains = {
            node.domain
            for graph in self._model.graphs()
            for node in graph
            if node.domain and node.domain not in BUILTIN_DOMAINS
        }
        return tuple(sorted(domains))

    # ==========================================
    # Dependency Traversal: Internal Helpers for Queries and Node Deletion
    # ==========================================

    @staticmethod
    def _nested_graphs(node):
        """Yield subgraphs directly held in node attributes, without recursion."""

        for attribute in node.attributes.values():
            if attribute.is_ref():
                continue
            if attribute.type == ir.AttributeType.GRAPH:
                yield attribute.as_graph()
            elif attribute.type == ir.AttributeType.GRAPHS:
                yield from attribute.as_graphs()

    @classmethod
    def _node_dependencies(cls, node):
        """Yield inputs and subgraph dependencies, conservatively covering outer captures."""

        yield from (value for value in node.inputs if value is not None)
        # A live control-flow node can depend on outer values captured anywhere
        # in its bodies. Conservatively retain every captured dependency.
        for graph in cls._nested_graphs(node):
            yield from graph.outputs
            for child in graph:
                yield from cls._node_dependencies(child)

    # ==========================================
    # Candidate Editing: Public Entry Point and Internal Implementations
    # ==========================================

    @measured("clone_and_edit")
    def apply_candidate(self, candidate: Candidate) -> OnnxModel:
        """Verify the base and apply edits to a clone; the engine decides acceptance."""

        if candidate.base_digest is not None and candidate.base_digest != self.canonical_hash():
            raise EditApplicationError("candidate base digest does not match the model")
        result = self.copy()
        # Each edit addresses the result of the preceding edit.
        for edit_index, edit in enumerate(candidate.edits):
            if isinstance(edit, DeleteNodes):
                result._delete_nodes(
                    edit.indices, inference_source=self if edit_index == 0 else result
                )
            elif isinstance(edit, DropOutputs):
                result._drop_outputs(edit.names)
            elif isinstance(edit, DropInputs):
                result._drop_inputs(edit.names)
            elif isinstance(edit, DropInitializers):
                result._drop_initializers(edit.names)
            else:
                raise EditApplicationError(f"unsupported edit: {type(edit).__name__}")
            # An intermediate edit may have acquired a serialized snapshot for
            # inference. Subsequent edits must never see that stale snapshot.
            result.__dict__.pop("_payload", None)
            result.__dict__.pop("_digest", None)
        return result

    def _delete_nodes(self, indices: Iterable[int], *, inference_source: OnnxModel) -> None:
        """Delete main-graph nodes and promote still-used boundary outputs to graph inputs."""

        graph = self._model.graph
        # 1. Resolve indices in current main-graph order and deduplicate selections.
        nodes = list(graph)
        selected = []
        selected_ids: set[int] = set()
        for index in indices:
            if index < 0 or index >= len(nodes):
                raise EditApplicationError(f"node index {index} is out of range")
            node = nodes[index]
            if id(node) not in selected_ids:
                selected_ids.add(id(node))
                selected.append(node)
        if not selected:
            return

        # 2. Include attribute-subgraph nodes whose input references must also be detached.
        detached_nodes = list(selected)

        def include_nested(node):
            for nested in self._nested_graphs(node):
                for child in nested:
                    selected_ids.add(id(child))
                    detached_nodes.append(child)
                    include_nested(child)

        for node in selected:
            include_nested(node)

        # 3. Find boundary outputs still needed outside the deleted region.
        promoted_values: list[ir.Value] = []
        for node in selected:
            for value in node.outputs:
                if value is None:
                    continue
                has_external_consumer = any(
                    id(usage.node) not in selected_ids for usage in value.uses()
                )
                if has_external_consumer or value.is_graph_output():
                    promoted_values.append(value)

        # 4. Infer missing information before detaching, then replace boundaries with inputs.
        inferred_types = inference_source._infer_value_types(
            value for value in promoted_values if value.type is None or value.shape is None
        )
        # Remove use-def edges from deleted attribute graphs too. Otherwise
        # captured inputs/weights retain ghost consumers until another clone.
        for node in detached_nodes:
            for input_index, input_value in enumerate(node.inputs):
                if input_value is not None:
                    node.replace_input_with(input_index, None)
        for promoted, value in enumerate(promoted_values):
            inferred = inferred_types.get(value.name, value)
            replacement = ir.Value(
                name=value.name or f"onnx_reduce_promoted_{promoted}",
                type=inferred.type,
                shape=inferred.shape,
                doc_string=value.doc_string,
            )
            value.replace_all_uses_with(replacement, replace_graph_outputs=True)
            graph.inputs.append(replacement)
        # 5. Remove nodes and restore topological order; leave input/weight cleanup separate.
        graph.remove(selected)
        graph.sort()
        self._refresh_node_indices()

    def _drop_outputs(self, names: Iterable[str]) -> None:
        """Drop named graph outputs, keeping at least one and preserving producer nodes."""

        graph = self._model.graph
        selected = set(names)
        outputs = [value for value in graph.outputs if value.name not in selected]
        if not outputs:
            raise EditApplicationError("cannot drop every graph output")
        missing = selected.difference(value.name for value in graph.outputs)
        if missing:
            raise EditApplicationError(f"unknown graph output(s): {', '.join(sorted(missing))}")
        graph.outputs.clear()
        graph.outputs.extend(outputs)

    def _drop_inputs(self, names: Iterable[str]) -> None:
        """Drop named graph inputs only if they are unused."""

        selected = set(names)
        if not selected <= set(self.unused_input_names()):
            raise EditApplicationError("can only drop unused graph inputs")
        graph = self._model.graph
        remaining = [value for value in graph.inputs if value.name not in selected]
        graph.inputs.clear()
        graph.inputs.extend(remaining)

    def _drop_initializers(self, names: Iterable[str]) -> None:
        """Drop named initializers only if they are unused."""

        selected = set(names)
        if not selected <= set(self.unused_initializer_names()):
            raise EditApplicationError("can only drop unused initializers")
        for name in selected:
            del self._model.graph.initializers[name]

    def _infer_value_types(self, values: Iterable[ir.Value]) -> dict[str, ir.Value]:
        """Infer types and shapes for main-graph promotions; return an empty mapping on failure."""

        missing = [value for value in values if value.name]
        if not missing:
            return {}
        caches = current_preparation()
        key = metadata_key("shape", self.canonical_hash()) if caches is not None else None
        summary = caches.shapes.get(key) if caches is not None else None
        if summary is None:
            summary = self._infer_type_summary()
            if caches is not None:
                caches.shapes.put(key, summary)
        # Empty bytes are a cached failure, distinct from a cache miss (None).
        graph = onnx.GraphProto.FromString(summary)
        wanted = {value.name for value in missing}
        return {
            value.name: ir.serde.deserialize_value_info_proto(value, None)
            for value in graph.value_info
            if value.name in wanted
        }

    @measured("shape_inference")
    def _infer_type_summary(self) -> bytes:
        """Retain only main-graph names and types, without weights or IR references."""

        try:
            inferred_proto = onnx.shape_inference.infer_shapes(self._payload)
            # Match IR import semantics, including main-graph outputs whose types
            # come from graph.output instead of intermediate value_info.
            inferred_model = ir.from_proto(inferred_proto)
            if not isinstance(inferred_model, ir.Model):
                return b""
            graph = inferred_model.graph
            available = {value.name: value for value in graph.inputs}
            for node in graph:
                available.update({value.name: value for value in node.outputs if value.name})
            summary = onnx.GraphProto()
            for value in available.values():
                info = summary.value_info.add()
                info.name = value.name
                info.type.CopyFrom(ir.serde.serialize_value(value).type)
            return summary.SerializeToString(deterministic=True)
        except Exception:
            count("shape_inference_failures")
            # An untyped promotion will be rejected by the configured checker.
            return b""

    # ==========================================
    # Cost Calculation and Model Validation
    # ==========================================

    def cost(self) -> ModelCost:
        """Build a lexicographic cost: nodes, bytes, outputs, inputs, and initializers."""

        graph = self._model.graph
        return ModelCost(
            node_count=graph.num_nodes(),
            byte_size=len(self._payload),
            output_count=len(graph.outputs),
            input_count=len(graph.inputs),
            initializer_count=len(graph.initializers),
        )

    @measured("validation")
    def validate(self, mode: str = "strict") -> ValidationResult:
        """Check the model and report validity, the failure reason, and the effective mode."""

        if mode not in {"strict", "structural", "permissive"}:
            raise ValueError(f"unknown validation mode: {mode}")
        graph = self._model.graph
        domains = self.custom_domains()
        if len(graph.outputs) == 0:
            return ValidationResult(False, "graph has no outputs", mode, domains)
        output_names = [value.name for value in graph.outputs]
        if any(not name for name in output_names):
            return ValidationResult(False, "graph output has an empty name", mode, domains)
        if len(set(output_names)) != len(output_names):
            return ValidationResult(False, "graph output names are not unique", mode, domains)
        for value in graph.outputs:
            if (
                value.producer() is None
                and not value.is_graph_input()
                and not value.is_initializer()
            ):
                return ValidationResult(
                    False,
                    f"graph output {value.name!r} has no producer",
                    mode,
                    domains,
                )

        # Downgrade strict to structural when custom domains are present.
        # Permissive mode performs only the basic output checks above.
        effective_mode = "structural" if mode == "strict" and domains else mode
        if effective_mode == "permissive":
            return ValidationResult(True, None, effective_mode, domains)
        try:
            onnx.checker.check_model(
                self._payload,
                full_check=True,
                check_custom_domain=effective_mode == "strict",
            )
        except Exception as exc:  # Checker failures describe an invalid candidate.
            return ValidationResult(False, f"onnx checker: {exc}", effective_mode, domains)
        return ValidationResult(True, None, effective_mode, domains)

    # ==========================================
    # Serialized Snapshots, Content Identity, and File Output
    # ==========================================

    @cached_property
    @measured("serialization")
    def _payload(self) -> bytes:
        """Cache deterministic bytes on first access for cost, hashing, and file output."""

        return self.to_proto().SerializeToString(deterministic=True)

    @cached_property
    @measured("hash")
    def _digest(self) -> str:
        """Cache the SHA-256 of serialized bytes as the exact content identity."""

        return hashlib.sha256(self._payload).hexdigest()

    def canonical_hash(self) -> str:
        """Return the exact serialized-content hash, without semantic canonicalization."""
        return self._digest

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically write the serialized snapshot to the destination."""

        atomic_write(Path(path), self._payload)

    def materialize(self, path: str | Path) -> ModelArtifact:
        """Save a self-contained ONNX file with weights and return its path, hash, and size."""
        destination = Path(path).resolve()
        self.save(destination)
        # Cost, digest and file are derived from exactly the same snapshot.
        return ModelArtifact(destination, (), self._digest, len(self._payload))
