"""ONNX model adapter implemented on top of ``onnxscript.ir``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import onnx
from google.protobuf.message import DecodeError
from onnxscript import ir

from onnx_reduce.edits import Candidate, DeleteNodes, DropInitializers, DropInputs, DropOutputs
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
    producer: int | None
    consumers: tuple[int, ...]
    is_graph_input: bool
    is_graph_output: bool
    is_initializer: bool


class OnnxModel:
    """An immutable public view of an owned ONNX IR model.

    Edits mutate only a fresh clone, before its serialized snapshot is read.
    Callers passing an IR model to the constructor transfer ownership and must
    not subsequently mutate it or its tensors. to_proto() returns a fresh copy.
    """

    def __init__(self, model: ir.Model):
        self._model = model

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> OnnxModel:
        # M1 uses self-contained tensors. This intentionally loads weights into
        # memory; lazy/COW storage can replace it without changing the engine.
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
        owned = onnx.ModelProto()
        owned.CopyFrom(proto)
        return cls._from_owned_proto(owned)

    @classmethod
    def _from_owned_proto(cls, proto: onnx.ModelProto) -> OnnxModel:
        converted = ir.from_proto(proto)
        if not isinstance(converted, ir.Model):
            raise TypeError("proto did not convert to an onnxscript IR Model")
        return cls(converted)

    def copy(self) -> OnnxModel:
        return OnnxModel(self._model.clone())

    @property
    def graph_name(self) -> str:
        return self._model.graph.name or ""

    @property
    def node_count(self) -> int:
        return self._model.graph.num_nodes()

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(value.name or "" for value in self._model.graph.outputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(value.name or "" for value in self._model.graph.inputs)

    @property
    def op_types(self) -> tuple[str, ...]:
        return tuple(node.op_type for node in self._model.graph)

    def to_proto(self) -> onnx.ModelProto:
        converted = ir.to_proto(self._model)
        if not isinstance(converted, onnx.ModelProto):
            raise TypeError("onnxscript IR did not convert to a ModelProto")
        return converted

    def save(self, path: str | os.PathLike[str]) -> None:
        atomic_write(Path(path), self._payload)

    @cached_property
    def _payload(self) -> bytes:
        return self.to_proto().SerializeToString(deterministic=True)

    @cached_property
    def _digest(self) -> str:
        return hashlib.sha256(self._payload).hexdigest()

    def cost(self) -> ModelCost:
        graph = self._model.graph
        return ModelCost(
            node_count=graph.num_nodes(),
            byte_size=len(self._payload),
            output_count=len(graph.outputs),
            input_count=len(graph.inputs),
            initializer_count=len(graph.initializers),
        )

    def materialize(self, path: str | Path) -> ModelArtifact:
        """M1 produces a self-contained ONNX file, including tensor contents."""
        destination = Path(path).resolve()
        self.save(destination)
        # Cost, digest and file are derived from exactly the same snapshot.
        return ModelArtifact(destination, (), self._digest, len(self._payload))

    def tensor_index(self) -> dict[str, TensorLinks]:
        """Return producer/consumer links for values in the main graph."""

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

    def custom_domains(self) -> tuple[str, ...]:
        domains = {
            node.domain
            for graph in self._model.graphs()
            for node in graph
            if node.domain and node.domain not in BUILTIN_DOMAINS
        }
        return tuple(sorted(domains))

    def validate(self, mode: str = "strict") -> ValidationResult:
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

        effective_mode = "structural" if mode == "strict" and domains else mode
        if effective_mode == "permissive":
            return ValidationResult(True, None, effective_mode, domains)
        try:
            proto = self.to_proto()
            onnx.checker.check_model(
                proto,
                full_check=True,
                check_custom_domain=effective_mode == "strict",
            )
        except Exception as exc:  # Checker failures describe an invalid candidate.
            return ValidationResult(False, f"onnx checker: {exc}", effective_mode, domains)
        return ValidationResult(True, None, effective_mode, domains)

    def apply_candidate(self, candidate: Candidate) -> OnnxModel:
        if candidate.base_digest is not None and candidate.base_digest != self.canonical_hash():
            raise EditApplicationError("candidate base digest does not match the model")
        result = self.copy()
        # Each edit addresses the result of the preceding edit.
        for edit in candidate.edits:
            if isinstance(edit, DeleteNodes):
                result._delete_nodes(edit.indices)
            elif isinstance(edit, DropOutputs):
                result._drop_outputs(edit.names)
            elif isinstance(edit, DropInputs):
                result._drop_inputs(edit.names)
            elif isinstance(edit, DropInitializers):
                result._drop_initializers(edit.names)
            else:
                raise EditApplicationError(f"unsupported edit: {type(edit).__name__}")
        return result

    def dead_node_indices(self) -> tuple[int, ...]:
        """Return main-graph nodes outside the backward closure of all outputs."""

        needed_values = list(self._model.graph.outputs)
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

    @staticmethod
    def _nested_graphs(node):
        for attribute in node.attributes.values():
            if attribute.is_ref():
                continue
            if attribute.type == ir.AttributeType.GRAPH:
                yield attribute.as_graph()
            elif attribute.type == ir.AttributeType.GRAPHS:
                yield from attribute.as_graphs()

    @classmethod
    def _node_dependencies(cls, node):
        yield from (value for value in node.inputs if value is not None)
        # A live control-flow node can depend on outer values captured anywhere
        # in its bodies. Conservatively retain every captured dependency.
        for graph in cls._nested_graphs(node):
            yield from graph.outputs
            for child in graph:
                yield from cls._node_dependencies(child)

    def _delete_nodes(self, indices: Iterable[int]) -> None:
        graph = self._model.graph
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

        detached_nodes = list(selected)

        def include_nested(node):
            for nested in self._nested_graphs(node):
                for child in nested:
                    selected_ids.add(id(child))
                    detached_nodes.append(child)
                    include_nested(child)

        for node in selected:
            include_nested(node)

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

        inferred_types = self._infer_value_types(
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
        graph.remove(selected)
        graph.sort()

    def _infer_value_types(self, values: Iterable[ir.Value]) -> dict[str, ir.Value]:
        missing = [value for value in values if value.name]
        if not missing:
            return {}
        try:
            inferred_proto = onnx.shape_inference.infer_shapes(self.to_proto())
            inferred_model = ir.from_proto(inferred_proto)
            if not isinstance(inferred_model, ir.Model):
                return {}
            # Promotions are main-graph edits. Nested scopes can reuse local
            # names, so never flatten their value names into this lookup.
            graph = inferred_model.graph
            available = {value.name: value for value in graph.inputs}
            for node in graph:
                available.update({value.name: value for value in node.outputs if value.name})
            return {
                value.name: available[value.name] for value in missing if value.name in available
            }
        except Exception:
            # An untyped promotion will be rejected by the configured checker.
            # This is preferable to failing the whole reduction run.
            return {}

    def _drop_outputs(self, names: Iterable[str]) -> None:
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

    def unused_input_names(self) -> tuple[str, ...]:
        return tuple(
            value.name
            for value in self._model.graph.inputs
            if not value.uses() and not value.is_graph_output()
        )

    def unused_initializer_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in self._model.graph.initializers.items()
            if not value.uses() and not value.is_graph_output()
        )

    def _drop_inputs(self, names: Iterable[str]) -> None:
        selected = set(names)
        if not selected <= set(self.unused_input_names()):
            raise EditApplicationError("can only drop unused graph inputs")
        graph = self._model.graph
        remaining = [value for value in graph.inputs if value.name not in selected]
        graph.inputs.clear()
        graph.inputs.extend(remaining)

    def _drop_initializers(self, names: Iterable[str]) -> None:
        selected = set(names)
        if not selected <= set(self.unused_initializer_names()):
            raise EditApplicationError("can only drop unused initializers")
        for name in selected:
            del self._model.graph.initializers[name]

    def canonical_hash(self) -> str:
        """Exact serialized-content identity, not semantic canonicalization."""
        return self._digest
