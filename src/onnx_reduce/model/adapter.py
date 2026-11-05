"""ONNX model adapter implemented on top of ``onnxscript.ir``."""

from __future__ import annotations

import os
from collections.abc import Iterable
from copy import copy
from functools import cached_property
from pathlib import Path
from types import MappingProxyType

import onnx
from google.protobuf.message import DecodeError
from onnxscript import ir

from onnx_reduce.edits import (
    Candidate,
)
from onnx_reduce.model import editing, inference, protobuf, queries, validation
from onnx_reduce.model.queries import TensorLinks
from onnx_reduce.model.snapshot import ModelSnapshot
from onnx_reduce.performance import count, measured
from onnx_reduce.storage import atomic_write
from onnx_reduce.types import (
    EditOptions,
    ModelArtifact,
    ModelCost,
    RegionGraph,
    ValidationResult,
)


class OnnxModel:
    """Own an ONNX representation and expose queries, edits and validation.

    Edits operate on fresh clones. Callers transferring an IR model must stop
    mutating it and its tensors; to_proto() returns an independent copy.
    """

    def __init__(
        self,
        model: ir.Model,
        *,
        _node_ids: tuple[int, ...] | None = None,
        edit_options: EditOptions | None = None,
    ):
        """Take ownership of the IR model; callers must stop mutating it and its tensors."""

        self.edit_options = edit_options or EditOptions()
        self._ir_model = model  # Owned IR; edits are applied only to fresh clones
        self._proto_snapshot = None
        ids = range(model.graph.num_nodes()) if _node_ids is None else _node_ids
        self._node_ids = dict(zip(model.graph, ids, strict=True))
        self._refresh_node_indices()

    @property
    def _model(self):
        if self._ir_model is None:
            self._ir_model = self._import_snapshot()
            self._node_ids = dict(zip(self._ir_model.graph, self.node_ids, strict=True))
        return self._ir_model

    @measured("lazy_import")
    def _import_snapshot(self):
        count("lazy_ir_imports")
        return ir.from_proto(self._snapshot)

    @classmethod
    def _from_snapshot(cls, proto, ids, *, edit_options=None):
        obj = cls.__new__(cls)
        obj.edit_options = edit_options or EditOptions()
        obj._ir_model = None
        obj._proto_snapshot = ModelSnapshot(proto)
        obj._ordered_node_ids = ids
        obj._id_to_index = MappingProxyType({item: i for i, item in enumerate(ids)})
        return obj

    @property
    def _snapshot(self):
        return self.snapshot.proto

    @cached_property
    @measured("export")
    def snapshot(self):
        if self._proto_snapshot is not None:
            return self._proto_snapshot
        converted = ir.to_proto(self._model)
        if not isinstance(converted, onnx.ModelProto):
            raise TypeError("onnxscript IR did not convert to a ModelProto")
        return ModelSnapshot(converted)

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

        return OnnxModel(
            self._model.clone(), _node_ids=self.node_ids, edit_options=self.edit_options
        )

    def with_edit_options(self, options: EditOptions) -> OnnxModel:
        """Return a policy view; model content and lineage remain unchanged."""
        result = copy(self)
        result.edit_options = options
        return result

    def candidate_snapshot(self):
        """Return the shared read-only-by-contract ONNX view for pass queries."""
        return self._snapshot

    def to_proto(self) -> onnx.ModelProto:
        """Export an independently mutable copy of the normalized snapshot."""
        proto = onnx.ModelProto()
        proto.CopyFrom(self._snapshot)
        return proto

    @property
    def graph_name(self) -> str:
        """Return the main-graph name, or an empty string if unnamed."""

        return self._snapshot.graph.name if self._ir_model is None else self._model.graph.name or ""

    @property
    def node_count(self) -> int:
        """Return the main-graph node count, excluding nodes inside attribute subgraphs."""

        return (
            len(self._snapshot.graph.node)
            if self._ir_model is None
            else self._model.graph.num_nodes()
        )

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

        if self._ir_model is None:
            return tuple(v.name for v in self._snapshot.graph.output)
        return tuple(value.name or "" for value in self._model.graph.outputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return main-graph input names in their current order."""

        if self._ir_model is None:
            return tuple(v.name for v in self._snapshot.graph.input)
        return tuple(value.name or "" for value in self._model.graph.inputs)

    @property
    def op_types(self) -> tuple[str, ...]:
        """Return operator types in main-graph node order."""

        if self._ir_model is None:
            return tuple(n.op_type for n in self._snapshot.graph.node)
        return tuple(node.op_type for node in self._model.graph)

    def tensor_index(self) -> dict[str, TensorLinks]:
        return queries.tensor_index(self)

    @cached_property
    def _extraction_view(self):
        return queries.extraction_view(self)

    @cached_property
    def _region_graph(self) -> RegionGraph:
        return queries._region_graph(self)

    def region_graph(self) -> RegionGraph:
        return queries.region_graph(self)

    def dead_node_indices(self) -> tuple[int, ...]:
        return queries.dead_node_indices(self)

    def unused_input_names(self) -> tuple[str, ...]:
        return queries.unused_input_names(self)

    def unused_initializer_names(self) -> tuple[str, ...]:
        return queries.unused_initializer_names(self)

    def custom_domains(self) -> tuple[str, ...]:
        return validation.custom_domains(self)

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

    def apply_candidate(self, candidate: Candidate) -> OnnxModel:
        return editing.apply_candidate(self, candidate)

    def _delete_nodes(self, indices: Iterable[int], *, inference_source: OnnxModel) -> None:
        return editing._delete_nodes(self, indices, inference_source=inference_source)

    @classmethod
    def _graph_ordered(cls, graph):
        return editing._graph_ordered(cls, graph)

    def _drop_outputs(self, names: Iterable[str]) -> None:
        return editing._drop_outputs(self, names)

    def _drop_inputs(self, names: Iterable[str]) -> None:
        return editing._drop_inputs(self, names)

    def _drop_initializers(self, names: Iterable[str]) -> None:
        return editing._drop_initializers(self, names)

    def _infer_value_types(self, values: Iterable[ir.Value]) -> dict[str, ir.Value]:
        return inference._infer_value_types(self, values)

    def _infer_type_summary(self) -> bytes:
        return inference._infer_type_summary(self)

    def cost(self) -> ModelCost:
        return validation.cost(self)

    def validate(self, mode: str = "strict", *, use_cache: bool = True) -> ValidationResult:
        return validation.validate(self, mode, use_cache=use_cache)

    @cached_property
    def _payload(self) -> bytes:
        """Cache deterministic bytes on first access for cost, hashing, and file output."""

        return self.snapshot.payload

    @cached_property
    def _digest(self) -> str:
        """Cache the SHA-256 of serialized bytes as the exact content identity."""

        return self.snapshot.digest

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

    def validate_artifact(self, artifact: ModelArtifact, mode: str) -> ValidationResult:
        return validation.validate_artifact(self, artifact, mode)

    @cached_property
    def _fast_capability(self):
        return protobuf._fast_capability(self)

    @cached_property
    def _proto_infos(self):
        return protobuf._proto_infos(self)

    def _try_fast_delete_nodes(self, indices):
        return protobuf._try_fast_delete_nodes(self, indices)

    def _proto_links(self):
        return protobuf._proto_links(self)

    def _proto_dead(self):
        return protobuf._proto_dead(self)

    def _proto_unused(self, kind):
        return protobuf._proto_unused(self, kind)

    def _proto_region_graph(self):
        return protobuf._proto_region_graph(self)
