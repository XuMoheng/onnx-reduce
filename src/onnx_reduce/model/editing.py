"""Apply ordered edits to owned clones; preserve node identities and invalidate snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from onnxscript import ir

from onnx_reduce.edits import (
    BypassNode,
    Candidate,
    CleanupGraph,
    DeleteNodes,
    DeleteSubgraphNodes,
    DropInitializers,
    DropInputs,
    DropNodeAttribute,
    DropOutputs,
    ResizeOperator,
    ShrinkDimension,
)
from onnx_reduce.performance import count, measured
from onnx_reduce.types import (
    EditApplicationError,
)

if TYPE_CHECKING:
    from onnx_reduce.model.adapter import OnnxModel

from onnx_reduce.model.snapshot import invalidate_derived


@measured("clone_and_edit")
def apply_candidate(model: OnnxModel, candidate: Candidate) -> OnnxModel:
    """Verify the base and apply edits to a clone; the engine decides acceptance."""

    from onnx_reduce.model.adapter import OnnxModel

    if candidate.base_digest is not None and candidate.base_digest != model.canonical_hash():
        raise EditApplicationError("candidate base digest does not match the model")
    if (
        model.edit_options.mode == "auto"
        and len(candidate.edits) == 1
        and isinstance(candidate.edits[0], DeleteNodes)
    ):
        fast = model._try_fast_delete_nodes(candidate.edits[0].indices)
        if fast is not None:
            return fast
    result = model.copy()
    # Each edit addresses the result of the preceding edit.
    for edit_index, edit in enumerate(candidate.edits):
        if isinstance(edit, DeleteNodes):
            result._delete_nodes(
                edit.indices, inference_source=model if edit_index == 0 else result
            )
        elif isinstance(edit, DropOutputs):
            result._drop_outputs(edit.names)
        elif isinstance(edit, DropInputs):
            result._drop_inputs(edit.names)
        elif isinstance(edit, DropInitializers):
            result._drop_initializers(edit.names)
        elif isinstance(
            edit,
            (
                BypassNode,
                DropNodeAttribute,
                ShrinkDimension,
                ResizeOperator,
                DeleteSubgraphNodes,
                CleanupGraph,
            ),
        ):
            from onnx_reduce.model.operations.rewrites import apply_rewrite

            proto = result.to_proto()
            if isinstance(edit, ResizeOperator):
                from onnx_reduce.model.operations.local_tensors import resize_operator

                resize_operator(proto, edit)
            elif isinstance(edit, ShrinkDimension):
                from onnx_reduce.model.operations.tensors import shrink_dimension

                shrink_dimension(proto, edit.old, edit.new)
            elif isinstance(edit, (DeleteSubgraphNodes, CleanupGraph)):
                from onnx_reduce.model.operations.graphs import apply_graph_edit

                apply_graph_edit(proto, edit)
            else:
                apply_rewrite(proto, edit)
            ids = result.node_ids
            if isinstance(edit, BypassNode) and not edit.graph_path:
                ids = ids[: edit.index] + ids[edit.index + 1 :]
            try:
                result = OnnxModel(
                    ir.from_proto(proto), _node_ids=ids, edit_options=model.edit_options
                )
            except (ir.serde.SerdeError, ValueError, TypeError) as exc:
                raise EditApplicationError(f"cannot import rewritten graph: {exc}") from exc
        else:
            raise EditApplicationError(f"unsupported edit: {type(edit).__name__}")
        # An intermediate edit may have acquired a serialized snapshot for
        # inference. Subsequent edits must never see that stale snapshot.
        invalidate_derived(result)
    return result


def _delete_nodes(model: OnnxModel, indices: Iterable[int], *, inference_source: OnnxModel) -> None:
    """Delete main-graph nodes and promote still-used boundary outputs to graph inputs."""

    graph = model._model.graph
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
        for nested in model._nested_graphs(node):
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
    if model.edit_options.mode == "auto" and model._graph_ordered(graph):
        count("sort_skips")
    else:
        graph.sort()
    model._refresh_node_indices()


def _graph_ordered(model_type, graph):
    available = set(graph.inputs) | set(graph.initializers.values())
    for node in graph:
        if any(True for _ in model_type._nested_graphs(node)):
            return False
        if any(value not in available for value in node.inputs if value is not None):
            return False
        available.update(node.outputs)
    return True


def _drop_outputs(model: OnnxModel, names: Iterable[str]) -> None:
    """Drop named graph outputs, keeping at least one and preserving producer nodes."""

    graph = model._model.graph
    selected = set(names)
    outputs = [value for value in graph.outputs if value.name not in selected]
    if not outputs:
        raise EditApplicationError("cannot drop every graph output")
    missing = selected.difference(value.name for value in graph.outputs)
    if missing:
        raise EditApplicationError(f"unknown graph output(s): {', '.join(sorted(missing))}")
    graph.outputs.clear()
    graph.outputs.extend(outputs)


def _drop_inputs(model: OnnxModel, names: Iterable[str]) -> None:
    """Drop named graph inputs only if they are unused."""

    selected = set(names)
    if not selected <= set(model.unused_input_names()):
        raise EditApplicationError("can only drop unused graph inputs")
    graph = model._model.graph
    remaining = [value for value in graph.inputs if value.name not in selected]
    graph.inputs.clear()
    graph.inputs.extend(remaining)


def _drop_initializers(model: OnnxModel, names: Iterable[str]) -> None:
    """Drop named initializers only if they are unused."""

    selected = set(names)
    if not selected <= set(model.unused_initializer_names()):
        raise EditApplicationError("can only drop unused initializers")
    for name in selected:
        del model._model.graph.initializers[name]
