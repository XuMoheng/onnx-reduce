"""Bounded dependency fragments with same-type bypasses checked as one edit path."""

from bisect import bisect_left
from itertools import chain, islice

import onnx

from onnx_reduce.edits import BypassNode, Candidate, ExtractNodes
from onnx_reduce.passes.combinations import spread_indices


def _tensor_type(info):
    if not info.type.HasField("tensor_type"):
        return None
    tensor = info.type.tensor_type
    if not tensor.elem_type or not tensor.HasField("shape"):
        return None
    if any(not dim.HasField("dim_value") for dim in tensor.shape.dim):
        return None
    return tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim)


class CompactFragmentsPass:
    """Keep shape-changing chains while bypassing compatible branch plumbing.

    This is a reducer heuristic, not an equivalence rewrite. A whole path is
    validated and failure-checked atomically by the normal candidate executor.
    Unresolved types and control-flow graphs are left to the existing passes.
    """

    name = "compact_fragments"
    limit = 8
    max_bypasses = 8
    max_visits = 4096
    max_inference_bytes = 16 * 1024 * 1024

    def generate(self, model):
        if self.limit <= 0 or self.max_visits <= 0 or not hasattr(model, "candidate_snapshot"):
            return
        proto = model.candidate_snapshot()
        graph = proto.graph
        if any(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in graph.node
            for attr in node.attribute
        ):
            return
        nodes = graph.node
        if len(nodes) < 2 or not any(len(node.input) > 1 or node.attribute for node in nodes):
            return
        types = {
            info.name: kind
            for info in chain(graph.input, graph.value_info, graph.output)
            if (kind := _tensor_type(info)) is not None
        }
        functions = {(function.domain, function.name) for function in proto.functions}
        constants = {tensor.name for tensor in graph.initializer}
        for tensor in graph.initializer:
            types[tensor.name] = tensor.data_type, tuple(tensor.dims)
        for tensor in graph.sparse_initializer:
            constants.add(tensor.values.name)
            types[tensor.values.name] = tensor.values.data_type, tuple(tensor.dims)
        # Only the dense Constant representation understood by ExtractNodes
        # can be omitted: extraction retains its exact payload as an initializer.
        for node in nodes:
            if (
                node.domain not in {"", "ai.onnx"}
                or node.op_type != "Constant"
                or (node.domain, node.op_type) in functions
            ):
                continue
            if len(node.output) != 1 or not node.output[0]:
                continue
            tensor = next(
                (
                    a.t
                    for a in node.attribute
                    if a.name == "value" and a.type == onnx.AttributeProto.TENSOR
                ),
                None,
            )
            if tensor is not None:
                constants.add(node.output[0])
                types[node.output[0]] = tensor.data_type, tuple(tensor.dims)
        missing = [name for node in nodes for name in node.output if name and name not in types]
        if (
            missing
            and hasattr(model, "infer_value_info")
            and len(nodes) <= self.max_visits
            and proto.ByteSize() <= self.max_inference_bytes
        ):
            types.update(
                (name, kind)
                for name, info in model.infer_value_info(missing).items()
                if (kind := _tensor_type(info)) is not None
            )
        producers = {name: i for i, node in enumerate(nodes) for name in node.output if name}
        outputs = {value.name for value in graph.output}
        bypasses = {}
        reductions, boundaries = [], []
        for index, node in enumerate(nodes):
            if (
                node.domain not in {"", "ai.onnx"}
                or (node.domain, node.op_type) in functions
                or len(node.output) != 1
                or (kind := types.get(node.output[0])) is None
                or node.output[0] in constants
            ):
                continue
            compatible = [
                i
                for i, name in enumerate(node.input)
                if name
                and name != node.output[0]
                and types.get(name) == kind
                and not (node.output[0] in outputs and name in outputs)
            ]
            if compatible:
                bypasses[index] = min(compatible, key=lambda i: (node.input[i] in constants, i))
                continue
            ranks = [len(types[name][1]) for name in node.input if name in types]
            if ranks:
                (reductions if len(kind[1]) < max(ranks) else boundaries).append(index)

        # Diverse anchors, without letting a long suffix hide an interior chain.
        anchors = chain(
            (reductions[i] for i in spread_indices(len(reductions))),
            (boundaries[i] for i in spread_indices(len(boundaries))),
        )
        remaining = self.max_visits
        for anchor in islice(anchors, self.limit):
            selected, removed = set(), set()
            pending, seen = [nodes[anchor].output[0]], set()
            while pending and remaining:
                name = pending.pop()
                remaining -= 1
                index = producers.get(name)
                if index is None or name in constants or index in seen:
                    continue
                seen.add(index)
                if index != anchor and index in bypasses:
                    removed.add(index)
                    if len(removed) > self.max_bypasses:
                        break
                    pending.append(nodes[index].input[bypasses[index]])
                else:
                    selected.add(index)
                    pending.extend(name for name in nodes[index].input if name)
            if not remaining:
                return
            if pending or not removed or len(removed) > self.max_bypasses or len(selected) < 2:
                continue
            removed = sorted(removed)
            # Descending bypasses still address original indices. Extraction
            # addresses the graph after all those nodes have been removed.
            edits = tuple(BypassNode(i, bypasses[i]) for i in reversed(removed))
            retained = tuple(i - bisect_left(removed, i) for i in sorted(selected))
            yield Candidate((*edits, ExtractNodes(retained, outputs=(nodes[anchor].output[0],))))

    def generate_stage(self, model, stage):
        if stage == "coarse":
            yield from self.generate(model)
