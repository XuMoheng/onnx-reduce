"""Try complete live fragments, rather than deleting nodes behind fixed outputs."""

from collections import defaultdict, deque
from itertools import islice, zip_longest

from onnx_reduce.edits import Candidate, ExtractNodes
from onnx_reduce.passes.delete_nodes import _granularities
from onnx_reduce.passes.groups import is_coarse


class ExtractNodesPass:
    name = "extract_nodes"

    @staticmethod
    def _groups(nodes):
        count = len(nodes)
        families = defaultdict(list)
        for index, node in enumerate(nodes):
            families[(node.domain, node.op_type)].append(index)
        # Rare families offer cheap witnesses. Interleave partitions so many
        # distinct operator types cannot starve bisection on a large graph.
        # Among equally rare families, shape consumers can expose a useful
        # constant-boundary witness before a coarse partition is accepted and
        # resets this stream. This also fits within the schedulers' pass turns.
        representatives = sorted(
            families.values(),
            key=lambda items: (
                len(items),
                not (
                    nodes[items[0]].domain == ""
                    and nodes[items[0]].op_type in {"Expand", "Reshape"}
                ),
                items[0],
            ),
        )
        groups = ((items[0],) for items in representatives)
        partitions = (
            tuple(range(start, min(start + size, count)))
            for size in _granularities(count)
            for start in range(0, count, size)
        )
        seen = set()
        for pair in zip_longest(groups, partitions):
            for group in pair:
                if group and group not in seen:
                    seen.add(group)
                    priority = (
                        len(group) == 1
                        and len(families[(nodes[group[0]].domain, nodes[group[0]].op_type)]) == 1
                    )
                    yield group, priority

    def generate(self, model):
        if not hasattr(model, "candidate_snapshot"):
            return
        graph = model.candidate_snapshot().graph
        producers = {name: i for i, node in enumerate(graph.node) for name in node.output if name}
        weights = {tensor.name for tensor in graph.initializer}
        pending = deque()
        regions = 0
        half = max(1, len(graph.node) // 2)
        region_quantum = (len(graph.node) + half - 1) // half
        for group, priority in self._groups(graph.node):
            selected = set(group)
            materialize = any(
                node.domain == ""
                and node.op_type in {"Expand", "Reshape"}
                and len(node.input) > 1
                and node.input[1] not in weights
                and node.input[1] in producers
                and producers[node.input[1]] not in selected
                and graph.node[producers[node.input[1]]].op_type != "Constant"
                for node in (graph.node[i] for i in group)
            )
            if materialize:
                yield Candidate((ExtractNodes(group, materialize_shapes=True),))
            yield Candidate((ExtractNodes(group),))
            variants = [None]
            if len(group) == 1:
                names = tuple(name for name in graph.node[group[0]].output if name)
                # A single named output is already represented by the default
                # projection; multi-output nodes need per-output witnesses.
                variants = [None] + ([(name,) for name in names] if len(names) > 1 else [])
                for outputs in variants[1:]:
                    if materialize:
                        yield Candidate(
                            (ExtractNodes(group, outputs=outputs, materialize_shapes=True),)
                        )
                    yield Candidate((ExtractNodes(group, outputs=outputs),))
            else:
                regions += 1
            for outputs in variants:
                if len(group) > 1 or (len(outputs) if outputs is not None else len(names)) == len(
                    graph.output
                ):
                    edit = ExtractNodes(group, outputs=outputs, preserve_interface=True)
                    if priority:
                        yield Candidate((edit,))
                    else:
                        pending.append(edit)
            # Include the odd-sized tail of the initial partition before paying
            # for deferred aliases. Bound lookahead in long singleton streams.
            if regions >= region_quantum or len(pending) >= 8:
                while pending:
                    yield Candidate((pending.popleft(),))
                regions = 0
        while pending:
            yield Candidate((pending.popleft(),))

    def generate_stage(self, model, stage):
        if stage == "cleanup":
            return
        for candidate in self.generate(model):
            size = len(candidate.edits[0].indices)
            # Direct witnesses get an early turn, even on large models.
            if (stage == "coarse") == (size == 1 or is_coarse(size, model.node_count)):
                yield candidate


class ProbeFragmentsPass:
    """A bounded early opportunity; full enumeration remains a later pass."""

    name = "probe_fragments"
    limit = 16

    def generate(self, model):
        yield from islice(ExtractNodesPass().generate(model), self.limit)

    def generate_stage(self, model, stage):
        if stage == "coarse":
            yield from self.generate(model)
