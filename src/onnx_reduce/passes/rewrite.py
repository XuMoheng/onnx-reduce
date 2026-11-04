"""ONNX-specific pass rules over optional model query views."""

from onnx_reduce.passes.candidates import rewrite_candidates


class BypassNodesPass:
    name = "bypass_nodes"

    def generate(self, model):
        yield from rewrite_candidates(model, "bypass")


class SimplifyAttributesPass:
    name = "simplify_attributes"

    def generate(self, model):
        yield from rewrite_candidates(model, "attributes")


class ShrinkTensorsPass:
    name = "shrink_tensors"

    def generate(self, model):
        yield from rewrite_candidates(model, "tensors")


class ReduceSubgraphsPass:
    name = "reduce_subgraphs"

    def generate(self, model):
        yield from rewrite_candidates(model, "subgraphs")
