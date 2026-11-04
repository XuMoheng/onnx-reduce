"""ONNX-specific rewrites; generic model adapters may decline this capability."""


class BypassNodesPass:
    name = "bypass_nodes"

    def generate(self, model):
        if hasattr(model, "rewrite_candidates"):
            yield from model.rewrite_candidates("bypass")


class SimplifyAttributesPass:
    name = "simplify_attributes"

    def generate(self, model):
        if hasattr(model, "rewrite_candidates"):
            yield from model.rewrite_candidates("attributes")


class ShrinkTensorsPass:
    name = "shrink_tensors"

    def generate(self, model):
        if hasattr(model, "rewrite_candidates"):
            yield from model.rewrite_candidates("tensors")


class ReduceSubgraphsPass:
    name = "reduce_subgraphs"

    def generate(self, model):
        if hasattr(model, "rewrite_candidates"):
            yield from model.rewrite_candidates("subgraphs")
