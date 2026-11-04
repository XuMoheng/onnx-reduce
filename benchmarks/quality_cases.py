"""Deterministic architectural fixtures, with simulated failure conditions."""

import numpy as np
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper

from onnx_reduce.failure_check.contract import FailureCheckResult
from onnx_reduce.model.operations.graphs import walk_graphs


def value(name, shape=(4, 4), dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def model(nodes, inputs, outputs, weights=()):
    return h.make_model(
        h.make_graph(nodes, "quality", inputs, outputs, list(weights)),
        opset_imports=[h.make_opsetid("", 18)],
    )


def quality_cases():
    weight = numpy_helper.from_array(np.arange(16, dtype=np.float32).reshape(4, 4), "w")
    yield (
        "matmul",
        model([h.make_node("MatMul", ["x", "w"], ["y"])], [value("x")], [value("y")], [weight]),
        {"MatMul"},
    )
    shape = numpy_helper.from_array(np.array([4, 4], dtype=np.int64), "shape")
    yield (
        "reshape",
        model([h.make_node("Reshape", ["x", "shape"], ["y"])], [value("x")], [value("y")], [shape]),
        {"Reshape"},
    )
    branch = h.make_graph(
        [h.make_node("Neg", ["x"], ["mid"]), h.make_node("Relu", ["mid"], ["out"])],
        "branch",
        [],
        [value("out")],
    )
    yield (
        "if",
        model(
            [h.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)],
            [value("x"), value("cond", (), T.BOOL)],
            [value("y")],
        ),
        {"If", "Relu"},
    )
    body = h.make_graph(
        [
            h.make_node("Identity", ["cond_in"], ["cond_out"]),
            h.make_node("Neg", ["state"], ["mid"]),
            h.make_node("Relu", ["mid"], ["state_out"]),
        ],
        "body",
        [value("iter", (), T.INT64), value("cond_in", (), T.BOOL), value("state")],
        [value("cond_out", (), T.BOOL), value("state_out")],
    )
    yield (
        "loop",
        model(
            [h.make_node("Loop", ["trip", "cond", "x"], ["y"], body=body)],
            [value("trip", (), T.INT64), value("cond", (), T.BOOL), value("x")],
            [value("y")],
        ),
        {"Loop", "Relu"},
    )
    body = h.make_graph(
        [
            h.make_node("Identity", ["state"], ["state_out"]),
            h.make_node("Neg", ["item"], ["mid"]),
            h.make_node("Relu", ["mid"], ["out"]),
        ],
        "body",
        [value("state"), value("item")],
        [value("state_out"), value("out")],
    )
    yield (
        "scan",
        model(
            [
                h.make_node(
                    "Scan", ["initial", "items"], ["last", "y"], body=body, num_scan_inputs=1
                )
            ],
            [value("initial"), value("items", (3, 4, 4))],
            [value("last"), value("y", (3, 4, 4))],
        ),
        {"Scan", "Relu"},
    )
    yield (
        "paired_bypass",
        model(
            [
                h.make_node("Neg", ["x"], ["a"]),
                h.make_node("Relu", ["a"], ["b"]),
                h.make_node("Neg", ["b"], ["c"]),
                h.make_node("Sigmoid", ["c"], ["y"]),
            ],
            [value("x")],
            [value("y")],
        ),
        {"Relu", "Sigmoid"},
    )


class SimulatedFailure:
    def __init__(self, required_ops, *, paired=False):
        self.required_ops = required_ops
        self.paired = paired

    def run(self, path):
        import onnx

        proto = onnx.load(path)
        nodes = [n for _, graph in walk_graphs(proto) for n in graph.node]
        ops = [n.op_type for n in nodes]
        interesting = self.required_ops <= set(ops)
        if self.paired:
            interesting &= ops.count("Neg") % 2 == 0 and [v.name for v in proto.graph.input] == [
                "x"
            ]
        return FailureCheckResult.INTERESTING if interesting else FailureCheckResult.NOT_INTERESTING
