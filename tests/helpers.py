"""Shared synthetic ONNX fixtures."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper


def tensor_value(name: str, dtype: int = TensorProto.FLOAT, shape=(1,)):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def chain_model(node_count: int = 6, trigger_index: int = 2) -> onnx.ModelProto:
    if not 0 <= trigger_index < node_count:
        raise ValueError("trigger_index must be inside the chain")
    current = "x"
    nodes = []
    for index in range(node_count):
        output = "y" if index == node_count - 1 else f"v{index}"
        op_type = "Relu" if index == trigger_index else "Identity"
        nodes.append(helper.make_node(op_type, [current], [output], name=f"n{index}"))
        current = output
    graph = helper.make_graph(nodes, "chain", [tensor_value("x")], [tensor_value("y")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def dynamic_expand_model(prefix=0, tail=1):
    """Constant-derived Expand shape, with no intermediate value_info (opset 12)."""
    source_shape = [1, 6, 2, 2, 12, 32, 32]
    target_shape = [1, 6, 3, 2, 2, 12, 32, 32]
    nodes = []
    current = "corr"
    for i in range(prefix):
        nodes.append(helper.make_node("Identity", [current], [f"prefix{i}"]))
        current = f"prefix{i}"

    def constant(name, values):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, TensorProto.INT64, [len(values)], values),
        )

    nodes.extend(
        [
            helper.make_node("Unsqueeze", [current], ["expanded_input"], axes=[2]),
            constant("target", target_shape),
            constant("rank", [len(target_shape)]),
            helper.make_node(
                "ConstantOfShape",
                ["rank"],
                ["ones"],
                value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
            ),
            constant("minus_one", [-1]),
            helper.make_node("Mul", ["ones", "minus_one"], ["negative"]),
            helper.make_node("Equal", ["target", "negative"], ["condition"]),
            helper.make_node("Where", ["condition", "ones", "target"], ["expand_shape"]),
            helper.make_node("Expand", ["expanded_input", "expand_shape"], ["expanded"]),
        ]
    )
    current = "expanded"
    for i in range(tail):
        nodes.append(helper.make_node("Identity", [current], [f"tail{i}"]))
        current = f"tail{i}"
    return helper.make_model(
        helper.make_graph(
            nodes,
            "constant-derived-expand",
            [tensor_value("corr", shape=source_shape)],
            [tensor_value(current, shape=target_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 12)],
        ir_version=7,
    )


def qdq_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["q"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero_point"], ["y"]),
    ]
    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [0.05]),
        helper.make_tensor("zero_point", TensorProto.UINT8, [], [0]),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq",
        [tensor_value("x")],
        [tensor_value("y", TensorProto.FLOAT)],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def control_flow_model() -> onnx.ModelProto:
    then_constant = helper.make_tensor("then_value", TensorProto.FLOAT, [], [1.0])
    else_constant = helper.make_tensor("else_value", TensorProto.FLOAT, [], [0.0])
    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["then_out"], value=then_constant)],
        "then",
        [],
        [tensor_value("then_out", shape=[])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["else_out"], value=else_constant)],
        "else",
        [],
        [tensor_value("else_out", shape=[])],
    )
    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph(
        [if_node],
        "control_flow",
        [tensor_value("cond", TensorProto.BOOL)],
        [tensor_value("y", shape=[])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def custom_domain_model() -> onnx.ModelProto:
    node = helper.make_node("Special", ["x"], ["y"], domain="com.example")
    graph = helper.make_graph([node], "custom", [tensor_value("x")], [tensor_value("y")])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
    )


def external_data_model() -> onnx.ModelProto:
    weights = np.arange(2048, dtype=np.float32)
    initializer = onnx.numpy_helper.from_array(weights, name="weights")
    nodes = [helper.make_node("Identity", ["weights"], ["y"])]
    graph = helper.make_graph(
        nodes,
        "external",
        [],
        [tensor_value("y", shape=[2048])],
        initializer=[initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
