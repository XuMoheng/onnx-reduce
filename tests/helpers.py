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
