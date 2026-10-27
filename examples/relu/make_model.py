"""Create a four-node teaching model; no real backend bug is implied."""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    destination = Path(sys.argv[1])
    nodes = [
        helper.make_node("Identity", ["x"], ["v0"]),
        helper.make_node("Relu", ["v0"], ["v1"]),
        helper.make_node("Identity", ["v1"], ["v2"]),
        helper.make_node("Identity", ["v2"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "learning_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    main()
