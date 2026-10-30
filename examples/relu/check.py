"""Teaching failure check: interesting iff the main graph contains Relu."""

import sys

import onnx

try:
    model = onnx.load(sys.argv[-1])
except Exception as exc:
    print(f"cannot load candidate: {exc}", file=sys.stderr)
    raise SystemExit(125) from exc

has_relu = any(node.op_type == "Relu" for node in model.graph.node)
if has_relu:
    print("assertion: DEMO_RELU")
raise SystemExit(0 if has_relu else 1)
