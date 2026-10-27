"""Spike 1/2: freeze the IR and validity semantics with executable evidence."""

from __future__ import annotations

import onnx
from onnxscript import ir

from onnx_reduce.model import OnnxModel
from tests.helpers import control_flow_model, custom_domain_model, external_data_model, qdq_model


def roundtrip(proto: onnx.ModelProto, tmp_path, name: str):
    path = tmp_path / name
    onnx.checker.check_model(proto)
    onnx.save_model(proto, path)
    model = ir.load(path)
    converted = ir.to_proto(model)
    onnx.checker.check_model(converted, full_check=True)
    return OnnxModel(model), converted


def test_control_flow_roundtrip_preserves_graph_and_subgraph(tmp_path):
    original = control_flow_model()
    _, converted = roundtrip(original, tmp_path, "control-flow.onnx")
    assert converted.graph.node[0].op_type == "If"
    attribute_names = {attribute.name for attribute in converted.graph.node[0].attribute}
    assert {"then_branch", "else_branch"} <= attribute_names


def test_qdq_roundtrip_preserves_operator_and_initializers(tmp_path):
    _, converted = roundtrip(qdq_model(), tmp_path, "qdq.onnx")
    assert [node.op_type for node in converted.graph.node] == [
        "QuantizeLinear",
        "DequantizeLinear",
    ]
    assert {initializer.name for initializer in converted.graph.initializer} == {
        "scale",
        "zero_point",
    }


def test_external_data_roundtrip_and_hash_are_stable(tmp_path):
    model_path = tmp_path / "external.onnx"
    onnx.save_model(
        external_data_model(),
        model_path,
        save_as_external_data=True,
        location="external.onnx.data",
        size_threshold=0,
        all_tensors_to_one_file=True,
    )
    loaded = onnx.load_model(model_path, load_external_data=True)
    assert len(loaded.graph.initializer[0].raw_data) == 2048 * 4

    model = OnnxModel.load(model_path)
    converted = model.to_proto()
    roundtrip_path = tmp_path / "external-roundtrip.onnx"
    onnx.save_model(converted, roundtrip_path)
    onnx.checker.check_model(roundtrip_path, full_check=True)
    assert model.canonical_hash() == model.copy().canonical_hash()


def test_custom_domain_checker_semantics():
    adapter = OnnxModel.from_proto(custom_domain_model())
    strict = adapter.validate("strict")
    structural = adapter.validate("structural")
    permissive = adapter.validate("permissive")
    assert strict.valid and strict.checker_mode == "structural"
    assert structural.valid and structural.checker_mode == "structural"
    assert permissive.valid and permissive.checker_mode == "permissive"
    assert strict.custom_domains == ("com.example",)
