"""Derived model state must never survive an intermediate edit."""

from functools import cached_property

from onnx_reduce.model import OnnxModel
from onnx_reduce.model.snapshot import invalidate_derived
from tests.helpers import chain_model


def test_new_derived_properties_participate_in_invalidation_automatically():
    class ExtendedModel(OnnxModel):
        @cached_property
        def extra_summary(self):
            return object()

    model = ExtendedModel.from_proto(chain_model())
    previous = model.extra_summary
    digest = model.canonical_hash()
    payload = model._payload
    model.region_graph()
    invalidate_derived(model)
    assert model.extra_summary is not previous
    assert model.canonical_hash() == digest
    assert model._payload == payload
    assert model.region_graph().node_ids == model.node_ids
