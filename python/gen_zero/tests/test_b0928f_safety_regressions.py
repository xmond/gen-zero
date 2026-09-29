from dataclasses import replace

import struct
import zlib

import numpy as np
import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType
from gen_zero.provenance.auditor import DecisionProvenanceAuditor
from gen_zero.gate.differentiable_safety_layer import DifferentiableSafetyLayer
from gen_zero.nanocore.spdk_nvme_fleet import SpdkNanoCoreSerializer, HEADER_SIZE
from gen_zero.client import GenZeroClient


def test_callable_without_decide_rejected():
    client = object.__new__(GenZeroClient)
    client.registered_nanocores = {}
    with pytest.raises(TypeError, match='callable decide'):
        client.register_nanocore(1, core=lambda state, candidates: None)


def test_ndarray_middle_mutation_and_timestamp_detected():
    auditor = DecisionProvenanceAuditor('explicit-test-secret')
    context = {'values': np.arange(3000, dtype=np.float32)}
    record = auditor.generate_provenance([], {}, context, [])
    assert auditor.verify_provenance(record, input_context=context)[0]
    context['values'][1500] = -99
    assert not auditor.verify_provenance(record, input_context=context)[0]
    assert not auditor.verify_provenance(replace(record, timestamp=record.timestamp + 1))[0]
    with pytest.raises(TypeError):
        DecisionProvenanceAuditor()


def test_descriptor_nested_aliases_and_export_are_isolated():
    schema = {'properties': {'a': {'enum': ['x']}}}
    descriptor = CapabilityDescriptor('tool:x', CapabilityType.TOOL, 'x', input_schema=schema)
    before = descriptor.digest()
    schema['properties']['a']['enum'].append('y')
    with pytest.raises(TypeError):
        descriptor.input_schema['properties']['a']['enum'] = ('y',)
    exported = descriptor.to_dict()
    exported['input_schema']['properties']['a']['enum'].append('z')
    assert descriptor.digest() == before
    assert descriptor.to_dict()['input_schema']['properties']['a']['enum'] == ['x']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_proposal_rejected(bad):
    layer = DifferentiableSafetyLayer(2)
    with pytest.raises(ValueError, match='Non-finite'):
        layer.forward(np.array([bad, 0.5]))


def test_spdk_crc_and_sha_fail_closed():
    core = {'weights': {'x': np.arange(8, dtype=np.float32)}}
    raw, _ = SpdkNanoCoreSerializer.serialize_core_to_chunks('check', core)
    corrupted = bytearray(raw)
    corrupted[HEADER_SIZE + 10] ^= 1
    with pytest.raises(ValueError, match='CRC32'):
        SpdkNanoCoreSerializer.deserialize_chunks_to_core(corrupted)

    # Preserve chunk CRC while altering serialized weight bytes: SHA256 must catch it.
    corrupted_sha = bytearray(raw)
    payload_size = struct.unpack_from("!I", corrupted_sha, 17)[0]
    weight_offset = HEADER_SIZE + payload_size - 1
    corrupted_sha[weight_offset] ^= 1
    crc = zlib.crc32(corrupted_sha[HEADER_SIZE:HEADER_SIZE + payload_size]) & 0xFFFFFFFF
    struct.pack_into("!I", corrupted_sha, 21, crc)
    with pytest.raises(ValueError, match='SHA256'):
        SpdkNanoCoreSerializer.deserialize_chunks_to_core(corrupted_sha)
