"""Gen-Zero Provenance Layer: Cryptographic Decision Provenance and Immutable Auditing.

RFC-079 Implementation:
Generates deterministic, tamper-proof trace tokens binding:
1. Candidate capabilities snapshot hash (SHA-256 over canonical manifest)
2. Policy and risk threshold fingerprint
3. Input context digest
4. Selected execution sequence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Union, Any, Tuple
import hashlib
import hmac
import json
import time

import numpy as np

from gen_zero.capability.descriptor import CapabilityDescriptor


@dataclass(frozen=True)
class DecisionProvenanceRecord:
    """Immutable audit record providing cryptographic proof of a decision's provenance."""
    candidate_snapshot_hash: str
    policy_fingerprint: str
    input_context_digest: str
    selected_sequence: List[str]
    trace_token: str
    payload_manifest: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "policy_fingerprint": self.policy_fingerprint,
            "input_context_digest": self.input_context_digest,
            "selected_sequence": list(self.selected_sequence),
            "trace_token": self.trace_token,
            "payload_manifest": self.payload_manifest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionProvenanceRecord":
        return cls(
            candidate_snapshot_hash=str(data["candidate_snapshot_hash"]),
            policy_fingerprint=str(data["policy_fingerprint"]),
            input_context_digest=str(data["input_context_digest"]),
            selected_sequence=list(data.get("selected_sequence", [])),
            trace_token=str(data["trace_token"]),
            payload_manifest=str(data["payload_manifest"]),
            timestamp=float(data.get("timestamp", time.time())),
        )


class DecisionProvenanceAuditor:
    """Manages cryptographic provenance tokens and tamper verification."""

    def __init__(self, audit_secret: str) -> None:
        if not isinstance(audit_secret, str) or not audit_secret:
            raise ValueError("audit_secret must be explicitly supplied and non-empty")
        self.audit_secret = audit_secret

    @staticmethod
    def _canonicalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object dtype arrays cannot be audited deterministically")
            shape = list(value.shape)
            arr = np.ascontiguousarray(value)
            return ["ndarray", arr.dtype.descr if arr.dtype.fields else arr.dtype.str,
                    shape, arr.tobytes().hex()]
        if isinstance(value, np.generic):
            return DecisionProvenanceAuditor._canonicalize(np.asarray(value))
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("Audit dictionary keys must be strings")
            return ["dict", [[k, DecisionProvenanceAuditor._canonicalize(v)]
                             for k, v in sorted(value.items())]]
        if isinstance(value, list):
            return ["list", [DecisionProvenanceAuditor._canonicalize(v) for v in value]]
        if isinstance(value, tuple):
            return ["tuple", [DecisionProvenanceAuditor._canonicalize(v) for v in value]]
        if value is None:
            return ["null"]
        if isinstance(value, bool):
            return ["bool", value]
        if isinstance(value, str):
            return ["str", value]
        if isinstance(value, int):
            return ["int", str(value)]
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("Non-finite audit value")
            return ["float", value.hex()]
        raise TypeError(f"Unsupported audit context type: {type(value)!r}")

    def compute_context_digest(self, input_context: Union[str, Dict[str, Any], Sequence[Any]]) -> str:
        """Computes deterministic SHA-256 digest of input prompt / context payload."""
        raw = json.dumps(self._canonicalize(input_context), sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def compute_candidate_snapshot_hash(self, active_descriptors: Sequence[CapabilityDescriptor]) -> str:
        """Computes canonical hash of participating candidate capabilities."""
        canonical = [
            {"id": c.capability_id, "digest": c.digest()}
            for c in sorted(active_descriptors, key=lambda x: x.capability_id)
        ]
        raw = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def compute_policy_fingerprint(self, policy_thresholds: Dict[str, Any]) -> str:
        """Computes canonical fingerprint of active safety gates and policy parameters."""
        raw = json.dumps(self._canonicalize(policy_thresholds), sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def generate_provenance(
        self,
        active_descriptors: Sequence[CapabilityDescriptor],
        policy_thresholds: Dict[str, Any],
        input_context: Union[str, Dict[str, Any], Sequence[Any]],
        selected_sequence: Sequence[str],
    ) -> DecisionProvenanceRecord:
        """Generates a tamper-proof decision provenance record using canonical HMAC-SHA256."""
        cand_hash = self.compute_candidate_snapshot_hash(active_descriptors)
        policy_fp = self.compute_policy_fingerprint(policy_thresholds)
        ctx_digest = self.compute_context_digest(input_context)
        seq_canonical = json.dumps(list(selected_sequence), separators=(",", ":"))

        # Assemble canonical raw trace payload with unambiguous structured sequence
        timestamp = time.time()
        raw_manifest = f"{cand_hash}|{policy_fp}|{ctx_digest}|{seq_canonical}|{timestamp.hex()}"
        secret_bytes = self.audit_secret.encode("utf-8")
        trace_token = hmac.new(secret_bytes, raw_manifest.encode("utf-8"), hashlib.sha256).hexdigest()

        return DecisionProvenanceRecord(
            candidate_snapshot_hash=cand_hash,
            policy_fingerprint=policy_fp,
            input_context_digest=ctx_digest,
            selected_sequence=list(selected_sequence),
            trace_token=trace_token,
            payload_manifest=raw_manifest,
            timestamp=timestamp,
        )

    def verify_provenance(
        self,
        record: DecisionProvenanceRecord,
        active_descriptors: Optional[Sequence[CapabilityDescriptor]] = None,
        policy_thresholds: Optional[Dict[str, Any]] = None,
        input_context: Optional[Union[str, Dict[str, Any], Sequence[Any]]] = None,
    ) -> Tuple[bool, str]:
        """Validates that a record has not been tampered with and matches claimed environment states."""
        # 1. Verify internal token cryptographic integrity using HMAC-SHA256
        seq_canonical = json.dumps(list(record.selected_sequence), separators=(",", ":"))
        expected_manifest = f"{record.candidate_snapshot_hash}|{record.policy_fingerprint}|{record.input_context_digest}|{seq_canonical}|{record.timestamp.hex()}"
        if record.payload_manifest != expected_manifest:
            return False, "Payload manifest inconsistency detected (tampered sequence or hashes)."

        secret_bytes = self.audit_secret.encode("utf-8")
        expected_token = hmac.new(secret_bytes, expected_manifest.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(record.trace_token, expected_token):
            return False, "Trace token signature verification failed (invalid audit secret or tampered token)."

        # 2. Check external capability snapshot if provided
        if active_descriptors is not None:
            actual_cand_hash = self.compute_candidate_snapshot_hash(active_descriptors)
            if actual_cand_hash != record.candidate_snapshot_hash:
                return False, f"Capability snapshot hash mismatch: expected {actual_cand_hash}, recorded {record.candidate_snapshot_hash}."

        # 3. Check policy thresholds if provided
        if policy_thresholds is not None:
            actual_policy_fp = self.compute_policy_fingerprint(policy_thresholds)
            if actual_policy_fp != record.policy_fingerprint:
                return False, f"Policy fingerprint mismatch: expected {actual_policy_fp}, recorded {record.policy_fingerprint}."

        # 4. Check context digest if provided
        if input_context is not None:
            actual_ctx_digest = self.compute_context_digest(input_context)
            if actual_ctx_digest != record.input_context_digest:
                return False, f"Input context digest mismatch: expected {actual_ctx_digest}, recorded {record.input_context_digest}."

        return True, "Provenance verified successfully: 100% cryptographic integrity guaranteed."
