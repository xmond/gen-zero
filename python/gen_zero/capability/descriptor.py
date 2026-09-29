"""Gen-Zero Capability Layer: Unified Heterogeneous Capability Descriptor.

RFC-079 Implementation:
Defines standard descriptors for Tool, Skill, Subagent, and CLI capabilities,
unifying disparate dispatch targets under a single type-safe mathematical manifold.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np


class CapabilityType(str, Enum):
    """Classification of heterogeneous capabilities in Gen-Zero."""
    TOOL = "tool"          # Single-step external tool (e.g. MCP tool)
    SKILL = "skill"        # Progressive multi-step expert workflow / micro-agent (RFC-027)
    SUBAGENT = "subagent"  # Long-horizon privileged autonomous agent (RFC-013, RFC-023)
    CLI = "cli"            # Host system native CLI command


class RiskLevel(str, Enum):
    """Safety and permission isolation tiers."""
    READ_ONLY = "read_only"      # Pure side-effect-free observation / queries
    WRITE_SAFE = "write_safe"    # Idempotent writes, workspace sandboxed edits
    NETWORK = "network"          # Outbound network requests
    DESTRUCTIVE = "destructive"  # High-risk destructive mutations (rm, kill, drop table)

    @property
    def level_rank(self) -> int:
        return _RISK_RANKS.get(self.value, 99)

    def is_within(self, max_allowed: "RiskLevel") -> bool:
        """Returns True if this risk level does not exceed max_allowed."""
        return _RISK_RANKS.get(self.value, 99) <= _RISK_RANKS.get(max_allowed.value, 99)


_RISK_RANKS = {
    "read_only": 0,
    "write_safe": 1,
    "network": 2,
    "destructive": 3,
}


def _canon_str(value: str) -> str:
    """Exact builtin-str copy of any str subclass (np.str_, user subclasses).

    str.__str__ bypasses a subclass's overridden __str__, so the content is
    preserved verbatim; JSON round-trips only ever yield builtin str.
    """
    return str.__str__(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, str):
        return _canon_str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object dtype metadata arrays are unsupported")
        return _freeze(value.tolist())
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Mapping):
        frozen = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Mapping keys must be strings, got {type(k).__name__}: {k!r}"
                )
            # str subclasses (e.g. np.str_) must normalize to the builtin str
            # so the frozen key domain matches what a JSON round-trip yields;
            # otherwise digest() diverges between np.str_("x") and "x" keys.
            norm_k = _canon_str(k)
            if norm_k in frozen:
                raise ValueError(
                    f"Mapping key collision after str-subclass normalization: {norm_k!r}"
                )
            frozen[norm_k] = _freeze(v)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


_THAW_MARKERS = ("__bytes_hex__", "__set__", "__map__")


def _thaw(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, (set, frozenset)):
        elements = [_thaw(v) for v in value]
        elements.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return {"__set__": elements}
    if isinstance(value, Mapping):
        thawed = {k: _thaw(v) for k, v in value.items()}
        if len(thawed) == 1 and next(iter(thawed)) in _THAW_MARKERS:
            # A genuine user mapping happens to collide with our marker shape
            # (e.g. metadata={"raw": {"__bytes_hex__": "61"}}). Escape it so
            # _restore_thawed can tell a real marker from user data.
            return {"__map__": [[k, v] for k, v in thawed.items()]}
        return thawed
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _restore_thawed(value: Any) -> Any:
    """Inverse of _thaw: recovers bytes/set/escaped-mapping markers lost by JSON-safe thawing."""
    if isinstance(value, Mapping):
        if len(value) == 1 and "__bytes_hex__" in value:
            return bytes.fromhex(value["__bytes_hex__"])
        if len(value) == 1 and "__set__" in value:
            return frozenset(_restore_thawed(v) for v in value["__set__"])
        if len(value) == 1 and "__map__" in value:
            return {k: _restore_thawed(v) for k, v in value["__map__"]}
        return {k: _restore_thawed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        # Tuple, not list: elements may feed a frozenset() call above us
        # (a set of tuples), which requires hashable members.
        return tuple(_restore_thawed(v) for v in value)
    return value


def _typed(value: Any) -> Any:
    """Unambiguous digest encoding, including user supplied marker-shaped mappings."""
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Mapping):
        items = sorted(
            ((_typed(k), _typed(v)) for k, v in value.items()),
            key=lambda kv: json.dumps(kv[0], sort_keys=True),
        )
        return ["map", [[k, v] for k, v in items]]
    if isinstance(value, (set, frozenset)):
        elements = sorted(
            (_typed(v) for v in value),
            key=lambda x: json.dumps(x, sort_keys=True),
        )
        return ["set", elements]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_typed(v) for v in value]]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    # Tag by canonical builtin type, never type(value).__name__: a str/int/float
    # subclass (np.str_, IntEnum, ...) would otherwise get a tag that a JSON
    # round-trip cannot reproduce, and digest() would drift.
    if isinstance(value, str):
        return ["str", _canon_str(value)]
    if isinstance(value, int):
        return ["int", int(value)]
    if isinstance(value, float):
        return ["float", float(value)]
    raise TypeError(f"Unsupported descriptor value: {type(value)!r}")


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Strongly typed contract for an orchestratable capability."""
    capability_id: str                              # Globally unique namespace ID, e.g. "tool:mcp_grep", "skill:refactor"
    capability_type: CapabilityType                 # Heterogeneous classification
    description: str                                # Semantic alignment description for latent space Choice routing
    input_schema: Dict[str, Any] = field(default_factory=dict)   # JSON schema for parameter validation
    output_schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema for return contracts
    domain: str = "general"                         # Primary functional domain: system, dev, web, verify, data
    cost_weight: float = 1.0                        # Computational / latency cost multiplier for beam planning
    risk_level: RiskLevel = RiskLevel.READ_ONLY     # Security risk tier
    required_permissions: Sequence[str] = field(default_factory=tuple) # Required OS/Security ACLs (immutable)
    is_idempotent: bool = True                      # Whether invoking multiple times produces identical outcomes
    timeout_seconds: float = 30.0                   # Execution deadline
    version: str = "1.0.0"                          # Semantic capability version
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every string field is pinned to the builtin str domain that a JSON
        # round-trip yields, so == and digest() agree before and after.
        for name in ("capability_id", "description", "domain", "version"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, _canon_str(value) if isinstance(value, str) else str(value)
            )
        for name in ("input_schema", "output_schema", "metadata"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(
            self,
            "required_permissions",
            tuple(
                _canon_str(p) if isinstance(p, str) else str(p)
                for p in self.required_permissions
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes descriptor into dictionary representation."""
        return {
            "capability_id": self.capability_id,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "input_schema": _thaw(self.input_schema),
            "output_schema": _thaw(self.output_schema),
            "domain": self.domain,
            "cost_weight": float(self.cost_weight),
            "risk_level": self.risk_level.value,
            "required_permissions": list(self.required_permissions),
            "is_idempotent": bool(self.is_idempotent),
            "timeout_seconds": float(self.timeout_seconds),
            "version": str(self.version),
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityDescriptor":
        """Reconstructs descriptor from dictionary representation."""
        return cls(
            capability_id=str(data["capability_id"]),
            capability_type=CapabilityType(data["capability_type"]),
            description=str(data.get("description", "")),
            input_schema=_restore_thawed(dict(data.get("input_schema", {}))),
            output_schema=_restore_thawed(dict(data.get("output_schema", {}))),
            domain=str(data.get("domain", "general")),
            cost_weight=float(data.get("cost_weight", 1.0)),
            risk_level=RiskLevel(data.get("risk_level", "read_only")),
            required_permissions=list(data.get("required_permissions", [])),
            is_idempotent=bool(data.get("is_idempotent", True)),
            timeout_seconds=float(data.get("timeout_seconds", 30.0)),
            version=str(data.get("version", "1.0.0")),
            metadata=_restore_thawed(dict(data.get("metadata", {}))),
        )

    def digest(self) -> str:
        """Computes deterministic SHA-256 fingerprint of the descriptor contract."""
        payload = json.dumps(_typed({**self.to_dict(), "input_schema": self.input_schema,
                                     "output_schema": self.output_schema, "metadata": self.metadata}), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
