"""Gen-Zero Unified Capability Layer (RFC-079).

Exports:
- CapabilityType: Enum of heterogeneous capability classifications (TOOL, SKILL, SUBAGENT, CLI)
- RiskLevel: Enum of safety risk tiers (READ_ONLY, WRITE_SAFE, NETWORK, DESTRUCTIVE)
- CapabilityDescriptor: Strongly typed contract definition
- CapabilityRegistry: Thread-safe registry and Stage-1 coarse filtering engine
"""

from .descriptor import (
    CapabilityType,
    RiskLevel,
    CapabilityDescriptor,
)
from .registry import (
    CapabilityRegistry,
)

__all__ = [
    "CapabilityType",
    "RiskLevel",
    "CapabilityDescriptor",
    "CapabilityRegistry",
]
