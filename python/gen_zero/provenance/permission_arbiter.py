"""Gen-Zero Provenance Layer: Neuro-Symbolic Permission Arbiter and Privilege Gate.

RFC-079 Implementation:
Enforces Zero-Privilege model isolation:
1. Neural policy / LLM generates only candidate proposals without holding system credentials.
2. Permission Arbiter verifies capability permissions against granted ACLs and risk boundaries.
3. Violations trigger strict Fail-Closed veto and Google OR-Tools CP-SAT 0-1 ILP blocking.
"""

from typing import Set, Sequence, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from gen_zero.capability.descriptor import CapabilityDescriptor, RiskLevel
from gen_zero.gate.cpsat_formal_solver import CPSATVerdict


@dataclass
class PermissionVerdict:
    """Outcome of a permission evaluation across an action sequence."""
    is_authorized: bool
    authorized_capabilities: List[CapabilityDescriptor]
    blocked_capabilities: List[CapabilityDescriptor]
    violations: List[str]
    max_allowed_risk: RiskLevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_authorized": self.is_authorized,
            "authorized_count": len(self.authorized_capabilities),
            "blocked_count": len(self.blocked_capabilities),
            "violations": self.violations,
            "max_allowed_risk": self.max_allowed_risk.value,
        }


class PermissionArbiter:
    """Security arbitration gate enforcing host ACL boundaries and risk isolation."""

    def __init__(
        self,
        granted_permissions: Optional[Sequence[str]] = None,
        max_allowed_risk: RiskLevel = RiskLevel.WRITE_SAFE,
        revoked_capability_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if granted_permissions is not None:
            self.granted_permissions: Set[str] = set(granted_permissions)
        else:
            self.granted_permissions = {"fs:read", "fs:write", "process:exec"}
        self.max_allowed_risk: RiskLevel = max_allowed_risk
        self.revoked_capability_ids: Set[str] = set(revoked_capability_ids) if revoked_capability_ids is not None else set()

    def grant_permission(self, permission: str) -> None:
        self.granted_permissions.add(permission)

    def revoke_permission(self, permission: str) -> None:
        self.granted_permissions.discard(permission)

    def revoke_capability(self, capability_id: str) -> None:
        self.revoked_capability_ids.add(capability_id)

    def evaluate_capability(self, descriptor: CapabilityDescriptor) -> Tuple[bool, Optional[str]]:
        """Evaluates whether a single capability is authorized to execute."""
        cid = descriptor.capability_id
        if cid in self.revoked_capability_ids:
            return False, f"Capability '{cid}' is explicitly revoked by administrative policy."

        # Risk level boundary check
        if not descriptor.risk_level.is_within(self.max_allowed_risk):
            return False, (
                f"Capability '{cid}' requires risk level '{descriptor.risk_level.value}', "
                f"which exceeds maximum allowed boundary '{self.max_allowed_risk.value}'."
            )

        # Permissions check
        req_perms = set(descriptor.required_permissions)
        missing = req_perms - self.granted_permissions
        if missing:
            return False, f"Capability '{cid}' missing required host permissions: {sorted(missing)}."

        return True, None

    def evaluate_plan(self, plan_descriptors: Sequence[CapabilityDescriptor]) -> PermissionVerdict:
        """Evaluates an entire proposed sequence, ensuring all steps satisfy safety ACLs."""
        authorized: List[CapabilityDescriptor] = []
        blocked: List[CapabilityDescriptor] = []
        violations: List[str] = []

        if not plan_descriptors:
            violations.append("Empty plan grants no authorization.")

        for desc in plan_descriptors:
            is_ok, reason = self.evaluate_capability(desc)
            if is_ok:
                authorized.append(desc)
            else:
                blocked.append(desc)
                violations.append(reason or f"Blocked {desc.capability_id}")

        return PermissionVerdict(
            is_authorized=(len(authorized) > 0 and len(blocked) == 0),
            authorized_capabilities=authorized,
            blocked_capabilities=blocked,
            violations=violations,
            max_allowed_risk=self.max_allowed_risk,
        )
