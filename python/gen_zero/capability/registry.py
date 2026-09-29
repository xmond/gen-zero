"""Gen-Zero Capability Layer: Unified Capability Registry and Discovery Bus.

RFC-079 Implementation:
Maintains the centralized registry of Tool, Skill, Subagent, and CLI capabilities,
supporting Stage-1 coarse domain filtering (< 0.05ms) and snapshot digest hashing.
"""

from typing import Dict, List, Optional, Sequence, Set, Any
import threading
import hashlib
import json
import shutil
import os
import heapq

from .descriptor import CapabilityDescriptor, CapabilityType, RiskLevel


class CapabilityRegistry:
    """Thread-safe registry for orchestratable heterogeneous capabilities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._capabilities: Dict[str, CapabilityDescriptor] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._type_index: Dict[CapabilityType, Set[str]] = {t: set() for t in CapabilityType}
        self._domain_sorted: Dict[str, List[CapabilityDescriptor]] = {}

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Registers or updates a capability descriptor in the registry."""
        with self._lock:
            cid = descriptor.capability_id
            domain_key = descriptor.domain.lower()
            # Remove from old indexes if re-registering
            if cid in self._capabilities:
                old = self._capabilities[cid]
                self._domain_index.get(old.domain.lower(), set()).discard(cid)
                self._type_index[old.capability_type].discard(cid)

            self._capabilities[cid] = descriptor
            self._domain_index.setdefault(domain_key, set()).add(cid)
            self._type_index[descriptor.capability_type].add(cid)
            self._domain_sorted.clear()

    def unregister(self, capability_id: str) -> bool:
        """Removes a capability from the registry."""
        with self._lock:
            if capability_id not in self._capabilities:
                return False
            old = self._capabilities.pop(capability_id)
            self._domain_index.get(old.domain.lower(), set()).discard(capability_id)
            self._type_index[old.capability_type].discard(capability_id)
            self._domain_sorted.clear()
            return True

    def get(self, capability_id: str) -> Optional[CapabilityDescriptor]:
        """Retrieves a capability descriptor by ID."""
        with self._lock:
            return self._capabilities.get(capability_id)

    def list_all(self) -> List[CapabilityDescriptor]:
        """Returns all registered capability descriptors sorted by capability_id."""
        with self._lock:
            return sorted(self._capabilities.values(), key=lambda c: c.capability_id)

    def count(self) -> int:
        with self._lock:
            return len(self._capabilities)

    def filter(
        self,
        capability_type: Optional[CapabilityType] = None,
        domain: Optional[str] = None,
        max_risk: Optional[RiskLevel] = None,
        required_permissions: Optional[Set[str]] = None,
    ) -> List[CapabilityDescriptor]:
        """Filters capabilities by type, domain, maximum allowed risk, and available permissions."""
        with self._lock:
            candidates = list(self._capabilities.values())

            if capability_type is not None:
                candidates = [c for c in candidates if c.capability_type == capability_type]

            if domain is not None:
                candidates = [c for c in candidates if c.domain.lower() == domain.lower()]

            if max_risk is not None:
                candidates = [c for c in candidates if c.risk_level.is_within(max_risk)]

            if required_permissions is not None:
                candidates = [
                    c for c in candidates
                    if set(c.required_permissions).issubset(required_permissions)
                ]

            return sorted(candidates, key=lambda c: c.capability_id)

    def coarse_filter_stage1(
        self,
        domain: Optional[str] = None,
        max_risk: Optional[RiskLevel] = None,
        max_candidates: int = 16,
    ) -> List[CapabilityDescriptor]:
        """Stage 1: Sub-0.05ms coarse domain and risk gating to prune thousands of candidates down to N <= 16."""
        with self._lock:
            domain_key = domain.lower() if domain else None
            if domain_key:
                if domain_key not in self._domain_sorted:
                    cids = self._domain_index.get(domain_key, set())
                    cands = [self._capabilities[cid] for cid in cids if cid in self._capabilities]
                    cands.sort(key=lambda c: (c.cost_weight, c.capability_id))
                    self._domain_sorted[domain_key] = cands
                pool = self._domain_sorted[domain_key]
            else:
                if "_all" not in self._domain_sorted:
                    cands = list(self._capabilities.values())
                    cands.sort(key=lambda c: (c.cost_weight, c.capability_id))
                    self._domain_sorted["_all"] = cands
                pool = self._domain_sorted["_all"]

            if max_risk is None:
                return pool[:max_candidates]

            max_rank = max_risk.level_rank
            results: List[CapabilityDescriptor] = []
            for c in pool:
                if c.risk_level.level_rank <= max_rank:
                    results.append(c)
                    if len(results) >= max_candidates:
                        break
            return results

    def snapshot_hash(self, active_subset: Optional[Sequence[CapabilityDescriptor]] = None) -> str:
        """Computes deterministic SHA-256 snapshot hash across active capabilities."""
        with self._lock:
            subset = active_subset if active_subset is not None else self.list_all()
            canonical_records = []
            for c in sorted(subset, key=lambda x: x.capability_id):
                canonical_records.append({
                    "id": c.capability_id,
                    "type": c.capability_type.value,
                    "digest": c.digest(),
                })
            raw_bytes = json.dumps(canonical_records, sort_keys=True).encode("utf-8")
            return hashlib.sha256(raw_bytes).hexdigest()

    def auto_discover_mcp_tools(self, tools_schemas: Sequence[Dict[str, Any]]) -> int:
        """Automatically registers tools from MCP tools/list schemas."""
        count = 0
        with self._lock:
            for tool in tools_schemas:
                name = tool.get("name", "unknown")
                cid = f"tool:mcp_{name}" if not name.startswith("tool:") else name
                desc = tool.get("description", f"MCP Tool {name}")
                in_schema = tool.get("inputSchema", {})
                
                # Deduce domain and risk level from name and description
                domain = "system"
                risk = RiskLevel.READ_ONLY
                lower_name = name.lower()
                lower_desc = desc.lower()

                if any(w in lower_name for w in ["write", "create", "edit", "patch", "modify", "save"]):
                    risk = RiskLevel.WRITE_SAFE
                    domain = "dev"
                elif any(w in lower_name for w in ["delete", "remove", "drop", "kill", "purge", "rm"]):
                    risk = RiskLevel.DESTRUCTIVE
                    domain = "system"
                elif any(w in lower_name for w in ["http", "fetch", "web", "url", "download"]):
                    risk = RiskLevel.NETWORK
                    domain = "web"
                elif any(w in lower_name for w in ["test", "verify", "check", "assert"]):
                    domain = "verify"

                descriptor = CapabilityDescriptor(
                    capability_id=cid,
                    capability_type=CapabilityType.TOOL,
                    description=desc,
                    input_schema=in_schema,
                    domain=domain,
                    cost_weight=1.0,
                    risk_level=risk,
                    is_idempotent=(risk == RiskLevel.READ_ONLY),
                    timeout_seconds=30.0,
                )
                self.register(descriptor)
                count += 1
        return count

    def auto_discover_host_cli(
        self,
        command_names: Optional[Sequence[str]] = None,
    ) -> int:
        """Non-invasive auto-discovery of host system CLI commands in PATH."""
        default_commands = [
            ("git", "Git version control CLI", "dev", RiskLevel.WRITE_SAFE, ["fs:write"]),
            ("docker", "Docker containerization daemon CLI", "system", RiskLevel.DESTRUCTIVE, ["docker:admin"]),
            ("pytest", "Pytest automated test runner", "verify", RiskLevel.READ_ONLY, []),
            ("python3", "Python runtime interpreter", "dev", RiskLevel.WRITE_SAFE, ["process:exec"]),
            ("cargo", "Rust cargo build and test manager", "dev", RiskLevel.WRITE_SAFE, ["fs:write"]),
            ("curl", "HTTP command-line client", "web", RiskLevel.NETWORK, ["network:outbound"]),
            ("jq", "JSON command-line stream processor", "data", RiskLevel.READ_ONLY, []),
            ("tar", "Archive utility", "system", RiskLevel.WRITE_SAFE, ["fs:write"]),
            ("grep", "Pattern search utility", "data", RiskLevel.READ_ONLY, []),
        ]

        target_cmds = default_commands
        if command_names is not None:
            target_cmds = [
                (c, f"Host command {c}", "system", RiskLevel.WRITE_SAFE, ["process:exec"])
                for c in command_names
            ]

        count = 0
        with self._lock:
            for name, desc, domain, risk, perms in target_cmds:
                path = shutil.which(name)
                if path and os.path.isfile(path) and os.access(path, os.X_OK):
                    cid = f"cli:{name}"
                    descriptor = CapabilityDescriptor(
                        capability_id=cid,
                        capability_type=CapabilityType.CLI,
                        description=f"{desc} (Executable: {path})",
                        input_schema={"type": "object", "properties": {"args": {"type": "array", "items": {"type": "string"}}}},
                        domain=domain,
                        cost_weight=1.5,
                        risk_level=risk,
                        required_permissions=perms,
                        is_idempotent=(risk == RiskLevel.READ_ONLY),
                        timeout_seconds=60.0,
                        metadata={"binary_path": path},
                    )
                    self.register(descriptor)
                    count += 1
        return count

    def export_manifest(self) -> Dict[str, Any]:
        """Exports full registry state as a portable JSON manifest."""
        with self._lock:
            return {
                "version": "1.0.0",
                "total_capabilities": len(self._capabilities),
                "snapshot_hash": self.snapshot_hash(),
                "capabilities": [c.to_dict() for c in self.list_all()],
            }

    def import_manifest(self, manifest: Dict[str, Any]) -> int:
        """Imports capabilities from a manifest dictionary."""
        caps = manifest.get("capabilities", [])
        count = 0
        with self._lock:
            for c_dict in caps:
                desc = CapabilityDescriptor.from_dict(c_dict)
                self.register(desc)
                count += 1
        return count
