"""Gen-Zero Harness Layer: Non-invasive Host Environment Auto-Discovery and Setup.

RFC-079 Implementation:
Automates host capability scanning and registration:
1. Probes host PATH for pre-installed development tools (git, docker, python3, pytest, cargo, etc.).
2. Creates standardized CapabilityDescriptors without manual glue code.
3. Generates a local capability manifest and snapshot hash for deterministic audit trails.
"""

from typing import Dict, List, Optional, Any
import os
import json
import time

from gen_zero.capability.registry import CapabilityRegistry
from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType, RiskLevel


def setup_host_harness(
    output_path: Optional[str] = None,
    custom_commands: Optional[List[str]] = None,
    extra_descriptors: Optional[List[CapabilityDescriptor]] = None,
) -> Dict[str, Any]:
    """Executes automated, non-invasive scan of the host environment.

    Args:
        output_path: Path to write the exported capability manifest JSON (defaults to '.gen_zero_capabilities.json').
        custom_commands: Optional list of additional command binary names to probe in PATH.
        extra_descriptors: Optional list of pre-configured descriptors to register.

    Returns:
        Summary dict containing total detected capabilities, snapshot hash, and manifest file path.
    """
    t0 = time.perf_counter()
    registry = CapabilityRegistry()

    # 1. Discover host CLI tools
    cli_count = registry.auto_discover_host_cli(command_names=custom_commands)

    # 2. Register core Gen-Zero system capabilities (skills, subagents, built-in tools)
    core_skills = [
        CapabilityDescriptor(
            capability_id="skill:code_refactor",
            capability_type=CapabilityType.SKILL,
            description="Progressive AST-safe code refactoring and dead-code pruning workflow",
            domain="dev",
            cost_weight=2.0,
            risk_level=RiskLevel.WRITE_SAFE,
            required_permissions=["fs:read", "fs:write"],
            timeout_seconds=120.0,
        ),
        CapabilityDescriptor(
            capability_id="skill:benchmark_audit",
            capability_type=CapabilityType.SKILL,
            description="Automated latency, memory, and regression performance evaluation",
            domain="verify",
            cost_weight=1.5,
            risk_level=RiskLevel.READ_ONLY,
            timeout_seconds=60.0,
        ),
        CapabilityDescriptor(
            capability_id="subagent:continuous_verifier",
            capability_type=CapabilityType.SUBAGENT,
            description="Continuous verification and evidence-source locking autonomous agent",
            domain="verify",
            cost_weight=3.0,
            risk_level=RiskLevel.READ_ONLY,
            required_permissions=["fs:read"],
            timeout_seconds=180.0,
        ),
    ]

    for s in core_skills:
        registry.register(s)

    # 3. Register any additional user descriptors
    if extra_descriptors:
        for d in extra_descriptors:
            registry.register(d)

    # 4. Export manifest
    target_file = output_path or os.path.abspath(".gen_zero_capabilities.json")
    manifest = registry.export_manifest()
    
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "SUCCESS",
        "total_capabilities": registry.count(),
        "host_cli_detected": cli_count,
        "core_skills_registered": len(core_skills),
        "snapshot_hash": registry.snapshot_hash(),
        "manifest_path": target_file,
        "setup_time_ms": round(elapsed_ms, 3),
        "capabilities": [c.to_dict() for c in registry.list_all()],
    }
