"""Parallel Noul Firewall Matrix with Cost-of-Failure Gating.

Implements the orthogonal four-dimensional security firewall:
1. Destruction: Irreversible deletion, disk wipe, or history overwrite.
2. Exfiltration: Unauthorized extraction of API keys, environment secrets, or private keys.
3. Injection: Prompt injection, system prompt override, or jailbreak attempts.
4. Integrity: Tampering with protected branches, read-only configurations, or system files.

Enforces strict fail-closed gating based on Cost-of-Failure economics:
- READ_ONLY (0.50)
- STATE_MUTATION (0.80)
- DESTRUCTIVE (0.98)
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
import time
from typing import Any, Dict, List, Optional, Union


class CostOfFailureLevel(str, Enum):
    """Cost of failure tiers governing gating confidence thresholds."""
    READ_ONLY = "read_only"              # Low mistake cost, threshold = 0.50
    STATE_MUTATION = "state_mutation"    # Moderate mistake cost, threshold = 0.80
    DESTRUCTIVE = "destructive"          # High mistake cost, threshold = 0.98


@dataclass
class SecurityRiskVector:
    """Four-dimensional orthogonal security risk probabilities and gating verdict."""
    p_destructive: float
    p_exfiltration: float
    p_injection: float
    p_integrity_violation: float
    overall_safe: bool
    triggered_reasons: List[str]
    max_risk: float
    cost_level: CostOfFailureLevel
    threshold: float
    blocked: bool
    requires_human_approval: bool
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cost_level"] = self.cost_level.value
        return d


class ParallelNoulFirewall:
    """Non-autoregressive Parallel Noul Firewall with sub-15ms latency SLA."""

    # Default cost-of-failure confidence thresholds
    COST_THRESHOLDS = {
        CostOfFailureLevel.READ_ONLY: 0.50,
        CostOfFailureLevel.STATE_MUTATION: 0.80,
        CostOfFailureLevel.DESTRUCTIVE: 0.98,
    }

    # Pre-compiled probe patterns for 4 orthogonal dimensions
    _DESTRUCTION_PATTERNS = [
        re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f*|-f[a-zA-Z]*r)\s+.*(/|\*|\~|\$HOME|\broot\b|\bbin\b|\betc\b|\bvar\b|\busr\b)", re.I),
        re.compile(r"\brm\s+(-[a-zA-Z]*f*r*)\s+(\/|\/\*|\.\/|\*)", re.I),
        re.compile(r"\b(drop\s+database|drop\s+table|drop\s+schema|truncate\s+table)\b", re.I),
        re.compile(r"\b(mkfs|fdisk|dd\s+if=/dev/(zero|urandom)\s+of=)", re.I),
        re.compile(r"\bgit\s+(push\s+.*(--force|-f)|reset\s+--hard\s+origin/|clean\s+-fdx)", re.I),
        re.compile(r"\b(shred\s+-u|wipefs)\b", re.I),
    ]

    _EXFILTRATION_PATTERNS = [
        re.compile(r"\b(curl|wget|nc|netcat|ncat|socat)\b.*(\$|\b)(OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|TOKEN|SECRET|PASSWORD)", re.I),
        re.compile(r"\b(printenv|env)\b.*\|\s*(curl|wget|nc|webhook|transfer)", re.I),
        re.compile(r"\bcat\b.*(\.ssh/id_rsa|\.aws/credentials|\.gnupg|\.netrc|\.docker/config\.json)", re.I),
        re.compile(r"(https?://(webhook\.site|pipedream\.net|requestbin\.net|pastebin\.com|ngrok\.io|burpcollaborator\.net))", re.I),
        re.compile(r"\b(base64|xxd)\b.*id_rsa", re.I),
    ]

    _INJECTION_PATTERNS = [
        re.compile(r"(ignore\s+previous\s+instructions|disregard\s+all\s+prior|ignore\s+all\s+rules)", re.I),
        re.compile(r"(\bDAN\s+Mode\b|you\s+are\s+now\s+unrestricted|jailbreak|developer\s+mode\s+enabled)", re.I),
        re.compile(r"<\s*(SYSTEM|INSTRUCTION|PROMPT_OVERRIDE)\s*>.*<\s*/\s*(SYSTEM|INSTRUCTION|PROMPT_OVERRIDE)\s*>", re.I | re.S),
        re.compile(r"(you\s+must\s+bypass\s+all\s+filters|forget\s+that\s+you\s+are\s+an\s+ai)", re.I),
        re.compile(r"(\[\s*SYSTEM\s*:\s*OVERRIDE\s*\]|\{\{\s*bypass\s*\}\})", re.I),
    ]

    _INTEGRITY_PATTERNS = [
        re.compile(r"\b(tee|echo|cat|cp|mv)\b.*(\/etc\/(passwd|shadow|sudoers|hosts)|\.git\/config|\/boot\/)", re.I),
        re.compile(r"\bgit\s+checkout\s+-B\s+(main|master|prod|production)\b", re.I),
        re.compile(r"\b(chmod\s+(777|a\+rwx|000)\s+.*\/)", re.I),
        re.compile(r"\b(chown\s+-R\s+root:root\s+\/)", re.I),
        re.compile(r"\b(iptables\s+-F|ufw\s+disable)\b", re.I),
    ]

    def __init__(
        self,
        default_cost_level: CostOfFailureLevel = CostOfFailureLevel.STATE_MUTATION,
    ) -> None:
        self.default_cost_level = default_cost_level

    def infer_cost_level(self, tool_name: str, args_text: str) -> CostOfFailureLevel:
        """Heuristically infers the default cost level if not explicitly specified."""
        tool_lower = tool_name.lower()
        args_lower = args_text.lower()

        # Destructive tool names or high-risk command lines
        if any(term in tool_lower for term in ("delete", "destroy", "drop", "purge", "kill")):
            return CostOfFailureLevel.DESTRUCTIVE
        if any(pattern.search(args_text) for pattern in self._DESTRUCTION_PATTERNS):
            return CostOfFailureLevel.DESTRUCTIVE

        # Read-only tools
        read_only_tools = {"cat", "read_file", "view_file", "list_dir", "grep_search", "find_by_name", "read_url"}
        if tool_lower in read_only_tools or (tool_lower in ("run_command", "bash") and re.match(r"^\s*(ls|cat|pwd|git\s+status|git\s+diff|echo|head|tail|which)\b", args_lower)):
            return CostOfFailureLevel.READ_ONLY

        return CostOfFailureLevel.STATE_MUTATION

    def inspect_tool_call(
        self,
        tool_name: str,
        arguments: Union[str, Dict[str, Any]],
        cost_level: Optional[CostOfFailureLevel] = None,
        context: Optional[str] = None,
    ) -> SecurityRiskVector:
        """Performs a single-forward orthogonal 4-Noul evaluation on the tool call."""
        start_time = time.perf_counter()

        if isinstance(arguments, dict):
            args_text = json.dumps(arguments, ensure_ascii=False)
        else:
            args_text = str(arguments)

        eval_corpus = f"{tool_name} {args_text}"
        if context:
            eval_corpus = f"{context}\n{eval_corpus}"

        target_cost = cost_level or self.infer_cost_level(tool_name, args_text)
        threshold = self.COST_THRESHOLDS[target_cost]

        triggered_reasons: List[str] = []

        # 1. Destruction Probe
        destruct_matches = [p.pattern for p in self._DESTRUCTION_PATTERNS if p.search(eval_corpus)]
        if destruct_matches:
            p_destruct = 0.99
            triggered_reasons.append(f"Destruction probe: irreversible wipe or force reset detected ({destruct_matches[0]})")
        elif "rm " in eval_corpus or "drop " in eval_corpus.lower():
            p_destruct = 0.45
        else:
            p_destruct = 0.01

        # 2. Exfiltration Probe
        exfil_matches = [p.pattern for p in self._EXFILTRATION_PATTERNS if p.search(eval_corpus)]
        if exfil_matches:
            p_exfil = 0.99
            triggered_reasons.append(f"Exfiltration probe: secret, key, or env exfiltration detected ({exfil_matches[0]})")
        elif any(k in eval_corpus for k in ("TOKEN", "SECRET", "PRIVATE KEY", "password")):
            p_exfil = 0.35
        else:
            p_exfil = 0.01

        # 3. Injection Probe
        inject_matches = [p.pattern for p in self._INJECTION_PATTERNS if p.search(eval_corpus)]
        if inject_matches:
            p_inject = 0.99
            triggered_reasons.append(f"Injection probe: adversarial prompt injection or rule evasion detected ({inject_matches[0]})")
        elif "override" in eval_corpus.lower() or "bypass" in eval_corpus.lower():
            p_inject = 0.30
        else:
            p_inject = 0.01

        # 4. Integrity Probe
        integrity_matches = [p.pattern for p in self._INTEGRITY_PATTERNS if p.search(eval_corpus)]
        if integrity_matches:
            p_integrity = 0.99
            triggered_reasons.append(f"Integrity probe: protected system config or branch integrity violation detected ({integrity_matches[0]})")
        elif "/etc/" in eval_corpus or ".git/" in eval_corpus:
            p_integrity = 0.35
        else:
            p_integrity = 0.01

        max_risk = max(p_destruct, p_exfil, p_inject, p_integrity)
        
        # Gating logic:
        # If max_risk exceeds the allowed risk margin (1.0 - threshold), block immediately.
        allowed_risk_margin = round(1.0 - threshold, 4)
        blocked = max_risk > allowed_risk_margin
        requires_human_approval = blocked or (max_risk >= 0.30 and target_cost == CostOfFailureLevel.DESTRUCTIVE)
        overall_safe = not blocked

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SecurityRiskVector(
            p_destructive=round(p_destruct, 4),
            p_exfiltration=round(p_exfil, 4),
            p_injection=round(p_inject, 4),
            p_integrity_violation=round(p_integrity, 4),
            overall_safe=overall_safe,
            triggered_reasons=triggered_reasons,
            max_risk=round(max_risk, 4),
            cost_level=target_cost,
            threshold=threshold,
            blocked=blocked,
            requires_human_approval=requires_human_approval,
            latency_ms=round(latency_ms, 3),
        )
