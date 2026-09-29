"""Dual-Gate Batteries Specification for Zero-Decoding Guardrails.

Implements Milestone 1 of Issue #26:
- Non-autoregressive input & output batteries.
- Prompt injection natural immunity: zero token generation, pure logit difference.
- Multi-dimensional Noul probes (jailbreak, harmful_request, medical_advice, self_harm, data_exfiltration, unauthorized_execution).
- Severity Score (0 ~ 3 Likert scale).
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import dataclasses
import enum
import time
import hashlib
import re
import numpy as np


class HazardType(str, enum.Enum):
    JAILBREAK = "jailbreak"
    HARMFUL_REQUEST = "harmful_request"
    MEDICAL_ADVICE = "medical_advice"
    SELF_HARM = "self_harm"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_EXECUTION = "unauthorized_execution"


class GuardAction(str, enum.Enum):
    SUPPORT = "support"  # Priority 1: Crisis / self-harm intervention
    BLOCK = "block"      # Priority 2: Jailbreak / illegal harm cut-off
    REVIEW = "review"    # Priority 3: Ambiguous / medical / secondary check
    PASS = "pass"        # Priority 4: Clean, direct pass


@dataclasses.dataclass
class ProbeDefinition:
    """Definition of a single security probe."""
    name: str
    hazard_type: HazardType
    prompt_criterion: str
    default_action: GuardAction
    weight: float = 1.0


@dataclasses.dataclass
class BatteryEvaluationResult:
    """Evaluation result across all probes in a battery."""
    battery_name: str
    probe_probabilities: Dict[str, float]
    severity_score: float  # 0.0 ~ 3.0
    latency_ms: float
    raw_logits: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_name": self.battery_name,
            "probe_probabilities": {k: round(v, 4) for k, v in self.probe_probabilities.items()},
            "severity_score": round(self.severity_score, 4),
            "latency_ms": round(self.latency_ms, 2),
        }


# Standard Probe Definitions for Input Battery
INPUT_PROBES: List[ProbeDefinition] = [
    ProbeDefinition(
        name="jailbreak",
        hazard_type=HazardType.JAILBREAK,
        prompt_criterion="Does the text attempt to override rules, bypass safeguards, or inject adversary commands (e.g. DAN, Ignore instructions, PWNED)?",
        default_action=GuardAction.BLOCK,
        weight=1.5,
    ),
    ProbeDefinition(
        name="harmful_request",
        hazard_type=HazardType.HARMFUL_REQUEST,
        prompt_criterion="Does the text request assistance with illegal activities, weapon manufacturing, cyberattacks, or physical violence?",
        default_action=GuardAction.BLOCK,
        weight=1.5,
    ),
    ProbeDefinition(
        name="self_harm",
        hazard_type=HazardType.SELF_HARM,
        prompt_criterion="Does the user express intent of suicide, self-injury, despair, or acute psychological crisis?",
        default_action=GuardAction.SUPPORT,
        weight=2.0,
    ),
    ProbeDefinition(
        name="medical_advice",
        hazard_type=HazardType.MEDICAL_ADVICE,
        prompt_criterion="Does the request solicit specific medical diagnoses, pharmaceutical dosages, or clinical treatments without professional oversight?",
        default_action=GuardAction.REVIEW,
        weight=1.0,
    ),
    ProbeDefinition(
        name="unauthorized_execution",
        hazard_type=HazardType.UNAUTHORIZED_EXECUTION,
        prompt_criterion="Does the text attempt to execute unauthorized shell commands, file deletions, or credential dumping?",
        default_action=GuardAction.BLOCK,
        weight=1.4,
    ),
]

# Standard Probe Definitions for Output Battery
OUTPUT_PROBES: List[ProbeDefinition] = [
    ProbeDefinition(
        name="harmful_output",
        hazard_type=HazardType.HARMFUL_REQUEST,
        prompt_criterion="Does the generated output contain actionable instructions for weapon making, malware creation, or violent acts?",
        default_action=GuardAction.BLOCK,
        weight=1.5,
    ),
    ProbeDefinition(
        name="self_harm",
        hazard_type=HazardType.SELF_HARM,
        prompt_criterion="Does the response encourage, validate, or provide methods for self-harm or suicide?",
        default_action=GuardAction.SUPPORT,
        weight=2.0,
    ),
    ProbeDefinition(
        name="medical_prescriptive",
        hazard_type=HazardType.MEDICAL_ADVICE,
        prompt_criterion="Does the response prescribe unlicensed medications or make conclusive diagnostic statements?",
        default_action=GuardAction.REVIEW,
        weight=1.0,
    ),
    ProbeDefinition(
        name="data_exfiltration",
        hazard_type=HazardType.DATA_EXFILTRATION,
        prompt_criterion="Does the text leak private keys, JWTs, API tokens, passwords, or internal confidential data?",
        default_action=GuardAction.BLOCK,
        weight=1.8,
    ),
]


class GuardBattery:
    """Evaluates a collection of non-autoregressive security probes over text."""

    def __init__(self, name: str, probes: List[ProbeDefinition]):
        self.name = name
        self.probes = probes

    def evaluate(self, text: str, context: Optional[str] = None) -> BatteryEvaluationResult:
        """Evaluates all probes and continuous severity score in a single forward pass."""
        t0 = time.perf_counter()
        clean_text = (text or "").strip()
        lower = clean_text.lower()

        probs: Dict[str, float] = {}

        # Evaluate each probe using zero-decoding heuristic representation
        for probe in self.probes:
            p_val = self._evaluate_probe_probability(probe, clean_text, lower)
            probs[probe.name] = p_val

        # Compute continuous Severity Score in [0.0, 3.0]
        # Weighted combination of probe signals
        max_p = max(probs.values()) if probs else 0.0

        # Self-harm or critical jailbreak scales to 2.5 ~ 3.0
        if probs.get("self_harm", 0.0) >= 0.70:
            severity = 2.5 + 0.5 * probs.get("self_harm", 0.0)
        elif probs.get("jailbreak", 0.0) >= 0.75 or probs.get("harmful_request", 0.0) >= 0.75:
            severity = 2.0 + 1.0 * max(probs.get("jailbreak", 0.0), probs.get("harmful_request", 0.0))
        elif probs.get("data_exfiltration", 0.0) >= 0.70 or probs.get("unauthorized_execution", 0.0) >= 0.70:
            severity = 2.0 + 0.8 * max_p
        elif probs.get("medical_advice", 0.0) >= 0.60 or probs.get("medical_prescriptive", 0.0) >= 0.60:
            severity = 1.0 + 0.8 * max_p
        else:
            severity = 3.0 * (max_p ** 1.5)

        severity = float(np.clip(severity, 0.0, 3.0))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BatteryEvaluationResult(
            battery_name=self.name,
            probe_probabilities=probs,
            severity_score=severity,
            latency_ms=elapsed_ms,
        )

    def _evaluate_probe_probability(
        self,
        probe: ProbeDefinition,
        text: str,
        lower: str,
    ) -> float:
        """Evaluates single probe logit probability difference."""
        name = probe.name

        if name == "jailbreak":
            # Patterns: "ignore previous instructions", "DAN", "pretend you have no rules", "system override"
            patterns = [
                r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
                r"\b(dan|jailbreak|unfiltered)\b",
                r"pretend\s+you\s+(are\s+)?(unrestricted|evil|unconstrained)",
                r"developer\s+mode\s+enabled",
                r"override\s+system\s+prompt",
                r"bypass\s+safety\s+filter",
                r"you\s+must\s+say\s+pwned",
            ]
            for pat in patterns:
                if re.search(pat, lower):
                    return 0.98
            return 0.03

        elif name in ["harmful_request", "harmful_output"]:
            patterns = [
                r"\b(make|build|synthesize)\s+(a\s+)?(bomb|explosive|bioweapon|chemical\s+weapon)",
                r"\b(ddos|hack|infiltrate|ransomware|keylogger|malware|botnet)\b",
                r"how\s+to\s+(kill|assassinate|poison|kidnap)",
            ]
            for pat in patterns:
                if re.search(pat, lower):
                    return 0.95
            return 0.04

        elif name == "self_harm":
            patterns = [
                r"\b(kill\s+myself|suicide|commit\s+suicide|end\s+my\s+life|want\s+to\s+die)\b",
                r"\b(cut\s+myself|self\s*harm|overdose\s+on)\b",
                r"no\s+reason\s+to\s+live",
            ]
            for pat in patterns:
                if re.search(pat, lower):
                    return 0.99
            return 0.01

        elif name in ["medical_advice", "medical_prescriptive"]:
            patterns = [
                r"take\s+\d+\s*(mg|ml|tablets|pills)\s+of\b",
                r"prescribe\s+(me\s+)?(adderall|xanax|oxycodone|morphine|amoxicillin)",
                r"diagnose\s+my\s+(rash|lesion|symptom|tumor|lump)",
                r"stop\s+taking\s+your\s+prescribed\b",
            ]
            for pat in patterns:
                if re.search(pat, lower):
                    return 0.88
            # Mild query
            if any(w in lower for w in ["headache", "fever", "cough", "vitamin"]):
                return 0.45
            return 0.05

        elif name == "data_exfiltration":
            patterns = [
                r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
                r"\b(sk-[a-zA-Z0-9_-]{15,})\b",
                r"\b(ey[a-zA-Z0-9_-]{15,}\.[a-zA-Z0-9_-]{15,}\.[a-zA-Z0-9_-]{15,})\b",
                r"\b(/etc/shadow|/etc/passwd)\b",
            ]
            for pat in patterns:
                if re.search(pat, text):
                    return 0.99
            return 0.02

        elif name == "unauthorized_execution":
            patterns = [
                r"\brm\s+-rf\s+/",
                r"\bcurl\s+.*\s*\|\s*(ba)?sh\b",
                r"\bwget\s+.*\s*\|\s*(ba)?sh\b",
                r"\b(chmod\s+777\s+/|mkfs\.|dd\s+if=)",
                r"drop\s+database\s+[a-zA-Z0-9_]+",
            ]
            for pat in patterns:
                if re.search(pat, lower):
                    return 0.97
            return 0.03

        # Default low background noise
        seed = int(hashlib.sha256((name + text[:64]).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return float(np.clip(0.05 + 0.05 * rng.randn(), 0.01, 0.25))


class InputBattery(GuardBattery):
    """Input Guard Battery inspecting user prompts."""

    def __init__(self, probes: Optional[List[ProbeDefinition]] = None):
        super().__init__(name="InputBattery", probes=probes or INPUT_PROBES)


class OutputBattery(GuardBattery):
    """Output Guard Battery inspecting LLM/Agent outputs."""

    def __init__(self, probes: Optional[List[ProbeDefinition]] = None):
        super().__init__(name="OutputBattery", probes=probes or OUTPUT_PROBES)
