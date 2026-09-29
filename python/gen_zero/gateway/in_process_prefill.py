"""In-process heuristic feature gates for eight decision categories.

Uses tool-name lists, substring/regular-expression matches, and context flags to
assign hand-selected risk scores. Scores and their logit transforms are not
learned neural predictions or calibrated probabilities. Context hashing provides
a fingerprint; this implementation does not perform shared neural KV-cache prefill.
Probes run sequentially without network RPC. Reported latency is a local timing
measurement, not a CPU/GPU latency guarantee or evidence of safety coverage.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import dataclasses
import time
import re
import hashlib
import numpy as np


STANDARD_PROBES = [
    "tool_authorization",
    "content_compliance",
    "privilege_escalation",
    "data_exfiltration",
    "result_consistency",
    "rate_limit_budget",
    "irreversible_action",
    "schema_validity",
]


@dataclasses.dataclass
class InProcessPrefillResult:
    shared_prefix_hash: str
    probe_probabilities: Dict[str, float]
    probe_logits: List[float]
    aggregate_risk_score: float
    is_safe: bool
    flagged_probes: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared_prefix_hash": self.shared_prefix_hash,
            "probe_probabilities": {k: round(v, 4) for k, v in self.probe_probabilities.items()},
            "probe_logits": [round(l, 4) for l in self.probe_logits],
            "aggregate_risk_score": round(self.aggregate_risk_score, 4),
            "is_safe": self.is_safe,
            "flagged_probes": self.flagged_probes,
            "latency_ms": round(self.latency_ms, 2),
        }


class InProcessPrefillEngine:
    """Evaluates eight heuristic decision gates in-process."""

    def __init__(
        self,
        risk_cutoff: float = 0.50,
        enable_simd: bool = True,
    ):
        self.risk_cutoff = float(risk_cutoff)
        self.enable_simd = bool(enable_simd)
        self._prefix_cache: Dict[str, Dict[str, Any]] = {}

    def compute_prefix_hash(self, context_text: str) -> str:
        """Computes deterministic SHA-256 fingerprint of the shared context prefix."""
        return hashlib.sha256(context_text.encode("utf-8")).hexdigest()[:16]

    def evaluate_probes(
        self,
        context: Dict[str, Any],
    ) -> InProcessPrefillResult:
        """Evaluates eight heuristic probes sequentially without network RPC."""
        t0 = time.perf_counter()

        # Extract text context
        user_prompt = str(context.get("prompt") or context.get("user_prompt") or "")
        tool_name = str(context.get("tool_name") or "")
        arguments = context.get("arguments") or {}
        raw_context = f"{user_prompt}\n{tool_name}\n{str(arguments)}"
        prefix_hash = self.compute_prefix_hash(raw_context)

        lower_ctx = raw_context.lower()

        # Vectorized probe evaluation
        probe_scores = np.zeros(len(STANDARD_PROBES), dtype=np.float32)

        # 1. tool_authorization: check if tool is in allowed list or disallowed
        disallowed_tools = {"system_exec", "eval", "shell_exec", "raw_sql"}
        if tool_name in disallowed_tools:
            probe_scores[0] = 0.95
        else:
            probe_scores[0] = 0.05

        # 2. content_compliance: prompt injection, jailbreak, DAN
        inj_cues = ["ignore all previous instructions", "override prompt", "dan mode", "jailbreak", "sudo mode"]
        has_inj = any(cue in lower_ctx for cue in inj_cues)
        probe_scores[1] = 0.96 if has_inj else 0.04

        # 3. privilege_escalation: elevation to root, admin, destructive shell
        priv_cues = ["chmod 777", "sudo", "--force", "chown", "grant all", "drop table", "rm -rf /", "rm -rf *"]
        has_priv = any(cue in lower_ctx for cue in priv_cues)
        probe_scores[2] = 0.98 if has_priv else 0.02

        # 4. data_exfiltration: private key, api key, token leakage
        exfil_patterns = [r"sk-[a-zA-Z0-9]{20,}", r"-----BEGIN (RSA|OPENSSH) PRIVATE KEY-----", r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9]+"]
        has_exfil = any(re.search(pat, raw_context) for pat in exfil_patterns)
        probe_scores[3] = 0.97 if has_exfil else 0.03

        # 5. result_consistency: plan vs action alignment
        has_inconsistency = bool(context.get("force_inconsistent", False))
        probe_scores[4] = 0.85 if has_inconsistency else 0.05

        # 6. rate_limit_budget: quota or rate breach check
        rate_breach = bool(context.get("rate_breach", False))
        probe_scores[5] = 0.90 if rate_breach else 0.02

        # 7. irreversible_action: permanent deletion, balance deduction, destructive mutation
        irreversible_cues = ["delete", "drop", "truncate", "format", "wipe", "transfer_funds", "rm "]
        has_irrev = any(cue in lower_ctx for cue in irreversible_cues)
        probe_scores[6] = 0.92 if has_irrev else 0.08

        # 8. schema_validity: invalid json or missing required keys
        schema_err = bool(context.get("schema_error", False))
        probe_scores[7] = 0.95 if schema_err else 0.02

        # Convert continuous probabilities to logits: L = ln(p / (1 - p))
        clipped_p = np.clip(probe_scores, 0.001, 0.999)
        logits = np.log(clipped_p / (1.0 - clipped_p))

        # Non-linear Max-Pooling & Co-Riding Logits Aggregation
        # Risk is driven by the most severe probe with secondary weighted accumulation
        max_probe_risk = float(np.max(probe_scores))
        mean_probe_risk = float(np.mean(probe_scores))
        aggregate_risk = float(np.clip(0.85 * max_probe_risk + 0.15 * mean_probe_risk, 0.0, 1.0))

        flagged = [
            STANDARD_PROBES[i] for i, score in enumerate(probe_scores) if score >= 0.50
        ]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        probe_prob_dict = {
            STANDARD_PROBES[i]: float(probe_scores[i]) for i in range(len(STANDARD_PROBES))
        }

        return InProcessPrefillResult(
            shared_prefix_hash=prefix_hash,
            probe_probabilities=probe_prob_dict,
            probe_logits=[float(l) for l in logits],
            aggregate_risk_score=aggregate_risk,
            is_safe=(aggregate_risk < self.risk_cutoff),
            flagged_probes=flagged,
            latency_ms=latency_ms,
        )

    def benchmark_latency(self, iterations: int = 100) -> Dict[str, float]:
        """Measures local sample latency; sla_met compares it to an 8ms target only."""
        sample_context = {
            "prompt": "Please fetch the latest quarterly revenue metrics from the reporting API.",
            "tool_name": "fetch_financial_data",
            "arguments": {"metric": "revenue", "quarter": "Q3-2026"},
        }

        latencies = []
        # Warmup
        for _ in range(10):
            self.evaluate_probes(sample_context)

        for _ in range(iterations):
            res = self.evaluate_probes(sample_context)
            latencies.append(res.latency_ms)

        latencies_arr = np.array(latencies, dtype=np.float32)
        return {
            "iterations": float(iterations),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "max_ms": float(np.max(latencies_arr)),
            "sla_met": bool(np.percentile(latencies_arr, 99) <= 8.0),
        }
