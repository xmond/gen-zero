"""Gen-Zero Layer 5: Semantic Perturbation & Invariance Evaluation Harness.

Evaluates decision models across 4 perturbation dimensions:
- Dim 1: Option Order Invariance (Permutation Equivariance, DFR == 0.0%, TVD <= 1e-4)
- Dim 2: Criterion Semantic Wrapper (Paraphrase robustness, DFR <= 1.5%, TVD <= 0.05)
- Dim 3: Irrelevant Context Injection (Noise resistance, DFR <= 1.0%, TVD <= 0.04)
- Dim 4: Missing Evidence / Insufficient Data (Active abstain recall >= 98.0%, conf < 0.70)
"""

import os
import json
import math
from typing import Dict, List, Any, Optional, Callable, Tuple


class PerturbationEvaluator:
    """Automated evaluation harness for the 108-case perturbation stability suite."""

    def __init__(self, benchmark_file: Optional[str] = None):
        if benchmark_file is None:
            # Default to benchmark suite path in repo
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            benchmark_file = os.path.join(base_dir, "data", "benchmarks", "perturbation_suite_108.jsonl")
        self.benchmark_file = benchmark_file
        self.cases: List[Dict[str, Any]] = []
        self._load_cases()

    def _load_cases(self):
        if os.path.exists(self.benchmark_file):
            with open(self.benchmark_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.cases.append(json.loads(line))

    @staticmethod
    def compute_tvd(probs1: Dict[str, float], probs2: Dict[str, float]) -> float:
        """Computes Total Variation Distance (TVD) between two probability distributions:
        TVD = 0.5 * sum(|p1(c) - p2(c)|).
        """
        all_keys = set(probs1.keys()).union(set(probs2.keys()))
        diff_sum = sum(abs(probs1.get(k, 0.0) - probs2.get(k, 0.0)) for k in all_keys)
        return 0.5 * diff_sum

    def evaluate_model(
        self,
        decide_fn: Callable[[str, List[str]], Dict[str, Any]],
        tier: int = 2
    ) -> Dict[str, Any]:
        """Evaluates a decision function over the perturbation suite.
        
        Args:
            decide_fn: Callable accepting (state: str, candidates: List[str]) -> Dict with keys:
                       - 'action': selected candidate action string
                       - 'probs': Dict[str, float]
                       - 'confidence': float in [0, 1]
            tier: 1 for fast in-loop micro-gate (Dim 1 & 4), 2 for full 108-case promotion gate.
        
        Returns:
            Dict containing multi-dimensional DFR, TVD, and Abstain recall metrics.
        """
        if not self.cases:
            self._load_cases()

        # Group cases by base_id
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for c in self.cases:
            b_id = c["base_id"]
            if b_id not in grouped:
                grouped[b_id] = {}
            dim = c["dimension"]
            # Store list or dict per dimension
            if dim not in grouped[b_id]:
                grouped[b_id][dim] = []
            grouped[b_id][dim].append(c)

        dim1_flips = 0
        dim1_tvds: List[float] = []
        dim1_evals = 0

        dim2_flips = 0
        dim2_tvds: List[float] = []
        dim2_evals = 0

        dim3_flips = 0
        dim3_tvds: List[float] = []
        dim3_evals = 0

        dim4_abstains = 0
        dim4_non_abstain_confidences: List[float] = []
        dim4_evals = 0

        total_evaluated = 0

        for b_id, dims in grouped.items():
            # Evaluate canonical reference
            canon_list = dims.get("canonical", [])
            if not canon_list:
                continue
            canon_case = canon_list[0]
            canon_res = decide_fn(canon_case["state"], canon_case["candidates"])
            canon_action = canon_res.get("action")
            canon_probs = canon_res.get("probs", {})
            total_evaluated += 1

            # 1. Dim 1: Order Reversal & Permutation
            dim1_cases = dims.get("dim1_order_reversal", []) + dims.get("dim1_alt_order", [])
            for c1 in dim1_cases:
                r1 = decide_fn(c1["state"], c1["candidates"])
                a1 = r1.get("action")
                p1 = r1.get("probs", {})
                if a1 != canon_action:
                    dim1_flips += 1
                dim1_tvds.append(self.compute_tvd(canon_probs, p1))
                dim1_evals += 1
                total_evaluated += 1

            # 4. Dim 4: Missing Evidence / Insufficient Data
            dim4_cases = dims.get("dim4_missing_evidence", [])
            for c4 in dim4_cases:
                r4 = decide_fn(c4["state"], c4["candidates"])
                a4 = r4.get("action")
                conf4 = r4.get("confidence", 0.0)
                if a4 == "ABSTAIN":
                    dim4_abstains += 1
                else:
                    dim4_non_abstain_confidences.append(conf4)
                dim4_evals += 1
                total_evaluated += 1

            if tier == 2:
                # 2. Dim 2: Criterion Semantic Wrapper
                dim2_cases = dims.get("dim2_semantic_wrapper", [])
                for c2 in dim2_cases:
                    r2 = decide_fn(c2["state"], c2["candidates"])
                    a2 = r2.get("action")
                    p2 = r2.get("probs", {})
                    if a2 != canon_action:
                        dim2_flips += 1
                    dim2_tvds.append(self.compute_tvd(canon_probs, p2))
                    dim2_evals += 1
                    total_evaluated += 1

                # 3. Dim 3: Irrelevant Context Noise Injection
                dim3_cases = dims.get("dim3_noise_injection", [])
                for c3 in dim3_cases:
                    r3 = decide_fn(c3["state"], c3["candidates"])
                    a3 = r3.get("action")
                    p3 = r3.get("probs", {})
                    if a3 != canon_action:
                        dim3_flips += 1
                    dim3_tvds.append(self.compute_tvd(canon_probs, p3))
                    dim3_evals += 1
                    total_evaluated += 1

        dim1_dfr = (dim1_flips / max(1, dim1_evals))
        dim1_mean_tvd = (sum(dim1_tvds) / max(1, len(dim1_tvds))) if dim1_tvds else 0.0

        dim2_dfr = (dim2_flips / max(1, dim2_evals)) if dim2_evals else 0.0
        dim2_mean_tvd = (sum(dim2_tvds) / max(1, len(dim2_tvds))) if dim2_tvds else 0.0

        dim3_dfr = (dim3_flips / max(1, dim3_evals)) if dim3_evals else 0.0
        dim3_mean_tvd = (sum(dim3_tvds) / max(1, len(dim3_tvds))) if dim3_tvds else 0.0

        dim4_abstain_recall = (dim4_abstains / max(1, dim4_evals))
        dim4_max_conf = max(dim4_non_abstain_confidences) if dim4_non_abstain_confidences else 0.0

        if total_evaluated == 0 or not self.cases:
            return {
                "dim1_dfr": 1.0,
                "dim1_tvd": 1.0,
                "dim2_dfr": 1.0,
                "dim2_tvd": 1.0,
                "dim3_dfr": 1.0,
                "dim3_tvd": 1.0,
                "dim4_abstain_recall": 0.0,
                "dim4_max_confidence": 1.0,
                "tier": tier,
                "total_evaluated": 0,
                "is_valid": False,
                "error": "EMPTY_OR_MISSING_BENCHMARK_SUITE"
            }

        return {
            "dim1_dfr": round(dim1_dfr, 5),
            "dim1_tvd": round(dim1_mean_tvd, 6),
            "dim2_dfr": round(dim2_dfr, 4),
            "dim2_tvd": round(dim2_mean_tvd, 4),
            "dim3_dfr": round(dim3_dfr, 4),
            "dim3_tvd": round(dim3_mean_tvd, 4),
            "dim4_abstain_recall": round(dim4_abstain_recall, 4),
            "dim4_max_confidence": round(dim4_max_conf, 4),
            "tier": tier,
            "total_evaluated": total_evaluated,
            "is_valid": True
        }
