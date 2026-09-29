"""Candidate-order sensitivity benchmark for the `gen-zero reflex` CLI.

Measures how often the chosen action changes when the candidate list is shuffled.
The observed flip rate is an empirical count over the trials below. It is not a proof,
and this suite makes no guarantee claim: a 0 here only means 0 flips in these
trials, on these candidate pools and contexts.

Fail-closed rules:
- A call that exits non-zero, prints non-JSON, or omits `meta.chosen_action` is an
  ERROR SAMPLE. It is never coerced to an empty choice, so two failures cannot
  masquerade as "no flip".
- A trial whose canonical (base) call fails is skipped, and every shuffle in it is
  counted as invalid.
- If no comparison is valid, the suite raises instead of reporting a rate.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_ERROR_DETAILS = 20
MAX_RESHUFFLE_ATTEMPTS = 50


class PermutationEquivarianceSuite:
    """Empirically measures candidate-order flips of the reflex decision."""

    def __init__(
        self,
        binary_path: str = "./target/release/gen-zero",
        num_trials: int = 50,
        shuffles_per_trial: int = 10,
        seed: int = 0,
    ):
        if num_trials < 1 or shuffles_per_trial < 1:
            raise ValueError("num_trials and shuffles_per_trial must be >= 1")
        self.binary_path = Path(binary_path)
        self.num_trials = num_trials
        self.shuffles_per_trial = shuffles_per_trial
        self.seed = seed

    def run(self) -> Dict[str, Any]:
        print("  Running Candidate-Order Sensitivity Benchmark...")
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"gen-zero binary not found: {self.binary_path}; build it, no numbers are invented")

        rng = random.Random(self.seed)
        candidate_pools = [
            ["execute_transfer", "flag_risk", "request_mfa", "audit_log"],
            ["proceed_route_a", "re_route_b", "emergency_stop", "escalate_human"],
            ["grant_read_only", "grant_admin", "revoke_session", "isolate_agent"],
            ["cache_result", "evict_oldest", "compress_context", "drop_frame"],
            ["allow_packet", "drop_packet", "throttle_bandwidth", "deep_inspect"],
        ]

        valid_comparisons = 0
        flips = 0
        error_samples: List[Dict[str, str]] = []
        error_count = 0
        skipped_shuffles = 0

        def record_error(trial: int, order: List[str], reason: str) -> None:
            nonlocal error_count
            error_count += 1
            if len(error_samples) < MAX_ERROR_DETAILS:
                error_samples.append({"trial": str(trial), "order": ",".join(order), "reason": reason})

        for trial in range(self.num_trials):
            candidates = candidate_pools[trial % len(candidate_pools)]
            context = f"Security arbiter decision scenario {trial}: candidate assessment under adversarial constraint"

            base_choice, err = self._call_zero(context, candidates)
            if base_choice is None:
                record_error(trial, candidates, f"base call failed: {err}")
                skipped_shuffles += self.shuffles_per_trial
                continue

            for _ in range(self.shuffles_per_trial):
                shuffled = self._distinct_shuffle(rng, candidates)
                shuffled_choice, err = self._call_zero(context, shuffled)
                if shuffled_choice is None:
                    record_error(trial, shuffled, f"shuffled call failed: {err}")
                    continue
                valid_comparisons += 1
                if shuffled_choice != base_choice:
                    flips += 1

        if valid_comparisons == 0:
            raise RuntimeError(
                f"equivariance suite produced 0 valid comparisons ({error_count} error samples); "
                f"first errors: {error_samples[:3]}"
            )

        flip_rate = flips / valid_comparisons * 100.0
        return {
            "is_synthetic": True,
            "num_trials": self.num_trials,
            "shuffles_per_trial": self.shuffles_per_trial,
            "seed": self.seed,
            "valid_comparisons": valid_comparisons,
            "observed_flips": flips,
            "flip_rate_percent": round(flip_rate, 4),
            "error_samples": error_count,
            "skipped_shuffles_after_base_failure": skipped_shuffles,
            "error_details": error_samples,
            "status": "ok" if error_count == 0 else "partial_errors",
            "scope": (
                f"{len(candidate_pools)} fixed candidate pools of 4, one context per trial; "
                "empirical count only, no equivariance proof is claimed"
            ),
        }

    @staticmethod
    def _distinct_shuffle(rng: random.Random, candidates: List[str]) -> List[str]:
        """Shuffle to an order different from the canonical one (identity order tests nothing)."""
        for _ in range(MAX_RESHUFFLE_ATTEMPTS):
            shuffled = candidates.copy()
            rng.shuffle(shuffled)
            if shuffled != candidates:
                return shuffled
        raise RuntimeError(f"could not draw a non-identity permutation of {candidates}")

    def _call_zero(self, context: str, candidates: List[str]) -> Tuple[Optional[str], str]:
        """Return (chosen_action, "") on success, or (None, reason) on any failure."""
        res = subprocess.run(
            [str(self.binary_path), "reflex", "--context", context, "--candidates", ",".join(candidates)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return None, f"exit {res.returncode}: {res.stderr.strip()[:200]}"
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError as err:
            return None, f"stdout is not JSON: {err}"
        meta = data.get("meta") if isinstance(data, dict) else None
        chosen = meta.get("chosen_action") if isinstance(meta, dict) else None
        if not isinstance(chosen, str) or not chosen:
            tier = meta.get("tier") if isinstance(meta, dict) else None
            is_error = data.get("is_error") if isinstance(data, dict) else None
            return None, f"response has no non-empty meta.chosen_action (is_error={is_error}, tier={tier})"
        if data.get("is_error") or chosen not in candidates:
            return None, "response is an error or chosen_action is outside the candidate set"
        return chosen, ""
