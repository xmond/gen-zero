"""Contrastive Fact Mutation, Verifier Hard-Negative Mining, and Tone Invariance Pipeline.

Implements Milestone 1 of Issue #17:
1. ContrastiveMutationEngine: Rule-driven minimal factual perturbations (condition invert,
   temporal flip, entity substitution) producing balanced positive/negative pairs.
2. VerifierFeedbackMiner: Extracts real test failures, stack traces, and false completion
   trajectories from Issue #13 Verifier to create hard negative decision pairs.
3. ToneInvariancePairGenerator: Creates adversarial tone pairs (neutral technical vs
   panicky/hyperbolic tone) guaranteeing 100% label invariance.
4. Spans across 11 core mission-critical decision domains.
"""

from typing import Dict, List, Any, Optional, Tuple, Sequence, Set
import json
import re
import hashlib
import time


CORE_DECISION_DOMAINS = [
    "ops_alert",
    "code_review",
    "access_control",
    "action_routing",
    "session_prune",
    "financial_risk",
    "database_ops",
    "workflow_heal",
    "multimodal_nav",
    "causal_policy",
    "verifier_audit",
]


class ContrastiveMutationEngine:
    """Generates synthetic minimal factual perturbations for contrastive decision training."""

    CONDITION_INVERSIONS = [
        (r"\b(CPU|Memory|Disk)\s*>\s*(\d+)%", r"\1 <= \2%"),
        (r"\b(CPU|Memory|Disk)\s*<=\s*(\d+)%", r"\1 > \2%"),
        (r"\bstatus\s*==\s*['\"]?healthy['\"]?", "status != 'healthy'"),
        (r"\bstatus\s*==\s*['\"]?failed['\"]?", "status == 'healthy'"),
        (r"\berror_count\s*==\s*0\b", "error_count > 0"),
        (r"\bexit_code\s*==\s*0\b", "exit_code != 0"),
        (r"\bis_authenticated\s*=\s*True\b", "is_authenticated = False"),
        (r"\bis_admin\s*=\s*True\b", "is_admin = False"),
    ]

    TEMPORAL_FLIPS = [
        (r"refund processed before item return", "refund requested after item return"),
        (r"token expired after request completed", "token expired before request sent"),
        (r"schema migrated before query execution", "schema migrated after query execution"),
        (r"lock acquired before writing state", "state written before lock acquired"),
    ]

    ENTITY_SUBSTITUTIONS = [
        (r"\buser_role:\s*admin\b", "user_role: guest"),
        (r"\benvironment:\s*production\b", "environment: staging"),
        (r"\bendpoint:\s*/api/v1/delete_user\b", "endpoint: /api/v1/view_profile"),
        (r"\bcluster:\s*prod-us-east-1\b", "cluster: dev-sandbox"),
    ]

    def mutate_fact(self, prompt: str, target_decision: str) -> Tuple[str, str, str]:
        """Applies a minimal factual perturbation to flip the correct decision.

        Returns:
            Tuple of (mutated_prompt, new_target_decision, mutation_type)
        """
        # 1. Condition Inversion
        for pattern, replacement in self.CONDITION_INVERSIONS:
            if re.search(pattern, prompt, re.IGNORECASE):
                mutated = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                inverted_decision = self._invert_decision(target_decision)
                return mutated, inverted_decision, "condition_inversion"

        # 2. Temporal Flip
        for pattern, replacement in self.TEMPORAL_FLIPS:
            if re.search(pattern, prompt, re.IGNORECASE):
                mutated = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                inverted_decision = self._invert_decision(target_decision)
                return mutated, inverted_decision, "temporal_flip"

        # 3. Entity Substitution
        for pattern, replacement in self.ENTITY_SUBSTITUTIONS:
            if re.search(pattern, prompt, re.IGNORECASE):
                mutated = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
                inverted_decision = self._invert_decision(target_decision)
                return mutated, inverted_decision, "entity_substitution"

        # Generic fallback negation
        mutated = prompt.rstrip() + " [CRITICAL: Invariant violated, check assertion failed]"
        return mutated, "ABSTAIN", "generic_negation"

    def _invert_decision(self, decision: str) -> str:
        inversion_map = {
            "PROCEED": "ESCALATE",
            "APPROVE": "REJECT",
            "EXECUTE": "ABSTAIN",
            "ALLOW": "DENY",
            "SCALE_UP": "MAINTAIN",
            "KEEP": "PRUNE",
            "COMPLETE": "INCOMPLETE",
        }
        dec_upper = decision.upper()
        if dec_upper in inversion_map:
            return inversion_map[dec_upper]
        for k, v in inversion_map.items():
            if v == dec_upper:
                return k
        return "ABSTAIN"


class VerifierFeedbackMiner:
    """Mines hard negative decision samples directly from Issue #13 Verifier traces."""

    def mine_from_verifier_log(self, verifier_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Converts an execution failure/loop record from Issue #13 into a hard negative training pair.

        Expected verifier_record schema:
        - prompt / task: original prompt
        - claimed_completion: bool or candidate
        - test_exit_code: int (non-zero indicates failure)
        - error_trace: Optional[str]
        - failed_criteria: Optional[List[str]]
        """
        task = verifier_record.get("task") or verifier_record.get("prompt") or ""
        if not task:
            return None

        exit_code = verifier_record.get("test_exit_code", 0)
        error_trace = verifier_record.get("error_trace", "")
        failed_criteria = verifier_record.get("failed_criteria", [])

        # Hard negative case: Agent claims "COMPLETE" but tests failed or criteria unmet
        if exit_code != 0 or error_trace or failed_criteria:
            state_text = (
                f"Task: {task}\n"
                f"Execution Output: {error_trace[:200] if error_trace else 'Exit code ' + str(exit_code)}\n"
                f"Failed Criteria: {', '.join(failed_criteria) if failed_criteria else 'Assertions failed'}"
            )
            return {
                "domain": "verifier_audit",
                "prompt": state_text,
                "candidates": ["COMPLETE", "REVISE", "ABSTAIN"],
                "target_choice": "REVISE",
                "hard_negative_reason": "False completion blocked by test/assertion failure",
                "is_hard_negative": True,
            }

        return None


class ToneInvariancePairGenerator:
    """Generates subjective tone perturbation pairs with guaranteed decision invariance."""

    EMOTIONAL_NOISE_PATTERNS = [
        "OMG this is an absolute disaster, we are totally doomed! ",
        "Why is everything so broken?! I hate this service: ",
        "URGENT EMERGENCY! Everyone is screaming in Slack! ",
        "Honestly just give up, this code is complete garbage... ",
        "Please please please make this pass or I'll lose my job: ",
    ]

    SARCASTIC_NOISE_PATTERNS = [
        "Brilliant job team, another magnificent masterpiece of bugs: ",
        "Surely nothing could ever go wrong here, right? ",
        "Oh wonderful, another flawless production deployment: ",
    ]

    def generate_tone_invariance_pair(
        self,
        base_prompt: str,
        candidates: List[str],
        ground_truth: str,
        domain: str = "general",
    ) -> List[Dict[str, Any]]:
        """Produces neutral and emotionally perturbed pairs with the exact same ground truth choice."""
        # 1. Neutral pair
        neutral_item = {
            "domain": domain,
            "prompt": base_prompt,
            "candidates": candidates,
            "target_choice": ground_truth,
            "tone": "neutral",
            "pair_id": hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()[:12],
        }

        # 2. Emotional perturbation
        noise = self.EMOTIONAL_NOISE_PATTERNS[hash(base_prompt) % len(self.EMOTIONAL_NOISE_PATTERNS)]
        emotional_prompt = f"{noise}{base_prompt}"

        perturbed_item = {
            "domain": domain,
            "prompt": emotional_prompt,
            "candidates": candidates,
            "target_choice": ground_truth,  # Invariant target
            "tone": "emotional_adversarial",
            "pair_id": neutral_item["pair_id"],
        }

        return [neutral_item, perturbed_item]


class ContrastiveDataPipeline:
    """Orchestrates end-to-end dataset generation across 11 domains with contrastive pairs."""

    def __init__(self):
        self.mutation_engine = ContrastiveMutationEngine()
        self.verifier_miner = VerifierFeedbackMiner()
        self.tone_generator = ToneInvariancePairGenerator()

    def generate_benchmark_corpus(self, samples_per_domain: int = 10) -> List[Dict[str, Any]]:
        """Generates structured benchmark corpus across all 11 domains with balanced contrast."""
        corpus: List[Dict[str, Any]] = []

        domain_templates = {
            "ops_alert": ("Cluster CPU > 95% on worker-node-2", ["SCALE_UP", "MAINTAIN", "DRAIN"], "SCALE_UP"),
            "code_review": ("User input directly concatenated into raw SQL string", ["REJECT", "APPROVE", "COMMENT"], "REJECT"),
            "access_control": ("Requesting /api/v1/delete_user with user_role: admin", ["ALLOW", "DENY", "ESCALATE"], "ALLOW"),
            "action_routing": ("User asked to scrape prices from URL", ["web_fetch", "bash_exec", "file_write"], "web_fetch"),
            "session_prune": ("Turn 15 resolved intermediate grep dump of 500 lines", ["PRUNE", "KEEP", "SUMMARIZE"], "PRUNE"),
            "financial_risk": ("Portfolio delta exposure exceeds limit, drawdown > 5%", ["HEDGE", "HOLD", "LEVERAGE"], "HEDGE"),
            "database_ops": ("Postgres replication lag > 60s, replica read traffic rising", ["ROUTE_PRIMARY", "MAINTAIN", "RESTART_REPLICA"], "ROUTE_PRIMARY"),
            "workflow_heal": ("Unit test failed with AssertionError at line 42", ["SELF_HEAL", "PROCEED", "ABORT"], "SELF_HEAL"),
            "multimodal_nav": ("Modal dialog button 'Confirm Payment' visible at (300, 450)", ["CLICK_CONFIRM", "SCROLL_DOWN", "CLOSE_TAB"], "CLICK_CONFIRM"),
            "causal_policy": ("Exogenous shock applied: treatment group shows significant lift", ["APPLY_TREATMENT", "MAINTAIN_CONTROL", "ABSTAIN"], "APPLY_TREATMENT"),
            "verifier_audit": ("All pytest assertions passed with exit_code == 0", ["COMPLETE", "REVISE", "ABSTAIN"], "COMPLETE"),
        }

        for domain in CORE_DECISION_DOMAINS:
            template = domain_templates.get(domain, ("Default system check", ["PROCEED", "HALT"], "PROCEED"))
            base_text, candidates, default_target = template

            for i in range(samples_per_domain):
                # 1. Base item with neutral tone
                sample_prompt = f"[{domain.upper()} #{i+1}] {base_text}"
                tone_pairs = self.tone_generator.generate_tone_invariance_pair(
                    base_prompt=sample_prompt,
                    candidates=candidates,
                    ground_truth=default_target,
                    domain=domain,
                )
                corpus.extend(tone_pairs)

                # 2. Mutated contrastive item (flipping decision)
                mutated_prompt, flipped_target, m_type = self.mutation_engine.mutate_fact(sample_prompt, default_target)
                if flipped_target not in candidates:
                    candidates_with_flipped = list(candidates) + [flipped_target]
                else:
                    candidates_with_flipped = candidates

                corpus.append({
                    "domain": domain,
                    "prompt": mutated_prompt,
                    "candidates": candidates_with_flipped,
                    "target_choice": flipped_target,
                    "tone": "mutated_contrastive",
                    "mutation_type": m_type,
                    "pair_id": tone_pairs[0]["pair_id"],
                })

        return corpus
