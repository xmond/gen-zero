"""Gen-Zero Layer 3: Counterfactual Synthetic Augmentation Engine.

Implements Pearl's "Generative What-If" twin sample generation:
1. Locks abduced historical exogenous noise U_t from real rollouts.
2. Intervenes on alternative actions do(A_t = a*).
3. Verifies physical and topological viability via Process Reward Model (PRM).
4. Synthesizes high-fidelity training targets (s_t, pi_CF, V_CF) without running extra real environments.
5. Injects verified synthetic twins into Stability Replay Buffer, doubling sample efficiency.
"""

import copy
from typing import Dict, List, Any, Optional, Tuple

from .counterfactual_engine import StructuralCausalModel
from ..model.prm import ProcessRewardModel
from ..train.replay_buffer import StabilityReplayBuffer


class CounterfactualSyntheticGenerator:
    """Generates PRM-verified counterfactual twin training samples."""

    def __init__(
        self,
        scm: Optional[StructuralCausalModel] = None,
        prm: Optional[ProcessRewardModel] = None,
        min_viability_threshold: float = 0.35
    ):
        self.scm = scm or StructuralCausalModel()
        self.prm = prm or ProcessRewardModel()
        self.min_viability_threshold = min_viability_threshold

    def generate_twins_from_step(
        self,
        state: Any,
        factual_action: str,
        observed_next_state: Any,
        candidates: List[str],
        factual_value: float,
        step_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """Generates counterfactual twin samples for a single decision step."""
        twins: List[Dict[str, Any]] = []

        # 1. Abduce historical exogenous disturbance
        noise = self.scm.abduce_exogenous_noise(state, factual_action, observed_next_state)

        # 2. Iterate through alternative counterfactual actions
        for a_cf in candidates:
            if a_cf == factual_action:
                continue

            cf_next_state, step_r = self.scm.intervene_and_predict(state, a_cf, noise)

            # 3. Verify physical and topological viability using PRM
            is_done = False
            prm_res = self.prm.verify_step(
                parent_state=state,
                action=a_cf,
                next_state=cf_next_state,
                is_done=is_done,
                step_reward=step_r
            )
            should_prune = prm_res.get("should_prune", False) if isinstance(prm_res, dict) else getattr(prm_res, "should_prune", False)
            viability_score = float(prm_res.get("viability", 0.5) if isinstance(prm_res, dict) else getattr(prm_res, "viability_score", 0.5))

            # Check if this counterfactual is safe and viable
            if not should_prune and viability_score >= self.min_viability_threshold:
                # 4. Synthesize soft policy target biased towards this viable alternative
                soft_tgt = {}
                base_prob = 1.0 / max(1, len(candidates))
                boost = min(0.6, viability_score * 0.5)

                for c in candidates:
                    if c == a_cf:
                        soft_tgt[c] = base_prob + boost
                    else:
                        soft_tgt[c] = max(0.02, (base_prob - boost / max(1, len(candidates) - 1)))

                tot_p = sum(soft_tgt.values())
                soft_tgt = {c: p / tot_p for c, p in soft_tgt.items()}

                # Synthesize counterfactual value target
                cf_value = min(1.0, max(-1.0, factual_value + viability_score * 0.4 + step_r))

                twins.append({
                    "id": f"syn_cf:{step_idx}:{a_cf}",
                    "type": "choice",
                    "state": state,
                    "candidate_ids": candidates,
                    "soft_target": soft_tgt,
                    "value_target": round(cf_value, 4),
                    "is_hard_sample": False,
                    "is_synthetic": True,
                    "cf_action": a_cf,
                    "viability_score": round(viability_score, 4),
                    "environment_id": "env_synthetic_counterfactual"
                })

        return twins

    def augment_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        final_outcome: str,
        final_score: float
    ) -> List[Dict[str, Any]]:
        """Augments an entire trajectory with PRM-verified counterfactual twin samples."""
        all_twins = []
        is_failure = final_outcome in ("collision", "trapped", "fail", "timeout")
        base_v = 1.0 if not is_failure else -1.0

        for i, step in enumerate(trajectory):
            s = step.get("state", {})
            a = step.get("action", "unknown")
            s_next = step.get("next_state") or s
            cands = step.get("candidates", [a])

            twins = self.generate_twins_from_step(
                state=s,
                factual_action=a,
                observed_next_state=s_next,
                candidates=cands,
                factual_value=base_v,
                step_idx=i
            )
            all_twins.extend(twins)

        return all_twins

    def augment_and_inject(
        self,
        replay_buffer: StabilityReplayBuffer,
        trajectory: List[Dict[str, Any]],
        final_outcome: str,
        final_score: float
    ) -> Dict[str, Any]:
        """Synthesizes twin samples and directly injects them into the replay buffer."""
        twins = self.augment_trajectory(trajectory, final_outcome, final_score)
        if twins:
            replay_buffer.add_synthetic_samples(twins)

        factual_steps = len(trajectory)
        synthetic_steps = len(twins)
        efficiency_multiplier = round((factual_steps + synthetic_steps) / max(1, factual_steps), 2)

        return {
            "factual_steps": factual_steps,
            "synthetic_twins_generated": synthetic_steps,
            "sample_efficiency_multiplier": efficiency_multiplier,
            "buffer_total": len(replay_buffer)
        }


from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import random


class CausalInterventionType(str, Enum):
    RELEVANT = "relevant"      # Decisive fact intervened -> decision MUST flip
    IRRELEVANT = "irrelevant"  # Cosmetic/peripheral context intervened -> decision MUST NOT flip


@dataclass
class CausalMinimalPair:
    pair_id: str
    intervention_type: CausalInterventionType
    decisive_key: str
    base_sample: Dict[str, Any]
    counterfactual_sample: Dict[str, Any]
    expected_flip: bool
    distractors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intervention_type"] = self.intervention_type.value
        return d


class CausalMinimalPairGenerator:
    """Synthesizes minimal policy pairs for causal verification and anti-shortcut training.

    RFC Requirements:
    1. Relevant Intervention: 微扰关键因果变量 (阈值/状态)，强制要求决策反转。
    2. Irrelevant Intervention: 扰动修饰词/冗余状态，强制要求决策保持不变。
    3. 解耦“以上皆非”捷径: 动态平衡 ABSTAIN / 干扰项先验概率。
    """

    IRRELEVANT_MODIFIERS = [
        "quick", "standard", "legacy", "verified", "background",
        "cached", "high-priority", "low-priority", "batch_mode"
    ]

    NEGATIVE_DISTRACTORS = [
        "ABSTAIN", "NONE_OF_THE_ABOVE", "FALLBACK_RETRY", "DEFER_TO_HUMAN"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_minimal_pair(
        self,
        base_state: Dict[str, Any],
        candidates: List[str],
        optimal_action: str,
        intervention_type: CausalInterventionType,
        decisive_key: str = "threshold",
        pair_id: Optional[str] = None
    ) -> CausalMinimalPair:
        """Generates a verified causal minimal pair."""
        pid = pair_id or f"cmp_{self.rng.randint(10000, 99999)}"
        base_sample = {
            "id": f"{pid}_base",
            "state": copy.deepcopy(base_state),
            "candidates": list(candidates),
            "optimal_action": optimal_action
        }

        cf_state = copy.deepcopy(base_state)
        cf_candidates = list(candidates)

        if intervention_type == CausalInterventionType.RELEVANT:
            # Modify the decisive causal variable -> Decision must flip
            expected_flip = True
            current_val = cf_state.get(decisive_key, 10)
            if isinstance(current_val, bool):
                cf_state[decisive_key] = not current_val
            elif isinstance(current_val, (int, float)):
                # Flip threshold boundary
                cf_state[decisive_key] = current_val * 2 if current_val > 0 else 10
            elif isinstance(current_val, str):
                cf_state[decisive_key] = f"inverted_{current_val}"
            else:
                cf_state[decisive_key] = "flipped_condition"

            # Optimal action must flip to another valid candidate
            alternative_cands = [c for c in candidates if c != optimal_action]
            cf_optimal_action = alternative_cands[0] if alternative_cands else "ABSTAIN"
            if cf_optimal_action not in cf_candidates:
                cf_candidates.append(cf_optimal_action)

        else:
            # Irrelevant intervention -> Decision must remain strictly invariant
            expected_flip = False
            cf_optimal_action = optimal_action
            # Perturb cosmetics, styling, comments, or unimportant metadata
            noise_key = f"_meta_noise_{self.rng.randint(1, 10)}"
            cf_state[noise_key] = self.rng.choice(self.IRRELEVANT_MODIFIERS)
            if "description" in cf_state:
                cf_state["description"] = f"{cf_state['description']} (tag: {self.rng.choice(self.IRRELEVANT_MODIFIERS)})"
            else:
                cf_state["_style"] = self.rng.choice(self.IRRELEVANT_MODIFIERS)

        cf_sample = {
            "id": f"{pid}_cf",
            "state": cf_state,
            "candidates": cf_candidates,
            "optimal_action": cf_optimal_action
        }

        return CausalMinimalPair(
            pair_id=pid,
            intervention_type=intervention_type,
            decisive_key=decisive_key,
            base_sample=base_sample,
            counterfactual_sample=cf_sample,
            expected_flip=expected_flip,
            distractors=[]
        )

    def inject_balanced_distractors(
        self,
        pair: CausalMinimalPair,
        distractor: str = "ABSTAIN",
        positive_prior: float = 0.5
    ) -> CausalMinimalPair:
        """Injects distractors into both base and counterfactual to prevent shortcut learning."""
        pair_copy = copy.deepcopy(pair)
        for sample in (pair_copy.base_sample, pair_copy.counterfactual_sample):
            if distractor not in sample["candidates"]:
                sample["candidates"].append(distractor)

        # In positive_prior cases, make distractor the valid choice to prevent shortcut learning
        if self.rng.random() < positive_prior:
            if pair_copy.intervention_type == CausalInterventionType.RELEVANT:
                # Decisive flip leads to unresolvable state -> ABSTAIN is optimal
                pair_copy.counterfactual_sample["optimal_action"] = distractor
            else:
                # Context requires abstention in both base and counterfactual
                pair_copy.base_sample["optimal_action"] = distractor
                pair_copy.counterfactual_sample["optimal_action"] = distractor

        pair_copy.distractors.append(distractor)
        return pair_copy

    def generate_balanced_suite(
        self,
        num_pairs: int = 10,
        include_distractors: bool = True
    ) -> List[CausalMinimalPair]:
        """Synthesizes a 50/50 balanced suite of Relevant and Irrelevant minimal pairs."""
        suite: List[CausalMinimalPair] = []
        domain_actions = ["NAVIGATE_TARGET", "SUBMIT_FORM", "DISMISS_POPUP", "SCROLL_DOWN", "CLICK_CHECKBOX"]

        for i in range(num_pairs):
            is_relevant = (i % 2 == 0)
            itype = CausalInterventionType.RELEVANT if is_relevant else CausalInterventionType.IRRELEVANT

            base_state = {
                "user_intent": "checkout_cart",
                "cart_total": 45.0,
                "is_authenticated": True,
                "target_element_visible": True,
                "security_clearance": 2
            }
            cands = list(domain_actions[:4])
            opt_act = domain_actions[0]

            decisive_keys = ["cart_total", "is_authenticated", "target_element_visible", "security_clearance"]
            chosen_key = decisive_keys[i % len(decisive_keys)]

            pair = self.create_minimal_pair(
                base_state=base_state,
                candidates=cands,
                optimal_action=opt_act,
                intervention_type=itype,
                decisive_key=chosen_key,
                pair_id=f"pair_{i:03d}"
            )

            if include_distractors:
                pair = self.inject_balanced_distractors(pair, distractor="ABSTAIN")

            suite.append(pair)

        return suite

    def export_dataset(self, suite: List[CausalMinimalPair], filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """Serializes minimal pairs to list of dictionaries or saves to JSON."""
        data = [p.to_dict() for p in suite]
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return data
