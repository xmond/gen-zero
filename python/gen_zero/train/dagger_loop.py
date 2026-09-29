"""DAgger (Dataset Aggregation) Expert Relabeling Loop & Covariate Shift Recovery.

Implements Milestone 3 of Issue #19:
1. Student Rollout: System 1 exploratory rollout with probability p_student in [0.70, 0.95].
2. Drift Detection: Identifies off-policy drifting states where:
   - Closed-form confidence C(s) < tau_conf (default: 0.60), OR
   - Attention entropy H_norm > tau_entropy (default: 0.85), OR
   - Student action diverged from expert optimal action.
3. Expert Relabeling: Calls System 2 / CP-SAT / Oracle expert to relabel drifting states
   with optimal action distribution y* = Expert(s, A).
4. Priority Injection: Injects recovery samples into StabilityReplayBuffer to eliminate
   compounding error cascades without catastrophic forgetting.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import dataclasses
import uuid
import time
import math


@dataclasses.dataclass
class DAggerState:
    """State snapshot captured during student rollout."""
    state: Any
    candidate_ids: List[str]
    student_action: Optional[str] = None
    student_confidence: float = 1.0
    student_entropy: float = 0.0
    step_idx: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DAggerRelabeledSample:
    """Sample relabeled by expert for stabilizing off-policy drift."""
    id: str
    state: Any
    candidate_ids: List[str]
    gold_action: str
    pi_target: Dict[str, float]
    priority_weight: float = 1.0
    expert_source: str = "cpsat_oracle"
    is_dagger_relabeled: bool = True
    student_action: Optional[str] = None
    student_confidence: float = 1.0
    student_entropy: float = 0.0
    value_target: float = 1.0
    created_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "choice",
            "state": self.state,
            "candidate_ids": self.candidate_ids,
            "pi_target": self.pi_target,
            "soft_target": self.pi_target,
            "gold_action": self.gold_action,
            "value_target": self.value_target,
            "priority_weight": self.priority_weight,
            "expert_source": self.expert_source,
            "is_dagger_relabeled": True,
            "is_hard_sample": True,  # High priority in stability replay buffer
            "student_action": self.student_action,
            "student_confidence": self.student_confidence,
            "student_entropy": self.student_entropy,
        }


class DAggerExpertRelabeler:
    """Online expert relabeler identifying off-policy drift and relabeling optimal recoveries."""

    def __init__(
        self,
        expert_oracle_fn: Callable[[Any, List[str]], Tuple[str, Dict[str, float]]],
        conf_threshold: float = 0.60,
        entropy_threshold: float = 0.85,
        base_priority: float = 2.0,
        expert_source: str = "cpsat_oracle"
    ):
        """
        Args:
            expert_oracle_fn: Callable(state, candidates) -> (best_action, pi_distribution)
            conf_threshold: States with confidence below this threshold are flagged as drifting.
            entropy_threshold: States with attention entropy above this threshold are flagged.
            base_priority: Sampling priority assigned to relabeled recovery samples.
            expert_source: Name or identifier of expert backend (e.g. "cpsat", "mcts", "oracle").
        """
        self.expert_oracle_fn = expert_oracle_fn
        self.conf_threshold = conf_threshold
        self.entropy_threshold = entropy_threshold
        self.base_priority = base_priority
        self.expert_source = expert_source

    def is_drift_state(
        self,
        confidence: float,
        entropy: float,
        student_action: Optional[str] = None,
        expert_action: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Evaluates whether the state has drifted into an off-policy or unsafe zone."""
        reasons = []
        if confidence < self.conf_threshold:
            reasons.append(f"low_conf({confidence:.2f}<{self.conf_threshold:.2f})")
        if entropy > self.entropy_threshold:
            reasons.append(f"high_entropy({entropy:.2f}>{self.entropy_threshold:.2f})")
        if student_action is not None and expert_action is not None and student_action != expert_action:
            reasons.append(f"suboptimal_action({student_action}!={expert_action})")

        is_drift = len(reasons) > 0
        return is_drift, "; ".join(reasons) if reasons else "on_policy_stable"

    def relabel_state(
        self,
        state: Any,
        candidates: List[str],
        student_action: Optional[str] = None,
        student_confidence: float = 1.0,
        student_entropy: float = 0.0,
        force_relabel: bool = False,
    ) -> Optional[DAggerRelabeledSample]:
        """Queries expert oracle and generates a relabeled recovery sample if drift is detected."""
        if not candidates:
            return None

        # Query expert oracle
        gold_action, pi_target = self.expert_oracle_fn(state, candidates)
        if not pi_target:
            # Fallback uniform smoothed target around gold_action
            eps = 0.05 / max(1, len(candidates) - 1)
            pi_target = {c: (1.0 - 0.05 if c == gold_action else eps) for c in candidates}

        is_drift, reason = self.is_drift_state(
            confidence=student_confidence,
            entropy=student_entropy,
            student_action=student_action,
            expert_action=gold_action,
        )

        if not is_drift and not force_relabel:
            return None

        # Priority increases with drift severity (divergence + diffuse entropy + low confidence)
        severity = 0.0
        if student_action is not None and student_action != gold_action:
            severity += 1.0
        if student_entropy > self.entropy_threshold:
            severity += (student_entropy - self.entropy_threshold) / max(1e-4, 1.0 - self.entropy_threshold)
        if student_confidence < self.conf_threshold:
            severity += (self.conf_threshold - student_confidence) / max(1e-4, self.conf_threshold)

        sample_id = f"dagger_{uuid.uuid4().hex[:10]}"
        return DAggerRelabeledSample(
            id=sample_id,
            state=state,
            candidate_ids=candidates,
            gold_action=gold_action,
            pi_target=pi_target,
            priority_weight=round(self.base_priority + severity, 3),
            expert_source=self.expert_source,
            is_dagger_relabeled=True,
            student_action=student_action,
            student_confidence=student_confidence,
            student_entropy=student_entropy,
        )

    def relabel_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        only_drifting: bool = True,
    ) -> List[DAggerRelabeledSample]:
        """Relabels an entire rollout trajectory, filtering for off-policy drift states."""
        relabeled: List[DAggerRelabeledSample] = []
        for step in trajectory:
            state = step.get("state")
            cands = step.get("candidate_ids") or step.get("candidates") or []
            act = step.get("student_action") or step.get("action")
            conf = float(step.get("student_confidence") or step.get("confidence") or 1.0)
            ent = float(step.get("student_entropy") or step.get("attention_entropy") or 0.0)

            sample = self.relabel_state(
                state=state,
                candidates=cands,
                student_action=act,
                student_confidence=conf,
                student_entropy=ent,
                force_relabel=not only_drifting,
            )
            if sample is not None:
                relabeled.append(sample)

        return relabeled


class DAggerCurriculumController:
    """Manages iterative DAgger curriculum, exploration decay, and relabeling yield."""

    def __init__(
        self,
        p_student_base: float = 0.85,
        decay_factor: float = 0.95,
        p_student_min: float = 0.50,
    ):
        self.p_student_base = p_student_base
        self.decay_factor = decay_factor
        self.p_student_min = p_student_min
        self.iteration: int = 0
        self.stats: Dict[str, Any] = {
            "total_rollouts": 0,
            "total_drift_states": 0,
            "total_relabeled": 0,
            "iterations_completed": 0,
        }

    def get_current_student_prob(self) -> float:
        """Computes beta_k student rollout probability for current curriculum iteration."""
        prob = self.p_student_base * (self.decay_factor ** self.iteration)
        return max(self.p_student_min, round(prob, 4))

    def advance_iteration(self, num_rollouts: int, num_drifts: int, num_relabeled: int) -> Dict[str, Any]:
        """Records iteration telemetry and advances the curriculum round."""
        self.stats["total_rollouts"] += num_rollouts
        self.stats["total_drift_states"] += num_drifts
        self.stats["total_relabeled"] += num_relabeled
        self.stats["iterations_completed"] += 1
        self.iteration += 1

        return {
            "iteration": self.iteration,
            "current_p_student": self.get_current_student_prob(),
            "stats": dict(self.stats),
        }
