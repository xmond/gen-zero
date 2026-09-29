"""Gen-Zero Structural Causal Model (SCM) & Counterfactual Inference Engine.

Grounds decision intelligence in Judea Pearl's Causality Ladder:
- Level 1 (Association / P(y|x)): Observation and pattern recognition.
- Level 2 (Intervention / P(y|do(x))): Active manipulation and policy execution.
- Level 3 (Counterfactual / P(y_x'|x, y)): Retrospective "What-If" under locked historical noise.

Enables:
1. Abduction-Intervention-Prediction 3-step counterfactual attribution.
2. Disentanglement of Action-Caused Failures (Decision Flaws) vs Exogenous Shocks.
3. Computation of Individual Treatment Effect (ITE) across alternative choices.
4. Identification of root-cause branching steps in irreversible trajectories.
"""

import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable


@dataclass
class CausalAttribution:
    """Attribution metadata for a single step decision."""
    step_idx: int
    factual_action: str
    factual_return: float
    best_counterfactual_action: Optional[str]
    best_counterfactual_return: float
    individual_treatment_effect: float       # ITE = Y_CF(do(a*)) - Y_Fact(a)
    is_decision_culprit: bool                # True if ITE > threshold (action was the flaw)
    is_exogenous_shock: bool                 # True if all actions fail due to unavoidable U
    abduced_noise_norm: float                # Magnitude of external disturbance ||U||
    causal_explanation: str


@dataclass
class CounterfactualResult:
    """Result of a counterfactual intervention query."""
    factual_state: Any
    factual_action: str
    factual_next_state: Any
    abduced_noise: Dict[str, Any]
    interventions: Dict[str, Dict[str, Any]]  # action -> {cf_next_state, cf_return, ite}
    best_action: str
    max_ite: float
    is_action_culprit: bool


class StructuralCausalModel:
    """Structural Causal Model with explicit endogenous-exogenous disentanglement.
    
    SCM Formula:
        S_{t+1} := f_S(S_t, do(A_t)) + U_t
    where:
        f_S: Endogenous deterministic transition mechanism.
        U_t: Exogenous invariant environment noise / disturbance.
    """

    def __init__(self, causal_noise_dim: int = 16):
        self.causal_noise_dim = causal_noise_dim

    def endogenous_transition(self, state: Any, action: str) -> Tuple[Any, float]:
        """Calculates deterministic nominal next state f_S(s, a) and nominal step reward."""
        if isinstance(state, dict):
            next_s = copy.deepcopy(state)
            r = 0.0
            # Grid / Coordinate based dynamics
            if "head" in state and isinstance(state["head"], (list, tuple)):
                hx, hy = state["head"][0], state["head"][1]
                delta = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}.get(action, (0, 0))
                next_s["head"] = [hx + delta[0], hy + delta[1]]
                if "food" in state and list(next_s["head"]) == list(state["food"]):
                    r = 1.0
                return next_s, r
            elif "pos" in state and isinstance(state["pos"], (list, tuple)):
                px, py = state["pos"][0], state["pos"][1]
                delta = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
                         "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(action, (0, 0))
                next_s["pos"] = [px + delta[0], py + delta[1]]
                return next_s, r
            elif "weights" in state:
                # Continuous portfolio
                return next_s, r
            # Generic structured state
            next_s["last_action"] = action
            return next_s, r
        elif isinstance(state, (list, tuple)):
            # Continuous numerical vector state
            next_s = list(state)
            return next_s, 0.0
        return state, 0.0

    def abduce_exogenous_noise(self, state: Any, action: str, observed_next_state: Any) -> Dict[str, Any]:
        """Step 1: Abduction.
        
        Calculates U_t = S_{t+1} - f_S(S_t, A_t) to reconstruct the exact historical
        realization of external disturbance (e.g. friction, slip, sudden price jump).
        """
        nominal_next, _ = self.endogenous_transition(state, action)
        noise: Dict[str, Any] = {"vector_diff": [], "norm": 0.0, "slip_occurred": False}

        if isinstance(observed_next_state, dict) and isinstance(nominal_next, dict):
            # Positional / coordinate drift
            for key in ("head", "pos"):
                if key in observed_next_state and key in nominal_next:
                    obs_p = observed_next_state[key]
                    nom_p = nominal_next[key]
                    dx = obs_p[0] - nom_p[0]
                    dy = obs_p[1] - nom_p[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    noise["coord_drift"] = (dx, dy)
                    noise["norm"] = dist
                    noise["slip_occurred"] = (dist > 1e-4)
                    break
        elif isinstance(observed_next_state, (list, tuple)) and isinstance(nominal_next, (list, tuple)):
            diffs = [obs - nom for obs, nom in zip(observed_next_state, nominal_next)]
            norm = math.sqrt(sum(d * d for d in diffs))
            noise["vector_diff"] = diffs
            noise["norm"] = norm

        return noise

    def intervene_and_predict(
        self,
        state: Any,
        counterfactual_action: str,
        abduced_noise: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Step 2 & 3: Intervention & Prediction.
        
        Evaluates S_{t+1}^{CF} = f_S(S_t, do(A_t = a*)) + U_t
        Clamps the abduced exogenous noise U_t to evaluate what would have occurred
        under the exact same historical external conditions.
        """
        nom_cf_next, nom_r = self.endogenous_transition(state, counterfactual_action)
        cf_next = copy.deepcopy(nom_cf_next)

        # Re-apply locked historical noise U_t
        if isinstance(cf_next, dict) and "coord_drift" in abduced_noise:
            dx, dy = abduced_noise["coord_drift"]
            for key in ("head", "pos"):
                if key in cf_next:
                    cf_next[key][0] += dx
                    cf_next[key][1] += dy
                    break
        elif isinstance(cf_next, list) and "vector_diff" in abduced_noise and abduced_noise["vector_diff"]:
            for i, diff in enumerate(abduced_noise["vector_diff"]):
                if i < len(cf_next):
                    cf_next[i] += diff

        return cf_next, nom_r


class CounterfactualEngine:
    """Engine for Pearlian Counterfactual Reasoning & Attribution."""

    def __init__(
        self,
        scm: Optional[StructuralCausalModel] = None,
        causal_threshold: float = 0.25,
        shock_noise_threshold: float = 1.2
    ):
        self.scm = scm or StructuralCausalModel()
        self.causal_threshold = causal_threshold
        self.shock_noise_threshold = shock_noise_threshold

    def compute_counterfactuals(
        self,
        state: Any,
        factual_action: str,
        observed_next_state: Any,
        candidate_actions: List[str],
        value_evaluator: Callable[[Any], float],
        factual_return: float
    ) -> CounterfactualResult:
        """Runs full Abduction-Intervention-Prediction cycle for a single step."""
        # 1. Abduction: Infer U_t
        abduced_noise = self.scm.abduce_exogenous_noise(state, factual_action, observed_next_state)

        interventions = {}
        best_act = factual_action
        max_cf_return = factual_return
        max_ite = 0.0

        # 2. Intervention: do(A = a*) for all alternative actions
        for a_cf in candidate_actions:
            if a_cf == factual_action:
                interventions[a_cf] = {
                    "next_state": observed_next_state,
                    "return": factual_return,
                    "ite": 0.0
                }
                continue

            cf_next, step_r = self.scm.intervene_and_predict(state, a_cf, abduced_noise)
            # Estimate projected downstream return of counterfactual state
            cf_val = value_evaluator(cf_next) + step_r
            ite = cf_val - factual_return

            interventions[a_cf] = {
                "next_state": cf_next,
                "return": cf_val,
                "ite": ite
            }

            if ite > max_ite:
                max_ite = ite
                best_act = a_cf
                max_cf_return = cf_val

        is_action_culprit = (max_ite >= self.causal_threshold)

        return CounterfactualResult(
            factual_state=state,
            factual_action=factual_action,
            factual_next_state=observed_next_state,
            abduced_noise=abduced_noise,
            interventions=interventions,
            best_action=best_act,
            max_ite=max_ite,
            is_action_culprit=is_action_culprit
        )

    def attribute_trajectory_failures(
        self,
        trajectory: List[Dict[str, Any]],
        final_outcome: str,
        final_score: float,
        value_evaluator: Optional[Callable[[Any], float]] = None
    ) -> List[CausalAttribution]:
        """Performs retrospective causal attribution across an entire trajectory.
        
        Separates Decision Errors from Environmental Shocks, pinpoints the root-cause
        branching step with maximum Individual Treatment Effect (ITE).
        """
        attributions: List[CausalAttribution] = []
        n_steps = len(trajectory)
        is_failure = final_outcome in ("collision", "trapped", "fail", "timeout")

        # Fallback evaluator using PRM or simple heuristics if none provided
        if value_evaluator is None:
            def default_eval(s: Any) -> float:
                if isinstance(s, dict):
                    # Distance to target/food or reachability
                    if "head" in s and "food" in s:
                        hx, hy = s["head"]
                        fx, fy = s["food"]
                        d = abs(hx - fx) + abs(hy - fy)
                        return -0.05 * d
                return 0.0
            value_evaluator = default_eval

        # Final factual outcome scalar
        y_fact = 1.0 if not is_failure else -1.0

        for idx, step in enumerate(trajectory):
            state = step.get("state", {})
            action = step.get("action", "unknown")
            next_state = step.get("next_state") or state
            candidates = step.get("candidates", [action])

            # Abduce external disturbance
            noise = self.scm.abduce_exogenous_noise(state, action, next_state)
            noise_norm = noise.get("norm", 0.0)

            # Check if this failure was due to an overwhelming uncontrollable exogenous shock
            is_shock = (noise_norm >= self.shock_noise_threshold)

            cf_res = self.compute_counterfactuals(
                state=state,
                factual_action=action,
                observed_next_state=next_state,
                candidate_actions=candidates,
                value_evaluator=value_evaluator,
                factual_return=y_fact
            )

            is_culprit = cf_res.is_action_culprit and not is_shock

            if is_culprit:
                explanation = (
                    f"Causal Decision Error: Action '{action}' led to failure, but counterfactual "
                    f"intervention do('{cf_res.best_action}') under identical conditions would yield ITE = +{cf_res.max_ite:.3f}."
                )
            elif is_shock:
                explanation = (
                    f"Exogenous Shock: External noise magnitude ||U||={noise_norm:.2f} exceeded controllable threshold. "
                    f"Failure was exogenously confounded."
                )
            else:
                explanation = "Action was conditionally aligned or no superior alternative counterfactual was found."

            attributions.append(CausalAttribution(
                step_idx=idx,
                factual_action=action,
                factual_return=y_fact,
                best_counterfactual_action=cf_res.best_action if is_culprit else None,
                best_counterfactual_return=cf_res.interventions.get(cf_res.best_action, {}).get("return", y_fact),
                individual_treatment_effect=cf_res.max_ite,
                is_decision_culprit=is_culprit,
                is_exogenous_shock=is_shock,
                abduced_noise_norm=noise_norm,
                causal_explanation=explanation
            ))

        return attributions

    def find_root_cause_step(self, attributions: List[CausalAttribution]) -> Optional[CausalAttribution]:
        """Finds the earliest high-ITE root-cause decision failure in the trajectory."""
        culprits = [attr for attr in attributions if attr.is_decision_culprit]
        if not culprits:
            return None
        # Sort by earliest step with substantial ITE
        culprits_sorted = sorted(culprits, key=lambda a: (-a.individual_treatment_effect, a.step_idx))
        return culprits_sorted[0]
