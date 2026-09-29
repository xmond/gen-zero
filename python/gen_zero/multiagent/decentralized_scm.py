"""Gen-Zero Layer 2 Multi-Agent Expert: Decentralized Structural Causal Model (D-SCM) & Asymmetric Bluffing Detector.

Implements Phase 4 Step 1:
1. Multi-Agent Structural Causal Modeling:
   S_{t+1} := f_S(S_t, do(A_{i,t}), do(A_{-i,t})) + U_t
   Disentangles main agent intervention A_i, opponent strategic action A_{-i}, and exogenous noise U_t.
2. Noise Abduction:
   Explicitly isolates environment shocks U_t from strategic moves.
3. Asymmetric Bluffing & Intent Classifier:
   Identifies whether opponent deviations stem from exogenous shocks, rational value bets, passive traps,
   or strategic bluffing / feints under stable conditions.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable

from ..planner.engines.cfr_nash_engine import OpponentBeliefTracker


class IntentType(Enum):
    """Categorization of opponent multi-agent strategic intent."""
    NORMAL_EQUILIBRIUM = "normal_equilibrium"        # Rational Nash baseline play
    STRATEGIC_BLUFF = "strategic_bluff"              # Aggressive commitment with weak backing under stable conditions
    VALUE_BET = "value_bet"                          # Genuine aggressive move backed by high underlying strength
    TRAP_PASSIVITY = "trap_passivity"                # Deceptive passivity masking strong underlying strength
    EXOGENOUS_CONFUSION = "exogenous_confusion"      # State deviation caused by external environment shock U_t


@dataclass
class AgentIntent:
    """Detailed intent analysis report for an opponent action."""
    opponent_id: str
    observed_action: str
    intent_type: IntentType
    bluff_probability: float                         # P(Bluff | A_{-i}, S_t, U_t) in [0.0, 1.0]
    abduced_shock_norm: float                        # ||U_t||
    is_environment_stable: bool                      # True if ||U_t|| < tau_shock
    counterfactual_payoff_gap: float                 # Payoff delta between aggressive action vs rational baseline
    recommended_counter_action: str                  # Counter-strategy: e.g. CALL_BLUFF, FOLD, PUNISH_FEINT
    confidence: float                                # Statistical confidence in [0.0, 1.0]
    rationale: str


@dataclass
class MultiAgentState:
    """State representation in a multi-agent environment."""
    features: Dict[str, float] = field(default_factory=dict)
    pot: float = 10.0
    pressure: float = 0.0
    volatility: float = 0.1
    history: List[str] = field(default_factory=list)


@dataclass
class MultiAgentTransition:
    """State transition record in a multi-agent environment."""
    state: Dict[str, float]
    ego_action: str
    opponent_action: str
    next_state: Dict[str, float]
    ego_reward: float
    opponent_reward: float
    info: Dict[str, Any] = field(default_factory=dict)


class DecentralizedSCM:
    r"""Decentralized Structural Causal Model for multi-agent game-theoretic systems.
    
    Equations:
        \hat{S}_{t+1} = f_S(S_t, A_i, A_{-i})
        U_t = S_{t+1} - \hat{S}_{t+1}
    """

    def __init__(
        self,
        shock_threshold: float = 0.25,
        transition_fn: Optional[Callable[[Dict[str, float], str, str], Dict[str, float]]] = None,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        self.shock_threshold = shock_threshold
        self.transition_fn = transition_fn or self._default_transition_model
        self.feature_weights = feature_weights or {
            "pot": 0.02,
            "pressure": 1.0,
            "volatility": 1.0
        }

    def _default_transition_model(
        self,
        state: Dict[str, float],
        ego_action: str,
        opponent_action: str
    ) -> Dict[str, float]:
        """Default first-order linear-bilinear multi-agent state prediction model."""
        next_state = dict(state)
        
        # Pot / stakes dynamics
        pot = state.get("pot", 10.0)
        pot_delta = 0.0
        if ego_action in ("raise", "bet", "aggressive_bid"):
            pot_delta += 10.0
        elif ego_action in ("call", "moderate_bid"):
            pot_delta += 5.0
            
        if opponent_action in ("raise", "bet", "aggressive_bid"):
            pot_delta += 10.0
        elif opponent_action in ("call", "moderate_bid"):
            pot_delta += 5.0
            
        next_state["pot"] = pot + pot_delta
        
        # Leverage / pressure dynamics
        pressure = state.get("pressure", 0.0)
        if opponent_action in ("raise", "bet"):
            pressure += 0.3
        elif opponent_action in ("check", "fold"):
            pressure = max(0.0, pressure - 0.2)
        next_state["pressure"] = min(1.0, max(0.0, pressure))
        
        # Resource liquidity / market volatility
        vol = state.get("volatility", 0.1)
        next_state["volatility"] = vol
        
        return next_state

    def abduce_noise(
        self,
        state: Dict[str, float],
        ego_action: str,
        opponent_action: str,
        actual_next_state: Dict[str, float]
    ) -> Tuple[Dict[str, float], float, bool]:
        """Abduces exogenous environment noise vector U_t and evaluates if shock occurred.
        
        Returns:
            (u_vector, shock_norm, is_shock)
        """
        predicted = self.transition_fn(state, ego_action, opponent_action)
        u_vector = {}
        sum_sq = 0.0
        
        for k in actual_next_state:
            pred_v = predicted.get(k, actual_next_state[k])
            actual_v = actual_next_state[k]
            diff = actual_v - pred_v
            u_vector[k] = diff
            w = self.feature_weights.get(k, 1.0)
            sum_sq += (diff * w) ** 2
            
        shock_norm = math.sqrt(sum_sq)
        is_shock = shock_norm >= self.shock_threshold
        return u_vector, shock_norm, is_shock

    def counterfactual_simulation(
        self,
        state: Dict[str, float],
        do_ego_action: str,
        do_opponent_action: str,
        abduced_u: Dict[str, float]
    ) -> Dict[str, float]:
        """Simulates counterfactual next state under locked historical exogenous noise U_t."""
        base_pred = self.transition_fn(state, do_ego_action, do_opponent_action)
        cf_next_state = {}
        for k, v in base_pred.items():
            cf_next_state[k] = v + abduced_u.get(k, 0.0)
        return cf_next_state


class AsymmetricBluffDetector:
    """Asymmetric Bluffing & Strategic Intent Classifier.
    
    Disentangles strategic deception from environmental volatility using D-SCM:
    P(Bluff | A_{-i}, S_t, U_t) = sigma( alpha * (Aggressiveness - Strength_Est) * w_stable + beta * KL(pi_hist || pi_nash) )
    """

    def __init__(
        self,
        d_scm: Optional[DecentralizedSCM] = None,
        belief_tracker: Optional[OpponentBeliefTracker] = None,
        alpha_pot_sensitivity: float = 3.2,
        beta_kl_sensitivity: float = 1.8,
        bluff_threshold: float = 0.58
    ):
        self.scm = d_scm or DecentralizedSCM()
        self.tracker = belief_tracker or OpponentBeliefTracker(prior_weight=2.0)
        self.alpha = alpha_pot_sensitivity
        self.beta = beta_kl_sensitivity
        self.bluff_threshold = bluff_threshold

    def _estimate_action_aggressiveness(self, action: str) -> float:
        """Assigns an aggressiveness quotient in [0.0, 1.0] to standard game actions."""
        agg_map = {
            "all_in": 1.0,
            "overbet": 0.90,
            "raise": 0.75,
            "aggressive_bid": 0.70,
            "bet": 0.60,
            "call": 0.35,
            "moderate_bid": 0.35,
            "check": 0.10,
            "fold": 0.0
        }
        return agg_map.get(action.lower(), 0.50)

    def analyze_intent(
        self,
        opponent_id: str,
        state: Dict[str, float],
        ego_action: str,
        opponent_action: str,
        actual_next_state: Dict[str, float],
        candidate_opponent_actions: Optional[List[str]] = None,
        opponent_revealed_strength: Optional[float] = None,
        nash_strategy: Optional[Dict[str, float]] = None
    ) -> AgentIntent:
        """Performs SCM-grounded intent inference and bluffing classification."""
        candidates = candidate_opponent_actions or ["fold", "check", "call", "raise"]
        info_set = f"state_pot_{int(state.get('pot', 10.0))}"
        
        # 1. Update belief tracker
        self.tracker.observe_action(info_set, opponent_action)
        profile = self.tracker.get_profile(info_set, candidates, nash_strategy)
        kl_div = profile.get("bias_kl", 0.0)
        stat_confidence = profile.get("confidence", 0.5)

        # 2. Abduce exogenous noise using Decentralized SCM
        u_vector, shock_norm, is_shock = self.scm.abduce_noise(
            state=state,
            ego_action=ego_action,
            opponent_action=opponent_action,
            actual_next_state=actual_next_state
        )
        
        # 3. Compute environmental stability factor w_stable in [0.0, 1.0]
        # Under massive external shocks, strategic intent is masked / confounded by noise
        w_stable = max(0.0, min(1.0, 1.0 - (shock_norm / (self.scm.shock_threshold * 2.0))))
        
        # 4. Estimate opponent underlying strength (latent card / reserve value in [0, 1])
        # If not explicitly known, estimate from historical equilibrium prior
        action_aggressiveness = self._estimate_action_aggressiveness(opponent_action)
        est_strength = opponent_revealed_strength if opponent_revealed_strength is not None else 0.45
        
        # Payoff gap: Discrepancy between high aggressiveness and true strength backing
        agg_gap = action_aggressiveness - est_strength
        
        # 5. Compute Bluff Logit & Probability
        # If environment is unstable (is_shock), damp the strategic attribution
        bluff_logit = (self.alpha * agg_gap * w_stable) + (self.beta * kl_div * w_stable)
        # Shift logit by baseline intercept
        bluff_logit -= 0.5
        
        bluff_prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, bluff_logit))))
        
        # 6. Intent Classification Logic
        if is_shock and shock_norm > self.scm.shock_threshold * 1.5:
            intent_type = IntentType.EXOGENOUS_CONFUSION
            bluff_prob *= 0.3  # De-escalate bluff probability under shock
            rec_counter = "DEFENSIVE_HOLD"
            rationale = (
                f"Severe exogenous shock detected (||U_t||={shock_norm:.3f} >= {self.scm.shock_threshold}). "
                "Observation discrepancy is driven by external turbulence rather than deliberate opponent deception."
            )
        elif action_aggressiveness >= 0.60 and est_strength <= 0.40 and bluff_prob >= self.bluff_threshold:
            intent_type = IntentType.STRATEGIC_BLUFF
            rec_counter = "CALL_BLUFF" if action_aggressiveness < 0.85 else "PUNISH_RE_RAISE"
            rationale = (
                f"High-confidence strategic bluff detected (P={bluff_prob:.3f}, Aggressiveness={action_aggressiveness:.2f} "
                f"vs Est.Strength={est_strength:.2f}). Environment is stable (||U_t||={shock_norm:.3f})."
            )
        elif action_aggressiveness >= 0.60 and est_strength > 0.60:
            intent_type = IntentType.VALUE_BET
            rec_counter = "FOLD" if est_strength > 0.75 else "DISCOUNTED_CALL"
            rationale = (
                f"Genuine value bet detected. Opponent displays strong backing (Est.Strength={est_strength:.2f}). "
                "Direct confrontation carries negative expected value."
            )
        elif action_aggressiveness <= 0.35 and est_strength >= 0.70:
            intent_type = IntentType.TRAP_PASSIVITY
            rec_counter = "CHECK_BEHIND"
            rationale = (
                f"Trap / Slow-play detected. Opponent shows deceptive passivity despite strong reserve ({est_strength:.2f}). "
                "Avoid unforced pot swelling."
            )
        else:
            intent_type = IntentType.NORMAL_EQUILIBRIUM
            rec_counter = "EQUILIBRIUM_MIX"
            rationale = (
                f"Action conforms to baseline equilibrium expectations (KL={kl_div:.3f}, P_bluff={bluff_prob:.3f})."
            )

        return AgentIntent(
            opponent_id=opponent_id,
            observed_action=opponent_action,
            intent_type=intent_type,
            bluff_probability=round(bluff_prob, 4),
            abduced_shock_norm=round(shock_norm, 4),
            is_environment_stable=not is_shock,
            counterfactual_payoff_gap=round(agg_gap, 4),
            recommended_counter_action=rec_counter,
            confidence=round(stat_confidence, 4),
            rationale=rationale
        )
