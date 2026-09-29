"""Gen-Zero Self-Perception & Dynamic Parameter Engine.

Eliminates all fixed hyperparameters across Layer 1 ~ Layer 5:
1. MCTS simulation budget N(s) in [8, 256]
2. MCTS exploration constant c_puct(s) in [0.8, 2.5]
3. A* heuristic weight lambda(s) in [0.1, 2.0]
4. MPC rolling horizon H_mpc in [3, 12] and sample budget in [16, 64]
5. GFlowNet sampling temperature T(s) in [0.3, 2.0]
6. CFR regret convergence early-stopping threshold (epsilon = 1e-4)
7. Relative normalized entropy mining threshold (H_norm > 0.70)
8. Value-cliff causal failure attribution window (detects exact drop step)
9. Replay buffer elastic forgetting ratio (adaptive hard-to-gold balance)
10. Dynamic ATR risk limits for quantitative trading
"""

import math
from typing import Dict, List, Any, Tuple, Optional


class AdaptiveParameterEngine:
    """Universal Parameter Self-Regulation Engine."""

    @staticmethod
    def get_normalized_entropy(probs: Dict[str, float]) -> float:
        """Computes Shannon entropy normalized by max possible entropy log(K)."""
        k = len(probs)
        if k <= 1:
            return 0.0
        h = -sum(p * math.log(max(1e-9, p)) for p in probs.values() if p > 1e-6)
        max_h = math.log(k)
        return float(max(0.0, min(1.0, h / max_h)))

    @staticmethod
    def dynamic_mcts_budget(
        state: Any,
        policy_entropy: Optional[float] = None,
        candidates_count: int = 4
    ) -> int:
        """Determines MCTS simulation budget N(s) in [8, 256].
        
        - Low entropy (clear action): N = 12 ~ 24 (sub-millisecond)
        - High entropy (ambivalent/complex): N = 96 ~ 256
        """
        ent = max(0.0, min(1.0, float(policy_entropy if policy_entropy is not None else 0.5)))
        # Quadratic scaling for high entropy
        raw_n = 16.0 + 180.0 * (ent ** 1.5)
        # Factor in action cardinality
        if candidates_count > 4:
            raw_n *= (1.0 + (candidates_count - 4) * 0.15)
        return int(max(8, min(256, round(raw_n))))

    @staticmethod
    def dynamic_mcts_cpuct(
        state: Any,
        td_surprise: Optional[float] = None,
        complexity_score: float = 0.5
    ) -> float:
        """Determines PUCT exploration constant c_puct in [0.8, 2.4].
        
        - Mature / low surprise states: c_puct ~ 1.0 (exploit proven trajectories)
        - High surprise / OOD states: c_puct ~ 2.2 (expand wide exploration)
        """
        base = 1.10
        surprise = td_surprise if td_surprise is not None else complexity_score
        c = base + 1.20 * max(0.0, min(1.0, surprise))
        return round(float(c), 3)

    @staticmethod
    def dynamic_astar_lambda(
        state: Any,
        policy_entropy: Optional[float] = None
    ) -> float:
        """Determines A* heuristic edge weight lambda(s) in [0.1, 2.0].
        
        - High entropy: lambda drops to ~0.2 (trust true topological distance)
        - Low entropy: lambda rises to ~1.8 (strongly follow model intuition)
        """
        ent = policy_entropy if policy_entropy is not None else 0.5
        # Inverted entropy
        lam = 1.8 * (1.0 - max(0.0, min(1.0, ent))) + 0.2
        return round(float(lam), 3)

    @staticmethod
    def dynamic_mpc_params(
        volatility: float = 0.01,
        complexity: float = 0.5
    ) -> Tuple[int, int]:
        """Determines (horizon, num_samples) for MPC CEM.
        
        - High volatility / sharp turns: horizon expands to 8~10, samples to 48~64
        - Calm straight trajectory: horizon = 3~4, samples = 16~24
        """
        vol_scale = max(0.0, min(1.0, volatility * 50.0 + complexity * 0.5))
        horizon = int(max(3, min(12, round(3.0 + 8.0 * vol_scale))))
        samples = int(max(16, min(64, round(16.0 + 44.0 * vol_scale))))
        return horizon, samples

    @staticmethod
    def dynamic_gflownet_temperature(
        policy_entropy: Optional[float] = None,
        mode_collapse_detected: bool = False
    ) -> float:
        """Determines GFlowNet sampling temperature T in [0.3, 2.2]."""
        if mode_collapse_detected:
            return 2.0  # Heat up to escape collapse
        ent = policy_entropy if policy_entropy is not None else 0.5
        t = 0.4 + 1.2 * max(0.0, min(1.0, ent))
        return round(float(t), 3)

    @staticmethod
    def cfr_should_early_stop(
        regret_delta: float,
        iteration: int,
        min_iterations: int = 15,
        target_epsilon: float = 1e-4
    ) -> bool:
        """Determines if CFR Nash Equilibrium has reached numerical convergence."""
        if iteration < min_iterations:
            return False
        return regret_delta < target_epsilon

    @staticmethod
    def dynamic_safe_exploitation_beta(
        confidence: float,
        bias_kl: float,
        safety_budget: float = 0.25,
        min_beta: float = 0.10
    ) -> float:
        """Determines safe interpolation weight beta in [min_beta, 1.0] for opponent exploitation.
        
        pi^* = (1 - beta) * BestResponse + beta * NashStrategy
        - Low confidence or zero bias => beta = 1.0 (pure unexploitable Nash defense)
        - High confidence & strong bias => beta drops to min_beta (active exploitation)
        """
        if confidence < 0.20 or bias_kl < 0.05:
            return 1.0  # Maintain rock-solid Nash defense

        signal = confidence * math.sqrt(bias_kl)
        raw_beta = math.exp(-signal / max(0.01, safety_budget))
        return round(float(max(min_beta, min(1.0, raw_beta))), 3)

    @staticmethod
    def dynamic_mining_criteria(
        model_probs: Dict[str, float],
        td_error: float,
        history_td_errors: Optional[List[float]] = None
    ) -> Tuple[bool, bool]:
        """Determines whether a sample is hard via relative entropy and dynamic percentile.
        
        Returns:
            (is_high_entropy, is_high_td_error)
        """
        norm_ent = AdaptiveParameterEngine.get_normalized_entropy(model_probs)
        is_high_entropy = norm_ent > 0.70  # Normalized relative threshold

        # Dynamic 80th percentile for TD error if history available
        if history_td_errors and len(history_td_errors) >= 10:
            sorted_errs = sorted(history_td_errors)
            p80_idx = int(len(sorted_errs) * 0.80)
            dynamic_td_thresh = max(0.25, sorted_errs[p80_idx])
        else:
            dynamic_td_thresh = 0.50

        is_high_td = td_error > dynamic_td_thresh
        return is_high_entropy, is_high_td

    @staticmethod
    def detect_value_cliff_step(trajectory: List[Dict[str, Any]]) -> int:
        """Finds the true causal turning point of failure by detecting value cliff."""
        n = len(trajectory)
        if n <= 1:
            return 0

        # Scan backwards for the step with the sharpest value degradation
        best_cliff_step = max(0, n - 3)
        max_drop = 0.0

        for i in range(n - 1, 0, -1):
            v_curr = trajectory[i].get("model_value", 0.0)
            v_prev = trajectory[i - 1].get("model_value", 0.0)
            drop = v_prev - v_curr
            if drop > max_drop:
                max_drop = drop
                best_cliff_step = i - 1

        # If significant cliff found (drop > 0.25), use it
        if max_drop > 0.25:
            return best_cliff_step
        return max(0, n - 5)

    @staticmethod
    def adjust_replay_hard_ratio(
        current_retention_rate: float,
        target_retention: float = 0.995,
        current_hard_ratio: float = 0.25
    ) -> float:
        """Dynamically adjusts hard-to-gold replay ratio based on catastrophic forgetting."""
        if current_retention_rate < target_retention:
            # Forgetting detected! Throttles hard ratio to restore foundation
            new_ratio = max(0.10, current_hard_ratio - 0.05)
        elif current_retention_rate >= 0.998:
            # Anchor is rock solid; boost hard sample learning rate
            new_ratio = min(0.50, current_hard_ratio + 0.05)
        else:
            new_ratio = current_hard_ratio
        return round(new_ratio, 3)

    @staticmethod
    def dynamic_quant_risk_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        window: int = 15
    ) -> Tuple[float, float]:
        """Calculates volatility-adaptive stop-loss and take-profit limits via ATR.
        
        Returns:
            (stop_loss_pct, take_profit_pct)
        """
        if (
            len(closes) < window + 1
            or len(highs) < window
            or len(lows) < window
            or closes[-1] <= 0
        ):
            return 0.015, 0.025  # Fallback 1.5% and 2.5%

        trs = []
        for i in range(-window, 0):
            h = highs[i]
            l = lows[i]
            prev_c = closes[i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        atr = sum(trs) / max(1, len(trs))
        atr_pct = atr / closes[-1]

        # Stop-loss at 2.0x ATR, Take-profit at 3.5x ATR
        stop_loss = max(0.005, min(0.04, atr_pct * 2.0))
        take_profit = max(0.010, min(0.08, atr_pct * 3.5))
        return round(stop_loss, 4), round(take_profit, 4)

    @staticmethod
    def dynamic_modality_decision(
        state_input: Any
    ) -> Tuple[str, bool, str]:
        """Formula 17: Input Modality Auto-Perception & LLM Forward Gating.
        
        Analyzes the mathematical structure of the state representation:
        - Pure numerical float dict / ndarray -> ('pure_numerical', False, 'bypass_llm_zero_latency')
        - Unstructured natural language -> ('unstructured_text', True, 'require_llm_forward')
        - Hybrid mixed sensor & prompt -> ('multimodal_hybrid', True, 'fuse_llm_and_telemetry')
        
        Returns:
            (modality_type, should_invoke_llm, rationale)
        """
        if isinstance(state_input, str):
            if any(state_input.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                return "vision_image", True, "require_vlm_forward"
            return "unstructured_text", True, "require_llm_forward"
        
        if hasattr(state_input, "size") and hasattr(state_input, "mode"):
            return "vision_image", True, "require_vlm_forward"
            
        if hasattr(state_input, "ndim") and state_input.ndim in (3, 4):
            return "vision_image", True, "require_vlm_forward"

        if isinstance(state_input, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in state_input):
                return "pure_numerical", False, "bypass_llm_zero_latency"
            return "unstructured_text", True, "require_llm_forward"
            
        if isinstance(state_input, dict):
            has_img = any(k in ("image", "frame", "visual", "screenshot") or (hasattr(v, "size") and hasattr(v, "mode")) for k, v in state_input.items())
            has_str = any(isinstance(v, str) for v in state_input.values())
            has_num = any(isinstance(v, (int, float, bool)) for v in state_input.values())
            
            if has_img:
                if has_str or has_num:
                    return "multimodal_vision_text", True, "require_vlm_forward"
                return "vision_image", True, "require_vlm_forward"
            if has_str and has_num:
                return "multimodal_hybrid", True, "fuse_llm_and_telemetry"
            elif has_str:
                return "unstructured_text", True, "require_llm_forward"
            else:
                return "pure_numerical", False, "bypass_llm_zero_latency"

        return "pure_numerical", False, "bypass_llm_zero_latency"
