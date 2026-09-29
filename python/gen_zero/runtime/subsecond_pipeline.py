"""Sub-Second High-Frequency Decision Pipeline with Late-Breaker Protection.

Implements Milestone 4 of Issue #18:
1. 300ms Strict Stage Budget Management:
   - Ingestion & Encoding: <= 20ms
   - Non-Autoregressive Scoring: <= 50ms
   - CP-SAT Formal Risk Gate: <= 10ms
   - Total Pipeline Target: <= 80ms
2. Late-Breaker Circuit: If overall latency exceeds safety threshold (>200ms),
   flags late=True and forces action='hold' to eliminate execution drift.
3. Event Frame Synthesis complying with Issue #18 SSE wire contract.
"""

from typing import Dict, List, Any, Optional, Tuple
import time

from gen_zero.model.market_state import L2OrderBookSnapshot, L2MarketStateEncoder
from gen_zero.gate.market_risk_gate import CPSATMarketRiskGate
from gen_zero.gateway.llamacpp_adapter import compute_closed_form_confidence


class SubSecondDecisionPipeline:
    """Orchestrates ingestion, reflex scoring, formal risk gating, and late-breaker enforcement."""

    def __init__(
        self,
        scorer_fn: Optional[Any] = None,
        risk_gate: Optional[CPSATMarketRiskGate] = None,
        encoder: Optional[L2MarketStateEncoder] = None,
        safety_deadline_ms: float = 200.0,
        latent_updater: Optional[Any] = None,
    ):
        self.encoder = encoder or L2MarketStateEncoder()
        self.risk_gate = risk_gate or CPSATMarketRiskGate()
        self.safety_deadline_ms = float(safety_deadline_ms)
        self.scorer_fn = scorer_fn or self._default_reflex_scorer
        # Optional zero-token latent-space reasoning sub-stage (System 1
        # "R>0" reflex depth). None by default: existing callers and tests
        # are unaffected. When set, must be a `gen_zero.model.LatentUpdater`
        # (or NumPy fallback) configured with input_dim=2, since it is fed
        # `[imbalance, spread_bps]` below.
        self.latent_updater = latent_updater

    def _default_reflex_scorer(
        self,
        state_text: str,
        candidates: List[str],
        imbalance: float,
    ) -> Dict[str, Any]:
        """High-speed microsecond heuristic scorer for L2 microstructure."""
        # Imbalance > +0.25 -> favors buy; Imbalance < -0.25 -> favors sell; else hold
        if imbalance > 0.25:
            probs = {"buy": round(0.55 + min(0.35, imbalance * 0.4), 4), "sell": 0.10, "hold": 0.0}
            probs["hold"] = round(1.0 - probs["buy"] - probs["sell"], 4)
            best = "buy"
        elif imbalance < -0.25:
            probs = {"sell": round(0.55 + min(0.35, abs(imbalance) * 0.4), 4), "buy": 0.10, "hold": 0.0}
            probs["hold"] = round(1.0 - probs["sell"] - probs["buy"], 4)
            best = "sell"
        else:
            probs = {"hold": 0.60, "buy": 0.20, "sell": 0.20}
            best = "hold"

        conf = compute_closed_form_confidence(probs)
        return {
            "action": best,
            "probabilities": probs,
            "confidence": round(conf, 4),
        }

    def process_tick(
        self,
        snapshot: L2OrderBookSnapshot,
        candidates: Optional[List[str]] = None,
        override_position: Optional[float] = None,
        simulate_artificial_delay_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Executes the sub-second decision cycle within the strict latency budget."""
        t0 = time.perf_counter()
        active_candidates = candidates or ["buy", "sell", "hold"]

        # Stage 1: Ingestion & Microstructural Feature Extraction (<= 20ms)
        t_ingest_start = time.perf_counter()
        state_text = self.encoder.encode_to_state_text(snapshot)
        features = self.encoder.encode_to_features(snapshot)
        imbalance = features["imbalance"]
        spread_bps = features["spread_bps"]
        ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

        # Stage 2a (optional): Zero-token latent-space reasoning (System 1
        # "R>0" reflex depth, G=0 generated tokens). Timed as its own
        # sub-stage, separate from `scoring` below, so the `scoring` number
        # stays unchanged whether or not this is configured. Default
        # LatentUpdater configs (small T / latent_dim) keep this well under
        # a few ms, comfortably inside the Stage 2 <= 50ms budget.
        latent_update_ms: Optional[float] = None
        latent_update_info: Optional[Dict[str, Any]] = None
        if self.latent_updater is not None:
            t_latent_start = time.perf_counter()
            _, latent_telemetry = self.latent_updater([imbalance, spread_bps])
            latent_update_ms = (time.perf_counter() - t_latent_start) * 1000.0
            latent_update_info = {
                "n_iterations": latent_telemetry.n_iterations,
                "final_residual": round(float(latent_telemetry.final_residual), 6),
                "converged": bool(latent_telemetry.converged),
            }

        # Stage 2: Non-Autoregressive Scoring (<= 50ms)
        t_score_start = time.perf_counter()
        score_res = self.scorer_fn(state_text, active_candidates, imbalance)
        proposed_action = score_res["action"]
        probabilities = score_res["probabilities"]
        confidence = score_res["confidence"]
        score_ms = (time.perf_counter() - t_score_start) * 1000.0

        # Stage 3: CP-SAT Formal Inventory & Risk Gate (<= 10ms)
        t_risk_start = time.perf_counter()
        risk_verdict = self.risk_gate.evaluate_risk(
            proposed_action=proposed_action,
            spread_bps=spread_bps,
            override_position=override_position,
        )
        final_action = risk_verdict["action"]
        risk_ms = (time.perf_counter() - t_risk_start) * 1000.0

        # Optional artificial delay for testing late breaker
        if simulate_artificial_delay_ms > 0:
            time.sleep(simulate_artificial_delay_ms / 1000.0)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Late-Breaker Check (> 200ms)
        is_late = bool(total_latency_ms > self.safety_deadline_ms)
        if is_late:
            # Overrides action to 'hold' to avoid execution drift slippage
            final_action = "hold"
            risk_verdict["passed"] = False
            risk_verdict["reason"] = f"late_breaker_triggered (latency {total_latency_ms:.1f}ms > {self.safety_deadline_ms:.1f}ms)"

        stage_breakdown_ms = {
            "ingest": round(ingest_ms, 2),
            "scoring": round(score_ms, 2),
            "risk_gate": round(risk_ms, 2),
        }
        if latent_update_ms is not None:
            stage_breakdown_ms["latent_update"] = round(latent_update_ms, 2)

        result = {
            "tick_id": snapshot.tick_id,
            "timestamp_ms": snapshot.timestamp_ms,
            "action": final_action,
            "probabilities": probabilities,
            "confidence": confidence,
            "latency_ms": round(total_latency_ms, 2),
            "late": is_late,
            "stage_breakdown_ms": stage_breakdown_ms,
            "risk_guard": {
                "passed": risk_verdict["passed"],
                "reason": risk_verdict["reason"],
            },
        }
        if latent_update_info is not None:
            result["latent_update"] = latent_update_info
        return result
