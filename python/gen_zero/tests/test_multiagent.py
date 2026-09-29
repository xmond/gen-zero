"""Unit tests for Gen-Zero Layer 2 Multi-Agent Causal Engine (Phase 4 Step 1)."""

import unittest
from typing import Dict, Any

from gen_zero.client import GenZero
from gen_zero.multiagent.decentralized_scm import (
    DecentralizedSCM,
    AsymmetricBluffDetector,
    IntentType,
    AgentIntent
)


class TestMultiAgentDecentralizedSCM(unittest.TestCase):
    """Tests for Decentralized SCM and Asymmetric Bluff Detector."""

    def setUp(self):
        self.scm = DecentralizedSCM(shock_threshold=0.25)
        self.detector = AsymmetricBluffDetector(d_scm=self.scm)
        self.client = GenZero()

    def test_noise_abduction_clean_vs_shock(self):
        """Verify SCM cleanly isolates zero-noise vs exogenous shocks."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        ego_action = "call"
        opp_action = "raise"
        
        # 1. Clean transition (matches transition_fn: pot -> 20 + 5 + 10 = 35, pressure -> 0.5)
        actual_clean = {"pot": 35.0, "pressure": 0.5, "volatility": 0.1}
        u_clean, shock_clean, is_shock_clean = self.scm.abduce_noise(
            state, ego_action, opp_action, actual_clean
        )
        self.assertFalse(is_shock_clean)
        self.assertAlmostEqual(shock_clean, 0.0, places=4)

        # 2. Shock transition (e.g. flash liquidity disruption or external rule penalty)
        actual_shock = {"pot": 35.0, "pressure": 0.95, "volatility": 0.6}
        u_shock, shock_val, is_shock_flag = self.scm.abduce_noise(
            state, ego_action, opp_action, actual_shock
        )
        self.assertTrue(is_shock_flag)
        self.assertGreater(shock_val, self.scm.shock_threshold)

    def test_counterfactual_simulation(self):
        """Test counterfactual intervention under locked exogenous noise."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        abduced_u = {"pot": 0.0, "pressure": 0.1, "volatility": 0.0}
        
        cf_next = self.scm.counterfactual_simulation(
            state=state,
            do_ego_action="fold",
            do_opponent_action="check",
            abduced_u=abduced_u
        )
        # Check that abduced noise is locked and added
        self.assertIn("pressure", cf_next)
        self.assertAlmostEqual(cf_next["pressure"], 0.1)

    def test_bluff_detection_under_stable_environment(self):
        """Opponent takes aggressive action with weak strength in stable environment -> STRATEGIC_BLUFF."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        actual_next = {"pot": 35.0, "pressure": 0.5, "volatility": 0.1}
        
        intent = self.detector.analyze_intent(
            opponent_id="shark_01",
            state=state,
            ego_action="call",
            opponent_action="raise",
            actual_next_state=actual_next,
            opponent_revealed_strength=0.20  # Weak hand!
        )
        
        self.assertEqual(intent.intent_type, IntentType.STRATEGIC_BLUFF)
        self.assertTrue(intent.is_environment_stable)
        self.assertGreaterEqual(intent.bluff_probability, 0.58)
        self.assertIn("CALL_BLUFF", intent.recommended_counter_action)

    def test_value_bet_detection(self):
        """Opponent takes aggressive action with strong hand -> VALUE_BET."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        actual_next = {"pot": 35.0, "pressure": 0.5, "volatility": 0.1}
        
        intent = self.detector.analyze_intent(
            opponent_id="shark_02",
            state=state,
            ego_action="call",
            opponent_action="raise",
            actual_next_state=actual_next,
            opponent_revealed_strength=0.85  # Strong hand!
        )
        
        self.assertEqual(intent.intent_type, IntentType.VALUE_BET)
        self.assertIn(intent.recommended_counter_action, ["FOLD", "DISCOUNTED_CALL"])

    def test_trap_passivity_detection(self):
        """Opponent takes passive action with monstrous strength -> TRAP_PASSIVITY."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        actual_next = {"pot": 20.0, "pressure": 0.0, "volatility": 0.1}
        
        intent = self.detector.analyze_intent(
            opponent_id="shark_03",
            state=state,
            ego_action="check",
            opponent_action="check",
            actual_next_state=actual_next,
            opponent_revealed_strength=0.92  # Trap!
        )
        
        self.assertEqual(intent.intent_type, IntentType.TRAP_PASSIVITY)
        self.assertEqual(intent.recommended_counter_action, "CHECK_BEHIND")

    def test_exogenous_confusion_isolation(self):
        """When large environmental shock is present, avoid mistaking noise for bluff."""
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        # Huge exogenous spike in volatility & pressure
        actual_next = {"pot": 35.0, "pressure": 0.95, "volatility": 0.85}
        
        intent = self.detector.analyze_intent(
            opponent_id="shark_04",
            state=state,
            ego_action="call",
            opponent_action="raise",
            actual_next_state=actual_next,
            opponent_revealed_strength=0.30
        )
        
        self.assertEqual(intent.intent_type, IntentType.EXOGENOUS_CONFUSION)
        self.assertFalse(intent.is_environment_stable)
        self.assertEqual(intent.recommended_counter_action, "DEFENSIVE_HOLD")

    def test_client_integration(self):
        """Test GenZero top-level client interface."""
        state = {"pot": 30.0, "pressure": 0.1, "volatility": 0.1}
        actual_next = {"pot": 45.0, "pressure": 0.4, "volatility": 0.1}
        
        intent = self.client.analyze_opponent_intent(
            opponent_id="bot_alpha",
            state=state,
            ego_action="call",
            opponent_action="raise",
            actual_next_state=actual_next,
            opponent_revealed_strength=0.15
        )
        self.assertIsInstance(intent, AgentIntent)
        self.assertEqual(intent.intent_type, IntentType.STRATEGIC_BLUFF)


if __name__ == "__main__":
    unittest.main()
