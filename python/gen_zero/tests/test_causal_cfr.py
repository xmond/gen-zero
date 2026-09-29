"""Unit tests for Gen-Zero Layer 2 Causal-CFR Engine (Phase 4 Step 2)."""

import unittest
from typing import Dict, List

from gen_zero.client import GenZero
from gen_zero.multiagent.causal_cfr_engine import CausalCFREngine, CausalCFROutcome
from gen_zero.multiagent.decentralized_scm import DecentralizedSCM


class TestCausalCFREngine(unittest.TestCase):
    """Tests for Causal-CFR Engine and Safe Exploitation."""

    def setUp(self):
        self.scm = DecentralizedSCM(shock_threshold=0.25)
        self.engine = CausalCFREngine(iterations=50, shock_threshold=0.25, d_scm=self.scm)
        self.client = GenZero()

    def test_shock_filtering_in_regret_updates(self):
        """Verify that exogenous noise ||U_t|| >= tau does not pollute the regret table."""
        info_set = "test_info_01"
        legal_actions = ["fold", "call", "raise"]
        utils = {"fold": 0.0, "call": 1.0, "raise": 2.0}

        # Severe shock (||U|| = 0.50 >= 0.25)
        w1 = self.engine.update_causal_regret(
            info_set=info_set,
            legal_actions=legal_actions,
            action_utilities=utils,
            abduced_shock_norm=0.50,
            iteration_index=1
        )
        self.assertEqual(w1, 0.0)
        # Regret table should remain pristine (all zero)
        regrets = self.engine.regret_table[info_set]
        self.assertTrue(all(v == 0.0 for v in regrets.values()))

        # Clean step (||U|| = 0.0)
        w2 = self.engine.update_causal_regret(
            info_set=info_set,
            legal_actions=legal_actions,
            action_utilities=utils,
            abduced_shock_norm=0.0,
            iteration_index=2
        )
        self.assertEqual(w2, 1.0)
        self.assertGreater(self.engine.regret_table[info_set]["raise"], 0.0)

    def test_cfr_plus_monotonic_convergence(self):
        """Test CFR+ convergence to low exploitability epsilon < 0.005."""
        info_set = "poker_turn_pot_40"
        legal_actions = ["fold", "call", "raise"]
        utils = {"fold": -1.0, "call": 1.0, "raise": 0.5}

        outcome = self.engine.solve_causal_game(
            info_set=info_set,
            legal_actions=legal_actions,
            base_utilities=utils,
            abduced_shock_norm=0.0
        )

        self.assertIsInstance(outcome, CausalCFROutcome)
        self.assertLess(outcome.exploitability_epsilon, 0.005)
        # Call should be the dominant action given the utilities
        self.assertEqual(outcome.recommended_action, "call")
        self.assertGreater(outcome.mixed_strategy["call"], 0.5)

    def test_safe_exploitation_blend(self):
        """Test Bayesian safe exploitation against persistent opponent bias."""
        info_set = "exploitable_fish_spot"
        legal_actions = ["fold", "call", "raise"]

        # Feed 15 consecutive over-aggressive actions from opponent to build confidence
        for _ in range(15):
            self.engine.tracker.observe_action(info_set, "raise")

        outcome = self.engine.solve_causal_game(
            info_set=info_set,
            legal_actions=legal_actions,
            base_utilities={"fold": 0.0, "call": 1.5, "raise": 0.2},
            observed_opponent_action="raise"
        )

        # Beta should be strictly positive due to detected opponent bias and statistical confidence
        self.assertGreater(outcome.safe_exploitation_beta, 0.0)
        self.assertLessEqual(outcome.safe_exploitation_beta, self.engine.max_exploitation_beta)

    def test_client_integration(self):
        """Test top-level GenZero client solve_causal_game invocation."""
        info_set = "client_matchup_01"
        legal_actions = ["conservative", "balanced", "aggressive"]
        
        outcome = self.client.solve_causal_game(
            info_set=info_set,
            legal_actions=legal_actions,
            base_utilities={"conservative": 0.8, "balanced": 1.2, "aggressive": 0.5},
            abduced_shock_norm=0.05
        )

        self.assertIsInstance(outcome, CausalCFROutcome)
        self.assertEqual(outcome.recommended_action, "balanced")
        self.assertLess(outcome.exploitability_epsilon, 0.005)


if __name__ == "__main__":
    unittest.main()
