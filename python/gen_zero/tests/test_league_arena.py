"""Unit tests for Gen-Zero Layer 2 League Arena & PFSP (Phase 4 Step 3)."""

import unittest
from typing import Dict, Any

from gen_zero.client import GenZero
from gen_zero.multiagent.league_arena import (
    LeagueArena,
    LeagueRole,
    AgentSnapshot,
    MatchResult
)


class TestLeagueArena(unittest.TestCase):
    """Tests for League Arena co-evolution and PFSP matchmaking."""

    def setUp(self):
        self.arena = LeagueArena(base_elo=1200.0, k_factor=32.0, pfsp_exponent=2.0)
        self.client = GenZero()

    def test_initial_roles_and_archive(self):
        """Verifies initial presence of Main Agent, Main Exploiter, and League Exploiter in archive."""
        self.assertEqual(len(self.arena.archive), 3)
        self.assertIn("main_gen_0", self.arena.archive)
        self.assertIn("main_exploiter_gen_0", self.arena.archive)
        self.assertIn("league_exploiter_gen_0", self.arena.archive)
        self.assertEqual(self.arena.main_agent.role, LeagueRole.MAIN_AGENT)
        self.assertEqual(self.arena.main_exploiter.role, LeagueRole.MAIN_EXPLOITER)
        self.assertEqual(self.arena.league_exploiter.role, LeagueRole.LEAGUE_EXPLOITER)

    def test_pfsp_matchmaking_sampling(self):
        """Verify PFSP correctly samples opponents without self-matching."""
        opp = self.arena.sample_pfsp_opponent(self.arena.main_agent)
        self.assertNotEqual(opp.snapshot_id, self.arena.main_agent.snapshot_id)
        self.assertIn(opp.role, [LeagueRole.MAIN_EXPLOITER, LeagueRole.LEAGUE_EXPLOITER])

    def test_play_match_and_elo_dynamics(self):
        """Simulate a match and check zero-sum Elo rating adjustments."""
        agent_a = self.arena.main_agent
        agent_b = self.arena.main_exploiter
        elo_a_before = agent_a.elo_rating
        elo_b_before = agent_b.elo_rating

        match_res = self.arena.play_match(agent_a, agent_b, num_rounds=10)
        self.assertIsInstance(match_res, MatchResult)
        self.assertEqual(match_res.rounds_played, 10)
        
        # Symmetrical Elo conservation: delta_a + delta_b approx 0
        self.assertAlmostEqual(match_res.elo_delta_a + match_res.elo_delta_b, 0.0, places=1)
        self.assertEqual(agent_a.elo_rating, elo_a_before + match_res.elo_delta_a)

    def test_generation_co_evolution(self):
        """Run 3 generations of co-evolution and check archive expansion."""
        for g in range(1, 4):
            rep = self.arena.evolve_league_generation(matches_per_agent=5)
            self.assertEqual(rep["generation"], g)
            self.assertGreater(rep["archive_size"], 3)

        robustness = self.arena.evaluate_historical_robustness()
        self.assertIn("mean_win_rate", robustness)
        self.assertIn("elo_delta", robustness)

    def test_client_run_league_evolution_cycle(self):
        """Test top-level GenZero client interface for league evolution."""
        cycle_res = self.client.run_league_evolution_cycle(generations=2, matches_per_agent=4)
        self.assertIn("evolution_history", cycle_res)
        self.assertEqual(len(cycle_res["evolution_history"]), 2)
        self.assertIn("final_robustness", cycle_res)
        self.assertIn("main_agent_elo", cycle_res)


if __name__ == "__main__":
    unittest.main()
