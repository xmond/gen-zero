"""Unit tests for two-stage hierarchical Simplex ETF routing (scenarios -> sub-intents).

Verifies:
1. The default taxonomy has 18 scenarios and 60 intents.
2. Every frame is an exact ETF and the two stages are orthogonal.
3. Clean encodings route back to their own (scenario, intent) for all 60 labels.
4. Stage-1 margin controls how many scenarios are expanded.
5. Sub-intent noise cannot change the scenario logits.
6. Input validation and determinism.
"""

import unittest

import numpy as np

from gen_zero.nanocore.hierarchical_simplex import (
    MASSIVE_TAXONOMY,
    HierarchicalRoute,
    HierarchicalSimplexRouter,
)


class TestHierarchicalSimplex(unittest.TestCase):
    """Test suite for HierarchicalSimplexRouter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.router = HierarchicalSimplexRouter()

    def test_01_default_taxonomy_shape(self):
        self.assertEqual(len(MASSIVE_TAXONOMY), 18)
        self.assertEqual(sum(len(v) for v in MASSIVE_TAXONOMY.values()), 60)
        self.assertEqual(len(self.router.scenarios), 18)

    def test_02_frames_are_exact_etfs_and_stages_orthogonal(self):
        report = self.router.verify()
        self.assertTrue(report.is_valid, report.to_dict())
        self.assertLess(report.stage1_max_deviation, 1e-12)
        self.assertLess(report.stage2_max_deviation, 1e-12)
        self.assertLess(report.cross_stage_max_overlap, 1e-12)

    def test_03_stage1_gram_is_minus_one_over_k_minus_one(self):
        anchors = np.stack([self.router.stage1_anchor(s) for s in self.router.scenarios])
        gram = anchors @ anchors.T
        off = gram[~np.eye(18, dtype=bool)]
        np.testing.assert_allclose(off, -1.0 / 17.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-12)

    def test_04_all_sixty_labels_round_trip(self):
        for scenario, intents in MASSIVE_TAXONOMY.items():
            for intent in intents:
                route = self.router.route(self.router.encode(scenario, intent))
                self.assertEqual((route.scenario, route.intent), (scenario, intent))

    def test_05_single_intent_scenario_skips_stage2(self):
        route = self.router.route(self.router.encode("weather", "weather_query"))
        self.assertEqual(route.intent, "weather_query")
        self.assertAlmostEqual(route.intent_prob, 1.0)
        self.assertAlmostEqual(route.joint_prob, route.scenario_prob)

    def test_06_confident_stage1_expands_one_scenario(self):
        route = self.router.route(self.router.encode("iot", "iot_wemo_on"))
        self.assertEqual(route.expanded_scenarios, ("iot",))
        self.assertGreaterEqual(route.stage1_margin, self.router.margin_threshold)

    def test_07_ambiguous_stage1_expands_beam(self):
        hidden = np.zeros(self.router.dim)  # zero vector: uniform stage 1, margin 0
        route = self.router.route(hidden)
        self.assertEqual(len(route.expanded_scenarios), self.router.beam)
        self.assertEqual(route.stage1_margin, 0.0)
        self.assertIsInstance(route, HierarchicalRoute)

    def test_08_beam_can_overturn_a_near_tied_stage1_call(self):
        """With a near-tied stage 1, joint scoring picks the scenario with sharper sub-intent evidence."""
        hidden = self.router.stage1_anchor("alarm") + 0.95 * self.router.stage1_anchor("cooking")
        hidden[17] = 1.0  # sub-intent evidence, read by whichever scenario is expanded

        greedy = HierarchicalSimplexRouter(temperature=0.5, margin_threshold=0.5, beam=1)
        beamed = HierarchicalSimplexRouter(temperature=0.5, margin_threshold=0.5, beam=2)
        self.assertEqual(greedy.route(hidden).scenario, "alarm")
        route = beamed.route(hidden)
        self.assertEqual(route.expanded_scenarios, ("alarm", "cooking"))
        self.assertEqual((route.scenario, route.intent), ("cooking", "cooking_recipe"))

    def test_09_intent_noise_does_not_move_scenario_probs(self):
        base = self.router.encode("music", "music_query")
        noisy = base + 3.0 * self.router.stage2_anchor("music", "music_likeness")
        np.testing.assert_allclose(
            self.router.scenario_probs(base), self.router.scenario_probs(noisy), atol=1e-12
        )

    def test_10_probabilities_are_normalized(self):
        hidden = np.random.RandomState(7).randn(self.router.dim)
        self.assertAlmostEqual(float(self.router.scenario_probs(hidden).sum()), 1.0)
        for scenario in self.router.scenarios:
            self.assertAlmostEqual(float(self.router.intent_probs(hidden, scenario).sum()), 1.0)

    def test_11_deterministic_and_batch_matches_single(self):
        rng = np.random.RandomState(11)
        batch = rng.randn(8, self.router.dim)
        first = [r.to_dict() for r in self.router.route_batch(batch)]
        second = [self.router.route(row).to_dict() for row in batch]
        self.assertEqual(first, second)

    def test_12_custom_taxonomy(self):
        router = HierarchicalSimplexRouter({"a": ["a1", "a2"], "b": ["b1"]}, dim=8)
        self.assertTrue(router.verify().is_valid)
        self.assertEqual(router.route(router.encode("a", "a2")).intent, "a2")
        self.assertEqual(router.route(router.encode("b", "b1")).scenario, "b")

    def test_13_single_scenario_taxonomy(self):
        router = HierarchicalSimplexRouter({"only": ["x", "y"]}, dim=4)
        route = router.route(router.encode("only", "y"))
        self.assertEqual((route.scenario, route.intent), ("only", "y"))
        self.assertAlmostEqual(route.scenario_prob, 1.0)

    def test_14_rejects_bad_taxonomies(self):
        with self.assertRaises(ValueError):
            HierarchicalSimplexRouter({})
        with self.assertRaises(ValueError):
            HierarchicalSimplexRouter({"a": []})
        with self.assertRaises(ValueError):
            HierarchicalSimplexRouter({"a": ["x", "x"]})
        with self.assertRaises(ValueError):
            HierarchicalSimplexRouter({"a": ["x"], "b": ["x"]})

    def test_15_rejects_small_dim(self):
        with self.assertRaises(ValueError):
            HierarchicalSimplexRouter(dim=24)  # needs 17 + 8 = 25
        HierarchicalSimplexRouter(dim=25)

    def test_16_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.router.route(np.zeros(self.router.dim + 1))
        bad = np.zeros(self.router.dim)
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            self.router.route(bad)
        with self.assertRaises(ValueError):
            self.router.route_batch(np.zeros(self.router.dim))
        with self.assertRaises(KeyError):
            self.router.intent_probs(np.zeros(self.router.dim), "nope")
        with self.assertRaises(KeyError):
            self.router.stage2_anchor("alarm", "weather_query")
        with self.assertRaises(KeyError):
            self.router.stage1_anchor("nope")


if __name__ == "__main__":
    unittest.main()
