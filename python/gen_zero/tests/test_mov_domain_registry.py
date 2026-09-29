"""Contract checks for the MoV scenario registry and domain routing boundary."""

import unittest

import numpy as np

from gen_zero.gateway.mov_fusion import (
    SCENARIO_DOMAINS,
    MicroCoreOutput,
    MoVDecisionLayer,
)


DIM = 64
REQUIRED_DOMAINS = {"Finance", "Safety", "CausalReasoning", "LogicDecision"}


class TestMovDomainRegistry(unittest.TestCase):
    def test_required_domains_have_structured_metadata(self):
        self.assertTrue(REQUIRED_DOMAINS.issubset(SCENARIO_DOMAINS))
        for domain, metadata in SCENARIO_DOMAINS.items():
            self.assertIsInstance(metadata, dict)
            for field in ("id", "display_name", "description", "default_intent", "intents", "risk_level", "capabilities"):
                self.assertIn(field, metadata, domain)
            self.assertTrue(metadata["id"])
            self.assertTrue(metadata["display_name"])
            self.assertTrue(metadata["description"])
            self.assertTrue(metadata["intents"])
            self.assertEqual(metadata["default_intent"], metadata["intents"][0])
            self.assertTrue(metadata["risk_level"])
            self.assertTrue(metadata["capabilities"])

    def test_default_prototypes_are_deterministic_and_match_registry(self):
        first = MoVDecisionLayer(dim=DIM)
        second = MoVDecisionLayer(dim=DIM)
        self.assertEqual(set(first.domain_prototypes), set(SCENARIO_DOMAINS))
        for domain in SCENARIO_DOMAINS:
            np.testing.assert_array_equal(
                first.domain_prototypes[domain], second.domain_prototypes[domain]
            )

    def test_unknown_domain_is_rejected(self):
        layer = MoVDecisionLayer(dim=DIM)
        state = np.zeros(DIM, dtype=np.float32)
        with self.assertRaises(ValueError):
            layer.compute_gating_weights(state, ["not_registered"])
        output = MicroCoreOutput(
            core_id="unknown",
            domain="not_registered",
            action_probabilities={"allow": 1.0},
            closed_form_confidence=1.0,
            feature_vector=np.zeros(DIM, dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            layer.fuse_decisions(state, [output], ["allow"])


if __name__ == "__main__":
    unittest.main()
