"""MoV domain prototypes come from HierarchicalSimplexRouter; fusion uses the domain posterior."""

import unittest

import numpy as np

from gen_zero.gateway.mov_fusion import (
    SCENARIO_DOMAINS,
    MicroCoreOutput,
    MoVDecisionLayer,
    rms_norm,
)

DIM = 64
CANDIDATES = ["BUY", "SELL", "HOLD"]


def _core(domain, conf, probs):
    return MicroCoreOutput(
        core_id=f"core_{domain}",
        domain=domain,
        action_probabilities=probs,
        closed_form_confidence=conf,
        feature_vector=np.ones(DIM, dtype=np.float32),
        expected_value=0.5,
    )


class TestSimplexPrototypes(unittest.TestCase):
    def setUp(self):
        self.layer = MoVDecisionLayer(dim=DIM)

    def test_prototypes_are_deterministic(self):
        other = MoVDecisionLayer(dim=DIM)
        for dom in SCENARIO_DOMAINS:
            np.testing.assert_array_equal(self.layer.domain_prototypes[dom], other.domain_prototypes[dom])

    def test_prototypes_are_equiangular_and_rms_normed(self):
        protos = [self.layer.domain_prototypes[d] for d in SCENARIO_DOMAINS]
        k = len(protos)
        for p in protos:
            self.assertAlmostEqual(float(np.sqrt(np.mean(p ** 2))), 1.0, places=4)
        for i in range(k):
            for j in range(i + 1, k):
                cos = float(np.dot(protos[i], protos[j]) / (np.linalg.norm(protos[i]) * np.linalg.norm(protos[j])))
                self.assertAlmostEqual(cos, -1.0 / (k - 1), places=4)

    def test_prototypes_match_router_anchor_direction(self):
        router = self.layer.simplex_router
        for dom in SCENARIO_DOMAINS:
            np.testing.assert_allclose(
                self.layer.domain_prototypes[dom], rms_norm(router.stage1_anchor(dom)), atol=1e-5
            )

    def test_custom_taxonomy_defines_domains(self):
        layer = MoVDecisionLayer(dim=DIM, domain_taxonomy={"A": ["a1", "a2"], "B": ["b1"]})
        self.assertEqual(set(layer.domain_prototypes), {"A", "B"})


class TestRouteHierarchical(unittest.TestCase):
    def setUp(self):
        self.layer = MoVDecisionLayer(dim=DIM)
        self.router = self.layer.simplex_router

    def test_route_hierarchical_recovers_domain(self):
        for dom in SCENARIO_DOMAINS:
            state = self.router.stage1_anchor(dom).astype(np.float32)
            route = self.layer.route_hierarchical(state)
            self.assertEqual(route.scenario, dom)
            self.assertEqual(route.intent, f"{dom}_default")

    def test_route_hierarchical_rejects_bad_state(self):
        with self.assertRaises(ValueError):
            self.layer.route_hierarchical(np.ones(DIM + 1))
        with self.assertRaises(ValueError):
            self.layer.route_hierarchical(np.full(DIM, np.nan))

    def _blend(self, w_code, w_ops):
        r = self.router
        return (w_code * r.stage1_anchor("Code") + w_ops * r.stage1_anchor("Ops")).astype(np.float32)

    def test_fuse_hierarchical_gates_toward_routed_domain(self):
        state = self._blend(1.0, 0.4)
        cores = [
            _core("Code", 0.8, {"BUY": 0.9, "SELL": 0.05, "HOLD": 0.05}),
            _core("Ops", 0.8, {"BUY": 0.05, "SELL": 0.9, "HOLD": 0.05}),
        ]
        decision = self.layer.fuse_hierarchical(state, cores, CANDIDATES)
        self.assertGreater(decision.gating_weights["Code"], decision.gating_weights["Ops"])
        self.assertEqual(decision.best_action, "BUY")

    def test_fuse_hierarchical_confidence_weighting(self):
        # Equal domain posterior: only the Bayesian confidence C_k breaks the tie.
        state = self._blend(1.0, 1.0)
        cores = [
            _core("Code", 0.05, {"BUY": 0.9, "SELL": 0.05, "HOLD": 0.05}),
            _core("Ops", 0.95, {"BUY": 0.05, "SELL": 0.9, "HOLD": 0.05}),
        ]
        decision = self.layer.fuse_hierarchical(state, cores, CANDIDATES)
        self.assertAlmostEqual(decision.gating_weights["Code"], decision.gating_weights["Ops"], places=6)
        self.assertEqual(decision.best_action, "SELL")

    def test_fuse_hierarchical_empty_inputs_abstain(self):
        state = self.router.stage1_anchor("Code").astype(np.float32)
        decision = self.layer.fuse_hierarchical(state, [], CANDIDATES)
        self.assertEqual(decision.best_action, "ABSTAIN")
        self.assertTrue(decision.fallback_escalated)


if __name__ == "__main__":
    unittest.main()
