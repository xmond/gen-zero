"""B01/B16/B14 result-contract tests for the HTTP service layer.

B01: /v1/decisions must never repackage an internal refusal (ABSTAIN, missing probs,
non-finite probs) as a normal successful answer -- no uniform-distribution fabrication,
no `cands[0]` fallback, no `1.0 - prob_true` reconstruction of an abstained noul.
B16: the trailing trajectory rollout inside GenZero.decide() must not force a full
second decision pass when only the bonus rollout fails.
B14: /v1/decide_step must reject an unknown `risk_profile` with 400 instead of silently
falling back to the default gate profile.
"""
import math
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gen_zero.service import app as service_app
from gen_zero.service.app import app


TOKEN = "b0928-contract-test-token"


def _abstain_result(status_code="INFEASIBLE_ABSTAIN", candidates=("true", "false")):
    return {
        "action": "ABSTAIN",
        "confidence": 0.0,
        "probs": {c: 0.0 for c in candidates},
        "value": -10.0,
        "status": status_code,
        "mode": "reflex",
        "latency_ms": 0.1,
    }


class _EnvAndGateMixin:
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.get("GENZERO_API_KEY")
        os.environ["GENZERO_API_KEY"] = TOKEN

    @classmethod
    def tearDownClass(cls):
        if cls._orig_env is not None:
            os.environ["GENZERO_API_KEY"] = cls._orig_env
        else:
            os.environ.pop("GENZERO_API_KEY", None)

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {TOKEN}"}
        self._gate_patch = patch.object(service_app.client, "weights_loaded_from_checkpoint", True)
        self._gate_patch.start()
        self.addCleanup(self._gate_patch.stop)


class TestB01NoulAbstainNotFabricated(_EnvAndGateMixin, unittest.TestCase):
    def test_single_state_abstain_kept_explicit(self):
        payload = {
            "state": "reactor core temperature unreadable",
            "questions": {"safe": {"type": "noul", "criteria": {"true": "safe", "false": "unsafe"}}},
        }
        with patch.object(service_app.client, "decide", return_value=_abstain_result()):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["safe"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "INFEASIBLE_ABSTAIN")
        self.assertNotIn("noul", ans)
        self.assertNotIn("confidence", ans)

    def test_batch_abstain_kept_explicit(self):
        payload = {
            "states": ["reactor A", "reactor B"],
            "questions": {"safe": {"type": "noul", "criteria": {"true": "safe", "false": "unsafe"}}},
        }
        with patch.object(
            service_app.client, "decide_batch",
            return_value=[_abstain_result(), _abstain_result()],
        ):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        for result in resp.json()["results"]:
            ans = result["answers"]["safe"]
            self.assertEqual(ans["status"], "ABSTAIN")
            self.assertNotIn("noul", ans)

    def test_noul_never_reconstructs_1_minus_prob_true_from_abstain(self):
        """The exact P0 bug: an internal {'true': 0, 'false': 0} abstain must never turn
        into a fabricated {'true': 0, 'false': 1} via `1.0 - prob_true`."""
        payload = {
            "state": "ambiguous",
            "questions": {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
        }
        with patch.object(service_app.client, "decide", return_value=_abstain_result()):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertNotIn("noul", ans)

    def test_noul_missing_false_prob_flagged_not_fabricated(self):
        """The kernel omitted 'false' entirely; must not become `1.0 - prob_true`."""
        bad = {
            "action": "true",
            "confidence": 0.9,
            "probs": {"true": 0.7},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
        }
        with patch.object(service_app.client, "decide", return_value=bad):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "MISSING_NOUL_PROB")
        self.assertNotIn("noul", ans)

    def test_noul_unnormalized_probs_flagged_not_used_directly(self):
        """true+false != 1: using 'false' as-is (or 1 - prob_true) would both be wrong;
        the kernel's own inconsistency must surface as an abstain, not a guess."""
        bad = {
            "action": "true",
            "confidence": 0.9,
            "probs": {"true": 0.7, "false": 0.7},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
        }
        with patch.object(service_app.client, "decide", return_value=bad):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "UNNORMALIZED_NOUL_PROBS")
        self.assertNotIn("noul", ans)

    def test_non_finite_probs_flagged_not_averaged(self):
        bad = {
            "action": "true",
            "confidence": 0.5,
            "probs": {"true": float("nan"), "false": 0.5},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
        }
        with patch.object(service_app.client, "decide", return_value=bad):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "NON_FINITE_PROBS")
        self.assertNotIn("noul", ans)


class TestDecideNoulProbsHelper(unittest.TestCase):
    """Direct unit coverage of the shared helper both noul call sites route through."""

    def test_success_sums_to_one(self):
        prob_true, prob_false, err = service_app._decide_noul_probs({"probs": {"true": 0.7, "false": 0.3}})
        self.assertEqual(err, {})
        self.assertEqual((prob_true, prob_false), (0.7, 0.3))

    def test_missing_false_key(self):
        _, _, err = service_app._decide_noul_probs({"probs": {"true": 0.7}})
        self.assertEqual(err["status"], "MISSING_NOUL_PROB")

    def test_missing_true_key(self):
        _, _, err = service_app._decide_noul_probs({"probs": {"false": 0.3}})
        self.assertEqual(err["status"], "MISSING_NOUL_PROB")

    def test_does_not_sum_to_one(self):
        _, _, err = service_app._decide_noul_probs({"probs": {"true": 0.7, "false": 0.7}})
        self.assertEqual(err["status"], "UNNORMALIZED_NOUL_PROBS")

    def test_real_false_diverging_from_1_minus_prob_true_is_caught(self):
        """The exact regression this helper exists to prevent: the kernel's real false=0.1
        disagrees with `1.0 - prob_true` (0.3) -- that disagreement must abstain, not silently
        pick whichever number `1.0 - prob_true` would have produced."""
        _, _, err = service_app._decide_noul_probs({"probs": {"true": 0.7, "false": 0.1}})
        self.assertEqual(err["status"], "UNNORMALIZED_NOUL_PROBS")


class TestB01ChoiceAbstainNotFabricated(_EnvAndGateMixin, unittest.TestCase):
    def test_single_state_choice_no_cands0_fallback(self):
        payload = {
            "state": "s",
            "questions": {"route": {"type": "choice", "criteria": {"x": "do x", "y": "do y"}}},
        }
        with patch.object(service_app.client, "decide", return_value=_abstain_result(candidates=("x", "y"))):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["route"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertNotIn("choice", ans)

    def test_batch_choice_no_uniform_fabrication(self):
        missing_probs = {"action": "ABSTAIN", "confidence": 0.0, "probs": {}, "mode": "reflex", "latency_ms": 0.1}
        payload = {
            "states": ["s1"],
            "questions": {"route": {"type": "choice", "criteria": {"x": "do x", "y": "do y"}}},
        }
        with patch.object(service_app.client, "decide_batch", return_value=[missing_probs]):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["results"][0]["answers"]["route"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertNotIn("probabilities", ans)


class TestB01ScoreAbstainNotFabricated(_EnvAndGateMixin, unittest.TestCase):
    def test_score_missing_probs_flagged(self):
        missing_probs = {"action": "ABSTAIN", "confidence": 0.0, "probs": {}, "mode": "reflex", "latency_ms": 0.1}
        payload = {
            "state": "s",
            "questions": {"sev": {"type": "score", "criteria": ["Low", "Medium", "High"]}},
        }
        with patch.object(service_app.client, "decide", return_value=missing_probs):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["sev"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertNotIn("score", ans)


class TestB01NormalAnswersUnaffected(_EnvAndGateMixin, unittest.TestCase):
    """A genuine successful decide() result must still be formatted normally (no
    regression from the abstain-detection gate added ahead of every branch)."""

    def test_noul_success_path(self):
        good = {
            "action": "true",
            "confidence": 0.9,
            "probs": {"true": 0.7, "false": 0.3},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
        }
        with patch.object(service_app.client, "decide", return_value=good):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertNotIn("status", ans)
        self.assertEqual(ans["noul"], 0.7)

    def test_choice_success_path(self):
        good = {
            "action": "x",
            "confidence": 0.8,
            "probs": {"x": 0.8, "y": 0.2},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"route": {"type": "choice", "criteria": {"x": "do x", "y": "do y"}}},
        }
        with patch.object(service_app.client, "decide", return_value=good):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["route"]
        self.assertNotIn("status", ans)
        self.assertEqual(ans["choice"], "x")

    def test_choice_extra_candidate_rejected(self):
        """Kernel returned an unexpected candidate outside admissible set; must ABSTAIN."""
        bad = {
            "action": "a",
            "confidence": 0.9,
            "probs": {"a": 0.4, "b": 0.4, "alien": 0.2},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}},
        }
        with patch.object(service_app.client, "decide", return_value=bad):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "EXTRA_CANDIDATE_PROBS")

    def test_choice_action_not_in_candidates_rejected(self):
        """Kernel picked an action that wasn't in candidates; must ABSTAIN."""
        bad = {
            "action": "alien",
            "confidence": 0.9,
            "probs": {"a": 0.5, "b": 0.5},
            "mode": "reflex",
            "latency_ms": 0.1,
        }
        payload = {
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}},
        }
        with patch.object(service_app.client, "decide", return_value=bad):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        ans = resp.json()["answers"]["q"]
        self.assertEqual(ans["status"], "ABSTAIN")
        self.assertEqual(ans["kernel_status"], "ACTION_NOT_IN_CANDIDATES")

    def test_real_kernel_end_to_end_all_three_types(self):
        """End-to-end integration: exercise the actual reflex kernel format through all 3 question types."""
        payload = {
            "state": "CPU at 96% with connection timeouts",
            "questions": {
                "escalate": {"type": "noul", "criteria": {"true": "page", "false": "wait"}},
                "route": {"type": "choice", "criteria": {"scale": "scale out", "alert": "alert oncall"}},
                "severity": {"type": "score", "criteria": ["Low", "Medium", "High"]},
            },
        }
        resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        answers = resp.json()["answers"]
        for q_id in ("escalate", "route", "severity"):
            self.assertIn(q_id, answers)
            ans = answers[q_id]
            if ans.get("status") == "ABSTAIN":
                continue
            self.assertIn("confidence", ans)
            self.assertTrue(math.isfinite(ans["confidence"]))
        # The reflex kernel normalizes noul over exactly ['true', 'false'], so a real
        # (unmocked) call must answer it, never silently ABSTAIN via UNNORMALIZED_NOUL_PROBS
        # or MISSING_NOUL_PROB -- if this ever fires, the kernel's own normalization broke,
        # and that must be visible as a test failure, not swallowed as "well, it abstained".
        self.assertNotEqual(answers["escalate"].get("status"), "ABSTAIN", answers["escalate"])
        self.assertIn("noul", answers["escalate"])


class TestB14UnknownRiskProfileRejected(_EnvAndGateMixin, unittest.TestCase):
    def test_unknown_risk_profile_is_400_not_silent_default(self):
        payload = {
            "state": "page loaded",
            "affordances": ["e1", "e2"],
            "goal": "click login",
            "risk_profile": "totally_bogus_profile",
        }
        resp = self.client.post("/v1/decide_step", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 400, resp.text)
        err = resp.json()["detail"]["error"]
        self.assertEqual(err["code"], "invalid_risk_profile")

    def test_missing_risk_profile_still_uses_default(self):
        payload = {"state": "page loaded", "affordances": ["e1", "e2"], "goal": "click login"}
        resp = self.client.post("/v1/decide_step", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_valid_risk_profile_still_works(self):
        payload = {
            "state": "page loaded",
            "affordances": ["e1", "e2"],
            "goal": "click login",
            "risk_profile": "read_only",
        }
        resp = self.client.post("/v1/decide_step", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)


class TestB16RolloutFailureScopedNotWholeRetry(unittest.TestCase):
    """GenZero.decide()'s trailing rollout ValueError must not force a second full
    decision pass (double CP-SAT solve, double arbiter call, etc)."""

    def test_decide_calls_cpsat_exactly_once_when_rollout_raises(self):
        from gen_zero.client import GenZero, UnsupportedStateError

        gz = GenZero()
        original_verify = gz.cp_sat_solver.verify_and_prune
        call_count = {"n": 0}

        def counting_verify(*args, **kwargs):
            call_count["n"] += 1
            return original_verify(*args, **kwargs)

        with patch.object(gz.cp_sat_solver, "verify_and_prune", side_effect=counting_verify):
            with patch("gen_zero.client.rollout", side_effect=UnsupportedStateError("unsupported shape")):
                res = gz.decide(
                    state="some free text state",
                    candidates=["a", "b"],
                    mode="reflex",
                    return_trajectory=True,
                )

        self.assertEqual(call_count["n"], 1, "decide() re-ran the whole pipeline instead of scoping the rollout failure")
        self.assertIsNone(res["trajectory"])
        self.assertIn("UNSUPPORTED_STATE_FOR_TRAJECTORY", res["trajectory_status"])
        self.assertIn(res["action"], ("a", "b", "ABSTAIN"))

    def test_decide_without_trajectory_unaffected(self):
        from gen_zero.client import GenZero

        gz = GenZero()
        res = gz.decide(state="some free text state", candidates=["a", "b"], mode="reflex")
        self.assertIn(res["action"], ("a", "b", "ABSTAIN"))
        self.assertNotIn("trajectory", res)

    def test_plain_value_error_from_rollout_propagates_not_swallowed(self):
        """A ValueError that is NOT UnsupportedStateError (a NaN reward, a bad horizon, a
        broken transition_fn) is a real contract violation, not a state/schema mismatch.
        decide() must let it propagate instead of repackaging it as trajectory_status."""
        from gen_zero.client import GenZero

        gz = GenZero()
        with patch("gen_zero.client.rollout", side_effect=ValueError("transition returned a non-finite reward: nan")):
            with self.assertRaises(ValueError):
                gz.decide(
                    state="some free text state",
                    candidates=["a", "b"],
                    mode="reflex",
                    return_trajectory=True,
                )


if __name__ == "__main__":
    unittest.main()
