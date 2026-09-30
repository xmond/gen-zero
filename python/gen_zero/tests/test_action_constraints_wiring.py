"""Caller action constraints wired into client.decide, the nanocore orchestrator and MCP `zero`.

Covers (T32 / audit A8):
- projection maths (exact zeros, unit sum, bounds, fail-closed statuses),
- production entry points really apply the constraints,
- `formal_verification` never claims a proof when OR-Tools is missing.
"""

import asyncio
import importlib.util
import json
import random
import sys
import unittest
from unittest.mock import patch

from gen_zero.client import GenZero
from gen_zero.gate.action_constraints import (
    PROJ_ALL_ZERO_INPUT,
    PROJ_INFEASIBLE_BOUNDS,
    PROJ_INSUFFICIENT_SUPPORT,
    PROJ_OK,
    parse_action_constraints,
    project_distribution,
)
from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler
from gen_zero.mcp.server import execute_zero
from gen_zero.nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator
from gen_zero.planner.engines.cpsat_formal_engine import CpSatFormalEngine

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
NO_ORTOOLS = {"ortools": None, "ortools.sat": None, "ortools.sat.python": None}
CANDS = ["reboot", "format", "wait"]
MUTEX = {"type": "mutually_exclusive", "actions": ["reboot", "format"]}


def project(specs, probs):
    return project_distribution(parse_action_constraints(specs), probs)


class TestSpecValidation(unittest.TestCase):
    def test_malformed_specs_raise(self):
        bad = [
            "reboot",
            {"type": "forbid", "actions": ["a"]},  # a dict, not a list
            [{"type": "nope", "actions": ["a", "b"]}],
            [{"actions": ["a", "b"]}],
            [{"type": "mutually_exclusive", "actions": ["a"]}],
            [{"type": "mutually_exclusive", "actions": ["a", "A"]}],
            [{"type": "mutually_exclusive", "actions": "ab"}],
            [{"type": "mutually_exclusive", "action": ["a", "b"]}],  # typo'd key
            [{"type": "forbid", "actions": ["a"], "extra": 1}],
            [{"type": "upper_bound", "action": "a", "value": 1.5}],
            [{"type": "upper_bound", "action": "a", "value": True}],
            [{"type": "lower_bound", "action": "a", "value": float("nan")}],
            [{"type": "lower_bound", "action": "", "value": 0.1}],
        ]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_action_constraints(spec)

    def test_empty_list_is_no_constraints(self):
        self.assertEqual(parse_action_constraints([]), ())


class TestProjection(unittest.TestCase):
    def test_mutex_zeroes_loser_exactly_and_renormalises(self):
        r = project([MUTEX], {"reboot": 0.3, "format": 0.5, "wait": 0.2})
        self.assertEqual(r.status, PROJ_OK)
        self.assertEqual(r.probs["reboot"], 0.0)
        self.assertAlmostEqual(r.probs["format"], 0.5 / 0.7, places=12)
        self.assertAlmostEqual(r.probs["wait"], 0.2 / 0.7, places=12)
        self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=12)
        self.assertEqual(r.mutex_dropped, ["reboot"])

    def test_forbid_zero_exact_and_case_insensitive(self):
        r = project([{"type": "forbid", "actions": [" FORMAT "]}], {"reboot": 0.2, "format": 0.7, "wait": 0.1})
        self.assertEqual(r.probs["format"], 0.0)
        self.assertAlmostEqual(r.probs["reboot"], 2 / 3, places=12)

    def test_overlapping_mutex_groups(self):
        specs = [
            {"type": "mutually_exclusive", "actions": ["a", "b"]},
            {"type": "mutually_exclusive", "actions": ["b", "c"]},
        ]
        r = project(specs, {"a": 0.4, "b": 0.35, "c": 0.25})
        self.assertEqual((r.probs["b"], r.probs["c"] > 0.0, r.probs["a"] > 0.0), (0.0, True, True))
        # b (higher than c) is visited second and loses to a; c is then free.
        self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=12)

    def test_upper_bound_caps_and_lifts_the_rest(self):
        r = project([{"type": "upper_bound", "action": "reboot", "value": 0.3}], {"reboot": 0.6, "format": 0.2, "wait": 0.2})
        self.assertAlmostEqual(r.probs["reboot"], 0.3, places=12)
        self.assertAlmostEqual(r.probs["format"], 0.35, places=12)
        self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=12)

    def test_lower_bound_floors(self):
        r = project([{"type": "lower_bound", "action": "wait", "value": 0.4}], {"reboot": 0.6, "format": 0.35, "wait": 0.05})
        self.assertAlmostEqual(r.probs["wait"], 0.4, places=12)
        self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=12)

    def test_random_constraints_hold_invariants(self):
        rng = random.Random(7)
        checked = 0
        for _ in range(300):
            n = rng.randint(2, 7)
            names = [f"a{i}" for i in range(n)]
            q = {a: rng.random() + 0.01 for a in names}
            specs = []
            for a in rng.sample(names, rng.randint(0, n - 1)):
                specs.append({"type": rng.choice(["upper_bound", "lower_bound"]), "action": a, "value": round(rng.uniform(0, 0.6), 3)})
            if n >= 3 and rng.random() < 0.7:
                specs.append({"type": "mutually_exclusive", "actions": rng.sample(names, 2)})
            r = project(specs, q)
            if r.status != PROJ_OK:
                self.assertIn(r.status, (PROJ_INFEASIBLE_BOUNDS, PROJ_INSUFFICIENT_SUPPORT))
                self.assertTrue(all(v == 0.0 for v in r.probs.values()))
                continue
            checked += 1
            self.assertAlmostEqual(sum(r.probs.values()), 1.0, places=9)
            for s in parse_action_constraints(specs):
                if s.kind == "upper_bound":
                    self.assertLessEqual(r.probs[s.actions[0].lower()], s.value + 1e-9)
                if s.kind == "lower_bound":
                    self.assertGreaterEqual(r.probs[s.actions[0].lower()], s.value - 1e-9)
                if s.kind == "mutually_exclusive":
                    live = [a for a in s.actions if r.probs[a.lower()] > 0.0]
                    self.assertLessEqual(len(live), 1)
        self.assertGreater(checked, 100)

    def test_fail_closed_statuses(self):
        self.assertEqual(project([], {"a": 0.0, "b": 0.0}).status, PROJ_ALL_ZERO_INPUT)
        # floors 0.6 + 0.6 > 1
        two_floors = [{"type": "lower_bound", "action": "a", "value": 0.6}, {"type": "lower_bound", "action": "b", "value": 0.6}]
        self.assertEqual(project(two_floors, {"a": 0.5, "b": 0.5, "c": 0.0}).status, PROJ_INFEASIBLE_BOUNDS)
        # all mass sits on the forbidden action: no mass is invented for the zero-probability ones
        r = project([{"type": "forbid", "actions": ["a"]}], {"a": 1.0, "b": 0.0})
        self.assertEqual(r.status, PROJ_INSUFFICIENT_SUPPORT)
        self.assertTrue(all(v == 0.0 for v in r.probs.values()))
        # a floor cannot lift an action the planner gave probability 0 (e.g. safety-pruned)
        r = project([{"type": "lower_bound", "action": "b", "value": 0.2}], {"a": 1.0, "b": 0.0})
        self.assertEqual(r.status, PROJ_INSUFFICIENT_SUPPORT)
        self.assertTrue(all(v == 0.0 for v in r.probs.values()))
        # a required floor on a non-candidate can never be met
        r = project([{"type": "lower_bound", "action": "ghost", "value": 0.1}], {"a": 1.0})
        self.assertEqual(r.status, PROJ_INFEASIBLE_BOUNDS)

    def test_unmatched_actions_are_reported_not_hidden(self):
        r = project([{"type": "forbid", "actions": ["reeboot"]}], {"reboot": 0.5, "wait": 0.5})
        self.assertEqual(r.status, PROJ_OK)
        self.assertEqual(r.unmatched_actions, ["REEBOOT"])

    def test_candidate_name_collision_raises(self):
        with self.assertRaises(ValueError):
            project([], {"Reboot": 0.5, "reboot ": 0.5})


class TestClientDecide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GenZero(action_effects={**{a: "EXECUTE" for a in CANDS},
                                       "approve": "EXECUTE", "hold": "READ_ONLY"})

    def _base(self):
        return self.g.decide("state", CANDS, mode="reflex")

    def test_no_constraints_reports_not_requested(self):
        base = self._base()
        self.assertEqual(base["formal_verification"], "not_requested")
        self.assertIsNone(base["constraint_projection"])

    def test_forbid_blocks_the_unconstrained_winner(self):
        base = self._base()
        top = base["action"]
        self.assertGreater(base["probs"][top], 0.0)
        res = self.g.decide("state", CANDS, mode="reflex", constraints=[{"type": "forbid", "actions": [top]}])
        self.assertNotEqual(res["action"], top)
        self.assertEqual(res["probs"][top], 0.0)
        self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=3)
        self.assertEqual(res["confidence"], res["probs"][res["action"]])

    def test_mutex_blocks_the_loser_and_keeps_the_winner(self):
        base = self._base()
        ranked = sorted(CANDS, key=lambda c: -base["probs"][c])
        winner, loser = ranked[0], ranked[1]
        res = self.g.decide(
            "state", CANDS, mode="reflex",
            constraints=[{"type": "mutually_exclusive", "actions": [loser, winner]}],
        )
        self.assertEqual(res["probs"][loser], 0.0)
        self.assertNotEqual(res["action"], loser)
        self.assertEqual(res["action"], winner)
        self.assertEqual(res["constraint_projection"]["mutex_dropped"], [loser])
        self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=3)

    def test_upper_bound_can_flip_the_action(self):
        base = self._base()
        top = base["action"]
        res = self.g.decide("state", CANDS, mode="reflex", constraints=[{"type": "upper_bound", "action": top, "value": 0.05}])
        self.assertLessEqual(res["probs"][top], 0.05 + 1e-4)
        self.assertNotEqual(res["action"], top)

    def test_unsatisfiable_constraints_abstain(self):
        specs = [{"type": "lower_bound", "action": "reboot", "value": 0.7}, {"type": "lower_bound", "action": "format", "value": 0.7}]
        res = self.g.decide("state", CANDS, mode="reflex", constraints=specs)
        self.assertEqual(res["action"], "ABSTAIN")
        self.assertTrue(all(v == 0.0 for v in res["probs"].values()))
        self.assertTrue(res["constraint_projection"]["status"].startswith("CONSTRAINTS_UNSATISFIABLE"))

    def test_lower_bound_cannot_resurrect_a_safety_pruned_action(self):
        # The default hard rule prunes 'approve' for an unauthorized state; a caller floor must not undo it.
        cands = ["approve", "wait", "hold"]
        state = "unauthorized request"
        res = self.g.decide(state, cands, mode="reflex", constraints=[{"type": "lower_bound", "action": "approve", "value": 0.3}])
        self.assertEqual(res["probs"]["approve"], 0.0)
        self.assertEqual(res["action"], "ABSTAIN")
        self.assertTrue(res["constraint_projection"]["status"].startswith("CONSTRAINTS_UNSATISFIABLE"))

    def test_malformed_constraints_raise_before_planning(self):
        with self.assertRaises(ValueError):
            self.g.decide("state", CANDS, mode="reflex", constraints=[{"type": "mutually_exclusive", "actions": ["reboot"]}])
        with self.assertRaises(ValueError):
            self.g.decide("state", CANDS, mode="reflex", constraints="forbid reboot")

    def test_missing_ortools_is_reported_never_faked(self):
        with patch.dict(sys.modules, NO_ORTOOLS):
            res = self.g.decide("state", CANDS, mode="reflex", constraints=[MUTEX])
        self.assertEqual(res["formal_verification"], "unavailable_missing_dependency")
        self.assertEqual(res["cpsat_solver_status"], "ortools_unavailable")
        # the gate itself is pure numpy and still applies
        self.assertTrue(any(v == 0.0 for v in (res["probs"][c] for c in ("reboot", "format"))))

    @unittest.skipUnless(HAS_ORTOOLS, "ortools not installed in this environment")
    def test_ortools_present_runs_real_cp_sat(self):
        res = self.g.decide("state", CANDS, mode="reflex", constraints=[MUTEX])
        self.assertEqual(res["formal_verification"], "cp_sat_verified")
        self.assertEqual(res["cpsat_solver_status"], "python_predicate_filter")

    def test_create_constraint_compiler_is_the_compiler_used(self):
        compiler = self.g.create_constraint_compiler()
        self.assertIsInstance(compiler, ConstraintLinearProjectionCompiler)
        self.assertEqual(compiler.compile_action_constraints([MUTEX]), 1)
        self.assertEqual(compiler.num_action_constraints, 1)


class TestPlannerEngineHonesty(unittest.TestCase):
    def test_engine_never_claims_cp_sat_without_ortools(self):
        with patch.dict(sys.modules, NO_ORTOOLS):
            res = CpSatFormalEngine().verify_and_prune("s", ["a", "b"])
            empty = CpSatFormalEngine().verify_and_prune("s", [])
        self.assertEqual(res["solver_status"], "ortools_unavailable")
        self.assertEqual(empty["solver_status"], "ortools_unavailable")

    @unittest.skipUnless(HAS_ORTOOLS, "ortools not installed in this environment")
    def test_engine_with_ortools_still_says_predicate_filter(self):
        res = CpSatFormalEngine().verify_and_prune("s", ["a", "b"])
        self.assertEqual(res["solver_status"], "python_predicate_filter")


class TestOrchestrator(unittest.TestCase):
    SAFE = staticmethod(lambda z, a: 0.9)

    def _run(self, constraints=None):
        # A fresh orchestrator per call: it carries latent state between steps.
        return WorldModelNanoCoreOrchestrator().imagine_and_orchestrate(
            "s", CANDS, safety_evaluator=self.SAFE, constraints=constraints
        )

    def test_forbid_blocks_the_selected_action(self):
        top = self._run().selected_action
        res = self._run([{"type": "forbid", "actions": [top]}])
        self.assertNotEqual(res.selected_action, top)
        self.assertIn(top, res.constraint_projection["disabled"])

    def test_mutex_drops_lower_scoring_member(self):
        orch = WorldModelNanoCoreOrchestrator()
        orig_step = orch.transition_model.step

        def step_mock(z, act, preserve_entropy=False):
            nz, r, info = orig_step(z, act, preserve_entropy=preserve_entropy)
            bonus = 1.0 if act == "format" else 0.0
            return nz, r + bonus, info

        with patch.object(orch.transition_model, "step", side_effect=step_mock):
            res = orch.imagine_and_orchestrate(
                "s", CANDS, safety_evaluator=self.SAFE,
                constraints=[{"type": "mutually_exclusive", "actions": ["reboot", "format"]}],
            )
            score = {t.action: t.predicted_value for t in res.imagined_trajectory}
            loser = min(("reboot", "format"), key=lambda c: score[c])
            self.assertEqual(loser, "reboot")
            self.assertEqual(res.constraint_projection["mutex_dropped"], ["reboot"])
            self.assertNotEqual(res.selected_action, "reboot")

    def test_bounds_and_malformed_rejected_before_state_changes(self):
        orch = WorldModelNanoCoreOrchestrator()
        for bad in ([{"type": "upper_bound", "action": "reboot", "value": 0.3}], [{"type": "forbid"}]):
            with self.assertRaises(ValueError):
                orch.imagine_and_orchestrate("s", CANDS, safety_evaluator=self.SAFE, constraints=bad)
        self.assertEqual(orch.step_count, 0)

    def test_missing_ortools_flag(self):
        with patch.dict(sys.modules, NO_ORTOOLS):
            res = self._run([MUTEX])
        self.assertEqual(res.formal_verification, "unavailable_missing_dependency")
        self.assertEqual(res.to_dict()["formal_verification"], "unavailable_missing_dependency")

    @unittest.skipUnless(HAS_ORTOOLS, "ortools not installed in this environment")
    def test_real_cp_sat_verifies_selection(self):
        res = self._run([MUTEX])
        self.assertEqual(res.formal_verification, "cp_sat_verified")

    def test_no_constraints_not_requested(self):
        self.assertEqual(self._run().formal_verification, "not_requested")


class TestMCP(unittest.IsolatedAsyncioTestCase):
    async def _call(self, args):
        res = await execute_zero(args)
        return res["isError"], res["content"][0]["text"]

    async def test_imagine_accepts_constraints_and_reports_them(self):
        with patch.dict(sys.modules, NO_ORTOOLS):
            is_err, text = await self._call(
                {"state": "s", "candidate_actions": CANDS, "constraints": [MUTEX]}
            )
        self.assertTrue(is_err)
        envelope = json.loads(text)
        self.assertEqual(envelope["error"], "SAFETY_INTERLOCKED")
        self.assertIn("result", envelope)
        self.assertEqual(envelope["result"]["formal_verification"], "unavailable_missing_dependency")

    async def test_imagine_rejects_malformed_constraints(self):
        is_err, text = await self._call(
            {"state": "s", "candidate_actions": CANDS, "constraints": [{"type": "mutually_exclusive", "actions": ["x"]}]}
        )
        self.assertTrue(is_err)
        self.assertIn("constraints", text)

    async def test_other_modes_refuse_instead_of_silently_dropping(self):
        for args in (
            {"action": "decide", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": ["c"]}}, "constraints": [MUTEX]},
            {"action": "stream", "observation": "o", "candidate_actions": CANDS, "constraints": [MUTEX]},
            {"action": "route", "task_goal": "g", "tools": [], "constraints": [MUTEX]},
        ):
            with self.subTest(action=args["action"]):
                is_err, text = await self._call(args)
                self.assertTrue(is_err)
                self.assertIn("not supported", text)


if __name__ == "__main__":
    unittest.main()
