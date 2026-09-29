"""Regression contract for ChatGPT 6 Pro round-2 review findings C07 and X-C01..X-C04.

C07:    project_latent_propositions and evaluate_condition/solve_safest_action must
        agree on every compound (AND) proposition's truth value -- never "projection
        reads the primary term, full evaluation reads all terms".
X-C01:  a z_latent requires an explicit schema_fingerprint matching the compiler's
        current variable->coordinate map; a missing or stale one fails closed
        (SchemaMismatchError), never a silent reinterpretation under a shifted map.
X-C02:  a z_latent shorter than a variable's bound coordinate must resolve that
        variable's predicates to Tristate.UNKNOWN, never zero-pad the missing
        dimension.
X-C03:  a threshold literal that overflows to +/-inf (e.g. `1e309`) must be
        rejected at compile time (UNSUPPORTED_SYNTAX), never silently become an
        always-false/always-true no-op rule.
X-C04:  a large integer threshold/metric near or beyond 2**53-1 must be compared
        exactly (no float() narrowing that flips the comparison result).
"""
import math

import numpy as np
import pytest

from gen_zero.gate.constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    SchemaMismatchError,
    Tristate,
)


def make_compiler(rules, latent_dim=8, hard_timeout_ms=100.0):
    c = ConstraintLinearProjectionCompiler(latent_dim=latent_dim, hard_timeout_ms=hard_timeout_ms)
    c.compile_rules(list(rules))
    return c


# ---------------------------------------------------------------------------
# C07: unified AND-of-comparisons semantics between projection and full eval.
# ---------------------------------------------------------------------------

class TestC07CompoundPredicateUnifiedSemantics:
    def test_project_and_full_eval_agree_when_second_term_is_false(self):
        """The exact reported counterexample: rule 'x > 0 and y > 0', x=1, y=-1.
        Before the fix, project_latent_propositions read only 'x > 0' (True) while
        evaluate_condition (via current_metrics) read both terms (False)."""
        c = make_compiler(["FORBID DANGER IF x > 0 and y > 0"])
        rule = c._rules[0]

        # Full evaluation via an explicit metrics dict (schema-independent).
        full_eval = c.evaluate_condition(rule, current_metrics={"x": 1.0, "y": -1.0})
        assert full_eval is False  # y > 0 is false, so the AND is false -> not active

        # Projection via the equivalent schema-bound latent vector (x=dim0, y=dim1).
        fp = c.schema_fingerprint
        z = np.array([1.0, -1.0])
        proj = c.project_latent_propositions(z, schema_fingerprint=fp)
        assert proj["x > 0 and y > 0"] is Tristate.FALSE  # must agree with full_eval, not TRUE

    def test_project_and_full_eval_agree_when_both_terms_true(self):
        c = make_compiler(["FORBID DANGER IF x > 0 and y > 0"])
        rule = c._rules[0]
        assert c.evaluate_condition(rule, current_metrics={"x": 1.0, "y": 1.0}) is True

        fp = c.schema_fingerprint
        z = np.array([1.0, 1.0])
        proj = c.project_latent_propositions(z, schema_fingerprint=fp)
        assert proj["x > 0 and y > 0"] is Tristate.TRUE

    def test_solve_safest_action_uses_full_compound_from_latent(self):
        """The compiled-rule solver (z_latent path) must forbid using ALL terms,
        not just the primary one."""
        c = make_compiler(["FORBID DANGER IF x > 0 and y > 0"])
        fp = c.schema_fingerprint

        # y <= 0: the AND is false, DANGER must NOT be forbidden.
        v_safe = c.solve_safest_action(
            {"DANGER": 10.0, "HOLD": 1.0}, z_latent=np.array([1.0, -1.0]), schema_fingerprint=fp
        )
        assert v_safe.is_safe and v_safe.selected_action == "DANGER"

        # Both > 0: the AND is true, DANGER must be forbidden.
        v_blocked = c.solve_safest_action(
            {"DANGER": 10.0, "HOLD": 1.0}, z_latent=np.array([1.0, 1.0]), schema_fingerprint=fp
        )
        assert v_blocked.is_safe and v_blocked.selected_action == "HOLD"

    def test_unsupported_or_returns_unknown_not_a_fabricated_boolean(self):
        """A genuinely unrepresentable compound (internal OR) must resolve to
        UNKNOWN in projection space too, never silently drop to a primary term."""
        c = make_compiler(["FORBID DANGER IF x > 0 or y > 0"])
        rule = c._rules[0]
        assert rule.rule_type == "UNSUPPORTED_SYNTAX"
        assert rule.sub_conditions == []
        fp = c.schema_fingerprint
        proj = c.project_latent_propositions(np.array([1.0, 1.0]), schema_fingerprint=fp)
        # The proposition key is the raw unsupported condition_expr; its only
        # entry must abstain, not silently evaluate the "x > 0" fragment.
        assert list(proj.values()) == [Tristate.UNKNOWN]


# ---------------------------------------------------------------------------
# X-C01: schema fingerprint binding.
# ---------------------------------------------------------------------------

class TestXC01SchemaFingerprintBinding:
    def test_reordering_coordinates_without_fingerprint_fails_closed(self):
        """The exact reported counterexample: a rule set with only 'x' is
        recompiled with an alphabetically-earlier 'a', shifting x from
        coordinate 0 to coordinate 1. The same raw vector z=[1, -1] must never
        be silently reinterpreted under the new map."""
        c = make_compiler(["FORBID DANGER IF x > 0"])
        assert c._variable_dimensions == {"x": 0}
        old_fp = c.schema_fingerprint
        z = np.array([1.0, -1.0])

        # Recompile: 'a' sorts before 'x', so x shifts from coordinate 0 to 1.
        c.compile_rules(["FORBID DANGER IF x > 0", "FORBID OTHER IF a > 0"])
        assert c._variable_dimensions == {"a": 0, "x": 1}
        new_fp = c.schema_fingerprint
        assert new_fp != old_fp

        # Using the stale fingerprint against the new schema fails closed.
        with pytest.raises(SchemaMismatchError):
            c.project_latent_propositions(z, schema_fingerprint=old_fp)
        # No fingerprint at all also fails closed.
        with pytest.raises(SchemaMismatchError):
            c.project_latent_propositions(z)
        # The correct, current fingerprint is required to proceed at all.
        proj = c.project_latent_propositions(z, schema_fingerprint=new_fp)
        assert proj["x > 0"] is Tristate.FALSE  # x is now z[1] = -1.0

    def test_solve_safest_action_requires_fingerprint_for_latent_only_calls(self):
        c = make_compiler(["FORBID DANGER IF x > 0"])
        z = np.array([1.0])
        with pytest.raises(SchemaMismatchError):
            c.solve_safest_action({"DANGER": 1.0, "HOLD": 0.0}, z_latent=z)

    def test_current_metrics_bypasses_fingerprint_requirement(self):
        """When current_metrics is supplied it is authoritative (schema-independent
        by construction); z_latent is only checked for NaN/Inf in that case."""
        c = make_compiler(["FORBID DANGER IF x > 0"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, z_latent=np.array([1.0]), current_metrics={"x": 1.0}
        )
        assert v.is_safe and v.selected_action == "HOLD"


# ---------------------------------------------------------------------------
# X-C02: missing latent dimensions fail closed, never zero-padded.
# ---------------------------------------------------------------------------

class TestXC02MissingDimensionFailsClosedNotZero:
    def test_short_latent_leaves_missing_variable_unknown(self):
        """Schema binds x->0, y->1. A vector of length 1 covers x but not y;
        'y == 0' must resolve to UNKNOWN, never be silently read as y=0.0 -> True."""
        c = make_compiler(["FORBID EXEC IF x > 0", "FORBID HALT IF y == 0"])
        assert c._variable_dimensions == {"x": 0, "y": 1}
        fp = c.schema_fingerprint

        short_z = np.array([1.0])  # only reaches coordinate 0 (x)
        proj = c.project_latent_propositions(short_z, schema_fingerprint=fp)
        assert proj["x > 0"] is Tristate.TRUE
        assert proj["y == 0"] is Tristate.UNKNOWN  # not Tristate.TRUE

        # The compiled-rule solver must not forbid HALT on a fabricated y=0.
        v = c.solve_safest_action(
            {"HALT": 10.0, "SAFE": 1.0}, z_latent=short_z, schema_fingerprint=fp
        )
        # FORBID_IF collapses UNKNOWN to fail-closed (forbid), so HALT is still
        # blocked -- but via the honest UNKNOWN path, not a fabricated y=0==True.
        assert v.selected_action == "SAFE"

    def test_full_length_latent_resolves_the_same_variable_normally(self):
        c = make_compiler(["FORBID HALT IF y == 0"])
        fp = c.schema_fingerprint
        z_present = np.zeros(8)
        proj = c.project_latent_propositions(z_present, schema_fingerprint=fp)
        assert proj["y == 0"] is Tristate.TRUE

    def test_orchestrator_rejects_short_raw_state_instead_of_zero_padding(self):
        """X-C02 at the production boundary: _encode_to_latent used to zero-pad a
        short raw state before handing it to the compiler, silently reopening the
        same bug one layer up (this is the real path the imagine_world_model MCP
        tool exposes to an external caller's 'state' field)."""
        from gen_zero.nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator

        orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=8, action_dim=4)

        with pytest.raises(ValueError, match="fewer than latent_dim"):
            orchestrator.imagine_and_orchestrate(
                state=[1.0, 2.0],  # a plain list, shorter than latent_dim=8
                candidate_actions=["A"],
                enforce_cpsat=False,
                safety_evaluator=lambda z, act: 0.9,
            )

        with pytest.raises(ValueError, match="fewer than latent_dim"):
            orchestrator.imagine_and_orchestrate(
                state=np.array([1.0, 2.0]),  # previously passed straight through unpadded
                candidate_actions=["A"],
                enforce_cpsat=False,
                safety_evaluator=lambda z, act: 0.9,
            )

    def test_orchestrator_accepts_full_length_raw_state(self):
        from gen_zero.nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator

        orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=8, action_dim=4)
        res = orchestrator.imagine_and_orchestrate(
            state=np.linspace(-1.0, 1.0, 8).astype(np.float32),
            candidate_actions=["A"],
            enforce_cpsat=False,
            safety_evaluator=lambda z, act: 0.9,
        )
        assert res.selected_action == "A"


# ---------------------------------------------------------------------------
# X-C03: overflowing threshold literals are rejected, not silently no-op.
# ---------------------------------------------------------------------------

class TestXC03OverflowingThresholdRejected:
    def test_1e309_threshold_is_unsupported_syntax_not_a_silent_noop(self):
        c = make_compiler(["FORBID DANGER IF risk > 1e309"])
        rule = c._rules[0]
        assert rule.rule_type == "UNSUPPORTED_SYNTAX"
        assert rule.sub_conditions == []

    def test_solve_safest_action_fails_closed_on_overflowing_threshold(self):
        c = make_compiler(["FORBID DANGER IF risk > 1e309"])
        v = c.solve_safest_action({"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"risk": 1e300})
        assert not v.is_safe
        assert v.solver_status == "UNSUPPORTED_SYNTAX_FAIL_CLOSED"

    def test_huge_int_threshold_is_also_rejected(self):
        c = make_compiler([f"FORBID DANGER IF risk > {10 ** 400}"])
        rule = c._rules[0]
        assert rule.rule_type == "UNSUPPORTED_SYNTAX"

    def test_negative_overflowing_threshold_is_rejected(self):
        c = make_compiler(["FORBID DANGER IF risk < -1e309"])
        rule = c._rules[0]
        assert rule.rule_type == "UNSUPPORTED_SYNTAX"


# ---------------------------------------------------------------------------
# X-C04: exact large-integer comparison, no float() narrowing.
# ---------------------------------------------------------------------------

class TestXC04LargeIntegerExactComparison:
    SAFE_INT = 9007199254740992  # 2**53, exactly representable in float64
    JUST_OVER = 9007199254740993  # 2**53 + 1, NOT exactly representable in float64

    def test_threshold_parsed_as_exact_int_not_narrowed_float(self):
        c = make_compiler([f"FORBID DANGER IF v >= {self.JUST_OVER}"])
        rule = c._rules[0]
        assert rule.threshold == self.JUST_OVER
        assert isinstance(rule.threshold, int)  # never silently cast to float

    def test_metric_one_below_threshold_is_false_not_narrowed_to_true(self):
        """Under float() narrowing, float(9007199254740992) >= float(9007199254740993)
        rounds both to the same value and wrongly reads True. Exact int comparison
        must read False."""
        assert float(self.SAFE_INT) >= float(self.JUST_OVER)  # sanity: narrowing DOES flip this
        c = make_compiler([f"FORBID DANGER IF v >= {self.JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": self.SAFE_INT}
        )
        assert v.is_safe and v.selected_action == "DANGER"  # not forbidden: v is genuinely less

    def test_metric_at_threshold_is_true(self):
        c = make_compiler([f"FORBID DANGER IF v >= {self.JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": self.JUST_OVER}
        )
        assert v.is_safe and v.selected_action == "HOLD"  # forbidden: v meets the exact threshold

    def test_float_metric_against_exact_int_threshold_still_exact(self):
        """A float metric equal to 2**53 compared against the int threshold
        2**53+1 must not be treated as equal via rounding."""
        c = make_compiler([f"FORBID DANGER IF v >= {self.JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": float(self.SAFE_INT)}
        )
        assert v.is_safe and v.selected_action == "DANGER"

    def test_numpy_integer_metric_is_not_read_as_unknown(self):
        """A numpy scalar (np.int64) in a caller-supplied metrics dict must be
        accepted, not silently downgraded to UNKNOWN just because it fails a
        bare `isinstance(val, (int, float))` check."""
        c = make_compiler([f"FORBID DANGER IF v >= {self.JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": np.int64(self.SAFE_INT)}
        )
        assert v.is_safe and v.selected_action == "DANGER"  # exact, still not forbidden

    def test_numpy_float32_nan_metric_still_fails_closed(self):
        """math.isfinite must be applied to a non-finite np.float32 metric too,
        even though np.float32 is not a Python `float` subclass. FORBID_IF's
        fail-closed collapse of UNKNOWN forbids DANGER, but HOLD remains a safe
        candidate -- so the verdict itself is still_safe, just not DANGER."""
        c = make_compiler(["FORBID DANGER IF v > 0"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": np.float32("nan")}
        )
        assert v.is_safe and v.selected_action == "HOLD"
