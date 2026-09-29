"""Regression contract for ChatGPT 6 Pro round-3 final-review finding T3-C01.

T3-C01: NumPy scalar/Latent mixed-type comparison silently narrows large
integers through float64, flipping the verdict near/beyond 2**53:

  - ``x < 9007199254740993`` evaluated against ``np.float64(9007199254740992)``
    reads FALSE (should be TRUE): NumPy upcasts the Python int threshold to
    float64 before comparing, rounding it down to the same value as ``x``.
  - ``x > 9007199254740992.0`` evaluated against ``np.int64(9007199254740993)``
    reads FALSE (should be TRUE): NumPy upcasts the int64 metric to float64,
    rounding it down to equal the float threshold.
  - A latent vector ``[9007199254740993]`` is silently narrowed to
    ``9007199254740992.0`` by ``_validate_latent``'s unconditional
    ``np.asarray(..., dtype=float64)`` cast, before any comparison even runs.

Root cause: Python's own mixed int/float rich comparison is exact (X-C04
already covers that), but NumPy's cross-type ufunc comparison is not -- it
upcasts both operands to a common dtype (float64) first. Any metric value
that arrives as a NumPy scalar (as every ``z_latent``-derived metric does,
via ``_bind_latent``) therefore re-opens the exact bug X-C04 closed for pure
Python int/float pairs.

Contract: comparator evaluation must normalize NumPy scalars to native Python
int/float (bit-exact, no precision change) *before* comparing, so Python's
native exact mixed-comparison semantics apply. Latent validation must reject
-- via ``LossyNumericConversionError`` -- any input integer whose magnitude
exceeds 2**53-1, the largest integer float64 can represent exactly, rather
than silently rounding it during the float64 cast.
"""
import numpy as np
import pytest

from gen_zero.gate.constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    LossyNumericConversionError,
    Tristate,
)

SAFE_INT = 9007199254740992  # 2**53, exactly representable in float64
JUST_OVER = 9007199254740993  # 2**53 + 1, NOT exactly representable in float64


def make_compiler(rules, latent_dim=8, hard_timeout_ms=100.0):
    c = ConstraintLinearProjectionCompiler(latent_dim=latent_dim, hard_timeout_ms=hard_timeout_ms)
    c.compile_rules(list(rules))
    return c


# ---------------------------------------------------------------------------
# T3-C01 counterexample A: np.float64 metric vs. int threshold, "<".
# ---------------------------------------------------------------------------

class TestT3C01NumpyFloatVsIntThreshold:
    def test_raw_numpy_upcast_bug_reproduces(self):
        """Sanity: confirms the underlying NumPy behavior this test guards against."""
        assert (np.float64(SAFE_INT) < JUST_OVER) is np.bool_(False)

    def test_sub_condition_reads_true_not_narrowed_false(self):
        c = make_compiler([f"FORBID DANGER IF v < {JUST_OVER}"])
        cond = c._rules[0].sub_conditions[0]
        outcome = c._evaluate_sub_condition(cond, {"v": np.float64(SAFE_INT)})
        assert outcome is Tristate.TRUE

    def test_solve_safest_action_forbids_via_numpy_float_metric(self):
        c = make_compiler([f"FORBID DANGER IF v < {JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": np.float64(SAFE_INT)}
        )
        assert v.is_safe and v.selected_action == "HOLD"  # v < JUST_OVER is TRUE -> forbidden


# ---------------------------------------------------------------------------
# T3-C01 counterexample B: np.int64 metric vs. float threshold, ">".
# ---------------------------------------------------------------------------

class TestT3C01NumpyIntVsFloatThreshold:
    def test_raw_numpy_upcast_bug_reproduces(self):
        assert (np.int64(JUST_OVER) > float(SAFE_INT)) is np.bool_(False)

    def test_sub_condition_reads_true_not_narrowed_false(self):
        c = make_compiler([f"FORBID DANGER IF v > {float(SAFE_INT)}"])
        cond = c._rules[0].sub_conditions[0]
        outcome = c._evaluate_sub_condition(cond, {"v": np.int64(JUST_OVER)})
        assert outcome is Tristate.TRUE

    def test_solve_safest_action_forbids_via_numpy_int_metric(self):
        c = make_compiler([f"FORBID DANGER IF v > {float(SAFE_INT)}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": np.int64(JUST_OVER)}
        )
        assert v.is_safe and v.selected_action == "HOLD"  # v > SAFE_INT is TRUE -> forbidden


# ---------------------------------------------------------------------------
# T3-C01 counterexample: numpy mixed-type "==" false positive.
# ---------------------------------------------------------------------------

class TestT3C01NumpyEqualityFalsePositive:
    def test_raw_numpy_upcast_bug_reproduces(self):
        assert (np.float64(SAFE_INT) == JUST_OVER) is np.bool_(True)

    def test_sub_condition_reads_false_not_narrowed_true(self):
        c = make_compiler([f"FORBID DANGER IF v == {JUST_OVER}"])
        cond = c._rules[0].sub_conditions[0]
        outcome = c._evaluate_sub_condition(cond, {"v": np.float64(SAFE_INT)})
        assert outcome is Tristate.FALSE

    def test_solve_safest_action_does_not_falsely_forbid(self):
        c = make_compiler([f"FORBID DANGER IF v == {JUST_OVER}"])
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0}, current_metrics={"v": np.float64(SAFE_INT)}
        )
        assert v.is_safe and v.selected_action == "DANGER"  # SAFE_INT != JUST_OVER -> not forbidden


# ---------------------------------------------------------------------------
# T3-C01 counterexample C: latent vector narrowing.
# ---------------------------------------------------------------------------

class TestT3C01LatentVectorNarrowing:
    def test_validate_latent_rejects_int64_array_beyond_safe_range(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent(np.array([JUST_OVER]))

    def test_validate_latent_rejects_plain_python_list(self):
        """The exact reported repro: latent = [9007199254740993]."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([JUST_OVER])

    def test_validate_latent_rejects_negative_out_of_range_int(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([-JUST_OVER])

    def test_validate_latent_rejects_bignum_beyond_int64_object_dtype(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([10 ** 30])

    def test_validate_latent_rejects_mixed_int_float_list(self):
        """[2**53+1, 1.5]: a plain Python list mixing an out-of-range int with a
        float. np.asarray() on this list upcasts the whole thing to float64
        immediately (dtype inference itself is the lossy step) -- the guard
        must walk the raw Python elements before ever calling np.asarray."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([JUST_OVER, 1.5])

    def test_validate_latent_rejects_torch_int64_tensor(self):
        """A caller may reasonably pass a torch tensor (world_model_orchestrator's
        own _encode_to_latent produces torch tensors on several paths); an
        int64 tensor beyond the safe range must be rejected the same as a
        numpy array, not silently skipped because it isn't an np.ndarray."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent(torch.tensor([JUST_OVER], dtype=torch.int64))

    def test_validate_latent_accepts_bare_scalar(self):
        """A 0-d array-like (bare scalar) must not crash the guard."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent(np.asarray(SAFE_INT))
        assert z[0] == float(SAFE_INT)

    def test_validate_latent_accepts_in_range_int_unchanged(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent(np.array([SAFE_INT]))
        assert z[0] == float(SAFE_INT)

    def test_project_latent_propositions_fails_closed_on_lossy_latent(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        fp = c.schema_fingerprint
        with pytest.raises(LossyNumericConversionError):
            c.project_latent_propositions(np.array([JUST_OVER]), schema_fingerprint=fp)

    def test_solve_safest_action_fails_closed_on_lossy_latent(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        fp = c.schema_fingerprint
        with pytest.raises(LossyNumericConversionError):
            c.solve_safest_action(
                {"DANGER": 1.0, "HOLD": 0.0},
                z_latent=np.array([JUST_OVER]),
                schema_fingerprint=fp,
            )


# ---------------------------------------------------------------------------
# T3-C01 round-4 counterexample: an np.ndarray LEAF nested inside a list/tuple
# bypasses the walk entirely. `_walk` recursed into list/tuple but routed
# every other leaf straight to `_check_scalar`, which only recognizes
# `(int, np.integer)` -- a 0-d or multi-dim ndarray matched neither branch,
# so `_check_scalar` silently did nothing and `_reject_lossy_integers`
# returned clean. `_validate_latent`'s subsequent
# `np.asarray(z_latent, dtype=np.float64)` then narrowed the value with no
# guard having ever inspected it.
# ---------------------------------------------------------------------------

class TestT3C01NestedNdarrayLeafBypass:
    def test_rejects_zero_d_int64_array_inside_list(self):
        """The exact WebGPT round-4 repro: [np.array(2**53+1, dtype=np.int64)]."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([np.array(JUST_OVER, dtype=np.int64)])

    def test_rejects_one_d_int64_array_inside_list(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([np.array([JUST_OVER], dtype=np.int64)])

    def test_rejects_zero_d_array_inside_tuple(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent((np.array(JUST_OVER),))

    def test_rejects_array_nested_two_levels_deep(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([[np.array(JUST_OVER)]])

    def test_rejects_uint64_array_mixed_with_scalar_in_list(self):
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([0, [np.array(JUST_OVER, dtype=np.uint64)]])

    def test_zero_d_array_leaf_in_range_still_accepted(self):
        """Positive control: an in-range 0-d array leaf must not be rejected."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent([np.array(SAFE_INT, dtype=np.int64)])
        assert z[0] == float(SAFE_INT)

    def test_reject_lossy_integers_direct_call_on_nested_array(self):
        """Direct unit coverage of the guard itself, not just via _validate_latent."""
        with pytest.raises(LossyNumericConversionError):
            ConstraintLinearProjectionCompiler._reject_lossy_integers(
                [np.array(JUST_OVER, dtype=np.int64)]
            )


# ---------------------------------------------------------------------------
# R5-N01 (WebGPT round 5): a torch.Tensor LEAF nested inside a list/tuple
# bypasses the walk entirely. The round-4 fix above special-cased np.ndarray
# leaves, but a torch.Tensor is neither a list/tuple nor an np.ndarray, so it
# fell into the same `_check_scalar` gap that round 4 closed for ndarrays --
# `_check_scalar` only recognizes `(int, np.integer)`, so the Tensor leaf
# silently passed through to `_validate_latent`'s float64 cast unchecked.
# The top-level (non-list) branch already converts any non-ndarray argument
# via the array protocol, so a *bare* top-level torch.Tensor was already
# rejected (see test_validate_latent_rejects_torch_int64_tensor above) --
# only a Tensor nested inside a list/tuple was vulnerable.
# ---------------------------------------------------------------------------

class TestR5N01NestedTensorLeafBypass:
    def test_rejects_scalar_int64_tensor_inside_list(self):
        """The exact WebGPT round-5 repro: [torch.tensor(2**53+1, dtype=torch.int64)]."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([torch.tensor(JUST_OVER, dtype=torch.int64)])

    def test_rejects_one_d_int64_tensor_inside_list(self):
        """The exact WebGPT round-5 repro: [torch.tensor([2**53+1], dtype=torch.int64)]."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([torch.tensor([JUST_OVER], dtype=torch.int64)])

    def test_rejects_tensor_inside_tuple(self):
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent((torch.tensor(JUST_OVER, dtype=torch.int64),))

    def test_rejects_tensor_nested_two_levels_deep(self):
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([[torch.tensor(JUST_OVER, dtype=torch.int64)]])

    def test_rejects_mixed_ndarray_and_tensor_leaves_in_list(self):
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([np.array(1), torch.tensor(JUST_OVER, dtype=torch.int64)])

    def test_reject_lossy_integers_direct_call_on_nested_tensor(self):
        """Direct unit coverage of the guard itself, not just via _validate_latent."""
        torch = pytest.importorskip("torch")
        with pytest.raises(LossyNumericConversionError):
            ConstraintLinearProjectionCompiler._reject_lossy_integers(
                [torch.tensor(JUST_OVER, dtype=torch.int64)]
            )

    def test_in_range_tensor_leaf_in_list_still_accepted(self):
        """Positive control: an in-range Tensor leaf must not be rejected."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent([torch.tensor(SAFE_INT, dtype=torch.int64)])
        assert z[0] == float(SAFE_INT)

    def test_float_dtype_tensor_leaf_in_list_still_accepted(self):
        """A float-dtype Tensor leaf carries no exact-int guarantee to check;
        it must pass through unrejected by this guard (NaN/Inf is a separate
        check performed later in _validate_latent)."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent([torch.tensor([1.5], dtype=torch.float32)])
        assert z[0] == pytest.approx(1.5)

    def test_plain_scalars_in_list_unaffected_by_array_protocol_branch(self):
        """Regression guard: plain Python int/float/bool leaves must keep
        going through `_check_scalar` (they have no `__array__`), not the
        new array-protocol branch."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        z = c._validate_latent([3, 4.5, True])
        assert list(z) == [3.0, 4.5, 1.0]

    def test_numpy_scalar_in_list_still_rejected_when_out_of_range(self):
        """np.integer scalars DO implement __array__, so they now route
        through the new branch too -- confirm they are still caught."""
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        with pytest.raises(LossyNumericConversionError):
            c._validate_latent([np.int64(JUST_OVER)])

    def test_project_latent_propositions_fails_closed_on_nested_tensor(self):
        """Public-entry-point mirror: the guard must actually be reachable from
        project_latent_propositions, not just from the private _validate_latent
        helper exercised by the tests above."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        fp = c.schema_fingerprint
        with pytest.raises(LossyNumericConversionError):
            c.project_latent_propositions(
                [torch.tensor(JUST_OVER, dtype=torch.int64)], schema_fingerprint=fp
            )

    def test_solve_safest_action_fails_closed_on_nested_tensor(self):
        """Public-entry-point mirror: solve_safest_action must reject a
        nested-Tensor z_latent via _bind_latent before CP-SAT is ever
        touched, so this holds regardless of whether ortools is installed
        in the test environment."""
        torch = pytest.importorskip("torch")
        c = make_compiler([f"FORBID DANGER IF v >= {JUST_OVER}"])
        fp = c.schema_fingerprint
        with pytest.raises(LossyNumericConversionError):
            c.solve_safest_action(
                {"DANGER": 1.0, "HOLD": 0.0},
                z_latent=[torch.tensor(JUST_OVER, dtype=torch.int64)],
                schema_fingerprint=fp,
            )


# ---------------------------------------------------------------------------
# Positive controls: ordinary small values still behave exactly as before.
# ---------------------------------------------------------------------------

class TestT3C01PositiveControlsUnaffected:
    def test_small_numpy_float_metric_still_compares_normally(self):
        c = make_compiler(["FORBID DANGER IF v > 5"])
        cond = c._rules[0].sub_conditions[0]
        assert c._evaluate_sub_condition(cond, {"v": np.float64(10.0)}) is Tristate.TRUE
        assert c._evaluate_sub_condition(cond, {"v": np.float64(1.0)}) is Tristate.FALSE

    def test_small_numpy_int_metric_still_compares_normally(self):
        c = make_compiler(["FORBID DANGER IF v > 5"])
        cond = c._rules[0].sub_conditions[0]
        assert c._evaluate_sub_condition(cond, {"v": np.int64(10)}) is Tristate.TRUE
        assert c._evaluate_sub_condition(cond, {"v": np.int64(1)}) is Tristate.FALSE

    def test_small_latent_vector_still_round_trips(self):
        c = make_compiler(["FORBID DANGER IF v > 5"])
        z = c._validate_latent(np.array([10.0]))
        assert z[0] == 10.0
        z2 = c._validate_latent([3, 4, 5])
        assert list(z2) == [3.0, 4.0, 5.0]

    def test_solve_safest_action_with_ordinary_latent_still_works(self):
        c = make_compiler(["FORBID DANGER IF v > 5"])
        fp = c.schema_fingerprint
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0},
            z_latent=np.array([10.0]),
            schema_fingerprint=fp,
        )
        assert v.is_safe and v.selected_action == "HOLD"
