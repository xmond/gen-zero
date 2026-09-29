"""Regression contract for ChatGPT 6 Pro round-7 audit finding R7-N01.

R7-N01: "np.longdouble is still silently narrowed".

Before this fix, the float whitelist in ``_reject_lossy_integers`` was:

  ``if isinstance(value, (float, np.floating)): return``
  ``elif kind == "f": return``

Both branches accept ANY float-kind value regardless of width. On this
platform ``np.longdouble`` (float128, 80-bit extended precision padded to 16
bytes) reports ``dtype.kind == "f"`` -- the SAME kind as float16/32/64, not a
distinct kind -- so it sailed through the whitelist and into
``_validate_latent``'s unconditional ``np.asarray(..., dtype=np.float64)``
cast, which silently discards the 11 extra mantissa bits (63 vs float64's 52):

  - ``np.longdouble(1) + np.ldexp(np.longdouble(1), -60)`` is a value
    strictly greater than 1 (``1.0000000000000000009``) that rounds to
    exactly ``1.0`` once cast -- flipping a ``> 1`` comparison from True to
    False.
  - ``np.ldexp(np.longdouble(1), -1075)`` is a strictly positive subnormal
    that underflows to ``0.0`` in float64 -- flipping a ``> 0`` comparison
    from True to False.

Root cause: kind alone cannot distinguish extended precision from standard
IEEE-754 float64 on this platform; itemsize is the only reliable signal
(float16/32/64 -> itemsize 2/4/8; longdouble/float128/float96 -> itemsize
> 8). Contract: any floating value/array/tensor whose itemsize exceeds 8
bytes must fail closed with ``TypeError`` in both ``_check_scalar`` and
``_check_ndarray``, before it can reach the lossy float64 cast -- never
silently narrow or underflow.
"""
import numpy as np
import pytest

from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler


def make_compiler(latent_dim=8, hard_timeout_ms=100.0):
    c = ConstraintLinearProjectionCompiler(latent_dim=latent_dim, hard_timeout_ms=hard_timeout_ms)
    c.compile_rules(["FORBID DANGER IF v >= 1"])
    return c


# ---------------------------------------------------------------------------
# R7-N01 counterexamples: each must fail closed with TypeError, never
# silently narrow/underflow through the float64 cast.
# ---------------------------------------------------------------------------

class TestR7N01RejectsLongdouble:
    def test_precision_loss_probe_ndarray_rejected(self):
        """np.longdouble(1) + ldexp(1, -60): > 1 before, == 1.0 after a
        silent float64 cast -- must be rejected, not silently narrowed."""
        c = make_compiler()
        x = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
        assert x > 1  # sanity: the probe is genuinely distinguishable from 1.0
        with pytest.raises(TypeError):
            c._validate_latent(np.array([x], dtype=np.longdouble))

    def test_precision_loss_probe_bare_scalar_rejected(self):
        c = make_compiler()
        x = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
        with pytest.raises(TypeError):
            c._validate_latent(x)

    def test_underflow_probe_rejected(self):
        """ldexp(longdouble(1), -1075): strictly positive, underflows to
        0.0 in float64 -- must be rejected, not silently zeroed."""
        c = make_compiler()
        y = np.ldexp(np.longdouble(1), -1075)
        assert y > 0  # sanity: the probe is genuinely nonzero
        with pytest.raises(TypeError):
            c._validate_latent(np.array([y], dtype=np.longdouble))

    def test_longdouble_scalar_leaf_in_list_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent([np.longdouble(1.5), 2.0])

    def test_float96_or_float128_dtype_array_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent(np.array([1.0], dtype=np.longdouble))

    def test_reject_lossy_integers_direct_call_matches_validate_latent(self):
        """Direct unit coverage of the guard itself for each R7-N01 probe."""
        x = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
        y = np.ldexp(np.longdouble(1), -1075)
        for probe in (
            np.array([x], dtype=np.longdouble),
            x,
            np.array([y], dtype=np.longdouble),
            [np.longdouble(1.5)],
        ):
            with pytest.raises(TypeError):
                ConstraintLinearProjectionCompiler._reject_lossy_integers(probe)


class TestR7N01FailsClosedAtPublicEntryPoints:
    """The guard must be reachable from the public API, not just the private
    _validate_latent helper exercised above."""

    def test_project_latent_propositions_rejects_longdouble_array(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        x = np.longdouble(1) + np.ldexp(np.longdouble(1), -60)
        with pytest.raises(TypeError):
            c.project_latent_propositions(
                np.array([x], dtype=np.longdouble), schema_fingerprint=fp
            )

    def test_solve_safest_action_rejects_longdouble_array(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        y = np.ldexp(np.longdouble(1), -1075)
        with pytest.raises(TypeError):
            c.solve_safest_action(
                {"DANGER": 1.0, "HOLD": 0.0},
                z_latent=np.array([y], dtype=np.longdouble),
                schema_fingerprint=fp,
            )


# ---------------------------------------------------------------------------
# Positive controls: standard IEEE-754 float16/32/64 must keep compiling and
# evaluating exactly as before -- the itemsize gate must not over-reject
# ordinary numeric input.
# ---------------------------------------------------------------------------

class TestR7N01PositiveControlsUnaffected:
    def test_python_float_accepted(self):
        c = make_compiler()
        z = c._validate_latent([1.0, 2.0])
        assert list(z) == [1.0, 2.0]

    def test_numpy_float16_accepted(self):
        c = make_compiler()
        z = c._validate_latent(np.array([1.0, 2.0], dtype=np.float16))
        assert list(z) == [1.0, 2.0]

    def test_numpy_float32_accepted(self):
        c = make_compiler()
        z = c._validate_latent(np.array([1.0, 2.0], dtype=np.float32))
        assert list(z) == [1.0, 2.0]

    def test_numpy_float64_accepted(self):
        c = make_compiler()
        z = c._validate_latent(np.array([1.0, 2.0], dtype=np.float64))
        assert list(z) == [1.0, 2.0]

    def test_numpy_float64_scalar_leaf_accepted(self):
        c = make_compiler()
        z = c._validate_latent([np.float64(1.5), np.float32(2.5)])
        assert list(z) == [1.5, 2.5]

    def test_torch_float32_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        z = c._validate_latent(torch.tensor([1.5, 2.5], dtype=torch.float32))
        assert z == pytest.approx([1.5, 2.5])

    def test_torch_float64_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        z = c._validate_latent(torch.tensor([1.5, 2.5], dtype=torch.float64))
        assert z == pytest.approx([1.5, 2.5])

    def test_project_latent_propositions_still_works_on_real_latent(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        result = c.project_latent_propositions(np.array([2.0]), schema_fingerprint=fp)
        assert result

    def test_solve_safest_action_still_works_on_real_latent(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        v = c.solve_safest_action(
            {"DANGER": 1.0, "HOLD": 0.0},
            z_latent=np.array([2.0]),
            schema_fingerprint=fp,
        )
        assert v.is_safe and v.selected_action == "HOLD"
