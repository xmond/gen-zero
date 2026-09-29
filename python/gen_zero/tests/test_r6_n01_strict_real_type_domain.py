"""Regression contract for ChatGPT 6 Pro round-6 audit finding R6-N01.

R6-N01: "non-agreed-upon real-number types still silently lossy-convert".

Before this fix, ``_reject_lossy_integers``'s ``_check_scalar`` /
``_check_ndarray`` helpers only ever inspected values that were ALREADY known
to be ``(int, np.integer)`` -- anything else (a ``str``, ``Decimal``,
``Fraction``, or Python ``complex``) fell through every branch with no check
at all, reaching ``_validate_latent``'s unconditional
``np.asarray(..., dtype=np.float64)`` cast completely unguarded:

  - ``["9007199254740993"]`` (a numeric string) was silently parsed by NumPy
    and rounded to ``9007199254740992.0``.
  - ``[Decimal(2**53 + 1)]`` / ``[Fraction(2**53 + 1, 1)]`` were silently
    narrowed via ``__float__`` to ``9007199254740992.0``.
  - ``np.array([1+2j])`` / ``torch.tensor([1+2j])`` were silently cast to
    float64, DROPPING THE IMAGINARY PART entirely (``1+2j`` -> ``1.0``).

Root cause: ``src/gen_zero/gate/constraint_compiler.py`` had no closed
whitelist of permitted leaf types -- only a magnitude check that assumed its
input was already an int. Contract: every leaf that is not a Python
``bool``/``int``/``float`` or a NumPy real integer/float scalar, and every
array/tensor whose dtype kind is not in ``('i', 'u', 'f')``, must fail closed
with ``TypeError`` before it can reach the lossy float64 cast -- never
silently convert, parse, or truncate.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler


def make_compiler(latent_dim=8, hard_timeout_ms=100.0):
    c = ConstraintLinearProjectionCompiler(latent_dim=latent_dim, hard_timeout_ms=hard_timeout_ms)
    c.compile_rules(["FORBID DANGER IF v >= 1"])
    return c


# ---------------------------------------------------------------------------
# R6-N01 counterexamples: each must fail closed with TypeError, not silently
# narrow/truncate. Exercised at all three z_latent-consuming entry points.
# ---------------------------------------------------------------------------

class TestR6N01RejectsLossyRealTypes:
    def test_numeric_string_leaf_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent(["9007199254740993"])

    def test_decimal_leaf_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent([Decimal(2 ** 53 + 1)])

    def test_fraction_leaf_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent([Fraction(2 ** 53 + 1, 1)])

    def test_numpy_complex_array_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent(np.array([1 + 2j]))

    def test_torch_complex_tensor_rejected(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent(torch.tensor([1 + 2j]))

    def test_bare_python_complex_scalar_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent([1 + 2j])

    def test_numpy_string_array_rejected(self):
        c = make_compiler()
        with pytest.raises(TypeError):
            c._validate_latent(np.array(["9007199254740993"]))

    def test_reject_lossy_integers_direct_call_matches_validate_latent(self):
        """Direct unit coverage of the guard itself for each R6-N01 probe."""
        for probe in (
            ["9007199254740993"],
            [Decimal(2 ** 53 + 1)],
            [Fraction(2 ** 53 + 1, 1)],
            np.array([1 + 2j]),
        ):
            with pytest.raises(TypeError):
                ConstraintLinearProjectionCompiler._reject_lossy_integers(probe)


class TestR6N01FailsClosedAtPublicEntryPoints:
    """The guard must be reachable from the public API, not just the private
    _validate_latent helper exercised above."""

    def test_project_latent_propositions_rejects_string_latent(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        with pytest.raises(TypeError):
            c.project_latent_propositions(["9007199254740993"], schema_fingerprint=fp)

    def test_project_latent_propositions_rejects_decimal_latent(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        with pytest.raises(TypeError):
            c.project_latent_propositions([Decimal(2 ** 53 + 1)], schema_fingerprint=fp)

    def test_project_latent_propositions_rejects_complex_ndarray(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        with pytest.raises(TypeError):
            c.project_latent_propositions(np.array([1 + 2j]), schema_fingerprint=fp)

    def test_solve_safest_action_rejects_fraction_latent(self):
        c = make_compiler()
        fp = c.schema_fingerprint
        with pytest.raises(TypeError):
            c.solve_safest_action(
                {"DANGER": 1.0, "HOLD": 0.0},
                z_latent=[Fraction(2 ** 53 + 1, 1)],
                schema_fingerprint=fp,
            )

    def test_solve_safest_action_rejects_complex_torch_tensor(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        fp = c.schema_fingerprint
        with pytest.raises(TypeError):
            c.solve_safest_action(
                {"DANGER": 1.0, "HOLD": 0.0},
                z_latent=torch.tensor([1 + 2j]),
                schema_fingerprint=fp,
            )


# ---------------------------------------------------------------------------
# Positive controls: legitimate real-scalar/array/tensor inputs must keep
# compiling and evaluating exactly as before -- the whitelist must not
# over-reject ordinary numeric input.
# ---------------------------------------------------------------------------

class TestR6N01PositiveControlsUnaffected:
    def test_python_float_list_accepted(self):
        c = make_compiler()
        z = c._validate_latent([1.0, 2.0])
        assert list(z) == [1.0, 2.0]

    def test_python_int_list_accepted(self):
        c = make_compiler()
        z = c._validate_latent([1, 2, 3])
        assert list(z) == [1.0, 2.0, 3.0]

    def test_python_bool_leaf_accepted(self):
        c = make_compiler()
        z = c._validate_latent([True, False])
        assert list(z) == [1.0, 0.0]

    def test_numpy_float64_ndarray_accepted(self):
        c = make_compiler()
        z = c._validate_latent(np.array([1.0, 2.0], dtype=np.float64))
        assert list(z) == [1.0, 2.0]

    def test_numpy_int_ndarray_accepted(self):
        c = make_compiler()
        z = c._validate_latent(np.array([1, 2, 3], dtype=np.int64))
        assert list(z) == [1.0, 2.0, 3.0]

    def test_torch_float32_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        z = c._validate_latent(torch.tensor([1.5, 2.5], dtype=torch.float32))
        assert z == pytest.approx([1.5, 2.5])

    def test_torch_int64_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        c = make_compiler()
        z = c._validate_latent(torch.tensor([1, 2, 3], dtype=torch.int64))
        assert list(z) == [1.0, 2.0, 3.0]

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
