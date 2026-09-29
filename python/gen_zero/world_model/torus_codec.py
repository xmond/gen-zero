"""Orthonormal code and relative-action encoding for the torus world.

Lives inside the package so wheel installs of ``gen_zero`` work without the
repo's ``benchmarks/`` tree. ``benchmarks/suites/deadlock_torus_env.py`` and
``scripts/extract_trajectories.py`` import from here too, so the encoder that
trains the world model and the one that serves it cannot drift apart.
"""
from __future__ import annotations

import numpy as np

TORUS_ACTION_NAMES = ("left", "straight", "right")
TORUS_ACTION_DIM = 16
TORUS_ACTION_SEED = 20260926


def latent_codes(n_states: int, dim: int, seed: int) -> np.ndarray:
    """Random orthonormal state codes (rows). This is the grid's 'encoder'."""
    if dim < n_states:
        raise ValueError("dim must be >= number of states for orthonormal codes")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, n_states)))
    return Q.T.astype(np.float64)


def encode_torus_action(action: str) -> np.ndarray:
    """Named relative action ("left" | "straight" | "right") -> float32 (16,) code."""
    try:
        idx = TORUS_ACTION_NAMES.index(action)
    except ValueError:
        raise KeyError(f"{action!r} is not a torus action {TORUS_ACTION_NAMES}") from None
    codes = latent_codes(len(TORUS_ACTION_NAMES), TORUS_ACTION_DIM, TORUS_ACTION_SEED)
    return codes[idx].astype(np.float32)
