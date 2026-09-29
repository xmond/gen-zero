"""Consumer bridge: ManifoldAnchorDistiller -> MCP `zero` NanoCore ask requests.

Targets the **operator-loaded** NanoCore route in
``crates/gen-zero-service/src/zero.rs``, selected by the presence of
``nanocore_domain`` and served by ``nanocore_ask`` (zero.rs:2246-2360). That
route is distinct from the *caller-supplied* ``engine: "nanocore"`` route
(which additionally requires ``nanocore_core`` + ``decision_state`` and is
handled elsewhere, zero.rs:2556+): mixing the two fields is explicitly
rejected by zero.rs:1591-1602 ("nanocore_domain(s) cannot combine with
engine/head decision backends"), confirmed by the
``operator_and_caller_nanocore_routes_do_not_mix`` regression test at
zero.rs:5462-5472. The payload this module builds must therefore omit
``engine`` -- see the accepted shape at zero.rs:5562-5569
(``{"verb": "ask", "candidates": [...], "nanocore_domain": 0, "nanocore_state": [...]}``).

This module is the actual consumer of a fitted ``ManifoldAnchorDistiller``
artifact: it loads the artifact, projects a raw 8192-dim hidden
representation down to 128 dims, and assembles the MCP ``tools/call zero``
request body ``nanocore_ask`` expects (128 finite floats under
``nanocore_state``, gated at zero.rs:2261-2279).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .manifold_anchor_distiller import ManifoldAnchorDistiller

__all__ = ["NanocoreAnchorBridge"]

NANOCORE_STATE_DIM = 128
# crates/gen-zero-core/src/types.rs: LocalActionFrame::new caps candidates at 16
# (fixed-size stack arrays); nanocore_ask (zero.rs) refuses beyond that.
MAX_NANOCORE_CANDIDATES = 16
# zero.rs:526-541: nanocore_domain must parse via `as_u64().and_then(u32::try_from)`.
MAX_NANOCORE_DOMAIN_ID = 0xFFFF_FFFF


class NanocoreAnchorBridge:
    """Projects raw hidden reps into the 128-d ``nanocore_state`` MCP expects."""

    def __init__(self, artifact_path: str | Path, *, core_manifest=None):
        self._distiller = ManifoldAnchorDistiller.load(str(artifact_path))
        if self._distiller.output_dim != NANOCORE_STATE_DIM:
            raise ValueError(
                f"NanocoreAnchorBridge requires an artifact with output_dim="
                f"{NANOCORE_STATE_DIM} (the NanoCore engine's fixed state "
                f"width), got output_dim={self._distiller.output_dim}"
            )

        space = self._distiller.space
        required = {"source_model", "layer", "norm", "GCCA_map", "anchor_basis", "core", "domain_id"}
        if not isinstance(space, dict) or set(space) != required or self._distiller.gcca is None:
            raise ValueError("artifact lacks complete source -> GCCA -> anchor -> core identity")
        if core_manifest != space:
            raise ValueError("core manifest does not match anchor space")
        self.space = dict(space)

    def project_to_nanocore_state(self, raw_hidden: np.ndarray) -> List[float]:
        if not isinstance(raw_hidden, dict) or set(raw_hidden) != {"values", "space"}:
            raise ValueError("input requires values and explicit source space identity")
        x = np.asarray(raw_hidden["values"], dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(
                f"raw_hidden must be a single {self._distiller.input_dim}-dim "
                f"vector (shape (D,) or (1, D)), got shape {np.asarray(raw_hidden).shape}"
            )
        z = self._distiller.transform_source(x, raw_hidden["space"])[0]
        if z.shape != (NANOCORE_STATE_DIM,):
            raise ValueError(
                f"projected state has shape {z.shape}, expected ({NANOCORE_STATE_DIM},)"
            )
        if not np.all(np.isfinite(z)):
            raise ValueError("projected nanocore_state contains non-finite values; refusing to emit")
        # nanocore_ask (zero.rs) reads each coordinate as f32 (CompressedLatent
        # is f32-backed): a float64 value finite at f64 precision can still
        # overflow to +/-inf once narrowed to f32, so this must be checked
        # BEFORE the narrowing cast -- checked as a float64 magnitude compare
        # against float32's max representable value, never by casting first
        # and inspecting the (already-overflowed, RuntimeWarning-raising)
        # result.
        if np.any(np.abs(z) > np.finfo(np.float32).max):
            raise ValueError("projected state contains non-finite values after float32 conversion")
        z32 = z.astype(np.float32)
        return z32.tolist()

    def generate_mcp_ask_payload(
        self,
        raw_hidden: np.ndarray,
        domain_id: int,
        candidates: List[str],
    ) -> dict:
        if (
            isinstance(domain_id, bool)
            or not isinstance(domain_id, (int, np.integer))
            or not (0 <= domain_id <= MAX_NANOCORE_DOMAIN_ID)
        ):
            raise ValueError(
                f"domain_id must be an integer in [0, {MAX_NANOCORE_DOMAIN_ID}] "
                f"(a u32 domain ID), got {domain_id!r}"
            )
        # A bare str/bytes is iterable, so `all(isinstance(c, str) ...)` below
        # would silently accept candidates="stop" and split it into the four
        # single-character candidates ["s","t","o","p"] -- reject the type
        # outright before anything downstream treats it as an iterable.
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, (list, tuple)):
            raise TypeError("candidates must be a list or tuple of strings")
        if not candidates or not all(isinstance(c, str) and len(c.strip()) > 0 for c in candidates):
            raise ValueError("candidates must contain non-empty strings")
        if len(candidates) > MAX_NANOCORE_CANDIDATES:
            raise ValueError(
                f"candidates has {len(candidates)} entries, exceeding the NanoCore "
                f"engine's fixed capacity of {MAX_NANOCORE_CANDIDATES}"
            )
        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"candidates contains duplicates; candidate actions must be unique, got {list(candidates)!r}"
            )

        if int(domain_id) != self.space["domain_id"]:
            raise ValueError("domain does not match bound core")
        state = self.project_to_nanocore_state(raw_hidden)
        return {
            "verb": "ask",
            "nanocore_domain": int(domain_id),
            "nanocore_state": state,
            "candidates": list(candidates),
        }
