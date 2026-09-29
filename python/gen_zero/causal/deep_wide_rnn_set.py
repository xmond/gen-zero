"""Deep & wide Parallel RNN + deep Set-Transformer, with paired cross-difference
interaction: pure NumPy CPU runtime.

Extends the single-layer scorer in `rnn_set_adapter.py` along two axes:

  Width:  d up to 1024 (vs. the original 256), a plain shape parameter.
  Depth:  L >= 2 residual Parallel RNN layers, each with its OWN low-rank
          transition A_l = a_scale_l * (diag(sigmoid(lam_l)) + U_A_l V_A_l^T),
          independently spectral-clamped to sigma_max(A_l) <= rho_max_l < 1
          (a per-layer rho_max lets layers sit at different timescales, e.g.
          0.95/0.85/0.70: fast layers forget quickly, slow layers hold longer
          context). N >= 2 Set-Transformer layers, pre-norm MHA (n_heads up
          to 8) + SwiGLU FFN (silu(x Wg^T) * (x Wu^T)) Wd^T, no bias, no
          positional encoding so scores stay permutation equivariant.

Paired cross-difference interaction (the mechanism, not a trained result):
for two already-encoded latents z_a, z_b in R^d, `_pair_features` builds

    u = [z_a ; z_b ; z_a - z_b ; z_a * z_b]   in R^(4d)
    z_pair = rms(u @ W_pair)                  in R^d

and feeds z_pair into the same think/score path `score` uses for a single
query. The diff term is antisymmetric under (a, b) -> (b, a); the product
term is symmetric. A linear read-out on top of both can therefore separate
order-sensitive relations (entail/contradict, premise-vs-hypothesis) from a
symmetric relation, which a mean-pooled or single-vector encoding of the pair
cannot express without a bilinear term. This module does not claim a trained
accuracy number on any real dataset: `benchmarks/tests/test_deep_wide_rnn_set.py`
verifies the structural properties above on synthetic vectors only.

Every layer's Lyapunov clamp is re-derived from its raw factors by SVD on
load and must match the stored `a_scale` (rtol 1e-4); this certifies that
each layer's own think loop is a contraction (sigma_max(A_l) < 1), not that
the depth-wise residual map h -> h + f_l(h) is one -- see `layer_spectral_radii`.

Pure NumPy, float32 runtime / float64 stability audits. Every public entry
point rejects non-finite input (fail-closed). No text, tokens, or labels are
inspected anywhere in this module; it operates on numeric vectors only.
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

_ARRAYS = (
    "mu", "W_in",
    "lam", "U_A", "V_A", "U_B", "V_B",          # (L, ...) RNN layers
    "W_pair",                                     # (4d, d) cross-difference projection
    "W_s",                                        # (d, d) final scorer
    "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",  # (N, ...) Set-Transformer MHA
    "w_gate", "w_up", "w_down",                    # (N, ...) Set-Transformer SwiGLU FFN
)
_INTS = ("in_dim", "d", "rank", "think_steps", "rnn_layers", "n_heads", "set_layers", "ffn_dim")


def _finite(value, name, *, dtype=np.float32) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return a


def _rms(x: np.ndarray, eps: np.float32) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_difference_features(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """[z_a ; z_b ; z_a - z_b ; z_a * z_b], the paired interaction tensor.

    Batched over any leading axes; z_a and z_b must share shape (..., d).
    Deterministic algebra, exposed standalone so it can be unit-tested
    (order-sensitivity of the diff term, order-invariance of the product
    term) without constructing a full runtime.
    """
    z_a = _finite(z_a, "z_a", dtype=np.float32)
    z_b = _finite(z_b, "z_b", dtype=np.float32)
    if z_a.shape != z_b.shape:
        raise ValueError("cross_difference_features: z_a and z_b must share shape")
    return np.concatenate([z_a, z_b, z_a - z_b, z_a * z_b], axis=-1)


class DeepWideRNNSetRuntime:
    def __init__(self, arrays: Dict[str, np.ndarray], config: Dict[str, int],
                 a_scale, rho_max, rms_eps: float, meta: dict) -> None:
        self.cfg = dict(config)
        self.p = {k: _finite(v, k) for k, v in arrays.items()}
        self.a_scale = _finite(a_scale, "a_scale").astype(np.float64)
        self.rho_max = _finite(rho_max, "rho_max").astype(np.float64)
        self.eps = np.float32(rms_eps)
        self.meta = meta
        self._check_shapes()
        self._check_stability()
        # sigmoid decay per RNN layer, cached in float32 for the think loop
        self._dvec = (1.0 / (1.0 + np.exp(-self.p["lam"].astype(np.float64)))).astype(np.float32)

    @classmethod
    def from_npz(cls, path) -> "DeepWideRNNSetRuntime":
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in _ARRAYS + _INTS + ("a_scale", "rho_max", "rms_eps", "meta_json")
                       if k not in z.files]
            if missing:
                raise ValueError(f"from_npz: missing keys {missing}")
            arrays = {k: np.asarray(z[k], dtype=np.float32) for k in _ARRAYS}
            config = {k: int(z[k]) for k in _INTS}
            a_scale = np.asarray(z["a_scale"], dtype=np.float64)
            rho_max = np.asarray(z["rho_max"], dtype=np.float64)
            rms_eps = float(z["rms_eps"])
            meta = json.loads(str(z["meta_json"]))
        return cls(arrays, config, a_scale, rho_max, rms_eps, meta)

    # -------------------------------------------------------------- shapes

    def _check_shapes(self) -> None:
        c, p = self.cfg, self.p
        D, d, r = c["in_dim"], c["d"], c["rank"]
        L, N, F, H = c["rnn_layers"], c["set_layers"], c["ffn_dim"], c["n_heads"]
        if min(c.values()) <= 0 or d % H:
            raise ValueError("config: sizes must be positive and d divisible by n_heads")
        want = {
            "mu": (D,), "W_in": (D, d),
            "lam": (L, d), "U_A": (L, d, r), "V_A": (L, d, r), "U_B": (L, d, r), "V_B": (L, d, r),
            "W_pair": (4 * d, d), "W_s": (d, d),
            "wq": (N, d, d), "wk": (N, d, d), "wv": (N, d, d), "wo": (N, d, d),
            "bq": (N, d), "bk": (N, d), "bv": (N, d), "bo": (N, d),
            "w_gate": (N, F, d), "w_up": (N, F, d), "w_down": (N, d, F),
        }
        for k, shape in want.items():
            if p[k].shape != shape:
                raise ValueError(f"{k}: expected shape {shape}, got {p[k].shape}")
        if self.a_scale.shape != (L,) or self.rho_max.shape != (L,):
            raise ValueError(f"a_scale/rho_max: expected shape ({L},)")

    def _check_stability(self) -> None:
        """Re-derive every layer's clamp from its raw factors (float64 SVD).

        Fails closed on any mismatch or on sigma_max(A_l) >= 1. Also records
        the per-layer spectral radius (max |eigenvalue|, <= sigma_max by
        definition) and the compound contraction product(rho_l).
        """
        L = self.cfg["rnn_layers"]
        if np.any(~np.isfinite(self.a_scale)) or np.any((self.a_scale <= 0.0) | (self.a_scale > 1.0)):
            raise ValueError("a_scale entries must be in (0, 1]")
        if np.any(~np.isfinite(self.rho_max)) or np.any((self.rho_max <= 0.0) | (self.rho_max >= 1.0)):
            raise ValueError("rho_max entries must be strictly between 0 and 1")
        p = self.p
        radii = np.empty(L, dtype=np.float64)
        for l in range(L):
            dvec = 1.0 / (1.0 + np.exp(-p["lam"][l].astype(np.float64)))
            raw = np.diag(dvec) + p["U_A"][l].astype(np.float64) @ p["V_A"][l].astype(np.float64).T
            sigma_raw = float(np.linalg.svd(raw, compute_uv=False)[0])
            expect = min(1.0, self.rho_max[l] / sigma_raw) if sigma_raw > 0.0 else 1.0
            if not np.isclose(self.a_scale[l], expect, rtol=1e-4, atol=0.0):
                raise ValueError(
                    f"a_scale[{l}] {self.a_scale[l]:.6g} disagrees with SVD-derived {expect:.6g}")
            A_l = self.a_scale[l] * raw
            sigma_max = self.a_scale[l] * sigma_raw
            if not sigma_max < 1.0:
                raise ValueError(f"layer {l}: unstable transition sigma_max(A) = {sigma_max:.6g} >= 1")
            radii[l] = float(np.max(np.abs(np.linalg.eigvals(A_l))))
        self.spectral_radii = radii
        self.composite_contraction = float(np.prod(radii))

    def layer_spectral_radii(self) -> np.ndarray:
        """rho(A_l) per RNN layer, float64, computed once in `_check_stability`."""
        return self.spectral_radii.copy()

    # ------------------------------------------------------------- forward

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return _rms((x - self.p["mu"]) @ self.p["W_in"], self.eps)

    def _pair_features(self, z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
        u = cross_difference_features(z_a.astype(np.float32), z_b.astype(np.float32))
        return _rms(u @ self.p["W_pair"], self.eps)

    def _think(self, z: np.ndarray) -> np.ndarray:
        """L residual layers; each runs its own T-step diag+low-rank think loop.

        Per layer: u_l = U_B_l (V_B_l^T h), then T steps of
        state = a_scale_l*(dvec_l*state + U_A_l (V_A_l^T state)) + u_l,
        h <- h + state_T. The dense (d, d) A_l is never formed here.
        """
        p, T = self.p, self.cfg["think_steps"]
        h = z
        for l in range(self.cfg["rnn_layers"]):
            u = p["U_B"][l] @ (p["V_B"][l].T @ h)
            state = np.zeros_like(h)
            scale, dvec = np.float32(self.a_scale[l]), self._dvec[l]
            for _ in range(T):
                state = scale * (dvec * state + p["U_A"][l] @ (p["V_A"][l].T @ state)) + u
            h = h + state
        return h

    def _set_block(self, X: np.ndarray) -> np.ndarray:
        p, K, d = self.p, X.shape[0], self.cfg["d"]
        nh = self.cfg["n_heads"]
        dh = d // nh
        for i in range(self.cfg["set_layers"]):
            Y = _rms(X, self.eps)

            def heads(w: str, b: str) -> np.ndarray:
                return (Y @ p[w][i].T + p[b][i]).reshape(K, nh, dh).transpose(1, 0, 2)

            Q, Kh, V = heads("wq", "bq"), heads("wk", "bk"), heads("wv", "bv")
            att = _softmax((Q @ Kh.transpose(0, 2, 1)) / np.float32(np.sqrt(dh)))
            O = (att @ V).transpose(1, 0, 2).reshape(K, d)
            X = X + O @ p["wo"][i].T + p["bo"][i]
            Z = _rms(X, self.eps)
            gated = _silu(Z @ p["w_gate"][i].T) * (Z @ p["w_up"][i].T)
            X = X + gated @ p["w_down"][i].T
        return X

    def score(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        D = self.cfg["in_dim"]
        q, C = _finite(q, "q"), _finite(C, "C")
        if q.shape != (D,):
            raise ValueError(f"q must have shape ({D},)")
        if C.ndim != 2 or C.shape[1] != D or C.shape[0] < 1:
            raise ValueError(f"C must have shape (K>=1, {D})")
        q_out = self._think(self._encode(q))
        H = self._set_block(self._encode(C))
        return (H @ (q_out @ self.p["W_s"]) / np.float32(np.sqrt(self.cfg["d"]))).astype(np.float32)

    def score_pair(self, x_a: np.ndarray, x_b: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Like `score`, but the query is the cross-difference of two inputs
        (premise/hypothesis, text A/text B) instead of one."""
        D = self.cfg["in_dim"]
        x_a, x_b, C = _finite(x_a, "x_a"), _finite(x_b, "x_b"), _finite(C, "C")
        if x_a.shape != (D,) or x_b.shape != (D,):
            raise ValueError(f"x_a and x_b must have shape ({D},)")
        if C.ndim != 2 or C.shape[1] != D or C.shape[0] < 1:
            raise ValueError(f"C must have shape (K>=1, {D})")
        z_pair = self._pair_features(self._encode(x_a), self._encode(x_b))
        q_out = self._think(z_pair)
        H = self._set_block(self._encode(C))
        return (H @ (q_out @ self.p["W_s"]) / np.float32(np.sqrt(self.cfg["d"]))).astype(np.float32)

    def predict(self, q: np.ndarray, C: np.ndarray) -> int:
        return int(np.argmax(self.score(q, C)))

    def working_set_bytes(self, k: int = 8) -> int:
        c = self.cfg
        params = sum(a.nbytes for a in self.p.values())
        acts = ((k + 1) * c["in_dim"] + 7 * k * c["d"] + 2 * k * c["ffn_dim"]
                + c["n_heads"] * k * k + 3 * c["d"])
        return int(params + 4 * acts)
