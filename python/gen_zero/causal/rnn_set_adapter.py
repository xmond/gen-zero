"""Parallel RNN + Set-Attention candidate scorer: pure NumPy CPU runtime.

Mirrors benchmarks/suites/rnn_set_adapter_torch.py (the GPU trainer) in
float32. Given a teacher query state q (D,) and K candidate states C (K, D):

    z      = rms((x - mu) @ W_in)                       D -> d projection
    h_t    = a_scale*(sig(lam)*h + U_A (V_A^T h)) + U_B (V_B^T zq),  t=1..T, h_0=0
    q_out  = zq + h_T
    H      = n_layers x [X += MHA(rms X); X += W2 relu(W1 rms X + b1) + b2]
    s_k    = (q_out @ W_s) . H_k / sqrt(d)

The think loop uses the diag + low-rank step form, as in
ParallelRNNLoRAAdapter.step: it never forms the dense (d, d) A, so each of
the T steps costs O(d*r). The set block has no positional encoding, so the
scores are permutation equivariant over the K candidates.

CPU cost per record: the projection dominates at O((K+1)*D*d) multiply-adds
(D=4096, d=256: ~1M per candidate). The set block is O(n_layers*(K*d^2 + K^2*d))
and the think loop O(T*d*r). No state persists between calls.

On load, the spectral clamp is re-derived from the raw factors by SVD and
must match the stored `a_scale` (rtol 1e-4), and sigma_max(A) must be < 1
(Lyapunov stability). Every entry point rejects non-finite input (fail-closed).
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

_ARRAYS = ("mu", "W_in", "lam", "U_A", "V_A", "U_B", "V_B", "W_s",
           "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "w1", "b1", "w2", "b2")
_INTS = ("in_dim", "d", "rank", "think_steps", "n_heads", "n_layers", "ffn_dim")


def _finite(value, name, *, dtype=np.float32) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return a


def _rms(x: np.ndarray, eps: np.float32) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class RNNSetAdapterRuntime:
    def __init__(self, arrays: Dict[str, np.ndarray], config: Dict[str, int],
                 a_scale: float, rho_max: float, rms_eps: float, meta: dict) -> None:
        self.cfg = dict(config)
        self.p = {k: _finite(v, k) for k, v in arrays.items()}
        self.a_scale = np.float32(a_scale)
        self.rho_max = float(rho_max)
        self.eps = np.float32(rms_eps)
        self.meta = meta
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"rms_eps must be a finite value > 0, got {rms_eps!r}")
        if not np.isfinite(self.rho_max) or self.rho_max <= 0:
            raise ValueError(f"rho_max must be a finite value > 0, got {rho_max!r}")
        self._check_shapes()
        self._check_stability()
        self._dvec = (1.0 / (1.0 + np.exp(-self.p["lam"].astype(np.float64)))).astype(np.float32)

    @classmethod
    def from_npz(cls, path) -> "RNNSetAdapterRuntime":
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in _ARRAYS + _INTS + ("a_scale", "rho_max", "rms_eps", "meta_json")
                       if k not in z.files]
            if missing:
                raise ValueError(f"from_npz: missing keys {missing}")
            arrays = {k: np.asarray(z[k], dtype=np.float32) for k in _ARRAYS}
            config = {k: int(z[k]) for k in _INTS}
            a_scale, rho_max, rms_eps = float(z["a_scale"]), float(z["rho_max"]), float(z["rms_eps"])
            meta = json.loads(str(z["meta_json"]))
        return cls(arrays, config, a_scale, rho_max, rms_eps, meta)

    def _check_shapes(self) -> None:
        c, p = self.cfg, self.p
        D, d, r, L, F = c["in_dim"], c["d"], c["rank"], c["n_layers"], c["ffn_dim"]
        if min(c.values()) <= 0 or d % c["n_heads"]:
            raise ValueError("config: sizes must be positive and d divisible by n_heads")
        want = {"mu": (D,), "W_in": (D, d), "lam": (d,), "U_A": (d, r), "V_A": (d, r),
                "U_B": (d, r), "V_B": (d, r), "W_s": (d, d),
                "wq": (L, d, d), "wk": (L, d, d), "wv": (L, d, d), "wo": (L, d, d),
                "bq": (L, d), "bk": (L, d), "bv": (L, d), "bo": (L, d),
                "w1": (L, F, d), "b1": (L, F), "w2": (L, d, F), "b2": (L, d)}
        for k, shape in want.items():
            if p[k].shape != shape:
                raise ValueError(f"{k}: expected shape {shape}, got {p[k].shape}")

    def _check_stability(self) -> None:
        """Re-derive the clamp from raw factors (float64 SVD); fail closed on mismatch."""
        if not np.isfinite(self.a_scale) or not (0.0 < self.a_scale <= 1.0):
            raise ValueError("a_scale must be in (0, 1]")
        p = self.p
        dvec = 1.0 / (1.0 + np.exp(-p["lam"].astype(np.float64)))
        raw = np.diag(dvec) + p["U_A"].astype(np.float64) @ p["V_A"].astype(np.float64).T
        sigma_raw = float(np.linalg.svd(raw, compute_uv=False)[0])
        expect = min(1.0, self.rho_max / sigma_raw) if sigma_raw > 0.0 else 1.0
        if not np.isclose(float(self.a_scale), expect, rtol=1e-4, atol=0.0):
            raise ValueError(f"a_scale {float(self.a_scale):.6g} disagrees with SVD-derived {expect:.6g}")
        self.sigma_max_A = float(self.a_scale) * sigma_raw
        if not self.sigma_max_A < 1.0:
            raise ValueError(f"unstable transition: sigma_max(A) = {self.sigma_max_A:.6g} >= 1")

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return _rms((x - self.p["mu"]) @ self.p["W_in"], self.eps)

    def _think(self, zq: np.ndarray) -> np.ndarray:
        """T diag + low-rank steps, O(d*r) each; the dense A is never formed."""
        p = self.p
        u = p["U_B"] @ (p["V_B"].T @ zq)
        h = np.zeros_like(zq)
        for _ in range(self.cfg["think_steps"]):
            h = self.a_scale * (self._dvec * h + p["U_A"] @ (p["V_A"].T @ h)) + u
        return zq + h

    def _set_block(self, X: np.ndarray) -> np.ndarray:
        p, K, d = self.p, X.shape[0], self.cfg["d"]
        nh = self.cfg["n_heads"]
        dh = d // nh
        for i in range(self.cfg["n_layers"]):
            Y = _rms(X, self.eps)

            def heads(w: str, b: str) -> np.ndarray:
                return (Y @ p[w][i].T + p[b][i]).reshape(K, nh, dh).transpose(1, 0, 2)

            Q, Kh, V = heads("wq", "bq"), heads("wk", "bk"), heads("wv", "bv")
            att = _softmax((Q @ Kh.transpose(0, 2, 1)) / np.float32(np.sqrt(dh)))
            O = (att @ V).transpose(1, 0, 2).reshape(K, d)
            X = X + O @ p["wo"][i].T + p["bo"][i]
            hid = np.maximum(_rms(X, self.eps) @ p["w1"][i].T + p["b1"][i], 0.0)
            X = X + hid @ p["w2"][i].T + p["b2"][i]
        return X

    def score(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        D = self.cfg["in_dim"]
        q = _finite(q, "q")
        C = _finite(C, "C")
        if q.shape != (D,):
            raise ValueError(f"q must have shape ({D},)")
        if C.ndim != 2 or C.shape[1] != D or C.shape[0] < 1:
            raise ValueError(f"C must have shape (K>=1, {D})")
        q_out = self._think(self._encode(q))
        H = self._set_block(self._encode(C))
        out = (H @ (q_out @ self.p["W_s"]) / np.float32(np.sqrt(self.cfg["d"]))).astype(np.float32)
        if not np.all(np.isfinite(out)):
            raise ValueError("score produced non-finite values (numeric overflow or unstable intermediate)")
        return out

    def predict(self, q: np.ndarray, C: np.ndarray) -> int:
        s = self.score(q, C)
        if s.size == 0 or not np.all(np.isfinite(s)):
            raise ValueError("predict: score is empty or non-finite, refusing to argmax")
        return int(np.argmax(s))

    def working_set_bytes(self, k: int = 77) -> int:
        """Parameters plus float32 activations for one record with k candidates.

        Activations: inputs (k+1)*D, residual/normed/Q/K/V/O ~ 7*k*d, FFN
        hidden k*ffn_dim, attention n_heads*k^2, think state 3*d.
        """
        c = self.cfg
        params = sum(a.nbytes for a in self.p.values())
        acts = ((k + 1) * c["in_dim"] + 7 * k * c["d"] + k * c["ffn_dim"]
                + c["n_heads"] * k * k + 3 * c["d"])
        return int(params + 4 * acts)
