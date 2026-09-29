"""LoRA side-car parallel RNN and the semigroup associative-scan operator.

Spec: docs/research/lora_parallel_rnn_and_fractal_cpu_zero_token.md, section 2.

The state recurrence is h_t = A_t h_{t-1} + B_t x_t. Writing a transition step
as the tuple g_t = (A_t, u_t) with u_t = B_t x_t, composition is:

    g_later (x) g_earlier = (A_later @ A_earlier, A_later @ u_earlier + u_later)

This composition is associative (matrix multiplication is associative and
distributes over the affine update) but NOT commutative, so scan order along
the sequence axis must be preserved exactly. `associative_scan` performs a
Blelloch work-efficient scan (up-sweep + down-sweep, O(log T) depth, O(T)
work); `sequential_scan` is the O(T) reference recursion used to check it.

`ParallelRNNLoRAAdapter` parameterises A as a *time-invariant* matrix
diag(sigmoid(lambda)) + U_A @ V_A^T (a diagonal decay plus a rank-r update,
per the spec formula in section 2.2 -- the formula has no x_t dependence, so
A does not vary with the input here) and B as U_B @ V_B^T. The spectral norm
of A is clamped strictly below 1.0 (Lyapunov stability, spec section 5.3) by
a single uniform rescale computed once at construction. `step()` never forms
the dense d x d matrix: it applies diag/low-rank factors directly, so the
per-step working set is O(d + r), not O(d^2), and holds no growing cache.

Pure NumPy. Every public entry point rejects non-finite input immediately
(fail-closed); nothing here imports `re` or inspects text/labels.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Element = Tuple[np.ndarray, np.ndarray]


def _finite(value, name, *, dtype=float) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return a


def _square_stack(value, name) -> np.ndarray:
    a = _finite(value, name)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"{name}: expected shape (T, d, d)")
    return a


def _vector_stack(value, name) -> np.ndarray:
    a = _finite(value, name)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected shape (T, d)")
    return a


def combine(earlier: Element, later: Element) -> Element:
    """Semigroup operator: compose `earlier` then `later`.

    (A_e, u_e) applied first, (A_l, u_l) applied second. Returns the single
    transition equivalent to running both in sequence:
        A = A_l @ A_e,  u = A_l @ u_e + u_l
    Batched over any number of leading axes; the trailing two axes of A are
    the (d, d) matrix and the trailing axis of u is the d-vector.
    """
    a_e, u_e = earlier
    a_l, u_l = later
    a_e = _finite(a_e, "earlier.A")
    u_e = _finite(u_e, "earlier.u")
    a_l = _finite(a_l, "later.A")
    u_l = _finite(u_l, "later.u")
    if a_e.shape != a_l.shape or a_e.shape[:-1] != u_e.shape or u_e.shape != u_l.shape:
        raise ValueError("combine: mismatched shapes between earlier/later elements")
    a = np.einsum("...ij,...jk->...ik", a_l, a_e)
    u = np.einsum("...ij,...j->...i", a_l, u_e) + u_l
    return a, u


def sequential_scan(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """O(T) reference recursion: h_t = A_t @ h_{t-1} + u_t, h_{-1} = 0."""
    A = _square_stack(A, "A")
    u = _vector_stack(u, "u")
    if A.shape[0] != u.shape[0] or A.shape[1] != u.shape[1]:
        raise ValueError("sequential_scan: A and u disagree on T or d")
    T, d = u.shape
    h = np.zeros(d, dtype=u.dtype)
    out = np.empty_like(u)
    for t in range(T):
        h = A[t] @ h + u[t]
        out[t] = h
    return out


def associative_scan(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Blelloch parallel scan: O(log T) depth, O(T) work, exact vs `sequential_scan`.

    Up-sweep builds inclusive combines bottom-up over a padded, power-of-two
    length array (identity element (I, 0) fills the pad). Down-sweep then
    turns those into the exclusive prefix for every original index; the
    inclusive result is recovered as h_t = A_t @ exclusive_u_t + u_t.
    """
    A = _square_stack(A, "A")
    u = _vector_stack(u, "u")
    if A.shape[0] != u.shape[0] or A.shape[1] != u.shape[1]:
        raise ValueError("associative_scan: A and u disagree on T or d")
    T, d = u.shape
    n = 1 << (T - 1).bit_length() if T > 1 else 1
    eye = np.eye(d, dtype=A.dtype)
    Apad = np.tile(eye, (n, 1, 1))
    Upad = np.zeros((n, d), dtype=u.dtype)
    Apad[:T], Upad[:T] = A, u

    stride = 1
    while stride < n:
        i = np.arange(stride * 2 - 1, n, stride * 2)
        j = i - stride
        Apad[i], Upad[i] = combine((Apad[j], Upad[j]), (Apad[i], Upad[i]))
        stride *= 2

    Apad[-1] = eye
    Upad[-1] = 0.0

    stride = n // 2
    while stride >= 1:
        i = np.arange(stride * 2 - 1, n, stride * 2)
        j = i - stride
        left = (Apad[j].copy(), Upad[j].copy())
        parent = (Apad[i].copy(), Upad[i].copy())
        Apad[j], Upad[j] = parent
        Apad[i], Upad[i] = combine(parent, left)
        stride //= 2

    excl_u = Upad[:T]
    return np.einsum("tij,tj->ti", A, excl_u) + u


def spectral_normalize(matrix: np.ndarray, rho_max: float) -> np.ndarray:
    """Rescale a square matrix so its largest singular value is <= rho_max.

    Uniform scaling preserves the eigenvector/singular-vector structure and
    guarantees sigma_max(c * M) = c * sigma_max(M) exactly, so this is a
    provable (not heuristic) Lyapunov-stability enforcement: with
    0 < rho_max < 1, the returned matrix has spectral radius < 1.0.
    """
    matrix = _finite(matrix, "matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("spectral_normalize: matrix must be square")
    if not np.isfinite(rho_max) or not (0.0 < rho_max < 1.0):
        raise ValueError("spectral_normalize: rho_max must be strictly between 0 and 1")
    sigma_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
    if sigma_max <= 0.0:
        return matrix.copy()
    scale = min(1.0, rho_max / sigma_max)
    return matrix * scale


class ParallelRNNLoRAAdapter:
    """Time-invariant LoRA side-car linear recurrence, h_t = A h_{t-1} + B x_t.

    A = scale * (diag(sigmoid(lambda)) + U_A @ V_A^T), spectral-normalised so
    sigma_max(A) <= rho_max < 1.0 (strict Lyapunov stability). B = U_B @ V_B^T.
    Parameters are O(dim * rank); `step()` never materialises the dense
    (dim, dim) matrix, so its working set is O(dim + rank), constant across
    calls, with no dynamically-growing cache of any kind.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        input_dim: Optional[int] = None,
        rho_max: float = 0.95,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if rank > dim:
            raise ValueError("rank must not exceed dim (this is a low-rank side-car)")
        input_dim = dim if input_dim is None else input_dim
        if isinstance(input_dim, bool) or not isinstance(input_dim, (int, np.integer)) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not np.isfinite(rho_max) or not (0.0 < rho_max < 1.0):
            raise ValueError("rho_max must be strictly between 0 and 1")

        self.dim = int(dim)
        self.rank = int(rank)
        self.input_dim = int(input_dim)
        self.rho_max = float(rho_max)
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        scale_init = 1.0 / np.sqrt(self.rank)
        self.lambda_ = rng.normal(0.0, 1.0, self.dim).astype(dtype)
        self.U_A = (rng.normal(0.0, scale_init, (self.dim, self.rank))).astype(dtype)
        self.V_A = (rng.normal(0.0, scale_init, (self.dim, self.rank))).astype(dtype)
        self.U_B = (rng.normal(0.0, scale_init, (self.dim, self.rank))).astype(dtype)
        self.V_B = (rng.normal(0.0, scale_init, (self.input_dim, self.rank))).astype(dtype)

        self._scale = self._compute_scale()

    def _dvec(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.lambda_))

    def _compute_scale(self) -> float:
        raw = np.diag(self._dvec()) + self.U_A @ self.V_A.T
        sigma_max = float(np.linalg.svd(raw, compute_uv=False)[0])
        if sigma_max <= 0.0:
            return 1.0
        return min(1.0, self.rho_max / sigma_max)

    def transition_matrix(self) -> np.ndarray:
        """Dense (dim, dim) A, for audit / `forward`. Not used by `step`."""
        raw = np.diag(self._dvec()) + self.U_A @ self.V_A.T
        return self._scale * raw

    def input_matrix(self) -> np.ndarray:
        """Dense (dim, input_dim) B, for audit / `forward`."""
        return self.U_B @ self.V_B.T

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """One O(dim*rank) step; never allocates a (dim, dim) array."""
        x_t = _finite(x_t, "x_t", dtype=self.dtype)
        h_prev = _finite(h_prev, "h_prev", dtype=self.dtype)
        if x_t.shape != (self.input_dim,):
            raise ValueError(f"x_t must have shape ({self.input_dim},)")
        if h_prev.shape != (self.dim,):
            raise ValueError(f"h_prev must have shape ({self.dim},)")
        a_term = self._dvec() * h_prev + self.U_A @ (self.V_A.T @ h_prev)
        b_term = self.U_B @ (self.V_B.T @ x_t)
        return self._scale * a_term + b_term

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """Full-sequence h_1..h_T via `associative_scan`. O(T) dense audit path."""
        x_seq = _finite(x_seq, "x_seq", dtype=self.dtype)
        if x_seq.ndim != 2 or x_seq.shape[1] != self.input_dim:
            raise ValueError(f"x_seq must have shape (T, {self.input_dim})")
        T = x_seq.shape[0]
        A = np.broadcast_to(self.transition_matrix(), (T, self.dim, self.dim)).copy()
        u = x_seq @ self.input_matrix().T
        return associative_scan(A, u)

    def save(self, path) -> None:
        """Round-trip everything the constructor derives state from: the raw
        arrays (not `transition_matrix()`/`input_matrix()`, which are derived),
        plus the scalar shape/stability parameters needed to validate them on load."""
        with open(path, "wb") as stream:
            np.savez(stream, lambda_=self.lambda_, U_A=self.U_A, V_A=self.V_A,
                     U_B=self.U_B, V_B=self.V_B, dim=np.int64(self.dim),
                     rank=np.int64(self.rank), input_dim=np.int64(self.input_dim),
                     rho_max=np.float64(self.rho_max), dtype=np.dtype(self.dtype).name)

    @classmethod
    def load(cls, path) -> "ParallelRNNLoRAAdapter":
        required = {"lambda_", "U_A", "V_A", "U_B", "V_B", "dim", "rank", "input_dim", "rho_max", "dtype"}
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                raise ValueError("invalid ParallelRNNLoRAAdapter checkpoint schema")
            dim = int(data["dim"])
            rank = int(data["rank"])
            input_dim = int(data["input_dim"])
            rho_max = float(data["rho_max"])
            dtype = np.dtype(str(data["dtype"]))
            lambda_ = _finite(data["lambda_"], "lambda_", dtype=dtype)
            u_a = _finite(data["U_A"], "U_A", dtype=dtype)
            v_a = _finite(data["V_A"], "V_A", dtype=dtype)
            u_b = _finite(data["U_B"], "U_B", dtype=dtype)
            v_b = _finite(data["V_B"], "V_B", dtype=dtype)
        if dim <= 0 or rank <= 0 or input_dim <= 0 or rank > dim:
            raise ValueError("ParallelRNNLoRAAdapter checkpoint has invalid dim/rank/input_dim")
        if not (0.0 < rho_max < 1.0):
            raise ValueError("rho_max must be strictly between 0 and 1")
        if (lambda_.shape != (dim,) or u_a.shape != (dim, rank) or v_a.shape != (dim, rank)
                or u_b.shape != (dim, rank) or v_b.shape != (input_dim, rank)):
            raise ValueError("ParallelRNNLoRAAdapter checkpoint shapes are inconsistent")
        adapter = cls.__new__(cls)
        adapter.dim, adapter.rank, adapter.input_dim = dim, rank, input_dim
        adapter.rho_max, adapter.dtype = rho_max, dtype
        adapter.lambda_, adapter.U_A, adapter.V_A = lambda_, u_a, v_a
        adapter.U_B, adapter.V_B = u_b, v_b
        adapter._scale = adapter._compute_scale()
        return adapter

    def working_set_bytes(self) -> int:
        """Bytes resident during `step`: parameters + one h buffer + one x buffer.

        Excludes `transition_matrix()`/`forward()`, which are O(dim^2) audit
        paths, never called from the constant-memory inference loop.
        """
        params = sum(
            arr.nbytes
            for arr in (self.lambda_, self.U_A, self.V_A, self.U_B, self.V_B)
        )
        itemsize = np.dtype(self.dtype).itemsize
        return int(params + self.dim * itemsize + self.input_dim * itemsize)
