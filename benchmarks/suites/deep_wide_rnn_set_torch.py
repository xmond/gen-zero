"""Deep & wide Parallel RNN + deep Set-Transformer adapter: PyTorch module.

Standalone (numpy + torch + stdlib only): mirrors rnn_set_adapter_torch.py's
convention of living next to the GPU extraction scripts, so it must not
import gen_zero. Trainable end-to-end with autograd; `export_npz` writes the
exact schema `gen_zero.causal.deep_wide_rnn_set.DeepWideRNNSetRuntime.from_npz`
reads, so the NumPy CPU runtime is a bit-parity mirror of this module, not an
independent reimplementation.

    z        = rms((x - mu) @ W_in)                          D -> d
    A_l      = scale_l * (diag(sigmoid(lam_l)) + U_A_l V_A_l^T),
               scale_l = min(1, rho_max_l / sigma_max(raw_l))  per-layer clamp
    h        = z; for l in 0..L-1: h = h + think_l(h)          residual depth
    z_pair   = rms([z_a; z_b; z_a - z_b; z_a * z_b] @ W_pair)  cross-difference
    H        = SetBlock(Z_C)   N pre-norm layers: MHA + SwiGLU FFN, no bias,
               no positional encoding (permutation equivariant over K)
    s_k      = (q_out @ W_s) . H_k / sqrt(d)

Each layer's think loop uses the Hillis-Steele doubling scan
(`lti_prefix_scan`, identical operator to parallel_rnn_lora.combine), so the
per-layer clamp is recomputed on every forward: sigma_max(A_l) <= rho_max_l
holds for whatever weights the optimiser produces, not just at init.
"""
from __future__ import annotations

import json
import math
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-6  # shared with the NumPy runtime; keep in sync


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)


def lti_prefix_scan(A: torch.Tensor, u: torch.Tensor, steps: int) -> torch.Tensor:
    """All prefixes h_1..h_T of h_t = A h_{t-1} + u (h_0 = 0), batched over B.

    A: (d, d) constant transition, u: (B, d) constant input. Returns (T, B, d).
    Identical Hillis-Steele doubling scan to rnn_set_adapter_torch.py.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    A_s = A.unsqueeze(0).expand(steps, -1, -1)
    u_s = u.unsqueeze(0).expand(steps, -1, -1)
    stride = 1
    while stride < steps:
        a_l, a_e = A_s[stride:], A_s[:-stride]
        A_s = torch.cat([A_s[:stride], a_l @ a_e], dim=0)
        u_s = torch.cat(
            [u_s[:stride], torch.einsum("tij,tbj->tbi", a_l, u_s[:-stride]) + u_s[stride:]],
            dim=0,
        )
        stride *= 2
    return u_s


class _RNNLayer(nn.Module):
    """One residual low-rank recurrent layer with its own Lyapunov clamp."""

    def __init__(self, d: int, rank: int, think_steps: int, rho_max: float) -> None:
        super().__init__()
        if not math.isfinite(rho_max) or not (0.0 < rho_max < 1.0):
            raise ValueError("rho_max must be strictly between 0 and 1")
        self.think_steps = think_steps
        self.rho_max = float(rho_max)
        s = 1.0 / math.sqrt(d)
        self.lam = nn.Parameter(torch.randn(d))
        self.U_A = nn.Parameter(torch.randn(d, rank) * s)
        self.V_A = nn.Parameter(torch.randn(d, rank) * s)
        self.U_B = nn.Parameter(torch.randn(d, rank) * s)
        self.V_B = nn.Parameter(torch.randn(d, rank) * s)

    def transition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = torch.diag(torch.sigmoid(self.lam)) + self.U_A @ self.V_A.T
        sigma = torch.linalg.matrix_norm(raw, ord=2)
        scale = torch.clamp(torch.tensor(self.rho_max, device=raw.device) / sigma, max=1.0)
        return scale * raw, scale

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h + think(h): the residual update contributed by this layer."""
        A, _ = self.transition()
        u = h @ (self.U_B @ self.V_B.T).T
        state = lti_prefix_scan(A, u, self.think_steps)[-1]
        return h + state

    @torch.no_grad()
    def spectral_radius(self) -> float:
        A, _ = self.transition()
        return float(torch.linalg.eigvals(A.detach().cpu().double()).abs().max())


class _SetLayer(nn.Module):
    """X = X + MHA(rms(X)); X = X + SwiGLU(rms(X)). Pre-norm, no bias in the FFN."""

    def __init__(self, d: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.w_gate = nn.Linear(d, ffn_dim, bias=False)
        self.w_up = nn.Linear(d, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d, bias=False)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, K, d = X.shape
        nh, dh = self.n_heads, d // self.n_heads
        Y = rms_normalize(X)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, K, nh, dh).transpose(1, 2)

        Q, Kt, V = heads(self.wq(Y)), heads(self.wk(Y)), heads(self.wv(Y))
        att = (Q @ Kt.transpose(-1, -2)) / math.sqrt(dh)
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
        att = torch.softmax(att, dim=-1)
        O = (att @ V).transpose(1, 2).reshape(B, K, d)
        X = X + self.wo(O)
        Z = rms_normalize(X)
        gated = F.silu(self.w_gate(Z)) * self.w_up(Z)
        return X + self.w_down(gated)


class DeepWideRNNSetAdapter(nn.Module):
    def __init__(self, in_dim: int, d: int = 512, rank: int = 16,
                 think_steps: int = 6, rnn_layers: int = 2,
                 rho_max_schedule: List[float] = None,
                 n_heads: int = 8, set_layers: int = 2, ffn_mult: int = 4) -> None:
        super().__init__()
        for name, v in (("in_dim", in_dim), ("d", d), ("rank", rank), ("think_steps", think_steps),
                        ("rnn_layers", rnn_layers), ("n_heads", n_heads),
                        ("set_layers", set_layers), ("ffn_mult", ffn_mult)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d > in_dim:
            raise ValueError("d must not exceed in_dim")
        if rank > d:
            raise ValueError("rank must not exceed d")
        if d % n_heads:
            raise ValueError("d must be divisible by n_heads")
        if rho_max_schedule is None:
            # Distinct per-layer timescales by default: earlier layers decay
            # faster (short memory), later layers decay slower (long memory).
            rho_max_schedule = [0.95 - 0.1 * l for l in range(rnn_layers)]
            rho_max_schedule = [max(0.5, r) for r in rho_max_schedule]
        if len(rho_max_schedule) != rnn_layers:
            raise ValueError("rho_max_schedule must have one entry per rnn_layer")

        self.in_dim, self.d, self.rank = in_dim, d, rank
        self.think_steps, self.n_heads = think_steps, n_heads
        self.rnn_layers_n, self.set_layers_n = rnn_layers, set_layers
        self.ffn_dim = ffn_mult * d
        self.rho_max_schedule = [float(r) for r in rho_max_schedule]

        self.register_buffer("mu", torch.zeros(in_dim))
        self.W_in = nn.Parameter(torch.randn(in_dim, d) / math.sqrt(in_dim))

        self.rnn = nn.ModuleList(
            _RNNLayer(d, rank, think_steps, rho_max_schedule[l]) for l in range(rnn_layers)
        )
        s = 1.0 / math.sqrt(4 * d)
        self.W_pair = nn.Parameter(torch.randn(4 * d, d) * s)

        self.set_block_layers = nn.ModuleList(
            _SetLayer(d, n_heads, self.ffn_dim) for _ in range(set_layers)
        )
        self.W_s = nn.Parameter(torch.eye(d))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize((x.float() - self.mu) @ self.W_in)

    def think(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.rnn:
            h = layer(h)
        return h

    def pair_features(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        u = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b], dim=-1)
        return rms_normalize(u @ self.W_pair)

    def set_block(self, Z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = Z
        for layer in self.set_block_layers:
            X = layer(X, mask)
        return X

    def forward(self, q: torch.Tensor, C: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_out = self.think(self.encode(q))
        H = self.set_block(self.encode(C), mask)
        logits = torch.einsum("bkd,bd->bk", H, q_out @ self.W_s) / math.sqrt(self.d)
        return logits.masked_fill(~mask, -1e9)

    def forward_pair(self, x_a: torch.Tensor, x_b: torch.Tensor, C: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        z_pair = self.pair_features(self.encode(x_a), self.encode(x_b))
        q_out = self.think(z_pair)
        H = self.set_block(self.encode(C), mask)
        logits = torch.einsum("bkd,bd->bk", H, q_out @ self.W_s) / math.sqrt(self.d)
        return logits.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def spectral_radii(self) -> List[float]:
        return [layer.spectral_radius() for layer in self.rnn]

    @torch.no_grad()
    def export_npz(self, path, meta: dict) -> None:
        def f32(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().float().numpy()

        def stack_rnn(get: Callable[[_RNNLayer], torch.Tensor]) -> np.ndarray:
            return np.stack([f32(get(layer)) for layer in self.rnn])

        def stack_set(get: Callable[[_SetLayer], torch.Tensor]) -> np.ndarray:
            return np.stack([f32(get(layer)) for layer in self.set_block_layers])

        a_scale = np.array([float(layer.transition()[1]) for layer in self.rnn], dtype=np.float64)
        rho_max = np.array(self.rho_max_schedule, dtype=np.float64)

        np.savez(
            path,
            mu=f32(self.mu), W_in=f32(self.W_in),
            lam=stack_rnn(lambda m: m.lam), U_A=stack_rnn(lambda m: m.U_A),
            V_A=stack_rnn(lambda m: m.V_A), U_B=stack_rnn(lambda m: m.U_B),
            V_B=stack_rnn(lambda m: m.V_B),
            W_pair=f32(self.W_pair), W_s=f32(self.W_s),
            wq=stack_set(lambda m: m.wq.weight), bq=stack_set(lambda m: m.wq.bias),
            wk=stack_set(lambda m: m.wk.weight), bk=stack_set(lambda m: m.wk.bias),
            wv=stack_set(lambda m: m.wv.weight), bv=stack_set(lambda m: m.wv.bias),
            wo=stack_set(lambda m: m.wo.weight), bo=stack_set(lambda m: m.wo.bias),
            w_gate=stack_set(lambda m: m.w_gate.weight),
            w_up=stack_set(lambda m: m.w_up.weight),
            w_down=stack_set(lambda m: m.w_down.weight),
            a_scale=a_scale, rho_max=rho_max,
            in_dim=np.int64(self.in_dim), d=np.int64(self.d), rank=np.int64(self.rank),
            think_steps=np.int64(self.think_steps), rnn_layers=np.int64(self.rnn_layers_n),
            n_heads=np.int64(self.n_heads), set_layers=np.int64(self.set_layers_n),
            ffn_dim=np.int64(self.ffn_dim), rms_eps=np.float64(RMS_EPS),
            meta_json=np.array(json.dumps(meta, sort_keys=True)),
        )
