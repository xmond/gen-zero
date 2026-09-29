"""Gen-Zero Deep Equilibrium (DEQ) Fixed-Point Thinking Module.

Implements a Deep Equilibrium Model for iterative reasoning:
- Forward: Anderson-accelerated fixed-point iteration z* = f(z*, x)
- Backward: Implicit Function Theorem (IFT) gradient via custom autograd Function
  that re-solves the fixed-point equation in the backward pass for O(1) memory training.

Key components:
1. DEQThinkingBlock: residual-MLP reasoning layer parameterizing f(z, x).
2. AndersonAccelerator: NumPy Anderson mixing (reference implementation).
3. DEQThinkingFunction: torch.autograd.Function with IFT backward. It returns
   gradients for x AND for every block parameter.
4. DEQThinkingModule: nn.Module wrapper for seamless training.

Dropout is deliberately absent: a stochastic f has no well-defined fixed point.

Reference:
  Bai et al. "Deep Equilibrium Models" (NeurIPS 2019).
  Anderson "Iterative Procedures for Nonlinear Integral Equations" (JACM 1965).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Pure NumPy / dataclass types (no torch dependency)
# ---------------------------------------------------------------------------

@dataclass
class DEQSolverState:
    """Telemetry from a DEQ fixed-point solve."""
    converged: bool
    n_iter: int
    final_residual: float
    residual_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "final_residual": round(self.final_residual, 8),
            "residual_history": [round(r, 8) for r in self.residual_history],
        }


class AndersonAccelerator:
    """Anderson acceleration for fixed-point iteration.

    Maintains a history of m previous iterates and their residuals, solving a
    small least-squares problem to produce an accelerated next iterate:

        z_{k+1} = (1 - sum_i alpha_i) * f(z_k) + sum_i alpha_i * f(z_{k-i})

    where alpha solves min ||F_k * alpha||_2 s.t. sum alpha_i = 1,
    F_k = [f(z_k) - z_k, ..., f(z_{k-m+1}) - z_{k-m+1}].

    Args:
        m: History size (number of past iterates to mix). Default 5.
        beta: Damping factor in (0, 1]. Default 1.0 (no damping).
    """

    def __init__(self, m: int = 5, beta: float = 1.0):
        if m < 1:
            raise ValueError(f"History size m must be >= 1, got {m}")
        self.m = m
        self.beta = beta
        self._reset()

    def _reset(self) -> None:
        self._z_history: List[np.ndarray] = []   # past iterates
        self._f_history: List[np.ndarray] = []   # past f(z) values

    def step(self, z: np.ndarray, fz: np.ndarray) -> np.ndarray:
        """Compute Anderson-accelerated next iterate.

        Args:
            z: Current iterate, shape (d,).
            fz: f(z), shape (d,).

        Returns:
            Accelerated next iterate, shape (d,).
        """
        self._z_history.append(z.copy())
        self._f_history.append(fz.copy())

        # Keep at most m entries
        if len(self._z_history) > self.m:
            self._z_history.pop(0)
            self._f_history.pop(0)

        k = len(self._z_history)
        if k == 1:
            # First step: no history to mix, return f(z) directly.
            return fz.copy()

        # Build residual matrix F = f(z_i) - z_i, shape (d, k)
        F_mat = np.column_stack([
            self._f_history[i] - self._z_history[i] for i in range(k)
        ])

        # Solve min ||F * alpha||_2  s.t. sum alpha = 1
        # Equivalent to: solve (F^T F + lambda * 11^T) alpha = lambda * 1
        # via normal equations with constraint.
        FtF = F_mat.T @ F_mat  # (k, k)
        # Regularise for numerical stability
        reg = 1e-10 * np.eye(k)
        ones = np.ones((k, 1))

        try:
            # Solve the KKT system
            # [ 2*FtF   1 ] [ alpha   ] = [ 0 ]
            # [   1^T   0 ] [ lambda' ] = [ 1 ]
            lhs = np.block([
                [2 * FtF + reg, ones],
                [ones.T,         np.zeros((1, 1))],
            ])
            rhs = np.zeros(k + 1)
            rhs[-1] = 1.0
            sol = np.linalg.solve(lhs, rhs)
            alpha = sol[:k]
        except np.linalg.LinAlgError:
            # Fallback: equal weights = plain Picard
            alpha = np.ones(k) / k

        # Damped mixing
        alpha = self.beta * alpha + (1.0 - self.beta) * np.ones(k) / k

        # z_new = (1 - sum alpha_i) * f(z_k) + sum_i alpha_i * f(z_{k-i})
        z_new = np.zeros_like(z)
        for i in range(k):
            z_new += alpha[i] * self._f_history[i]
        z_new += (1.0 - np.sum(alpha)) * fz

        return z_new


def _np_solve_fixed_point(
    f_fn: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    anderson: AndersonAccelerator,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, DEQSolverState]:
    """Run Anderson-accelerated fixed-point iteration (NumPy).

    Args:
        f_fn: The function f(z) whose fixed point we seek.
        z0: Initial guess, shape (d,).
        anderson: Pre-configured Anderson accelerator.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on relative residual.

    Returns:
        (z_star, solver_state)
    """
    z = z0.copy()
    residuals: List[float] = []

    for it in range(max_iter):
        fz = f_fn(z)
        residual = float(np.linalg.norm(fz - z) / (np.linalg.norm(z) + 1e-8))
        residuals.append(residual)

        if residual < tol:
            return z, DEQSolverState(
                converged=True,
                n_iter=it + 1,
                final_residual=residual,
                residual_history=residuals,
            )

        z_next = anderson.step(z, fz)
        z = z_next

    return z, DEQSolverState(
        converged=False,
        n_iter=max_iter,
        final_residual=residuals[-1] if residuals else float("inf"),
        residual_history=residuals,
    )




# ---------------------------------------------------------------------------
# Torch-dependent components
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class DEQThinkingBlock(nn.Module):
        """Single reasoning step f(z, x).

            z' = LayerNorm(h + MLP(h)),   h = LayerNorm(z + Proj(x))

        Args:
            dim: Dimensionality of the fixed-point state z.
            context_dim: Dimensionality of the conditioning input x.
            hidden_mult: MLP hidden expansion factor. Default 4.
        """

        def __init__(self, dim: int, context_dim: int, hidden_mult: int = 4):
            super().__init__()
            self.dim = dim
            self.context_dim = context_dim
            self.context_proj = nn.Linear(context_dim, dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * hidden_mult),
                nn.GELU(),
                nn.Linear(dim * hidden_mult, dim),
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            # Small output weights keep f near a contraction at init, so the
            # solver converges from step one. Training may move away from it.
            with torch.no_grad():
                self.mlp[2].weight.mul_(0.1)

        def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(z + self.context_proj(x))
            return self.norm2(h + self.mlp(h))

        def init_z0(
            self, batch_size: int, device: torch.device, dtype: torch.dtype,
        ) -> torch.Tensor:
            """Zero initial guess, shape (batch_size, dim)."""
            return torch.zeros(batch_size, self.dim, device=device, dtype=dtype)


    def _anderson_solve(
        f: Callable[[torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        m: int,
        max_iter: int,
        tol: float,
        lam: float = 1e-6,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, DEQSolverState]:
        """Batched Anderson solve of z = f(z). Never builds an autograd graph.

        Each sample (dim 0) gets its own least-squares mixing weights, so
        samples stay independent. Works for affine f too, which is how the
        backward pass reuses it for the IFT linear system.

        Returns the iterate with the lowest worst-case relative residual.
        """
        with torch.no_grad():
            shape = z0.shape
            bsz = shape[0]
            z = z0.reshape(bsz, -1)
            d = z.shape[1]
            m = max(1, m)

            def apply_f(v: torch.Tensor) -> torch.Tensor:
                return f(v.reshape(shape)).reshape(bsz, -1)

            xs = torch.zeros(bsz, m, d, device=z.device, dtype=z.dtype)
            fs = torch.zeros_like(xs)
            eye = torch.eye(m + 1, device=z.device, dtype=z.dtype)
            history: List[float] = []
            best_z, best_res = z, float("inf")

            for k in range(max_iter):
                fz = apply_f(z)
                res = (fz - z).norm(dim=1) / (fz.norm(dim=1) + 1e-8)
                worst = float(res.max())
                history.append(worst)
                if worst < best_res:
                    best_z, best_res = z, worst
                if worst < tol:
                    return z.reshape(shape), DEQSolverState(True, k + 1, worst, history)

                slot = k % m
                xs[:, slot] = z
                fs[:, slot] = fz
                n = min(k + 1, m)
                g = fs[:, :n] - xs[:, :n]                       # (B, n, d)
                gram = g @ g.transpose(1, 2)                    # (B, n, n)
                scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
                kkt = torch.zeros(bsz, n + 1, n + 1, device=z.device, dtype=z.dtype)
                kkt[:, 0, 1:] = 1.0
                kkt[:, 1:, 0] = 1.0
                kkt[:, 1:, 1:] = gram + lam * scale[:, None, None] * eye[1:n + 1, 1:n + 1]
                rhs = torch.zeros(bsz, n + 1, 1, device=z.device, dtype=z.dtype)
                rhs[:, 0] = 1.0
                sol, info = torch.linalg.solve_ex(kkt, rhs)
                alpha = sol[:, 1:, :].transpose(1, 2)           # (B, 1, n)
                bad = (info != 0) | ~torch.isfinite(alpha).all(dim=(1, 2))
                if bool(bad.any()):
                    alpha = torch.where(bad[:, None, None], torch.full_like(alpha, 1.0 / n), alpha)
                z = beta * (alpha @ fs[:, :n]).squeeze(1) \
                    + (1.0 - beta) * (alpha @ xs[:, :n]).squeeze(1)

            fz = apply_f(z)
            res = float(((fz - z).norm(dim=1) / (fz.norm(dim=1) + 1e-8)).max())
            history.append(res)
            if res < best_res:
                best_z, best_res = z, res
            return best_z.reshape(shape), DEQSolverState(
                best_res < tol, max_iter, best_res, history,
            )


    class DEQThinkingFunction(torch.autograd.Function):
        """Fixed-point solve with an implicit-function-theorem backward.

        Forward runs entirely without autograd, so memory does not grow with
        the iteration count. Only (x, z*, params) are saved.

        Backward: at z* = f(z*, x; W), for a loss gradient v = dL/dz*,

            u = (I - J^T)^{-1} v,   J = df/dz at z*
            dL/dx = u^T df/dx,      dL/dW = u^T df/dW

        The linear solve is Anderson-accelerated fixed-point iteration on
        u = J^T u + v, with J^T u taken from vjp calls on one cached graph of
        a single f evaluation. It needs rho(J) < 1.

        Second-order gradients are not supported.

        Call as: apply(x, z0, block, max_iter, tol, anderson_m, stats, *params)
        where params = tuple(block.parameters()). Passing them as inputs is
        what makes autograd route their gradients here.
        """

        @staticmethod
        def forward(ctx, x, z0, block, max_iter, tol, anderson_m, stats, *params):
            if z0 is None:
                z0 = block.init_z0(x.shape[0], x.device, x.dtype)
            z_star, state = _anderson_solve(
                lambda z: block(z, x), z0, anderson_m, max_iter, tol,
            )
            # One extra application puts z* exactly on f's range.
            z_star = block(z_star, x)
            if stats is not None:
                stats["forward"] = state
            ctx.block = block
            ctx.stats = stats
            ctx.cfg = (max_iter, tol, anderson_m)
            ctx.n_params = len(params)
            ctx.save_for_backward(x, z_star, *params)
            return z_star

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            x, z_star, *params = ctx.saved_tensors
            max_iter, tol, anderson_m = ctx.cfg
            need = ctx.needs_input_grad
            need_x = need[0]
            need_p = list(need[7:])

            with torch.enable_grad():
                z_leaf = z_star.detach().requires_grad_(True)
                x_leaf = x.detach().requires_grad_(need_x)
                fz = ctx.block(z_leaf, x_leaf)

                def vjp_z(u: torch.Tensor) -> torch.Tensor:
                    (jtu,) = torch.autograd.grad(fz, z_leaf, grad_outputs=u, retain_graph=True)
                    return jtu

                u, state = _anderson_solve(
                    lambda u: vjp_z(u) + grad_output, grad_output, anderson_m, max_iter, tol,
                )
                if ctx.stats is not None:
                    ctx.stats["backward"] = state

                targets = ([x_leaf] if need_x else []) \
                    + [p for p, n in zip(params, need_p) if n]
                grads = iter(torch.autograd.grad(fz, targets, grad_outputs=u, allow_unused=True)) \
                    if targets else iter(())

            grad_x = next(grads) if need_x else None
            grad_p = [next(grads) if n else None for n in need_p]
            # (x, z0, block, max_iter, tol, anderson_m, stats, *params)
            return (grad_x, None, None, None, None, None, None, *grad_p)


    class DEQThinkingModule(nn.Module):
        """Deep Equilibrium thinking module with O(1)-memory training.

        Args:
            dim: Dimensionality of the fixed-point state z.
            context_dim: Dimensionality of the conditioning input x.
            hidden_mult: MLP hidden expansion factor. Default 4.
            max_iter: Maximum solver iterations (forward and backward). Default 50.
            anderson_m: Anderson history size. Default 5.
            tol: Relative-residual convergence tolerance. Default 1e-5.
        """

        def __init__(
            self,
            dim: int,
            context_dim: int,
            hidden_mult: int = 4,
            max_iter: int = 50,
            anderson_m: int = 5,
            tol: float = 1e-5,
        ):
            super().__init__()
            self.block = DEQThinkingBlock(dim, context_dim, hidden_mult)
            self.max_iter = max_iter
            self.anderson_m = anderson_m
            self.tol = tol
            self._stats: Dict[str, DEQSolverState] = {}

        @property
        def last_state(self) -> Optional[DEQSolverState]:
            """Telemetry of the most recent forward solve."""
            return self._stats.get("forward")

        @property
        def last_backward_state(self) -> Optional[DEQSolverState]:
            """Telemetry of the most recent backward (IFT) solve."""
            return self._stats.get("backward")

        def forward(
            self, x: torch.Tensor, z0: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Solve z* = f(z*, x) and return z*, shape (batch, dim)."""
            return DEQThinkingFunction.apply(
                x, z0, self.block, self.max_iter, self.tol, self.anderson_m,
                self._stats, *self.block.parameters(),
            )

else:
    class DEQThinkingBlock:  # type: ignore[no-redef]
        pass

    class DEQThinkingFunction:  # type: ignore[no-redef]
        pass

    class DEQThinkingModule:  # type: ignore[no-redef]
        pass
