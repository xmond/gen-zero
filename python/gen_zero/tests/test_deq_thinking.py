"""Tests for the Gen-Zero DEQ fixed-point thinking module.

Covers:
1. NumPy Anderson accelerator on linear fixed points.
2. Batched torch Anderson solver, including affine maps.
3. IFT backward: gradients for x AND block parameters match an unrolled
   reference and finite differences.
4. O(1) memory: the forward solve builds no autograd graph.
5. Hygiene: source and tests hold no private filesystem paths.
"""

import re
import unittest
from pathlib import Path

import numpy as np

from gen_zero.causal.deq_thinking import (
    AndersonAccelerator,
    DEQSolverState,
    _np_solve_fixed_point,
)

try:
    import torch
    from gen_zero.causal.deq_thinking import (
        DEQThinkingBlock,
        DEQThinkingFunction,
        DEQThinkingModule,
        _anderson_solve,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestAndersonAccelerator(unittest.TestCase):
    """Test the Anderson accelerator on simple linear systems."""

    def test_simple_linear_fixed_point(self):
        """Solve z = A*z + b with spectral radius < 1."""
        accel = AndersonAccelerator(m=5, beta=1.0)

        # A has spectral radius 0.5 -> contractive
        A = np.array([[0.3, 0.1], [0.1, 0.3]], dtype=np.float64)
        b = np.array([1.0, 2.0], dtype=np.float64)

        def f(z):
            return A @ z + b

        z0 = np.zeros(2, dtype=np.float64)
        z_star, state = _np_solve_fixed_point(f, z0, accel, max_iter=100, tol=1e-10)

        # Analytical solution: z = (I - A)^{-1} b
        expected = np.linalg.solve(np.eye(2) - A, b)
        np.testing.assert_allclose(z_star, expected, rtol=1e-8)
        self.assertTrue(state.converged)
        # Anderson should converge faster than plain Picard
        self.assertLess(state.n_iter, 80)

    def test_non_convergent_reports_false(self):
        """When the mapping is not contractive, the solver reports non-converged."""
        accel = AndersonAccelerator(m=5, beta=1.0)

        # A has spectral radius ~1.5 -> divergent
        A = np.array([[1.0, 0.8], [0.8, 1.0]], dtype=np.float64)
        b = np.array([0.0, 0.0], dtype=np.float64)

        def f(z):
            return A @ z + b

        z0 = np.array([1.0, 1.0], dtype=np.float64)
        _z_star, state = _np_solve_fixed_point(f, z0, accel, max_iter=20, tol=1e-12)

        self.assertFalse(state.converged)
        self.assertEqual(state.n_iter, 20)

    def test_anderson_better_than_picard(self):
        """Anderson (m=5) needs far fewer iterations than plain Picard (m=1)."""
        A = np.array([[0.9, 0.05], [0.05, 0.9]], dtype=np.float64)  # rho = 0.95
        b = np.array([3.0, -1.0], dtype=np.float64)

        def f(z):
            return A @ z + b

        z0 = np.zeros(2, dtype=np.float64)
        _, s_and = _np_solve_fixed_point(f, z0.copy(), AndersonAccelerator(m=5), 500, 1e-8)
        _, s_pic = _np_solve_fixed_point(f, z0.copy(), AndersonAccelerator(m=1), 500, 1e-8)

        self.assertTrue(s_and.converged)
        self.assertTrue(s_pic.converged)
        self.assertLess(s_and.n_iter * 5, s_pic.n_iter)

    def test_invalid_history(self):
        with self.assertRaises(ValueError):
            AndersonAccelerator(m=0)

    def test_damping_beta(self):
        """Beta < 1 damps the Anderson mixing, still converges."""
        accel = AndersonAccelerator(m=5, beta=0.5)

        A = np.array([[0.4, 0.0], [0.0, 0.4]], dtype=np.float64)
        b = np.array([5.0, -3.0], dtype=np.float64)

        def f(z):
            return A @ z + b

        z0 = np.zeros(2, dtype=np.float64)
        z_star, state = _np_solve_fixed_point(f, z0, accel, max_iter=100, tol=1e-10)
        expected = np.linalg.solve(np.eye(2) - A, b)
        np.testing.assert_allclose(z_star, expected, rtol=1e-8)
        self.assertTrue(state.converged)

    def test_history_maintenance(self):
        """AndersonAccelerator correctly maintains at most m history entries."""
        accel = AndersonAccelerator(m=3)
        z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        fz = np.array([1.1, 2.1, 3.1], dtype=np.float64)

        for _ in range(10):
            z = accel.step(z, fz)
            fz = z * 0.9  # dummy contraction

        self.assertEqual(len(accel._z_history), 3)
        self.assertEqual(len(accel._f_history), 3)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestTorchAndersonSolve(unittest.TestCase):
    def test_affine_solve_matches_direct(self):
        torch.manual_seed(0)
        d = 12
        A = 0.4 * torch.randn(d, d, dtype=torch.float64) / d ** 0.5
        b = torch.randn(3, d, dtype=torch.float64)
        z, state = _anderson_solve(lambda z: z @ A.T + b, torch.zeros(3, d, dtype=torch.float64),
                                   m=5, max_iter=100, tol=1e-12)
        expected = torch.linalg.solve(torch.eye(d, dtype=torch.float64) - A, b.T).T
        self.assertTrue(state.converged)
        torch.testing.assert_close(z, expected, rtol=1e-9, atol=1e-9)

    def test_divergent_map_reports_not_converged(self):
        A = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
        # Fixed point exists at 0, but the start is away from it and A is expansive;
        # a constant offset makes the true fixed point unreachable by Picard.
        b = torch.ones(1, 2, dtype=torch.float64)
        _, state = _anderson_solve(lambda z: z @ A.T * 3.0 + b, torch.ones(1, 2, dtype=torch.float64),
                                   m=1, max_iter=10, tol=1e-12)
        self.assertFalse(state.converged)
        self.assertEqual(state.n_iter, 10)

    def test_samples_are_independent(self):
        torch.manual_seed(1)
        d = 6
        A = 0.3 * torch.randn(d, d, dtype=torch.float64) / d ** 0.5
        b = torch.randn(4, d, dtype=torch.float64)
        f = lambda z: z @ A.T + b
        z_all, _ = _anderson_solve(f, torch.zeros(4, d, dtype=torch.float64), 5, 100, 1e-12)
        for i in range(4):
            bi = b[i:i + 1]
            z_i, _ = _anderson_solve(lambda z: z @ A.T + bi, torch.zeros(1, d, dtype=torch.float64),
                                     5, 100, 1e-12)
            torch.testing.assert_close(z_all[i:i + 1], z_i, rtol=1e-8, atol=1e-8)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestDEQThinkingBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block = DEQThinkingBlock(dim=16, context_dim=8)

    def test_forward_shape(self):
        out = self.block(torch.randn(4, 16), torch.randn(4, 8))
        self.assertEqual(out.shape, (4, 16))

    def test_init_z0_is_zero(self):
        z0 = self.block.init_z0(3, torch.device("cpu"), torch.float32)
        self.assertEqual(z0.shape, (3, 16))
        self.assertTrue(torch.all(z0 == 0))

    def test_contractive_at_init(self):
        """Jacobian df/dz at the fixed point has spectral radius < 1."""
        module = DEQThinkingModule(16, 8).double()
        x = torch.randn(1, 8, dtype=torch.float64)
        z = module(x).detach()
        jac = torch.autograd.functional.jacobian(lambda z_: module.block(z_, x), z)
        rho = torch.linalg.eigvals(jac.reshape(16, 16)).abs().max().item()
        self.assertLess(rho, 1.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestDEQThinkingModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = DEQThinkingModule(16, 8, tol=1e-10).double()
        self.x = torch.randn(4, 8, dtype=torch.float64)

    def test_forward_shape_and_convergence(self):
        z = self.module(self.x)
        self.assertEqual(z.shape, (4, 16))
        self.assertTrue(self.module.last_state.converged)

    def test_output_is_fixed_point(self):
        with torch.no_grad():
            z = self.module(self.x)
            fz = self.module.block(z, self.x)
        rel = (fz - z).norm() / z.norm()
        self.assertLess(rel.item(), 1e-8)

    def test_z0_does_not_change_solution(self):
        with torch.no_grad():
            a = self.module(self.x)
            b = self.module(self.x, z0=torch.randn(4, 16, dtype=torch.float64))
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_batch_independence(self):
        with torch.no_grad():
            full = self.module(self.x)
            single = self.module(self.x[1:2])
        torch.testing.assert_close(full[1:2], single, rtol=1e-7, atol=1e-7)

    def _unrolled_grads(self, w, steps=400):
        x = self.x.clone().requires_grad_(True)
        z = torch.zeros(4, 16, dtype=torch.float64)
        for _ in range(steps):
            z = self.module.block(z, x)
        self.module.zero_grad()
        (z * w).sum().backward()
        return x.grad.clone(), {n: p.grad.clone() for n, p in self.module.named_parameters()}

    def test_ift_matches_unrolled_for_x_and_all_params(self):
        w = torch.randn(4, 16, dtype=torch.float64)
        x = self.x.clone().requires_grad_(True)
        self.module.zero_grad()
        (self.module(x) * w).sum().backward()
        self.assertTrue(self.module.last_backward_state.converged)
        gx = x.grad.clone()
        gp = {n: p.grad for n, p in self.module.named_parameters()}

        ref_x, ref_p = self._unrolled_grads(w)
        torch.testing.assert_close(gx, ref_x, rtol=1e-6, atol=1e-8)
        for name, ref in ref_p.items():
            self.assertIsNotNone(gp[name], f"{name} got no gradient")
            self.assertGreater(ref.abs().max().item(), 0.0, f"reference for {name} is zero")
            torch.testing.assert_close(gp[name], ref, rtol=1e-6, atol=1e-8, msg=name)

    def test_finite_difference_on_input(self):
        w = torch.randn(4, 16, dtype=torch.float64)
        x = self.x.clone().requires_grad_(True)
        (self.module(x) * w).sum().backward()
        eps = 1e-6
        for idx in [(0, 0), (2, 5), (3, 7)]:
            xp, xm = self.x.clone(), self.x.clone()
            xp[idx] += eps
            xm[idx] -= eps
            with torch.no_grad():
                fd = ((self.module(xp) * w).sum() - (self.module(xm) * w).sum()) / (2 * eps)
            self.assertAlmostEqual(x.grad[idx].item(), fd.item(), places=5)

    def test_finite_difference_on_parameter(self):
        w = torch.randn(4, 16, dtype=torch.float64)
        self.module.zero_grad()
        (self.module(self.x) * w).sum().backward()
        p = self.module.block.context_proj.weight
        analytic = p.grad[3, 2].item()
        eps = 1e-6
        with torch.no_grad():
            p[3, 2] += eps
            up = (self.module(self.x) * w).sum().item()
            p[3, 2] -= 2 * eps
            down = (self.module(self.x) * w).sum().item()
            p[3, 2] += eps
        self.assertAlmostEqual(analytic, (up - down) / (2 * eps), places=5)

    def test_no_input_grad_needed(self):
        """x without requires_grad still trains the parameters."""
        self.module.zero_grad()
        self.module(self.x).sum().backward()
        for name, p in self.module.named_parameters():
            self.assertIsNotNone(p.grad, name)

    def test_frozen_parameter_gets_no_grad(self):
        self.module.block.norm1.weight.requires_grad_(False)
        self.module.zero_grad()
        self.module(self.x).sum().backward()
        self.assertIsNone(self.module.block.norm1.weight.grad)
        self.assertIsNotNone(self.module.block.norm2.weight.grad)

    def test_optimizer_step_reduces_loss(self):
        module = DEQThinkingModule(16, 8)
        x = torch.randn(8, 8)
        target = torch.randn(8, 16)
        opt = torch.optim.Adam(module.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = ((module(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_forward_builds_no_autograd_graph(self):
        """O(1) memory: every f call inside the solve runs with grad disabled."""
        modes = []
        handle = self.module.block.register_forward_hook(
            lambda *_: modes.append(torch.is_grad_enabled()))
        try:
            x = self.x.clone().requires_grad_(True)
            z = self.module(x)
        finally:
            handle.remove()
        self.assertGreater(len(modes), 3)
        self.assertFalse(any(modes))
        self.assertEqual(type(z.grad_fn).__name__, "DEQThinkingFunctionBackward")

    def test_graph_size_independent_of_iterations(self):
        """Autograd graph size is the same for 3 and 60 solver iterations."""
        def graph_nodes(root):
            seen, stack = set(), [root]
            while stack:
                node = stack.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                stack.extend(fn for fn, _ in node.next_functions)
            return len(seen)

        sizes = []
        for max_iter in (3, 60):
            m = DEQThinkingModule(16, 8, max_iter=max_iter, tol=0.0).double()
            z = m(self.x.clone().requires_grad_(True))
            sizes.append(graph_nodes(z.grad_fn))
        self.assertEqual(sizes[0], sizes[1])

    def test_early_exit(self):
        module = DEQThinkingModule(16, 8, max_iter=200, tol=1e-3)
        module(torch.randn(2, 8))
        self.assertTrue(module.last_state.converged)
        self.assertLess(module.last_state.n_iter, 200)

    def test_unconverged_forward_reported(self):
        module = DEQThinkingModule(16, 8, max_iter=1, tol=1e-12)
        module(torch.randn(2, 8))
        self.assertFalse(module.last_state.converged)

    def test_solver_state_to_dict(self):
        d = DEQSolverState(True, 3, 1.23456789012, [0.5, 0.1]).to_dict()
        self.assertEqual(d["n_iter"], 3)
        self.assertEqual(d["final_residual"], 1.23456789)


class TestNoPrivatePathLeaks(unittest.TestCase):
    """Zero private filesystem paths in the module and its tests."""

    PATTERNS = [r"/home/\w+", r"/Users/\w+", r"/ebs/", r"C:\\Users", r"\.claude", r"~/inbox"]

    def test_no_private_paths(self):
        here = Path(__file__).resolve()
        files = [here, here.parents[1] / "causal" / "deq_thinking.py"]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                if "PATTERNS" in line:
                    continue
                for pat in self.PATTERNS:
                    self.assertIsNone(re.search(pat, line),
                                      f"{path.name}:{line_no} matches {pat}")


if __name__ == "__main__":
    unittest.main()
