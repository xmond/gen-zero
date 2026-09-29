"""Supervised fixed-point calibration of frozen features; no text parsing.

Only U_A, V_A, U_B, V_B are optimized. The diagonal is frozen. Radial
spectral projection preserves the diagonal-plus-low-rank representation.
Offline fitting/auditing uses dense matrices and is NOT cache bounded.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .parallel_rnn_lora import ParallelRNNLoRAAdapter, _finite

FACTORS = ('U_A', 'V_A', 'U_B', 'V_B')
MAX_DEPLOYMENT_WORKING_SET_BYTES = 5632  # 5.5 KiB


class CalibratedDynamics(ParallelRNNLoRAAdapter):
    """Deployment adapter, with no labels accepted by the inference path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refresh_deployment_cache()

    def _refresh_deployment_cache(self):
        self._deployment_diag = self._dvec()
        self._rank1_step = ((self._deployment_diag, self.U_A[:, 0], self.V_A[:, 0],
                             self.U_B[:, 0], self.V_B[:, 0], self._scale)
                            if self.rank == 1 else None)

    def step(self, x_t, h_prev):
        """Validated deployment step with an exact rank-one fast path."""
        x_t = np.asarray(x_t)
        h_prev = np.asarray(h_prev)
        if x_t.dtype != self.dtype:
            x_t = x_t.astype(self.dtype)
        if h_prev.dtype != self.dtype:
            h_prev = h_prev.astype(self.dtype)
        if x_t.shape != (self.input_dim,):
            raise ValueError(f'x_t must have shape ({self.input_dim},)')
        if h_prev.shape != (self.dim,):
            raise ValueError(f'h_prev must have shape ({self.dim},)')
        if not np.isfinite(x_t).all() or not np.isfinite(h_prev).all():
            raise ValueError('step inputs must contain only finite values')
        if self.rank == 1:
            diagonal, u_a, v_a, u_b, v_b, scale = self._rank1_step
            a_term = diagonal * h_prev + u_a * np.dot(v_a, h_prev)
            return scale * a_term + u_b * np.dot(v_b, x_t)
        return super().step(x_t, h_prev)

    def forcing(self, x_t):
        """Validate an input once and return its constant Bx recurrence term."""
        x_t = np.asarray(x_t)
        if x_t.dtype != self.dtype:
            x_t = x_t.astype(self.dtype)
        if x_t.shape != (self.input_dim,) or not np.isfinite(x_t).all():
            raise ValueError(f'x_t must be a finite vector of shape ({self.input_dim},)')
        if self.rank == 1:
            _, _, _, u_b, v_b, _ = self._rank1_step
            return u_b * np.dot(v_b, x_t)
        return self.U_B @ (self.V_B.T @ x_t)

    def step_forcing(self, forcing, h_prev):
        """Step with a previously validated forcing vector (fixed-input loop)."""
        forcing = np.asarray(forcing)
        h_prev = np.asarray(h_prev)
        if forcing.shape != (self.dim,) or h_prev.shape != (self.dim,):
            raise ValueError(f'forcing and state must have shape ({self.dim},)')
        if not np.isfinite(forcing).all() or not np.isfinite(h_prev).all():
            raise ValueError('step inputs must contain only finite values')
        if self.rank == 1:
            diagonal, u_a, v_a, _, _, scale = self._rank1_step
            return scale * (diagonal * h_prev + u_a * np.dot(v_a, h_prev)) + forcing
        a_term = self._deployment_diag * h_prev + self.U_A @ (self.V_A.T @ h_prev)
        return self._scale * a_term + forcing

    def _step_cached_forcing(self, forcing, h_prev):
        """Internal path: ``forcing`` was produced by :meth:`forcing`."""
        h_prev = np.asarray(h_prev)
        if h_prev.shape != (self.dim,) or not np.isfinite(h_prev).all():
            raise ValueError(f'state must be a finite vector of shape ({self.dim},)')
        diagonal, u_a, v_a, _, _, scale = self._rank1_step
        return scale * (diagonal * h_prev + u_a * np.dot(v_a, h_prev)) + forcing

    def _compute_scale(self):
        raw = np.diag(self._dvec()) + self.U_A @ self.V_A.T
        rho = float(np.max(np.abs(np.linalg.eigvals(raw))))
        sigma = float(np.linalg.svd(raw, compute_uv=False)[0])
        # Float32 roundoff margin, followed by an independent certificate.
        scale = min(1., .55 / max(rho, 1e-30), .95 / max(sigma, 1e-30))
        return scale * (1. - 1e-5)

    def certify(self):
        a = self.transition_matrix().astype(np.float64)
        rho = float(np.max(np.abs(np.linalg.eigvals(a))))
        sigma = float(np.linalg.svd(a, compute_uv=False)[0])
        if not (rho <= .55 and sigma <= .95):
            raise ValueError('spectral certificate failed')
        if self.working_set_bytes() > MAX_DEPLOYMENT_WORKING_SET_BYTES:
            raise ValueError('numeric inference working set must be <=5.5 KiB')
        return dict(rho=rho, sigma=sigma, working_set_bytes=self.working_set_bytes())

    def working_set_bytes(self):
        """Conservative numeric payload bound, including step temporaries.

        Counts parameters, 12 state-sized, 4 input-sized, 4 rank-sized
        buffers and a scalar. Excludes Python objects, BLAS internal workspace,
        instructions and cache conflicts; this is not a hardware residency claim.
        """
        params = sum(getattr(self, n).nbytes for n in ('lambda_',) + FACTORS)
        return int(params + np.dtype(self.dtype).itemsize *
                   (12 * self.dim + 4 * self.input_dim + 4 * self.rank + 1))

    def fixed_points(self, features):
        """Dense offline audit only; online inference uses inherited step()."""
        x = _finite(features, 'features', dtype=self.dtype)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('feature shape mismatch')
        return np.linalg.solve(np.eye(self.dim) - self.transition_matrix(),
                               (x @ self.input_matrix().T).T).T

    def save(self, path):
        self.certify()
        if not getattr(self, 'provenance', None):
            raise ValueError('uncalibrated adapter cannot be exported')
        metadata = dict(version=1, dim=self.dim, rank=self.rank,
                        input_dim=self.input_dim, provenance=self.provenance)
        with open(path, 'wb') as stream:
            np.savez(stream, metadata=json.dumps(metadata), lambda_=self.lambda_,
                     **{n: getattr(self, n) for n in FACTORS})

    @classmethod
    def load(cls, path, *, encoder_id):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'metadata', 'lambda_', *FACTORS}:
                raise ValueError('invalid weight schema')
            meta = json.loads(str(data['metadata']))
            if meta['version'] != 1 or meta['provenance']['encoder_id'] != encoder_id:
                raise ValueError('version or frozen encoder mismatch')
            obj = cls(meta['dim'], meta['rank'], meta['input_dim'], dtype=np.float32, seed=0)
            for name in ('lambda_',) + FACTORS:
                value = data[name]
                if value.dtype != np.float32 or value.shape != getattr(obj, name).shape:
                    raise ValueError('invalid weight shape/dtype')
                setattr(obj, name, _finite(value, name, dtype=np.float32).copy())
            obj.provenance = meta['provenance']
        # Do not silently repair corrupted/unstable serialized weights: the scale
        # is deterministic from factors, as it is during every optimization step.
        obj._scale = obj._compute_scale()
        obj._refresh_deployment_cache()
        obj.certify()
        return obj


class CausalDynamicsCalibrator:
    def __init__(self, dim, rank, input_dim, *, seed=0):
        self.adapter = CalibratedDynamics(dim, rank, input_dim, seed=seed, dtype=np.float32)
        self.adapter.certify()

    def fit(self, features, labels, *, sample_ids, source, split, encoder_id,
            epochs=1000, lr=.01):
        """Fit ONLY explicitly declared train/calibration data.

        Provenance is an attestation from the feature producer, not proof of
        source truth. Features must be extracted without answers/labels in the
        encoder input. No evaluation set or evaluation labels are accepted.
        """
        import torch
        if split not in ('train', 'calibration'):
            raise ValueError('only train/calibration splits may be fitted')
        if not source or not encoder_id:
            raise ValueError('source and encoder_id are required')
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError('epochs must be positive integer')
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be positive finite')
        a = self.adapter
        x = _finite(features, 'features', dtype=np.float32)
        y = np.asarray(labels)
        ids = np.asarray(sample_ids, dtype=str)
        if x.ndim != 2 or x.shape[1] != a.input_dim or y.shape != (len(x),):
            raise ValueError('training shapes mismatch')
        if y.dtype.kind not in 'iu' or ids.shape != y.shape or len(set(ids)) != len(ids):
            raise ValueError('integer labels and unique sample ids required')
        classes, y = np.unique(y, return_inverse=True)
        k = len(classes)
        if k < 2 or k > a.dim:
            raise ValueError('need 2..dim classes')
        # Fixed centered simplex; only training labels select regression targets.
        codebook = np.zeros((k, a.dim), dtype=np.float32)
        codebook[:, :k] = (np.eye(k) - np.ones((k, k)) / k) / np.sqrt(1 - 1/k)
        target = torch.tensor(codebook[y])
        xt = torch.tensor(x)
        diag = torch.tensor(a._dvec())
        params = {n: torch.nn.Parameter(torch.tensor(getattr(a, n))) for n in FACTORS}
        # A failed refit must not leave an old successful export attestation.
        a.provenance = None
        optimizer = torch.optim.Adam(list(params.values()), lr=lr)
        history = []
        for _ in range(epochs):
            optimizer.zero_grad()
            raw = torch.diag(diag) + params['U_A'] @ params['V_A'].T
            # Piecewise differentiable radial projection; no detached surrogate.
            rho = torch.linalg.eigvals(raw).abs().max()
            sigma = torch.linalg.matrix_norm(raw, ord=2)
            scale = torch.minimum(torch.ones(()), torch.minimum(.55/rho.clamp_min(1e-30),
                                  .95/sigma.clamp_min(1e-30))) * (1 - 1e-5)
            force = xt @ params['V_B'] @ params['U_B'].T
            fixed = torch.linalg.solve(torch.eye(a.dim) - scale * raw, force.T).T
            loss = (fixed - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite calibration loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
                raise FloatingPointError('nonfinite calibration gradient')
            optimizer.step()
            for n, p in params.items():
                setattr(a, n, _finite(p.detach().numpy(), n, dtype=np.float32).copy())
            a._scale = a._compute_scale()
            a.certify()
            history.append(float(loss.detach()))
        digest = hashlib.sha256()
        for data in (x, classes[y], ids):
            digest.update(np.ascontiguousarray(data).tobytes())
        a.provenance = dict(source=source, split=split, encoder_id=encoder_id,
                            training_sha256=digest.hexdigest(), samples=len(x),
                            classes=classes.tolist(), epochs=epochs, lr=lr)
        a._refresh_deployment_cache()
        return history

    def fit_candidate_geometry(self, features, candidates, positive_indices, *,
                               sample_ids, source, split, encoder_id, epochs=1000,
                               lr=.01, margin=.25, residual_weight=1.):
        """Fit attraction to a positive branch and orthogonal rejection of alternatives.

        ``features`` are frozen compact encoder states ``(N, input_dim)``.
        ``candidates`` are frozen branch states ``(N, K, dim)`` and
        ``positive_indices`` identifies the entailed/correct branch.  The encoder
        inputs themselves must remain label-free; supervision is used only here.
        No text, task names, answer strings, or evaluation formats are inspected.
        """
        import torch
        if split not in ('train', 'calibration'):
            raise ValueError('only train/calibration splits may be fitted')
        if not source or not encoder_id:
            raise ValueError('source and encoder_id are required')
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError('epochs must be positive integer')
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be positive finite')
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('margin must be positive finite')
        if not np.isfinite(residual_weight) or residual_weight <= 0:
            raise ValueError('residual_weight must be positive finite')

        a = self.adapter
        x = _finite(features, 'features', dtype=np.float32)
        c = _finite(candidates, 'candidates', dtype=np.float32)
        pos = np.asarray(positive_indices)
        ids = np.asarray(sample_ids, dtype=str)
        n = len(x)
        if x.ndim != 2 or x.shape[1] != a.input_dim:
            raise ValueError('feature shape mismatch')
        if c.ndim != 3 or c.shape[0] != n or c.shape[1] < 2 or c.shape[2] != a.dim:
            raise ValueError('candidates must have shape (N, K>=2, dim)')
        if np.any(np.linalg.norm(c, axis=2) <= 1e-12):
            raise ValueError('candidate vectors must have nonzero norm')
        if pos.dtype.kind not in 'iu' or pos.shape != (n,) or np.any(pos < 0) or np.any(pos >= c.shape[1]):
            raise ValueError('positive_indices must be valid integer candidate indices')
        if ids.shape != (n,) or len(set(ids)) != n:
            raise ValueError('unique sample ids required')

        xt = torch.tensor(x)
        ct = torch.tensor(c)
        pt = torch.tensor(pos.astype(np.int64))
        diag = torch.tensor(a._dvec())
        params = {name: torch.nn.Parameter(torch.tensor(getattr(a, name))) for name in FACTORS}
        a.provenance = None
        optimizer = torch.optim.Adam(list(params.values()), lr=lr)
        history = []
        row = torch.arange(n)
        for _ in range(epochs):
            optimizer.zero_grad()
            raw = torch.diag(diag) + params['U_A'] @ params['V_A'].T
            rho = torch.linalg.eigvals(raw).abs().max()
            sigma = torch.linalg.matrix_norm(raw, ord=2)
            scale = torch.minimum(torch.ones(()), torch.minimum(
                .55 / rho.clamp_min(1e-30), .95 / sigma.clamp_min(1e-30))) * (1 - 1e-5)
            force = xt @ params['V_B'] @ params['U_B'].T
            fixed = torch.linalg.solve(torch.eye(a.dim) - scale * raw, force.T).T

            fixed_unit = fixed / fixed.norm(dim=1, keepdim=True).clamp_min(1e-12)
            candidate_unit = ct / ct.norm(dim=2, keepdim=True).clamp_min(1e-12)
            positive = candidate_unit[row, pt]
            # Unit-sphere attraction is scale invariant: B cannot win by merely
            # inflating its output norm.
            attraction = 1. - (fixed_unit * positive).sum(dim=1)
            direction = positive
            residual = fixed_unit[:, None, :] - candidate_unit
            # Rejection evidence must live outside the positive branch direction;
            # mere movement farther along that direction cannot satisfy the margin.
            orthogonal = residual - (residual * direction[:, None, :]).sum(
                dim=2, keepdim=True) * direction[:, None, :]
            neg_residual = orthogonal.square().sum(dim=2)
            negative_mask = torch.ones_like(neg_residual, dtype=torch.bool)
            negative_mask[row, pt] = False
            rejection = torch.relu(margin - neg_residual)
            loss = attraction.mean() + residual_weight * rejection[negative_mask].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite calibration loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
                raise FloatingPointError('nonfinite calibration gradient')
            optimizer.step()
            for name, parameter in params.items():
                setattr(a, name, _finite(parameter.detach().numpy(), name,
                                         dtype=np.float32).copy())
            a._scale = a._compute_scale()
            a.certify()
            history.append(float(loss.detach()))

        digest = hashlib.sha256()
        for data in (x, c, pos, ids):
            digest.update(np.ascontiguousarray(data).tobytes())
        a.provenance = dict(source=source, split=split, encoder_id=encoder_id,
                            objective='candidate_geometry_v1',
                            training_sha256=digest.hexdigest(), samples=n,
                            candidates=int(c.shape[1]), epochs=epochs, lr=lr,
                            margin=margin, residual_weight=residual_weight)
        a._refresh_deployment_cache()
        return history
