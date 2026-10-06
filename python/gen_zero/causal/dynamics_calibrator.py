"""Calibrated fixed-point dynamics of frozen features; no text parsing.

Inference only: U_A, V_A, U_B, V_B are static factors loaded from an artifact.
The diagonal is frozen. Radial spectral projection preserves the
diagonal-plus-low-rank representation. Dense auditing is NOT cache bounded.
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
            loaded = {}
            for name in ('lambda_',) + FACTORS:
                value = data[name]
                if value.dtype != np.float32 or value.shape != getattr(obj, name).shape:
                    raise ValueError('invalid weight shape/dtype')
                loaded[name] = _finite(value, name, dtype=np.float32).copy()
            obj.lambda_ = loaded['lambda_']
            obj.U_A, obj.V_A = loaded['U_A'], loaded['V_A']
            obj.U_B, obj.V_B = loaded['U_B'], loaded['V_B']
            obj.provenance = meta['provenance']
        # Do not silently repair corrupted/unstable serialized weights: the scale
        # is deterministic from factors, as it is on every load.
        obj._scale = obj._compute_scale()
        obj._refresh_deployment_cache()
        obj.certify()
        return obj


class CausalDynamicsCalibrator:
    """Holds a calibrated adapter built from static, pre-computed factors.

    Inference only: no fitting happens in this package. Factors come from an
    exported artifact or are supplied directly by the caller.
    """

    def __init__(self, dim, rank, input_dim, *, seed=0):
        self.adapter = CalibratedDynamics(dim, rank, input_dim, seed=seed, dtype=np.float32)
        self.adapter.certify()

    def set_static_parameters(self, U_A, V_A, U_B, V_B, *, source, split, encoder_id):
        """Install fixed factors and attest their provenance.

        Provenance is an attestation from the factor producer, not proof of
        source truth. Evaluation splits are rejected.
        """
        if split not in ('train', 'calibration'):
            raise ValueError('only train/calibration splits may be attested')
        if not source or not encoder_id:
            raise ValueError('source and encoder_id are required')
        a = self.adapter
        values = (U_A, V_A, U_B, V_B)
        checked = []
        for name, value in zip(FACTORS, values):
            value = _finite(value, name, dtype=np.float32)
            if value.shape != getattr(a, name).shape:
                raise ValueError('invalid factor shape')
            checked.append(value.copy())
        a.provenance = None
        a.U_A, a.V_A, a.U_B, a.V_B = checked
        a._scale = a._compute_scale()
        a.certify()
        digest = hashlib.sha256()
        for value in checked:
            digest.update(np.ascontiguousarray(value).tobytes())
        a.provenance = dict(source=source, split=split, encoder_id=encoder_id,
                            weights_sha256=digest.hexdigest())
        a._refresh_deployment_cache()
        return a.provenance
