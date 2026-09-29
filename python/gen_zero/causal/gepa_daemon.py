"""Stateful GEPA boundary-trace processor with covariance-based basis rotation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .gepa_evolution_loop import (BoundaryTraceScanner,
    CounterfactualAdversarialSynthesizer, IncrementalManifoldCodebookPatcher)
from .universal_manifold_extractor import StreamingCovarianceAccumulator


class GepaEvolutionDaemon:
    def __init__(self, codebook, A_operator, basis, *, scanner=None, synthesizer=None,
                 patcher=None, reorthogonalize_every=100, orthogonality_tolerance=1e-8):
        self.codebook = np.asarray(codebook, dtype=np.float64).copy()
        self.A_operator = np.asarray(A_operator, dtype=np.float64).copy()
        self.basis = np.asarray(basis, dtype=np.float64).copy()
        if self.basis.ndim != 2:
            raise ValueError('basis must be 2D')
        self.dim = self.basis.shape[1]
        if self.codebook.ndim != 2 or self.codebook.shape[1] != self.dim or self.A_operator.shape != (self.dim, self.dim):
            raise ValueError('codebook, operator and basis dimensions disagree')
        if not all(np.isfinite(x).all() for x in (self.codebook, self.A_operator, self.basis)):
            raise ValueError('state must be finite')
        if np.linalg.norm(self.basis.T @ self.basis - np.eye(self.dim), ord=2) > 1e-5:
            raise ValueError('initial basis must be orthonormal')
        if not isinstance(reorthogonalize_every, int) or reorthogonalize_every < 1 or orthogonality_tolerance <= 0:
            raise ValueError('invalid recalibration threshold')
        self.reorthogonalize_every = reorthogonalize_every
        self.orthogonality_tolerance = orthogonality_tolerance
        self.scanner = scanner or BoundaryTraceScanner()
        self.synthesizer = synthesizer or CounterfactualAdversarialSynthesizer()
        if self.synthesizer.epsilon > 0.05:
            raise ValueError('counterfactual epsilon exceeds 0.05')
        self.patcher = patcher or IncrementalManifoldCodebookPatcher()
        self.covariance = StreamingCovarianceAccumulator(self.dim)
        self.n_scanned = self.n_boundary = self.n_patched = self.n_recalibrated = 0
        self._since_recalibration = 0
        self._certify()

    def _certify(self):
        rho = float(np.max(np.abs(np.linalg.eigvals(self.A_operator))))
        if not np.isfinite(rho) or rho >= 1.0:
            raise ValueError('operator is not contractive')
        return rho

    def process_traces(self, traces):
        traces = list(traces)
        boundary = self.scanner.scan_traces(traces)
        for trace in boundary:
            state = np.asarray(trace.input_state, dtype=np.float64)
            candidates = np.asarray(trace.candidate_states, dtype=np.float64)
            idx = trace.ground_truth_idx if trace.ground_truth_idx is not None else trace.predicted_idx
            if state.shape != (self.dim,) or candidates.ndim != 2 or candidates.shape[1] != self.dim or not 0 <= idx < min(len(candidates), len(self.codebook)) or not np.isfinite(state).all() or not np.isfinite(candidates).all():
                raise ValueError('invalid boundary trace geometry or index')
            _, delta = self.synthesizer.synthesize_perturbation(state, candidates[idx])
            if not np.isfinite(delta).all() or np.linalg.norm(delta) > 0.05 + 1e-12:
                raise ValueError('counterfactual delta exceeds bound')
            gradient = np.outer(delta, state) / (np.dot(state, state) + 1e-8)
            new_codebook = self.patcher.patch_codebook_attractors(self.codebook, idx, delta)
            new_operator, _ = self.patcher.retune_lyapunov_operator(self.A_operator, gradient)
            self.codebook, self.A_operator = new_codebook, new_operator
            self.covariance.update(np.stack((state, state + delta)))
            self.n_patched += 1
            self._since_recalibration += 1
            if (self._since_recalibration >= self.reorthogonalize_every or
                self.orthogonality_error() > self.orthogonality_tolerance):
                self.recalibrate()
        self.n_scanned += len(traces)
        self.n_boundary += len(boundary)
        return len(boundary)

    def orthogonality_error(self):
        return float(np.linalg.norm(self.basis.T @ self.basis - np.eye(self.dim), ord=2))

    def recalibrate(self):
        if self.covariance.n_samples < 2:
            raise ValueError('insufficient states to recalibrate')
        rotation, _ = self.covariance.compute_principal_basis(self.dim)
        candidate = self.basis @ rotation
        # Polar factor corrects numerical drift without changing the span.
        left, _, right = np.linalg.svd(candidate, full_matrices=False)
        corrected = left @ right
        transform = self.basis.T @ corrected
        # Reproject all latent state and dynamics coordinates into the new basis.
        self.codebook = self.codebook @ transform
        self.A_operator = transform.T @ self.A_operator @ np.linalg.inv(transform.T)
        self.basis = corrected
        self.covariance = StreamingCovarianceAccumulator(self.dim)
        self.n_recalibrated += 1
        self._since_recalibration = 0
        rho = self._certify()
        if self.orthogonality_error() > self.orthogonality_tolerance:
            raise ArithmeticError('basis remains nonorthogonal after recalibration')
        return rho

    def run(self, trace_batches, *, max_batches=None):
        """Consume a live or finite iterable; caller controls lifetime and polling."""
        if max_batches is not None and max_batches < 0:
            raise ValueError('max_batches must be nonnegative')
        for index, batch in enumerate(trace_batches):
            if max_batches is not None and index >= max_batches:
                break
            self.process_traces(batch)

    def export_audit_log(self, path=None):
        rho = self._certify()
        digest = hashlib.sha256()
        for array in (self.basis, self.codebook, self.A_operator):
            digest.update(np.ascontiguousarray(array).tobytes())
        report = {'n_scanned': self.n_scanned, 'n_boundary': self.n_boundary,
                  'n_patched': self.n_patched, 'n_recalibrated': self.n_recalibrated,
                  'orthogonality_error': self.orthogonality_error(),
                  'lyapunov_spectral_radius': rho, 'contractive_certified': rho < 1.0,
                  'sha256': digest.hexdigest()}
        if path is not None:
            Path(path).write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        return report
