"""Vocabulary independent geometry bridge for paired, calibrated hidden states.

A shared coordinate system requires separately calibrated bases for each model.
This module does not infer semantic correspondence from vocabulary sizes.
"""
from __future__ import annotations

import numpy as np


class CrossVocabManifoldAdapter:
    def __init__(self, teacher_basis, teacher_mean, student_basis, student_mean,
                 *, teacher_vocab_size=248077, student_vocab_size=151643,
                 orthogonality_tolerance=1e-5):
        self.teacher_basis = self._basis(teacher_basis, teacher_mean, orthogonality_tolerance)
        self.student_basis = self._basis(student_basis, student_mean, orthogonality_tolerance)
        if self.teacher_basis.shape[1] != self.student_basis.shape[1]:
            raise ValueError('manifold dimensions must match')
        self.teacher_mean = self._mean(teacher_mean, self.teacher_basis.shape[0])
        self.student_mean = self._mean(student_mean, self.student_basis.shape[0])
        if teacher_vocab_size <= 0 or student_vocab_size <= 0:
            raise ValueError('vocabulary sizes must be positive')
        self.teacher_vocab_size = int(teacher_vocab_size)
        self.student_vocab_size = int(student_vocab_size)

    @staticmethod
    def _mean(value, dim):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (dim,) or not np.isfinite(result).all():
            raise ValueError('mean must be a finite vector matching basis rows')
        return result.copy()

    @classmethod
    def _basis(cls, value, mean, tolerance):
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 2 or not 0 < result.shape[1] <= result.shape[0] or not np.isfinite(result).all():
            raise ValueError('basis must be a finite tall matrix')
        cls._mean(mean, result.shape[0])
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('orthogonality_tolerance must be positive')
        if np.linalg.norm(result.T @ result - np.eye(result.shape[1]), ord=2) > tolerance:
            raise ValueError('basis columns must be orthonormal')
        return result.copy()

    def _pair(self, model):
        if model == 'teacher':
            return self.teacher_basis, self.teacher_mean
        if model == 'student':
            return self.student_basis, self.student_mean
        raise ValueError("model must be 'teacher' or 'student'")

    def project(self, hidden, *, model='teacher'):
        basis, mean = self._pair(model)
        h = np.asarray(hidden, dtype=np.float64)
        if h.ndim < 1 or h.shape[-1] != basis.shape[0] or not np.isfinite(h).all():
            raise ValueError('hidden states must be finite with the model hidden dimension')
        return (h - mean) @ basis

    def reconstruct(self, latent, *, model='teacher'):
        basis, mean = self._pair(model)
        z = np.asarray(latent, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != basis.shape[1] or not np.isfinite(z).all():
            raise ValueError('latent states must be finite with the manifold dimension')
        return z @ basis.T + mean

    def map_hidden(self, hidden, *, source='teacher', destination='student'):
        if source == destination:
            raise ValueError('source and destination must differ')
        self._pair(destination)
        return self.reconstruct(self.project(hidden, model=source), model=destination)

    def projection_energy_ratio(self, hidden, *, model='teacher'):
        basis, mean = self._pair(model)
        centered = np.asarray(hidden, dtype=np.float64) - mean
        if centered.shape[-1] != basis.shape[0] or not np.isfinite(centered).all():
            raise ValueError('invalid hidden states')
        denominator = np.sum(centered * centered, axis=-1)
        if np.any(denominator <= 0):
            raise ValueError('zero centered energy')
        ratio = np.sum((centered @ basis) ** 2, axis=-1) / denominator
        return np.clip(ratio, 0.0, 1.0)

    def orthogonality_error(self, *, model='teacher'):
        basis, _ = self._pair(model)
        return float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord=2))

    @staticmethod
    def geodesic_cosine(a, b):
        x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if x.shape != y.shape or x.ndim < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('states must be finite with matching shapes')
        norms = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
        if np.any(norms <= 0):
            raise ValueError('geodesic cosine undefined for zero vectors')
        return np.clip(np.sum(x * y, axis=-1) / norms, -1.0, 1.0)

    @staticmethod
    def bures_wasserstein_distance(a, b):
        x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if x.ndim != 2 or x.shape != y.shape or x.shape[0] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('paired batches must be finite, 2D, and contain at least two states')
        mx, my = x.mean(axis=0), y.mean(axis=0)
        cx, cy = np.cov(x, rowvar=False, bias=True), np.cov(y, rowvar=False, bias=True)
        def sqrt_psd(c):
            values, vectors = np.linalg.eigh(np.atleast_2d(c))
            return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
        root = sqrt_psd(cx)
        cross = sqrt_psd(root @ np.atleast_2d(cy) @ root)
        squared = np.sum((mx - my) ** 2) + np.trace(cx) + np.trace(cy) - 2 * np.trace(cross)
        return float(np.sqrt(max(0.0, squared)))

    def alignment_objective(self, teacher_hidden, student_hidden):
        teacher = self.project(teacher_hidden, model='teacher')
        student = self.project(student_hidden, model='student')
        if teacher.shape != student.shape or teacher.ndim != 2:
            raise ValueError('paired hidden batches must have matching sample counts')
        cosine = float(np.mean(self.geodesic_cosine(teacher, student)))
        distance = self.bures_wasserstein_distance(teacher, student)
        return {'geodesic_cosine': cosine, 'bures_wasserstein_distance': distance,
                'bures_wasserstein_similarity': float(np.exp(-distance))}
