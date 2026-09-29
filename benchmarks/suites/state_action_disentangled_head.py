"""Spec 23: NumPy-only bilinear scorer over independently supplied embeddings.

No text encoder, candidate generator, reward oracle or production latency guarantee.
fit() is offline ridge regression on paired outer products, not LLM training.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _array(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be real numeric data')
    x = np.array(raw, dtype=np.float64, order='C', copy=True)
    if x.ndim != ndim or not x.size or not np.isfinite(x).all():
        raise ValueError(f'{name} must be nonempty, finite and {ndim}-D')
    return x


def _unit(x):
    # Scaling first avoids overflow for large finite inputs. Zero has no cosine.
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero vectors have no cosine similarity')
    y = x / scale
    return y / np.linalg.norm(y, axis=-1, keepdims=True)


def _frozen(x):
    # Immutable bytes backing, so callers cannot re-enable WRITEABLE on a view.
    return np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)


class CachedCPUHead:
    """Standalone deployment artifact; no training or encoder dependencies."""

    def __init__(self, weights, bias, action_ids, normalize=False):
        w = _array(weights, 2, 'weights')
        b = _array(bias, 1, 'bias')
        ids = np.asarray(action_ids)
        if b.shape != (len(w),) or ids.shape != (len(w),):
            raise ValueError('bias and action_ids must match weight rows')
        if ids.dtype.kind != 'U' or len(set(ids.tolist())) != len(ids):
            raise ValueError('action_ids must be unique Unicode strings')
        if any(not s for s in ids.tolist()):
            raise ValueError('action_ids cannot be empty')
        if not isinstance(normalize, (bool, np.bool_)):
            raise ValueError('normalize must be boolean')
        self._weights, self._bias = _frozen(w), _frozen(b)
        self._action_ids = _frozen(ids)
        self._normalize = bool(normalize)

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def action_ids(self):
        return self._action_ids

    def score_cached(self, state):
        """Validated batch-one API: normalization (optional), one GEMV + bias."""
        z = _array(state, 1, 'state')
        if z.shape != (self._weights.shape[1],):
            raise ValueError('state dimension mismatch')
        if self._normalize:
            z = _unit(z)
        with np.errstate(over='raise', invalid='raise'):
            return self._weights @ z + self._bias

    def export_cpu(self, path):
        """Uncompressed .npz; float64 preserves reference numerical precision.

        Explicit file handle preserves the exact requested path. No pickle.
        """
        path = Path(path)
        with path.open('wb') as f:
            np.savez(f, version=np.array(1, dtype=np.int64),
                     weights=self._weights, bias=self._bias,
                     action_ids=self._action_ids,
                     normalize=np.array(self._normalize))
        return path

    @classmethod
    def load_cpu(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'version', 'weights', 'bias', 'action_ids', 'normalize'}:
                raise ValueError('invalid artifact fields')
            if data['version'].shape != () or data['version'].dtype.kind not in 'iu' or data['version'].item() != 1:
                raise ValueError('unsupported artifact version')
            if data['normalize'].shape != () or data['normalize'].dtype.kind != 'b':
                raise ValueError('invalid normalize metadata')
            return cls(data['weights'], data['bias'], data['action_ids'],
                       data['normalize'].item())


class StateActionDisentangledHead:
    """S(s,a) = state.T @ metric @ action + bias; ds and da may differ.

    normalize=True normalizes input vectors BEFORE applying the metric; M=I,
    b=0 then gives cosine. Public parameters are immutable; fit invalidates cache.
    """

    def __init__(self, metric, bias=0.0, *, normalize=False):
        m = _array(metric, 2, 'metric')
        b = _array(bias, 0, 'bias')
        if not isinstance(normalize, (bool, np.bool_)):
            raise ValueError('normalize must be boolean')
        self._metric, self._bias = _frozen(m), float(b)
        self._normalize = bool(normalize)
        self._cache = None

    @property
    def metric(self):
        return self._metric

    @property
    def bias(self):
        return self._bias

    def _features(self, value, dim, name):
        x = _array(value, 2, name)
        if x.shape[1] != dim:
            raise ValueError(f'{name} dimension mismatch')
        return _unit(x) if self._normalize else x

    def score(self, states, actions):
        """All-pairs uncached reference: output shape (N,K)."""
        x = self._features(states, self._metric.shape[0], 'states')
        a = self._features(actions, self._metric.shape[1], 'actions')
        with np.errstate(over='raise', invalid='raise'):
            return (x @ self._metric) @ a.T + self._bias

    def cache_actions(self, actions, action_ids=None):
        """Compile W=A M.T offline; row identities are preserved, never sorted."""
        a = self._features(actions, self._metric.shape[1], 'actions')
        if action_ids is None:
            action_ids = np.array([str(i) for i in range(len(a))])
        with np.errstate(over='raise', invalid='raise'):
            cache = CachedCPUHead(a @ self._metric.T,
                                  np.full(len(a), self._bias), action_ids,
                                  self._normalize)
        self._cache = cache
        return self

    def score_cached(self, state):
        if self._cache is None:
            raise RuntimeError('cache_actions required after construction or fit')
        return self._cache.score_cached(state)

    def export_cpu(self, path):
        if self._cache is None:
            raise RuntimeError('cache_actions required before export')
        return self._cache.export_cpu(path)

    def fit(self, states, actions, targets, *, ridge=1e-6, max_design_elements=10_000_000):
        """Fit real-valued rewards on N PAIRED examples, not an N x N label grid.

        min ||D m + b - y||² + ridge ||m||², D_i=vec(s_i a_i.T).
        Bias is unpenalized. SVD least squares handles rank deficiency with no
        hidden fallback. Explicit allocation limit: large encoders need offline
        dimensionality reduction or a different, explicitly selected optimizer.
        """
        x = self._features(states, self._metric.shape[0], 'states')
        a = self._features(actions, self._metric.shape[1], 'actions')
        y = _array(targets, 1, 'targets')
        if len(x) != len(a) or len(x) != len(y) or len(x) < 2:
            raise ValueError('need at least two aligned state/action/target rows')
        if not np.isscalar(ridge) or not np.isfinite(ridge) or ridge < 0:
            raise ValueError('ridge must be finite and nonnegative')
        if isinstance(max_design_elements, bool) or not isinstance(max_design_elements, (int, np.integer)) or max_design_elements < 1:
            raise ValueError('max_design_elements must be a positive integer')
        n, p = len(x), self._metric.size
        # Account for the augmented ridge design too; refuse before allocation.
        if n * p + ((n + p) * p if ridge else 0) > max_design_elements:
            raise ValueError('offline design exceeds max_design_elements')
        with np.errstate(over='raise', invalid='raise'):
            design = np.einsum('ni,nj->nij', x, a).reshape(n, p)
            mean_d, mean_y = design.mean(axis=0), y.mean()
            centered, target = design - mean_d, y - mean_y
            if ridge:
                centered = np.vstack((centered, np.sqrt(ridge) * np.eye(p)))
                target = np.concatenate((target, np.zeros(p)))
            coefficient = np.linalg.lstsq(centered, target, rcond=None)[0]
            metric = _array(coefficient.reshape(self._metric.shape), 2, 'fitted metric')
            bias = float(_array(mean_y - mean_d @ coefficient, 0, 'fitted bias'))
        # Commit only a successful fit. A failed fit leaves the old model intact.
        self._metric, self._bias = _frozen(metric), bias
        self._cache = None
        return self
