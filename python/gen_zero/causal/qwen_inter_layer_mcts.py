"""Inter-layer hooks, episodic manifold memory and MCTS-guided compute pruning for Qwen2.

Three pieces, all pure inference (nothing here computes a gradient or updates a weight):

* ``QwenInterLayerHook`` registers a forward hook on every decoder layer of a Qwen2 model
  (``model.model.layers`` or ``model.layers``; a layer may return a Tensor or a tuple). It
  counts real layer calls, records the relative kinetic step of the last token
  ``delta_l = ||h_l - h_{l-1}|| / (||h_l|| + eps)``, can add a low-rank steering step
  (``action == 0`` or ``eta == 0`` returns None, so the output is bit-identical), and can
  stop the forward pass by raising ``_LayerEarlyExit``. The exception leaves the layer loop,
  so the remaining layers are never called.
* ``EpisodicManifoldMemory`` is a non-parametric store of ``(L2Norm(h_l P), label
  distribution)`` pairs read by a cosine-kernel softmax (Modern Hopfield / Nadaraya-Watson
  retrieval). ``loo_query`` masks each entry's own id, and ``select_memory_lambda`` keeps a
  memory mixing weight only when its paired out-of-fold gain clears 2 standard errors.
* ``MctsLayerwiseQwenRunner`` reads a verbalizer probe (final norm + the label rows of
  ``lm_head``) at checkpoint layers and fuses it with the memory into a label distribution
  ``p_l``. The exit certificate lives in that probability simplex, so both sides of the
  test have one unit: with drift ``d_l = ||p_l - p_prev||_inf`` and the empirical
  contraction ``g_l = min(0.95, max(gamma, d_l / d_prev))`` the remaining-depth bound is
  ``B_l = 2 * tau * g_l / (1 - g_l) * d_l``. The pass exits only at a layer
  ``>= min_exit_layer`` after at least two checkpoints, when the top-1 label is the same
  at both, ``LatentMctsPlanner`` picks it with a visit-share margin above
  ``mcts_margin_min``, and the top-1 minus top-2 probability exceeds ``B_l``. The bound
  is a geometric-tail heuristic on observed drift, not a proof about unseen layers.
  ``fit_memories`` also stores per-layer anchor statistics ``(mean, std)`` of the probe
  logits; the readout then uses ``(z - mean) / std``. An intermediate layer read through
  the final norm and ``lm_head`` carries a large constant label offset (it is not where
  the head was trained to read), and this removes it with anchor data only.
  ``score_candidates_mcts_pruned`` runs every candidate through the shallow layers only, keeps
  the top-m by MCTS visits, and resumes just those from the cached shallow state, so the
  pruned candidates never run their deep layers.

PyTorch is imported lazily: the memory works on NumPy alone, and a missing torch raises on
first use of a model path, never silently.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .latent_mcts import LatentMctsPlanner, _softmax

__all__ = [
    "QwenInterLayerHook",
    "EpisodicManifoldMemory",
    "MctsLayerwiseQwenRunner",
    "VerbalizerProbe",
    "LayerExitDecision",
    "PrunedCandidateResult",
    "select_memory_lambda",
    "simplex_etf",
    "decoder_layers",
    "QwenSparseFfnHook",
    "score_ffn_channels",
    "select_top_ffn_channels",
]

_EPS = 1e-6
_GAMMA_MAX = 0.95


def _torch():
    try:
        import torch
    except ImportError as exc:  # fail closed: hooks need a real torch model
        raise ImportError("qwen_inter_layer_mcts needs PyTorch to hook a Qwen2 model") from exc
    return torch


def _backbone(model):
    """The decoder stack: ``model.model`` for a causal-LM wrapper, else the model itself."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if hasattr(model, "layers"):
        return model
    raise ValueError("model has neither model.model.layers nor model.layers")


def decoder_layers(model):
    return _backbone(model).layers


_SWIGLU_ERROR = "decoder layer has no SwiGLU mlp (gate_proj, up_proj, down_proj)"


def _swiglu(layer):
    mlp = getattr(layer, "mlp", None)
    if mlp is None or any(not hasattr(mlp, name) for name in
                          ("gate_proj", "up_proj", "down_proj")):
        raise ValueError(_SWIGLU_ERROR)
    return mlp


def score_ffn_channels(model, anchors: Sequence[Any], layers: Sequence[int]) -> dict[int, np.ndarray]:
    torch = _torch()
    selected = sorted(set(layers))
    decoders = list(decoder_layers(model))
    if not anchors or not selected or any(not isinstance(i, int) or not 0 <= i < len(decoders)
                                          for i in selected):
        raise ValueError("anchors and valid decoder layers must be nonempty")
    mlps = {i: _swiglu(decoders[i]) for i in selected}
    sums = {i: torch.zeros(mlps[i].gate_proj.out_features, dtype=torch.float64) for i in selected}
    counts = {i: 0 for i in selected}
    handles = []
    def make_hook(i):
        def hook(_module, args):
            x = args[0]
            mlp = mlps[i]
            activated = torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            sums[i].add_(activated.detach().abs().reshape(-1, activated.shape[-1]).double().sum(dim=0).cpu())
            counts[i] += activated.numel() // activated.shape[-1]
        return hook
    try:
        for i, mlp in mlps.items():
            handles.append(mlp.register_forward_pre_hook(make_hook(i)))
        with torch.no_grad():
            for ids in anchors:
                if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
                    raise ValueError("each anchor must be [1, T] token ids")
                _backbone(model)(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for i, mlp in mlps.items():
        if counts[i] == 0:
            raise ValueError("anchor forward did not execute requested FFN layer")
        norm = mlp.down_proj.weight.detach().double().norm(dim=0).cpu()
        importance = (sums[i] / counts[i] * norm).numpy()
        if not np.all(np.isfinite(importance)):
            raise FloatingPointError("FFN channel importance is non-finite")
        result[i] = importance
    return result


def select_top_ffn_channels(importance: Mapping[int, np.ndarray],
                            keep_ratios: float | Mapping[int, float]) -> dict[int, np.ndarray]:
    result = {}
    for layer, raw in importance.items():
        imp = np.asarray(raw, dtype=np.float64)
        rho = keep_ratios[layer] if isinstance(keep_ratios, Mapping) else keep_ratios
        if imp.ndim != 1 or not len(imp) or not np.all(np.isfinite(imp)):
            raise ValueError("importance must be a nonempty finite vector")
        if not np.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError("keep ratio must be finite and in (0, 1]")
        k = max(1, int(round(float(rho) * len(imp))))
        result[layer] = np.sort(np.argsort(-imp, kind="stable")[:k]).astype(np.int64)
    return result


class QwenSparseFfnHook:
    """Execute selected SwiGLU channels with physically sliced linear weights."""

    def __init__(self, model, channel_masks: Mapping[int, Sequence[int] | np.ndarray]):
        torch = _torch()
        self.layers = list(decoder_layers(model))
        self.channel_masks = {}
        self._slices = {}
        self._originals = {}
        for layer, raw in channel_masks.items():
            if not isinstance(layer, int) or not 0 <= layer < len(self.layers):
                raise ValueError("FFN mask layer out of range")
            mlp = _swiglu(self.layers[layer])
            idx = np.asarray(raw)
            width = mlp.gate_proj.weight.shape[0]
            if (idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu" or
                    np.any(idx < 0) or np.any(idx >= width) or np.any(np.diff(idx) <= 0)):
                raise ValueError("FFN mask needs unique ascending in-range channel indices")
            self.channel_masks[layer] = idx.astype(np.int64)
            tensor_idx = torch.as_tensor(idx.copy(), device=mlp.gate_proj.weight.device, dtype=torch.long)
            def bias(proj, sliced=False):
                b = proj.bias
                return None if b is None else (b.detach()[tensor_idx].contiguous() if sliced else b.detach())
            self._slices[layer] = (
                mlp.gate_proj.weight.detach()[tensor_idx].contiguous(), bias(mlp.gate_proj, True),
                mlp.up_proj.weight.detach()[tensor_idx].contiguous(), bias(mlp.up_proj, True),
                mlp.down_proj.weight.detach()[:, tensor_idx].contiguous(), bias(mlp.down_proj),
            )
        if channel_masks:
            for layer in self.layers:
                _swiglu(layer)
        self.reset()

    def reset(self):
        self.ffn_layer_calls = [0] * len(self.layers)
        self.ffn_channels_executed = 0
        self.ffn_dense_channels = 0
        self.ffn_macs_executed = 0
        self.ffn_dense_macs = 0

    def attach(self):
        import torch.nn.functional as F
        if self._originals:
            raise RuntimeError("FFN hook already attached")
        self.reset()
        for i, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            original = mlp.forward
            self._originals[i] = original
            def forward(x, *args, _i=i, _original=original, **kwargs):
                layer_mlp = self.layers[_i].mlp
                width = layer_mlp.gate_proj.weight.shape[0]
                hidden = layer_mlp.gate_proj.weight.shape[1]
                kept = len(self.channel_masks[_i]) if _i in self.channel_masks else width
                self.ffn_layer_calls[_i] += 1
                self.ffn_channels_executed += kept
                self.ffn_dense_channels += width
                self.ffn_macs_executed += 3 * hidden * kept
                self.ffn_dense_macs += 3 * hidden * width
                if _i not in self.channel_masks:
                    return _original(x, *args, **kwargs)
                wg, bg, wu, bu, wd, bd = self._slices[_i]
                def align(t):
                    return t if t is None or (t.device == x.device and t.dtype == x.dtype) else t.to(device=x.device, dtype=x.dtype)
                return F.linear(F.silu(F.linear(x, align(wg), align(bg))) *
                                F.linear(x, align(wu), align(bu)), align(wd), align(bd))
            mlp.forward = forward
        return self

    def detach(self):
        for i, original in self._originals.items():
            self.layers[i].mlp.forward = original
        self._originals.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()


def _layer_hidden(output):
    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"decoder layer returned {type(output).__name__}, expected Tensor or tuple")


def _replace_hidden(output, hidden):
    if isinstance(output, (tuple, list)):
        return type(output)((hidden, *output[1:]))
    return hidden


class _LayerEarlyExit(Exception):
    """Raised inside a layer hook to leave the decoder loop after ``layer_idx``."""

    def __init__(self, layer_idx: int, hidden_state) -> None:
        super().__init__(f"early exit after layer {layer_idx}")
        self.layer_idx = layer_idx
        self.hidden_state = hidden_state


ExitPolicy = Callable[[int, Any, np.ndarray], bool]


class QwenInterLayerHook:
    """Forward hooks on every decoder layer: count, measure, steer, and stop.

    ``exit_policy(layer_idx, last_token_state[B, d], delta[B]) -> bool`` is asked at the
    layers in ``exit_layers`` (all but the last when None); True raises ``_LayerEarlyExit``.
    ``steering_basis`` is ``(r, d)``; rows are orthonormalised, and action ``a`` in
    ``1..r`` adds ``eta * u_{a-1}`` to every position at ``steering_layers``.
    """

    def __init__(
        self,
        model,
        capture_layers: Sequence[int] | None = None,
        exit_policy: ExitPolicy | None = None,
        exit_layers: Sequence[int] | None = None,
        steering_basis: np.ndarray | None = None,
        steering_layers: Sequence[int] | None = None,
        eta: float = 0.0,
        action: int = 0,
        eps: float = _EPS,
    ) -> None:
        torch = _torch()
        # Snapshot the modules: hooks stay on the real layers even if the list is later
        # patched with replay stand-ins, so replayed layers are never counted as executed.
        self.layers = list(decoder_layers(model))
        self.num_layers = len(self.layers)
        if self.num_layers == 0:
            raise ValueError("model has no decoder layers")
        self.capture_layers = self._check_layers(capture_layers, "capture_layers")
        self.exit_layers = (
            set(range(self.num_layers - 1)) if exit_layers is None
            else self._check_layers(exit_layers, "exit_layers")
        )
        self.exit_policy = exit_policy
        if not np.isfinite(eta):
            raise ValueError("eta must be finite")
        if action < 0:
            raise ValueError("action must be >= 0")
        self.eta, self.action, self.eps = float(eta), int(action), float(eps)
        self._steer = None
        if steering_basis is not None:
            basis = np.asarray(steering_basis, dtype=np.float64)
            if basis.ndim != 2 or not np.all(np.isfinite(basis)):
                raise ValueError("steering_basis must be a finite (r, d) matrix")
            q, _ = np.linalg.qr(basis.T)
            self._steer = torch.as_tensor(q.T.copy(), dtype=torch.float32)
        if self.action > 0 and self.eta != 0.0:
            if self._steer is None or self.action > self._steer.shape[0]:
                raise ValueError("action needs a steering_basis row (1-based)")
        self.steering_layers = self._check_layers(steering_layers, "steering_layers") or set()
        self._handles: list[Any] = []
        self.reset()

    def _check_layers(self, layers, name) -> set[int]:
        if layers is None:
            return set()
        out = {int(i) for i in layers}
        if any(not 0 <= i < self.num_layers for i in out):
            raise ValueError(f"{name} must index the {self.num_layers} decoder layers")
        return out

    def reset(self) -> None:
        self.layers_executed = 0
        self.layer_calls = [0] * self.num_layers
        self.deltas: dict[int, np.ndarray] = {}
        self.captured: dict[int, Any] = {}

    def attach(self) -> "QwenInterLayerHook":
        if self._handles:
            raise RuntimeError("hook already attached")
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(idx), with_kwargs=True))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "QwenInterLayerHook":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def steering_delta(self, idx: int, hidden):
        """The steering step for ``hidden`` at layer ``idx``, or None for the identity."""
        if self.action == 0 or self.eta == 0.0 or idx not in self.steering_layers:
            return None
        u = self._steer[self.action - 1].to(dtype=hidden.dtype, device=hidden.device)
        return self.eta * u

    def _make_hook(self, idx: int):
        def hook(module, args, kwargs, output):
            self.layers_executed += 1
            self.layer_calls[idx] += 1
            hidden = _layer_hidden(output)
            h_in = args[0] if args else kwargs["hidden_states"]
            new_output = None
            step = self.steering_delta(idx, hidden)
            if step is not None:
                hidden = hidden + step
                new_output = _replace_hidden(output, hidden)
            last = hidden[:, -1, :].detach().float()
            prev = h_in[:, -1, :].detach().float()
            delta = ((last - prev).norm(dim=-1) / (last.norm(dim=-1) + self.eps)).cpu().numpy()
            self.deltas[idx] = delta
            if idx in self.capture_layers:
                self.captured[idx] = last.clone()
            if self.exit_policy is not None and idx in self.exit_layers and idx < self.num_layers - 1:
                if self.exit_policy(idx, last, delta):
                    raise _LayerEarlyExit(idx, hidden.detach())
            return new_output

        return hook


def simplex_etf(k: int) -> np.ndarray:
    """``k`` unit rows of a simplex equiangular tight frame in ``k`` dimensions."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return np.ones((1, 1))
    return math.sqrt(k / (k - 1)) * (np.eye(k) - 1.0 / k)


class EpisodicManifoldMemory:
    """Cosine-kernel associative memory over ``L2Norm((h - center) P)`` keys.

    ``query`` returns ``softmax(beta * K q)^T V``. ``center`` is fixed when the memory is
    built (``fit_memories`` uses the anchor mean): raw decoder states share a few huge
    outlier dimensions, so uncentred cosines all sit near 1 and the read degrades to the
    class prior. ``exclude_ids`` masks entries written
    under those ids (leave-one-out). The store is a FIFO ring of ``capacity`` entries.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        capacity: int = 256,
        beta: float = 12.0,
        projection: np.ndarray | None = None,
        center: np.ndarray | None = None,
    ) -> None:
        if dim < 1 or num_labels < 1 or capacity < 3:
            raise ValueError("dim and num_labels must be >= 1; capacity must be >= 3")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and > 0")
        self.dim, self.num_labels, self.capacity, self.beta = dim, num_labels, capacity, float(beta)
        self.projection = None
        if projection is not None:
            p = np.asarray(projection, dtype=np.float64)
            if p.ndim != 2 or p.shape[0] != dim or not np.all(np.isfinite(p)):
                raise ValueError(f"projection must be a finite ({dim}, r) matrix")
            self.projection = p
        self.center = None
        if center is not None:
            c = np.asarray(center, dtype=np.float64)
            if c.shape != (dim,) or not np.all(np.isfinite(c)):
                raise ValueError(f"center must be a finite ({dim},) vector")
            self.center = c
        key_dim = dim if self.projection is None else self.projection.shape[1]
        self._keys = np.zeros((0, key_dim))
        self._raw_keys = np.zeros((0, dim))
        self._values = np.zeros((0, num_labels))
        self._ids: list[Any] = []

    def __len__(self) -> int:
        return len(self._ids)

    def encode(self, hidden) -> np.ndarray:
        h = np.atleast_2d(np.asarray(hidden, dtype=np.float64))
        if h.shape[1] != self.dim or not np.all(np.isfinite(h)):
            raise ValueError(f"hidden states must be finite with last dim {self.dim}")
        if self.center is not None:
            h = h - self.center
        z = h if self.projection is None else h @ self.projection
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        if np.any(norm == 0) or not np.all(np.isfinite(norm)):
            raise ValueError("cannot key a zero or overflowing state")
        return z / norm

    def write(self, hidden, targets, ids: Sequence[Any]) -> None:
        raw = np.atleast_2d(np.asarray(hidden, dtype=np.float64))
        if raw.shape[1] != self.dim or not np.all(np.isfinite(raw)):
            raise ValueError("hidden states must be finite with the memory dimension")
        if len(set(ids)) != len(ids) or any(i in self._ids for i in ids):
            raise ValueError("duplicate memory ID")
        t = np.asarray(targets)
        if t.ndim == 1:
            if t.dtype.kind not in "iu" or np.any((t < 0) | (t >= self.num_labels)):
                raise ValueError("label index out of range")
            values = np.eye(self.num_labels)[t.astype(int)]
        else:
            values = np.asarray(t, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != self.num_labels:
                raise ValueError("target distributions must have shape (n, num_labels)")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("target distributions must be finite and non-negative")
            row_sums = values.sum(axis=1, keepdims=True)
            if not np.all((row_sums > 0) & np.isfinite(row_sums)):
                raise ValueError("every target distribution needs a positive mass")
            values = values / row_sums
        if not len(raw) == len(values) == len(ids):
            raise ValueError("hidden, targets and ids must have the same length")
        new_raw = np.vstack([self._raw_keys, raw])[-self.capacity:]
        new_values = np.vstack([self._values, values])[-self.capacity:]
        new_ids = (self._ids + list(ids))[-self.capacity:]
        old_center = self.center
        if len(new_raw):
            self.center = new_raw.mean(axis=0)
        try:
            if len(new_raw) == 1:
                original = new_raw if self.projection is None else new_raw @ self.projection
                norm = np.linalg.norm(original)
                if norm == 0 or not np.isfinite(norm):
                    raise ValueError("cannot key a zero or overflowing state")
                new_keys = np.zeros((1, self._keys.shape[1]))
            else:
                new_keys = self.encode(new_raw)
        except ValueError:
            self.center = old_center
            raise
        self._raw_keys, self._values, self._ids, self._keys = new_raw, new_values, new_ids, new_keys

    def query(self, hidden, exclude_ids: Sequence[Any] | None = None) -> np.ndarray:
        if exclude_ids is not None and self.center is not None:
            raw = np.atleast_2d(np.asarray(hidden, dtype=np.float64))
            if raw.shape[1] != self.dim or not np.all(np.isfinite(raw)):
                raise ValueError(f"hidden states must be finite with last dim {self.dim}")
            if len(exclude_ids) != len(raw):
                raise ValueError("one exclude id per query row")
            if not len(self):
                raise ValueError("memory is empty")
            stored = np.asarray(self._ids, dtype=object)
            result = np.empty((len(raw), self.num_labels))
            for i, (q, excluded) in enumerate(zip(raw, exclude_ids)):
                keep = stored != excluded
                remaining = self._raw_keys[keep]
                if not len(remaining):
                    result[i] = 1.0 / self.num_labels
                    continue
                center = remaining.mean(axis=0)
                def normalized(rows):
                    z = rows - center
                    if self.projection is not None:
                        z = z @ self.projection
                    norms = np.linalg.norm(z, axis=1, keepdims=True)
                    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
                        raise ValueError("cannot key a zero or overflowing state")
                    return z / norms
                logits = self.beta * (normalized(q[None, :]) @ normalized(remaining).T)
                result[i] = _softmax(logits, axis=1)[0] @ self._values[keep]
            return result
        return self._read(self.encode(hidden), exclude_ids)

    def _read(self, q: np.ndarray, exclude_ids: Sequence[Any] | None) -> np.ndarray:
        if len(self) == 0:
            raise ValueError("memory is empty")
        sims = self.beta * (q @ self._keys.T)
        if exclude_ids is None:
            return _softmax(sims, axis=1) @ self._values
        if len(exclude_ids) != len(q):
            raise ValueError("one exclude id per query row")
        stored = np.asarray(self._ids, dtype=object)
        result = np.empty((len(q), self.num_labels))
        for i, excluded in enumerate(exclude_ids):
            valid_mask = stored != excluded
            if not np.any(valid_mask):
                result[i] = 1.0 / self.num_labels
                continue
            valid_logits = sims[i, valid_mask]
            result[i] = _softmax(valid_logits) @ self._values[valid_mask]
        return result

    def loo_query(self, hidden=None, ids: Sequence[Any] | None = None) -> np.ndarray:
        """Read every supplied anchor against retained slots, excluding matching IDs.

        When a center is configured, refit it for each exclusion for both query
        and remaining keys. With no center, retain the uncentered geometry.
        Supplying all anchors preserves rows that have fallen out of the FIFO window.
        """
        raw = self._raw_keys if hidden is None else np.atleast_2d(np.asarray(hidden, dtype=np.float64))
        query_ids = self._ids if ids is None else list(ids)
        if raw.ndim != 2 or raw.shape[1] != self.dim or not np.all(np.isfinite(raw)):
            raise ValueError("hidden states must be finite with the memory dimension")
        if len(raw) != len(query_ids):
            raise ValueError("one ID per query row")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("duplicate query ID")
        if not len(self):
            raise ValueError("memory is empty")
        return self.query(raw, exclude_ids=query_ids)


def select_memory_lambda(
    base_probs: np.ndarray,
    memory_probs: np.ndarray,
    labels: Sequence[int],
    grid: Sequence[float] = tuple(np.round(np.linspace(0.1, 1.0, 10), 2)),
    se_mult: float = 2.0,
    null_probs: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Pick the memory weight by paired out-of-fold log-likelihood gain over ``lambda=0``.

    A ``lambda`` passes only when its mean per-sample gain exceeds ``se_mult`` standard
    errors. With ``null_probs`` (e.g. the leave-one-out class prior) the memory must also
    beat that null mixed at the same ``lambda`` by ``se_mult`` standard errors: a saturated
    base probe gains from mixing in anything, so the first test alone would accept a
    memory that carries no more than the label frequencies. No pass returns 0.
    """
    base = np.asarray(base_probs, dtype=np.float64)
    mem = np.asarray(memory_probs, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    n = len(y)
    if n < 2:
        raise ValueError("lambda selection needs at least 2 out-of-fold samples")
    if base.shape != mem.shape or base.shape[0] != n:
        raise ValueError("base_probs, memory_probs and labels disagree in shape")
    nulls = {"uniform": np.full_like(base, 1.0 / base.shape[1])}
    counts = np.bincount(y, minlength=base.shape[1]).astype(np.float64)
    nulls["laplace"] = (counts[None, :] - np.eye(base.shape[1])[y] + 1.0) / (n - 1 + base.shape[1])
    if null_probs is not None:
        null = np.asarray(null_probs, dtype=np.float64)
        if null.shape != base.shape or not np.all(np.isfinite(null)):
            raise ValueError("null_probs must be finite with the shape of base_probs")
        nulls["provided"] = null
    rows = np.arange(n)
    ll0 = np.log(np.clip(base[rows, y], 1e-12, None))
    report: dict[str, Any] = {"n": n, "candidates": []}
    best_lam, best_gain = 0.0, 0.0
    for lam in grid:
        mix = (1 - lam) * base + lam * mem
        gain = np.log(np.clip(mix[rows, y], 1e-12, None)) - ll0
        mean = float(gain.mean())
        se = float(gain.std(ddof=1) / math.sqrt(n))
        if not (np.isfinite(mean) and np.isfinite(se)):
            raise ValueError("non-finite out-of-fold gain")
        passed = mean > 0 and mean > se_mult * se
        entry = {"lambda": float(lam), "mean_gain": mean, "se": se}
        for null_name, null in nulls.items():
            null_mix = (1 - lam) * base + lam * null
            excess = np.log(np.clip(mix[rows, y], 1e-12, None)) - np.log(np.clip(null_mix[rows, y], 1e-12, None))
            excess_mean = float(excess.mean())
            excess_se = float(excess.std(ddof=1) / math.sqrt(n))
            if not (np.isfinite(excess_mean) and np.isfinite(excess_se)):
                raise ValueError("non-finite out-of-fold gain over the null")
            passed = passed and excess_mean > 0 and excess_mean > se_mult * excess_se
            entry[f"gain_over_{null_name}"] = excess_mean
            entry[f"gain_over_{null_name}_se"] = excess_se
            if null_name == "provided":
                entry.update(gain_over_null=excess_mean, gain_over_null_se=excess_se)
        entry["passed"] = passed
        report["candidates"].append(entry)
        if passed and mean > best_gain:
            best_lam, best_gain = float(lam), mean
    report["lambda"] = best_lam
    return best_lam, report


def _anchor_logit_stats(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-label mean and population std of anchor logits, each ``(1, K)``.

    Raises when a variance is not finite or not above ``1e-10`` times the mean square: a
    constant column such as ``[0.1, 0.1, 0.1]`` rounds to a tiny positive variance, and
    its std would blow the standardised logits up instead of failing.
    """
    mean = z.mean(axis=0, keepdims=True)
    second = (z ** 2).mean(axis=0, keepdims=True)
    var = z.var(axis=0, keepdims=True)
    if not (np.all(np.isfinite(second)) and np.all(np.isfinite(var))) or np.any(
            var <= np.maximum(0.0, 1e-10 * second)):
        raise ValueError("degenerate anchor logits with zero variance")
    return mean, np.sqrt(var)


def _loo_mean(z: np.ndarray) -> np.ndarray:
    """Row ``i``: mean over the other rows (closed form)."""
    n = len(z)
    return (z.sum(axis=0, keepdims=True) - z) / (n - 1)


def _loo_std(z: np.ndarray) -> np.ndarray:
    """Row ``i``: population std over the other rows (closed form).

    Raises when any leave-one-out variance is not finite or not above ``1e-10`` times its
    mean square. The closed form ``second - mu ** 2`` cancels: leaving row 0 out of
    ``[0.3, 1.1, 1.1]`` gives ``2.2e-16`` instead of 0, and a std of ``1.5e-8`` would
    standardise the logits to about ``5e7``. A zero std would give inf or NaN.
    """
    n = len(z)
    mu = _loo_mean(z)
    second = ((z ** 2).sum(axis=0, keepdims=True) - z ** 2) / (n - 1)
    var = second - mu ** 2
    if not (np.all(np.isfinite(second)) and np.all(np.isfinite(var))) or np.any(
            var <= np.maximum(0.0, 1e-10 * second)):
        raise ValueError("degenerate leave-one-out anchor logits with zero variance")
    return np.sqrt(var)


class VerbalizerProbe:
    """Label logits from a pre-norm residual: final norm, then only the label rows of lm_head.

    Each label's logit is the logsumexp of its token ids.
    """

    def __init__(self, model, label_token_groups: Sequence[Sequence[int]]) -> None:
        torch = _torch()
        if not label_token_groups or any(len(g) == 0 for g in label_token_groups):
            raise ValueError("every label needs at least one token id")
        self.norm = _backbone(model).norm
        head = model.lm_head
        self.groups = [list(map(int, g)) for g in label_token_groups]
        ids = torch.tensor([i for g in self.groups for i in g], dtype=torch.long, device=head.weight.device)
        if int(ids.max()) >= head.weight.shape[0]:
            raise ValueError("label token id outside the vocabulary")
        self.weight = head.weight.detach()[ids].float()
        self.bias = None if head.bias is None else head.bias.detach()[ids].float()

    @property
    def num_labels(self) -> int:
        return len(self.groups)

    def __call__(self, residual) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            dev = self.norm.weight.device
            x = self.norm(residual.to(device=dev, dtype=self.norm.weight.dtype)).float()
            w = self.weight if self.weight.device == x.device else self.weight.to(device=x.device)
            logits = x @ w.T
            if self.bias is not None:
                b = self.bias if self.bias.device == x.device else self.bias.to(device=x.device)
                logits = logits + b
        out, start = [], 0
        for g in self.groups:
            out.append(torch.logsumexp(logits[:, start:start + len(g)], dim=-1))
            start += len(g)
        result = torch.stack(out, dim=-1).cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("probe produced non-finite logits")
        return result


@dataclass
class LayerExitDecision:
    label: int
    probs: np.ndarray
    exit_layer: int
    early_exit: bool
    layers_executed: int
    margin: float = float("nan")
    threshold: float = float("nan")
    value: float = float("nan")
    layer_calls: list[int] = field(default_factory=list)
    mcts_margin: float = float("nan")
    drift: float = float("nan")
    gamma_hat: float = float("nan")
    ffn_channels_executed: int = 0
    ffn_dense_channels: int = 0
    ffn_macs_executed: int = 0
    ffn_dense_macs: int = 0
    active_hook_weights: dict[int, float] = field(default_factory=dict)


@dataclass
class PrunedCandidateResult:
    best: int
    survivors: list[int]
    shallow_values: np.ndarray
    deep_values: dict[int, float]
    layers_executed: int
    full_layers: int
    mcts_visits: np.ndarray


def _replay_layer_cls():
    torch = _torch()

    class _ReplayLayer(torch.nn.Module):
        """Stands in for an already-computed layer: returns the cached residual."""

        def __init__(self, cached, orig_layer) -> None:
            super().__init__()
            self.cached = cached
            object.__setattr__(self, "_orig_layer", orig_layer)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(object.__getattribute__(self, "_orig_layer"), name)

        def forward(self, hidden_states, *args, **kwargs):
            if hidden_states.shape != self.cached.shape:
                raise ValueError("replayed residual does not match the input shape")
            return self.cached

    return _ReplayLayer


class MctsLayerwiseQwenRunner:
    """MCTS-certified early exit and candidate pruning over a Qwen2 causal LM.

    ``memories`` maps a layer index to ``(EpisodicManifoldMemory, lambda)``; the probe
    distribution at that layer becomes ``(1 - lambda) * p_probe + lambda * p_mem``.
    ``probe_stats`` maps a layer index to ``(logit_mean, logit_std)``, each ``(1, K)``,
    fitted on anchors; with it ``p_probe = softmax((z - logit_mean) / logit_std)``. Once
    any stats are set, every layer read must have them, so one decision never mixes
    calibrated and raw readouts. Without stats the raw probe is used unchanged.
    ``checkpoints`` are the layers where the label distribution is read; exit is
    considered from the second checkpoint on and only at layers ``>= min_exit_layer``
    (default ``max(2, num_layers // 2)``).
    """

    def __init__(
        self,
        model,
        probe: VerbalizerProbe,
        checkpoints: Sequence[int],
        memories: Mapping[int, tuple[EpisodicManifoldMemory, float]] | None = None,
        probe_stats: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
        planner: LatentMctsPlanner | None = None,
        tau: float = 1.0,
        gamma: float = 0.5,
        min_exit_layer: int | None = None,
        mcts_margin_min: float = 0.2,
        layer_weights: Mapping[int, float] | None = None,
        ffn_channel_masks: Mapping[int, Sequence[int] | np.ndarray] | None = None,
    ) -> None:
        _torch()
        if not 0 < gamma < _GAMMA_MAX:
            raise ValueError(f"gamma must be in (0, {_GAMMA_MAX})")
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and >= 0")
        if not np.isfinite(mcts_margin_min) or not 0 <= mcts_margin_min < 1:
            raise ValueError("mcts_margin_min must be in [0, 1)")
        self.model = model
        self.backbone = _backbone(model)
        self.num_layers = len(self.backbone.layers)
        self.probe = probe
        self.checkpoints = sorted({int(c) for c in checkpoints})
        if any(not 0 <= c < self.num_layers - 1 for c in self.checkpoints):
            raise ValueError("checkpoints must lie before the last layer")
        self.memories = dict(memories or {})
        for lam in (lam for _, lam in self.memories.values()):
            if not 0 <= lam <= 1:
                raise ValueError("memory lambda must be in [0, 1]")
        k = probe.num_labels
        self.probe_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for layer, (mean, std) in (probe_stats or {}).items():
            self.probe_stats[int(layer)] = self._check_probe_stats(int(layer), mean, std, k)
        self.planner = planner or LatentMctsPlanner(num_simulations=32, temperature=1.0, max_depth=2)
        self.tau, self.gamma = float(tau), float(gamma)
        self.min_exit_layer = (max(2, self.num_layers // 2) if min_exit_layer is None
                               else int(min_exit_layer))
        if not 0 <= self.min_exit_layer < self.num_layers:
            raise ValueError("min_exit_layer must index a decoder layer")
        self.mcts_margin_min = float(mcts_margin_min)
        self.labels = [str(i) for i in range(k)]
        self.etf = simplex_etf(k)
        self.layer_weights = self._normalize_layer_weights(layer_weights or {})
        self.base_utility_weights: dict[int, float] = {}
        self.ffn_channel_masks = dict(ffn_channel_masks or {})
        self._sparse_hook = QwenSparseFfnHook(self.model, self.ffn_channel_masks)
        self._sparse_masks_snapshot = {i: np.asarray(v).copy() for i, v in self.ffn_channel_masks.items()}

    def _normalize_layer_weights(self, weights):
        if not weights:
            return {}
        if any(not isinstance(i, int) or not 0 <= i < self.num_layers or
               not np.isfinite(v) or v < 0 for i, v in weights.items()):
            raise ValueError("layer weights need valid layers and finite nonnegative values")
        total = float(sum(weights.values()))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("layer weights need positive finite total")
        return {i: float(v / total) for i, v in weights.items()}

    def fit_layer_weights(self, z_by_layer: Mapping[int, np.ndarray], labels: Sequence[int],
                          top_k_hooks: int = 4, reward_scale: float = 10.0,
                          memory_loo_by_layer: Mapping[int, np.ndarray] | None = None) -> dict[int, float]:
        layers = sorted(z_by_layer)
        y = np.asarray(labels)
        if not layers or top_k_hooks < 1 or not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("need layers, positive top_k_hooks and reward_scale")
        if any(not isinstance(i, int) or not 0 <= i < self.num_layers for i in layers):
            raise ValueError("hook layer out of range")
        utility = []
        for layer in layers:
            z = np.asarray(z_by_layer[layer], dtype=np.float64)
            if (z.ndim != 2 or z.shape != (len(y), self.probe.num_labels) or len(y) < 3 or
                    not np.all(np.isfinite(z)) or np.any(y < 0) or np.any(y >= z.shape[1])):
                raise ValueError("finite anchor logits and matching labels required")
            p = _softmax((z - _loo_mean(z)) / _loo_std(z), axis=1)
            entry = self.memories.get(layer)
            if entry is not None and entry[1] > 0:
                if memory_loo_by_layer is None or layer not in memory_loo_by_layer:
                    raise ValueError("anchor-aligned memory LOO probabilities required")
                memory_loo = np.asarray(memory_loo_by_layer[layer])
                if memory_loo.shape != p.shape:
                    raise ValueError("memory LOO probabilities must match anchor rows")
                p = (1 - entry[1]) * p + entry[1] * memory_loo
            acc = np.mean(np.argmax(p, axis=1) == y)
            ll = np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
            utility.append(float(acc + 0.1 * ll))
        reward = reward_scale * (np.asarray(utility) - np.mean(utility))
        base = _softmax(reward)
        self.base_utility_weights = {layer: float(base[j]) for j, layer in enumerate(layers)}
        planner = LatentMctsPlanner(num_simulations=max(64, self.planner.num_simulations),
                                    c_puct=self.planner.c_puct, temperature=0.25, max_depth=2)
        _, policy, _ = planner.plan(reward, [str(i) for i in layers], simplex_etf(len(layers)))
        keep = np.argsort(-policy, kind="stable")[:min(top_k_hooks, len(layers))]
        weights = {layer: float(policy[j]) if j in keep else 0.0 for j, layer in enumerate(layers)}
        self.layer_weights = self._normalize_layer_weights(weights)
        return dict(self.layer_weights)

    def fit_ffn_pruning(self, anchors: Sequence[Any], layers: Sequence[int],
                        keep_ratio: float | Mapping[int, float] = 0.75) -> dict[int, np.ndarray]:
        masks = select_top_ffn_channels(score_ffn_channels(self.model, anchors, layers), keep_ratio)
        self._sparse_hook = QwenSparseFfnHook(self.model, masks)
        self.ffn_channel_masks = masks
        self._sparse_masks_snapshot = {i: np.asarray(v).copy() for i, v in masks.items()}
        return masks

    def _check_probe_stats(self, layer: int, mean, std, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"probe_stats layer {layer} does not index a decoder layer")
        m = np.asarray(mean, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
        if m.shape != (1, k) or s.shape != (1, k):
            raise ValueError(f"probe_stats for layer {layer} must be two (1, {k}) arrays")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s))) or np.any(s <= 0):
            raise ValueError(f"probe_stats for layer {layer} must be finite with std > 0")
        return m, s

    # -- readout --------------------------------------------------------------------------
    def probe_logits(self, layer: int, residual_last) -> np.ndarray:
        """Probe logits at ``layer``, standardised by that layer's anchor stats when set."""
        z = self.probe(residual_last)
        if not self.probe_stats:
            return z
        stats = self.probe_stats.get(layer)
        if stats is None:
            raise ValueError(f"probe_stats are set but layer {layer} has none; refit with it")
        mean, std = stats
        return (z - mean) / std

    def fused_probs(self, layer: int, residual_last, exclude_id: Any = None) -> np.ndarray:
        p = _softmax(self.probe_logits(layer, residual_last), axis=1)
        entry = self.memories.get(layer)
        if entry is not None and entry[1] > 0:
            memory, lam = entry
            excl = None if exclude_id is None else [exclude_id] * len(p)
            p = (1 - lam) * p + lam * memory.query(residual_last.cpu().numpy(), exclude_ids=excl)
        return p

    def mcts_margin(self, probs: np.ndarray) -> tuple[int, float, float]:
        """MCTS over the label simplex: (best label, visit-share margin, root value)."""
        z = np.log(np.clip(probs, 1e-12, None))
        z = z - z.mean()
        best, policy, info = self.planner.plan(z, self.labels, self.etf)
        second = np.partition(policy, -2)[-2] if len(policy) > 1 else 0.0
        return best, float(policy[best] - second), float(info["root_q"][best])

    def fit_memories(
        self,
        anchors: Sequence[Any],
        labels: Sequence[int],
        layers: Sequence[int],
        beta: float = 12.0,
        capacity: int = 256,
        fit_mcts_layer_weights: bool = False,
        top_k_hooks: int = 4,
    ) -> dict[int, dict[str, Any]]:
        """Fill one memory per layer from full forward passes over labelled anchors.

        Each layer's mixing weight comes from ``select_memory_lambda`` on leave-one-out
        reads, so an anchor never retrieves itself, against the leave-one-out class prior
        as the null. Keys are centred on the anchor mean of that layer. The anchor passes
        are calibration cost, not counted against inference.

        The same pass sets ``probe_stats[layer]`` to the anchor mean and std of the probe
        logits; a zero variance, in the full or any leave-one-out set, raises ``ValueError``.
        The base distribution that the lambda gate scores uses leave-one-out stats, so an
        anchor is not standardised with its own logits.
        """
        layers = sorted({int(layer) for layer in layers})
        if capacity < 3:
            raise ValueError("capacity must be >= 3")
        if len(anchors) != len(labels) or len(anchors) < 3:
            raise ValueError("need at least 3 anchors, one label each")
        states: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        for ids in anchors:
            hook = QwenInterLayerHook(self.model, capture_layers=layers)
            if self._run(hook, ids) is not None:
                raise RuntimeError("anchor pass exited early")
            for layer in layers:
                states[layer].append(hook.captured[layer].cpu().numpy())
        reports = {}
        z_by_layer = {}
        memory_loo_by_layer = {}
        torch = _torch()
        y = np.asarray(labels)
        onehot = np.eye(self.probe.num_labels)[y]
        loo_prior = (onehot.sum(axis=0)[None, :] - onehot) / (len(y) - 1)
        for layer in layers:
            h = np.concatenate(states[layer], axis=0)
            memory = EpisodicManifoldMemory(h.shape[1], self.probe.num_labels, capacity, beta,
                                            center=h.mean(axis=0))
            memory.write(h, y, ids=list(range(len(labels))))
            z = self.probe(torch.as_tensor(h, dtype=torch.float32))
            z_by_layer[layer] = z
            mean, std = _anchor_logit_stats(z)
            base = _softmax((z - _loo_mean(z)) / _loo_std(z), axis=1)
            memory_loo_by_layer[layer] = memory.loo_query(h, ids=list(range(len(labels))))
            lam, report = select_memory_lambda(base, memory_loo_by_layer[layer], labels,
                                               null_probs=loo_prior)
            self.memories[layer] = (memory, lam)
            self.probe_stats[layer] = self._check_probe_stats(layer, mean, std, self.probe.num_labels)
            report["probe_logit_mean"] = mean[0].tolist()
            report["probe_logit_std"] = std[0].tolist()
            reports[layer] = report
        if fit_mcts_layer_weights:
            weights = self.fit_layer_weights(z_by_layer, labels, top_k_hooks=top_k_hooks,
                                             memory_loo_by_layer=memory_loo_by_layer)
            for layer in layers:
                reports[layer]["mcts_layer_weight"] = weights[layer]
        return reports

    def ablation_predict(self, arm: str, input_ids) -> LayerExitDecision:
        """Evaluate a calibrated component ablation with the same exit rule."""
        old_weights, old_min_exit = self.layer_weights, self.min_exit_layer
        try:
            if arm == "multihook_weighted_full28_no_early_exit":
                self.min_exit_layer = self.num_layers
            elif arm == "multihook_no_mcts_margin_early_exit":
                if not self.base_utility_weights:
                    raise ValueError("base utility weights require a calibrated fit")
                self.layer_weights = self.base_utility_weights
            elif arm == "multihook_equal_weight_full28":
                if not self.probe_stats:
                    raise ValueError("equal-weight ablation requires a calibrated fit")
                self.layer_weights = {layer: 1.0 / len(self.probe_stats) for layer in self.probe_stats}
                self.min_exit_layer = self.num_layers
            else:
                raise ValueError(f"unknown ablation arm: {arm}")
            return self.predict(input_ids)
        finally:
            self.layer_weights, self.min_exit_layer = old_weights, old_min_exit

    def contraction(self, drift: float, prev_drift: float | None) -> float:
        """Empirical drift ratio, floored at ``gamma`` and capped at 0.95."""
        if prev_drift is None:
            return self.gamma
        return min(_GAMMA_MAX, max(self.gamma, drift / max(prev_drift, _EPS)))

    def tail_bound(self, drift: float, gamma_hat: float) -> float:
        """Remaining probability movement ``2 tau g / (1 - g) * drift`` (sup-norm, x2 for a gap)."""
        return 2.0 * self.tau * gamma_hat / (1.0 - gamma_hat) * float(drift)

    def _run(self, hook: QwenInterLayerHook, input_ids):
        torch = _torch()
        if (self.ffn_channel_masks.keys() != self._sparse_masks_snapshot.keys() or
                any(not np.array_equal(v, self._sparse_masks_snapshot[i])
                    for i, v in self.ffn_channel_masks.items())):
            self._sparse_hook = QwenSparseFfnHook(self.model, self.ffn_channel_masks)
            self._sparse_masks_snapshot = {i: np.asarray(v).copy() for i, v in self.ffn_channel_masks.items()}
        self._last_sparse_hook = self._sparse_hook
        with torch.no_grad(), self._sparse_hook, hook:
            try:
                self.backbone(input_ids=input_ids, use_cache=False)
            except _LayerEarlyExit as stop:
                return stop
        return None

    # -- layerwise early exit -------------------------------------------------------------
    def predict(self, input_ids, exclude_id: Any = None) -> LayerExitDecision:
        if input_ids.shape[0] != 1:
            raise ValueError("early exit decides per sequence; pass one row")
        seen: dict[int, dict[str, Any]] = {}
        prev: dict[str, Any] = {}
        cum_w = 0.0
        cum_z = None
        effective_steps = 0

        def trajectory(idx, last):
            nonlocal cum_w, cum_z
            instant = self.fused_probs(idx, last, exclude_id)
            if not self.layer_weights:
                return instant[0]
            z = self.probe_logits(idx, last)
            entry = self.memories.get(idx)
            if entry is not None and entry[1] > 0:
                z = np.log(np.clip(instant, 1e-12, None))
            weight = self.layer_weights.get(idx, 0.0)
            if weight > 1e-12:
                cum_w += weight
                cum_z = weight * z if cum_z is None else cum_z + weight * z
            return _softmax(cum_z / cum_w, axis=1)[0] if cum_w > 0 else instant[0]

        def policy(idx: int, last, delta: np.ndarray) -> bool:
            nonlocal effective_steps
            probs = trajectory(idx, last)
            record: dict[str, Any] = {"probs": probs}
            seen[idx] = record
            if self.layer_weights:
                if self.layer_weights.get(idx, 0.0) <= 1e-12:
                    return False
                effective_steps += 1
            if not prev:
                prev.update(probs=probs, drift=None)
                return False
            drift = float(np.max(np.abs(probs - prev["probs"])))
            gamma_hat = self.contraction(drift, prev["drift"])
            bound = self.tail_bound(drift, gamma_hat)
            top = int(np.argmax(probs))
            stable = top == int(np.argmax(prev["probs"]))
            prev.update(probs=probs, drift=drift)
            ordered = np.sort(probs)
            gap = float(ordered[-1] - ordered[-2])
            record.update(drift=drift, gamma_hat=gamma_hat, threshold=bound, margin=gap)
            if idx < self.min_exit_layer or not stable or gap <= bound or (self.layer_weights and effective_steps < 2):
                return False
            best, visit_margin, value = self.mcts_margin(probs)
            record.update(mcts_margin=visit_margin, value=value)
            return best == top and visit_margin > self.mcts_margin_min

        last_layer = self.num_layers - 1
        hook = QwenInterLayerHook(
            self.model, capture_layers=[last_layer], exit_policy=policy, exit_layers=self.checkpoints
        )
        stop = self._run(hook, input_ids)
        sparse = self._last_sparse_hook
        metrics = dict(ffn_channels_executed=sparse.ffn_channels_executed,
                       ffn_dense_channels=sparse.ffn_dense_channels,
                       ffn_macs_executed=sparse.ffn_macs_executed,
                       ffn_dense_macs=sparse.ffn_dense_macs,
                       active_hook_weights=dict(self.layer_weights))
        if stop is not None:
            r = seen[stop.layer_idx]
            return LayerExitDecision(int(np.argmax(r["probs"])), r["probs"], stop.layer_idx, True,
                                     hook.layers_executed, r["margin"], r["threshold"], r["value"],
                                     list(hook.layer_calls), r["mcts_margin"], r["drift"],
                                     r["gamma_hat"], **metrics)
        probs = trajectory(last_layer, hook.captured[last_layer])
        return LayerExitDecision(int(np.argmax(probs)), probs, last_layer, False,
                                 hook.layers_executed, layer_calls=list(hook.layer_calls), **metrics)

    # -- multi-candidate pruning ----------------------------------------------------------
    def score_candidates_mcts_pruned(
        self,
        candidates: Sequence[Any],
        target_label: int,
        shallow_layers: int,
        top_m: int,
    ) -> PrunedCandidateResult:
        """Rank ``candidates`` (each ``[1, T]`` ids) by the probability of ``target_label``.

        Every candidate runs layers ``0..shallow_layers-1`` only. MCTS over the shallow
        values keeps ``top_m`` by visit count; those resume from their cached residual through
        the remaining layers, the rest never run them.
        """
        k = len(candidates)
        if not 1 <= top_m <= k:
            raise ValueError("top_m must be in [1, len(candidates)]")
        if not 1 <= shallow_layers < self.num_layers:
            raise ValueError("shallow_layers must be in [1, num_layers)")
        if not 0 <= target_label < self.probe.num_labels:
            raise ValueError("target_label out of range")
        cut = shallow_layers - 1
        executed = 0
        cached, shallow = [], np.zeros(k)
        for i, ids in enumerate(candidates):
            if ids.shape[0] != 1:
                raise ValueError("each candidate is one row")
            stop, layers_run = self._shallow_pass(ids, cut)
            executed += layers_run
            cached.append(stop.hidden_state)
            shallow[i] = self.fused_probs(cut, stop.hidden_state[:, -1, :].float())[0, target_label]

        z = np.log(np.clip(shallow, 1e-12, None))
        z = z - z.mean()
        planner = LatentMctsPlanner(num_simulations=max(8 * k, self.planner.num_simulations),
                                    c_puct=self.planner.c_puct, temperature=1.0, max_depth=1)
        _, _, info = planner.plan(z, [str(i) for i in range(k)], simplex_etf(k))
        visits = info["root_visits"]
        order = np.lexsort((info["root_priors"], info["root_q"], visits))[::-1]
        survivors = [int(i) for i in order[:top_m]]

        last_layer = self.num_layers - 1
        deep: dict[int, float] = {}
        for i in survivors:
            final, layers_run = self._resume_pass(candidates[i], cached[i], shallow_layers)
            executed += layers_run
            deep[i] = float(self.fused_probs(last_layer, final)[0, target_label])
        best = max(deep, key=deep.get)
        return PrunedCandidateResult(best, survivors, shallow, deep, executed,
                                     k * self.num_layers, np.asarray(visits).copy())

    def _shallow_pass(self, input_ids, cut: int) -> tuple[_LayerEarlyExit, int]:
        """Run layers ``0..cut`` only; return the stop (with the residual) and layers run."""
        hook = QwenInterLayerHook(self.model, exit_policy=lambda *_: True, exit_layers=[cut])
        stop = self._run(hook, input_ids)
        if stop is None or stop.layer_idx != cut:
            raise RuntimeError("shallow pass did not stop at the cut layer")
        return stop, hook.layers_executed

    def _resume_pass(self, input_ids, cached, shallow_layers: int):
        """Finish the forward from a cached shallow residual: the first ``shallow_layers``
        layers are replaced by replay stand-ins, so only the deep layers execute. Returns the
        last-token state of the last layer and the number of real layers run."""
        replay = _replay_layer_cls()
        layers = self.backbone.layers
        originals = [layers[j] for j in range(shallow_layers)]
        last_layer = self.num_layers - 1
        hook = QwenInterLayerHook(self.model, capture_layers=[last_layer])
        try:
            for j in range(shallow_layers):
                layers[j] = replay(cached, originals[j])
            if self._run(hook, input_ids) is not None:
                raise RuntimeError("deep pass exited early")
        finally:
            for j, layer in enumerate(originals):
                layers[j] = layer
        return hook.captured[last_layer], hook.layers_executed
