"""Late causal consensus ensemble over independent RNN+Set adapters (pure NumPy, CPU).

Each teacher model (e.g. Qwen-72B, LLaMA-70B) keeps its own frozen hidden-state
space and its own independently trained RNNSetAdapterRuntime (own in_dim, own
adapter_dim, own weights). Nothing here shares a projection, concatenates
features, or trains a joint head across models: every model scores the same
K candidates in its own space, and only the *outputs* (logits / probabilities)
are combined. That is what "分开,不合在一起" (kept separate, not merged)
means at the code level -- see score_ensemble's per-model loop, which never
touches another model's arrays.

Fusion strategies, given per-model logits L_m over K candidates:

    logits_sum     L = sum_m L_m                        (raw logit addition)
    log_prob_sum   L = sum_m log_softmax(L_m)            (log-probability addition;
                                                           equal to logits_sum only
                                                           up to a per-model additive
                                                           constant, so it is reported
                                                           separately)
    consensus_veto If every model's argmax agrees, fuse by log_prob_sum (the
                   agreeing case has no ambiguity to arbitrate). If models
                   disagree, defer to the single model with the largest
                   normalized margin (top1-prob - top2-prob for that model) and
                   flag agreement=False, so a caller can see the tie was broken
                   by confidence rather than by summed evidence.

Every entry point is fail-closed on shape mismatch: a model whose own input
does not match its own in_dim raises immediately naming that model. There is
no fallback that silently drops a failed model from the ensemble.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .rnn_set_adapter import RNNSetAdapterRuntime

_STRATEGIES = ("logits_sum", "log_prob_sum", "consensus_veto")
_NEG_INF = np.float32(-np.inf)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _entropy_normalized(p: np.ndarray) -> float:
    """Shannon entropy of p, divided by log(K_valid) so it lands in [0, 1]."""
    valid = p > 0.0
    k = int(np.count_nonzero(valid))
    if k <= 1:
        return 0.0
    h = float(-np.sum(p[valid] * np.log(p[valid])))
    return h / float(np.log(k))


def _margin(sorted_desc: np.ndarray) -> float:
    if sorted_desc.shape[0] < 2:
        return float(sorted_desc[0])
    return float(sorted_desc[0] - sorted_desc[1])


def score_single(adapter: RNNSetAdapterRuntime, query: np.ndarray, candidates: np.ndarray,
                  mask: np.ndarray | None = None) -> Dict[str, object]:
    """Score one model's candidates in its own space; returns logits/probs/pred/confidence.

    ``mask`` (K,) bool marks which of the K rows of ``candidates`` are real;
    masked rows never win argmax and carry zero softmax mass, matching the
    padded-batch convention used elsewhere in this codebase (-1e9-scale
    logits at masked slots), but they are never fed through the adapter, so a
    masked row does not need to satisfy in_dim at all.
    """
    candidates = np.asarray(candidates)
    k_total = candidates.shape[0]
    if mask is None:
        mask = np.ones(k_total, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (k_total,):
            raise ValueError(f"mask must have shape ({k_total},), got {mask.shape}")
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("score_single: mask selects zero candidates")

    raw = adapter.score(query, candidates[valid])
    logits = np.full(k_total, _NEG_INF, dtype=np.float32)
    logits[valid] = raw

    log_probs = np.full(k_total, _NEG_INF, dtype=np.float32)
    log_probs[valid] = _log_softmax(raw)
    probs = np.zeros(k_total, dtype=np.float32)
    probs[valid] = np.exp(log_probs[valid])

    pred = int(valid[np.argmax(raw)])
    order = np.argsort(raw)[::-1]
    return {
        "logits": logits,
        "log_probs": log_probs,
        "probs": probs,
        "pred": pred,
        "margin": _margin(raw[order]),
        "entropy": _entropy_normalized(probs),
        "mask": mask,
    }


def _fuse_logits_sum(per_model: Mapping[str, Dict[str, object]]) -> np.ndarray:
    stacked = np.stack([r["logits"] for r in per_model.values()], axis=0)
    return np.sum(stacked, axis=0)


def _fuse_log_prob_sum(per_model: Mapping[str, Dict[str, object]]) -> np.ndarray:
    stacked = np.stack([r["log_probs"] for r in per_model.values()], axis=0)
    return np.sum(stacked, axis=0)


def score_ensemble(adapters: Mapping[str, RNNSetAdapterRuntime],
                    inputs: Mapping[str, Mapping[str, np.ndarray]],
                    fusion_strategy: str = "logits_sum") -> Dict[str, object]:
    """Late-fuse K-candidate scores from independently-run adapters.

    ``adapters`` maps model name -> RNNSetAdapterRuntime (own dims, own weights).
    ``inputs`` maps the *same* model names -> {"query": (D_m,), "candidates":
    (K, D_m), "mask": optional (K,)}. K (the number of candidate slots) must
    match across models -- they are scoring the same K candidate texts, just
    embedded independently in each model's own space. Each model's own
    (query, candidates) shape is checked only against that model's own
    in_dim: there is no cross-model dimension check, because the spaces are
    never combined before the softmax.
    """
    if fusion_strategy not in _STRATEGIES:
        raise ValueError(f"fusion_strategy must be one of {_STRATEGIES}, got {fusion_strategy!r}")
    if set(adapters) != set(inputs):
        raise ValueError(f"adapters and inputs must name the same models: "
                          f"{sorted(adapters)} vs {sorted(inputs)}")
    if not adapters:
        raise ValueError("score_ensemble: at least one model is required")

    per_model: Dict[str, Dict[str, object]] = {}
    k_ref: int | None = None
    for name, adapter in adapters.items():
        rec = inputs[name]
        try:
            result = score_single(adapter, rec["query"], rec["candidates"], rec.get("mask"))
        except ValueError as exc:
            raise ValueError(f"model {name!r}: {exc}") from exc
        k_total = result["logits"].shape[0]
        if k_ref is None:
            k_ref = k_total
        elif k_total != k_ref:
            raise ValueError(f"model {name!r}: has {k_total} candidate slots, "
                              f"expected {k_ref} (all models must score the same K)")
        per_model[name] = result

    preds = {name: r["pred"] for name, r in per_model.items()}
    agreement = len(set(preds.values())) == 1

    pairwise = None
    if len(per_model) > 2:
        names = list(per_model)
        pairwise = {a: {b: bool(preds[a] == preds[b]) for b in names} for a in names}

    if fusion_strategy == "logits_sum":
        fused_logits = _fuse_logits_sum(per_model)
        fused_pred = int(np.argmax(fused_logits))
    elif fusion_strategy == "log_prob_sum":
        fused_logits = _fuse_log_prob_sum(per_model)
        fused_pred = int(np.argmax(fused_logits))
    else:  # consensus_veto
        if agreement:
            fused_logits = _fuse_log_prob_sum(per_model)
            fused_pred = int(np.argmax(fused_logits))
        else:
            best_name = max(per_model, key=lambda n: _prob_margin(per_model[n]["probs"]))
            fused_logits = per_model[best_name]["logits"].copy()
            fused_pred = int(preds[best_name])

    return {
        "per_model": per_model,
        "preds": preds,
        "agreement": bool(agreement),
        "pairwise_agreement": pairwise,
        "fused_logits": fused_logits,
        "pred": fused_pred,
        "fusion_strategy": fusion_strategy,
    }


def _prob_margin(probs: np.ndarray) -> float:
    order = np.sort(probs)[::-1]
    if order.shape[0] < 2:
        return float(order[0])
    return float(order[0] - order[1])


def load_adapters(paths: Mapping[str, object]) -> Dict[str, RNNSetAdapterRuntime]:
    """Load one or more independent adapter checkpoints by model name.

    Each path is loaded through RNNSetAdapterRuntime.from_npz, so every
    adapter goes through the same shape/stability checks as a solo model
    (spectral clamp re-derived by SVD, non-finite rejection). Failing to load
    one model raises immediately naming that model; there is no partial
    ensemble.
    """
    out: Dict[str, RNNSetAdapterRuntime] = {}
    for name, path in paths.items():
        try:
            out[name] = RNNSetAdapterRuntime.from_npz(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"model {name!r}: failed to load {path}: {exc}") from exc
    return out
