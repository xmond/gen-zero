"""Spec 19 enhanced heads for benchmark_sota_ensemble.py (Phase 1: deep residual adapter,
Phase 2 direction 4: SupCon joint training head, Phase 3: RDA / Nystrom), plus the Spec 20 P3
Formulation B folded residual adapter (FoldedResidualAdapterBHead, "adapter_b": frozen linear
baseline f0 + C GELU(Ux + a), alternating proximal optimization with a monotone objective trace) and
the Spec 20 P4 dual-manifold head (DualManifoldHead, "dual_manifold": Qwen principal subspace + Gemma
orthogonal innovation projection, S3.3, algebraically folded into two GEMVs for CPU, S6.2).

DeepResidualAdapterHead, Spec 19 S2 / S6.1. On the standardized full-context vector z
(the exact `lin_full` feature map of sota_ensemble_experts.LinearProbe):

    h = z + W_up GELU(W_down z + b_down) + b_up          W_down (r, D), W_up (D, r)
    s = W_h h + b_h                                       W_h (K, D)

W_up and b_up start at zero, so at construction s is the linear probe's logits exactly.
Because the head is linear, the trained model folds (S2.3) into

    s = W_h z + W_fold GELU(W_down z + b_down) + b_fold,  W_fold = W_h W_up, b_fold = W_h b_up + b_h

which is what save() exports and scores() runs: NumPy only, torch is never imported at
inference. The device is a fit-time knob only: the weights are copied back to host NumPy
before export, and the saved .npz does not record it. GELU is the tanh form on BOTH sides (torch approximate="tanh"); the erf form
differs by ~1e-3 and would break the fold identity.

Training (fit): fp32 on `device` ("cpu", "cuda" or "auto"; sx.resolve_device), the train_rnn recipe of sota_ensemble_experts (15% early-stop
slice drawn first, AdamW lr 1e-3, wd 0.01 on ndim >= 2 / 0 on biases, batch 32, clip 1.0,
patience 10, max 80 epochs), plus lambda_up * ||W_up||_F^2 in the loss (shrink toward the
identity map) and an optional spectral cap on ||W_up||_2 after every step.

Two decisions that keep the comparison with lin_full fair:
  * The warm-start LinearProbe is fitted on the 85% training slice only, never on the
    early-stop rows. Otherwise its early-stop accuracy is in-sample, no epoch can beat it,
    and the adapter silently collapses to the probe on every task.
  * The init state (the probe itself) is scored as "epoch 0" and competes for best_state.
    If no epoch beats it on the held-out slice, the head returned IS the linear probe
    (W_fold = 0), which is S2.2(c)'s degenerate direction.

On pair tasks the adapter reads only the full-context third of the source vector, as
lin_full does.

SupConHead (Spec 19 S5.3) is the same network and the same export. Only the training loss
differs:  L = CE(s, y) + lambda_supcon * SupCon(u, y, tau),  u = h / ||h||_2.  Two feature-level
dropout views of each record go through the residual projection; same-label views are
positives, all other views negatives. CE reads the clean (undropped) view, so the head is
trained on the distribution it sees at inference. A batch with no class holding two distinct
records has no cross-record positive: the SupCon term is skipped for it (CE only, no error)
and counted in info["supcon_degraded_batches"]. The saved .npz has the exact layout of the
adapter (W_fold = W_h W_up), so inference is the same NumPy-only fold.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np

import sota_ensemble_experts as sx

GELU_C = math.sqrt(2.0 / math.pi)


def gelu_tanh(z: np.ndarray) -> np.ndarray:
    """The tanh GELU, identical in form to torch.nn.functional.gelu(approximate="tanh")."""
    return 0.5 * z * (1.0 + np.tanh(GELU_C * (z + 0.044715 * (z ** 3))))


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 64                       # S2.5: nested CV over {32, 64, 128}
    lambda_up: float = 1e-3              # shrink W_up toward 0, i.e. the adapter toward identity
    spectral_cap: Optional[float] = None  # if set, ||W_up||_2 <= cap after each step
    warm_start_linear: bool = True       # init W_h, b_h from LinearProbe.fit on the same rows
    fold_for_inference: bool = True      # export W_h @ W_up, never the unfolded pair
    device: str = "auto"                 # fit-time only: "cpu", "cuda" (raises if absent) or "auto"
    lr: float = 1e-3                     # AdamW learning rate (tune_spec19_adapter_a100.py per-task sweep)

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive int, got {self.rank!r}")
        if not math.isfinite(self.lambda_up) or self.lambda_up < 0.0:
            raise ValueError(f"lambda_up must be finite and >= 0, got {self.lambda_up!r}")
        if self.spectral_cap is not None and (not math.isfinite(self.spectral_cap) or self.spectral_cap <= 0.0):
            raise ValueError(f"spectral_cap must be positive and finite, got {self.spectral_cap!r}")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        sx.check_device_name(self.device)


class DeepResidualAdapterHead:
    """FittableHead / ExpertCore of Spec 19 S6.1. Inference is NumPy only."""

    head_type: ClassVar[str] = "adapter"
    CONFIG_CLS: ClassVar[type] = AdapterConfig
    FOLDED_KEYS = ("mu_full", "sd_full", "W_h", "W_fold", "W_down", "b_down", "b_fold")
    UNFOLDED_KEYS = ("mu_full", "sd_full", "W_h", "b_h", "W_up", "b_up", "W_down", "b_down")

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.cfg = dict(cfg)
        if self.cfg.get("head_type", self.head_type) != self.head_type:
            raise ValueError(f"{type(self).__name__} cannot load a {self.cfg['head_type']!r} head")
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim, self.D, self.rank = int(cfg["in_dim"]), int(cfg["D"]), int(cfg["rank"])
        self.folded = bool(cfg["folded"])
        need = self.FOLDED_KEYS if self.folded else self.UNFOLDED_KEYS
        missing = [k for k in need if k not in arrays]
        if missing:
            raise ValueError(f"adapter arrays missing {missing}")
        self.a = {k: np.ascontiguousarray(arrays[k], dtype=np.float32) for k in need}
        D, r, K = self.D, self.rank, self.K
        shapes = {"mu_full": (D,), "sd_full": (D,), "W_h": (K, D), "W_fold": (K, r), "b_fold": (K,),
                  "b_h": (K,), "W_up": (D, r), "b_up": (D,), "W_down": (r, D), "b_down": (r,)}
        for k, v in self.a.items():
            if v.shape != shapes[k]:
                raise ValueError(f"adapter array {k} has shape {v.shape}, expected {shapes[k]}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"adapter array {k} holds non-finite values")
        if self.in_dim != (3 * D if self.pair else D):
            raise ValueError(f"in_dim {self.in_dim} does not match D={D}, pair={self.pair}")
        self.info: dict = {}

    # ------------------------------------------------------------ construction

    @classmethod
    def from_unfolded(cls, stats: Dict[str, np.ndarray], W_down, b_down, W_up, b_up, W_h, b_h, *,
                      K: int, pair: bool, in_dim: int, config: AdapterConfig) -> "DeepResidualAdapterHead":
        """Export trained (or initial) parameters; the fold is done in float64, then cast."""
        f64 = lambda v: np.asarray(v, dtype=np.float64)  # noqa: E731
        W_down, b_down, W_up, b_up, W_h, b_h = map(f64, (W_down, b_down, W_up, b_up, W_h, b_h))
        D, r = W_down.shape[1], W_down.shape[0]
        cfg = {"K": int(K), "pair": bool(pair), "in_dim": int(in_dim), "D": int(D), "rank": int(r),
               "folded": bool(config.fold_for_inference), "head_type": cls.head_type,
               "gelu": "tanh",
               # device is a fit-time knob, not a model property: leaving it out keeps the saved cfg_json as before
               "config": {k: v for k, v in asdict(config).items() if k != "device"}}
        arrays = {"mu_full": stats["mu_full"], "sd_full": stats["sd_full"], "W_h": W_h,
                  "W_down": W_down, "b_down": b_down}
        if config.fold_for_inference:
            arrays.update(W_fold=W_h @ W_up, b_fold=W_h @ b_up + b_h)
        else:
            arrays.update(W_up=W_up, b_up=b_up, b_h=b_h)
        return cls(arrays, cfg)

    @classmethod
    def from_linear_probe(cls, lp: "sx.LinearProbe", config: AdapterConfig,
                          rng: np.random.Generator) -> "DeepResidualAdapterHead":
        """The untrained adapter on top of a fitted `lin_full` probe: W_up = 0, b_up = 0.

        W_down ~ U(-1/sqrt(D), 1/sqrt(D)) (nn.Linear's bound), b_down = 0. Its logits equal
        lp.scores() exactly, whatever W_down is.
        """
        if lp.cfg.get("kind") != "full" or lp.cfg.get("pca_k"):
            raise ValueError("the adapter warm-starts only from a raw 'full' LinearProbe (no PCA)")
        W_h, b_h = lp.a["W"], lp.a["b"]
        K, D = W_h.shape
        r = config.rank
        bound = 1.0 / math.sqrt(D)
        W_down = rng.uniform(-bound, bound, size=(r, D))
        return cls.from_unfolded({"mu_full": lp.a["mu_full"], "sd_full": lp.a["sd_full"]},
                                 W_down, np.zeros(r), np.zeros((D, r)), np.zeros(D), W_h, b_h,
                                 K=K, pair=lp.pair, in_dim=lp.in_dim, config=config)

    # -------------------------------------------------------------- inference

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (1, 2) or X.shape[-1] != self.in_dim:
            raise ValueError(f"adapter expects (..., {self.in_dim}) input, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("adapter input holds non-finite values (NaN or Inf)")
        full, _, _ = sx.split_source(X, self.pair)
        return (full - self.a["mu_full"]) / self.a["sd_full"]

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, in_dim) -> (N, K) float32 logits, or (in_dim,) -> (K,). NumPy only."""
        z = self._standardize(X)
        a = self.a
        act = gelu_tanh(z @ a["W_down"].T + a["b_down"])
        if self.folded:
            out = z @ a["W_h"].T + act @ a["W_fold"].T + a["b_fold"]
        else:
            h = z + act @ a["W_up"].T + a["b_up"]
            out = h @ a["W_h"].T + a["b_h"]
        return out.astype(np.float32)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"adapter trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores(x)

    def supervised_param_count(self) -> int:
        """Spec 19 S2.1: 2Dr + D + r (adapter) + K(D + 1) (head)."""
        D, r, K = self.D, self.rank, self.K
        return 2 * D * r + D + r + K * (D + 1)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a)

    @classmethod
    def load(cls, path: Path) -> "DeepResidualAdapterHead":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json"}
            return cls(arrays, json.loads(str(z["cfg_json"])))

    # ---------------------------------------------------------------- training

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, K: int, *, pair: bool, seed: int,
            config: Optional[AdapterConfig] = None, rng_key: Optional[Sequence[int]] = None,
            probe_fit=None, device: Optional[str] = None) -> "DeepResidualAdapterHead":
        """Train on the rows given, and only those.

        device overrides config.device ("cpu" | "cuda" | "auto"); both the warm-start probe and the
        adapter training run there, and the head that comes back is plain NumPy either way.

        rng_key seeds the early-stop split, W_down and the batch order (default [TRAIN_SEED, seed]).
        probe_fit(X, y) -> LinearProbe overrides the warm-start probe fit (default:
        LinearProbe.fit kind="full", no PCA, inner-CV C); it only ever sees the 85% slice.
        """
        cfg = config or cls.CONFIG_CLS()
        if type(cfg) is not cls.CONFIG_CLS:
            raise TypeError(f"{cls.__name__}.fit needs a {cls.CONFIG_CLS.__name__}, got {type(cfg).__name__}")
        dev = sx.resolve_device(cfg.device if device is None else device)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be (N, in_dim) with N == len(y), got {X.shape} and {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("adapter training rows hold non-finite values (NaN or Inf)")
        if not np.issubdtype(y.dtype, np.integer) or len(y) == 0 or y.min() < 0 or y.max() >= K:
            raise ValueError(f"labels must be integers in [0, {K}), got {y.dtype} range "
                             f"[{y.min() if len(y) else None}, {y.max() if len(y) else None}]")
        rng = np.random.default_rng(list(rng_key) if rng_key is not None else [sx.TRAIN_SEED, int(seed)])
        n = len(y)
        es = rng.random(n) < sx.EARLY_STOP_FRACTION
        tr_idx, es_idx = np.flatnonzero(~es), np.flatnonzero(es)
        if len(es_idx) == 0 or len(np.unique(y[tr_idx])) < 2:
            raise ValueError(f"too few rows to train the adapter: {len(tr_idx)} train / {len(es_idx)} early-stop")
        if probe_fit is None:
            probe_fit = lambda Xp, yp: sx.LinearProbe.fit(Xp, yp, K, pair=pair, kind="full",  # noqa: E731
                                                          pca_k=None, seed=int(seed), device=dev)
        lp = probe_fit(X[tr_idx], y[tr_idx])
        if not cfg.warm_start_linear:
            # Keep the probe's standardization (unsupervised) but start the head from zero,
            # except the absent-class bias, so a class missing from the rows stays unreachable.
            absent = np.setdiff1d(np.arange(K), np.unique(y[tr_idx]))
            b0 = np.zeros(K, dtype=np.float32)
            b0[absent] = lp.a["b"][absent]
            lp = sx.LinearProbe(dict(lp.a, W=np.zeros_like(lp.a["W"]), b=b0), lp.cfg)
        init = cls.from_linear_probe(lp, replace(cfg, fold_for_inference=False), rng)
        t0 = time.perf_counter()
        extra = cls._extra_objective(cfg, rng, dev)
        head, info = _train(init, lp, X, y, tr_idx, es_idx, cfg, rng, dev, extra)
        info.update(device=dev, lr=cfg.lr, train_seconds=time.perf_counter() - t0, n_train=int(len(tr_idx)),
                    n_early_stop=int(len(es_idx)), probe_C=lp.cfg.get("C"),
                    params=head.supervised_param_count())
        head.info = info
        return head

    @classmethod
    def _extra_objective(cls, cfg: AdapterConfig, rng: np.random.Generator, dev: str):
        """Loss term added to the CE inside _train; None = plain adapter. Called after the
        split and probe draws, so a subclass may draw from rng without moving them."""
        return None


def torch_embedding(net, z):
    """h = z + W_up GELU(W_down z + b_down) + b_up, the residual projection."""
    import torch.nn.functional as F
    return z + F.gelu(z @ net["W_down"].T + net["b_down"], approximate="tanh") @ net["W_up"].T + net["b_up"]


def torch_unfolded_logits(net, z):
    """The training-time forward: h = torch_embedding(z); s = W_h h + b_h."""
    return torch_embedding(net, z) @ net["W_h"].T + net["b_h"]


def _train(init: DeepResidualAdapterHead, lp, X, y, tr_idx, es_idx, cfg: AdapterConfig,
           rng: np.random.Generator, dev: str = "cpu", extra=None):
    """Train on the concrete torch device `dev` (already resolved); return a NumPy-only head.

    extra(net, z_batch, y_batch) -> scalar tensor is added to the CE (SupCon); it may expose info()."""
    import torch
    import torch.nn.functional as F
    a = init.a
    net = {k: torch.nn.Parameter(torch.as_tensor(np.array(a[k], dtype=np.float32), device=dev))
           for k in ("W_down", "b_down", "W_up", "b_up", "W_h", "b_h")}
    params = list(net.values())
    z_all = torch.as_tensor(init._standardize(X), device=dev)   # the probe's own standardization
    y_t = torch.as_tensor(y.astype(np.int64), device=dev)
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": sx.WD},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=cfg.lr)
    es_b = torch.as_tensor(es_idx, device=dev)

    def es_key():
        with torch.no_grad():
            lg = torch_unfolded_logits(net, z_all[es_b])
            return (float((lg.argmax(-1) == y_t[es_b]).float().mean()), float(F.cross_entropy(lg, y_t[es_b])))

    def snapshot():
        return {k: v.detach().to("cpu", copy=True) for k, v in net.items()}   # host copy, off the GPU

    # Epoch 0 is the warm start itself (the linear probe): training must beat it on held-out rows.
    identity_err = float(np.max(np.abs(init.scores(X[es_idx]) - lp.scores(X[es_idx]))))
    best, best_state, best_epoch, stale = es_key(), snapshot(), 0, 0
    init_key = best
    epochs_run = 0
    for epoch in range(1, sx.EPOCHS + 1):
        epochs_run = epoch
        perm = torch.as_tensor(rng.permutation(tr_idx), device=dev)
        for s in range(0, len(perm), sx.BATCH_TRAIN):
            b = perm[s:s + sx.BATCH_TRAIN]
            loss = F.cross_entropy(torch_unfolded_logits(net, z_all[b]), y_t[b])
            if extra is not None:
                loss = loss + extra(net, z_all[b], y_t[b])
            if cfg.lambda_up > 0.0:
                loss = loss + cfg.lambda_up * net["W_up"].pow(2).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"adapter: non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if cfg.spectral_cap is not None:
                with torch.no_grad():
                    sn = float(torch.linalg.matrix_norm(net["W_up"], ord=2))
                    if sn > cfg.spectral_cap:
                        net["W_up"].mul_(cfg.spectral_cap / sn)
        key = es_key()
        if (key[0], -key[1]) > (best[0], -best[1]):
            best, best_state, best_epoch, stale = key, snapshot(), epoch, 0
        else:
            stale += 1
            if stale >= sx.PATIENCE:
                break
    st = {k: v.detach().cpu().numpy() for k, v in best_state.items()}
    stats = {"mu_full": init.a["mu_full"], "sd_full": init.a["sd_full"]}
    head = type(init).from_unfolded(stats, st["W_down"], st["b_down"], st["W_up"], st["b_up"],
                                                 st["W_h"], st["b_h"], K=init.K, pair=init.pair,
                                                 in_dim=init.in_dim, config=cfg)
    # Self-check of the export: the NumPy head must reproduce the torch forward it came from.
    with torch.no_grad():
        ref = torch_unfolded_logits(best_state, z_all[es_b].cpu()).numpy()   # host forward of the host snapshot
    got = head.scores(X[es_idx])
    fold_err = float(np.max(np.abs(got - ref)))
    if not fold_err <= 1e-4 * (1.0 + float(np.max(np.abs(ref)))):
        raise FloatingPointError(f"adapter export does not match its torch forward: max abs err {fold_err}")
    extra_info = extra.info() if hasattr(extra, "info") else {}
    return head, {**extra_info, "best_epoch": best_epoch, "epochs_run": epochs_run, "early_stop_acc": best[0],
                  "early_stop_ce": best[1], "early_stop_acc_at_init": init_key[0],
                  "early_stop_ce_at_init": init_key[1], "identity_max_abs_err_at_init": identity_err,
                  "export_max_abs_err_es": fold_err,
                  "w_up_fro": float(np.linalg.norm(st["W_up"])),
                  "w_up_spectral": float(np.linalg.norm(st["W_up"], 2)) if st["W_up"].size else 0.0}


# ------------------------------------------------- Phase 2 direction 4: SupCon head

@dataclass(frozen=True)
class SupConConfig(AdapterConfig):
    """AdapterConfig plus the contrastive knobs (it keeps spectral_cap and warm_start_linear)."""
    tau: float = 0.1                     # SupCon temperature
    lambda_supcon: float = 0.5           # weight of the contrastive term next to the CE
    feature_dropout: float = 0.1         # drop rate of the two augmented views (feature level)

    def __post_init__(self) -> None:
        for name in ("rank", "lr", "lambda_up", "tau", "lambda_supcon", "feature_dropout"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a real number, got {v!r}")
        if not isinstance(self.fold_for_inference, bool):
            raise ValueError(f"fold_for_inference must be a bool, got {self.fold_for_inference!r}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}")
        super().__post_init__()
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(f"tau must be finite and > 0, got {self.tau!r}")
        if not math.isfinite(self.lambda_supcon) or self.lambda_supcon < 0.0:
            raise ValueError(f"lambda_supcon must be finite and >= 0, got {self.lambda_supcon!r}")
        if not math.isfinite(self.feature_dropout) or not 0.0 <= self.feature_dropout < 1.0:
            raise ValueError(f"feature_dropout must be in [0, 1), got {self.feature_dropout!r}")


def supcon_loss(u1, u2, y, tau: float):
    """Supervised contrastive loss (Khosla et al. 2020) over two views; None if it cannot apply.

    u1, u2: (B, d) unit vectors, the two views of the same B records; y: (B,) labels. Every
    view is an anchor; its positives are all other views with its label (its own second view
    included), its negatives every other view. Anchors whose class has no second record in the
    batch would only do instance discrimination, so they are skipped; with no anchor left the
    result is None (the caller falls back to CE alone).
    """
    import torch
    B = y.shape[0]
    u = torch.cat([u1, u2])
    lab = torch.cat([y, y])
    rec = torch.arange(B, device=y.device).repeat(2)
    eye = torch.eye(2 * B, dtype=torch.bool, device=y.device)
    same = lab[:, None] == lab[None, :]
    pos = same & ~eye
    valid = (same & (rec[:, None] != rec[None, :])).any(dim=1)
    if not bool(valid.any()):
        return None
    sim = (u @ u.T / tau).masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(log_prob * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
    return per_anchor[valid].mean()


class _SupConObjective:
    """lambda_supcon * SupCon on two dropout views of the batch, with degradation bookkeeping."""

    def __init__(self, cfg: SupConConfig, seed: int, dev: str) -> None:
        import torch
        self.cfg, self.dev = cfg, dev
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed))
        self.batches = self.degraded = 0
        self.loss_sum = 0.0

    def _view(self, z):
        import torch
        p = self.cfg.feature_dropout
        if p == 0.0:
            return z
        keep = torch.rand(z.shape, generator=self.gen, device=self.dev) >= p
        return z * keep.to(z.dtype) / (1.0 - p)

    def __call__(self, net, z, y):
        import torch
        import torch.nn.functional as F
        self.batches += 1
        sup = None
        if self.cfg.lambda_supcon > 0.0:
            u1 = F.normalize(torch_embedding(net, self._view(z)), dim=-1)
            u2 = F.normalize(torch_embedding(net, self._view(z)), dim=-1)
            sup = supcon_loss(u1, u2, y, self.cfg.tau)
        if sup is None:
            self.degraded += 1
            return z.new_zeros(())
        self.loss_sum += float(sup.detach())
        return self.cfg.lambda_supcon * sup

    def info(self) -> dict:
        used = self.batches - self.degraded
        return {"supcon_batches": self.batches, "supcon_degraded_batches": self.degraded,
                "supcon_mean_loss": self.loss_sum / used if used else None}


class SupConHead(DeepResidualAdapterHead):
    """Spec 19 S5.3: the residual adapter trained with CE + lambda_supcon * SupCon; same fold and export."""

    head_type: ClassVar[str] = "supcon"
    CONFIG_CLS: ClassVar[type] = SupConConfig

    @classmethod
    def _extra_objective(cls, cfg: SupConConfig, rng: np.random.Generator, dev: str):
        return _SupConObjective(cfg, int(rng.integers(0, 2 ** 31 - 1)), dev)


# ------------------------------------------ Phase 3 direction 3: RDA + Nystrom (Spec 19 S4.2/4.3)

def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance via the Gram identity (no (N, M, D) intermediate).

    (D,), (M, D) -> (M,); (N, D), (M, D) -> (N, M)."""
    a2 = np.sum(a * a, axis=-1)
    b2 = np.sum(b * b, axis=-1)
    ab = a @ b.T
    d2 = (a2 + b2 - 2.0 * ab) if a.ndim == 1 else (a2[:, None] + b2[None, :] - 2.0 * ab)
    return np.clip(d2, 0.0, None)          # roundoff can make a==b slightly negative


@dataclass(frozen=True)
class RDAConfig:
    pca_k: int = 128
    beta_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)  # 0.0 = LDA = linear head
    device: str = "cpu"                  # RDAHead is NumPy/sklearn only, no other backend exists

    def __post_init__(self) -> None:
        if not isinstance(self.pca_k, int) or isinstance(self.pca_k, bool) or self.pca_k < 1:
            raise ValueError(f"pca_k must be a positive int, got {self.pca_k!r}")
        if not self.beta_grid:
            raise ValueError("beta_grid must not be empty")
        for b in self.beta_grid:
            if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0.0 <= b <= 1.0:
                raise ValueError(f"beta_grid values must be finite in [0, 1], got {b!r}")
        if self.device != "cpu":
            raise ValueError(f"RDAHead has no backend but CPU NumPy/sklearn, got device={self.device!r}")


class RDAHead:
    """FittableHead / ExpertCore of Spec 19 S4.2 / S6.1.

    Friedman (1989) regularized discriminant analysis on a whitened PCA subspace, with
    Ledoit & Wolf (2004) closed-form covariance shrinkage:

      1. Standardize the full-context vector, then PCA-whiten it to k = min(pca_k, n_fit - 1)
         dims (the same whitened-PCA recipe as LinearProbe's pca_k path).
      2. Per class c: mu_c = mean, S_c = Ledoit-Wolf-shrunk empirical covariance toward its own
         scaled identity (`sklearn.covariance.ledoit_wolf`, exactly the closed-form Ledoit & Wolf
         derive). S_pooled = the same shrinkage applied to all classes' centered residuals pooled
         into one sample.
      3. Sigma_c(beta) = beta * S_c + (1 - beta) * S_pooled; beta is chosen from `beta_grid` by
         accuracy on an inner StratifiedKFold split of the fit rows (first max wins ties, and the
         grid's ascending order puts the least complex beta=0 first). beta=0 makes every class
         share S_pooled, so the quadratic term x^T Sigma^-1 x is identical across classes and
         cancels in every pairwise comparison: the decision surface is then exactly the one of a
         shared-covariance linear discriminant (LDA), not merely close to it.
      4. g_c(x) = -0.5 (x - mu_c)^T Sigma_c^-1 (x - mu_c) - 0.5 log|Sigma_c| + log pi_c, pi_c the
         empirical class frequency on the fit rows. A class absent from the fit rows gets
         Sigma_c = S_pooled and log pi_c = sx.NEG (-1e4), far below any reachable score.

    Inference is NumPy only: mu_c, the precomputed precision matrices Sigma_c^-1, log|Sigma_c|
    and log pi_c are exported to the .npz, never Sigma_c itself.
    """

    head_type: ClassVar[str] = "rda"
    CONFIG_CLS: ClassVar[type] = RDAConfig
    ARRAY_KEYS = ("mu_full", "sd_full", "pca_mu", "pca_P", "mu_c", "prec_c", "logdet_c", "log_prior")

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.cfg = dict(cfg)
        if self.cfg.get("head_type", self.head_type) != self.head_type:
            raise ValueError(f"{type(self).__name__} cannot load a {self.cfg['head_type']!r} head")
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim, self.D, self.k = int(cfg["in_dim"]), int(cfg["D"]), int(cfg["k"])
        missing = [key for key in self.ARRAY_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"rda arrays missing {missing}")
        self.a = {key: np.ascontiguousarray(arrays[key], dtype=np.float32) for key in self.ARRAY_KEYS}
        D, k, K = self.D, self.k, self.K
        shapes = {"mu_full": (D,), "sd_full": (D,), "pca_mu": (D,), "pca_P": (D, k), "mu_c": (K, k),
                  "prec_c": (K, k, k), "logdet_c": (K,), "log_prior": (K,)}
        for key, v in self.a.items():
            if v.shape != shapes[key]:
                raise ValueError(f"rda array {key} has shape {v.shape}, expected {shapes[key]}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"rda array {key} holds non-finite values")
        if self.in_dim != (3 * D if self.pair else D):
            raise ValueError(f"in_dim {self.in_dim} does not match D={D}, pair={self.pair}")
        self.info: dict = {}

    # -------------------------------------------------------------- inference

    def _project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (1, 2) or X.shape[-1] != self.in_dim:
            raise ValueError(f"rda expects (..., {self.in_dim}) input, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("rda input holds non-finite values (NaN or Inf)")
        full, _, _ = sx.split_source(X, self.pair)
        z = (full - self.a["mu_full"]) / self.a["sd_full"]
        return (z - self.a["pca_mu"]) @ self.a["pca_P"]

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, in_dim) -> (N, K) float32 logits, or (in_dim,) -> (K,). NumPy only."""
        z = self._project(X).astype(np.float64)
        mu_c, prec_c = self.a["mu_c"].astype(np.float64), self.a["prec_c"].astype(np.float64)
        if z.ndim == 1:
            diff = mu_c - z[None, :]
            quad = np.einsum("ki,kij,kj->k", diff, prec_c, diff)
        else:
            diff = mu_c[None, :, :] - z[:, None, :]
            quad = np.einsum("nki,kij,nkj->nk", diff, prec_c, diff)
        out = -0.5 * quad - 0.5 * self.a["logdet_c"].astype(np.float64) + self.a["log_prior"].astype(np.float64)
        return out.astype(np.float32)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"rda trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores(x)

    def supervised_param_count(self) -> int:
        """Spec 19 S4.2: K * (k + k(k+1)/2), mean plus the free entries of one k x k covariance."""
        k, K = self.k, self.K
        return K * (k + k * (k + 1) // 2)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a)

    @classmethod
    def load(cls, path: Path) -> "RDAHead":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json"}
            return cls(arrays, json.loads(str(z["cfg_json"])))

    # ---------------------------------------------------------------- training

    @staticmethod
    def _fit_discriminant(z: np.ndarray, y: np.ndarray, K: int, beta: float):
        """mu_c, prec_c, logdet_c, log_prior for every c in [0, K), on rows (z, y)."""
        from sklearn.covariance import ledoit_wolf
        n, k = z.shape
        classes_present = np.unique(y)
        resid = np.empty_like(z)
        for c in classes_present:
            idx = y == c
            resid[idx] = z[idx] - z[idx].mean(0)
        cov_pool, _ = ledoit_wolf(resid, assume_centered=True)
        mu_c = np.zeros((K, k), dtype=np.float64)
        prec_c = np.zeros((K, k, k), dtype=np.float64)
        logdet_c = np.zeros(K, dtype=np.float64)
        log_prior = np.zeros(K, dtype=np.float64)
        eye = np.eye(k)
        for c in range(K):
            idx = y == c
            nc = int(idx.sum())
            if nc == 0:
                sigma, log_prior[c] = cov_pool, float(sx.NEG)
            else:
                mu_c[c] = z[idx].mean(0)
                log_prior[c] = math.log(nc / n)
                cov_c = ledoit_wolf(z[idx] - mu_c[c], assume_centered=True)[0] if nc >= 2 else cov_pool
                sigma = beta * cov_c + (1.0 - beta) * cov_pool
            sigma = sigma + (1e-9 * np.trace(sigma) / k + 1e-12) * eye     # numerical floor, keeps it invertible
            prec_c[c] = np.linalg.inv(sigma)
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                raise FloatingPointError(f"rda: class {c} covariance is not positive definite (beta={beta})")
            logdet_c[c] = logdet
        return mu_c, prec_c, logdet_c, log_prior

    @staticmethod
    def _predict_from_params(z: np.ndarray, mu_c, prec_c, logdet_c, log_prior) -> np.ndarray:
        diff = mu_c[None, :, :] - z[:, None, :]
        quad = np.einsum("nki,kij,nkj->nk", diff, prec_c, diff)
        g = -0.5 * quad - 0.5 * logdet_c[None, :] + log_prior[None, :]
        return g.argmax(1)

    @staticmethod
    def _select_beta(z: np.ndarray, y: np.ndarray, K: int, cfg: RDAConfig, seed: int) -> float:
        grid = cfg.beta_grid
        if len(grid) == 1:
            return float(grid[0])
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            return float(grid[0])          # too few rows for an inner split: fall back to the simplest beta
        from sklearn.model_selection import StratifiedKFold
        n_splits = int(max(2, min(4, counts.min())))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        mean_acc = []
        for beta in grid:
            accs = []
            for tr_i, va_i in cv.split(z, y):
                if len(np.unique(y[tr_i])) < 2:
                    continue
                params = RDAHead._fit_discriminant(z[tr_i], y[tr_i], K, beta)
                pred = RDAHead._predict_from_params(z[va_i], *params)
                accs.append(float(np.mean(pred == y[va_i])))
            mean_acc.append(float(np.mean(accs)) if accs else -math.inf)
        return float(grid[int(np.argmax(mean_acc))])

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, K: int, *, pair: bool, seed: int,
            config: Optional[RDAConfig] = None, rng_key: Optional[Sequence[int]] = None,
            device: Optional[str] = None) -> "RDAHead":
        """Train on the rows given, and only those. Inner CV over `config.beta_grid`."""
        cfg = config or cls.CONFIG_CLS()
        if type(cfg) is not cls.CONFIG_CLS:
            raise TypeError(f"{cls.__name__}.fit needs a {cls.CONFIG_CLS.__name__}, got {type(cfg).__name__}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be (N, in_dim) with N == len(y), got {X.shape} and {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("rda training rows hold non-finite values (NaN or Inf)")
        if not np.issubdtype(y.dtype, np.integer) or len(y) == 0 or y.min() < 0 or y.max() >= K:
            raise ValueError(f"labels must be integers in [0, {K}), got {y.dtype} range "
                             f"[{y.min() if len(y) else None}, {y.max() if len(y) else None}]")
        full, _, _ = sx.split_source(X, pair)
        n, D = full.shape
        if n < 2:
            raise ValueError(f"rda needs at least 2 training rows, got {n}")
        mu_full, sd_full = full.mean(0), full.std(0) + 1e-6
        z_std = (full - mu_full) / sd_full
        pca_mu = z_std.mean(0)
        _, s, Vt = np.linalg.svd(z_std - pca_mu, full_matrices=False)
        # Same clamp as LinearProbe's pca_k path (sota_ensemble_experts.py): the achievable rank is
        # min(n, D), not pca_k alone, and Vt.shape[0] holds the true min(n, D) after the SVD ran.
        k = min(cfg.pca_k, Vt.shape[0] - 1)
        if k < 1:
            raise ValueError(f"rda needs at least 2 usable PCA dimensions, got k={k} from n={n}, D={D}")
        sd_pc = s[:k] / math.sqrt(max(n - 1, 1))
        pca_P = Vt[:k].T / (sd_pc + 1e-6)
        z = (z_std - pca_mu) @ pca_P

        beta = cls._select_beta(z, y, K, cfg, seed)
        mu_c, prec_c, logdet_c, log_prior = cls._fit_discriminant(z, y, K, beta)

        cfg_out = {"K": int(K), "pair": bool(pair), "in_dim": int(X.shape[1]), "D": int(D), "k": int(k),
                   "head_type": cls.head_type, "beta": float(beta),
                   "config": {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in asdict(cfg).items()}}
        arrays = {"mu_full": mu_full, "sd_full": sd_full, "pca_mu": pca_mu, "pca_P": pca_P,
                  "mu_c": mu_c, "prec_c": prec_c, "logdet_c": logdet_c, "log_prior": log_prior}
        head = cls(arrays, cfg_out)
        head.info = {"beta": float(beta), "k": int(k)}
        return head


@dataclass(frozen=True)
class NystromConfig:
    landmarks: int = 256
    rank: int = 128
    gamma_multipliers: Tuple[float, ...] = (0.25, 1.0, 4.0)     # times the median heuristic
    class_stratified_landmarks: bool = True                     # if True, landmarks count as supervised
    device: str = "cpu"                  # NystromHead is NumPy/sklearn only, no other backend exists

    def __post_init__(self) -> None:
        if not isinstance(self.landmarks, int) or isinstance(self.landmarks, bool) or self.landmarks < 2:
            raise ValueError(f"landmarks must be an int >= 2, got {self.landmarks!r}")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError(f"rank must be a positive int, got {self.rank!r}")
        if not self.gamma_multipliers:
            raise ValueError("gamma_multipliers must not be empty")
        for g in self.gamma_multipliers:
            if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or g <= 0.0:
                raise ValueError(f"gamma_multipliers values must be finite and > 0, got {g!r}")
        if not isinstance(self.class_stratified_landmarks, bool):
            raise ValueError(f"class_stratified_landmarks must be a bool, got {self.class_stratified_landmarks!r}")
        if self.device != "cpu":
            raise ValueError(f"NystromHead has no backend but CPU NumPy/sklearn, got device={self.device!r}")


class NystromHead:
    """FittableHead / ExpertCore of Spec 19 S4.3 / S6.1.

    Williams & Seeger (2001) Nystrom low-rank approximation of the RBF kernel, followed by a
    `sx.LinearProbe` fitted on the low-rank feature map (exactly the "linear probe on psi(x)"
    S4.3 calls for):

      1. Landmarks Z: m = min(landmarks, n_fit) rows of the standardized fit rows, drawn by class-
         stratified sampling (each present class gets at least one landmark, quota proportional to
         its frequency) when `class_stratified_landmarks`, else a uniform sample.
      2. Bandwidth: gamma_0 = 1 / median(pairwise squared distance among Z, i != j); the final
         gamma is gamma_0 * one of `gamma_multipliers`, chosen by accuracy on an inner
         StratifiedKFold split of the fit rows (the landmarks and their kernel eigendecomposition
         are recomputed per candidate gamma; only Z itself is fixed before the search).
      3. W = exp(-gamma ||Z_i - Z_j||^2) (m, m); its top r = min(rank, m) eigenpairs give the
         projection Proj = Lambda_r^{-1/2} U_r^T (r, m), so psi(x) = Proj @ exp(-gamma ||x - Z_j||^2)
         in R^r (Williams & Seeger's low-rank feature map).
      4. `sx.LinearProbe.fit(psi(X_fit), y, ..., kind="full", pca_k=None)` trains the classification
         head on psi(x); its own inner-CV-selected C and its absent-class handling (`sx.NEG`) carry
         over unchanged.

    Inference is NumPy only: Z, Proj and the embedded LinearProbe arrays are exported to the .npz.
    Supervised parameter count is K(r+1) plus the m landmarks when they were class-stratified
    (S4.3's honesty requirement: a label-informed landmark draw is not a free, unsupervised map).
    """

    head_type: ClassVar[str] = "nystrom"
    CONFIG_CLS: ClassVar[type] = NystromConfig
    PROBE_PREFIX = "probe__"
    ARRAY_KEYS = ("mu_full", "sd_full", "Z", "Proj")

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.cfg = dict(cfg)
        if self.cfg.get("head_type", self.head_type) != self.head_type:
            raise ValueError(f"{type(self).__name__} cannot load a {self.cfg['head_type']!r} head")
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim, self.D = int(cfg["in_dim"]), int(cfg["D"])
        self.m, self.r, self.gamma = int(cfg["m"]), int(cfg["r"]), float(cfg["gamma"])
        missing = [key for key in self.ARRAY_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"nystrom arrays missing {missing}")
        self.a = {key: np.ascontiguousarray(arrays[key], dtype=np.float32) for key in self.ARRAY_KEYS}
        D, m, r = self.D, self.m, self.r
        shapes = {"mu_full": (D,), "sd_full": (D,), "Z": (m, D), "Proj": (r, m)}
        for key, v in self.a.items():
            if v.shape != shapes[key]:
                raise ValueError(f"nystrom array {key} has shape {v.shape}, expected {shapes[key]}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"nystrom array {key} holds non-finite values")
        if self.in_dim != (3 * D if self.pair else D):
            raise ValueError(f"in_dim {self.in_dim} does not match D={D}, pair={self.pair}")
        probe_arrays = {key[len(self.PROBE_PREFIX):]: v for key, v in arrays.items()
                        if key.startswith(self.PROBE_PREFIX)}
        if not probe_arrays or "probe_cfg" not in cfg:
            raise ValueError("nystrom arrays/cfg missing the embedded probe__* LinearProbe")
        self.probe = sx.LinearProbe(probe_arrays, cfg["probe_cfg"])
        self.info: dict = {}

    # -------------------------------------------------------------- inference

    def _psi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (1, 2) or X.shape[-1] != self.in_dim:
            raise ValueError(f"nystrom expects (..., {self.in_dim}) input, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("nystrom input holds non-finite values (NaN or Inf)")
        full, _, _ = sx.split_source(X, self.pair)
        z = ((full - self.a["mu_full"]) / self.a["sd_full"]).astype(np.float64)
        Z = self.a["Z"].astype(np.float64)
        k_zx = np.exp(-self.gamma * _sqdist(z, Z))
        return (k_zx @ self.a["Proj"].astype(np.float64).T).astype(np.float32)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, in_dim) -> (N, K) float32 logits, or (in_dim,) -> (K,). NumPy only."""
        return self.probe.scores(self._psi(X))

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"nystrom trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores(x)

    def supervised_param_count(self) -> int:
        """Spec 19 S4.3: K(r + 1), plus the m landmarks when they were drawn class-stratified."""
        p = self.K * (self.r + 1)
        if self.cfg.get("class_stratified_landmarks", True):
            p += self.m
        return p

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        arrays = dict(self.a)
        arrays.update({f"{self.PROBE_PREFIX}{k}": v for k, v in self.probe.a.items()})
        cfg = dict(self.cfg, probe_cfg=self.probe.cfg)
        np.savez(path, cfg_json=np.array(json.dumps(cfg)), **arrays)

    @classmethod
    def load(cls, path: Path) -> "NystromHead":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json"}
            return cls(arrays, json.loads(str(z["cfg_json"])))

    # ---------------------------------------------------------------- training

    @staticmethod
    def _sample_landmarks(y: np.ndarray, m: int, rng: np.random.Generator, stratified: bool) -> np.ndarray:
        n = len(y)
        if not stratified:
            return rng.choice(n, size=m, replace=False)
        classes, counts = np.unique(y, return_counts=True)
        quota = np.maximum(np.floor(counts / n * m).astype(int), 1)
        while quota.sum() > m:
            quota[int(np.argmax(quota))] -= 1
        idx: list = []
        for c, q in zip(classes, quota):
            pool = np.flatnonzero(y == c)
            idx.extend(rng.choice(pool, size=min(int(q), len(pool)), replace=False).tolist())
        remaining = m - len(idx)
        if remaining > 0:
            rest = np.setdiff1d(np.arange(n), idx)
            idx.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())
        return np.array(sorted(idx))

    @staticmethod
    def _nystrom_projection(Z: np.ndarray, gamma: float, rank: int) -> Tuple[np.ndarray, int]:
        m = Z.shape[0]
        W = np.exp(-gamma * _sqdist(Z, Z))
        W = 0.5 * (W + W.T)
        eigvals, eigvecs = np.linalg.eigh(W)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        r = min(rank, m)
        floor = 1e-10 * max(float(eigvals[0]), 1e-300)
        eigvals_r = np.clip(eigvals[:r], floor, None)
        Proj = (eigvecs[:, :r] / np.sqrt(eigvals_r)[None, :]).T
        return Proj, r

    @staticmethod
    def _select_gamma(z: np.ndarray, y: np.ndarray, K: int, Z: np.ndarray, gamma0: float,
                      cfg: NystromConfig, seed: int) -> float:
        grid = cfg.gamma_multipliers
        if len(grid) == 1:
            return gamma0 * grid[0]
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            return gamma0 * grid[0]
        from sklearn.model_selection import StratifiedKFold
        n_splits = int(max(2, min(4, counts.min())))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        mean_acc = []
        for gmul in grid:
            gamma = gamma0 * gmul
            Proj, _ = NystromHead._nystrom_projection(Z, gamma, cfg.rank)
            psi = np.exp(-gamma * _sqdist(z, Z)) @ Proj.T
            accs = []
            for tr_i, va_i in cv.split(psi, y):
                if len(np.unique(y[tr_i])) < 2:
                    continue
                lp = sx.LinearProbe.fit(psi[tr_i], y[tr_i], K, pair=False, kind="full", pca_k=None,
                                        seed=seed, device="cpu")
                pred = lp.scores(psi[va_i]).argmax(1)
                accs.append(float(np.mean(pred == y[va_i])))
            mean_acc.append(float(np.mean(accs)) if accs else -math.inf)
        return float(gamma0 * grid[int(np.argmax(mean_acc))])

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, K: int, *, pair: bool, seed: int,
            config: Optional[NystromConfig] = None, rng_key: Optional[Sequence[int]] = None,
            device: Optional[str] = None) -> "NystromHead":
        """Train on the rows given, and only those. Inner CV over `config.gamma_multipliers`."""
        cfg = config or cls.CONFIG_CLS()
        if type(cfg) is not cls.CONFIG_CLS:
            raise TypeError(f"{cls.__name__}.fit needs a {cls.CONFIG_CLS.__name__}, got {type(cfg).__name__}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be (N, in_dim) with N == len(y), got {X.shape} and {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("nystrom training rows hold non-finite values (NaN or Inf)")
        if not np.issubdtype(y.dtype, np.integer) or len(y) == 0 or y.min() < 0 or y.max() >= K:
            raise ValueError(f"labels must be integers in [0, {K}), got {y.dtype} range "
                             f"[{y.min() if len(y) else None}, {y.max() if len(y) else None}]")
        full, _, _ = sx.split_source(X, pair)
        n, D = full.shape
        if n < 2:
            raise ValueError(f"nystrom needs at least 2 training rows, got {n}")
        mu_full, sd_full = full.mean(0), full.std(0) + 1e-6
        z = (full - mu_full) / sd_full

        rng = np.random.default_rng(list(rng_key) if rng_key is not None else [sx.TRAIN_SEED, int(seed)])
        m = min(cfg.landmarks, n)
        landmark_idx = cls._sample_landmarks(y, m, rng, cfg.class_stratified_landmarks)
        Z = z[landmark_idx]

        d2_zz = _sqdist(Z, Z)
        offdiag = d2_zz[~np.eye(m, dtype=bool)]
        med = float(np.median(offdiag)) if offdiag.size else 1.0
        gamma0 = 1.0 / med if math.isfinite(med) and med > 0.0 else 1.0

        best_gamma = cls._select_gamma(z, y, K, Z, gamma0, cfg, seed)
        Proj, r = cls._nystrom_projection(Z, best_gamma, cfg.rank)
        psi = np.exp(-best_gamma * _sqdist(z, Z)) @ Proj.T
        lp = sx.LinearProbe.fit(psi, y, K, pair=False, kind="full", pca_k=None, seed=seed, device="cpu")

        cfg_out = {"K": int(K), "pair": bool(pair), "in_dim": int(X.shape[1]), "D": int(D), "m": int(m),
                   "r": int(r), "gamma": float(best_gamma), "head_type": cls.head_type,
                   "class_stratified_landmarks": bool(cfg.class_stratified_landmarks), "probe_cfg": lp.cfg,
                   "config": {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in asdict(cfg).items()}}
        arrays = {"mu_full": mu_full, "sd_full": sd_full, "Z": Z, "Proj": Proj}
        arrays.update({f"{cls.PROBE_PREFIX}{k}": v for k, v in lp.a.items()})
        head = cls(arrays, cfg_out)
        head.info = {"gamma": float(best_gamma), "gamma0": float(gamma0), "m": int(m), "r": int(r),
                    "probe_C": lp.cfg.get("C")}
        return head


# --------------------------------------------- Spec 20 P3: Formulation B (adapter_b)

@dataclass(frozen=True)
class AdapterBConfig:
    """Spec 20 S4.2 Formulation B: f(x) = f0(x) + C GELU(U x + a) on the frozen linear baseline f0."""
    rank: int = 64                       # r: candidates 32/64, 128 pre-registered extension (S4.3)
    lambda_C: float = 1e-3               # lambda_C / 2 * ||C||_F^2
    lambda_theta: float = 1e-3           # lambda_theta / 2 * ||theta - theta_0||_F^2, theta = (U, a)
    lr: float = 1e-3                     # initial proximal step size eta of the theta step (grows 2x per accepted step, backtracked by halving)
    max_epochs: int = 50                 # outer alternating iterations (C step + theta step)
    device: str = "cpu"                  # fit-time only: "cpu", "cuda" (raises if absent) or "auto"
    lbfgs_max_iter: int = 100            # L-BFGS iterations of one C step
    max_backtracks: int = 20             # halvings of eta before a theta step is rejected
    rel_tol: float = 1e-6                # stop when (J_prev - J) / max(J_prev, 1e-12) < rel_tol

    def __post_init__(self) -> None:
        for name in ("rank", "max_epochs", "lbfgs_max_iter", "max_backtracks"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        for name in ("lambda_C", "lambda_theta"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
        for name in ("lr", "rel_tol"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
        sx.check_device_name(self.device)


class FoldedResidualAdapterBHead:
    """FittableHead / ExpertCore of Spec 20 S4.2 (Formulation B). Inference is NumPy only.

    On the standardized full-context vector z (the `lin_full` feature map):

        logits = z W0^T + b0 + GELU(z U^T + a) C^T        W0 (K, D), C (K, r), U (r, D), a (r,)

    f0(x) = W0 x + b0 is a multinomial logistic regression (sx.LinearProbe, kind="full") fitted
    on the SAME rows fit() receives and then frozen. C starts at exactly 0, so the initial head
    IS f0 (bit-exact: the residual term adds an all-zero float32 array to the probe's own
    float32 matmul). U starts at the top-r PCA directions of the unlabeled standardized fit
    rows (each scaled to unit variance on those rows), a at 0; theta_0 = (U_0, a_0) is the
    centre of the theta penalty.

    Training minimizes, in float64 and always on the full fit set (never a minibatch),

        J(C, theta) = mean_i CE(y_i, z0_i + C GELU(U z_i + a)) + lambda_C/2 ||C||_F^2
                      + lambda_theta/2 ||theta - theta_0||_F^2,          J(0, theta_0) = J_0 = CE(f0),

    by alternating (1) a convex C step (torch L-BFGS, strong Wolfe; a backtracking gradient step
    if L-BFGS did not lower J) and (2) a proximal gradient step on theta with backtracking on
    the standard quadratic upper bound. A step is accepted only if the full-train J did not
    increase, else the previous iterate is kept and the rejection is logged. `objective_trace`
    holds J_0 followed by the J of every accepted step, so J_t <= J_{t-1} <= J_0 by
    construction (S4.2): a bound on TRAINING cross-entropy, not on accuracy or test loss.
    No early-stop split is used (info["early_stop_scope"] == "none").
    """

    head_type: ClassVar[str] = "adapter_b"
    CONFIG_CLS: ClassVar[type] = AdapterBConfig
    ARRAY_KEYS = ("mu_full", "sd_full", "W0", "b0", "C", "U", "a")

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.cfg = dict(cfg)
        if self.cfg.get("head_type") != self.head_type:
            raise ValueError(f"{type(self).__name__} cannot load a {self.cfg.get('head_type')!r} head")
        missing_cfg = [k for k in ("K", "pair", "in_dim", "D", "rank") if k not in self.cfg]
        if missing_cfg:
            raise ValueError(f"adapter_b cfg missing {missing_cfg}")
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim, self.D, self.rank = int(cfg["in_dim"]), int(cfg["D"]), int(cfg["rank"])
        missing = [k for k in self.ARRAY_KEYS if k not in arrays]
        if missing:
            raise ValueError(f"adapter_b arrays missing {missing}")
        self.a = {k: np.ascontiguousarray(arrays[k], dtype=np.float32) for k in self.ARRAY_KEYS}
        D, r, K = self.D, self.rank, self.K
        shapes = {"mu_full": (D,), "sd_full": (D,), "W0": (K, D), "b0": (K,), "C": (K, r), "U": (r, D), "a": (r,)}
        for k, v in self.a.items():
            if v.shape != shapes[k]:
                raise ValueError(f"adapter_b array {k} has shape {v.shape}, expected {shapes[k]}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"adapter_b array {k} holds non-finite values")
        if self.in_dim != (3 * D if self.pair else D):
            raise ValueError(f"in_dim {self.in_dim} does not match D={D}, pair={self.pair}")
        if K < 1 or r < 1 or D < 1:
            raise ValueError(f"adapter_b needs K, rank, D >= 1, got K={K}, rank={r}, D={D}")
        self.info: dict = {}

    # ------------------------------------------------------------ construction

    @classmethod
    def from_arrays(cls, lp: "sx.LinearProbe", C, U, a, *, config: AdapterBConfig) -> "FoldedResidualAdapterBHead":
        """Wrap the frozen probe lp (W0, b0 and its standardization) with the residual (C, U, a)."""
        if lp.cfg.get("kind") != "full" or lp.cfg.get("pca_k"):
            raise ValueError("adapter_b needs a raw 'full' LinearProbe (no PCA) as its baseline f0")
        K, D = lp.a["W"].shape
        cfg = {"K": int(K), "pair": bool(lp.pair), "in_dim": int(lp.in_dim), "D": int(D), "rank": int(config.rank),
               "head_type": cls.head_type, "gelu": "tanh", "probe_C": lp.cfg.get("C"),
               "config": {k: v for k, v in asdict(config).items() if k != "device"}}
        arrays = {"mu_full": lp.a["mu_full"], "sd_full": lp.a["sd_full"], "W0": lp.a["W"], "b0": lp.a["b"],
                  "C": np.asarray(C), "U": np.asarray(U), "a": np.asarray(a)}
        return cls(arrays, cfg)

    @staticmethod
    def pca_init(z: np.ndarray, rank: int) -> np.ndarray:
        """U_0 (rank, D): top-`rank` principal directions of the (unlabeled) standardized rows z,
        each scaled so the corresponding projection has unit variance on z. Labels never enter."""
        z = np.asarray(z, dtype=np.float64)
        n, D = z.shape
        if rank > min(n - 1, D):
            raise ValueError(f"adapter_b rank {rank} exceeds the {min(n - 1, D)} PCA directions "
                             f"available from {n} rows of dimension {D}")
        zc = z - z.mean(0)
        _, s, Vt = np.linalg.svd(zc, full_matrices=False)
        sd = s[:rank] / math.sqrt(max(n - 1, 1))
        if not np.all(sd > 0.0):
            raise ValueError("adapter_b PCA init: a requested principal direction has zero variance")
        return Vt[:rank] / sd[:, None]

    # -------------------------------------------------------------- inference

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (1, 2) or X.shape[-1] != self.in_dim:
            raise ValueError(f"adapter_b expects (..., {self.in_dim}) input, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("adapter_b input holds non-finite values (NaN or Inf)")
        full, _, _ = sx.split_source(X, self.pair)
        return (full - self.a["mu_full"]) / self.a["sd_full"]

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, in_dim) -> (N, K) float32 logits, or (in_dim,) -> (K,). NumPy only.

        The baseline matmul is kept as its own float32 operation (the exact expression
        sx.LinearProbe.scores evaluates) and the residual is added afterwards, so C == 0 gives
        the probe's logits bit for bit."""
        return self._scores_from_standardized(self._standardize(X))

    def _scores_from_standardized(self, z: np.ndarray) -> np.ndarray:
        a = self.a
        z = np.asarray(z, dtype=np.float32)
        base = (z @ a["W0"].T + a["b0"]).astype(np.float32)
        resid = gelu_tanh(z @ a["U"].T + a["a"]) @ a["C"].T
        return (base + resid).astype(np.float32)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"adapter_b trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores(x)

    def supervised_param_count(self) -> int:
        """Spec 20 S4.3: K(D + 1) frozen baseline + r(D + 1) for (U, a) + Kr for C."""
        D, r, K = self.D, self.rank, self.K
        return K * (D + 1) + r * (D + 1) + K * r

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a)

    @classmethod
    def load(cls, path: Path) -> "FoldedResidualAdapterBHead":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json"}
            return cls(arrays, json.loads(str(z["cfg_json"])))

    # ---------------------------------------------------------------- training

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, K: int, *, pair: bool, seed: int,
            config: Optional[AdapterBConfig] = None, rng_key: Optional[Sequence[int]] = None,
            probe_fit=None, device: Optional[str] = None) -> "FoldedResidualAdapterBHead":
        """Train on the rows given, and only those; every row is a training row (no early-stop split).

        device overrides config.device. probe_fit(X, y) -> LinearProbe overrides the baseline fit
        (default: sx.LinearProbe.fit kind="full", no PCA, inner-CV C on the same rows).
        rng_key is accepted for interface parity: the algorithm is deterministic given the rows.
        """
        cfg = config or cls.CONFIG_CLS()
        if type(cfg) is not cls.CONFIG_CLS:
            raise TypeError(f"{cls.__name__}.fit needs a {cls.CONFIG_CLS.__name__}, got {type(cfg).__name__}")
        dev = sx.resolve_device(cfg.device if device is None else device)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be (N, in_dim) with N == len(y), got {X.shape} and {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("adapter_b training rows hold non-finite values (NaN or Inf)")
        if not np.issubdtype(y.dtype, np.integer) or len(y) == 0 or y.min() < 0 or y.max() >= K:
            raise ValueError(f"labels must be integers in [0, {K}), got {y.dtype} range "
                             f"[{y.min() if len(y) else None}, {y.max() if len(y) else None}]")
        if len(np.unique(y)) < 2:
            raise ValueError("adapter_b needs at least two classes among the training rows")
        if probe_fit is None:
            probe_fit = lambda Xp, yp: sx.LinearProbe.fit(Xp, yp, K, pair=pair, kind="full",  # noqa: E731
                                                          pca_k=None, seed=int(seed), device=dev)
        t0 = time.perf_counter()
        lp = probe_fit(X, y)
        full, _, _ = sx.split_source(X, pair)
        z = (full - lp.a["mu_full"]) / lp.a["sd_full"]   # the probe's own standardization, float32
        U0 = cls.pca_init(z, cfg.rank)
        init = cls.from_arrays(lp, np.zeros((K, cfg.rank)), U0, np.zeros(cfg.rank), config=cfg)
        identity_err = float(np.max(np.abs(init.scores(X) - lp.scores(X))))   # at (C=0, theta_0)
        head, info = _train_adapter_b(lp, z, y, K, U0, cfg, dev)
        info.update(device=dev, lr=cfg.lr, train_seconds=time.perf_counter() - t0, n_train=int(len(y)),
                    n_early_stop=0, early_stop_scope="none", probe_C=lp.cfg.get("C"),
                    identity_max_abs_err_at_init=identity_err, params=head.supervised_param_count())
        head.info = info
        return head


def _train_adapter_b(lp, z: np.ndarray, y: np.ndarray, K: int, U0: np.ndarray, cfg: AdapterBConfig, dev: str):
    """Spec 20 S4.2 alternating proximal optimization, float64, full batch, on torch device `dev`.

    Returns (head, info). The head carries the last ACCEPTED (C, U, a); info["objective_trace"] is
    [J_0, J after each accepted step] and is non-increasing by construction."""
    import torch
    import torch.nn.functional as F
    dt = torch.float64
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64), device=dev, dtype=dt)
    yt = torch.as_tensor(y.astype(np.int64), device=dev)
    W0 = torch.as_tensor(lp.a["W"].astype(np.float64), device=dev, dtype=dt)
    b0 = torch.as_tensor(lp.a["b"].astype(np.float64), device=dev, dtype=dt)
    z0 = zt @ W0.T + b0                               # frozen baseline logits, float64
    r = cfg.rank
    U_init = torch.as_tensor(np.asarray(U0, dtype=np.float64), device=dev, dtype=dt)
    a_init = torch.zeros(r, device=dev, dtype=dt)
    C = torch.zeros((K, r), device=dev, dtype=dt, requires_grad=True)
    U = U_init.clone().requires_grad_(True)
    a = a_init.clone().requires_grad_(True)
    lamC, lamT = float(cfg.lambda_C), float(cfg.lambda_theta)

    def act_of(U_, a_):
        return F.gelu(zt @ U_.T + a_, approximate="tanh")

    def ce_of(C_, act_):
        return F.cross_entropy(z0 + act_ @ C_.T, yt)

    def R_theta(U_, a_):
        return 0.5 * lamT * ((U_ - U_init).pow(2).sum() + (a_ - a_init).pow(2).sum())

    def J_of(C_, U_, a_):
        return ce_of(C_, act_of(U_, a_)) + 0.5 * lamC * C_.pow(2).sum() + R_theta(U_, a_)

    with torch.no_grad():
        J0 = float(ce_of(C, act_of(U, a)))            # == J(0, theta_0): both penalties vanish
    if not math.isfinite(J0):
        raise FloatingPointError("adapter_b: the baseline objective J_0 is not finite")
    trace = [J0]
    step_log = []
    rejected = 0
    backtrack_failures = 0
    lbfgs_rejections = 0            # L-BFGS results that did not lower J (rescued by backtracking GD or rejected)
    status = "max_epochs"
    J_cur = J0
    eta_theta = float(cfg.lr)     # last accepted proximal step size; cfg.lr is its start value

    for epoch in range(1, cfg.max_epochs + 1):
        J_epoch_start = J_cur
        # ---- (1) C step: convex in C for the fixed feature matrix act = GELU(U z + a).
        with torch.no_grad():
            act = act_of(U, a)
        C_prev = C.detach().clone()
        J_before = J_cur

        def c_objective(C_):
            return ce_of(C_, act) + 0.5 * lamC * C_.pow(2).sum() + float(R_theta(U, a).detach())

        opt = torch.optim.LBFGS([C], lr=1.0, max_iter=cfg.lbfgs_max_iter, tolerance_grad=1e-12,
                                tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")
        n_evals = [0]

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = c_objective(C)
            loss.backward()
            n_evals[0] += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            J_c = float(J_of(C, U, a))
        c_accepted = math.isfinite(J_c) and J_c <= J_before
        c_method = "lbfgs"
        if not c_accepted:
            lbfgs_rejections += 1
            # Second attempt: a backtracking gradient step on C from the previous iterate.
            with torch.no_grad():
                C.copy_(C_prev)
            Cg = C.detach().clone().requires_grad_(True)
            gC = torch.autograd.grad(c_objective(Cg), Cg)[0]
            eta = 1.0
            for _ in range(cfg.max_backtracks):
                with torch.no_grad():
                    cand = C_prev - eta * gC
                    J_c = float(J_of(cand, U, a))
                if math.isfinite(J_c) and J_c <= J_before:
                    with torch.no_grad():
                        C.copy_(cand)
                    c_accepted, c_method = True, "backtracking_gd"
                    break
                eta *= 0.5
        if c_accepted:
            J_cur = J_c
            trace.append(J_cur)
        else:
            with torch.no_grad():
                C.copy_(C_prev)
            rejected += 1
        step_log.append({"epoch": epoch, "step": "C", "method": c_method, "accepted": bool(c_accepted),
                         "J_before": J_before, "J_after": J_c if c_accepted else None, "lbfgs_evals": n_evals[0]})

        # ---- (2) theta step: proximal gradient on L = J - R_theta with backtracking (S4.2 items 3-4).
        J_before = J_cur
        U_prev, a_prev = U.detach().clone(), a.detach().clone()
        Ug = U_prev.clone().requires_grad_(True)
        ag = a_prev.clone().requires_grad_(True)
        L_val = ce_of(C.detach(), act_of(Ug, ag)) + 0.5 * lamC * C.detach().pow(2).sum()
        gU, ga = torch.autograd.grad(L_val, (Ug, ag))
        L_prev = float(L_val.detach())
        eta = 2.0 * eta_theta        # try a larger step than last time, backtrack down
        t_accepted = False
        J_t = None
        n_bt = 0
        for n_bt in range(cfg.max_backtracks + 1):
            with torch.no_grad():
                U_new = (U_prev - eta * gU + eta * lamT * U_init) / (1.0 + eta * lamT)
                a_new = (a_prev - eta * ga + eta * lamT * a_init) / (1.0 + eta * lamT)
                dU, da = U_new - U_prev, a_new - a_prev
                L_new = float(ce_of(C, act_of(U_new, a_new)) + 0.5 * lamC * C.pow(2).sum())
                bound = L_prev + float((gU * dU).sum() + (ga * da).sum()) + float(dU.pow(2).sum() + da.pow(2).sum()) / (2.0 * eta)
                J_t = float(L_new + R_theta(U_new, a_new))
            if math.isfinite(L_new) and L_new <= bound and math.isfinite(J_t) and J_t <= J_before:
                t_accepted = True
                break
            eta *= 0.5
        if t_accepted:
            with torch.no_grad():
                U.copy_(U_new)
                a.copy_(a_new)
            J_cur = J_t
            eta_theta = eta
            trace.append(J_cur)
        else:
            rejected += 1
            backtrack_failures += 1
        step_log.append({"epoch": epoch, "step": "theta", "accepted": bool(t_accepted), "eta": eta,
                         "backtracks": n_bt, "J_before": J_before, "J_after": J_t if t_accepted else None})

        if not c_accepted and not t_accepted:
            status = "stalled"
            break
        if (J_epoch_start - J_cur) / max(abs(J_epoch_start), 1e-12) < cfg.rel_tol:
            status = "converged"
            break

    if len(trace) == 1:
        status = "no_step_accepted"
    # The trace is monotone by construction; assert it rather than trust it.
    if any(b > a_ for a_, b in zip(trace, trace[1:])):
        raise FloatingPointError("adapter_b: objective_trace is not non-increasing (internal error)")

    C_np, U_np, a_np = (t.detach().cpu().numpy() for t in (C, U, a))
    head = FoldedResidualAdapterBHead.from_arrays(lp, C_np, U_np, a_np, config=cfg)
    # Export self-check: the float32 NumPy head must reproduce the float64 torch forward.
    with torch.no_grad():
        ref = (z0 + act_of(U, a) @ C.T).cpu().numpy()
    got = head._scores_from_standardized(z)
    export_err = float(np.max(np.abs(got - ref)))
    if not export_err <= 1e-4 * (1.0 + float(np.max(np.abs(ref)))):
        raise FloatingPointError(f"adapter_b export does not match its torch forward: max abs err {export_err}")
    with torch.no_grad():
        train_ce = float(ce_of(C, act_of(U, a)))
        gmap = float((gU.pow(2).sum() + ga.pow(2).sum()).sqrt())
    info = {"objective_trace": trace, "J0": J0, "J_final": J_cur, "train_ce_final": train_ce,
            "train_ce_baseline": J0, "optimizer_status": status, "rejected_steps": rejected,
            "theta_backtrack_failures": backtrack_failures, "lbfgs_rejections": lbfgs_rejections,
            "epochs_run": len(step_log) // 2,
            "accepted_steps": len(trace) - 1, "step_log": step_log, "export_max_abs_err": export_err,
            "theta_grad_norm_before_last_step": gmap, "c_fro": float(np.linalg.norm(C_np)),
            "theta_dist_from_init": float(math.sqrt(np.sum((U_np - U0) ** 2) + np.sum(a_np ** 2)))}
    return head, info


# --------------------------------------------- Spec 20 P4: dual-manifold orthogonal innovation projection

@dataclass(frozen=True)
class DualManifoldConfig:
    """Spec 20 S3.3 / S6.2: Qwen principal subspace + Gemma orthogonal innovation, folded for CPU.

    k_qwen / k_gemma are BUDGETS: fit() truncates each to the rank the fit rows can identify
    (<= n_fit - 1, <= source width, positive singular values only) and records the actual ranks.
    No zero padding (S3.3). gate is the constant g multiplying the innovation block (the
    no-learning control of S3.3); a conditional gate would break the linear fold and is not offered here.
    """
    k_qwen: int = 1536                   # budget of Qwen principal directions k_q
    k_gemma: int = 512                   # budget of Gemma innovation directions k_g
    target_dim: int = 2048               # k_q + k_g <= target_dim (S3.4: a cost cap, not a "lossless" claim)
    regularization: float = 1e-4         # lambda of the ridge regression B; 0 = OLS (exact orthogonality)
    gate: float = 1.0                    # constant g on the innovation block
    rank_rtol: float = 1e-3              # drop directions with singular value < rank_rtol * largest (see fit)
    device: str = "cpu"                  # fit-time only: where the top LinearProbe is fitted

    def __post_init__(self) -> None:
        for name in ("k_qwen", "k_gemma", "target_dim"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.k_qwen + self.k_gemma > self.target_dim:
            raise ValueError(f"k_qwen + k_gemma = {self.k_qwen + self.k_gemma} exceeds target_dim {self.target_dim}")
        for name, lo in (("regularization", 0.0), ("gate", 0.0)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo:
                raise ValueError(f"{name} must be finite and >= {lo}, got {v!r}")
        v = self.rank_rtol
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 < v < 1.0:
            raise ValueError(f"rank_rtol must be in (0, 1), got {v!r}")
        sx.check_device_name(self.device)


def fold_dual_manifold(u: Dict[str, np.ndarray], gate: float, dtype=np.float64) -> Dict[str, np.ndarray]:
    """Spec 20 S6.2 algebraic fold of the unfolded chain `u` into (W_fold_Q, W_fold_G, b_fold).

    The unfolded chain, all affine, is
        Xqs = (X_Q - mu_Q) / sd_Q,  Xgs = (X_G - mu_G) / sd_G
        Z_Q = Xqs P_Q,  E_G = Xgs - Z_Q B,  z = [Z_Q ; g E_G R_G]
        s = ((z - z_mu) / z_sd) W_top^T + b_top
    so with W_eff = W_top / z_sd = [W_zQ, W_zG] and b_eff = b_top - W_eff z_mu:
        W_fold_Q = (W_zQ P_Q^T - g W_zG R_G^T B^T P_Q^T) / sd_Q
        W_fold_G = g W_zG R_G^T / sd_G
        b_fold   = b_eff - W_fold_Q mu_Q - W_fold_G mu_G
    Every product is taken in the order that never materializes a D_Q x D_G matrix.
    """
    c = {k: np.asarray(v, dtype=dtype) for k, v in u.items()}
    k_q = c["P_Q"].shape[1]
    W_eff = c["W_top"] / c["z_sd"]
    b_eff = c["b_top"] - W_eff @ c["z_mu"]
    W_zQ, W_zG = W_eff[:, :k_q], W_eff[:, k_q:]
    g = dtype(gate)
    RW = c["R_G"] @ W_zG.T                                   # (D_G, K)
    A_Q = c["P_Q"] @ (W_zQ.T - g * (c["B"] @ RW))            # (D_Q, K)
    W_fold_Q = A_Q.T / c["sd_Q"]
    W_fold_G = g * RW.T / c["sd_G"]
    b_fold = b_eff - W_fold_Q @ c["mu_Q"] - W_fold_G @ c["mu_G"]
    return {"W_fold_Q": W_fold_Q, "W_fold_G": W_fold_G, "b_fold": b_fold}


class DualManifoldHead:
    """FittableHead / ExpertCore of Spec 20 P4 (S3.3 + S6.2). Inference is NumPy only, two GEMVs:

        logits = X_Q W_fold_Q^T + X_G W_fold_G^T + b_fold        W_fold_Q (K, D_Q), W_fold_G (K, D_G)

    fit() standardizes each source on the fit rows, takes the top-k_q whitened principal directions
    of Qwen (P_Q), ridge-regresses standardized Gemma on Z_Q (B), keeps the top-k_g directions of
    the innovation residual E_G = Xgs - Z_Q B (R_G, orthonormal columns), and fits a
    sx.LinearProbe (kind "full", inner-CV C) on z = [Z_Q ; g E_G R_G]. Everything is fitted on the
    rows fit() receives and nothing else: the dispatcher hands it one fold's training rows, so no
    projection ever sees held-out rows or labels (S3.3, S7.2). Labels enter the top probe only.

    Z_Q^T E_G = lambda B exactly; it is 0 only for lambda = 0 (S3.3). fit() records the actual
    ranks, the variance each block keeps and the share of Gemma variance Qwen explains, and
    checks the exported float32 fold against the float64 unfolded chain on the fit rows.

    Wire protocol: the ensemble's .score(x, C) / .in_dim contract takes ONE vector, so the dual
    expert's x is the concatenation [x_Q ; x_G] (width in_dim_q + in_dim_g) and score() splits it.
    scores(X_Q, X_G) is the explicit two-source API. On pair tasks each source contributes its
    full-context third only, as lin_full does.
    """

    head_type: ClassVar[str] = "dual_manifold"
    CONFIG_CLS: ClassVar[type] = DualManifoldConfig
    FOLD_KEYS = ("W_fold_Q", "W_fold_G", "b_fold")
    UNFOLD_KEYS = ("mu_Q", "sd_Q", "P_Q", "B", "mu_G", "sd_G", "R_G", "z_mu", "z_sd", "W_top", "b_top")
    CFG_KEYS = ("K", "pair", "in_dim_q", "in_dim_g", "D_Q", "D_G", "k_q", "k_g", "gate")

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict,
                 unfolded: Optional[Dict[str, np.ndarray]] = None) -> None:
        self.cfg = dict(cfg)
        if self.cfg.get("head_type") != self.head_type:
            raise ValueError(f"{type(self).__name__} cannot load a {self.cfg.get('head_type')!r} head")
        missing_cfg = [k for k in self.CFG_KEYS if k not in self.cfg]
        if missing_cfg:
            raise ValueError(f"dual_manifold cfg missing {missing_cfg}")
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim_q, self.in_dim_g = int(cfg["in_dim_q"]), int(cfg["in_dim_g"])
        self.D_Q, self.D_G, self.k_q, self.k_g = (int(cfg[k]) for k in ("D_Q", "D_G", "k_q", "k_g"))
        self.gate = float(cfg["gate"])
        if not math.isfinite(self.gate) or self.gate < 0.0:
            raise ValueError(f"dual_manifold gate must be finite and >= 0, got {self.gate}")
        K, D_Q, D_G, k_q, k_g = self.K, self.D_Q, self.D_G, self.k_q, self.k_g
        if min(K, D_Q, D_G, k_q, k_g) < 1:
            raise ValueError(f"dual_manifold needs K, D_Q, D_G, k_q, k_g >= 1, got {K}, {D_Q}, {D_G}, {k_q}, {k_g}")
        for name, in_dim, D in (("q", self.in_dim_q, D_Q), ("g", self.in_dim_g, D_G)):
            if in_dim != (3 * D if self.pair else D):
                raise ValueError(f"in_dim_{name} {in_dim} does not match D={D}, pair={self.pair}")
        missing = [k for k in self.FOLD_KEYS if k not in arrays]
        if missing:
            raise ValueError(f"dual_manifold arrays missing {missing}")
        self.a = {k: np.ascontiguousarray(arrays[k], dtype=np.float32) for k in self.FOLD_KEYS}
        self._check_shapes(self.a, {"W_fold_Q": (K, D_Q), "W_fold_G": (K, D_G), "b_fold": (K,)})
        self.u: Optional[Dict[str, np.ndarray]] = None
        if unfolded is not None:
            missing = [k for k in self.UNFOLD_KEYS if k not in unfolded]
            if missing:
                raise ValueError(f"dual_manifold unfolded arrays missing {missing}")
            self.u = {k: np.ascontiguousarray(unfolded[k], dtype=np.float64) for k in self.UNFOLD_KEYS}
            kz = k_q + k_g
            self._check_shapes(self.u, {"mu_Q": (D_Q,), "sd_Q": (D_Q,), "P_Q": (D_Q, k_q), "B": (k_q, D_G),
                                        "mu_G": (D_G,), "sd_G": (D_G,), "R_G": (D_G, k_g), "z_mu": (kz,),
                                        "z_sd": (kz,), "W_top": (K, kz), "b_top": (K,)})
        self.info: dict = {}

    @staticmethod
    def _check_shapes(arrays: Dict[str, np.ndarray], shapes: Dict[str, tuple]) -> None:
        for k, shape in shapes.items():
            v = arrays[k]
            if v.shape != shape:
                raise ValueError(f"dual_manifold array {k} has shape {v.shape}, expected {shape}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"dual_manifold array {k} holds non-finite values")

    @property
    def in_dim(self) -> int:
        """Width of the concatenated wire vector [x_Q ; x_G] (sx.EngineExpert reads this)."""
        return self.in_dim_q + self.in_dim_g

    # -------------------------------------------------------------- inference

    def _full_pair(self, X_Q: np.ndarray, X_G: np.ndarray, what: str) -> Tuple[np.ndarray, np.ndarray]:
        X_Q, X_G = np.asarray(X_Q, dtype=np.float32), np.asarray(X_G, dtype=np.float32)
        if X_Q.ndim != X_G.ndim or X_Q.ndim not in (1, 2):
            raise ValueError(f"dual_manifold {what}: X_Q and X_G must both be 1-D or both 2-D, got {X_Q.shape} and {X_G.shape}")
        if X_Q.ndim == 2 and X_Q.shape[0] != X_G.shape[0]:
            raise ValueError(f"dual_manifold {what}: X_Q has {X_Q.shape[0]} rows, X_G has {X_G.shape[0]}")
        if X_Q.shape[-1] != self.in_dim_q or X_G.shape[-1] != self.in_dim_g:
            raise ValueError(f"dual_manifold {what} expects widths ({self.in_dim_q}, {self.in_dim_g}), "
                             f"got ({X_Q.shape[-1]}, {X_G.shape[-1]})")
        if not np.all(np.isfinite(X_Q)) or not np.all(np.isfinite(X_G)):
            raise ValueError(f"dual_manifold {what} holds non-finite values (NaN or Inf)")
        fq, _, _ = sx.split_source(X_Q, self.pair)
        fg, _, _ = sx.split_source(X_G, self.pair)
        return fq, fg

    def scores(self, X_Q: np.ndarray, X_G: np.ndarray) -> np.ndarray:
        """(N, in_dim_q), (N, in_dim_g) -> (N, K) float32 logits; 1-D rows -> (K,). Two GEMVs, NumPy only."""
        fq, fg = self._full_pair(X_Q, X_G, "input")
        a = self.a
        return (fq @ a["W_fold_Q"].T + fg @ a["W_fold_G"].T + a["b_fold"]).astype(np.float32)

    def split_concat(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        if X.shape[-1] != self.in_dim:
            raise ValueError(f"dual_manifold expects a concatenated (..., {self.in_dim}) = "
                             f"(in_dim_q {self.in_dim_q} + in_dim_g {self.in_dim_g}) vector, got {X.shape}")
        return X[..., :self.in_dim_q], X[..., self.in_dim_q:]

    def scores_concat(self, X: np.ndarray) -> np.ndarray:
        return self.scores(*self.split_concat(X))

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Ensemble contract: x = [x_Q ; x_G] concatenated, C the (K, ...) candidate table."""
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"dual_manifold trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores_concat(x)

    # ------------------------------------------------- unfolded chain (verification only)

    def _need_unfolded(self) -> Dict[str, np.ndarray]:
        if self.u is None:
            raise ValueError("dual_manifold: the unfolded chain was not kept (save(include_unfolded=True) to keep it)")
        return self.u

    def fused_features(self, X_Q: np.ndarray, X_G: np.ndarray) -> np.ndarray:
        """(N, in_dim_q), (N, in_dim_g) -> z = [Z_Q ; g E_G R_G] (N, k_q + k_g), float64, materialized.
        Verification / analysis only: inference never builds z (S6.2)."""
        u = self._need_unfolded()
        fq, fg = self._full_pair(X_Q, X_G, "input")
        Xqs = (fq.astype(np.float64) - u["mu_Q"]) / u["sd_Q"]
        Xgs = (fg.astype(np.float64) - u["mu_G"]) / u["sd_G"]
        Z_Q = Xqs @ u["P_Q"]
        E_G = Xgs - Z_Q @ u["B"]
        return np.concatenate([Z_Q, self.gate * (E_G @ u["R_G"])], axis=-1)

    def scores_unfolded(self, X_Q: np.ndarray, X_G: np.ndarray) -> np.ndarray:
        """The explicit chain: materialize z, then the top probe's own affine map. float64."""
        u = self._need_unfolded()
        z = self.fused_features(X_Q, X_G)
        return ((z - u["z_mu"]) / u["z_sd"]) @ u["W_top"].T + u["b_top"]

    def supervised_param_count(self) -> int:
        """Trained-by-labels parameters: the top probe only, K (k_q + k_g + 1). The projections are
        label-free (S6.2: the training count is what the training process fitted)."""
        return self.K * (self.k_q + self.k_g + 1)

    def exported_param_count(self) -> int:
        """Parameters the CPU path reads: K (D_Q + D_G + 1)."""
        return self.K * (self.D_Q + self.D_G + 1)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path, include_unfolded: bool = False) -> None:
        """Folded matrices always; the unfolded chain only on request (P_Q alone is D_Q x k_q)."""
        extra = {}
        if include_unfolded:
            extra = {f"unfolded__{k}": v for k, v in self._need_unfolded().items()}
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a, **extra)

    @classmethod
    def load(cls, path: Path) -> "DualManifoldHead":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json" and not k.startswith("unfolded__")}
            unfolded = {k[len("unfolded__"):]: z[k] for k in z.files if k.startswith("unfolded__")}
            return cls(arrays, json.loads(str(z["cfg_json"])), unfolded or None)

    # ---------------------------------------------------------------- training

    @staticmethod
    def _truncated_rank(s: np.ndarray, budget: int, cap: int, D: int, tol: float) -> int:
        """Rank actually identifiable: min(budget, cap, D, #singular values > tol).

        The tolerance is not cosmetic. Z_Q is whitened (divided by each singular value), so a
        direction with s_j = 1e-6 s_max multiplies its noise by 1e6 and the float32 two-GEMV fold
        no longer matches the float64 chain (seen at n_fit = 30, budget 29: max abs err 0.3). Such
        directions are not identifiable from the rows anyway (S3.3): drop them and record it."""
        return int(min(budget, cap, D, int(np.sum(s > tol))))

    @staticmethod
    def rank_caps(n: int, k_qwen: int, k_gemma: int) -> Tuple[int, int]:
        """The n fit rows identify at most n - 1 centered directions IN TOTAL across both sources:
        Z_Q with k_q = n - 1 spans every row, E_G is then identically zero and there is no
        innovation left to keep. The budgets are therefore split in their own ratio:
        cap_q = ceil((n - 1) k_qwen / (k_qwen + k_gemma)), cap_g = n - 1 - cap_q (S3.3: record the
        actual rank, never claim the budget)."""
        total = n - 1
        cap_q = max(1, min(total - 1, math.ceil(total * k_qwen / (k_qwen + k_gemma))))
        return cap_q, total - cap_q

    @classmethod
    def fit(cls, X_Q: np.ndarray, X_G: np.ndarray, y: np.ndarray, K: int, *, pair: bool, seed: int,
            config: Optional[DualManifoldConfig] = None, rng_key: Optional[Sequence[int]] = None,
            probe_fit=None, device: Optional[str] = None) -> "DualManifoldHead":
        """Fit projections and top probe on the rows given, and only those (no early-stop split).

        device overrides config.device (top probe only). probe_fit(z, y) -> LinearProbe overrides
        the top-probe fit (default sx.LinearProbe.fit, kind="full", no PCA, inner-CV C). rng_key is
        accepted for interface parity: the projections are deterministic given the rows.
        """
        cfg = config or cls.CONFIG_CLS()
        if type(cfg) is not cls.CONFIG_CLS:
            raise TypeError(f"{cls.__name__}.fit needs a {cls.CONFIG_CLS.__name__}, got {type(cfg).__name__}")
        dev = sx.resolve_device(cfg.device if device is None else device)
        X_Q, X_G, y = np.asarray(X_Q, dtype=np.float32), np.asarray(X_G, dtype=np.float32), np.asarray(y)
        if X_Q.ndim != 2 or X_G.ndim != 2:
            raise ValueError(f"X_Q and X_G must be 2-D, got {X_Q.shape} and {X_G.shape}")
        if X_Q.shape[0] != X_G.shape[0] or X_Q.shape[0] != len(y):
            raise ValueError(f"X_Q, X_G and y must share N, got {X_Q.shape[0]}, {X_G.shape[0]}, {len(y)}")
        if not np.all(np.isfinite(X_Q)) or not np.all(np.isfinite(X_G)):
            raise ValueError("dual_manifold training rows hold non-finite values (NaN or Inf)")
        if not np.issubdtype(y.dtype, np.integer) or len(y) == 0 or y.min() < 0 or y.max() >= K:
            raise ValueError(f"labels must be integers in [0, {K}), got {y.dtype} range "
                             f"[{y.min() if len(y) else None}, {y.max() if len(y) else None}]")
        if len(np.unique(y)) < 2:
            raise ValueError("dual_manifold needs at least two classes among the training rows")
        n = len(y)
        if n < 3:
            raise ValueError(f"dual_manifold needs at least 3 fit rows, got {n}")
        t0 = time.perf_counter()
        fq, _, _ = sx.split_source(X_Q, pair)
        fg, _, _ = sx.split_source(X_G, pair)
        fq, fg = fq.astype(np.float64), fg.astype(np.float64)
        D_Q, D_G = fq.shape[1], fg.shape[1]

        # Qwen principal subspace, whitened: Z_Q has unit variance per column on the fit rows.
        mu_Q, sd_Q = fq.mean(0), fq.std(0) + 1e-6
        Xqs = (fq - mu_Q) / sd_Q
        _, s_q, Vt_q = np.linalg.svd(Xqs, full_matrices=False)
        cap_q, cap_g = cls.rank_caps(n, cfg.k_qwen, cfg.k_gemma)
        k_q = cls._truncated_rank(s_q, cfg.k_qwen, cap_q, D_Q, cfg.rank_rtol * s_q[0])
        if k_q < 1:
            raise ValueError("dual_manifold: the Qwen fit rows span no direction (all rows identical?)")
        P_Q = Vt_q[:k_q].T / (s_q[:k_q] / math.sqrt(n - 1))
        Z_Q = Xqs @ P_Q
        # Gemma explained by Z_Q (ridge), and its innovation residual.
        mu_G, sd_G = fg.mean(0), fg.std(0) + 1e-6
        Xgs = (fg - mu_G) / sd_G
        G = Z_Q.T @ Z_Q + cfg.regularization * np.eye(k_q)
        B = np.linalg.solve(G, Z_Q.T @ Xgs)
        E_G = Xgs - Z_Q @ B
        _, s_e, Vt_e = np.linalg.svd(E_G, full_matrices=False)
        # Innovation directions are measured against GEMMA's own scale (top singular value of Xgs),
        # not against E_G's: when Qwen explains Gemma entirely, E_G is float noise whose own top
        # direction would pass a self-relative test.
        s_g0 = float(np.linalg.norm(Xgs, 2))
        k_g = cls._truncated_rank(s_e, cfg.k_gemma, min(cap_g, n - 1 - k_q), D_G, cfg.rank_rtol * s_g0)
        if k_g < 1:
            raise ValueError("dual_manifold: Gemma has no innovation beyond the Qwen subspace on the fit rows "
                             "(E_G is numerically zero); refusing to fake a dual-source head")
        R_G = Vt_e[:k_g].T
        z = np.concatenate([Z_Q, cfg.gate * (E_G @ R_G)], axis=1)
        if probe_fit is None:
            probe_fit = lambda zp, yp: sx.LinearProbe.fit(zp, yp, K, pair=False, kind="full",  # noqa: E731
                                                          pca_k=None, seed=int(seed), device=dev)
        lp = probe_fit(z.astype(np.float32), y)
        if lp.cfg.get("kind") != "full" or lp.cfg.get("pca_k") or lp.a["W"].shape != (K, k_q + k_g):
            raise ValueError("dual_manifold needs a raw 'full' LinearProbe (no PCA) of width k_q + k_g on z")
        unfolded = {"mu_Q": mu_Q, "sd_Q": sd_Q, "P_Q": P_Q, "B": B, "mu_G": mu_G, "sd_G": sd_G, "R_G": R_G,
                    "z_mu": lp.a["mu_full"].astype(np.float64), "z_sd": lp.a["sd_full"].astype(np.float64),
                    "W_top": lp.a["W"].astype(np.float64), "b_top": lp.a["b"].astype(np.float64)}
        folded = fold_dual_manifold(unfolded, cfg.gate, np.float64)
        head_cfg = {"K": int(K), "pair": bool(pair), "in_dim_q": int(X_Q.shape[1]), "in_dim_g": int(X_G.shape[1]),
                    "D_Q": int(D_Q), "D_G": int(D_G), "k_q": int(k_q), "k_g": int(k_g), "gate": float(cfg.gate),
                    "head_type": cls.head_type, "probe_C": lp.cfg.get("C"),
                    "config": {k: v for k, v in asdict(cfg).items() if k != "device"}}
        head = cls(folded, head_cfg, unfolded)
        # Export self-check on the fit rows: float32 two-GEMV fold vs the float64 unfolded chain.
        ref = head.scores_unfolded(X_Q, X_G)
        got = head.scores(X_Q, X_G)
        export_err = float(np.max(np.abs(got - ref)))
        if not export_err <= 1e-4 * (1.0 + float(np.max(np.abs(ref)))):
            raise FloatingPointError(f"dual_manifold fold does not match its unfolded chain: max abs err {export_err}")
        tot_q, tot_e, tot_g = float(np.sum(s_q ** 2)), float(np.sum(s_e ** 2)), float(np.sum(Xgs ** 2))
        cross = float(np.max(np.abs(Z_Q.T @ E_G - cfg.regularization * B)))
        head.info = {
            "device": dev, "train_seconds": time.perf_counter() - t0, "n_train": int(n), "n_early_stop": 0,
            "early_stop_scope": "none", "probe_C": lp.cfg.get("C"), "params": head.supervised_param_count(),
            "exported_params": head.exported_param_count(), "source_dimensions": [int(D_Q), int(D_G)],
            "actual_projection_rank": [int(k_q), int(k_g)], "budget_rank": [cfg.k_qwen, cfg.k_gemma],
            "rank_truncated": bool(k_q < cfg.k_qwen or k_g < cfg.k_gemma),
            "rank_cap_by_rows": [int(cap_q), int(cap_g)],
            "singular_value_ratio_kept": [float(s_q[k_q - 1] / s_q[0]), float(s_e[k_g - 1] / s_g0)],
            "qwen_variance_kept": float(np.sum(s_q[:k_q] ** 2) / tot_q) if tot_q > 0 else 0.0,
            "innovation_variance_kept": float(np.sum(s_e[:k_g] ** 2) / tot_e) if tot_e > 0 else 0.0,
            "gemma_variance_explained_by_qwen": 1.0 - tot_e / tot_g if tot_g > 0 else 0.0,
            "orthogonality_identity_max_abs_err": cross, "export_max_abs_err": export_err,
            "train_acc_folded": float(np.mean(got.argmax(1) == y)),
        }
        return head


HEADS = {"adapter": DeepResidualAdapterHead, "supcon": SupConHead, "rda": RDAHead, "nystrom": NystromHead,
         "adapter_b": FoldedResidualAdapterBHead, "dual_manifold": DualManifoldHead}
CONFIGS = {"adapter": AdapterConfig, "supcon": SupConConfig, "rda": RDAConfig, "nystrom": NystromConfig,
           "adapter_b": AdapterBConfig, "dual_manifold": DualManifoldConfig}


def head_config(spec: dict):
    """A pipeline spec {"type": "adapter", "rank": 64, ...} -> its frozen config dataclass."""
    kw = {k: v for k, v in spec.items() if k != "type"}
    return CONFIGS[spec["type"]](**kw)
