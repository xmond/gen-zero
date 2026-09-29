"""Simplex ETF linear-probe training for frozen Qwen3.5-9B hidden states.

Implements Phase 2 of docs/architecture/13-simplex-etf-linear-probe-post-training-recipe.md.

The 9B backbone is never loaded here. Phase 1 (hidden-state extraction) is run once
elsewhere and cached as feature arrays, so the backbone is frozen by construction.
Only ``W_proj`` (hidden_dim x hidden_dim, 4096 x 4096 by default) is trained:

    q  = normalize(W_proj @ (h - mean))     # mean is stored in the probe; feed RAW h
    z  = q @ V_K^T                      # cosine to each Simplex ETF vertex

    L  = ce_weight * CE(z / tau)                      # cosine ETF cross-entropy
       + margin_weight * relu(m - (z_y - max_{k!=y} z_k))   # ETF target margin
       + alpha * SupCon(q, y; tau_con)                # supervised contrastive

Label ids index the ETF vertices, so class order is fixed by the canonical (sorted)
label set. Reordering candidates at inference cannot change the logits.

Feature sources (``load_features``):
    * ``.npz`` with ``features`` and ``labels`` arrays (optional ``label_names``)
    * ``.npy`` features plus a separate ``.npy`` labels file
    * ``synthetic``: anisotropic class-clustered vectors, for CI and smoke tests

Example:
    python -m gen_zero.train.train_simplex_probe --source synthetic --epochs 5
    python -m gen_zero.train.train_simplex_probe --source features.npz --out probe.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gen_zero.nanocore.action_etf_embedding import generate_simplex_etf

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


QWEN_HIDDEN_DIM = 4096


@dataclass
class SimplexProbeConfig:
    """Hyperparameters. Defaults follow the recipe doc, section 3 and 5."""
    hidden_dim: int = QWEN_HIDDEN_DIM
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    tau: float = 0.07            # ETF cross-entropy temperature
    tau_con: float = 0.10        # SupCon temperature
    alpha: float = 0.35          # SupCon weight
    ce_weight: float = 1.0
    margin_weight: float = 0.5
    margin: float = 0.5          # required cosine gap: target vertex vs best distractor
    val_fraction: float = 0.2
    center: bool = True          # subtract train mean; removes the anisotropic cone offset
    seed: int = 0


class SimplexProbe(nn.Module if HAS_TORCH else object):
    """Bias-free linear probe onto a fixed Simplex ETF. ``w_proj`` is the only parameter."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = QWEN_HIDDEN_DIM,
        seed: int = 0,
        init_weights: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("torch is required for simplex probe training")
        if num_classes < 2:
            raise ValueError("Simplex ETF probe needs at least 2 classes")
        if hidden_dim < num_classes - 1:
            raise ValueError("hidden_dim must be >= num_classes - 1")
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if init_weights:
            # Orthogonal init (QR of Gaussian noise) scaled by sqrt(2/d), per the recipe.
            gen = torch.Generator().manual_seed(seed)
            q, r = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim, generator=gen))
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # fix QR sign ambiguity
            self.w_proj = nn.Parameter(q * math.sqrt(2.0 / hidden_dim))
        else:  # caller overwrites the weights (checkpoint load); skip the O(d^3) QR
            self.w_proj = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        vertices = torch.from_numpy(generate_simplex_etf(num_classes, hidden_dim)).float()
        self.register_buffer("etf", vertices, persistent=True)
        self.register_buffer("feature_mean", torch.zeros(hidden_dim), persistent=True)

    def embed(self, h: "torch.Tensor") -> "torch.Tensor":
        """Unit-norm query q on the hypersphere. Takes RAW backbone features; centers inside."""
        return F.normalize((h - self.feature_mean) @ self.w_proj.t(), dim=-1, eps=1e-8)

    def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Returns (q, cosines) where cosines[i, k] = <q_i, v_k>."""
        q = self.embed(h)
        return q, q @ self.etf.t()


def supervised_contrastive_loss(q: "torch.Tensor", labels: "torch.Tensor", tau_con: float) -> "torch.Tensor":
    """SupCon over a batch of unit vectors. Anchors with no positive are skipped."""
    n = q.shape[0]
    sim = (q @ q.t()) / tau_con
    self_mask = torch.eye(n, dtype=torch.bool, device=q.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return q.sum() * 0.0  # keeps the graph alive, contributes no gradient signal
    # masked_fill, not multiply: -inf on the diagonal would give nan via 0 * -inf.
    pos_log_prob = log_prob.masked_fill(~pos, 0.0).sum(dim=1)
    per_anchor = -pos_log_prob[valid] / pos_count[valid]
    return per_anchor.mean()


def etf_margin_loss(cosines: "torch.Tensor", labels: "torch.Tensor", margin: float) -> "torch.Tensor":
    """Hinge: target-vertex cosine must beat the best distractor by ``margin``."""
    target = cosines.gather(1, labels.unsqueeze(1)).squeeze(1)
    distractors = cosines.scatter(1, labels.unsqueeze(1), float("-inf"))
    gap = target - distractors.max(dim=1).values
    return F.relu(margin - gap).mean()


def probe_loss(
    probe: "SimplexProbe",
    h: "torch.Tensor",
    labels: "torch.Tensor",
    cfg: SimplexProbeConfig,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    q, cosines = probe(h)
    ce = F.cross_entropy(cosines / cfg.tau, labels)
    margin = etf_margin_loss(cosines, labels, cfg.margin)
    supcon = supervised_contrastive_loss(q, labels, cfg.tau_con)
    total = cfg.ce_weight * ce + cfg.margin_weight * margin + cfg.alpha * supcon
    return total, {"ce": ce.item(), "margin": margin.item(), "supcon": supcon.item()}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    features: np.ndarray          # (N, hidden_dim) float32
    labels: np.ndarray            # (N,) int64, ids index ETF vertices
    label_names: List[str]        # sorted canonical names, len == num_classes

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def encode_labels(raw: Sequence[Any], names: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Map raw labels to ids using the sorted label set, so vertex assignment is canonical."""
    raw_arr = np.asarray(raw)
    if names is None:
        names = sorted({str(x) for x in raw_arr.tolist()})
    else:
        names = [str(x) for x in names]
    index = {name: i for i, name in enumerate(names)}
    try:
        ids = np.array([index[str(x)] for x in raw_arr.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label {exc} is not in label_names") from None
    return ids, list(names)


def _first_key(data: Any, keys: Sequence[str], what: str) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"{what} array not found; expected one of {list(keys)}")


def make_synthetic_features(
    num_classes: int = 6,
    samples_per_class: int = 200,
    hidden_dim: int = QWEN_HIDDEN_DIM,
    signal: float = 1.0,
    seed: int = 0,
) -> FeatureSet:
    """Anisotropic stand-in for Qwen hidden states.

    Mimics the recipe's "narrow cone" finding: a large shared mean, noise energy that
    decays across dimensions, and a class signal that is small next to both. The raw
    zero-shot ETF projection therefore scores near chance, while a probe can recover it.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    rng = np.random.default_rng(seed)
    d = hidden_dim
    mean = rng.standard_normal(d)
    mean *= 8.0 / np.linalg.norm(mean)

    spectrum = 1.0 / np.sqrt(1.0 + np.arange(d) / 8.0)  # decaying noise scale
    spectrum *= 0.5 / spectrum.mean()

    centers = rng.standard_normal((num_classes, d))
    centers *= signal * 2.0 / np.linalg.norm(centers, axis=1, keepdims=True)

    n = num_classes * samples_per_class
    labels = np.repeat(np.arange(num_classes, dtype=np.int64), samples_per_class)
    noise = rng.standard_normal((n, d)) * spectrum
    feats = (mean + centers[labels] + noise).astype(np.float32)
    return FeatureSet(feats, labels, [f"class_{i:02d}" for i in range(num_classes)])


def load_features(
    source: str,
    labels_path: Optional[str] = None,
    hidden_dim: int = QWEN_HIDDEN_DIM,
    synthetic_classes: int = 6,
    synthetic_per_class: int = 200,
    seed: int = 0,
) -> FeatureSet:
    """Load cached backbone features, or build the synthetic set when source == "synthetic"."""
    if source == "synthetic":
        return make_synthetic_features(synthetic_classes, synthetic_per_class, hidden_dim, seed=seed)

    if not os.path.isfile(source):
        raise FileNotFoundError(f"feature file not found: {os.path.basename(source)}")

    names: Optional[Sequence[str]] = None
    if source.endswith(".npz"):
        with np.load(source, allow_pickle=False) as data:
            feats = np.asarray(_first_key(data, ("features", "hidden", "h", "X"), "features"))
            raw = np.asarray(_first_key(data, ("labels", "y", "Y"), "labels"))
            if "label_names" in data:
                names = [str(x) for x in data["label_names"].tolist()]
    elif source.endswith(".npy"):
        if not labels_path:
            raise ValueError(".npy features need --labels pointing at a labels .npy file")
        feats = np.load(source, allow_pickle=False)
        raw = np.load(labels_path, allow_pickle=False)
    else:
        raise ValueError("unsupported feature file; use .npz, or .npy plus --labels")

    if feats.ndim != 2 or feats.shape[1] != hidden_dim:
        raise ValueError(f"features must have shape (N, {hidden_dim}), got {feats.shape}")
    if raw.shape[0] != feats.shape[0]:
        raise ValueError(f"{feats.shape[0]} feature rows but {raw.shape[0]} labels")
    if not np.isfinite(feats).all():
        raise ValueError("features contain NaN or inf")

    ids, names = encode_labels(raw, names)
    return FeatureSet(np.ascontiguousarray(feats, dtype=np.float32), ids, names)


def extract_features(
    model_name: str,
    prompts: Sequence[str],
    labels: Sequence[Any],
    out_npz: str,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> None:
    """Phase 1: run the FROZEN backbone once and cache last-layer, last-token hidden states.

    Needs ``transformers`` and the model weights. The backbone is put in eval mode with
    ``requires_grad_(False)`` and runs under ``torch.no_grad()``; nothing here can update it.
    Writes an ``.npz`` that ``load_features`` reads. Not covered by unit tests (needs weights).
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for feature extraction")
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must have the same length")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for feature extraction") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # last position is then the last real token for every row
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.requires_grad_(False)

    rows: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = tokenizer(list(prompts[i:i + batch_size]), return_tensors="pt", padding=True).to(device)
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1][:, -1, :]
            rows.append(hidden.float().cpu().numpy())
    feats = np.concatenate(rows, axis=0)
    ids, names = encode_labels(labels)
    np.savez(out_npz, features=feats, labels=ids, label_names=np.array(names))


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class seeded split; every class with >= 2 samples keeps at least one in train."""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_val = int(round(len(idx) * val_fraction)) if len(idx) > 1 else 0
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx), dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(probe: "SimplexProbe", h: "torch.Tensor", labels: "torch.Tensor") -> float:
    if labels.numel() == 0:
        return float("nan")
    with torch.no_grad():
        _, cosines = probe(h)
    return float((cosines.argmax(dim=1) == labels).float().mean())


def train_probe(
    data: FeatureSet,
    cfg: Optional[SimplexProbeConfig] = None,
    log: Optional[Any] = None,
) -> Tuple["SimplexProbe", Dict[str, Any]]:
    """Train W_proj on cached features. Returns (probe, report)."""
    if not HAS_TORCH:
        raise ImportError("torch is required for simplex probe training")
    cfg = cfg or SimplexProbeConfig()
    if data.features.shape[1] != cfg.hidden_dim:
        raise ValueError(f"features have dim {data.features.shape[1]}, config expects {cfg.hidden_dim}")

    torch.manual_seed(cfg.seed)
    train_idx, val_idx = stratified_split(data.labels, cfg.val_fraction, cfg.seed)
    feats = torch.from_numpy(data.features)
    labels = torch.from_numpy(data.labels)
    x_train, y_train = feats[train_idx], labels[train_idx]
    x_val, y_val = feats[val_idx], labels[val_idx]

    probe = SimplexProbe(data.num_classes, cfg.hidden_dim, cfg.seed)
    if cfg.center:
        probe.feature_mean.copy_(x_train.mean(dim=0))
    trainable = [p for p in probe.parameters() if p.requires_grad]
    if len(trainable) != 1 or trainable[0] is not probe.w_proj:
        raise RuntimeError("only W_proj may be trainable")

    # Zero-shot baseline: identity projection onto the raw ETF (uncalibrated, recipe section 1).
    with torch.no_grad():
        saved_w, saved_mean = probe.w_proj.detach().clone(), probe.feature_mean.clone()
        probe.w_proj.copy_(torch.eye(cfg.hidden_dim))
        probe.feature_mean.zero_()
        zero_shot_val = evaluate(probe, x_val, y_val)
        probe.w_proj.copy_(saved_w)
        probe.feature_mean.copy_(saved_mean)

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []
    n_train = x_train.shape[0]
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        sums = {"loss": 0.0, "ce": 0.0, "margin": 0.0, "supcon": 0.0}
        batches = 0
        for i in range(0, n_train, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            if b.numel() < 2:  # SupCon needs a pair
                continue
            loss, parts = probe_loss(probe, x_train[b], y_train[b], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += v
            batches += 1
        probe.eval()
        row = {k: v / max(batches, 1) for k, v in sums.items()}
        row["epoch"] = epoch
        row["train_acc"] = evaluate(probe, x_train, y_train)
        row["val_acc"] = evaluate(probe, x_val, y_val)
        history.append(row)
        if log is not None:
            log(
                f"epoch {epoch:3d}/{cfg.epochs} loss={row['loss']:.4f} ce={row['ce']:.4f} "
                f"margin={row['margin']:.4f} supcon={row['supcon']:.4f} "
                f"train_acc={row['train_acc']:.3f} val_acc={row['val_acc']:.3f}"
            )

    probe.eval()
    val_margin = float("nan")
    if len(val_idx):
        with torch.no_grad():
            _, cos_val = probe(x_val)
            distract = cos_val.scatter(1, y_val.unsqueeze(1), float("-inf")).max(dim=1).values
            val_margin = float((cos_val.gather(1, y_val.unsqueeze(1)).squeeze(1) - distract).mean())

    report: Dict[str, Any] = {
        "config": asdict(cfg),
        "num_classes": data.num_classes,
        "label_names": data.label_names,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "zero_shot_val_acc": zero_shot_val,
        "probe_val_acc": history[-1]["val_acc"] if history else zero_shot_val,
        "probe_train_acc": history[-1]["train_acc"] if history else float("nan"),
        "val_target_margin": val_margin,
        "train_seconds": time.perf_counter() - start,
        "history": history,
    }
    return probe, report


def save_probe(probe: "SimplexProbe", report: Dict[str, Any], out_path: str) -> None:
    """Checkpoint holds tensors and plain JSON-able metadata only, so it loads with weights_only=True."""
    torch.save(
        {
            "w_proj": probe.w_proj.detach().cpu(),
            "feature_mean": probe.feature_mean.cpu(),
            "num_classes": probe.num_classes,
            "hidden_dim": probe.hidden_dim,
            "label_names": report["label_names"],
            "config": report["config"],
        },
        out_path,
    )


def load_probe(path: str) -> "SimplexProbe":
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    probe = SimplexProbe(ckpt["num_classes"], ckpt["hidden_dim"], init_weights=False)
    with torch.no_grad():
        probe.w_proj.copy_(ckpt["w_proj"])
        probe.feature_mean.copy_(ckpt["feature_mean"])
    return probe.eval()


def main(argv: Optional[Sequence[str]] = None) -> int:
    d = SimplexProbeConfig()
    p = argparse.ArgumentParser(description="Train a Simplex ETF linear probe on frozen Qwen3.5-9B features.")
    p.add_argument("--source", default="synthetic", help="'synthetic', a .npz file, or a .npy features file")
    p.add_argument("--labels", default=None, help="labels .npy (only with a .npy --source)")
    p.add_argument("--out", default=None, help="checkpoint path (.pt); omitted means do not save")
    p.add_argument("--report", default=None, help="write the JSON report here")
    p.add_argument("--hidden-dim", type=int, default=d.hidden_dim)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--alpha", type=float, default=d.alpha, help="SupCon weight")
    p.add_argument("--margin", type=float, default=d.margin)
    p.add_argument("--margin-weight", type=float, default=d.margin_weight)
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--tau-con", type=float, default=d.tau_con)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--no-center", action="store_true", help="skip train-mean centering")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--synthetic-classes", type=int, default=6)
    p.add_argument("--synthetic-per-class", type=int, default=200)
    args = p.parse_args(argv)

    cfg = SimplexProbeConfig(
        hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, margin=args.margin, margin_weight=args.margin_weight, tau=args.tau,
        tau_con=args.tau_con, val_fraction=args.val_fraction, center=not args.no_center, seed=args.seed,
    )
    data = load_features(
        args.source, args.labels, cfg.hidden_dim,
        args.synthetic_classes, args.synthetic_per_class, args.seed,
    )
    print(f"source={'synthetic' if args.source == 'synthetic' else os.path.basename(args.source)} "
          f"samples={len(data.labels)} classes={data.num_classes} dim={cfg.hidden_dim}")
    probe, report = train_probe(data, cfg, log=print)
    print(f"zero-shot val acc={report['zero_shot_val_acc']:.4f}  "
          f"probe val acc={report['probe_val_acc']:.4f}  "
          f"trainable params={report['trainable_params']}")
    if args.out:
        save_probe(probe, report, args.out)
        print(f"saved probe to {os.path.basename(args.out)}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
