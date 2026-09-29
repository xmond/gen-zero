"""NanoCore Layer Scaling Benchmark & Block Influence Evaluation.

Implements Milestone 1 of Issue #23:
- Block Influence (BI): Measures layer-to-layer representation cosine distance:
  BI(l) = 1 - (1/T) sum_{t=1}^T cos(x_t^l, x_t^{l+1})
- Linear Separability: Computes representation separability to avoid premature shallow truncation
  in anisotropic embedding cones.
- Pareto Frontier Profiler: Evaluates L1, L2, L4, L6 trade-offs across latency, memory, ECE, and accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple
import dataclasses
import math
import time
import numpy as np

from gen_zero.manifold import MasterClosedFormSolver


@dataclasses.dataclass
class LayerMetrics:
    layer_depth: int
    block_influence: float
    linear_separability: float
    accuracy: float
    p99_latency_ms: float
    memory_mb: float
    ece_10bin: float
    pareto_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_depth": self.layer_depth,
            "block_influence": round(self.block_influence, 4),
            "linear_separability": round(self.linear_separability, 4),
            "accuracy": round(self.accuracy, 4),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "memory_mb": round(self.memory_mb, 2),
            "ece_10bin": round(self.ece_10bin, 4),
            "pareto_score": round(self.pareto_score, 4),
        }


def compute_block_influence(
    hidden_states_l: np.ndarray,
    hidden_states_next: np.ndarray,
) -> float:
    """Computes Block Influence (BI) between two adjacent layer hidden representations.

    BI(l) = 1 - (1/T) sum_{t=1}^T cos(x_t^l, x_t^{l+1})

    Args:
        hidden_states_l: Array of shape [T, D] or [B, T, D]
        hidden_states_next: Array of shape [T, D] or [B, T, D]
    Returns:
        Scalar BI in [0.0, 2.0] where higher implies greater transformation contribution.
    """
    raw_x1 = np.asarray(hidden_states_l, dtype=np.float32)
    raw_x2 = np.asarray(hidden_states_next, dtype=np.float32)
    if raw_x1.ndim < 2 or raw_x2.ndim < 2 or raw_x1.shape != raw_x2.shape:
        raise ValueError(
            "adjacent hidden states must have matching non-empty [.., tokens, features] shapes"
        )
    if raw_x1.shape[-1] == 0 or raw_x1.size == 0:
        raise ValueError("hidden states must contain at least one token and feature")
    if not np.isfinite(raw_x1).all() or not np.isfinite(raw_x2).all():
        raise ValueError("hidden states must contain only finite values")
    x1 = raw_x1.reshape(-1, raw_x1.shape[-1])
    x2 = raw_x2.reshape(-1, raw_x2.shape[-1])

    # Norms
    norm1 = np.linalg.norm(x1, axis=-1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(x2, axis=-1, keepdims=True) + 1e-8

    u1 = x1 / norm1
    u2 = x2 / norm2

    # Cosine similarities
    cos_sim = np.clip(np.sum(u1 * u2, axis=-1), -1.0, 1.0)
    mean_cos = float(np.mean(cos_sim))

    # BI is 1 - mean_cos
    bi = 1.0 - mean_cos
    return max(0.0, bi)


def compute_linear_separability(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    test_fraction: float = 0.25,
    random_state: int = 0,
) -> float:
    """Computes held-out linear-probe accuracy on frozen embeddings.

    The previous implementation fitted and scored on the same rows, allowing a
    high-dimensional probe to report memorization as ``separability``.  A
    deterministic stratified holdout keeps this metric usable in small CPU
    benchmarks while making the reported value an out-of-sample measurement.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("embeddings must be 2-D and labels must align as a 1-D array")
    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0 or num_classes <= 1:
        raise ValueError(
            "compute_linear_separability requires non-empty embeddings and 2 classes; "
            f"got n_samples={n_samples}, num_classes={num_classes}"
        )
    if not np.isfinite(X).all():
        raise ValueError("embeddings must contain only finite values")
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("labels must be integer class indices in [0, num_classes)")
    if not np.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    observed_classes = np.unique(y)
    if len(observed_classes) < 2:
        raise ValueError("compute_linear_separability requires at least two observed classes")
    if any(np.sum(y == cls) < 2 for cls in observed_classes):
        raise ValueError("each observed class needs at least two samples for a held-out probe")

    rng = np.random.default_rng(random_state)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for cls in observed_classes:
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        test_count = max(1, int(round(len(cls_indices) * test_fraction)))
        test_count = min(test_count, len(cls_indices) - 1)
        test_indices.extend(int(i) for i in cls_indices[:test_count])
        train_indices.extend(int(i) for i in cls_indices[test_count:])

    train_idx = np.asarray(train_indices, dtype=np.intp)
    test_idx = np.asarray(test_indices, dtype=np.intp)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit preprocessing on the training split only, then apply it to held-out
    # rows.  This avoids leaking test distribution statistics into the probe.
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_norm = (X_train - mean) / (std + 1e-6)
    X_test_norm = (X_test - mean) / (std + 1e-6)

    # Simple multiclass ridge pseudo-inverse probe
    Y_onehot = np.zeros((len(train_idx), num_classes), dtype=np.float32)
    for i, label in enumerate(y_train):
        Y_onehot[i, label] = 1.0

    # Ridge regression (standardised features, so no intercept), solved by Cholesky.
    probe = MasterClosedFormSolver()
    probe.fit(X_train_norm, Y_onehot, lambda_reg=1e-2, fit_intercept=False)
    preds = np.argmax(probe.predict(X_test_norm), axis=1)
    acc = float(np.mean(preds == y_test))

    return float(np.clip(acc, 0.0, 1.0))


class NanoCoreLayerScalingBenchmark:
    """Evaluates pareto scaling frontier for specialized micro-cores."""

    def __init__(self, target_domains: Optional[List[str]] = None):
        self.target_domains = target_domains or ["DOM", "Code", "Ops", "Market"]

    def evaluate_pareto_frontier(
        self,
        measurements: Dict[int, Dict[str, float]],
    ) -> List[LayerMetrics]:
        """Runs Pareto scaling evaluation across caller-supplied per-depth measurements.

        Args:
            measurements: Mapping of layer depth (e.g. 1, 2, 4, 6) to a dict of
                measured metrics with keys: block_influence, linear_separability,
                accuracy, p99_latency_ms, memory_mb, ece_10bin.

        Raises:
            ValueError: If measurements is empty or any entry is missing a required key.
        """
        required_keys = {
            "block_influence", "linear_separability", "accuracy",
            "p99_latency_ms", "memory_mb", "ece_10bin",
        }

        if not measurements:
            raise ValueError("evaluate_pareto_frontier requires at least one depth measurement")

        results: List[LayerMetrics] = []

        for d in sorted(measurements.keys()):
            m = measurements[d]
            missing = required_keys - m.keys()
            if missing:
                raise ValueError(
                    f"measurements[{d}] is missing required keys: {sorted(missing)}"
                )

            bi = float(m["block_influence"])
            sep = float(m["linear_separability"])
            acc = float(m["accuracy"])
            p99_lat = float(m["p99_latency_ms"])
            mem = float(m["memory_mb"])
            ece = float(m["ece_10bin"])

            values = {
                "block_influence": bi,
                "linear_separability": sep,
                "accuracy": acc,
                "p99_latency_ms": p99_lat,
                "memory_mb": mem,
                "ece_10bin": ece,
            }
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(f"measurements[{d}] must contain only finite values")
            if not 0.0 <= bi <= 2.0:
                raise ValueError(f"measurements[{d}]['block_influence'] must be in [0, 2]")
            if not 0.0 <= sep <= 1.0 or not 0.0 <= acc <= 1.0 or not 0.0 <= ece <= 1.0:
                raise ValueError(
                    f"measurements[{d}] requires linear_separability, accuracy and ece_10bin in [0, 1]"
                )
            if p99_lat <= 0.0 or mem <= 0.0:
                raise ValueError(f"measurements[{d}] requires positive latency and memory")

            # Composite Pareto Score: rewards acc and separability, penalizes latency, memory, and ECE
            # Pareto Score = Acc * Sep / (Latency_ms * (Memory_MB / 10) * (1 + 10 * ECE))
            pareto_score = (acc * sep) / (p99_lat * (mem / 20.0) * (1.0 + 10.0 * ece))

            results.append(LayerMetrics(
                layer_depth=d,
                block_influence=bi,
                linear_separability=sep,
                accuracy=acc,
                p99_latency_ms=p99_lat,
                memory_mb=mem,
                ece_10bin=ece,
                pareto_score=pareto_score,
            ))

        return results

    def generate_pareto_report_markdown(self, metrics: List[LayerMetrics]) -> str:
        """Generates markdown Pareto frontier evaluation table from measured metrics."""
        lines = [
            "# NanoCore Layer Scaling & Block Influence Benchmark Report",
            "",
            "| Layer Depth | Block Influence | Linear Sep | Top-1 Acc | P99 Latency (ms) | Memory (MB) | 10-Bin ECE | Pareto Score | SLA Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        sla_passing = [
            m for m in metrics
            if m.p99_latency_ms <= 1.4 and m.memory_mb <= 20.0 and m.ece_10bin <= 0.035
        ]
        best_score = max((m.pareto_score for m in sla_passing), default=-1.0)
        for m in metrics:
            sla_pass = (m.p99_latency_ms <= 1.4 and m.memory_mb <= 20.0 and m.ece_10bin <= 0.035)
            is_optimal = (m.pareto_score == best_score)
            verdict = "PASS (SWEET-SPOT)" if (is_optimal and sla_pass) else ("PASS" if sla_pass else "VIOLATES_SLA")

            lines.append(
                f"| **L{m.layer_depth}** | {m.block_influence:.4f} | {m.linear_separability:.4f} | "
                f"{m.accuracy * 100:.1f}% | {m.p99_latency_ms:.2f}ms | {m.memory_mb:.1f}MB | "
                f"{m.ece_10bin:.4f} | **{m.pareto_score:.2f}** | `{verdict}` |"
            )

        # Derive conclusions from the measured metrics instead of asserting fixed numbers.
        conclusion_lines = ["", "## Architecture Conclusions"]
        sla_passing_depths = sorted(m.layer_depth for m in sla_passing)
        if sla_passing_depths:
            best_depth = next(m.layer_depth for m in metrics if m.pareto_score == best_score)
            conclusion_lines.append(
                f"- **Depths meeting SLA (P99 <= 1.4ms, Memory <= 20MB, ECE <= 0.035)**: "
                f"{', '.join(f'L{d}' for d in sla_passing_depths)}."
            )
            conclusion_lines.append(
                f"- **Pareto-optimal depth among SLA-passing candidates**: L{best_depth} "
                f"(pareto score {best_score:.2f})."
            )
        else:
            conclusion_lines.append(
                "- **No candidate depth meets the SLA** (P99 <= 1.4ms, Memory <= 20MB, ECE <= 0.035) "
                "for the supplied measurements."
            )
        violating_depths = sorted(
            m.layer_depth for m in metrics
            if not (m.p99_latency_ms <= 1.4 and m.memory_mb <= 20.0 and m.ece_10bin <= 0.035)
        )
        if violating_depths:
            conclusion_lines.append(
                f"- **Depths violating SLA**: {', '.join(f'L{d}' for d in violating_depths)}."
            )
        lines.extend(conclusion_lines)

        return "\n".join(lines)
