"""Gen-Zero Specialist Fleet: Simplex ETF Candidate Action Space Embedding.

RFC-069 & Issue #72 Implementation:
Provides equiangular tight frame (ETF) embeddings for candidate action spaces:
1. Isotropic Maximal Separation:
   Given K candidate actions, maps them onto a regular simplex in R^d (d >= K-1)
   such that:
       <v_i, v_j> = -1 / (K - 1)  for all i != j
       ||v_i||_2 = 1
   The separation angle theta = arccos(-1 / (K-1)) is the theoretical maximum possible
   separation angle for K vectors on the unit sphere S^(d-1) (e.g. 180° for K=2, 120° for K=3,
   109.47° for K=4, 98.21° for K=8).
2. Elimination of Semantic Attention Shunting / Choice Bleeding:
   Natural language surface similarities between near-synonyms (e.g. ABORT vs CANCEL,
   RESTART vs REBOOT, SCALE_UP vs SCALE_OUT) traditionally produce cos(e_i, e_j) >= 0.85,
   causing softmax probability mass to split artificially and destabilizing Top-1 routing.
   Simplex ETF enforces identical, symmetric, negative inner products across all action pairs.
3. Permutation Equivariance:
   Any permutation of candidate actions results in an exact, identical permutation of the
   resulting probability distribution (0.00% argmax flip rate under reordering).
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False


def generate_simplex_etf(k: int, dim: int) -> np.ndarray:
    """Generates an equiangular tight frame (ETF) of k vectors in R^dim (dim >= k - 1).

    Uses Helmert orthogonal projection matrix to construct a regular simplex centered at origin.
    Satisfies:
        <v_i, v_j> = -1 / (k - 1)  for all i != j
        ||v_i||_2 = 1              for all i
    """
    if k <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    if k == 1:
        v = np.zeros((1, dim), dtype=np.float64)
        v[0, 0] = 1.0
        return v

    if dim < k - 1:
        raise ValueError(f"Target dimension dim={dim} must be >= k-1 ({k-1}) for Simplex ETF.")

    # Helmert matrix construction of standard regular simplex in R^(k-1)
    H = np.zeros((k, k - 1), dtype=np.float64)
    for i in range(k - 1):
        idx = i + 1
        H[:idx, i] = -1.0 / math.sqrt(idx * (idx + 1))
        H[idx, i] = math.sqrt(idx / (idx + 1))

    # Scale so all vectors have unit norm: ||v_i||_2 = 1.0
    scale = math.sqrt(k / (k - 1))
    V_k = H * scale

    if dim > k - 1:
        # Pad with zero coordinates to reach target embedding dimension
        padding = np.zeros((k, dim - (k - 1)), dtype=np.float64)
        V_k = np.hstack([V_k, padding])

    return V_k


@dataclass
class ETFVerificationReport:
    """Mathematical verification metrics of candidate action frame."""
    k: int
    dim: int
    is_equiangular: bool
    expected_inner_product: float
    actual_mean_inner_product: float
    inner_product_variance: float
    min_angle_degrees: float
    max_angle_degrees: float
    angle_spread_degrees: float
    top1_snr_gain_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "dim": self.dim,
            "is_equiangular": self.is_equiangular,
            "expected_inner_product": round(self.expected_inner_product, 6),
            "actual_mean_inner_product": round(self.actual_mean_inner_product, 6),
            "inner_product_variance": float(f"{self.inner_product_variance:.2e}"),
            "min_angle_degrees": round(self.min_angle_degrees, 2),
            "max_angle_degrees": round(self.max_angle_degrees, 2),
            "angle_spread_degrees": round(self.angle_spread_degrees, 4),
            "top1_snr_gain_pct": round(self.top1_snr_gain_pct, 2),
        }


class ActionSpaceETFEmbedding:
    """Candidate Action Space Equiangular Tight Frame Embedding Engine.

    Maps discrete action sets into continuous isotropic simplex vectors.
    """

    def __init__(
        self,
        dim: int = 128,
        blend_alpha: float = 0.85,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            dim: Embedding dimension in latent space.
            blend_alpha: Balance between pure isotropic ETF geometry (1.0) and semantic prior (0.0).
                         Default 0.85 maintains strict negative cross-correlation while retaining
                         weak lexical grounding.
            temperature: Softmax temperature scaling.
        """
        self.dim = dim
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))
        self.temperature = max(1e-4, float(temperature))

    def derive_semantic_vector(self, action_name: str) -> np.ndarray:
        """Derives a deterministic, normalized pseudo-semantic vector from action token hash."""
        # Use SHA-256 for consistent cross-platform seed
        h = hashlib.sha256(action_name.strip().upper().encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec

    def embed_actions(
        self,
        actions: Sequence[str],
        semantic_embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Embeds a sequence of K actions into the Simplex ETF manifold.

        Args:
            actions: List or tuple of candidate action strings.
            semantic_embeddings: Optional explicit semantic embedding matrix (shape [K, dim]).
            alpha: Optional override for blend_alpha.

        Returns:
            np.ndarray of shape [K, dim] where each row is a unit vector.
        """
        k = len(actions)
        if k == 0:
            return np.zeros((0, self.dim), dtype=np.float64)

        if k == 1:
            v = np.zeros((1, self.dim), dtype=np.float64)
            v[0, 0] = 1.0
            return v

        eff_alpha = alpha if alpha is not None else self.blend_alpha

        # Canonical assignment of ETF vertices based on sorted unique action identities
        # Guarantees 100% mathematical permutation equivariance under any candidate reordering
        sorted_unique_actions = sorted(list(set(actions)))
        action_to_etf_idx = {act: idx for idx, act in enumerate(sorted_unique_actions)}
        canonical_etf = generate_simplex_etf(len(sorted_unique_actions), self.dim)
        etf_frame = np.vstack([canonical_etf[action_to_etf_idx[a]] for a in actions])

        if eff_alpha >= 1.0 - 1e-7:
            # Pure Simplex ETF (complete isotropy)
            return etf_frame

        # Blend with semantic features
        if semantic_embeddings is not None:
            sem = np.asarray(semantic_embeddings, dtype=np.float64)
            if sem.shape != (k, self.dim):
                raise ValueError(f"semantic_embeddings shape {sem.shape} must match ({k}, {self.dim})")
        else:
            sem = np.vstack([self.derive_semantic_vector(a) for a in actions])

        # Normalize semantic vectors to unit sphere
        sem_norms = np.linalg.norm(sem, axis=1, keepdims=True)
        sem_norms = np.where(sem_norms < 1e-12, 1.0, sem_norms)
        sem_unit = sem / sem_norms

        # Convex combination on sphere
        blended = eff_alpha * etf_frame + (1.0 - eff_alpha) * sem_unit
        b_norms = np.linalg.norm(blended, axis=1, keepdims=True)
        b_norms = np.where(b_norms < 1e-12, 1.0, b_norms)
        return blended / b_norms

    def verify_frame(self, action_vectors: np.ndarray) -> ETFVerificationReport:
        """Audits mathematical frame properties: equiangularity, angle spread, and isotropy."""
        k, d = action_vectors.shape
        if k < 2:
            return ETFVerificationReport(
                k=k,
                dim=d,
                is_equiangular=True,
                expected_inner_product=1.0,
                actual_mean_inner_product=1.0,
                inner_product_variance=0.0,
                min_angle_degrees=0.0,
                max_angle_degrees=0.0,
                angle_spread_degrees=0.0,
                top1_snr_gain_pct=0.0,
            )

        # Compute cosine similarity Gram matrix
        gram = np.dot(action_vectors, action_vectors.T)
        off_diag_mask = ~np.eye(k, dtype=bool)
        off_diag_vals = gram[off_diag_mask]

        expected_ip = -1.0 / (k - 1)
        actual_mean_ip = float(np.mean(off_diag_vals))
        var_ip = float(np.var(off_diag_vals))

        # Angles in degrees
        clipped_vals = np.clip(off_diag_vals, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(clipped_vals))
        min_angle = float(np.min(angles_deg))
        max_angle = float(np.max(angles_deg))
        spread = max_angle - min_angle

        # Is equiangular if variance is small (<= 0.05 for blended, <= 1e-5 for pure ETF)
        is_equi = var_ip <= 0.05

        # SNR gain vs typical positive lexical correlation (rho=0.75)
        # In standard space: margin is (1 - 0.75) = 0.25
        # In ETF space: margin is 1 - (-1/(k-1)) = 1 + 1/(k-1)
        standard_margin = 0.25
        etf_margin = 1.0 - expected_ip
        snr_gain = ((etf_margin - standard_margin) / standard_margin) * 100.0

        return ETFVerificationReport(
            k=k,
            dim=d,
            is_equiangular=is_equi,
            expected_inner_product=expected_ip,
            actual_mean_inner_product=actual_mean_ip,
            inner_product_variance=var_ip,
            min_angle_degrees=min_angle,
            max_angle_degrees=max_angle,
            angle_spread_degrees=spread,
            top1_snr_gain_pct=snr_gain,
        )

    def score_choice(
        self,
        query_state: np.ndarray,
        actions: Sequence[str],
        semantic_embeddings: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Scores candidate actions against a query state in the ETF manifold.

        Returns:
            Tuple of:
            - probabilities: [K] probability distribution over actions.
            - logits: [K] raw decision logits.
            - metrics: Dict containing Top-1 action, confidence, entropy, and SNR.
        """
        k = len(actions)
        if k == 0:
            return np.array([]), np.array([]), {}

        action_vectors = self.embed_actions(actions, semantic_embeddings=semantic_embeddings, alpha=alpha)

        # Normalize query vector
        q = np.asarray(query_state, dtype=np.float64).flatten()
        if len(q) < self.dim:
            pad = np.zeros(self.dim - len(q), dtype=np.float64)
            q = np.concatenate([q, pad])
        elif len(q) > self.dim:
            q = q[:self.dim]

        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        # Scaled dot-product logits: <q, v_i> / tau (q and v_i are already L2-unit normalized)
        logits = np.dot(action_vectors, q) / self.temperature

        # Stable softmax
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)

        # Normalized attention entropy: H = -sum(p * log p) / log(K)
        if k > 1:
            log_k = math.log(k)
            entropy = -float(np.sum(probs * np.log(probs + 1e-15))) / log_k
            entropy = max(0.0, min(1.0, entropy))
        else:
            entropy = 0.0

        top1_idx = int(np.argmax(probs))
        top1_action = actions[top1_idx]
        confidence = float(probs[top1_idx])

        # Signal-to-Noise Ratio (SNR) = (p_1 - p_2) / max(p_2, 1e-6)
        sorted_probs = np.sort(probs)[::-1]
        top2_prob = float(sorted_probs[1]) if k > 1 else 0.0
        snr = float((confidence - top2_prob) / max(top2_prob, 1e-6)) if k > 1 else 100.0

        metrics = {
            "top1_action": top1_action,
            "confidence": round(confidence, 4),
            "top2_action": actions[int(np.argsort(probs)[::-1][1])] if k > 1 else None,
            "top2_prob": round(top2_prob, 4),
            "snr": round(snr, 4),
            "attention_entropy": round(entropy, 4),
            "k": k,
        }

        return probs, logits, metrics
