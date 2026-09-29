"""Gen-Zero Streaming Spatial-Temporal World Model Engine.

Integrates:
1. RollingVisionKVCache (Streaming Attention Sinks + Sliding Window).
2. LatentTransitionModel (Non-autoregressive dynamics & Pearl causal shock).
3. ImaginationMCTSPlanner (Ultra-fast <1.5ms latent Monte Carlo Tree Search).
4. Direct candidate scoring and streaming decision loop.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from collections import OrderedDict
import hashlib
import logging
import time
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .rolling_kv_cache import RollingVisionKVCache
from .latent_dynamics import LatentTransitionModel, resolve_torch_device
from .imagination_planner import ImaginationMCTSPlanner

logger = logging.getLogger("gen_zero.world_model.streaming_engine")

# Observation provenance. Only RAW_FEATURE_VECTOR is a real latent, passed through unchanged
# because it already has exactly latent_dim entries. Anything else is a degraded placeholder:
# the hash prior for text/objects, or the dimension projector for a numeric vector whose length
# does not match latent_dim (see PROVENANCE_PROJECTED_FEATURE_VECTOR below).
PROVENANCE_RAW_FEATURE_VECTOR: str = "raw_feature_vector"
PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR: str = "untrained_text_hash_prior"
DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR: str = "UNTRAINED_TEXT_HASH_PRIOR"

# A numeric vector whose length != latent_dim used to be silently zero-padded or truncated
# and still tagged PROVENANCE_RAW_FEATURE_VECTOR (degraded=False): padding fabricates entries
# out of nothing and truncation drops real ones, both while claiming to be the real thing.
# Now any length mismatch instead goes through an explicit, deterministic linear projector
# (see _get_dim_projector_matrix) and is tagged with this provenance and degraded=True: it is
# a real linear map over every input dimension, but it is an untrained, uncalibrated map, not
# a learned manifold encoder.
PROVENANCE_PROJECTED_FEATURE_VECTOR: str = "projected_untrained_linear_projection"
DEGRADATION_DIM_MISMATCH_PROJECTED: str = "DIM_MISMATCH_LINEAR_PROJECTED"

# _dim_projectors caches one latent_dim x in_dim matrix per distinct input length ever seen.
# Unbounded, a caller sending many distinct observation lengths (or one absurdly long one)
# could grow this cache without limit. MAX_DIM_PROJECTOR_CACHE_ENTRIES bounds the cache to an
# LRU of this many distinct dimensions; MAX_PROJECTOR_INPUT_DIM rejects any single dimension
# large enough to make even one matrix expensive (32768 * 1024 floats = 128MiB per entry).
MAX_DIM_PROJECTOR_CACHE_ENTRIES: int = 32
MAX_PROJECTOR_INPUT_DIM: int = 32768


class StreamingWorldModelEngine:
    """High-level streaming decision engine with spatial-temporal lookahead."""

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 32,
        num_sink_tokens: int = 4,
        window_size: int = 8,
        max_simulations: int = 64,
        device: str = "cpu"
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = resolve_torch_device(device)

        # 1. Rolling KV-Cache
        self.kv_cache = RollingVisionKVCache(
            num_sink_tokens=num_sink_tokens,
            window_size=window_size,
            feature_dim=latent_dim,
            device=self.device
        )

        # 2. Latent Dynamics
        self.transition_model = LatentTransitionModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            device=self.device
        )

        # 3. Latent MCTS Planner
        self.planner = ImaginationMCTSPlanner(
            transition_model=self.transition_model,
            max_simulations=max_simulations,
            max_depth=4
        )

        # Episode state tracker
        self.last_latent: Optional[Any] = None
        self.last_action: Optional[Any] = None
        self.step_count = 0

        # Cache of deterministic dimension-mismatch projector matrices, keyed by input length.
        # LRU-bounded to MAX_DIM_PROJECTOR_CACHE_ENTRIES; see _get_dim_projector_matrix.
        self._dim_projectors: "OrderedDict[int, Any]" = OrderedDict()

    def reset_stream(self) -> None:
        """Flushes the stream cache and state for a new episode."""
        self.kv_cache.reset()
        self.last_latent = None
        self.last_action = None
        self.step_count = 0

    def step_stream(
        self,
        observation: Any,
        candidate_actions: List[Any],
        action_priors: Optional[Dict[Any, float]] = None,
        context_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingests a new streaming observation, updates spatial-temporal state,

        computes Pearl causal shock, and plans the optimal action via latent MCTS.
        
        Args:
            observation: Image path, raw tensor, or feature array representing the current frame.
            candidate_actions: List of possible candidate actions.
            action_priors: Optional prior probability map from fast reflex head.
            context_prompt: Optional textual context.
        """
        t0 = time.perf_counter()
        degradations: List[str] = []

        # 1. Extract / Normalize Latent State z_t (1024-dim)
        t_enc_start = time.perf_counter()
        curr_z, provenance = self._encode_observation(observation)
        t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
        degraded = provenance in (PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR, PROVENANCE_PROJECTED_FEATURE_VECTOR)
        if provenance == PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR:
            logger.warning(
                "%s: observation of type %s is not a feature vector; the latent is a hash prior "
                "and the transition model has no language-manifold calibration.",
                DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR, type(observation).__name__,
            )
            degradations.append(DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR)
        elif provenance == PROVENANCE_PROJECTED_FEATURE_VECTOR:
            logger.warning(
                "%s: observation vector length does not match latent_dim=%d; routed through an "
                "untrained deterministic linear projector instead of truncating or zero-padding.",
                DEGRADATION_DIM_MISMATCH_PROJECTED, self.latent_dim,
            )
            degradations.append(DEGRADATION_DIM_MISMATCH_PROJECTED)

        # 2. Calculate Pearl Exogenous Causal Shock ||u_t|| if previous step exists
        causal_shock_norm = 0.0
        if self.last_latent is not None and self.last_action is not None:
            causal_shock_norm, _ = self.transition_model.compute_causal_shock(
                prior_latent=self.last_latent,
                action=self.last_action,
                real_next_latent=curr_z
            )

        # 3. Latent Space Imagination MCTS
        t_plan_start = time.perf_counter()
        plan_res = self.planner.plan(
            root_latent=curr_z,
            candidate_actions=candidate_actions,
            action_priors=action_priors
        )
        t_plan_ms = (time.perf_counter() - t_plan_start) * 1000.0
        best_action = plan_res["best_action"]

        # 4. Commit session state only now that decoding, validation and planning all
        # succeeded: a request that raises above must leave step_count, the rolling KV-cache
        # and the causal-shock anchor exactly as they were.
        self.kv_cache.append(key=curr_z, value=curr_z, timestamp=time.time())
        kv_stats = self.kv_cache.get_temporal_context()
        self.last_latent = curr_z
        self.last_action = best_action
        self.step_count += 1

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "step": self.step_count,
            "selected_action": best_action,
            "degraded": degraded,
            "provenance": provenance,
            "degradations": degradations,
            "action_probabilities": plan_res["action_probabilities"],
            "expected_value": plan_res["expected_value"],
            "causal_shock_norm": round(causal_shock_norm, 4),
            "imagined_trajectory": plan_res["imagined_trajectory"],
            "kv_cache_stats": kv_stats,
            "latency_breakdown_ms": {
                "encoding_ms": round(t_enc_ms, 3),
                "imagination_mcts_ms": round(t_plan_ms, 3),
                "total_ms": round(total_elapsed_ms, 3)
            }
        }

    def _get_dim_projector_matrix(self, in_dim: int) -> Any:
        """Deterministic Johnson-Lindenstrauss-style random projection matrix R^{latent_dim x in_dim}.

        Cached per input length. The seed depends only on (in_dim, latent_dim), never on the
        data being projected, so the same input length always gets the same matrix within a
        process (and across processes/runs). This is NOT a trained encoder -- it exists only
        so a dimension mismatch is handled by a real linear map over every input dimension,
        instead of the previous silent truncate/zero-pad that fabricated or dropped entries
        while still claiming ``PROVENANCE_RAW_FEATURE_VECTOR``. Callers of this projection are
        always tagged ``PROVENANCE_PROJECTED_FEATURE_VECTOR`` and ``degraded=True``.

        The cache is bounded to ``MAX_DIM_PROJECTOR_CACHE_ENTRIES`` distinct dimensions (LRU
        eviction) and refuses to build a matrix for ``in_dim > MAX_PROJECTOR_INPUT_DIM``: both
        guard against a caller growing server memory without bound by sending many distinct
        observation lengths, or one absurdly long one.
        """
        cached = self._dim_projectors.get(in_dim)
        if cached is not None:
            self._dim_projectors.move_to_end(in_dim)
            return cached
        if in_dim <= 0:
            raise ValueError(f"cannot project an empty observation vector (in_dim={in_dim})")
        if in_dim > MAX_PROJECTOR_INPUT_DIM:
            raise ValueError(
                f"observation vector length {in_dim} exceeds the maximum supported dimension "
                f"{MAX_PROJECTOR_INPUT_DIM}; refusing to allocate a "
                f"{self.latent_dim}x{in_dim} projector matrix."
            )
        seed = (in_dim * 1_000_003 + self.latent_dim * 97 + 1) & 0xFFFFFFFF
        scale = 1.0 / math.sqrt(in_dim)
        if HAS_TORCH:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            w = (torch.randn(self.latent_dim, in_dim, generator=gen) * scale).to(
                device=self.device, dtype=torch.float32
            )
        elif HAS_NUMPY:
            rng = np.random.default_rng(seed)
            w = rng.standard_normal((self.latent_dim, in_dim)).astype(np.float32) * np.float32(scale)
        else:
            import random as _random
            r = _random.Random(seed)
            w = [[r.gauss(0.0, 1.0) * scale for _ in range(in_dim)] for _ in range(self.latent_dim)]
        self._dim_projectors[in_dim] = w
        if len(self._dim_projectors) > MAX_DIM_PROJECTOR_CACHE_ENTRIES:
            self._dim_projectors.popitem(last=False)
        return w

    def _apply_dim_projector(self, values: Any, in_dim: int) -> Any:
        """Projects a length-``in_dim`` numeric vector onto R^latent_dim via the cached matrix."""
        w = self._get_dim_projector_matrix(in_dim)
        if HAS_TORCH and isinstance(w, torch.Tensor):
            x = values if isinstance(values, torch.Tensor) else torch.tensor(
                list(values), dtype=torch.float32
            )
            x = x.to(device=self.device, dtype=torch.float32)
            return w @ x
        elif HAS_NUMPY and isinstance(w, np.ndarray):
            x = values if isinstance(values, np.ndarray) else np.array(list(values), dtype=np.float32)
            return w @ x.astype(np.float32)
        else:
            x = list(values)
            return [sum(w_row[i] * x[i] for i in range(in_dim)) for w_row in w]

    def _encode_observation(self, obs: Any) -> Tuple[Any, str]:
        """Encodes or normalizes incoming observation into a latent_dim-dim latent vector.

        Returns ``(latent, provenance)``. A numeric vector already of length latent_dim passes
        through as ``PROVENANCE_RAW_FEATURE_VECTOR`` (not degraded). A numeric vector of any
        other length is routed through the deterministic projector above and tagged
        ``PROVENANCE_PROJECTED_FEATURE_VECTOR`` (degraded: see _get_dim_projector_matrix).
        Anything else (str, dict, arbitrary object) becomes a deterministic hash prior tagged
        ``PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR``: there is no text encoder here and the
        transition model was never calibrated on language.
        """
        if HAS_TORCH and isinstance(obs, torch.Tensor):
            t = obs.to(device=self.device, dtype=torch.float32)
            if t.ndim > 1:
                t = t.flatten()
            if len(t) == 0:
                raise ValueError("observation tensor must not be empty")
            if not bool(torch.isfinite(t).all()):
                raise ValueError("observation tensor must contain only finite values")
            if len(t) == self.latent_dim:
                return t, PROVENANCE_RAW_FEATURE_VECTOR
            return self._apply_dim_projector(t, len(t)), PROVENANCE_PROJECTED_FEATURE_VECTOR
        elif HAS_NUMPY and isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32).flatten()
            if len(arr) == 0:
                raise ValueError("observation array must not be empty")
            if not bool(np.isfinite(arr).all()):
                raise ValueError("observation array must contain only finite values")
            if len(arr) == self.latent_dim:
                return arr, PROVENANCE_RAW_FEATURE_VECTOR
            return self._apply_dim_projector(arr, len(arr)), PROVENANCE_PROJECTED_FEATURE_VECTOR
        elif isinstance(obs, (list, tuple)):
            arr = list(obs)
            if len(arr) == 0:
                raise ValueError("observation list must not be empty")
            for idx, v in enumerate(arr):
                try:
                    finite = math.isfinite(float(v))
                except (OverflowError, TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError(f"observation[{idx}] must be a finite number")
            if len(arr) == self.latent_dim:
                if HAS_TORCH:
                    return torch.tensor(arr, device=self.device, dtype=torch.float32), PROVENANCE_RAW_FEATURE_VECTOR
                elif HAS_NUMPY:
                    return np.array(arr, dtype=np.float32), PROVENANCE_RAW_FEATURE_VECTOR
                return arr, PROVENANCE_RAW_FEATURE_VECTOR
            return self._apply_dim_projector(arr, len(arr)), PROVENANCE_PROJECTED_FEATURE_VECTOR

        # Hash prior for text / objects. sha256, not hash(): str hashing is salted per process
        # and would make the same observation produce a different latent on every run.
        h = int.from_bytes(hashlib.sha256(str(obs).encode("utf-8")).digest()[:8], "big")
        seed_vals = [(math.sin(h * (i + 1)) * 0.5) for i in range(self.latent_dim)]
        if HAS_TORCH:
            return torch.tensor(seed_vals, device=self.device, dtype=torch.float32), PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR
        elif HAS_NUMPY:
            return np.array(seed_vals, dtype=np.float32), PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR
        return seed_vals, PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR
