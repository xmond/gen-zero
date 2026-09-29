"""Zero: the standalone CPU model, text in, causal decision out.

Pipeline (one CPU process, configurable threads, no GPU, no second model):

    text --Qwen tokenizer--> ids --Zero forward (INT8 weight-only / BF16 / FP32)-->
    last-token hidden state (896) --ZCA manifold (fit on calibration only)-->
    64-D unit-sphere causal state --continuous_causal_reasoning_expert (or task head)--> decision

What Zero is
    Zero's backbone is the Qwen2.5-0.5B transformer trunk (24 layers, hidden 896)
    without a language-model head. The default precision is INT8 weight-only
    with FP32 compute: every ``nn.Linear`` weight is stored as int8 with one
    FP32 scale per output row and is dequantized into a shared scratch buffer
    at call time. The embedding table is stored in BF16 and up-cast on lookup.
    This keeps the resident tensors near 0.6 GB and the last-token hidden state
    within cosine 0.99 of the FP32 model (see the test suite for the number).

What Zero is not
    There is no call to any 9B/70B model, no remote service, no fallback
    model, no mock. If the weights or the manifold are missing the runtime
    raises. Nothing in this module inspects task names, answer strings or
    text patterns: the decision is made on vectors only.

Honesty notes
    * The manifold is unsupervised (ZCA whitening in the top-64 principal
      subspace) and is fitted only on hidden states of the calibration split.
      No label is read anywhere in this module.
    * Dynamic (activation) INT8 quantization was measured and rejected: it
      drops the last-token cosine to 0.05-0.58 because Qwen hidden states carry
      massive-activation outliers. Weight-only INT8 keeps cosine >= 0.99.
    * The optional task head (``ZeroTaskHead``, ``task_head_path``) is a
      supervised bilinear scorer fit offline on the labeled calibration split
      (disjoint from the frozen test set; see benchmarks/data/CALIBRATION_SPLIT.md
      and gen_zero.train.train_zero_task_head). Training reads labels; this
      module's ``decide()`` does not -- it reads two already-computed vectors
      (``z0``, ``zc``) and nothing else, exactly as the unsupervised path does.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .continuous_causal_reasoning_expert import (
    ContinuousCausalReasoningExpert,
    continuous_causal_reasoning_expert,
)
from .bifurcated_fractal_engine import BifurcatedFractalEngine
from .parallel_rnn_lora import ParallelRNNLoRAAdapter
from .zero_task_head import ZeroTaskHead
from .zero_trunk import (
    ZERO_ENCODER_FAMILY,
    PRECISIONS,
    KVCache,
    build_int8_artifact,
    load_trunk,
    malloc_trim,
)

__all__ = [
    "MANIFOLD_DIM",
    "ZERO_ENCODER_FAMILY",
    "ZeroManifold",
    "ZeroTaskHead",
    "ZeroDecision",
    "ZeroStandaloneRuntime",
    "build_int8_artifact",
    "enforce_single_core",
    "find_local_snapshot",
]

MANIFOLD_DIM = 64
MANIFOLD_VERSION = 1
DEFAULT_MODEL_CACHE = Path(os.environ.get(
    "ZERO_MODEL_CACHE", str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B")))


# --------------------------------------------------------------------------
# Single-core enforcement
# --------------------------------------------------------------------------

def enforce_single_core() -> None:
    """Pin PyTorch to one intra-op and one inter-op thread.

    ``set_num_interop_threads`` can only be called once per process; if an
    earlier caller already pinned it to 1 that is fine, any other value is an
    error because the latency numbers would not be single-core numbers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("inter-op thread pool was already started with more than one thread")
    if torch.get_num_threads() != 1:
        raise RuntimeError("could not pin PyTorch to a single thread")


def find_local_snapshot(model_cache: Path = DEFAULT_MODEL_CACHE) -> Path:
    """Return the complete local HF snapshot directory or raise (never download)."""
    if all((model_cache / name).exists() for name in ("config.json", "model.safetensors", "tokenizer.json")):
        return model_cache
    snapshots = Path(model_cache) / "snapshots"
    candidates = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
    complete = [p for p in candidates if (p / "config.json").exists()
                and (p / "model.safetensors").exists() and (p / "tokenizer.json").exists()]
    if not complete:
        raise FileNotFoundError(f"no complete local Zero backbone snapshot under {snapshots}")
    return complete[-1]


# --------------------------------------------------------------------------
# 64-D manifold
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ZeroManifold:
    """ZCA whitening restricted to the top-``dim`` principal subspace, then sphered.

    ``project(h)`` = normalize(((h - mean) @ basis) * scale @ basis.T @ basis)
    which, because ``basis`` is orthonormal, equals normalize(((h - mean) @ basis) * scale)
    expressed in the 64 principal coordinates. Whitening equalizes the
    variance of the 64 kept directions so no single outlier dimension of the
    backbone dominates the causal geometry.
    """

    mean: np.ndarray      # (hidden,)
    basis: np.ndarray     # (hidden, dim), orthonormal columns
    scale: np.ndarray     # (dim,)
    provenance: dict

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def hidden(self) -> int:
        return int(self.basis.shape[0])

    @classmethod
    def fit(cls, states: np.ndarray, *, dim: int = MANIFOLD_DIM, shrink: float = 0.1,
            encoder_id: str, source: str, split: str) -> "ZeroManifold":
        """Fit on label-free hidden states of a train/calibration split only."""
        if split not in ("train", "calibration"):
            raise ValueError("the manifold may only be fitted on train/calibration states")
        if not source or not encoder_id:
            raise ValueError("source and encoder_id are required")
        x = np.asarray(states, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] <= dim or not np.isfinite(x).all():
            raise ValueError(f"need a finite (N > {dim}, hidden) matrix of states")
        if not (0.0 < shrink < 1.0):
            raise ValueError("shrink must be in (0, 1)")
        mean = x.mean(axis=0)
        centered = x - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        eig = (singular ** 2) / x.shape[0]
        if eig[dim - 1] <= 0.0:
            raise ValueError("calibration states do not span the requested manifold dimension")
        kept = eig[:dim]
        eps = float(shrink * kept.mean())
        scale = 1.0 / np.sqrt(kept + eps)
        digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()
        provenance = {
            "version": MANIFOLD_VERSION,
            "encoder_id": encoder_id,
            "source": source,
            "split": split,
            "samples": int(x.shape[0]),
            "states_sha256": digest,
            "energy_kept": float(kept.sum() / eig.sum()),
            "shrink": shrink,
            "eps": eps,
        }
        return cls(mean=mean.astype(np.float32), basis=vt[:dim].T.astype(np.float32).copy(),
                   scale=scale.astype(np.float32), provenance=provenance)

    def project(self, states: np.ndarray) -> np.ndarray:
        """(N, hidden) or (hidden,) -> unit vectors on the 64-sphere, float64."""
        h = np.asarray(states, dtype=np.float64)
        squeeze = h.ndim == 1
        if squeeze:
            h = h[None, :]
        if h.ndim != 2 or h.shape[1] != self.hidden or not np.isfinite(h).all():
            raise ValueError(f"states must be finite with hidden size {self.hidden}")
        z = ((h - self.mean) @ self.basis) * self.scale
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("a state collapsed to the manifold origin; cannot sphere it")
        z = z / norms
        return z[0] if squeeze else z

    def save(self, path: Path) -> None:
        with open(path, "wb") as stream:
            np.savez(stream, metadata=json.dumps(self.provenance), mean=self.mean,
                     basis=self.basis, scale=self.scale)

    @classmethod
    def load(cls, path: Path, *, encoder_id: str) -> "ZeroManifold":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"metadata", "mean", "basis", "scale"}:
                raise ValueError("invalid manifold schema")
            meta = json.loads(str(data["metadata"]))
            if meta.get("version") != MANIFOLD_VERSION or meta.get("encoder_id") != encoder_id:
                raise ValueError("manifold version or frozen encoder mismatch")
            if meta.get("split") not in ("train", "calibration"):
                raise ValueError("manifold provenance split is not train/calibration")
            mean, basis, scale = data["mean"], data["basis"], data["scale"]
        for name, value in (("mean", mean), ("basis", basis), ("scale", scale)):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"manifold {name} must be finite float32")
        if basis.ndim != 2 or mean.shape != (basis.shape[0],) or scale.shape != (basis.shape[1],):
            raise ValueError("manifold shapes are inconsistent")
        gram = basis.astype(np.float64).T @ basis.astype(np.float64)
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-4):
            raise ValueError("manifold basis is not orthonormal")
        if np.any(scale <= 0):
            raise ValueError("manifold scale must be positive")
        return cls(mean=mean, basis=basis, scale=scale, provenance=meta)


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ZeroDecision:
    index: int
    scores: np.ndarray            # (K,) higher is better
    prompt_state: np.ndarray      # (64,)
    candidate_states: np.ndarray  # (K, 64)
    prompt_tokens: int
    candidate_tokens: int         # summed over candidates
    tokenize_ms: float
    forward_ms: float
    manifold_ms: float
    dynamics_ms: float
    route_weights: Optional[dict] = None
    tangent_weights: Optional[dict] = None
    task_head_used: bool = False

    @property
    def total_ms(self) -> float:
        return self.tokenize_ms + self.forward_ms + self.manifold_ms + self.dynamics_ms


class ZeroStandaloneRuntime:
    """One CPU process and one model, with configurable compute threads."""

    def __init__(self, *, model_cache: Path = DEFAULT_MODEL_CACHE, precision: str = "int8",
                 int8_artifact: Optional[Path] = None, manifold_path: Optional[Path] = None,
                 dynamics_path: Optional[Path] = None, task_head_path: Optional[Path] = None,
                 adapter_path: Optional[Path] = None, rnn_path: Optional[Path] = None,
                 max_length: int = 1024, single_core: bool = False,
                 num_threads: Optional[int] = None,
                 candidate_chunk_size: Optional[int] = None,
                 thinking_mode: Optional[str] = None) -> None:
        if thinking_mode not in (None, "fractal", "continuous", "lora_rnn"):
            raise ValueError("thinking_mode must be None, fractal, continuous, or lora_rnn")
        if thinking_mode == "lora_rnn" and rnn_path is None:
            raise ValueError("lora_rnn thinking mode requires rnn_path")
        if thinking_mode is not None and task_head_path is not None:
            raise ValueError("thinking_mode and task_head_path select different decision scorers")
        self.thinking_mode = thinking_mode
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if candidate_chunk_size is not None and candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive or None")
        if num_threads is not None and num_threads <= 0:
            raise ValueError("num_threads must be positive or None")
        if single_core:
            enforce_single_core()
        else:
            torch.set_num_threads(num_threads or min(8, os.cpu_count() or 4))
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError("Zero must not run in a process that has initialized CUDA")
        self.snapshot = find_local_snapshot(model_cache)
        self.precision = precision
        self.max_length = int(max_length)
        self.candidate_chunk_size = candidate_chunk_size
        self.encoder_id = f"{ZERO_ENCODER_FAMILY}:{precision}:last-token"

        from tokenizers import Tokenizer  # the Rust tokenizer only; no transformers import
        self.tokenizer = Tokenizer.from_file(str(self.snapshot / "tokenizer.json"))
        self.model, self.weight_metadata = load_trunk(self.snapshot, precision, int8_artifact)
        self.hidden_size = self.model.cfg.hidden_size

        self.manifold: Optional[ZeroManifold] = None
        manifold_sha256: Optional[str] = None
        if manifold_path is not None:
            manifold_path = Path(manifold_path)
            self.manifold = ZeroManifold.load(manifold_path, encoder_id=self.encoder_id)
            if self.manifold.hidden != self.hidden_size:
                raise ValueError("manifold hidden size does not match the backbone")
            manifold_sha256 = hashlib.sha256(manifold_path.read_bytes()).hexdigest()
        self.calibrated_expert: Optional[ContinuousCausalReasoningExpert] = None
        if dynamics_path is not None:
            self.calibrated_expert = ContinuousCausalReasoningExpert.from_file(
                Path(dynamics_path), encoder_id=self.encoder_id)
        self.task_head: Optional[ZeroTaskHead] = None
        if task_head_path is not None:
            if self.manifold is None:
                raise ValueError("a task head requires a manifold_path (it scores manifold vectors)")
            self.task_head = ZeroTaskHead.load(Path(task_head_path), encoder_id=self.encoder_id)
            if self.task_head.dim != self.manifold.dim:
                raise ValueError("task head dimension does not match the causal manifold")
            if self.task_head.provenance.get("manifold_sha256") != manifold_sha256:
                raise ValueError("task head was calibrated against a different manifold artifact "
                                 f"(loaded manifold sha256 {manifold_sha256}, head expects "
                                 f"{self.task_head.provenance.get('manifold_sha256')})")
        self.adapter = None
        if adapter_path is not None:
            if self.manifold is None:
                raise ValueError("an adapter requires a manifold_path (it extends the manifold's "
                                 "linear skip)")
            from .deep_projection_adapter import DeepProjectionAdapter  # avoid the circular import at module load
            self.adapter = DeepProjectionAdapter(self.manifold, encoder_id=self.encoder_id)
            self.adapter.load_state_dict(torch.load(Path(adapter_path), map_location="cpu", weights_only=True))
            self.adapter.eval()
        self.rnn = None
        if rnn_path is not None:
            from .parallel_rnn_lora import ParallelRNNLoRAAdapter
            self.rnn = ParallelRNNLoRAAdapter.load(Path(rnn_path))
        malloc_trim()

    def tensor_bytes(self) -> int:
        return self.model.tensor_bytes()

    # -- tokenization -------------------------------------------------------

    def token_ids(self, text: str) -> List[int]:
        """Token ids for one text. Sequences over ``max_length`` raise; never truncated."""
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a nonempty string")
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) > self.max_length:
            raise ValueError(f"{len(ids)} tokens exceeds max_length={self.max_length}; refusing to truncate")
        if not ids:
            raise ValueError("text tokenized to nothing")
        return ids

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return ids, mask

    # -- encoding -----------------------------------------------------------

    def rollout_latent(self, *, steps: int, prompt: Optional[str] = None,
                       inputs_embeds: Optional[Tensor] = None) -> Tensor:
        """Return K successive continuous hidden states (B, K, H) on CPU.

        Each previous hidden state becomes the next input embedding directly.
        This is an untrained identity bridge, not a calibrated reasoning policy.
        """
        if (prompt is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of prompt and inputs_embeds")
        if steps < 1 or steps > self.max_length:
            raise ValueError("steps must be within 1..max_length")
        if prompt is not None:
            ids = torch.tensor([self.token_ids(prompt)], dtype=torch.long)
            seed_length = ids.shape[1]
        else:
            ids = None
            seed_length = inputs_embeds.shape[1] if inputs_embeds.ndim == 3 else 0
        if seed_length + steps > self.max_length:
            raise ValueError("latent rollout exceeds max_length")
        with torch.inference_mode():
            hidden, cache = self.model(ids, inputs_embeds=inputs_embeds, return_cache=True)
            latent = hidden[:, -1:, :]
            mask = torch.ones((hidden.shape[0], seed_length), dtype=torch.bool)
            states = []
            for _ in range(steps):
                latent, cache, mask = self.model.forward_latent_step(latent, cache, mask)
                if not torch.isfinite(latent).all():
                    raise FloatingPointError("Zero produced a non-finite latent state")
                states.append(latent)
        return torch.cat(states, dim=1)

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, dict]:
        """Last-token hidden states (N, hidden) float32 for independent full sequences."""
        t0 = time.perf_counter()
        rows = [self.token_ids(t) for t in texts]
        ids, mask = self._pad(rows)
        t1 = time.perf_counter()
        with torch.inference_mode():
            hidden, _ = self.model(ids, mask)
            last = self.model.last_valid(hidden, mask).to(torch.float32)
        t2 = time.perf_counter()
        states = last.numpy()
        if not np.isfinite(states).all():
            raise FloatingPointError("Zero produced a non-finite hidden state")
        return states, {"tokenize_ms": (t1 - t0) * 1000.0, "forward_ms": (t2 - t1) * 1000.0,
                        "tokens": int(mask.sum()), "width": int(ids.shape[1])}

    @staticmethod
    def candidate_text(candidate: str) -> str:
        """Candidate continuation text; the same rule for every task and language."""
        return " " + candidate.replace("_", " ")

    def encode_prompt_with_candidates(self, prompt: str, candidates: Sequence[str], *,
                                      candidate_chunk_size: Optional[int] = None
                                      ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Prompt state and K candidate-terminal states with one prompt pass.

        The prompt is run once with a KV cache; each candidate's tokens then
        continue from that cache in a batch. With causal attention this is the
        same computation as K full ``prompt + candidate`` sequences (the test
        suite checks the equality), at a fraction of the cost.

        ``candidate_chunk_size`` bounds how many candidates share one
        expanded-KV-cache forward pass at a time (default: all candidates).
        Set a positive chunk size when memory is scarce. Results are numerically equivalent to the
        unchunked path up to float rounding (chunking only changes batch
        size, not the computation; int8 matmuls are batch-size-sensitive at
        the ULP level, see test_candidate_chunking_matches_unchunked).
        """
        t0 = time.perf_counter()
        prompt_ids = self.token_ids(prompt)
        cand_rows = [self.tokenizer.encode(self.candidate_text(c), add_special_tokens=False).ids
                     for c in candidates]
        if any(not r for r in cand_rows):
            raise ValueError("a candidate tokenized to nothing")
        if any(len(prompt_ids) + len(r) > self.max_length for r in cand_rows):
            raise ValueError("prompt + candidate exceeds max_length; refusing to truncate")
        cand_ids, cand_mask = self._pad(cand_rows)
        t1 = time.perf_counter()
        if candidate_chunk_size is None:
            candidate_chunk_size = self.candidate_chunk_size
        k = len(candidates)
        chunk = candidate_chunk_size if candidate_chunk_size else k
        with torch.inference_mode():
            p_ids = torch.tensor([prompt_ids], dtype=torch.long)
            p_mask = torch.ones_like(p_ids)
            hidden, cache = self.model(p_ids, p_mask, return_cache=True)
            q0 = hidden[0, -1].to(torch.float32)
            chunk_states = []
            for start in range(0, k, chunk):
                end = min(start + chunk, k)
                n = end - start
                expanded: KVCache = [(kk.expand(n, -1, -1, -1), vv.expand(n, -1, -1, -1)) for kk, vv in cache]
                past_mask = p_mask.expand(n, -1)
                hidden_c, _ = self.model(cand_ids[start:end], cand_mask[start:end],
                                         past=expanded, past_mask=past_mask)
                chunk_states.append(self.model.last_valid(hidden_c, cand_mask[start:end]).to(torch.float32))
            c_states = torch.cat(chunk_states, dim=0)
        t2 = time.perf_counter()
        q0_np, c_np = q0.numpy(), c_states.numpy()
        if not (np.isfinite(q0_np).all() and np.isfinite(c_np).all()):
            raise FloatingPointError("Zero produced a non-finite hidden state")
        info = {"tokenize_ms": (t1 - t0) * 1000.0, "forward_ms": (t2 - t1) * 1000.0,
                "prompt_tokens": len(prompt_ids), "candidate_tokens": int(cand_mask.sum())}
        return q0_np, c_np, info

    # -- decision -----------------------------------------------------------

    def decide(self, prompt: str, candidates: Sequence[str], *, rank: int = 4,
               seed: int = 0, gateway=None, codebook_runtime=None,
               codebook_task_id=None, codebook_x_projected=None,
               codebook_c_projected=None, fractal_engine_kwargs: Optional[dict] = None) -> ZeroDecision:
        if self.manifold is None:
            raise RuntimeError("no manifold loaded; decisions are disabled (fit one on calibration data)")
        candidates = list(candidates)
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError("need at least two distinct candidates")
        q0, c_states, info = self.encode_prompt_with_candidates(prompt, candidates)
        t0 = time.perf_counter()
        if self.adapter is not None:
            with torch.inference_mode():
                z0_t = self.adapter(torch.from_numpy(q0).float())
                zc_t = self.adapter(torch.from_numpy(c_states).float())
            z0 = z0_t.detach().numpy().astype(np.float64)
            zc = zc_t.detach().numpy().astype(np.float64)
        else:
            z0 = self.manifold.project(q0)
            zc = self.manifold.project(c_states)
        route_weights = tangent_weights = None
        if gateway is not None:
            if codebook_runtime is None or codebook_task_id is None or codebook_x_projected is None:
                raise ValueError("routed decision requires a codebook runtime, task id, and projected input")
            routed = codebook_runtime.infer_routed(
                codebook_task_id, codebook_x_projected, z0, codebook_c_projected,
                gateway=gateway)
            if routed.decision_state.shape != z0.shape:
                raise ValueError("shared routed state must match the Zero decision dimension")
            z0 = routed.decision_state
            zc = zc @ gateway.zero_to_shared.T
            route_weights = routed.weights
            tangent_weights = routed.tangent_weights
        t1 = time.perf_counter()
        if not (np.isfinite(z0).all() and np.isfinite(zc).all()):
            raise FloatingPointError("decision received non-finite causal states")
        if self.thinking_mode == "fractal":
            adapter = self.rnn if self.rnn is not None else ParallelRNNLoRAAdapter(
                dim=z0.size, rank=min(rank, z0.size), rho_max=0.5, seed=seed)
            engine_kwargs = {"n_repulsors": max(0, len(candidates) - 1)}
            engine_kwargs.update(fractal_engine_kwargs or {})
            engine = BifurcatedFractalEngine(dim=z0.size, adapter=adapter, **engine_kwargs)
            scores = np.empty(len(candidates), dtype=np.float64)
            for i, target in enumerate(zc):
                repulsors = zc[np.arange(len(candidates)) != i] - z0
                result = engine.think(z0, target - z0, target, repulsors=repulsors)
                if result.survivor is None or result.micro is None or not result.micro.converged:
                    raise RuntimeError(f"fractal dynamics failed to converge for candidate {i}: {result.status}")
                macro_cost = result.race.cost_history[-1][result.survivor]
                scores[i] = -macro_cost - result.micro.fixed_point_gap
        elif self.thinking_mode == "lora_rnn":
            evolved = self.rollout_state(z0, steps=3)
            result = continuous_causal_reasoning_expert(
                evolved, zc, domain_prototype=None, rank=rank, seed=seed,
                calibrated_expert=self.calibrated_expert)
            scores = np.asarray(result.scores, dtype=np.float64)
        elif self.task_head is not None:
            # Supervised path: trained on the labeled calibration split (disjoint
            # from the frozen test set), no task name or text seen here or at
            # training time -- only the same label-free manifold vectors.
            scores = np.asarray(self.task_head.score(z0, zc), dtype=np.float64)
        else:
            result = continuous_causal_reasoning_expert(
                z0, zc, domain_prototype=None, rank=rank, seed=seed,
                calibrated_expert=self.calibrated_expert)
            scores = np.asarray(result.scores, dtype=np.float64)
        t2 = time.perf_counter()
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise FloatingPointError("decision produced an invalid score vector")
        return ZeroDecision(index=int(np.argmax(scores)), scores=scores, prompt_state=z0,
                            candidate_states=zc, prompt_tokens=info["prompt_tokens"],
                            candidate_tokens=info["candidate_tokens"],
                            tokenize_ms=info["tokenize_ms"], forward_ms=info["forward_ms"],
                            manifold_ms=(t1 - t0) * 1000.0, dynamics_ms=(t2 - t1) * 1000.0,
                            route_weights=route_weights, tangent_weights=tangent_weights,
                            task_head_used=self.task_head is not None)

    # -- cheap multi-step rollout (opt-in, never used by decide()) ---------

    def rollout_state(self, z0: np.ndarray, steps: int) -> np.ndarray:
        """Multi-step state rollout via the low-dim ``ParallelRNNLoRAAdapter`` side-car.

        O(steps * (dim + rank)); this never re-invokes the 24-layer backbone --
        each step is one ``self.rnn.step`` call in the RNN's own (dim,) space.
        Starts from ``h = 0`` and folds in ``z0`` at every step, returning only
        the final state after ``steps`` iterations (not the full trajectory).
        Standalone and opt-in: ``decide()``'s default single-shot path never
        calls this, so every existing accuracy number is unaffected.
        """
        if self.rnn is None:
            raise RuntimeError("no rnn loaded; pass rnn_path to enable rollout_state")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        z0 = np.asarray(z0, dtype=self.rnn.dtype)
        h = np.zeros(self.rnn.dim, dtype=self.rnn.dtype)
        for _ in range(steps):
            h = self.rnn.step(z0, h)
        return h

    # -- manifold fitting (offline) ----------------------------------------

    def fit_manifold(self, states: np.ndarray, *, source: str, split: str,
                     dim: int = MANIFOLD_DIM) -> ZeroManifold:
        manifold = ZeroManifold.fit(states, dim=dim, encoder_id=self.encoder_id,
                                    source=source, split=split)
        self.manifold = manifold
        return manifold
