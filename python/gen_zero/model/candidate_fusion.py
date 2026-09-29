"""Gen-Zero Layer 1: Candidate-Conditioned Batch Forward + Multi-Layer Latent Fusion.

Fuses two 0-token scoring strategies into one calibrated confidence per candidate,
using EXACTLY ONE batched `model(**batch, output_hidden_states=True)` forward pass.
`generate()` is never invoked; zero autoregressive tokens are ever produced.

Strategy 1 (Candidate-Conditioned Batch Forward):
    Tokenizes `prompt + candidate_i` for every candidate into one right-padded batch
    (shared prompt prefix, variable-length candidate suffix) and reads off the
    length-normalized teacher-forced log-likelihood of each candidate's own tokens:
        (1 / |c_i|^alpha) * sum_{t in c_i} log P(t | prefix, c_i[<t])
    where `alpha` is the configurable `length_penalty_alpha` (1.0 = plain mean).

Strategy 2 (Multi-Layer Latent Fusion):
    Pulls the terminal-token hidden state from several intermediate layers
    (default offsets -8, -4, -2, -1, spanning semantic / relational / output stages
    in a 36-layer Qwen stack) and, at each layer, computes the cosine alignment
    between the prompt's boundary token and the candidate's terminal token. The
    per-layer alignments are averaged into one multi-layer alignment score per
    candidate, then converted to a permutation-equivariant relative margin against
    the best competing candidate.

The two signals are combined with `calibrate_temperature_and_entropy` (the same
calibration primitive used by `ActionETFChoiceHead`) into one fused, calibrated
confidence distribution over candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gen_zero.nanocore.choice_head import calibrate_temperature_and_entropy

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False


DEFAULT_LAYER_OFFSETS: Tuple[int, ...] = (-8, -4, -2, -1)


@dataclass
class CandidateScore:
    """Per-candidate breakdown of the fused 0-token score."""
    candidate: str
    sequence_logprob: float   # Strategy 1: length-normalized teacher-forced log-likelihood
    etf_alignment: float      # Strategy 2: multi-layer prompt-candidate cosine alignment
    etf_margin: float         # Strategy 2: alignment margin over the best competing candidate
    fused_confidence: float   # calibrated probability after fusing Strategy 1 + Strategy 2
    num_candidate_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate,
            "sequence_logprob": round(self.sequence_logprob, 6),
            "etf_alignment": round(self.etf_alignment, 6),
            "etf_margin": round(self.etf_margin, 6),
            "fused_confidence": round(self.fused_confidence, 6),
            "num_candidate_tokens": self.num_candidate_tokens,
        }


@dataclass
class CandidateFusionResult:
    """Outcome of one `CandidateMultiLayerFusionEngine.score` call. `tokens_generated` is 0 by construction."""
    scores: List[CandidateScore]
    best_index: int
    layer_offsets: Tuple[int, ...]
    calibration_entropy: float
    tokens_generated: int = 0
    forward_pass_count: int = 1
    # Row 0's hidden state / logits at the prompt boundary, reused so callers don't need a
    # second standalone `model(prompt)` forward just to get these. Only populated when row 0
    # had no joint-tokenization boundary merge (see `score()`); `score()` raises otherwise
    # rather than populating a value computed over a shorter, wrong prefix.
    prompt_hidden_state: Optional[np.ndarray] = None
    prompt_last_logits: Optional[Any] = None
    # Wall-clock time (ms) of just the `self.model(...)` call, GPU-synchronized on CUDA.
    forward_ms: Optional[float] = None
    # Per-candidate terminal hidden state at the last layer, (K, D). Same layer and same
    # gather (`terminal_idx`) `_strategy2_multilayer_latent` already computes for its cosine
    # alignment; that computation reduces each row to one scalar and discards the vector.
    # This field keeps the vector instead, for callers that need real per-candidate
    # representations (not a fabricated one) -- see continuous_causal_reasoning_expert.py.
    candidate_hidden_states: Optional[np.ndarray] = None

    @property
    def best_candidate(self) -> str:
        return self.scores[self.best_index].candidate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.to_dict() for s in self.scores],
            "best_index": self.best_index,
            "best_candidate": self.best_candidate,
            "layer_offsets": list(self.layer_offsets),
            "calibration_entropy": round(self.calibration_entropy, 6),
            "tokens_generated": self.tokens_generated,
            "forward_pass_count": self.forward_pass_count,
        }


class CandidateMultiLayerFusionEngine:
    """Scores candidate continuations of a prompt with a single 0-token batched forward pass."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer_offsets: Sequence[int] = DEFAULT_LAYER_OFFSETS,
        likelihood_weight: float = 0.7,
        geometry_weight: float = 0.3,
        fusion_temperature: float = 1.0,
        entropy_penalty_weight: float = 0.25,
        max_confidence_cap: float = 0.88,
        length_penalty_alpha: float = 1.0,
        joint_tokenization: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("CandidateMultiLayerFusionEngine requires PyTorch")
        if len(layer_offsets) == 0:
            raise ValueError("layer_offsets must contain at least one layer index")
        self.model = model
        self.tokenizer = tokenizer
        self.layer_offsets = tuple(layer_offsets)
        self.likelihood_weight = float(likelihood_weight)
        self.geometry_weight = float(geometry_weight)
        self.fusion_temperature = max(1e-4, float(fusion_temperature))
        self.entropy_penalty_weight = float(entropy_penalty_weight)
        self.max_confidence_cap = float(max_confidence_cap)
        self.length_penalty_alpha = float(length_penalty_alpha)
        self.joint_tokenization = bool(joint_tokenization)
        self.device = device if device is not None else self._infer_device(model)

    @staticmethod
    def _infer_device(model: Any) -> Any:
        if hasattr(model, "device") and model.device is not None:
            return model.device
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
        return "cpu"

    def _pad_token_id(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        return int(pad_id) if pad_id is not None else 0

    def _joint_boundary_row(
        self, prompt: str, candidate: str, prompt_ids: Sequence[int]
    ) -> Tuple[List[int], int]:
        """Tokenizes `prompt + candidate` as one string; returns (row, prompt_len_for_row).

        Naively concatenating `encode(prompt)` and `encode(candidate)` freezes the boundary
        between the two strings, so a non-compositional BPE tokenizer that would merge
        characters across that boundary (e.g. a candidate that continues mid-subword) never
        gets the chance to. Tokenizing the joint string instead lets those boundary merges
        happen. The row returned IS the joint tokenization (never a concatenation of two
        separately-tokenized halves, which would duplicate or corrupt the merged boundary
        token); `prompt_len_for_row` is the longest common prefix length with the standalone
        `prompt_ids`, i.e. how many leading tokens of the row are unaffected by the merge and
        therefore still "prompt" rather than "candidate" for scoring purposes.
        """
        joint_ids = list(self.tokenizer.encode(prompt + candidate, add_special_tokens=False))
        common_len = 0
        max_common = min(len(prompt_ids), len(joint_ids))
        while common_len < max_common and prompt_ids[common_len] == joint_ids[common_len]:
            common_len += 1
        if common_len == 0 or common_len == len(joint_ids):
            # Fail closed: strictly no silent fallbacks. Raise descriptive error.
            raise ValueError(
                f"Joint tokenization failed for candidate {candidate!r}: "
                f"shared prefix length {common_len} (prompt tokens: {len(prompt_ids)}, joint tokens: {len(joint_ids)}). "
                "If non-joint concatenation is explicitly desired, initialize with joint_tokenization=False."
            )
        return joint_ids, common_len

    def build_batch(self, prompt: str, candidates: Sequence[str]) -> Dict[str, Any]:
        """Tokenizes `prompt + candidate_i` for every candidate into one right-padded batch.

        Returns input_ids/attention_mask/position_ids of shape [K, max row length], plus
        `prompt_len` (the standalone-tokenized prompt length, for reference), `prompt_lens`
        (the actual per-row prompt boundary length used for scoring; see below), and
        `candidate_lens` (per-row candidate token counts, i.e. row length minus its
        `prompt_lens` entry).

        With `joint_tokenization` disabled, every row is `prompt_ids + encode(candidate_i)`
        and `prompt_lens` is `prompt_len` repeated K times, exactly as before. With it
        enabled (the default), each row is instead the direct tokenization of
        `prompt + candidate_i` (see `_joint_boundary_row`): a merge across the boundary can
        absorb what would otherwise be the prompt's last token or two, so `prompt_lens[i]`
        can be shorter than `prompt_len` for that row.
        """
        if len(candidates) == 0:
            raise ValueError("candidates must be non-empty")

        prompt_ids: List[int] = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if len(prompt_ids) == 0:
            raise ValueError("prompt must tokenize to at least one token")

        rows: List[List[int]] = []
        prompt_lens: List[int] = []
        for c in candidates:
            if self.joint_tokenization:
                row, row_prompt_len = self._joint_boundary_row(prompt, c, prompt_ids)
            else:
                row = list(prompt_ids) + list(self.tokenizer.encode(c, add_special_tokens=False))
                row_prompt_len = len(prompt_ids)
            rows.append(row)
            prompt_lens.append(row_prompt_len)

        candidate_lens = [len(row) - p_len for row, p_len in zip(rows, prompt_lens)]
        for i, clen in enumerate(candidate_lens):
            if clen <= 0:
                raise ValueError(f"candidate at index {i} tokenized to zero tokens")

        k = len(candidates)
        seq_len = max(len(row) for row in rows)
        pad_id = self._pad_token_id()

        input_ids = torch.full((k, seq_len), fill_value=pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((k, seq_len), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((k, seq_len), dtype=torch.long, device=self.device)

        for i, row in enumerate(rows):
            real_len = len(row)
            input_ids[i, :real_len] = torch.tensor(row, dtype=torch.long, device=self.device)
            attention_mask[i, :real_len] = 1
            position_ids[i, :real_len] = torch.arange(real_len, device=self.device)
            if real_len < seq_len:
                # Clamp padded positions so RoPE never extrapolates past the real sequence.
                position_ids[i, real_len:] = real_len - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_len": len(prompt_ids),
            "prompt_lens": prompt_lens,
            "candidate_lens": candidate_lens,
        }

    def score(self, prompt: str, candidates: Sequence[str]) -> CandidateFusionResult:
        """Scores every candidate with exactly one batched forward pass. Zero tokens generated."""
        candidates = list(candidates)
        batch = self.build_batch(prompt, candidates)
        prompt_lens = batch["prompt_lens"]
        candidate_lens = batch["candidate_lens"]
        c_max = max(candidate_lens)

        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()

        cuda_timed = bool(torch.cuda.is_available())
        if cuda_timed:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            if hasattr(self.model, "train") and was_training:
                self.model.train(True)
        if cuda_timed:
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - _t0) * 1000.0

        # Row 0's hidden state / logits at its own prompt boundary are only a valid
        # stand-in for a standalone `model(prompt)` forward when candidate 0's joint
        # tokenization did not merge across the prompt/candidate boundary (see
        # `_joint_boundary_row`). Fail closed rather than silently handing a caller a
        # value computed over a shorter, wrong prefix.
        if prompt_lens[0] != batch["prompt_len"]:
            raise ValueError(
                "Cannot extract prompt_hidden_state/prompt_last_logits: candidate 0's "
                f"joint tokenization merged across the prompt boundary (prompt_lens[0]="
                f"{prompt_lens[0]}, standalone prompt_len={batch['prompt_len']}). Row 0's "
                "hidden state/logits at position prompt_lens[0]-1 correspond to a shorter, "
                "different prefix than the standalone prompt and cannot be reused. Reorder "
                "candidates so a non-merging one is first, or compute a standalone forward."
            )
        prompt_boundary_pos = prompt_lens[0] - 1
        prompt_hidden_state = (
            outputs.hidden_states[-1][0, prompt_boundary_pos, :].detach().float().cpu().numpy()
        )
        prompt_last_logits = outputs.logits[0, prompt_boundary_pos, :].detach().float()

        seq_logprobs = self._strategy1_sequence_logprobs(
            outputs.logits, batch["input_ids"], prompt_lens, candidate_lens, c_max
        )
        latent_alignment, latent_margin = self._strategy2_multilayer_latent(
            outputs.hidden_states, prompt_lens, candidate_lens
        )
        # Same terminal position `_strategy2_multilayer_latent` gathers at its last layer
        # offset, kept as a vector instead of reduced to a cosine scalar.
        terminal_idx_last = (
            torch.tensor(prompt_lens, dtype=torch.long, device=outputs.hidden_states[-1].device)
            + torch.tensor(candidate_lens, dtype=torch.long, device=outputs.hidden_states[-1].device)
            - 1
        )
        candidate_hidden_states = (
            outputs.hidden_states[-1][torch.arange(len(candidates)), terminal_idx_last, :]
            .detach().float().cpu().numpy()
        )

        seq_logprobs_np = seq_logprobs.detach().cpu().numpy().astype(np.float64)
        alignment_np = latent_alignment.detach().cpu().numpy().astype(np.float64)
        margin_np = latent_margin.detach().cpu().numpy().astype(np.float64)

        fused_logits = (
            self.likelihood_weight * seq_logprobs_np + self.geometry_weight * margin_np
        )
        calibrated_probs, _calibrated_conf, _ = calibrate_temperature_and_entropy(
            fused_logits,
            temperature=self.fusion_temperature,
            entropy_penalty=self.entropy_penalty_weight,
            # A singleton is necessarily [1]; a multi-choice cap < 1 is
            # mathematically inapplicable, not evidence of uncertainty.
            max_confidence_cap=self.max_confidence_cap if len(candidates) > 1 else None,
        )

        # Accurate Shannon entropy of final returned distribution
        k = len(candidates)
        p = np.clip(calibrated_probs, 1e-12, 1.0)
        norm_factor = np.log(k) if k > 1 else 1.0
        entropy = float(-np.sum(p * np.log(p)) / norm_factor)

        scores = [
            CandidateScore(
                candidate=candidates[i],
                sequence_logprob=float(seq_logprobs_np[i]),
                etf_alignment=float(alignment_np[i]),
                etf_margin=float(margin_np[i]),
                fused_confidence=float(calibrated_probs[i]),
                num_candidate_tokens=candidate_lens[i],
            )
            for i in range(len(candidates))
        ]
        best_index = int(np.argmax(calibrated_probs))

        return CandidateFusionResult(
            scores=scores,
            best_index=best_index,
            layer_offsets=self.layer_offsets,
            calibration_entropy=entropy,
            tokens_generated=0,
            forward_pass_count=1,
            prompt_hidden_state=prompt_hidden_state,
            prompt_last_logits=prompt_last_logits,
            forward_ms=forward_ms,
            candidate_hidden_states=candidate_hidden_states,
        )

    def _strategy1_sequence_logprobs(
        self,
        logits: "torch.Tensor",       # [K, L, V]
        input_ids: "torch.Tensor",    # [K, L]
        prompt_lens: Sequence[int],   # per-row prompt boundary length (see build_batch)
        candidate_lens: Sequence[int],
        c_max: int,
    ) -> "torch.Tensor":
        """Length-normalized candidate log-likelihood with pre-gathered log_softmax to prevent VRAM explosion.

        Normalizes by `|c_i|^length_penalty_alpha` rather than a plain mean, so callers can
        tune how strongly candidate length scales the negative log-probability sum
        (alpha=1.0 is the plain per-token mean; alpha<1.0 approaches unnormalized joint log-prob,
        penalizing longer sequences more; alpha>1.0 dampens length penalties).

        `prompt_lens` varies per row under joint tokenization (a boundary merge can shift
        where the candidate actually starts), so prediction/target positions are gathered
        per-row rather than sliced with one shared range.
        """
        device = logits.device
        seq_len = logits.shape[1]
        vocab_size = logits.shape[-1]
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)  # [K]
        step_idx = torch.arange(c_max, device=device).unsqueeze(0)                  # [1, c_max]

        pred_idx = (prompt_lens_t.unsqueeze(1) - 1 + step_idx).clamp(0, seq_len - 1)  # [K, c_max]
        tgt_idx = (prompt_lens_t.unsqueeze(1) + step_idx).clamp(0, seq_len - 1)       # [K, c_max]

        # Gather only candidate prediction positions in native dtype BEFORE float conversion to save massive VRAM
        gather_idx = pred_idx.unsqueeze(-1).expand(-1, -1, vocab_size)
        cand_logits = torch.gather(logits, 1, gather_idx).float()                    # [K, c_max, V]
        pred_log_probs = F.log_softmax(cand_logits, dim=-1)                          # [K, c_max, V]

        targets = torch.gather(input_ids, 1, tgt_idx)                                # [K, c_max]
        gathered = torch.gather(pred_log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)  # [K, c_max]

        lens = torch.tensor(candidate_lens, dtype=torch.long, device=device)
        valid_mask = (step_idx < lens.unsqueeze(1)).to(gathered.dtype)              # [K, c_max]

        summed = (gathered * valid_mask).sum(dim=1)
        denom = lens.clamp(min=1).to(summed.dtype) ** self.length_penalty_alpha
        return summed / denom

    def _strategy2_multilayer_latent(
        self,
        hidden_states: Sequence["torch.Tensor"],  # tuple of [K, L, D], one per layer (+embeddings)
        prompt_lens: Sequence[int],   # per-row prompt boundary length (see build_batch)
        candidate_lens: Sequence[int],
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Permutation-equivariant multi-layer prompt-candidate cosine alignment and relative margin."""
        k = len(candidate_lens)
        num_layers_available = len(hidden_states)
        device = hidden_states[0].device

        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)  # [K]
        prompt_idx = prompt_lens_t - 1  # [K], per-row: joint tokenization can shift the boundary
        candidate_lens_t = torch.tensor(candidate_lens, dtype=torch.long, device=device)
        terminal_idx = prompt_lens_t + candidate_lens_t - 1  # [K]

        layer_alignments: List["torch.Tensor"] = []
        for offset in self.layer_offsets:
            try:
                layer_hs = hidden_states[offset]  # [K, L, D]
            except IndexError:
                raise ValueError(
                    f"layer offset {offset} out of range for {num_layers_available} hidden-state layers"
                ) from None
            d = layer_hs.shape[-1]

            # Prompt boundary representation at this layer [K, D]
            gather_idx_prompt = prompt_idx.view(k, 1, 1).expand(k, 1, d)
            p_vec = torch.gather(layer_hs, 1, gather_idx_prompt).squeeze(1)  # [K, D]
            p_norm = F.normalize(p_vec.float(), p=2, dim=-1)

            # Candidate terminal representation at this layer [K, D]
            gather_idx = terminal_idx.view(k, 1, 1).expand(k, 1, d)
            c_vec = torch.gather(layer_hs, 1, gather_idx).squeeze(1)  # [K, D]
            c_norm = F.normalize(c_vec.float(), p=2, dim=-1)

            # Cosine alignment between prompt and candidate at this layer [K]
            # Depends strictly on prompt and candidate content, 100% permutation equivariant
            cos_sim = (p_norm * c_norm).sum(dim=-1)
            layer_alignments.append(cos_sim)

        # Multi-layer average alignment [K]
        multi_alignment = torch.stack(layer_alignments, dim=0).mean(dim=0)  # [K]

        if k == 1:
            margin = multi_alignment.clone()
        else:
            # Permutation-equivariant relative margin over competing candidates
            # margin_i = alignment_i - max_{j != i} alignment_j
            # When candidates are permuted, margin permutes equivariantly with zero slot bias.
            diff_matrix = multi_alignment.unsqueeze(1) - multi_alignment.unsqueeze(0)  # [K, K]: diff[i, j] = a_i - a_j
            diff_matrix.fill_diagonal_(float("inf"))
            margin = diff_matrix.min(dim=1).values  # min_{j != i} (a_i - a_j) = a_i - max_{j != i} a_j

        return multi_alignment, margin

    # Backwards-compatibility alias
    _strategy2_multilayer_etf = _strategy2_multilayer_latent


__all__ = [
    "CandidateScore",
    "CandidateFusionResult",
    "CandidateMultiLayerFusionEngine",
    "DEFAULT_LAYER_OFFSETS",
    "HAS_TORCH",
]
