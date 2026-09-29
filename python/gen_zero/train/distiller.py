"""Gen-Zero Layer 4: Multi-Task Dual-Head Policy+Value Distiller.

Implements AlphaZero-style soft target distillation:
Loss = L_policy(KL / Soft-Cross-Entropy) + lambda_v * MSE(z, V_hat)
Lightweight iterations (1000~1500 steps) designed for GPU and CPU execution.
"""

import math
from typing import Dict, List, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from .replay_buffer import StabilityReplayBuffer


class GenZeroDistiller:
    """Multi-task Policy-Value Distillation Trainer."""
    def __init__(
        self,
        model: Any,
        replay_buffer: StabilityReplayBuffer,
        lr: float = 2e-5,
        value_weight: float = 0.5,
        device: str = "cuda"
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.value_weight = value_weight
        self.device = device if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        if HAS_TORCH and hasattr(model, "parameters"):
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = None

    def bind_model(self, new_model: Any) -> None:
        """Re-binds the distiller and creates a fresh optimizer pointing to new_model parameters."""
        self.model = new_model
        if HAS_TORCH and hasattr(new_model, "parameters"):
            self.optimizer = torch.optim.AdamW(new_model.parameters(), lr=self.lr, weight_decay=1e-4)
        else:
            self.optimizer = None

    def clone_for_candidate(self, candidate_model: Any) -> "GenZeroDistiller":
        """Spawns an isolated distiller instance dedicated strictly to candidate_model."""
        return GenZeroDistiller(
            model=candidate_model,
            replay_buffer=self.replay_buffer,
            lr=self.lr,
            value_weight=self.value_weight,
            device=self.device
        )

    def train_step(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single optimization step across Policy CE and Value MSE."""
        return self._train_step_impl(batch)

    distill_step = train_step

    def _train_step_impl(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:


        if not batch:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "optimized": False, "status": "EMPTY_BATCH"}

        if HAS_TORCH:
            if self.optimizer is None or self.model is None:
                return {
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "optimized": False,
                    "status": "NO_OPTIMIZER_OR_MODEL"
                }
        else:
            # Portable CPU / Mock environment without PyTorch
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "optimized": False,
                "status": "SKIPPED_NO_TRAINING_BACKEND"
            }

        # Validate that batch contains meaningful supervision targets
        has_supervision = any(
            (ex.get("pi_target") and (any(float(v) != 0 for v in ex["pi_target"].values()) if isinstance(ex["pi_target"], dict) else any(float(v) != 0 for v in ex["pi_target"]))) or
            (ex.get("value_target") is not None and "value_target" in ex)
            for ex in batch
        )
        if not has_supervision:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "optimized": False,
                "status": "MISSING_SUPERVISION"
            }

        self.model.train()
        self.optimizer.zero_grad()

        # Check if batch contains tensor-ready examples (leaf_tokens)
        has_leaf_tokens = all("leaf_tokens" in ex and "candidate_ids" in ex for ex in batch)
        if has_leaf_tokens and hasattr(self.model, "scalar"):
            try:
                pad_token = getattr(self.model, "pad_token_id", 0)
                logits, valid, values = self.model(batch, pad_token=pad_token, return_value=True)

                # Mask out invalid candidate positions before log_softmax so probability mass
                # is distributed strictly across valid candidate actions (Probe R8_T01)
                if hasattr(valid, "dtype") and valid.dtype == torch.bool:
                    logits = logits.masked_fill(~valid, -1e9)
                log_probs = F.log_softmax(logits, dim=-1)
                target_probs = torch.zeros_like(logits)
                device = logits.device
                has_policy_mask = torch.zeros(len(batch), dtype=torch.bool, device=device)
                has_value_mask = torch.zeros(len(batch), dtype=torch.bool, device=device)
                target_values = torch.zeros((len(batch), 1), device=device)

                for i, ex in enumerate(batch):
                    cands = ex.get("candidate_ids", [])
                    pi_tgt = ex.get("pi_target")
                    if pi_tgt is not None:
                        # Probe T06: verify all target probabilities are finite (not NaN, not Inf)
                        tgt_items = []
                        if isinstance(pi_tgt, dict):
                            tgt_items = list(pi_tgt.values())
                        elif isinstance(pi_tgt, (list, tuple)):
                            tgt_items = list(pi_tgt)
                        elif hasattr(pi_tgt, "tolist"):
                            tgt_items = pi_tgt.tolist()
                            if isinstance(tgt_items, (int, float)):
                                tgt_items = [tgt_items]

                        for v in tgt_items:
                            try:
                                fv = float(v)
                                if not math.isfinite(fv):
                                    return {
                                        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                                        "optimized": False, "status": "NON_FINITE_TARGETS"
                                    }
                            except (TypeError, ValueError):
                                return {
                                    "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                                    "optimized": False, "status": "NON_FINITE_TARGETS"
                                }

                        row_sum = 0.0
                        for a_idx, act in enumerate(cands):
                            if a_idx < target_probs.shape[1]:
                                val = 0.0
                                if isinstance(pi_tgt, dict):
                                    val = float(pi_tgt.get(act, 0.0))
                                elif isinstance(pi_tgt, (list, tuple)) and a_idx < len(pi_tgt):
                                    val = float(pi_tgt[a_idx])
                                if not math.isfinite(val):
                                    return {
                                        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                                        "optimized": False, "status": "NON_FINITE_TARGETS"
                                    }
                                if val < 0.0:
                                    return {
                                        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                                        "optimized": False, "status": "NEGATIVE_TARGET_PROBABILITIES"
                                    }
                                # Probe T14/T15: candidate valid mask alignment
                                is_cand_valid = True
                                if hasattr(valid, "dtype") and valid.dtype == torch.bool:
                                    is_cand_valid = bool(valid[i, a_idx].item())
                                if is_cand_valid:
                                    target_probs[i, a_idx] = val
                                    row_sum += val
                                else:
                                    target_probs[i, a_idx] = 0.0

                        # Only mark policy supervision if probability mass on valid candidates is strictly positive
                        if row_sum > 1e-6:
                            has_policy_mask[i] = True
                            target_probs[i] = target_probs[i] / row_sum
                        else:
                            has_policy_mask[i] = False

                    if "value_target" in ex and ex.get("value_target") is not None:
                        v_tgt = float(ex["value_target"])
                        if not math.isfinite(v_tgt):
                            return {
                                "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                                "optimized": False, "status": "NON_FINITE_TARGETS"
                            }
                        target_values[i, 0] = v_tgt
                        has_value_mask[i] = True

                # If no effective supervision exists on any sample for either head, skip step
                if not has_policy_mask.any() and not has_value_mask.any():
                    return {
                        "loss": 0.0,
                        "policy_loss": 0.0,
                        "value_loss": 0.0,
                        "optimized": False,
                        "status": "MISSING_SUPERVISION"
                    }

                # Compute policy loss strictly on samples with valid policy supervision
                if has_policy_mask.any():
                    valid_mask = valid[has_policy_mask].float()
                    p_loss = -(target_probs[has_policy_mask] * log_probs[has_policy_mask] * valid_mask).sum(dim=-1).mean()
                else:
                    p_loss = torch.tensor(0.0, device=device)

                # Compute value loss strictly on samples with explicit value supervision
                if values is not None and has_value_mask.any():
                    v_pred = values.view(-1, 1)[has_value_mask]
                    v_tgt_sub = target_values[has_value_mask]
                    v_loss = F.mse_loss(v_pred, v_tgt_sub)
                else:
                    v_loss = torch.tensor(0.0, device=device)

                total_loss = p_loss + self.value_weight * v_loss

                # Reject non-finite loss before backward or optimization
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    return {
                        "loss": 0.0,
                        "policy_loss": 0.0,
                        "value_loss": 0.0,
                        "optimized": False,
                        "status": "NON_FINITE_LOSS"
                    }

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Verify parameters remain finite after optimization
                for p in self.model.parameters():
                    if not torch.isfinite(p).all():
                        return {
                            "loss": 0.0,
                            "policy_loss": 0.0,
                            "value_loss": 0.0,
                            "optimized": False,
                            "status": "NON_FINITE_WEIGHTS"
                        }

                return {
                    "loss": round(float(total_loss.item()), 4),
                    "policy_loss": round(float(p_loss.item()), 4),
                    "value_loss": round(float(v_loss.item()), 4),
                    "optimized": True,
                    "status": "OPTIMIZED"
                }
            except Exception as e:
                # Real training failure must raise or report meaningful error
                raise RuntimeError(f"GenZeroDistiller optimization step failed: {e}") from e

        # Explicit fallback if batch does not contain leaf_tokens or model cannot be trained
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "optimized": False, "status": "MISSING_LEAF_TOKENS"}


    def run_iteration(self, steps: int = 50, batch_size: int = 16) -> Dict[str, Any]:
        """Executes a bounded training iteration tracking actual optimizer steps."""
        losses = []
        actual_steps = 0
        for _ in range(steps):
            batch = self.replay_buffer.sample_batch(batch_size=batch_size)
            if not batch:
                continue
            metrics = self.train_step(batch)
            if metrics.get("optimized", False):
                actual_steps += 1
                losses.append(metrics["loss"])

        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        return {
            "steps_trained": actual_steps,
            "mean_loss": round(avg_loss, 4),
            "buffer_stats": self.replay_buffer.stats,
            "status": "SUCCESS" if actual_steps > 0 else "NO_EFFECTIVE_STEPS"
        }
