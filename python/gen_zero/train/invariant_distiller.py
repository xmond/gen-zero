"""Gen-Zero Layer 4: Invariant Risk Minimization (IRM) Causal Distiller.

Implements Causal Invariant Distillation based on Arjovsky et al. (IRMv1) combined with
Judea Pearl's Counterfactual Individual Treatment Effect (ITE) weighting.

Objective:
    min_{Phi, w} sum_{e in E} L^e(w . Phi(x)) + lambda_irm * ||grad_{w|w=1.0} L^e(w . Phi(x))||^2
    with causal weighting: w_sample = (1.0 + beta * ITE) for causal decision errors.

Minimizes empirical policy/value risk with an IRMv1 gradient penalty across
environments. This objective does not establish out-of-distribution invariance.
The compatibility metric ``invariance_score`` is a clipped transform of the
training penalty, not a measured OOD accuracy or invariance probability.
"""

from typing import Dict, List, Any, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from .replay_buffer import StabilityReplayBuffer


class InvariantCausalDistiller:
    """Invariant Risk Minimization (IRM) Causal Policy-Value Distiller."""

    def __init__(
        self,
        model: Any,
        replay_buffer: StabilityReplayBuffer,
        lr: float = 2e-5,
        value_weight: float = 0.5,
        irm_lambda: float = 0.1,
        causal_ite_beta: float = 1.5,
        device: str = "cuda"
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.value_weight = value_weight
        self.irm_lambda = irm_lambda
        self.causal_ite_beta = causal_ite_beta
        self.device = device if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        if HAS_TORCH and hasattr(model, "parameters"):
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = None

    def partition_into_environments(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Partitions batch into environment subsets E based on metadata or disturbance regimes."""
        envs: Dict[str, List[Dict[str, Any]]] = {}

        for item in batch:
            # Check explicit environment_id
            env_id = item.get("environment_id")
            if not env_id:
                # Dynamically partition by causal culprit vs nominal anchor
                if item.get("is_causal_culprit", False):
                    env_id = "env_causal_decision_error"
                elif item.get("is_hard_sample", False):
                    env_id = "env_high_entropy_hard"
                else:
                    env_id = "env_gold_anchor"

            if env_id not in envs:
                envs[env_id] = []
            envs[env_id].append(item)

        return envs

    def compute_environment_loss(
        self,
        env_batch: List[Dict[str, Any]]
    ) -> Tuple[Any, Any, Any]:
        """Computes empirical task loss and IRM gradient norm penalty for an environment subset."""
        if not HAS_TORCH:
            raise RuntimeError("Torch is required to compute invariant distillation loss")
        if not env_batch:
            raise ValueError("Environment batch must not be empty")
        if not all("leaf_tokens" in ex and "candidate_ids" in ex for ex in env_batch):
            raise ValueError("Environment samples require leaf_tokens and candidate_ids")
        if not hasattr(self.model, "scalar"):
            raise TypeError("Invariant distillation requires a model with a scalar head")

        pad_token = getattr(self.model, "pad_token_id", 0)
        logits, valid, values = self.model(env_batch, pad_token=pad_token, return_value=True)

        # Multiplier scalar for IRM gradient penalty w=1.0
        dummy_w = torch.tensor(1.0, device=logits.device, requires_grad=True)
        scaled_logits = logits * dummy_w

        log_probs = F.log_softmax(scaled_logits, dim=-1)
        target_probs = torch.zeros_like(logits)
        target_values = torch.zeros((len(env_batch), 1), device=logits.device)
        sample_weights = torch.ones(len(env_batch), device=logits.device)

        for i, ex in enumerate(env_batch):
            cands = ex.get("candidate_ids", [])
            pi_tgt = ex.get("soft_target") or ex.get("pi_target", {})
            v_tgt = float(ex.get("value_target", 0.0))
            target_values[i, 0] = v_tgt

            # Causal ITE weighting: scale up true decision errors
            if ex.get("is_causal_culprit", False):
                ite = float(ex.get("ite", 0.5))
                sample_weights[i] = 1.0 + self.causal_ite_beta * ite

            for a_idx, act in enumerate(cands):
                if a_idx < target_probs.shape[1]:
                    if isinstance(pi_tgt, dict):
                        target_probs[i, a_idx] = float(pi_tgt.get(act, 0.0))
                    elif isinstance(pi_tgt, (list, tuple)) and a_idx < len(pi_tgt):
                        target_probs[i, a_idx] = float(pi_tgt[a_idx])

        valid_mask = valid.float()
        target_probs = target_probs * valid_mask
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        per_sample_p_loss = -(target_probs * log_probs * valid_mask).sum(dim=-1)
        weighted_p_loss = (per_sample_p_loss * sample_weights).mean()
        if values is not None and target_values is not None:
            t_val = target_values.squeeze(-1) if target_values.dim() > values.dim() else target_values
            v_val = (values * dummy_w).squeeze(-1) if (values * dummy_w).dim() > t_val.dim() else (values * dummy_w)
            weighted_v_loss = F.mse_loss(v_val, t_val)
        else:
            weighted_v_loss = logits.new_tensor(0.0)

        env_task_loss = weighted_p_loss + self.value_weight * weighted_v_loss

        # Calculate IRM gradient norm with respect to dummy scalar w
        grad_w = torch.autograd.grad(env_task_loss, dummy_w, create_graph=True)[0]
        irm_penalty = (grad_w ** 2).sum()

        return weighted_p_loss, weighted_v_loss, irm_penalty

    def train_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Executes one Invariant Risk Minimization optimization step across environments."""
        if not HAS_TORCH or self.optimizer is None:
            raise RuntimeError("Torch and an active optimizer are required for invariant distillation training step")
        if not batch:
            raise ValueError("Training batch must not be empty")
        envs = self.partition_into_environments(batch)

        self.model.train()
        self.optimizer.zero_grad()

        total_erm = torch.tensor(0.0, device=self.device)
        total_irm_penalty = torch.tensor(0.0, device=self.device)

        for env_batch in envs.values():
            p_loss, v_loss, irm_pen = self.compute_environment_loss(env_batch)
            task_loss = p_loss + self.value_weight * v_loss
            total_erm = total_erm + task_loss
            total_irm_penalty = total_irm_penalty + irm_pen

        total_loss = (total_erm + self.irm_lambda * total_irm_penalty) / len(envs)
        if not total_loss.requires_grad:
            raise RuntimeError("Invariant distillation loss is detached from autograd")
        if not torch.isfinite(total_loss).all():
            raise RuntimeError("Invariant distillation loss is not finite")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.optimizer.step()

        loss_val = float(total_loss.item())
        erm_val = float(total_erm.item()) / len(envs)
        irm_pen_val = float(total_irm_penalty.item()) / len(envs)

        invariance_score = max(0.0, min(1.0, 1.0 - irm_pen_val * 2.0))

        return {
            "loss": round(loss_val, 4),
            "erm_loss": round(erm_val, 4),
            "irm_penalty": round(irm_pen_val, 4),
            "invariance_score": round(invariance_score, 4),
            "num_environments": len(envs)
        }

    def run_iteration(self, steps: int = 50, batch_size: int = 16) -> Dict[str, Any]:
        """Executes a bounded IRM distillation iteration."""
        if steps <= 0 or batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        losses = []
        erm_losses = []
        irm_penalties = []
        invariance_scores = []

        for _ in range(steps):
            batch = self.replay_buffer.sample_batch(batch_size=batch_size)
            metrics = self.train_step(batch)
            losses.append(metrics["loss"])
            erm_losses.append(metrics["erm_loss"])
            irm_penalties.append(metrics["irm_penalty"])
            invariance_scores.append(metrics["invariance_score"])

        avg_loss = sum(losses) / max(1, len(losses))
        avg_erm = sum(erm_losses) / max(1, len(erm_losses))
        avg_irm = sum(irm_penalties) / max(1, len(irm_penalties))
        avg_inv = sum(invariance_scores) / max(1, len(invariance_scores))

        return {
            "steps_trained": steps,
            "mean_loss": round(avg_loss, 4),
            "mean_erm_loss": round(avg_erm, 4),
            "mean_irm_penalty": round(avg_irm, 4),
            "invariance_score": round(avg_inv, 4),
            "buffer_stats": self.replay_buffer.stats
        }
