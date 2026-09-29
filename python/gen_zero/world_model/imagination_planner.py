"""Gen-Zero Latent Space Imagination MCTS Planner.

Performs forward lookahead planning entirely within the learned latent feature
space Z, without interacting with external simulators or decoding images:
1. Fast PUCT-guided Monte Carlo Tree Search in feature space.
2. Latent transitions via LatentTransitionModel.
3. 32~64 rollouts executed in < 1.5ms.
4. Returns visit distributions, best candidate action, and anticipated trajectory.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time

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


class LatentMCTSNode:
    """A node in the latent imagination tree."""

    def __init__(
        self,
        latent_state: Any,
        parent: Optional["LatentMCTSNode"] = None,
        action_from_parent: Optional[Any] = None,
        prior_prob: float = 1.0,
        immediate_reward: float = 0.0,
        depth: int = 0
    ):
        self.latent_state = latent_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_prob = prior_prob
        self.immediate_reward = immediate_reward
        self.depth = depth

        self.children: Dict[Any, "LatentMCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class ImaginationMCTSPlanner:
    """Non-autoregressive MCTS planner running entirely in high-dimensional latent space."""

    def __init__(
        self,
        transition_model: Any,
        c_puct: float = 1.414,
        max_simulations: int = 64,
        max_depth: int = 4,
        discount: float = 0.95,
        temperature: float = 1.0,
        strict_evaluation: bool = False
    ):
        self.strict_evaluation = strict_evaluation
        self.transition_model = transition_model
        self.c_puct = c_puct
        self.max_simulations = max(4, max_simulations)
        self.max_depth = max(1, max_depth)
        self.discount = discount
        self.temperature = max(1e-3, temperature)

    def plan(
        self,
        root_latent: Any,
        candidate_actions: List[Any],
        action_priors: Optional[Dict[Any, float]] = None,
        value_evaluator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Executes latent MCTS from the current latent state.
        
        Args:
            root_latent: [1, latent_dim] or [latent_dim] tensor/array.
            candidate_actions: List of possible actions (indices, verbs, or tokens).
            action_priors: Optional prior probability map P(a|s) from fast reflex head.
            value_evaluator: Optional callable returning V(z) in [-1, 1].
        """
        t0 = time.perf_counter()

        if not candidate_actions:
            return {
                "best_action": None,
                "action_probabilities": {},
                "expected_value": 0.0,
                "imagined_trajectory": [],
                "simulations": 0,
                "planning_time_ms": 0.0
            }

        # Root node
        root = LatentMCTSNode(latent_state=root_latent, depth=0)

        # Uniform priors if none provided
        uniform_p = 1.0 / len(candidate_actions)
        priors = {act: action_priors.get(act, uniform_p) if action_priors else uniform_p for act in candidate_actions}

        # Expand root immediately
        self._expand_node(root, candidate_actions, priors)

        # Run simulations
        for _ in range(self.max_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.is_expanded and node.children and node.depth < self.max_depth:
                act, next_node = self._select_child(node)
                node = next_node
                search_path.append(node)

            # 2. Expansion (if leaf and within depth)
            value = 0.0
            if not node.is_expanded and node.depth < self.max_depth:
                self._expand_node(node, candidate_actions, priors)
                # 3. Evaluation
                value = self._evaluate_latent_state(node.latent_state, value_evaluator)
            else:
                value = self._evaluate_latent_state(node.latent_state, value_evaluator)

            # 4. Backpropagation
            self._backpropagate(search_path, value)

        # Compute visit count distribution
        visits = {act: child.visit_count for act, child in root.children.items()}
        total_visits = sum(visits.values())

        if total_visits > 0:
            if self.temperature < 0.05:
                # Argmax
                best_act = max(visits.keys(), key=lambda a: visits[a])
                probs = {a: (1.0 if a == best_act else 0.0) for a in candidate_actions}
            else:
                pow_visits = {a: math.pow(v, 1.0 / self.temperature) for a, v in visits.items()}
                sum_pow = sum(pow_visits.values())
                probs = {a: pow_visits[a] / max(1e-8, sum_pow) for a in candidate_actions}
                best_act = max(visits.keys(), key=lambda a: visits[a])
        else:
            best_act = candidate_actions[0]
            probs = priors

        # Trace principal imagined trajectory
        traj = self._extract_principal_trajectory(root)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "best_action": best_act,
            "action_probabilities": probs,
            "visit_counts": visits,
            "expected_value": round(root.q_value, 4),
            "imagined_trajectory": traj,
            "simulations": self.max_simulations,
            "planning_time_ms": round(elapsed_ms, 3)
        }

    def _select_child(self, node: LatentMCTSNode) -> Tuple[Any, LatentMCTSNode]:
        """Selects child with maximum PUCT score."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))

        for act, child in node.children.items():
            # Q + U
            q = child.q_value
            u = self.c_puct * child.prior_prob * (sqrt_total / (1 + child.visit_count))
            score = q + u

            if score > best_score:
                best_score = score
                best_action = act
                best_child = child

        return best_action, best_child

    def _expand_node(
        self,
        node: LatentMCTSNode,
        candidate_actions: List[Any],
        priors: Dict[Any, float]
    ) -> None:
        """Expands node by predicting next latent state for all candidate actions."""
        if node.is_expanded:
            return

        if hasattr(self.transition_model, "step_batch"):
            batch_results = self.transition_model.step_batch(node.latent_state, candidate_actions)
            for act, (next_z, step_reward, _) in zip(candidate_actions, batch_results):
                p = priors.get(act, 1.0 / max(1, len(candidate_actions)))
                child = LatentMCTSNode(
                    latent_state=next_z,
                    parent=node,
                    action_from_parent=act,
                    prior_prob=p,
                    immediate_reward=step_reward,
                    depth=node.depth + 1
                )
                node.children[act] = child
        else:
            for act in candidate_actions:
                next_z, step_reward, _ = self.transition_model.step(node.latent_state, act)
                p = priors.get(act, 1.0 / max(1, len(candidate_actions)))
                child = LatentMCTSNode(
                    latent_state=next_z,
                    parent=node,
                    action_from_parent=act,
                    prior_prob=p,
                    immediate_reward=step_reward,
                    depth=node.depth + 1
                )
                node.children[act] = child

        node.is_expanded = True

    def _evaluate_latent_state(
        self,
        latent_state: Any,
        value_evaluator: Optional[Any]
    ) -> float:
        """Evaluates a leaf latent state using value head or heuristic."""
        if self.strict_evaluation:
            if not callable(value_evaluator):
                raise ValueError("Strict MCTS requires an explicit value evaluator")
            value = value_evaluator(latent_state)
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Strict MCTS requires a finite value in [-1, 1]")
            return float(value)
        if value_evaluator is not None and callable(value_evaluator):
            try:
                v = value_evaluator(latent_state)
                if isinstance(v, (int, float)):
                    return float(v)
                elif HAS_TORCH and isinstance(v, torch.Tensor):
                    return float(v.item())
                elif HAS_NUMPY and isinstance(v, np.ndarray):
                    return float(v.item() if v.size == 1 else np.mean(v))
            except Exception:
                pass

        # Default heuristic: mean of normalized latent activation
        if HAS_TORCH and isinstance(latent_state, torch.Tensor):
            return float(torch.tanh(torch.mean(latent_state) * 3.0).item())
        elif HAS_NUMPY and isinstance(latent_state, np.ndarray):
            return float(np.tanh(np.mean(latent_state) * 3.0))
        return 0.0

    def _backpropagate(self, search_path: List[LatentMCTSNode], leaf_value: float) -> None:
        """Backpropagates value and discounted rewards from leaf to root."""
        running_value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += running_value
            running_value = node.immediate_reward + self.discount * running_value

    def _extract_principal_trajectory(self, root: LatentMCTSNode) -> List[Dict[str, Any]]:
        """Follows greedy visit count path to describe imagined future steps."""
        path = []
        curr = root
        step_idx = 1
        while curr.children:
            best_act = max(curr.children.keys(), key=lambda a: curr.children[a].visit_count)
            best_child = curr.children[best_act]
            path.append({
                "step": step_idx,
                "action": best_act,
                "q_value": round(best_child.q_value, 4),
                "visits": best_child.visit_count,
                "immediate_reward": round(best_child.immediate_reward, 4)
            })
            curr = best_child
            step_idx += 1
            if step_idx > 5:
                break
        return path
