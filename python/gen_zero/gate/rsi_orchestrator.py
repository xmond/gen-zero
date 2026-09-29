"""Gen-Zero Layer 5: Recursive Self-Improvement (RSI) Flywheel Orchestrator.

Orchestrates the continuous online learning flywheel:
1. Online Agent Rollout (MCTS + Current Model)
2. Hard Sample Mining (Collision, High Entropy, TD Error)
3. 1:3 Stability Replay Buffer Update
4. Multi-Task Policy-Value Distillation
5. Automated Safety Gate Evaluation
6. Meta^n Dynamic Stopping Criterion (I-24 Principle)
"""

import time
from typing import Dict, List, Any, Optional
from ..config import GenZeroConfig
from ..rollout.runner import UnifiedEnvironmentRunner
from ..rollout.hard_miner import HardSampleMiner
from ..train.replay_buffer import StabilityReplayBuffer
from ..train.distiller import GenZeroDistiller
from .safety_gate import SafetyGate, GateVerdict


class RSIOrchestrator:
    """Master Orchestrator for Gen-Zero Recursive Self-Improvement."""
    def __init__(
        self,
        config: Optional[GenZeroConfig] = None,
        runner: Optional[UnifiedEnvironmentRunner] = None,
        replay_buffer: Optional[StabilityReplayBuffer] = None,
        distiller: Optional[GenZeroDistiller] = None,
        gate: Optional[SafetyGate] = None
    ):
        self.config = config or GenZeroConfig()
        self.runner = runner if runner is not None else UnifiedEnvironmentRunner()
        self.replay_buffer = replay_buffer if replay_buffer is not None else StabilityReplayBuffer(hard_ratio=self.config.hard_to_gold_ratio)
        self.distiller = distiller if distiller is not None else GenZeroDistiller(model=None, replay_buffer=self.replay_buffer)
        if distiller is not None and distiller.replay_buffer is not self.replay_buffer:
            # Synchronize distiller's buffer reference to ensure shared replay pool
            distiller.replay_buffer = self.replay_buffer
        self.gate = gate if gate is not None else SafetyGate(
            min_acc_retention=self.config.gate_min_accuracy_retention,
            min_score_gain=self.config.gate_min_score_gain
        )
        self.round_history: List[Dict[str, Any]] = []

    def execute_flywheel_round(
        self,
        round_idx: int,
        env: Any,
        policy_fn: Any,
        eval_fn: Any,
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Executes a complete single round of the Gen-Zero self-evolution flywheel."""
        t0 = time.time()
        # Checkpoint baseline model weights if model exists
        model = getattr(self.distiller, "model", None)
        checkpoint = None
        if model is not None and hasattr(model, "state_dict"):
            import copy
            checkpoint = copy.deepcopy(model.state_dict())

        # 1. Online Rollout & Hard Mining
        rollout_res = self.runner.run_episode(env, policy_fn, max_steps=256)
        mined_samples = rollout_res["mined_samples"]
        print(f"  [1. Rollout & Mining] Steps: {rollout_res['steps']}, Outcome: {rollout_res['outcome']}, Mined Hard Samples: {len(mined_samples)}")

        # 2. Add to Replay Buffer (1:3 mixing)
        self.replay_buffer.add_mined_samples(mined_samples)
        buffer_stats = self.replay_buffer.stats
        print(f"  [2. Replay Buffer] Pool status: {buffer_stats['hard_samples']} hard, {buffer_stats['gold_samples']} gold, Total: {buffer_stats['total_samples']}")

        # 3. Dual-Head Distillation
        train_res = self.distiller.run_iteration(steps=30, batch_size=self.config.batch_size)
        print(f"  [3. Multi-Task Distillation] Mean Loss: {train_res['mean_loss']}")

        # 4. Automated Safety Gate Evaluation on Frozen Benchmark
        candidate_metrics = eval_fn()
        verdict = self.gate.evaluate_candidate(baseline_metrics, candidate_metrics)
        print(f"  [4. Safety Gate] Passed: {verdict.passed} -> Action: {verdict.action}")
        print(f"     Accuracy Delta: {verdict.accuracy_delta}%, Score Delta: {verdict.score_delta}, Collision Delta: {verdict.collision_delta}")

        # True rollback if candidate failed safety gate
        if not verdict.passed:
            if checkpoint is not None and model is not None:
                model.load_state_dict(checkpoint)
                print("  [Safety Rollback] Restored previous checkpoint weights.")
            self.replay_buffer.pop_recent_mined_samples(len(mined_samples))
            acc_retention = candidate_metrics.get("accuracy", 0.0) / max(1e-4, baseline_metrics.get("accuracy", 1.0))
            new_ratio = self.replay_buffer.update_retention_feedback(acc_retention)
            print(f"  [Safety Rollback] Purged {len(mined_samples)} rejected hard samples. Adjusted hard_ratio to {new_ratio}.")

        elapsed = time.time() - t0
        round_record = {
            "round": round_idx,
            "mined_samples_count": len(mined_samples),
            "train_loss": train_res["mean_loss"],
            "candidate_metrics": candidate_metrics,
            "gate_verdict": {
                "passed": verdict.passed,
                "action": verdict.action,
                "accuracy_delta": verdict.accuracy_delta,
                "score_delta": verdict.score_delta
            },
            "elapsed_seconds": round(elapsed, 2)
        }
        self.round_history.append(round_record)
        return round_record

    def check_metan_convergence(self) -> bool:
        """I-24 Meta^n Principle: Checks if incremental gain has saturated (< delta for 2 rounds)."""
        if len(self.round_history) < 2:
            return False

        r1 = self.round_history[-2]["gate_verdict"]["score_delta"]
        r2 = self.round_history[-1]["gate_verdict"]["score_delta"]
        delta = abs(r2 - r1)
        if delta < self.config.metan_convergence_delta and self.round_history[-1]["gate_verdict"]["passed"]:
            print(f"\n[Meta^n Convergence Criterion Met] Delta gain ({delta:.4f}) < threshold ({self.config.metan_convergence_delta}). Saturation reached. Stopping flywheel.")
            return True
        return False
