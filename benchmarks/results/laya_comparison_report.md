# Laya vs Gen-Zero (RFC-072) measured report

> **Synthetic benchmark:** this report measures generated grid/trap and parity fixtures; it does not support a real-world or general capability claim.

Generated 2026-09-23T22:14:15+0900; load average at report (11.25, 9.82666015625, 12.84912109375).

| Arm | Trap survival | Regular acc | ECE | Brier | Median ms wall (Gen-Zero head-only, Laya end-to-end) | Median ms CPU | MCTS rate |
|---|---|---|---|---|---|---|---|
| A: Laya convaiinnovations/laya-typed-decisions (421M ModernBERT, single-step) | 0/30 (0.0%) | 43.3% | 0.212 | 0.249 | 841.440 | 3351.869 | 0.0% |
| A: Laya convaiinnovations/laya (421M ModernBERT, single-step) | 0/30 (0.0%) | 46.7% | 0.267 | 0.277 | 891.202 | 3510.746 | 0.0% |
| B: Parallel RNN + Set-Transformer (no MCTS) | 0/30 (0.0%) | 100.0% | 0.187 | 0.191 | 0.321 | 0.347 | 0.0% |
| Full: Causal MCTS + RNN + Set (adaptive gate) | 30/30 (100.0%) | 100.0% | 0.166 | 0.096 | 4.813 | 4.815 | 22.2% |
| Ablation: always MCTS | 30/30 (100.0%) | 100.0% | 0.140 | 0.088 | 17.529 | 17.532 | 100.0% |
| Ablation: entropy-only gate | 8/30 (26.7%) | 100.0% | 0.272 | 0.179 | 0.631 | 0.633 | 14.2% |
| Ablation: one rollout per action, no tree | 30/30 (100.0%) | 93.3% | 0.414 | 0.270 | 3.059 | 3.061 | 0.0% |
| Ablation: dynamics fit from 1500 transitions | 6/30 (20.0%) | 95.0% | 0.084 | 0.152 | 5.492 | 5.524 | 41.2% |
| Ablation: dynamics fit from 4000 transitions | 25/30 (83.3%) | 93.3% | 0.115 | 0.126 | 4.952 | 5.011 | 31.5% |

| MCQ arm | Items | Accuracy | ECE | Brier (binary) | Median ms (wall) |
|---|---|---|---|---|---|
| Gen-Zero B | parity odd half (100) | 66.0% (all 200: 65.0%) | 0.105 | 0.182 | 1.187 (head only) |
| Gen-Zero Full | parity odd half (100) | 66.0% (all 200: 65.0%) | 0.103 | 0.183 | 10.775 (head only) |
| Laya laya-typed-decisions | 200 other val rows | 24.5% | 0.088 | 0.191 | 989.0 (text end-to-end) |
| Laya laya | 200 other val rows | 20.5% | 0.342 | 0.302 | 999.9 (text end-to-end) |

## Findings

- One greedy rollout per action with no tree survives 30/30 traps (Full: 30/30). Survival comes from model-based look-ahead; the tree adds regular accuracy (93.3% -> 100.0%) and calibration (ECE 0.414 -> 0.166), not survival.
- Entropy-only gating survives 8/30: deceptive traps are confident mistakes, so a low-entropy gate sends them down the fast path. The probe-rollout trigger fires on 12.6% of Full's decisions.
- Dynamics from 1500 transitions (60% state-action coverage): survival 6/30.
- Dynamics from 4000 transitions (75% state-action coverage): survival 25/30.
- Latency: Full fast path median 4.55 ms, MCTS path median 27.1 ms (RFC-072 table claims 2.3-7.4 ms). Lie step: 20.8 us single at d=408, 2.1 us at d=64 (RFC claims <=3.2 us).
- ACT steps: grid 9-16, MCQ 4-4; the halting step tracks the contraction of A, not task difficulty.
- MCQ: general Laya 20.5%, grid regular accuracy 46.7%, trap survival 0/30.
- Learned dynamics: 81.3% of all state-action pairs decode to the true next state. Of the 224 wrong pairs, 223 start on the always-lethal border ring (never observed as a source, terminal for the planner) and 1 elsewhere; 0 wrong pairs start from a non-lethal state reachable within depth+3 steps of a test trap. Damper: off-manifold norm after a 16-step rollout is 2.12e-07 with it, 1.52e-07 without, so in this benchmark it has no measurable effect; its contraction is verified by unit tests only.
- MCQ: Laya 24.5% on 200 val rows vs Gen-Zero B 65.0% on 200 different val rows. Both Laya checkpoints (typed-decisions specialist and general) are near chance on these tasks with this prompt format; Laya's own published numbers are on its typed-decisions benchmark, which was not run here.
- Regular grid states are selected so that one-step greedy is optimal; 100% there shows MCTS does not hurt easy cases, not general competence.

## Caveats

- Traps are built so that the one-step signal points into the corridor; B is trained on that one-step signal only. B failing deep traps is the expected behaviour of a single-step mapper, measured, not asserted.
- The dynamics are exact permutations on orthonormal codes, which is why a rotation can model them. A rotation cannot model many-to-one transitions; irreversibility here comes from terminal readouts.
- With full exploration the learned model is essentially exact, so MCTS then plans on a near-perfect simulator; the limited-exploration ablations show what happens when it is not.
- MCQ: Laya and Gen-Zero are scored on different rows of the same split, and Gen-Zero latency excludes the 9B feature extractor.
- Grid latency is head-only for Gen-Zero: per-layout feature and readout construction (GridContext, ~10 ms per map) is precomputed and not timed. Laya's time is end-to-end (tokenise + 421M forward). The latency columns are not like-for-like.

See laya_comparison_report.json for per-item records.
