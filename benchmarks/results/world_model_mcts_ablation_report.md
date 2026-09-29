# World model MCTS paired ablation

This diagnostic uses privileged exact graph dynamics, not a trained neural model. Results cannot establish neural-world-model gains or production planner gains.

Episodes: 100; seed range: 0–99.

| Policy | Success % | Trap % | Mean steps | Mean decision latency ms | Model calls |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 100.00 | 1.00 | 0.0106 | 0 |
| world_model | 100.00 | 0.00 | 4.71 | 0.6816 | 145364 |

Exact paired McNemar: baseline-only=0, model-only=100, p=1.57772e-30.

Paired per-seed outcomes and timing are in the adjacent JSON.
