# World model MCTS paired ablation

This diagnostic uses privileged exact graph dynamics, not a trained neural model. Results cannot establish neural-world-model gains or production planner gains.

Episodes: 100; seed range: 0–99.

| Policy | Success % | Trap % | Mean steps | Mean decision latency ms | Model calls |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 100.00 | 1.00 | 0.0097 | 0 |
| world_model | 100.00 | 0.00 | 4.71 | 0.5202 | 145364 |

Exact paired McNemar: baseline-only=0, model-only=100, p=1.57772e-30.

Paired per-seed outcomes and timing are in the adjacent JSON.

## Production engine options

```
==============================================================================
MctsEngine options ablation (production engine.plan, budget 32, depth 6)
==============================================================================
[exact_graph_dict] 471 paired decisions
Config            |  Mean ms |  Med ms | Transitions | Sims run | Success % | Agree % | Ratio
default           |    0.646 |   0.690 |        72.3 |     32.0 |    100.00 |       - |     -
early_stop        |    0.616 |   0.652 |        69.0 |     24.6 |    100.00 |   100.0 |  0.91
cache             |    0.746 |   0.782 |        41.3 |     32.0 |    100.00 |   100.0 |  1.15
early_stop+cache  |    0.715 |   0.753 |        40.7 |     24.6 |    100.00 |   100.0 |  1.06
[latent_codes_dim4096] 471 paired decisions
Config            |  Mean ms |  Med ms | Transitions | Sims run | Success % | Agree % | Ratio
default           |   14.467 |  14.147 |        72.3 |     32.0 |    100.00 |       - |     -
early_stop        |   13.697 |  14.158 |        69.0 |     24.6 |    100.00 |   100.0 |  0.98
cache             |   10.537 |  10.370 |        41.3 |     32.0 |    100.00 |   100.0 |  0.75
early_stop+cache  |   10.200 |  10.252 |        40.7 |     24.6 |    100.00 |   100.0 |  0.76
------------------------------------------------------------------------------
Distractor-trap hit rate % (n=200 per cell, chance 12.5%)
prior=lure    sims=  8 | PUCT    0.0 | PUCT-uniform   15.0 | SeqHalving   41.5 | p_vs_PUCT=2.07e-25 p_vs_uniform=3.2e-09
prior=lure    sims= 16 | PUCT    0.0 | PUCT-uniform   24.5 | SeqHalving   77.5 | p_vs_PUCT=4.38e-47 p_vs_uniform=4.7e-23
prior=lure    sims= 32 | PUCT    0.0 | PUCT-uniform   66.5 | SeqHalving   84.5 | p_vs_PUCT=2.67e-51 p_vs_uniform=1.41e-05
prior=helpful sims=  8 | PUCT  100.0 | PUCT-uniform   15.0 | SeqHalving   47.5 | p_vs_PUCT=4.93e-32 p_vs_uniform=1.85e-12
prior=helpful sims= 16 | PUCT  100.0 | PUCT-uniform   24.5 | SeqHalving   85.5 | p_vs_PUCT=3.73e-09 p_vs_uniform=2.57e-28
prior=helpful sims= 32 | PUCT  100.0 | PUCT-uniform   66.5 | SeqHalving   92.5 | p_vs_PUCT=6.1e-05 p_vs_uniform=5.75e-11
------------------------------------------------------------------------------
LatentMctsPlanner early_stop (n=100, budget 200): agreement 100.0%, sims 107.4, ms 13.514 -> 7.876
==============================================================================
```
