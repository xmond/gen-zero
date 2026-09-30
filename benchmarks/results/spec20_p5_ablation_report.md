# Spec 20 P5: combined ablation & final grand scorecard (real data where it exists, explicit not_evaluated everywhere else)

Generated: 2026-09-24T09:29:39Z  
Spec: `docs/zero/20-triad-deep-enhancement-multi-dimensional-analysis-spec.md`

## 0. Headline

q9b_diff_compact (8192-D) x {linear_probe, adapter_formulation_a, supcon} with strict nested 1-SE selection = macro 76.06% over 13 tasks, beating Jev (76.0%, delta +0.06pp) and Nimble (74.8%, delta +1.26pp).

**This +0.06pp vs Jev is carried in part by 1 collapsed task(s): civil_comments (majority-class train prior=92.1%, our accuracy=88.67%). Excluding them, macro over the remaining 12 tasks is ours=75.005% vs Jev=75.583% (delta -0.58pp) vs Nimble=75.192% (delta -0.19pp): the sign vs Jev FLIPS from positive to negative. The 13-task bootstrap CI on our own macro is [66.3, 84.03]; Jev and Nimble sit inside it too, i.e. those reference scores are not distinguishable from our macro at this sample size.**

**Everything else below with `status: not_evaluated` has no accuracy number in this report on purpose.**

## 1. Gating table (this script's assessment; the spec table itself has no status column)

| Phase | Status |
|---|---|
| P0_evidence_baseline | claimed passed in the task brief; partially spot-checked here (evidence: tasks.<task>.leakage_gate.q9b_diff_16_24_compact.{id_overlap,text_overlap,family_overlap} == 0 for all 13 tasks in 01png_sota_ensemble_report_phase4.json (checked programmatically below); full nested/group-CV audit not re-verified by this script) |
| P1_long_context_pooling | engineering done + unit tests pass; accuracy not evaluated (S8:445) |
| P2_layer_dynamics_diff | passed with real accuracy: this IS the 76.06% phase4 result |
| P3_folded_heads | engineering done + unit tests pass; accuracy not evaluated (S8:445) |
| P4_dual_source_fusion | engineering done + unit tests pass; accuracy not evaluated (S8:444-445), Gemma features never extracted |
| P5_combined_ablation | this report: only P0+P2 have a real jointly-evaluated cell; P1/P3/P4 cannot be honestly combined into it yet (S7.1: 'do not blind-search the full combination grid ... only combine axes that passed inner-layer training validation' -- P1/P3/P4 never ran the inner nested-CV/1-SE ladder on real data at all) |

P0 leakage-gate spot check: 13 tasks, all_clean=True, nonzero_overlap=[]

## 2. Axis grid

### axis1_representation

| Cell | Status | Detail |
|---|---|---|
| q9b_diff_compact_8192d | evaluated | synthesize_layer_diff_features.py:15 compact = [h24; h24-h16], 8192-D; evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "command (--features-dir ...q9b_diff_compact) and tasks.<task>.leakage_gate.q9b_diff_16_24_compact", "command_string": "D:\\genz\\benchmarks\\suites\\evaluate_full_13_grand_scorecard.py --features-dir D:\\genz\\features_uncap_v1\\q9b_diff_compact\\features --ranks 32,64,128 --device cuda --results-dir D:\\genz\\benchmarks\\results_phase4_compact"} |
| q9b_mid_8192d | not_evaluated | Spec 20 S8 'not verified'/'not done' (docs/zero/20-tri...; unit tests: n/a |
| q9b_diff_full_12288d | not_evaluated | Spec 20 S8 'not verified'/'not done' (docs/zero/20-tri...; unit tests: n/a |

### axis2_head

| Cell | Status | Detail |
|---|---|---|
| linear_probe | evaluated | evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "aggregate.macro_avg_13"} |
| adapter_formulation_a | evaluated | evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "aggregate.macro_avg_13"} |
| supcon | evaluated | evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "aggregate.macro_avg_13"} |
| adapter_formulation_b | not_evaluated | Spec 20 S8 'not verified'/'not done' (docs/zero/20-tri...; unit tests: 44 passed, 14 warnings in 8.63s |

### axis3_fusion

| Cell | Status | Detail |
|---|---|---|
| single_qwen | evaluated | evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "aggregate.macro_avg_13"} |
| dual_manifold_qwen_gemma | not_evaluated | Spec 20 S8 'not verified'/'not done' (docs/zero/20-tri...; unit tests: 20 passed, 4 warnings in 6.37s |

### pooling_axis

| Cell | Status | Detail |
|---|---|---|
| uniform_mean | evaluated_implicitly | The 13-task phase4 evaluation uses cached mean/last-token pooling baked into the q9b_diff_16_24_compact feature cache itself (synthesize_layer_diff_features.py), not a separate pooling module call; no task in the 13-task suite exercises >1536 tokens, so this is not a stress test of pooling choice.; evidence: {"file": "benchmarks/results/01png_sota_ensemble_report_phase4.json", "key": "aggregate.macro_avg_13"} |
| partition_anchor_pooling | not_evaluated | Spec 20 S8 'not verified'/'not done' (docs/zero/20-tri...; unit tests: 38 passed in 1.69s |

## 3. Per-task real scorecard (13 tasks, q9b_diff_compact + 1-SE selection)

Real trained-weight CPU decision latency (median/p95 µs) is phase4's own measurement for the head that actually won each task -- not this script's random-weight microbench.

| task | n | acc% | wilson95 | strategy | vs Jev | vs Nimble | collapsed | Jev in our CI | Nimble in our CI | latency median µs | latency p95 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aegis_safety | 250 | 73.6 | [67.81, 78.68] | linear_probe | -6.80 | -7.60 | False | False | False | 6.299989763647318 | 6.699992809444666 |
| boolq | 300 | 82.0 | [77.26, 85.93] | linear_probe | -7.70 | -4.00 | False | False | False | 6.200018106028438 | 6.599993503186852 |
| civil_comments | 300 | 88.67 | [84.58, 91.78] | adapter_r128 | +7.67 | +18.37 | True | False | False | 46.00000102072954 | 70.79998758854344 |
| helpsteer2 | 249 | 38.96 | [33.11, 45.14] | supcon_r128 | +4.86 | -0.04 | False | True | True | 73.89998063445091 | 87.59999764151871 |
| massive_de | 350 | 88.0 | [84.18, 91.0] | linear_probe | +1.10 | +4.60 | False | True | False | 9.500014130026102 | 11.520000407472258 |
| massive_en | 350 | 82.57 | [78.25, 86.19] | linear_probe | -4.83 | -4.33 | False | False | False | 12.79998105019331 | 15.00999787822366 |
| multinli | 299 | 83.28 | [78.63, 87.08] | linear_probe | +0.38 | -2.02 | False | True | True | 7.800001185387373 | 8.200004231184721 |
| paws | 250 | 88.8 | [84.29, 92.14] | adapter_r64 | -0.40 | +6.00 | False | True | False | 37.299992982298136 | 42.919993575196706 |
| pubmedqa | 250 | 68.4 | [62.4, 73.85] | adapter_r64 | -8.80 | -7.20 | False | False | False | 52.3999915458262 | 57.61498614447186 |
| squad2 | 299 | 86.96 | [82.67, 90.31] | adapter_r128 | +4.06 | +6.36 | False | True | False | 45.69999873638153 | 52.519992459565394 |
| summeval_consistency | 144 | 86.11 | [79.52, 90.83] | supcon_r32 | +4.91 | +10.41 | False | True | False | 45.39999645203352 | 55.680012155789875 |
| summeval_relevance | 240 | 41.25 | [35.21, 47.57] | adapter_r128 | +6.25 | -7.95 | False | False | False | 49.649999709799886 | 55.15001103049143 |
| vitaminc | 599 | 80.13 | [76.75, 83.13] | linear_probe | +0.03 | +3.53 | False | True | False | 7.800001185387373 | 8.200004231184721 |

## 4. Macro summary

- macro_avg_13 = **76.06%** (task-level bootstrap 95% CI [66.3, 84.03])
- vs Jev: +0.06pp, vs Nimble: +1.26pp
- collapsed tasks: ['civil_comments']
- **excluding collapsed tasks ['civil_comments']** (12 tasks): ours=75.005% vs Jev=75.583% (delta -0.58pp) vs Nimble=75.192% (delta -0.19pp)

## 5. 1-SE selector distribution (13 tasks, real)

linear_probe=6, adapter_formulation_a=5, supcon=2 (sum=13 of 13)

## 6. CPU latency (fresh measurement this run, µs, batch=1, float32, random weights)

Methodology: batch=1, float32, random weights (dense-kernel latency is weight-value-independent, only shape-dependent; same principle as spec19_head_latency_microbench.py); BLAS pinned to 1 thread (OMP/OPENBLAS/MKL_NUM_THREADS=1); reps=2000 warm=200 for head GEMVs, reps=200 warm=200 for O(T) pooling; median_us / p95_us over independently timed calls on this host, this run.

Host: host load was below cpu_count at run time.

Caveat: `linear_probe` below is a bare `x @ W.T + b` GEMV; `adapter_b` and `dual_manifold` are timed through their production `.scores()`, which also does input validation (`isfinite`, shape checks) and standardization. The ratio between them is not a pure FLOPs comparison; it is the real, honest cost of calling the production API.

| head | K | median_us | p95_us |
|---|---|---|---|
| adapter_a (=supcon) D8192 r64 | K2 | 168.59 | 225.41 |
| adapter_a (=supcon) D8192 r64 | K3 | 169.84 | 231.4 |
| adapter_a (=supcon) D8192 r64 | K18 | 207.89 | 271.99 |
| adapter_b D8192 r64 | K2 | 175.29 | 235.69 |
| adapter_b D8192 r64 | K3 | 175.89 | 241.91 |
| adapter_b D8192 r64 | K18 | 201.94 | 268.9 |
| dual_manifold DQ8192+DG2816 | K2 | 25.6 | 38.8 |
| dual_manifold DQ8192+DG2816 | K3 | 25.7 | 46.2 |
| dual_manifold DQ8192+DG2816 | K18 | 52.7 | 81.7 |
| linear_probe D8192 | K2 | 5.3 | 7.1 |
| linear_probe D8192 | K3 | 6.8 | 9.2 |
| linear_probe D8192 | K18 | 16.1 | 32.2 |

### Pooling (separate latency boundary: GPU-extraction-time pooling cost (paid once per document when caching features), NOT a per-query CPU decision-head cost -- do not compare directly to the head latencies above (S7.3: report latency boundaries as they are).)

| pooling | median_us | p95_us |
|---|---|---|
| partition_anchor_pooling T1536 D4096 | 84586.4 | 93902.75 |
| uniform_mean T1536 D4096 | 2162.12 | 2727.14 |

## 7. Unit test evidence (re-run live by this script, not cached)

| suite | exit_code | summary |
|---|---|---|
| P1_partition_anchor_pooling | 0 | 38 passed in 1.69s |
| P3_formulation_b_adapter | 0 | 44 passed, 14 warnings in 8.63s |
| P4_dual_manifold | 0 | 20 passed, 4 warnings in 6.37s |

## 8. Final verdict

q9b_diff_compact (8192-D) x {linear_probe, adapter_formulation_a, supcon} with strict nested 1-SE selection = macro 76.06% over 13 tasks, beating Jev (76.0%, delta +0.06pp) and Nimble (74.8%, delta +1.26pp).

**This +0.06pp vs Jev is carried in part by 1 collapsed task(s): civil_comments (majority-class train prior=92.1%, our accuracy=88.67%). Excluding them, macro over the remaining 12 tasks is ours=75.005% vs Jev=75.583% (delta -0.58pp) vs Nimble=75.192% (delta -0.19pp): the sign vs Jev FLIPS from positive to negative. The 13-task bootstrap CI on our own macro is [66.3, 84.03]; Jev and Nimble sit inside it too, i.e. those reference scores are not distinguishable from our macro at this sample size.**

Not yet combinable: q9b_mid_8192d, q9b_diff_full_12288d, adapter_formulation_b, dual_manifold_qwen_gemma, partition_anchor_pooling

Each has real, passing algorithmic-correctness unit tests (see unit_test_evidence) but was never run through the nested-CV training/selection ladder on the real 13-task data. Fabricating an accuracy delta for them would violate Spec 20 S7.2's own instruction ('do not fake dual-source success in the inference path based on ...') and this task's anti-cheating mandate. Real P5 numbers for these axes require the not-done work S8:451 names: GPU extraction, CPU training, and a new nested-CV test evaluation -- none of which this report-generation script performs.
