# Spec 29: PubMedQA and Aegis 2.0 Evaluation Findings -- Corrections

- Date: 2026-09-26
- Original evaluation: [Unified evaluation report](../../benchmarks/results/unified_manifold_pareto_eval.md) and [machine-readable results](../../benchmarks/results/unified_manifold_pareto_eval.json)
- Scope: independent Python evaluation suite; `evaluate_manifold_pareto_ensemble.py`, referenced by the report, is missing from the current working tree and is not tracked by git, so this correction cannot re-run it. `bb38d1e` is not proof that this suite has been merged into the trunk or deployed.

## Observed results and statistical boundaries

| Task and basis | Selected result | Same-basis comparison | Supportable conclusion |
| --- | ---: | ---: | --- |
| PubMedQA, 250 test questions | Fusion `fuse0.75+bbp\|raw` 78.40% (196/250) | Qwen single-model 77.20% (193/250); LLaMA single-model 77.60% (194/250) | 3 more correct answers than Qwen, 2 more than LLaMA; both are descriptive differences only |
| Aegis Track A, full 250 questions | OOF-selected `llama+bbp\|raw` 81.60% (204/250) | Qwen single-model 81.20%; external Jev 80.40%, Nimble 81.20% | Only Track A's denominator may be placed side by side with the full external figures; whether the protocols fully match still needs verification |
| Aegis Track B, 225 questions after excluding 25 `Needs Caution` items | Historical report cites 84.44% (corresponding to 190/225; this repository has no row-level or result-file evidence for it) | No external baseline on the same subset | **Must never be compared cross-basis against the full 250-question Jev/Nimble results** |

The 78.40% figure for PubMedQA is a test-set observation from this OOF model-selection run, not an established SOTA. The original report's peak macro-average-accuracy target check (>=82.5) evaluates to **False**; this is not a significance test for the PubMedQA task alone. The paired bootstrap 95% CI of `[-0.20, +1.60]` pp reported corresponds to `peak_minus_peak_single` on the **macro average across both tasks** (point estimate +0.60 pp, 5,000 resamples over test rows within each task), **not** a PubMedQA-only interval. That interval contains 0. This repository has no verified per-item paired interval for PubMedQA alone, so a claim of statistically significant superiority over the single models or the external systems cannot be made. External Jev/Nimble lack matching per-item predictions from this run, so paired significance cannot be computed for them.

Aegis Track A's `concat+lda` fixed control reaches 82.80% on the same test set. This is a **post hoc control observation**, not a formally OOF-selected result, and must not be listed as the "highest point on the full test set" or as SOTA. The historically reported Track B figure of 84.44% is a conditional-subset accuracy that currently lacks source-result evidence; cross-denominator claims such as "exceeds Jev by +4.04 pp" are retracted, and the excluded samples are not characterized as confirmed noise.

## Method naming and scope of applicability

- The track originally labeled "Certified Robust" is in fact an empirical selection based on **OOF balanced accuracy and F1, with a 1-SE rule**. This is not a formal robustness proof; its two-task macro-average test balanced accuracy is 70.32% and F1 is 69.74%, and both target checks evaluate to False.
- The Aegis Margin Gate uses an empirically fixed threshold `theta=1.0`, abstaining or escalating when `|m|<theta`. It does **not** carry a split-conformal calibration-set quantile guarantee and must not be called a "conformal guarantee." A separate prediction-set experiment reports a marginal coverage of 92.4% and a minimum per-class coverage of 88.8% on PubMedQA; these coverage figures cannot be transferred to the Margin Gate as a guarantee.
- Where the original Aegis gating statistics report 0 missed high-risk samples, that holds only for that specific batch of test rows and the corresponding gating view. `oracle_high_risk` uses gold-standard labels and is a diagnostic view only; it must not be presented as a deployable detector or as zero missed detections in production.
- `aegis_dual_track.py` is invoked by the standalone evaluation entry point; this does not prove that the Rust CLI, HTTP, MCP, or production scheduling entry points invoke it or the Margin Gate. Production integration and deployment status are not verified by this report.

## Reproducibility references

The original run commands are recorded at the top of the [unified evaluation report](../../benchmarks/results/unified_manifold_pareto_eval.md); that historical report has no verifiable raw exit-code log, and this correction does not fabricate an `EXIT=0`. The PubMedQA and Track A figures appear in that report's `Per task`, `Per-task fixed controls`, `Paired bootstrap`, `Target check`, and `Conformal` sections. The Track B definition, denominator, and baseline-comparability flag are in [`aegis_dual_track.py`](../../benchmarks/suites/aegis_dual_track.py). A full re-run of the scripts, data, and environment is out of scope for this documentation correction.
