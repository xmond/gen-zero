"""Markdown rendering for benchmark_sota_ensemble.py. Reads only the report dict."""
from __future__ import annotations

import json
from typing import List, Tuple


def _p(x: float) -> str:
    return f"{x:.3g}"


def train_rows_text(rep: dict) -> str:
    """Train-row count per task from protocol.train_rows_per_task: 'N' if all tasks match, else 'min-max'.
    Never a hard-coded cap: the cap is configurable (GC_N_TRAIN*), so the text must follow the data."""
    counts = [int(v) for v in rep["protocol"].get("train_rows_per_task", {}).values()]
    if not counts:
        return "an unrecorded number of"
    lo, hi = min(counts), max(counts)
    return f"{lo:,}" if lo == hi else f"{lo:,} to {hi:,}"


def png_refs(rep: dict) -> Tuple[float, float, float]:
    """(jev, nimble, laya) reference macro accuracies, read from the report so no number is written twice."""
    ref = rep["protocol"]["png_reference_avg"]
    return float(ref["jev"]), float(ref["nimble"]), float(rep["aggregate"]["laya_a100_reported_macro_13"])


def render_md(rep: dict) -> str:
    a, rows = rep["aggregate"], rep["tasks"]
    jev, nimble, laya = png_refs(rep)
    L = [f"# {rep['title']}", "",
         f"Generated {rep['generated_utc']} by `{rep['command']}`. Raw data: "
         "`benchmarks/results/01png_sota_ensemble_report.json`.", "", "## Headline", ""]
    L.append(f"- **Ensemble macro {a['macro_avg_evaluated']}%** over {len(rows)} tasks (micro {a['micro_acc']}%). "
             f"Jev {jev}% (delta {a['delta_macro_vs_jev']}), Nimble {nimble}% (delta {a['delta_macro_vs_nimble']}), "
             f"Laya {laya}% (delta {a['delta_macro_vs_laya']}).")
    if "prior_baseline_macro" in a:
        L.append(f"- Prior single heads on the same test rows: Baseline {a['prior_baseline_macro']}%, "
                 f"Deep-Wide {a.get('prior_deep_wide_macro')}%.")
        L.append(f"- Exact McNemar p<0.05 vs prior Baseline: better on "
                 f"{', '.join(a['tasks_sig_better_than_prior_baseline_p05']) or 'none'}; worse on "
                 f"{', '.join(a['tasks_sig_worse_than_prior_baseline_p05']) or 'none'}.")
    L.append(f"- Beats Jev on: {', '.join(a['tasks_beating_jev']) or 'none'}. "
             f"Beats the better of Nimble/Jev on: {', '.join(a['tasks_beating_best_01png']) or 'none'} "
             "(majority-class collapses excluded from this count, per Spec 19 S6.5).")
    if a.get("tasks_collapsed"):
        L.append(f"- **Majority-class collapse** (max predicted class > 95% of predictions, K>=2): "
                 f"{', '.join(a['tasks_collapsed'])}. Of those, "
                 f"{', '.join(a['tasks_beating_best_01png_collapsed']) or 'none'} were numerically ahead of the "
                 "01.PNG commercial baseline but are NOT counted as a win: the head learned a class prior, not "
                 "discriminative ability.")
    L.append(f"- CPU decision latency (all selected experts + fusion, features given): median over tasks "
             f"{a['decision_latency_ms_median_over_tasks']:.2f} ms, worst task p95 {a['decision_latency_ms_max_p95']:.2f} ms.")
    L += ["", "## Per task (accuracy %, 95% Wilson CI)", "",
          "| Task | n | Ensemble | Balanced Acc | Macro F1 | vs 01PNG | "
          "chosen strategy (nested-CV acc on TRAIN) | Prior Baseline | Δ | McNemar p | "
          "Prior Deep-Wide | Δ | McNemar p | Majority | Nimble | Jev | Δ vs Jev |",
          "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in rows.items():
        b = r["vs_prior_heads"].get("baseline", {})
        d = r["vs_prior_heads"].get("deep_wide", {})

        def cell(x: dict) -> str:
            if not x:
                return "- | - | -"
            return f"{x['accuracy']} | {x['delta']:+.2f} | {_p(x['mcnemar']['p_value'])}"
        L.append(f"| {r['dataset']} | {r['n']} | {r['accuracy']} [{r['wilson95'][0]}, {r['wilson95'][1]}] | "
                 f"{r['balanced_accuracy']} | {r['macro_f1']} | {r['win_marker']} | "
                 f"`{r['chosen_strategy']}` ({r['nested_cv_acc_chosen']}) | {cell(b)} | {cell(d)} | "
                 f"{r['majority_class_train_prior_acc']} | {r['nimble']} | {r['jev']} | {r['delta_vs_jev']:+.2f} |")
    L += ["", "## CPU decision latency (ms per record)", "",
          "| Task | median | p95 | experts run | encoder batch-1 median (per source) |", "|---|---:|---:|---:|---|"]
    for t, r in rows.items():
        lat = r["decision_latency_ms"]
        enc = "; ".join(f"{s}: {v['whole_context_median']:.0f}" if v["whole_context_median"] is not None
                        else f"{s}: n/a" for s, v in r["encoder_latency_ms_batch1"].items())
        L.append(f"| {r['dataset']} | {lat['median']:.2f} | {lat['p95']:.2f} | {lat['n_experts_run']} | {enc} |")
    h = rep["host"]
    L += ["", f"Host loadavg at eval start/end: {h['loadavg_at_eval_start']} / {h['loadavg_at_eval_end']} on "
          f"{h['cpu_count']} CPUs; BLAS env {h['blas_threads_env']}. Latency excludes the text encoder.", "",
          "## Post-hoc test accuracy of every expert (NOT used for any selection)", ""]
    for t, r in rows.items():
        best = sorted(r["post_hoc_test_acc_per_expert_NOT_used_for_selection"].items(), key=lambda kv: -kv[1])
        L.append(f"- {r['dataset']}: " + ", ".join(f"{k} {v}" for k, v in best))
    L += ["", "## Protocol", "", "```json", json.dumps(rep["protocol"], indent=1, ensure_ascii=False), "```", ""]
    L += status_section(rep)
    return "\n".join(L)


def status_section(rep: dict) -> List[str]:
    a, rows = rep["aggregate"], rep["tasks"]
    n_ok = all(r["n"] == r["n_expected_01png"] for r in rows.values())
    gate_zero = all(g["id_overlap"] == 0 and g["text_overlap"] == 0 and g["family_overlap"] == 0
                    for r in rows.values() for g in r["leakage_gate"].values())
    repro = all(x.get("reproduces_prior_report") for r in rows.values() for x in r["vs_prior_heads"].values())
    jev = png_refs(rep)[0]
    above = a["macro_avg_evaluated"] > jev
    n_rows = train_rows_text(rep)
    return [
        "## Verdict", "",
        f"Ensemble macro **{a['macro_avg_evaluated']}%** is {'above' if above else 'below'} Jev ({jev}%) by "
        f"{a['delta_macro_vs_jev']} points" + ("" if above else "; the SOTA target is NOT reached") + ".", "",
        "## Status (three classes)", "",
        "### Implemented, with evidence", "",
        f"- Test set = 01.PNG record counts on every task: {n_ok} (sha256 per file in JSON `test_file`).",
        f"- Leakage gate id/family/normalized-text overlap zero for every source: {gate_zero}.",
        f"- The reused Baseline/Deep-Wide heads reproduce the prior report's accuracy on every task: {repro} "
        "(JSON `vs_prior_heads.*.reproduces_prior_report`).",
        "- Selection used TRAIN only: 5-fold cross-fitted expert scores, nested 5-fold CV over them for the "
        "strategy choice (JSON `nested_cv_acc_all_strategies`); test labels are read after all predictions.",
        "- Fusion ran through the unmodified `ensemble_causal_engine.score_ensemble` and "
        "`causal_moe_engine.MultiModelCausalMoE` on CPU (NumPy).", "",
        "### Not verified", "",
        "- Latency: shared host, one run, see loadavg; the text encoder is excluded and is far slower than the head.",
        "- One seed per expert; run-to-run variance not measured.",
        f"- The nested-CV accuracy of the chosen strategy is the max over ~20 candidates on {n_rows} train rows "
        "per task, so it is optimistic; the test number is the unbiased one.", "",
        "### Not done", "",
        "- Nimble/Jev publish aggregates only: no per-item McNemar against them.",
        "- Features from larger encoders are only used if a matching --source was given (see `protocol.sources`).", "",
        "### Comparability", "",
        f"- Nimble and Jev are zero-shot judges; every expert here is supervised on {n_rows} leakage-gated public "
        "train rows per task. A higher number here is not the same claim as theirs.", ""]
