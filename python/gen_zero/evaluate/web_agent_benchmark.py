"""Gen-Zero Layer 4: Decide-and-Fill Action-Selection Smoke Test (Issue #15).

This is a synthetic smoke test, not a real-world benchmark: it runs the
Decide-and-Fill Pipeline plus Semantic Action Compression over 3 hand-written
scenarios and checks that action selection and literal Word-Span argument
filling behave as intended. It does not compare against any LLM baseline
and its numbers are not comparable to real-world web-agent benchmarks.

Outputs:
- results/gen_zero/web_agent_action_selection_smoke_results.json
- results/gen_zero/web_agent_action_selection_smoke_report.md
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gen_zero.harness.action_pipeline import (
    DecideAndFillPipeline,
    ActionSpec,
    WordSpanExtractor,
    PipelineDecision
)
from gen_zero.harness.semantic_synthesizer import (
    SemanticActionSynthesizer,
    SynthesizedAction,
    DOMClosureHandle
)


@dataclass
class WebScenario:
    """Represents a standardized multi-step web agent task."""
    scenario_id: str
    task_objective: str
    initial_elements: List[Dict[str, Any]]
    expected_actions: List[str]
    expected_word_span_text: Optional[str] = None
    viewport_overlays: List[Dict[str, Any]] = field(default_factory=list)


def create_standard_web_scenarios() -> List[WebScenario]:
    """Creates representative benchmark scenarios across e-commerce, forms, and catalogs."""
    return [
        WebScenario(
            scenario_id="scenario_01_ecommerce_search_filter",
            task_objective='Search for "wireless noise-canceling headphones" and filter by electronics category',
            initial_elements=[
                {
                    "tag": "input",
                    "role": "searchbox",
                    "aria_label": "Search products",
                    "id": "global_search",
                    "bounding_box": {"x": 200.0, "y": 20.0, "w": 400.0, "h": 36.0}
                },
                {
                    "tag": "button",
                    "role": "button",
                    "text": "Filter Category",
                    "aria_label": "Category filter dropdown",
                    "id": "filter_cat",
                    "bounding_box": {"x": 50.0, "y": 120.0, "w": 120.0, "h": 30.0}
                },
                {
                    "tag": "button",
                    "role": "button",
                    "text": "Add to Cart",
                    "aria_label": "Add first item to cart",
                    "id": "btn_add_cart_1",
                    "bounding_box": {"x": 300.0, "y": 300.0, "w": 140.0, "h": 40.0}
                }
            ],
            expected_actions=["search_products", "filter_catalog", "add_to_cart"],
            expected_word_span_text="wireless noise-canceling headphones"
        ),
        WebScenario(
            scenario_id="scenario_02_employee_onboarding_form",
            task_objective='Enter employee title "Lead Platform Systems Engineer" and submit application form',
            initial_elements=[
                {
                    "tag": "input",
                    "role": "textbox",
                    "aria_label": "Job Title Input",
                    "name": "employee_title",
                    "bounding_box": {"x": 100.0, "y": 80.0, "w": 300.0, "h": 32.0}
                },
                {
                    "tag": "button",
                    "role": "button",
                    "text": "Submit Application",
                    "aria_label": "Submit form",
                    "id": "btn_submit_app",
                    "bounding_box": {"x": 100.0, "y": 250.0, "w": 160.0, "h": 38.0}
                }
            ],
            expected_actions=["submit_form"],
            expected_word_span_text="Lead Platform Systems Engineer"
        ),
        WebScenario(
            scenario_id="scenario_03_secure_auth_login",
            task_objective='Log in with email and password and proceed to checkout',
            initial_elements=[
                {
                    "tag": "input",
                    "role": "textbox",
                    "aria_label": "User Email",
                    "id": "input_email",
                    "bounding_box": {"x": 150.0, "y": 100.0, "w": 250.0, "h": 30.0}
                },
                {
                    "tag": "input",
                    "role": "textbox",
                    "aria_label": "User Password",
                    "id": "input_pwd",
                    "bounding_box": {"x": 150.0, "y": 140.0, "w": 250.0, "h": 30.0}
                },
                {
                    "tag": "button",
                    "role": "button",
                    "text": "Checkout Now",
                    "aria_label": "Proceed to checkout",
                    "id": "btn_checkout",
                    "bounding_box": {"x": 150.0, "y": 220.0, "w": 180.0, "h": 36.0}
                }
            ],
            expected_actions=["submit_login", "proceed_to_checkout"],
            expected_word_span_text=None
        )
    ]


@dataclass
class BenchmarkParadigmResult:
    """Measured metrics for the Decide-and-Fill action-selection smoke test."""
    paradigm_name: str
    task_success_rate_pct: float
    average_steps_per_task: Optional[float]
    average_step_latency_ms: float
    description: str


class WebAgentBenchmarkSuite:
    """Runs the Decide-and-Fill action-selection smoke test."""

    @classmethod
    def evaluate(cls, repeat_trials: int = 10) -> Dict[str, Any]:
        """Runs the Decide-and-Fill pipeline over the 3 synthetic scenarios."""
        if not isinstance(repeat_trials, int) or isinstance(repeat_trials, bool) or repeat_trials <= 0:
            raise ValueError("repeat_trials must be a positive integer")
        scenarios = create_standard_web_scenarios()
        synthesizer = SemanticActionSynthesizer()

        pipeline_successes = 0
        action_mismatches = 0
        span_mismatches = 0
        total_evals = 0
        actual_steps = []
        actual_latencies = []

        for trial in range(repeat_trials):
            for sc in scenarios:
                total_evals += 1
                # 1. Synthesize semantic tools
                actions = synthesizer.synthesize_actions(sc.initial_elements, sc.viewport_overlays)
                action_specs = [
                    ActionSpec(
                        name=a.action_name,
                        description=a.description,
                        requires_arguments=("query" in a.parameters_schema or "username" in a.parameters_schema),
                        parameter_schema=a.parameters_schema,
                        category=a.category
                    )
                    for a in actions
                ]
                pipeline = DecideAndFillPipeline(available_actions=action_specs)

                # 2. Decide and fill
                t_start = time.perf_counter()
                dec = pipeline.decide_and_fill(
                    utterance=sc.task_objective,
                    state="Page Loaded with 3 main sections."
                )
                lat_ms = (time.perf_counter() - t_start) * 1000.0
                actual_latencies.append(lat_ms)

                # Check action validity, then (if applicable) exact Word-Span match
                if dec.action in sc.expected_actions:
                    if sc.expected_word_span_text:
                        extracted = dec.arguments.get("text", "").strip()
                        if extracted == sc.expected_word_span_text:
                            pipeline_successes += 1
                        else:
                            span_mismatches += 1
                    else:
                        pipeline_successes += 1
                else:
                    action_mismatches += 1

                actual_steps.append(len(actions))

        exp_acc = round((pipeline_successes / max(1, total_evals)) * 100.0, 2)
        exp_avg_steps = round(sum(actual_steps) / max(1, len(actual_steps)), 1)
        exp_latency_ms = round(sum(actual_latencies) / max(1, len(actual_latencies)), 2)

        res_exp = BenchmarkParadigmResult(
            paradigm_name="Decide-and-Fill + Semantic Action Compression",
            task_success_rate_pct=exp_acc,
            average_steps_per_task=None,  # no browser actions are executed
            average_step_latency_ms=exp_latency_ms,
            description="Two-stage System 1 prefill action selection + literal Word-Span argument filling."
        )

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_synthetic": True,
            "is_synthetic_smoke_test": True,
            "average_synthesized_actions_per_task": exp_avg_steps,
            "evaluated_scenarios_count": len(scenarios),
            "total_trials": repeat_trials * len(scenarios),
            "decide_and_fill": asdict(res_exp),
            "failure_breakdown": {
                "total_evaluations": total_evals,
                "successes": pipeline_successes,
                "action_mismatches": action_mismatches,
                "span_mismatches": span_mismatches,
            },
        }

        # Write output files
        out_dir = Path("results/gen_zero")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "web_agent_action_selection_smoke_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        md_path = out_dir / "web_agent_action_selection_smoke_report.md"
        md_content = f"""# Gen-Zero Decide-and-Fill Action Selection Smoke Test Report

Evaluation Time: {results['timestamp']} · Trials: {results['total_trials']} experiments

**Note**: This report is a synthetic smoke test covering 3 handwritten scenarios,
designed to verify that the Decide-and-Fill action selection and literal Word-Span
parameter extraction pipeline operate as expected.
It is not a real-world web agent benchmark and must not be compared to LLM baselines.

## Results

| Metric | Value |
| :--- | :---: |
| Task Success Rate (action match + span exact match) | {res_exp.task_success_rate_pct}% |
| Candidate Actions per Scenario (not execution steps) | {exp_avg_steps} |
| Steps per Task | Not measured |
| Single-Step Decision Latency (mean) | {res_exp.average_step_latency_ms} ms |
| Total Evaluations | {results['failure_breakdown']['total_evaluations']} |
| Action Mismatches | {results['failure_breakdown']['action_mismatches']} |
| Span Mismatches | {results['failure_breakdown']['span_mismatches']} |
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return results


def main():
    print("=================================================================")
    print("   GEN-ZERO DECIDE-AND-FILL ACTION-SELECTION SMOKE TEST (ISSUE #15)    ")
    print("=================================================================")
    res = WebAgentBenchmarkSuite.evaluate(repeat_trials=10)
    r = res["decide_and_fill"]
    fb = res["failure_breakdown"]
    print(f"\nSuccess rate: {r['task_success_rate_pct']}% ({fb['successes']}/{fb['total_evaluations']})")
    print(f"Action mismatches: {fb['action_mismatches']} | Span mismatches: {fb['span_mismatches']}")
    print(f"Average step latency: {r['average_step_latency_ms']} ms")
    print("\nSaved artifacts to results/gen_zero/ (synthetic smoke test over 3 hand-written scenarios)")


if __name__ == "__main__":
    main()
