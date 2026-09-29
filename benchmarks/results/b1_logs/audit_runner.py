"""Reproduce B1 diagnostics without changing benchmark or production behavior.

The exit status is nonzero when real-model acceptance is unavailable. Legacy
planner predicates are measured but are never promoted to solution accuracy.
"""
import hashlib
import inspect
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python'))
OUT = ROOT / 'benchmarks/results'
from gen_zero.scripts.run_12_planners_benchmark import TwelvePlannerStressBenchmark
from gen_zero.train.contrastive_data_pipeline import ContrastiveDataPipeline
from gen_zero.train.qwen_post_trainer import QwenPostTrainer

class ObservedBenchmark(TwelvePlannerStressBenchmark):
    def _record(self, name, latencies, success_rate, category):
        raw = list(latencies)
        super()._record(name, latencies, success_rate, category)
        self.results[name]['raw_invocation_latency_ms'] = raw
        self.results[name]['completed_invocations'] = len(raw)
        self.results[name]['latency_scope'] = 'whole benchmark invocation, NOT search step'


def main():
    payload = {
        'schema_version': 1,
        'status': 'failed_acceptance',
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': sys.version,
        'platform': platform.platform(),
        'utc_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'commands': json.loads((OUT / 'b1_logs/commands.json').read_text()),
        'decision_foundation': {},
        'planners': [],
    }
    corpus = ContrastiveDataPipeline().generate_benchmark_corpus(samples_per_domain=15)
    corpus_path = OUT / 'b1_logs/foundation_corpus.json'
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + '\n')
    trainer = QwenPostTrainer()
    reason = ('REJECTED: benchmark calls simulate_decision_forward (regex action associations '
              'and hash logits), not model inference; no model/tokenizer loaded. '
              'Do not report simulation accuracy as Qwen accuracy.')
    print(reason, flush=True)
    payload['decision_foundation'] = {
        'status': 'invalid_model_benchmark', 'reason': reason,
        'configured_model_name_only': trainer.config.model_name,
        'model_loaded': trainer.model is not None, 'tokenizer_loaded': trainer.tokenizer is not None,
        'generated_samples': len(corpus), 'evaluated_real_model_samples': 0,
        'domains': sorted(set(x['domain'] for x in corpus)),
        'corpus_path': str(corpus_path.relative_to(ROOT)),
        'corpus_sha256': hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        'accuracy': None, 'latency_ms': None, 'entropy_distribution': None,
        'tier_distribution': None, 'model_results': [],
        'missing_metrics_reason': 'No real inference; original benchmark also has no tier or latency instrumentation.',
    }
    bench = ObservedBenchmark()
    print('DIAGNOSTIC ONLY: default GenZero weights_loaded_from_checkpoint =',
          bench.client.weights_loaded_from_checkpoint, flush=True)
    print('WARNING: legacy success predicates do not establish solution correctness; independent cases continue after logged exceptions.', flush=True)
    methods = [name for name, member in TwelvePlannerStressBenchmark.__dict__.items()
               if name.startswith('_test_') and callable(member)]
    for name in methods:
        method = getattr(bench, name)
        row = {'method': name, 'source': 'python/gen_zero/scripts/run_12_planners_benchmark.py:' +
               str(inspect.getsourcelines(method)[1]), 'requested_iterations': 50,
               'score': None, 'solution_rate': None, 'search_steps': None,
               'per_search_step_latency_ms': None,
               'missing_metrics_reason': 'Existing benchmark does not measure these quantities; success_rate is only its legacy predicate.',
               'accepted_as_real_model_evidence': False}
        before = set(bench.results)
        print('BEGIN', name, flush=True)
        start = time.perf_counter()
        try:
            method(50)
            row['status'] = 'diagnostic_completed'
            row['legacy_metrics'] = {k: v for k, v in bench.results.items() if k not in before}
        except Exception as exc:
            row['status'] = 'error'
            row['error_type'] = type(exc).__name__
            row['error'] = str(exc)
            row['traceback'] = traceback.format_exc()
            traceback.print_exc()
        row['wall_seconds'] = time.perf_counter() - start
        payload['planners'].append(row)
        print('END', name, row['status'], flush=True)
        (OUT / 'decision_foundation_eval_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    payload['planner_diagnostic_completion'] = {
        'completed': sum(x['status'] == 'diagnostic_completed' for x in payload['planners']),
        'errored': sum(x['status'] == 'error' for x in payload['planners']),
        'attempted': len(methods),
        'is_solution_pass_rate': False,
    }
    (OUT / 'decision_foundation_eval_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload['planner_diagnostic_completion']), flush=True)
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
