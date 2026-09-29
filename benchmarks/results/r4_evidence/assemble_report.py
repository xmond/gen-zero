"""Assemble evidence without converting failed benchmark exits into success."""
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy
import ortools

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent

def read(name):
    return json.loads((evidence / name).read_text())

def audit(report):
    return {k: v for k, v in report['solver_audit'].items() if k != 'trials'}

def source(path, marker):
    p = root / path
    lines = p.read_text().splitlines()
    line = next(i for i, text in enumerate(lines, 1) if marker in text)
    return {'location': f'{path}:{line}', 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

before = read('constraints_original_audited_report.json')
after = read('constraints_after_report.json')
planners = read('planners_after_report.json')
league = read('league_after_report.json')
trials = after['solver_audit']['trials']
worst = max(trials, key=lambda t: t['wall_ms'])
records = {name: read(name + '.json') for name in [
    'planners_before', 'constraints_before', 'league_before',
    'constraints_original_audited', 'constraints_after', 'planners_after',
    'league_after', 'tests_after', 'tests_final', 'tests_verified']}

report = {
    'overall_passed': False,
    'status': 'incomplete_targets_fail_closed',
    'workspace': str(root),
    'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'environment': {'python': sys.version, 'executable': sys.executable,
                    'ortools': ortools.__version__, 'numpy': numpy.__version__,
                    'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
                    'affinity': sorted(os.sched_getaffinity(0))},
    'implemented': [
        'CEM benchmark uses the required state and discrete candidate arguments; reward-sensitive regression passes.',
        'Client bidirectional API invokes the actual bidirectional engine method; returned path regression passes.',
        'CP-SAT uses one worker, compact Boolean protobuf model, exactly-one constraint, presolve enabled, probing/symmetry/linearization disabled.',
        'Ordinal utility coefficients preserve exact single-choice ordering without four-decimal truncation.',
        'No argmax release on timeout, unknown solver status, missing dependency or solver exception; return is_safe=false with explicit error/status.',
        'Latent validation is included in timing; unknown compound conditions cannot borrow a random projection as a proof.',
        'QR allocation uses the actual proposition count instead of a dense latent-dimension square.',
        'Benchmark preserves all 1000 wall timings, verdicts, scenarios, safety oracle, rejection counts and process exit codes.',
        'League uses observed Elo window stability, preserves training state during historical evaluation, and no longer asserts 85% wins or prints monotonic certification.',
        '116 focused and expanded regression tests passed with raw exit code 0.'
    ],
    'unverified': [
        'A trained dual-head Reflex checkpoint was not supplied; random-init weights are explicitly not accepted.',
        'No live GPU arbiter acceptance or production deployment was performed.',
        'Finite synthetic runs do not certify universal safety, arbitrary natural-language reasoning, Nash equilibrium or hard real-time scheduling.',
        'The user-reported historical 103/1000 timeouts has no original log in this task and is not presented as a reproduced measurement.'
    ],
    'incomplete': [
        f"12/12 target: final strict run passed {planners['passed_count']}/12; process exit code {records['planners_after']['exit_code']}.",
        f"2.0ms / zero-fallback target: {audit(after)['fallback_count']}/1000 fallbacks and {audit(after)['wall_deadline_misses']}/1000 wall deadline misses; process exit code {records['constraints_after']['exit_code']}.",
        'Shared-host scheduler delay remains observable; solver parameters alone do not establish a hard completion-time bound.'
    ],
    'commands_and_raw_exit_codes': records,
    'constraint_comparison': {
        'same_scenario_oracle_and_environment': True,
        'comparison_note': 'Original HEAD compiler and final compiler use the updated audited synthetic scenarios with real OR-Tools; shared host load is not controlled.',
        'user_reported_historical_fallbacks': {'count': 103, 'trials': 1000, 'independently_verified': False},
        'original_audited': {'source': before['source'], 'audit': audit(before),
                             'time': before['time'], 'safety': before['safety_and_blocking']},
        'final': {'audit': audit(after), 'time': after['time'], 'safety': after['safety_and_blocking'],
                  'adaptation': after['zero_training_adaptation']},
        'worst_final_trial': {'index': worst['index'], 'wall_ms': worst['wall_ms'],
                              'thread_cpu_ms': worst['thread_cpu_ms'], 'verdict': worst['verdict']},
        'median_wall_ms': statistics.median(t['wall_ms'] for t in trials),
        'median_thread_cpu_ms': statistics.median(t['thread_cpu_ms'] for t in trials),
        'full_before_trials': 'benchmarks/results/r4_evidence/constraints_original_audited_report.json',
        'full_after_trials': 'benchmarks/results/r4_evidence/constraints_after_report.json'
    },
    'planners': {**{k:v for k,v in planners.items() if k != 'planners'},
                 'metrics': {k:{field:value for field,value in v.items() if field != 'latencies_ms'}
                             for k,v in planners['planners'].items()}},
    'league': {k:v for k,v in league.items() if k != 'history'},
    'parameter_experiments': {
        name: [{k:v for k,v in row.items() if k != 'trials'} for row in read(name+'.json')]
        for name in ['tune_cpsat','tune_compact']
    },
    'source_evidence': [
        source('python/gen_zero/client.py', 'return self.bidirectional_planner.plan_bidirectional('),
        source('python/gen_zero/scripts/run_12_planners_benchmark.py', 'state={"price": 100.0}'),
        source('python/gen_zero/gate/constraint_compiler.py', 'def solve_safest_action('),
        source('python/gen_zero/gate/constraint_compiler.py', 'solver.parameters.num_search_workers'),
        source('python/gen_zero/scripts/benchmark_issue_76_constraint_compiler.py', 'report_dict["solver_audit"]'),
        source('python/gen_zero/run_league_benchmark.py', 'def elo_convergence('),
        source('python/gen_zero/multiagent/league_arena.py', 'def evaluate_historical_robustness('),
        source('python/gen_zero/tests/test_r4_planner_constraint_regressions.py', 'def test_deadline_rejects')
    ],
    'compatibility_changes': [
        'Compiler failures no longer release an argmax fallback as is_safe=true. Callers must respect is_safe and fallback_used.',
        'Ambiguous normalized action IDs and nonfinite utilities now raise ValueError.',
        'Unsupported conditions fail closed even when a latent vector is supplied.',
        'Historical robustness removes zero_elo_degradation; callers receive measured elo_delta and win_rates instead.',
        'Full repository test suite and other OR-Tools versions were not tested.'
    ]
}
report['evidence_sha256'] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(evidence.iterdir()) if p.suffix in ('.log','.json','.py','.md')
}
out = root / 'benchmarks/results/planners_and_constraints_fixed_results.json'
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
assert report['commands_and_raw_exit_codes']['tests_verified']['exit_code'] == 0
assert report['constraint_comparison']['final']['safety']['after_cpsat_violations'] == 0
assert not report['overall_passed']
print(out)
print('Persisted incomplete target status and original failing benchmark exit codes.')
