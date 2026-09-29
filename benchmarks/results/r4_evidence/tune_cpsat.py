"""Reproducible CP-SAT parameter experiment; every trial runs OR-Tools."""
import json
import random
import time
from pathlib import Path
from ortools.sat.python import cp_model

rows = []
for workers, presolve, compressed in [(0, True, False), (1, True, False), (1, True, True), (1, False, True), (2, True, True), (2, False, True)]:
    rng = random.Random(42)
    trials = []
    for trial in range(1000):
        utilities = [rng.random() for _ in range(11)]
        allowed = [i for i in range(11) if rng.random() > .3]
        if not allowed:
            allowed = [0]
        start = time.perf_counter()
        model = cp_model.CpModel()
        indices = allowed if compressed else range(11)
        variables = {i: model.NewBoolVar(str(i)) for i in indices}
        model.AddExactlyOne(list(variables.values()))
        if not compressed:
            for i in variables:
                if i not in allowed:
                    model.Add(variables[i] == 0)
        ranks = {u: i for i, u in enumerate(sorted(utilities))}
        model.Maximize(sum(ranks[utilities[i]] * v for i, v in variables.items()))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = workers
        solver.parameters.cp_model_presolve = presolve
        solver.parameters.linearization_level = 0
        solver.parameters.max_time_in_seconds = max(0.000001, .002 - (time.perf_counter() - start))
        built = time.perf_counter()
        status = solver.Solve(model)
        end = time.perf_counter()
        optimal_correct = status == cp_model.OPTIMAL and any(solver.Value(v) and utilities[i] == max(utilities[a] for a in allowed) for i, v in variables.items())
        trials.append({'wall_ms': (end-start)*1000, 'build_ms': (built-start)*1000,
                       'solve_ms': (end-built)*1000, 'status': solver.StatusName(status), 'correct': optimal_correct})
    wall = sorted(t['wall_ms'] for t in trials)
    row = {'workers': workers, 'presolve': presolve, 'compressed': compressed,
           'deadline_misses': sum(t['wall_ms'] > 2 for t in trials),
           'not_optimal': sum(not t['correct'] for t in trials),
           'p50_ms': wall[500], 'p99_ms': wall[990], 'max_ms': max(wall), 'trials': trials}
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k != 'trials'}), flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows, indent=2)+'\n')
