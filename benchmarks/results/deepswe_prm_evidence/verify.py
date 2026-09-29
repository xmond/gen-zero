"""Independent metric checks against saved fresh-inference evidence."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path[:0] = ['/tmp/clmrepro/repo/evaluation', '/tmp/clmrepro/repo/src', '/tmp/clmrepro/repo/preprocessing']
from bon_eval import best_of_n
base = Path(__file__).resolve().parent
r = json.loads((base.parent / 'deepswe_prm_eval_results.json').read_text())
t = pd.read_csv(base / 'trajectory_scores.csv')
m = pd.read_parquet(base / 'step_scores.parquet')
assert len(m) == 44409 and len(t) == 449 and t.task_id.nunique() == 113
for label, report in r['results'].items():
    tasks = set(report['per_task'])
    for frame, col, name in [(t, 'score', 'trajectory_correlation'), (m, 'clm_score', 'step_correlation')]:
        sub = frame[frame.task_id.isin(tasks)]
        assert abs(roc_auc_score(sub.reward, sub[col]) - report[name]['roc_auc']) < 1e-12
cases = 0
for scores in itertools.product([0., 1.], repeat=4):
    for rewards in itertools.product([0, 1], repeat=4):
        candidates = list(zip(scores, rewards))
        for n in range(1, 5):
            expected = []
            for subset in itertools.combinations(candidates, n):
                top = max(s for s, _ in subset)
                winners = [y for s, y in subset if s == top]
                expected.append(np.mean(winners))
            assert abs(best_of_n(candidates, n)['selected'] - np.mean(expected)) < 1e-12
            cases += 1
print('PASS: row/trajectory/task counts; six sklearn AUC comparisons;', cases, 'exhaustive subset/tie cases')
assert r['results']['all_113']['bon']['8']['strict_n_all_tasks_rate'] is None
assert r['results']['all_113']['bon']['4']['strict_n_all_tasks_rate'] is None
print('PASS: unavailable strict N=4/N=8 not presented as measured rates')
