#!/usr/bin/env python3
"""Independent report check against the local released CLM evaluator.

This does not train, select models, or change any score. It cross-checks the
saved candidate-level results using the release's combinatorial evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True)
    p.add_argument('--reference-repo', required=True)
    args = p.parse_args()
    for part in ('src', 'preprocessing', 'evaluation'):
        sys.path.insert(0, str(Path(args.reference_repo) / part))
    from bon_eval import best_of_n

    report = json.loads(Path(args.report).read_text())
    rows_by_method = {}
    for method in ('baseline', 'enhanced'):
        result = report[method]
        all_candidates = [c for t in result['per_task'] for c in t['candidates']]
        rows_by_method[method] = {c['trajectory_id']: c['reward'] for c in all_candidates}
        assert len(all_candidates) == len(rows_by_method[method]) == result['trajectories']
        assert len(result['per_task']) == result['tasks'] == 38
        for n in (1, 2, 4):
            expectation = sum(best_of_n([(c['score'], c['reward']) for c in task['candidates']],
                                        min(n, len(task['candidates'])))['selected']
                              for task in result['per_task'])
            assert abs(expectation - result['bon'][str(n)]['solved_task_equivalents']) < 1e-10
        y, s = [c['reward'] for c in all_candidates], [c['score'] for c in all_candidates]
        assert abs(roc_auc_score(y, s) - result['auc']) < 1e-12
        assert abs(spearmanr(y, s).statistic - result['spearman']) < 1e-12
        expected_wrong = []
        for task in result['per_task']:
            highest = max(c['score'] for c in task['candidates'])
            picked = [c for c in task['candidates'] if c['score'] == highest]
            assert [c['trajectory_id'] for c in picked] == task['selected']
            assert np.mean([c['reward'] for c in picked]) == task['selected_reward']
            if task['selected_reward'] < 1:
                expected_wrong.append(task)
        assert expected_wrong == result['wrong_tasks']
    assert rows_by_method['baseline'] == rows_by_method['enhanced']
    provenance = report['provenance']
    for path, key in [(provenance['checkpoint'], 'checkpoint_sha256'),
                      ('benchmarks/eval_deepswe_enhanced_prm.py', 'source_sha256')]:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == provenance[key]
    assert report['target_achieved'] == (report['enhanced']['bon']['4']['solved_task_equivalents'] >= 32)
    print('PASS: candidate identities, released BoN combinatorics, ties, AUC, Spearman, failures, source/checkpoint hashes')
    print('reference_evaluator_sha256=' + hashlib.sha256(
        (Path(args.reference_repo) / 'evaluation/bon_eval.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
