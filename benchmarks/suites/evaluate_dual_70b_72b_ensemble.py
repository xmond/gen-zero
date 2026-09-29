#!/usr/bin/env python3
"""Cross-fitted probability fusion of aligned Qwen-72B and Llama-70B features.

This is a deliberately bounded ridge-probe ensemble, separate from the much larger
Spec 21 per-source head search. All fusion choices use training OOF predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.random_projection import GaussianRandomProjection

import grand_challenge_data as gd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDIR = Path(os.environ.get('MASTER_QWEN_DIR', '/ebs/data/extracted_features/qwen72b/features'))
DEFAULT_LDIR = Path(os.environ.get('MASTER_LLAMA_DIR', '/ebs/data/extracted_features/llama70b'))
# Backward-compatible module-level defaults; callers that import this module and use
# load()/run_task() without explicit dirs (e.g. evaluate_dual_70b_72b_advanced_ensemble.py)
# still get these paths.
QDIR = DEFAULT_QDIR
LDIR = DEFAULT_LDIR
DEFAULT_OUT = ROOT / 'benchmarks/results/spec21_dual_70b_72b_ensemble_report'
OUT = DEFAULT_OUT
SEED = 20260925
FOLDS = 5
WIDTH = 256
ALPHA = 100.0
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(task: str, qdir: Path = QDIR, ldir: Path = LDIR):
    paths = (Path(qdir) / f'{task}.npz', Path(ldir) / f'{task}.npz')
    data = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            data.append({k: z[k] for k in ('train_full', 'test_full', 'train_label', 'train_ids', 'test_ids', 'cands')})
    a, b = data
    for key in ('train_ids', 'test_ids', 'train_label'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'{task}: source mismatch: {key}')
    if len(set(a['train_ids'])) != len(a['train_ids']) or set(a['train_ids']) & set(a['test_ids']):
        raise ValueError(f'{task}: duplicate or overlapping row ids')
    if a['cands'].shape != b['cands'].shape:
        raise ValueError(f'{task}: candidate shape mismatch')
    for d in data:
        if not np.isfinite(d['train_full']).all() or not np.isfinite(d['test_full']).all():
            raise ValueError(f'{task}: non-finite feature')
    return paths, data


def project(d: dict, seed: int):
    n = len(d['train_full'])
    X = np.concatenate((d['train_full'], d['test_full']), axis=0)
    model = GaussianRandomProjection(n_components=WIDTH, random_state=seed)
    Z = model.fit_transform(X).astype(np.float64)
    return Z[:n], Z[n:]


def fit_scores(X, y, Q, K):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.maximum(sd, 1e-6)
    x, q = (X - mu) / sd, (Q - mu) / sd
    x = np.column_stack((x, np.ones(len(x))))
    q = np.column_stack((q, np.ones(len(q))))
    Y = np.eye(K)[y]
    reg = np.eye(x.shape[1]) * ALPHA
    reg[-1, -1] = 0.0
    coef = solve(x.T @ x + reg, x.T @ Y, assume_a='pos', check_finite=False)
    return q @ coef


def probs(score, scale):
    return softmax(score / max(scale, 1e-8), axis=1)


def metrics(y, pred):
    return {'accuracy': 100 * accuracy_score(y, pred),
            'balanced_accuracy': 100 * balanced_accuracy_score(y, pred),
            'macro_f1': 100 * f1_score(y, pred, average='macro', zero_division=0),
            'correct': int(np.sum(y == pred)), 'n': int(len(y))}


def run_task(task, qdir: Path = QDIR, ldir: Path = LDIR):
    paths, data = load(task, qdir, ldir)
    y = data[0]['train_label'].astype(int)
    K = len(data[0]['cands'])
    if y.min() < 0 or y.max() >= K:
        raise ValueError(f'{task}: training label outside candidate range')
    Z = [project(d, SEED + 100 * gd.TASKS.index(task) + i) for i, d in enumerate(data)]
    folds = np.random.default_rng([SEED, gd.TASKS.index(task)]).permutation(len(y)) % FOLDS
    oof = [np.empty((len(y), K)) for _ in data]
    for k in range(FOLDS):
        tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        for i in range(2):
            oof[i][ho] = fit_scores(Z[i][0][tr], y[tr], Z[i][0][ho], K)
    # Temperature is a label-free OOF logit scale; the only supervised fusion
    # selection is the weight, picked by OOF accuracy with a fixed tie order.
    scales = [float(np.std(s)) for s in oof]
    op = [probs(s, scale) for s, scale in zip(oof, scales)]
    cv = {str(w): metrics(y, (w * op[0] + (1 - w) * op[1]).argmax(1))['accuracy'] for w in WEIGHTS}
    chosen = max((0.5, 0.25, 0.75, 0.0, 1.0), key=lambda w: cv[str(w)])
    test_scores = [fit_scores(ztr, y, zte, K) for ztr, zte in Z]
    tp = [probs(s, scale) for s, scale in zip(test_scores, scales)]
    # Test labels enter only after every fit, score and weight selection.
    records = gd.load_test(task)
    ids = [r['id'] for r in records]
    if ids != data[0]['test_ids'].tolist():
        raise ValueError(f'{task}: test ids differ from features')
    cands = records[0]['candidates']
    if len(cands) != K or any(r['candidates'] != cands for r in records):
        raise ValueError(f'{task}: candidate lists differ')
    gold = np.array([cands.index(r['ground_truth']) for r in records])
    results = {'qwen_probe': metrics(gold, tp[0].argmax(1)),
               'llama_probe': metrics(gold, tp[1].argmax(1)),
               'equal_fusion': metrics(gold, (0.5 * tp[0] + 0.5 * tp[1]).argmax(1)),
               'oof_weight_fusion': metrics(gold, (chosen * tp[0] + (1 - chosen) * tp[1]).argmax(1))}
    return {'n_train': len(y), 'n_test': len(gold), 'K': K, 'weight_qwen': chosen,
            'oof_accuracy_by_qwen_weight': cv, 'oof_logit_scale': scales,
            'feature_files': {str(p): digest(p) for p in paths},
            'test_file_sha256': digest(gd.TEST_DIR / f'{task}.jsonl'), 'metrics': results}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-dir', type=Path, default=DEFAULT_QDIR,
                         help='Directory with per-task Qwen-72B feature .npz files '
                              '(default: $MASTER_QWEN_DIR or %(default)s)')
    parser.add_argument('--llama-dir', type=Path, default=DEFAULT_LDIR,
                         help='Directory with per-task Llama-70B feature .npz files '
                              '(default: $MASTER_LLAMA_DIR or %(default)s)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                         help='Output path stem for the .json/.md report (default: %(default)s)')
    parser.add_argument('--tasks', nargs='+', default=None,
                         help='Subset of tasks to evaluate (default: all 13 tasks). '
                              f'Choices: {", ".join(gd.TASKS)}')
    return parser.parse_args(argv)


def resolve_tasks(requested):
    if requested is None:
        return list(gd.TASKS)
    unknown = [t for t in requested if t not in gd.TASKS]
    if unknown:
        raise ValueError(f'Unknown task(s): {", ".join(unknown)}. Valid tasks: {", ".join(gd.TASKS)}')
    return list(requested)


def check_inputs(qdir: Path, ldir: Path, tasks):
    """Fail fast with an explicit, non-zero-exit error instead of a confusing stack trace."""
    missing = []
    if not qdir.is_dir():
        missing.append(f'Qwen feature directory not found: {qdir}')
    if not ldir.is_dir():
        missing.append(f'Llama feature directory not found: {ldir}')
    if missing:
        raise FileNotFoundError('; '.join(missing))
    for task in tasks:
        for label, d in (('Qwen', qdir), ('Llama', ldir)):
            p = d / f'{task}.npz'
            if not p.is_file():
                missing.append(f'{label} feature file missing for task {task!r}: {p}')
    if missing:
        raise FileNotFoundError('; '.join(missing))


def main(argv=None):
    args = parse_args(argv)
    try:
        tasks = resolve_tasks(args.tasks)
        check_inputs(args.qwen_dir, args.llama_dir, tasks)
    except (ValueError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    is_subset = tasks != list(gd.TASKS)
    rows = {}
    for task in tasks:
        rows[task] = run_task(task, args.qwen_dir, args.llama_dir)
        print(task, json.dumps(rows[task]['metrics'], sort_keys=True), flush=True)
    baselines = {n: json.loads((ROOT / f'benchmarks/results/spec21_{n}_scorecard.json').read_text())
                 for n in ('qwen72b', 'llama70b')}
    methods = ('qwen_probe', 'llama_probe', 'equal_fusion', 'oof_weight_fusion')
    aggregate = {m: {metric: float(np.mean([rows[t]['metrics'][m][metric] for t in rows]))
                     for metric in ('accuracy', 'balanced_accuracy', 'macro_f1')} for m in methods}
    published_tasks = {t: {n: baselines[n]['tasks'][t]['accuracy'] for n in baselines} for t in rows}
    for task, ref in published_tasks.items():
        rows[task]['published_single_model_reference_accuracy'] = ref
        rows[task]['equal_fusion_minus_best_published_pp'] = (
            rows[task]['metrics']['equal_fusion']['accuracy'] - max(ref.values()))
    scope = {'tasks': tasks, 'n_tasks': len(tasks), 'is_subset': is_subset}
    title = ('13-task' if not is_subset else f'{len(tasks)}-task subset') + \
            ' Qwen-72B + Llama-70B ridge-probe probability ensemble'
    report = {'title': title,
              'scope': scope,
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'command': ' '.join(sys.argv),
              'feature_dirs': {'qwen': str(args.qwen_dir), 'llama': str(args.llama_dir)},
              'protocol': {'folds': FOLDS, 'seed': SEED, 'projection': 'Gaussian random projection, 256 dimensions per source, label-free',
                           'head': 'standardized ridge one-hot classifier', 'alpha': ALPHA,
                           'weight_grid_qwen': WEIGHTS, 'weight_selection': 'training OOF accuracy; tie order 0.5,0.25,0.75,0,1',
                           'probability': 'softmax of ridge scores divided by OOF score standard deviation',
                           'baseline_caveat': 'Spec21 references use different selected expert families and are not matched head comparisons.'},
              'published_single_model_reference': {n: baselines[n]['aggregate']['macro_all'] for n in baselines},
              'aggregate_macro': aggregate, 'tasks': rows}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    refs = report['published_single_model_reference']
    scope_note = (f"Scope: full 13-task suite." if not is_subset else
                  f"Scope: {len(tasks)}-task subset ({', '.join(tasks)}); aggregates below are over this subset only, not the full 13-task suite.")
    lines = [f'# Dual 70B/72B ensemble: {title.split(" Qwen")[0]}', '',
             scope_note, '',
             'Cross-fitted ridge probes on aligned frozen features; the Qwen fusion weight is selected on training OOF accuracy. Test labels are read after selection.', '',
             f"The published Qwen {refs['qwen72b']:.2f}% / Llama {refs['llama70b']:.2f}% scorecards use different selected expert families. Their numbers are external references, not matched probe controls.", '',
             'The equal fusion does not beat either published macro reference. Some tasks have high accuracy but much lower balanced accuracy and macro F1, indicating class imbalance or class collapse.', '',
             '## Macro metrics (%)', '', '| Method | Accuracy | Balanced accuracy | Macro F1 |', '|---|---:|---:|---:|']
    for m in methods:
        v = aggregate[m]
        lines.append(f"| {m} | {v['accuracy']:.2f} | {v['balanced_accuracy']:.2f} | {v['macro_f1']:.2f} |")
    lines += ['', '## Per-task accuracy (%)', '', '| Task | Qwen probe | Llama probe | Equal fusion | OOF-weight fusion | Matched winner | Qwen weight | Published Qwen | Published Llama | Equal minus best published |', '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for t, r in rows.items():
        v = {m: r['metrics'][m]['accuracy'] for m in methods}
        top = max(v.values())
        winner = ', '.join(m for m in methods if abs(v[m] - top) < 1e-10)
        ref = r['published_single_model_reference_accuracy']
        lines.append(f"| {t} | {v['qwen_probe']:.2f} | {v['llama_probe']:.2f} | {v['equal_fusion']:.2f} | {v['oof_weight_fusion']:.2f} | {winner} | {r['weight_qwen']:.2f} | {ref['qwen72b']:.2f} | {ref['llama70b']:.2f} | {r['equal_fusion_minus_best_published_pp']:+.2f} |")
    lines += ['', 'All full metrics, OOF weights, feature hashes, and test-file hashes are in the JSON report.', '']
    args.out.with_suffix('.md').write_text('\n'.join(lines))
    print('WROTE', args.out.with_suffix('.json'), args.out.with_suffix('.md'), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
