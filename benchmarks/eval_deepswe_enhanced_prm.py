#!/usr/bin/env python3
"""Train a task-disjoint temporal/embedding kernel ranker, then evaluate once.

No outcome, task identity, fold, or supplied prm_score enters the features.
Geometric temporal associations are not causal identification. Step rows are
pooled into trajectories; they are not independent labelled examples.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def splits(args):
    train = json.loads(Path(args.train_tasks).read_text())
    held = json.loads(Path(args.heldout_tasks).read_text())
    require(len(set(train)) == len(train) == 75, 'expected 75 unique train tasks')
    require(len(set(held)) == len(held) == 38, 'expected 38 unique heldout tasks')
    require(not set(train) & set(held), 'train/heldout overlap')
    return set(train), set(held)


def unit(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    require(np.isfinite(x).all() and (norms > 0).all(), 'nonfinite or zero embedding')
    return x / norms


def features(s, a):
    """All statistics use observed steps, with backward-only differences."""
    su, au = unit(s), unit(a)
    sn, an = np.linalg.norm(s, axis=1), np.linalg.norm(a, axis=1)
    # First-step differences are defined as zero (no previous observation).
    ds = np.vstack([np.zeros_like(s[:1]), np.diff(s, axis=0)])
    da = np.vstack([np.zeros_like(a[:1]), np.diff(a, axis=0)])
    dsn, dan = np.linalg.norm(ds, axis=1), np.linalg.norm(da, axis=1)
    energy = np.sum((su - au) ** 2, axis=1)
    base = np.column_stack([
        np.log(sn), np.log(an), an / sn, np.sum(su * au, axis=1), energy,
        dsn / sn, dan / an,
        np.r_[0., np.sum(su[1:] * su[:-1], axis=1)],
        np.r_[0., np.sum(au[1:] * au[:-1], axis=1)],
        np.r_[0., np.diff(energy)],
        np.sum(ds * da, axis=1) / np.maximum(dsn * dan, 1e-12),
        np.r_[0., np.sum(ds[1:] * au[:-1], axis=1) / np.maximum(dsn[1:], 1e-12)],
    ])
    g = [np.array([np.log1p(len(s))])]
    means = []
    for w in (1, 4, 12, 32, len(s)):
        z = base[-w:]
        means.append(z.mean(0))
        g.extend([z.mean(0), z.std(0), z.min(0), z.max(0), z[-1] - z[0]])
    g.extend([means[i] - means[i + 1] for i in range(4)])
    semantic = np.concatenate([unit(z.mean(0, keepdims=True))[0]
                               for z in (su, au, su[-12:], au[-12:])]) / 2
    return np.concatenate(g).astype('float64'), semantic.astype('float64')


def prepare(args):
    train, held = splits(args)
    # Select training records before materializing any embedding feature or label.
    table = pq.read_table(args.data)
    meta = table.select(['trajectory_id', 'task_id', 'step_idx', 'reward']).to_pandas()
    require(set(meta.task_id) == train | held, 'unexpected/missing tasks')
    require(not meta.duplicated(['trajectory_id', 'step_idx']).any(), 'duplicate step')
    mask = meta.task_id.isin(train).to_numpy()
    if args.partition == 'heldout':
        mask = ~mask
    indices = np.flatnonzero(mask)
    selected = table.take(indices)
    m = meta.iloc[indices].reset_index(drop=True)
    s = np.asarray(selected['state_embedding'].to_pylist(), dtype=np.float32)
    a = np.asarray(selected['action_embedding'].to_pylist(), dtype=np.float32)
    require(s.shape == a.shape and s.ndim == 2, 'embedding shape mismatch')
    rows, geom, sem = [], [], []
    for tid, group in m.groupby('trajectory_id', sort=True):
        group = group.sort_values('step_idx')
        require(group.task_id.nunique() == group.reward.nunique() == 1,
                'trajectory task/outcome inconsistent')
        require(group.reward.iloc[0] in (0., 1.), 'nonbinary outcome')
        ix = group.index.to_numpy()
        g, e = features(s[ix], a[ix])
        geom.append(g)
        sem.append(e)
        rows.append(dict(trajectory_id=tid, task_id=group.task_id.iloc[0],
                         reward=int(group.reward.iloc[0]), steps=len(group)))
    out = dict(geometry=np.stack(geom), semantic=np.stack(sem), rows=rows,
               partition=args.partition, data_sha256=digest(args.data),
               train_sha256=digest(args.train_tasks), heldout_sha256=digest(args.heldout_tasks))
    torch.save(out, args.output)
    print(json.dumps(dict(partition=args.partition, steps=len(m), trajectories=len(rows),
                          tasks=m.task_id.nunique(), feature_shapes=[out['geometry'].shape,
                                                                   out['semantic'].shape])), flush=True)


def transform(g, mean, std):
    return (g - mean) / std / math.sqrt(g.shape[1])


def kernel(g, e, rg, re, config):
    kind, bandwidth = config
    if kind == 'semantic':
        return e @ re.T
    distance = np.maximum((g*g).sum(1)[:, None] + (rg*rg).sum(1)[None, :] - 2*g@rg.T, 0)
    kg = g @ rg.T if bandwidth == 0 else np.exp(-distance / (2 * bandwidth**2))
    return kg if kind == 'geometry' else (kg + e @ re.T) / 2


def fit(k, rows, alpha, outcome_weight=1.):
    """Squared outcome loss + task-balanced squared pairwise margin loss + L2.

    Solve in a kernel eigenfeature basis; all supplied trajectories contribute
    outcome loss, and mixed-outcome tasks additionally contribute ranking loss.
    """
    y = np.array([r['reward'] * 2 - 1 for r in rows], dtype=float)
    tasks = np.array([r['task_id'] for r in rows])
    require(alpha > 0 and outcome_weight > 0, 'positive regularization/outcome weight required')
    scale_outcome = math.sqrt(outcome_weight / len(rows))
    b = [np.eye(len(rows)) * scale_outcome]
    targets = [y * scale_outcome]
    mixed = []
    for task in sorted(set(tasks)):
        pos, neg = np.flatnonzero((tasks == task) & (y == 1)), np.flatnonzero((tasks == task) & (y == -1))
        if len(pos) and len(neg):
            mixed.append(list(itertools.product(pos, neg)))
    require(bool(mixed), 'no within-task contrastive pairs')
    for pairs in mixed:
        d = np.zeros((len(pairs), len(rows)))
        for i, (p, n) in enumerate(pairs):
            d[i, p], d[i, n] = 1, -1
        scale = math.sqrt(len(mixed) * len(pairs))
        b.append(d / scale)
        targets.append(np.full(len(pairs), 2. / scale))
    b, target = np.concatenate(b), np.concatenate(targets)
    eig, u = np.linalg.eigh((k + k.T) / 2)
    require(eig.min() > -1e-6, 'kernel is not positive semidefinite')
    root = np.sqrt(np.maximum(eig, 0))
    phi = u * root
    design = b @ phi
    weight = solve(design.T @ design + alpha * np.eye(len(rows)), design.T @ target,
                   assume_a='pos')
    # Equivalent representer coefficients; tiny eigenvalues are null directions.
    inverse = np.zeros_like(root)
    np.divide(1., root, out=inverse, where=root > 1e-10)
    return u @ (weight * inverse)


def bon(scores, rewards, n):
    require(len(scores) == len(rewards) and len(scores) > 0, 'empty/misaligned candidates')
    require(np.isfinite(scores).all() and set(rewards) <= {0, 1}, 'invalid scores/outcomes')
    # Match released protocol: exact uniform subset expectation; ties uniform.
    n = min(n, len(scores))
    results = []
    for ix in itertools.combinations(range(len(scores)), n):
        best = max(scores[i] for i in ix)
        results.append(np.mean([rewards[i] for i in ix if scores[i] == best]))
    return float(np.mean(results))


def metrics(rows, scores):
    scores = np.asarray(scores)
    require(np.isfinite(scores).all(), 'nonfinite predictions')
    y = np.array([r['reward'] for r in rows])
    task = np.array([r['task_id'] for r in rows])
    rates, failures, details = {}, [], []
    for n in (1, 2, 4):
        values = [bon(scores[task == t], y[task == t], n) for t in sorted(set(task))]
        rates[str(n)] = dict(accuracy=float(np.mean(values)), solved_task_equivalents=float(sum(values)))
    for t in sorted(set(task)):
        ix = np.flatnonzero(task == t)
        selected = ix[scores[ix] == scores[ix].max()]
        record = dict(task_id=t, selected=[rows[i]['trajectory_id'] for i in selected],
                      selected_reward=float(y[selected].mean()), oracle=int(y[ix].max()),
                      candidates=[dict(trajectory_id=rows[i]['trajectory_id'], reward=int(y[i]),
                                       score=float(scores[i])) for i in ix])
        details.append(record)
        if record['selected_reward'] < 1:
            failures.append(record)
    require(len(set(y)) == 2 and np.std(scores) > 0, 'AUC/Spearman undefined')
    return dict(bon=rates, auc=float(roc_auc_score(y, scores)),
                spearman=float(spearmanr(y, scores).statistic), tasks=len(set(task)),
                trajectories=len(rows), oracle_tasks=sum(d['oracle'] for d in details),
                wrong_tasks=failures, per_task=details)


def load_features(path, partition, args):
    data = torch.load(path, weights_only=False)
    train, held = splits(args)
    require(data['partition'] == partition, 'wrong feature partition')
    for key, path in [('data', args.data), ('train', args.train_tasks), ('heldout', args.heldout_tasks)]:
        require(data[key + '_sha256'] == digest(path), 'feature provenance mismatch: ' + key)
    require({r['task_id'] for r in data['rows']} == (train if partition == 'train' else held),
            'feature task mismatch')
    require(np.isfinite(data['geometry']).all() and np.isfinite(data['semantic']).all(), 'invalid features')
    return data


def train(args):
    data = load_features(args.features, 'train', args)
    g, e, rows = data['geometry'], data['semantic'], data['rows']
    tasks = sorted({r['task_id'] for r in rows})
    np.random.default_rng(1234).shuffle(tasks)
    folds = {t: i % 5 for i, t in enumerate(tasks)}
    assignment = np.array([folds[r['task_id']] for r in rows])
    # Fixed, predeclared search; no heldout features or labels loaded here.
    configs = [('semantic', 0)] + [(kind, bw) for kind in ('geometry', 'combined') for bw in (0, .5, 1., 2.)]
    candidates = [(cfg, alpha, ow) for cfg in configs for alpha in (.001, .01, .1, 1.)
                  for ow in (.01, .1, 1.)]
    cv = []
    for config, alpha, outcome_weight in candidates:
        pred = np.empty(len(rows))
        for fold in range(5):
            tr, va = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
            mean, std = g[tr].mean(0), np.maximum(g[tr].std(0), 1e-6)
            gt, gv = transform(g[tr], mean, std), transform(g[va], mean, std)
            coef = fit(kernel(gt, e[tr], gt, e[tr], config), [rows[i] for i in tr], alpha, outcome_weight)
            pred[va] = kernel(gv, e[va], gt, e[tr], config) @ coef
        result = metrics(rows, pred)
        # Primary selection: task-macro BoN=4, secondary: pairwise accuracy.
        pair = []
        for t in tasks:
            ix = [i for i, r in enumerate(rows) if r['task_id'] == t]
            v = [float(pred[p] > pred[n]) + .5 * float(pred[p] == pred[n])
                 for p in ix for n in ix if rows[p]['reward'] == 1 and rows[n]['reward'] == 0]
            if v:
                pair.append(np.mean(v))
        cv.append(dict(config=config, alpha=alpha, outcome_weight=outcome_weight,
                       bon4=result['bon']['4']['accuracy'],
                       pairwise_accuracy=float(np.mean(pair)), auc=result['auc']))
        print(json.dumps(cv[-1]), flush=True)
    best = max(cv, key=lambda x: (x['bon4'], x['pairwise_accuracy'], x['auc']))
    mean, std = g.mean(0), np.maximum(g.std(0), 1e-6)
    gt = transform(g, mean, std)
    coef = fit(kernel(gt, e, gt, e, best['config']), rows, best['alpha'], best['outcome_weight'])
    checkpoint = dict(mean=mean, std=std, reference_geometry=gt, reference_semantic=e,
                      coefficients=coef, selected=best, cv=cv, folds=folds,
                      training_tasks=tasks, training_steps=sum(r['steps'] for r in rows),
                      training_trajectories=len(rows), feature_sha256=digest(args.features),
                      source_sha256=digest(__file__), data_sha256=data['data_sha256'],
                      train_sha256=data['train_sha256'], heldout_sha256=data['heldout_sha256'])
    torch.save(checkpoint, args.output)
    write_json(args.output + '.selection.json', {k: checkpoint[k] for k in
               ('selected', 'cv', 'folds', 'training_steps', 'training_trajectories', 'source_sha256')})
    print('FROZEN ' + json.dumps(dict(path=args.output, sha256=digest(args.output), selected=best)), flush=True)


class BaselineHead(nn.Module):
    """Strict implementation of the supplied checkpoint's released MLP."""
    def __init__(self, cfg):
        super().__init__()
        require(cfg['activation'] == 'gelu' and cfg['layernorm'] and not cfg['residual'],
                'unsupported baseline architecture')
        width = cfg['width']
        self.inp = nn.Linear(cfg['hidden_size'], width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(cfg['depth'] - 2))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.hidden)
        self.out = nn.Linear(width, cfg['projection_dim'])

    def forward(self, x):
        x = F.gelu(self.inp(x))
        for lin, norm in zip(self.hidden, self.norms):
            x = F.gelu(norm(lin(x)))
        return self.out(x)


def baseline(args, rows):
    ck = torch.load(args.baseline, map_location='cpu', weights_only=False)
    sh, ah = BaselineHead(ck['cfg']), BaselineHead(ck['cfg'])
    sh.load_state_dict(ck['state_head'], strict=True)
    ah.load_state_dict(ck['action_head'], strict=True)
    sh.eval(); ah.eval()
    table = pq.read_table(args.data)
    meta = table.select(['trajectory_id', 'step_idx']).to_pandas()
    selected = []
    for row in rows:
        ix = meta[meta.trajectory_id == row['trajectory_id']].sort_values('step_idx').index[-12:]
        selected.extend(ix)
    selected = np.asarray(selected)
    subset = table.take(selected)
    s = np.asarray(subset['state_embedding'].to_pylist(), dtype=np.float32)
    a = np.asarray(subset['action_embedding'].to_pylist(), dtype=np.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(s), 256):
            scores.extend((F.normalize(sh(torch.from_numpy(s[i:i+256])), dim=-1) *
                           F.normalize(ah(torch.from_numpy(a[i:i+256])), dim=-1)).sum(-1).tolist())
    ids = meta.iloc[selected].trajectory_id.to_numpy()
    scores = np.array(scores)
    return np.array([scores[ids == r['trajectory_id']].mean() for r in rows])


def evaluate(args):
    require(not Path(args.output).exists(), 'evaluation output exists; refusing silent repeated holdout evaluation')
    ck = torch.load(args.checkpoint, weights_only=False)
    require(ck['source_sha256'] == digest(__file__), 'source changed after model freeze')
    data = load_features(args.features, 'heldout', args)
    train_tasks, _ = splits(args)
    require(set(ck['training_tasks']) == train_tasks, 'checkpoint training partition mismatch')
    for key in ('data_sha256', 'train_sha256', 'heldout_sha256'):
        require(ck[key] == data[key], 'checkpoint provenance mismatch: ' + key)
    g = transform(data['geometry'], ck['mean'], ck['std'])
    scores = kernel(g, data['semantic'], ck['reference_geometry'], ck['reference_semantic'],
                    ck['selected']['config']) @ ck['coefficients']
    enhanced = metrics(data['rows'], scores)
    reference = metrics(data['rows'], baseline(args, data['rows']))
    differences = np.array([e['selected_reward'] - b['selected_reward']
                            for e, b in zip(enhanced['per_task'], reference['per_task'])])
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(differences, (20000, len(differences)), replace=True).mean(1)
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    report = dict(enhanced=enhanced, baseline=reference, selected=ck['selected'],
                  target_achieved=enhanced['bon']['4']['solved_task_equivalents'] >= 32,
                  training_steps=ck['training_steps'], training_trajectories=ck['training_trajectories'],
                  provenance=dict(checkpoint=args.checkpoint, checkpoint_sha256=digest(args.checkpoint),
                                  baseline_sha256=digest(args.baseline), source_sha256=digest(__file__),
                                  data_sha256=data['data_sha256'], train_sha256=data['train_sha256'],
                                  heldout_sha256=data['heldout_sha256']),
                  comparison=dict(paired_wins=wins, paired_losses=losses,
                                  delta_bon4=float(differences.mean()),
                                  task_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist()),
                  protocol=dict(bon='exact uniform N-subsets; uniform top-score ties; min(N,candidates)',
                                auc_spearman_unit='trajectory', baseline='supplied best_head.pt; final 12 step cosine mean',
                                selection='5-fold task-grouped CV on 75 training tasks only',
                                limitations=['Embedding associations do not establish causality.',
                                             '30372 training rows are steps of 298 trajectories, not independent outcomes.',
                                             'Training-task embeddings may retain task-specific information.',
                                             'Historical baseline holdout results were already disclosed by the user.',
                                             'A 38-task point estimate does not establish general statistical superiority.']))
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ('target_achieved', 'comparison', 'selected')}), flush=True)
    for name in ('baseline', 'enhanced'):
        print(name, json.dumps({k: report[name][k] for k in ('bon', 'auc', 'spearman', 'oracle_tasks')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    p.add_argument('--data', default='/tmp/clmrepro/eval/deepswe_eval_opus5max.parquet')
    p.add_argument('--train-tasks', default='/tmp/clmrepro/heads/train_tasks.json')
    p.add_argument('--heldout-tasks', default='/tmp/clmrepro/heads/heldout_tasks.json')
    p.add_argument('--baseline', default='/tmp/clmrepro/heads/best_head.pt')
    p.add_argument('--partition', choices=['train', 'heldout'], default='train')
    p.add_argument('--features')
    p.add_argument('--checkpoint')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    print('COMMAND ' + ' '.join(sys.argv), flush=True)
    print('DEVICE cpu (explicit); no GPU requested or fallback', flush=True)
    if args.command in ('train', 'evaluate'):
        require(args.features is not None, '--features is required')
    if args.command == 'evaluate':
        require(args.checkpoint is not None, '--checkpoint is required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
