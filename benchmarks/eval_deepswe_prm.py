"""Recompute released-head offline PRM metrics; never infer missing rollouts."""
import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import spearmanr, rankdata
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def correlation(scores, rewards):
    scores, rewards = np.asarray(scores), np.asarray(rewards)
    require(np.isfinite(scores).all(), 'nonfinite correlation input')
    require(set(rewards) == {0, 1}, 'AUC requires both binary classes')
    pos = rewards == 1
    auc = (rankdata(scores)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    rho, p = spearmanr(scores, rewards)
    require(math.isfinite(rho), 'undefined Spearman correlation')
    return dict(roc_auc=float(auc), spearman_rho=float(rho), spearman_pvalue=float(p), count=len(scores))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--data', default='/tmp/clmrepro/eval/deepswe_eval_opus5max.parquet')
    ap.add_argument('--head', default='/tmp/clmrepro/heads/best_head.pt')
    ap.add_argument('--reference', default='/tmp/clmrepro/repo')
    ap.add_argument('--output', default='benchmarks/results/deepswe_prm_eval_results.json')
    args = ap.parse_args()
    out = Path(args.output)
    evidence = out.parent / 'deepswe_prm_evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(Path(args.reference) / p) for p in ('src', 'preprocessing', 'evaluation')]
    from clm.heads import make_head
    from bon_eval import best_of_n, aggregate
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print('Explicit execution device: CPU, float32, 4 threads; fresh inference, no cached scores', flush=True)
    pf = pq.ParquetFile(args.data)
    cols = ['trajectory_id', 'step_idx', 'task_id', 'model', 'config', 'reward']
    require(set(cols + ['state_embedding', 'action_embedding']) <= set(pf.schema_arrow.names), 'missing required columns')
    m = pf.read(columns=cols).to_pandas()
    require(len(m) == 44409 and m.task_id.nunique() == 113, 'unexpected dataset dimensions')
    require(not m.isnull().any().any(), 'null metadata')
    require(set(m.reward.unique()) == {0.0, 1.0}, 'reward is not binary')
    require(not m.duplicated(['trajectory_id', 'step_idx']).any(), 'duplicate trajectory step')
    require((m.step_idx >= 0).all() and (m.step_idx % 1 == 0).all(), 'invalid step indices')
    for key in ['task_id', 'reward', 'model', 'config']:
        require((m.groupby('trajectory_id')[key].nunique() == 1).all(), 'inconsistent trajectory ' + key)
    require(m.config.nunique() == 1, 'multiple policies need explicit grouping')
    ck = torch.load(args.head, map_location='cpu', weights_only=True)
    cfg = ck['cfg']
    kw = {k: cfg[k] for k in ['width', 'depth', 'activation', 'layernorm', 'residual']}
    kw.update(proj=cfg['projection_dim'], hidden=cfg['hidden_size'])
    heads = [make_head(**kw) for _ in range(2)]
    for head, name in zip(heads, ['state_head', 'action_head']):
        require(all(torch.isfinite(t).all() for t in ck[name].values()), 'nonfinite checkpoint weights')
        head.load_state_dict(ck[name], strict=True)
        head.eval()
    print('Checkpoint strictly loaded:', cfg, flush=True)
    scores = []
    with torch.inference_mode():
        for batch in pf.iter_batches(batch_size=1024, columns=['state_embedding', 'action_embedding']):
            d = batch.to_pydict()
            projected = []
            for head, col in zip(heads, ['state_embedding', 'action_embedding']):
                x = torch.tensor(np.asarray(d[col], dtype=np.float32))
                require(x.shape == (batch.num_rows, cfg['hidden_size']), 'embedding dimension mismatch')
                require(torch.isfinite(x).all(), 'nonfinite embeddings')
                z = head(x)
                require(torch.isfinite(z).all() and (z.norm(dim=-1) > 0).all(), 'invalid projected vectors')
                projected.append(F.normalize(z, dim=-1))
            scores.extend((projected[0] * projected[1]).sum(-1).tolist())
            print(f'Scored {len(scores)}/{len(m)} rows', flush=True)
    require(len(scores) == len(m), 'score count mismatch')
    m['clm_score'] = scores
    m.to_parquet(evidence / 'step_scores.parquet', index=False)
    trajectories = []
    for tid, g in m.groupby('trajectory_id', sort=True):
        g = g.sort_values('step_idx')
        trajectories.append(dict(trajectory_id=tid, task_id=g.task_id.iloc[0], reward=int(g.reward.iloc[0]),
                                 steps=len(g), score=aggregate(g.clm_score.tolist(), 12)))
    import pandas as pd
    t = pd.DataFrame(trajectories)
    t.to_csv(evidence / 'trajectory_scores.csv', index=False)
    splitdir = Path(args.head).parent
    train = set(json.loads((splitdir / 'train_tasks.json').read_text()))
    held = set(json.loads((splitdir / 'heldout_tasks.json').read_text()))
    require(not train & held and train | held == set(t.task_id), 'invalid train/heldout partition')
    reports = {}
    for label, tasks in [('all_113', set(t.task_id)), ('heldout', held), ('train_partition', train)]:
        tt = t[t.task_id.isin(tasks)]
        groups = list(tt.groupby('task_id'))
        per_task = {}
        for task, g in groups:
            candidates = list(zip(g.score, g.reward))
            row = dict(candidates=len(g), passed=int(g.reward.sum()), budgets={})
            for n in (1, 2, 4, 8):
                budget = min(n, len(g))
                r = best_of_n(candidates, budget)
                # Independent exhaustive subset check: small real candidate pools allow exact validation.
                brute = []
                for subset in itertools.combinations(candidates, budget):
                    top = max(s for s, _ in subset)
                    winners = [v for s, v in subset if s == top]
                    brute.append(sum(winners) / len(winners))
                require(abs(r['selected'] - np.mean(brute)) < 1e-12, 'BoN exact enumeration disagreement')
                row['budgets'][str(n)] = dict(effective_n=budget, **r)
            per_task[task] = row
        bon = {}
        for n in (2, 4, 8):
            short = {task: r['candidates'] for task, r in per_task.items() if r['candidates'] < n}
            rate = float(np.mean([r['budgets'][str(n)]['selected'] for r in per_task.values()]))
            bon[str(n)] = dict(strict_n_all_tasks_rate=None if short else rate,
                              available_candidate_capped_rate=rate, short_tasks=short,
                              expected_resolved_tasks=rate * len(groups),
                              oracle_available_candidate_capped_rate=float(np.mean([r['budgets'][str(n)]['oracle'] for r in per_task.values()])))
            if short:
                print(f'UNAVAILABLE strict BoN={n} on {label}: {len(short)}/{len(groups)} short groups; capped metric explicitly separate', flush=True)
        mm = m[m.task_id.isin(tasks)]
        reports[label] = dict(n_tasks=len(groups), n_trajectories=len(tt),
            pass_at_1_random_trajectory_task_macro=float(np.mean([r['budgets']['1']['random'] for r in per_task.values()])),
            greedy=dict(value=None, reason='No designated greedy decoding rollout or decoding metadata; random offline Pass@1 is not greedy.'),
            bon=bon, trajectory_correlation=correlation(tt.score, tt.reward),
            step_correlation=correlation(mm.clm_score, mm.reward), per_task=per_task)
    paths = [Path(args.data), Path(args.head), Path(__file__), splitdir/'train_tasks.json', splitdir/'heldout_tasks.json',
             splitdir/'README.md', Path(args.reference)/'evaluation/bon_eval.py', Path(args.reference)/'src/clm/heads.py',
             evidence/'step_scores.parquet', evidence/'trajectory_scores.csv']
    claim_rate = reports['heldout']['bon']['4']['available_candidate_capped_rate']
    result = dict(status='completed_with_explicit_unavailable_metrics',
        evaluation_type='offline PRM selection over supplied recorded trajectories; no environment execution; not a B2 policy rollout',
        dataset=dict(rows=len(m), tasks=t.task_id.nunique(), trajectories=len(t), columns=pf.schema_arrow.names,
                     candidate_count_histogram={str(k):int(v) for k,v in t.groupby('task_id').size().value_counts().items()},
                     models=m.model.unique().tolist(), configs=m.config.unique().tolist()),
        protocol=dict(score='cosine(state_head(state_embedding), action_head(action_embedding))',
                      aggregation='mean of final 12 available steps sorted by step_idx',
                      selection='Exact expectation over uniform subsets without replacement; uniform top-score ties; macro average over tasks',
                      reward='Dataset binary reward, required constant per trajectory; not re-executed in environment',
                      correlations='Pooled trajectory primary; pooled step secondary repeats terminal labels and weights long trajectories; p-values assume independent rows',
                      device='cpu', dtype='float32', threads=4),
        integrity=dict(files=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p)) for p in paths],
                       checkpoint_strict_load=True, all_embeddings_read_finite=True,
                       published_head_sha256_match=digest(args.head)=='554989fe88635606cb978dc45a1ce083be1990c4a51e551ea3b6055ead1a029a',
                       parquet_publisher_checksum='Unavailable: local SHA256 is reproducibility evidence, not independent authenticity proof'),
        software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pyarrow=pyarrow.__version__, scipy=scipy.__version__),
        results=reports,
        comparison_81_6=dict(claim_percent=81.6, source=str(splitdir/'README.md'), source_protocol='heldout 38 tasks, offline BoN=4, final 12 steps',
                             measured_heldout_percent=100*claim_rate, difference_percentage_points=100*claim_rate-81.6,
                             all_113_difference_percentage_points=100*reports['all_113']['bon']['4']['available_candidate_capped_rate']-81.6,
                             comparable_full_environment=False,
                             caveat='All 113 includes 75 training-partition tasks. Only heldout subset matches the local release claim protocol; source dataset identity beyond local metadata is not independently authenticated.'),
        limitations=['Strict BoN=8 impossible: at most 4 recorded candidates per task.',
                     'Strict BoN=4 over all 113 impossible: 3 tasks have only 3 candidates; capped metric follows reference explicitly.',
                     'Greedy decoding accuracy unavailable.', 'No new B2 inference policy or environment benchmark was run.',
                     'Existing parquet prm_score is not used; every score recomputed with supplied best_head.pt.'])
    require(result['integrity']['published_head_sha256_match'], 'checkpoint checksum differs from release README')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a != 'per_task'} for k,v in reports.items()}, indent=2), flush=True)
    print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
