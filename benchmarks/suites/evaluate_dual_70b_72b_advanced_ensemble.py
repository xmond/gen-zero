#!/usr/bin/env python3
"""Five-fold training-only expert and probability-fusion selection on paired features."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import grand_challenge_data as gd
import evaluate_dual_70b_72b_ensemble as base
from spec21_advanced_heads import BBPAdaptiveProbe, LedoitWolfLDAHead, compute_class_priors, logit_adjust

DEFAULT_OUT = base.ROOT / 'benchmarks/results/spec21_dual_70b_72b_advanced_ensemble_report'
OUT = DEFAULT_OUT
HEADS = ('ridge', 'weighted_ridge', 'logistic', 'weighted_logistic', 'lw_lda', 'bbp')
TAUS = (0.0, 0.5, 1.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def score(y, pred):
    return {'accuracy': 100 * accuracy_score(y, pred),
            'balanced_accuracy': 100 * balanced_accuracy_score(y, pred),
            'macro_f1': 100 * f1_score(y, pred, average='macro', zero_division=0)}


def fit_head(name, X, y, Q, K, seed):
    if name in ('ridge', 'weighted_ridge'):
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        x, q = (X-mu)/sd, (Q-mu)/sd
        x, q = np.c_[x, np.ones(len(x))], np.c_[q, np.ones(len(q))]
        Y = np.eye(K)[y]
        w = np.bincount(y, minlength=K).astype(float)
        row_w = np.ones(len(y)) if name == 'ridge' else len(y)/(K*w[y])
        reg = np.eye(x.shape[1])*base.ALPHA
        reg[-1,-1] = 0
        coef = np.linalg.solve(x.T@(row_w[:,None]*x)+reg, x.T@(row_w[:,None]*Y))
        return q@coef
    if name in ('logistic', 'weighted_logistic'):
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced' if name == 'weighted_logistic' else None,
                                 random_state=seed).fit((X-mu)/sd, y)
        raw = clf.decision_function((Q-mu)/sd)
        return np.stack((-raw/2, raw/2), axis=1) if K == 2 else raw
    if name == 'lw_lda':
        return LedoitWolfLDAHead.fit(X, y, K).scores(Q)
    return BBPAdaptiveProbe.fit(X, y, K, C=1.0, seed=seed).scores(Q)


def named_scores(name, raw, priors, scale):
    return {f'{name}+la{tau:g}': softmax(logit_adjust(raw / scale, priors, tau), axis=1)
            for tau in TAUS}


def objective(y, pred, imbalanced):
    m = score(y, pred)
    return m['balanced_accuracy'] if imbalanced else m['accuracy']


def run_task(task, qdir=base.DEFAULT_QDIR, ldir=base.DEFAULT_LDIR):
    paths, data = base.load(task, qdir, ldir)
    y = data[0]['train_label'].astype(int)
    K = len(data[0]['cands'])
    if np.bincount(y, minlength=K).min() < 5:
        raise ValueError(f'{task}: insufficient class rows for five stratified folds')
    Z = [base.project(d, base.SEED + 100*gd.TASKS.index(task)+i) for i,d in enumerate(data)]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=base.SEED).split(Z[0][0], y))
    imbalance = np.bincount(y).max()/np.bincount(y).min() >= 3
    oof = [{f'{h}+la{t:g}': np.zeros((len(y),K)) for h in HEADS for t in TAUS} for _ in range(2)]
    failures = []
    for fi,(tr,ho) in enumerate(folds):
        priors = compute_class_priors(y[tr], K)
        for branch in range(2):
            for head in HEADS:
                try:
                    raw = fit_head(head, Z[branch][0][tr], y[tr], Z[branch][0][ho], K, base.SEED+fi)
                    scale = max(float(np.std(raw)), 1e-8)
                    for key,p in named_scores(head, raw, priors, scale).items():
                        oof[branch][key][ho] = p
                except Exception as exc:
                    failures.append(f'{branch}:{head}:fold{fi}:{type(exc).__name__}:{exc}')
                    for tau in TAUS:
                        oof[branch][f'{head}+la{tau:g}'][ho] = np.nan
    valid = [{k:v for k,v in branch.items() if np.isfinite(v).all()} for branch in oof]
    if not all(valid):
        raise RuntimeError(f'{task}: one branch has no valid heads: {failures}')
    options = []
    for branch in range(2):
        for key,p in valid[branch].items():
            options.append((objective(y,p.argmax(1),imbalance), {'kind':'single','branch':branch,'head':key}))
    for qk,qp in valid[0].items():
        for lk,lp in valid[1].items():
            for w in WEIGHTS[1:-1]:
                options.append((objective(y,(w*qp+(1-w)*lp).argmax(1),imbalance),
                                {'kind':'fusion','qwen_head':qk,'llama_head':lk,'qwen_weight':w}))
    # Deterministic tie order: singles first, then lexicographic pool traversal.
    best_value, selected = max(options, key=lambda item:item[0])
    singles = {}
    for branch,label in enumerate(('qwen','llama')):
        subset = [(objective(y,p.argmax(1),imbalance),key) for key,p in valid[branch].items()]
        singles[label] = max(subset, key=lambda item:item[0])[1]
    needed = {(0,singles['qwen']),(1,singles['llama'])}
    if selected['kind']=='single': needed.add((selected['branch'],selected['head']))
    else: needed.update(((0,selected['qwen_head']),(1,selected['llama_head'])))
    testp = {}
    for branch,key in needed:
        head,tau = key.split('+la')
        raw = fit_head(head,Z[branch][0],y,Z[branch][1],K,base.SEED)
        prior = compute_class_priors(y,K)
        # Full-fit score spread uses training predictions; no test labels.
        trainraw = fit_head(head,Z[branch][0],y,Z[branch][0],K,base.SEED)
        scale = max(float(np.std(trainraw)),1e-8)
        testp[branch,key] = softmax(logit_adjust(raw/scale,prior,float(tau)),axis=1)
    records = gd.load_test(task)
    if [r['id'] for r in records] != data[0]['test_ids'].tolist():
        raise ValueError(f'{task}: test ID mismatch')
    cands=records[0]['candidates']
    if len(cands)!=K or any(r['candidates']!=cands for r in records):
        raise ValueError(f'{task}: candidate mismatch')
    gold=np.array([cands.index(r['ground_truth']) for r in records])
    q,l=testp[0,singles['qwen']],testp[1,singles['llama']]
    if selected['kind']=='single': chosen=testp[selected['branch'],selected['head']]
    else: chosen=selected['qwen_weight']*testp[0,selected['qwen_head']]+(1-selected['qwen_weight'])*testp[1,selected['llama_head']]
    result={'n_train':len(y),'n_test':len(gold),'classes':K,'selection_metric':'balanced_accuracy' if imbalance else 'accuracy',
            'oof_selected_score':best_value,'selected':selected,'single_heads':singles,
            'metrics':{'qwen_best':score(gold,q.argmax(1)),'llama_best':score(gold,l.argmax(1)),
                       'selected':score(gold,chosen.argmax(1))},'fit_failures':failures,
            'feature_sha256':{str(p):base.digest(p) for p in paths},
            'test_sha256':base.digest(gd.TEST_DIR/f'{task}.jsonl')}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-dir', type=Path, default=base.DEFAULT_QDIR,
                         help='Directory with per-task Qwen-72B feature .npz files '
                              '(default: $MASTER_QWEN_DIR or %(default)s)')
    parser.add_argument('--llama-dir', type=Path, default=base.DEFAULT_LDIR,
                         help='Directory with per-task Llama-70B feature .npz files '
                              '(default: $MASTER_LLAMA_DIR or %(default)s)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                         help='Output path stem for the .json/.md report (default: %(default)s)')
    parser.add_argument('--tasks', nargs='+', default=None,
                         help='Subset of tasks to evaluate (default: all 13 tasks).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        tasks = base.resolve_tasks(args.tasks)
        base.check_inputs(args.qwen_dir, args.llama_dir, tasks)
    except (ValueError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    rows={}
    for task in tasks:
        rows[task]=run_task(task, args.qwen_dir, args.llama_dir)
        print(task,rows[task]['selected'],rows[task]['metrics']['selected'],flush=True)
    baseline={n:json.loads((base.ROOT/f'benchmarks/results/spec21_{n}_scorecard.json').read_text()) for n in ('qwen72b','llama70b')}
    agg={name:{metric:float(np.mean([r['metrics'][name][metric] for r in rows.values()])) for metric in ('accuracy','balanced_accuracy','macro_f1')}
         for name in ('qwen_best','llama_best','selected')}
    winners=Counter((r['selected'].get('head') if r['selected']['kind']=='single' else
                     r['selected']['qwen_head']+' / '+r['selected']['llama_head']) for r in rows.values())
    head_types=Counter()
    for r in rows.values():
        s=r['selected']
        for key in (('head',) if s['kind']=='single' else ('qwen_head','llama_head')):
            head_types[s[key].split('+')[0]] += 1
    lift={metric:agg['selected'][metric]-max(agg['qwen_best'][metric],agg['llama_best'][metric])
          for metric in ('accuracy','balanced_accuracy','macro_f1')}
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'command':' '.join(sys.argv),
            'protocol':{'folds':5,'seed':base.SEED,'projection_width':base.WIDTH,'selection':'training OOF only; balanced accuracy when train max/min class count >= 3, otherwise accuracy','test_labels':'loaded after selection; never passed to fit or selection','heads':HEADS,'logit_adjustment_tau':TAUS,'fusion_weights':WEIGHTS},
            'published_reference_macro_accuracy':{n:baseline[n]['aggregate']['macro_all'] for n in baseline},
            'aggregate_macro':agg,'macro_lift_vs_best_matched_single_pp':lift,
            'selected_expert_distribution':dict(winners),'selected_head_type_distribution':dict(head_types),'tasks':rows}
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# Advanced dual 70B/72B expert evaluation','','Five stratified training folds select the branch, expert and fusion weight. Test labels are loaded only after selection. Scores are percentages.','',
           'Published 81.07% / 81.09% single-model references used a different selection protocol, so the matched single-model columns are the direct controls.','',
           '| Method | Macro accuracy | Macro balanced accuracy | Macro F1 |','|---|---:|---:|---:|']
    for name,m in agg.items(): lines.append(f"| {name} | {m['accuracy']:.2f} | {m['balanced_accuracy']:.2f} | {m['macro_f1']:.2f} |")
    lines += ['', f"Selected macro accuracy {agg['selected']['accuracy']:.2f}% versus published Qwen {report['published_reference_macro_accuracy']['qwen72b']:.2f}% and Llama {report['published_reference_macro_accuracy']['llama70b']:.2f}%: the published threshold was not reached.", '', 'Macro lift versus the stronger matched single model (percentage points): '+', '.join(f'{k} {v:+.2f}' for k,v in lift.items()),'', '| Task | Qwen best head | Qwen acc | Llama best head | Llama acc | Selected expert | Selected acc | Selected balanced acc | Selected F1 |',
              '|---|---|---:|---|---:|---|---:|---:|---:|']
    for task,r in rows.items():
        s=r['selected']; label=(f"{['Qwen','Llama'][s['branch']]} {s['head']}" if s['kind']=='single' else f"{s['qwen_head']} / {s['llama_head']} @ {s['qwen_weight']:.2f}")
        m=r['metrics'];lines.append(f"| {task} | {r['single_heads']['qwen']} | {m['qwen_best']['accuracy']:.2f} | {r['single_heads']['llama']} | {m['llama_best']['accuracy']:.2f} | {label} | {m['selected']['accuracy']:.2f} | {m['selected']['balanced_accuracy']:.2f} | {m['selected']['macro_f1']:.2f} |")
    lines += ['','## Selected head type distribution','']+[f'- {k}: {v}' for k,v in head_types.items()]
    lines += ['','## Selected expert combinations','']+[f'- {k}: {v}' for k,v in winners.items()]
    lines += ['','Full hashes, OOF selections, metrics and fit failures are in the JSON report.','']
    args.out.with_suffix('.md').write_text('\n'.join(lines))
    print('WROTE',args.out.with_suffix('.json'),args.out.with_suffix('.md'),flush=True)
    return 0

if __name__=='__main__': sys.exit(main())
