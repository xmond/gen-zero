#!/usr/bin/env python3
"""Executable, train-only-fit BoolQ geometry probe (not a production predictor).

Default is an explicitly logged 2048-row training subsample for a local probe.
Use --train-rows 0 for all cached training rows. No synthetic data or fallback.
Existing fuse* means logit pooling; procrustes* additionally tests feature mixing.
Spherical distances are induced metrics, not inferred intrinsic manifold distances.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import svd
from scipy.stats import binomtest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

import confidence_adaptive_logit_adjustment as cala
import evaluate_manifold_pareto_ensemble as ensemble
import geometric_latent_fusion as glf
import spec21_advanced_heads as heads


def summary(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite statistic')
    return {'n': int(x.size), 'mean': float(x.mean()) if x.size else None,
            'quantiles_0_25_50_75_100': np.quantile(x, [0, .25, .5, .75, 1]).tolist() if x.size else []}


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (n <= 1e-12).any():
        raise ValueError('undefined spherical direction (zero/nonfinite vector)')
    return x / n


def angles(a, b):
    return np.arccos(np.clip(a @ b.T, -1, 1))


def geometry(train, y, blocks, neighbors, tangent):
    """Normalized arithmetic centroids; local tangent-PCA residual as curvature proxy.

    Neighbors come ONLY from fit rows. Project neighbor displacements into the
    query sphere tangent plane, then measure energy outside its top tangent axes.
    This is a dimension/noise-sensitive bending proxy, NOT sectional curvature.
    """
    tr = unit(train)
    centers = unit(np.stack([tr[y == c].mean(0) for c in range(2)]))
    if neighbors > len(tr) or tangent >= min(neighbors - 1, tr.shape[1] - 1):
        raise ValueError('neighbor/tangent settings incompatible with fit geometry')
    nn = NearestNeighbors(n_neighbors=neighbors, metric='euclidean', algorithm='brute').fit(tr)
    output = []
    for block in blocks:
        u = unit(block)
        ix = nn.kneighbors(u, return_distance=False)
        curvature = []
        for row, ids in zip(u, ix):
            neighbors_u = tr[ids]
            displacement = neighbors_u - (neighbors_u @ row)[:, None] * row
            displacement -= displacement.mean(0)
            s = svd(displacement, compute_uv=False)
            energy = float(s @ s)
            if energy <= 1e-20:
                raise ValueError('degenerate neighborhood; curvature undefined')
            curvature.append(float(np.sum(s[tangent:] ** 2) / energy))
        output.append((angles(u, centers), np.array(curvature)))
    return output


def paired(y, base, new):
    delta = (new == y).astype(float) - (base == y).astype(float)
    gain, loss = int((delta > 0).sum()), int((delta < 0).sum())
    se = float(delta.std(ddof=1) / np.sqrt(len(y)))
    return {'rescued': gain, 'harmed': loss, 'unchanged': int((delta == 0).sum()),
            'delta_percentage_points': float(100 * delta.mean()),
            'delta_95pct_normal_ci_pp': [100 * (float(delta.mean()) - 1.96 * se),
                                        100 * (float(delta.mean()) + 1.96 * se)],
            'mcnemar_exact_two_sided_p': float(binomtest(gain, gain + loss).pvalue) if gain + loss else 1.0}


def run(args):
    started = time.monotonic()
    paths, q, l = ensemble.load_pair('boolq', args.qwen_dir, args.llama_dir)
    y = q['train_label']
    if y.ndim != 1 or y.dtype.kind not in 'iu' or set(y.tolist()) != {0, 1}:
        raise ValueError('expected integer binary training labels')
    for view in (q, l):
        for split in ('train', 'test'):
            if view[f'{split}_full'].shape[0] != len(view[f'{split}_ids']):
                raise ValueError('feature/ID length mismatch')
            if len(set(view[f'{split}_ids'])) != len(view[f'{split}_ids']):
                raise ValueError('duplicate IDs')
        if len(y) != len(view['train_full']) or view['cands'].shape[0] != 2:
            raise ValueError('label/candidate shape mismatch')
    ids = np.arange(len(y))
    if args.train_rows:
        if not 16 <= args.train_rows <= len(y):
            raise ValueError('--train-rows must be 0 (all) or between 16 and available rows')
        if args.train_rows < len(y):
            ids, _ = train_test_split(ids, train_size=args.train_rows, stratify=y, random_state=args.seed)
    fit, cal = train_test_split(ids, test_size=.25, stratify=y[ids], random_state=args.seed)
    print(f'[scope] cached train={len(y)}; selected={len(ids)}; fit={len(fit)}; calibration={len(cal)}; test={len(q["test_ids"])}; all feature columns retained', flush=True)
    print('[fit] existing GeometricLatentFusion + existing LedoitWolfLDAHead; calibration labels unused', flush=True)
    fq, fl = q['train_full'][fit], l['train_full'][fit]
    model = glf.GeometricLatentFusion.fit(fq, fl)
    cq, cl = model.view_q.coords(fq), model.view_l.coords(fl)
    blocks_q = [model.view_q.coords(q['train_full'][cal]), model.view_q.coords(q['test_full'])]
    blocks_l = [model.view_l.coords(l['train_full'][cal]), model.view_l.coords(l['test_full'])]
    # Column spaces live in the COMMON paired-sample ambient space R^n, not
    # incompatible model feature axes. Rank deficiency must not create QR axes.
    bases, spectra = [], []
    for name, c in [('qwen', cq), ('llama', cl)]:
        u, s, _ = svd(c, full_matrices=False)
        if s[-1] <= s[0] * max(c.shape) * np.finfo(float).eps:
            raise ValueError(f'{name}: retained coordinates numerically rank deficient')
        bases.append(u)
        spectra.append(s.tolist())
    cosines = svd(bases[0].T @ bases[1], compute_uv=False)
    theta = np.arccos(np.clip(cosines, 0, 1))
    report = {'task': 'b0925m-t3-boolq-exp', 'status': 'completed',
              'command': shlex.join([sys.executable, *sys.argv]),
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'feature_sha256': {str(p): ensemble.digest(p) for p in paths},
              'code_sha256': {str(Path(m.__file__).resolve()): ensemble.digest(Path(m.__file__)) for m in [ensemble, glf, heads, cala]},
              'probe_sha256': ensemble.digest(Path(__file__)),
              'scope': {'cached_train': len(y), 'fit_ids': q['train_ids'][fit].tolist(),
                        'calibration_ids': q['train_ids'][cal].tolist(), 'test_n': len(q['test_ids']),
                        'feature_dimensions': [fq.shape[1], fl.shape[1]], 'seed': args.seed},
              'geometry': {'gd_q': model.view_q.gd_info, 'gd_l': model.view_l.gd_info,
                           'core_dim': model.core_dim, 'geo_dim': model.output_dim,
                           'retained_coordinate_singular_values_q_l': spectra,
                           'cross_covariance_singular_values': model.cross_singular_values_.tolist(),
                           'principal_cosines': cosines.tolist(), 'principal_angles_degrees': np.degrees(theta).tolist(),
                           'principal_angles_summary_degrees': summary(np.degrees(theta)),
                           'grassmann_distance_radians': float(np.linalg.norm(theta)),
                           'unmatched_dimensions': abs(cq.shape[1] - cl.shape[1]),
                           'ambient': 'paired fit sample space R^n; unequal-rank distance uses min-rank principal angles'},
              'limitations': [
                  'Cached quantized Q4_K_M last-token features, not fresh model extraction or full-precision model runs.',
                  'Training subset and 25% calibration holdout are explicit; no full-train/OOF selection claim.',
                  'LDA is trained on GD coordinates; these controls are not full-8192-dimensional ensemble heads.',
                  'fuse* is original logit pooling, with weighted product-sphere geometry; procrustes* is additional feature mixing.',
                  'Distances are spherical angles to normalized arithmetic training centroids, not intrinsic learned geodesics.',
                  'Curvature is a local tangent-PCA residual proxy, confounded by noise, rank and neighborhood size.',
                  'All gate parameters fixed before test labels; exploratory multiple comparisons are not selection or breakthrough evidence.',
                  'Jev 89.70 is user-supplied, without paired predictions or verified identical evaluation protocol.',
                  'Geometry is linear truncation/rotation/mixing; it creates no information or trained nonlinear elevation.',
                  'This executable experimental CLI calls existing solver modules; no production deployment claimed.'
              ], 'source_audit': [
                  {'path': 'benchmarks/suites/evaluate_manifold_pareto_ensemble.py:197', 'finding': 'fuse weights normalized head logits, not Procrustes feature coordinates'},
                  {'path': 'benchmarks/suites/geometric_latent_fusion.py:19', 'finding': 'Invertibility claim is false when core_dim > 0: averaging removes one direction per paired core axis; output_dim=rq+rl-core_dim'},
                  {'path': 'benchmarks/suites/evaluate_manifold_pareto_ensemble.py:259', 'finding': 'Engine catches head failures and records missing keys; this probe instead propagates every fitting failure'}
              ], 'representations': {}}
    # Full standardized spectra, as well as retained/cross spectra, are recorded.
    report['geometry']['full_standardized_singular_values'] = {}
    for name, view, x in [('qwen', model.view_q, fq), ('llama', model.view_l, fl)]:
        spectrum = svd((x - view.mean) / view.scale, compute_uv=False)
        report['geometry']['full_standardized_singular_values'][name] = spectrum.tolist()
    report['geometry']['discarded_core_difference_dimensions'] = model.core_dim
    aligned_train = cq @ model.procrustes_
    aligned_blocks = [x @ model.procrustes_ for x in blocks_q]
    denom = float(np.linalg.norm(cl))
    report['geometry']['procrustes_relative_residual'] = float(np.linalg.norm(aligned_train - cl) / denom)
    reps = {'qwen': (cq, blocks_q), 'llama': (cl, blocks_l)}
    for w in (.25, .5, .75):
        reps[f'procrustes{w:g}'] = (w * aligned_train + (1-w) * cl,
                                   [w*a + (1-w)*b for a,b in zip(aligned_blocks, blocks_l)])
    for name, w in [('geo035', .35), ('geo100', 1.)]:
        reps[name] = (model.transform(fq, fl, w),
                      [model.transform(q['train_full'][cal], l['train_full'][cal], w),
                       model.transform(q['test_full'], l['test_full'], w)])
    raw, geom = {}, {}
    for name, (tr, blocks) in reps.items():
        print(f'[representation] {name} dim={tr.shape[1]}', flush=True)
        head = heads.LedoitWolfLDAHead.fit(tr, y[fit], 2, standardize=False)
        z = [head.scores(x) for x in blocks]
        scale = float(z[0].std())
        if scale <= 1e-12 or not np.isfinite(scale):
            raise ValueError(f'{name}: degenerate calibration logit scale')
        raw[name] = [v / scale for v in z]
        geom[name] = geometry(tr, y[fit], blocks, args.neighbors, args.tangent_dim)
    for w in (.25, .5, .75):
        name = f'fuse{w:g}'
        z = [w*a + (1-w)*b for a,b in zip(raw['qwen'], raw['llama'])]
        scale = float(z[0].std())
        if scale <= 1e-12:
            raise ValueError('degenerate fused calibration logits')
        raw[name] = [v / scale for v in z]
        # Product of unit spheres with weighted metric, not a fictitious
        # feature space silently assigned to an existing logit fusion.
        geom[name] = [(np.sqrt(w*a[0]**2 + (1-w)*b[0]**2), w*a[1]+(1-w)*b[1])
                      for a,b in zip(geom['qwen'], geom['llama'])]
    priors = heads.compute_class_priors(y[fit], 2)
    predictions = {}
    for name, (_, z) in raw.items():
        calibration_curvature = geom[name][0][1]
        curvature = geom[name][1][1]
        kscale = float(np.median(calibration_curvature))
        if kscale <= 1e-12:
            raise ValueError(f'{name}: degenerate curvature scale')
        # More local bending -> lower margin confidence -> larger CALA gate.
        base_phi = cala.gate_from_logits(z, 1., 'margin')
        confidence = 1 - base_phi
        phi = 1 - confidence / (1 + args.curvature_alpha * curvature / kscale)
        predictions[name] = {
            'raw': z.argmax(1),
            'margin': cala.cala_adjust(z, priors, args.tau, base_phi).argmax(1),
            'curvature_margin': cala.cala_adjust(z, priors, args.tau, phi).argmax(1)}
        report['representations'][name] = {'curvature_calibration_median': kscale,
                                          'raw_logits': z.tolist(), 'margin_phi': base_phi.tolist(),
                                          'curvature_margin_phi': phi.tolist()}
    print('[freeze] all predictions fixed; reading gold solely for paired diagnostics', flush=True)
    import grand_challenge_data as gd
    records = gd.load_test('boolq')
    if not records or any(r['candidates'] != ['false', 'true'] for r in records):
        raise ValueError('BoolQ candidate order must be false,true')
    gold = ensemble.load_gold('boolq', q['test_ids'], 2)
    report['test_data'] = gd.test_file_digest('boolq')
    report['test_ids'] = q['test_ids'].tolist()
    report['gold'] = gold.tolist()
    report['gate_formula'] = 'phi=1-margin/(1+alpha*kappa/median(kappa_cal)); logits-tau*phi*log(train_priors)'
    report['train_priors'] = priors.tolist()
    for name, preds in predictions.items():
        row = report['representations'][name]
        distances, curvature = geom[name][1]
        row['distance_to_false_true_radians'] = distances.tolist()
        row['curvature_proxy'] = curvature.tolist()
        row['predictions'] = {key: value.tolist() for key,value in preds.items()}
        row['metrics'] = {key: ensemble.metrics_from_pred(gold, value, 2) for key,value in preds.items()}
        row['paired_vs_raw'] = {key: paired(gold, preds['raw'], value) for key,value in preds.items() if key != 'raw'}
        row['paired_curvature_vs_margin'] = paired(gold, preds['margin'], preds['curvature_margin'])
        row['accuracy_95pct_wilson_ci'] = {key: list(binomtest(int((value == gold).sum()), len(gold)).proportion_ci(method='wilson')) for key, value in preds.items()}
        row['paired_raw_vs_qwen'] = paired(gold, predictions['qwen']['raw'], preds['raw'])
        row['distance_distributions'] = {}
        for mode, pred in preds.items():
            row['distance_distributions'][mode] = {}
            for group, mask in {'FN': (gold == 1) & (pred == 0), 'FP': (gold == 0) & (pred == 1),
                                'TP': (gold == 1) & (pred == 1), 'TN': (gold == 0) & (pred == 0)}.items():
                row['distance_distributions'][mode][group] = {
                    'to_false': summary(distances[mask, 0]), 'to_true': summary(distances[mask, 1]),
                    'true_minus_wrong': summary((distances[np.arange(len(gold)), gold] - distances[np.arange(len(gold)), 1-gold])[mask]),
                    'curvature': summary(curvature[mask])}
        print(f'[result] {name}: raw={row["metrics"]["raw"]["accuracy"]:.3f}% margin={row["metrics"]["margin"]["accuracy"]:.3f}% curvature={row["metrics"]["curvature_margin"]["accuracy"]:.3f}% rescued/harmed={row["paired_vs_raw"]["curvature_margin"]["rescued"]}/{row["paired_vs_raw"]["curvature_margin"]["harmed"]}', flush=True)
    qcorrect = predictions['qwen']['raw'] == gold
    lcorrect = predictions['llama']['raw'] == gold
    report['single_head_complementarity'] = {
        'both_correct': int((qcorrect & lcorrect).sum()),
        'qwen_only_correct': int((qcorrect & ~lcorrect).sum()),
        'llama_only_correct': int((~qcorrect & lcorrect).sum()),
        'both_wrong': int((~qcorrect & ~lcorrect).sum()),
        'oracle_accuracy_diagnostic_only': float(100 * (qcorrect | lcorrect).mean())}
    report['seconds'] = time.monotonic() - started
    print('[angles]', json.dumps(report['geometry']['principal_angles_summary_degrees']), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(f'[complete] {args.out} seconds={report["seconds"]:.2f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-dir', type=Path, default=ensemble.DEFAULT_QDIR)
    p.add_argument('--llama-dir', type=Path, default=ensemble.DEFAULT_LDIR)
    p.add_argument('--out', type=Path, default=ensemble.ROOT / 'benchmarks/results/boolq_elevation_probe_report.json')
    p.add_argument('--train-rows', type=int, default=2048, help='explicit subsample; 0 uses all cached train rows')
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--neighbors', type=int, default=32)
    p.add_argument('--tangent-dim', type=int, default=4)
    p.add_argument('--tau', type=float, default=1.)
    p.add_argument('--curvature-alpha', type=float, default=1.)
    args = p.parse_args()
    if args.threads < 1 or args.neighbors < 3 or args.tangent_dim < 1 or any(not np.isfinite(v) or v < 0 for v in [args.tau, args.curvature_alpha]):
        p.error('invalid numeric settings')
    try:
        with threadpool_limits(limits=args.threads):
            run(args)
    except Exception as exc:
        # Never leave a stale success report looking like the current attempt.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'status': 'failed', 'command': shlex.join([sys.executable, *sys.argv]),
                                        'error': f'{type(exc).__name__}: {exc}'}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()  # exceptions propagate with traceback and nonzero process status
