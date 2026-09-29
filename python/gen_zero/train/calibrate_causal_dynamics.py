"""Offline calibration from provenance-attested frozen feature NPZ files.

Required arrays: features, sample_ids and either (labels) or
(candidates, positive_indices). Required scalar JSON metadata:
source, split (train/calibration), encoder_id, label_free_encoder_input (true).
The producer is responsible for honest provenance; split strings cannot prove it.
No synthetic fallback and no benchmark results ingestion.
"""
import argparse
import json

import numpy as np

from gen_zero.causal.dynamics_calibrator import CausalDynamicsCalibrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta.get('label_free_encoder_input') is not True:
            raise ValueError('producer must attest label-free encoder inputs')
        required = {'features', 'sample_ids', 'metadata'}
        if not required.issubset(data.files):
            raise ValueError(f'missing required arrays: {sorted(required - set(data.files))}')
        x, ids = data['features'], data['sample_ids']
        has_class = 'labels' in data.files
        has_geometry = {'candidates', 'positive_indices'}.issubset(data.files)
        if has_class == has_geometry:
            raise ValueError('provide exactly one of labels or candidate geometry')
        y = data['labels'] if has_class else None
        candidates = data['candidates'] if has_geometry else None
        positive_indices = data['positive_indices'] if has_geometry else None
    calibrator = CausalDynamicsCalibrator(args.dim, args.rank, x.shape[1], seed=args.seed)
    provenance = {n: meta[n] for n in ('source', 'split', 'encoder_id')}
    if has_geometry:
        history = calibrator.fit_candidate_geometry(
            x, candidates, positive_indices, sample_ids=ids, **provenance,
            epochs=args.epochs, lr=args.lr)
    else:
        history = calibrator.fit(x, y, sample_ids=ids, **provenance,
                                 epochs=args.epochs, lr=args.lr)
    calibrator.adapter.save(args.out)
    print(json.dumps(dict(certificate=calibrator.adapter.certify(),
                          initial_loss=history[0], final_preupdate_loss=history[-1],
                          artifact=args.out, provenance=calibrator.adapter.provenance)))


if __name__ == '__main__':
    main()
