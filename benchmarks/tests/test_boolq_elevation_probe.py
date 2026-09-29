"""Acceptance checks on actual probe artifacts; no mocked/synthetic success data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/suites'))
import probe_boolq_elevation as probe


class BoolQProbeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'benchmarks/results/boolq_elevation_probe_report.json').read_text())
        if cls.report['status'] != 'completed':
            raise ValueError('real completed probe report required')

    def test_provenance_and_split(self):
        r = self.report
        for path, digest in {**r['feature_sha256'], **r['code_sha256']}.items():
            self.assertEqual(probe.ensemble.digest(Path(path)), digest)
        self.assertEqual(probe.ensemble.digest(Path(probe.__file__)), r['probe_sha256'])
        fit, cal, test = map(set, [r['scope']['fit_ids'], r['scope']['calibration_ids'], r['test_ids']])
        self.assertFalse(fit & cal or fit & test or cal & test)
        self.assertEqual(len(test), len(r['gold']))
        self.assertEqual(probe.ensemble.digest(Path(r['test_data']['path'])), r['test_data']['sha256'])

    def test_per_sample_metrics_and_groups(self):
        y = np.array(self.report['gold'])
        for row in self.report['representations'].values():
            for mode, values in row['predictions'].items():
                pred = np.array(values)
                self.assertAlmostEqual(100 * np.mean(pred == y), row['metrics'][mode]['accuracy'])
                groups = row['distance_distributions'][mode]
                self.assertEqual(sum(g['to_false']['n'] for g in groups.values()), len(y))
                self.assertEqual(groups['FN']['to_false']['n'], int(((y == 1) & (pred == 0)).sum()))
                self.assertEqual(groups['FP']['to_false']['n'], int(((y == 0) & (pred == 1)).sum()))
            p = row['predictions']
            for mode in ['margin', 'curvature_margin']:
                delta = (np.array(p[mode]) == y).astype(int) - (np.array(p['raw']) == y).astype(int)
                pair = row['paired_vs_raw'][mode]
                self.assertEqual(pair['rescued'], int((delta == 1).sum()))
                self.assertEqual(pair['harmed'], int((delta == -1).sum()))

    def test_gate_and_distance_math(self):
        r = self.report
        for row in r['representations'].values():
            z = np.array(row['raw_logits'])
            probability = np.exp(z - z.max(1, keepdims=True))
            probability /= probability.sum(1, keepdims=True)
            margin = np.abs(probability[:, 0] - probability[:, 1])
            k = np.array(row['curvature_proxy'])
            phi = 1 - margin / (1 + r['arguments']['curvature_alpha'] * k / row['curvature_calibration_median'])
            np.testing.assert_allclose(phi, row['curvature_margin_phi'], atol=1e-12)
            pred = (z - r['arguments']['tau'] * phi[:, None] * np.log(np.array(r['train_priors']) + 1e-12)).argmax(1)
            np.testing.assert_array_equal(pred, row['predictions']['curvature_margin'])
            d = np.array(row['distance_to_false_true_radians'])
            self.assertTrue(np.isfinite(d).all() and (d >= 0).all() and (d <= np.pi).all())
        g = r['geometry']
        np.testing.assert_allclose(np.degrees(np.arccos(np.clip(g['principal_cosines'], 0, 1))), g['principal_angles_degrees'])
        self.assertEqual(g['geo_dim'], g['gd_q']['rank'] + g['gd_l']['rank'] - g['core_dim'])

    def test_fail_closed_missing_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report.json'
            result = subprocess.run([sys.executable, str(Path(probe.__file__)), '--qwen-dir', tmp,
                                     '--out', str(out)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileNotFoundError', result.stderr)
            self.assertEqual(json.loads(out.read_text())['status'], 'failed')
            print(f'Fail-closed missing-feature subprocess exit={result.returncode}; FileNotFoundError; report.status=failed')

    def test_fail_closed_undefined_geometry(self):
        # Corrupt actual saved logits to ensure invalid numerical inputs raise.
        x = np.array(self.report['representations']['qwen']['raw_logits'])
        x[0] = np.nan
        with self.assertRaisesRegex(ValueError, 'undefined spherical'):
            probe.unit(x)


if __name__ == '__main__':
    unittest.main(verbosity=2)
