"""Small mathematical fixtures; these are not benchmark evidence."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'enhanced_prm', Path(__file__).parents[1] / 'eval_deepswe_enhanced_prm.py')
prm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prm)


class EnhancedPRMTests(unittest.TestCase):
    def test_exact_subset_and_tie_expectation(self):
        self.assertAlmostEqual(prm.bon(np.array([0., 1., 2., 3.]), np.array([0, 1, 0, 1]), 2), 4/6)
        self.assertAlmostEqual(prm.bon(np.ones(4), np.array([0, 1, 0, 1]), 4), .5)
        self.assertEqual(prm.bon(np.array([2.]), np.array([1]), 4), 1.)

    def test_invalid_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            prm.bon(np.array([float('nan')]), np.array([1]), 1)
        with self.assertRaises(ValueError):
            prm.features(np.zeros((2, 3)), np.ones((2, 3)))

    def test_geometry_invariant_to_shared_rotation(self):
        rng = np.random.default_rng(40)
        s, a = rng.normal(size=(15, 8)), rng.normal(size=(15, 8))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        g, e = prm.features(s, a)
        gr, er = prm.features(s @ rotation, a @ rotation)
        np.testing.assert_allclose(g, gr, atol=1e-10)
        self.assertAlmostEqual(float(e @ e), float(er @ er))

    def test_temporal_order_is_used(self):
        rng = np.random.default_rng(12)
        s, a = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
        self.assertFalse(np.allclose(prm.features(s, a)[0], prm.features(s[::-1], a[::-1])[0]))

    def test_ranking_fit_uses_actual_outcomes(self):
        x = np.array([[-2.], [2.], [-1.], [1.]])
        rows = [dict(task_id=str(i//2), reward=i%2) for i in range(4)]
        k = x @ x.T
        pred = k @ prm.fit(k, rows, .01)
        self.assertGreater(pred[1], pred[0])
        self.assertGreater(pred[3], pred[2])
        opposite = [dict(task_id=r['task_id'], reward=1-r['reward']) for r in rows]
        np.testing.assert_allclose(k @ prm.fit(k, opposite, .01), -pred)

    def test_no_ranking_pairs_rejected(self):
        with self.assertRaises(ValueError):
            prm.fit(np.eye(2), [dict(task_id='a', reward=1), dict(task_id='b', reward=0)], .1)


if __name__ == '__main__':
    unittest.main()
