"""CPU regression checks; synthetic encodings are not GPU benchmark evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "remote_eval_v6", Path(__file__).resolve().parents[1] / "suites" / "run_remote_eval_v6.py")
v6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v6
SPEC.loader.exec_module(v6)


class PawsRegressionTests(unittest.TestCase):
    def test_signed_difference_preserves_roles(self):
        a, b = np.array([3., 1., 0.]), np.array([1., 3., 0.])
        delta = v6.relational_difference(a, b)
        np.testing.assert_array_equal(delta, [2., -2., 0.])
        np.testing.assert_array_equal(v6.relational_difference(b, a), -delta)
        np.testing.assert_array_equal(v6.relational_difference(a, a), np.zeros(3))
        z = np.tile([0., 0., 1.], (3, 1))
        features = v6.paws_probe_features(z, np.stack([delta, -delta, delta * 0]))
        self.assertEqual(features.shape, (3, 6))
        np.testing.assert_array_equal(features[0, 3:], -features[1, 3:])
        self.assertFalse(np.allclose(features[0], features[2]))

    def test_collection_uses_separate_sentences_and_accounts_for_latency(self):
        class Engine:
            def __init__(self):
                self.prompts = []

            def forward(self, prompt):
                self.prompts.append(prompt)
                return np.array([len(self.prompts), 1.], dtype=float), 2., None

        context = ('Sentence 1: Alice follows Bob.\nSentence 2: Bob follows Alice.\n'
                   'Do these two sentences have the exact same meaning?')
        self.assertEqual(v6.paws_sentences(context), ('Alice follows Bob.', 'Bob follows Alice.'))
        with self.assertRaises(ValueError):
            v6.paws_sentences('Sentence 1: only one sentence')
        engine = Engine()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'paws.jsonl').write_text(json.dumps({
                'context': context, 'candidates': [], 'ground_truth': 'not_paraphrase'}) + '\n')
            errors = {'exceptions': [], 'misses': []}
            records = v6.collect_records(engine, Path(directory), None, True, errors)
        self.assertEqual(errors['exceptions'], [])
        self.assertEqual(engine.prompts[1:], ['Sentence: Alice follows Bob.\n', 'Sentence: Bob follows Alice.\n'])
        np.testing.assert_array_equal(records[0].h_difference, [-1., 0.])
        self.assertEqual(records[0].forward_ms, 6.)

    def test_v6_pairwise_analysis_returns_sentence_pair_probabilities(self):
        records = v6._synthetic_records(np.random.default_rng(7))
        for record in records:
            if record.task == 'paws':
                record.h_difference = np.arange(record.h.size, dtype=float)
                # Isolate the label-free manifold expert so analyze's returned
                # probabilities are its probabilities, rather than a fused AR
                # distribution.  v6 requires this signed feature and computes
                # ``zt`` from it, while its final manifold scores currently use
                # the separately retained sentence states below.
                record.ar_scores = None
        paws = [record for record in records if record.task == 'paws']

        # Reproduce the production pairwise geometry and calibration from the
        # actual sentence states.  This is the numerical expected output for
        # the isolated manifold_alignment expert.
        s1 = np.stack([record.h_sentence1 for record in paws]).astype(np.float64)
        s2 = np.stack([record.h_sentence2 for record in paws]).astype(np.float64)
        pooled_mean = np.concatenate([s1, s2], axis=0).mean(axis=0)
        cos = np.sum(
            v6.l2_normalize(s1 - pooled_mean)
            * v6.l2_normalize(s2 - pooled_mean),
            axis=1,
        )
        g = (cos - cos.mean()) / max(float(cos.std()), 1e-6)
        para_idx, not_para_idx = v6.paws_candidate_indices(paws[0].candidates)
        geo_scores = np.zeros((len(paws), len(paws[0].candidates)))
        geo_scores[:, para_idx] = g
        geo_scores[:, not_para_idx] = -g
        expected_probs, expected_taus = v6.decoupled_temperature_calibrate(
            geo_scores, tau=1.0
        )

        result = v6.analyze(records, v6.AnalysisConfig())
        self.assertTrue(result['task_meta']['paws']['relational_features'])
        self.assertEqual(
            result['task_meta']['paws']['pairwise_head'],
            'contrastive_manifold_alignment',
        )
        result_by_index = {entry['i']: entry for entry in result['results']}
        paws_indices = [index for index, record in enumerate(records) if record.task == 'paws']
        for local, (record, index) in enumerate(zip(paws, paws_indices)):
            entry = result_by_index[index]
            self.assertEqual(set(entry['expert_pred']), {'manifold_alignment'})
            np.testing.assert_allclose(
                entry['probs'], expected_probs[local], rtol=1e-12, atol=1e-12
            )
            self.assertAlmostEqual(
                entry['expert_tau']['manifold_alignment'], expected_taus[local], places=12
            )

        # Ground-truth changes must not alter this label-free expert's output.
        for record in records:
            if record.candidates:
                current = record.candidates.index(record.ground_truth)
                record.ground_truth = record.candidates[(current + 1) % len(record.candidates)]
        relabeled = v6.analyze(records, v6.AnalysisConfig())
        relabeled_by_index = {entry['i']: entry for entry in relabeled['results']}
        for record, index in zip(paws, paws_indices):
            np.testing.assert_allclose(
                relabeled_by_index[index]['probs'],
                result_by_index[index]['probs'],
                rtol=1e-12,
                atol=1e-12,
            )
        with self.assertRaises(v6.ContractViolation):
            v6.analyze(records, v6.AnalysisConfig(probe_mode='loo'))
        records[0].h_difference = None
        with self.assertRaisesRegex(ValueError, 'Incomplete PAWS'):
            v6.analyze(records, v6.AnalysisConfig())

    def test_exception_does_not_disclose_private_paths(self):
        class Engine:
            def forward(self, prompt):
                raise OSError('cannot read ' + str(Path.home() / 'private-model' / 'weights'))

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'paws.jsonl').write_text('{}\n')
            errors = {'exceptions': [], 'misses': []}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(v6.collect_records(Engine(), Path(directory), None, True, errors), [])
        serialized = json.dumps(errors) + output.getvalue()
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn('private-model', serialized)
        self.assertEqual(errors['exceptions'][0]['error'], 'OSError')


if __name__ == '__main__':
    unittest.main()
