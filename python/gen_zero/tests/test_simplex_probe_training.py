"""Tests for the Simplex ETF linear-probe trainer (frozen-backbone features)."""

import os
import tempfile
import unittest

import numpy as np

import gen_zero.model  # noqa: F401  (import order: avoids a train/gate import cycle)
import torch

from gen_zero.train.train_simplex_probe import (
    SimplexProbe,
    SimplexProbeConfig,
    encode_labels,
    etf_margin_loss,
    load_features,
    load_probe,
    make_synthetic_features,
    save_probe,
    stratified_split,
    supervised_contrastive_loss,
    train_probe,
)

DIM = 64


class TestSimplexProbeLosses(unittest.TestCase):
    def test_etf_vertices_are_equiangular(self):
        probe = SimplexProbe(num_classes=5, hidden_dim=DIM)
        gram = probe.etf @ probe.etf.t()
        off = gram[~torch.eye(5, dtype=torch.bool)]
        self.assertTrue(torch.allclose(torch.diagonal(gram), torch.ones(5), atol=1e-5))
        self.assertTrue(torch.allclose(off, torch.full_like(off, -1 / 4), atol=1e-5))

    def test_only_w_proj_is_trainable_and_full_size(self):
        probe = SimplexProbe(num_classes=3, hidden_dim=DIM)
        params = [p for p in probe.parameters() if p.requires_grad]
        self.assertEqual(len(params), 1)
        self.assertEqual(tuple(params[0].shape), (DIM, DIM))

    def test_supcon_prefers_clustered_labels(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1.0, 0], [1, 0.01], [0, 1], [0.01, 1]])
        bad = torch.tensor([[1.0, 0], [0, 1], [1, 0.01], [0.01, 1]])
        norm = torch.nn.functional.normalize
        self.assertLess(
            supervised_contrastive_loss(norm(good, dim=1), labels, 0.1).item(),
            supervised_contrastive_loss(norm(bad, dim=1), labels, 0.1).item(),
        )

    def test_supcon_without_positives_is_finite_zero(self):
        q = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        loss = supervised_contrastive_loss(q, torch.tensor([0, 1, 2]), 0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)

    def test_margin_loss_zero_when_gap_met(self):
        cos = torch.tensor([[0.9, -0.1, -0.1]])
        self.assertEqual(etf_margin_loss(cos, torch.tensor([0]), 0.5).item(), 0.0)
        self.assertGreater(etf_margin_loss(cos, torch.tensor([1]), 0.5).item(), 0.0)


class TestSimplexProbeTraining(unittest.TestCase):
    def test_probe_beats_zero_shot_and_round_trips(self):
        data = make_synthetic_features(num_classes=4, samples_per_class=80, hidden_dim=DIM, seed=1)
        cfg = SimplexProbeConfig(hidden_dim=DIM, epochs=12, lr=1e-3, seed=1)
        probe, report = train_probe(data, cfg)
        self.assertEqual(report["trainable_params"], DIM * DIM)
        self.assertGreaterEqual(report["probe_val_acc"], 0.85)
        self.assertGreater(report["probe_val_acc"], report["zero_shot_val_acc"] + 0.3)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.pt")
            save_probe(probe, report, path)
            loaded = load_probe(path)
        x = torch.randn(5, DIM)
        self.assertTrue(torch.allclose(probe(x)[1], loaded(x)[1], atol=1e-6))

    def test_reloaded_probe_scores_raw_features_like_the_report(self):
        data = make_synthetic_features(num_classes=4, samples_per_class=80, hidden_dim=DIM, seed=2)
        cfg = SimplexProbeConfig(hidden_dim=DIM, epochs=8, lr=1e-3, seed=2)
        probe, report = train_probe(data, cfg)
        _, val_idx = stratified_split(data.labels, cfg.val_fraction, cfg.seed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.pt")
            save_probe(probe, report, path)
            loaded = load_probe(path)
        raw = torch.from_numpy(data.features[val_idx])  # raw, NOT pre-centered
        pred = loaded(raw)[1].argmax(dim=1)
        acc = (pred == torch.from_numpy(data.labels[val_idx])).float().mean().item()
        self.assertAlmostEqual(acc, report["probe_val_acc"], places=6)
        self.assertGreater(loaded.feature_mean.abs().sum().item(), 0.0)


class TestFeatureLoading(unittest.TestCase):
    def test_npz_loading_sorts_string_labels(self):
        feats = np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.npz")
            np.savez(path, features=feats, labels=np.array(["b", "a", "c", "a", "b", "c"]))
            data = load_features(path, hidden_dim=DIM)
        self.assertEqual(data.label_names, ["a", "b", "c"])  # canonical vertex order
        self.assertEqual(data.labels.tolist(), [1, 0, 2, 0, 1, 2])

    def test_loader_rejects_bad_shape_and_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.npz")
            np.savez(path, features=np.zeros((3, DIM + 1), np.float32), labels=np.array([0, 1, 0]))
            with self.assertRaises(ValueError):
                load_features(path, hidden_dim=DIM)
            nan = np.zeros((3, DIM), np.float32)
            nan[0, 0] = np.nan
            np.savez(path, features=nan, labels=np.array([0, 1, 0]))
            with self.assertRaises(ValueError):
                load_features(path, hidden_dim=DIM)

    def test_encode_labels_unknown_raises(self):
        with self.assertRaises(ValueError):
            encode_labels(["x"], names=["a", "b"])

    def test_stratified_split_keeps_every_class_in_train(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        train, val = stratified_split(labels, 0.5, seed=0)
        self.assertEqual(set(labels[train].tolist()), {0, 1, 2})
        self.assertTrue(set(train.tolist()).isdisjoint(val.tolist()))


if __name__ == "__main__":
    unittest.main()
