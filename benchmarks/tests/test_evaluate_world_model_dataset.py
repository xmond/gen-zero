"""Checks for metric semantics and complete-episode selection."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gen_zero.world_model.neural_dynamics import TransitionDataset

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_world_model_dataset.py"
spec = importlib.util.spec_from_file_location("evaluate_world_model_dataset", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_classification_confusion_and_auc():
    result = ev.classification(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.2]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["bce"] > 0


def test_single_class_auc_is_explicitly_undefined():
    assert ev.classification(np.ones(3), np.ones(3) * 0.8)["roc_auc"] is None


def test_nonfinite_prediction_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        ev.state_metrics(np.array([[np.nan]]), np.array([[1.0]]))


class DoublingModel:
    """step doubles the state; done fires once the state passes 5."""

    def step(self, state, action):
        nxt = state * 2
        return nxt, 1.0, bool(nxt[0] > 5)


def _ds():
    return TransitionDataset(
        np.array([[1], [2], [3], [6], [1]], dtype=np.float32),
        np.ones((5, 1), dtype=np.float32),
        np.array([[2], [4], [6], [12], [2]], dtype=np.float32),
        np.array([1, 1, 1, 0, 1], dtype=np.float32),
        episode_ids=np.array([0, 0, 1, 1, 2]),
    )


def test_rollout_rejects_partial_episode():
    ds = _ds()
    with pytest.raises(ValueError, match="episode 1 is split"):
        ev.rollout(DoublingModel(), ds, ds.episode_ids, np.array([0, 1, 2]), horizon=2)


def test_rollout_feeds_back_own_state_and_tracks_survival():
    ds = _ds()
    result = ev.rollout(DoublingModel(), ds, ds.episode_ids, np.arange(5), horizon=3)
    assert result["episodes"] == 3
    s1, s2, s3 = result["steps"]
    assert (s1["n"], s2["n"], s3["n"]) == (3, 2, 0)
    assert s2["reach_rate"] == pytest.approx(2 / 3)
    # Episode 0 and 1 are exact under doubling; episode 2 is exact at step 1.
    assert s1["mse"] == s2["mse"] == 0.0
    assert s1["cosine_similarity"] == pytest.approx(1.0)
    # Identity baseline predicts s_0: ep0 (1 vs 4)=9, ep1 (3 vs 12)=81 at step 2.
    assert s2["identity_baseline_mse"] == pytest.approx(45.0)
    # Episode 1 dies at step 2 (reward 0) and the model's done fires there (12 > 5).
    assert s2["true_survival_rate"] == 0.5 and s2["predicted_survival_rate"] == 0.5
    assert s2["survival_agreement"] == 1.0
    # Episode 1 step 1 predicts 6 > 5 (done) while reward is 1: disagreement.
    assert s1["predicted_survival_rate"] == pytest.approx(2 / 3)
    assert s1["survival_agreement"] == pytest.approx(2 / 3)


def test_checked_split_refuses_mismatched_training_report(tmp_path):
    ds = _ds()
    model_path, data_path = tmp_path / "m.pt", tmp_path / "d.npz"
    model_path.write_bytes(b"model")
    ds.save_npz(data_path)
    _, val_idx = ds.split_indices(0.5, 0)
    report = {"split_mode": "grouped_by_episode", "checkpoint_sha256": ev.sha256(model_path),
              "data_sha256": ev.sha256(data_path), "hyperparameters": {"val_fraction": 0.5, "seed": 0},
              "val_episode_ids": np.unique(ds.episode_ids[val_idx]).tolist()}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(report))
    splits, _ = ev._checked_split(ds, model_path, data_path, rp)
    np.testing.assert_array_equal(splits["val"], val_idx)
    for key, bad in (("checkpoint_sha256", "0" * 64), ("split_mode", "random_rows"), ("val_episode_ids", [99])):
        rp.write_text(json.dumps({**report, key: bad}))
        with pytest.raises(ValueError):
            ev._checked_split(ds, model_path, data_path, rp)
