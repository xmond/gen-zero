"""Tests for NeuralDynamicsWorldModel and its training pipeline."""

import importlib.util
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from gen_zero.world_model.neural_dynamics import (
    NeuralDynamicsWorldModel,
    TransitionDataset,
    joint_loss,
    make_synthetic_transitions,
    train_step,
)
from gen_zero.planner.engines.mcts_engine import MctsEngine

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIM = 6
ACTIONS = ["left", "right", "stay"]


def _load_train_script():
    path = REPO_ROOT / "scripts" / "train_world_model_dynamics.py"
    spec = importlib.util.spec_from_file_location("train_world_model_dynamics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model():
    return NeuralDynamicsWorldModel(
        state_dim=STATE_DIM, action_dim=len(ACTIONS), hidden_dim=32, action_vocab=ACTIONS
    )


def _trained_checkpoint(tmp_path: Path) -> Path:
    data = make_synthetic_transitions(num_samples=256, state_dim=STATE_DIM, action_vocab=ACTIONS, seed=0)
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        train_step(model, opt, data.as_tensors(), bce_weight=1.0)
    path = tmp_path / "wm.pt"
    model.save_checkpoint(path)
    return path


def test_forward_shapes_and_residual():
    model = _model()
    z = torch.randn(4, STATE_DIM)
    a = torch.eye(len(ACTIONS))[[0, 1, 2, 0]]
    next_z, r_prob, r_logit = model(z, a)
    assert next_z.shape == (4, STATE_DIM)
    assert r_prob.shape == (4,)
    assert r_logit.shape == (4,)
    assert torch.all((r_prob >= 0) & (r_prob <= 1))
    delta = model.transition_net(torch.cat([z, a], dim=-1))
    assert torch.allclose(next_z, z + delta)


def test_step_fails_closed_without_weights():
    model = _model()
    with pytest.raises(RuntimeError, match="World model weights not loaded - fail closed"):
        model.step(np.zeros(STATE_DIM, dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_step_fails_closed_after_saving_untrained_model(tmp_path):
    # Saving does not mark the in-memory instance as loaded.
    model = _model()
    model.save_checkpoint(tmp_path / "x.pt")
    with pytest.raises(RuntimeError, match="fail closed"):
        model.step(np.zeros(STATE_DIM, dtype=np.float32), "left")


def test_train_step_reduces_loss():
    data = make_synthetic_transitions(num_samples=512, state_dim=STATE_DIM, action_vocab=ACTIONS, seed=1)
    batch = data.as_tensors()
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        first = joint_loss(model, batch, bce_weight=1.0)[0].item()
    for _ in range(60):
        train_step(model, opt, batch, bce_weight=1.0)
    with torch.no_grad():
        last = joint_loss(model, batch, bce_weight=1.0)[0].item()
    assert last < first * 0.5, (first, last)


def test_save_load_roundtrip_is_exact(tmp_path):
    path = _trained_checkpoint(tmp_path)
    a = NeuralDynamicsWorldModel.from_checkpoint(path)
    b = NeuralDynamicsWorldModel.from_checkpoint(path)
    s = np.linspace(-1, 1, STATE_DIM).astype(np.float32)
    na, ra, da = a.step(s, "right")
    nb, rb, db = b.step(s, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert na.shape == (STATE_DIM,) and na.dtype == np.float32
    np.testing.assert_array_equal(na, nb)
    assert ra == rb and da == db
    assert 0.0 <= ra <= 1.0
    assert isinstance(da, bool)
    assert da == (ra < a.done_threshold)


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _model().load_checkpoint(tmp_path / "missing.pt")


def test_load_dimension_mismatch_raises(tmp_path):
    path = _trained_checkpoint(tmp_path)
    other = NeuralDynamicsWorldModel(state_dim=STATE_DIM + 1, action_dim=len(ACTIONS), hidden_dim=32)
    with pytest.raises(ValueError, match="state_dim"):
        other.load_checkpoint(path)
    with pytest.raises(RuntimeError, match="fail closed"):
        other.step(np.zeros(STATE_DIM + 1, dtype=np.float32), np.zeros(len(ACTIONS), dtype=np.float32))


def test_unknown_action_and_bad_shapes_raise(tmp_path):
    model = NeuralDynamicsWorldModel.from_checkpoint(_trained_checkpoint(tmp_path))
    with pytest.raises(KeyError):
        model.step(np.zeros(STATE_DIM, dtype=np.float32), "jump")
    with pytest.raises(ValueError):
        model.step(np.zeros(STATE_DIM + 2, dtype=np.float32), "left")
    with pytest.raises(ValueError):
        model.step(np.zeros(STATE_DIM, dtype=np.float32), np.zeros(5, dtype=np.float32))


def test_mcts_uses_dynamics_model_reward(tmp_path):
    model = NeuralDynamicsWorldModel.from_checkpoint(_trained_checkpoint(tmp_path))
    engine = MctsEngine(num_simulations=16, max_depth=3, dynamics_model=model)
    root = np.zeros(STATE_DIM, dtype=np.float32)
    result = engine.plan(root, ACTIONS, transition_fn=None, legal_actions_fn=lambda s: ACTIONS)
    assert result["nodes_expanded"] > 0
    assert result["value_source"] == "neural_dynamics"
    assert result["best_action"] in ACTIONS
    # The random-rollout generator must not be consumed when a dynamics model drives evaluation.
    engine_ref = MctsEngine(num_simulations=16, max_depth=3, dynamics_model=model)
    engine_ref.plan(root, ACTIONS, transition_fn=None, legal_actions_fn=lambda s: ACTIONS)
    assert engine.rng.uniform() == engine_ref.rng.uniform() == np.random.RandomState(42).uniform()


def test_mcts_unloaded_dynamics_model_fails_closed():
    engine = MctsEngine(num_simulations=4, dynamics_model=_model())
    with pytest.raises(RuntimeError, match="fail closed"):
        engine.plan(np.zeros(STATE_DIM, dtype=np.float32), ACTIONS, transition_fn=None, legal_actions_fn=lambda s: ACTIONS)


def test_genzero_client_mounts_checkpoint(tmp_path):
    from gen_zero.client import GenZero
    from gen_zero.config import GenZeroConfig

    path = _trained_checkpoint(tmp_path)
    gz = GenZero(GenZeroConfig(neural_dynamics_checkpoint=str(path)))
    assert isinstance(gz.neural_dynamics_model, NeuralDynamicsWorldModel)
    assert gz.mcts_engine.dynamics_model is gz.neural_dynamics_model

    with pytest.raises(FileNotFoundError):
        GenZero(GenZeroConfig(neural_dynamics_checkpoint=str(tmp_path / "nope.pt")))


def test_training_script_requires_opt_in_for_synthetic(tmp_path):
    script = _load_train_script()
    with pytest.raises(FileNotFoundError, match="--allow-synthetic"):
        script.main(["--data", str(tmp_path / "absent.npz"), "--checkpoint", str(tmp_path / "m.pt"),
                     "--report", str(tmp_path / "r.json")])


def test_training_script_end_to_end_synthetic(tmp_path):
    script = _load_train_script()
    ckpt = tmp_path / "wm.pt"
    report_path = tmp_path / "report.json"
    rc = script.main([
        "--data", str(tmp_path / "absent.npz"), "--allow-synthetic",
        "--checkpoint", str(ckpt), "--report", str(report_path),
        "--epochs", "15", "--synthetic-samples", "800", "--state-dim", str(STATE_DIM),
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["data_source"] == "synthetic"
    assert report["split_mode"] == "random_rows"
    assert report["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert report["num_train"] == 640 and report["num_val"] == 160
    for key in ("final_train_loss", "val_mse", "val_bce", "val_auc", "wall_time_s"):
        assert np.isfinite(report[key]), key
    model = NeuralDynamicsWorldModel.from_checkpoint(ckpt)
    nxt, r, done = model.step(np.zeros(STATE_DIM, dtype=np.float32), model.action_vocab[0])
    assert nxt.shape == (STATE_DIM,)


def test_training_script_reads_npz(tmp_path):
    script = _load_train_script()
    data = _with_episodes(make_synthetic_transitions(num_samples=300, state_dim=4, action_vocab=["a", "b"], seed=3))
    npz = tmp_path / "traj.npz"
    data.save_npz(npz)
    loaded = TransitionDataset.from_npz(npz)
    np.testing.assert_array_equal(loaded.states, data.states)
    np.testing.assert_array_equal(loaded.episode_ids, data.episode_ids)
    assert loaded.action_vocab == ["a", "b"]
    report_path = tmp_path / "r.json"
    rc = script.main([
        "--data", str(npz), "--checkpoint", str(tmp_path / "m.pt"),
        "--report", str(report_path), "--epochs", "3",
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["data_source"] == "file"
    assert report["data_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert report["split_mode"] == "grouped_by_episode"
    assert report["train_episodes"] + report["val_episodes"] == 60
    assert report["val_episodes"] == 12 == len(report["val_episode_ids"])
    val_rows = np.isin(data.episode_ids, report["val_episode_ids"])
    assert report["num_val"] == int(val_rows.sum())


def test_training_script_rejects_file_without_episode_ids(tmp_path):
    script = _load_train_script()
    npz = tmp_path / "traj.npz"
    make_synthetic_transitions(num_samples=100, state_dim=4, action_vocab=["a", "b"], seed=3).save_npz(npz)
    with pytest.raises(KeyError, match="episode_ids"):
        script.main(["--data", str(npz), "--checkpoint", str(tmp_path / "m.pt"),
                     "--report", str(tmp_path / "r.json"), "--epochs", "1"])


def _with_episodes(data, seed=0, n_episodes=60):
    """Attach contiguous episodes of random length (rows 0.. in step order)."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, len(data)), size=n_episodes - 1, replace=False))
    ids = np.zeros(len(data), dtype=np.int64)
    for c in cuts:
        ids[c:] += 1
    return TransitionDataset(data.states, data.actions, data.next_states, data.rewards, data.action_vocab, ids)


def test_grouped_split_keeps_whole_episodes_in_order():
    ds = _with_episodes(make_synthetic_transitions(num_samples=500, state_dim=4, action_vocab=["a", "b"], seed=4))
    train_idx, val_idx = ds.split_indices(0.2, seed=7)
    assert len(train_idx) + len(val_idx) == len(ds)
    assert np.intersect1d(train_idx, val_idx).size == 0
    train_eps, val_eps = set(ds.episode_ids[train_idx]), set(ds.episode_ids[val_idx])
    assert train_eps.isdisjoint(val_eps)
    assert len(val_eps) == 12 and len(train_eps) == 48
    for eid in val_eps:
        rows = np.flatnonzero(ds.episode_ids == eid)
        np.testing.assert_array_equal(val_idx[np.isin(val_idx, rows)], rows)
    train, val = ds.split(0.2, seed=7)
    np.testing.assert_array_equal(val.states, ds.states[val_idx])
    np.testing.assert_array_equal(val.episode_ids, ds.episode_ids[val_idx])
    assert len(train) == len(train_idx)


def test_grouped_split_is_deterministic_and_seed_dependent():
    ds = _with_episodes(make_synthetic_transitions(num_samples=500, state_dim=4, action_vocab=["a", "b"], seed=4))
    a, b, c = (ds.split_indices(0.2, seed=s)[1] for s in (1, 1, 2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grouped_split_without_episode_ids_raises():
    ds = make_synthetic_transitions(num_samples=100, state_dim=4, action_vocab=["a", "b"], seed=4)
    with pytest.raises(ValueError, match="episode_ids"):
        ds.split(0.2, seed=0)
    train, val = ds.split(0.2, seed=0, group_by_episode=False)
    assert (len(train), len(val)) == (80, 20)


def test_episode_ids_are_validated():
    ds = make_synthetic_transitions(num_samples=10, state_dim=4, action_vocab=["a", "b"], seed=4)
    with pytest.raises(ValueError, match="episode_ids"):
        TransitionDataset(ds.states, ds.actions, ds.next_states, ds.rewards, None, np.zeros(9, dtype=np.int64))
    with pytest.raises(ValueError, match="episode_ids"):
        TransitionDataset(ds.states, ds.actions, ds.next_states, ds.rewards, None, np.zeros(10, dtype=np.float32))
