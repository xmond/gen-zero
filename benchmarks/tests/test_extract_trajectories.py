"""Tests for scripts/extract_trajectories.py (benchmarks/suites/deadlock_torus_env.py).

Covers: state/action shapes and dtypes, the {0,1} reward alphabet, determinism
under a fixed seed, and -- the load-bearing check -- that the "doomed" (trap)
classifier used for reward/traps labeling is not decorative: a hand-built 6x6
trap layout must have every state inside its dead-end corridor marked non-viable
(doomed) and the cell just outside the corridor marked viable.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import deadlock_torus_env as dte  # noqa: E402
import extract_trajectories as et  # noqa: E402


def _small_extract(**overrides):
    kwargs = dict(n_torus=8, n_traps=6, n_regular=6, episodes_per_layout=4,
                  target_min=200, target_max=300, seed=7)
    kwargs.update(overrides)
    return et.extract(**kwargs)


def test_shapes_and_dtypes():
    data = _small_extract()
    n = data["states"].shape[0]
    assert n > 0
    assert data["states"].shape == (n, 64)
    assert data["next_states"].shape == (n, 64)
    assert data["actions"].shape == (n, 16)
    assert data["rewards"].shape == (n,)
    assert data["episode_ids"].shape == (n,)
    assert data["traps"].shape == (n,)
    assert data["states"].dtype == np.float32
    assert data["actions"].dtype == np.float32
    assert data["rewards"].dtype == np.float32


def test_reward_alphabet_and_trap_subset_of_zero_reward():
    data = _small_extract()
    assert set(np.unique(data["rewards"]).tolist()) <= {0.0, 1.0}
    assert set(np.unique(data["traps"]).tolist()) <= {0, 1}
    # every trap-flagged transition must carry reward 0.0 (doomed implies reward 0)
    trap_rows = data["traps"] == 1
    assert np.all(data["rewards"][trap_rows] == 0.0)


def test_multiple_episodes_present():
    data = _small_extract()
    n_episodes = len(set(data["episode_ids"].tolist()))
    assert n_episodes > 1


def test_determinism_same_seed():
    a = _small_extract(seed=42)
    b = _small_extract(seed=42)
    np.testing.assert_array_equal(a["states"], b["states"])
    np.testing.assert_array_equal(a["actions"], b["actions"])
    np.testing.assert_array_equal(a["rewards"], b["rewards"])
    np.testing.assert_array_equal(a["traps"], b["traps"])


def test_action_embeddings_are_distinct_and_fixed_dim():
    emb = et.action_embeddings()
    assert emb.shape == (dte.N_ACTIONS, et.ACTION_DIM)
    # the 3 action codes must be linearly distinguishable (not collapsed)
    for i in range(dte.N_ACTIONS):
        for j in range(i + 1, dte.N_ACTIONS):
            assert not np.allclose(emb[i], emb[j])


def test_doomed_classifier_on_validated_trap_layout():
    """The doomed (irreversible-trap) classification is not decorative.

    `make_trap` only ever returns a layout after checking, via `world.viable`,
    that its start is alive-and-viable while the deceptive greedy action (1 =
    straight) leads to a state that is alive but NOT viable (doomed). This test
    re-derives that same invariant independently to confirm `viable()` really
    draws the line where the trap's own construction says it must: reachable
    before the corridor mouth, unreachable-to-food one step past it.
    """
    world = dte.TorusWorld(n=10)
    lay = None
    for seed in range(500):
        rng = np.random.default_rng(seed)
        depth = int(rng.integers(1, world.n - 4))
        row = int(rng.integers(2, world.n - 4))
        k_rot = int(rng.integers(0, 4))
        lay = dte.make_trap(world, depth, row, k_rot, rng)
        if lay is not None:
            break
    assert lay is not None, "failed to construct any valid trap layout for the test"

    viable = world.viable(lay)
    assert viable[lay.start], "trap start must remain viable (a safe route exists)"

    s2, outcome = world.step(lay, lay.start, 1)  # 1 = straight, the deceptive action
    assert outcome == "alive", "the deceptive step must not be instant death"
    assert not viable[s2], "the state one step into the corridor must be doomed"

    # the deception itself: greedy one-step reward picks the doomed action
    rew = world.one_step_reward(lay, lay.start)
    assert int(np.argmax(rew)) == 1
    # yet no optimal (shortest safe path) action is the doomed one
    assert 1 not in world.optimal_actions(lay, lay.start)


def test_doomed_means_inevitable_death_independent_of_viable():
    """Brute-force check that does not call `viable()`: from the state one step
    into the trap, EVERY possible action sequence up to the corridor's own
    length dies before it can escape. This is the actual ground truth "doomed"
    is supposed to represent; `viable()` is only checked against it, not
    assumed correct because `viable()` says so.
    """
    world = dte.TorusWorld(n=10)
    lay = None
    for seed in range(500):
        rng = np.random.default_rng(seed)
        depth = int(rng.integers(1, world.n - 4))
        row = int(rng.integers(2, world.n - 4))
        k_rot = int(rng.integers(0, 4))
        lay = dte.make_trap(world, depth, row, k_rot, rng)
        if lay is not None:
            break
    assert lay is not None

    s2, outcome = world.step(lay, lay.start, 1)
    assert outcome == "alive"

    horizon = lay.depth + 3
    for seq in itertools.product(range(dte.N_ACTIONS), repeat=horizon):
        s = s2
        died = False
        for a in seq:
            s, st = world.step(lay, s, a)
            if st == "dead":
                died = True
                break
            if st == "goal":
                pytest.fail(f"escaped the trap via {seq[:len(seq)]} and reached food -- not doomed")
        assert died, f"action sequence {seq} neither died nor escaped within horizon={horizon}"

    # positive control: the safe route from the true start really does reach food
    dist = world.goal_distance(lay)
    s = lay.start
    for _ in range(world.S):
        if world.state_masks(lay)[1][s]:
            break
        opt = world.optimal_actions(lay, s, dist)
        assert opt, "start was claimed viable but has no optimal action"
        s, _ = world.step(lay, s, opt[0])
    else:
        pytest.fail("optimal-action rollout never reached the goal within world.S steps")
