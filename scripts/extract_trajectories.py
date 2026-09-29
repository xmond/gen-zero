"""Extract (s, a, s', r) transitions from the deadlock-torus environment.

Provenance note: benchmarks/suites/deadlock_torus_env.py was NOT present in this
worktree, on any git branch, or in git history (verified with `git log --all` and
`git ls-tree -r` across every branch). It existed only as an *untracked* file in
the main gen-zero checkout. Its API (TorusWorld, no-reverse left/straight/
right actions, irreversible dead-end traps via `viable()`, `make_trap`/
`make_regular` layouts) matches this task's spec exactly, so it was copied
verbatim into this worktree and is now tracked here for the first time. A second,
different, also-untracked draft of the same filename exists in the sibling
worktree feat/b0926c-t3-bench; that one was NOT used, since its API (plain (x, y)
coordinates, no heading, single fixed absorbing chamber) does not match the
"gridworld + heading + local wall-distance features" spec below. Reconcile this
divergence before the b0926c branches merge.

State s_t (64-D), built from TorusWorld/Layout ground truth, is continuous and
deterministic given (row, col, heading, layout):
  [0:4]   torus angle encoding: sin/cos of 2*pi*row/n, 2*pi*col/n
  [4:6]   row/n, col/n (linear position)
  [6:10]  heading one-hot (N, E, S, W)
  [10:12] heading unit vector (dr, dc)
  [12:16] wall ray-cast distance, absolute cardinal directions N/E/S/W (normalized, capped at n/2)
  [16:20] wall ray-cast distance, heading-relative directions front/right/back/left
  [20:44] local 5x5 lethal-wall patch centered on the agent, center cell excluded (24 cells)
  [44:48] food-relative: dx/maxd, dy/maxd, dist/maxd, cos(heading, direction-to-food)
  [48:50] safe-successor density at 1 and 2 steps (from TorusWorld.state_features)
  [50:53] neighbor wall density: 4-neighborhood, 8-neighborhood, distance-2 ring
  [53:57] lookahead: ahead-clear-to-food flag, plus blocked-in-1/2/3-steps flags
  [57:59] status flags: is-lethal, is-goal (both 0 for the states we ever emit as s_t)
  [59:64] bias term (1.0) + 4 zero-reserved slots

Action a_t (16-D) is a fixed, seeded orthonormal projection of the one-hot action
index (0=left, 1=straight, 2=right), reusing the env's own `latent_codes` helper
so the projection is deterministic and reproducible.

Reward r_t in {0.0, 1.0}: 0.0 if s_{t+1} is dead (collision with a lethal wall) or
"doomed" (alive but TorusWorld.viable(layout)[s_{t+1}] is False -- i.e. the step
just entered an irreversible dead-end trap with no path back to food); 1.0 for
every other transition, including the one that reaches food.

traps[i] = 1 iff the transition at row i entered a doomed-but-alive state (the
irreversible-trap case above); 0 otherwise (covers both safe steps and outright
death-by-wall-collision, which is a different failure mode from a trap).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "suites"))
import deadlock_torus_env as dte  # noqa: E402

STATE_DIM = 64
ACTION_DIM = 16
ACTION_PROJECTION_SEED = 20260926
MAX_STEPS_PER_EPISODE = 30


def action_embeddings() -> np.ndarray:
    """(3, 16) fixed orthonormal one-hot projection: row a is the embedding for action a."""
    return dte.latent_codes(n_states=dte.N_ACTIONS, dim=ACTION_DIM, seed=ACTION_PROJECTION_SEED)


def _ray_cast(lethal: np.ndarray, r: int, c: int, dr: int, dc: int, n: int, max_range: int) -> float:
    for k in range(1, max_range + 1):
        rr, cc = (r + dr * k) % n, (c + dc * k) % n
        if lethal[rr, cc]:
            return k / max_range
    return 1.0


def encode_state(world: "dte.TorusWorld", lay: "dte.Layout", s: int) -> np.ndarray:
    n = world.n
    L = lay.lethal
    r, c, h = world.decode(s)
    fr, fc = lay.food
    max_range = max(n // 2, 1)
    v = np.zeros(STATE_DIM, np.float32)

    v[0] = np.sin(2 * np.pi * r / n)
    v[1] = np.cos(2 * np.pi * r / n)
    v[2] = np.sin(2 * np.pi * c / n)
    v[3] = np.cos(2 * np.pi * c / n)
    v[4] = r / n
    v[5] = c / n
    v[6 + h] = 1.0
    hr, hc = dte.DIRS[h]
    v[10] = hr
    v[11] = hc

    for i, (dr, dc) in enumerate(dte.DIRS):  # N E S W absolute
        v[12 + i] = _ray_cast(L, r, c, dr, dc, n, max_range)
    for i in range(4):  # front, right, back, left relative to heading
        dr, dc = dte.DIRS[(h + i) % 4]
        v[16 + i] = _ray_cast(L, r, c, dr, dc, n, max_range)

    idx = 20
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if di == 0 and dj == 0:
                continue
            v[idx] = float(L[(r + di) % n, (c + dj) % n])
            idx += 1

    maxd = float(2 * n)
    dr_food, dc_food = dte.TorusWorld.delta(r, fr), dte.TorusWorld.delta(c, fc)
    dist = abs(dr_food) + abs(dc_food)
    v[44] = dr_food / maxd
    v[45] = dc_food / maxd
    v[46] = dist / maxd
    v[47] = (hr * dr_food + hc * dc_food) / dist if dist else 1.0

    lethal_s, goal_s = world.state_masks(lay)
    safe1 = float((~lethal_s[world.next[s]]).sum())
    safe2 = 0.0
    for a in range(dte.N_ACTIONS):
        s1 = int(world.next[s, a])
        if not lethal_s[s1]:
            safe2 += float((~lethal_s[world.next[s1]]).sum())
    v[48] = safe1 / 3.0
    v[49] = safe2 / 9.0

    nb4 = sum(L[(r + a) % n, (c + b) % n] for a, b in dte.DIRS)
    nb8 = nb4 + sum(L[(r + a) % n, (c + b) % n] for a in (-1, 1) for b in (-1, 1))
    ring2 = sum(L[(r + a) % n, (c + b) % n] for a in range(-2, 3) for b in range(-2, 3)
                if abs(a) + abs(b) == 2)
    v[50] = nb4 / 4.0
    v[51] = nb8 / 8.0
    v[52] = ring2 / 8.0

    ahead = 0.0
    blocked = [0.0, 0.0, 0.0]
    for k in range(1, 4):
        rr, cc = (r + hr * k) % n, (c + hc * k) % n
        if L[rr, cc]:
            blocked[k - 1] = 1.0
            break
        if (rr, cc) == (fr, fc):
            ahead = 1.0
            break
    v[53] = ahead
    v[54], v[55], v[56] = blocked

    v[57] = float(lethal_s[s])
    v[58] = float(goal_s[s])
    v[59] = 1.0
    return v


def make_layouts(world: "dte.TorusWorld", n_traps: int, n_regular: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    layouts = []
    n_trap_ok = 0
    attempts = 0
    while n_trap_ok < n_traps and attempts < n_traps * 60:
        attempts += 1
        depth = int(rng.integers(1, world.n - 4))
        row = int(rng.integers(2, world.n - 4))
        k_rot = int(rng.integers(0, 4))
        lay_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        lay = dte.make_trap(world, depth, row, k_rot, lay_rng, name=f"trap_{n_trap_ok}")
        if lay is not None:
            layouts.append(lay)
            n_trap_ok += 1
    n_regular_ok = 0
    attempts = 0
    while n_regular_ok < n_regular and attempts < n_regular * 60:
        attempts += 1
        lay_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        lay = dte.make_regular(world, lay_rng, name=f"regular_{n_regular_ok}")
        if lay is not None:
            layouts.append(lay)
            n_regular_ok += 1

    depths = sorted(lay.depth for lay in layouts if lay.kind == "trap")
    print(f"make_layouts: requested traps={n_traps} regular={n_regular}; "
          f"got traps={n_trap_ok} regular={n_regular_ok}; trap depths={depths}")
    if n_trap_ok < n_traps * 0.5 or n_regular_ok < n_regular * 0.5:
        raise RuntimeError(
            f"layout generation degraded badly: traps {n_trap_ok}/{n_traps}, "
            f"regular {n_regular_ok}/{n_regular} (fail-closed, not silently accepted)")
    return layouts


def choose_action(world: "dte.TorusWorld", lay: "dte.Layout", s: int, policy: str,
                   rng: np.random.Generator, dist: np.ndarray) -> int:
    if policy == "pure_random":
        return int(rng.integers(dte.N_ACTIONS))
    if policy == "greedy_deceived":
        rew = world.one_step_reward(lay, s)
        return int(np.argmax(rew))
    # epsilon_random: mostly optimal, occasionally random (covers wall bumps too)
    if rng.random() < 0.3:
        return int(rng.integers(dte.N_ACTIONS))
    opt = world.optimal_actions(lay, s, dist)
    return int(rng.choice(opt)) if opt else int(rng.integers(dte.N_ACTIONS))


def run_episode(world: "dte.TorusWorld", lay: "dte.Layout", policy: str,
                 rng: np.random.Generator, episode_id: int, action_emb: np.ndarray):
    viable = world.viable(lay)
    dist = world.goal_distance(lay)
    s = lay.start
    rows = []
    for _ in range(MAX_STEPS_PER_EPISODE):
        a = choose_action(world, lay, s, policy, rng, dist)
        s2, outcome = world.step(lay, s, a)
        doomed = (outcome == "alive") and (not viable[s2])
        reward = 0.0 if (outcome == "dead" or doomed) else 1.0
        rows.append((
            encode_state(world, lay, s),
            action_emb[a],
            encode_state(world, lay, s2),
            reward,
            episode_id,
            1 if doomed else 0,
        ))
        if outcome != "alive" or doomed:
            break
        s = s2
    return rows


def extract(n_torus: int, n_traps: int, n_regular: int, episodes_per_layout: int,
            target_min: int, target_max: int, seed: int):
    world = dte.TorusWorld(n=n_torus)
    action_emb = action_embeddings()
    layouts = make_layouts(world, n_traps, n_regular, seed)
    if not layouts:
        raise RuntimeError("no valid layouts generated")

    policies = ("greedy_deceived", "epsilon_random", "pure_random")
    states, actions, next_states, rewards, episode_ids, traps = [], [], [], [], [], []
    episode_id = 0
    rng = np.random.default_rng(seed + 1)
    round_robin = 0
    while len(states) < target_max:
        progressed = False
        for lay in layouts:
            if len(states) >= target_max:
                break
            for _ in range(episodes_per_layout):
                policy = policies[round_robin % len(policies)]
                round_robin += 1
                rows = run_episode(world, lay, policy, rng, episode_id, action_emb)
                if rows:
                    progressed = True
                for st, ac, st2, rw, eid, tr in rows:
                    states.append(st)
                    actions.append(ac)
                    next_states.append(st2)
                    rewards.append(rw)
                    episode_ids.append(eid)
                    traps.append(tr)
                episode_id += 1
                if len(states) >= target_max:
                    break
        if len(states) >= target_min:
            break
        if not progressed:
            raise RuntimeError("layouts stopped producing transitions before reaching target_min")

    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
        "rewards": np.asarray(rewards, np.float32),
        "episode_ids": np.asarray(episode_ids, np.int64),
        "traps": np.asarray(traps, np.int64),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/artifacts/zero/trajectories_v1.npz")
    ap.add_argument("--n-torus", type=int, default=10)
    ap.add_argument("--n-traps", type=int, default=45)
    ap.add_argument("--n-regular", type=int, default=45)
    ap.add_argument("--episodes-per-layout", type=int, default=6)
    ap.add_argument("--target-min", type=int, default=6000)
    ap.add_argument("--target-max", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = extract(args.n_torus, args.n_traps, args.n_regular, args.episodes_per_layout,
                    args.target_min, args.target_max, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)

    n = data["states"].shape[0]
    n_reward1 = int((data["rewards"] == 1.0).sum())
    n_trap = int((data["traps"] == 1).sum())
    n_dead = n - n_reward1 - n_trap
    n_episodes = len(set(data["episode_ids"].tolist()))
    print(f"wrote {out_path} : {n} transitions across {n_episodes} episodes")
    print(f"reward=1.0: {n_reward1} ({n_reward1 / n:.1%})  "
          f"trap(doomed): {n_trap} ({n_trap / n:.1%})  "
          f"dead(wall collision): {n_dead} ({n_dead / n:.1%})")
    for key in ("states", "actions", "next_states", "rewards", "episode_ids", "traps"):
        print(f"  {key}: shape={data[key].shape} dtype={data[key].dtype}")

    eids = data["episode_ids"]
    lengths = np.bincount(eids)
    lengths = lengths[lengths > 0]
    print(f"episode length: min={lengths.min()} median={int(np.median(lengths))} max={lengths.max()}")

    emb = action_embeddings()
    action_idx = np.argmax(data["actions"] @ emb.T, axis=1)
    counts = np.bincount(action_idx, minlength=dte.N_ACTIONS)
    print("action counts: " + ", ".join(
        f"{dte.ACTION_NAMES[i]}={int(counts[i])}" for i in range(dte.N_ACTIONS)))


if __name__ == "__main__":
    main()
