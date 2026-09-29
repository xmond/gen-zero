"""No-reverse torus gridworld with deceptive irreversible dead ends.

State s = (row, col, heading) on an n x n torus; headings N, E, S, W.
Actions are relative: 0 = turn left, 1 = straight, 2 = turn right; the agent
turns, then moves one cell. It can never reverse, so a width-1 corridor with a
closed end is an irreversible trap: once inside, every path dies.

Each action is a bijection of the 4*n*n states (a shift composed with a
heading rotation), so it is exactly a permutation matrix on orthonormal state
codes. That is why a learned rotation exp(Omega_a) can model it. Row 0 and
column 0 are lethal, so the torus behaves like a walled box.

Ground truth (viability, shortest paths) comes from exact graph search here.
It is used only to label test items and to score outcomes; that search never
enters a planner. The map itself (walls, food) is the observation every arm
receives: Laya reads it as ASCII text, Gen-Zero as linear readouts.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PKG = Path(__file__).resolve().parents[2] / "python"
if _PKG.is_dir() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from gen_zero.world_model.torus_codec import latent_codes  # noqa: E402,F401  (re-exported)

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))          # N E S W
HEAD_NAMES = ("north", "east", "south", "west")
ACTION_NAMES = ("left", "straight", "right")
N_ACTIONS = 3
FEATURE_DIM = 16


@dataclass
class Layout:
    n: int
    lethal: np.ndarray          # (n, n) bool
    food: Tuple[int, int]
    start: int                  # state index
    kind: str                   # "trap" | "regular"
    depth: int = 0              # corridor depth for traps
    name: str = ""


class TorusWorld:
    def __init__(self, n: int = 10) -> None:
        if n < 6:
            raise ValueError("n must be at least 6")
        self.n = n
        self.S = 4 * n * n
        nxt = np.empty((self.S, N_ACTIONS), np.int64)
        for s in range(self.S):
            r, c, h = self.decode(s)
            for a in range(N_ACTIONS):
                h2 = (h + a - 1) % 4
                dr, dc = DIRS[h2]
                nxt[s, a] = self.encode((r + dr) % n, (c + dc) % n, h2)
        for a in range(N_ACTIONS):
            if len(np.unique(nxt[:, a])) != self.S:
                raise AssertionError("action is not a permutation")
        self.next = nxt

    def encode(self, r: int, c: int, h: int) -> int:
        return (r * self.n + c) * 4 + h

    def decode(self, s: int) -> Tuple[int, int, int]:
        cell, h = divmod(int(s), 4)
        r, c = divmod(cell, self.n)
        return r, c, h

    # ------------------------------------------------------------------ ground truth
    def state_masks(self, lay: Layout) -> Tuple[np.ndarray, np.ndarray]:
        cells = np.arange(self.S) // 4
        lethal = lay.lethal.ravel()[cells]
        goal = cells == lay.food[0] * self.n + lay.food[1]
        return lethal, goal

    def viable(self, lay: Layout) -> np.ndarray:
        """Greatest fixed point: states from which the agent can live forever or reach food."""
        lethal, goal = self.state_masks(lay)
        ok = ~lethal
        while True:
            new = ok & (goal | ok[self.next].any(axis=1))
            if (new == ok).all():
                return ok
            ok = new

    def goal_distance(self, lay: Layout) -> np.ndarray:
        """Shortest safe path length to the food for every state (inf if none)."""
        lethal, goal = self.state_masks(lay)
        dist = np.full(self.S, np.inf)
        prev: Dict[int, List[int]] = {}
        for s in range(self.S):
            for a in range(N_ACTIONS):
                prev.setdefault(int(self.next[s, a]), []).append(s)
        dq = deque()
        for s in np.flatnonzero(goal):
            dist[s] = 0
            dq.append(int(s))
        while dq:
            t = dq.popleft()
            for s in prev.get(t, ()):
                if not lethal[s] and not goal[s] and dist[s] == np.inf:
                    dist[s] = dist[t] + 1
                    dq.append(s)
        return dist

    def optimal_actions(self, lay: Layout, s: int, dist: Optional[np.ndarray] = None) -> List[int]:
        dist = self.goal_distance(lay) if dist is None else dist
        lethal, _ = self.state_masks(lay)
        ds = [dist[self.next[s, a]] if not lethal[self.next[s, a]] else np.inf for a in range(N_ACTIONS)]
        best = min(ds)
        return [a for a in range(N_ACTIONS) if ds[a] == best and np.isfinite(best)]

    def step(self, lay: Layout, s: int, a: int) -> Tuple[int, str]:
        s2 = int(self.next[s, a])
        r, c, _ = self.decode(s2)
        if lay.lethal[r, c]:
            return s2, "dead"
        if (r, c) == lay.food:
            return s2, "goal"
        return s2, "alive"

    # ------------------------------------------------------------------ observation features
    @staticmethod
    def delta(a: int, b: int) -> int:
        """Plain signed offset. The lethal ring makes the torus wrap impassable,
        so wrap-around distance would point through a wall."""
        return b - a

    def state_features(self, lay: Layout) -> np.ndarray:
        """(S, FEATURE_DIM) local, map-derived features. No look-ahead beyond 2 steps."""
        n = self.n
        L = lay.lethal
        fr, fc = lay.food
        lethal_s, goal_s = self.state_masks(lay)
        safe1 = (~lethal_s[self.next]).sum(axis=1)                        # safe successors
        safe2 = np.zeros(self.S)
        for a in range(N_ACTIONS):
            s1 = self.next[:, a]
            safe2 += np.where(lethal_s[s1], 0, (~lethal_s[self.next[s1]]).sum(axis=1))
        maxd = float(2 * n)  # Manhattan distance inside the box < 2n
        G = np.zeros((self.S, FEATURE_DIM), np.float32)
        for s in range(self.S):
            r, c, h = self.decode(s)
            dr, dc = self.delta(r, fr), self.delta(c, fc)
            dist = abs(dr) + abs(dc)
            hr, hc = DIRS[h]
            cos = (hr * dr + hc * dc) / dist if dist else 1.0
            nb4 = sum(L[(r + a) % n, (c + b) % n] for a, b in DIRS)
            nb8 = nb4 + sum(L[(r + a) % n, (c + b) % n] for a in (-1, 1) for b in (-1, 1))
            ring2 = sum(L[(r + a) % n, (c + b) % n] for a in range(-2, 3) for b in range(-2, 3)
                        if abs(a) + abs(b) == 2)
            ahead = 0.0
            for k in range(1, 4):
                rr, cc = (r + hr * k) % n, (c + hc * k) % n
                if L[rr, cc]:
                    break
                if (rr, cc) == (fr, fc):
                    ahead = 1.0
                    break
            G[s, 0] = float(lethal_s[s])
            G[s, 1] = float(goal_s[s])
            G[s, 2] = dist / maxd
            G[s, 3] = dr / maxd
            G[s, 4] = dc / maxd
            G[s, 5] = cos
            G[s, 6 + h] = 1.0
            G[s, 10] = safe1[s] / 3.0
            G[s, 11] = safe2[s] / 9.0
            G[s, 12] = nb4 / 4.0
            G[s, 13] = nb8 / 8.0
            G[s, 14] = ring2 / 8.0
            G[s, 15] = ahead
        return G

    def one_step_reward(self, lay: Layout, s: int) -> np.ndarray:
        """Immediate signal only: +1 food, -1 death, else -0.1 * change in Manhattan distance."""
        fr, fc = lay.food
        r, c, _ = self.decode(s)
        d0 = abs(self.delta(r, fr)) + abs(self.delta(c, fc))
        out = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            s2, st = self.step(lay, s, a)
            if st == "dead":
                out[a] = -1.0
            elif st == "goal":
                out[a] = 1.0
            else:
                r2, c2, _ = self.decode(s2)
                out[a] = -0.1 * (abs(self.delta(r2, fr)) + abs(self.delta(c2, fc)) - d0)
        return out

    # ------------------------------------------------------------------ text (for Laya)
    def render_text(self, lay: Layout, s: int) -> Tuple[str, Dict[str, str]]:
        r, c, h = self.decode(s)
        arrow = "^>v<"[h]
        rows = []
        for i in range(self.n):
            line = []
            for j in range(self.n):
                if (i, j) == (r, c):
                    line.append(arrow)
                elif (i, j) == lay.food:
                    line.append("F")
                elif lay.lethal[i, j]:
                    line.append("#")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        state = ("Grid world, row 0 at the top. Legend: '#' is a deadly wall, '.' is free floor, "
                 "'F' is the food, and the agent is the arrow '" + arrow + "' facing " + HEAD_NAMES[h] +
                 f", at row {r}, column {c}. Each turn the agent turns (or not) and then moves exactly one "
                 "cell. It can never move backwards. Stepping onto '#' kills it instantly. "
                 "The goal is to reach F alive.\n" + "\n".join(rows))
        criteria = {}
        for a, name in enumerate(ACTION_NAMES):
            h2 = (h + a - 1) % 4
            dr, dc = DIRS[h2]
            rr, cc = (r + dr) % self.n, (c + dc) % self.n
            what = "a deadly wall" if lay.lethal[rr, cc] else ("the food" if (rr, cc) == lay.food else "free floor")
            verb = "keep going straight" if a == 1 else f"turn {name}"
            criteria[name] = f"{verb}: move {HEAD_NAMES[h2]} to row {rr}, column {cc} ({what})"
        return state, criteria


# --------------------------------------------------------------------------- layouts

def _ring(n: int) -> np.ndarray:
    L = np.zeros((n, n), bool)
    L[0, :] = True
    L[:, 0] = True
    return L


def _rotate(world: TorusWorld, interior: np.ndarray, pos: Tuple[int, int], head: int,
            food: Tuple[int, int], k: int) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """Rotate the (n-1)x(n-1) interior CCW k times; interior (i, j) -> torus (i+1, j+1)."""
    m = interior.shape[0]

    def rot(p):
        i, j = p
        for _ in range(k):
            i, j = m - 1 - j, i
        return i, j

    inner = np.rot90(interior, k)
    L = _ring(world.n)
    L[1:, 1:] = inner
    pi, pj = rot(pos)
    fi, fj = rot(food)
    return L, world.encode(pi + 1, pj + 1, (head - k) % 4), (fi + 1, fj + 1)


def make_trap(world: TorusWorld, depth: int, row: int, k_rot: int, rng: np.random.Generator,
              clutter: int = 4, name: str = "") -> Optional[Layout]:
    """Deceptive dead end: straight toward the food enters a closed width-1 corridor."""
    m = world.n - 1
    if not (1 <= depth <= m - 3) or not (2 <= row <= m - 3):
        return None
    I = np.zeros((m, m), bool)
    for j in range(1, depth + 1):
        I[row - 1, j] = I[row + 1, j] = True
    I[row, depth + 1] = True
    food = (row, depth + 2)
    protected = {(row, j) for j in range(0, depth + 3)}
    protected |= {(row + a, b) for a in (-2, -1, 1, 2) for b in (0, 1)}
    cand = [(i, j) for i in range(m) for j in range(m) if not I[i, j] and (i, j) not in protected]
    for idx in rng.permutation(len(cand))[:clutter]:
        I[cand[idx]] = True
    L, start, fpos = _rotate(world, I, (row, 0), 1, food, k_rot)
    lay = Layout(world.n, L, fpos, start, "trap", depth, name)
    via = world.viable(lay)
    enter, st = world.step(lay, start, 1)
    opt = world.optimal_actions(lay, start)
    # Valid trap: start is viable, straight is alive now but doomed, a safe route exists
    # and never starts with straight, and straight is the one-step greedy choice.
    if (not via[start] or st != "alive" or via[enter] or not opt or 1 in opt
            or int(np.argmax(world.one_step_reward(lay, start))) != 1):
        return None
    return lay


def make_regular(world: TorusWorld, rng: np.random.Generator, density: float = 0.12,
                 name: str = "") -> Optional[Layout]:
    """Random clutter; keep states where single-step information suffices.

    Regular means: a safe route to the food exists, and every one-step-greedy
    action is on a shortest safe route (no deception) and leads to a viable state.
    """
    n = world.n
    L = _ring(n)
    L[1:, 1:] = rng.random((n - 1, n - 1)) < density
    free = [(i, j) for i in range(1, n) for j in range(1, n) if not L[i, j]]
    if len(free) < 10:
        return None
    fi = free[rng.integers(len(free))]
    lay = Layout(n, L, fi, 0, "regular", 0, name)
    via = world.viable(lay)
    dist = world.goal_distance(lay)
    lethal_s, goal_s = world.state_masks(lay)
    starts = [s for s in range(world.S) if via[s] and not goal_s[s] and 3 <= dist[s] <= 9]
    rng.shuffle(starts)
    for s in starts[:20]:
        opt = world.optimal_actions(lay, s, dist)
        rew = world.one_step_reward(lay, s)
        greedy = np.flatnonzero(rew == rew.max())
        if not opt or len(opt) == N_ACTIONS:
            continue
        if all(g in opt and via[world.next[s, g]] for g in greedy):
            lay.start = int(s)
            return lay
    return None


# ---------------------------------------------------------------------------
# Directed torus mazes with deadlock branch (for MCTS ablation benchmark)
# ---------------------------------------------------------------------------

from random import Random

ACTIONS = ("north", "east", "south", "west")


@dataclass(frozen=True)
class DeadlockTorusEnv:
    seed: int
    size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    transitions: dict[tuple[int, int], dict[str, tuple[int, int]]]
    traps: frozenset[tuple[int, int]]

    @classmethod
    def generate(cls, seed: int, size: int = 9) -> "DeadlockTorusEnv":
        if size < 7:
            raise ValueError("size must be >= 7")
        rng = Random(seed)
        y = rng.randrange(size)
        x = rng.randrange(size)
        distance = rng.randrange(2, min(5, (size + 1) // 2))
        start = (x, y)
        goal = ((x + distance) % size, y)
        transitions = {}
        for xx in range(size):
            for yy in range(size):
                transitions[(xx, yy)] = {
                    "north": (xx, (yy - 1) % size), "east": ((xx + 1) % size, yy),
                    "south": (xx, (yy + 1) % size), "west": ((xx - 1) % size, yy),
                }
        chamber = ((x + 1) % size, y)
        transitions[start]["east"] = chamber
        transitions[chamber] = {action: chamber for action in ACTIONS}
        for state, edges in transitions.items():
            if state != start and state != chamber:
                for action, dest in list(edges.items()):
                    if dest == chamber:
                        edges[action] = state
        return cls(seed, size, start, goal, transitions, frozenset({chamber}))

    def step(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        return self.transitions[state][action]

    def distance(self, state: tuple[int, int]) -> int:
        x, y = state
        gx, gy = self.goal
        return min((x - gx) % self.size, (gx - x) % self.size) + min(
            (y - gy) % self.size, (gy - y) % self.size)
