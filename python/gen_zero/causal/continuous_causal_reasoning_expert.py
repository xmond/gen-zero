"""Continuous causal reasoning expert: macro race + micro hard forcing + counterfactual judge.

Wires three independently-tested engines into one label-free per-candidate score for
`probe_mode == 'none'` multi-choice inference, replacing a bare `ar_loglik` readout with
a genuine, geometry-driven decision:

  macro   `BifurcatedFractalEngine.bifurcate` / `.race` (bifurcated_fractal_engine.py):
          for candidate k, race a momentum-seeded branch pool from the prompt state q0
          toward that candidate's terminal state c_k, with every *other* candidate's
          direction from q0 registered as a repulsor. A candidate whose direction is not
          robustly separable from a competitor gets pruned or costed down by the race
          itself -- this is where the "origin saddle point" (the degenerate q0 == c_k
          case that plain cosine similarity cannot break) gets resolved by geometry,
          not by a label.
  micro   `forced_affine_scan.hard_forced_step` (forced_affine_scan.py): the macro
          survivor state is refined by a few explicit affine steps toward c_k, each one
          hard-projected (KKT-exact, `P = I - G^+ G`) onto the affine space through q0
          whose displacement is orthogonal to every rejected candidate's direction.
          G/b are built purely from candidate
          geometry (the *other* answer choices), never from problem text: this repo's
          only text->algebra parser (`symbolic_arithmetic_verifier.py`) needs
          pre-structured `EquationStep`s, and building those from free text would mean
          regex standing in for reasoning, which is out of scope here. An empty
          condition set (a single remaining candidate, or repulsion disabled) is the
          identity projector by `ConditionSet`'s own contract: no constraint, no rows.
  judge   `CounterfactualConstraintVerifier.repulsion` (counterfactual_constraint_verifier.py):
          only `repulsion()` is used, not `verify()` -- `verify()` is a fixed 3-vertex
          (entailment/neutral/contradiction) NLI simplex and cannot emit a K-ary
          candidate distribution. `repulsion(premise, hypothesis, cf_premise,
          cf_hypothesis)` needs a counterfactual premise that differs from the real
          premise (P0 == P gives exactly zero evidence, by the verifier's own
          docstring); the domain prototype (the label-free mean of other same-domain
          records, excluding the current one -- the same quantity `analyze()`'s
          `competence_gate` already computes) is reused as that P0. The candidate's
          micro-refined state is the hypothesis; the mean of the *other* candidates'
          micro-refined states is the counterfactual hypothesis. The resulting
          `negation_alignment` is a penalty, not a vote: it can only pull a score down.

Honesty notes:
  * `BifurcatedFractalEngine.think()` is not called. Its own docstring: an untrained
    adapter's micro fixed point "has nothing to do with c". This module constructs a
    `ParallelRNNLoRAAdapter` only to satisfy the engine's constructor (which needs one
    to type-check and to report a spectral radius), and calls `.bifurcate()`/`.race()`
    directly -- never `.micro()`. The real micro layer is `hard_forced_step`, which is
    parameter-free (a KKT projection, not a learned map) and needs no training.
  * Everything below operates in the same ZCA-whitened, unit-sphered space the caller
    already computes for every other expert (`z_all` in `run_remote_eval_v6.analyze`).
    That keeps `dim` at the ZCA rank (tens, not thousands), so no working-set budget
    from `bifurcated_fractal_engine.working_set_bytes` is at risk -- that check lives
    inside `.think()`, which this module does not call.
  * Nothing here reads `ground_truth`, `y`, or any label. The only inputs are the
    prompt's representation vector, the candidates' representation vectors, and
    (optionally) a domain prototype vector built the same label-free way
    `competence_gate` builds it.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import numpy as np

from .bifurcated_fractal_engine import BifurcatedFractalEngine
from .counterfactual_constraint_verifier import CounterfactualConstraintVerifier
from .forced_affine_scan import ConditionSet, hard_forced_step
from .fractal_multiscale_engine import _positive_int, _repulsor_matrix, _vector
from .parallel_rnn_lora import ParallelRNNLoRAAdapter
from .dynamics_calibrator import CalibratedDynamics

__all__ = [
    "STATUS_CONVERGED",
    "STATUS_PRUNED",
    "CandidateReasoningTrace",
    "ContinuousCausalReasoningResult",
    "ContinuousCausalReasoningExpert",
    "continuous_causal_reasoning_expert",
]

STATUS_CONVERGED = "CONVERGED"
STATUS_PRUNED = "PRUNED"

# Bounded saturation cap for the score penalty applied when the macro race
# hard-wipes a candidate's branch pool (`race_trace.pruned_all`). Genuine
# wipeouts on validated finite inputs need a non-finite race cost from *every*
# branch (BifurcatedFractalEngine.race()'s own rescue already keeps a finite
# candidate whenever one exists), so this path is a defense-in-depth floor,
# not the common case it used to be. `RACE_PENALTY_SCALE * tanh(x / scale)`
# saturates any finite or infinite race cost into [0, RACE_PENALTY_SCALE),
# so one degenerate candidate can never swamp the K-ary comparison the way a
# fixed -100/-1e9 floor did.
RACE_PENALTY_SCALE = 10.0


class ContinuousCausalReasoningExpert:
    """Inference owner for an optional, provenance-checked compact dynamics sidecar."""

    def __init__(self, dynamics: CalibratedDynamics):
        if not isinstance(dynamics, CalibratedDynamics):
            raise TypeError("dynamics must be CalibratedDynamics")
        if dynamics.dim != dynamics.input_dim:
            raise ValueError("reasoning dynamics requires equal state and input dimensions")
        dynamics.certify()
        self.dynamics = dynamics
        self._cached_feature = None
        self._cached_forcing = None

    @classmethod
    def from_file(cls, path, *, encoder_id: str):
        return cls(CalibratedDynamics.load(path, encoder_id=encoder_id))

    @property
    def dim(self) -> int:
        return self.dynamics.dim

    def step(self, feature: np.ndarray, state: np.ndarray) -> np.ndarray:
        """One allocation-bounded calibrated recurrence step."""
        feature = np.asarray(feature)
        if (self._cached_feature is None
                or not np.array_equal(feature, self._cached_feature)):
            self._cached_forcing = self.dynamics.forcing(feature)
            self._cached_feature = feature.astype(self.dynamics.dtype, copy=True)
        if self.dynamics.rank == 1:
            return self.dynamics._step_cached_forcing(self._cached_forcing, state)
        return self.dynamics.step_forcing(self._cached_forcing, state)

    def bind_feature(self, feature: np.ndarray) -> None:
        """Validate and bind the constant input used by an iterative trajectory."""
        self._cached_forcing = self.dynamics.forcing(feature)
        self._cached_feature = np.asarray(feature, dtype=self.dynamics.dtype).copy()

    def step_bound(self, state: np.ndarray) -> np.ndarray:
        """Advance one state after :meth:`bind_feature` computed the fixed Bx term."""
        if self._cached_forcing is None:
            raise RuntimeError("bind_feature must be called before step_bound")
        if self.dynamics.rank == 1:
            return self.dynamics._step_cached_forcing(self._cached_forcing, state)
        return self.dynamics.step_forcing(self._cached_forcing, state)


@dataclasses.dataclass(frozen=True)
class CandidateReasoningTrace:
    """Per-candidate diagnostics proving the engines actually ran (not decorative)."""

    status: str
    race_steps: int
    race_alive_history: Tuple[int, ...]
    macro_survivor_cost: float
    micro_residual_to_target: float
    judge_penalty: float
    score: float


@dataclasses.dataclass(frozen=True)
class ContinuousCausalReasoningResult:
    scores: np.ndarray                          # (K,) higher is better, always finite
    traces: Tuple[CandidateReasoningTrace, ...]  # len K


def _validate_candidates(q0: np.ndarray, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q0 = _vector(q0, "q0")
    c = np.asarray(candidates, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != q0.shape[0]:
        raise ValueError(f"candidates must be (K, {q0.shape[0]}) with K >= 1, got {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("candidates must contain only finite values")
    return q0, c.copy()


def continuous_causal_reasoning_expert(
    q0: np.ndarray,
    candidates: np.ndarray,
    *,
    domain_prototype: Optional[np.ndarray] = None,
    rank: int = 4,
    seed: int = 0,
    dt_micro: float = 0.2,
    micro_steps: int = 20,
    judge_weight: float = 1.0,
    enable_repulsion: bool = True,
    calibrated_expert: Optional[ContinuousCausalReasoningExpert] = None,
) -> ContinuousCausalReasoningResult:
    """Macro race + micro hard forcing + counterfactual judge, per candidate.

    `q0`: (dim,) prompt representation, ZCA-whitened and sphered.
    `candidates`: (K, dim) candidate terminal representations, same space as `q0`.
    `domain_prototype`: (dim,) label-free counterfactual premise reference (the mean
        representation of other same-domain records, excluding this one). `None`
        disables the judge layer entirely (no counterfactual reference to test
        against, which the verifier's own contract says yields zero evidence anyway).
    `enable_repulsion`: when False, no candidate ever sees a competitor as a repulsor
        and `hard_forced_step` always gets an empty (identity) condition set. This is
        the ablation switch: see the module's test suite for the discriminating test
        that this flag must actually change the output, or the "reasoning" is cosmetic.
    """
    q0, cands = _validate_candidates(q0, candidates)
    dim = q0.shape[0]
    k = cands.shape[0]
    rank = min(_positive_int(rank, "rank"), dim)
    if calibrated_expert is not None and calibrated_expert.dim != dim:
        raise ValueError("calibrated expert dimension does not match candidate geometry")

    adapter = ParallelRNNLoRAAdapter(dim=dim, rank=rank, rho_max=0.5, seed=seed)
    engine = BifurcatedFractalEngine(dim=dim, adapter=adapter)
    verifier = CounterfactualConstraintVerifier()

    final_states: List[Optional[np.ndarray]] = [None] * k
    pre_judge_scores = np.full(k, np.nan, dtype=np.float64)
    partial_traces: List[dict] = []

    for i in range(k):
        c_i = cands[i]
        other_idx = [j for j in range(k) if j != i]
        has_competitors = bool(other_idx) and enable_repulsion
        p0 = c_i - q0
        repulsors = _repulsor_matrix(np.stack([cands[j] - q0 for j in other_idx]), "repulsors", dim) \
            if has_competitors else None

        pool, _decision = engine.bifurcate(q0, p0, c_i, repulsors=repulsors)
        pool, race_trace = engine.race(pool, c_i, repulsors=repulsors)

        # `race()` itself never lets a branch with a finite cost go
        # unrescued (BifurcatedFractalEngine.race, soft least-cost fallback),
        # so `pruned_all` here means not one of the K branches had a finite
        # race cost at all -- a genuine numeric failure, not the common
        # "everything was over the hard repulsion threshold" case this used
        # to catch. There is no real macro survivor to refine in that case:
        # fall back to the unraced prompt state so the batch stays alive,
        # and price the failure itself into the score as a bounded penalty
        # instead of a non-physical floor.
        if race_trace.pruned_all:
            macro_state = q0
            macro_cost = float("inf")
        else:
            survivors = np.flatnonzero(pool.alive)
            survivor = int(min(survivors, key=lambda s: (float(pool.cost[s]), int(s))))
            macro_state = pool.q[survivor]
            macro_cost = float(pool.cost[survivor])

        if calibrated_expert is not None:
            macro_state = calibrated_expert.step(c_i.astype(np.float32),
                                                  macro_state.astype(np.float32)).astype(np.float64)

        if has_competitors:
            g_rows = np.stack([cands[j] - q0 for j in other_idx])
            norms = np.linalg.norm(g_rows, axis=1, keepdims=True)
            g_rows = g_rows / np.maximum(norms, 1e-12)
            # Repulsors are displacements from q0, so constrain G(s - q0) = 0.
            # Gs = 0 instead projects the absolute state toward the coordinate
            # origin and can penalize even a target already equal to q0.
            cond = ConditionSet.from_arrays(g_rows, g_rows @ q0)
        else:
            cond = ConditionSet.empty(dim)

        s = macro_state

        def _micro_step_fn(x_t: np.ndarray, s_prev: np.ndarray) -> np.ndarray:
            return s_prev + dt_micro * (x_t - s_prev)

        for _ in range(micro_steps):
            s = hard_forced_step(_micro_step_fn, c_i, s, cond)
        final_states[i] = s
        target_residual = float(np.linalg.norm(s - c_i))
        if race_trace.pruned_all:
            # Lyapunov-style continuous energy penalty for the race-failure
            # fallback only: `macro_cost` (here `inf`, no branch survived at
            # all) is the same `_cost()` energy every surviving candidate
            # would have been scored by. tanh saturates it into a finite,
            # bounded range so this one candidate can never swamp the K-ary
            # comparison the way a fixed -100/-1e9 floor did.
            penalty = RACE_PENALTY_SCALE * math.tanh(macro_cost / RACE_PENALTY_SCALE)
            pre_judge_scores[i] = -target_residual - penalty
        else:
            pre_judge_scores[i] = -target_residual
        partial_traces.append({
            "status": STATUS_PRUNED if race_trace.pruned_all else STATUS_CONVERGED,
            "race_steps": race_trace.steps,
            "race_alive_history": race_trace.alive_history,
            "macro_survivor_cost": macro_cost,
            "micro_residual_to_target": target_residual,
        })

    judge_penalty = np.zeros(k, dtype=np.float64)
    if domain_prototype is not None:
        proto = _vector(domain_prototype, "domain_prototype", dim)
        if float(np.linalg.norm(proto)) > 1e-12 and float(np.linalg.norm(proto - q0)) > 1e-12:
            for i in range(k):
                if final_states[i] is None:
                    continue
                others = [final_states[j] for j in range(k) if j != i and final_states[j] is not None]
                if not others:
                    continue
                cf_hypothesis = np.mean(np.stack(others), axis=0)
                try:
                    _, _penalty, neg = verifier.repulsion(q0, final_states[i], proto, cf_hypothesis)
                except ValueError:
                    neg = 0.0  # degenerate (zero-norm) geometry: fail open to no evidence
                judge_penalty[i] = neg

    scores = pre_judge_scores - judge_weight * judge_penalty

    traces = tuple(
        CandidateReasoningTrace(
            status=t["status"],
            race_steps=t["race_steps"],
            race_alive_history=t["race_alive_history"],
            macro_survivor_cost=t["macro_survivor_cost"],
            micro_residual_to_target=t.get("micro_residual_to_target", float("inf")),
            judge_penalty=float(judge_penalty[i]),
            score=float(scores[i]),
        )
        for i, t in enumerate(partial_traces)
    )
    return ContinuousCausalReasoningResult(scores=scores, traces=traces)
