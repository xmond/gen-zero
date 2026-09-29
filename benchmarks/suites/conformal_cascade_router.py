"""Conformal cascade router (task spec citing docs/zero/31 Spec §6, which is not present in
this worktree -- implemented from the task's inline specification, not independently verified
against that document): split-conformal prediction sets gate the dual-process cascade,
replacing the divergence heuristic with a coverage-guaranteed rule.

The 1 - alpha coverage guarantee below is on P(y in C_alpha(x)) per tier, marginally over the
calibration/test distribution. It is NOT a guarantee on the accuracy of committed labels:
conditioning on |C_alpha(x)| == 1 is a selection step that breaks the exchangeability the
guarantee relies on, so a tier's committed samples can run above or below 1 - alpha in practice
(`evaluate_cascade` reports `selective_accuracy` for that reason, distinct from
`empirical_coverage_by_tier`). `route_sample` also never calls a model lazily -- it takes every
tier's probabilities up front and simulates which ones the cascade would have used, for offline
evaluation; it accounts cost, it does not reduce it.

Each tier's model produces a class-probability vector. Split conformal calibration turns that
into a prediction set C_alpha(x) with the marginal guarantee P(y in C_alpha(x)) >= 1 - alpha
(under calibration/test exchangeability). The router walks the tiers in order:

    |C_alpha(x)| == 1   -> COMMIT here (single candidate, confident)
    |C_alpha(x)| != 1   -> escalate to the next tier (0 = anomalous/out-of-distribution point,
                           >=2 = genuine ambiguity)

If the last configured tier still fails to produce a singleton, the router ABSTAINs. It never
guesses a label to "unblock" a sample — on a safety task (e.g. aegis_safety) that means the
sample goes to human/Tier2 review, never a silent release.

Two nonconformity scores are available (`score_method`), and they disagree on what an "empty
set" means, so the choice matters for the routing state machine above:

  * "margin" (LAC, Sadinle et al. 2019; DEFAULT): score = 1 - p_y(x). A class conforms iff its
    probability clears 1 - q_hat. An empty set genuinely means "no class was confident enough" —
    a flat/out-of-distribution point — while a single dominant class always commits. This is the
    scoring the router uses by default because it is the only one of the two where the empty-set
    escalation branch in the spec is a real, reachable state without also misfiring on ordinary
    confident, correct samples.
  * "aps": textbook split-conformal set (Angelopoulos & Bates 2021, Algorithm 1) — classes are
    added in decreasing probability order up to and including the one that first makes the
    cumulative sum reach q_hat. This construction is *never* empty by design (the top class
    always conforms), so under "aps" the only escalation trigger is genuine ambiguity
    (|C| >= 2); there is no separate "anomalous empty set" state. Do not implement this as the
    naive "score(x, c) <= q_hat for every c independently" rule: that variant excludes the top
    class whenever p_top(x) > q_hat, which turns the *most* confident, correct samples into
    empty sets and forces them through the whole cascade to a terminal Abstain — the opposite of
    "confident samples commit at Tier 1, low cost."

Fail-closed rules (nothing here degrades silently):
  * Probability rows that are not finite, out of [0, 1], or don't sum to 1 raise ValueError.
  * A calibration/route/evaluate call with a tier count, class count, or label range mismatch
    raises ValueError.
  * A calibration set too small for the requested alpha (so the conformal quantile would need to
    exceed 1) raises ValueError naming the minimum calibration size required, instead of
    silently degrading every prediction set to "the full label set forever".
  * `evaluate_cascade` requires the router to already be calibrated with the SAME alpha it is
    called with — it never silently recalibrates or reuses a mismatched threshold.
  * `route_sample` on an unresolved cascade always returns Verdict.ABSTAIN with label=None,
    never a guessed label from the last tier's arbitrary top class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ALPHA = 0.05
DEFAULT_ENTROPY_THRESHOLD = 0.65
# Naive baseline threshold ("commit if the top class clears a coin-flip") for evaluate_cascade's
# comparison report only. It has no coverage guarantee and is never used for the router's own
# gating decision -- on a binary task it can commit 100% of samples at tier 1, which is exactly
# the failure mode the conformal gate exists to avoid.
DEFAULT_MAX_PROB_THRESHOLD = 0.5
_PROB_ATOL = 1e-7
_SET_ATOL = 1e-9


class ScoreMethod(str, Enum):
    APS = "aps"
    MARGIN = "margin"


class Verdict(str, Enum):
    COMMIT = "COMMIT"
    ABSTAIN = "ABSTAIN"


# ------------------------------------------------------------------ validation helpers

def _validate_prob_matrix(probs, name: str, n_classes: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (n_samples, n_classes), got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 classes, got {arr.shape[1]}")
    if n_classes is not None and arr.shape[1] != n_classes:
        raise ValueError(f"{name} has {arr.shape[1]} classes, expected {n_classes}")
    if (arr < -_PROB_ATOL).any() or (arr > 1 + _PROB_ATOL).any():
        raise ValueError(f"{name} values must lie in [0, 1]")
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows must sum to 1 (probabilities), row {bad} sums to {sums[bad]!r}")
    # Remove only machine-rounding excursions, then normalize the accepted rows.
    arr = np.clip(arr, 0.0, 1.0)
    sums = arr.sum(axis=1)
    arr = arr / sums[:, None]
    return arr


def _validate_prob_vector(probs, name: str, n_classes: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector, got shape {arr.shape}")
    return _validate_prob_matrix(arr[None, :], name, n_classes)[0]


def _validate_labels(labels, n_samples: int, n_classes: int) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {arr.shape}")
    if arr.size and not np.all(np.isfinite(arr.astype(np.float64))):
        raise ValueError("labels contains NaN or Inf")
    if arr.size and not np.all(np.mod(arr.astype(np.float64), 1) == 0):
        raise ValueError("labels must be integer class indices")
    arr = arr.astype(np.int64)
    if arr.size and ((arr < 0).any() or (arr >= n_classes).any()):
        raise ValueError(f"labels must be in [0, {n_classes}), got range [{arr.min()}, {arr.max()}]")
    return arr


def _check_cost(c) -> float:
    c = float(c)
    if not math.isfinite(c) or c < 0:
        raise ValueError(f"cost must be finite and >= 0, got {c}")
    return c


# ------------------------------------------------------------------ nonconformity scoring

def _nonconformity_scores(probs: np.ndarray, labels: np.ndarray, method: ScoreMethod) -> np.ndarray:
    n = probs.shape[0]
    p_true = probs[np.arange(n), labels]
    if method is ScoreMethod.MARGIN:
        return 1.0 - p_true
    # APS/HPS score: cumulative probability mass of every class at least as likely as the truth.
    ge = probs >= p_true[:, None]
    return (probs * ge).sum(axis=1)


def _min_calibration_n(alpha: float) -> int:
    n = max(1, math.ceil(1.0 / alpha) - 2)
    while math.ceil((n + 1) * (1.0 - alpha)) > n:
        n += 1
    return n


def _quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    n = scores.shape[0]
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        # Not enough calibration points to support this alpha: the (n+1)(1-alpha)/n quantile
        # would need to exceed the largest score, which makes every prediction set "the full
        # label set" forever. That is a silent, permanent full-escalation failure mode, so it
        # is fail-closed as an error rather than returned as a usable (inf) threshold.
        raise ValueError(
            f"calibration set too small for alpha={alpha}: need at least "
            f"{_min_calibration_n(alpha)} calibration samples for a finite conformal "
            f"threshold, got {n}")
    return float(np.quantile(scores, level, method="higher"))


def _prediction_set(probs_row: np.ndarray, q_hat: float, method: ScoreMethod) -> Tuple[int, ...]:
    if method is ScoreMethod.MARGIN:
        # LAC: a class conforms iff its own probability clears 1 - q_hat. Can be legitimately
        # empty (no class confident enough) -- see the module docstring.
        scores = 1.0 - probs_row
        return tuple(int(i) for i in np.where(scores <= q_hat + _SET_ATOL)[0])

    # APS: include classes in decreasing-probability order up to and including whichever one
    # first makes the cumulative mass reach q_hat (Angelopoulos & Bates 2021, Algorithm 1).
    # Ties are grouped by value (not sort position) so tied classes are always included/excluded
    # together. A class's "mass strictly ahead of it" determines inclusion; the top-probability
    # class group always has zero mass ahead of it, so this is never empty for q_hat >= 0.
    strictly_greater = probs_row[None, :] > probs_row[:, None]
    mass_ahead = (strictly_greater * probs_row[None, :]).sum(axis=1)
    included = mass_ahead < q_hat + _SET_ATOL
    if not included.any():
        top = probs_row.max()
        included = probs_row >= top - _SET_ATOL
    return tuple(int(i) for i in np.where(included)[0])


def _normalized_entropy(probs_row: np.ndarray) -> float:
    k = probs_row.shape[0]
    p = probs_row[probs_row > 0]
    h = -float((p * np.log(p)).sum())
    return h / math.log(k)


# ------------------------------------------------------------------ result types

@dataclass(frozen=True)
class TierCalibration:
    tier_name: str
    q_hat: float
    n_calibration: int


@dataclass(frozen=True)
class CascadeDecision:
    verdict: str
    label: Optional[int]
    prediction_set: Tuple[int, ...]
    committed_tier: Optional[str]
    tiers_visited: Tuple[str, ...]
    n_escalations: int
    cost: float
    task_name: str
    reason: str

    @property
    def committed(self) -> bool:
        return self.verdict == Verdict.COMMIT.value

    @property
    def abstained(self) -> bool:
        return self.verdict == Verdict.ABSTAIN.value


@dataclass(frozen=True)
class BaselineCascadeReport:
    name: str
    overall_accuracy: float
    selective_accuracy: float
    abstention_rate: float
    expected_cost: float
    escalation_rate_by_tier: Dict[str, float]


@dataclass(frozen=True)
class CascadeEvaluationReport:
    """`escalation_rate_by_tier` for the last configured tier is always 0.0 by construction --
    there is nowhere further to escalate to, so a non-singleton set there ends in Abstain, not
    escalation."""
    alpha: float
    score_method: str
    n_samples: int
    overall_accuracy: float
    selective_accuracy: float
    abstention_rate: float
    expected_cost: float
    empirical_coverage_by_tier: Dict[str, float]
    escalation_rate_by_tier: Dict[str, float]
    baseline_entropy: BaselineCascadeReport
    baseline_max_prob: BaselineCascadeReport


# ------------------------------------------------------------------ router

class ConformalCascadeRouter:
    """Split-conformal dual/multi-process cascade gate.

    `tier_names` and `costs`, if given, must match the number of tiers passed to `calibrate`.
    Costs default to a uniform 1.0 per tier (i.e. cost == number of tiers used) when omitted.
    """

    def __init__(self, tier_names: Optional[Sequence[str]] = None,
                 costs: Optional[Sequence[float]] = None,
                 score_method: str = "margin"):
        try:
            self._score_method = ScoreMethod(score_method)
        except ValueError as exc:
            allowed = [m.value for m in ScoreMethod]
            raise ValueError(f"score_method must be one of {allowed}, got {score_method!r}") from exc
        self._tier_names_override = None if tier_names is None else tuple(tier_names)
        self._default_costs = None if costs is None else [_check_cost(c) for c in costs]
        self.tier_calibrations: Optional[Tuple[TierCalibration, ...]] = None
        self.n_classes: Optional[int] = None
        self.n_tiers: Optional[int] = None
        self._alpha: Optional[float] = None

    def _require_calibrated(self) -> None:
        if self.tier_calibrations is None:
            raise ValueError("router is not calibrated: call calibrate() first")

    def _resolve_tier_names(self, n_tiers: int) -> Tuple[str, ...]:
        if self._tier_names_override is not None:
            if len(self._tier_names_override) != n_tiers:
                raise ValueError(
                    f"tier_names has {len(self._tier_names_override)} entries, expected {n_tiers}")
            return self._tier_names_override
        return tuple(f"tier_{i + 1}" for i in range(n_tiers))

    # -------------------------------------------------------------- calibrate

    def calibrate(self, calibration_probs: Sequence[np.ndarray], calibration_labels,
                  alpha: float = DEFAULT_ALPHA) -> "ConformalCascadeRouter":
        if not isinstance(calibration_probs, (list, tuple)) or len(calibration_probs) == 0:
            raise ValueError("calibration_probs must be a non-empty sequence of per-tier probability matrices")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n_tiers = len(calibration_probs)
        tier_names = self._resolve_tier_names(n_tiers)
        if self._default_costs is not None and len(self._default_costs) != n_tiers:
            raise ValueError(f"costs has {len(self._default_costs)} entries, expected {n_tiers} tiers")

        first = _validate_prob_matrix(calibration_probs[0], "calibration_probs[0]")
        n_calib, n_classes = first.shape
        labels = _validate_labels(calibration_labels, n_calib, n_classes)

        matrices = [first]
        for i in range(1, n_tiers):
            mat = _validate_prob_matrix(calibration_probs[i], f"calibration_probs[{i}]", n_classes)
            if mat.shape[0] != n_calib:
                raise ValueError(
                    f"calibration_probs[{i}] has {mat.shape[0]} rows, expected {n_calib} (calibration_probs[0])")
            matrices.append(mat)

        calibrations = []
        for name, mat in zip(tier_names, matrices):
            scores = _nonconformity_scores(mat, labels, self._score_method)
            q_hat = _quantile_threshold(scores, alpha)
            calibrations.append(TierCalibration(tier_name=name, q_hat=q_hat, n_calibration=n_calib))

        self.tier_calibrations = tuple(calibrations)
        self.n_classes = n_classes
        self.n_tiers = n_tiers
        self._alpha = alpha
        return self

    # -------------------------------------------------------------- routing

    def _predict_set(self, tier_idx: int, probs_row: np.ndarray) -> Tuple[int, ...]:
        return _prediction_set(probs_row, self.tier_calibrations[tier_idx].q_hat, self._score_method)

    def _route(self, probs_by_tier: Sequence[np.ndarray], costs: Sequence[float],
               task_name: str = "") -> CascadeDecision:
        tier_names = [c.tier_name for c in self.tier_calibrations]
        visited: List[str] = []
        cumulative_cost = 0.0
        last_set: Tuple[int, ...] = ()
        for i in range(self.n_tiers):
            visited.append(tier_names[i])
            cumulative_cost += costs[i]
            pred_set = self._predict_set(i, probs_by_tier[i])
            last_set = pred_set
            if len(pred_set) == 1:
                return CascadeDecision(
                    verdict=Verdict.COMMIT.value,
                    label=pred_set[0],
                    prediction_set=pred_set,
                    committed_tier=tier_names[i],
                    tiers_visited=tuple(visited),
                    n_escalations=i,
                    cost=cumulative_cost,
                    task_name=task_name,
                    reason=f"{tier_names[i]}_SINGLETON",
                )
        reason = "TERMINAL_ABSTAIN_EMPTY" if len(last_set) == 0 else "TERMINAL_ABSTAIN_AMBIGUOUS"
        return CascadeDecision(
            verdict=Verdict.ABSTAIN.value,
            label=None,
            prediction_set=last_set,
            committed_tier=None,
            tiers_visited=tuple(visited),
            n_escalations=self.n_tiers - 1,
            cost=cumulative_cost,
            task_name=task_name,
            reason=reason,
        )

    def route_sample(self, probs_by_tier: Sequence[np.ndarray], task_name: str = "") -> CascadeDecision:
        self._require_calibrated()
        if len(probs_by_tier) != self.n_tiers:
            raise ValueError(f"probs_by_tier has {len(probs_by_tier)} tiers, router calibrated for {self.n_tiers}")
        rows = [_validate_prob_vector(p, f"probs_by_tier[{i}]", self.n_classes)
                for i, p in enumerate(probs_by_tier)]
        costs = self._default_costs if self._default_costs is not None else [1.0] * self.n_tiers
        return self._route(rows, costs, task_name)

    # -------------------------------------------------------------- evaluation

    @staticmethod
    def _summarize(decisions: Sequence[CascadeDecision], labels_arr: np.ndarray,
                    tier_names: Sequence[str]):
        committed = np.array([d.committed for d in decisions])
        correct = np.array([d.committed and d.label == labels_arr[i] for i, d in enumerate(decisions)])
        overall_acc = float(correct.mean())
        sel_acc = float(correct[committed].mean()) if committed.any() else float("nan")
        abst_rate = float(1.0 - committed.mean())
        exp_cost = float(np.mean([d.cost for d in decisions]))
        esc_by_tier = {}
        for t, name in enumerate(tier_names):
            reached = np.array([len(d.tiers_visited) >= t + 1 for d in decisions])
            escalated_past = np.array([len(d.tiers_visited) >= t + 2 for d in decisions])
            esc_by_tier[name] = float(escalated_past.sum() / reached.sum()) if reached.any() else 0.0
        return overall_acc, sel_acc, abst_rate, exp_cost, esc_by_tier

    def _baseline_report(self, name: str, matrices: Sequence[np.ndarray], labels_arr: np.ndarray,
                          costs: Sequence[float], tier_names: Sequence[str],
                          rule: Callable[[np.ndarray], bool]) -> BaselineCascadeReport:
        n = matrices[0].shape[0]
        n_tiers = len(tier_names)
        committed_flags, corrects, visited_counts, total_costs = [], [], [], []
        for j in range(n):
            visited = 0
            cum_cost = 0.0
            committed = False
            label = None
            for t in range(n_tiers):
                visited = t + 1
                cum_cost += costs[t]
                probs_row = matrices[t][j]
                if rule(probs_row):
                    committed = True
                    label = int(np.argmax(probs_row))
                    break
            committed_flags.append(committed)
            corrects.append(committed and label == labels_arr[j])
            visited_counts.append(visited)
            total_costs.append(cum_cost)

        committed_arr = np.array(committed_flags)
        correct_arr = np.array(corrects)
        overall_acc = float(correct_arr.mean())
        sel_acc = float(correct_arr[committed_arr].mean()) if committed_arr.any() else float("nan")
        abst_rate = float(1.0 - committed_arr.mean())
        exp_cost = float(np.mean(total_costs))
        visited_arr = np.array(visited_counts)
        esc_by_tier = {}
        for t, tname in enumerate(tier_names):
            reached = visited_arr >= t + 1
            escalated_past = visited_arr >= t + 2
            esc_by_tier[tname] = float(escalated_past.sum() / reached.sum()) if reached.any() else 0.0
        return BaselineCascadeReport(name=name, overall_accuracy=overall_acc, selective_accuracy=sel_acc,
                                      abstention_rate=abst_rate, expected_cost=exp_cost,
                                      escalation_rate_by_tier=esc_by_tier)

    def evaluate_cascade(self, tiers_probs: Sequence[np.ndarray], labels, costs: Sequence[float],
                         alpha: float = DEFAULT_ALPHA,
                         entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
                         max_prob_threshold: float = DEFAULT_MAX_PROB_THRESHOLD) -> CascadeEvaluationReport:
        self._require_calibrated()
        alpha = float(alpha)
        if alpha != self._alpha:
            raise ValueError(
                f"alpha={alpha} does not match the calibrated alpha={self._alpha}; "
                "call calibrate() again with a matching alpha before evaluating")
        if len(tiers_probs) != self.n_tiers:
            raise ValueError(f"tiers_probs has {len(tiers_probs)} tiers, router calibrated for {self.n_tiers}")

        matrices = [_validate_prob_matrix(p, f"tiers_probs[{i}]", self.n_classes)
                    for i, p in enumerate(tiers_probs)]
        n = matrices[0].shape[0]
        for i, mat in enumerate(matrices[1:], start=1):
            if mat.shape[0] != n:
                raise ValueError(f"tiers_probs[{i}] has {mat.shape[0]} rows, expected {n} (tiers_probs[0])")
        labels_arr = _validate_labels(labels, n, self.n_classes)

        if len(costs) != self.n_tiers:
            raise ValueError(f"costs has {len(costs)} entries, expected {self.n_tiers} tiers")
        costs = [_check_cost(c) for c in costs]

        if not math.isfinite(entropy_threshold) or entropy_threshold <= 0.0:
            raise ValueError(f"entropy_threshold must be a positive finite number, got {entropy_threshold}")
        if not (0.0 <= max_prob_threshold <= 1.0):
            raise ValueError(f"max_prob_threshold must be in [0, 1], got {max_prob_threshold}")

        decisions = [self._route([matrices[t][i] for t in range(self.n_tiers)], costs)
                     for i in range(n)]

        tier_names = [c.tier_name for c in self.tier_calibrations]
        overall_acc, sel_acc, abst_rate, exp_cost, esc_by_tier = self._summarize(decisions, labels_arr, tier_names)

        coverage_by_tier = {}
        for i, name in enumerate(tier_names):
            sets = [self._predict_set(i, matrices[i][j]) for j in range(n)]
            covered = np.array([labels_arr[j] in sets[j] for j in range(n)])
            coverage_by_tier[name] = float(covered.mean())

        baseline_entropy = self._baseline_report(
            "normalized_entropy", matrices, labels_arr, costs, tier_names,
            rule=lambda p: _normalized_entropy(p) <= entropy_threshold)
        baseline_max_prob = self._baseline_report(
            "max_probability", matrices, labels_arr, costs, tier_names,
            rule=lambda p: float(p.max()) >= max_prob_threshold)

        return CascadeEvaluationReport(
            alpha=alpha,
            score_method=self._score_method.value,
            n_samples=n,
            overall_accuracy=overall_acc,
            selective_accuracy=sel_acc,
            abstention_rate=abst_rate,
            expected_cost=exp_cost,
            empirical_coverage_by_tier=coverage_by_tier,
            escalation_rate_by_tier=esc_by_tier,
            baseline_entropy=baseline_entropy,
            baseline_max_prob=baseline_max_prob,
        )
