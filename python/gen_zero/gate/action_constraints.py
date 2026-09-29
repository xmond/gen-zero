"""Structured action constraints for the decision path.

Caller-supplied constraints are plain dicts:

    {"type": "forbid", "actions": ["format"]}
    {"type": "mutually_exclusive", "actions": ["reboot", "format"]}
    {"type": "upper_bound", "action": "reboot", "value": 0.3}
    {"type": "lower_bound", "action": "backup", "value": 0.2}

Semantics (all fail-closed):
- Action names match case-insensitively (strip + upper), like the rule compiler.
- ``forbid`` and the losers of a ``mutually_exclusive`` group get probability exactly 0.
- Mutual exclusion is resolved greedily: candidates are visited by (has lower bound, then
  higher input probability, then candidate order). A candidate is dropped when an already
  kept candidate shares a group with it. Overlapping groups are handled by the same rule.
- The final distribution is ``p_i = clip(c * q_i, lo_i, hi_i)`` with ``c`` chosen so the sum is
  exactly 1. With no bounds this is "zero the disabled actions and renormalise
  proportionally". With bounds it is the box-constrained simplex scaling (the KL projection
  of q when every q_i > 0). No probability mass is ever invented for an action whose input
  probability is 0: if the constraints cannot be met without that, the projection reports
  INSUFFICIENT_SUPPORT and every probability is 0.
- Malformed specs raise ValueError. Unknown extra keys raise too, so a typo cannot pass silently.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# formal_verification values
FV_NOT_REQUESTED = "not_requested"
FV_UNAVAILABLE = "unavailable_missing_dependency"
FV_VERIFIED = "cp_sat_verified"
FV_REFUTED = "cp_sat_refuted"
FV_NO_ACTION = "not_applicable_no_action_selected"

# projection statuses
PROJ_OK = "OK"
PROJ_ALL_ZERO_INPUT = "ALL_ZERO_INPUT"
PROJ_INFEASIBLE_BOUNDS = "INFEASIBLE_BOUNDS"
PROJ_INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"

KIND_MUTEX = "mutually_exclusive"
KIND_FORBID = "forbid"
KIND_UPPER = "upper_bound"
KIND_LOWER = "lower_bound"

_KIND_ALIASES = {"mutual_exclusive": KIND_MUTEX}
_ALLOWED_KEYS = {
    KIND_MUTEX: {"type", "actions"},
    KIND_FORBID: {"type", "actions"},
    KIND_UPPER: {"type", "action", "value"},
    KIND_LOWER: {"type", "action", "value"},
}
_SUM_TOL = 1e-9


def norm_action(action: Any) -> str:
    return str(action).strip().upper()


@dataclass(frozen=True)
class ActionConstraintSpec:
    kind: str
    actions: Tuple[str, ...]  # normalised
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.kind, "actions": list(self.actions)}
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass
class ProjectionResult:
    status: str
    probs: Dict[str, float]  # keyed by the caller's candidate names
    disabled: List[str] = field(default_factory=list)  # zeroed by forbid / mutex / upper_bound 0
    mutex_dropped: List[str] = field(default_factory=list)
    unmatched_actions: List[str] = field(default_factory=list)  # named by a constraint, not a candidate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "disabled": list(self.disabled),
            "mutex_dropped": list(self.mutex_dropped),
            "unmatched_actions": list(self.unmatched_actions),
        }


def _action_list(spec: Mapping[str, Any], key: str, minimum: int, idx: int) -> Tuple[str, ...]:
    raw = spec.get(key)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"constraint #{idx}: {key!r} must be a list of action names, got {type(raw).__name__}")
    names = [norm_action(a) for a in raw if isinstance(a, str) and a.strip()]
    if len(names) != len(raw):
        raise ValueError(f"constraint #{idx}: every entry of {key!r} must be a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError(f"constraint #{idx}: {key!r} lists an action twice")
    if len(names) < minimum:
        raise ValueError(f"constraint #{idx}: {key!r} needs at least {minimum} action(s)")
    return tuple(names)


def parse_action_constraints(specs: Any) -> Tuple[ActionConstraintSpec, ...]:
    """Validate and normalise caller constraints. Raises ValueError on anything malformed."""
    if isinstance(specs, (str, bytes, Mapping)) or not isinstance(specs, (list, tuple)):
        raise ValueError(f"constraints must be a list of constraint objects, got {type(specs).__name__}")
    parsed: List[ActionConstraintSpec] = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"constraint #{idx}: must be an object, got {type(spec).__name__}")
        raw_kind = spec.get("type")
        if not isinstance(raw_kind, str):
            raise ValueError(f"constraint #{idx}: 'type' is required")
        kind = raw_kind.strip().lower()
        kind = _KIND_ALIASES.get(kind, kind)
        if kind not in _ALLOWED_KEYS:
            raise ValueError(
                f"constraint #{idx}: unknown type {raw_kind!r}; expected one of {sorted(_ALLOWED_KEYS)}"
            )
        extra = set(spec) - _ALLOWED_KEYS[kind]
        if extra:
            raise ValueError(f"constraint #{idx} ({kind}): unexpected key(s) {sorted(extra)}")
        if kind in (KIND_MUTEX, KIND_FORBID):
            actions = _action_list(spec, "actions", 2 if kind == KIND_MUTEX else 1, idx)
            parsed.append(ActionConstraintSpec(kind, actions))
            continue
        action = spec.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"constraint #{idx} ({kind}): 'action' must be a non-empty string")
        value = spec.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"constraint #{idx} ({kind}): 'value' must be a finite number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"constraint #{idx} ({kind}): 'value' must lie in [0, 1], got {value}")
        parsed.append(ActionConstraintSpec(kind, (norm_action(action),), float(value)))
    return tuple(parsed)


def _resolve_mask(
    specs: Sequence[ActionConstraintSpec],
    names: List[str],
    priority: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Tuple[Set[int], Set[int]]:
    """Apply forbid + mutex to ``hi`` in place. Returns (disabled indices, mutex-dropped indices)."""
    index = {n: i for i, n in enumerate(names)}
    for s in specs:
        if s.kind == KIND_FORBID:
            for a in s.actions:
                if a in index:
                    hi[index[a]] = 0.0
    groups = [[index[a] for a in s.actions if a in index] for s in specs if s.kind == KIND_MUTEX]
    groups = [g for g in groups if len(g) > 1]
    dropped: Set[int] = set()
    if groups:
        kept: List[int] = []
        for i in sorted(range(len(names)), key=lambda i: (lo[i] <= 0.0, -priority[i], i)):
            if hi[i] <= 0.0:
                continue  # cannot be chosen anyway, so it must not win a group
            if any(i in g and j in g for g in groups for j in kept):
                hi[i] = 0.0
                dropped.add(i)
            else:
                kept.append(i)
    disabled = {i for i in range(len(names)) if hi[i] <= 0.0}
    return disabled, dropped


def _clip_scale(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[np.ndarray]:
    """Exact p = clip(c*q, lo, hi) with sum(p) == 1, or None when no such c exists."""
    def total(c: float) -> float:
        return float(np.clip(c * q, lo, hi).sum())

    pos = q > 0.0
    bps = np.concatenate(([0.0], lo[pos] / q[pos], hi[pos] / q[pos]))
    bps = np.unique(bps[np.isfinite(bps)])
    vals = np.array([total(b) for b in bps])
    if vals[-1] < 1.0 - _SUM_TOL:
        return None
    k = int(np.argmax(vals >= 1.0 - _SUM_TOL))
    if k == 0:
        c = float(bps[0])
    else:
        c0, c1, f0, f1 = float(bps[k - 1]), float(bps[k]), float(vals[k - 1]), float(vals[k])
        c = c1 if f1 <= f0 else c0 + (1.0 - f0) / (f1 - f0) * (c1 - c0)
    p = np.clip(c * q, lo, hi)
    if abs(float(p.sum()) - 1.0) > 1e-7:
        logger.error("clip-scale solve missed the unit sum: sum=%r; reporting no feasible distribution", p.sum())
        return None
    return p


def resolve_disabled_actions(
    specs: Sequence[ActionConstraintSpec],
    priorities: Mapping[str, float],
    pre_disabled: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Hard-mask part only, for callers with scores instead of probabilities.

    Returns (disabled, mutex_dropped, unmatched_actions) in caller names; ``pre_disabled`` actions
    steer the mutex outcome but are not reported as disabled by the constraints. ``priorities`` orders
    the mutex winners (higher wins). Bound constraints are rejected: they need a distribution.
    """
    if any(s.kind in (KIND_UPPER, KIND_LOWER) for s in specs):
        raise ValueError("upper_bound / lower_bound constrain a probability distribution and are not valid here")
    originals = list(priorities)
    names = [norm_action(a) for a in originals]
    if len(set(names)) != len(names):
        raise ValueError(f"candidates collide after case-insensitive normalisation: {originals}")
    pre = {norm_action(a) for a in (pre_disabled or set())}
    prio = np.array([float(priorities[a]) for a in originals], dtype=np.float64)
    lo = np.zeros(len(names))
    hi = np.array([0.0 if n in pre else 1.0 for n in names])
    disabled, dropped = _resolve_mask(specs, names, prio, lo, hi)
    disabled = {i for i in disabled if names[i] not in pre}  # report only what the constraints did
    return (
        [originals[i] for i in sorted(disabled)],
        [originals[i] for i in sorted(dropped)],
        _unmatched(specs, set(names)),
    )


def _unmatched(specs: Sequence[ActionConstraintSpec], known: Set[str]) -> List[str]:
    seen: List[str] = []
    for s in specs:
        for a in s.actions:
            if a not in known and a not in seen:
                seen.append(a)
    return seen


def project_distribution(
    specs: Sequence[ActionConstraintSpec], probs: Mapping[str, float]
) -> ProjectionResult:
    """Project a candidate distribution onto the constraints. See the module docstring."""
    originals = list(probs)
    if not originals:
        raise ValueError("probs must be non-empty")
    names = [norm_action(a) for a in originals]
    if len(set(names)) != len(names):
        raise ValueError(f"candidates collide after case-insensitive normalisation: {originals}")
    q = np.array([probs[a] for a in originals], dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError(f"probs must be finite and non-negative, got {dict(probs)}")

    zeros = {a: 0.0 for a in originals}
    unmatched = _unmatched(specs, set(names))
    if float(q.sum()) <= 0.0:
        return ProjectionResult(PROJ_ALL_ZERO_INPUT, zeros, unmatched_actions=unmatched)

    index = {n: i for i, n in enumerate(names)}
    lo = np.zeros(len(names))
    hi = np.ones(len(names))
    for s in specs:
        for a in s.actions:
            if s.kind == KIND_UPPER and a in index:
                hi[index[a]] = min(hi[index[a]], s.value)
            elif s.kind == KIND_LOWER:
                if a in index:
                    lo[index[a]] = max(lo[index[a]], s.value)
                elif s.value > 0.0:
                    # A required floor on an action that is not a candidate can never be met.
                    return ProjectionResult(PROJ_INFEASIBLE_BOUNDS, zeros, unmatched_actions=unmatched)
    disabled, dropped = _resolve_mask(specs, names, q, lo, hi)
    disabled_names = [originals[i] for i in sorted(disabled)]
    dropped_names = [originals[i] for i in sorted(dropped)]

    def result(status: str, p: Optional[np.ndarray] = None) -> ProjectionResult:
        out = zeros if p is None else {a: float(p[i]) for i, a in enumerate(originals)}
        return ProjectionResult(status, out, disabled_names, dropped_names, unmatched)

    if np.any(lo > hi) or lo.sum() > 1.0 + _SUM_TOL or hi.sum() < 1.0 - _SUM_TOL:
        return result(PROJ_INFEASIBLE_BOUNDS)
    if np.any((q <= 0.0) & (lo > 0.0)):
        # A floor cannot conjure mass for an action the planner gave 0 (e.g. one the hard-safety
        # pre-filter already pruned). Fail closed instead of overriding that.
        return result(PROJ_INSUFFICIENT_SUPPORT)
    p = _clip_scale(q, lo, hi)
    if p is None:
        return result(PROJ_INSUFFICIENT_SUPPORT)
    return result(PROJ_OK, p)


def cpsat_verify_selection(
    specs: Sequence[ActionConstraintSpec],
    candidates: Sequence[str],
    disabled: Sequence[str],
    selected: str,
) -> str:
    """Run OR-Tools CP-SAT on the 0-1 encoding of the hard-mask constraints.

    Model: one Bool per candidate, exactly one true, disabled ones 0, every mutex pair at most
    one, the selected one 1. Feasible -> FV_VERIFIED. Infeasible -> FV_REFUTED. Never claims
    verification when the solver did not run. Probability bounds are continuous and are NOT
    part of this model (they are enforced numerically by the projection).
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        logger.warning("OR-Tools is not installed; CP-SAT verification was NOT run (%s).", FV_UNAVAILABLE)
        return FV_UNAVAILABLE
    try:
        names = [norm_action(c) for c in candidates]
        model = cp_model.CpModel()
        x = {n: model.NewBoolVar(f"x_{i}") for i, n in enumerate(names)}
        model.Add(sum(x.values()) == 1)
        for d in disabled:
            model.Add(x[norm_action(d)] == 0)
        for s in specs:
            if s.kind == KIND_MUTEX:
                members = [x[a] for a in s.actions if a in x]
                if len(members) > 1:
                    model.Add(sum(members) <= 1)
        model.Add(x[norm_action(selected)] == 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return FV_VERIFIED
        if status == cp_model.INFEASIBLE:
            return FV_REFUTED
        return f"cp_sat_error:status_{solver.StatusName(status)}"
    except Exception as exc:  # explicit, never a silent pass
        logger.error("CP-SAT verification raised %s: %s", type(exc).__name__, exc)
        return f"cp_sat_error:{type(exc).__name__}"


def cpsat_available() -> bool:
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        return False
    return True
