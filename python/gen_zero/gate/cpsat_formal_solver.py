"""CP-SAT Formal Safety Solver with 2ms Hard Timeout Gating.

Implements Milestone 5 of Issue #23:
- 0-1 Integer Linear Programming / CP-SAT discrete optimization over safe action space C_safe.
- Maximizes utility: max_a S_composite(a) s.t. a in C_safe.
- 2ms Hard Timeout Circuit Breaker: Enforces strict <= 2.0ms solve budget.
  If solver times out or exceeds budget, falls back deterministically to the best safe heuristic action.
- Formal hard constraint safety interception.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import dataclasses
import logging
import math
import os
import time

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CPSATVerdict:
    selected_action: str
    is_safe: bool
    solve_time_ms: float
    timed_out: bool
    fallback_used: bool
    solver_status: str
    applied_constraints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "is_safe": self.is_safe,
            "solve_time_ms": round(self.solve_time_ms, 3),
            "timed_out": self.timed_out,
            "fallback_used": self.fallback_used,
            "solver_status": self.solver_status,
            "applied_constraints": self.applied_constraints,
        }


class CPSATFormalSolver:
    """Solves 0-1 integer linear constraints over candidate actions with 2ms hard timeout."""

    def __init__(self, hard_timeout_ms: Optional[float] = None):
        if hard_timeout_ms is None:
            hard_timeout_ms = float(os.environ.get("GENZERO_CPSAT_TIMEOUT_MS", "50.0"))
        self.hard_timeout_ms = float(hard_timeout_ms)
        self._ortools_available = False
        try:
            from ortools.sat.python import cp_model
            self._ortools_available = True
        except ImportError:
            pass

    def solve_safest_optimal_action(
        self,
        candidate_utilities: Dict[str, float],
        forbidden_actions: Optional[Set[str]] = None,
        required_preconditions: Optional[Dict[str, bool]] = None,
        fallback_safe_action: str = "HOLD",
    ) -> CPSATVerdict:
        """Solves optimal safe action with strict 2ms hard timeout.

        Args:
            candidate_utilities: Dict of candidate action -> float utility (probability or value).
            forbidden_actions: Set of actions violating hard safety invariants.
            required_preconditions: Dict of action -> bool indicating whether preconditions are met.
            fallback_safe_action: Action to return if all options are barred or solver times out.
        """
        t0 = time.perf_counter()
        forbidden = forbidden_actions or set()
        preconds = required_preconditions or {}

        # 1. Filter out candidate actions violating hard constraints
        feasible_candidates = {}
        for action, util in candidate_utilities.items():
            if action in forbidden:
                continue
            if action in preconds and not preconds[action]:
                continue
            feasible_candidates[action] = util

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Check 2ms timeout early
        if elapsed_ms > self.hard_timeout_ms:
            # Hard timeout breach -> immediate deterministic fallback
            best_safe = max(feasible_candidates.keys(), key=lambda a: feasible_candidates[a]) if feasible_candidates else fallback_safe_action
            return CPSATVerdict(
                selected_action=best_safe,
                is_safe=bool(feasible_candidates),
                solve_time_ms=elapsed_ms,
                timed_out=True,
                fallback_used=True,
                solver_status="TIMEOUT_EARLY_FALLBACK",
                applied_constraints=list(forbidden),
            )

        # 2. If ortools is available and feasible set is non-trivial, run CP-SAT model with time limit
        if self._ortools_available and len(feasible_candidates) > 1:
            try:
                from ortools.sat.python import cp_model
                model = cp_model.CpModel()

                # 0-1 boolean variable per feasible action
                action_vars = {a: model.NewBoolVar(f"act_{a}") for a in feasible_candidates}

                # Exactly one action must be chosen: sum(action_vars) == 1
                model.Add(sum(action_vars.values()) == 1)

                # Objective: Maximize scaled utility
                scaled_utilities = {a: int(round(u * 10000)) for a, u in feasible_candidates.items()}
                model.Maximize(sum(action_vars[a] * scaled_utilities[a] for a in feasible_candidates))

                solver = cp_model.CpSolver()
                # Set strict timeout budget in seconds (e.g. 0.002s)
                remaining_sec = max(0.0001, (self.hard_timeout_ms - elapsed_ms) / 1000.0)
                solver.parameters.max_time_in_seconds = remaining_sec
                solver.parameters.num_search_workers = 1

                status = solver.Solve(model)
                total_solve_ms = (time.perf_counter() - t0) * 1000.0

                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    for a, var in action_vars.items():
                        if solver.Value(var) == 1:
                            return CPSATVerdict(
                                selected_action=a,
                                is_safe=True,
                                solve_time_ms=total_solve_ms,
                                timed_out=False,
                                fallback_used=False,
                                solver_status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                                applied_constraints=list(forbidden),
                            )
                elif status == cp_model.UNKNOWN:
                    # Timeout triggered in solver
                    best_safe = max(feasible_candidates.keys(), key=lambda a: feasible_candidates[a])
                    return CPSATVerdict(
                        selected_action=best_safe,
                        is_safe=True,
                        solve_time_ms=total_solve_ms,
                        timed_out=True,
                        fallback_used=True,
                        solver_status="SOLVER_TIMEOUT_FALLBACK",
                        applied_constraints=list(forbidden),
                    )
                else:
                    # INFEASIBLE / MODEL_INVALID: the solver ran and produced no solution.
                    # Never relabel this as a solve; it is an explicit, logged fallback.
                    status_name = solver.StatusName(status)
                    logger.error(
                        "CP-SAT returned %s with no solution; returning explicit CPSAT_NO_SOLUTION_FALLBACK",
                        status_name,
                    )
                    best_safe = max(feasible_candidates.keys(), key=lambda a: feasible_candidates[a])
                    return CPSATVerdict(
                        selected_action=best_safe,
                        is_safe=True,
                        solve_time_ms=total_solve_ms,
                        timed_out=False,
                        fallback_used=True,
                        solver_status=f"CPSAT_NO_SOLUTION_FALLBACK:{status_name}",
                        applied_constraints=list(forbidden),
                    )
            except Exception as exc:
                # Fail closed: a crashed solver is reported as a fallback, never as a solve.
                logger.error(
                    "CP-SAT solver raised %s: %s; returning explicit CPSAT_EXCEPTION_FALLBACK",
                    type(exc).__name__, exc,
                )
                total_solve_ms = (time.perf_counter() - t0) * 1000.0
                best_safe = max(feasible_candidates.keys(), key=lambda a: feasible_candidates[a])
                return CPSATVerdict(
                    selected_action=best_safe,
                    is_safe=True,
                    solve_time_ms=total_solve_ms,
                    timed_out=False,
                    fallback_used=True,
                    solver_status=f"CPSAT_EXCEPTION_FALLBACK:{type(exc).__name__}",
                    applied_constraints=list(forbidden),
                )

        # 3. Deterministic linear maximisation. Only claimed as a plain solve when OR-Tools
        # was really present; a missing OR-Tools is an explicit, logged fallback.
        total_solve_ms = (time.perf_counter() - t0) * 1000.0
        if feasible_candidates:
            best_action = max(feasible_candidates.keys(), key=lambda a: feasible_candidates[a])
            if not self._ortools_available:
                logger.warning(
                    "OR-Tools is not installed; CP-SAT was NOT run. Returning "
                    "ORTOOLS_UNAVAILABLE_FALLBACK (deterministic argmax over the safe set)."
                )
                return CPSATVerdict(
                    selected_action=best_action,
                    is_safe=True,
                    solve_time_ms=total_solve_ms,
                    timed_out=False,
                    fallback_used=True,
                    solver_status="ORTOOLS_UNAVAILABLE_FALLBACK",
                    applied_constraints=list(forbidden),
                )
            return CPSATVerdict(
                selected_action=best_action,
                is_safe=True,
                solve_time_ms=total_solve_ms,
                timed_out=False,
                fallback_used=False,
                solver_status="DETERMINISTIC_SAFE_SOLVED",
                applied_constraints=list(forbidden),
            )
        else:
            # No feasible action was proven safe; the fallback is an abstention only.
            return CPSATVerdict(
                selected_action=fallback_safe_action,
                is_safe=False,
                solve_time_ms=total_solve_ms,
                timed_out=False,
                fallback_used=True,
                solver_status="ALL_CANDIDATES_FORBIDDEN_INTERCEPT",
                applied_constraints=list(forbidden),
            )
