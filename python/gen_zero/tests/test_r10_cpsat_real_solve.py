"""Release gate: exercise an actual OR-Tools CP-SAT solve."""

from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver


def test_real_cpsat_selects_best_permitted_action():
    solver = CPSATFormalSolver(hard_timeout_ms=1000.0)
    assert solver._ortools_available

    verdict = solver.solve_safest_optimal_action(
        {"SAFE_READ": 0.4, "SAFE_WRITE": 0.9, "DANGEROUS_DROP": 1.0},
        forbidden_actions={"DANGEROUS_DROP"},
        fallback_safe_action="HOLD",
    )

    assert verdict.fallback_used is False
    assert verdict.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert verdict.is_safe is True
    assert verdict.selected_action == "SAFE_WRITE"
    assert verdict.selected_action != "DANGEROUS_DROP"
