# gen-zero-gate

Formal safety interlocks for Gen-Zero: a 0-1 ILP linear constraint compiler, a
4-tier `PolicyGate` state machine, and the Spec 25
geometric/sheaf-cohomology fast gate.

## Architecture

- `constraint`: 0-1 ILP linear constraint compiler and representation.
- `error`: gate error types (`GateError`).
- `policy`: the four-tier `PolicyGate` state machine and its arbiters.
- `risk`: maps a semantic risk probability (produced elsewhere, e.g. the
  Python classifier behind the `zero` bridge) to a `PolicyGate` tier. Never
  reads request text itself.
- `sheaf_gate`: Spec 25 geometric + sheaf-cohomology deterministic fast gate.
  A single operator-based quadratic relaxation loop supports weighted
  coboundaries and affine dynamics windows with soft observations, at
  `O(T*d^2 + P*d)` work per evaluation, without forming a dense global matrix.
- `two_stage`: two-stage dual-track answerability gateway. Stage 1 scores
  (`score_diff = null_score - best_span_score`) decide alone outside the
  ambiguity band; inside it a `CausalVerifier` returns three teacher
  projections and the gate computes `tri_sim`, `p_same_meaning` and keeps the
  candidate iff `tri_sim >= threshold`. Every stage 2 fault is a `GateError`.
  The production verifier is `gen_zero_model::TriTeacherPairDecider`, reached
  through `gen-zero qa-gate` and the `zero` verb `qa_gate` (MCP and HTTP).

## Key exports

- `LinearConstraint`, `RuleId`: ILP constraint representation.
- `GateError`: crate error type.
- `GateVerdict`, `PolicyGate`, `PolicyTier`: the policy gate state machine.
- `SemanticRisk`: semantic risk probability to tier mapping.
- `AcceptedState`, `Budget`, `CandidateState`, `CertifiedCandidate`,
  `DynamicsStep`, `EnergyRoseKind`, `ExplicitMatrixProblem`, `GeometryGate`,
  `LaplacianHeatFlowGate`, `ManifoldGuard`, `Pin`, `Reject`, `RelaxationStatus`,
  `SheafOperator`, `SheafProblem`, `WindowDynamicsProblem`: the sheaf/geometry
  gate and its problem types.

## Dependencies

- `gen-zero-core`: base types and traits.
- `gen-zero-lod`: manifold and geometry types used by `sheaf_gate`.
