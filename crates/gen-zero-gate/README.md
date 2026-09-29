# gen-zero-gate

Formal safety interlocks for Gen-Zero: a 0-1 ILP linear constraint compiler, a
4-tier `PolicyGate` state machine, dual-track promotion gates, and the Spec 25
geometric/sheaf-cohomology fast gate.

## Architecture

- `constraint`: 0-1 ILP linear constraint compiler and representation.
- `dual_track`: dual-track safety and capability ladder gate. Track A is a
  formal safety gate with zero-tolerance hard stop; Track B is a capability
  ladder requiring >= 99.5% retention against a frozen golden test suite.
- `error`: gate error types (`GateError`).
- `policy`: the four-tier `PolicyGate` state machine and its arbiters.
- `risk`: maps a semantic risk probability (produced elsewhere, e.g. the
  Python classifier behind the `zero` bridge) to a `PolicyGate` tier. Never
  reads request text itself.
- `sheaf_gate`: Spec 25 geometric + sheaf-cohomology deterministic fast gate.
  A single operator-based quadratic relaxation loop supports weighted
  coboundaries and affine dynamics windows with soft observations, at
  `O(T*d^2 + P*d)` work per evaluation, without forming a dense global matrix.

## Key exports

- `LinearConstraint`, `RuleId`: ILP constraint representation.
- `DualTrackVerifier`, `SafetyAuditReport`: dual-track promotion verification.
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
