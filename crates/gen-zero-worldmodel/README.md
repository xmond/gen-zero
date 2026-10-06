# gen-zero-worldmodel

Latent dynamics for Gen-Zero: Contact Hamiltonian integrators with Strang
splitting and symplectic (Stormer-Verlet) integrators.

## Architecture

- `compression` (private): versioned, dimension-bound zstd storage for finite
  f32 state streams, used by the contact and symplectic trajectory
  compression helpers.
- `conformal_dynamics`: `ConformalWorldModelDynamics`, `WorldModelDynamics` on
  the contact manifold: the symplectic well plus conformal damping `gamma >= 0`
  through `ContactIntegrator`. This is the `contact` dynamics of
  gen-zero-service.
- `contact`: Contact Hamiltonian dynamics and Strang splitting integrators.
  Models dissipative open systems on contact manifolds (`[q, p, s]^T`, 2N+1
  dimensions), with symmetric operator splitting, phase-volume contraction,
  and gauge clamping to eliminate floating-point drift.
- `dynamics`: `LatentDynamicsWorldModel`, implementing `WorldModelDynamics`.
  Its error type is `CoreError`, so it plugs directly into the planner's
  `Arc<dyn WorldModelDynamics<Error = CoreError>>`; failures are raised as
  `WorldModelError` and converted via `From<WorldModelError> for CoreError`.
- `error`: crate error type (`WorldModelError`).
- `symplectic`: symplectic (Stormer-Verlet) integrator for separable
  Hamiltonians `H(q, p) = 1/2 ||p||^2 + V(q)`. Conservative counterpart of the
  contact integrator: phase volume preserved exactly, energy error bounded,
  no heap allocation.
- `symplectic_dynamics`: `SymplecticWorldModelDynamics`, `WorldModelDynamics`
  on a Hamiltonian phase space. Reads the latent `z` as `(q, p)` with
  `q = z[..512]`, `p = z[512..]`; an action selects the Hamiltonian of its step.

## Key exports

- `compress_contact_trajectory_zstd`, `decompress_contact_trajectory_zstd`,
  `ContactIntegrator`, `ContactState`: contact Hamiltonian dynamics.
- `LatentDynamicsWorldModel`, `DONE_NORM`, `SAFETY_SOURCE_NORM_MARGIN`: the
  default latent dynamics world model.
- `WorldModelError`: crate error type.
- `compress_phase_trajectory_zstd`, `decompress_phase_trajectory_zstd`,
  `PhaseState`, `SymplecticIntegrator`: symplectic integrator.
- `PhaseTransition`, `SymplecticWorldModelDynamics`: symplectic world model
  dynamics.
- `ContactTransition`, `ConformalWorldModelDynamics`: conformal symplectic
  (contact) world model dynamics.

## Dependencies

- `gen-zero-core`: base types and traits (`WorldModelDynamics`, `CoreError`).
