//! gen-zero-core fundamental trait contracts for all subsystems.

use crate::types::{ActionId, FullLatent};

/// Per-step safety reading a world model attaches to a transition.
///
/// `safe_prob` lies in `[0, 1]`. `calibrated` is true only when the number was fit
/// against observed outcomes; a margin derived from the model's own termination rule
/// is not calibrated. `source` names what produced it, so callers can refuse a
/// source they do not trust.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafetyEstimate {
    pub safe_prob: f32,
    pub calibrated: bool,
    pub source: &'static str,
}

/// Latent Dynamics Trait modeling forward causal transitions: T(s, a) -> (s', r, d).
pub trait WorldModelDynamics: Send + Sync {
    type Error;

    /// State-dependent subset of candidate IDs, in candidate order. Implementations
    /// must be deterministic and may only remove candidates. Empty means dead end.
    /// Unconstrained dynamics explicitly permit the entire candidate frame.
    fn allowed_actions(
        &self,
        _state: &FullLatent,
        candidates: &[ActionId],
    ) -> Result<Vec<ActionId>, Self::Error> {
        Ok(candidates.to_vec())
    }

    /// Single-step forward transition in continuous latent space.
    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error>;

    /// Batch transition kernel for parallel MCTS rollouts and spectral projections (zero-alloc).
    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), Self::Error>;

    /// Safety of the transition that produced `next_state`. `None` means the model
    /// has no safety estimate; callers that need one must refuse, not assume safe.
    fn safety_estimate(
        &self,
        _next_state: &FullLatent,
        _reward: f32,
        _done: bool,
    ) -> Option<SafetyEstimate> {
        None
    }
}

/// Latent flow map `z <- Phi(z; context)` advanced one step in place.
///
/// Contract for rollout loops that repeat a step many times:
/// * `step_in_place` either leaves a finite latent or returns an error. It never
///   hands back NaN or Inf as a valid state.
/// * `contraction_rate` is the spectral radius of the linearised one-step map. Its
///   logarithm is the top Lyapunov exponent per step. Below 1 means perturbations
///   shrink asymptotically. Exactly 1 means a conservative flow.
/// * `is_dissipative` is true only when the implementor can prove the rate is
///   strictly below 1. Unknown or borderline cases must answer false.
pub trait LatentContraction: Send + Sync {
    type Error;

    /// Advance `z` by one step. `context` parameterises the flow (for example an
    /// attractor centre). Autonomous flows may ignore it.
    fn step_in_place(&self, z: &mut FullLatent, context: &FullLatent) -> Result<(), Self::Error>;

    /// Spectral radius of the linearised one-step map. Non-finite when undecidable.
    fn contraction_rate(&self) -> f64;

    /// True when the step strictly contracts phase-space perturbations.
    fn is_dissipative(&self) -> bool;

    /// Top Lyapunov exponent per step: `ln(contraction_rate)`.
    #[inline]
    fn lyapunov_exponent(&self) -> f64 {
        self.contraction_rate().ln()
    }
}

/// Bijective Lossless Invertible Encoder Trait: s <-> z with rigorous reconstruction guarantees.
pub trait LosslessInvertibleEncoder: Send + Sync {
    type Source;
    type Error;

    /// Lossless projection to latent representation: s -> z.
    fn encode(&self, source: &Self::Source) -> Result<FullLatent, Self::Error>;

    /// Exact inverse reconstruction from latent representation: z -> s.
    fn decode(&self, latent: &FullLatent) -> Result<Self::Source, Self::Error>;
}

/// Cognitive Graph Fact Provider Trait.
/// Exposes validated epistemic facts to formal Datalog / CP-SAT gate without circular dependencies.
pub trait GraphFactProvider: Send + Sync {
    type DepIter<'a>: Iterator<Item = (u64, u64)> + 'a
    where
        Self: 'a;

    /// Batch iterator yielding only Validated or Axiomatic dependency edges.
    fn active_validated_dependencies<'a>(&'a self) -> Self::DepIter<'a>;

    /// Visitor pattern method yielding active validated dependency edges without heap allocation.
    fn for_each_validated_dependency<F: FnMut(u64, u64)>(&self, mut f: F) {
        for edge in self.active_validated_dependencies() {
            f(edge.0, edge.1);
        }
    }

    /// Check if target entity has active revocation.
    fn is_revoked(&self, entity_id: u64) -> bool;

    /// Check privilege bitflags assigned to the target agent.
    fn has_privilege(&self, agent_id: u64, privilege: u32) -> bool;
}

impl GraphFactProvider for () {
    type DepIter<'a> = std::iter::Empty<(u64, u64)>;
    #[inline]
    fn active_validated_dependencies<'a>(&'a self) -> Self::DepIter<'a> {
        std::iter::empty()
    }
    #[inline]
    fn is_revoked(&self, _entity_id: u64) -> bool {
        false
    }
    #[inline]
    fn has_privilege(&self, _agent_id: u64, _privilege: u32) -> bool {
        false
    }
}
