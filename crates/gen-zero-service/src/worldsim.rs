//! World-model rollouts behind the `simulate`, `what_if` and `audit` verbs,
//! and the latent planner modes (`mcts`, `mpc_cem`, `astar`) of `decide`.
//!
//! A request picks its dynamics with `dynamics` (see [`parse_dynamics`]):
//! `residual` (default) is [`LatentDynamicsWorldModel`], a fixed residual map
//! with a hash-phased action term. `symplectic` is
//! [`SymplecticWorldModelDynamics`], a Hamiltonian well stepped by
//! Stormer-Verlet, whose steps also report their energy ledger. `contact` is
//! [`ConformalWorldModelDynamics`], the same well on the contact manifold with
//! a conformal damping rate `damping` (`gamma >= 0`): `gamma = 0` is the
//! symplectic flow, `gamma > 0` contracts each `(q_i, p_i)` pair by
//! `exp(-2 gamma dt)` per step, and the ledger reports both. None of the three
//! is **trained or calibrated**, so every output carries the dynamics'
//! provenance tag and no verdict in this module can be an approval. The
//! numbers describe a prior, not an environment.
//!
//! Inputs are numeric only. A latent state must be exactly [`LATENT_DIM`]
//! finite numbers: there is no text encoder into the latent space, so text is
//! refused and a short vector is refused, never padded or truncated.
//! Out-of-range horizons are refused, never clamped.
//!
//! Hazard means the model reported `done` (the state norm left the stable
//! ball). The model has no safety probability, and none is invented here.

use crate::cognitive::Rejection;
use crate::zero::{action_id, tier_name};
use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{PolicyGate, PolicyTier};
use gen_zero_lod::LodGraph;
use gen_zero_planner::{AStarEngine, AStarGoal, MctsEngine, MpcCemEngine, PlanningEngine};
use gen_zero_worldmodel::{
    ConformalWorldModelDynamics, LatentDynamicsWorldModel, SymplecticWorldModelDynamics,
};
use serde_json::{json, Value};

/// Width of a latent state (`FullLatent`).
pub const LATENT_DIM: usize = 1024;
/// Horizon used when a request names none (`simulate` uses its plan length).
pub const DEFAULT_HORIZON: usize = 5;
/// Largest horizon or plan length one request may ask for.
pub const MAX_HORIZON: usize = 256;
/// Most candidates one request may carry (`LocalActionFrame` capacity).
pub const MAX_CANDIDATES: usize = 16;
/// Which simulator produced a step. Stamped on every output of this module:
/// [`PROVENANCE`] for `residual`, [`PROVENANCE_SYMPLECTIC`] for `symplectic`,
/// [`PROVENANCE_CONTACT`] for `contact`.
pub const PROVENANCE: &str = "latent_residual_dynamics_untrained";
pub const PROVENANCE_SYMPLECTIC: &str = "symplectic_hamiltonian_dynamics_untrained";
pub const PROVENANCE_CONTACT: &str = "conformal_symplectic_contact_dynamics_untrained";
/// Values `dynamics` accepts.
pub const DYNAMICS_NAMES: [&str; 3] = ["residual", "symplectic", "contact"];
/// `_meta.engine` of the world-model verbs.
pub const ENGINE_WORLDMODEL: &str = "latent_dynamics_world_model";

const STAGE: &str = "worldmodel";

/// Result of a world-model verb: the `_meta` object and the one-line summary.
pub type WorldResult = Result<(Value, String), Rejection>;

fn invalid(detail: impl Into<String>) -> Rejection {
    Rejection::invalid(STAGE, detail)
}

fn failure(code: &str, detail: impl Into<String>) -> Rejection {
    Rejection {
        code: code.to_string(),
        stage: STAGE.to_string(),
        detail: detail.into(),
        http_status: 422,
    }
}

fn numerical(e: CoreError) -> Rejection {
    failure(
        "NumericalInstability",
        format!("world model transition failed: {e}"),
    )
}

/// The planner engine behind a latent `decide` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerMode {
    Mcts,
    MpcCem,
    AStar,
}

impl PlannerMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mcts => "mcts",
            Self::MpcCem => "mpc_cem",
            Self::AStar => "astar",
        }
    }
}

/// The world model a request rolls forward on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicsKind {
    Residual,
    Symplectic,
    /// Conformal symplectic flow on the contact manifold (`gamma >= 0`).
    Contact,
}

impl DynamicsKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Residual => "residual",
            Self::Symplectic => "symplectic",
            Self::Contact => "contact",
        }
    }

    pub fn provenance(self) -> &'static str {
        match self {
            Self::Residual => PROVENANCE,
            Self::Symplectic => PROVENANCE_SYMPLECTIC,
            Self::Contact => PROVENANCE_CONTACT,
        }
    }

    fn prior_label(self) -> &'static str {
        match self {
            Self::Residual => "untrained latent prior",
            Self::Symplectic => "untrained symplectic Hamiltonian prior",
            Self::Contact => "untrained conformal symplectic contact prior",
        }
    }

    /// Whether a rollout keeps every `(q, p)` state for `phase_trajectory`.
    fn has_phase_space(self) -> bool {
        matches!(self, Self::Symplectic | Self::Contact)
    }
}

/// `dynamics`: absent means `residual`. Anything but one of
/// [`DYNAMICS_NAMES`] is refused, never mapped to a default.
pub fn parse_dynamics(value: Option<&Value>) -> Result<DynamicsKind, Rejection> {
    let Some(value) = value else {
        return Ok(DynamicsKind::Residual);
    };
    match value.as_str() {
        Some("residual") => Ok(DynamicsKind::Residual),
        Some("symplectic") => Ok(DynamicsKind::Symplectic),
        Some("contact") => Ok(DynamicsKind::Contact),
        _ => Err(invalid(format!(
            "`dynamics` must be one of {DYNAMICS_NAMES:?}, got {value}"
        ))),
    }
}

/// The dynamics of a request together with its parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsSpec {
    pub kind: DynamicsKind,
    /// Conformal damping rate `gamma` of `contact`. `None` for the other kinds.
    pub damping: Option<f32>,
}

impl DynamicsSpec {
    /// The one spec with no request-level parameters: `residual`.
    pub const RESIDUAL: Self = Self {
        kind: DynamicsKind::Residual,
        damping: None,
    };
}

/// `dynamics` plus its parameters. `damping` (the conformal rate `gamma`) is
/// read only for `contact`; absent means
/// [`ConformalWorldModelDynamics::DEFAULT_DAMPING`]. With any other dynamics
/// the field is refused, as is a negative, non-finite or non-numeric value.
pub fn parse_dynamics_spec(args: &Value) -> Result<DynamicsSpec, Rejection> {
    let kind = parse_dynamics(args.get("dynamics"))?;
    let damping = match (kind, args.get("damping")) {
        (DynamicsKind::Contact, None) => Some(ConformalWorldModelDynamics::DEFAULT_DAMPING),
        (DynamicsKind::Contact, Some(value)) => {
            let gamma = value
                .as_f64()
                .ok_or_else(|| invalid(format!("`damping` must be a number >= 0, got {value}")))?;
            let narrowed = gamma as f32;
            if !narrowed.is_finite() || narrowed < 0.0 {
                return Err(invalid(format!(
                    "`damping` must be a finite number >= 0 (a conformal rate gamma), got {value}"
                )));
            }
            // A positive rate that rounds to zero in f32 would run the conservative flow
            // while the request asked for damping: refuse instead of degrading silently.
            if gamma > 0.0 && narrowed == 0.0 {
                return Err(invalid(format!(
                    "`damping` {value} is positive but underflows f32 to 0; use 0 or a rate \
                     the integrator can represent"
                )));
            }
            Some(narrowed)
        }
        (_, None) => None,
        (other, Some(_)) => {
            return Err(invalid(format!(
                "`damping` is read only with `dynamics: \"contact\"`, not `{}`",
                other.name()
            )))
        }
    };
    Ok(DynamicsSpec { kind, damping })
}

/// A built world model of any kind.
pub enum WorldDynamics {
    Residual(LatentDynamicsWorldModel),
    Symplectic(SymplecticWorldModelDynamics),
    Contact(ConformalWorldModelDynamics),
}

/// Contact-manifold bookkeeping of one step.
#[derive(Clone, Copy, Debug)]
struct ContactStep {
    /// Contact action `s` after the step (it starts at zero, it has no latent slot).
    action: f32,
    /// `exp(-2 gamma dt)`: the exact area factor of each `(q_i, p_i)` pair.
    volume_factor_per_pair: f64,
}

/// One step, with the Hamiltonian ledger when the dynamics have one.
struct StepOut {
    next: FullLatent,
    reward: f32,
    done: bool,
    /// `(H_a before, H_a after, drift)` of the step's action (symplectic and contact).
    energy: Option<(f64, f64, f64)>,
    /// Contact-only ledger.
    contact: Option<ContactStep>,
}

impl WorldDynamics {
    /// Builds the model. A `contact` damping outside the integrator's stability
    /// band is refused here, at load time, never clamped.
    pub fn new(spec: DynamicsSpec) -> Result<Self, Rejection> {
        match spec.kind {
            DynamicsKind::Residual => Ok(Self::Residual(LatentDynamicsWorldModel::default())),
            DynamicsKind::Symplectic => {
                Ok(Self::Symplectic(SymplecticWorldModelDynamics::default()))
            }
            DynamicsKind::Contact => {
                let gamma = spec.damping.ok_or_else(|| {
                    invalid("`contact` dynamics need a damping rate; none was resolved")
                })?;
                let model = ConformalWorldModelDynamics::new(
                    ConformalWorldModelDynamics::DEFAULT_DT,
                    ConformalWorldModelDynamics::DEFAULT_STIFFNESS,
                    ConformalWorldModelDynamics::DEFAULT_ACTION_SCALE,
                    gamma,
                )
                .map_err(|e| {
                    invalid(format!(
                        "`damping` {gamma} is refused by the contact integrator: {e}"
                    ))
                })?;
                Ok(Self::Contact(model))
            }
        }
    }

    pub fn kind(&self) -> DynamicsKind {
        match self {
            Self::Residual(_) => DynamicsKind::Residual,
            Self::Symplectic(_) => DynamicsKind::Symplectic,
            Self::Contact(_) => DynamicsKind::Contact,
        }
    }

    /// The model as the planner crate's trait object.
    pub fn as_dyn(&self) -> &dyn WorldModelDynamics<Error = CoreError> {
        match self {
            Self::Residual(m) => m,
            Self::Symplectic(m) => m,
            Self::Contact(m) => m,
        }
    }

    fn step(&self, state: &FullLatent, id: ActionId) -> Result<StepOut, Rejection> {
        match self {
            Self::Residual(m) => {
                let (next, reward, done) = m.step(state, id).map_err(numerical)?;
                Ok(StepOut {
                    next,
                    reward,
                    done,
                    energy: None,
                    contact: None,
                })
            }
            Self::Symplectic(m) => {
                let t = m.transition(state, id).map_err(|e| numerical(e.into()))?;
                let energy = Some((t.energy_before, t.energy_after, t.energy_drift()));
                Ok(StepOut {
                    next: t.next,
                    reward: t.reward,
                    done: t.done,
                    energy,
                    contact: None,
                })
            }
            Self::Contact(m) => {
                let t = m.transition(state, id).map_err(|e| numerical(e.into()))?;
                let energy = Some((t.energy_before, t.energy_after, t.energy_drift()));
                Ok(StepOut {
                    next: t.next,
                    reward: t.reward,
                    done: t.done,
                    energy,
                    contact: Some(ContactStep {
                        action: t.contact_action,
                        volume_factor_per_pair: t.phase_volume_factor_per_pair,
                    }),
                })
            }
        }
    }

    /// Contact-manifold parameters of the built model, for the ledger.
    fn contact_ledger_header(&self) -> Option<Value> {
        let Self::Contact(m) = self else {
            return None;
        };
        Some(json!({
            "damping_gamma": m.gamma(),
            "integrator_gamma": m.integrator_gamma(),
            "dt": m.dt(),
            "stiffness": m.stiffness(),
            "phase_volume_factor_per_pair": m.phase_volume_factor_per_pair(),
            "phase_volume_factor_projection": m.phase_volume_factor(),
            "contraction_rate": m.contraction_rate(),
            "conservative": m.gamma() == 0.0,
        }))
    }
}

/// Parse a latent state: a JSON array of exactly [`LATENT_DIM`] finite
/// numbers, or `{"latent": [...]}`.
pub fn parse_latent(value: Option<&Value>, key: &str) -> Result<FullLatent, Rejection> {
    let value = value.ok_or_else(|| {
        invalid(format!(
            "`{key}` is required: a JSON array of {LATENT_DIM} numbers"
        ))
    })?;
    let items = match value {
        Value::Array(items) => items,
        Value::Object(fields) => fields
            .get("latent")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("`{key}` object needs a `latent` array")))?,
        Value::String(_) => {
            return Err(invalid(format!(
                "`{key}` is text: there is no text encoder into the latent space, \
                 supply {LATENT_DIM} numbers"
            )))
        }
        _ => {
            return Err(invalid(format!(
                "`{key}` must be an array of {LATENT_DIM} numbers"
            )))
        }
    };
    if items.len() != LATENT_DIM {
        return Err(invalid(format!(
            "`{key}` has {} values, the latent space is {LATENT_DIM}-wide \
             (nothing is padded or truncated)",
            items.len()
        )));
    }
    let mut values = Vec::with_capacity(LATENT_DIM);
    for (i, item) in items.iter().enumerate() {
        let number = item
            .as_f64()
            .ok_or_else(|| invalid(format!("`{key}`[{i}] is not a number")))?;
        let narrowed = number as f32;
        if !narrowed.is_finite() {
            return Err(invalid(format!("`{key}`[{i}] is not a finite f32")));
        }
        values.push(narrowed);
    }
    FullLatent::from_slice(&values).map_err(|e| invalid(e.to_string()))
}

/// Parse a list of action names: a non-empty array of non-empty strings, at
/// most `max` long. `distinct` refuses repeats. Anything else is refused;
/// nothing is filtered out.
pub fn parse_names(
    value: Option<&Value>,
    key: &str,
    max: usize,
    distinct: bool,
) -> Result<Vec<String>, Rejection> {
    let items = value
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("`{key}` is required: a JSON array of action names")))?;
    if items.is_empty() {
        return Err(invalid(format!("`{key}` is empty")));
    }
    if items.len() > max {
        return Err(invalid(format!(
            "`{key}` has {} entries, the limit is {max}",
            items.len()
        )));
    }
    let mut names: Vec<String> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let name = item
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid(format!("`{key}`[{i}] is not a non-empty string")))?;
        if distinct && names.iter().any(|n| n == name) {
            return Err(invalid(format!("`{key}` repeats `{name}`")));
        }
        names.push(name.to_string());
    }
    Ok(names)
}

/// Horizon: absent means `default`; present must be an integer in
/// `1..=MAX_HORIZON`. Booleans, floats and negatives are refused.
pub fn parse_horizon(value: Option<&Value>, default: usize) -> Result<usize, Rejection> {
    let Some(value) = value else {
        return Ok(default);
    };
    let horizon = value
        .as_u64()
        .ok_or_else(|| invalid("`horizon` must be a positive integer"))?;
    if horizon == 0 || horizon > MAX_HORIZON as u64 {
        return Err(invalid(format!(
            "`horizon` {horizon} is outside 1..={MAX_HORIZON}"
        )));
    }
    Ok(horizon as usize)
}

struct Rollout {
    kind: DynamicsKind,
    trajectory: Vec<Value>,
    /// Full `(q, p)` after each step. Kept only when the caller asks for it.
    phase_states: Vec<FullLatent>,
    /// `(H_a before, H_a after)` per step, symplectic and contact only.
    energies: Vec<(f64, f64)>,
    /// Contact action `s` per step, contact only.
    contact_actions: Vec<f32>,
    /// Parameters of the contact model, contact only.
    contact_header: Option<Value>,
    /// Every step played the same action, so one Hamiltonian spans the run.
    single_action: bool,
    final_state: FullLatent,
    cumulative_return: f64,
    termination_step: Option<usize>,
    first_hazard_step: Option<usize>,
    survival_horizon: usize,
    worst_tier: PolicyTier,
    initial_energy: f64,
    final_energy: f64,
}

impl Rollout {
    fn summary(&self, with_final_state: bool) -> Value {
        let mut out = json!({
            "steps_simulated": self.trajectory.len(),
            "survival_horizon": self.survival_horizon,
            "cumulative_return": self.cumulative_return,
            "terminated_early": self.termination_step.is_some(),
            "termination_step": self.termination_step,
            "is_safe": self.first_hazard_step.is_none(),
            "first_hazard_step": self.first_hazard_step,
            "worst_tier": tier_name(self.worst_tier),
            "final_state_norm": self.final_state.l2_norm(),
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "energy_change": self.final_energy - self.initial_energy,
            "energy_increase": self.kind.has_phase_space() && self.final_energy > self.initial_energy,
            "hazard_detected": self.first_hazard_step.is_some()
                || (self.kind.has_phase_space() && self.final_energy > self.initial_energy),
            "admissible": self.first_hazard_step.is_none()
                && (!self.kind.has_phase_space() || self.final_energy <= self.initial_energy)
                && self.worst_tier != PolicyTier::Tier3HardStop,
            "trajectory": self.trajectory,
        });
        if with_final_state {
            out["final_state"] = json!(self.final_state.as_slice());
        }
        match self.kind {
            DynamicsKind::Symplectic => out["energy_ledger"] = self.energy_ledger(),
            DynamicsKind::Contact => out["energy_ledger"] = self.contact_ledger(),
            DynamicsKind::Residual => {}
        }
        if !self.phase_states.is_empty() {
            let half = LATENT_DIM / 2;
            let phase: Vec<Value> = self
                .phase_states
                .iter()
                .enumerate()
                .map(|(i, z)| {
                    let z = z.as_slice();
                    json!({"step": i + 1, "q": &z[..half], "p": &z[half..]})
                })
                .collect();
            out["phase_trajectory"] = json!(phase);
        }
        out
    }

    /// Whole-run `H_a` entry, only when one action (one Hamiltonian) spans
    /// every step: across different actions the forcing does work and the
    /// energies of different steps are not comparable.
    fn run_entry(&self) -> Value {
        let rel =
            |before: f64, after: f64| (after - before).abs() / before.abs().max(f64::MIN_POSITIVE);
        match (
            self.single_action,
            self.energies.first(),
            self.energies.last(),
        ) {
            (true, Some(&(h0, _)), Some(&(_, hn))) => json!({
                "initial": h0,
                "final": hn,
                "drift": hn - h0,
                "relative_drift": rel(h0, hn),
            }),
            _ => Value::Null,
        }
    }

    /// Ledger of a contact run. Nothing is claimed conserved unless the damping
    /// is zero; otherwise the entries show the decay and the exact per-step
    /// volume factor `exp(-2 gamma dt)` of the flow that produced them.
    fn contact_ledger(&self) -> Value {
        let rel =
            |before: f64, after: f64| (after - before).abs() / before.abs().max(f64::MIN_POSITIVE);
        let header = self
            .contact_header
            .clone()
            .unwrap_or_else(|| json!({"error": "contact ledger without a contact model"}));
        let conservative = header["conservative"] == Value::Bool(true);
        let max_abs = self
            .energies
            .iter()
            .map(|(b, a)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_rel = self
            .energies
            .iter()
            .map(|&(b, a)| rel(b, a))
            .fold(0.0_f64, f64::max);
        let dissipated: f64 = self.energies.iter().map(|(b, a)| b - a).sum();
        let contact_action: f64 = self.contact_actions.iter().map(|&s| f64::from(s)).sum();
        let mut out = json!({
            "manifold": "contact (q, p, s); s has no latent slot and restarts at 0 each step",
            "integrator": "strang_split_contact_V_half_D_half_T_D_half_V_half",
            "hamiltonian": "H_a = 1/2 |p|^2 + k/2 |q - c_a|^2 of each step's action",
            "friction": "dp/dt = -grad V - 2 gamma p",
            "conserved_quantity": if conservative {
                json!("H_a (gamma = 0: the split is exactly Stormer-Verlet)")
            } else {
                Value::Null
            },
            "max_abs_step_drift": max_abs,
            "max_relative_step_drift": max_rel,
            "dissipated_energy": dissipated,
            "contact_action_sum": contact_action,
            "single_action": self.single_action,
            "run": self.run_entry(),
        });
        if let Some(fields) = header.as_object() {
            for (k, v) in fields {
                out[k] = v.clone();
            }
        }
        out
    }

    /// Energy bookkeeping of a symplectic run. The whole-run drift is only
    /// reported when one action (one Hamiltonian) spans every step: across
    /// different actions the forcing does work and energy is not conserved.
    fn energy_ledger(&self) -> Value {
        let rel =
            |before: f64, after: f64| (after - before).abs() / before.abs().max(f64::MIN_POSITIVE);
        let max_abs = self
            .energies
            .iter()
            .map(|(b, a)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_rel = self
            .energies
            .iter()
            .map(|&(b, a)| rel(b, a))
            .fold(0.0_f64, f64::max);
        json!({
            "conserved_quantity": "H_a = 1/2 |p|^2 + k/2 |q - c_a|^2 of each step's action",
            "integrator": "stormer_verlet_kick_drift_kick",
            "max_abs_step_drift": max_abs,
            "max_relative_step_drift": max_rel,
            "single_action": self.single_action,
            "run": self.run_entry(),
        })
    }
}

/// Run up to `horizon` imagined steps. `pick(step, state)` names each action
/// (`step` starts at 1). Stops at the first `done`, which is also a hazard.
/// `keep_phase` keeps every intermediate state for `phase_trajectory`.
fn rollout<'a>(
    model: &WorldDynamics,
    gate: &PolicyGate,
    graph: &LodGraph,
    start: &FullLatent,
    horizon: usize,
    keep_phase: bool,
    mut pick: impl FnMut(usize, &FullLatent) -> Result<&'a str, Rejection>,
) -> Result<Rollout, Rejection> {
    let initial_energy = stored_energy(model, start);
    let mut current = start.clone();
    let mut trajectory = Vec::with_capacity(horizon);
    let mut phase_states = Vec::new();
    let mut energies = Vec::new();
    let mut contact_actions = Vec::new();
    let mut first_action: Option<&str> = None;
    let mut single_action = true;
    let mut cumulative_return = 0.0_f64;
    let mut termination_step = None;
    let mut worst_tier = PolicyTier::Tier0Proceed;
    for step in 1..=horizon {
        let name = pick(step, &current)?;
        single_action &= *first_action.get_or_insert(name) == name;
        let id = action_id(name);
        // The live graph's revocations reach every imagined step.
        let verdict = gate
            .evaluate(id, NormalizedEntropy::ZERO, Some(graph), None)
            .map_err(|e| failure("GateError", format!("policy gate failed: {e}")))?;
        worst_tier = worst_tier.max(verdict.tier);
        let StepOut {
            next,
            reward,
            done,
            energy,
            contact,
        } = model.step(&current, id)?;
        cumulative_return += f64::from(reward);
        let mut entry = json!({
            "step": step,
            "action": name,
            "action_id": id.0,
            "reward": reward,
            "done": done,
            "hazard": done,
            "tier": tier_name(verdict.tier),
            "state_norm": next.l2_norm(),
        });
        if let Some((before, after, drift)) = energy {
            let half = LATENT_DIM / 2;
            let norm = |xs: &[f32]| xs.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            entry["hamiltonian_before"] = json!(before);
            entry["hamiltonian_after"] = json!(after);
            entry["energy_drift"] = json!(drift);
            entry["q_norm"] = json!(norm(&next.as_slice()[..half]));
            entry["p_norm"] = json!(norm(&next.as_slice()[half..]));
            energies.push((before, after));
        }
        if let Some(c) = contact {
            entry["contact_action"] = json!(c.action);
            entry["phase_volume_factor_per_pair"] = json!(c.volume_factor_per_pair);
            contact_actions.push(c.action);
        }
        trajectory.push(entry);
        if keep_phase {
            phase_states.push(next.clone());
        }
        current = next;
        if done {
            termination_step = Some(step);
            break;
        }
    }
    // `done` is the only hazard signal this model has, so the first hazard
    // step is the termination step.
    let first_hazard_step = termination_step;
    let survival_horizon = first_hazard_step.map_or(trajectory.len(), |s| s - 1);
    let final_energy = stored_energy(model, &current);
    Ok(Rollout {
        kind: model.kind(),
        trajectory,
        phase_states,
        energies,
        contact_actions,
        contact_header: model.contact_ledger_header(),
        single_action,
        final_state: current,
        cumulative_return,
        termination_step,
        first_hazard_step,
        survival_horizon,
        worst_tier,
        initial_energy,
        final_energy,
    })
}

/// Unforced quadratic energy in the model's phase-space well. This common
/// reference makes energies comparable when different actions shift their wells.
fn stored_energy(model: &WorldDynamics, state: &FullLatent) -> f64 {
    let z = state.as_slice();
    let stiffness = match model {
        WorldDynamics::Symplectic(m) => f64::from(m.stiffness()),
        WorldDynamics::Contact(m) => f64::from(m.stiffness()),
        WorldDynamics::Residual(_) => 1.0,
    };
    let half = LATENT_DIM / 2;
    0.5 * (stiffness * z[..half].iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
        + z[half..].iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
}

/// Step 1 plays `first`; later steps take the one-step best of `set`:
/// non-hazard first, then higher reward, ties keep `set` order.
fn greedy_pick<'a>(
    model: &'a WorldDynamics,
    first: &'a str,
    set: &'a [String],
) -> impl FnMut(usize, &FullLatent) -> Result<&'a str, Rejection> + 'a {
    move |step, state| {
        if step == 1 {
            return Ok(first);
        }
        let mut best: Option<(&'a str, bool, f32)> = None;
        for name in set {
            let StepOut { reward, done, .. } = model.step(state, action_id(name))?;
            let better = best.is_none_or(|(_, best_done, best_reward)| {
                (!done, reward) > (!best_done, best_reward)
            });
            if better {
                best = Some((name.as_str(), done, reward));
            }
        }
        best.map(|(name, _, _)| name)
            .ok_or_else(|| invalid("greedy continuation needs a non-empty action set"))
    }
}

fn provenance_meta(spec: DynamicsSpec) -> Value {
    let mut meta = json!({
        "provenance": spec.kind.provenance(),
        "dynamics": spec.kind.name(),
        "trained": false,
        "calibrated": false,
        "hazard_definition": "world model reported `done` (state norm left the stable ball)",
    });
    if let Some(gamma) = spec.damping {
        meta["damping"] = json!(gamma);
    }
    meta
}

/// `simulate`: replay a fixed action plan from a latent state.
pub fn simulate(gate: &PolicyGate, graph: &LodGraph, args: &Value) -> WorldResult {
    let state = parse_latent(args.get("state"), "state")?;
    let actions = parse_names(args.get("actions"), "actions", MAX_HORIZON, false)?;
    let horizon = parse_horizon(args.get("horizon"), actions.len())?;
    if horizon > actions.len() {
        return Err(invalid(format!(
            "`horizon` {horizon} is longer than the {} given actions",
            actions.len()
        )));
    }
    let spec = parse_dynamics_spec(args)?;
    let kind = spec.kind;
    let model = WorldDynamics::new(spec)?;
    let keep_phase = kind.has_phase_space();
    let run = rollout(
        &model,
        gate,
        graph,
        &state,
        horizon,
        keep_phase,
        |step, _| Ok(actions[step - 1].as_str()),
    )?;
    let summary = format!(
        "Simulated {} of {horizon} step(s) on the {}: survival {}, return {:.4}",
        run.trajectory.len(),
        kind.prior_label(),
        run.survival_horizon,
        run.cumulative_return
    );
    let mut meta = provenance_meta(spec);
    meta["engine"] = json!(ENGINE_WORLDMODEL);
    meta["horizon"] = json!(horizon);
    meta["simulation"] = run.summary(true);
    Ok((meta, summary))
}

/// `what_if`: compare candidate first actions. Later steps follow the greedy
/// one-step policy over the candidates.
pub fn what_if(gate: &PolicyGate, graph: &LodGraph, args: &Value) -> WorldResult {
    let state = parse_latent(args.get("state"), "state")?;
    let candidates = parse_names(args.get("candidates"), "candidates", MAX_CANDIDATES, true)?;
    let horizon = parse_horizon(args.get("horizon"), DEFAULT_HORIZON)?;
    let spec = parse_dynamics_spec(args)?;
    let kind = spec.kind;
    let model = WorldDynamics::new(spec)?;
    let mut runs = Vec::with_capacity(candidates.len());
    for name in &candidates {
        let run = rollout(&model, gate, graph, &state, horizon, false, |_, _| {
            Ok(name.as_str())
        })?;
        runs.push(run);
    }
    // Energy-admissible before trapped, gate-clear before gate-blocked, then longer
    // survival, then higher return. Stable on ties.
    let is_admissible = |r: &Rollout| {
        r.first_hazard_step.is_none()
            && (!r.kind.has_phase_space() || r.final_energy <= r.initial_energy)
    };
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by(|&a, &b| {
        let key = |i: usize| {
            let r = &runs[i];
            (
                is_admissible(r),
                r.worst_tier != PolicyTier::Tier3HardStop,
                r.survival_horizon,
                r.cumulative_return,
            )
        };
        let (ka, kb) = (key(a), key(b));
        kb.0.cmp(&ka.0)
            .then(kb.1.cmp(&ka.1))
            .then(kb.2.cmp(&ka.2))
            .then(kb.3.total_cmp(&ka.3))
    });
    let ranking: Vec<&str> = order.iter().map(|&i| candidates[i].as_str()).collect();
    let top = order
        .first()
        .copied()
        .filter(|&i| is_admissible(&runs[i]) && runs[i].worst_tier != PolicyTier::Tier3HardStop);
    let outcomes: Vec<Value> = candidates
        .iter()
        .zip(&runs)
        .map(|(name, run)| {
            let mut o = run.summary(false);
            o["candidate"] = json!(name);
            o
        })
        .collect();
    let summary = match top {
        Some(i) => format!(
            "Ranked {} candidate(s) on the {}; top by that prior: '{}' \
             (advisory only, not a decision)",
            candidates.len(),
            kind.prior_label(),
            candidates[i]
        ),
        None => format!(
            "Ranked {} candidate(s) on the {}; none is both hazard-free \
             and gate-clear, so there is no top candidate",
            candidates.len(),
            kind.prior_label()
        ),
    };
    let mut meta = provenance_meta(spec);
    meta["engine"] = json!(ENGINE_WORLDMODEL);
    meta["horizon"] = json!(horizon);
    meta["continuation_policy"] = json!("repeat_candidate");
    meta["outcomes"] = json!(outcomes);
    meta["ranking"] = json!(ranking);
    meta["top_candidate"] = json!(top.map(|i| candidates[i].as_str()));
    meta["advisory_only"] = json!(true);
    Ok((meta, summary))
}

/// Verdicts of `audit`. None of them is an approval: the dynamics behind the
/// rollout are untrained.
pub const VERDICT_LETHAL: &str = "REJECT_LETHAL";
pub const VERDICT_POLICY: &str = "REJECT_POLICY";
pub const VERDICT_CONFIRM: &str = "REQUIRES_CONFIRMATION";
pub const VERDICT_UNVERIFIED: &str = "UNVERIFIED_UNTRAINED_DYNAMICS";

/// `audit`: roll one planned action forward and report a verdict.
/// `semantic_tier` and `semantic_meta` are the request-text risk check of the
/// audited action name, made by the engine.
pub fn audit(
    gate: &PolicyGate,
    graph: &LodGraph,
    args: &Value,
    semantic_tier: PolicyTier,
    semantic_meta: Value,
) -> WorldResult {
    let state = parse_latent(args.get("state"), "state")?;
    let target = args
        .get("target_action")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("`target_action` is required: a non-empty action name"))?;
    let horizon = parse_horizon(args.get("horizon"), DEFAULT_HORIZON)?;
    let continuation = match args.get("continuation_actions") {
        Some(v) => Some(parse_names(
            Some(v),
            "continuation_actions",
            MAX_CANDIDATES,
            true,
        )?),
        None => None,
    };
    let (policy_name, action_set) = match continuation {
        Some(set) => ("greedy_one_step_over_continuation_actions", set),
        None => ("repeat_audited_action", vec![target.to_string()]),
    };
    let spec = parse_dynamics_spec(args)?;
    let kind = spec.kind;
    let model = WorldDynamics::new(spec)?;
    let run = rollout(
        &model,
        gate,
        graph,
        &state,
        horizon,
        false,
        greedy_pick(&model, target, &action_set),
    )?;
    let tier = run.worst_tier.max(semantic_tier);
    let (verdict, explanation) = if let Some(step) = run.first_hazard_step {
        (
            VERDICT_LETHAL,
            format!(
                "Hazard at step {step} of {horizon}: the world model ended the episode; \
                 survives {} step(s).",
                run.survival_horizon
            ),
        )
    } else if tier == PolicyTier::Tier3HardStop {
        (
            VERDICT_POLICY,
            "The policy gate or the semantic risk check hard-stops this action.".to_string(),
        )
    } else if tier != PolicyTier::Tier0Proceed {
        (
            VERDICT_CONFIRM,
            format!(
                "No hazard in {} simulated step(s), but the policy gate or the semantic risk \
                 check requires confirmation (tier {}).",
                run.trajectory.len(),
                tier_name(tier)
            ),
        )
    } else {
        (
            VERDICT_UNVERIFIED,
            format!(
                "No hazard in {} simulated step(s) of horizon {horizon}. The {} dynamics are \
                 an untrained prior, so this is not a safety guarantee.",
                run.trajectory.len(),
                kind.name()
            ),
        )
    };
    let mut meta = provenance_meta(spec);
    meta["engine"] = json!(ENGINE_WORLDMODEL);
    meta["horizon"] = json!(horizon);
    meta["audited_action"] = json!(target);
    meta["verdict"] = json!(verdict);
    meta["explanation"] = json!(explanation);
    meta["hazard_detected"] = json!(run.first_hazard_step.is_some());
    meta["continuation_policy"] = json!(policy_name);
    meta["risk"] = semantic_meta;
    meta["audit_tier"] = json!(tier_name(tier));
    meta["rollout"] = run.summary(false);
    let summary = format!("Audit verdict for '{target}': {verdict}");
    Ok((meta, summary))
}

/// A latent planner's pick.
pub struct PlannedChoice {
    /// Index into the candidate list.
    pub index: usize,
    pub entropy: NormalizedEntropy,
    /// Name of the engine that decided.
    pub engine: &'static str,
}

/// Run one planner engine over `names` on a latent state. The planner crate's
/// engines are the only place a mode is decided: nothing here ranks actions.
pub fn plan_latent(
    mode: PlannerMode,
    spec: DynamicsSpec,
    gate: &PolicyGate,
    latent: &FullLatent,
    names: &[String],
) -> Result<PlannedChoice, Rejection> {
    if names.is_empty() || names.len() > MAX_CANDIDATES {
        return Err(invalid(format!(
            "planner modes take 1..={MAX_CANDIDATES} candidates, got {}",
            names.len()
        )));
    }
    let ids: Vec<ActionId> = names.iter().map(|n| action_id(n)).collect();
    let slices: Vec<&str> = names.iter().map(String::as_str).collect();
    let frame = LocalActionFrame::new(&slices, &ids)
        .ok_or_else(|| invalid("candidate frame could not be built"))?;
    let model = WorldDynamics::new(spec)?;
    let engine: Box<dyn PlanningEngine> = match mode {
        PlannerMode::Mcts => Box::new(MctsEngine::default()),
        PlannerMode::MpcCem => Box::new(MpcCemEngine::default()),
        PlannerMode::AStar => {
            let target = model.step(latent, ids[0]).map(|out| out.next).ok();
            Box::new(AStarEngine {
                goal: target.map(|t| AStarGoal::WithinDistance {
                    target: Box::new(t),
                    tolerance: 0.01,
                }),
                ..AStarEngine::default()
            })
        }
    };
    let (chosen, entropy) = engine
        .plan(latent, &frame, model.as_dyn(), gate)
        .map_err(|e| failure("PlannerFailure", format!("{} failed: {e}", engine.name())))?;
    let index = frame.to_local(chosen).ok_or_else(|| {
        failure(
            "PlannerFailure",
            "planner returned an action outside the frame",
        )
    })?;
    Ok(PlannedChoice {
        index,
        entropy,
        engine: engine.name(),
    })
}

/// Rollout of `chosen` on the latent prior (`decide --return-trajectory`).
/// Later steps follow the greedy policy over all candidates.
pub fn trajectory_for_choice(
    spec: DynamicsSpec,
    gate: &PolicyGate,
    graph: &LodGraph,
    latent: &FullLatent,
    chosen: &str,
    candidates: &[String],
    horizon: usize,
) -> Result<Value, Rejection> {
    let model = WorldDynamics::new(spec)?;
    let run = rollout(
        &model,
        gate,
        graph,
        latent,
        horizon,
        false,
        greedy_pick(&model, chosen, candidates),
    )?;
    let mut out = run.summary(false);
    out["provenance"] = json!(spec.kind.provenance());
    out["dynamics"] = json!(spec.kind.name());
    if let Some(gamma) = spec.damping {
        out["damping"] = json!(gamma);
    }
    out["continuation_policy"] = json!("greedy_one_step_over_candidates");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(fill: f64) -> Value {
        json!(vec![fill; LATENT_DIM])
    }

    #[test]
    fn short_state_is_refused_not_padded() {
        let e = parse_latent(Some(&json!([0.0, 1.0])), "state").unwrap_err();
        assert_eq!(e.code, "InvalidParams");
        assert!(e.detail.contains("padded"), "{}", e.detail);
    }

    #[test]
    fn text_state_is_refused() {
        let e = parse_latent(Some(&json!("hello")), "state").unwrap_err();
        assert!(e.detail.contains("no text encoder"), "{}", e.detail);
    }

    #[test]
    fn non_number_and_overflow_are_refused() {
        let mut v = vec![json!(0.0); LATENT_DIM];
        v[7] = json!("x");
        assert!(parse_latent(Some(&Value::Array(v)), "state").is_err());
        let mut v = vec![json!(0.0); LATENT_DIM];
        v[3] = json!(1e300);
        let e = parse_latent(Some(&Value::Array(v)), "state").unwrap_err();
        assert!(e.detail.contains("finite"), "{}", e.detail);
    }

    #[test]
    fn horizon_is_refused_not_clamped() {
        assert!(parse_horizon(Some(&json!(0)), 3).is_err());
        assert!(parse_horizon(Some(&json!(MAX_HORIZON + 1)), 3).is_err());
        assert!(parse_horizon(Some(&json!(true)), 3).is_err());
        assert!(parse_horizon(Some(&json!(-2)), 3).is_err());
        assert!(parse_horizon(Some(&json!(2.5)), 3).is_err());
        assert_eq!(parse_horizon(None, 3).unwrap(), 3);
        assert_eq!(parse_horizon(Some(&json!(7)), 3).unwrap(), 7);
    }

    #[test]
    fn names_are_strict() {
        assert!(parse_names(Some(&json!(["a", 1])), "actions", 8, false).is_err());
        assert!(parse_names(Some(&json!(["a", " "])), "actions", 8, false).is_err());
        assert!(parse_names(Some(&json!([])), "actions", 8, false).is_err());
        assert!(parse_names(Some(&json!(["a", "a"])), "candidates", 8, true).is_err());
        assert!(parse_names(Some(&json!(["a", "a"])), "actions", 8, false).is_ok());
        assert!(parse_names(Some(&json!(["a", "b", "c"])), "candidates", 2, true).is_err());
    }

    #[test]
    fn simulate_replays_the_plan_and_reports_provenance() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(0.0), "actions": ["a", "b", "c"]});
        let (meta, _) = simulate(&gate, &LodGraph::new(), &args).unwrap();
        assert_eq!(meta["provenance"], PROVENANCE);
        assert_eq!(meta["trained"], false);
        let sim = &meta["simulation"];
        assert_eq!(sim["steps_simulated"], 3);
        assert_eq!(sim["trajectory"][2]["action"], "c");
        assert_eq!(sim["final_state"].as_array().unwrap().len(), LATENT_DIM);
    }

    #[test]
    fn simulate_refuses_a_horizon_beyond_the_plan() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(0.0), "actions": ["a"], "horizon": 2});
        assert!(simulate(&gate, &LodGraph::new(), &args).is_err());
    }

    #[test]
    fn a_blown_up_state_is_a_hazard_at_step_one() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(10.0), "actions": ["a", "b"]});
        let (meta, _) = simulate(&gate, &LodGraph::new(), &args).unwrap();
        let sim = &meta["simulation"];
        assert_eq!(sim["is_safe"], false);
        assert_eq!(sim["first_hazard_step"], 1);
        assert_eq!(sim["steps_simulated"], 1);
        assert_eq!(sim["survival_horizon"], 0);
    }

    #[test]
    fn what_if_ranks_safe_before_hazard_and_is_advisory() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(0.0), "candidates": ["a", "b", "c"], "horizon": 3});
        let (meta, _) = what_if(&gate, &LodGraph::new(), &args).unwrap();
        assert_eq!(meta["ranking"].as_array().unwrap().len(), 3);
        assert_eq!(meta["advisory_only"], true);
        assert_eq!(meta["provenance"], PROVENANCE);
        assert!(meta["top_candidate"].is_string());
    }

    #[test]
    fn what_if_has_no_top_candidate_when_every_rollout_is_hazardous() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(10.0), "candidates": ["a", "b"]});
        let (meta, _) = what_if(&gate, &LodGraph::new(), &args).unwrap();
        assert!(meta["top_candidate"].is_null());
    }

    #[test]
    fn audit_never_approves() {
        let gate = PolicyGate::default();
        let args = json!({"state": latent(0.0), "target_action": "a", "horizon": 3});
        let (meta, _) = audit(
            &gate,
            &LodGraph::new(),
            &args,
            PolicyTier::Tier0Proceed,
            json!({"assessed": true}),
        )
        .unwrap();
        assert_eq!(meta["verdict"], VERDICT_UNVERIFIED);
        assert_ne!(meta["verdict"], "APPROVED");
        assert_eq!(meta["continuation_policy"], "repeat_audited_action");
    }

    #[test]
    fn audit_verdict_follows_hazard_then_tier() {
        let gate = PolicyGate::default();
        let hot = json!({"state": latent(10.0), "target_action": "a"});
        let (meta, _) = audit(
            &gate,
            &LodGraph::new(),
            &hot,
            PolicyTier::Tier0Proceed,
            json!({}),
        )
        .unwrap();
        assert_eq!(meta["verdict"], VERDICT_LETHAL);
        let calm = json!({"state": latent(0.0), "target_action": "a"});
        let (meta, _) = audit(
            &gate,
            &LodGraph::new(),
            &calm,
            PolicyTier::Tier2Escalate,
            json!({}),
        )
        .unwrap();
        assert_eq!(meta["verdict"], VERDICT_CONFIRM);
        let (meta, _) = audit(
            &gate,
            &LodGraph::new(),
            &calm,
            PolicyTier::Tier3HardStop,
            json!({}),
        )
        .unwrap();
        assert_eq!(meta["verdict"], VERDICT_POLICY);
    }

    /// A falsified graph node for action "a" hard-stops every simulated step of
    /// "a" and turns the audit into a policy rejection.
    #[test]
    fn graph_revocation_reaches_worldsim_rollouts() {
        use gen_zero_lod::{EpistemicStatus, LodBand, LodNode, MixedCurvatureCoord};
        let gate = PolicyGate::default();
        let graph = LodGraph::new();
        let entity = u64::from(action_id("a").0);
        graph
            .add_node(
                LodNode::new(
                    0,
                    LodBand::Lod0Atomic,
                    MixedCurvatureCoord::origin(),
                    "a",
                    entity,
                )
                .with_status(EpistemicStatus::Falsified),
            )
            .unwrap();
        let args = json!({"state": latent(0.0), "actions": ["a", "b"]});
        let (meta, _) = simulate(&gate, &graph, &args).unwrap();
        let steps = meta["simulation"]["trajectory"].as_array().unwrap();
        assert_eq!(steps[0]["tier"], "HardStop");
        assert_eq!(steps[1]["tier"], "Proceed");
        let (meta, _) = simulate(&gate, &LodGraph::new(), &args).unwrap();
        assert_eq!(meta["simulation"]["trajectory"][0]["tier"], "Proceed");

        let calm = json!({"state": latent(0.0), "target_action": "a"});
        let (meta, _) = audit(&gate, &graph, &calm, PolicyTier::Tier0Proceed, json!({})).unwrap();
        assert_eq!(meta["verdict"], VERDICT_POLICY);
    }

    #[test]
    fn every_planner_mode_picks_a_candidate() {
        let gate = PolicyGate::default();
        let state = parse_latent(Some(&latent(0.0)), "state").unwrap();
        let names = vec!["left".to_string(), "right".to_string(), "wait".to_string()];
        for mode in [PlannerMode::Mcts, PlannerMode::MpcCem, PlannerMode::AStar] {
            let choice = plan_latent(mode, DynamicsSpec::RESIDUAL, &gate, &state, &names).unwrap();
            assert!(choice.index < names.len(), "{mode:?}");
        }
    }

    #[test]
    fn planner_refuses_more_than_sixteen_candidates() {
        let gate = PolicyGate::default();
        let state = parse_latent(Some(&latent(0.0)), "state").unwrap();
        let names: Vec<String> = (0..17).map(|i| format!("a{i}")).collect();
        assert!(plan_latent(
            PlannerMode::AStar,
            DynamicsSpec::RESIDUAL,
            &gate,
            &state,
            &names
        )
        .is_err());
    }
}
