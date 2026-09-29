//! The Single Polymorphic `zero` Tool Router.
//!
//! Exposes 11 Core Cognitive Verbs:
//! 1. ask: Discrete decision over candidates, scored by the semantic bridge
//! 2. route: Tool ranking by semantic relevance to the intent
//! 3. imagine: Multi-step PUCT lookahead with semantic priors + formal feasibility
//! 4. stream: Tangent-space SSM scan of a numeric event window, gate-certified
//! 5. grep: Literal line search over request-supplied `lines`
//! 6. compact: zstd context compression
//! 7. entail: Asymmetric Busemann entailment on manifold coordinates
//! 8. causal_fold: Relation-chain fold on a caller-supplied learned semiring
//! 9. simulate: Fixed action plan rolled out on the latent world model
//! 10. what_if: Counterfactual comparison of candidate first actions on the latent world model
//! 11. audit: Shadow risk review of one planned action (never an approval)
//!
//! `ask` (alias `decide`) also takes a `mode`: `auto`/`reflex` run the semantic
//! ask; `mcts` runs the semantic PUCT lookahead (`imagine`) or, with a numeric
//! `latent`, the planner crate's MCTS; `mpc_cem` and `astar` need a `latent`.
//! The latent modes and the world-model verbs run on an untrained prior and say so.
//!
//! `ask`, `route` and `imagine` call the Python semantic scorer through
//! [`SemanticBridgeClient`]. When the bridge is off or unreachable they say so
//! in `_meta.engine` (`"local_fast_reflex_fallback"`) and never pretend to
//! have scored anything.
//!
//! Safety: the request text of those three verbs (context, intent, scenario)
//! goes to the multilingual semantic risk classifier first, and
//! [`PolicyGate::evaluate_semantic_risk`] turns the probability into a tier.
//! A request that cannot be assessed is escalated, never waved through.
//!
//! Every request first captures one immutable mount snapshot (Spec 25 §5.3,
//! [`crate::mount`]) and keeps it to the end. The bound generation is stamped
//! into `_meta.mount` of every outcome. A `tenant`/`workspace` pair with no
//! mount is refused, never served from a default. A request that names a
//! `mount_version` other than the captured one is `EpochMismatch`.
//!
//! Requests that carry numeric manifold coordinates (`cognitive`) go through
//! [`CognitiveRuntime`]: tangent map, parallel SSM scan, geometry gate and the
//! action verifier. Its refusals are typed ([`Rejection`]) and every entry
//! maps them to a non-success status. Text-only requests do not enter the
//! runtime (there is no text encoder into the manifold) and `_meta` says so.

use crate::bridge::{AskInput, BridgeError, SemanticBridgeClient, SemanticRiskResponse};
use crate::cognitive::{
    parse_decision, parse_entailment, parse_stream, CognitiveAssets, CognitiveRuntime, Rejection,
    ENGINE_COGNITIVE,
};
use crate::error::ServiceError;
use crate::imagine::{
    run_lookahead, BridgeOracle, LookaheadConfig, RootNoise, DEFAULT_C_PUCT, MAX_HORIZON,
    MAX_SIMULATIONS,
};
use crate::mount::{
    digest_hex, AssetDigests, AtomicMountRegistry, Budget, CandidateMount, MountKey, MountRegistry,
    MountSnapshot, Proposal, Reject, RequestBinding, SnapshotChange,
};
use crate::pipeline_verb::execute_pipeline;
use crate::worldsim::{self, PlannerMode};
use gen_zero_core::{
    ActionId, CompressedLatent, CoreError, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{PolicyGate, PolicyTier, SemanticRisk};
use gen_zero_lod::{
    AxiomWeights, FoldOutcome, Gender, LodGraph, LogProbSemiring, RelId, RelationKey,
    RelationSemiring, ResultSet, TropicalSemiring, WeightedFoldOutcome,
};
use gen_zero_model::{
    contains_raw_control_marker, ActionETFChoiceHead, MetricKind, DEFAULT_UNCALIBRATED_TEMPERATURE,
};
use gen_zero_nanocore::{
    DomainId, MoVFusionEngine, NanoCoreFleetScheduler, NanoCoreInstance, WatchdogConfig,
};
use gen_zero_provenance::{
    CapabilityArbiter, CapabilityFlags, CapabilityToken, DecisionAuditEntry, MmrInclusionProof,
    MmrLedger, MAX_AUDIT_LEAVES,
};
use gen_zero_storage::GoldenSnapshotManager;
use gen_zero_worldmodel::LatentDynamicsWorldModel;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// Polymorphic Verb in the `zero` tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZeroVerb {
    Ask,
    Route,
    Imagine,
    Stream,
    Grep,
    Compact,
    /// Asymmetric Busemann entailment `passage ⊃ question` on manifold coordinates.
    Entail,
    /// Relation-chain fold on a caller-supplied learned relation semiring
    /// (Spec 24 §8.6.2): S1 `left`, tiered dispatch or S3 `chart`.
    CausalFold,
    /// Latent world-model pipeline: `simulate`, `what_if`, `audit_action` and
    /// multi-mode `decide` over the engine's world model and PolicyGate.
    Pipeline,
    /// Fixed action plan rolled out on the latent world model.
    Simulate,
    /// Candidate first actions compared on the latent world model.
    WhatIf,
    /// Shadow risk review of one planned action.
    Audit,
}

impl ZeroVerb {
    /// Verbs that run on the latent world model and take numeric `state`.
    fn is_worldmodel(self) -> bool {
        matches!(self, Self::Simulate | Self::WhatIf | Self::Audit)
    }

    /// Inferred or parsed verb from input JSON.
    pub fn infer_from_input(val: &Value) -> Result<Self, ServiceError> {
        // Explicit "action" or "verb" parameter
        if let Some(act) = val
            .get("action")
            .or_else(|| val.get("verb"))
            .and_then(|v| v.as_str())
        {
            match act.to_lowercase().as_str() {
                "ask" | "decide" => return Ok(Self::Ask),
                "route" | "prune" => return Ok(Self::Route),
                "imagine" => return Ok(Self::Imagine),
                "simulate" => return Ok(Self::Simulate),
                "what_if" | "what-if" | "whatif" => return Ok(Self::WhatIf),
                "audit" | "audit_action" => return Ok(Self::Audit),
                "stream" | "step" => return Ok(Self::Stream),
                "grep" | "search" | "filter" => return Ok(Self::Grep),
                "compact" | "compress" => return Ok(Self::Compact),
                "entail" | "entailment" | "boolq" => return Ok(Self::Entail),
                "causal_fold" | "fold" => return Ok(Self::CausalFold),
                "pipeline" => return Ok(Self::Pipeline),
                _ => {}
            }
        }

        // Implicit intent inference by parameter heuristics (Doc 07 section 2)
        if val.get("pipeline").is_some() {
            Ok(Self::Pipeline)
        } else if val.get("causal_fold").is_some() {
            Ok(Self::CausalFold)
        } else if val.get("entailment").is_some() {
            Ok(Self::Entail)
        } else if val.get("actions").is_some() {
            Ok(Self::Simulate)
        } else if val.get("target_action").is_some() {
            Ok(Self::Audit)
        } else if val.get("context").is_some() && val.get("candidates").is_some() {
            Ok(Self::Ask)
        } else if val.get("intent").is_some() || val.get("tools").is_some() {
            Ok(Self::Route)
        } else if val.get("scenario").is_some()
            || val.get("simulation").is_some()
            || val.get("candidate_actions").is_some()
            || val.get("horizon").is_some()
            || val.get("enforce_cpsat").is_some()
        {
            Ok(Self::Imagine)
        } else if val.get("observation").is_some()
            || val.get("stream_id").is_some()
            || val.get("trajectory").is_some()
            || (val.get("cognitive").is_some() && val.get("candidates").is_none())
        {
            Ok(Self::Stream)
        } else if val.get("paths").is_some()
            || val.get("expr").is_some()
            || val.get("lines").is_some()
            || val.get("query").is_some()
        {
            Ok(Self::Grep)
        } else if val.get("messages").is_some()
            || val.get("head_lines").is_some()
            || val.get("tail_lines").is_some()
            || val.get("text").is_some()
        {
            Ok(Self::Compact)
        } else {
            // Default fallback is fast reflex ask
            Ok(Self::Ask)
        }
    }
}

/// Which engine answers an `ask` / `decide` request (`mode`, `latent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecideRoute {
    /// `auto`, `reflex` or no mode: the semantic ask.
    Plain,
    /// `mcts` on text: the semantic PUCT lookahead (`imagine`).
    SemanticLookahead,
    /// `mcts`, `mpc_cem` or `astar` on a numeric `latent`: the planner crate.
    Latent(PlannerMode),
}

/// Validate optional backend fields before the generic route can ignore them.
fn decision_backend(arguments: &Value, verb: ZeroVerb) -> Result<bool, Rejection> {
    let fields = [
        "engine",
        "head",
        "nanocore_core",
        "decision_state",
        "etf_rep",
        "candidate_reps",
        "etf_temperature",
        "etf_metric",
    ];
    let has = |field: &str| arguments.get(field).is_some();
    if verb != ZeroVerb::Ask {
        if let Some(field) = fields.iter().find(|f| has(f)) {
            return Err(Rejection::invalid(
                "request",
                format!("`{field}` is only accepted by ask/decide"),
            ));
        }
        return Ok(false);
    }
    let read = |field: &str, default: &'static str| -> Result<&str, Rejection> {
        match arguments.get(field) {
            None => Ok(default),
            Some(Value::String(s)) => Ok(s),
            _ => Err(Rejection::invalid(
                "request",
                format!("`{field}` must be a string"),
            )),
        }
    };
    let engine = read("engine", "generic")?;
    let head = read("head", "linear")?;
    if !matches!(engine, "generic" | "nanocore") || !matches!(head, "linear" | "etf") {
        return Err(Rejection::invalid(
            "request",
            "engine must be generic|nanocore and head must be linear|etf",
        ));
    }
    if engine != "nanocore" && (has("nanocore_core") || has("decision_state")) {
        return Err(Rejection::invalid(
            "request",
            "nanocore fields require engine=nanocore",
        ));
    }
    if head != "etf" && has("etf_rep") || engine == "nanocore" && has("etf_rep") {
        return Err(Rejection::invalid(
            "request",
            "etf_rep is only read by generic ETF",
        ));
    }
    if head != "etf" && (has("candidate_reps") || has("etf_temperature") || has("etf_metric")) {
        return Err(Rejection::invalid(
            "request",
            "candidate_reps, etf_temperature and etf_metric are only read by head=etf",
        ));
    }
    let specialized = engine == "nanocore" || head == "etf";
    if specialized
        && (has("cognitive")
            || has("latent")
            || has("return_trajectory") && arguments["return_trajectory"] == true
            || !matches!(
                arguments.get("mode").and_then(Value::as_str),
                None | Some("auto") | Some("reflex")
            ))
    {
        return Err(Rejection::invalid("request", "specialized decisions require auto/reflex mode without cognitive, latent or trajectory"));
    }
    Ok(specialized)
}

/// Largest whitening matrix `etf_metric` accepts, in f32 entries (4 MiB).
const MAX_WHITENING_ENTRIES: usize = 1 << 20;

/// Read one JSON array of f32 values for `etf_metric`. Refuses non-numbers,
/// non-finite values, values beyond f32 range, and nonzero values that
/// underflow to 0 in f32. Ordinary f32 rounding (e.g. 0.1) is accepted.
/// `candidate_reps` uses `parse_candidate_f32_array` instead, which keeps
/// the historical, more permissive underflow behavior callers rely on.
fn parse_f32_array(value: Option<&Value>, field: &str) -> Result<Vec<f32>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("`{field}` must be a numeric array"))?
        .iter()
        .map(|v| match v.as_f64() {
            Some(n)
                if n.is_finite() && n.abs() <= f32::MAX as f64 && (n == 0.0 || n as f32 != 0.0) =>
            {
                Ok(n as f32)
            }
            _ => Err(format!("`{field}` contains an invalid float")),
        })
        .collect()
}

/// Read one JSON array of f32 values for `candidate_reps`. Refuses
/// non-numbers, non-finite values, and values beyond f32 range. Unlike
/// `parse_f32_array`, a nonzero value that underflows to 0 in f32 (e.g.
/// `1e-60`) is accepted, matching pre-existing behavior for this field.
fn parse_candidate_f32_array(value: Option<&Value>, field: &str) -> Result<Vec<f32>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("`{field}` must be a numeric array"))?
        .iter()
        .map(|v| match v.as_f64() {
            Some(n) if n.is_finite() && n.abs() <= f32::MAX as f64 => Ok(n as f32),
            _ => Err(format!("`{field}` contains an invalid float")),
        })
        .collect()
}

/// Read optional `etf_metric` for a state of `width` floats. `None` means
/// the field is absent and the head stays isotropic. Unknown kinds and keys
/// are refused, not ignored. Value checks (positive precision, zero rows)
/// run again in the model constructors.
fn parse_etf_metric(arguments: &Value, width: usize) -> Result<Option<MetricKind>, String> {
    let Some(raw) = arguments.get("etf_metric") else {
        return Ok(None);
    };
    let obj = raw
        .as_object()
        .ok_or("`etf_metric` must be an object with a `kind`")?;
    let kind = obj
        .get("kind")
        .and_then(Value::as_str)
        .ok_or("`etf_metric.kind` must be isotropic|diagonal_mahalanobis|whitened")?;
    let allowed: &[&str] = match kind {
        "isotropic" => &["kind"],
        "diagonal_mahalanobis" => &["kind", "precision"],
        "whitened" => &["kind", "matrix", "out_dim"],
        other => {
            return Err(format!(
                "`etf_metric.kind` `{other}` is not isotropic|diagonal_mahalanobis|whitened"
            ))
        }
    };
    if let Some(extra) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(format!("`etf_metric.{extra}` is not read by kind={kind}"));
    }
    match kind {
        "isotropic" => Ok(Some(MetricKind::Isotropic)),
        "diagonal_mahalanobis" => {
            let precision = parse_f32_array(obj.get("precision"), "etf_metric.precision")?;
            if precision.len() != width {
                return Err(format!(
                    "`etf_metric.precision` has {} floats; the state has {width}",
                    precision.len()
                ));
            }
            Ok(Some(MetricKind::DiagonalMahalanobis(precision)))
        }
        _ => {
            let out_dim = obj
                .get("out_dim")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or("`etf_metric.out_dim` must be a positive integer")?;
            let entries = out_dim
                .checked_mul(width)
                .filter(|&n| n <= MAX_WHITENING_ENTRIES)
                .ok_or_else(|| {
                    format!("`etf_metric` whitening matrix exceeds {MAX_WHITENING_ENTRIES} entries")
                })?;
            // Length first, so an oversized array is refused before any float is parsed.
            let raw_len = obj.get("matrix").and_then(Value::as_array).map(Vec::len);
            if let Some(len) = raw_len.filter(|&len| len != entries) {
                return Err(format!(
                    "`etf_metric.matrix` has {len} floats; out_dim x state width is {entries}"
                ));
            }
            let matrix = parse_f32_array(obj.get("matrix"), "etf_metric.matrix")?;
            Ok(Some(MetricKind::WhitenedProjection { matrix, out_dim }))
        }
    }
}

/// Read `candidate_reps` (candidate name -> numeric vector) in `feasible` order.
/// Every feasible candidate needs a vector of the state's width; names outside
/// the request are refused. Reps of formally infeasible candidates are unread.
fn parse_candidate_reps(
    arguments: &Value,
    feasible: &[String],
    infeasible: &[&String],
    width: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let map = arguments
        .get("candidate_reps")
        .and_then(Value::as_object)
        .ok_or("head=etf requires `candidate_reps`: an object mapping each candidate to its numeric representation")?;
    if let Some(unknown) = map
        .keys()
        .find(|k| !feasible.contains(k) && !infeasible.contains(k))
    {
        return Err(format!(
            "`candidate_reps` names unknown candidate `{unknown}`"
        ));
    }
    feasible
        .iter()
        .map(|name| {
            let field = format!("candidate_reps.{name}");
            let rep = parse_candidate_f32_array(map.get(name), &field)?;
            if rep.len() != width {
                return Err(format!(
                    "`{field}` has {} floats; the state has {width}",
                    rep.len()
                ));
            }
            Ok(rep)
        })
        .collect()
}

const NANOCORE_PATH_ENV: &str = "GENZERO_NANOCORE_PATH";
const NANOCORE_PATHS_ENV: &str = "GENZERO_NANOCORE_PATHS";

/// Split the plural operator configuration using the platform path-list
/// separator. Empty entries are retained so configuration mistakes can be
/// reported with the setting name instead of being silently ignored.
fn split_nanocore_paths(raw: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(raw).collect()
}

/// Return the configured operator core files in registration order. The
/// plural setting takes precedence when present, while the original singular
/// setting remains the one-file compatibility path.
fn configured_nanocore_paths() -> Vec<PathBuf> {
    match std::env::var_os(NANOCORE_PATHS_ENV) {
        Some(raw) => split_nanocore_paths(&raw),
        None => std::env::var_os(NANOCORE_PATH_ENV)
            .into_iter()
            .map(PathBuf::from)
            .collect(),
    }
}

fn load_configured_nanocores(fleet: &NanoCoreFleetScheduler) {
    let plural_configured = std::env::var_os(NANOCORE_PATHS_ENV).is_some();
    let plural_empty = std::env::var_os(NANOCORE_PATHS_ENV)
        .map(|value| value.as_os_str().is_empty())
        .unwrap_or(false);
    let paths = configured_nanocore_paths();
    if plural_configured && (plural_empty || paths.is_empty()) {
        panic!("{NANOCORE_PATHS_ENV} is set but contains no core paths");
    }
    let source = if plural_configured {
        NANOCORE_PATHS_ENV
    } else {
        NANOCORE_PATH_ENV
    };
    load_nanocores_from_paths(fleet, paths, source);
}

fn load_nanocores_from_paths<I>(fleet: &NanoCoreFleetScheduler, paths: I, source: &str)
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut domains = BTreeSet::new();
    for path in paths {
        if path.as_os_str().is_empty() {
            panic!("{source} contains an empty core path");
        }
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {source} path {path:?}: {error}"));
        let core: NanoCoreInstance = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!("invalid operator Nanocore JSON at {source} path {path:?}: {error}")
        });
        core.validate().unwrap_or_else(|error| {
            panic!("invalid operator Nanocore at {source} path {path:?}: {error}")
        });
        if !domains.insert(core.domain_id) {
            panic!(
                "duplicate operator Nanocore domain {} configured at {path:?}",
                core.domain_id.0
            );
        }
        fleet.register_core(core).unwrap_or_else(|error| {
            panic!("cannot register operator Nanocore from {path:?}: {error}")
        });
    }
}

/// Parse the operator-core selector while retaining the old singular request
/// field. A request may select one domain with `nanocore_domain` or several
/// with `nanocore_domains`; mixing the fields or repeating a domain is
/// rejected so that the active mixture is always explicit.
fn nanocore_domains(arguments: &Value) -> Result<Option<Vec<DomainId>>, Rejection> {
    let singular = arguments.get("nanocore_domain");
    let plural = arguments.get("nanocore_domains");
    if singular.is_some() && plural.is_some() {
        return Err(Rejection::invalid(
            "nanocore",
            "provide either nanocore_domain or nanocore_domains, not both",
        ));
    }

    let parse_domain = |value: &Value, field: &str| {
        value
            .as_u64()
            .and_then(|number| u32::try_from(number).ok())
            .map(DomainId)
            .ok_or_else(|| {
                Rejection::invalid("nanocore", format!("`{field}` must be a u32 domain ID"))
            })
    };

    if let Some(value) = singular {
        return Ok(Some(vec![parse_domain(value, "nanocore_domain")?]));
    }
    let Some(value) = plural else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        Rejection::invalid(
            "nanocore",
            "`nanocore_domains` must be an array of u32 domain IDs",
        )
    })?;
    if values.is_empty() {
        return Err(Rejection::invalid(
            "nanocore",
            "`nanocore_domains` must contain at least one domain ID",
        ));
    }

    let mut domains = Vec::with_capacity(values.len());
    for value in values {
        let domain = parse_domain(value, "nanocore_domains")?;
        if domains.contains(&domain) {
            return Err(Rejection::invalid(
                "nanocore",
                format!("`nanocore_domains` repeats domain {}", domain.0),
            ));
        }
        domains.push(domain);
    }
    Ok(Some(domains))
}

impl DecideRoute {
    const MODES: &'static str = "auto, reflex, mcts, mpc_cem, astar";

    /// Fix the route or refuse. `mode`, `latent` and `return_trajectory` are
    /// read only by `ask`; anywhere else they would be dropped silently.
    fn resolve(arguments: &Value, verb: ZeroVerb) -> Result<Self, Rejection> {
        let has = |field: &str| arguments.get(field).is_some();
        let reads_dynamics = matches!(
            verb,
            ZeroVerb::Ask | ZeroVerb::Simulate | ZeroVerb::WhatIf | ZeroVerb::Audit
        );
        for field in ["dynamics", "damping"] {
            if has(field) && !reads_dynamics {
                return Err(Rejection::invalid(
                    "request",
                    format!(
                        "`{field}` is only accepted by simulate, what_if, audit and the latent \
                         ask modes, not {verb:?}"
                    ),
                ));
            }
        }
        if verb != ZeroVerb::Ask {
            for field in ["mode", "latent", "return_trajectory"] {
                if has(field) {
                    return Err(Rejection::invalid(
                        "request",
                        format!("`{field}` is only accepted by ask/decide, not {verb:?}"),
                    ));
                }
            }
            return Ok(Self::Plain);
        }
        if arguments
            .get("return_trajectory")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(Rejection::invalid(
                "request",
                "`return_trajectory` must be a boolean",
            ));
        }
        let mode = match arguments.get("mode") {
            None => "auto".to_string(),
            Some(Value::String(s)) => s.trim().to_ascii_lowercase(),
            Some(_) => {
                return Err(Rejection::invalid("request", "`mode` must be a string"));
            }
        };
        let latent = has("latent");
        let needs_latent = || {
            Rejection::invalid(
                "request",
                format!(
                    "mode `{mode}` needs a numeric `latent` state: there is no text encoder \
                     into the latent space"
                ),
            )
        };
        let route = match mode.as_str() {
            "auto" | "reflex" if latent => Err(Rejection::invalid(
                "request",
                "`latent` is read only by modes mcts, mpc_cem and astar; \
                 auto and reflex run the semantic ask",
            )),
            "auto" | "reflex" => Ok(Self::Plain),
            "mcts" if latent => Ok(Self::Latent(PlannerMode::Mcts)),
            "mcts" => Ok(Self::SemanticLookahead),
            "mpc_cem" if latent => Ok(Self::Latent(PlannerMode::MpcCem)),
            "astar" if latent => Ok(Self::Latent(PlannerMode::AStar)),
            "mpc_cem" | "astar" => Err(needs_latent()),
            other => Err(Rejection::invalid(
                "request",
                format!("unknown mode `{other}`; modes: {}", Self::MODES),
            )),
        }?;
        // Fields the chosen engine never reads are refused, not dropped.
        for field in ["dynamics", "damping"] {
            if !matches!(route, Self::Latent(_)) && has(field) {
                return Err(Rejection::invalid(
                    "request",
                    format!(
                        "`{field}` picks the world model of a latent mode (mcts, mpc_cem or \
                         astar with `latent`); the semantic ask has no latent dynamics"
                    ),
                ));
            }
        }
        if matches!(route, Self::Latent(_)) && has("cognitive") {
            return Err(Rejection::invalid(
                "request",
                "`cognitive` runs the cognitive runtime and a latent mode runs the planner \
                 crate: they are different engines, pick one",
            ));
        }
        if matches!(route, Self::Plain) && has("mode") && has("horizon") {
            return Err(Rejection::invalid(
                "request",
                "`horizon` is not read by the semantic ask (modes auto and reflex); \
                 it applies to mcts, or to a latent mode with `return_trajectory`",
            ));
        }
        Ok(route)
    }

    /// Say which engine answered, and what `return_trajectory` could not do.
    fn annotate(self, arguments: &Value, meta: &mut Value) {
        if let Some(requested) = arguments.get("mode").and_then(Value::as_str) {
            let resolved = match self {
                Self::Plain => "semantic_ask".to_string(),
                Self::SemanticLookahead => "semantic_puct_lookahead".to_string(),
                Self::Latent(m) => format!("latent_planner:{}", m.name()),
            };
            meta["mode"] = json!({"requested": requested, "resolved": resolved});
        }
        let wants_trajectory = arguments.get("return_trajectory") == Some(&Value::Bool(true));
        if wants_trajectory && !matches!(self, Self::Latent(_)) {
            meta["trajectory_status"] = json!(
                "unsupported_without_latent_state: a text state has no latent dynamics, \
                 so no trajectory was produced"
            );
        }
    }
}

/// Unified Result from executing a `zero` tool call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroToolOutcome {
    pub verb: ZeroVerb,
    pub is_error: bool,
    pub content: Vec<ZeroContentBlock>,
    pub meta: Value,
    /// Typed refusal. Set on every runtime, mount or input refusal; entries
    /// map it to a non-success status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection: Option<Rejection>,
}

impl ZeroToolOutcome {
    /// Refusal outcome. `meta` keeps whatever trace was produced before the
    /// refusal, so the caller sees how far the request got.
    pub fn rejected(verb: ZeroVerb, rejection: Rejection, mut meta: Value) -> Self {
        if !meta.is_object() {
            meta = json!({});
        }
        meta["reject"] = json!(rejection);
        Self {
            verb,
            is_error: true,
            content: text_block(format!("{}: {}", rejection.code, rejection.detail)),
            meta,
            rejection: Some(rejection),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

/// Receipt for one verified append to the audit ledger.
struct AuditLeaf {
    index: u64,
    leaf_count: usize,
    root: [u8; 32],
    window: usize,
}

#[derive(Clone, Copy)]
struct AuditAppendInput {
    request_hash: [u8; 32],
    result_hash: [u8; 32],
    action: ActionId,
    tier: u8,
    timestamp: u64,
}

/// Configuration that affects engine-local durable audit state.
#[derive(Clone, Debug, Default)]
pub struct ZeroEngineConfig {
    /// Optional checksummed MMR snapshot path. When set, startup loads the key, root and
    /// retained proof material from this path, and every decision append is synced there.
    pub mmr_persist_path: Option<PathBuf>,
}

impl ZeroEngineConfig {
    /// Read the optional audit snapshot path from the process environment.
    pub fn from_env() -> Result<Self, ServiceError> {
        let path = std::env::var_os("GENZERO_MMR_PERSIST_PATH")
            .filter(|value| !value.to_string_lossy().trim().is_empty())
            .map(PathBuf::from);
        Ok(Self {
            mmr_persist_path: path,
        })
    }

    pub fn with_mmr_persist_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmr_persist_path = Some(path.into());
        self
    }
}

/// Process and cross-process ownership of one durable audit snapshot. A lock file is held for
/// the lifetime of the engine, so two service instances cannot overwrite one append history.
struct AuditPersistence {
    path: PathBuf,
    _lock: File,
}

fn acquire_audit_lock(path: &Path) -> Result<File, ServiceError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    options.mode(0o600);
    let lock = options.open(path).map_err(|error| {
        ServiceError::Core(format!(
            "open audit persistence lock {}: {error}",
            path.display()
        ))
    })?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|error| {
        ServiceError::Core(format!(
            "cannot exclusively lock audit persistence path {}: {error}",
            path.display()
        ))
    })?;
    // Keep the inode on disk: unlinking it would allow a second lock domain.
    // Closing this handle releases ownership, including after a process crash.
    Ok(lock)
}

impl AuditPersistence {
    fn open(path: PathBuf, generated_key: [u8; 32]) -> Result<(Self, MmrLedger), ServiceError> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let file_name = path.file_name().ok_or_else(|| {
            ServiceError::Core(format!(
                "GENZERO_MMR_PERSIST_PATH {} has no file name",
                path.display()
            ))
        })?;
        let lock_path = parent.join(format!(".{}.lock", file_name.to_string_lossy()));
        let lock = acquire_audit_lock(&lock_path)?;

        let persistence = Self { path, _lock: lock };
        let ledger = if persistence.path.exists() {
            MmrLedger::load_snapshot(&persistence.path, Some(MAX_AUDIT_LEAVES))
                .map_err(|e| ServiceError::Core(e.to_string()))?
        } else {
            let ledger = MmrLedger::with_window(generated_key, MAX_AUDIT_LEAVES)
                .map_err(|e| ServiceError::Core(e.to_string()))?;
            persistence
                .persist(&ledger)
                .map_err(|e| ServiceError::Core(e.to_string()))?;
            ledger
        };
        Ok((persistence, ledger))
    }

    fn persist(&self, ledger: &MmrLedger) -> Result<(), gen_zero_provenance::ProvenanceError> {
        ledger.persist_snapshot(&self.path)
    }
}

/// Central Polymorphic Zero Engine.
pub struct PolymorphicZeroEngine {
    /// Shared admission budget for queued and running blocking CPU jobs.
    cpu_slots: Arc<tokio::sync::Semaphore>,
    gate: Arc<PolicyGate>,
    graph: Arc<LodGraph>,
    /// Transition model behind the `pipeline` verb.
    world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
    bridge: Option<Arc<SemanticBridgeClient>>,
    mounts: Arc<AtomicMountRegistry>,
    runtime: Arc<CognitiveRuntime>,
    pub(crate) golden_snapshots: Mutex<GoldenSnapshotManager>,
    /// Operator-loaded micro-cores only. Caller-supplied cores (`engine: "nanocore"`) run
    /// directly and are never registered here: `register_core` overwrites by domain.
    nano_fleet: Arc<NanoCoreFleetScheduler>,
    /// The one process-local decision ledger: `ask` and `pipeline decide` both append here,
    /// and `/audit/ledger` reads it. Windowed to [`MAX_AUDIT_LEAVES`] provable leaves.
    audit_ledger: Arc<Mutex<MmrLedger>>,
    /// Serializes durable writer transactions without making readers wait on fsync/rename.
    audit_writer: Arc<Mutex<()>>,
    audit_key: [u8; 32],
    audit_arbiter: CapabilityArbiter,
    audit_token: CapabilityToken,
    audit_persistence: Option<Arc<AuditPersistence>>,
    pub(crate) metrics: crate::server::ServiceMetrics,
}

impl Default for PolymorphicZeroEngine {
    /// Gate defaults plus the semantic bridge configured from the environment
    /// (`GENZERO_PYTHON_ENDPOINT`, see [`crate::bridge::BridgeConfig::from_env`]).
    fn default() -> Self {
        Self::new()
    }
}

impl PolymorphicZeroEngine {
    /// Construct an engine from environment configuration, propagating persistence failures to
    /// callers that can report a startup error. Self::new retains the historical infallible
    /// constructor and fails closed with a startup panic when this returns an error.
    pub fn try_new() -> Result<Self, ServiceError> {
        Self::try_from_config(ZeroEngineConfig::from_env()?)
    }

    /// Construct an engine with explicit durable audit configuration.
    pub fn try_from_config(config: ZeroEngineConfig) -> Result<Self, ServiceError> {
        let generated_audit_key = rand::random();
        let (audit_persistence, audit_key, audit_ledger) =
            if let Some(path) = config.mmr_persist_path {
                let (persistence, ledger) = AuditPersistence::open(path, generated_audit_key)?;
                let key = ledger.key();
                (Some(Arc::new(persistence)), key, ledger)
            } else {
                let ledger = MmrLedger::with_window(generated_audit_key, MAX_AUDIT_LEAVES)
                    .map_err(|e| ServiceError::Core(e.to_string()))?;
                (None, generated_audit_key, ledger)
            };
        let audit_arbiter = CapabilityArbiter::new(audit_key);
        audit_arbiter.register_agent("service_audit_writer", CapabilityFlags::AUDIT_ADMIN);
        let audit_token = audit_arbiter
            .issue_token("service_audit_writer", rand::random())
            .expect("registered audit writer must receive a token");
        let nano_fleet = Arc::new(NanoCoreFleetScheduler::new(
            gen_zero_nanocore::DEFAULT_RAM_BUDGET_BYTES,
        ));
        load_configured_nanocores(&nano_fleet);
        Ok(Self {
            cpu_slots: Arc::new(tokio::sync::Semaphore::new(
                std::thread::available_parallelism().map_or(2, |n| n.get().saturating_mul(2)),
            )),
            gate: Arc::new(PolicyGate::default()),
            graph: Arc::new(LodGraph::new()),
            world_model: Arc::new(LatentDynamicsWorldModel::default()),
            bridge: SemanticBridgeClient::from_env().map(Arc::new),
            mounts: Arc::new(default_mounts()),
            runtime: Arc::new(CognitiveRuntime::new()),
            golden_snapshots: Mutex::new(GoldenSnapshotManager::new(3, rand::random(), 16)),
            nano_fleet,
            audit_ledger: Arc::new(Mutex::new(audit_ledger)),
            audit_writer: Arc::new(Mutex::new(())),
            audit_key,
            audit_arbiter,
            audit_token,
            audit_persistence,
            metrics: crate::server::ServiceMetrics::default(),
        })
    }

    /// Alias for hosts that use configuration-driven construction without the `try_` naming.
    pub fn from_config(config: ZeroEngineConfig) -> Result<Self, ServiceError> {
        Self::try_from_config(config)
    }
}

/// Tenant and workspace used when a request names neither.
pub const DEFAULT_TENANT: &str = "default";
pub const DEFAULT_WORKSPACE: &str = "default";

/// Registry holding the built-in generation of the default key. Its digests
/// name what this binary actually serves today: the compiled engine, and no
/// atlas, geometry or graph asset (none is loaded yet). They are identities,
/// not measurements of a trained model.
fn default_mounts() -> AtomicMountRegistry {
    let label = |slot: &str, what: &str| -> [u8; 32] {
        use sha2::{Digest, Sha256};
        Sha256::digest(format!("gen-zero/genesis/{slot}/{what}").as_bytes()).into()
    };
    let digests = AssetDigests {
        model: label(
            "model",
            concat!("builtin-engine/", env!("CARGO_PKG_VERSION")),
        ),
        geometry: label("geometry", "none-loaded"),
        atlas: label("atlas", "empty"),
        graph: label("graph", "empty"),
        policy: label("policy", "builtin-gate"),
    };
    let registry = AtomicMountRegistry::new();
    let genesis = MountSnapshot::genesis(
        MountKey::new(DEFAULT_TENANT, DEFAULT_WORKSPACE),
        digests,
        0,
        Arc::from(Vec::new()),
    )
    .expect("genesis digests are SHA-256 outputs, never all-zero");
    registry
        .register(genesis)
        .expect("a fresh registry has no default mount yet");
    registry
}

/// `tenant` / `workspace` of a request. Absent means the default key; present
/// but not a non-empty string is an error, never a silent default.
fn mount_key(arguments: &Value) -> Result<MountKey, Rejection> {
    let field = |name: &str, default: &str| -> Result<String, Rejection> {
        match arguments.get(name) {
            None => Ok(default.to_string()),
            Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.trim().to_string()),
            Some(_) => Err(Rejection::invalid(
                "mount_capture",
                format!("`{name}` must be a non-empty string"),
            )),
        }
    };
    Ok(MountKey::new(
        field("tenant", DEFAULT_TENANT)?,
        field("workspace", DEFAULT_WORKSPACE)?,
    ))
}

fn mount_meta(snapshot: &MountSnapshot) -> Value {
    json!({
        "tenant": snapshot.key().tenant,
        "workspace": snapshot.key().workspace,
        "version": snapshot.version().0,
        "digest": digest_hex(snapshot.digest()),
        "watermark": snapshot.watermark(),
        "has_cognitive_assets": !snapshot.assets().is_empty(),
    })
}

const ENGINE_SEMANTIC: &str = "semantic_bridge";
const ENGINE_FALLBACK: &str = "local_fast_reflex_fallback";
const ENGINE_LATENT_PLANNER: &str = "latent_planner";
const DEFAULT_IMAGINE_HORIZON: usize = 3;
const DEFAULT_IMAGINE_SIMULATIONS: usize = 16;
/// Largest edge-chain `causal_fold` accepts. `chart_fold_chain` is a cubic-
/// time CYK-style DP over the chain length, and this endpoint is network-
/// exposed, so the length is capped rather than left unbounded.
const MAX_CAUSAL_FOLD_EDGES: usize = 64;
const MAX_CAUSAL_FOLD_SET_RELATIONS: usize = 64;
const MAX_CAUSAL_FOLD_TOTAL_SET_RELATIONS: usize = 256;
/// Most distinct relation ids that may appear across all axiom `result`
/// lists. Every chart span of two or more edges holds only such ids, so this
/// bounds each span set, and each split costs at most this value squared in
/// lookups. Without it one axiom with a 65,536-id result makes a 4-edge
/// chart do billions of lookups.
const MAX_CAUSAL_FOLD_RESULT_RELATIONS: usize = 64;
/// Most axiom entries one request may carry.
const MAX_CAUSAL_FOLD_AXIOMS: usize = 4096;

/// One axiom entry of a `causal_fold` request: `T(r1, r2, gender) = result`.
/// Unknown keys are refused, not ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CausalFoldAxiomWire {
    r1: RelId,
    r2: RelId,
    gender: Gender,
    result: Vec<RelId>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct CausalFoldAxiomWeightWire {
    r1: RelId,
    r2: RelId,
    gender: Gender,
    relation: RelId,
    count: u64,
}

/// Request: `{edges, genders, axioms?, strategy?}` or `{sets, gender, axioms?}`.
/// Unknown keys are refused, not ignored. Missing `axioms` means an empty
/// table; missing `strategy` means `"tiered"` with weights, otherwise `"chart"`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CausalFoldWire {
    #[serde(default)]
    weights: Option<Vec<CausalFoldAxiomWeightWire>>,
    #[serde(default)]
    margin_threshold: Option<f64>,
    #[serde(default)]
    sets: Option<Vec<Vec<RelId>>>,
    #[serde(default)]
    gender: Option<Gender>,
    #[serde(default)]
    edges: Vec<RelId>,
    #[serde(default)]
    genders: Vec<Gender>,
    #[serde(default)]
    axioms: Vec<CausalFoldAxiomWire>,
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    semiring: Option<String>,
}

/// Stable identifier for a candidate name. Used only as a key for the
/// PolicyGate (constraints and confirm lists), never as a semantic signal.
pub(crate) fn action_id(name: &str) -> ActionId {
    let hash = blake3::hash(name.as_bytes());
    let bytes: [u8; 4] = hash.as_bytes()[0..4]
        .try_into()
        .expect("blake3 digest is 32 bytes");
    ActionId(u32::from_le_bytes(bytes))
}

/// Explicit outcome state of the gate, so a caller never has to infer
/// "needs confirmation" from an HTTP status or a numeric error code.
const GATE_STATUS_PROCEED: &str = "proceed";
const GATE_STATUS_REQUIRES_CONFIRMATION: &str = "requires_confirmation";
const GATE_STATUS_HARD_STOP: &str = "hard_stop";

fn gate_status(tier: PolicyTier) -> &'static str {
    match tier {
        PolicyTier::Tier0Proceed => GATE_STATUS_PROCEED,
        PolicyTier::Tier1Confirm | PolicyTier::Tier2Escalate => GATE_STATUS_REQUIRES_CONFIRMATION,
        PolicyTier::Tier3HardStop => GATE_STATUS_HARD_STOP,
    }
}

pub(crate) fn tier_name(tier: PolicyTier) -> &'static str {
    match tier {
        PolicyTier::Tier0Proceed => "Proceed",
        PolicyTier::Tier1Confirm => "Confirm",
        PolicyTier::Tier2Escalate => "Escalate",
        PolicyTier::Tier3HardStop => "HardStop",
    }
}

fn text_block(text: impl Into<String>) -> Vec<ZeroContentBlock> {
    vec![ZeroContentBlock {
        block_type: "text".to_string(),
        text: text.into(),
    }]
}

fn string_list(val: Option<&Value>, field: &str) -> Result<Vec<String>, Rejection> {
    match val {
        None => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        Rejection::invalid(field, "expected an array of non-empty strings")
                    })
            })
            .collect(),
        Some(_) => Err(Rejection::invalid(
            field,
            "expected an array of non-empty strings",
        )),
    }
}

/// First non-empty string among `keys`.
fn first_text<'a>(arguments: &'a Value, keys: &[&str]) -> &'a str {
    keys.iter()
        .filter_map(|k| arguments.get(*k).and_then(|v| v.as_str()))
        .map(str::trim)
        .find(|s| !s.is_empty())
        .unwrap_or("")
}

fn tool_name(tool: &Value) -> Option<String> {
    tool.as_str()
        .or_else(|| tool.get("name").and_then(|n| n.as_str()))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Request-local hard masks. Names use the Python compiler's trim/uppercase convention.
/// Mutex groups keep the first feasible candidate, including across all lookahead depths.
#[derive(Default)]
struct RequestSafety {
    forbidden: BTreeSet<String>,
    mutex: Vec<BTreeSet<String>>,
    enforce_cpsat: bool,
}

impl RequestSafety {
    fn parse(arguments: &Value) -> Result<Self, Rejection> {
        let mut safety = Self {
            enforce_cpsat: true,
            ..Self::default()
        };
        if let Some(value) = arguments.get("enforce_cpsat") {
            safety.enforce_cpsat = value
                .as_bool()
                .ok_or_else(|| Rejection::invalid("enforce_cpsat", "expected a boolean"))?;
        }
        let names = |value: &Value,
                     field: &str,
                     minimum: usize|
         -> Result<BTreeSet<String>, Rejection> {
            let list = value
                .as_array()
                .ok_or_else(|| Rejection::invalid(field, "expected an array of action names"))?;
            let mut names = BTreeSet::new();
            for value in list {
                let name = value
                    .as_str()
                    .filter(|name| !name.trim().is_empty())
                    .ok_or_else(|| Rejection::invalid(field, "expected non-empty action names"))?;
                if !names.insert(name.trim().to_uppercase()) {
                    return Err(Rejection::invalid(field, "duplicate action name"));
                }
            }
            if names.len() < minimum {
                return Err(Rejection::invalid(field, "too few action names"));
            }
            Ok(names)
        };
        if let Some(value) = arguments.get("forbidden_actions") {
            safety.forbidden = names(value, "forbidden_actions", 0)?;
        }
        if let Some(value) = arguments.get("constraints") {
            let specs = value.as_array().ok_or_else(|| {
                Rejection::invalid("constraints", "expected an array of constraint objects")
            })?;
            for spec in specs {
                let obj = spec.as_object().ok_or_else(|| {
                    Rejection::invalid("constraints", "expected a constraint object")
                })?;
                if obj.keys().any(|key| key != "type" && key != "actions") {
                    return Err(Rejection::invalid(
                        "constraints",
                        "unexpected constraint key; probability bounds are unsupported",
                    ));
                }
                let kind = obj
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                let minimum = match kind.as_str() {
                    "forbid" => 1,
                    "mutually_exclusive" | "mutual_exclusive" => 2,
                    _ => {
                        return Err(Rejection::invalid(
                            "constraints",
                            "expected forbid or mutually_exclusive",
                        ))
                    }
                };
                let actions = names(
                    obj.get("actions").unwrap_or(&Value::Null),
                    "constraints",
                    minimum,
                )?;
                if kind == "forbid" {
                    safety.forbidden.extend(actions);
                } else {
                    safety.mutex.push(actions);
                }
            }
        }
        Ok(safety)
    }

    fn apply(&self, names: &[String], feasible: &mut [bool]) {
        let mut kept = BTreeSet::new();
        for (name, ok) in names.iter().zip(feasible.iter_mut()) {
            let name = name.trim().to_uppercase();
            *ok &= !self.forbidden.contains(&name)
                && !self
                    .mutex
                    .iter()
                    .any(|group| group.contains(&name) && !group.is_disjoint(&kept));
            if *ok {
                kept.insert(name);
            }
        }
    }
}

const BRIDGE_DISABLED: &str = "semantic bridge disabled (GENZERO_PYTHON_ENDPOINT=off)";

/// Why a semantic verb produced no semantic score.
pub(crate) enum Unscored {
    /// A local precondition failed; the bridge was never called.
    Skipped(String),
    /// The bridge was called and failed (transport, HTTP status, validation).
    Bridge(BridgeError),
}

impl Unscored {
    fn skipped(reason: &str) -> Self {
        Self::Skipped(reason.to_string())
    }

    /// `fallback_reason` for local skips, `bridge_error` for real call failures.
    fn record(&self, meta: &mut Value) {
        match self {
            Self::Skipped(reason) => meta["fallback_reason"] = json!(reason),
            Self::Bridge(err) => meta["bridge_error"] = json!(err.to_string()),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Skipped(reason) => reason.clone(),
            Self::Bridge(err) => err.to_string(),
        }
    }
}

/// Semantic safety check of one request text.
pub(crate) enum RiskCheck {
    /// No request text: nothing that could carry an intent.
    NotApplicable,
    Assessed {
        resp: SemanticRiskResponse,
        tier: PolicyTier,
    },
    /// The classifier could not answer. `tier` is the gate's fail-closed tier.
    Unassessed { why: Unscored, tier: PolicyTier },
}

impl RiskCheck {
    fn tier(&self) -> PolicyTier {
        match self {
            Self::NotApplicable => PolicyTier::Tier0Proceed,
            Self::Assessed { tier, .. } | Self::Unassessed { tier, .. } => *tier,
        }
    }

    fn meta(&self) -> Value {
        match self {
            Self::NotApplicable => json!({"assessed": false, "reason": "no request text"}),
            Self::Assessed { resp, tier } => json!({
                "assessed": true,
                "tier": tier_name(*tier),
                "p_dangerous": resp.p_dangerous,
                "thresholds": resp.thresholds,
                "classifier": resp.classifier,
                "windows": resp.windows,
                "forward_ms": resp.forward_ms,
            }),
            Self::Unassessed { why, tier } => json!({
                "assessed": false,
                "tier": tier_name(*tier),
                "fail_closed": true,
                "reason": why.describe(),
            }),
        }
    }

    /// Why the risk check raised the tier (only called when it did).
    fn message(&self) -> String {
        match self {
            Self::Assessed { resp, tier } if *tier == PolicyTier::Tier3HardStop => format!(
                "SafetyInterlockRejected: the semantic risk classifier rates this request as \
                 dangerous (p={:.3} >= hard-stop {:.3})",
                resp.p_dangerous, resp.thresholds.hard_stop
            ),
            Self::Assessed { resp, .. } => format!(
                "ConfirmationRequired: the semantic risk classifier rates this request as \
                 possibly dangerous (p={:.3} >= escalate {:.3})",
                resp.p_dangerous, resp.thresholds.escalate
            ),
            Self::Unassessed { why, .. } => format!(
                "ConfirmationRequired: the safety of this request could not be assessed ({})",
                why.describe()
            ),
            Self::NotApplicable => String::new(),
        }
    }
}

impl PolymorphicZeroEngine {
    /// Reject overload without creating an unbounded queue. The closure owns the
    /// permit: cancelling the request cannot free capacity while CPU work runs.
    fn spawn_cpu<F, T>(&self, work: F) -> Result<tokio::task::JoinHandle<T>, ServiceError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let permit = Arc::clone(&self.cpu_slots)
            .try_acquire_owned()
            .map_err(|_| ServiceError::Overloaded)?;
        Ok(tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work()
        }))
    }

    /// Current in-process audit commitment. Callers must retain this root externally.
    pub fn audit_ledger(&self) -> (usize, [u8; 32]) {
        let ledger = self.audit_ledger.lock();
        (ledger.len(), ledger.get_root())
    }

    pub fn audit_proof(
        &self,
        index: u64,
    ) -> Result<MmrInclusionProof, gen_zero_provenance::ProvenanceError> {
        self.audit_ledger.lock().generate_proof(index)
    }

    pub fn verify_audit_proof(&self, proof: &MmrInclusionProof, trusted_root: &[u8; 32]) -> bool {
        let ledger = self.audit_ledger.lock();
        let Ok(index) = usize::try_from(proof.leaf_index) else {
            return false;
        };
        let Some(entry) = ledger.get_entry(index) else {
            return false;
        };
        ledger.verify_inclusion_against_root(
            proof,
            trusted_root,
            &entry.hash_entry(&self.audit_key),
        )
    }
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|error| panic!("cannot initialize Gen-Zero engine: {error}"))
    }

    /// Replace the semantic bridge (`None` disables it).
    pub fn with_bridge(mut self, bridge: Option<Arc<SemanticBridgeClient>>) -> Self {
        self.bridge = bridge;
        self
    }

    pub fn with_gate(mut self, gate: PolicyGate) -> Self {
        self.gate = Arc::new(gate);
        self
    }

    /// Supply the live graph used by the production pipeline's policy gate.
    pub fn with_lod_graph(mut self, graph: Arc<LodGraph>) -> Self {
        self.graph = graph;
        self
    }

    /// Replace the world model the `pipeline` verb rolls forward.
    pub fn with_world_model(
        mut self,
        world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
    ) -> Self {
        self.world_model = world_model;
        self
    }

    /// Replace the mount registry (tests, or a host that preloads mounts).
    pub fn with_mounts(mut self, mounts: Arc<AtomicMountRegistry>) -> Self {
        self.mounts = mounts;
        self
    }

    /// Install a fleet containing externally loaded micro-core weights.
    pub fn with_nano_fleet(mut self, fleet: Arc<NanoCoreFleetScheduler>) -> Self {
        self.nano_fleet = fleet;
        self
    }

    pub fn bridge(&self) -> Option<&SemanticBridgeClient> {
        self.bridge.as_deref()
    }

    pub fn mounts(&self) -> &Arc<AtomicMountRegistry> {
        &self.mounts
    }

    /// Dispatch and execute the requested `zero` tool call.
    ///
    /// The mount snapshot is captured here, once, before any verb runs, and
    /// every verb reads only that snapshot.
    pub async fn execute(&self, arguments: &Value) -> Result<ZeroToolOutcome, ServiceError> {
        let started = Instant::now();
        let result = self.execute_inner(arguments).await;
        self.metrics
            .record_execution(arguments, &result, started.elapsed().as_secs_f64());
        result
    }

    async fn execute_inner(&self, arguments: &Value) -> Result<ZeroToolOutcome, ServiceError> {
        let mut verb = ZeroVerb::infer_from_input(arguments)?;
        // Explicit decision backends are restricted to the ask route. Reject
        // unknown values and unused inputs before any generic dispatch.
        let specialized = decision_backend(arguments, verb);
        if let Err(rej) = specialized {
            return Ok(ZeroToolOutcome::rejected(verb, rej, json!({})));
        }
        let binding = match self.capture(arguments) {
            Ok(b) => b,
            Err(rej) => return Ok(ZeroToolOutcome::rejected(verb, rej, json!({}))),
        };
        // `decide` modes: which engine really answers is fixed here, before
        // any verb runs, and reported back in `_meta.mode`.
        let route = match DecideRoute::resolve(arguments, verb) {
            Ok(route) => route,
            Err(rej) => {
                let meta = json!({"mount": mount_meta(binding.snapshot())});
                return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
            }
        };
        if matches!(route, DecideRoute::SemanticLookahead) {
            verb = ZeroVerb::Imagine;
        }
        let safety = match RequestSafety::parse(arguments) {
            Ok(safety) => safety,
            Err(rejection) => {
                return Ok(ZeroToolOutcome::rejected(
                    verb,
                    rejection,
                    json!({"formal_checked": false}),
                ))
            }
        };
        if !matches!(verb, ZeroVerb::Ask | ZeroVerb::Imagine)
            && ["constraints", "forbidden_actions", "enforce_cpsat"]
                .iter()
                .any(|field| arguments.get(field).is_some())
        {
            return Ok(ZeroToolOutcome::rejected(
                verb,
                Rejection::invalid(
                    "constraints",
                    "safety constraints are supported only for ask/decide and imagine",
                ),
                json!({"formal_checked": false}),
            ));
        }
        // Only ask and stream read `cognitive`. Anywhere else the geometry
        // would be dropped without a word, so the request is refused.
        if arguments.get("cognitive").is_some() && !matches!(verb, ZeroVerb::Ask | ZeroVerb::Stream)
        {
            let rej = Rejection::invalid(
                "request",
                format!("`cognitive` is only accepted by ask and stream, not {verb:?}"),
            );
            let meta = json!({"mount": mount_meta(binding.snapshot())});
            return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
        }

        // Only entail reads `entailment`; elsewhere it would be dropped silently.
        if arguments.get("entailment").is_some() && verb != ZeroVerb::Entail {
            let rej = Rejection::invalid(
                "request",
                format!("`entailment` is only accepted by entail, not {verb:?}"),
            );
            let meta = json!({"mount": mount_meta(binding.snapshot())});
            return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
        }

        // Only causal_fold reads `causal_fold`; elsewhere it would be dropped silently.
        if arguments.get("causal_fold").is_some() && verb != ZeroVerb::CausalFold {
            let rej = Rejection::invalid(
                "request",
                format!("`causal_fold` is only accepted by causal_fold, not {verb:?}"),
            );
            let meta = json!({"mount": mount_meta(binding.snapshot())});
            return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
        }

        // Only pipeline reads `pipeline`; elsewhere it would be dropped silently.
        if arguments.get("pipeline").is_some() && verb != ZeroVerb::Pipeline {
            let rej = Rejection::invalid(
                "request",
                format!("`pipeline` is only accepted by pipeline, not {verb:?}"),
            );
            let meta = json!({"mount": mount_meta(binding.snapshot())});
            return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
        }

        // The world-model fields are read only by the world-model verbs;
        // elsewhere they would be dropped without a word.
        if !verb.is_worldmodel() {
            for field in ["actions", "target_action", "continuation_actions"] {
                if arguments.get(field).is_some() {
                    let rej = Rejection::invalid(
                        "request",
                        format!(
                            "`{field}` is only accepted by simulate, what_if and audit, not {verb:?}"
                        ),
                    );
                    let meta = json!({"mount": mount_meta(binding.snapshot())});
                    return Ok(ZeroToolOutcome::rejected(verb, rej, meta));
                }
            }
        }
        let operator_domains = match nanocore_domains(arguments) {
            Ok(domains) => domains,
            Err(rej) => return Ok(ZeroToolOutcome::rejected(verb, rej, json!({}))),
        };
        if arguments.get("nanocore_state").is_some() && operator_domains.is_none() {
            return Ok(ZeroToolOutcome::rejected(
                verb,
                Rejection::invalid(
                    "nanocore",
                    "nanocore_state requires nanocore_domain or nanocore_domains",
                ),
                json!({}),
            ));
        }
        if operator_domains.is_some()
            && (verb != ZeroVerb::Ask || matches!(route, DecideRoute::Latent(_)))
        {
            return Ok(ZeroToolOutcome::rejected(
                verb,
                Rejection::invalid(
                    "nanocore",
                    "nanocore_domain(s) are only accepted by text ask",
                ),
                json!({}),
            ));
        }
        // Two micro-core routes exist: operator-loaded (`nanocore_domain`) and caller-supplied
        // (`engine`/`head`). Refuse the mix instead of letting one silently shadow the other.
        if operator_domains.is_some() && matches!(specialized, Ok(true)) {
            return Ok(ZeroToolOutcome::rejected(
                verb,
                Rejection::invalid(
                    "nanocore",
                    "nanocore_domain(s) cannot combine with engine/head decision backends",
                ),
                json!({}),
            ));
        }
        let mut outcome = match verb {
            ZeroVerb::Ask => match route {
                DecideRoute::Latent(mode) => {
                    self.handle_latent_plan(arguments, mode, &safety).await?
                }
                _ => self.handle_ask(arguments, &binding, &safety).await?,
            },
            ZeroVerb::Route => self.handle_route(arguments).await?,
            ZeroVerb::Imagine => self.handle_imagine(arguments, &safety).await?,
            ZeroVerb::Stream => self.handle_stream(arguments, &binding).await?,
            ZeroVerb::Grep => self.handle_grep(arguments),
            ZeroVerb::Compact => self.handle_compact(arguments).await?,
            ZeroVerb::Entail => {
                let args = arguments.clone();
                let snapshot = Arc::clone(binding.snapshot());
                let runtime = Arc::clone(&self.runtime);
                self.spawn_cpu(move || Self::handle_entail(&runtime, &args, snapshot))?
                    .await
                    .map_err(|e| ServiceError::Core(format!("entail task failed: {e}")))?
            }
            ZeroVerb::CausalFold => match Self::validate_causal_fold(arguments) {
                Err(rejection) => ZeroToolOutcome::rejected(
                    verb,
                    rejection,
                    json!({"engine": "relation_semiring_fold"}),
                ),
                Ok(block) => {
                    let block = block.clone();
                    self.spawn_cpu(move || Self::handle_causal_fold(&block))?
                        .await
                        .map_err(|e| ServiceError::Core(format!("causal fold task failed: {e}")))?
                }
            },
            ZeroVerb::Pipeline => {
                let args = arguments.clone();
                let model = Arc::clone(&self.world_model);
                let gate = Arc::clone(&self.gate);
                let graph = Arc::clone(&self.graph);
                self.spawn_cpu(move || Self::handle_pipeline(model, gate, graph, &args))?
                    .await
                    .map_err(|e| ServiceError::Core(format!("pipeline task failed: {e}")))?
            }
            ZeroVerb::Simulate => {
                self.run_worldmodel(verb, arguments, worldsim::simulate)
                    .await?
            }
            ZeroVerb::WhatIf => {
                self.run_worldmodel(verb, arguments, worldsim::what_if)
                    .await?
            }
            ZeroVerb::Audit => self.handle_audit(arguments).await?,
        };
        if !outcome.meta.is_object() {
            outcome.meta = json!({});
        }
        if verb == ZeroVerb::Ask {
            outcome.meta["best_action"] = outcome.meta["chosen_action"].clone();
            let mut probs = serde_json::Map::new();
            if let Some(candidates) = outcome.meta["candidates"].as_array() {
                for candidate in candidates {
                    if let (Some(name), Some(probability)) =
                        (candidate["name"].as_str(), candidate.get("probability"))
                    {
                        probs.insert(name.to_owned(), probability.clone());
                    }
                }
            }
            if outcome.meta.get("confidence").is_none() {
                outcome.meta["confidence"] = outcome.meta["chosen_action"]
                    .as_str()
                    .and_then(|name| probs.get(name))
                    .cloned()
                    .unwrap_or(json!(0.0));
            }
            outcome.meta["probs"] = Value::Object(probs);
        }
        if verb == ZeroVerb::Ask && !outcome.is_error {
            let action = outcome
                .meta
                .get("action_id")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ServiceError::Core("successful ask has no action_id for audit".into())
                })?;
            let action = u32::try_from(action).map_err(|e| ServiceError::Core(e.to_string()))?;
            let tier = match outcome.meta.get("tier").and_then(Value::as_str) {
                Some("Proceed") => PolicyTier::Tier0Proceed,
                Some("Confirm") => PolicyTier::Tier1Confirm,
                Some("Escalate") => PolicyTier::Tier2Escalate,
                Some("HardStop") => PolicyTier::Tier3HardStop,
                other => {
                    return Err(ServiceError::Core(format!(
                        "unknown decision tier: {other:?}"
                    )))
                }
            };
            self.record_decision(arguments, &mut outcome.meta, ActionId(action), tier)
                .await?;
        }
        outcome.meta["mount"] = mount_meta(binding.snapshot());
        route.annotate(arguments, &mut outcome.meta);
        if verb == ZeroVerb::Pipeline
            && arguments.pointer("/pipeline/op").and_then(Value::as_str) == Some("decide")
            && outcome.meta.pointer("/pipeline/decision").is_some()
        {
            let Some(action) = outcome
                .meta
                .pointer("/pipeline/decision/action")
                .and_then(Value::as_u64)
                .and_then(|a| u32::try_from(a).ok())
            else {
                return Err(ServiceError::Core(
                    "pipeline decision missing valid action for audit".into(),
                ));
            };
            let tier: PolicyTier =
                serde_json::from_value(outcome.meta["pipeline"]["decision"]["gate_tier"].clone())
                    .map_err(|e| {
                    ServiceError::Core(format!("pipeline decision missing gate tier: {e}"))
                })?;
            let request =
                serde_json::to_vec(arguments).map_err(|e| ServiceError::Core(e.to_string()))?;
            let result =
                serde_json::to_vec(&outcome.meta).map_err(|e| ServiceError::Core(e.to_string()))?;
            let leaf = self
                .append_audit_leaf(
                    *blake3::hash(&request).as_bytes(),
                    *blake3::hash(&result).as_bytes(),
                    ActionId(action),
                    tier as u8,
                )
                .await?;
            outcome.meta["audit_ledger"] = json!({"leaf_index": leaf.index, "leaf_count": leaf.leaf_count, "root": digest_hex(&leaf.root), "proof_window": leaf.window, "persistence": self.audit_persistence_name(), "formal_certificate": "unavailable", "tier": tier_name(tier), "gate_status": gate_status(tier)});
        }
        if arguments.get("cognitive").is_none() && verb != ZeroVerb::Entail && !verb.is_worldmodel()
        {
            outcome.meta["cognitive_runtime"] =
                json!("not_engaged: request carries no numeric manifold coordinates");
        }
        Ok(outcome)
    }

    /// Capture the request's snapshot and enforce a requested generation.
    fn capture(&self, arguments: &Value) -> Result<RequestBinding, Rejection> {
        let key = mount_key(arguments)?;
        let binding = RequestBinding::capture(&*self.mounts, &key).map_err(|r| {
            Rejection::reject(
                r,
                "mount_capture",
                format!("{r}: tenant={} workspace={}", key.tenant, key.workspace),
            )
        })?;
        if let Some(v) = arguments.get("mount_version") {
            let wanted = v.as_u64().ok_or_else(|| {
                Rejection::invalid("mount_capture", "mount_version must be a u64")
            })?;
            let live = binding.snapshot().version().0;
            if wanted != live {
                return Err(Rejection::reject(
                    Reject::EpochMismatch,
                    "mount_capture",
                    format!(
                        "request is bound to mount version {wanted}, the mounted version is {live}"
                    ),
                ));
            }
        }
        Ok(binding)
    }

    /// Publish a new generation of cognitive assets (Spec 25 §5.3): validate
    /// the assets, derive the next snapshot from the named base, seal it and
    /// compare-and-swap it in. A stale `base_version` is `CasConflict`; the
    /// caller must re-read the mount and decide again.
    pub fn publish_assets(&self, body: &Value) -> Result<Value, Rejection> {
        let key = mount_key(body)?;
        let base_version = body
            .get("base_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| Rejection::invalid("publish", "base_version must be a u64"))?;
        let reason = body
            .get("reason")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Rejection::invalid("publish", "reason must say why this generation exists")
            })?;
        let assets = CognitiveAssets::from_json(
            body.get("assets")
                .ok_or_else(|| Rejection::invalid("publish", "assets is required"))?,
        )?;
        let publish = |r: Reject| Rejection::reject(r, "publish", r.to_string());

        let base = self.mounts.load(&key).map_err(publish)?;
        if base.version().0 != base_version {
            return Err(Rejection::reject(
                Reject::CasConflict,
                "publish",
                format!(
                    "base_version {base_version} is not the mounted version {}",
                    base.version().0
                ),
            ));
        }
        let fam = assets.families();
        let change = SnapshotChange {
            model: Some(fam.model),
            geometry: Some(fam.geometry),
            policy: Some(fam.policy),
            assets: Some(Arc::from(assets.canonical_bytes())),
            ..SnapshotChange::default()
        };
        let next = base.derive(&change).map_err(publish)?;
        let proposal =
            Proposal::new(Arc::clone(&base), Arc::from(reason.as_bytes())).map_err(publish)?;
        let candidate = CandidateMount::new(Arc::new(next), &proposal);
        let gate = assets.gate();
        let budget = Budget {
            max_steps: gate.max_steps,
            max_time_ns: 1_000_000_000,
            max_bytes: 1 << 20,
            residual_limit: gate.residual_limit,
            numeric_error: gate.numeric_error,
            policy: fam.policy,
        };
        let sealed = self
            .mounts
            .validate(proposal, candidate, &budget)
            .map_err(publish)?;
        let published = self
            .mounts
            .compare_and_mount(&key, sealed)
            .map_err(publish)?;
        let live = self.mounts.load(&key).map_err(publish)?;
        tracing::info!(
            tenant = %key.tenant,
            workspace = %key.workspace,
            from = published.from.0,
            to = published.to.0,
            digest = %digest_hex(&published.digest),
            "cognitive assets published"
        );
        Ok(json!({
            "published": {
                "from": published.from.0,
                "to": published.to.0,
                "digest": digest_hex(&published.digest),
            },
            // SHA-256 family digests sealed into the new epochs. `geometry`
            // covers the entailment preset, so a width change is a new seal.
            "sealed": {
                "model": digest_hex(&fam.model),
                "geometry": digest_hex(&fam.geometry),
                "policy": digest_hex(&fam.policy),
            },
            "mount": mount_meta(&live),
            "assets": assets.summary(),
        }))
    }

    /// Retain complete graph-aware verdicts, including confirmation and revocation.
    fn candidate_verdicts(
        &self,
        names: &[String],
        entropy: NormalizedEntropy,
    ) -> Result<Vec<gen_zero_gate::GateVerdict>, ServiceError> {
        names
            .iter()
            .map(|name| {
                self.gate
                    .evaluate(action_id(name), entropy, Some(self.graph.as_ref()), None)
                    .map_err(|e| ServiceError::Core(e.to_string()))
            })
            .collect()
    }

    fn formal_feasibility(&self, names: &[String]) -> Vec<bool> {
        names
            .iter()
            .map(|name| {
                self.gate
                    .evaluate(
                        action_id(name),
                        NormalizedEntropy::ZERO,
                        Some(self.graph.as_ref()),
                        None,
                    )
                    .is_ok_and(|v| v.tier != PolicyTier::Tier3HardStop)
            })
            .collect()
    }

    fn request_feasibility(&self, names: &[String], safety: &RequestSafety) -> Vec<bool> {
        // Disabling the formal solver never disables the operator's gate or caller prohibitions.
        let mut feasible = if safety.enforce_cpsat {
            self.formal_feasibility(names)
        } else {
            names
                .iter()
                .map(|name| {
                    self.gate
                        .evaluate_basic(action_id(name), NormalizedEntropy::ZERO)
                        .is_ok_and(|verdict| verdict.tier != PolicyTier::Tier3HardStop)
                })
                .collect()
        };
        safety.apply(names, &mut feasible);
        feasible
    }

    /// Semantic risk of the request text through the bridge and the gate.
    async fn assess_risk(&self, text: &str) -> RiskCheck {
        let text = text.trim();
        if text.is_empty() {
            return RiskCheck::NotApplicable;
        }
        let answer = match self.bridge() {
            Some(bridge) => bridge.semantic_risk(text).await.map_err(Unscored::Bridge),
            None => Err(Unscored::skipped(BRIDGE_DISABLED)),
        };
        match answer {
            Ok(resp) => {
                let tier = self.gate.evaluate_semantic_risk(Some(SemanticRisk {
                    p_dangerous: resp.p_dangerous as f32,
                    escalate_threshold: resp.thresholds.escalate as f32,
                    hard_stop_threshold: resp.thresholds.hard_stop as f32,
                }));
                RiskCheck::Assessed { resp, tier }
            }
            Err(why) => {
                tracing::warn!("request risk not assessed: {}", why.describe());
                let tier = self.gate.evaluate_semantic_risk(None);
                RiskCheck::Unassessed { why, tier }
            }
        }
    }

    /// Fail-closed exit before any scoring when the request itself is a hard stop.
    fn risk_hard_stop(verb: ZeroVerb, risk: &RiskCheck) -> Option<ZeroToolOutcome> {
        (risk.tier() == PolicyTier::Tier3HardStop).then(|| {
            Self::gated_outcome(
                verb,
                PolicyTier::Tier0Proceed,
                risk,
                json!({"engine": "semantic_risk_gate", "semantic_scoring": false}),
                String::new(),
            )
        })
    }

    /// Map the gate verdict and the request risk to the MCP outcome. The final
    /// tier is the stricter of the two. `meta` is kept in every branch so the
    /// caller always sees which engine produced the decision.
    pub(crate) fn gated_outcome(
        verb: ZeroVerb,
        gate_tier: PolicyTier,
        risk: &RiskCheck,
        mut meta: Value,
        success: String,
    ) -> ZeroToolOutcome {
        let tier = gate_tier.max(risk.tier());
        meta["tier"] = json!(tier_name(tier));
        meta["risk"] = risk.meta();
        let risk_decides = risk.tier() == tier && tier != PolicyTier::Tier0Proceed;
        meta["gate_status"] = json!(gate_status(tier));
        let (is_error, text) = match tier {
            PolicyTier::Tier3HardStop => {
                meta["error_code"] = json!(-32001);
                meta["fail_closed"] = json!(true);
                let text = if risk_decides {
                    risk.message()
                } else {
                    "SafetyInterlockRejected: PolicyGate hard stop triggered".to_string()
                };
                (true, text)
            }
            PolicyTier::Tier2Escalate => {
                meta["error_code"] = json!(-32002);
                meta["requires_confirmation"] = json!(true);
                let text = if risk_decides {
                    risk.message()
                } else {
                    "ConfirmationRequired: PolicyGate epistemic uncertainty requires escalation"
                        .to_string()
                };
                (true, text)
            }
            PolicyTier::Tier1Confirm => {
                meta["error_code"] = json!(-32002);
                meta["requires_confirmation"] = json!(true);
                (
                    true,
                    "ConfirmationRequired: Irreversible action requires human confirmation"
                        .to_string(),
                )
            }
            PolicyTier::Tier0Proceed => (false, success),
        };
        ZeroToolOutcome {
            verb,
            is_error,
            content: text_block(text),
            meta,
            rejection: None,
        }
    }

    /// Verb 1: `ask` (semantic decision, PolicyGate-checked)
    async fn handle_ask(
        &self,
        arguments: &Value,
        binding: &RequestBinding,
        safety: &RequestSafety,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let context_str = first_text(arguments, &["context", "questions", "state"]);
        let candidates = match string_list(arguments.get("candidates"), "candidates") {
            Ok(candidates) => candidates,
            Err(rejection) => {
                return Ok(ZeroToolOutcome::rejected(
                    ZeroVerb::Ask,
                    rejection,
                    json!({}),
                ))
            }
        };
        if contains_raw_control_marker(context_str)
            || candidates.iter().any(|c| contains_raw_control_marker(c))
        {
            return Ok(ZeroToolOutcome::rejected(
                ZeroVerb::Ask,
                Rejection::invalid(
                    "request",
                    "raw model control delimiter in context or candidate",
                ),
                json!({}),
            ));
        }

        // Without context the candidates themselves are the request.
        let risk_text = if context_str.is_empty() {
            candidates.join("\n")
        } else {
            context_str.to_string()
        };
        let risk = self.assess_risk(&risk_text).await;
        if let Some(stop) = Self::risk_hard_stop(ZeroVerb::Ask, &risk) {
            return Ok(stop);
        }

        let cognitive = arguments.get("cognitive");
        if cognitive.is_some() && candidates.is_empty() {
            return Ok(ZeroToolOutcome::rejected(
                ZeroVerb::Ask,
                Rejection::invalid("request", "a cognitive ask needs candidates"),
                json!({"engine": ENGINE_COGNITIVE, "risk": risk.meta()}),
            ));
        }

        // No candidates means nothing to choose from. Never invent an action.
        if candidates.is_empty() {
            return Ok(ZeroToolOutcome::rejected(
                ZeroVerb::Ask,
                Rejection::invalid("request", "ask needs at least one candidate"),
                json!({"risk": risk.meta()}),
            ));
        }

        let feasible_mask = self.request_feasibility(&candidates, safety);
        let feasible: Vec<String> = candidates
            .iter()
            .zip(&feasible_mask)
            .filter(|(_, ok)| **ok)
            .map(|(c, _)| c.clone())
            .collect();
        let infeasible: Vec<&String> = candidates
            .iter()
            .zip(&feasible_mask)
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| c)
            .collect();
        if feasible.is_empty() {
            return Ok(Self::gated_outcome(
                ZeroVerb::Ask,
                PolicyTier::Tier3HardStop,
                &risk,
                json!({"engine": "formal_filter", "formally_infeasible": infeasible}),
                String::new(),
            ));
        }

        if let Some(domains) = match nanocore_domains(arguments) {
            Ok(domains) => domains,
            Err(rej) => return Ok(ZeroToolOutcome::rejected(ZeroVerb::Ask, rej, json!({}))),
        } {
            if cognitive.is_some() {
                return Ok(ZeroToolOutcome::rejected(
                    ZeroVerb::Ask,
                    Rejection::invalid(
                        "nanocore",
                        "nanocore_domain(s) cannot combine with cognitive decision",
                    ),
                    json!({}),
                ));
            }
            return self.nanocore_ask(arguments, &feasible, &domains, &risk);
        }

        if decision_backend(arguments, ZeroVerb::Ask).expect("validated at dispatch") {
            return self.specialized_ask(arguments, &feasible, &infeasible, &risk);
        }

        if let Some(block) = cognitive {
            return self
                .certified_ask(block, binding, &feasible, &infeasible, &risk)
                .await;
        }

        let state = arguments.get("state").filter(|s| !s.is_string());
        let scored = if feasible.len() < 2 {
            Err(Unscored::skipped(
                "fewer than two feasible candidates; nothing to score",
            ))
        } else if context_str.is_empty() {
            Err(Unscored::skipped(
                "no context text to score candidates against",
            ))
        } else {
            match self.bridge() {
                Some(bridge) => bridge
                    .semantic_ask(&AskInput {
                        context: context_str,
                        candidates: &feasible,
                        state,
                        history: &[],
                        return_embedding: false,
                    })
                    .await
                    .map_err(Unscored::Bridge),
                None => Err(Unscored::skipped(BRIDGE_DISABLED)),
            }
        };

        let (chosen, probs, mut meta) = match scored {
            Ok(resp) => {
                let probs: Vec<f64> = resp.candidates.iter().map(|c| c.probability).collect();
                let meta = json!({
                    "engine": ENGINE_SEMANTIC,
                    "semantic_scoring": true,
                    "bridge_endpoint": self.bridge().map(|b| b.endpoint()),
                    "scorer": resp.scorer,
                    "candidates": resp.candidates,
                    "embedding_dim": resp.embedding_dim,
                    "bridge_timing_ms": resp.timing_ms,
                });
                (resp.chosen, probs, meta)
            }
            Err(why) => {
                tracing::warn!("ask falls back to local reflex: {}", why.describe());
                let uniform = 1.0 / feasible.len() as f64;
                let mut meta = json!({
                    "engine": ENGINE_FALLBACK,
                    "semantic_scoring": false,
                    "bridge_endpoint": self.bridge().map(|b| b.endpoint()),
                    "decision_rule": "first_feasible_candidate_uniform_prior",
                    "candidates": feasible
                        .iter()
                        .map(|c| json!({"name": c, "probability": uniform}))
                        .collect::<Vec<_>>(),
                });
                why.record(&mut meta);
                (feasible[0].clone(), vec![uniform; feasible.len()], meta)
            }
        };

        let p32: Vec<f32> = probs.iter().map(|&p| p as f32).collect();
        let entropy = NormalizedEntropy::from_probabilities(&p32);
        let chosen_id = action_id(&chosen);
        let confidence = meta["candidates"]
            .as_array()
            .and_then(|candidates| {
                candidates
                    .iter()
                    .find(|candidate| candidate["name"] == chosen)
            })
            .and_then(|candidate| candidate["probability"].as_f64())
            .unwrap_or(0.0);
        meta["chosen_action"] = json!(chosen);
        meta["action_id"] = json!(chosen_id.0);
        meta["confidence"] = json!(confidence);
        meta["entropy"] = json!(entropy.0);
        meta["formally_infeasible"] = json!(infeasible);

        let verdict = self
            .gate
            .evaluate(chosen_id, entropy, Some(self.graph.as_ref()), None)
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        Ok(Self::gated_outcome(
            ZeroVerb::Ask,
            verdict.tier,
            &risk,
            meta,
            format!("Selected action: {}", chosen),
        ))
    }

    /// Explicit operator micro-core inference. Requires one or more registered
    /// cores and 128 finite coordinates; selected cores are fused by the
    /// Nanocore MoV engine in the order requested by the caller.
    fn nanocore_ask(
        &self,
        arguments: &Value,
        feasible: &[String],
        domains: &[DomainId],
        risk: &RiskCheck,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let domain_values: Vec<u32> = domains.iter().map(|domain| domain.0).collect();
        let refuse = |detail: String| {
            ZeroToolOutcome::rejected(
                ZeroVerb::Ask,
                Rejection::invalid("nanocore", detail),
                json!({"engine": "nanocore", "domains": domain_values.clone()}),
            )
        };
        let state: Vec<f32> = match arguments.get("nanocore_state").and_then(Value::as_array) {
            Some(v) if v.len() == 128 => match v
                .iter()
                .map(|x| x.as_f64().filter(|f| f.is_finite()).map(|f| f as f32))
                .collect::<Option<Vec<_>>>()
            {
                Some(v) if v.iter().all(|x| x.is_finite()) => v,
                _ => {
                    return Ok(refuse(
                        "nanocore_state must contain 128 finite numbers".into(),
                    ))
                }
            },
            _ => {
                return Ok(refuse(
                    "nanocore_state must contain 128 finite numbers".into(),
                ))
            }
        };
        let mut cores = Vec::with_capacity(domains.len());
        for domain in domains {
            let core = match self.nano_fleet.get_core(*domain) {
                Ok(core) => core,
                Err(e) => {
                    return Ok(refuse(format!(
                        "micro-core unavailable (domain {}): {e}",
                        domain.0
                    )))
                }
            };
            if let Err(error) = core.validate() {
                return Ok(refuse(format!(
                    "micro-core domain {} is invalid: {error}",
                    domain.0
                )));
            }
            cores.push(core);
        }
        if cores.is_empty() {
            return Ok(refuse(
                "micro-core mixture must select at least one core".into(),
            ));
        }
        let core_refs: Vec<&NanoCoreInstance> = cores.iter().map(Arc::as_ref).collect();
        let names: Vec<&str> = feasible.iter().map(String::as_str).collect();
        let ids: Vec<ActionId> = feasible.iter().map(|s| action_id(s)).collect();
        let Some(frame) = LocalActionFrame::new(&names, &ids) else {
            return Ok(refuse("micro-core supports at most 16 candidates".into()));
        };
        let state =
            CompressedLatent::from_slice(&state).map_err(|e| ServiceError::Core(e.to_string()))?;
        let decision = match MoVFusionEngine::new(WatchdogConfig::default())
            .fuse(&core_refs, &state, &frame)
        {
            Ok(d) => d,
            Err(e) => {
                return Ok(refuse(format!(
                    "micro-core mixture inference refused for domains {:?}: {e}",
                    domain_values
                )))
            }
        };
        let Some(index) = ids.iter().position(|id| *id == decision.selected_action) else {
            return Err(ServiceError::Core(
                "micro-core returned an unknown action".into(),
            ));
        };
        let verdict = self
            .gate
            .evaluate(
                decision.selected_action,
                decision.composite_entropy,
                Some(self.graph.as_ref()),
                None,
            )
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        let weights: Vec<Value> = decision
            .weights
            .iter()
            .map(|(domain, weight)| json!({"domain": domain.0, "weight": weight}))
            .collect();
        let candidates_meta: Vec<Value> = feasible
            .iter()
            .zip(decision.probabilities.iter())
            .map(|(name, probability)| json!({"name": name, "probability": probability}))
            .collect();
        let mut meta = json!({
            "engine": "nanocore",
            "domains": domain_values,
            "core_count": domains.len(),
            "weights": weights,
            "chosen_action": feasible[index],
            "best_action": feasible[index],
            "action_id": decision.selected_action.0,
            "probabilities": decision.probabilities,
            "candidates": candidates_meta,
            "confidence": decision.composite_confidence,
            "composite_value": decision.composite_value,
            "entropy": decision.composite_entropy.0
        });
        // Keep the singular response field stable for existing callers.
        if let Some(domain) = domains.first().filter(|_| domains.len() == 1) {
            meta["domain"] = json!(domain.0);
        }
        Ok(Self::gated_outcome(
            ZeroVerb::Ask,
            verdict.tier,
            risk,
            meta,
            format!("Selected action: {}", feasible[index]),
        ))
    }

    /// Decision evidence. The response exposes the root and leaf for verification. Appends and
    /// root reads are O(1) amortized under the lock; only the last proof window leaves stay
    /// provable. A configured persistence path commits the bounded snapshot before publication.
    async fn record_decision(
        &self,
        arguments: &Value,
        meta: &mut Value,
        action: ActionId,
        tier: PolicyTier,
    ) -> Result<(), ServiceError> {
        let state = serde_json::to_vec(arguments).map_err(|e| ServiceError::Core(e.to_string()))?;
        let candidates = serde_json::to_vec(&arguments.get("candidates"))
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        let leaf = self
            .append_audit_leaf(
                *blake3::hash(&state).as_bytes(),
                *blake3::hash(&candidates).as_bytes(),
                action,
                tier as u8,
            )
            .await?;
        let scope = if self.audit_persistence.is_some() {
            "durable"
        } else {
            "process_local"
        };
        meta["decision_audit"] = json!({"scope": scope, "leaf_index": leaf.index, "leaf_count": leaf.leaf_count, "root": digest_hex(&leaf.root), "proof_window": leaf.window, "persistence": self.audit_persistence_name(), "formal_proof": false});
        Ok(())
    }

    fn audit_persistence_name(&self) -> &'static str {
        if self.audit_persistence.is_some() {
            "durable"
        } else {
            "in_process_only"
        }
    }

    /// Append one leaf to the shared audit ledger under the audit writer's capability, then
    /// prove and verify it before the caller may report it. Durable writes run in a bounded
    /// blocking worker; a separate writer lock serializes candidates while the ledger lock is
    /// held only for short clone/publish operations.
    async fn append_audit_leaf(
        &self,
        request_hash: [u8; 32],
        result_hash: [u8; 32],
        action: ActionId,
        tier: u8,
    ) -> Result<AuditLeaf, ServiceError> {
        self.audit_arbiter
            .authorize(&self.audit_token, CapabilityFlags::AUDIT_ADMIN)
            .map_err(|e| ServiceError::Core(format!("audit writer denied: {e}")))?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ServiceError::Core(e.to_string()))?
            .as_nanos();
        let timestamp = u64::try_from(timestamp).map_err(|e| ServiceError::Core(e.to_string()))?;
        let input = AuditAppendInput {
            request_hash,
            result_hash,
            action,
            tier,
            timestamp,
        };
        let ledger = Arc::clone(&self.audit_ledger);
        let audit_key = self.audit_key;
        if let Some(persistence) = self.audit_persistence.clone() {
            let writer = Arc::clone(&self.audit_writer);
            // The blocking closure owns the writer guard and candidate across the complete
            // snapshot write, file sync, atomic rename, and directory sync.
            // Once started, Tokio does not cancel `spawn_blocking`; a cancelled request can
            // therefore not leave a durable snapshot without the matching in-memory commit.
            self.spawn_cpu(move || {
                let _writer = writer.lock();
                let mut candidate = ledger.lock().clone();
                let leaf = Self::append_audit_leaf_to_ledger(&mut candidate, audit_key, input)?;
                persistence
                    .persist(&candidate)
                    .map_err(|e| ServiceError::Core(e.to_string()))?;
                let mut ledger = ledger.lock();
                *ledger = candidate;
                Ok(leaf)
            })?
            .await
            .map_err(|e| ServiceError::Core(format!("durable audit task failed: {e}")))?
        } else {
            let mut ledger = ledger.lock();
            Self::append_audit_leaf_to_ledger(&mut ledger, audit_key, input)
        }
    }

    /// Append and verify one audit leaf in a ledger. Durable callers pass a cloned candidate
    /// and publish it only after its complete snapshot has been persisted.
    fn append_audit_leaf_to_ledger(
        ledger: &mut MmrLedger,
        audit_key: [u8; 32],
        input: AuditAppendInput,
    ) -> Result<AuditLeaf, ServiceError> {
        let index = ledger.len() as u64;
        let entry = DecisionAuditEntry::new(
            index,
            ledger.get_root(),
            input.request_hash,
            input.result_hash,
            [0; 32],
            input.action,
            input.tier,
            input.timestamp,
        );
        let expected_leaf_hash = entry.hash_entry(&audit_key);
        let appended_index = ledger.append(entry);
        if appended_index != index {
            return Err(ServiceError::Core(format!(
                "audit ledger append index {} does not match expected index {index}",
                appended_index
            )));
        }
        let root = ledger.get_root();
        let proof = ledger
            .generate_proof(index)
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        if !MmrLedger::verify_against_root(&audit_key, &proof, &root, &expected_leaf_hash)
            .map_err(|e| ServiceError::Core(e.to_string()))?
        {
            return Err(ServiceError::Core(
                "audit inclusion verification failed".into(),
            ));
        }
        Ok(AuditLeaf {
            index,
            leaf_count: ledger.len(),
            root,
            window: ledger.window(),
        })
    }

    fn specialized_ask(
        &self,
        arguments: &Value,
        feasible: &[String],
        infeasible: &[&String],
        risk: &RiskCheck,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let reject = |detail: String| {
            ZeroToolOutcome::rejected(
                ZeroVerb::Ask,
                Rejection::invalid("request", detail),
                json!({"engine": arguments.get("engine"), "head": arguments.get("head")}),
            )
        };
        let engine = arguments
            .get("engine")
            .and_then(Value::as_str)
            .unwrap_or("generic");
        let head = arguments
            .get("head")
            .and_then(Value::as_str)
            .unwrap_or("linear");
        let parse_vec = |field: &str| -> Result<Vec<f32>, String> {
            let raw = arguments
                .get(field)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("`{field}` must be a numeric array"))?;
            raw.iter()
                .map(|v| {
                    let n = v
                        .as_f64()
                        .ok_or_else(|| format!("`{field}` contains a non-number"))?;
                    if !n.is_finite() || n.abs() > f32::MAX as f64 {
                        return Err(format!("`{field}` contains an invalid float"));
                    }
                    Ok(n as f32)
                })
                .collect()
        };
        // Canonical ascending-ActionId order makes every reduction below (ETF,
        // normalization, argmax ties) independent of the caller's candidate order.
        // NanoCore channels are bound by name through `action_vocab`, not by slot.
        let mut canonical_candidates = feasible.to_vec();
        canonical_candidates.sort_by_key(|name| action_id(name).0);
        let feasible = canonical_candidates.as_slice();
        let mut nano_logits = None;
        let mut nano_meta = json!(null);
        let state = if engine == "nanocore" {
            let values = match parse_vec("decision_state") {
                Ok(v) => v,
                Err(e) => return Ok(reject(e)),
            };
            let arr: [f32; 128] = match values.try_into() {
                Ok(v) => v,
                Err(_) => {
                    return Ok(reject(
                        "`decision_state` must have exactly 128 floats".into(),
                    ))
                }
            };
            let core: NanoCoreInstance = match arguments
                .get("nanocore_core")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
            {
                Some(v) => v,
                None => {
                    return Ok(reject(
                        "`nanocore_core` must provide serialized core parameters".into(),
                    ))
                }
            };
            if let Err(e) = core.validate() {
                return Ok(reject(format!("invalid nanocore: {e}")));
            }
            // Caller weights run directly and never enter `nano_fleet`: registering them there
            // would overwrite an operator-loaded core of the same domain.
            let latent = CompressedLatent { values: arr };
            let names: Vec<&str> = feasible.iter().map(String::as_str).collect();
            match core.forward(&latent, &names) {
                Ok((logits, confidence, value)) => {
                    nano_logits = Some(logits);
                    nano_meta = json!({"domain_id": core.domain_id.0, "confidence": confidence, "value": value});
                }
                Err(e) => return Ok(reject(format!("nanocore forward failed: {e}"))),
            }
            Some(latent)
        } else {
            None
        };
        let mut etf_meta = json!(null);
        let etf_probs = if head == "etf" {
            let rep = if let Some(ref state) = state {
                state.as_slice().to_vec()
            } else {
                match parse_vec("etf_rep") {
                    Ok(v) => v,
                    Err(e) => return Ok(reject(e)),
                }
            };
            if rep.is_empty() || rep.len() > 4096 {
                return Ok(reject("ETF state has invalid dimension".into()));
            }
            let reps = match parse_candidate_reps(arguments, feasible, infeasible, rep.len()) {
                Ok(v) => v,
                Err(e) => return Ok(reject(e)),
            };
            let (temperature, temperature_source) = match arguments.get("etf_temperature") {
                None => (DEFAULT_UNCALIBRATED_TEMPERATURE, "default_uncalibrated"),
                Some(v) => match v.as_f64() {
                    Some(t) if t.is_finite() && t > 0.0 && t <= f32::MAX as f64 => {
                        (t as f32, "caller")
                    }
                    _ => {
                        return Ok(reject(
                            "`etf_temperature` must be a finite positive number".into(),
                        ))
                    }
                },
            };
            let names: Vec<&str> = feasible.iter().map(String::as_str).collect();
            let ids: Vec<ActionId> = feasible.iter().map(|s| action_id(s)).collect();
            let frame = match LocalActionFrame::new(&names, &ids) {
                Some(v) => v,
                None => return Ok(reject("ETF supports at most 16 candidates".into())),
            };
            let metric = match parse_etf_metric(arguments, rep.len()) {
                Ok(v) => v,
                Err(e) => return Ok(reject(e)),
            };
            let metric_source = if metric.is_some() {
                "caller"
            } else {
                "default"
            };
            let metric = metric.unwrap_or(MetricKind::Isotropic);
            let choice = match ActionETFChoiceHead::with_metric(rep.len(), temperature, &metric) {
                Ok(v) => v,
                Err(e) => return Ok(reject(format!("ETF head failed: {e}"))),
            };
            let rep_refs: Vec<&[f32]> = reps.iter().map(Vec::as_slice).collect();
            match choice.evaluate(&rep, &rep_refs, &frame) {
                Ok(scores) => {
                    let scoring = match choice.metric() {
                        MetricKind::Isotropic => "cosine(state, candidate_rep) / temperature",
                        MetricKind::DiagonalMahalanobis(_) => {
                            "cosine_w(state, candidate_rep), <u,v>_w = sum w_i u_i v_i, / temperature"
                        }
                        MetricKind::WhitenedProjection { .. } => {
                            "cosine(W state, W candidate_rep) / temperature"
                        }
                    };
                    etf_meta = json!({
                        "scoring": scoring,
                        "metric": choice.metric().name(),
                        "metric_source": metric_source,
                        "metric_input_dim": choice.dimension(),
                        "metric_output_dim": choice.metric_dimension(),
                        "temperature": temperature,
                        "temperature_source": temperature_source,
                        "similarities": feasible.iter().zip(&scores.similarities)
                            .map(|(name, s)| json!({"name": name, "similarity": s}))
                            .collect::<Vec<_>>(),
                    });
                    Some(scores.probabilities)
                }
                Err(e) => return Ok(reject(format!("ETF head failed: {e}"))),
            }
        } else {
            None
        };
        let mut probs = if let Some(logits) = nano_logits {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>()
        } else {
            vec![1.0; feasible.len()]
        };
        if let Some(etf) = etf_probs {
            for (p, e) in probs.iter_mut().zip(etf) {
                *p *= e;
            }
        }
        let sum: f32 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Ok(reject("invalid decision probability mass".into()));
        }
        for p in &mut probs {
            *p /= sum;
        }
        let best = (0..probs.len())
            .max_by(|&a, &b| {
                probs[a]
                    .total_cmp(&probs[b])
                    .then_with(|| action_id(&feasible[b]).0.cmp(&action_id(&feasible[a]).0))
            })
            .unwrap();
        let chosen = &feasible[best];
        let entropy = NormalizedEntropy::from_probabilities(&probs);
        let verdict = self
            .gate
            .evaluate(action_id(chosen), entropy, Some(self.graph.as_ref()), None)
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        Ok(Self::gated_outcome(
            ZeroVerb::Ask,
            verdict.tier,
            risk,
            json!({"engine": engine, "head": head, "chosen_action": chosen, "action_id": action_id(chosen).0,
                "candidates": feasible.iter().zip(&probs).map(|(name,p)| json!({"name":name,"probability":p})).collect::<Vec<_>>(),
                "entropy": entropy.0, "nanocore": nano_meta, "etf": etf_meta, "formally_infeasible": infeasible,
                "training_status": "caller_supplied_parameters_unverified"}),
            format!("Selected action: {chosen}"),
        ))
    }

    /// `ask` with manifold coordinates: every feasible candidate's control
    /// window runs through [`CognitiveRuntime::decide`] on the captured
    /// snapshot. Only a certified action reaches the PolicyGate; any refusal
    /// is typed and no action is chosen.
    async fn certified_ask(
        &self,
        block: &Value,
        binding: &RequestBinding,
        feasible: &[String],
        infeasible: &[&String],
        risk: &RiskCheck,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let snapshot = Arc::clone(binding.snapshot());
        let mut meta = json!({
            "engine": ENGINE_COGNITIVE,
            "semantic_scoring": false,
            "formally_infeasible": infeasible,
            "certified_action": Value::Null,
            "committed": false,
            "risk": risk.meta(),
        });
        let refuse =
            |r: Rejection, meta: Value| Ok(ZeroToolOutcome::rejected(ZeroVerb::Ask, r, meta));
        let assets = match CognitiveAssets::from_snapshot(&snapshot) {
            Ok(a) => a,
            Err(r) => return refuse(r, meta),
        };
        meta["assets"] = assets.summary();
        let req = match parse_decision(block, feasible) {
            Ok(r) => r,
            Err(r) => return refuse(r, meta),
        };
        let runtime = Arc::clone(&self.runtime);
        let snap = Arc::clone(&snapshot);
        let decided = self
            .spawn_cpu(move || runtime.decide(&snap, &assets, &req))?
            .await
            .map_err(|e| ServiceError::Core(format!("cognitive runtime task failed: {e}")))?;
        let decision = match decided {
            Ok(d) => d,
            Err(r) => return refuse(r, meta),
        };
        meta["cognitive"] = decision.trace();
        let (cert, probs) = match decision.chosen {
            Ok(c) => c,
            Err(r) => return refuse(r, meta),
        };
        // Commit point (§5.3 step 5): an action certified on a generation
        // that was replaced mid-request is not committed on the new one.
        if let Err(r) = binding.ensure_current(&*self.mounts) {
            meta["stale_certificate"] = json!(cert);
            return refuse(
                Rejection::reject(
                    r,
                    "commit",
                    format!(
                        "certified on mount v{} but the mount changed before commit",
                        cert.mount_version
                    ),
                ),
                meta,
            );
        }

        let p32: Vec<f32> = probs.iter().map(|&p| p as f32).collect();
        let entropy = NormalizedEntropy::from_probabilities(&p32);
        let chosen_id = action_id(&cert.action);
        meta["decision_rule"] = json!("min_certified_geodesic_energy_to_goal");
        meta["candidates"] = json!(feasible
            .iter()
            .zip(&probs)
            .map(|(c, p)| json!({"name": c, "probability": p}))
            .collect::<Vec<_>>());
        meta["chosen_action"] = json!(cert.action);
        meta["action_id"] = json!(chosen_id.0);
        meta["entropy"] = json!(entropy.0);
        meta["certified_action"] = json!(cert);
        let verdict = self
            .gate
            .evaluate(chosen_id, entropy, Some(self.graph.as_ref()), None)
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        let text = format!(
            "Certified action: {} (mount v{}, certificate {})",
            cert.action, cert.mount_version, cert.certificate
        );
        let mut out = Self::gated_outcome(ZeroVerb::Ask, verdict.tier, risk, meta, text);
        // Certified by geometry, but held back by the policy tier or the
        // request risk: nothing is committed until someone confirms.
        out.meta["committed"] = json!(!out.is_error);
        Ok(out)
    }

    /// `ask` in a latent planner mode: one of the planner crate's engines
    /// picks among the feasible candidates from a numeric `latent` state on
    /// the untrained latent prior. The request text still goes through the
    /// risk classifier, and the pick through the PolicyGate.
    async fn handle_latent_plan(
        &self,
        arguments: &Value,
        mode: PlannerMode,
        safety: &RequestSafety,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let dynamics = worldsim::parse_dynamics_spec(arguments);
        let mut meta = json!({
            "engine": ENGINE_LATENT_PLANNER,
            "planner_mode": mode.name(),
            "semantic_scoring": false,
            "provenance": dynamics.as_ref().ok().map(|s| s.kind.provenance()),
            "dynamics": dynamics.as_ref().ok().map(|s| s.kind.name()),
            "damping": dynamics.as_ref().ok().and_then(|s| s.damping),
            "trained": false,
            "calibrated": false,
        });
        let refuse =
            |r: Rejection, meta: Value| Ok(ZeroToolOutcome::rejected(ZeroVerb::Ask, r, meta));
        let dynamics = match dynamics {
            Ok(k) => k,
            Err(r) => return refuse(r, meta),
        };
        let want_trajectory = arguments.get("return_trajectory") == Some(&Value::Bool(true));
        if arguments.get("horizon").is_some() && !want_trajectory {
            return refuse(
                Rejection::invalid(
                    "request",
                    "`horizon` is read only with `return_trajectory` in a latent planner mode",
                ),
                meta,
            );
        }
        let parsed = worldsim::parse_latent(arguments.get("latent"), "latent").and_then(|latent| {
            let names = worldsim::parse_names(
                arguments.get("candidates"),
                "candidates",
                worldsim::MAX_CANDIDATES,
                true,
            )?;
            let horizon =
                worldsim::parse_horizon(arguments.get("horizon"), worldsim::DEFAULT_HORIZON)?;
            Ok((latent, names, horizon))
        });
        let (latent, candidates, horizon) = match parsed {
            Ok(p) => p,
            Err(r) => return refuse(r, meta),
        };

        let context = first_text(arguments, &["context"]);
        let risk_text = if context.is_empty() {
            candidates.join("\n")
        } else {
            context.to_string()
        };
        let risk = self.assess_risk(&risk_text).await;
        if let Some(stop) = Self::risk_hard_stop(ZeroVerb::Ask, &risk) {
            return Ok(stop);
        }

        let feasible_mask = self.request_feasibility(&candidates, safety);
        let feasible: Vec<String> = candidates
            .iter()
            .zip(&feasible_mask)
            .filter(|(_, ok)| **ok)
            .map(|(c, _)| c.clone())
            .collect();
        let infeasible: Vec<&String> = candidates
            .iter()
            .zip(&feasible_mask)
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| c)
            .collect();
        if feasible.is_empty() {
            return Ok(Self::gated_outcome(
                ZeroVerb::Ask,
                PolicyTier::Tier3HardStop,
                &risk,
                json!({"engine": "formal_filter", "formally_infeasible": infeasible}),
                String::new(),
            ));
        }
        meta["formally_infeasible"] = json!(infeasible);

        let gate = Arc::clone(&self.gate);
        let planned = {
            let (latent, feasible) = (latent.clone(), feasible.clone());
            self.spawn_cpu(move || {
                worldsim::plan_latent(mode, dynamics, &gate, &latent, &feasible)
            })?
            .await
            .map_err(|e| ServiceError::Core(format!("latent planner task failed: {e}")))?
        };
        let choice = match planned {
            Ok(c) => c,
            Err(r) => return refuse(r, meta),
        };
        let chosen = feasible[choice.index].clone();
        let chosen_id = action_id(&chosen);
        meta["planner"] = json!(choice.engine);
        meta["chosen_action"] = json!(chosen);
        meta["action_id"] = json!(chosen_id.0);
        meta["entropy"] = json!(choice.entropy.0);
        meta["candidates"] = json!(feasible);
        if want_trajectory {
            let gate = Arc::clone(&self.gate);
            let (latent, chosen, feasible) = (latent.clone(), chosen.clone(), feasible.clone());
            let trajectory = self
                .spawn_cpu(move || {
                    worldsim::trajectory_for_choice(
                        dynamics, &gate, &latent, &chosen, &feasible, horizon,
                    )
                })?
                .await
                .map_err(|e| ServiceError::Core(format!("trajectory task failed: {e}")))?;
            match trajectory {
                Ok(t) => meta["trajectory"] = t,
                Err(r) => return refuse(r, meta),
            }
        }
        let verdict = self
            .gate
            .evaluate(chosen_id, choice.entropy, Some(self.graph.as_ref()), None)
            .map_err(|e| ServiceError::Core(e.to_string()))?;
        Ok(Self::gated_outcome(
            ZeroVerb::Ask,
            verdict.tier,
            &risk,
            meta,
            format!("Selected action: {chosen}"),
        ))
    }

    /// `simulate` / `what_if`: numeric state and action names only. No
    /// request text is classified, and `_meta.risk` says so.
    async fn run_worldmodel(
        &self,
        verb: ZeroVerb,
        arguments: &Value,
        op: fn(&PolicyGate, &Value) -> worldsim::WorldResult,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let gate = Arc::clone(&self.gate);
        let args = arguments.clone();
        let done = self
            .spawn_cpu(move || op(&gate, &args))?
            .await
            .map_err(|e| ServiceError::Core(format!("world model task failed: {e}")))?;
        let risk = json!({
            "assessed": false,
            "reason": "numeric state and action names only; no request text was risk-classified",
        });
        let dynamics = worldsim::parse_dynamics(arguments.get("dynamics")).ok();
        Ok(Self::worldmodel_outcome(verb, done, risk, dynamics))
    }

    /// `audit`: the audited action name goes through the semantic risk check
    /// first. An unassessed name stays unassessed in the verdict, it is never
    /// waved through.
    async fn handle_audit(&self, arguments: &Value) -> Result<ZeroToolOutcome, ServiceError> {
        let risk = self
            .assess_risk(first_text(arguments, &["target_action"]))
            .await;
        let gate = Arc::clone(&self.gate);
        let args = arguments.clone();
        let (tier, risk_meta) = (risk.tier(), risk.meta());
        let done = self
            .spawn_cpu(move || worldsim::audit(&gate, &args, tier, risk_meta))?
            .await
            .map_err(|e| ServiceError::Core(format!("world model task failed: {e}")))?;
        let dynamics = worldsim::parse_dynamics(arguments.get("dynamics")).ok();
        Ok(Self::worldmodel_outcome(
            ZeroVerb::Audit,
            done,
            risk.meta(),
            dynamics,
        ))
    }

    /// A completed world-model verb is a success even when its verdict is a
    /// rejection: the audit ran. A refused request is a typed rejection; its
    /// provenance is null when the `dynamics` field itself was refused.
    fn worldmodel_outcome(
        verb: ZeroVerb,
        done: worldsim::WorldResult,
        risk: Value,
        dynamics: Option<worldsim::DynamicsKind>,
    ) -> ZeroToolOutcome {
        match done {
            Ok((mut meta, summary)) => {
                if meta.get("risk").is_none() {
                    meta["risk"] = risk;
                }
                ZeroToolOutcome {
                    verb,
                    is_error: false,
                    content: text_block(summary),
                    meta,
                    rejection: None,
                }
            }
            Err(rejection) => ZeroToolOutcome::rejected(
                verb,
                rejection,
                json!({
                    "engine": worldsim::ENGINE_WORLDMODEL,
                    "provenance": dynamics.map(|k| k.provenance()),
                    "dynamics": dynamics.map(|k| k.name()),
                }),
            ),
        }
    }

    /// Verb 2: `route` (tool ranking by semantic relevance to the intent)
    async fn handle_route(&self, arguments: &Value) -> Result<ZeroToolOutcome, ServiceError> {
        let tools: Vec<Value> = arguments
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let top_k = arguments
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(3)
            .max(1) as usize;
        let intent = first_text(arguments, &["intent", "task_goal", "context", "query"]);

        let risk = self.assess_risk(intent).await;
        if let Some(stop) = Self::risk_hard_stop(ZeroVerb::Route, &risk) {
            return Ok(stop);
        }

        let named: Vec<(String, Value)> = tools
            .into_iter()
            .filter_map(|t| tool_name(&t).map(|n| (n, t)))
            .collect();
        let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
        let verdicts = self.candidate_verdicts(&names, NormalizedEntropy::ZERO)?;
        let feasible_mask = verdicts.iter().map(|v| v.tier != PolicyTier::Tier3HardStop);
        let (feasible, infeasible): (Vec<_>, Vec<_>) = named
            .into_iter()
            .zip(feasible_mask)
            .partition(|(_, ok)| *ok);
        let feasible: Vec<(String, Value)> = feasible.into_iter().map(|(t, _)| t).collect();
        let infeasible: Vec<String> = infeasible.into_iter().map(|((n, _), _)| n).collect();
        let feasible_names: Vec<String> = feasible.iter().map(|(n, _)| n.clone()).collect();
        let feasible_tools: Vec<Value> = feasible.iter().map(|(_, t)| t.clone()).collect();

        let state = arguments.get("state");
        let ranked = if feasible.len() < 2 {
            Err(Unscored::skipped(
                "fewer than two feasible tools; nothing to rank",
            ))
        } else if intent.is_empty() {
            Err(Unscored::skipped("no intent text to rank tools against"))
        } else {
            match self.bridge() {
                Some(bridge) => bridge
                    .semantic_route(intent, &feasible_tools, &feasible_names, top_k, state)
                    .await
                    .map_err(Unscored::Bridge),
                None => Err(Unscored::skipped(BRIDGE_DISABLED)),
            }
        };

        let (order, degraded, mut meta): (Vec<String>, Option<String>, Value) = match ranked {
            Ok(resp) => (
                resp.ranked.iter().map(|r| r.name.clone()).collect(),
                None,
                json!({
                    "engine": ENGINE_SEMANTIC,
                    "semantic_scoring": true,
                    "ranking": "semantic_relevance",
                    "bridge_endpoint": self.bridge().map(|b| b.endpoint()),
                    "scorer": resp.scorer,
                    "ranked": resp.ranked,
                    "entropy": resp.entropy,
                    "bridge_timing_ms": resp.timing_ms,
                }),
            ),
            Err(why) => {
                tracing::warn!("semantic route unavailable: {}", why.describe());
                let mut meta = json!({
                    "engine": ENGINE_FALLBACK,
                    "semantic_scoring": false,
                    "ranking": "unavailable",
                    "bridge_endpoint": self.bridge().map(|b| b.endpoint()),
                });
                why.record(&mut meta);
                (Vec::new(), Some(why.describe()), meta)
            }
        };

        let selected: Vec<Value> = order
            .iter()
            .take(top_k)
            .filter_map(|n| {
                feasible
                    .iter()
                    .find(|(name, _)| name == n)
                    .map(|(_, t)| t.clone())
            })
            .collect();
        let selected_names: Vec<String> = selected.iter().filter_map(tool_name).collect();
        let entropy = NormalizedEntropy(meta["entropy"].as_f64().unwrap_or(f64::NAN) as f32);
        let selected_verdicts = self.candidate_verdicts(&selected_names, entropy)?;
        let selected_tier = selected_verdicts
            .iter()
            .map(|v| v.tier)
            .max()
            .unwrap_or_else(|| {
                verdicts
                    .iter()
                    .map(|v| v.tier)
                    .max()
                    .unwrap_or(PolicyTier::Tier3HardStop)
            });
        meta["candidate_verdicts"] = json!(verdicts);
        meta["selected_verdicts"] = json!(selected_verdicts);
        meta["selected_tools"] = json!(selected);
        meta["count"] = json!(selected.len());
        meta["formally_infeasible"] = json!(infeasible);

        let summary = if degraded.is_some() {
            "Semantic ranking unavailable; selected no tools".to_owned()
        } else {
            format!(
                "Ranked {} tools, kept top {} ({})",
                feasible_names.len(),
                selected.len(),
                meta["ranking"].as_str().unwrap_or("")
            )
        };
        let mut outcome = Self::gated_outcome(ZeroVerb::Route, selected_tier, &risk, meta, summary);
        if let Some(reason) = degraded {
            // No semantic ranking is available: report an error and expose no
            // selected tool, whatever the risk tier.
            outcome.is_error = true;
            outcome.meta["degraded"] = json!(true);
            if outcome.meta.get("error_code").is_none() {
                outcome.meta["error_code"] = json!(-32003);
            }
            if outcome.meta.get("gate_status") == Some(&json!("proceed")) {
                outcome.meta["gate_status"] = json!("unavailable");
            }
            let note = format!(
                "SemanticRankingUnavailable: no tools selected or ranked by relevance \
                 ({reason})"
            );
            if outcome.meta["tier"] == tier_name(PolicyTier::Tier0Proceed) {
                outcome.content = text_block(note);
            } else {
                outcome.content.extend(text_block(note));
            }
        }
        Ok(outcome)
    }

    /// Verb 3: `imagine` (multi-step semantic lookahead, formally filtered)
    async fn handle_imagine(
        &self,
        arguments: &Value,
        safety: &RequestSafety,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let horizon = arguments
            .get("horizon")
            .and_then(|v| v.as_u64())
            .map(|h| h as usize)
            .unwrap_or(DEFAULT_IMAGINE_HORIZON)
            .clamp(1, MAX_HORIZON);
        let simulations = arguments
            .get("simulations")
            .and_then(|v| v.as_u64())
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_IMAGINE_SIMULATIONS)
            .clamp(1, MAX_SIMULATIONS);
        let parsed = string_list(arguments.get("candidate_actions"), "candidate_actions").and_then(
            |candidates| {
                if candidates.is_empty() {
                    string_list(arguments.get("candidates"), "candidates")
                } else {
                    Ok(candidates)
                }
            },
        );
        let candidates = match parsed {
            Ok(candidates) => candidates,
            Err(rejection) => {
                return Ok(ZeroToolOutcome::rejected(
                    ZeroVerb::Imagine,
                    rejection,
                    json!({}),
                ))
            }
        };
        let scenario = first_text(arguments, &["scenario", "context", "simulation", "state"]);
        let state = arguments.get("state").filter(|s| !s.is_string());

        let feasible = self.request_feasibility(&candidates, safety);
        let infeasible: Vec<&String> = candidates
            .iter()
            .zip(&feasible)
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| c)
            .collect();

        let unavailable = |why: Unscored, engine: &str, risk: Option<&RiskCheck>| {
            let mut meta = json!({
                "engine": engine,
                "semantic_scoring": false,
                "horizon": horizon,
                "sequence_likelihood": Value::Null,
                "formal_checked": safety.enforce_cpsat && !candidates.is_empty(),
                "formally_infeasible": infeasible,
                "bridge_endpoint": self.bridge().map(|b| b.endpoint()),
            });
            why.record(&mut meta);
            let mut content = text_block(format!("LookaheadUnavailable: {}", why.describe()));
            if let Some(risk) = risk {
                meta["risk"] = risk.meta();
                if risk.tier() != PolicyTier::Tier0Proceed {
                    // Tier3 exited earlier via `risk_hard_stop`; what is left
                    // needs confirmation and must say so (428, not 422).
                    meta["requires_confirmation"] = json!(true);
                    meta["gate_status"] = json!(GATE_STATUS_REQUIRES_CONFIRMATION);
                    meta["error_code"] = json!(-32002);
                    content.extend(text_block(risk.message()));
                }
            }
            ZeroToolOutcome {
                verb: ZeroVerb::Imagine,
                is_error: true,
                content,
                meta,
                rejection: None,
            }
        };

        if candidates.len() < 2 {
            return Ok(unavailable(
                Unscored::skipped("imagine needs at least two distinct candidate_actions"),
                "none_invalid_input",
                None,
            ));
        }
        if scenario.is_empty() {
            return Ok(unavailable(
                Unscored::skipped("imagine needs a scenario or context text"),
                "none_invalid_input",
                None,
            ));
        }
        if !feasible.iter().any(|&ok| ok) {
            return Ok(unavailable(
                Unscored::skipped("every candidate action is formally infeasible"),
                "formal_filter",
                None,
            ));
        }

        let risk = self.assess_risk(scenario).await;
        if let Some(stop) = Self::risk_hard_stop(ZeroVerb::Imagine, &risk) {
            return Ok(stop);
        }
        let Some(bridge) = self.bridge() else {
            return Ok(unavailable(
                Unscored::skipped(BRIDGE_DISABLED),
                ENGINE_FALLBACK,
                Some(&risk),
            ));
        };

        let oracle = BridgeOracle {
            client: bridge,
            scenario,
            state,
            candidates: &candidates,
        };
        let root_noise = RootNoise {
            seed: arguments
                .get("seed")
                .and_then(|v| v.as_u64())
                .unwrap_or(RootNoise::default().seed),
            ..RootNoise::default()
        };
        let cfg = LookaheadConfig {
            horizon,
            simulations,
            c_puct: DEFAULT_C_PUCT,
            root_noise,
        };
        let result = match run_lookahead(&oracle, &feasible, cfg).await {
            Ok(r) => r,
            Err(err) => {
                tracing::warn!("imagine has no semantic oracle: {err}");
                return Ok(unavailable(
                    Unscored::Bridge(err),
                    ENGINE_FALLBACK,
                    Some(&risk),
                ));
            }
        };

        let mut plan = Vec::with_capacity(result.principal_variation.len());
        // The request risk is a floor for the whole plan.
        let mut worst = risk.tier();
        for step in &result.principal_variation {
            let name = &candidates[step.action];
            let verdict = self
                .gate
                .evaluate_basic(action_id(name), step.entropy)
                .map_err(|e| ServiceError::Core(e.to_string()))?;
            if verdict.tier > worst {
                worst = verdict.tier;
            }
            plan.push(json!({
                "action": name,
                "probability": step.probability,
                "entropy": step.entropy.0,
                "tier": tier_name(verdict.tier),
            }));
        }
        let root: Vec<Value> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "action": c,
                    "feasible": feasible[i],
                    "visits": result.root_visits[i],
                    "prior_with_noise": result.root_noisy_priors[i],
                    "sequence_likelihood": result.root_values[i],
                })
            })
            .collect();
        let best = &candidates[result.best_action];
        let meta = json!({
            "engine": ENGINE_SEMANTIC,
            "semantic_scoring": true,
            "planner": "puct_mcts",
            // A language-model likelihood of the action sequence given the
            // scenario text. Not an environment reward, not observed feedback.
            "value_model": "geometric_mean_step_likelihood",
            "value_is_environment_reward": false,
            "c_puct": DEFAULT_C_PUCT,
            "root_noise": {
                "kind": "dirichlet",
                "alpha": root_noise.alpha,
                "epsilon": root_noise.epsilon,
                "seed": root_noise.seed,
            },
            "bridge_endpoint": bridge.endpoint(),
            "horizon": horizon,
            "simulations": result.simulations,
            "oracle_calls": result.oracle_calls,
            "best_action": best,
            "sequence_likelihood": result.sequence_likelihood,
            "confidence": result.confidence,
            "plan": plan,
            "root": root,
            "formal_checked": safety.enforce_cpsat,
            "formally_infeasible": infeasible,
            "plan_tier": tier_name(worst),
        });
        let summary = format!(
            "Lookahead over {} steps ({} simulations): best first action '{}' \
                 (visit share {:.2}, sequence likelihood {:.3}; a language-model likelihood, \
                 not an environment reward)",
            horizon, result.simulations, best, result.confidence, result.sequence_likelihood
        );
        // `worst` already folds the request risk into the plan tier. The same
        // mapping as ask and route decides the outcome: a Tier1/Tier2 plan is
        // a ConfirmationRequired error (428) even when the request text was
        // Tier0, and a Tier3 plan is a hard stop. The plan stays in `_meta`
        // for the person who confirms; the summary follows the verdict text.
        let mut out = Self::gated_outcome(ZeroVerb::Imagine, worst, &risk, meta, summary.clone());
        if out.is_error {
            out.content.extend(text_block(summary));
        }
        Ok(out)
    }

    /// Verb 4: `stream`: one numeric event window through the cognitive
    /// runtime (tangent map, parallel scan, geometry gate) on the captured
    /// snapshot. Text observations are refused: there is no encoder that puts
    /// text into the manifold, so nothing would be computed.
    async fn handle_stream(
        &self,
        arguments: &Value,
        binding: &RequestBinding,
    ) -> Result<ZeroToolOutcome, ServiceError> {
        let mut meta = json!({"engine": ENGINE_COGNITIVE});
        let refuse =
            |r: Rejection, meta: Value| Ok(ZeroToolOutcome::rejected(ZeroVerb::Stream, r, meta));
        let Some(block) = arguments.get("cognitive") else {
            return refuse(
                Rejection::invalid(
                    "stream",
                    "stream needs a numeric `cognitive` window {state, window_start_ns, events}; \
                     text observations have no encoder into the manifold and are not ingested",
                ),
                meta,
            );
        };
        let snapshot = Arc::clone(binding.snapshot());
        let assets = match CognitiveAssets::from_snapshot(&snapshot) {
            Ok(a) => a,
            Err(r) => return refuse(r, meta),
        };
        meta["assets"] = assets.summary();
        let (req, events) = match parse_stream(block) {
            Ok(p) => p,
            Err(r) => return refuse(r, meta),
        };
        let runtime = Arc::clone(&self.runtime);
        let snap = Arc::clone(&snapshot);
        let ran = self
            .spawn_cpu(move || runtime.simulate(&snap, &assets, &req, &events))?
            .await;
        let traj = match ran {
            Ok(Ok(t)) => t,
            Ok(Err(r)) => return refuse(r, meta),
            Err(e) => {
                return refuse(
                    Rejection::reject(
                        Reject::BackendUnavailable,
                        "stream",
                        format!("cognitive runtime task failed: {e}"),
                    ),
                    meta,
                )
            }
        };
        let text = format!(
            "Scanned {} steps on mount v{} ({}); gate: {}",
            traj.scan["steps"],
            snapshot.version().0,
            traj.scan["backend"].as_str().unwrap_or(""),
            traj.gate.status
        );
        meta["trajectory"] = traj.trace();
        Ok(ZeroToolOutcome {
            verb: ZeroVerb::Stream,
            is_error: false,
            content: text_block(text),
            meta,
            rejection: None,
        })
    }

    /// Verb 7: `entail`: the asymmetric Busemann test `passage ⊃ question`
    /// on the preset product geometry the captured snapshot seals. Input is two
    /// points of the sealed width (64, 128 or 256 coordinates, per preset); a
    /// different width is `FiberMismatch`. There is no text encoder, so text
    /// is refused. A
    /// `false` verdict is a valid answer; a refusal means no verdict exists.
    /// With `passage_events` and `question_events` the verdict runs on the
    /// fiber cross-difference SSM path (scheme 2, see `cognitive`).
    fn handle_entail(
        runtime: &CognitiveRuntime,
        arguments: &Value,
        snapshot: Arc<MountSnapshot>,
    ) -> ZeroToolOutcome {
        let mut meta = json!({"engine": ENGINE_COGNITIVE, "semantic_scoring": false});
        let refuse =
            |r: Rejection, meta: Value| ZeroToolOutcome::rejected(ZeroVerb::Entail, r, meta);
        let Some(block) = arguments.get("entailment") else {
            return refuse(
                Rejection::invalid(
                    "entail",
                    "entail needs numeric `entailment` {passage, question}, each with the \
                     width of the mounted topology preset (compact_64d: 64, balanced_128d / \
                     boolq_128d: 128, extended_256d: 256); text has no encoder into the \
                     manifold and is not embedded",
                ),
                meta,
            );
        };
        let assets = match CognitiveAssets::from_snapshot(&snapshot) {
            Ok(a) => a,
            Err(r) => return refuse(r, meta),
        };
        meta["assets"] = assets.summary();
        let req = match parse_entailment(block) {
            Ok(r) => r,
            Err(r) => return refuse(r, meta),
        };
        let verdict = match runtime.evaluate_entailment(&snapshot, &assets, &req) {
            Ok(v) => v,
            Err(r) => return refuse(r, meta),
        };
        let text = format!(
            "Entailed: {} (mode {}, margin {:.6}, mount v{}); not calibrated",
            verdict.is_entailed(),
            verdict.mode(),
            verdict.score.confidence,
            snapshot.version().0
        );
        meta["entailment"] = verdict.trace();
        ZeroToolOutcome {
            verb: ZeroVerb::Entail,
            is_error: false,
            content: text_block(text),
            meta,
            rejection: None,
        }
    }

    /// Verb 8: `causal_fold`: fold a chain of relation ids through the
    /// caller-supplied learned relation semiring (Spec 24 §8.6.2), closing
    /// under S1 (`left`), tiered dispatch or S3 (`chart`, the default,
    /// a CYK-style chart over every bracketing). The axiom table is exactly
    /// what the caller supplies in `axioms`; a missing table means every
    /// composition is `∅`, so a chain of two or more edges always refuses.
    /// A conflict key (two or more results for one `(r1, r2, gender)`)
    /// makes `left` refuse. `chart` carries every
    /// candidate forward instead, so it concludes only when exactly one
    /// relation survives at the root and refuses when two or more do; it
    /// never picks among survivors. `meta.causal_fold.conflict_keys` counts
    /// the conflict keys in the table so a caller can see one was present.
    /// Verb 9: `pipeline`. Runs the planner's `ProductionPipeline` on this engine's
    /// world model and gate. Every refusal is a typed outcome, never a default answer.
    fn handle_pipeline(
        world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
        gate: Arc<PolicyGate>,
        graph: Arc<LodGraph>,
        arguments: &Value,
    ) -> ZeroToolOutcome {
        let meta = json!({"engine": "production_pipeline"});
        let Some(block) = arguments.get("pipeline") else {
            let rej = Rejection::invalid(
                "pipeline",
                "pipeline needs {op: simulate | what_if | audit_action | decide, state: [1024 \
                 numbers], ...}",
            );
            return ZeroToolOutcome::rejected(ZeroVerb::Pipeline, rej, meta);
        };
        match execute_pipeline(world_model, gate, graph, block) {
            Ok((summary, result)) => {
                let mut meta = meta;
                meta["pipeline"] = result;
                if let Some(decision) = meta.pointer("/pipeline/decision") {
                    let tier: PolicyTier =
                        match serde_json::from_value(decision["gate_tier"].clone()) {
                            Ok(tier) => tier,
                            Err(e) => {
                                return ZeroToolOutcome::rejected(
                                    ZeroVerb::Pipeline,
                                    Rejection::invalid(
                                        "pipeline",
                                        format!("invalid decision gate tier: {e}"),
                                    ),
                                    meta,
                                );
                            }
                        };
                    return Self::gated_outcome(
                        ZeroVerb::Pipeline,
                        tier,
                        &RiskCheck::NotApplicable,
                        meta,
                        summary,
                    );
                }
                ZeroToolOutcome {
                    verb: ZeroVerb::Pipeline,
                    is_error: false,
                    content: text_block(summary),
                    meta,
                    rejection: None,
                }
            }
            Err(rej) => {
                tracing::warn!(code = %rej.code, "pipeline refused: {}", rej.detail);
                ZeroToolOutcome::rejected(ZeroVerb::Pipeline, rej, meta)
            }
        }
    }

    fn validate_causal_fold(arguments: &Value) -> Result<&Value, Rejection> {
        let Some(block) = arguments.get("causal_fold") else {
            return Err(Rejection::invalid(
                "causal_fold",
                "causal_fold needs {edges: [relation ids], genders: [node genders, \
                     edges.len() + 1 of them], axioms?: [{r1, r2, gender, result: [relation \
                     ids]}] (missing means an empty table), strategy?: \"chart\" (default), \
                     \"tiered\" or \"left\"}",
            ));
        };
        // serde reads `"strategy": null` as absent; refuse it instead of
        // quietly running the default chart strategy.
        if block.get("strategy").is_some_and(Value::is_null) {
            return Err(Rejection::invalid(
                "causal_fold",
                "causal_fold.strategy must not be null",
            ));
        }
        if block.get("sets").is_some_and(Value::is_null) {
            return Err(Rejection::invalid(
                "causal_fold",
                "causal_fold.sets must not be null",
            ));
        }
        // Inspect lengths in the borrowed JSON before cloning/deserializing or
        // allocating any relation sets. HTTP JSON parsing precedes this gate.
        let invalid = |detail: &str| Err(Rejection::invalid("causal_fold", detail));
        if let Some(sets) = block.get("sets").and_then(Value::as_array) {
            if sets.len() > MAX_CAUSAL_FOLD_EDGES {
                return invalid("causal_fold.sets must contain 1 to 64 sets");
            }
            let mut total = 0;
            for ids in sets.iter().filter_map(Value::as_array) {
                if ids.len() > MAX_CAUSAL_FOLD_SET_RELATIONS {
                    return invalid("causal_fold.sets exceeds single-set relation cap (64)");
                }
                total += ids.len();
                if total > MAX_CAUSAL_FOLD_TOTAL_SET_RELATIONS {
                    return invalid("causal_fold.sets exceeds total relation cap (256)");
                }
            }
        }
        for (field, cap) in [
            ("weights", MAX_CAUSAL_FOLD_AXIOMS),
            ("axioms", MAX_CAUSAL_FOLD_AXIOMS),
            ("edges", MAX_CAUSAL_FOLD_EDGES),
            ("genders", MAX_CAUSAL_FOLD_EDGES + 1),
        ] {
            if block
                .get(field)
                .and_then(Value::as_array)
                .is_some_and(|a| a.len() > cap)
            {
                return invalid(&format!(
                    "causal_fold.{field} has more than the cap of {cap} entries"
                ));
            }
        }
        if let Some(axioms) = block.get("axioms").and_then(Value::as_array) {
            if axioms.iter().any(|a| {
                a.get("result")
                    .and_then(Value::as_array)
                    .is_some_and(|r| r.len() > MAX_CAUSAL_FOLD_RESULT_RELATIONS)
            }) {
                return invalid("causal_fold.axioms result has more than 64 entries; no more than 64 distinct relation ids allowed");
            }
        }
        if block
            .get("margin_threshold")
            .is_some_and(|v| v.as_f64().is_none_or(|n| !n.is_finite() || n < 0.0))
        {
            return invalid("causal_fold.margin_threshold must be finite and >= 0");
        }
        Ok(block)
    }

    fn handle_causal_fold(block: &Value) -> ZeroToolOutcome {
        let meta = json!({"engine": "relation_semiring_fold"});
        let refuse =
            |r: Rejection, meta: Value| ZeroToolOutcome::rejected(ZeroVerb::CausalFold, r, meta);
        let invalid =
            |detail: &str| refuse(Rejection::invalid("causal_fold", detail), meta.clone());
        let wire: CausalFoldWire = match serde_json::from_value(block.clone()) {
            Ok(w) => w,
            Err(e) => {
                return refuse(
                    Rejection::invalid("causal_fold", format!("invalid causal_fold request: {e}")),
                    meta,
                )
            }
        };
        if wire.strategy.as_deref() == Some("bidirectional") {
            return invalid("strategy 'bidirectional' has been deprecated and retired; use 'chart', 'tiered', or 'weighted_logprob'");
        }
        // Set chains have one uniform composition gender and always use the chart.
        let set_input = if let Some(sets) = &wire.sets {
            let invalid =
                |detail: String| refuse(Rejection::invalid("causal_fold", detail), meta.clone());
            if !wire.edges.is_empty() {
                return invalid(
                    "cannot provide both causal_fold.edges and causal_fold.sets".into(),
                );
            }
            let Some(gender) = wire.gender else {
                return invalid("causal_fold.gender is required when sets is provided".into());
            };
            if sets.is_empty() || sets.len() > MAX_CAUSAL_FOLD_EDGES {
                return invalid("causal_fold.sets must contain 1 to 64 sets".into());
            }
            if !wire.genders.is_empty() {
                return invalid(
                    "causal_fold.genders cannot be provided with sets; use gender".into(),
                );
            }
            if wire.strategy.as_deref().is_some_and(|s| s != "chart") {
                return invalid("causal_fold.sets requires chart strategy".into());
            }
            let mut chain = Vec::with_capacity(sets.len());
            for (i, ids) in sets.iter().enumerate() {
                if ids.is_empty() {
                    return invalid(format!("causal_fold.sets[{i}] must not be empty"));
                }
                let unique: std::collections::HashSet<_> = ids.iter().collect();
                if unique.len() != ids.len() {
                    return invalid(format!("causal_fold.sets[{i}] repeats a relation id"));
                }
                chain.push(ResultSet::from_ids(ids.iter().copied()));
            }
            Some((chain, gender))
        } else {
            if wire.edges.is_empty() {
                return refuse(
                    Rejection::invalid("causal_fold", "causal_fold.edges must not be empty"),
                    meta,
                );
            }
            if wire.edges.len() > MAX_CAUSAL_FOLD_EDGES {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!(
                            "causal_fold.edges has {} entries, more than the cap of {} \
                             (chart_fold_chain is cubic time and this endpoint is network-exposed)",
                            wire.edges.len(),
                            MAX_CAUSAL_FOLD_EDGES
                        ),
                    ),
                    meta,
                );
            }
            if wire.genders.len() != wire.edges.len() + 1 {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!(
                            "causal_fold.genders must have edges.len() + 1 = {} entries, got {}",
                            wire.edges.len() + 1,
                            wire.genders.len()
                        ),
                    ),
                    meta,
                );
            }
            None
        };
        let strategy = match wire.strategy.as_deref() {
            None if wire.weights.is_some() => "tiered",
            None | Some("chart") => "chart",
            Some("tiered") => "tiered",
            Some("left") => "left",
            Some("weighted_tropical") => "weighted_tropical",
            Some("weighted_logprob") => "weighted_logprob",
            Some(other) => {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!(
                            "unknown causal_fold.strategy {other:?}; expected chart, \
                             tiered, left, weighted_tropical or weighted_logprob"
                        ),
                    ),
                    meta,
                )
            }
        };

        let weighted = matches!(strategy, "weighted_tropical" | "weighted_logprob");
        if weighted && wire.weights.as_ref().is_none_or(Vec::is_empty) {
            return invalid("causal_fold.weights required for weighted strategies");
        }
        if !weighted
            && strategy != "tiered"
            && (block.get("weights").is_some() || block.get("margin_threshold").is_some())
        {
            return invalid("causal_fold.weights and margin_threshold require a weighted strategy");
        }
        if block.get("weights").is_some() && wire.weights.as_ref().is_none_or(Vec::is_empty) {
            return invalid("causal_fold.weights must be a non-empty array");
        }
        if block.get("semiring").is_some()
            && (strategy != "tiered"
                || !matches!(wire.semiring.as_deref(), Some("logprob" | "tropical")))
        {
            return invalid(
                "causal_fold.semiring requires tiered strategy and must be logprob or tropical",
            );
        }
        if set_input.is_some() && strategy != "chart" {
            return invalid(
                "causal_fold.sets requires chart strategy; tiered uses edges and genders",
            );
        }
        if wire.axioms.len() > MAX_CAUSAL_FOLD_AXIOMS {
            return refuse(
                Rejection::invalid(
                    "causal_fold",
                    format!(
                        "causal_fold.axioms has {} entries, more than the cap of {}",
                        wire.axioms.len(),
                        MAX_CAUSAL_FOLD_AXIOMS
                    ),
                ),
                meta,
            );
        }
        let mut result_relations: BTreeSet<RelId> = BTreeSet::new();
        let mut seen_keys: BTreeSet<(RelId, RelId, Gender)> = BTreeSet::new();
        let mut sem = RelationSemiring::new();
        for (i, ax) in wire.axioms.iter().enumerate() {
            if ax.result.is_empty() {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!("causal_fold.axioms[{i}].result must not be empty"),
                    ),
                    meta,
                );
            }
            let distinct: BTreeSet<RelId> = ax.result.iter().copied().collect();
            if distinct.len() != ax.result.len() {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!("causal_fold.axioms[{i}].result repeats a relation id"),
                    ),
                    meta,
                );
            }
            result_relations.extend(distinct);
            if result_relations.len() > MAX_CAUSAL_FOLD_RESULT_RELATIONS {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!(
                            "causal_fold.axioms results name more than {} distinct relation \
                             ids (the chart cost grows with the square of this number)",
                            MAX_CAUSAL_FOLD_RESULT_RELATIONS
                        ),
                    ),
                    meta,
                );
            }
            let key = (ax.r1, ax.r2, ax.gender);
            if !seen_keys.insert(key) {
                return refuse(
                    Rejection::invalid(
                        "causal_fold",
                        format!(
                            "causal_fold.axioms[{i}] duplicates key (r1={}, r2={}, gender={:?}); \
                             a duplicate axiom key is refused, never overwritten",
                            ax.r1, ax.r2, ax.gender
                        ),
                    ),
                    meta,
                );
            }
            sem.insert_axiom(
                RelationKey::new(ax.r1, ax.r2, ax.gender),
                ResultSet::from_ids(ax.result.iter().copied()),
            );
        }

        let axioms_count = wire.axioms.len();
        let conflict_keys = sem.conflict_key_count();
        let edges_count = wire.sets.as_ref().map_or(wire.edges.len(), Vec::len);
        // Validate counts with integer operations before the discrete band. No energies
        // or logarithms are computed unless dispatch actually reaches Band 1.
        if let Some(weights) = &wire.weights {
            let mut seen = BTreeSet::new();
            for w in weights {
                if w.count == 0 || !seen.insert((w.r1, w.r2, w.gender, w.relation)) {
                    return invalid(
                        "invalid causal_fold.weights: zero count or duplicate relation",
                    );
                }
            }
        }
        let band0 =
            (strategy == "tiered").then(|| sem.chart_fold_chain(&wire.edges, &wire.genders));
        let escalate =
            matches!(&band0, Some(FoldOutcome::Refused { .. })) && wire.weights.is_some();
        if weighted || escalate {
            let mut raw_counts =
                std::collections::BTreeMap::<RelationKey, Vec<(RelId, u64)>>::new();
            for w in wire.weights.as_ref().expect("validated weights") {
                raw_counts
                    .entry(RelationKey::new(w.r1, w.r2, w.gender))
                    .or_default()
                    .push((w.relation, w.count));
            }
            // Plain relative frequencies: no reserved probability mass.
            let pseudo_count = 0.0;
            let axiom_weights = match AxiomWeights::from_counts(raw_counts, pseudo_count) {
                Ok(weights) => weights,
                Err(e) => return invalid(&format!("invalid causal_fold.weights: {e:?}")),
            };
            let threshold = wire.margin_threshold.unwrap_or(0.0);
            let outcome = if strategy == "weighted_tropical"
                || wire.semiring.as_deref() == Some("tropical")
            {
                sem.weighted_chart_fold_chain::<TropicalSemiring>(
                    &axiom_weights,
                    &wire.edges,
                    &wire.genders,
                    threshold,
                )
            } else {
                sem.weighted_chart_fold_chain::<LogProbSemiring>(
                    &axiom_weights,
                    &wire.edges,
                    &wire.genders,
                    threshold,
                )
            };
            let mut meta = meta;
            meta["causal_fold"] = json!({"strategy": strategy, "axioms": axioms_count,
                "conflict_keys": conflict_keys, "edges": edges_count, "pseudo_count": pseudo_count});
            if let Some(FoldOutcome::Refused {
                step_failed,
                reason,
            }) = &band0
            {
                meta["causal_fold"]["dispatched_band"] = json!(1);
                meta["causal_fold"]["band0_refusal"] =
                    json!({"step_failed": step_failed, "reason": reason});
                meta["causal_fold"]["semiring"] =
                    json!(wire.semiring.as_deref().unwrap_or("logprob"));
            }
            match outcome {
                WeightedFoldOutcome::Concluded {
                    predicted,
                    steps,
                    proof_path,
                    energy,
                    margin,
                    confidence,
                    candidates,
                } => {
                    let cf = &mut meta["causal_fold"];
                    cf["predicted"] = json!(predicted);
                    cf["steps"] = json!(steps);
                    cf["proof_path"] = json!(proof_path);
                    cf["energy"] = json!(energy);
                    // JSON has no infinity: encode the single-candidate margin explicitly.
                    cf["margin"] = if margin == f64::INFINITY {
                        json!("infinity")
                    } else {
                        json!(margin)
                    };
                    cf["confidence"] = json!(confidence);
                    cf["candidates"] = json!(candidates);
                    return ZeroToolOutcome { verb: ZeroVerb::CausalFold, is_error: false,
                        content: text_block(format!("Concluded: relation {predicted} in {steps} steps (strategy {strategy})")),
                        meta, rejection: None };
                }
                WeightedFoldOutcome::Refused {
                    step_failed,
                    reason,
                    candidates,
                } => {
                    meta["causal_fold"]["step_failed"] = json!(step_failed);
                    meta["causal_fold"]["reason"] = json!(reason);
                    meta["causal_fold"]["candidates"] = json!(candidates);
                    return refuse(
                        Rejection {
                            code: "CausalFoldRefused".into(),
                            stage: "causal_fold".into(),
                            detail: reason,
                            http_status: 422,
                        },
                        meta,
                    );
                }
            }
        }
        let outcome = if let Some(outcome) = band0 {
            outcome
        } else if let Some((chain_sets, gender)) = set_input {
            sem.chart_fold_sets(&chain_sets, gender)
        } else {
            match strategy {
                "left" => sem.left_fold_chain(&wire.edges, &wire.genders),
                _ => sem.chart_fold_chain(&wire.edges, &wire.genders),
            }
        };

        match outcome {
            FoldOutcome::Concluded {
                predicted,
                steps,
                proof_path,
            } => {
                let mut meta = json!({
                    "engine": "relation_semiring_fold",
                    "causal_fold": {
                        "strategy": strategy,
                        "predicted": predicted,
                        "steps": steps,
                        "proof_path": proof_path,
                        "axioms": axioms_count,
                        "conflict_keys": conflict_keys,
                        "edges": edges_count,
                    },
                });
                if strategy == "tiered" {
                    meta["causal_fold"]["dispatched_band"] = json!(0);
                }
                ZeroToolOutcome {
                    verb: ZeroVerb::CausalFold,
                    is_error: false,
                    content: text_block(format!(
                        "Concluded: relation {predicted} in {steps} steps (strategy {strategy})"
                    )),
                    meta,
                    rejection: None,
                }
            }
            FoldOutcome::Refused {
                step_failed,
                reason,
            } => {
                let mut meta = meta;
                meta["causal_fold"] = json!({
                    "strategy": strategy,
                    "step_failed": step_failed,
                    "axioms": axioms_count,
                    "conflict_keys": conflict_keys,
                    "edges": edges_count,
                });
                let reason = if strategy == "tiered" {
                    meta["causal_fold"]["dispatched_band"] = json!(0);
                    format!("{reason}; provide weights to escalate to Band 1")
                } else {
                    reason
                };
                refuse(
                    Rejection {
                        code: "CausalFoldRefused".to_string(),
                        stage: "causal_fold".to_string(),
                        detail: reason,
                        http_status: 422,
                    },
                    meta,
                )
            }
        }
    }

    /// Verb 5: `grep`: literal, case-sensitive line search over the
    /// request's own `lines`. Not semantic. `paths` (no file backend in a
    /// network service) and `expr` (no boolean evaluator) are refused, never
    /// ignored.
    fn handle_grep(&self, arguments: &Value) -> ZeroToolOutcome {
        let meta = json!({"engine": "literal_line_search", "semantic": false});
        let refuse = |r: Rejection| ZeroToolOutcome::rejected(ZeroVerb::Grep, r, meta.clone());
        let query = match arguments.get("query") {
            Some(Value::String(q)) if !q.is_empty() => q.as_str(),
            _ => {
                return refuse(Rejection::invalid(
                    "grep",
                    "query must be a non-empty string",
                ))
            }
        };
        if arguments.get("expr").is_some() {
            return refuse(Rejection::invalid(
                "grep",
                "boolean `expr` is not implemented; it is refused, not ignored",
            ));
        }
        if arguments.get("paths").is_some() {
            return refuse(Rejection::reject(
                Reject::BackendUnavailable,
                "grep",
                "this service has no file search backend and never reads `paths`; \
                 send the text to search as `lines`",
            ));
        }
        let Some(lines) = arguments.get("lines").and_then(Value::as_array) else {
            return refuse(Rejection::invalid(
                "grep",
                "lines must be an array of strings",
            ));
        };
        let mut hits = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(text) = line.as_str() else {
                return refuse(Rejection::invalid(
                    "grep",
                    format!("lines[{i}] must be a string"),
                ));
            };
            if text.contains(query) {
                hits.push(json!({"line": i + 1, "text": text}));
            }
        }
        let mut meta = meta.clone();
        meta["match_mode"] = json!("literal_substring_case_sensitive");
        meta["query"] = json!(query);
        meta["lines_searched"] = json!(lines.len());
        meta["matches"] = json!(hits.len());
        meta["hits"] = json!(hits);
        ZeroToolOutcome {
            verb: ZeroVerb::Grep,
            is_error: false,
            content: text_block(format!(
                "{} of {} lines contain '{}'",
                hits.len(),
                lines.len(),
                query
            )),
            meta,
            rejection: None,
        }
    }

    /// Verb 6: `compact` (Context compression)
    async fn handle_compact(&self, arguments: &Value) -> Result<ZeroToolOutcome, ServiceError> {
        let text = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let input = text.to_owned();
        let compressed = self
            .spawn_cpu(move || zstd::encode_all(input.as_bytes(), 3))?
            .await
            .map_err(|e| ServiceError::Core(format!("compression task failed: {e}")))?
            .map_err(|e| ServiceError::Core(e.to_string()))?;

        Ok(ZeroToolOutcome {
            verb: ZeroVerb::Compact,
            is_error: false,
            content: vec![ZeroContentBlock {
                block_type: "text".to_string(),
                text: format!(
                    "Compressed {} bytes to {} bytes (ratio: {:.1}x)",
                    text.len(),
                    compressed.len(),
                    (text.len().max(1) as f32) / (compressed.len().max(1) as f32)
                ),
            }],
            meta: json!({
                "original_bytes": text.len(),
                "compressed_bytes": compressed.len()
            }),
            rejection: None,
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::bridge::BridgeConfig;

    /// Engine whose request-risk classifier is a local HTTP endpoint answering
    /// p_dangerous = 0, so the tier below reflects the decision head alone.
    pub(crate) async fn engine_with_benign_risk(
    ) -> (PolymorphicZeroEngine, tokio::task::JoinHandle<()>) {
        use axum::{routing::post, Json, Router};
        let app = Router::new().route(
            "/v1/semantic_risk",
            post(|| async {
                Json(json!({
                    "p_dangerous":0.0, "log_odds":-10.0, "windows":1,
                    "thresholds":{"escalate":0.5,"hard_stop":0.9}, "classifier":{}, "forward_ms":1.0
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let engine = PolymorphicZeroEngine::new().with_bridge(Some(Arc::new(
            SemanticBridgeClient::new(BridgeConfig::new(endpoint)).unwrap(),
        )));
        (engine, server)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::BridgeConfig;
    use gen_zero_gate::{LinearConstraint, RuleId};
    use gen_zero_nanocore::core_type::fixtures::synthetic_core;

    #[test]
    fn request_masks_preserve_operator_gate_and_resolve_overlapping_mutexes() {
        let mut gate = PolicyGate::default();
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(42),
            "operator",
            action_id("operator-blocked"),
        ));
        let engine = offline_engine().with_gate(gate);
        let names: Vec<String> = ["operator-blocked", " Delete ", "a", "b", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for enforce in [true, false] {
            let safety = RequestSafety::parse(&json!({
                "enforce_cpsat": enforce,
                "forbidden_actions": ["DELETE"],
                "constraints": [
                    {"type":"mutually_exclusive", "actions":["a", "b"]},
                    {"type":"mutual_exclusive", "actions":["b", "c"]}
                ]
            }))
            .unwrap();
            assert_eq!(
                engine.request_feasibility(&names, &safety),
                [false, false, true, false, true]
            );
        }
        assert_eq!(
            engine.request_feasibility(&names, &RequestSafety::parse(&json!({})).unwrap()),
            [false, true, true, true, true]
        );
    }

    #[tokio::test]
    async fn lookahead_never_simulates_request_disabled_actions() {
        struct Oracle;
        impl crate::imagine::PriorOracle for Oracle {
            fn priors<'a>(&'a self, history: &'a [usize]) -> crate::imagine::PriorFuture<'a> {
                Box::pin(async move {
                    assert!(history.iter().all(|action| *action == 1));
                    Ok(vec![0.8, 0.1, 0.1])
                })
            }
        }
        let safety = RequestSafety::parse(&json!({
            "forbidden_actions":["delete"],
            "constraints":[{"type":"mutually_exclusive", "actions":["backup", "restart"]}]
        }))
        .unwrap();
        let names = vec!["delete".into(), "backup".into(), "restart".into()];
        let mask = offline_engine().request_feasibility(&names, &safety);
        let result = run_lookahead(
            &Oracle,
            &mask,
            LookaheadConfig {
                horizon: 4,
                simulations: 32,
                c_puct: DEFAULT_C_PUCT,
                root_noise: RootNoise::default(),
            },
        )
        .await
        .unwrap();
        assert_eq!(result.best_action, 1);
        assert!(result
            .principal_variation
            .iter()
            .all(|step| step.action == 1));
        assert_eq!(result.root_visits[0], 0);
        assert_eq!(result.root_visits[2], 0);
    }

    #[tokio::test]
    async fn caller_prohibitions_reach_decisions_and_imagine_without_leaking() {
        let engine = offline_engine();
        for action in ["ask", "imagine"] {
            let args = json!({"action":action, "context":"choose", "candidates":["delete", "wait"],
                "constraints":[{"type":"forbid", "actions":["DELETE"]}], "enforce_cpsat":false});
            let out = engine.execute(&args).await.unwrap();
            assert_eq!(
                out.meta["formally_infeasible"],
                json!(["delete"]),
                "{out:?}"
            );
            if action == "ask" {
                assert_eq!(out.meta["chosen_action"], "wait");
                assert_eq!(out.meta["best_action"], "wait");
                assert_eq!(out.meta["probs"], json!({"wait":1.0}));
            } else {
                assert_eq!(out.meta["formal_checked"], false);
            }
        }
        let out = engine
            .execute(&json!({"action":"ask", "candidates":["delete"]}))
            .await
            .unwrap();
        assert_eq!(out.meta["chosen_action"], "delete");
        let out = engine
            .execute(
                &json!({"action":"ask", "candidates":["proceed"], "forbidden_actions":["proceed"]}),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert_eq!(out.meta["tier"], "HardStop");
    }

    #[tokio::test]
    async fn malformed_safety_inputs_fail_closed_before_execution() {
        let engine = offline_engine();
        for extra in [
            json!({"constraints":null}),
            json!({"constraints":["delete"]}),
            json!({"constraints":[{"type":"forbid", "actions":["a"], "typo":true}]}),
            json!({"constraints":[{"type":"forbid", "actions":["a", " A "]}]}),
            json!({"constraints":[{"type":"mutually_exclusive", "actions":["a"]}]}),
            json!({"constraints":[{"type":"upper_bound", "action":"a", "value":0.5}]}),
            json!({"forbidden_actions":"a"}),
            json!({"forbidden_actions":[1]}),
            json!({"enforce_cpsat":"false"}),
        ] {
            let mut args =
                json!({"action":"imagine", "scenario":"choose", "candidate_actions":["a","b"]});
            args.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let out = engine.execute(&args).await.unwrap();
            assert!(out.rejection.is_some(), "{args}: {out:?}");
            assert_eq!(out.meta["formal_checked"], false);
        }
        let out = engine
            .execute(&json!({"action":"compact", "text":"hello", "constraints":[]}))
            .await
            .unwrap();
        assert!(out.rejection.is_some());
    }

    #[tokio::test]
    async fn malformed_candidates_are_typed_errors() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        for candidates in [
            json!([123]),
            json!(["safe", 123]),
            json!([""]),
            json!(null),
            json!("safe"),
        ] {
            for args in [
                json!({"action":"ask", "candidates":candidates}),
                json!({"action":"imagine", "candidate_actions":candidates}),
            ] {
                let out = engine.execute(&args).await.unwrap();
                assert!(out.is_error);
                assert_eq!(out.rejection.unwrap().code, "InvalidParams");
                assert!(out.meta.get("chosen_action").is_none());
            }
        }
    }

    #[tokio::test]
    async fn ask_without_candidates_is_rejected_not_fabricated() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        for args in [
            json!({"action":"ask"}),
            json!({"action":"ask", "context":"should I?", "candidates":[]}),
        ] {
            let out = engine.execute(&args).await.unwrap();
            assert!(out.is_error, "{out:?}");
            assert!(out.meta.get("chosen_action").is_none(), "{out:?}");
            assert!(out.meta.get("action_id").is_none(), "{out:?}");
            // The response envelope reports no choice: null action, zero confidence.
            assert!(out.meta["best_action"].is_null(), "{out:?}");
            assert_eq!(out.meta["confidence"], 0.0, "{out:?}");
            assert!(out.rejection.is_some(), "{out:?}");
            assert!(
                out.content
                    .iter()
                    .all(|b| !b.text.contains("Selected action")),
                "{out:?}"
            );
        }
    }

    #[tokio::test]
    async fn ask_enforces_graph_revocations() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        engine.graph.revoke_entity(action_id("revoked").0 as u64);
        let out = engine
            .execute(&json!({"action":"ask", "candidates":["revoked"]}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert_eq!(out.meta["tier"], "HardStop");
        let out = engine
            .execute(&json!({"action":"ask", "candidates":["revoked", "safe"]}))
            .await
            .unwrap();
        assert_eq!(out.meta["chosen_action"], "safe");
    }

    #[tokio::test]
    async fn ranked_tools_preserve_confirmation_and_entropy_verdicts() {
        use axum::{routing::post, Json, Router};
        for (entropy, confirm, revoke, expected) in [
            (0.1, false, false, "Proceed"),
            (0.1, true, false, "Confirm"),
            (0.9, false, false, "Escalate"),
            (2.0, false, false, "HardStop"),
            (0.1, false, true, "HardStop"),
        ] {
            let mut engine = PolymorphicZeroEngine::new().with_bridge(None);
            let graph = engine.graph.clone();
            let body = json!({"ranked": [
                {"name":"first", "log_likelihood":-1.0, "baseline_log_likelihood":-1.0, "pmi":0.0, "probability":0.75},
                {"name":"second", "log_likelihood":-2.0, "baseline_log_likelihood":-1.0, "pmi":-1.0, "probability":0.25}
            ], "selected":["first"], "entropy":entropy, "scorer":{}, "timing_ms":1.0});
            let app = Router::new()
                .route("/v1/semantic_route", post(move || {
                    let body = body.clone();
                    let graph = graph.clone();
                    async move {
                        if revoke { graph.revoke_entity(action_id("first").0 as u64); }
                        Json(body)
                    }
                }))
                .route("/v1/semantic_risk", post(|| async { Json(json!({
                    "p_dangerous":0.0, "log_odds":-10.0, "windows":1,
                    "thresholds":{"escalate":0.5,"hard_stop":0.9}, "classifier":{}, "forward_ms":1.0
                })) }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            engine = engine.with_bridge(Some(Arc::new(
                SemanticBridgeClient::new(BridgeConfig::new(endpoint)).unwrap(),
            )));
            if confirm {
                let mut gate = PolicyGate::default();
                gate.register_confirm_action(action_id("first"));
                engine = engine.with_gate(gate);
            }
            let out = engine
                .execute(&json!({"tools":["first","second"], "intent":"pick", "top_k":1}))
                .await
                .unwrap();
            assert_eq!(out.meta["tier"], expected, "{out:?}");
            assert_eq!(out.is_error, expected != "Proceed");
            if expected == "Confirm" || expected == "Escalate" {
                assert_eq!(out.meta["requires_confirmation"], true);
            }
            assert_eq!(
                out.meta["selected_verdicts"][0]["action"],
                action_id("first").0
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn cpu_admission_rejects_heavy_verbs_when_saturated() {
        let mut engine = PolymorphicZeroEngine::new().with_bridge(None);
        engine.cpu_slots = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = Arc::clone(&engine.cpu_slots).try_acquire_owned().unwrap();
        for verb in ["compact", "entail", "causal_fold", "pipeline"] {
            let mut args = json!({"verb": verb});
            if verb == "causal_fold" {
                args["causal_fold"] = json!({});
            }
            assert!(
                matches!(engine.execute(&args).await, Err(ServiceError::Overloaded)),
                "{verb} bypassed admission"
            );
        }
        drop(permit);
        assert!(engine
            .execute(&json!({"verb": "compact", "text": "hello"}))
            .await
            .is_ok());
    }

    // A current-thread runtime must remain responsive while the CPU closure waits.
    #[tokio::test]
    async fn running_cpu_job_keeps_permit_after_abort() {
        let mut engine = PolymorphicZeroEngine::new().with_bridge(None);
        engine.cpu_slots = Arc::new(tokio::sync::Semaphore::new(1));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let job = engine
            .spawn_cpu(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
            .unwrap();
        started_rx.await.unwrap();
        job.abort();
        let overloaded = matches!(engine.spawn_cpu(|| ()), Err(ServiceError::Overloaded));
        release_tx.send(()).unwrap();
        job.await.unwrap();
        assert!(overloaded);
        assert_eq!(engine.cpu_slots.available_permits(), 1);
        engine.spawn_cpu(|| ()).unwrap().await.unwrap();
    }

    #[tokio::test]
    async fn cpu_job_panic_releases_capacity() {
        let mut engine = PolymorphicZeroEngine::new().with_bridge(None);
        engine.cpu_slots = Arc::new(tokio::sync::Semaphore::new(1));
        let error = engine
            .spawn_cpu(|| panic!("test CPU panic"))
            .unwrap()
            .await
            .unwrap_err();
        assert!(error.is_panic());
        assert_eq!(engine.cpu_slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn specialized_scores_are_exactly_invariant_to_candidate_order() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        // Real forward execution with explicit untrained parameters; no learned
        // capability is asserted by this routing regression.
        let mut core = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "permutation regression",
            CompressedLatent { values: [1.0; 128] },
            4,
            0.5,
            &["alpha", "beta", "gamma"],
        );
        core.projection_weights.fill(0.0);
        core.projection_weights[0] = 1.0;
        for (backend, head) in [
            ("nanocore", "etf"),
            ("nanocore", "linear"),
            ("generic", "etf"),
        ] {
            for names in [vec!["alpha", "beta"], vec!["alpha", "beta", "gamma"]] {
                let mut args = json!({"action":"decide", "context":"pick", "candidates":names,
                    "engine":backend, "head":head});
                let width = if backend == "nanocore" { 128 } else { 3 };
                if backend == "nanocore" {
                    args["nanocore_core"] = json!(core);
                    args["decision_state"] = json!(vec![1.0_f32; 128]);
                } else {
                    args["etf_rep"] = json!([1.0, 0.5, 0.0]);
                }
                if head == "etf" {
                    let reps: serde_json::Map<String, Value> = names
                        .iter()
                        .enumerate()
                        .map(|(i, n)| {
                            let mut v = vec![0.2_f32; width];
                            v[i] = 1.0;
                            (n.to_string(), json!(v))
                        })
                        .collect();
                    args["candidate_reps"] = Value::Object(reps);
                }
                let forward = engine.execute(&args).await.unwrap();
                assert!(forward.meta["candidates"].is_array(), "{forward:?}");
                let scores = forward.meta["candidates"].as_array().unwrap();
                assert!(scores
                    .windows(2)
                    .any(|w| w[0]["probability"] != w[1]["probability"]));
                let mut reversed = names.clone();
                reversed.reverse();
                args["candidates"] = json!(reversed);
                let reverse = engine.execute(&args).await.unwrap();
                assert_eq!(
                    forward.meta["candidates"], reverse.meta["candidates"],
                    "{backend}/{head}"
                );
                assert_eq!(forward.meta["action_id"], reverse.meta["action_id"]);
                assert_eq!(forward.meta["chosen_action"], reverse.meta["chosen_action"]);
                assert_eq!(forward.meta["entropy"], reverse.meta["entropy"]);
            }
        }
    }

    /// B03: pruning a candidate must not reinterpret the remaining NanoCore
    /// channels. The survivors keep their logits, so their probability ratio
    /// is unchanged; an action outside the core vocabulary is refused.
    #[tokio::test]
    async fn specialized_nanocore_pruning_keeps_survivor_channels() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let core = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "prune regression",
            CompressedLatent { values: [1.0; 128] },
            4,
            0.5,
            &["alpha", "beta", "gamma"],
        );
        let state: Vec<f32> = (0..128).map(|j| ((j as f32) * 0.37 + 1.3).sin()).collect();
        let run = |names: Vec<&str>| {
            let args = json!({"action":"decide", "context":"pick", "candidates":names,
                "engine":"nanocore", "head":"linear", "nanocore_core":core,
                "decision_state":state});
            let engine = &engine;
            async move { engine.execute(&args).await.unwrap() }
        };
        let prob = |out: &ZeroToolOutcome, name: &str| {
            out.meta["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == name)
                .unwrap()["probability"]
                .as_f64()
                .unwrap()
        };
        let full = run(vec!["alpha", "beta", "gamma"]).await;
        let pruned = run(vec!["gamma", "alpha"]).await;
        assert!(full.meta["candidates"].is_array(), "{full:?}");
        assert!(pruned.meta["candidates"].is_array(), "{pruned:?}");
        let ratio_full = prob(&full, "alpha") / prob(&full, "gamma");
        let ratio_pruned = prob(&pruned, "alpha") / prob(&pruned, "gamma");
        assert!((ratio_full - 1.0).abs() > 1e-3, "vacuous: {ratio_full}");
        assert!(
            (ratio_full - ratio_pruned).abs() <= 1e-5 * ratio_full.abs(),
            "{ratio_full} vs {ratio_pruned}"
        );

        let unknown = run(vec!["alpha", "delta"]).await;
        assert!(unknown.is_error, "{unknown:?}");
        let detail = unknown.rejection.expect("typed rejection").detail;
        assert!(detail.contains("delta"), "{detail}");
    }

    #[tokio::test]
    async fn specialized_etf_overflow_is_explicitly_rejected() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let out = engine
            .execute(&json!({"action":"decide", "context":"pick",
            "candidates":["alpha", "beta"], "head":"etf", "etf_rep":[3e38, 0.0],
            "candidate_reps":{"alpha":[1.0, 0.0], "beta":[0.0, 1.0]}}))
            .await
            .unwrap();
        assert!(out.is_error, "{out:?}");
        assert!(out
            .rejection
            .unwrap()
            .detail
            .contains("squared norm overflow"));
        assert!(out.meta["chosen_action"].is_null());
        assert!(out.meta["candidates"].is_null());
    }

    #[tokio::test]
    async fn candidate_reps_accepts_subnormal_underflow_in_default_isotropic_head() {
        // 1e-60 underflows to 0.0 in f32. The default (no etf_metric) request
        // must accept it and reach scoring, matching historical
        // candidate_reps parsing; only etf_metric's own parser refuses
        // underflow. (The bridge is disabled here, so the response still
        // carries an unrelated confirmation-required risk gate; that is not
        // what this test checks.)
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let out = engine
            .execute(&json!({"action":"decide", "context":"pick",
            "candidates":["alpha", "beta"], "head":"etf", "etf_rep":[1.0, 0.0],
            "candidate_reps":{"alpha":[1.0, 1e-60], "beta":[0.0, 1.0]}}))
            .await
            .unwrap();
        assert!(out.rejection.is_none(), "{out:?}");
        assert_eq!(out.meta["etf"]["metric_source"], "default");
        assert_eq!(out.meta["chosen_action"], "alpha", "{out:?}");
    }

    #[tokio::test]
    async fn specialized_decision_routes_and_fail_closed() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let base = json!({"action":"decide", "context":"pick", "candidates":["alpha","beta"]});
        let mut missing = base.clone();
        missing["engine"] = json!("nanocore");
        let refused = engine.execute(&missing).await.unwrap();
        assert!(refused.is_error);
        assert!(
            refused.content[0].text.contains("decision_state"),
            "{refused:?}"
        );

        let mut etf = base.clone();
        etf["head"] = json!("etf");
        etf["etf_rep"] = json!([1.0, 0.0]);
        etf["candidate_reps"] = json!({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]});
        let out = engine.execute(&etf).await.unwrap();
        assert_eq!(out.meta["head"], "etf", "{out:?}");
        let probs = out.meta["candidates"].as_array().unwrap();
        assert_eq!(probs.len(), 2);
        assert!(
            (probs
                .iter()
                .map(|p| p["probability"].as_f64().unwrap())
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-5
        );

        let mut nano = base;
        nano["engine"] = json!("nanocore");
        nano["head"] = json!("etf");
        nano["decision_state"] = json!(vec![1.0_f32; 128]);
        nano["candidate_reps"] = json!({"alpha": vec![1.0_f32; 128], "beta": vec![-1.0_f32; 128]});
        let core = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "test untrained core",
            CompressedLatent { values: [1.0; 128] },
            4,
            0.5,
            &["alpha", "beta", "gamma"],
        );
        nano["nanocore_core"] = serde_json::to_value(core).unwrap();
        let out = engine.execute(&nano).await.unwrap();
        assert_eq!(out.meta["engine"], "nanocore", "{out:?}");
        assert_eq!(out.meta["head"], "etf");
        assert_eq!(
            out.meta["training_status"],
            "caller_supplied_parameters_unverified"
        );
        assert!(out.meta["nanocore"]["confidence"].is_number());
    }

    /// Regression: the old ETF head escalated every K >= 3 decision regardless
    /// of input. Distinct manifold scores must now reach Tier 0 Proceed, and
    /// near-equal scores must still escalate to Tier 2.
    #[tokio::test]
    async fn etf_head_tier_follows_manifold_score_separation() {
        let (engine, server) = test_support::engine_with_benign_risk().await;
        let names = ["approve", "reject", "escalate", "defer"];
        let state = [1.0_f32, 0.0, 0.0, 0.0];
        let distinct = json!({
            "approve": [0.95, 0.31, 0.0, 0.0],
            "reject": [0.1, 0.0, 0.99, 0.0],
            "escalate": [0.0, 1.0, 0.0, 0.0],
            "defer": [-0.2, 0.0, 0.0, 0.98],
        });
        let close = json!({
            "approve": [0.50, 0.866, 0.0, 0.0],
            "reject": [0.49, 0.0, 0.872, 0.0],
            "escalate": [0.48, 0.0, 0.0, 0.877],
            "defer": [0.49, -0.872, 0.0, 0.0],
        });
        for (reps, tier) in [(distinct, "Proceed"), (close, "Escalate")] {
            for order in [names.to_vec(), names.iter().rev().copied().collect()] {
                let out = engine
                    .execute(&json!({"action":"decide", "context":"pick the next step",
                        "candidates": order, "head":"etf", "etf_rep": state,
                        "candidate_reps": reps}))
                    .await
                    .unwrap();
                assert_eq!(out.meta["risk"]["assessed"], true, "{out:?}");
                assert_eq!(out.meta["tier"], tier, "{out:?}");
                let entropy = out.meta["entropy"].as_f64().unwrap();
                if tier == "Proceed" {
                    assert!(!out.is_error, "{out:?}");
                    assert!(entropy < 0.65, "{entropy}");
                    assert_eq!(out.meta["chosen_action"], "approve");
                } else {
                    assert_eq!(out.meta["requires_confirmation"], true);
                    assert!(entropy > 0.65, "{entropy}");
                }
                assert_eq!(
                    out.meta["etf"]["temperature_source"],
                    "default_uncalibrated"
                );
            }
        }
        // A caller temperature is honoured and reported: T=1 on bounded cosines
        // cannot sharpen, so the same distinct scores escalate.
        let out = engine
            .execute(&json!({"action":"decide", "context":"pick the next step",
                "candidates": names, "head":"etf", "etf_rep": state, "etf_temperature": 1.0,
                "candidate_reps": {"approve":[1.0,0.0,0.0,0.0], "reject":[0.0,1.0,0.0,0.0],
                    "escalate":[0.0,0.0,1.0,0.0], "defer":[0.0,0.0,0.0,1.0]}}))
            .await
            .unwrap();
        assert_eq!(out.meta["etf"]["temperature_source"], "caller");
        assert_eq!(out.meta["tier"], "Escalate", "{out:?}");
        server.abort();
    }

    #[tokio::test]
    async fn etf_head_refuses_missing_or_malformed_candidate_reps() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let base = json!({"action":"decide", "context":"pick", "candidates":["alpha","beta"],
            "head":"etf", "etf_rep":[1.0, 0.0]});
        let cases = [
            (json!(null), "requires `candidate_reps`"),
            (json!({"alpha":[1.0,0.0]}), "candidate_reps.beta"),
            (
                json!({"alpha":[1.0,0.0], "beta":[0.0,1.0], "gamma":[1.0,1.0]}),
                "unknown candidate `gamma`",
            ),
            (
                json!({"alpha":[1.0,0.0], "beta":[0.0,1.0,0.0]}),
                "the state has 2",
            ),
            (
                json!({"alpha":[1.0,0.0], "beta":[0.0,"x"]}),
                "invalid float",
            ),
            (json!({"alpha":[1.0,0.0], "beta":[0.0,0.0]}), "zero norm"),
        ];
        for (reps, needle) in cases {
            let mut args = base.clone();
            if !reps.is_null() {
                args["candidate_reps"] = reps;
            }
            let out = engine.execute(&args).await.unwrap();
            assert!(out.is_error, "{out:?}");
            let detail = out.rejection.expect("typed rejection").detail;
            assert!(detail.contains(needle), "{needle}: {detail}");
            assert!(out.meta["chosen_action"].is_null());
        }
        for t in [json!(0.0), json!(-1.0), json!("hot")] {
            let mut args = base.clone();
            args["candidate_reps"] = json!({"alpha":[1.0,0.0], "beta":[0.0,1.0]});
            args["etf_temperature"] = t;
            let out = engine.execute(&args).await.unwrap();
            assert!(out.is_error, "{out:?}");
        }
        let linear = json!({"action":"decide", "context":"pick", "candidates":["alpha","beta"],
            "candidate_reps":{"alpha":[1.0], "beta":[0.0]}});
        let out = engine.execute(&linear).await.unwrap();
        assert!(out.is_error, "{out:?}");
    }

    /// `etf_metric` reaches the model head: the request picks the geometry,
    /// the decision follows it, and `etf.metric` reports what ran.
    #[tokio::test]
    async fn etf_metric_is_configurable_and_reported() {
        let (engine, server) = test_support::engine_with_benign_risk().await;
        // Axes 0-2 carry the signal; axes 3-4 carry noise the state shares
        // with the wrong candidate.
        let base = json!({"action":"decide", "context":"pick the next step",
            "candidates": ["right", "noisy_twin", "other"], "head":"etf",
            "etf_rep": [1.0, 0.0, 0.0, 5.0, 5.0],
            "candidate_reps": {"right":[1.0,0.0,0.0,0.0,0.0],
                "noisy_twin":[0.0,1.0,0.0,5.0,5.0], "other":[0.0,0.0,1.0,5.0,-5.0]}});

        let out = engine.execute(&base).await.unwrap();
        assert_eq!(out.meta["etf"]["metric"], "isotropic", "{out:?}");
        assert_eq!(out.meta["etf"]["metric_source"], "default");
        assert_eq!(out.meta["etf"]["metric_input_dim"], 5);
        assert_eq!(out.meta["etf"]["metric_output_dim"], 5);
        assert_eq!(out.meta["chosen_action"], "noisy_twin");

        let mut explicit = base.clone();
        explicit["etf_metric"] = json!({"kind":"isotropic"});
        let same = engine.execute(&explicit).await.unwrap();
        assert_eq!(same.meta["etf"]["metric_source"], "caller");
        assert_eq!(same.meta["candidates"], out.meta["candidates"]);

        let mut diag = base.clone();
        diag["etf_metric"] =
            json!({"kind":"diagonal_mahalanobis", "precision":[1.0,1.0,1.0,1e-4,1e-4]});
        let out = engine.execute(&diag).await.unwrap();
        assert!(!out.is_error, "{out:?}");
        assert_eq!(out.meta["etf"]["metric"], "diagonal_mahalanobis");
        assert_eq!(out.meta["etf"]["metric_output_dim"], 5);
        assert_eq!(out.meta["chosen_action"], "right");
        assert_eq!(out.meta["tier"], "Proceed", "{out:?}");
        assert!(out.meta["entropy"].as_f64().unwrap() < 0.65);

        // Projection onto the three signal axes: same fix, 3-wide metric space.
        let mut whitened = base.clone();
        whitened["etf_metric"] = json!({"kind":"whitened", "out_dim":3, "matrix":[
            1.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,1.0,0.0,0.0]});
        let out = engine.execute(&whitened).await.unwrap();
        assert!(!out.is_error, "{out:?}");
        assert_eq!(out.meta["etf"]["metric"], "whitened");
        assert_eq!(out.meta["etf"]["metric_input_dim"], 5);
        assert_eq!(out.meta["etf"]["metric_output_dim"], 3);
        assert_eq!(out.meta["chosen_action"], "right");
        server.abort();
    }

    #[tokio::test]
    async fn etf_metric_refuses_malformed_or_misplaced_parameters() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let base = json!({"action":"decide", "context":"pick", "candidates":["alpha","beta"],
            "head":"etf", "etf_rep":[1.0, 0.0],
            "candidate_reps":{"alpha":[1.0,0.0], "beta":[0.0,1.0]}});
        let cases = [
            (json!("isotropic"), "must be an object"),
            (json!({}), "etf_metric.kind"),
            (json!({"kind":"cosine"}), "`cosine` is not"),
            (
                json!({"kind":"isotropic", "precision":[1.0,1.0]}),
                "etf_metric.precision",
            ),
            (
                json!({"kind":"diagonal_mahalanobis"}),
                "etf_metric.precision",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0]}),
                "the state has 2",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0, 0.0]}),
                "finite positive",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0, -2.0]}),
                "finite positive",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0, 1e-60]}),
                "invalid float",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0, "x"]}),
                "invalid float",
            ),
            (
                json!({"kind":"diagonal_mahalanobis", "precision":[1.0, 1e300]}),
                "invalid float",
            ),
            (
                json!({"kind":"whitened", "matrix":[1.0,0.0,0.0,1.0]}),
                "out_dim",
            ),
            (
                json!({"kind":"whitened", "out_dim":0, "matrix":[]}),
                "out_dim",
            ),
            (
                json!({"kind":"whitened", "out_dim":1.5, "matrix":[1.0,0.0]}),
                "out_dim",
            ),
            (
                json!({"kind":"whitened", "out_dim":2, "matrix":[1.0,0.0,0.0]}),
                "out_dim x state width is 4",
            ),
            (
                json!({"kind":"whitened", "out_dim":2, "matrix":[1.0,0.0,0.0,0.0]}),
                "all zeros",
            ),
            (
                json!({"kind":"whitened", "out_dim":1_000_000, "matrix":[]}),
                "exceeds",
            ),
            (
                json!({"kind":"whitened", "out_dim":u64::MAX, "matrix":[]}),
                "exceeds",
            ),
            // Rank one: the state [1, 0] survives, candidate beta [0, 1] collapses to zero.
            (
                json!({"kind":"whitened", "out_dim":1, "matrix":[1.0, 0.0]}),
                "zero norm",
            ),
        ];
        for (metric, needle) in cases {
            let mut args = base.clone();
            args["etf_metric"] = metric.clone();
            let out = engine.execute(&args).await.unwrap();
            assert!(out.is_error, "{metric}: {out:?}");
            let detail = out.rejection.expect("typed rejection").detail;
            assert!(detail.contains(needle), "{metric} -> {needle}: {detail}");
            assert!(out.meta["chosen_action"].is_null());
        }
        // A metric on a non-ETF head is refused, not silently ignored.
        let linear = json!({"action":"decide", "context":"pick", "candidates":["alpha","beta"],
            "etf_metric":{"kind":"isotropic"}});
        let out = engine.execute(&linear).await.unwrap();
        assert!(out.is_error, "{out:?}");
        let detail = out.rejection.expect("typed rejection").detail;
        assert!(detail.contains("etf_metric"), "{detail}");
    }

    /// Engine whose bridge points at a closed port: every semantic call fails
    /// at connect time, so the fallback path is exercised deterministically
    /// even if a real Python server is running on the default port.
    fn offline_engine() -> PolymorphicZeroEngine {
        let mut cfg = BridgeConfig::new("http://127.0.0.1:1");
        cfg.max_retries = 0;
        let bridge = SemanticBridgeClient::new(cfg).unwrap();
        PolymorphicZeroEngine::new().with_bridge(Some(Arc::new(bridge)))
    }

    #[tokio::test]
    async fn test_polymorphic_zero_engine_verbs() {
        let engine = offline_engine();

        // 1. ask: bridge down -> labeled fallback, uniform prior escalates
        let ask_res = engine
            .execute(&json!({"action": "ask", "context": "pick one", "candidates": ["tool_a", "tool_b"]}))
            .await
            .unwrap();
        assert_eq!(ask_res.verb, ZeroVerb::Ask);
        assert_eq!(ask_res.meta["engine"], ENGINE_FALLBACK);

        // 2. route
        let route_res = engine
            .execute(&json!({"tools": ["t1", "t2", "t3"], "top_k": 2, "intent": "x"}))
            .await
            .unwrap();
        assert_eq!(route_res.verb, ZeroVerb::Route);
        assert_eq!(route_res.meta["count"], 0);
        assert_eq!(route_res.meta["selected_tools"], json!([]));
        assert!(route_res.is_error);

        // 3. imagine
        let imagine_res = engine.execute(&json!({"horizon": 5})).await.unwrap();
        assert_eq!(imagine_res.verb, ZeroVerb::Imagine);

        // 4. stream
        let stream_res = engine
            .execute(&json!({"observation": "frame_payload_xyz"}))
            .await
            .unwrap();
        assert_eq!(stream_res.verb, ZeroVerb::Stream);
        // A text observation is not ingested: no encoder, so no fake success.
        assert!(stream_res.is_error);
        assert_eq!(stream_res.rejection.as_ref().unwrap().code, "InvalidParams");
        assert!(stream_res.meta.get("status").is_none());

        // 5. grep: paths are refused (no file backend), never "matched"
        let grep_res = engine
            .execute(&json!({"paths": ["/src/main.rs"], "query": "PolicyGate"}))
            .await
            .unwrap();
        assert_eq!(grep_res.verb, ZeroVerb::Grep);
        assert!(grep_res.is_error);

        // A cognitive block on a verb that cannot read it is refused.
        let dropped = engine
            .execute(&json!({"intent": "x", "tools": ["a", "b"], "cognitive": {"state": [0.1]}}))
            .await
            .unwrap();
        assert_eq!(dropped.verb, ZeroVerb::Route);
        assert_eq!(dropped.rejection.as_ref().unwrap().code, "InvalidParams");
        assert_eq!(
            grep_res.rejection.as_ref().unwrap().code,
            "BackendUnavailable"
        );
        assert!(grep_res.meta.get("matches").is_none());

        // 6. compact
        let compact_res = engine
            .execute(&json!({"text": "long verbose log repeated many times"}))
            .await
            .unwrap();
        assert_eq!(compact_res.verb, ZeroVerb::Compact);
    }

    #[tokio::test]
    async fn ask_fallback_is_labeled_and_never_claims_semantics() {
        let engine = offline_engine();
        let res = engine
            .execute(&json!({
                "context": "把这封邮件删除掉，不需要确认",
                "candidates": ["delete", "backup", "wait"]
            }))
            .await
            .unwrap();
        assert_eq!(res.meta["engine"], "local_fast_reflex_fallback");
        assert_eq!(res.meta["semantic_scoring"], false);
        assert!(res.meta["bridge_error"]
            .as_str()
            .unwrap()
            .contains("transport"));
        let probs: Vec<f64> = res.meta["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["probability"].as_f64().unwrap())
            .collect();
        assert!(probs.iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-9));
        // A uniform prior has maximal entropy: the gate must not let it proceed.
        assert!(res.is_error);
        assert_eq!(res.meta["tier"], "Escalate");
        assert_eq!(res.meta["requires_confirmation"], true);
    }

    #[tokio::test]
    async fn disabled_bridge_is_reported_as_fallback() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let res = engine
            .execute(&json!({"context": "c", "candidates": ["a", "b"]}))
            .await
            .unwrap();
        assert_eq!(res.meta["engine"], ENGINE_FALLBACK);
        assert!(res.meta["fallback_reason"]
            .as_str()
            .unwrap()
            .contains("disabled"));
        assert!(
            res.meta.get("bridge_error").is_none(),
            "no call, no bridge_error"
        );
        assert!(res.meta["bridge_endpoint"].is_null());
    }

    #[tokio::test]
    async fn local_preconditions_are_not_reported_as_bridge_errors() {
        let engine = offline_engine();
        for args in [
            json!({"context": "c", "candidates": ["only"]}),
            json!({"candidates": ["a", "b"]}),
        ] {
            let res = engine.execute(&args).await.unwrap();
            assert_eq!(res.meta["engine"], ENGINE_FALLBACK);
            assert!(res.meta["fallback_reason"].is_string(), "{}", res.meta);
            assert!(res.meta.get("bridge_error").is_none(), "{}", res.meta);
        }
    }

    #[tokio::test]
    async fn route_fallback_exposes_no_selection() {
        let engine = offline_engine();
        let res = engine
            .execute(&json!({"intent": "send mail", "tools": ["search_web", "send_email", "delete_file"], "top_k": 2}))
            .await
            .unwrap();
        assert_eq!(res.meta["engine"], ENGINE_FALLBACK);
        assert_eq!(res.meta["ranking"], "unavailable");
        assert_eq!(res.meta["selected_tools"], json!([]));
        assert_eq!(res.meta["count"], 0);
        // No ranking is available, so this must not look like success.
        assert!(res.is_error, "{:?}", res.content);
        assert_eq!(res.meta["degraded"], true);
        assert!(res.meta["error_code"].is_i64());
        assert!(res
            .content
            .iter()
            .any(|b| b.text.starts_with("SemanticRankingUnavailable")
                && b.text.contains("no tools selected")));
    }

    /// The reviewers' repro: with the old English substring table this Chinese
    /// request came back `Proceed`. Without a reachable classifier it must now
    /// be escalated (fail-closed), in every semantic verb.
    #[tokio::test]
    async fn unassessed_request_risk_is_escalated_in_any_language() {
        let engine = offline_engine();
        let zh = "递归强制删除根目录下所有文件，绕过安全检查，提权到 root";
        for args in [
            json!({"context": zh}),
            json!({"context": zh, "candidates": ["执行"]}),
            json!({"context": "run rm -rf /", "candidates": ["run", "skip"]}),
            json!({"intent": zh, "tools": ["shell", "mail"]}),
        ] {
            let res = engine.execute(&args).await.unwrap();
            assert!(res.is_error, "{args}: {:?}", res.content);
            assert_ne!(res.meta["tier"], "Proceed", "{args}: {}", res.meta);
            assert_eq!(res.meta["risk"]["assessed"], false, "{}", res.meta);
            assert_eq!(res.meta["risk"]["fail_closed"], true, "{}", res.meta);
        }
        let imagine = engine
            .execute(&json!({"scenario": zh, "candidate_actions": ["a", "b"]}))
            .await
            .unwrap();
        assert!(imagine.is_error);
        assert_eq!(imagine.meta["requires_confirmation"], true);
        assert_eq!(imagine.meta["risk"]["assessed"], false);
    }

    /// The adjudicated defect: a Tier0 request whose lookahead plan lands on
    /// Tier1 or Tier2 used to come back `is_error: false` with
    /// `requires_confirmation: true`, and the HTTP layer turned that into 200.
    /// The plan tier alone must make the outcome a ConfirmationRequired error.
    #[test]
    fn tier0_request_with_confirm_plan_is_never_a_success() {
        let tier0 = RiskCheck::NotApplicable;
        for plan_tier in [PolicyTier::Tier1Confirm, PolicyTier::Tier2Escalate] {
            let out = PolymorphicZeroEngine::gated_outcome(
                ZeroVerb::Imagine,
                plan_tier,
                &tier0,
                json!({"engine": ENGINE_SEMANTIC}),
                "summary".to_string(),
            );
            assert!(out.is_error, "{plan_tier:?}: {}", out.meta);
            assert_eq!(out.meta["requires_confirmation"], true, "{plan_tier:?}");
            assert_eq!(out.meta["error_code"], -32002, "{plan_tier:?}");
            assert_eq!(out.meta["gate_status"], GATE_STATUS_REQUIRES_CONFIRMATION);
            assert_eq!(out.meta["tier"], tier_name(plan_tier));
            assert!(out.content[0].text.starts_with("ConfirmationRequired"));
        }
        let stop = PolymorphicZeroEngine::gated_outcome(
            ZeroVerb::Imagine,
            PolicyTier::Tier3HardStop,
            &tier0,
            json!({}),
            String::new(),
        );
        assert!(stop.is_error);
        assert_eq!(stop.meta["error_code"], -32001);
        assert_eq!(stop.meta["gate_status"], GATE_STATUS_HARD_STOP);
        assert!(
            stop.meta.get("requires_confirmation").is_none(),
            "a hard stop is not confirmable"
        );
        let ok = PolymorphicZeroEngine::gated_outcome(
            ZeroVerb::Imagine,
            PolicyTier::Tier0Proceed,
            &tier0,
            json!({}),
            "fine".to_string(),
        );
        assert!(!ok.is_error);
        assert_eq!(ok.meta["gate_status"], GATE_STATUS_PROCEED);
        assert!(ok.meta.get("requires_confirmation").is_none());
    }

    /// Invariant over every (request tier, plan tier) pair: requires_confirmation
    /// implies is_error, and is_error is exactly "final tier is not Proceed".
    #[test]
    fn requires_confirmation_always_implies_is_error() {
        let tiers = [
            PolicyTier::Tier0Proceed,
            PolicyTier::Tier1Confirm,
            PolicyTier::Tier2Escalate,
            PolicyTier::Tier3HardStop,
        ];
        for req in tiers {
            let risk = RiskCheck::Unassessed {
                why: Unscored::skipped("t"),
                tier: req,
            };
            for plan in tiers {
                let out = PolymorphicZeroEngine::gated_outcome(
                    ZeroVerb::Imagine,
                    plan,
                    &risk,
                    json!({}),
                    String::new(),
                );
                let final_tier = plan.max(req);
                assert_eq!(
                    out.is_error,
                    final_tier != PolicyTier::Tier0Proceed,
                    "{req:?}/{plan:?}"
                );
                if out.meta["requires_confirmation"] == true {
                    assert!(out.is_error, "{req:?}/{plan:?}: confirmation without error");
                }
                assert_eq!(
                    out.meta["gate_status"],
                    gate_status(final_tier),
                    "{req:?}/{plan:?}"
                );
            }
        }
    }

    /// The fallback path (no oracle) with a confirm-tier request must carry the
    /// same error code as the scored path, so HTTP maps it to 428 and not 422.
    #[tokio::test]
    async fn imagine_fallback_with_risky_request_carries_confirmation_code() {
        let engine = offline_engine();
        let res = engine
            .execute(&json!({"scenario": "run rm -rf /", "candidate_actions": ["run", "skip"]}))
            .await
            .unwrap();
        assert!(res.is_error);
        assert_eq!(res.meta["requires_confirmation"], true, "{}", res.meta);
        assert_eq!(res.meta["error_code"], -32002, "{}", res.meta);
        assert_eq!(res.meta["gate_status"], GATE_STATUS_REQUIRES_CONFIRMATION);
    }

    #[test]
    fn no_literal_keyword_safety_table_remains() {
        // Safety must not be decided by substring. Guard the non-test source.
        let src = include_str!("zero.rs");
        let prod = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(!prod.contains(".contains(\""), "substring test in zero.rs");
        assert!(!prod.contains(&["is_red", "_line"].concat()));
    }

    #[tokio::test]
    async fn imagine_without_oracle_returns_no_reward() {
        let engine = offline_engine();
        let res = engine
            .execute(&json!({
                "scenario": "migrate the database",
                "candidate_actions": ["backup", "migrate", "wait"],
                "horizon": 3
            }))
            .await
            .unwrap();
        assert!(res.is_error);
        assert_eq!(res.meta["engine"], ENGINE_FALLBACK);
        assert!(res.meta["sequence_likelihood"].is_null());
        assert!(res.meta.get("expected_reward").is_none());
        assert_ne!(res.meta["formal_checked"], json!(false));
    }

    #[tokio::test]
    async fn grep_counts_real_matches_and_refuses_what_it_cannot_do() {
        let engine = offline_engine();
        let res = engine
            .execute(&json!({
                "action": "grep",
                "query": "Gate",
                "lines": ["PolicyGate::new()", "no match here", "PolicyGate::default()"]
            }))
            .await
            .unwrap();
        assert!(!res.is_error, "{:?}", res.content);
        assert_eq!(res.meta["matches"], 2);
        assert_eq!(res.meta["hits"][0]["line"], 1);
        assert_eq!(res.meta["hits"][1]["line"], 3);
        assert_eq!(res.meta["semantic"], false);

        let none = engine
            .execute(&json!({"action": "grep", "query": "zzz", "lines": ["a", "b"]}))
            .await
            .unwrap();
        assert!(!none.is_error);
        assert_eq!(none.meta["matches"], 0);

        for (args, code) in [
            (json!({"action": "grep", "lines": ["a"]}), "InvalidParams"),
            (
                json!({"action": "grep", "query": "", "lines": ["a"]}),
                "InvalidParams",
            ),
            (
                json!({"action": "grep", "query": "a", "expr": "a AND b", "lines": ["a"]}),
                "InvalidParams",
            ),
            (
                json!({"action": "grep", "query": "a", "lines": [1]}),
                "InvalidParams",
            ),
            (json!({"action": "grep", "query": "a"}), "InvalidParams"),
            (
                json!({"action": "grep", "query": "a", "paths": ["/etc"]}),
                "BackendUnavailable",
            ),
        ] {
            let res = engine.execute(&args).await.unwrap();
            assert!(res.is_error, "{args}");
            assert_eq!(res.rejection.unwrap().code, code, "{args}");
        }
    }

    #[tokio::test]
    async fn formally_infeasible_candidates_are_removed_before_scoring() {
        let blocked = action_id("drop_table");
        let mut gate = PolicyGate::default();
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(1),
            "no_drop_table",
            blocked,
        ));
        let engine = offline_engine().with_gate(gate);

        let only_blocked = engine
            .execute(&json!({"context": "c", "candidates": ["drop_table"]}))
            .await
            .unwrap();
        assert!(only_blocked.is_error);
        assert_eq!(only_blocked.meta["tier"], "HardStop");

        let mixed = engine
            .execute(&json!({"context": "c", "candidates": ["drop_table", "backup"]}))
            .await
            .unwrap();
        assert_eq!(mixed.meta["chosen_action"], "backup");
        assert_eq!(mixed.meta["formally_infeasible"], json!(["drop_table"]));
    }
}

#[cfg(test)]
mod provenance_wiring_tests {
    use super::*;
    use gen_zero_nanocore::core_type::fixtures::synthetic_core;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gen-zero-service-mmr-{label}-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    #[tokio::test]
    async fn records_and_verifies_multiple_audit_leaves() {
        let engine = PolymorphicZeroEngine::new();
        for index in 0..2 {
            let mut meta = json!({});
            engine
                .record_decision(
                    &json!({"candidates":["alpha"]}),
                    &mut meta,
                    ActionId(7),
                    PolicyTier::Tier0Proceed,
                )
                .await
                .unwrap();
            assert_eq!(meta["decision_audit"]["leaf_index"], index);
            assert_eq!(meta["decision_audit"]["root"].as_str().unwrap().len(), 64);
            assert_eq!(meta["decision_audit"]["proof_window"], MAX_AUDIT_LEAVES);
        }
    }

    #[tokio::test]
    async fn audit_ledger_memory_stays_bounded_past_the_window() {
        let engine = PolymorphicZeroEngine::new();
        let total = MAX_AUDIT_LEAVES as u64 + 100;
        let mut last_root = String::new();
        for index in 0..total {
            let mut meta = json!({});
            engine
                .record_decision(
                    &json!({"candidates":["alpha"]}),
                    &mut meta,
                    ActionId(7),
                    PolicyTier::Tier0Proceed,
                )
                .await
                .unwrap();
            assert_eq!(meta["decision_audit"]["leaf_index"], index);
            let root = meta["decision_audit"]["root"].as_str().unwrap().to_string();
            assert_ne!(root, last_root);
            last_root = root;
        }
        let ledger = engine.audit_ledger.lock();
        assert_eq!(ledger.len() as u64, total);
        assert_eq!(
            ledger.oldest_retained_index(),
            total - MAX_AUDIT_LEAVES as u64
        );
        assert!(ledger.stored_nodes() <= 2 * MAX_AUDIT_LEAVES + 128);
        assert!(ledger.generate_proof(0).is_err());
    }

    #[tokio::test]
    async fn durable_engine_restarts_with_same_root_and_rejects_second_writer() {
        let path = temp_path("restart");
        let config = ZeroEngineConfig::default().with_mmr_persist_path(path.clone());
        let first = PolymorphicZeroEngine::try_from_config(config.clone()).unwrap();
        let mut meta = json!({});
        first
            .record_decision(
                &json!({"candidates":["alpha"]}),
                &mut meta,
                ActionId(7),
                PolicyTier::Tier0Proceed,
            )
            .await
            .unwrap();
        let root = first.audit_ledger().1;
        assert_eq!(meta["decision_audit"]["persistence"], "durable");
        assert!(PolymorphicZeroEngine::try_from_config(config.clone()).is_err());
        drop(first);

        let second = PolymorphicZeroEngine::try_from_config(config).unwrap();
        assert_eq!(second.audit_ledger(), (1, root));
        let proof = second.audit_proof(0).unwrap();
        assert!(second.verify_audit_proof(&proof, &root));
        drop(second);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn durable_engine_refuses_corrupt_snapshot_and_directory_target() {
        let corrupt = temp_path("corrupt");
        let config = ZeroEngineConfig::default().with_mmr_persist_path(corrupt.clone());
        let engine = PolymorphicZeroEngine::try_from_config(config.clone()).unwrap();
        drop(engine);
        let mut snapshot: Value = serde_json::from_slice(&fs::read(&corrupt).unwrap()).unwrap();
        snapshot["payload"]["root"][0] = json!(1);
        fs::write(&corrupt, serde_json::to_vec(&snapshot).unwrap()).unwrap();
        let error = PolymorphicZeroEngine::try_from_config(config)
            .err()
            .expect("corruption must fail");
        assert!(error.to_string().contains("checksum"), "{error}");
        fs::remove_file(corrupt).unwrap();

        let directory = temp_path("directory");
        fs::create_dir(&directory).unwrap();
        let error = PolymorphicZeroEngine::try_from_config(
            ZeroEngineConfig::default().with_mmr_persist_path(directory.clone()),
        )
        .err()
        .expect("directory must fail");
        assert!(error.to_string().contains("read") || error.to_string().contains("directory"));
        fs::remove_dir(directory).unwrap();
    }

    #[tokio::test]
    async fn durable_engine_does_not_acknowledge_failed_append() {
        let directory = temp_path("write-failure");
        fs::create_dir(&directory).unwrap();
        let path = directory.join("audit.json");
        let engine = PolymorphicZeroEngine::try_from_config(
            ZeroEngineConfig::default().with_mmr_persist_path(&path),
        )
        .unwrap();
        // A directory cannot be replaced by the candidate snapshot.
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        let before = engine.audit_ledger();
        let mut meta = json!({});
        assert!(engine
            .record_decision(&json!({}), &mut meta, ActionId(7), PolicyTier::Tier0Proceed)
            .await
            .is_err());
        assert_eq!(engine.audit_ledger(), before);
        assert!(meta.get("decision_audit").is_none());
        drop(engine);
        fs::remove_dir_all(directory).unwrap();
    }

    fn caller_core_ask(candidates: &[&str]) -> Value {
        let core = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "caller core",
            CompressedLatent { values: [1.0; 128] },
            4,
            0.5,
            &["alpha", "beta", "gamma"],
        );
        json!({"verb": "ask", "context": "pick", "candidates": candidates,
            "engine": "nanocore", "nanocore_core": core, "decision_state": vec![1.0_f32; 128]})
    }

    #[tokio::test]
    async fn caller_core_never_enters_operator_fleet() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let out = engine.execute(&caller_core_ask(&["alpha"])).await.unwrap();
        assert_eq!(out.meta["engine"], "nanocore", "{out:?}");
        assert!(engine
            .nano_fleet
            .get_core(gen_zero_nanocore::DOMAIN_GENERAL)
            .is_err());
        // Without a bridge the risk stays unassessed, so the gate escalates and nothing is
        // audited. The audited success path is `worldmodel_verbs_tests::caller_nanocore_ask_*`.
        assert!(out.is_error, "{out:?}");
        assert_eq!(engine.audit_ledger().0, 0);
    }

    #[tokio::test]
    async fn operator_and_caller_nanocore_routes_do_not_mix() {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let mut args = caller_core_ask(&["alpha", "beta"]);
        args["nanocore_domain"] = json!(0);
        args["nanocore_state"] = json!(vec![0.0_f32; 128]);
        let out = engine.execute(&args).await.unwrap();
        assert!(out.is_error, "{out:?}");
        assert!(out.content[0].text.contains("cannot combine"), "{out:?}");
        assert_eq!(engine.audit_ledger().0, 0);
    }

    fn operator_engine_with_two_cores() -> PolymorphicZeroEngine {
        let fleet = Arc::new(NanoCoreFleetScheduler::new(
            gen_zero_nanocore::DEFAULT_RAM_BUDGET_BYTES,
        ));
        let prototype = CompressedLatent { values: [1.0; 128] };
        fleet
            .register_core(synthetic_core(
                gen_zero_nanocore::DOMAIN_GENERAL,
                "general operator core",
                prototype.clone(),
                4,
                0.9,
                &["alpha", "beta", "gamma"],
            ))
            .unwrap();
        let mut trading = synthetic_core(
            gen_zero_nanocore::DOMAIN_TRADING,
            "trading operator core",
            prototype,
            4,
            0.8,
            &["alpha", "beta", "gamma"],
        );
        // Make the second expert's prediction deliberately distinct so the
        // regression fails if the service silently runs only the first core.
        trading.projection_weights.fill(0.0);
        fleet.register_core(trading).unwrap();
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_nano_fleet(fleet)
    }

    /// B03 through the Ask handler: a permuted candidate list yields the same
    /// choice and bit-identical per-name probabilities.
    #[tokio::test]
    async fn operator_nanocore_ask_is_permutation_equivariant() {
        let engine = operator_engine_with_two_cores();
        let state: Vec<f32> = (0..128)
            .map(|j| 1.0 + 0.05 * ((j as f32) * 0.21).sin())
            .collect();
        let ask = |names: Vec<&str>| {
            let args = json!({"verb": "ask", "context": "pick an action", "candidates": names,
                "nanocore_domains": [0, 5], "nanocore_state": state});
            let engine = &engine;
            async move { engine.execute(&args).await.unwrap() }
        };
        let by_name = |out: &ZeroToolOutcome| {
            out.meta["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c["name"].as_str().unwrap().to_string(),
                        c["probability"].clone(),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        let forward = ask(vec!["alpha", "beta", "gamma"]).await;
        let permuted = ask(vec!["gamma", "alpha", "beta"]).await;
        assert!(forward.meta["candidates"].is_array(), "{forward:?}");
        assert_eq!(
            forward.meta["chosen_action"],
            permuted.meta["chosen_action"]
        );
        assert_eq!(forward.meta["action_id"], permuted.meta["action_id"]);
        assert_eq!(forward.meta["entropy"], permuted.meta["entropy"]);
        let probs = by_name(&forward);
        assert_eq!(probs, by_name(&permuted));
        assert!(
            probs.values().any(|p| p != &probs["alpha"]),
            "vacuous: {probs:?}"
        );

        let unknown = ask(vec!["alpha", "delta"]).await;
        assert!(unknown.is_error, "{unknown:?}");
        assert!(unknown.content[0].text.contains("delta"), "{unknown:?}");
    }

    #[tokio::test]
    async fn operator_nanocore_mixture_fuses_selected_domains_and_keeps_single_compatibility() {
        let engine = operator_engine_with_two_cores();
        let state = vec![1.0_f32; 128];
        let mixture = engine
            .execute(&json!({
                "verb": "ask",
                "context": "pick an action",
                "candidates": ["alpha", "beta"],
                "nanocore_domains": [5, 0],
                "nanocore_state": state
            }))
            .await
            .unwrap();
        assert_eq!(mixture.meta["domains"], json!([5, 0]), "{mixture:?}");
        assert_eq!(mixture.meta["core_count"], 2, "{mixture:?}");
        let weights = mixture.meta["weights"].as_array().unwrap();
        assert_eq!(weights.len(), 2, "{mixture:?}");
        let weight_sum: f64 = weights
            .iter()
            .map(|weight| weight["weight"].as_f64().unwrap())
            .sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "{mixture:?}");
        assert!(mixture.meta["confidence"].is_number(), "{mixture:?}");

        let general = engine
            .nano_fleet
            .get_core(gen_zero_nanocore::DOMAIN_GENERAL)
            .unwrap();
        let trading = engine
            .nano_fleet
            .get_core(gen_zero_nanocore::DOMAIN_TRADING)
            .unwrap();
        let names = ["alpha", "beta"];
        let ids = [action_id("alpha"), action_id("beta")];
        let frame = LocalActionFrame::new(&names, &ids).unwrap();
        let expected = MoVFusionEngine::new(WatchdogConfig::default())
            .fuse(
                &[trading.as_ref(), general.as_ref()],
                &CompressedLatent { values: [1.0; 128] },
                &frame,
            )
            .unwrap();
        let actual_probabilities = mixture.meta["probabilities"].as_array().unwrap();
        assert_eq!(actual_probabilities.len(), expected.probabilities.len());
        for (actual, expected) in actual_probabilities
            .iter()
            .zip(expected.probabilities.iter())
        {
            assert!((actual.as_f64().unwrap() - f64::from(*expected)).abs() < 1e-6);
        }
        assert!(
            (mixture.meta["confidence"].as_f64().unwrap()
                - f64::from(expected.composite_confidence))
            .abs()
                < 1e-6
        );

        let single = engine
            .execute(&json!({
                "verb": "ask",
                "context": "pick an action",
                "candidates": ["alpha", "beta"],
                "nanocore_domain": 0,
                "nanocore_state": vec![1.0_f32; 128]
            }))
            .await
            .unwrap();
        assert_eq!(single.meta["domain"], 0, "{single:?}");
        assert_eq!(single.meta["domains"], json!([0]), "{single:?}");
        assert_eq!(single.meta["core_count"], 1, "{single:?}");
        assert_ne!(mixture.meta["probabilities"], single.meta["probabilities"]);
    }

    #[tokio::test]
    async fn operator_nanocore_mixture_reports_missing_selected_domain() {
        let engine = operator_engine_with_two_cores();
        let out = engine
            .execute(&json!({
                "verb": "ask",
                "context": "pick an action",
                "candidates": ["alpha", "beta"],
                "nanocore_domains": [0, 99],
                "nanocore_state": vec![1.0_f32; 128]
            }))
            .await
            .unwrap();
        assert!(out.is_error, "{out:?}");
        assert!(
            out.content[0]
                .text
                .contains("micro-core unavailable (domain 99)"),
            "{out:?}"
        );
        assert_eq!(out.meta["domains"], json!([0, 99]), "{out:?}");
    }

    #[tokio::test]
    async fn operator_nanocore_selector_errors_are_explicit() {
        let engine = operator_engine_with_two_cores();
        for (domains, singular, expected_detail) in [
            (json!([0, 0]), Value::Null, "repeats domain 0"),
            (json!([]), Value::Null, "at least one domain ID"),
            (
                json!([0]),
                json!(0),
                "either nanocore_domain or nanocore_domains",
            ),
        ] {
            let mut request = json!({
                "verb": "ask",
                "context": "pick an action",
                "candidates": ["alpha", "beta"],
                "nanocore_domains": domains,
                "nanocore_state": vec![1.0_f32; 128]
            });
            if !singular.is_null() {
                request["nanocore_domain"] = singular;
            }
            let out = engine.execute(&request).await.unwrap();
            assert!(out.is_error, "{out:?}");
            assert!(out.content[0].text.contains(expected_detail), "{out:?}");
        }
    }

    #[test]
    fn operator_nanocore_path_loader_reads_multiple_files_without_environment_mutation() {
        let root = std::env::temp_dir().join(format!(
            "gen-zero-nanocore-test-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&root).unwrap();
        let first_path = root.join("general.json");
        let second_path = root.join("trading.json");
        let prototype = CompressedLatent { values: [1.0; 128] };
        let first = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "general file core",
            prototype.clone(),
            4,
            0.9,
            &["alpha", "beta", "gamma"],
        );
        let second = synthetic_core(
            gen_zero_nanocore::DOMAIN_TRADING,
            "trading file core",
            prototype,
            4,
            0.8,
            &["alpha", "beta", "gamma"],
        );
        std::fs::write(&first_path, serde_json::to_vec(&first).unwrap()).unwrap();
        std::fs::write(&second_path, serde_json::to_vec(&second).unwrap()).unwrap();

        let fleet = NanoCoreFleetScheduler::new(gen_zero_nanocore::DEFAULT_RAM_BUDGET_BYTES);
        load_nanocores_from_paths(
            &fleet,
            vec![first_path.clone(), second_path.clone()],
            "test Nanocore paths",
        );
        assert_eq!(
            fleet
                .get_core(gen_zero_nanocore::DOMAIN_GENERAL)
                .unwrap()
                .name,
            first.name
        );
        assert_eq!(
            fleet
                .get_core(gen_zero_nanocore::DOMAIN_TRADING)
                .unwrap()
                .name,
            second.name
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nanocore_path_list_uses_platform_separator() {
        let paths = vec![PathBuf::from("vision.json"), PathBuf::from("dynamics.json")];
        let joined = std::env::join_paths(&paths).unwrap();
        assert_eq!(split_nanocore_paths(&joined), paths);
    }

    #[test]
    #[should_panic(expected = "duplicate operator Nanocore domain")]
    fn operator_nanocore_path_loader_rejects_duplicate_domains() {
        let root = std::env::temp_dir().join(format!(
            "gen-zero-nanocore-duplicate-test-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&root).unwrap();
        let first_path = root.join("first.json");
        let second_path = root.join("second.json");
        let prototype = CompressedLatent { values: [1.0; 128] };
        let first = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "first duplicate",
            prototype.clone(),
            4,
            0.9,
            &["alpha", "beta", "gamma"],
        );
        let second = synthetic_core(
            gen_zero_nanocore::DOMAIN_GENERAL,
            "second duplicate",
            prototype,
            4,
            0.8,
            &["alpha", "beta", "gamma"],
        );
        std::fs::write(&first_path, serde_json::to_vec(&first).unwrap()).unwrap();
        std::fs::write(&second_path, serde_json::to_vec(&second).unwrap()).unwrap();
        let fleet = NanoCoreFleetScheduler::new(gen_zero_nanocore::DEFAULT_RAM_BUDGET_BYTES);
        load_nanocores_from_paths(
            &fleet,
            vec![first_path, second_path],
            "test duplicate Nanocore paths",
        );
    }
}
