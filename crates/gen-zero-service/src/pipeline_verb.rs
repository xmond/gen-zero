//! `pipeline` verb: the planner's [`ProductionPipeline`] behind `zero`, MCP, HTTP
//! (`POST /v1/pipeline/{op}`) and the CLI.
//!
//! Request block: `{"op": "simulate" | "what_if" | "audit_action" | "decide", ...}`.
//! `state` is exactly 1024 finite numbers; nothing is padded or truncated. Unknown
//! keys are refused, so a misspelt field never falls back to a default unseen.

use crate::cognitive::Rejection;
use crate::graph_verb::graph_rejection;
use gen_zero_core::{ActionId, CoreError, FullLatent, NormalizedEntropy, WorldModelDynamics};
use gen_zero_gate::PolicyGate;
use gen_zero_lod::{LodError, LodGraph, ReflectionRevocation};
use gen_zero_planner::{
    AuditReport, DecideMode, DecideRequest, Decision, GraphContext, PlannerConfig, PlannerError,
    ProductionPipeline, PrunedAction, Rollout, WhatIfReport, DEFAULT_WARN_RISK,
};
use serde_json::{json, Map, Value};
use std::sync::Arc;

const STAGE: &str = "pipeline";
/// Horizon used by `what_if`, `audit_action` and `decide` trajectories when the
/// request names none. Reported back in every response.
pub const DEFAULT_PIPELINE_HORIZON: usize = 5;

pub const PIPELINE_OPS: [&str; 4] = ["simulate", "what_if", "audit_action", "decide"];

/// Run one pipeline request. `Ok` holds a one-line summary and the result object.
///
/// `decide` alone may carry a `planner_config` object overriding the MCTS /
/// MPC-CEM / A* / router hyperparameters (see [`PlannerConfig`]) for this call.
/// Every other op refuses the field outright: `simulate`, `what_if` and
/// `audit_action` never call an engine (they replay a fixed plan or a one-step
/// greedy policy), so `planner_config` would have no effect there — accepting
/// and silently dropping it would be exactly the fake-knob failure mode this
/// field exists to avoid.
pub fn execute_pipeline(
    world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
    gate: Arc<PolicyGate>,
    graph: Arc<LodGraph>,
    block: &Value,
) -> Result<(String, Value), Rejection> {
    let obj = block
        .as_object()
        .ok_or_else(|| invalid("pipeline must be an object with an `op` field"))?;
    let op = obj
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("pipeline.op must be one of {PIPELINE_OPS:?}")))?;
    let pipeline = match op {
        "decide" => {
            let config = parse_planner_config(obj)?;
            let pipeline = ProductionPipeline::new_with_config(world_model, gate, config)
                .map_err(planner_rejection)?
                .with_graph(Arc::clone(&graph));
            match obj.get("astar_goal") {
                None => pipeline,
                Some(value) => {
                    let goal = value
                        .as_object()
                        .ok_or_else(|| invalid("astar_goal must contain state and tolerance"))?;
                    allow_keys(goal, &["state", "tolerance"])?;
                    let target = parse_state(goal)?;
                    let tolerance = goal
                        .get("tolerance")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid("astar_goal.tolerance is required"))?
                        as f32;
                    if !tolerance.is_finite()
                        || tolerance < 0.0
                        || target.as_slice().iter().any(|x| !x.is_finite())
                    {
                        return Err(invalid(
                            "astar_goal must be finite with nonnegative tolerance",
                        ));
                    }
                    pipeline.with_astar_goal(gen_zero_planner::AStarGoal::WithinDistance {
                        target: Box::new(target),
                        tolerance,
                    })
                }
            }
        }
        _ => ProductionPipeline::new(world_model, gate).with_graph(Arc::clone(&graph)),
    };
    let pipeline = &pipeline;
    let auto_reflect = match obj.get("auto_reflect") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid("pipeline.auto_reflect must be a boolean"))?,
    };
    match op {
        "simulate" => {
            allow_keys(obj, &["op", "state", "actions", "horizon", "auto_reflect"])?;
            let state = parse_state(obj)?;
            let actions = parse_actions(obj, "actions")?;
            let horizon = parse_opt_usize(obj, "horizon")?;
            let roll = pipeline
                .simulate(&state, &actions, horizon)
                .map_err(planner_rejection)?;
            let summary = format!(
                "simulate: {} step(s), survival_horizon {}, terminated_early {}",
                roll.steps_simulated(),
                roll.survival_horizon,
                roll.terminated_early
            );
            Ok((
                summary,
                json!({"op": op, "simulation": rollout_json(&roll),
                    "graph_reflection": reflect_rollouts(&graph, auto_reflect, op, &state, &[&roll], &[])?}),
            ))
        }
        "what_if" => {
            allow_keys(
                obj,
                &["op", "state", "candidates", "horizon", "auto_reflect"],
            )?;
            let state = parse_state(obj)?;
            let candidates = parse_actions(obj, "candidates")?;
            let horizon = parse_opt_usize(obj, "horizon")?.unwrap_or(DEFAULT_PIPELINE_HORIZON);
            let report = pipeline
                .what_if(&state, &candidates, horizon)
                .map_err(planner_rejection)?;
            let summary = format!(
                "what_if: best_candidate {}, traps_detected {:?}",
                report.best_candidate.0,
                ids(&report.traps_detected)
            );
            Ok((
                summary,
                json!({"op": op, "what_if": what_if_json(&report),
                "graph_reflection": reflect_rollouts(&graph, auto_reflect, op, &state,
                    &report.outcomes.iter().map(|o| &o.rollout).collect::<Vec<_>>(), &report.gate_blocked)?}),
            ))
        }
        "audit_action" => {
            allow_keys(
                obj,
                &[
                    "op",
                    "state",
                    "action",
                    "horizon",
                    "continuation_actions",
                    "warn_risk",
                    "auto_reflect",
                ],
            )?;
            let state = parse_state(obj)?;
            let action = parse_action_id(obj.get("action"), "action")?;
            let horizon = parse_opt_usize(obj, "horizon")?.unwrap_or(DEFAULT_PIPELINE_HORIZON);
            let continuation = match obj.get("continuation_actions") {
                None => None,
                Some(_) => Some(parse_actions(obj, "continuation_actions")?),
            };
            let warn_risk = match obj.get("warn_risk") {
                None => DEFAULT_WARN_RISK,
                Some(v) => v
                    .as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| invalid("pipeline.warn_risk must be a number"))?,
            };
            let report = pipeline
                .audit_action(&state, action, horizon, continuation.as_deref(), warn_risk)
                .map_err(planner_rejection)?;
            let summary = format!(
                "audit_action: {} (risk_score {:.4})",
                report.verdict.as_str(),
                report.risk_score
            );
            Ok((
                summary,
                json!({"op": op, "audit": audit_json(&report, horizon, warn_risk),
                    "graph_reflection": reflect_rollouts(&graph, auto_reflect, op, &state,
                        &report.trajectory.iter().collect::<Vec<_>>(),
                        &if report.gate_tier == gen_zero_gate::PolicyTier::Tier3HardStop {
                            vec![PrunedAction { action, tier: report.gate_tier, violated_rules: vec![], reason: report.reasons.join("; ") }]
                        } else { vec![] })?}),
            ))
        }
        "decide" => {
            allow_keys(
                obj,
                &[
                    "op",
                    "state",
                    "candidates",
                    "mode",
                    "entropy",
                    "active_context",
                    "return_trajectory",
                    "horizon",
                    "planner_config",
                    "budget_ms",
                    "astar_goal",
                ],
            )?;
            let state = parse_state(obj)?;
            let candidates = parse_actions(obj, "candidates")?;
            let mode: DecideMode = obj
                .get("mode")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid("pipeline.mode is required: auto, mcts, mpc_cem, astar, manifold_gflownet, cfr_nash or reflex")
                })?
                .parse()
                .map_err(planner_rejection)?;
            let entropy = obj
                .get("entropy")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("pipeline.entropy is required: a number in [0, 1]"))?
                as f32;
            let return_trajectory = match obj.get("return_trajectory") {
                None => false,
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| invalid("pipeline.return_trajectory must be a boolean"))?,
            };
            let horizon = parse_opt_usize(obj, "horizon")?.unwrap_or(DEFAULT_PIPELINE_HORIZON);
            let decision = pipeline
                .decide(&DecideRequest {
                    active_context: match obj.get("active_context") {
                        None => Vec::new(),
                        Some(_) => parse_actions(obj, "active_context")?,
                    },
                    deadline: None,
                    budget_ms: obj
                        .get("budget_ms")
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| invalid("pipeline.budget_ms must be a number"))
                        })
                        .transpose()?,
                    state: &state,
                    candidates: &candidates,
                    mode,
                    entropy: NormalizedEntropy(entropy),
                    return_trajectory,
                    horizon,
                })
                .map_err(planner_rejection)?;
            let summary = format!(
                "decide[{}]: action {} via {}",
                decision.mode.as_str(),
                decision.action.0,
                decision.engine
            );
            Ok((
                summary,
                json!({"op": op, "decision": decision_json(&decision)}),
            ))
        }
        other => Err(invalid(format!(
            "unknown pipeline.op {other:?}; expected one of {PIPELINE_OPS:?}"
        ))),
    }
}

/// Deposits only the actual hazardous transition, not an innocent first action
/// whose continuation later failed. The full state bits and policy/source are
/// recorded so deduplication never conflates equal-norm but different states.
fn reflect_rollouts(
    graph: &LodGraph,
    enabled: bool,
    op: &str,
    initial: &FullLatent,
    rolls: &[&Rollout],
    blocked: &[PrunedAction],
) -> Result<Value, Rejection> {
    if !enabled {
        return Ok(json!({"enabled": false, "observations": []}));
    }
    let mut observations = Vec::new();
    for roll in rolls {
        if let Some(step) = roll.steps.iter().find(|s| s.hazard) {
            let before = if step.step_idx == 1 {
                initial
            } else {
                &roll.steps[step.step_idx - 2].state
            };
            let payload = json!({"kind": "model_terminal_observation", "op": op,
                "action": step.action.0, "step": step.step_idx,
                "state_before": before.as_slice(), "state_after": step.state.as_slice(),
                "reward": step.reward, "continuation_policy": roll.continuation_policy,
                "safety_sources": roll.safety_sources, "calibrated": roll.safety_calibrated,
                "claim": "model diagnostic; not proof of real-world causality"})
            .to_string();
            observations.push(deposit_reflection(graph, step.action, &payload)?);
        }
    }
    for action in blocked {
        let payload = json!({"kind": "policy_hard_stop", "op": op,
            "action": action.action.0, "step": 0, "state_before": initial.as_slice(),
            "reason": action.reason, "rules": action.violated_rules})
        .to_string();
        observations.push(deposit_reflection(graph, action.action, &payload)?);
    }
    Ok(json!({"enabled": true, "observations": observations}))
}

fn deposit_reflection(
    graph: &LodGraph,
    action: ActionId,
    payload: &str,
) -> Result<Value, Rejection> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| invalid(format!("reflection clock error: {e}")))?;
    let timestamp =
        u64::try_from(now.as_nanos()).map_err(|_| invalid("reflection timestamp overflow"))?;
    let reflection = graph
        .reflect_failure(action.0, payload, timestamp)
        .map_err(|e| reflection_rejection(action, e))?;
    let report = &reflection.evolution;
    for block in &report.adapted_blocks {
        tracing::warn!(
            action = action.0,
            nodes = ?block.nodes,
            requested_gamma = block.requested_gamma,
            applied_gamma = block.applied_gamma,
            requested_contraction = block.requested_contraction,
            contraction = block.contraction,
            "reflection met a non-contractive cycle; its internal falsifier gain was lowered"
        );
    }
    let revocation = match reflection.revocation {
        ReflectionRevocation::Evolution => "evolution",
        ReflectionRevocation::Quarantine => {
            tracing::warn!(
                action = action.0,
                confidence = reflection.target_confidence,
                "reflection evolution left the action above the revocation threshold; quarantined"
            );
            "quarantine"
        }
    };
    let adapted: Vec<Value> = report
        .adapted_blocks
        .iter()
        .map(|b| {
            json!({"nodes": b.nodes, "requested_gamma": b.requested_gamma,
                "applied_gamma": b.applied_gamma,
                "requested_contraction": b.requested_contraction, "contraction": b.contraction})
        })
        .collect();
    let (evidence, target) = (reflection.evidence, reflection.target);
    Ok(
        json!({"evidence_node_id": evidence, "action_node_id": target,
        "falsification_edge": {"source": evidence, "target": target, "type": "Falsifies"},
        "revoked_entities": [action.0], "newly_revoked_entities": report.revoked_entities,
        "revocation": revocation, "action_confidence": reflection.target_confidence,
        "evolution": {"converged": true, "beta": report.beta, "gamma": report.gamma,
            "tolerance": report.tolerance, "theta_lo": report.theta_lo, "theta_hi": report.theta_hi,
            "scc_count": report.scc_count, "cyclic_scc_count": report.cyclic_scc_count,
            "trivial_scc_count": report.trivial_scc_count, "max_scc_size": report.max_scc_size,
            "contraction": report.contraction, "adapted_blocks": adapted,
            "iterations": report.iterations, "residual": report.residual,
            "error_bound": report.error_bound, "node_updates": report.node_updates,
            "falsification_edges": report.falsification_edges}}),
    )
}

/// A failed reflection quarantines the action and is refused with the status of
/// its cause: an engine fault (CSR, checkpoint) is 500, a refused evolution 422,
/// a conflict with what the graph holds (an axiom, a colliding or incomplete
/// earlier observation) 409.
fn reflection_rejection(action: ActionId, e: LodError) -> Rejection {
    let conflict = matches!(e, LodError::InvalidQuery(_));
    let mut rejection = graph_rejection(e);
    if conflict {
        rejection.http_status = 409;
    }
    Rejection {
        code: "GraphReflectionFailed".into(),
        stage: STAGE.into(),
        detail: format!(
            "reflection failed; action {} quarantined: {}",
            action.0, rejection.detail
        ),
        http_status: rejection.http_status,
    }
}

fn invalid(detail: impl Into<String>) -> Rejection {
    Rejection::invalid(STAGE, detail)
}

/// Input faults are 400. A request that is well formed but has no admissible answer
/// (all actions gated, no safety estimate, the model diverged) is 422. A world-model
/// failure is 500.
fn planner_rejection(e: PlannerError) -> Rejection {
    let (code, status) = match &e {
        PlannerError::InvalidInput(_)
        | PlannerError::MissingSearchGoal
        | PlannerError::InvalidHorizon { .. }
        | PlannerError::UnknownMode(_) => ("InvalidParams", 400),
        PlannerError::DivergentState(_) => ("DivergentState", 422),
        PlannerError::NoFeasibleAction => ("NoFeasibleAction", 422),
        PlannerError::MissingSafetyEstimate { .. } => ("MissingSafetyEstimate", 422),
        PlannerError::InvalidSafetyEstimate(_) => ("InvalidSafetyEstimate", 500),
        PlannerError::Core(_) => ("WorldModelError", 500),
        _ => ("PlannerError", 500),
    };
    Rejection {
        code: code.to_string(),
        stage: STAGE.to_string(),
        detail: e.to_string(),
        http_status: status,
    }
}

fn allow_keys(obj: &Map<String, Value>, allowed: &[&str]) -> Result<(), Rejection> {
    match obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(invalid(format!(
            "unknown field pipeline.{k} for this op; allowed: {allowed:?}"
        ))),
        None => Ok(()),
    }
}

fn parse_state(obj: &Map<String, Value>) -> Result<FullLatent, Rejection> {
    let arr = obj
        .get("state")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("pipeline.state must be an array of 1024 numbers"))?;
    let mut state = FullLatent::zeros();
    if arr.len() != state.dim() {
        return Err(invalid(format!(
            "pipeline.state has {} values, expected {}",
            arr.len(),
            state.dim()
        )));
    }
    for (i, (slot, v)) in state.as_mut_slice().iter_mut().zip(arr).enumerate() {
        *slot = v
            .as_f64()
            .ok_or_else(|| invalid(format!("pipeline.state[{i}] is not a number")))?
            as f32;
    }
    Ok(state)
}

fn parse_action_id(v: Option<&Value>, field: &str) -> Result<ActionId, Rejection> {
    v.and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .map(ActionId)
        .ok_or_else(|| invalid(format!("pipeline.{field} must be an action id (u32)")))
}

fn parse_actions(obj: &Map<String, Value>, field: &str) -> Result<Vec<ActionId>, Rejection> {
    obj.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("pipeline.{field} must be an array of action ids")))?
        .iter()
        .map(|v| parse_action_id(Some(v), field))
        .collect()
}

/// `pipeline.planner_config`: absent means every engine keeps its shipped
/// default (see `PlannerConfig::default`). A malformed object (unknown field,
/// wrong type) or an out-of-range value is refused rather than partially
/// applied or silently ignored.
fn parse_planner_config(obj: &Map<String, Value>) -> Result<PlannerConfig, Rejection> {
    let config = match obj.get("planner_config") {
        None => PlannerConfig::default(),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| invalid(format!("pipeline.planner_config: {e}")))?,
    };
    config.validate().map_err(planner_rejection)?;
    Ok(config)
}

fn parse_opt_usize(obj: &Map<String, Value>, field: &str) -> Result<Option<usize>, Rejection> {
    match obj.get(field) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| invalid(format!("pipeline.{field} must be a non-negative integer"))),
    }
}

fn ids(actions: &[gen_zero_core::ActionId]) -> Vec<u32> {
    actions.iter().map(|a| a.0).collect()
}

fn pruned_json(pruned: &[PrunedAction]) -> Value {
    pruned
        .iter()
        .map(|p| {
            json!({
                "action": p.action.0,
                "tier": p.tier,
                "violated_rules": p.violated_rules,
                "reason": p.reason,
            })
        })
        .collect()
}

pub fn rollout_json(roll: &Rollout) -> Value {
    let steps: Vec<Value> = roll
        .steps
        .iter()
        .map(|s| {
            json!({
                "step_idx": s.step_idx,
                "action": s.action.0,
                "state": s.state.as_slice(),
                "reward": s.reward,
                "safe_prob": s.safe_prob,
                "done": s.done,
                "hazard": s.hazard,
                "gate_tier": s.gate_tier,
            })
        })
        .collect();
    json!({
        "trajectory": steps,
        "steps_simulated": roll.steps_simulated(),
        "survival_horizon": roll.survival_horizon,
        "cumulative_return": roll.cumulative_return,
        "terminated_early": roll.terminated_early,
        "termination_step": roll.termination_step,
        "is_safe": roll.is_safe(),
        "first_hazard_step": roll.first_hazard_step,
        "min_safe_prob": roll.min_safe_prob,
        "safety_coverage": roll.safety_coverage,
        "safety_calibrated": roll.safety_calibrated,
        "safety_sources": roll.safety_sources,
        "continuation_policy": roll.continuation_policy,
        "final_state": roll.final_state.as_slice(),
    })
}

fn what_if_json(r: &WhatIfReport) -> Value {
    let outcomes: Vec<Value> = r
        .outcomes
        .iter()
        .map(|o| {
            let mut v = rollout_json(&o.rollout);
            v["action"] = json!(o.action.0);
            v
        })
        .collect();
    json!({
        "candidate_outcomes": outcomes,
        "best_candidate": r.best_candidate.0,
        "safety_ranking": ids(&r.safety_ranking),
        "traps_detected": ids(&r.traps_detected),
        "all_candidates_trapped": r.all_candidates_trapped,
        "gate_blocked": pruned_json(&r.gate_blocked),
        "horizon": r.horizon,
    })
}

fn audit_json(r: &AuditReport, horizon: usize, warn_risk: f32) -> Value {
    json!({
        "action": r.action.0,
        "verdict": r.verdict.as_str(),
        "risk_score": r.risk_score,
        // Null for a gate hard stop (no rollout); false for the default model's margin.
        "safety_calibrated": r.trajectory.as_ref().and_then(|t| t.safety_calibrated),
        "reasons": r.reasons,
        "gate_tier": r.gate_tier,
        "is_safe": r.is_safe,
        "survival_horizon": r.survival_horizon,
        "first_hazard_step": r.first_hazard_step,
        "continuation_pruned": pruned_json(&r.continuation_pruned),
        "horizon": horizon,
        "warn_risk": warn_risk,
        "trajectory": r.trajectory.as_ref().map(rollout_json),
    })
}

fn decision_json(d: &Decision) -> Value {
    json!({
        "timed_out": d.timed_out,
        "action": d.action.0,
        "entropy": d.entropy.0,
        "mode": d.mode.as_str(),
        "engine": d.engine,
        "hazard_detected": d.hazard_detected,
        "hazardous_actions": ids(&d.hazardous_actions),
        "routing_tier": d.routing_tier.map(|t| format!("{t:?}")),
        "gate_tier": d.gate_tier,
        "requires_confirmation": d.requires_confirmation,
        "feasible": ids(&d.feasible),
        "pruned": pruned_json(&d.pruned),
        "trajectory": d.trajectory.as_ref().map(rollout_json),
        "graph_context": graph_context_json(&d.graph_context),
    })
}

/// PPR context plus the hierarchical facts consumed by graph prior gating.
fn graph_context_json(c: &GraphContext) -> Value {
    match c {
        GraphContext::Diffused {
            seed_entities,
            facts,
            iterations,
            converged,
        } => json!({
            "status": "diffused",
            "advisory": true,
            "used_in_choice": false,
            "hierarchical_prior_used_in_gate": true,
            "method": "personalized_pagerank",
            "seed_entities": seed_entities,
            "iterations": iterations,
            "converged": converged,
            "facts": facts.iter().map(|f| json!({
                "entity_id": f.entity_id,
                "label": f.label,
                "status": f.status,
                "score": f.score,
                "band": f.band,
                "confidence": f.confidence,
                "used_in_gate": f.hierarchical_prior,
            })).collect::<Vec<_>>(),
        }),
        GraphContext::Unavailable { reason } => json!({
            "status": "unavailable",
            "advisory": true,
            "used_in_choice": false,
            "hierarchical_prior_used_in_gate": true,
            "reason": reason,
        }),
    }
}
