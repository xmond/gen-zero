//! The production subcommands of the `gen-zero` binary: `decide`, `simulate`,
//! `what-if` and `audit`. Each test runs the real binary and checks the exit
//! code and the JSON it prints.
//!
//! `simulate`, `what-if` and `audit` run on the untrained latent prior, and
//! every answer says so. These tests prove wiring, validation and fail-closed
//! exits, not the quality of the prior.
//!
//! `decide` needs a request-risk verdict before it may exit 0. Where a test
//! needs one it starts a local stub scorer that answers with canned numbers,
//! so the test checks the CLI -> engine -> bridge -> gate -> exit chain, not
//! a real scorer.

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::process::{Command, Output};

const DIM: usize = 1024;

fn latent(fill: f64) -> String {
    json!(vec![fill; DIM]).to_string()
}

/// Canned scorer on a background thread: every text is low risk, the first
/// candidate wins with 0.9 (confident enough to stay under the gate's entropy
/// escalation). Returns its endpoint.
fn stub_scorer() -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let app = Router::new()
                .route(
                    "/v1/semantic_risk",
                    post(|Json(_): Json<Value>| async {
                        Json(json!({
                            "p_dangerous": 0.01, "log_odds": -4.6, "windows": 1,
                            "thresholds": {"escalate": 0.5, "hard_stop": 0.9},
                            "classifier": {"name": "test-stub"}, "forward_ms": 0.0,
                        }))
                    }),
                )
                .route(
                    "/v1/semantic_ask",
                    post(|Json(req): Json<Value>| async move {
                        let names: Vec<String> = req["candidates"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|c| c.as_str().unwrap().to_string())
                            .collect();
                        let n = names.len();
                        let prob = |i: usize| match (n, i) {
                            (1, _) => 1.0,
                            (_, 0) => 0.9,
                            _ => 0.1 / (n - 1) as f64,
                        };
                        let scores: Vec<Value> = names
                            .iter()
                            .enumerate()
                            .map(|(i, name)| {
                                json!({
                                    "name": name, "log_likelihood": -1.0,
                                    "baseline_log_likelihood": -1.0, "pmi": 0.0,
                                    "probability": prob(i),
                                })
                            })
                            .collect();
                        Json(json!({
                            "chosen": names[0], "chosen_index": 0, "candidates": scores,
                            "entropy": 0.5, "scorer": {"name": "test-stub"},
                            "embedding_dim": 0, "timing_ms": 0.0,
                        }))
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    format!("http://{}", rx.recv().unwrap())
}

/// Run `gen-zero <args>`. `endpoint` is the scorer; `None` turns the bridge off.
fn gen_zero(args: &[&str], endpoint: Option<&str>) -> (Option<i32>, Value, String) {
    let out: Output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(args)
        .env("GENZERO_PYTHON_ENDPOINT", endpoint.unwrap_or("off"))
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    eprintln!(
        "exit={:?}\nstdout(head)={}\nstderr={stderr}",
        out.status.code(),
        stdout.chars().take(600).collect::<String>()
    );
    let json = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    (out.status.code(), json, stderr)
}

// ----------------------------------------------------------------- simulate

#[test]
fn simulate_exits_zero_with_a_structured_untrained_rollout() {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(&["simulate", "--state", &state, "--actions", "a,b,c"], None);
    assert_eq!(code, Some(0));
    assert_eq!(out["verb"], "simulate");
    assert_eq!(out["is_error"], false);
    assert_eq!(
        out["meta"]["provenance"],
        "latent_residual_dynamics_untrained"
    );
    assert_eq!(out["meta"]["simulation"]["steps_simulated"], 3);
    assert_eq!(
        out["meta"]["simulation"]["final_state"]
            .as_array()
            .unwrap()
            .len(),
        DIM
    );
}

#[test]
fn simulate_takes_json_actions_a_state_file_and_a_shorter_horizon() {
    let path = std::env::temp_dir().join(format!("gen_zero_state_{}.json", std::process::id()));
    std::fs::write(&path, latent(0.0)).unwrap();
    let file_arg = format!("@{}", path.display());
    let (code, out, _) = gen_zero(
        &[
            "simulate",
            "--state",
            &file_arg,
            "--actions",
            r#"["a","b","c"]"#,
            "--horizon",
            "2",
        ],
        None,
    );
    std::fs::remove_file(&path).ok();
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["simulation"]["steps_simulated"], 2);
}

#[test]
fn simulate_refuses_a_short_state_instead_of_padding_it() {
    let (code, out, stderr) = gen_zero(
        &["simulate", "--state", "[0.0, 1.0]", "--actions", "a"],
        None,
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
    assert!(stderr.contains("refused"), "{stderr}");
}

#[test]
fn simulate_refuses_a_horizon_past_the_plan() {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            "a",
            "--horizon",
            "5",
        ],
        None,
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
}

#[test]
fn an_empty_action_name_stops_the_command_before_the_engine() {
    let state = latent(0.0);
    let (code, out, stderr) = gen_zero(&["simulate", "--state", &state, "--actions", "a,,b"], None);
    assert_ne!(code, Some(0));
    assert!(out.is_null(), "no outcome is printed");
    assert!(stderr.contains("empty name"), "{stderr}");
}

// ------------------------------------------------------------------ what-if

#[test]
fn what_if_exits_zero_and_ranks_every_candidate() {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "what-if",
            "--state",
            &state,
            "--candidates",
            "a,b,c",
            "--horizon",
            "3",
        ],
        None,
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["verb"], "what_if");
    assert_eq!(out["meta"]["ranking"].as_array().unwrap().len(), 3);
    assert_eq!(out["meta"]["advisory_only"], true);
    assert_eq!(
        out["meta"]["provenance"],
        "latent_residual_dynamics_untrained"
    );
}

#[test]
fn what_if_refuses_repeated_candidates() {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(&["what-if", "--state", &state, "--candidates", "a,a"], None);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
}

// -------------------------------------------------------------------- audit

#[test]
fn audit_exits_zero_and_never_approves() {
    let endpoint = stub_scorer();
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "audit",
            "--state",
            &state,
            "--action",
            "a",
            "--horizon",
            "3",
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["verb"], "audit");
    assert_eq!(out["meta"]["risk"]["assessed"], true);
    assert_eq!(out["meta"]["verdict"], "UNVERIFIED_UNTRAINED_DYNAMICS");
    assert_ne!(out["meta"]["verdict"], "APPROVED");
}

#[test]
fn audit_without_a_scorer_reports_unassessed_and_asks_for_confirmation() {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(&["audit", "--state", &state, "--action", "a"], None);
    // The audit ran, so the command succeeded; the verdict is the finding.
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["verdict"], "REQUIRES_CONFIRMATION");
    assert_eq!(out["meta"]["risk"]["assessed"], false);
    assert_eq!(out["meta"]["risk"]["fail_closed"], true);
}

#[test]
fn audit_reports_a_lethal_rollout() {
    let endpoint = stub_scorer();
    let state = latent(10.0);
    let (code, out, _) = gen_zero(
        &["audit", "--state", &state, "--action", "a"],
        Some(&endpoint),
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["verdict"], "REJECT_LETHAL");
    assert_eq!(out["meta"]["hazard_detected"], true);
    assert_eq!(out["meta"]["rollout"]["first_hazard_step"], 1);
}

#[test]
fn audit_takes_a_continuation_set() {
    let endpoint = stub_scorer();
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "audit",
            "--state",
            &state,
            "--action",
            "a",
            "--continuation",
            "a,b,c",
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(0));
    assert_eq!(
        out["meta"]["continuation_policy"],
        "greedy_one_step_over_continuation_actions"
    );
}

// ------------------------------------------------------------------- decide

#[test]
fn decide_exits_zero_in_auto_and_reflex_mode_with_a_risk_verdict() {
    let endpoint = stub_scorer();
    for mode in ["auto", "reflex"] {
        let (code, out, _) = gen_zero(
            &[
                "decide",
                "--context",
                "pick a move",
                "--candidates",
                "left,right",
                "--mode",
                mode,
            ],
            Some(&endpoint),
        );
        assert_eq!(code, Some(0), "{mode}");
        assert_eq!(out["verb"], "ask");
        assert_eq!(out["meta"]["engine"], "semantic_bridge");
        assert_eq!(out["meta"]["chosen_action"], "left");
        assert_eq!(out["meta"]["mode"]["resolved"], "semantic_ask");
    }
}

#[test]
fn decide_without_a_scorer_is_held_back_not_waved_through() {
    let (code, out, stderr) = gen_zero(
        &[
            "decide",
            "--context",
            "pick a move",
            "--candidates",
            "left,right",
        ],
        None,
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["is_error"], true);
    assert_eq!(out["meta"]["gate_status"], "requires_confirmation");
    assert!(stderr.contains("refused"), "{stderr}");
}

#[test]
fn decide_mcts_on_text_runs_the_semantic_lookahead() {
    let endpoint = stub_scorer();
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "back up before deleting",
            "--candidates",
            "delete,backup,wait",
            "--mode",
            "mcts",
            "--horizon",
            "2",
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["verb"], "imagine");
    assert_eq!(out["meta"]["planner"], "puct_mcts");
    assert_eq!(out["meta"]["mode"]["resolved"], "semantic_puct_lookahead");
}

/// One `decide --latent` run: (exit code, output).
fn decide_latent(mode: &str, endpoint: &str) -> (Option<i32>, Value) {
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "pick a move",
            "--candidates",
            "left,right,wait",
            "--mode",
            mode,
            "--latent",
            &state,
            "--return-trajectory",
            "--horizon",
            "3",
        ],
        Some(endpoint),
    );
    (code, out)
}

#[test]
fn decide_runs_the_latent_planners_and_returns_a_trajectory() {
    let endpoint = stub_scorer();
    // A* is absent here: its near-tied costs escalate, see the test below.
    let (code, out) = decide_latent("mpc_cem", &endpoint);
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["engine"], "latent_planner");
    assert_eq!(out["meta"]["tier"], "Proceed");
    let chosen = out["meta"]["chosen_action"].as_str().unwrap();
    assert!(["left", "right", "wait"].contains(&chosen), "{chosen}");
    assert_eq!(out["meta"]["trajectory"]["trajectory"][0]["action"], chosen);
    assert_eq!(out["meta"]["trajectory"]["steps_simulated"], 3);
}

#[test]
fn decide_holds_back_a_latent_planner_that_is_unsure() {
    // MCTS visits over the untrained prior are near uniform, and A* costs are
    // near-tied, so its Boltzmann entropy is high too. The gate escalates and
    // the command exits 1. The pick stays in the JSON.
    let endpoint = stub_scorer();
    for (mode, planner) in [("mcts", "MctsEngine"), ("astar", "AStarEngine")] {
        let (code, out) = decide_latent(mode, &endpoint);
        assert_eq!(code, Some(1), "{mode}");
        assert_eq!(out["meta"]["planner"], planner);
        assert_eq!(
            out["meta"]["gate_status"], "requires_confirmation",
            "{mode}"
        );
        assert!(out["meta"]["chosen_action"].is_string(), "{mode}");
    }
}

#[test]
fn decide_return_trajectory_on_text_says_it_is_unsupported() {
    let endpoint = stub_scorer();
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "pick a move",
            "--candidates",
            "left,right",
            "--return-trajectory",
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(0));
    assert!(out["meta"]["trajectory_status"]
        .as_str()
        .unwrap()
        .starts_with("unsupported_without_latent_state"));
    assert!(out["meta"].get("trajectory").is_none());
}

#[test]
fn decide_refuses_planner_modes_without_a_latent_state() {
    let endpoint = stub_scorer();
    for mode in ["mpc_cem", "astar"] {
        let (code, out, stderr) = gen_zero(
            &["decide", "--context", "x", "--mode", mode],
            Some(&endpoint),
        );
        assert_eq!(code, Some(1), "{mode}");
        assert_eq!(out["rejection"]["code"], "InvalidParams");
        assert!(stderr.contains("latent"), "{stderr}");
    }
}

#[test]
fn decide_refuses_a_latent_in_a_semantic_mode() {
    let endpoint = stub_scorer();
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "x",
            "--mode",
            "auto",
            "--latent",
            &state,
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
}

#[test]
fn decide_rejects_an_unknown_mode_at_the_command_line() {
    let (code, out, stderr) = gen_zero(&["decide", "--context", "x", "--mode", "bogus"], None);
    assert_ne!(code, Some(0));
    assert!(out.is_null());
    assert!(stderr.contains("bogus"), "{stderr}");
}

#[test]
fn decide_refuses_a_cognitive_block_in_a_latent_mode() {
    let endpoint = stub_scorer();
    let state = latent(0.0);
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "x",
            "--mode",
            "astar",
            "--latent",
            &state,
            "--cognitive",
            r#"{"state":[0.0]}"#,
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
    assert!(out["rejection"]["detail"]
        .as_str()
        .unwrap()
        .contains("cognitive"));
}

#[test]
fn decide_refuses_a_horizon_the_semantic_ask_would_drop() {
    let endpoint = stub_scorer();
    let (code, out, _) = gen_zero(
        &[
            "decide",
            "--context",
            "x",
            "--mode",
            "auto",
            "--horizon",
            "5",
        ],
        Some(&endpoint),
    );
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
    assert!(out["rejection"]["detail"]
        .as_str()
        .unwrap()
        .contains("horizon"));
}

// ------------------------------------------------------- --dynamics symplectic

fn phase_latent() -> String {
    let v: Vec<f64> = (0..DIM).map(|i| 0.1 * ((i as f64) * 0.013).cos()).collect();
    json!(v).to_string()
}

#[test]
fn simulate_with_symplectic_dynamics_conserves_energy_over_ten_steps() {
    let state = phase_latent();
    let plan = ["push"; 10].join(",");
    let (code, out, _) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            &plan,
            "--dynamics",
            "symplectic",
        ],
        None,
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["dynamics"], "symplectic");
    assert_eq!(
        out["meta"]["provenance"],
        "symplectic_hamiltonian_dynamics_untrained"
    );
    let sim = &out["meta"]["simulation"];
    assert_eq!(sim["steps_simulated"], 10);
    let run = &sim["energy_ledger"]["run"];
    let rel = run["relative_drift"].as_f64().unwrap();
    eprintln!(
        "cli H0={} H10={} rel_drift={rel:.3e}",
        run["initial"], run["final"]
    );
    assert!(rel < 1e-4, "relative drift {rel}");
    let phase = sim["phase_trajectory"].as_array().unwrap();
    assert_eq!(phase.len(), 10);
    assert_eq!(phase[0]["q"].as_array().unwrap().len(), DIM / 2);
    assert_eq!(phase[0]["p"].as_array().unwrap().len(), DIM / 2);
}

#[test]
fn what_if_with_symplectic_dynamics_reports_it() {
    let state = phase_latent();
    let (code, out, _) = gen_zero(
        &[
            "what-if",
            "--state",
            &state,
            "--candidates",
            "a,b",
            "--dynamics",
            "symplectic",
        ],
        None,
    );
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["dynamics"], "symplectic");
    assert!(out["meta"]["outcomes"][0]["energy_ledger"].is_object());
}

#[test]
fn unknown_dynamics_is_refused_by_the_cli() {
    let state = phase_latent();
    let (code, _, stderr) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            "a",
            "--dynamics",
            "conformal_x",
        ],
        None,
    );
    assert_ne!(code, Some(0));
    assert!(stderr.contains("conformal_x"), "{stderr}");
}

#[test]
fn simulate_with_contact_dynamics_reports_the_contact_ledger() {
    let state = phase_latent();
    let actions: Vec<&str> = vec!["push"; 10];
    let (code, out, stderr) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            &actions.join(","),
            "--dynamics",
            "contact",
            "--damping",
            "0.5",
        ],
        None,
    );
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(out["meta"]["dynamics"], "contact");
    assert_eq!(out["meta"]["damping"], 0.5);
    assert_eq!(
        out["meta"]["provenance"],
        "conformal_symplectic_contact_dynamics_untrained"
    );
    let ledger = &out["meta"]["simulation"]["energy_ledger"];
    assert_eq!(ledger["conservative"], false);
    assert!(ledger["conserved_quantity"].is_null());
    let factor = ledger["phase_volume_factor_per_pair"].as_f64().unwrap();
    assert!(
        (factor - (-2.0 * 0.5 * 0.01_f64).exp()).abs() < 1e-6,
        "{factor}"
    );
    assert!(ledger["run"]["drift"].as_f64().unwrap() < 0.0, "{ledger}");
    assert_eq!(
        out["meta"]["simulation"]["phase_trajectory"]
            .as_array()
            .unwrap()
            .len(),
        10
    );
}

#[test]
fn negative_damping_and_damping_without_contact_are_refused_by_the_cli() {
    let state = phase_latent();
    let (code, _, stderr) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            "a",
            "--dynamics",
            "contact",
            "--damping=-0.1",
        ],
        None,
    );
    assert_ne!(code, Some(0));
    assert!(stderr.contains("damping"), "{stderr}");
    let (code, _, stderr) = gen_zero(
        &[
            "simulate",
            "--state",
            &state,
            "--actions",
            "a",
            "--dynamics",
            "symplectic",
            "--damping",
            "0.1",
        ],
        None,
    );
    assert_ne!(code, Some(0));
    assert!(stderr.contains("damping"), "{stderr}");
}

// ------------------------------------------------------------ audit-ledger

/// `audit-ledger` prints the outcome JSON followed by a plain-text ledger
/// line, so unlike the other commands its stdout is not one JSON document;
/// this reads only the first.
fn first_json_value(text: &str) -> Value {
    serde_json::Deserializer::from_str(text)
        .into_iter::<Value>()
        .next()
        .expect("at least one JSON value on stdout")
        .expect("first stdout document is valid JSON")
}

fn audit_ledger(args: &[&str]) -> (Option<i32>, Value) {
    let out: Output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(args)
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    eprintln!(
        "exit={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status.code()
    );
    (out.status.code(), first_json_value(&stdout))
}

/// `--entropy-threshold-high` reaches the router the same way
/// `planner_config` does over HTTP: lowering it below the request's entropy
/// flips `auto` from `K2Pipeline` to `K3Committee`, proving the CLI flag is
/// wired to `ProductionPipeline::new_with_config`, not just parsed and dropped.
#[test]
fn audit_ledger_entropy_threshold_flag_changes_the_routing_tier() {
    use gen_zero_core::{ActionId, FullLatent, WorldModelDynamics};
    let (target, _, _) = gen_zero_worldmodel::LatentDynamicsWorldModel::default()
        .step(&FullLatent::zeros(), ActionId(1))
        .unwrap();

    let request = json!({
        "op": "decide",
        "astar_goal": {"state": target.as_slice(), "tolerance": 0.00001},
        "state": vec![0.0; DIM],
        "candidates": [1, 2, 3],
        "mode": "auto",
        "entropy": 0.5,
    })
    .to_string();

    let (code, out) = audit_ledger(&["audit-ledger", "--request", &request]);
    assert_eq!(code, Some(0));
    assert_eq!(
        out["meta"]["pipeline"]["decision"]["routing_tier"],
        "K2Pipeline"
    );

    let (code, out) = audit_ledger(&[
        "audit-ledger",
        "--request",
        &request,
        "--entropy-threshold-high",
        "0.3",
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(
        out["meta"]["pipeline"]["decision"]["routing_tier"],
        "K3Committee"
    );
}

/// `--c-puct` and the other engine-tuning flags are refused, not silently
/// dropped, once the value is out of range: the same fail-closed contract
/// `PlannerConfig::validate` enforces over HTTP.
#[test]
fn audit_ledger_refuses_an_invalid_c_puct() {
    let request = json!({
        "op": "decide",
        "state": vec![0.0; DIM],
        "candidates": [1, 2],
        "mode": "mcts",
        "entropy": 0.5,
    })
    .to_string();
    let (code, out) = audit_ledger(&["audit-ledger", "--request", &request, "--c-puct=-1.0"]);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams");
}

/// A CLI flag and a `planner_config` already embedded in `--request` disagree
/// about which value wins; refuse rather than pick one silently.
#[test]
fn audit_ledger_refuses_conflicting_config_sources() {
    let request = json!({
        "op": "decide",
        "state": vec![0.0; DIM],
        "candidates": [1, 2],
        "mode": "mcts",
        "entropy": 0.5,
        "planner_config": {"mcts_c_puct": 2.0},
    })
    .to_string();
    let out: Output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["audit-ledger", "--request", &request, "--c-puct", "1.0"])
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    assert_ne!(out.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("planner_config"), "{stderr}");
}

#[test]
fn audit_ledger_refuses_unknown_planner_config_fields() {
    for field in ["mcts_c_puct_typo", "virtual_loss", "arena_capacity"] {
        let request = json!({
            "op": "decide", "state": vec![0.0; DIM], "candidates": [1, 2],
            "mode": "mcts", "entropy": 0.5,
            "planner_config": {"mcts_c_puct": 2.0, (field): -1},
        })
        .to_string();
        let (code, out) = audit_ledger(&["audit-ledger", "--request", &request]);
        assert_eq!(code, Some(1), "{out}");
        assert_eq!(out["rejection"]["code"], "InvalidParams", "{out}");
        let message = out["rejection"]["detail"].as_str().unwrap();
        assert!(
            message.contains("unknown field") && message.contains(field),
            "{out}"
        );
    }
}

#[test]
fn simulate_rejects_misaligned_dimensions_and_out_of_range_horizons() {
    for dim in [DIM - 1, DIM + 1] {
        let state = json!(vec![0.0; dim]).to_string();
        let (code, out, stderr) =
            gen_zero(&["simulate", "--state", &state, "--actions", "a"], None);
        assert_eq!(code, Some(1), "dim={dim}: {out} {stderr}");
        assert_eq!(out["is_error"], true);
        assert_eq!(out["rejection"]["code"], "InvalidParams");
        assert!(out["meta"].get("simulation").is_none());
        assert!(stderr.contains("refused"), "{stderr}");
    }
    let state = latent(0.0);
    for horizon in ["0", "257", "18446744073709551615"] {
        let (code, out, stderr) = gen_zero(
            &[
                "simulate",
                "--state",
                &state,
                "--actions",
                "a",
                "--horizon",
                horizon,
            ],
            None,
        );
        assert_eq!(code, Some(1), "horizon={horizon}: {out} {stderr}");
        assert_eq!(out["is_error"], true);
        assert_eq!(out["rejection"]["code"], "InvalidParams");
        assert!(out["meta"].get("simulation").is_none());
        assert!(stderr.contains("refused"), "{stderr}");
    }
}
