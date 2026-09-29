//! The `gen-zero reflex` binary: a refused or gated decision exits non-zero,
//! and `--cognitive` reaches the same cognitive runtime as HTTP and MCP.

use serde_json::{json, Value};
use std::process::{Command, Output};

const ASSETS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../gen-zero-service/tests/fixtures/cognitive_assets_linear2d.json"
);

fn reflex(args: &[&str]) -> (Option<i32>, Value, String) {
    let out: Output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .arg("reflex")
        .args(args)
        // No scorer in tests: the bridge is off, so request text is unassessed.
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
    let json = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    (out.status.code(), json, stderr)
}

fn cognitive(state: [f64; 2], u: f64) -> String {
    let events: Vec<Value> = (1..=10u64)
        .map(|i| json!({"time_ns": i * 100_000_000, "input": [u, 0.0]}))
        .collect();
    json!({
        "state": state, "goal": [0.5, 0.0], "window_start_ns": 0,
        "controls": {"go": events},
    })
    .to_string()
}

#[test]
fn gated_reflex_exits_one_not_zero() {
    let (code, out, stderr) = reflex(&["--context", "deploy now"]);
    assert_eq!(code, Some(1));
    assert_eq!(out["is_error"], true);
    assert!(stderr.contains("refused"), "{stderr}");
}

#[test]
fn cognitive_domain_violation_exits_one() {
    let cog = cognitive([0.6, 0.8], 1.0);
    let (code, out, _) = reflex(&[
        "--context",
        "move",
        "--candidates",
        "go",
        "--mount-assets",
        ASSETS,
        "--cognitive",
        &cog,
    ]);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "DomainViolation");
    assert_eq!(out["meta"]["mount"]["version"], 2);
}

#[test]
fn cognitive_energy_rise_exits_one_with_no_action() {
    let cog = cognitive([0.1, 0.0], -1.0);
    let (code, out, _) = reflex(&[
        "--context",
        "move",
        "--candidates",
        "go",
        "--mount-assets",
        ASSETS,
        "--cognitive",
        &cog,
    ]);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "EnergyRose");
    assert!(out["meta"]["certified_action"].is_null());
}

#[test]
fn cognitive_certified_action_is_still_held_without_a_risk_classifier() {
    let cog = cognitive([0.1, 0.0], 1.0);
    let (code, out, _) = reflex(&[
        "--context",
        "move",
        "--candidates",
        "go",
        "--mount-assets",
        ASSETS,
        "--cognitive",
        &cog,
    ]);
    // Geometry certified the action; the unassessed request text escalates.
    assert_eq!(code, Some(1));
    assert_eq!(out["meta"]["certified_action"]["action"], "go");
    assert_eq!(out["meta"]["committed"], false);
    assert_eq!(out["meta"]["tier"], "Escalate");
}

#[test]
fn unreadable_assets_stop_the_command() {
    let (code, out, stderr) = reflex(&["--context", "x", "--mount-assets", "/nonexistent/a.json"]);
    assert_ne!(code, Some(0));
    assert!(out.is_null(), "no decision is printed");
    assert!(stderr.contains("/nonexistent/a.json"), "{stderr}");
}
