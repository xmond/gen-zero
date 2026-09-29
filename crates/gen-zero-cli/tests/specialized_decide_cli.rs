use gen_zero_core::CompressedLatent;
use gen_zero_nanocore::core_type::fixtures::synthetic_core;
use gen_zero_nanocore::DOMAIN_GENERAL;
use serde_json::{json, Value};
use std::process::Command;

fn decide(extra: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["decide", "--context", "pick", "--candidates", "alpha,beta"])
        .args(extra)
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .unwrap();
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    (output.status.code().unwrap(), body)
}

#[test]
fn etf_flag_reaches_service_head() {
    let reps = r#"{"alpha":[1.0,0.0],"beta":[0.0,1.0]}"#;
    let (code, out) = decide(&[
        "--head",
        "etf",
        "--etf-rep",
        "[1.0,0.0]",
        "--candidate-reps",
        reps,
        "--etf-temperature",
        "0.2",
    ]);
    assert!(matches!(code, 0 | 1)); // gate status is independent of routing
    assert_eq!(out["meta"]["head"], "etf", "{out}");
    assert_eq!(out["meta"]["candidates"].as_array().unwrap().len(), 2);
    assert_eq!(out["meta"]["chosen_action"], "alpha");
    assert_eq!(out["meta"]["etf"]["temperature_source"], "caller");

    let (code, refused) = decide(&["--head", "etf", "--etf-rep", "[1.0,0.0]"]);
    assert_eq!(code, 1);
    assert!(
        refused["rejection"]["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("candidate_reps")
            || refused["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("candidate_reps")
    );
}

#[test]
fn nanocore_flag_runs_supplied_core_and_missing_core_fails_closed() {
    let (code, refused) = decide(&["--engine", "nanocore"]);
    assert_eq!(code, 1);
    assert_eq!(refused["is_error"], true);
    assert!(refused["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("decision_state"));

    let core = synthetic_core(
        DOMAIN_GENERAL,
        "test untrained core",
        CompressedLatent { values: [1.0; 128] },
        4,
        0.5,
        &["alpha", "beta"],
    );
    let core_json = serde_json::to_string(&core).unwrap();
    let state_json = json!(vec![1.0_f32; 128]).to_string();
    let reps_json = json!({"alpha": vec![1.0_f32; 128], "beta": vec![-1.0_f32; 128]}).to_string();
    let (_, out) = decide(&[
        "--engine",
        "nanocore",
        "--head",
        "etf",
        "--nanocore-core",
        &core_json,
        "--decision-state",
        &state_json,
        "--candidate-reps",
        &reps_json,
    ]);
    assert_eq!(out["meta"]["engine"], "nanocore", "{out}");
    assert_eq!(out["meta"]["head"], "etf");
    assert!(out["meta"]["nanocore"]["confidence"].is_number());
}
