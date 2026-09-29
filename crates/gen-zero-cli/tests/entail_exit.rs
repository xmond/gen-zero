//! `gen-zero entail` reaches the same cognitive runtime as HTTP and MCP, and a
//! refusal exits non-zero.

use serde_json::Value;
use std::process::Command;

const ASSETS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../gen-zero-service/tests/fixtures/cognitive_assets_boolq_entailment.json"
);

fn point(r: f64, theta: f64) -> String {
    let mut v = vec![0.0; 128];
    v[0] = r * theta.cos();
    v[1] = r * theta.sin();
    v[104] = 1.0;
    serde_json::to_string(&v).unwrap()
}

fn entail(passage: &str, question: &str, assets: Option<&str>) -> (Option<i32>, Value) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gen-zero"));
    cmd.args(["entail", "--passage", passage, "--question", question])
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS");
    if let Some(a) = assets {
        cmd.args(["--mount-assets", a]);
    }
    let out = cmd.output().expect("run gen-zero");
    eprintln!(
        "exit={:?}\nstdout={}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    (out.status.code(), json)
}

#[test]
fn entailed_pair_exits_zero_and_reverse_is_a_false_verdict() {
    let (general, specific) = (point(0.3, 0.0), point(0.8, 0.02));
    let (code, out) = entail(&general, &specific, Some(ASSETS));
    assert_eq!(code, Some(0));
    assert_eq!(out["meta"]["entailment"]["is_entailed"], true);
    let (code, out) = entail(&specific, &general, Some(ASSETS));
    assert_eq!(code, Some(0), "a false verdict is an answer, not a failure");
    assert_eq!(out["meta"]["entailment"]["is_entailed"], false);
}

#[test]
fn missing_geometry_and_boundary_points_exit_one() {
    let (code, out) = entail(&point(0.3, 0.0), &point(0.8, 0.0), None);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "BackendUnavailable");
    let (code, out) = entail(&point(0.3, 0.0), &point(1.0, 0.0), Some(ASSETS));
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "DomainViolation");
}

/// The scheme 1 fixture plus `entailment.dynamics` (`A = 0`, `B = I`),
/// written to a per-process temp file.
fn fiber_assets_file() -> std::path::PathBuf {
    let mut assets: Value =
        serde_json::from_str(&std::fs::read_to_string(ASSETS).unwrap()).unwrap();
    let square = |d: f64| -> Vec<Vec<f64>> {
        (0..128)
            .map(|i| (0..128).map(|j| if i == j { d } else { 0.0 }).collect())
            .collect()
    };
    assets["entailment"]["dynamics"] = serde_json::json!({"a": square(0.0), "b": square(1.0)});
    let path =
        std::env::temp_dir().join(format!("gen-zero-fiber-assets-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&assets).unwrap()).unwrap();
    path
}

fn events(input: &[f64]) -> String {
    serde_json::json!([{"time_ns": 1_000_000_000u64, "input": input}]).to_string()
}

fn run(args: &[&str]) -> (Option<i32>, Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(args)
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    eprintln!(
        "exit={:?}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    (out.status.code(), json)
}

#[test]
fn fiber_events_run_the_cross_diff_ssm_path_from_the_cli() {
    let assets = fiber_assets_file();
    let assets = assets.to_str().unwrap();
    let p = point(0.3, 0.0);
    let zero = vec![0.0; 128];
    let mut radial = vec![0.0; 128];
    radial[0] = 0.5;
    let (pe, qe) = (events(&zero), events(&radial));

    let (code, out) = run(&[
        "entail",
        "--passage",
        &p,
        "--question",
        &p,
        "--passage-events",
        &pe,
        "--question-events",
        &qe,
        "--window-start-ns",
        "0",
        "--mount-assets",
        assets,
    ]);
    assert_eq!(code, Some(0), "{out}");
    let e = &out["meta"]["entailment"];
    assert_eq!(e["mode"], "fiber_cross_diff_ssm", "{out}");
    assert_eq!(e["is_entailed"], true, "{out}");
    assert_eq!(e["fiber_ssm"]["direct"]["is_entailed"], false, "{out}");
    assert_eq!(e["fiber_ssm"]["scan"]["steps"], 1, "{out}");

    // One channel alone is refused, not run as scheme 1.
    let (code, out) = run(&[
        "entail",
        "--passage",
        &p,
        "--question",
        &p,
        "--passage-events",
        &pe,
        "--window-start-ns",
        "0",
        "--mount-assets",
        assets,
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "InvalidParams");

    // Streams against a mount with no dynamics: BackendUnavailable.
    let (code, out) = run(&[
        "entail",
        "--passage",
        &p,
        "--question",
        &p,
        "--passage-events",
        &pe,
        "--question-events",
        &qe,
        "--window-start-ns",
        "0",
        "--mount-assets",
        ASSETS,
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "BackendUnavailable");
    let _ = std::fs::remove_file(assets);
}
