//! `gen-zero fold` reaches the same `causal_fold` verb as HTTP and MCP, and a
//! refusal exits non-zero. Tables mirror the fixtures in
//! `gen-zero-lod/src/semiring.rs` `#[cfg(test)] mod tests`.

use serde_json::{json, Value};
use std::process::Command;

/// Write `axioms` (a JSON array of `{r1, r2, gender, result}`) to a
/// per-process temp file and return its path.
fn axioms_file(axioms: Value, tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "gen-zero-causal-fold-axioms-{tag}-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, serde_json::to_vec(&axioms).unwrap()).unwrap();
    path
}

fn fold(args: &[&str]) -> (Option<i32>, Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .arg("fold")
        .args(args)
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    eprintln!(
        "exit={:?}\nstdout={}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    (out.status.code(), json)
}

// ------------------------------------------------------------------ (a)

fn two_plus_two_axioms() -> Value {
    json!([
        {"r1": 10, "r2": 11, "gender": "Female", "result": [14]},
        {"r1": 12, "r2": 13, "gender": "Unknown", "result": [15]},
        {"r1": 14, "r2": 15, "gender": "Unknown", "result": [16]},
    ])
}

#[test]
fn chart_closes_a_2plus2_chain_that_left_fold_refuses() {
    let axioms = axioms_file(two_plus_two_axioms(), "a");
    let axioms = axioms.to_str().unwrap();
    let edges = "[10, 11, 12, 13]";
    let genders = r#"["Male", "Male", "Female", "Male", "Unknown"]"#;

    let (code, out) = fold(&[
        "--edges",
        edges,
        "--genders",
        genders,
        "--axioms",
        axioms,
        "--strategy",
        "left",
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "CausalFoldRefused", "{out}");

    let (code, out) = fold(&[
        "--edges",
        edges,
        "--genders",
        genders,
        "--axioms",
        axioms,
        "--strategy",
        "chart",
    ]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["causal_fold"]["predicted"], 16, "{out}");
    assert_eq!(out["meta"]["causal_fold"]["steps"], 3, "{out}");
    assert_eq!(
        out["meta"]["causal_fold"]["proof_path"],
        json!([10, 11, 14, 12, 13, 15, 16]),
        "{out}"
    );

    // No --strategy given: default is chart, same result.
    let (code, out) = fold(&["--edges", edges, "--genders", genders, "--axioms", axioms]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["causal_fold"]["predicted"], 16, "{out}");
    assert_eq!(out["meta"]["causal_fold"]["strategy"], "chart", "{out}");

    let _ = std::fs::remove_file(axioms);
}

// ------------------------------------------------------------------ (b)

fn inner_first_axioms() -> Value {
    json!([
        {"r1": 11, "r2": 12, "gender": "Female", "result": [17]},
        {"r1": 17, "r2": 13, "gender": "Male", "result": [18]},
        {"r1": 10, "r2": 18, "gender": "Male", "result": [19]},
    ])
}

#[test]
fn retired_strategy_refuses_where_chart_closes() {
    let axioms = axioms_file(inner_first_axioms(), "b");
    let axioms = axioms.to_str().unwrap();
    let edges = "[10, 11, 12, 13]";
    let genders = r#"["Male", "Female", "Unknown", "Female", "Male"]"#;

    let (code, out) = fold(&[
        "--edges",
        edges,
        "--genders",
        genders,
        "--axioms",
        axioms,
        "--strategy",
        "bidirectional",
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "InvalidParams", "{out}");

    let (code, out) = fold(&[
        "--edges",
        edges,
        "--genders",
        genders,
        "--axioms",
        axioms,
        "--strategy",
        "chart",
    ]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["causal_fold"]["predicted"], 19, "{out}");
    assert_eq!(out["meta"]["causal_fold"]["steps"], 3, "{out}");
    assert_eq!(
        out["meta"]["causal_fold"]["proof_path"],
        json!([10, 11, 12, 17, 13, 18, 19]),
        "{out}"
    );

    let _ = std::fs::remove_file(axioms);
}

// ------------------------------------------------------------------ (c)

#[test]
fn chart_refuses_when_bracketings_disagree() {
    let axioms = axioms_file(
        json!([
            {"r1": 0, "r2": 0, "gender": "Male", "result": [1]},
            {"r1": 1, "r2": 0, "gender": "Male", "result": [2]},
            {"r1": 0, "r2": 1, "gender": "Male", "result": [3]},
        ]),
        "c",
    );
    let axioms = axioms.to_str().unwrap();

    let (code, out) = fold(&[
        "--edges",
        "[0, 0, 0]",
        "--genders",
        r#"["Male", "Male", "Male", "Male"]"#,
        "--axioms",
        axioms,
        "--strategy",
        "chart",
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["is_error"], true, "{out}");
    assert_eq!(out["rejection"]["code"], "CausalFoldRefused", "{out}");
    let detail = out["rejection"]["detail"].as_str().unwrap_or("");
    assert!(detail.contains("bracketings disagree: [2, 3]"), "{detail}");

    let _ = std::fs::remove_file(axioms);
}

// ------------------------------------------------------------------ (d)

#[test]
fn a_conflict_key_refuses() {
    let axioms = axioms_file(
        json!([{"r1": 10, "r2": 11, "gender": "Male", "result": [14, 15]}]),
        "d",
    );
    let axioms = axioms.to_str().unwrap();

    let (code, out) = fold(&[
        "--edges",
        "[10, 11]",
        "--genders",
        r#"["Male", "Male", "Male"]"#,
        "--axioms",
        axioms,
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "CausalFoldRefused", "{out}");

    let _ = std::fs::remove_file(axioms);
}

// ------------------------------------------------------------------ (e)

#[test]
fn malformed_edges_json_exits_one() {
    let (code, out) = fold(&["--edges", "not json", "--genders", "[\"Male\"]"]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.is_null(), "no outcome is printed on a CLI parse error");
}

#[test]
fn genders_length_mismatch_is_invalid_params() {
    let (code, out) = fold(&["--edges", "[10, 11]", "--genders", r#"["Male", "Male"]"#]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "InvalidParams", "{out}");
}

#[test]
fn unknown_strategy_is_invalid_params() {
    let (code, out) = fold(&[
        "--edges",
        "[10, 11]",
        "--genders",
        r#"["Male", "Male", "Male"]"#,
        "--strategy",
        "sideways",
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "InvalidParams", "{out}");
}

#[test]
fn missing_axioms_file_exits_one() {
    let (code, out) = fold(&[
        "--edges",
        "[10, 11]",
        "--genders",
        r#"["Male", "Male", "Male"]"#,
        "--axioms",
        "/nonexistent/axioms.json",
    ]);
    assert_eq!(code, Some(1));
    assert!(
        out.is_null(),
        "no outcome is printed when the axioms file is unreadable"
    );
}

#[test]
fn no_axioms_with_a_2edge_chain_refuses() {
    let (code, out) = fold(&[
        "--edges",
        "[10, 11]",
        "--genders",
        r#"["Male", "Male", "Male"]"#,
    ]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "CausalFoldRefused", "{out}");
}

#[test]
fn chart_sets_reaches_service_and_returns_proof() {
    let path = axioms_file(
        json!([
            {"r1": 1, "r2": 3, "gender": "Male", "result": [9]}
        ]),
        "sets",
    );
    let (code, out) = fold(&[
        "--sets",
        "[[1], [2, 3]]",
        "--gender",
        "Male",
        "--axioms",
        path.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["causal_fold"]["predicted"], 9);
    assert_eq!(out["meta"]["causal_fold"]["proof_path"], json!([1, 3, 9]));
    assert_eq!(out["meta"]["causal_fold"]["edges"], 2);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn sets_cli_refuses_missing_gender_and_conflicting_edges() {
    for args in [
        vec!["--sets", "[[1]]"],
        vec!["--sets", "[[1]]", "--edges", "[1]", "--gender", "Male"],
    ] {
        let (code, _) = fold(&args);
        assert_eq!(code, Some(2));
    }
    let (code, out) = fold(&["--sets", "[[1], [2]]", "--gender", "Male"]);
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "CausalFoldRefused");
}

#[test]
fn weighted_cli_success_and_margin_refusal() {
    let axioms = axioms_file(
        json!([{"r1":10,"r2":11,"gender":"Male","result":[14,15]}]),
        "weighted-support",
    );
    let weights = axioms_file(
        json!([
            {"r1":10,"r2":11,"gender":"Male","relation":14,"count":9},
            {"r1":10,"r2":11,"gender":"Male","relation":15,"count":1}
        ]),
        "weighted-counts",
    );
    for (threshold, expected) in [("0.1", 0), ("3", 1)] {
        for strategy in ["weighted_tropical", "tiered"] {
            let (code, out) = fold(&[
                "--edges",
                "[10,11]",
                "--genders",
                r#"["Male","Male","Male"]"#,
                "--axioms",
                axioms.to_str().unwrap(),
                "--weights",
                weights.to_str().unwrap(),
                "--strategy",
                strategy,
                "--margin-threshold",
                threshold,
            ]);
            assert_eq!(code, Some(expected), "{out}");
            if expected == 0 {
                assert_eq!(out["meta"]["causal_fold"]["predicted"], 14);
                assert!(
                    (out["meta"]["causal_fold"]["energy"].as_f64().unwrap() + 0.9_f64.ln()).abs()
                        < 1e-12
                );
            } else {
                assert_eq!(out["rejection"]["code"], "CausalFoldRefused");
                assert_eq!(
                    out["meta"]["causal_fold"]["candidates"]
                        .as_array()
                        .unwrap()
                        .len(),
                    2
                );
            }
        }
    }
    let (code, out) = fold(&[
        "--edges",
        "[10,11]",
        "--genders",
        r#"["Male","Male","Male"]"#,
        "--axioms",
        axioms.to_str().unwrap(),
        "--weights",
        weights.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["causal_fold"]["strategy"], "tiered");
    assert_eq!(out["meta"]["causal_fold"]["dispatched_band"], 1);
    std::fs::remove_file(axioms).unwrap();
    std::fs::remove_file(weights).unwrap();
}
