//! `gen-zero causal-plan` reaches the same `causal_plan` verb as HTTP and
//! MCP, reads a file or standard input, and a refusal exits non-zero.

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

fn chain(budget: f64) -> Value {
    json!({
        "nodes": [
            {"id": 1, "cost": 1.0},
            {"id": 2, "cost": 2.0, "and_parents": [1]},
            {"id": 3, "cost": 3.0, "or_parents": [[2, 4]]},
            {"id": 4, "cost": 9.0}
        ],
        "target": 3,
        "budget": budget
    })
}

/// A chain of `n` nodes (i needs i-1, node 0 a free root), listed in reverse
/// (node n-1 first, node 0 last). Exercises the planner's linear closure
/// over a request shape that used to drive the old full-scan fixpoint into
/// one pass per node.
fn reverse_chain(n: u32, target: u32) -> Value {
    let nodes: Vec<Value> = (0..n)
        .rev()
        .map(|i| {
            if i == 0 {
                json!({"id": i, "cost": 1.0})
            } else {
                json!({"id": i, "cost": 1.0, "and_parents": [i - 1]})
            }
        })
        .collect();
    json!({"nodes": nodes, "target": target, "budget": 1e9})
}

fn causal_plan(input: &str, stdin: Option<&Value>) -> (Option<i32>, Value, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["causal-plan", "--input", input])
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run gen-zero");
    {
        let mut pipe = child.stdin.take().unwrap();
        if let Some(body) = stdin {
            pipe.write_all(body.to_string().as_bytes()).unwrap();
        }
    }
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    eprintln!(
        "exit={:?}\nstdout={}\nstderr={stderr}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
    );
    let json = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    (out.status.code(), json, stderr)
}

#[test]
fn file_input_prints_the_exact_plan() {
    let path =
        std::env::temp_dir().join(format!("gen-zero-causal-plan-{}.json", std::process::id()));
    std::fs::write(&path, chain(6.0).to_string()).unwrap();
    let (code, out, _) = causal_plan(path.to_str().unwrap(), None);
    std::fs::remove_file(&path).ok();
    assert_eq!(code, Some(0));
    assert_eq!(out["verb"], "causal_plan");
    let plan = &out["meta"]["causal_plan"];
    assert_eq!(plan["path"], json!([1, 2, 3]), "{out}");
    assert_eq!(plan["total_cost"], 6.0, "{out}");
}

#[test]
fn stdin_input_over_budget_exits_one_with_budget_exceeded() {
    let (code, out, stderr) = causal_plan("-", Some(&chain(5.0)));
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "BudgetExceeded", "{out}");
    assert!(stderr.contains("refused (BudgetExceeded)"), "{stderr}");
}

#[test]
fn unknown_field_exits_one_with_invalid_params() {
    let mut body = chain(6.0);
    body["heuristic"] = json!("greedy");
    let (code, out, _) = causal_plan("-", Some(&body));
    assert_eq!(code, Some(1));
    assert_eq!(out["rejection"]["code"], "InvalidParams", "{out}");
}

#[test]
fn reverse_listed_65536_chain_reaches_target_fast() {
    // A whole-process timing check (debug build + JSON parse of ~65k nodes),
    // separate from the in-process timing in gen-zero-planner's own tests.
    let path = std::env::temp_dir().join(format!(
        "gen-zero-causal-plan-{}-rev4.json",
        std::process::id()
    ));
    std::fs::write(&path, reverse_chain(65_536, 4).to_string()).unwrap();
    let start = std::time::Instant::now();
    let (code, out, _) = causal_plan(path.to_str().unwrap(), None);
    let elapsed = start.elapsed();
    std::fs::remove_file(&path).ok();
    eprintln!("reverse_listed_65536_chain_reaches_target_fast took {elapsed:?}");
    assert_eq!(code, Some(0), "{out}");
    let plan = &out["meta"]["causal_plan"];
    assert_eq!(plan["path"], json!([0, 1, 2, 3, 4]), "{out}");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "took {elapsed:?}"
    );
}

#[test]
fn reverse_listed_65536_chain_deep_target_is_refused_fast() {
    let path = std::env::temp_dir().join(format!(
        "gen-zero-causal-plan-{}-rev65535.json",
        std::process::id()
    ));
    std::fs::write(&path, reverse_chain(65_536, 65_535).to_string()).unwrap();
    let start = std::time::Instant::now();
    let (code, out, _) = causal_plan(path.to_str().unwrap(), None);
    let elapsed = start.elapsed();
    std::fs::remove_file(&path).ok();
    eprintln!("reverse_listed_65536_chain_deep_target_is_refused_fast took {elapsed:?}");
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "TargetTooComplex", "{out}");
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "took {elapsed:?}"
    );
}
