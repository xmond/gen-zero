//! `gen-zero qa-gate` loads the tri-teacher verifier before it decides, so a
//! missing artifact is a non-zero exit, never a stage 1 answer. The real run
//! is `#[ignore]`d: the trained adapter (~117 MB) and Qwen2.5-0.5B are not
//! shipped in this repository.

use serde_json::Value;
use std::process::{Command, Output};

fn qa_gate(model: &str, adapter: &str, best: &str, null: &str) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["qa-gate", "--model", model, "--adapter", adapter])
        .args(["--context", "The Eiffel Tower is in Paris, France."])
        .args(["--question", "Where is the Eiffel Tower?"])
        .args(["--candidate", "Paris"])
        .args(["--best-span-score", best, "--null-score", null])
        .env_remove("GENZERO_QWEN_TOKENIZER_PATH")
        .output()
        .expect("run gen-zero");
    eprintln!(
        "exit={:?}\nstdout={}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

#[test]
fn missing_adapter_exits_non_zero_without_a_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("adapter.safetensors");
    // A fast-pass score: even here the verifier must load first.
    let out = qa_gate(
        dir.path().to_str().unwrap(),
        missing.to_str().unwrap(),
        "3.0",
        "0.0",
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty(), "no evidence may be printed");
    assert!(String::from_utf8_lossy(&out.stderr).contains("load tri-teacher verifier"));
}

fn resolve_adapter(path_str: &str) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path_str);
    if p.exists() {
        return p;
    }
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let candidate = repo_root.join(path_str);
    if candidate.exists() {
        return candidate;
    }
    p
}

fn resolve_qwen_base() -> Option<std::path::PathBuf> {
    if let Ok(v) = std::env::var("GENZERO_QWEN_BASE_DIR") {
        let p = std::path::PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Runs end-to-end stage 1 and stage 2 when Qwen2.5-0.5B base is available locally.
#[test]
fn real_verifier_runs_stage_2_only_inside_the_band() {
    let model = match resolve_qwen_base() {
        Some(m) => m,
        None => {
            eprintln!("skipping real_verifier_runs_stage_2: Qwen2.5-0.5B not found in cache");
            return;
        }
    };
    let adapter_str = std::env::var("GENZERO_TRI_TEACHER_ADAPTER")
        .unwrap_or_else(|_| "examples/weights/tri_teacher_demo.safetensors".into());
    let adapter = resolve_adapter(&adapter_str);
    assert!(
        adapter.exists(),
        "adapter not found at {}",
        adapter.display()
    );

    let model_str = model.to_str().unwrap();
    let adapter_str = adapter.to_str().unwrap();

    let out = qa_gate(model_str, adapter_str, "3.0", "0.0");
    assert_eq!(out.status.code(), Some(0));
    let fast: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(fast["evidence"]["fast_pass"], true);
    assert_eq!(fast["evidence"]["stage2_triggered"], false);
    assert_eq!(fast["evidence"]["final_answer"], "Paris");

    let out = qa_gate(model_str, adapter_str, "1.0", "1.0");
    assert_eq!(out.status.code(), Some(0));
    let slow: Value = serde_json::from_slice(&out.stdout).unwrap();
    let ev = &slow["evidence"];
    assert_eq!(ev["stage2_triggered"], true);
    assert_eq!(ev["verifier_type"], "tri_teacher");
    let tri_sim = ev["tri_sim"].as_f64().expect("stage 2 reports tri_sim");
    let threshold = slow["config"]["tri_teacher_threshold"].as_f64().unwrap();
    // The verdict follows the measured similarity, whichever way it falls.
    assert_eq!(ev["is_answerable"].as_bool().unwrap(), tri_sim >= threshold);
    println!(
        "tri_sim {tri_sim} threshold {threshold} answerable {}",
        ev["is_answerable"]
    );
}
