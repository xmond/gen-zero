//! End-to-end acceptance test for the `gen-zero reflex-*` subcommands,
//! against the real compiled binary (not the library in-process): checkpoint
//! files on disk, a real SQLite database, real subprocess exit codes.
//!
//! Covers: differential patch create (with its own round-trip verification)
//! -> inspect -> apply; SQLite feedback record -> status -> online
//! adaptation cycle -> hot-swapped checkpoint -> status again -> prune; and a
//! concurrent-load micro-benchmark including a live patch hot-swap.

use gen_zero_model::reflex::{ReflexHead, ReflexOperator, ReflexOperatorConfig, ReflexPlugin};
use gen_zero_storage::{ReflexTraceRecord, SqliteFeedbackStore};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

struct Xorshift32(u32);
impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

const INPUT_DIM: usize = 12;

fn build_plugin(task: &str, seed: u32) -> ReflexPlugin {
    let rank = 4;
    let mut rng = Xorshift32(seed);
    let config = ReflexOperatorConfig {
        input_dim: INPUT_DIM,
        lora_rank: rank,
        alpha: 0.5,
        steps: 3,
        epsilon: 1e-6,
    };
    let operator = ReflexOperator::new(
        config,
        rng.vec(INPUT_DIM * rank),
        rng.vec(INPUT_DIM * rank),
        rng.vec(2 * rank * rank),
        rng.vec(rank),
        rng.vec(2 * rank * rank),
        rng.vec(rank),
        rng.vec(rank * INPUT_DIM),
    )
    .unwrap();
    let head = ReflexHead::new(
        "main",
        INPUT_DIM,
        rng.vec(2 * INPUT_DIM),
        rng.vec(2),
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    ReflexPlugin::new(task, operator, vec![head], Some("main".into())).unwrap()
}

fn bumped(plugin: &ReflexPlugin, bump: f32) -> ReflexPlugin {
    let ops = plugin.operator().weight_arrays();
    let bumped: Vec<Vec<f32>> = ops
        .iter()
        .map(|a| a.iter().map(|&v| v + bump).collect())
        .collect();
    let operator = ReflexOperator::new(
        plugin.operator().config(),
        bumped[0].clone(),
        bumped[1].clone(),
        bumped[2].clone(),
        bumped[3].clone(),
        bumped[4].clone(),
        bumped[5].clone(),
        bumped[6].clone(),
    )
    .unwrap();
    let heads: Vec<ReflexHead> = plugin
        .heads()
        .map(|h| {
            ReflexHead::new(
                h.name.clone(),
                plugin.input_dim(),
                h.weight.iter().map(|&w| w + bump).collect(),
                h.bias.iter().map(|&b| b + bump).collect(),
                h.candidates.clone(),
            )
            .unwrap()
        })
        .collect();
    ReflexPlugin::new(
        plugin.task().to_string(),
        operator,
        heads,
        plugin.default_head().map(String::from),
    )
    .unwrap()
}

fn run(args: &[&str]) -> (Option<i32>, Value, String) {
    let out: Output = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(args)
        .output()
        .expect("run gen-zero");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    eprintln!(
        "$ gen-zero {}\nexit={:?}\nstdout={stdout}\nstderr={stderr}",
        args.join(" "),
        out.status.code()
    );
    let json = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    (out.status.code(), json, stderr)
}

fn path_str(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn patch_create_inspect_apply_round_trip_via_cli() {
    let dir = TempDir::new().unwrap();
    let base = build_plugin("patchdemo", 1);
    let target = bumped(&base, 0.03);
    let base_path = dir.path().join("base.bin");
    let target_path = dir.path().join("target.bin");
    let patch_path = dir.path().join("patch.bin");
    let applied_path = dir.path().join("applied.bin");
    std::fs::write(&base_path, base.to_bytes().unwrap()).unwrap();
    std::fs::write(&target_path, target.to_bytes().unwrap()).unwrap();

    let (code, out, _) = run(&[
        "reflex-patch-create",
        "--base",
        path_str(&base_path),
        "--target",
        path_str(&target_path),
        "--out",
        path_str(&patch_path),
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["verified_round_trip"], true);
    let base_hash_hex = base
        .sha256()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let target_hash_hex = target
        .sha256()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(out["base_sha256"], base_hash_hex);
    assert_eq!(out["target_sha256"], target_hash_hex);

    let (code, out, _) = run(&["reflex-patch-inspect", "--patch", path_str(&patch_path)]);
    assert_eq!(code, Some(0));
    assert_eq!(out["base_sha256"], base_hash_hex);
    assert_eq!(out["target_sha256"], target_hash_hex);
    assert!(out["disk_bytes"].as_u64().unwrap() > 0);
    assert_eq!(out["heads"][0]["head"], "main");

    let (code, out, _) = run(&[
        "reflex-patch-apply",
        "--base",
        path_str(&base_path),
        "--patch",
        path_str(&patch_path),
        "--out",
        path_str(&applied_path),
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["target_sha256"], target_hash_hex);

    let applied_bytes = std::fs::read(&applied_path).unwrap();
    let applied = ReflexPlugin::from_bytes(&applied_bytes).unwrap();
    assert_eq!(applied.sha256().unwrap(), target.sha256().unwrap());

    // Verified predict output: applying the patch out-of-band must produce
    // bit-identical predictions to the independently constructed target.
    let mut rng = Xorshift32(555);
    let x = rng.vec(INPUT_DIM);
    let applied_decision = applied.predict(&x, None).unwrap();
    let target_decision = target.predict(&x, None).unwrap();
    assert_eq!(
        applied_decision.probabilities,
        target_decision.probabilities
    );
}

#[test]
fn reflex_bench_reports_latency_and_hot_swap_via_cli() {
    let dir = TempDir::new().unwrap();
    let base = build_plugin("benchdemo", 2);
    // An independently seeded target: many elements change sign, so this
    // needs `reflex-patch-create`'s exact fix-ups to verify at all.
    let target = build_plugin("benchdemo", 99);
    let base_path = dir.path().join("base.bin");
    let target_path = dir.path().join("target.bin");
    let patch_path = dir.path().join("patch.bin");
    std::fs::write(&base_path, base.to_bytes().unwrap()).unwrap();
    std::fs::write(&target_path, target.to_bytes().unwrap()).unwrap();

    let (code, created, stderr) = run(&[
        "reflex-patch-create",
        "--base",
        path_str(&base_path),
        "--target",
        path_str(&target_path),
        "--out",
        path_str(&patch_path),
    ]);
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(created["verified_round_trip"], true);
    assert!(created["exact_fixups"].as_u64().unwrap() > 0, "{created}");

    let (code, out, _) = run(&[
        "reflex-bench",
        "--plugin",
        path_str(&base_path),
        "--threads",
        "4",
        "--iterations",
        "500",
        "--patch",
        path_str(&patch_path),
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["threads"], 4);
    assert!(out["total_predictions"].as_u64().unwrap() >= 2000);
    assert!(out["predict_latency_us"]["p50"].as_u64().is_some());
    assert!(out["predict_latency_us"]["p99"].as_u64().is_some());
    assert!(out["hot_swap_latency_us"].as_u64().is_some(), "{out}");
}

#[test]
fn feedback_lifecycle_and_online_adapt_via_cli() {
    let dir = TempDir::new().unwrap();
    let base = build_plugin("adaptdemo", 3);
    let base_path = dir.path().join("base.bin");
    let adapted_path = dir.path().join("adapted.bin");
    let db_path = dir.path().join("feedback.sqlite3");
    std::fs::write(&base_path, base.to_bytes().unwrap()).unwrap();

    // Record traces directly (this is the production predict call site's
    // job, which lives outside the CLI); driving feedback attachment,
    // status, and adaptation entirely through the CLI subprocess below.
    let store = SqliteFeedbackStore::open(&db_path, INPUT_DIM).unwrap();
    let mut rng = Xorshift32(2024);
    let x = rng.vec(INPUT_DIM);
    let pre_decision = base.predict(&x, None).unwrap();
    let (head_input, _telemetry) = base.restored_features(&x).unwrap();
    let target_label = pre_decision.label.clone();
    let target_idx = pre_decision.class_index;

    for i in 0..15 {
        let trace_id = format!("trace-{i}");
        store
            .insert_trace(&ReflexTraceRecord {
                trace_id: trace_id.clone(),
                created_at_unix_ms: 1_700_000_000_000 + i,
                task: "adaptdemo".into(),
                head: pre_decision.head.clone(),
                plugin_version: "v1".into(),
                z0: x.clone(),
                head_input: head_input.clone(),
                pred_label: pre_decision.label.clone(),
                pred_confidence: pre_decision.confidence as f64,
                max_gamma: pre_decision.telemetry.max_gamma.map(|g| g as f64),
            })
            .unwrap();
    }
    drop(store); // release the SQLite connection before the CLI subprocess opens it

    let db = path_str(&db_path);
    let dim = INPUT_DIM.to_string();

    // Before any feedback is attached: 15 unlabeled rows.
    let (code, out, _) = run(&["reflex-feedback-status", "--db", db, "--input-dim", &dim]);
    assert_eq!(code, Some(0));
    assert_eq!(out["total"], 15);
    assert_eq!(out["unlabeled"], 15);
    assert_eq!(out["unconsumed"], 0);

    for i in 0..15 {
        let trace_id = format!("trace-{i}");
        let (code, _out, _) = run(&[
            "reflex-feedback-record",
            "--db",
            db,
            "--input-dim",
            &dim,
            "--trace-id",
            &trace_id,
            "--label",
            &target_label,
        ]);
        assert_eq!(code, Some(0));
    }

    let (code, out, _) = run(&["reflex-feedback-status", "--db", db, "--input-dim", &dim]);
    assert_eq!(code, Some(0));
    assert_eq!(out["unlabeled"], 0);
    assert_eq!(out["unconsumed"], 15);
    assert_eq!(out["trained"], 0);

    let (code, out, _) = run(&[
        "reflex-adapt",
        "--plugin",
        path_str(&base_path),
        "--out",
        path_str(&adapted_path),
        "--db",
        db,
        "--task",
        "adaptdemo",
        "--head",
        "main",
        "--learning-rate",
        "0.5",
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["adapted"], true);
    assert_eq!(out["trace_count"], 15);
    let loss_before = out["loss_before"].as_f64().unwrap();
    let loss_after = out["loss_after"].as_f64().unwrap();
    assert!(
        loss_after <= loss_before,
        "{loss_after} should be <= {loss_before}"
    );

    // Registry hot-swap -> verified predict output: the checkpoint written by
    // reflex-adapt must be strictly more confident in the trained-toward
    // label, on the exact same input, than the pre-adaptation base plugin.
    let adapted_bytes = std::fs::read(&adapted_path).unwrap();
    let adapted = ReflexPlugin::from_bytes(&adapted_bytes).unwrap();
    let target_sha256_hex = out["target_sha256"].as_str().unwrap();
    let adapted_hash_hex: String = adapted
        .sha256()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(adapted_hash_hex, target_sha256_hex);

    let post_decision = adapted.predict(&x, None).unwrap();
    assert_eq!(post_decision.class_index, target_idx);
    assert!(
        post_decision.probabilities[target_idx] >= pre_decision.probabilities[target_idx],
        "post-adapt confidence {} should be >= pre-adapt confidence {}",
        post_decision.probabilities[target_idx],
        pre_decision.probabilities[target_idx]
    );

    // All rows are now trained.
    let (code, out, _) = run(&["reflex-feedback-status", "--db", db, "--input-dim", &dim]);
    assert_eq!(code, Some(0));
    assert_eq!(out["unconsumed"], 0);
    assert_eq!(out["trained"], 15);

    // A second adapt cycle is a no-op: nothing left to consume.
    let (code, out, _) = run(&[
        "reflex-adapt",
        "--plugin",
        path_str(&adapted_path),
        "--out",
        path_str(&adapted_path),
        "--db",
        db,
        "--task",
        "adaptdemo",
        "--head",
        "main",
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["adapted"], false);

    // Prune: default (non-forced) prune of trained rows before a future cutoff succeeds.
    let far_future_ms = 9_999_999_999_999i64;
    let (code, out, _) = run(&[
        "reflex-feedback-prune",
        "--db",
        db,
        "--input-dim",
        &dim,
        "--before-unix-ms",
        &far_future_ms.to_string(),
    ]);
    assert_eq!(code, Some(0));
    assert_eq!(out["rows_deleted"], 15);

    let (code, out, _) = run(&["reflex-feedback-status", "--db", db, "--input-dim", &dim]);
    assert_eq!(code, Some(0));
    assert_eq!(out["total"], 0);
}
