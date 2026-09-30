//! Implementations for the `gen-zero reflex-*` subcommands: SQLite feedback
//! store administration, differential patch create/apply/inspect, and a
//! concurrent hot-swap micro-benchmark. Kept out of `main.rs` because each of
//! these does real, multi-step work (file I/O, hashing, threading), not a
//! one-shot request built from flags and handed to `PolymorphicZeroEngine`
//! like the rest of the CLI's subcommands.

use anyhow::Context;
use gen_zero_model::reflex::ReflexPlugin;
use gen_zero_model::ReflexPatch;
use gen_zero_service::{ReflexOnlineAdapter, ReflexRegistry};
use gen_zero_storage::SqliteFeedbackStore;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_millis() as i64
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn load_plugin(path: &Path) -> anyhow::Result<ReflexPlugin> {
    let bytes = std::fs::read(path).with_context(|| format!("read plugin checkpoint {path:?}"))?;
    ReflexPlugin::from_bytes(&bytes).with_context(|| format!("parse plugin checkpoint {path:?}"))
}

fn save_plugin(path: &Path, plugin: &ReflexPlugin) -> anyhow::Result<()> {
    let bytes = plugin.to_bytes().context("serialize plugin checkpoint")?;
    std::fs::write(path, bytes).with_context(|| format!("write plugin checkpoint {path:?}"))
}

fn load_patch(path: &Path) -> anyhow::Result<ReflexPatch> {
    let bytes = std::fs::read(path).with_context(|| format!("read patch {path:?}"))?;
    ReflexPatch::from_bytes(&bytes).with_context(|| format!("parse patch {path:?}"))
}

// ---------------------------------------------------------------------------
// Feedback store administration
// ---------------------------------------------------------------------------

pub fn feedback_status(db: &Path, input_dim: usize, task: Option<&str>) -> anyhow::Result<()> {
    let store = SqliteFeedbackStore::open(db, input_dim)
        .with_context(|| format!("open feedback store {db:?}"))?;
    let status = store.feedback_status(task)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "db": db,
            "task": task,
            "total": status.total,
            "unlabeled": status.unlabeled,
            "unconsumed": status.unconsumed,
            "trained": status.trained,
        }))?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn feedback_record(
    db: &Path,
    input_dim: usize,
    trace_id: &str,
    label: &str,
    feedback_type: &str,
    joined_at_unix_ms: Option<i64>,
) -> anyhow::Result<()> {
    let store = SqliteFeedbackStore::open(db, input_dim)
        .with_context(|| format!("open feedback store {db:?}"))?;
    let joined_at = joined_at_unix_ms.unwrap_or_else(now_unix_ms);
    store
        .record_feedback(trace_id, label, feedback_type, joined_at)
        .with_context(|| format!("record feedback for trace {trace_id:?}"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "trace_id": trace_id,
            "feedback_label": label,
            "feedback_type": feedback_type,
            "joined_at_unix_ms": joined_at,
        }))?
    );
    Ok(())
}

pub fn feedback_prune(
    db: &Path,
    input_dim: usize,
    before_unix_ms: i64,
    force: bool,
) -> anyhow::Result<()> {
    let store = SqliteFeedbackStore::open(db, input_dim)
        .with_context(|| format!("open feedback store {db:?}"))?;
    let deleted = store.prune_before(before_unix_ms, force)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "db": db,
            "before_unix_ms": before_unix_ms,
            "force": force,
            "rows_deleted": deleted,
        }))?
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Differential patch create / apply / inspect
// ---------------------------------------------------------------------------

pub fn patch_create(
    base_path: &Path,
    target_path: &Path,
    out_path: &Path,
    metadata: Option<String>,
) -> anyhow::Result<()> {
    let base = load_plugin(base_path)?;
    let target = load_plugin(target_path)?;
    let metadata = metadata
        .unwrap_or_else(|| format!("diff {} -> {}", base_path.display(), target_path.display()));
    let patch =
        ReflexPatch::diff(&base, &target, metadata).context("compute differential patch")?;

    // Never ship a patch that does not reproduce its own promised target:
    // verified before the file is written, not left for the first
    // `reflex-patch-apply` to discover.
    let applied = base
        .apply_patch(&patch)
        .context("verify patch reproduces target before writing it")?;
    anyhow::ensure!(
        applied.sha256()? == target.sha256()?,
        "internal error: patch verified but hash still differs"
    );

    let bytes = patch.to_bytes().context("serialize patch")?;
    let patch_len = bytes.len();
    std::fs::write(out_path, &bytes).with_context(|| format!("write patch {out_path:?}"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "base_sha256": hex(&patch.base_sha256),
            "target_sha256": hex(&patch.target_sha256),
            "patch_path": out_path,
            "patch_bytes": patch_len,
            "exact_fixups": patch.fixup_count(),
            "verified_round_trip": true,
        }))?
    );
    Ok(())
}

pub fn patch_apply(base_path: &Path, patch_path: &Path, out_path: &Path) -> anyhow::Result<()> {
    let base = load_plugin(base_path)?;
    let patch = load_patch(patch_path)?;
    let applied = base
        .apply_patch(&patch)
        .context("apply patch to base checkpoint")?;
    save_plugin(out_path, &applied)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "base_sha256": hex(&patch.base_sha256),
            "target_sha256": hex(&applied.sha256()?),
            "out_path": out_path,
        }))?
    );
    Ok(())
}

pub fn patch_inspect(patch_path: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(patch_path).with_context(|| format!("read patch {patch_path:?}"))?;
    let patch =
        ReflexPatch::from_bytes(&bytes).with_context(|| format!("parse patch {patch_path:?}"))?;

    let operator_lens = [
        patch.delta_w_down.len(),
        patch.delta_w_ctx.len(),
        patch.delta_w_gate.len(),
        patch.delta_b_gate.len(),
        patch.delta_w_cand.len(),
        patch.delta_b_cand.len(),
        patch.delta_w_up.len(),
    ];
    let operator_elements: usize = operator_lens.iter().sum();
    let operator_nonzero: usize = [
        &patch.delta_w_down,
        &patch.delta_w_ctx,
        &patch.delta_w_gate,
        &patch.delta_b_gate,
        &patch.delta_w_cand,
        &patch.delta_b_cand,
        &patch.delta_w_up,
    ]
    .iter()
    .map(|arr| arr.iter().filter(|&&v| v != 0.0).count())
    .sum();

    let head_summaries: Vec<serde_json::Value> = patch
        .head_deltas
        .iter()
        .map(|(name, d)| {
            let nonzero = d.delta_weight.iter().filter(|&&v| v != 0.0).count()
                + d.delta_bias.iter().filter(|&&v| v != 0.0).count();
            serde_json::json!({
                "head": name,
                "weight_len": d.delta_weight.len(),
                "bias_len": d.delta_bias.len(),
                "nonzero_elements": nonzero,
            })
        })
        .collect();

    let disk_bytes = bytes.len();
    // Reference point: an uncompressed f32-per-element cost for every changed
    // array, i.e. what shipping full arrays (rather than this packed binary
    // patch, which is the same packing, no zstd) would already look like.
    // `reflex-patch-inspect` reports the patch's real on-disk size, not a
    // fabricated "compression ratio" against a format nothing here produces.
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "patch_path": patch_path,
            "format": gen_zero_model::PATCH_FORMAT,
            "base_sha256": hex(&patch.base_sha256),
            "target_sha256": hex(&patch.target_sha256),
            "metadata": patch.metadata,
            "disk_bytes": disk_bytes,
            "operator_delta_elements": operator_elements,
            "operator_delta_nonzero_elements": operator_nonzero,
            "operator_delta_sparsity": 1.0 - (operator_nonzero as f64 / operator_elements.max(1) as f64),
            "exact_fixups": patch.fixup_count(),
            "heads": head_summaries,
        }))?
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Online adaptation: SQLite feedback -> gradient step -> hot-swap -> checkpoint
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn adapt(
    plugin_path: &Path,
    out_path: &Path,
    db: &Path,
    task: &str,
    head: &str,
    limit: usize,
    learning_rate: f32,
) -> anyhow::Result<()> {
    let plugin = load_plugin(plugin_path)?;
    anyhow::ensure!(
        plugin.task() == task,
        "checkpoint {plugin_path:?} is task {:?}, not {task:?}",
        plugin.task()
    );
    let input_dim = plugin.input_dim();
    let registry = Arc::new(ReflexRegistry::new());
    registry.register(plugin);
    let store = SqliteFeedbackStore::open(db, input_dim)
        .with_context(|| format!("open feedback store {db:?}"))?;
    let adapter = ReflexOnlineAdapter::new(store, Arc::clone(&registry), learning_rate);

    match adapter.run_cycle(task, head, limit)? {
        None => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "task": task,
                    "head": head,
                    "adapted": false,
                    "reason": "no unconsumed feedback",
                }))?
            );
        }
        Some(report) => {
            let patched = registry
                .route(task)
                .expect("apply_patch_live just registered this task's result");
            save_plugin(out_path, &patched)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "task": report.task,
                    "head": report.head,
                    "adapted": true,
                    "trace_count": report.trace_ids.len(),
                    "trace_ids": report.trace_ids,
                    "loss_before": report.loss_before,
                    "loss_after": report.loss_after,
                    "base_sha256": hex(&report.base_sha256),
                    "target_sha256": hex(&report.target_sha256),
                    "out_path": out_path,
                }))?
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Micro-benchmark: predict latency under concurrent load, plus optional
// live-patch hot-swap latency.
// ---------------------------------------------------------------------------

fn percentile(sorted_us: &[u64], p: f64) -> u64 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

pub fn bench(
    plugin_path: &Path,
    threads: usize,
    iterations_per_thread: usize,
    patch_path: Option<&Path>,
) -> anyhow::Result<()> {
    anyhow::ensure!(threads >= 1, "--threads must be at least 1");
    anyhow::ensure!(
        iterations_per_thread >= 1,
        "--iterations must be at least 1"
    );

    let plugin = load_plugin(plugin_path)?;
    let task = plugin.task().to_string();
    let input_dim = plugin.input_dim();
    let registry = Arc::new(ReflexRegistry::new());
    registry.register(plugin);

    let patch = patch_path.map(load_patch).transpose()?;
    let has_patch = patch.is_some();

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for t in 0..threads {
        let registry = Arc::clone(&registry);
        let task = task.clone();
        let barrier = Arc::clone(&barrier);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || -> anyhow::Result<Vec<u64>> {
            let mut rng_state: u32 = 0x9E37_79B9 ^ (t as u32).wrapping_mul(2654435761).max(1);
            let mut next_f32 = move || {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 17;
                rng_state ^= rng_state << 5;
                ((rng_state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
            };
            let x: Vec<f32> = (0..input_dim).map(|_| next_f32()).collect();
            let mut latencies_us = Vec::with_capacity(iterations_per_thread);
            barrier.wait();
            let mut i = 0;
            while i < iterations_per_thread || (has_patch && !stop.load(Ordering::Relaxed)) {
                let call_start = Instant::now();
                registry
                    .predict(&task, &x, None)
                    .context("predict during bench")?;
                latencies_us.push(call_start.elapsed().as_micros() as u64);
                i += 1;
                if i >= iterations_per_thread && !has_patch {
                    break;
                }
            }
            Ok(latencies_us)
        }));
    }

    barrier.wait();
    let bench_start = Instant::now();
    let swap_latency_us = if let Some(patch) = &patch {
        // Give readers a moment to actually be in-flight before the swap, so
        // the concurrent-hot-swap path is genuinely exercised, not raced
        // against threads still spinning up.
        std::thread::sleep(Duration::from_millis(5));
        let swap_start = Instant::now();
        registry
            .apply_patch_live(&task, patch)
            .context("apply_patch_live during bench")?;
        let elapsed = swap_start.elapsed().as_micros() as u64;
        stop.store(true, Ordering::Relaxed);
        Some(elapsed)
    } else {
        None
    };

    let mut all_latencies_us: Vec<u64> = Vec::new();
    for h in handles {
        let latencies = h
            .join()
            .map_err(|_| anyhow::anyhow!("bench reader thread panicked"))??;
        all_latencies_us.extend(latencies);
    }
    let wall_clock = bench_start.elapsed();
    all_latencies_us.sort_unstable();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "task": task,
            "input_dim": input_dim,
            "threads": threads,
            "total_predictions": all_latencies_us.len(),
            "wall_clock_ms": wall_clock.as_secs_f64() * 1000.0,
            "predict_latency_us": {
                "p50": percentile(&all_latencies_us, 0.50),
                "p90": percentile(&all_latencies_us, 0.90),
                "p99": percentile(&all_latencies_us, 0.99),
                "max": all_latencies_us.last().copied().unwrap_or(0),
            },
            "hot_swap_latency_us": swap_latency_us,
        }))?
    );
    Ok(())
}
