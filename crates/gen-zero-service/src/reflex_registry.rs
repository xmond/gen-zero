//! `ReflexRegistry`: a lock-free, hot-swappable serving table of
//! [`ReflexPlugin`]s keyed by task name.
//!
//! Structure mirrors [`crate::mount::AtomicMountRegistry`]: an outer
//! `ArcSwap<HashMap<String, SharedCell>>` maps a task to a per-task
//! `ArcSwap<ReflexPlugin>` cell. Registering a new task replaces the outer
//! map (rare; a CAS-retry loop, since the whole table is cloned); routing and
//! predicting only ever load the outer map and then the per-task cell, both
//! plain atomic loads with no lock. Hot-swapping a plugin (`compare_and_swap`,
//! `apply_patch_live`) only ever touches the per-task cell, so predicting on
//! task A never contends with a patch landing on task B.
//!
//! Every swap is copy-on-write: a reader that has already loaded an `Arc`
//! keeps serving a complete, self-consistent plugin for the lifetime of its
//! call, even if a patch lands mid-flight. There is no "half patched" state a
//! concurrent reader can observe.

use gen_zero_model::reflex::{ReflexDecision, ReflexPlugin};
use gen_zero_model::{ModelError, ReflexPatch};
use gen_zero_storage::StorageError;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use arc_swap::{ArcSwap, Guard};

/// Errors from routing, predicting, or hot-swapping a [`ReflexRegistry`], or
/// from the online adaptation cycle built on top of it.
#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("no reflex plugin registered for task {0:?}")]
    UnknownTask(String),
    #[error("task {task:?} has no head named {head:?}")]
    UnknownHead { task: String, head: String },
    #[error(
        "compare_and_swap on task {task:?} expected sha256 {expected}, registry holds {actual}"
    )]
    StaleBase {
        task: String,
        expected: String,
        actual: String,
    },
    #[error("registry CAS contention on task {0:?} exceeded the retry budget")]
    CasContention(String),
    #[error("online adaptation safety gate rejected the update for task {task:?}: {reason}")]
    SafetyGateRejected { task: String, reason: String },
    #[error("feedback label {label:?} is not among head {head:?}'s candidates")]
    LabelNotInCandidates { head: String, label: String },
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Storage(#[from] StorageError),
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

type SharedCell = Arc<ArcSwap<ReflexPlugin>>;

/// Swap `new` into `cell` only if it still holds exactly `expected` (pointer
/// identity, not value equality). Returns whether the swap took effect.
fn swap_if_current<T>(cell: &ArcSwap<T>, expected: &Arc<T>, new: Arc<T>) -> bool {
    let previous: Arc<T> = Guard::into_inner(cell.compare_and_swap(expected, new));
    Arc::ptr_eq(&previous, expected)
}

/// Bound on retrying a CAS loop against contention from other writers on the
/// same task. Not a correctness mechanism (a legitimate base mismatch fails
/// closed immediately, it never loops); this only guards against an
/// unbounded loop under a pathological, permanently-contended writer storm.
const MAX_CAS_ATTEMPTS: usize = 1024;

/// Lock-free registry of [`ReflexPlugin`]s, keyed by task.
#[derive(Default)]
pub struct ReflexRegistry {
    cells: ArcSwap<HashMap<String, SharedCell>>,
}

impl ReflexRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) the plugin serving `plugin.task()`. Always
    /// succeeds: this is the initial-load / full-reload path, not a
    /// version-checked hot-swap (use [`Self::compare_and_swap`] or
    /// [`Self::apply_patch_live`] for that).
    pub fn register(&self, plugin: ReflexPlugin) {
        let task = plugin.task().to_string();
        let cell: SharedCell = Arc::new(ArcSwap::from_pointee(plugin));
        loop {
            let current = self.cells.load_full();
            let mut table = HashMap::clone(&current);
            table.insert(task.clone(), Arc::clone(&cell));
            if swap_if_current(&self.cells, &current, Arc::new(table)) {
                return;
            }
        }
    }

    fn cell(&self, task: &str) -> Result<SharedCell, ReflexError> {
        self.cells
            .load()
            .get(task)
            .cloned()
            .ok_or_else(|| ReflexError::UnknownTask(task.to_string()))
    }

    /// Look up the plugin currently serving `task`, if any. A lock-free
    /// atomic load: no mutex, no blocking on a concurrent writer.
    pub fn route(&self, task: &str) -> Option<Arc<ReflexPlugin>> {
        self.cells.load().get(task).map(|cell| cell.load_full())
    }

    /// Route to `task`'s plugin and run `predict`. The routing load and the
    /// plugin's own `predict` together are the sub-microsecond hot path this
    /// registry exists for.
    pub fn predict(
        &self,
        task: &str,
        input: &[f32],
        head: Option<&str>,
    ) -> Result<ReflexDecision, ReflexError> {
        let plugin = self
            .route(task)
            .ok_or_else(|| ReflexError::UnknownTask(task.to_string()))?;
        Ok(plugin.predict(input, head)?)
    }

    /// Replace `task`'s plugin with `new_plugin`, but only if the plugin
    /// currently registered for `task` hashes to `expected_sha256`. A single
    /// attempt: unlike `apply_patch_live`, a fixed `new_plugin` was computed
    /// against one specific expected base, so a mismatch (whether from a
    /// genuinely stale caller or from a concurrent writer winning the race)
    /// is reported, never silently retried against a different base.
    pub fn compare_and_swap(
        &self,
        task: &str,
        expected_sha256: [u8; 32],
        new_plugin: ReflexPlugin,
    ) -> Result<(), ReflexError> {
        let cell = self.cell(task)?;
        let current = cell.load_full();
        let current_hash = current.sha256()?;
        if current_hash != expected_sha256 {
            return Err(ReflexError::StaleBase {
                task: task.to_string(),
                expected: hex(&expected_sha256),
                actual: hex(&current_hash),
            });
        }
        if swap_if_current(&cell, &current, Arc::new(new_plugin)) {
            Ok(())
        } else {
            Err(ReflexError::StaleBase {
                task: task.to_string(),
                expected: hex(&expected_sha256),
                actual: "concurrently replaced during swap".to_string(),
            })
        }
    }

    /// Apply `patch` to `task`'s currently registered plugin and hot-swap the
    /// result in, atomically and with zero downtime: every `predict` call in
    /// flight keeps running against whichever plugin it already loaded, and
    /// every call after the swap sees the fully patched plugin. Never the
    /// reverse: no reader ever observes a partially patched plugin.
    ///
    /// `ReflexPlugin::apply_patch` (not `apply_patch_in_place`) does the
    /// actual patching: it clones the weight buffers, verifies
    /// `patch.base_sha256` against the loaded plugin and `patch.target_sha256`
    /// against its own output, and never touches the loaded `Arc`'s plugin in
    /// place. Retries only when a concurrent writer swapped the cell between
    /// our load and our compare-and-swap (spurious CAS loss); a genuine base
    /// mismatch (this patch was computed against a different version than
    /// whatever is live now) fails closed immediately, on the first or any
    /// later attempt, without retrying further.
    pub fn apply_patch_live(
        &self,
        task: &str,
        patch: &ReflexPatch,
    ) -> Result<Arc<ReflexPlugin>, ReflexError> {
        let cell = self.cell(task)?;
        for _ in 0..MAX_CAS_ATTEMPTS {
            let current = cell.load_full();
            let new_plugin = Arc::new(current.apply_patch(patch)?);
            if swap_if_current(&cell, &current, Arc::clone(&new_plugin)) {
                return Ok(new_plugin);
            }
        }
        Err(ReflexError::CasContention(task.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_model::reflex::{ReflexHead, ReflexOperator, ReflexOperatorConfig};
    use gen_zero_model::ReflexHeadDelta;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::thread;

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

    fn build_plugin(task: &str, seed: u32) -> ReflexPlugin {
        let input_dim = 16;
        let rank = 4;
        let mut rng = Xorshift32(seed);
        let config = ReflexOperatorConfig {
            input_dim,
            lora_rank: rank,
            alpha: 0.5,
            steps: 3,
            epsilon: 1e-6,
        };
        let operator = ReflexOperator::new(
            config,
            rng.vec(input_dim * rank),
            rng.vec(input_dim * rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(rank * input_dim),
        )
        .unwrap();
        let head = ReflexHead::new(
            "main",
            input_dim,
            rng.vec(2 * input_dim),
            rng.vec(2),
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        ReflexPlugin::new(task, operator, vec![head], Some("main".into())).unwrap()
    }

    /// Build a patch that bumps every weight by a small, fixed amount via
    /// plain scalar addition, then constructs the target the exact same way,
    /// so `base + delta == target` bit-exact without any fix-ups.
    fn bump_patch(base: &ReflexPlugin, bump: f32) -> (ReflexPlugin, ReflexPatch) {
        let ops = base.operator().weight_arrays();
        let bumped: Vec<Vec<f32>> = ops
            .iter()
            .map(|a| a.iter().map(|&v| v + bump).collect())
            .collect();
        let target_operator = ReflexOperator::new(
            base.operator().config(),
            bumped[0].clone(),
            bumped[1].clone(),
            bumped[2].clone(),
            bumped[3].clone(),
            bumped[4].clone(),
            bumped[5].clone(),
            bumped[6].clone(),
        )
        .unwrap();
        let mut head_deltas = BTreeMap::new();
        let mut target_heads = Vec::new();
        for head in base.heads() {
            let delta_weight = vec![bump; head.weight.len()];
            let delta_bias = vec![bump; head.bias.len()];
            let new_weight: Vec<f32> = head.weight.iter().map(|&w| w + bump).collect();
            let new_bias: Vec<f32> = head.bias.iter().map(|&b| b + bump).collect();
            target_heads.push(
                ReflexHead::new(
                    head.name.clone(),
                    base.input_dim(),
                    new_weight,
                    new_bias,
                    head.candidates.clone(),
                )
                .unwrap(),
            );
            head_deltas.insert(
                head.name.clone(),
                ReflexHeadDelta {
                    delta_weight,
                    delta_bias,
                    weight_fixups: Vec::new(),
                    bias_fixups: Vec::new(),
                },
            );
        }
        let target = ReflexPlugin::new(
            base.task().to_string(),
            target_operator,
            target_heads,
            base.default_head().map(String::from),
        )
        .unwrap();
        let patch = ReflexPatch {
            base_sha256: base.sha256().unwrap(),
            target_sha256: target.sha256().unwrap(),
            metadata: "test bump".into(),
            delta_w_down: vec![bump; ops[0].len()],
            delta_w_ctx: vec![bump; ops[1].len()],
            delta_w_gate: vec![bump; ops[2].len()],
            delta_b_gate: vec![bump; ops[3].len()],
            delta_w_cand: vec![bump; ops[4].len()],
            delta_b_cand: vec![bump; ops[5].len()],
            delta_w_up: vec![bump; ops[6].len()],
            operator_fixups: Default::default(),
            head_deltas,
        };
        (target, patch)
    }

    #[test]
    fn route_returns_none_for_unknown_task() {
        let registry = ReflexRegistry::new();
        assert!(registry.route("nope").is_none());
    }

    #[test]
    fn register_then_route_returns_the_plugin() {
        let registry = ReflexRegistry::new();
        registry.register(build_plugin("t1", 1));
        let plugin = registry.route("t1").unwrap();
        assert_eq!(plugin.task(), "t1");
    }

    #[test]
    fn predict_on_unknown_task_fails_closed() {
        let registry = ReflexRegistry::new();
        let err = registry.predict("nope", &[0.0; 16], None).unwrap_err();
        assert!(matches!(err, ReflexError::UnknownTask(_)));
    }

    #[test]
    fn predict_routes_and_matches_direct_plugin_call() {
        let registry = ReflexRegistry::new();
        let plugin = build_plugin("t1", 2);
        let mut rng = Xorshift32(99);
        let x = rng.vec(16);
        let direct = plugin.predict(&x, None).unwrap();
        registry.register(plugin);
        let routed = registry.predict("t1", &x, None).unwrap();
        assert_eq!(direct.label, routed.label);
        assert_eq!(direct.probabilities, routed.probabilities);
    }

    #[test]
    fn compare_and_swap_replaces_on_matching_hash() {
        let registry = ReflexRegistry::new();
        let base = build_plugin("t1", 3);
        let base_hash = base.sha256().unwrap();
        registry.register(base);
        let replacement = build_plugin("t1", 4);
        registry
            .compare_and_swap("t1", base_hash, replacement.clone())
            .unwrap();
        assert_eq!(
            registry.route("t1").unwrap().sha256().unwrap(),
            replacement.sha256().unwrap()
        );
    }

    #[test]
    fn compare_and_swap_fails_closed_on_stale_hash() {
        let registry = ReflexRegistry::new();
        registry.register(build_plugin("t1", 5));
        let wrong_hash = [0xAB; 32];
        let err = registry
            .compare_and_swap("t1", wrong_hash, build_plugin("t1", 6))
            .unwrap_err();
        assert!(matches!(err, ReflexError::StaleBase { .. }));
    }

    #[test]
    fn apply_patch_live_hot_swaps_and_is_bit_exact() {
        let registry = ReflexRegistry::new();
        let base = build_plugin("t1", 7);
        let base_task = base.task().to_string();
        registry.register(base.clone());
        let (target, patch) = bump_patch(&base, 0.01);
        let swapped = registry.apply_patch_live(&base_task, &patch).unwrap();
        assert_eq!(swapped.sha256().unwrap(), target.sha256().unwrap());
        assert_eq!(
            registry.route(&base_task).unwrap().sha256().unwrap(),
            target.sha256().unwrap()
        );
    }

    #[test]
    fn apply_patch_live_fails_closed_on_wrong_base() {
        let registry = ReflexRegistry::new();
        let base = build_plugin("t1", 8);
        registry.register(base.clone());
        let (_target, mut patch) = bump_patch(&base, 0.01);
        patch.base_sha256[0] ^= 0xFF;
        let err = registry.apply_patch_live("t1", &patch).unwrap_err();
        assert!(matches!(
            err,
            ReflexError::Model(ModelError::ReflexPatch(_))
        ));
        // Registry must be untouched: same hash as before the failed patch.
        assert_eq!(
            registry.route("t1").unwrap().sha256().unwrap(),
            base.sha256().unwrap()
        );
    }

    /// Concurrent hot-swap acceptance test: reader threads predict in a tight
    /// loop while a writer thread applies a live patch. Every reader
    /// observation must be bit-exact for either the base or the target
    /// plugin, never a mix, and no thread may panic.
    #[test]
    fn concurrent_predict_during_apply_patch_live_never_panics_and_is_bit_exact() {
        let registry = Arc::new(ReflexRegistry::new());
        let base = build_plugin("hot", 42);
        registry.register(base.clone());
        let (target, patch) = bump_patch(&base, 0.02);

        let base_hash = base.sha256().unwrap();
        let target_hash = target.sha256().unwrap();

        let mut rng = Xorshift32(777);
        let inputs: Vec<Vec<f32>> = (0..8).map(|_| rng.vec(16)).collect();
        let base_outputs: Vec<ReflexDecision> = inputs
            .iter()
            .map(|x| base.predict(x, None).unwrap())
            .collect();
        let target_outputs: Vec<ReflexDecision> = inputs
            .iter()
            .map(|x| target.predict(x, None).unwrap())
            .collect();

        let readers_ready = Arc::new(Barrier::new(9));
        let stop = Arc::new(AtomicBool::new(false));
        let mixed_observations = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let registry = Arc::clone(&registry);
            let inputs = inputs.clone();
            let base_outputs = base_outputs.clone();
            let target_outputs = target_outputs.clone();
            let readers_ready = Arc::clone(&readers_ready);
            let stop = Arc::clone(&stop);
            let mixed_observations = Arc::clone(&mixed_observations);
            handles.push(thread::spawn(move || {
                readers_ready.wait();
                while !stop.load(Ordering::Relaxed) {
                    for (i, x) in inputs.iter().enumerate() {
                        let decision = registry.predict("hot", x, None).unwrap();
                        let matches_base = decision.probabilities == base_outputs[i].probabilities;
                        let matches_target =
                            decision.probabilities == target_outputs[i].probabilities;
                        if !matches_base && !matches_target {
                            mixed_observations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }));
        }

        readers_ready.wait();
        let swap_start = std::time::Instant::now();
        let swapped = registry.apply_patch_live("hot", &patch).unwrap();
        let swap_elapsed = swap_start.elapsed();
        assert_eq!(swapped.sha256().unwrap(), target_hash);
        assert_ne!(swapped.sha256().unwrap(), base_hash);

        // A few more predicts post-swap must be exactly the target's output.
        for (i, x) in inputs.iter().enumerate() {
            let decision = registry.predict("hot", x, None).unwrap();
            assert_eq!(decision.probabilities, target_outputs[i].probabilities);
        }

        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            mixed_observations.load(Ordering::Relaxed),
            0,
            "every observed prediction must exactly match either the base or the target plugin"
        );
        // Debug builds (and a loaded CI box) are not representative of
        // production latency, so only hold the sub-millisecond SLA in
        // release mode, same convention as gen-zero-model's own
        // `production_scale_8192_dim_runs_and_converges_or_exhausts_budget`.
        // `reflex-bench` (gen-zero-cli) is the release-mode evidence source
        // for this number outside the test suite.
        if cfg!(debug_assertions) {
            assert!(
                swap_elapsed.as_millis() < 50,
                "hot-swap took {swap_elapsed:?}, expected a lock-free pointer swap to be fast"
            );
        } else {
            assert!(
                swap_elapsed.as_micros() < 1000,
                "hot-swap took {swap_elapsed:?}, expected sub-millisecond under release optimizations"
            );
        }
    }
}
