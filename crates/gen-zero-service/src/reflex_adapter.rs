//! `ReflexOnlineAdapter`: closes the loop between [`SqliteFeedbackStore`] and
//! [`ReflexRegistry`] — fetch labeled-but-unconsumed traces, take one batch
//! gradient step of multinomial logistic regression on the named head's
//! linear readout (`weight`, `bias`; the LoRA operator is frozen), verify a
//! fail-closed safety gate, hot-swap the patched plugin into the registry,
//! and only then mark the consumed rows trained.
//!
//! What this is, precisely: last-layer-only online SGD on softmax
//! cross-entropy loss, computed against
//! [`ReflexPlugin::restored_features`]'s exact output for each trace (the
//! `head_input` column `SqliteFeedbackStore` records for exactly this
//! reason). It is not a claim about the LoRA recurrence itself: `w_down`,
//! `w_ctx`, `w_gate`, `b_gate`, `w_cand`, `b_cand`, and `w_up` never change
//! here, only the named head's `weight`/`bias`.
//!
//! Ordering matters for crash safety: the patch is verified and hot-swapped
//! into the registry *before* the source rows are marked trained. A crash
//! between those two steps re-delivers the same batch next cycle (the rows
//! are still "unconsumed"), which lands on the *already-patched* plugin and
//! is caught by [`ReflexRegistry::apply_patch_live`]'s base-hash check the
//! moment it no longer matches — it never silently reapplies a stale patch or
//! loses the rows.

use crate::reflex_registry::{ReflexError, ReflexRegistry};
use gen_zero_model::reflex::{ReflexHead, ReflexOperator, ReflexPlugin};
use gen_zero_model::ReflexHeadDelta;
use gen_zero_storage::{ReflexFeedbackBatchItem, SqliteFeedbackStore};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Default batch step size. Small and conservative: this runs online, in
/// production, against whatever unconsumed feedback has accumulated, not in
/// an offline loop with a learning-rate schedule to tune.
pub const DEFAULT_LEARNING_RATE: f32 = 0.05;

/// What one successful [`ReflexOnlineAdapter::run_cycle`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptationReport {
    pub task: String,
    pub head: String,
    pub trace_ids: Vec<String>,
    /// Mean softmax cross-entropy loss over the batch, before the update.
    pub loss_before: f64,
    /// Mean softmax cross-entropy loss over the *same* batch, after the
    /// update (the safety gate requires `loss_after <= loss_before`).
    pub loss_after: f64,
    pub base_sha256: [u8; 32],
    pub target_sha256: [u8; 32],
}

/// Coordinates a [`SqliteFeedbackStore`] and a [`ReflexRegistry`] into one
/// online adaptation loop.
pub struct ReflexOnlineAdapter {
    store: SqliteFeedbackStore,
    registry: Arc<ReflexRegistry>,
    learning_rate: f32,
}

impl ReflexOnlineAdapter {
    pub fn new(
        store: SqliteFeedbackStore,
        registry: Arc<ReflexRegistry>,
        learning_rate: f32,
    ) -> Self {
        Self {
            store,
            registry,
            learning_rate,
        }
    }

    pub fn store(&self) -> &SqliteFeedbackStore {
        &self.store
    }

    pub fn registry(&self) -> &Arc<ReflexRegistry> {
        &self.registry
    }

    /// Run one adaptation cycle for `task`/`head_name`: fetch up to `limit`
    /// unconsumed feedback rows for `task`, take one batch gradient step on
    /// `head_name`, and hot-swap the result in if the safety gate passes.
    ///
    /// Returns `Ok(None)` when there is no unconsumed feedback for `task` (a
    /// no-op, not an error: an idle task is the normal steady state). Returns
    /// `Err(SafetyGateRejected)` when the update is well-formed but makes the
    /// batch worse or produces a non-finite weight; in that case nothing is
    /// hot-swapped and nothing is marked trained, so the same rows are picked
    /// up again next cycle (by a human relabeling them, or a future cycle
    /// with more data diluting a bad batch).
    pub fn run_cycle(
        &self,
        task: &str,
        head_name: &str,
        limit: usize,
    ) -> Result<Option<AdaptationReport>, ReflexError> {
        let batch = self.store.fetch_unconsumed(limit)?;
        let items: Vec<ReflexFeedbackBatchItem> = batch
            .into_iter()
            .filter(|b| b.task == task && b.head == head_name)
            .collect();
        if items.is_empty() {
            return Ok(None);
        }

        let base = self
            .registry
            .route(task)
            .ok_or_else(|| ReflexError::UnknownTask(task.to_string()))?;
        let head =
            base.heads()
                .find(|h| h.name == head_name)
                .ok_or_else(|| ReflexError::UnknownHead {
                    task: task.to_string(),
                    head: head_name.to_string(),
                })?;
        let input_dim = base.input_dim();
        let k = head.candidates.len();

        let mut examples: Vec<(Vec<f32>, usize)> = Vec::with_capacity(items.len());
        for item in &items {
            let class_idx = head
                .candidates
                .iter()
                .position(|c| c == &item.feedback_label)
                .ok_or_else(|| ReflexError::LabelNotInCandidates {
                    head: head_name.to_string(),
                    label: item.feedback_label.clone(),
                })?;
            examples.push((item.head_input.clone(), class_idx));
        }

        let loss_before = batch_loss(&head.weight, &head.bias, k, input_dim, &examples);
        let (delta_weight, delta_bias) = gradient_step(
            &head.weight,
            &head.bias,
            k,
            input_dim,
            &examples,
            self.learning_rate,
        );

        if !delta_weight.iter().all(|v| v.is_finite()) || !delta_bias.iter().all(|v| v.is_finite())
        {
            return Err(ReflexError::SafetyGateRejected {
                task: task.to_string(),
                reason: "gradient step produced a non-finite delta".to_string(),
            });
        }

        let new_weight: Vec<f32> = head
            .weight
            .iter()
            .zip(&delta_weight)
            .map(|(&w, &d)| w + d)
            .collect();
        let new_bias: Vec<f32> = head
            .bias
            .iter()
            .zip(&delta_bias)
            .map(|(&b, &d)| b + d)
            .collect();
        let loss_after = batch_loss(&new_weight, &new_bias, k, input_dim, &examples);

        if !loss_after.is_finite() || loss_after > loss_before {
            return Err(ReflexError::SafetyGateRejected {
                task: task.to_string(),
                reason: format!(
                    "batch loss regressed on its own training batch: {loss_before} -> {loss_after}"
                ),
            });
        }

        // Build the target plugin: every other head and the whole operator
        // carried over unchanged (via public constructors, not the crate's
        // internal in-place mutators), only `head_name` replaced.
        let new_head = ReflexHead::new(
            head_name,
            input_dim,
            new_weight,
            new_bias,
            head.candidates.clone(),
        )?;
        let mut new_heads = Vec::with_capacity(base.heads().count());
        for h in base.heads() {
            if h.name == head_name {
                new_heads.push(new_head.clone());
            } else {
                new_heads.push(ReflexHead::new(
                    h.name.clone(),
                    input_dim,
                    h.weight.clone(),
                    h.bias.clone(),
                    h.candidates.clone(),
                )?);
            }
        }
        let ops = base.operator().weight_arrays();
        let unchanged_operator = ReflexOperator::new(
            base.operator().config(),
            ops[0].to_vec(),
            ops[1].to_vec(),
            ops[2].to_vec(),
            ops[3].to_vec(),
            ops[4].to_vec(),
            ops[5].to_vec(),
            ops[6].to_vec(),
        )?;
        let target_plugin = ReflexPlugin::new(
            base.task().to_string(),
            unchanged_operator,
            new_heads,
            base.default_head().map(String::from),
        )?;

        let base_sha256 = base.sha256()?;
        let target_sha256 = target_plugin.sha256()?;

        let mut head_deltas = BTreeMap::new();
        head_deltas.insert(
            head_name.to_string(),
            ReflexHeadDelta {
                delta_weight,
                delta_bias,
                weight_fixups: Vec::new(),
                bias_fixups: Vec::new(),
            },
        );

        let patch = gen_zero_model::ReflexPatch {
            base_sha256,
            target_sha256,
            metadata: format!(
                "online-adapt task={task} head={head_name} n={} lr={}",
                examples.len(),
                self.learning_rate
            ),
            delta_w_down: vec![0.0; ops[0].len()],
            delta_w_ctx: vec![0.0; ops[1].len()],
            delta_w_gate: vec![0.0; ops[2].len()],
            delta_b_gate: vec![0.0; ops[3].len()],
            delta_w_cand: vec![0.0; ops[4].len()],
            delta_b_cand: vec![0.0; ops[5].len()],
            delta_w_up: vec![0.0; ops[6].len()],
            operator_fixups: Default::default(),
            head_deltas,
        };

        // Hot-swap before marking trained: see the module doc's crash-safety
        // note on why this order is load-bearing, not incidental.
        self.registry.apply_patch_live(task, &patch)?;

        let trace_ids: Vec<String> = items.iter().map(|i| i.trace_id.clone()).collect();
        self.store.mark_trained(&trace_ids)?;

        Ok(Some(AdaptationReport {
            task: task.to_string(),
            head: head_name.to_string(),
            trace_ids,
            loss_before,
            loss_after,
            base_sha256,
            target_sha256,
        }))
    }
}

// ---------------------------------------------------------------------------
// Plain softmax cross-entropy gradient, computed head-locally.
//
// Deliberately not a call into `gen_zero_model::reflex`'s private `softmax`:
// that function is not exported (`reflex.rs` keeps its numerical primitives
// module-private), and this is a small, independently-defined, independently
// testable primitive, not a copy that risks drifting from a hidden contract.
// ---------------------------------------------------------------------------

fn head_logits(weight: &[f32], bias: &[f32], k: usize, d: usize, x: &[f32]) -> Vec<f32> {
    (0..k)
        .map(|row| {
            let w = &weight[row * d..(row + 1) * d];
            let dot: f32 = w.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            dot + bias[row]
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Mean softmax cross-entropy loss `-log(p[y])` over `examples`, in `f64` for
/// numerically stable accumulation across a batch.
fn batch_loss(
    weight: &[f32],
    bias: &[f32],
    k: usize,
    d: usize,
    examples: &[(Vec<f32>, usize)],
) -> f64 {
    let mut total = 0.0f64;
    for (x, y) in examples {
        let probs = softmax(&head_logits(weight, bias, k, d, x));
        total += -(probs[*y].max(1e-12) as f64).ln();
    }
    total / examples.len().max(1) as f64
}

/// One batch gradient descent step: `dW = mean((p - onehot(y)) . x^T)`,
/// `db = mean(p - onehot(y))`, `delta = -lr * grad`. Returns `(delta_weight,
/// delta_bias)` in the same row-major `(K, d)` / `(K,)` layout as
/// [`gen_zero_model::reflex::ReflexHead`].
fn gradient_step(
    weight: &[f32],
    bias: &[f32],
    k: usize,
    d: usize,
    examples: &[(Vec<f32>, usize)],
    learning_rate: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = examples.len().max(1) as f32;
    let mut grad_w = vec![0.0f32; k * d];
    let mut grad_b = vec![0.0f32; k];
    for (x, y) in examples {
        let probs = softmax(&head_logits(weight, bias, k, d, x));
        for row in 0..k {
            let dz = probs[row] - if row == *y { 1.0 } else { 0.0 };
            grad_b[row] += dz;
            let w_row = &mut grad_w[row * d..(row + 1) * d];
            for (j, &xj) in x.iter().enumerate() {
                w_row[j] += dz * xj;
            }
        }
    }
    let delta_weight: Vec<f32> = grad_w.iter().map(|&g| -learning_rate * g / n).collect();
    let delta_bias: Vec<f32> = grad_b.iter().map(|&g| -learning_rate * g / n).collect();
    (delta_weight, delta_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_model::reflex::ReflexOperatorConfig;
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

    fn build_plugin(task: &str, input_dim: usize, seed: u32) -> ReflexPlugin {
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

    fn open_temp_store(input_dim: usize) -> (TempDir, SqliteFeedbackStore) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("feedback.sqlite3");
        let store = SqliteFeedbackStore::open(&path, input_dim).unwrap();
        (dir, store)
    }

    /// Insert one trace by actually calling `predict`/`restored_features` on
    /// `plugin`, the exact way a real serving call site would, then attach
    /// feedback. This is the full, honest loop: predict -> record -> label,
    /// not a hand-built fixture that assumes the shape of a trace.
    fn record_labeled_trace(
        store: &SqliteFeedbackStore,
        plugin: &ReflexPlugin,
        trace_id: &str,
        x: &[f32],
        true_label: &str,
        created_at_ms: i64,
    ) {
        let decision = plugin.predict(x, None).unwrap();
        let (head_input, _telemetry) = plugin.restored_features(x).unwrap();
        store
            .insert_trace(&gen_zero_storage::ReflexTraceRecord {
                trace_id: trace_id.to_string(),
                created_at_unix_ms: created_at_ms,
                task: plugin.task().to_string(),
                head: decision.head.clone(),
                plugin_version: "v1".to_string(),
                z0: x.to_vec(),
                head_input,
                pred_label: decision.label.clone(),
                pred_confidence: decision.confidence as f64,
                max_gamma: decision.telemetry.max_gamma.map(|g| g as f64),
            })
            .unwrap();
        store
            .record_feedback(trace_id, true_label, "test_harness", created_at_ms + 1)
            .unwrap();
    }

    #[test]
    fn run_cycle_is_none_when_no_unconsumed_feedback() {
        let (_dir, store) = open_temp_store(8);
        let registry = Arc::new(ReflexRegistry::new());
        registry.register(build_plugin("t1", 8, 1));
        let adapter = ReflexOnlineAdapter::new(store, registry, DEFAULT_LEARNING_RATE);
        assert!(adapter.run_cycle("t1", "main", 100).unwrap().is_none());
    }

    #[test]
    fn run_cycle_rejects_label_outside_head_candidates() {
        let (_dir, store) = open_temp_store(8);
        let plugin = build_plugin("t1", 8, 2);
        let mut rng = Xorshift32(55);
        let x = rng.vec(8);
        record_labeled_trace(&store, &plugin, "t1", &x, "not-a-real-label", 100);

        let registry = Arc::new(ReflexRegistry::new());
        registry.register(plugin);
        let adapter = ReflexOnlineAdapter::new(store, registry, DEFAULT_LEARNING_RATE);
        let err = adapter.run_cycle("t1", "main", 100).unwrap_err();
        assert!(matches!(err, ReflexError::LabelNotInCandidates { .. }));
    }

    /// End-to-end acceptance test: SQLite feedback record -> fetch -> online
    /// update -> patch apply -> registry hot-swap -> verified predict output.
    ///
    /// Feeds the same, correctly-labeled example enough times that a
    /// gradient step measurably sharpens the head's confidence on it (a real,
    /// checkable learning effect, not just "the code ran without error").
    #[test]
    fn end_to_end_feedback_to_hot_swapped_prediction() {
        let input_dim = 12;
        let (_dir, store) = open_temp_store(input_dim);
        let plugin = build_plugin("classify", input_dim, 99);
        let mut rng = Xorshift32(2024);
        let x = rng.vec(input_dim);

        let pre_decision = plugin.predict(&x, None).unwrap();
        // Whichever label the head favors, drive training toward it hard so
        // the direction of the effect is unambiguous either way.
        let target_label = pre_decision.label.clone();
        let target_idx = pre_decision.class_index;

        for i in 0..20 {
            record_labeled_trace(
                &store,
                &plugin,
                &format!("trace-{i}"),
                &x,
                &target_label,
                1_700_000_000_000 + i,
            );
        }

        let registry = Arc::new(ReflexRegistry::new());
        registry.register(plugin.clone());
        let base_sha256 = plugin.sha256().unwrap();

        let adapter = ReflexOnlineAdapter::new(store, registry.clone(), 0.5);
        let report = adapter
            .run_cycle("classify", "main", 100)
            .unwrap()
            .expect("20 unconsumed labeled traces must yield a cycle");

        assert_eq!(report.trace_ids.len(), 20);
        assert_eq!(report.base_sha256, base_sha256);
        assert!(report.loss_after <= report.loss_before);

        let swapped = registry.route("classify").unwrap();
        assert_eq!(swapped.sha256().unwrap(), report.target_sha256);
        assert_ne!(swapped.sha256().unwrap(), base_sha256);

        // The hot-swapped plugin must now be *more* confident in the label it
        // was just trained toward, on the exact same input: a real, checkable
        // learning effect, not merely "some weights changed".
        let post_decision = swapped.predict(&x, None).unwrap();
        assert_eq!(post_decision.class_index, target_idx);
        assert!(
            post_decision.probabilities[target_idx] >= pre_decision.probabilities[target_idx],
            "post-update confidence {} should be >= pre-update confidence {}",
            post_decision.probabilities[target_idx],
            pre_decision.probabilities[target_idx]
        );

        // Rows must be marked trained: a second cycle with no new feedback
        // is a no-op, not a re-application of the same patch.
        assert!(adapter
            .run_cycle("classify", "main", 100)
            .unwrap()
            .is_none());
    }

    /// Proves the safety gate actually rejects, not merely that the code path
    /// exists: an absurd learning rate drives the gradient delta to
    /// non-finite values, and the gate must refuse to hot-swap or mark
    /// anything trained. A gate that only ever observes a passing update is
    /// unverified, not fail-closed.
    #[test]
    fn run_cycle_rejects_non_finite_update_from_a_pathological_learning_rate_and_changes_nothing() {
        let input_dim = 8;
        let (_dir, store) = open_temp_store(input_dim);
        let plugin = build_plugin("unstable", input_dim, 11);
        let mut rng = Xorshift32(4321);
        let x = rng.vec(input_dim);
        record_labeled_trace(&store, &plugin, "trace-0", &x, "a", 100);

        let registry = Arc::new(ReflexRegistry::new());
        registry.register(plugin.clone());
        let base_sha256 = plugin.sha256().unwrap();

        let adapter = ReflexOnlineAdapter::new(store, Arc::clone(&registry), f32::MAX);
        let err = adapter.run_cycle("unstable", "main", 100).unwrap_err();
        assert!(
            matches!(err, ReflexError::SafetyGateRejected { .. }),
            "{err:?}"
        );

        // Registry must be untouched: same plugin, same hash.
        assert_eq!(
            registry.route("unstable").unwrap().sha256().unwrap(),
            base_sha256
        );
        // The row must not have been marked trained: it is still fetchable.
        assert_eq!(adapter.store().fetch_unconsumed(10).unwrap().len(), 1);
    }

    #[test]
    fn gradient_step_reduces_loss_on_a_toy_batch() {
        let d = 4;
        let k = 2;
        let weight = vec![0.0f32; k * d];
        let bias = vec![0.0f32; k];
        let examples = vec![
            (vec![1.0, 0.0, 0.0, 0.0], 0usize),
            (vec![0.0, 1.0, 0.0, 0.0], 0usize),
            (vec![0.0, 0.0, 1.0, 0.0], 1usize),
            (vec![0.0, 0.0, 0.0, 1.0], 1usize),
        ];
        let before = batch_loss(&weight, &bias, k, d, &examples);
        let (dw, db) = gradient_step(&weight, &bias, k, d, &examples, 1.0);
        let new_weight: Vec<f32> = weight.iter().zip(&dw).map(|(&w, &d)| w + d).collect();
        let new_bias: Vec<f32> = bias.iter().zip(&db).map(|(&b, &d)| b + d).collect();
        let after = batch_loss(&new_weight, &new_bias, k, d, &examples);
        assert!(after < before, "{after} should be < {before}");
    }
}
