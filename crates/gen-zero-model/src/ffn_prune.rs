//! Physical SwiGLU FFN channel pruning for [`QwenModel`].
//!
//! A Qwen2 MLP computes `down(silu(gate(x)) * up(x))` with `gate`, `up` of
//! shape `[I, H]` and `down` of shape `[H, I]`. Channel `c` contributes
//! `down[:, c] * silu(gate[c] x) * up[c] x`, so dropping it means deleting
//! row `c` of `gate` and `up` and column `c` of `down`. Pruning here does
//! exactly that: the kept rows and columns are copied into new contiguous
//! `[K, H]` and `[H, K]` matrices and the old ones are freed, so memory,
//! activation width and FFN multiply-accumulates all shrink to `K / I`.
//!
//! Channel importance has two sources (ported from
//! `python/gen_zero/causal/qwen_inter_layer_mcts.py`, `score_ffn_channels`,
//! plus a weight-only variant the task specification adds):
//! * activation: `I[c] = E_tokens |silu(gate x)_c * (up x)_c| * ||down[:, c]||_2`
//!   over full forward passes of anchor prompts;
//! * static weights: `I[c] = (||gate[c, :]|| * ||up[c, :]||)^(1/2) * ||down[:, c]||`,
//!   which needs no data and is what `GENZERO_QWEN_FFN_KEEP_RATIO` uses at load.
//!
//! Pruning is lossy: it changes every score the model produces. Callers that
//! cache by model id must include the keep ratio in the id.

use crate::error::ModelError;
use crate::qwen::{candle_err, QwenModel};
use candle_core::Tensor;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

/// Channel importance per layer.
pub type FfnImportance = BTreeMap<usize, Vec<f64>>;
/// Kept channel indices per layer, strictly ascending.
pub type FfnChannelMasks = BTreeMap<usize, Vec<usize>>;

/// What [`QwenModel::prune_ffn_channels`] changed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FfnPruneReport {
    /// Requested keep ratio, when the masks came from one.
    pub keep_ratio: Option<f64>,
    /// `activation` or `weight_norm`, when the masks came from a scorer.
    pub importance: Option<String>,
    pub layers: Vec<usize>,
    /// Channels per pruned layer before and after.
    pub channels_before: Vec<usize>,
    pub channels_after: Vec<usize>,
    /// FFN weight parameters of the pruned layers before and after
    /// (`3 * H * channels` each).
    pub ffn_params_before: u64,
    pub ffn_params_after: u64,
    /// f32 bytes released by the slicing.
    pub bytes_freed: u64,
}

/// The upper layers pruned by default: `22..28` for the 28-layer Qwen2.5-1.5B,
/// the same `11 / 14` depth fraction for other depths.
pub fn default_upper_ffn_layers(num_layers: usize) -> Range<usize> {
    let start = (num_layers * 11 + 7) / 14;
    start.min(num_layers)..num_layers
}

/// Top `round(keep_ratio * I)` channels (at least one) of every layer, by
/// descending importance with ties kept in index order, returned ascending.
pub fn select_top_ffn_channels(
    importance: &FfnImportance,
    keep_ratio: f64,
) -> Result<FfnChannelMasks, ModelError> {
    check_keep_ratio(keep_ratio)?;
    importance
        .iter()
        .map(|(&layer, imp)| {
            if imp.is_empty() || imp.iter().any(|v| !v.is_finite()) {
                return Err(ModelError::QwenInput(format!(
                    "layer {layer}: FFN importance must be a nonempty finite vector"
                )));
            }
            let k = ((keep_ratio * imp.len() as f64).round() as usize).max(1);
            let mut order: Vec<usize> = (0..imp.len()).collect();
            // Stable sort on descending importance: equal scores keep index order.
            order.sort_by(|&a, &b| imp[b].total_cmp(&imp[a]));
            let mut kept = order[..k].to_vec();
            kept.sort_unstable();
            Ok((layer, kept))
        })
        .collect()
}

fn check_keep_ratio(keep_ratio: f64) -> Result<(), ModelError> {
    if keep_ratio.is_finite() && keep_ratio > 0.0 && keep_ratio <= 1.0 {
        Ok(())
    } else {
        Err(ModelError::QwenInput(format!(
            "FFN keep ratio {keep_ratio} must be finite and in (0, 1]"
        )))
    }
}

fn check_layers(model: &QwenModel, layers: &[usize]) -> Result<BTreeSet<usize>, ModelError> {
    let n = model.config().num_layers;
    let set: BTreeSet<usize> = layers.iter().copied().collect();
    if set.is_empty() || set.iter().any(|&l| l >= n) {
        return Err(ModelError::QwenInput(format!(
            "FFN layers {layers:?} must be nonempty and index the {n} decoder layers"
        )));
    }
    Ok(set)
}

/// L2 norm of every row (`dim = 1`) or column (`dim = 0`) of a matrix, in f64.
fn axis_norms(m: &Tensor, dim: usize) -> Result<Vec<f64>, ModelError> {
    m.sqr()
        .and_then(|s| s.sum(dim))
        .and_then(|s| s.to_vec1::<f32>())
        .map(|v| v.into_iter().map(|x| (x as f64).sqrt()).collect())
        .map_err(candle_err("ffn norms"))
}

fn finite_or_err(layer: usize, imp: Vec<f64>) -> Result<Vec<f64>, ModelError> {
    if imp.iter().all(|v| v.is_finite()) {
        Ok(imp)
    } else {
        Err(ModelError::Inference(format!(
            "layer {layer}: FFN channel importance is non-finite"
        )))
    }
}

/// `sqrt(||gate[c, :]|| * ||up[c, :]||) * ||down[:, c]||` of one layer.
fn weight_norm_importance(
    layer: usize,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
) -> Result<Vec<f64>, ModelError> {
    let (g, u, d) = (
        axis_norms(gate, 1)?,
        axis_norms(up, 1)?,
        axis_norms(down, 0)?,
    );
    let imp = g
        .iter()
        .zip(&u)
        .zip(&d)
        .map(|((g, u), d)| (g * u).sqrt() * d)
        .collect();
    finite_or_err(layer, imp)
}

/// Rows `kept` of `gate` and `up` and columns `kept` of `down`, each copied
/// into a new contiguous matrix so the dense originals can be freed.
fn slice_ffn_channels(
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    kept: &[usize],
) -> Result<(Tensor, Tensor, Tensor), ModelError> {
    let err = candle_err("prune ffn");
    let idx: Vec<u32> = kept.iter().map(|&c| c as u32).collect();
    let idx = Tensor::from_vec(idx, kept.len(), gate.device()).map_err(&err)?;
    let rows = |m: &Tensor| m.index_select(&idx, 0).and_then(|t| t.contiguous());
    let down = down
        .index_select(&idx, 1)
        .and_then(|t| t.contiguous())
        .map_err(&err)?;
    Ok((rows(gate).map_err(&err)?, rows(up).map_err(&err)?, down))
}

impl FfnPruneReport {
    fn empty(keep_ratio: Option<f64>, importance: Option<&str>) -> Self {
        Self {
            keep_ratio,
            importance: importance.map(str::to_string),
            layers: Vec::new(),
            channels_before: Vec::new(),
            channels_after: Vec::new(),
            ffn_params_before: 0,
            ffn_params_after: 0,
            bytes_freed: 0,
        }
    }

    fn record_layer(&mut self, layer: usize, hidden: u64, before: usize, after: usize) {
        self.layers.push(layer);
        self.channels_before.push(before);
        self.channels_after.push(after);
        self.ffn_params_before += 3 * hidden * before as u64;
        self.ffn_params_after += 3 * hidden * after as u64;
    }

    fn finish(&mut self) {
        self.bytes_freed = 4 * (self.ffn_params_before - self.ffn_params_after);
    }
}

/// Weight-norm pruning applied while a model loads, one decoder layer at a
/// time: the loader hands over each layer's dense FFN right after
/// dequantizing it and gets back the sliced matrices, so the dense upper
/// FFNs never coexist in memory. Same importance, selection and slicing as
/// [`QwenModel::prune_upper_ffn_by_weight_norm`], hence bit-exact with it.
pub(crate) struct StreamingFfnPruner {
    keep_ratio: f64,
    layers: Range<usize>,
    hidden: u64,
    report: FfnPruneReport,
}

impl StreamingFfnPruner {
    pub(crate) fn new(
        num_layers: usize,
        hidden_size: usize,
        keep_ratio: f64,
    ) -> Result<Self, ModelError> {
        check_keep_ratio(keep_ratio)?;
        let layers = default_upper_ffn_layers(num_layers);
        if layers.is_empty() {
            return Err(ModelError::QwenInput(format!(
                "a {num_layers}-layer model has no upper FFN layers to prune"
            )));
        }
        Ok(Self {
            keep_ratio,
            layers,
            hidden: hidden_size as u64,
            report: FfnPruneReport::empty(Some(keep_ratio), Some("weight_norm")),
        })
    }

    /// The sliced `(gate, up, down)` of `layer`, or `None` for a layer that
    /// stays dense. Layers must arrive in ascending order.
    pub(crate) fn prune_layer(
        &mut self,
        layer: usize,
        gate: &Tensor,
        up: &Tensor,
        down: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>, ModelError> {
        if !self.layers.contains(&layer) {
            return Ok(None);
        }
        if self.report.layers.last().is_some_and(|&l| l >= layer) {
            return Err(ModelError::QwenLoad(format!(
                "streaming FFN pruning got layer {layer} out of order"
            )));
        }
        let width = gate.dims2().map_err(candle_err("ffn width"))?.0;
        let imp = BTreeMap::from([(layer, weight_norm_importance(layer, gate, up, down)?)]);
        let kept = &select_top_ffn_channels(&imp, self.keep_ratio)?[&layer];
        let sliced = slice_ffn_channels(gate, up, down, kept)?;
        self.report
            .record_layer(layer, self.hidden, width, kept.len());
        Ok(Some(sliced))
    }

    /// The report, refusing a load that skipped any upper layer.
    pub(crate) fn finish(mut self) -> Result<FfnPruneReport, ModelError> {
        if !self.report.layers.iter().copied().eq(self.layers.clone()) {
            return Err(ModelError::QwenLoad(format!(
                "streaming FFN pruning covered layers {:?}, expected {:?}",
                self.report.layers, self.layers
            )));
        }
        self.report.finish();
        Ok(self.report)
    }
}

impl QwenModel {
    /// Activation importance of every FFN channel at `layers`, from full
    /// forward passes over `anchors` (each a non-empty token row).
    pub fn score_ffn_channels(
        &self,
        anchors: &[Vec<u32>],
        layers: &[usize],
    ) -> Result<FfnImportance, ModelError> {
        let set = check_layers(self, layers)?;
        if anchors.is_empty() || anchors.iter().any(Vec::is_empty) {
            return Err(ModelError::QwenInput(
                "FFN scoring needs nonempty anchor token rows".into(),
            ));
        }
        let sums = self.ffn_activation_sums(anchors, &set)?;
        sums.into_iter()
            .map(|(layer, acc)| {
                if acc.rows == 0 {
                    return Err(ModelError::Inference(format!(
                        "anchor forward did not execute FFN layer {layer}"
                    )));
                }
                let (_, _, down) = self.ffn_matrices(layer)?;
                let col = axis_norms(down, 0)?;
                let imp = acc
                    .sums
                    .iter()
                    .zip(&col)
                    .map(|(s, n)| s / acc.rows as f64 * n)
                    .collect();
                Ok((layer, finite_or_err(layer, imp)?))
            })
            .collect()
    }

    /// Weight-only importance `sqrt(||gate[c]|| * ||up[c]||) * ||down[:, c]||`.
    pub fn ffn_weight_norm_importance(
        &self,
        layers: &[usize],
    ) -> Result<FfnImportance, ModelError> {
        check_layers(self, layers)?
            .into_iter()
            .map(|layer| {
                let (gate, up, down) = self.ffn_matrices(layer)?;
                Ok((layer, weight_norm_importance(layer, gate, up, down)?))
            })
            .collect()
    }

    /// Keep only `masks[layer]` channels of each listed layer: `gate` and `up`
    /// become contiguous `[K, H]`, `down` contiguous `[H, K]`. Every mask is
    /// checked before any layer changes, so an error leaves the model intact.
    pub fn prune_ffn_channels(
        &mut self,
        masks: &FfnChannelMasks,
    ) -> Result<FfnPruneReport, ModelError> {
        let hidden = self.config().hidden_size as u64;
        for (&layer, kept) in masks {
            let width = self.ffn_width(layer)?;
            let ascending = kept.windows(2).all(|w| w[0] < w[1]);
            if kept.is_empty() || !ascending || kept.last().is_some_and(|&c| c >= width) {
                return Err(ModelError::QwenInput(format!(
                    "layer {layer}: FFN mask needs unique ascending channels below {width}"
                )));
            }
        }
        let mut report = FfnPruneReport::empty(None, None);
        for (&layer, kept) in masks {
            let width = self.ffn_width(layer)?;
            let (gate, up, down) = self.ffn_matrices(layer)?;
            let (gate, up, down) = slice_ffn_channels(gate, up, down, kept)?;
            self.replace_ffn(layer, gate, up, down)?;
            report.record_layer(layer, hidden, width, kept.len());
        }
        report.finish();
        Ok(report)
    }

    /// Activation-scored pruning of `layers` to `keep_ratio` of their channels.
    pub fn prune_ffn_by_activation(
        &mut self,
        anchors: &[Vec<u32>],
        layers: &[usize],
        keep_ratio: f64,
    ) -> Result<FfnPruneReport, ModelError> {
        check_keep_ratio(keep_ratio)?;
        let masks =
            select_top_ffn_channels(&self.score_ffn_channels(anchors, layers)?, keep_ratio)?;
        let mut report = self.prune_ffn_channels(&masks)?;
        report.keep_ratio = Some(keep_ratio);
        report.importance = Some("activation".into());
        Ok(report)
    }

    /// Weight-norm pruning of the upper layers ([`default_upper_ffn_layers`])
    /// to `keep_ratio`. Needs no data, so it can run at load time.
    pub fn prune_upper_ffn_by_weight_norm(
        &mut self,
        keep_ratio: f64,
    ) -> Result<FfnPruneReport, ModelError> {
        check_keep_ratio(keep_ratio)?;
        let layers: Vec<usize> = default_upper_ffn_layers(self.config().num_layers).collect();
        let masks =
            select_top_ffn_channels(&self.ffn_weight_norm_importance(&layers)?, keep_ratio)?;
        let mut report = self.prune_ffn_channels(&masks)?;
        report.keep_ratio = Some(keep_ratio);
        report.importance = Some("weight_norm".into());
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::tests::tiny_model;
    use crate::qwen::HookAction;

    #[test]
    fn upper_layers_default_to_22_through_27_for_28_layers() {
        assert_eq!(default_upper_ffn_layers(28), 22..28);
        assert_eq!(default_upper_ffn_layers(24), 19..24);
        assert_eq!(default_upper_ffn_layers(1), 1..1);
    }

    #[test]
    fn select_top_channels_keeps_highest_in_index_order() {
        let imp = BTreeMap::from([(3, vec![0.1, 5.0, 2.0, 2.0, 9.0, 0.0, 1.0, 3.0])]);
        let masks = select_top_ffn_channels(&imp, 0.5).unwrap();
        assert_eq!(masks[&3], vec![1, 2, 4, 7]);
        assert_eq!(select_top_ffn_channels(&imp, 0.01).unwrap()[&3], vec![4]);
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(select_top_ffn_channels(&imp, bad).is_err(), "{bad}");
        }
        let nan = BTreeMap::from([(0, vec![1.0, f64::NAN])]);
        assert!(select_top_ffn_channels(&nan, 0.5).is_err());
    }

    #[test]
    fn weight_norm_importance_matches_definition() {
        let model = tiny_model(2);
        let imp = model.ffn_weight_norm_importance(&[1]).unwrap();
        let (gate, up, down) = model.ffn_matrices(1).unwrap();
        let (g, u, d) = (
            gate.to_vec2::<f32>().unwrap(),
            up.to_vec2::<f32>().unwrap(),
            down.to_vec2::<f32>().unwrap(),
        );
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for c in 0..g.len() {
            let gr: Vec<f64> = g[c].iter().map(|&x| x as f64).collect();
            let ur: Vec<f64> = u[c].iter().map(|&x| x as f64).collect();
            let dc: Vec<f64> = d.iter().map(|row| row[c] as f64).collect();
            let want = (norm(&gr) * norm(&ur)).sqrt() * norm(&dc);
            assert!((imp[&1][c] - want).abs() <= 1e-5 * want.max(1.0), "{c}");
        }
    }

    #[test]
    fn activation_scores_are_positive_and_finite() {
        let model = tiny_model(3);
        let imp = model
            .score_ffn_channels(&[vec![1, 2, 3], vec![4, 5, 6, 7]], &[0, 2])
            .unwrap();
        assert_eq!(imp.keys().copied().collect::<Vec<_>>(), vec![0, 2]);
        assert!(imp
            .values()
            .all(|v| v.len() == 32 && v.iter().all(|x| x.is_finite() && *x > 0.0)));
        assert!(model.score_ffn_channels(&[], &[0]).is_err());
        assert!(model.score_ffn_channels(&[vec![1]], &[9]).is_err());
    }

    #[test]
    fn pruning_slices_matrices_and_matches_masked_dense_ffn() {
        let model = tiny_model(2);
        let rows = vec![vec![1, 5, 7, 9]];
        let masks =
            select_top_ffn_channels(&model.score_ffn_channels(&rows, &[0, 1]).unwrap(), 0.75)
                .unwrap();
        // Reference: dense weights with the dropped rows of gate/up zeroed.
        // silu(0) * 0 = 0, so those channels add nothing, exactly as if removed.
        let mut zeroed = model.clone();
        for (&layer, kept) in &masks {
            let (gate, up, down) = zeroed.ffn_matrices(layer).unwrap();
            let keep: Vec<f32> = (0..32)
                .map(|c| if kept.contains(&c) { 1.0 } else { 0.0 })
                .collect();
            let m = Tensor::from_vec(keep, (32, 1), model.device()).unwrap();
            let (g, u, d) = (
                gate.broadcast_mul(&m).unwrap(),
                up.broadcast_mul(&m).unwrap(),
                down.clone(),
            );
            zeroed.replace_ffn(layer, g, u, d).unwrap();
        }
        let mut pruned = model.clone();
        let report = pruned.prune_ffn_channels(&masks).unwrap();
        assert_eq!(report.channels_before, vec![32, 32]);
        assert_eq!(report.channels_after, vec![24, 24]);
        assert_eq!(report.ffn_params_before, 2 * 3 * 16 * 32);
        assert_eq!(report.ffn_params_after, 2 * 3 * 16 * 24);
        assert_eq!(report.bytes_freed, 4 * 2 * 3 * 16 * 8);
        for layer in 0..2 {
            let (g, u, d) = pruned.ffn_matrices(layer).unwrap();
            assert_eq!(
                (g.dims(), u.dims(), d.dims()),
                (&[24, 16][..], &[24, 16][..], &[16, 24][..])
            );
            assert!(g.is_contiguous() && u.is_contiguous() && d.is_contiguous());
            assert_eq!(pruned.ffn_width(layer).unwrap(), 24);
            assert_eq!(pruned.ffn_dense_width(layer).unwrap(), 32);
        }
        let (a, _) = pruned.forward(&rows, None).unwrap();
        let (b, _) = zeroed.forward(&rows, None).unwrap();
        let (a, b) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "pruned vs masked dense differ by {worst}");
        let (dense, _) = model.forward(&rows, None).unwrap();
        assert_ne!(dense.flatten_all().unwrap().to_vec1::<f32>().unwrap(), a);
    }

    #[test]
    fn hooked_forward_counts_exactly_keep_ratio_of_ffn_work() {
        let mut model = tiny_model(4);
        let rows = vec![vec![2, 4, 6]];
        let dense = model
            .forward_with_hook(&rows, &mut |_| Ok(HookAction::Continue))
            .unwrap()
            .ffn;
        assert_eq!(dense.ffn_channels_executed, dense.ffn_dense_channels);
        assert_eq!(dense.ffn_dense_channels, 4 * 32);
        assert_eq!(dense.ffn_dense_macs, 4 * 3 * 16 * 32 * 3);
        let layers: Vec<usize> = (0..4).collect();
        let report = model
            .prune_ffn_by_activation(&[vec![1, 2, 3]], &layers, 0.75)
            .unwrap();
        assert_eq!(report.importance.as_deref(), Some("activation"));
        let work = model
            .forward_with_hook(&rows, &mut |_| Ok(HookAction::Continue))
            .unwrap()
            .ffn;
        assert!(work.ffn_channels_executed > 0 && work.ffn_macs_executed > 0);
        assert_eq!(work.ffn_channels_executed * 4, work.ffn_dense_channels * 3);
        assert_eq!(work.ffn_macs_executed * 4, work.ffn_dense_macs * 3);
        assert_eq!(work.ffn_dense_macs, dense.ffn_dense_macs);
        // An exit after layer 1 counts only the two executed layers.
        let early = model
            .forward_with_hook(&rows, &mut |obs| {
                Ok(if obs.layer == 1 {
                    HookAction::Exit
                } else {
                    HookAction::Continue
                })
            })
            .unwrap();
        assert_eq!(early.ffn.ffn_channels_executed, 2 * 24);
        assert_eq!(early.ffn.ffn_dense_channels, 2 * 32);
    }

    #[test]
    fn upper_weight_norm_pruning_touches_only_upper_layers() {
        let mut model = tiny_model(28);
        let report = model.prune_upper_ffn_by_weight_norm(0.75).unwrap();
        assert_eq!(report.layers, (22..28).collect::<Vec<_>>());
        assert_eq!(report.importance.as_deref(), Some("weight_norm"));
        for layer in 0..28 {
            let want = if layer >= 22 { 24 } else { 32 };
            assert_eq!(model.ffn_width(layer).unwrap(), want, "layer {layer}");
        }
        let work = model
            .forward_with_hook(&[vec![3, 1]], &mut |_| Ok(HookAction::Continue))
            .unwrap()
            .ffn;
        assert_eq!(work.ffn_channels_executed, 22 * 32 + 6 * 24);
        assert_eq!(work.ffn_dense_channels, 28 * 32);
        assert!(model.prune_upper_ffn_by_weight_norm(0.0).is_err());
    }

    // Bit patterns, not f32 values: `==` on floats treats +0.0 and -0.0 as equal.
    fn flat(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Streaming pruning at load must equal load-then-prune bit for bit:
    /// same report, same dense widths, same matrices, same forward output.
    fn assert_streaming_matches_load_then_prune(
        path: &std::path::Path,
        keep: f64,
        rows: &[Vec<u32>],
    ) -> (QwenModel, FfnPruneReport) {
        let mut reference = QwenModel::load(path).unwrap();
        let want = reference.prune_upper_ffn_by_weight_norm(keep).unwrap();
        let (streamed, got) = QwenModel::load_with_weight_norm_pruning(path, keep).unwrap();
        assert_eq!(got, want, "{}", path.display());
        let n = reference.config().num_layers;
        assert_eq!(got.layers, default_upper_ffn_layers(n).collect::<Vec<_>>());
        assert_eq!(got.importance.as_deref(), Some("weight_norm"));
        assert_eq!(got.keep_ratio, Some(keep));
        for layer in 0..n {
            assert_eq!(
                streamed.ffn_width(layer).unwrap(),
                reference.ffn_width(layer).unwrap()
            );
            assert_eq!(
                streamed.ffn_dense_width(layer).unwrap(),
                reference.ffn_dense_width(layer).unwrap()
            );
            let (a, b) = (
                streamed.ffn_matrices(layer).unwrap(),
                reference.ffn_matrices(layer).unwrap(),
            );
            for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                assert_eq!(x.dims(), y.dims(), "layer {layer}");
                assert!(x.is_contiguous(), "layer {layer}");
                assert_eq!(flat(x), flat(y), "layer {layer}");
            }
        }
        let (a, _) = streamed.forward(rows, None).unwrap();
        let (b, _) = reference.forward(rows, None).unwrap();
        assert_eq!(flat(&a), flat(&b));
        (streamed, got)
    }

    #[test]
    fn streaming_load_pruning_is_bit_exact_with_load_then_prune() {
        let model = tiny_model(28);
        let dir = tempfile::tempdir().unwrap();
        let (hf, gguf) = crate::qwen::tests::write_tiny_checkpoints(&model, dir.path());
        let rows = vec![vec![1, 5, 7, 9, 30], vec![2, 4, 6, 8, 10]];
        // The files hold the in-memory weights exactly.
        for path in [&hf, &gguf] {
            assert_eq!(
                flat(
                    &QwenModel::load(path)
                        .unwrap()
                        .forward(&rows, None)
                        .unwrap()
                        .0
                ),
                flat(&model.forward(&rows, None).unwrap().0)
            );
        }
        for keep in [0.75, 0.5] {
            for path in [&hf, &gguf] {
                let (streamed, report) =
                    assert_streaming_matches_load_then_prune(path, keep, &rows);
                let k = ((keep * 32.0).round()) as usize;
                assert_eq!(report.channels_after, vec![k; 6]);
                assert_eq!(streamed.ffn_width(22).unwrap(), k);
                assert_eq!(streamed.ffn_width(21).unwrap(), 32);
                assert_eq!(streamed.ffn_dense_width(22).unwrap(), 32);
                let work = streamed
                    .forward_with_hook(&rows, &mut |_| Ok(HookAction::Continue))
                    .unwrap()
                    .ffn;
                assert_eq!(work.ffn_channels_executed, 22 * 32 + 6 * k as u64);
                assert_eq!(work.ffn_dense_channels, 28 * 32);
            }
        }
        assert!(QwenModel::load_with_weight_norm_pruning(&hf, 0.0).is_err());
        assert!(QwenModel::load_with_weight_norm_pruning(&gguf, 1.5).is_err());
    }

    /// `GENZERO_QWEN_STREAM_WEIGHTS=<.gguf or HF dir> cargo test -p gen-zero-model
    /// --lib streaming_pruning_bit_exact_on_real_weights -- --ignored --nocapture`.
    /// A missing variable fails the test.
    #[test]
    #[ignore = "needs Qwen2 weights at $GENZERO_QWEN_STREAM_WEIGHTS"]
    fn streaming_pruning_bit_exact_on_real_weights() {
        let path = std::env::var("GENZERO_QWEN_STREAM_WEIGHTS")
            .expect("set GENZERO_QWEN_STREAM_WEIGHTS to a Qwen2 .gguf file or HF directory");
        let rows = vec![vec![785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315]];
        let (_, report) =
            assert_streaming_matches_load_then_prune(std::path::Path::new(&path), 0.75, &rows);
        println!("streaming == load-then-prune on {path}: {report:?}");
    }

    #[test]
    fn streaming_pruner_refuses_skipped_or_reordered_layers() {
        let model = tiny_model(28);
        let (gate, up, down) = model.ffn_matrices(22).unwrap();
        let mut pruner = StreamingFfnPruner::new(28, 16, 0.75).unwrap();
        assert!(pruner.prune_layer(3, gate, up, down).unwrap().is_none());
        assert!(pruner.prune_layer(23, gate, up, down).unwrap().is_some());
        assert!(pruner.prune_layer(22, gate, up, down).is_err());
        assert!(pruner.finish().is_err());
        assert!(StreamingFfnPruner::new(1, 16, 0.75).is_err());
        assert!(StreamingFfnPruner::new(28, 16, f64::NAN).is_err());
    }

    #[test]
    fn bad_masks_leave_model_untouched() {
        let mut model = tiny_model(2);
        for bad in [vec![], vec![3, 1], vec![1, 1], vec![0, 32]] {
            let masks = BTreeMap::from([(0, vec![0, 1]), (1, bad.clone())]);
            assert!(model.prune_ffn_channels(&masks).is_err(), "{bad:?}");
            assert_eq!(model.ffn_width(0).unwrap(), 32);
        }
        assert!(model
            .prune_ffn_channels(&BTreeMap::from([(5, vec![0])]))
            .is_err());
    }

    #[test]
    fn non_f32_projection_refuses_pruning() {
        let q = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((32, 32), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            candle_core::quantized::GgmlDType::Q8_0,
        )
        .unwrap();
        let m = candle_core::quantized::QMatMul::QTensor(std::sync::Arc::new(q));
        assert!(crate::qwen::dense_matrix(&m).is_err());
    }
}
