//! Inter-layer hooks, anchor manifold memory with ridge drift correction,
//! MCTS layer weights and a heuristic subspace-step early exit for Qwen2.
//!
//! Sources. The shape follows `python/gen_zero/causal/qwen_inter_layer_mcts.py`
//! (`MctsLayerwiseQwenRunner`: checkpoint readout, trajectory of weighted
//! label log-odds, `fit_layer_weights` by UCT over layer candidates with the
//! LOO utility `acc + 0.1 * ll`, MCTS label confirmation). Three parts are
//! defined by the task specification, not by that file, and differ from it:
//! * `default_hook_layers`: `[12, 16, 20, 22, 24, 26, 27]` for 28 layers;
//! * `InterLayerMemory`: per hook layer, the centred final-norm states of the
//!   labelled anchors (the residual memory) and a dual ridge map from them to
//!   the full model's final label logits (the drift correction), with the
//!   ridge strength chosen by exact leave-one-out error;
//! * the heuristic exit rule `p_(1) - p_(2) > 2 L_f delta_l / tau`, replacing the
//!   Python geometric-tail bound.
//!
//! The subspace-step bound. At hook layer `l` the corrected logits are
//! `z_l = ybar + W^T (f_l - mu)` with `f_l` the final-norm state of the last
//! token, so `||dz||_2 <= L_f ||df||_2` where `L_f = ||W||_2`. For
//! `p = softmax(z / tau)` every probability moves by at most
//! `||dz||_inf / (2 tau)`, so the top-2 gap shrinks by at most
//! `L_f ||df|| / tau`, and the factor 2 in the rule is extra slack. With
//! `delta_l = ||W_l^T(f_l - f_prev)|| / (L_f sqrt(H/K))`, with `prev` the
//! preceding hook checkpoint, the rule estimates that the label
//! cannot flip under a further readout-space move as large as the last one.
//! The dimension normalization and assumption about unseen layers are
//! heuristics, not a proof; the exit also needs a stable
//! top-1 across two checkpoints, `layer >= min_exit_layer` and an MCTS visit
//! margin, as in the Python runner. With layer weights the bound uses the
//! full `L_f` of the current layer, an upper bound for its `w_l / sum w`
//! share of the weighted trajectory.
//!
//! The calibrated readout ([`InterLayerMctsRunner::fit_calibrated`]) is a
//! port of the Python `fit_memories` + `predict` path instead:
//! * [`VerbalizerLogitStats`]: each label's logit is the `logsumexp` of its
//!   token group, standardised per layer by the anchor mean and std. An
//!   intermediate layer read through the final norm and `lm_head` carries a
//!   large constant label offset; `(z - mean) / std` removes it with two
//!   numbers per label instead of a map fitted in `H = 1536` dimensions;
//! * [`EpisodicManifoldMemory`]: cosine-kernel softmax retrieval over the
//!   centred, L2-normalised raw residuals, mixed in with a weight from
//!   [`select_memory_lambda`] (out-of-fold gain over `lambda = 0` and over the
//!   leave-one-out class prior, both by 2 standard errors, else 0);
//! * the exit rule is the Python geometric-tail bound on probability drift,
//!   `2 tau g / (1 - g) * d_l`; `tau` here scales that bound only, it is not
//!   a softmax temperature.

use crate::error::ModelError;
use crate::ffn_prune::{select_top_ffn_channels, FfnChannelMasks, FfnImportance};
use crate::latent_mcts::{simplex_etf, softmax, LatentMctsPlanner};
use crate::qwen::{candle_err, FfnWork, HookAction, QwenModel};
use candle_core::Tensor;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PROB_FLOOR: f64 = 1e-12;
/// Ridge strengths, relative to the mean squared anchor feature norm.
const RIDGE_GRID: [f64; 6] = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0];

/// Hook layers for a decoder of `num_layers`: `[12, 16, 20, 22, 24, 26, 27]`
/// for the 28-layer Qwen2.5-1.5B, the same depth fractions of the last layer
/// otherwise, strictly increasing and always ending with the last layer.
pub fn default_hook_layers(num_layers: usize) -> Vec<usize> {
    const QWEN_1_5B: [usize; 7] = [12, 16, 20, 22, 24, 26, 27];
    if num_layers == 0 {
        return Vec::new();
    }
    let last = num_layers - 1;
    let mut out: Vec<usize> = QWEN_1_5B
        .iter()
        .map(|&l| ((l * last) as f64 / 27.0).round() as usize)
        .collect();
    out.dedup();
    if out.last() != Some(&last) {
        out.push(last);
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn top2_gap(p: &[f64]) -> f64 {
    let mut s = p.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s.get(1).copied().unwrap_or(0.0)
}

fn check_finite(what: &str, v: &[f64]) -> Result<(), ModelError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::Inference(format!("{what} is non-finite")))
    }
}

/// Inverse of a symmetric positive definite matrix by Cholesky.
fn spd_inverse(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModelError> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
                    return Err(ModelError::Inference(
                        "ridge system is not positive definite".into(),
                    ));
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Solve L L^T X = I column by column.
    let mut columns = vec![vec![0.0; n]; n];
    for (c, column) in columns.iter_mut().enumerate() {
        let mut y = vec![0.0; n];
        for (i, row) in l.iter().enumerate() {
            let rhs = if i == c { 1.0 } else { 0.0 };
            y[i] = (rhs - (0..i).map(|k| row[k] * y[k]).sum::<f64>()) / row[i];
        }
        for i in (0..n).rev() {
            column[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * column[k]).sum::<f64>()) / l[i][i];
        }
    }
    Ok((0..n)
        .map(|i| columns.iter().map(|column| column[i]).collect())
        .collect())
}

/// Largest eigenvalue of a small symmetric positive semidefinite matrix.
fn top_eigenvalue(m: &[Vec<f64>]) -> f64 {
    let k = m.len();
    if k == 0 {
        return 0.0;
    }
    let mut best = 0.0_f64;
    for basis in 0..k {
        let mut v = vec![0.0; k];
        v[basis] = 1.0;
        let mut lambda = 0.0;
        for _ in 0..1000 {
            let w: Vec<f64> = m.iter().map(|row| dot(row, &v)).collect();
            let norm = dot(&w, &w).sqrt();
            if norm == 0.0 {
                break;
            }
            v = w.iter().map(|x| x / norm).collect();
            let next = m
                .iter()
                .zip(&v)
                .map(|(row, x)| x * dot(row, &v))
                .sum::<f64>();
            if (next - lambda).abs() <= 1e-12 * next.abs().max(1.0) {
                lambda = next;
                break;
            }
            lambda = next;
        }
        best = best.max(lambda);
    }
    best
}

/// One hook layer's anchor memory and drift-corrected readout.
#[derive(Clone, Debug)]
pub struct InterLayerMemory {
    pub layer: usize,
    /// Anchor mean of the final-norm states (the centre).
    center: Vec<f64>,
    /// Centred anchor states, one row per anchor (the residual memory).
    anchors: Vec<Vec<f64>>,
    /// Dual ridge coefficients, `[n, K]`.
    alpha: Vec<Vec<f64>>,
    /// Anchor mean of the final label logits.
    target_mean: Vec<f64>,
    pub ridge_lambda: f64,
    pub loo_mse: f64,
    /// `||W||_2` of the readout `W = anchors^T alpha`, `[H, K]`.
    pub lipschitz: f64,
    /// Exact leave-one-out predicted logits, `[n, K]`.
    pub loo_logits: Vec<Vec<f64>>,
}

impl InterLayerMemory {
    /// Fit the map `features [n, H] -> targets [n, K]` by centred dual ridge.
    /// The hat matrix with an unpenalised intercept is
    /// `11^T / n + G (G + lambda I)^-1` (`G` the centred Gram), so the LOO
    /// residual is `e_i / (lambda [A^-1]_ii - 1/n)` with `e = lambda alpha`.
    pub fn fit(
        layer: usize,
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<Self, ModelError> {
        let n = features.len();
        let k = targets.first().map_or(0, Vec::len);
        let h = features.first().map_or(0, Vec::len);
        if n < 3 || targets.len() != n || k == 0 || h == 0 {
            return Err(ModelError::QwenInput(format!(
                "layer {layer}: memory needs >= 3 anchors with matching targets"
            )));
        }
        if features.iter().any(|f| f.len() != h) || targets.iter().any(|t| t.len() != k) {
            return Err(ModelError::QwenInput(format!(
                "layer {layer}: ragged anchor features or targets"
            )));
        }
        for row in features.iter().chain(targets) {
            check_finite("anchor feature or target", row)?;
        }
        let mean = |rows: &[Vec<f64>], d: usize| -> Vec<f64> {
            (0..d)
                .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n as f64)
                .collect()
        };
        let center = mean(features, h);
        let target_mean = mean(targets, k);
        let anchors: Vec<Vec<f64>> = features
            .iter()
            .map(|f| f.iter().zip(&center).map(|(a, b)| a - b).collect())
            .collect();
        let yc: Vec<Vec<f64>> = targets
            .iter()
            .map(|t| t.iter().zip(&target_mean).map(|(a, b)| a - b).collect())
            .collect();
        let gram: Vec<Vec<f64>> = anchors
            .iter()
            .map(|a| anchors.iter().map(|b| dot(a, b)).collect())
            .collect();
        let scale = (0..n).map(|i| gram[i][i]).sum::<f64>() / n as f64;
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(ModelError::Inference(format!(
                "layer {layer}: anchor states are identical; nothing to regress on"
            )));
        }
        type RidgeCandidate = (f64, f64, Vec<Vec<f64>>, Vec<Vec<f64>>);
        let mut best: Option<RidgeCandidate> = None;
        for rel in RIDGE_GRID {
            let lambda = rel * scale;
            let mut a = gram.clone();
            for (i, row) in a.iter_mut().enumerate() {
                row[i] += lambda;
            }
            let inv = spd_inverse(&a)?;
            let alpha: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..k)
                        .map(|c| (0..n).map(|j| inv[i][j] * yc[j][c]).sum())
                        .collect()
                })
                .collect();
            let mut loo = Vec::with_capacity(n);
            let mut sse = 0.0;
            for i in 0..n {
                let denom = lambda * inv[i][i] - 1.0 / n as f64;
                if !denom.is_finite() || denom <= 1e-12 {
                    return Err(ModelError::Inference(format!(
                        "layer {layer}: anchor {i} has leverage 1; LOO is undefined"
                    )));
                }
                let row: Vec<f64> = (0..k)
                    .map(|c| {
                        let r = lambda * alpha[i][c] / denom;
                        sse += r * r;
                        targets[i][c] - r
                    })
                    .collect();
                loo.push(row);
            }
            let mse = sse / (n * k) as f64;
            if !mse.is_finite() {
                return Err(ModelError::Inference(format!(
                    "layer {layer}: non-finite LOO error at ridge {lambda}"
                )));
            }
            if best.as_ref().is_none_or(|b| mse < b.1) {
                best = Some((lambda, mse, alpha, loo));
            }
        }
        let (ridge_lambda, loo_mse, alpha, loo_logits) =
            best.ok_or_else(|| ModelError::Inference("empty ridge grid".into()))?;
        // W^T W = alpha^T G alpha, a K x K matrix.
        let g_alpha: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|c| (0..n).map(|j| gram[i][j] * alpha[j][c]).sum())
                    .collect()
            })
            .collect();
        let wtw: Vec<Vec<f64>> = (0..k)
            .map(|a| {
                (0..k)
                    .map(|b| (0..n).map(|i| alpha[i][a] * g_alpha[i][b]).sum())
                    .collect()
            })
            .collect();
        let lipschitz = top_eigenvalue(&wtw).max(0.0).sqrt();
        check_finite("ridge Lipschitz constant", &[lipschitz])?;
        Ok(Self {
            layer,
            center,
            anchors,
            alpha,
            target_mean,
            ridge_lambda,
            loo_mse,
            lipschitz,
            loo_logits,
        })
    }

    /// Drift-corrected final label logits for one final-norm state.
    pub fn predict(&self, feature: &[f64]) -> Result<Vec<f64>, ModelError> {
        if feature.len() != self.center.len() {
            return Err(ModelError::QwenInput(format!(
                "layer {}: state of width {} for a memory of width {}",
                self.layer,
                feature.len(),
                self.center.len()
            )));
        }
        let x: Vec<f64> = feature
            .iter()
            .zip(&self.center)
            .map(|(a, b)| a - b)
            .collect();
        let kernel: Vec<f64> = self.anchors.iter().map(|a| dot(a, &x)).collect();
        let z: Vec<f64> = (0..self.target_mean.len())
            .map(|c| {
                self.target_mean[c]
                    + kernel
                        .iter()
                        .zip(&self.alpha)
                        .map(|(kv, a)| kv * a[c])
                        .sum::<f64>()
            })
            .collect();
        check_finite("drift-corrected logits", &z)?;
        Ok(z)
    }

    /// Dimension-normalized displacement in this layer's readout subspace.
    pub fn projected_delta(&self, current: &[f64], previous: &[f64]) -> Result<f64, ModelError> {
        if current.len() != self.center.len() || previous.len() != self.center.len() {
            return Err(ModelError::QwenInput(format!(
                "layer {}: invalid feature width",
                self.layer
            )));
        }
        let df: Vec<f64> = current.iter().zip(previous).map(|(a, b)| a - b).collect();
        let kernel: Vec<f64> = self.anchors.iter().map(|a| dot(a, &df)).collect();
        let norm = (0..self.target_mean.len())
            .map(|c| {
                let dz: f64 = kernel
                    .iter()
                    .zip(&self.alpha)
                    .map(|(v, row)| v * row[c])
                    .sum();
                dz * dz
            })
            .sum::<f64>()
            .sqrt();
        let scale = (self.center.len() as f64 / self.target_mean.len() as f64).sqrt();
        let delta = norm / (self.lipschitz.max(1e-12) * scale);
        check_finite("projected checkpoint delta", &[delta])?;
        Ok(delta)
    }

    pub fn num_anchors(&self) -> usize {
        self.anchors.len()
    }
}

/// Mean accuracy and log-likelihood of `softmax(z / tau)` against `labels`.
fn utility_terms(logits: &[Vec<f64>], labels: &[usize], tau: f64) -> (f64, f64) {
    let n = labels.len() as f64;
    let mut acc = 0.0;
    let mut ll = 0.0;
    for (z, &y) in logits.iter().zip(labels) {
        let p = softmax(&z.iter().map(|v| v / tau).collect::<Vec<_>>());
        if argmax(&p) == y {
            acc += 1.0;
        }
        ll += p[y].max(PROB_FLOOR).ln();
    }
    (acc / n, ll / n)
}

/// Sparse layer weights by UCT over the layer candidates: each layer's
/// reward is `reward_scale * (u_l - mean u)`, the top `top_k` by root visit
/// share keep their share, the rest get 0, and the result sums to 1.
pub fn fit_layer_weights(
    utilities: &BTreeMap<usize, f64>,
    top_k: usize,
    reward_scale: f64,
    planner: &LatentMctsPlanner,
) -> Result<BTreeMap<usize, f64>, ModelError> {
    mcts_layer_weights(utilities, top_k, reward_scale, 1.0, planner)
}

/// [`fit_layer_weights`] with the search temperature as a parameter: the
/// ridge runner uses 1.0, the calibrated runner 0.25 as the Python
/// `fit_layer_weights` does.
fn mcts_layer_weights(
    utilities: &BTreeMap<usize, f64>,
    top_k: usize,
    reward_scale: f64,
    temperature: f64,
    planner: &LatentMctsPlanner,
) -> Result<BTreeMap<usize, f64>, ModelError> {
    if utilities.is_empty() || top_k == 0 || !(reward_scale.is_finite() && reward_scale > 0.0) {
        return Err(ModelError::QwenInput(
            "layer weights need utilities, top_k >= 1 and a positive reward scale".into(),
        ));
    }
    let layers: Vec<usize> = utilities.keys().copied().collect();
    let u: Vec<f64> = utilities.values().copied().collect();
    check_finite("layer utility", &u)?;
    let mean = u.iter().sum::<f64>() / u.len() as f64;
    let reward: Vec<f64> = u.iter().map(|v| reward_scale * (v - mean)).collect();
    let search = LatentMctsPlanner {
        num_simulations: planner.num_simulations.max(64),
        c_puct: planner.c_puct,
        temperature,
        max_depth: 2,
        damping: planner.damping,
    };
    let plan = search.plan(&reward, &simplex_etf(layers.len())?)?;
    let mut order: Vec<usize> = (0..layers.len()).collect();
    order.sort_by(|&a, &b| plan.policy[b].total_cmp(&plan.policy[a]));
    let keep: BTreeSet<usize> = order.into_iter().take(top_k.min(layers.len())).collect();
    let total: f64 = keep.iter().map(|&j| plan.policy[j]).sum();
    if !(total > 0.0 && total.is_finite()) {
        return Err(ModelError::Inference(
            "MCTS gave the kept layers no visits".into(),
        ));
    }
    Ok(layers
        .iter()
        .enumerate()
        .map(|(j, &l)| {
            (
                l,
                if keep.contains(&j) {
                    plan.policy[j] / total
                } else {
                    0.0
                },
            )
        })
        .collect())
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// Label logits of one final-norm state: `logsumexp` of `rows . f` over each
/// label's consecutive group of `group_sizes[c]` rows.
fn group_logits(
    rows: &[Vec<f64>],
    group_sizes: &[usize],
    feature: &[f64],
) -> Result<Vec<f64>, ModelError> {
    let token_logits: Vec<f64> = rows.iter().map(|r| dot(r, feature)).collect();
    VerbalizerLogitStats::group_logits(&token_logits, group_sizes)
}

fn column_moments(z: &[Vec<f64>], what: &str) -> Result<(usize, usize), ModelError> {
    let n = z.len();
    let k = z.first().map_or(0, Vec::len);
    if n < 2 || k == 0 || z.iter().any(|r| r.len() != k) {
        return Err(ModelError::QwenInput(format!(
            "{what} needs >= 2 rows of one nonzero width"
        )));
    }
    for row in z {
        check_finite(what, row)?;
    }
    Ok((n, k))
}

/// Per-label anchor statistics of the verbalizer logits at one layer.
///
/// The readout is `(z - mean) / std` with the population std. A variance
/// that is not finite or not above `1e-10` times the mean square fails:
/// rounding gives a constant column a tiny positive variance, and its std
/// would blow the standardised logits up instead of failing.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VerbalizerLogitStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl VerbalizerLogitStats {
    /// Label logits from token logits: the `logsumexp` of each label's
    /// consecutive group of `group_sizes[c]` token logits.
    pub fn group_logits(
        token_logits: &[f64],
        group_sizes: &[usize],
    ) -> Result<Vec<f64>, ModelError> {
        if group_sizes.is_empty()
            || group_sizes.contains(&0)
            || group_sizes.iter().sum::<usize>() != token_logits.len()
        {
            return Err(ModelError::QwenInput(format!(
                "token groups {group_sizes:?} do not cover {} token logits",
                token_logits.len()
            )));
        }
        check_finite("verbalizer token logits", token_logits)?;
        let mut start = 0;
        let z: Vec<f64> = group_sizes
            .iter()
            .map(|&g| {
                let v = logsumexp(&token_logits[start..start + g]);
                start += g;
                v
            })
            .collect();
        check_finite("verbalizer label logits", &z)?;
        Ok(z)
    }

    /// Mean and population std of anchor logits `z [n, K]`.
    pub fn fit(z: &[Vec<f64>]) -> Result<Self, ModelError> {
        let (n, k) = column_moments(z, "anchor verbalizer logits")?;
        let mut mean = vec![0.0; k];
        let mut std = vec![0.0; k];
        for c in 0..k {
            let mu = z.iter().map(|r| r[c]).sum::<f64>() / n as f64;
            let second = z.iter().map(|r| r[c] * r[c]).sum::<f64>() / n as f64;
            let var = z.iter().map(|r| (r[c] - mu).powi(2)).sum::<f64>() / n as f64;
            if !(second.is_finite() && var.is_finite()) || var <= (1e-10 * second).max(0.0) {
                return Err(ModelError::Inference(format!(
                    "degenerate anchor logits with zero variance in label {c}"
                )));
            }
            mean[c] = mu;
            std[c] = var.sqrt();
        }
        Ok(Self { mean, std })
    }

    pub fn standardize(&self, z: &[f64]) -> Result<Vec<f64>, ModelError> {
        if z.len() != self.mean.len() {
            return Err(ModelError::QwenInput(format!(
                "{} logits for stats of {} labels",
                z.len(),
                self.mean.len()
            )));
        }
        let out: Vec<f64> = z
            .iter()
            .zip(self.mean.iter().zip(&self.std))
            .map(|(v, (m, s))| (v - m) / s)
            .collect();
        check_finite("standardised verbalizer logits", &out)?;
        Ok(out)
    }

    /// Row `i`: the mean over the other rows (closed form).
    pub fn loo_mean(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModelError> {
        let (n, k) = column_moments(z, "leave-one-out logits")?;
        let total: Vec<f64> = (0..k).map(|c| z.iter().map(|r| r[c]).sum()).collect();
        Ok(z.iter()
            .map(|r| (0..k).map(|c| (total[c] - r[c]) / (n - 1) as f64).collect())
            .collect())
    }

    /// Row `i`: the population std over the other rows (closed form
    /// `second - mean^2`). Fails when any leave-one-out variance is not
    /// finite or not above `1e-10` times its mean square: the closed form
    /// cancels, so leaving row 0 out of `[0.3, 1.1, 1.1]` gives `2.2e-16`
    /// instead of 0.
    pub fn loo_std(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModelError> {
        let (n, k) = column_moments(z, "leave-one-out logits")?;
        let mu = Self::loo_mean(z)?;
        let total_sq: Vec<f64> = (0..k)
            .map(|c| z.iter().map(|r| r[c] * r[c]).sum())
            .collect();
        z.iter()
            .zip(&mu)
            .map(|(r, m)| {
                (0..k)
                    .map(|c| {
                        let second = (total_sq[c] - r[c] * r[c]) / (n - 1) as f64;
                        let var = second - m[c] * m[c];
                        if !(second.is_finite() && var.is_finite())
                            || var <= (1e-10 * second).max(0.0)
                        {
                            Err(ModelError::Inference(format!(
                                "degenerate leave-one-out anchor logits with zero variance \
                                 in label {c}"
                            )))
                        } else {
                            Ok(var.sqrt())
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Each row standardised by the stats of the other rows, so no anchor is
    /// scored with its own logits.
    pub fn loo_standardize(z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModelError> {
        let mu = Self::loo_mean(z)?;
        let sd = Self::loo_std(z)?;
        Ok(z.iter()
            .zip(mu.iter().zip(&sd))
            .map(|(r, (m, s))| {
                r.iter()
                    .zip(m.iter().zip(s))
                    .map(|(v, (a, b))| (v - a) / b)
                    .collect()
            })
            .collect())
    }
}

/// Cosine-kernel associative memory over `L2Norm(h - center)` keys
/// (Modern Hopfield / Nadaraya-Watson read): `query` returns
/// `softmax(beta * K q)^T V`. Raw decoder states share a few huge outlier
/// dimensions, so uncentred cosines all sit near 1 and the read degrades to
/// the class prior; `center` (the anchor mean in [`InterLayerMctsRunner::fit_calibrated`])
/// removes them. The store is a FIFO ring of `capacity` entries; the
/// calibrated fit centres on the retained anchors.
///
/// With a centre, [`Self::loo_query`] treats it as the mean of the stored
/// states: entry `i` is read against the mean of the other entries, so the
/// out-of-fold read never sees `h_i` through the centre either.
#[derive(Clone, Debug)]
pub struct EpisodicManifoldMemory {
    dim: usize,
    num_labels: usize,
    capacity: usize,
    beta: f64,
    center: Option<Vec<f64>>,
    raw: Vec<Vec<f64>>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    ids: Vec<usize>,
}

impl EpisodicManifoldMemory {
    pub fn new(
        dim: usize,
        num_labels: usize,
        capacity: usize,
        beta: f64,
        center: Option<Vec<f64>>,
    ) -> Result<Self, ModelError> {
        if dim == 0 || num_labels == 0 || capacity < 3 {
            return Err(ModelError::QwenInput(
                "dim and num_labels must be >= 1; capacity must be >= 3".into(),
            ));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(ModelError::QwenInput("beta must be finite and > 0".into()));
        }
        if let Some(c) = &center {
            if c.len() != dim {
                return Err(ModelError::QwenInput(format!(
                    "center must have width {dim}"
                )));
            }
            check_finite("memory center", c)?;
        }
        Ok(Self {
            dim,
            num_labels,
            capacity,
            beta,
            center,
            raw: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
            ids: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The unit key of one state.
    pub fn encode(&self, hidden: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.encode_with(hidden, self.center.as_deref())
    }

    fn encode_with(&self, hidden: &[f64], center: Option<&[f64]>) -> Result<Vec<f64>, ModelError> {
        if hidden.len() != self.dim {
            return Err(ModelError::QwenInput(format!(
                "state of width {} for a memory of width {}",
                hidden.len(),
                self.dim
            )));
        }
        check_finite("memory state", hidden)?;
        let h: Vec<f64> = match center {
            Some(c) => hidden.iter().zip(c).map(|(a, b)| a - b).collect(),
            None => hidden.to_vec(),
        };
        let norm = dot(&h, &h).sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return Err(ModelError::Inference(
                "cannot key a zero or overflowing state".into(),
            ));
        }
        Ok(h.iter().map(|v| v / norm).collect())
    }

    /// Store one one-hot label per state under `ids`.
    pub fn write(
        &mut self,
        hidden: &[Vec<f64>],
        labels: &[usize],
        ids: &[usize],
    ) -> Result<(), ModelError> {
        if hidden.len() != labels.len() || hidden.len() != ids.len() {
            return Err(ModelError::QwenInput(
                "hidden, labels and ids must have the same length".into(),
            ));
        }
        if let Some(bad) = labels.iter().find(|&&y| y >= self.num_labels) {
            return Err(ModelError::QwenInput(format!(
                "label index {bad} out of range"
            )));
        }
        let mut seen: std::collections::BTreeSet<usize> = self.ids.iter().copied().collect();
        if ids.iter().any(|id| !seen.insert(*id)) {
            return Err(ModelError::QwenInput("duplicate memory ID".into()));
        }
        for h in hidden {
            if h.len() != self.dim {
                return Err(ModelError::QwenInput("memory state width mismatch".into()));
            }
            check_finite("memory state", h)?;
        }
        let mut raw = self.raw.clone();
        let mut values = self.values.clone();
        let mut retained_ids = self.ids.clone();
        for ((h, &y), &id) in hidden.iter().zip(labels).zip(ids) {
            let mut value = vec![0.0; self.num_labels];
            value[y] = 1.0;
            raw.push(h.clone());
            values.push(value);
            retained_ids.push(id);
        }
        let drop = retained_ids.len().saturating_sub(self.capacity);
        raw.drain(..drop);
        values.drain(..drop);
        retained_ids.drain(..drop);
        let center = (!raw.is_empty()).then(|| {
            (0..self.dim)
                .map(|j| raw.iter().map(|h| h[j]).sum::<f64>() / raw.len() as f64)
                .collect::<Vec<_>>()
        });
        let keys = if raw.len() == 1 {
            vec![vec![0.0; self.dim]]
        } else {
            raw.iter()
                .map(|h| self.encode_with(h, center.as_deref()))
                .collect::<Result<Vec<_>, _>>()?
        };
        self.raw = raw;
        self.values = values;
        self.ids = retained_ids;
        self.center = center;
        self.keys = keys;
        Ok(())
    }

    fn read(&self, q: &[f64], exclude: Option<usize>) -> Result<Vec<f64>, ModelError> {
        self.read_keys(&self.keys, q, exclude)
    }

    fn read_keys(
        &self,
        keys: &[Vec<f64>],
        q: &[f64],
        exclude: Option<usize>,
    ) -> Result<Vec<f64>, ModelError> {
        if self.is_empty() {
            return Err(ModelError::Inference("memory is empty".into()));
        }
        let sims: Vec<f64> = keys
            .iter()
            .zip(&self.ids)
            .map(|(k, &id)| {
                if Some(id) == exclude {
                    f64::NEG_INFINITY
                } else {
                    self.beta * dot(q, k)
                }
            })
            .collect();
        if sims.iter().all(|s| *s == f64::NEG_INFINITY) {
            return Err(ModelError::Inference(
                "leave-one-out removed every memory entry".into(),
            ));
        }
        let w = softmax(&sims);
        let out: Vec<f64> = (0..self.num_labels)
            .map(|c| w.iter().zip(&self.values).map(|(a, v)| a * v[c]).sum())
            .collect();
        check_finite("memory read", &out)?;
        Ok(out)
    }

    /// Label distribution retrieved for `hidden`, skipping entries stored
    /// under `exclude_id`.
    pub fn query(&self, hidden: &[f64], exclude_id: Option<usize>) -> Result<Vec<f64>, ModelError> {
        self.read(&self.encode(hidden)?, exclude_id)
    }

    /// Each stored entry's own read with itself masked out (out-of-fold).
    /// With a centre, entry `i` and every other key are re-centred on
    /// `mu_-i = (sum - h_i) / (m - 1)`, the mean of the other `m - 1` states.
    pub fn loo_query(&self) -> Result<Vec<Vec<f64>>, ModelError> {
        if self.center.is_none() {
            return self
                .keys
                .iter()
                .zip(&self.ids)
                .map(|(k, &id)| self.read(k, Some(id)))
                .collect();
        }
        let m = self.raw.len();
        if m < 2 {
            return Err(ModelError::Inference(
                "leave-one-out removed every memory entry".into(),
            ));
        }
        let sum: Vec<f64> = (0..self.dim)
            .map(|j| self.raw.iter().map(|r| r[j]).sum())
            .collect();
        self.raw
            .iter()
            .zip(&self.ids)
            .map(|(h, &id)| {
                let mu: Vec<f64> = sum
                    .iter()
                    .zip(h)
                    .map(|(s, v)| (s - v) / (m - 1) as f64)
                    .collect();
                // Entries sharing id with h are masked, so their keys never
                // count; a placeholder skips encoding a possibly zero vector.
                let keys = self
                    .raw
                    .iter()
                    .zip(&self.ids)
                    .map(|(r, &other)| {
                        if other == id {
                            Ok(vec![0.0; self.dim])
                        } else {
                            self.encode_with(r, Some(&mu))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.read_keys(&keys, &self.encode_with(h, Some(&mu))?, Some(id))
            })
            .collect()
    }

    /// One out-of-fold read per written anchor, in `ids` order, even when
    /// the FIFO kept only the last `capacity` of them. An anchor still in
    /// the store gets its [`Self::loo_query`] row (itself masked, centre
    /// `mu_-i`). An anchor the FIFO already dropped is not in the store, so
    /// it reads every retained entry, centred on the retained mean. The rows
    /// stay 1-to-1 with the anchors' labels and base probabilities. Both
    /// row kinds are out-of-fold (no row reads its own entry), but a dropped
    /// anchor's read sees one more entry and the full retained centre.
    pub fn loo_query_for_all_anchors(
        &self,
        hidden: &[Vec<f64>],
        ids: &[usize],
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        if hidden.len() != ids.len() {
            return Err(ModelError::QwenInput(
                "hidden and ids must have the same length".into(),
            ));
        }
        let unique: std::collections::BTreeSet<usize> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            return Err(ModelError::QwenInput("duplicate query ID".into()));
        }
        if self.is_empty() {
            return Err(ModelError::Inference("memory is empty".into()));
        }
        let retained: BTreeMap<usize, usize> = self
            .ids
            .iter()
            .enumerate()
            .map(|(j, &id)| (id, j))
            .collect();
        let in_store = ids.iter().any(|id| retained.contains_key(id));
        let loo = if in_store {
            self.loo_query()?
        } else {
            Vec::new()
        };
        let dropped = match &self.center {
            None => None,
            Some(_) => {
                let m = self.raw.len() as f64;
                let mu: Vec<f64> = (0..self.dim)
                    .map(|j| self.raw.iter().map(|r| r[j]).sum::<f64>() / m)
                    .collect();
                let keys = self
                    .raw
                    .iter()
                    .map(|r| self.encode_with(r, Some(&mu)))
                    .collect::<Result<Vec<_>, _>>()?;
                Some((mu, keys))
            }
        };
        hidden
            .iter()
            .zip(ids)
            .map(|(h, id)| match (retained.get(id), &dropped) {
                (Some(&j), _) => Ok(loo[j].clone()),
                (None, None) => self.read(&self.encode(h)?, None),
                (None, Some((mu, keys))) => {
                    self.read_keys(keys, &self.encode_with(h, Some(mu))?, None)
                }
            })
            .collect()
    }
}

/// Memory mixing weights tried by [`select_memory_lambda`]: 0.1 to 1.0.
pub const DEFAULT_LAMBDA_GRID: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
/// Standard errors the memory gain must clear.
pub const MEMORY_SE_MULT: f64 = 2.0;

#[derive(Clone, Debug, Serialize)]
pub struct LambdaCandidate {
    pub lambda: f64,
    pub mean_gain: f64,
    pub se: f64,
    pub gain_over_null: Option<f64>,
    pub gain_over_null_se: Option<f64>,
    /// Gain over the uniform `1/K` row mixed at the same `lambda`.
    pub gain_over_uniform: f64,
    pub gain_over_uniform_se: f64,
    /// Gain over the leave-one-out Laplace prior mixed at the same
    /// `lambda`; set together with `gain_over_null`.
    pub gain_over_laplace: Option<f64>,
    pub gain_over_laplace_se: Option<f64>,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LambdaReport {
    pub n: usize,
    pub lambda: f64,
    pub candidates: Vec<LambdaCandidate>,
}

/// Mean and standard error (`ddof = 1`) of per-sample gains.
fn mean_se(gain: &[f64]) -> (f64, f64) {
    let n = gain.len() as f64;
    let mean = gain.iter().sum::<f64>() / n;
    let var = gain.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// The memory weight with the best paired out-of-fold log-likelihood gain
/// over `lambda = 0`, among those that pass; 0 when none does.
///
/// A `lambda` passes when its mean per-sample gain is above 0 and above
/// `se_mult` standard errors. With `null_probs` (the leave-one-out class
/// prior in [`InterLayerMctsRunner::fit_calibrated`]) the memory must also
/// beat that null mixed at the same `lambda` by `se_mult` standard errors: a
/// saturated base probe gains from mixing in anything, so the first test
/// alone would accept a memory that carries no more than the label
/// frequencies.
///
/// The memory must also beat the uniform row `1/K` mixed at the same
/// `lambda` (always), and, with `null_probs`, the leave-one-out Laplace
/// prior `(count_c - [c = y_i] + 1) / (n - 1 + K)`. On balanced labels the
/// leave-one-out prior puts `< 1/2` on the true label, so a constant
/// `[0.5, 0.5]` memory beats it while carrying no information; against the
/// uniform row its gain is exactly 0 and it fails.
pub fn select_memory_lambda(
    base_probs: &[Vec<f64>],
    memory_probs: &[Vec<f64>],
    labels: &[usize],
    grid: &[f64],
    se_mult: f64,
    null_probs: Option<&[Vec<f64>]>,
) -> Result<(f64, LambdaReport), ModelError> {
    let n = labels.len();
    if n < 2 {
        return Err(ModelError::QwenInput(
            "lambda selection needs at least 2 out-of-fold samples".into(),
        ));
    }
    let k = base_probs.first().map_or(0, Vec::len);
    let shaped = |p: &[Vec<f64>]| p.len() == n && p.iter().all(|r| r.len() == k);
    if k == 0
        || !shaped(base_probs)
        || !shaped(memory_probs)
        || null_probs.is_some_and(|p| !shaped(p))
    {
        return Err(ModelError::QwenInput(
            "base, memory and null probabilities and labels disagree in shape".into(),
        ));
    }
    if labels.iter().any(|&y| y >= k) {
        return Err(ModelError::QwenInput("label out of range".into()));
    }
    for row in base_probs
        .iter()
        .chain(memory_probs)
        .chain(null_probs.unwrap_or(&[]))
    {
        check_finite("lambda selection probabilities", row)?;
    }
    if !(se_mult.is_finite() && se_mult >= 0.0)
        || grid
            .iter()
            .any(|l| !(l.is_finite() && (0.0..=1.0).contains(l)))
    {
        return Err(ModelError::QwenInput(
            "se_mult must be finite >= 0 and every lambda in [0, 1]".into(),
        ));
    }
    let ln = |p: f64| p.max(PROB_FLOOR).ln();
    let mixed = |other: &[Vec<f64>], lam: f64, i: usize| {
        (1.0 - lam) * base_probs[i][labels[i]] + lam * other[i][labels[i]]
    };
    let uniform = vec![vec![1.0 / k as f64; k]; n];
    let mut counts = vec![0.0; k];
    for &y in labels {
        counts[y] += 1.0;
    }
    let laplace: Vec<Vec<f64>> = labels
        .iter()
        .map(|&y| {
            (0..k)
                .map(|c| (counts[c] - if c == y { 1.0 } else { 0.0 } + 1.0) / (n - 1 + k) as f64)
                .collect()
        })
        .collect();
    // Paired gain of the memory mixture over `other` mixed at `lam`.
    let excess_over = |other: &[Vec<f64>], lam: f64, what: &str| {
        let excess: Vec<f64> = (0..n)
            .map(|i| ln(mixed(memory_probs, lam, i)) - ln(mixed(other, lam, i)))
            .collect();
        let (em, es) = mean_se(&excess);
        if !(em.is_finite() && es.is_finite()) {
            return Err(ModelError::Inference(format!(
                "non-finite out-of-fold gain over the {what}"
            )));
        }
        Ok((em, es, em > 0.0 && em > se_mult * es))
    };
    let mut report = LambdaReport {
        n,
        lambda: 0.0,
        candidates: Vec::new(),
    };
    let mut best = (0.0, 0.0);
    for &lam in grid {
        let gain: Vec<f64> = (0..n)
            .map(|i| ln(mixed(memory_probs, lam, i)) - ln(base_probs[i][labels[i]]))
            .collect();
        let (mean, se) = mean_se(&gain);
        if !(mean.is_finite() && se.is_finite()) {
            return Err(ModelError::Inference("non-finite out-of-fold gain".into()));
        }
        let (um, us, uniform_ok) = excess_over(&uniform, lam, "uniform row")?;
        let mut passed = mean > 0.0 && mean > se_mult * se && uniform_ok;
        let mut candidate = LambdaCandidate {
            lambda: lam,
            mean_gain: mean,
            se,
            gain_over_null: None,
            gain_over_null_se: None,
            gain_over_uniform: um,
            gain_over_uniform_se: us,
            gain_over_laplace: None,
            gain_over_laplace_se: None,
            passed: false,
        };
        if let Some(null) = null_probs {
            let (em, es, null_ok) = excess_over(null, lam, "null")?;
            let (lm, ls, laplace_ok) = excess_over(&laplace, lam, "Laplace prior")?;
            passed = passed && null_ok && laplace_ok;
            candidate.gain_over_null = Some(em);
            candidate.gain_over_null_se = Some(es);
            candidate.gain_over_laplace = Some(lm);
            candidate.gain_over_laplace_se = Some(ls);
        }
        candidate.passed = passed;
        report.candidates.push(candidate);
        if passed && mean > best.1 {
            best = (lam, mean);
        }
    }
    report.lambda = best.0;
    Ok((best.0, report))
}

/// Settings of [`InterLayerMctsRunner::fit`].
#[derive(Clone, Debug)]
pub struct InterLayerConfig {
    /// Hook layers, the last decoder layer included; default
    /// [`default_hook_layers`].
    pub hook_layers: Option<Vec<usize>>,
    /// Softmax temperature of the label logits.
    pub tau: f64,
    /// No exit below this layer; default `max(2, num_layers / 2)`.
    pub min_exit_layer: Option<usize>,
    /// Required MCTS root visit-share margin of the exit label.
    pub mcts_margin_min: f64,
    /// Fit sparse MCTS layer weights; without them the readout is the
    /// current layer's corrected logits.
    pub fit_layer_weights: bool,
    pub top_k_hooks: usize,
    pub reward_scale: f64,
}

impl Default for InterLayerConfig {
    fn default() -> Self {
        Self {
            hook_layers: None,
            tau: 1.0,
            min_exit_layer: None,
            mcts_margin_min: 0.2,
            fit_layer_weights: true,
            top_k_hooks: 4,
            reward_scale: 10.0,
        }
    }
}

/// Per hook layer numbers of a fit.
#[derive(Clone, Debug, Serialize)]
pub struct LayerFitReport {
    pub layer: usize,
    /// `None` for the last layer, which reads the model's own logits.
    pub ridge_lambda: Option<f64>,
    pub loo_mse: Option<f64>,
    pub lipschitz: Option<f64>,
    pub loo_accuracy: f64,
    pub loo_log_likelihood: f64,
    /// `loo_accuracy + 0.1 * loo_log_likelihood`.
    pub utility: f64,
    pub weight: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InterLayerFitReport {
    pub anchors: usize,
    pub hook_layers: Vec<usize>,
    pub min_exit_layer: usize,
    pub tau: f64,
    pub layers: Vec<LayerFitReport>,
}

/// One checkpoint read during [`InterLayerMctsRunner::predict`].
#[derive(Clone, Debug, Serialize)]
pub struct CheckpointTrace {
    pub layer: usize,
    pub probs: Vec<f64>,
    pub gap: f64,
    /// `2 L_f delta_l / tau`; `None` at the last layer.
    pub bound: Option<f64>,
    pub delta: Option<f64>,
    /// Calibrated readout only: the contraction `g_l` of the tail bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gamma_hat: Option<f64>,
    pub stable: bool,
    pub mcts_margin: Option<f64>,
}

/// Outcome of one heuristic subspace-step early-exit pass.
#[derive(Clone, Debug, Serialize)]
pub struct LayerExitDecision {
    pub label: usize,
    pub probs: Vec<f64>,
    pub exit_layer: usize,
    pub early_exit: bool,
    pub layers_executed: usize,
    pub num_layers: usize,
    pub margin: f64,
    pub threshold: Option<f64>,
    pub lipschitz: Option<f64>,
    pub delta: Option<f64>,
    pub mcts_margin: Option<f64>,
    pub mcts_value: Option<f64>,
    #[serde(flatten)]
    pub ffn: FfnWork,
    pub active_hook_weights: BTreeMap<usize, f64>,
    pub checkpoints: Vec<CheckpointTrace>,
}

/// Fitted memories, layer weights and exit rule over one Qwen2 model.
#[derive(Clone, Debug)]
pub struct InterLayerMctsRunner {
    num_layers: usize,
    hook_layers: Vec<usize>,
    /// Output-head rows of the label tokens, grouped label by label.
    label_rows: Vec<Vec<f64>>,
    /// Tokens per label in `label_rows`; a label's logit is the
    /// `logsumexp` over its group (one token each for [`Self::fit`]).
    group_sizes: Vec<usize>,
    memories: BTreeMap<usize, InterLayerMemory>,
    /// Set by [`Self::fit_calibrated`]; replaces the ridge readout and
    /// exit rule in [`Self::predict`].
    calibrated: Option<CalibratedReadout>,
    layer_weights: BTreeMap<usize, f64>,
    tau: f64,
    min_exit_layer: usize,
    mcts_margin_min: f64,
    planner: LatentMctsPlanner,
}

fn to_f64(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(f64::from).collect()
}

impl InterLayerMctsRunner {
    /// Run every anchor through all layers, fit one [`InterLayerMemory`] per
    /// hook layer below the last against the full model's final label logits,
    /// and (optionally) MCTS layer weights on the anchors' LOO utilities.
    pub fn fit(
        model: &QwenModel,
        anchors: &[Vec<u32>],
        labels: &[usize],
        label_token_ids: &[u32],
        config: &InterLayerConfig,
    ) -> Result<(Self, InterLayerFitReport), ModelError> {
        let num_layers = model.config().num_layers;
        let k = label_token_ids.len();
        if k < 2 || anchors.len() != labels.len() || anchors.len() < 3 {
            return Err(ModelError::QwenInput(
                "inter-layer fit needs >= 2 labels and >= 3 labelled anchors".into(),
            ));
        }
        if let Some(bad) = labels.iter().find(|&&y| y >= k) {
            return Err(ModelError::QwenInput(format!("anchor label {bad} >= {k}")));
        }
        if !(config.tau.is_finite() && config.tau > 0.0) {
            return Err(ModelError::QwenInput("tau must be finite and > 0".into()));
        }
        if !(0.0..1.0).contains(&config.mcts_margin_min) {
            return Err(ModelError::QwenInput(
                "mcts_margin_min must be in [0, 1)".into(),
            ));
        }
        let hook_layers = config
            .hook_layers
            .clone()
            .unwrap_or_else(|| default_hook_layers(num_layers));
        let ascending = hook_layers.windows(2).all(|w| w[0] < w[1]);
        if hook_layers.len() < 2 || !ascending || hook_layers.last() != Some(&(num_layers - 1)) {
            return Err(ModelError::QwenInput(format!(
                "hook layers {hook_layers:?} must ascend, hold >= 2 layers and end at {}",
                num_layers - 1
            )));
        }
        let min_exit_layer = config
            .min_exit_layer
            .unwrap_or_else(|| 2.max(num_layers / 2));
        if min_exit_layer >= num_layers {
            return Err(ModelError::QwenInput(
                "min_exit_layer must index a layer".into(),
            ));
        }
        let label_rows: Vec<Vec<f64>> = model
            .lm_head_rows(label_token_ids)?
            .into_iter()
            .map(to_f64)
            .collect();
        let hooks: BTreeSet<usize> = hook_layers.iter().copied().collect();
        let mut states: BTreeMap<usize, Vec<Vec<f64>>> =
            hooks.iter().map(|&l| (l, Vec::new())).collect();
        for ids in anchors {
            let out = model.forward_with_hook(std::slice::from_ref(ids), &mut |obs| {
                if let Some(rows) = states.get_mut(&obs.layer) {
                    rows.push(to_f64(model.final_norm_last(obs.residual)?));
                }
                Ok(HookAction::Continue)
            })?;
            if out.exited_at.is_some() {
                return Err(ModelError::Inference("anchor pass exited early".into()));
            }
        }
        let last = num_layers - 1;
        let singletons = vec![1; k];
        let final_logits = states[&last]
            .iter()
            .map(|f| group_logits(&label_rows, &singletons, f))
            .collect::<Result<Vec<_>, _>>()?;
        let mut memories = BTreeMap::new();
        let mut layers = Vec::new();
        let mut utilities = BTreeMap::new();
        for &layer in &hook_layers {
            let (loo, mem) = if layer == last {
                (final_logits.clone(), None)
            } else {
                let mem = InterLayerMemory::fit(layer, &states[&layer], &final_logits)?;
                (mem.loo_logits.clone(), Some(mem))
            };
            let (acc, ll) = utility_terms(&loo, labels, config.tau);
            let utility = acc + 0.1 * ll;
            utilities.insert(layer, utility);
            layers.push(LayerFitReport {
                layer,
                ridge_lambda: mem.as_ref().map(|m| m.ridge_lambda),
                loo_mse: mem.as_ref().map(|m| m.loo_mse),
                lipschitz: mem.as_ref().map(|m| m.lipschitz),
                loo_accuracy: acc,
                loo_log_likelihood: ll,
                utility,
                weight: None,
            });
            if let Some(mem) = mem {
                memories.insert(layer, mem);
            }
        }
        let planner = LatentMctsPlanner {
            num_simulations: 32,
            temperature: 1.0,
            max_depth: 2,
            ..Default::default()
        };
        let layer_weights = if config.fit_layer_weights {
            fit_layer_weights(
                &utilities,
                config.top_k_hooks,
                config.reward_scale,
                &planner,
            )?
        } else {
            BTreeMap::new()
        };
        for row in &mut layers {
            row.weight = layer_weights.get(&row.layer).copied();
        }
        let report = InterLayerFitReport {
            anchors: anchors.len(),
            hook_layers: hook_layers.clone(),
            min_exit_layer,
            tau: config.tau,
            layers,
        };
        Ok((
            Self {
                num_layers,
                hook_layers,
                label_rows,
                group_sizes: vec![1; k],
                memories,
                calibrated: None,
                layer_weights,
                tau: config.tau,
                min_exit_layer,
                mcts_margin_min: config.mcts_margin_min,
                planner,
            },
            report,
        ))
    }

    pub fn layer_weights(&self) -> &BTreeMap<usize, f64> {
        &self.layer_weights
    }

    pub fn memory(&self, layer: usize) -> Option<&InterLayerMemory> {
        self.memories.get(&layer)
    }

    /// MCTS over the label simplex: (best label, visit-share margin, root Q).
    fn mcts_margin(&self, probs: &[f64]) -> Result<(usize, f64, f64), ModelError> {
        let z: Vec<f64> = probs.iter().map(|p| p.max(PROB_FLOOR).ln()).collect();
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        let z: Vec<f64> = z.iter().map(|v| v - mean).collect();
        let plan = self.planner.plan(&z, &simplex_etf(z.len())?)?;
        let mut shares = plan.policy.clone();
        shares.sort_by(|a, b| b.total_cmp(a));
        let second = shares.get(1).copied().unwrap_or(0.0);
        Ok((
            plan.best,
            plan.policy[plan.best] - second,
            plan.root_q[plan.best],
        ))
    }

    /// One pass over `ids` that reads the label distribution at the hook
    /// layers and stops after the first layer that passes the heuristic subspace-step rule.
    pub fn predict(&self, model: &QwenModel, ids: &[u32]) -> Result<LayerExitDecision, ModelError> {
        if model.config().num_layers != self.num_layers {
            return Err(ModelError::QwenInput(
                "runner was fitted on another depth".into(),
            ));
        }
        if let Some(cal) = &self.calibrated {
            return self.predict_calibrated(model, ids, cal);
        }
        let last = self.num_layers - 1;
        let hooks: BTreeSet<usize> = self.hook_layers.iter().copied().collect();
        let mut prev_feature: Option<Vec<f64>> = None;
        let mut cum_z: Option<Vec<f64>> = None;
        let mut cum_w = 0.0;
        // Feature and top-1 label of the last checkpoint with weight
        // > WEIGHT_EPS: a zero-weight checkpoint is no observation, so it is
        // no reference for the step `delta` or for stability.
        let mut prev_top: Option<usize> = None;
        let mut trace: Vec<CheckpointTrace> = Vec::new();
        let mut exit: Option<(f64, f64, f64, f64, f64)> = None;
        let out = model.forward_with_hook(&[ids.to_vec()], &mut |obs| {
            let layer = obs.layer;
            if !hooks.contains(&layer) {
                return Ok(HookAction::Continue);
            }
            let feature = to_f64(model.final_norm_last(obs.residual)?);
            let layer_w = self.layer_weights.get(&layer).copied().unwrap_or(0.0);
            let weighted_ok = self.layer_weights.is_empty() || layer_w > WEIGHT_EPS;
            let prev = if weighted_ok {
                prev_feature.replace(feature.clone())
            } else {
                None
            };
            let (z, mem) = match self.memories.get(&layer) {
                Some(mem) => (mem.predict(&feature)?, Some(mem)),
                None => (self.probe_logits(&feature)?, None),
            };
            let readout = if self.layer_weights.is_empty() {
                z
            } else {
                let w = self.layer_weights.get(&layer).copied().unwrap_or(0.0);
                if w > WEIGHT_EPS && layer != last {
                    cum_w += w;
                    let acc = cum_z.get_or_insert_with(|| vec![0.0; z.len()]);
                    for (a, v) in acc.iter_mut().zip(&z) {
                        *a += w * v;
                    }
                }
                if layer == last {
                    z.clone()
                } else {
                    match &cum_z {
                        Some(acc) if cum_w > 0.0 => acc.iter().map(|a| a / cum_w).collect(),
                        _ => z,
                    }
                }
            };
            let probs = softmax(&readout.iter().map(|v| v / self.tau).collect::<Vec<_>>());
            check_finite("checkpoint probabilities", &probs)?;
            let gap = top2_gap(&probs);
            let top = argmax(&probs);
            let stable = weighted_ok && prev_top == Some(top);
            if weighted_ok {
                prev_top = Some(top);
            }
            let delta = match (mem, prev.as_deref()) {
                (Some(m), Some(p)) => Some(m.projected_delta(&feature, p)?),
                _ => None,
            };
            let bound = match (mem, delta) {
                (Some(m), Some(d)) => Some(2.0 * m.lipschitz * d / self.tau),
                _ => None,
            };
            let mut row = CheckpointTrace {
                layer,
                probs,
                gap,
                bound,
                delta,
                gamma_hat: None,
                stable,
                mcts_margin: None,
            };
            let mut action = HookAction::Continue;
            if let (Some(b), Some(m)) = (bound, mem) {
                if layer < last && layer >= self.min_exit_layer && stable && weighted_ok && gap > b
                {
                    let (best, margin, value) = self.mcts_margin(&row.probs)?;
                    row.mcts_margin = Some(margin);
                    if best == top && margin > self.mcts_margin_min {
                        exit = Some((b, m.lipschitz, row.delta.unwrap_or(0.0), margin, value));
                        action = HookAction::Exit;
                    }
                }
            }
            trace.push(row);
            Ok(action)
        })?;
        let final_row = trace
            .last()
            .ok_or_else(|| ModelError::Inference("no hook layer was read".into()))?;
        let early_exit = out.exited_at.is_some();
        if early_exit != exit.is_some() || (!early_exit && final_row.layer != last) {
            return Err(ModelError::Inference(
                "hooked pass and exit decision disagree".into(),
            ));
        }
        let (threshold, lipschitz, delta, mcts_margin, mcts_value) = match exit {
            Some((b, l, d, m, v)) => (Some(b), Some(l), Some(d), Some(m), Some(v)),
            None => (None, None, None, None, None),
        };
        Ok(LayerExitDecision {
            label: argmax(&final_row.probs),
            probs: final_row.probs.clone(),
            exit_layer: final_row.layer,
            early_exit,
            layers_executed: out.layers_executed,
            num_layers: self.num_layers,
            margin: final_row.gap,
            threshold,
            lipschitz,
            delta,
            mcts_margin,
            mcts_value,
            ffn: out.ffn,
            active_hook_weights: self.layer_weights.clone(),
            checkpoints: trace,
        })
    }
}

/// Settings of [`InterLayerMctsRunner::fit_calibrated`]; the defaults are
/// those of the Python `MctsLayerwiseQwenRunner` constructor and
/// `fit_memories`.
#[derive(Clone, Debug)]
pub struct CalibratedConfig {
    /// Hook layers, the last decoder layer included; the others are the
    /// exit checkpoints. Default [`default_hook_layers`].
    pub hook_layers: Option<Vec<usize>>,
    /// Scale of the tail bound `2 tau g / (1 - g) * drift` (not a softmax
    /// temperature).
    pub tau: f64,
    /// Floor of the empirical contraction `g`, in `(0, 0.95)`.
    pub gamma: f64,
    /// No exit below this layer; default `max(2, num_layers / 2)`.
    pub min_exit_layer: Option<usize>,
    /// Required MCTS root visit-share margin of the exit label.
    pub mcts_margin_min: f64,
    pub fit_layer_weights: bool,
    pub top_k_hooks: usize,
    pub reward_scale: f64,
    pub memory_beta: f64,
    pub memory_capacity: usize,
}

impl Default for CalibratedConfig {
    fn default() -> Self {
        Self {
            hook_layers: None,
            tau: 1.0,
            gamma: 0.5,
            min_exit_layer: None,
            mcts_margin_min: 0.2,
            fit_layer_weights: true,
            top_k_hooks: 4,
            reward_scale: 10.0,
            memory_beta: 12.0,
            memory_capacity: 256,
        }
    }
}

impl CalibratedConfig {
    /// The checks that need no model, so callers can fail before any
    /// forward pass. The memory keeps at least 2 entries: with one, the
    /// leave-one-out read of that entry has nothing left to read.
    pub fn validate(&self) -> Result<(), ModelError> {
        if !(self.tau.is_finite() && self.tau >= 0.0) {
            return Err(ModelError::QwenInput("tau must be finite and >= 0".into()));
        }
        if !(self.gamma > 0.0 && self.gamma < GAMMA_MAX) {
            return Err(ModelError::QwenInput(format!(
                "gamma must be in (0, {GAMMA_MAX})"
            )));
        }
        if !(0.0..1.0).contains(&self.mcts_margin_min) {
            return Err(ModelError::QwenInput(
                "mcts_margin_min must be in [0, 1)".into(),
            ));
        }
        if self.memory_capacity < 3 {
            return Err(ModelError::QwenInput(format!(
                "memory_capacity must be >= 3, got {}",
                self.memory_capacity
            )));
        }
        if !(self.memory_beta.is_finite() && self.memory_beta > 0.0) {
            return Err(ModelError::QwenInput(
                "memory_beta must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

const GAMMA_MAX: f64 = 0.95;
const DRIFT_EPS: f64 = 1e-6;
/// A hook weight at or below this adds nothing to the weighted readout, so
/// that checkpoint is no new observation for the exit rule.
const WEIGHT_EPS: f64 = 1e-12;
/// Search temperature of the calibrated layer-weight MCTS (Python value).
const CALIBRATED_WEIGHT_TEMPERATURE: f64 = 0.25;

/// One checkpoint of the calibrated weighted readout.
#[derive(Clone, Debug)]
struct CheckpointEval {
    probs: Vec<f64>,
    gap: f64,
    /// `2 tau g / (1 - g) * drift`; `None` unless this checkpoint is at
    /// least the second effective one.
    bound: Option<f64>,
    drift: Option<f64>,
    gamma_hat: Option<f64>,
    stable: bool,
}

impl CheckpointEval {
    /// The drift part of the exit rule: the top-1 label held since the
    /// previous effective checkpoint and its gap beats the tail bound.
    fn passes_bound(&self) -> bool {
        self.stable && self.bound.is_some_and(|b| self.gap > b)
    }
}

/// Running state of the calibrated weighted readout along one pass.
///
/// With weights, only a checkpoint of weight `> WEIGHT_EPS` is an
/// observation: it moves the weighted logit mean, counts in
/// `effective_steps` and updates the drift history. A zero-weight
/// checkpoint leaves the mean unchanged, so its drift would be 0 and its
/// bound 0; it must not count as evidence of convergence. The first
/// effective checkpoint only seeds the history, so the exit rule needs two.
#[derive(Clone, Debug)]
struct WeightedExitTracker {
    weighted: bool,
    gamma: f64,
    tau: f64,
    cum_w: f64,
    cum_z: Option<Vec<f64>>,
    prev: Option<(Vec<f64>, Option<f64>)>,
    effective_steps: usize,
}

impl WeightedExitTracker {
    fn new(weighted: bool, gamma: f64, tau: f64) -> Self {
        Self {
            weighted,
            gamma,
            tau,
            cum_w: 0.0,
            cum_z: None,
            prev: None,
            effective_steps: 0,
        }
    }

    /// Fold in one checkpoint of weight `w`, trajectory logits `traj` and
    /// instant distribution `instant`. `exit_checkpoint` is false at the
    /// last layer, which reads the readout but feeds no exit rule.
    fn observe(
        &mut self,
        w: f64,
        traj: &[f64],
        instant: Vec<f64>,
        exit_checkpoint: bool,
    ) -> Result<CheckpointEval, ModelError> {
        let effective = !self.weighted || w > WEIGHT_EPS;
        if self.weighted && effective {
            self.cum_w += w;
            let acc = self.cum_z.get_or_insert_with(|| vec![0.0; traj.len()]);
            for (a, v) in acc.iter_mut().zip(traj) {
                *a += w * v;
            }
        }
        let probs = match &self.cum_z {
            Some(acc) if self.weighted && self.cum_w > 0.0 => {
                softmax(&acc.iter().map(|a| a / self.cum_w).collect::<Vec<_>>())
            }
            _ => instant,
        };
        check_finite("checkpoint probabilities", &probs)?;
        let mut eval = CheckpointEval {
            gap: top2_gap(&probs),
            probs,
            bound: None,
            drift: None,
            gamma_hat: None,
            stable: false,
        };
        if !(effective && exit_checkpoint) {
            return Ok(eval);
        }
        self.effective_steps += 1;
        let prev_drift = match self.prev.take() {
            None => None,
            Some((prev_probs, prev_drift)) => {
                let drift = eval
                    .probs
                    .iter()
                    .zip(&prev_probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                let g = match prev_drift {
                    None => self.gamma,
                    Some(d) => (drift / d.max(DRIFT_EPS)).max(self.gamma).min(GAMMA_MAX),
                };
                eval.bound = Some(2.0 * self.tau * g / (1.0 - g) * drift);
                eval.drift = Some(drift);
                eval.gamma_hat = Some(g);
                eval.stable = argmax(&prev_probs) == argmax(&eval.probs);
                Some(drift)
            }
        };
        debug_assert!(eval.bound.is_none() || self.effective_steps >= 2);
        self.prev = Some((eval.probs.clone(), prev_drift));
        Ok(eval)
    }
}

/// Last-position states of one full pass at the hook layers.
#[derive(Clone, Debug)]
pub struct HookStates {
    /// Residual after the layer, before the final norm: the memory key.
    pub raw: BTreeMap<usize, Vec<f64>>,
    /// Final-norm state: the verbalizer probe input.
    pub normed: BTreeMap<usize, Vec<f64>>,
    pub ffn: FfnWork,
    pub tokens: usize,
}

impl HookStates {
    fn layer(&self, layer: usize) -> Result<(&[f64], &[f64]), ModelError> {
        match (self.raw.get(&layer), self.normed.get(&layer)) {
            (Some(r), Some(n)) => Ok((r, n)),
            _ => Err(ModelError::QwenInput(format!(
                "anchor states lack hook layer {layer}"
            ))),
        }
    }
}

fn residual_last(residual: &Tensor) -> Result<Vec<f64>, ModelError> {
    let err = candle_err("residual last");
    let (b, t, _) = residual.dims3().map_err(&err)?;
    if b != 1 {
        return Err(ModelError::QwenInput(format!(
            "hook states need one row, got {b}"
        )));
    }
    residual
        .narrow(1, t - 1, 1)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .map(to_f64)
        .map_err(&err)
}

#[derive(Clone, Debug)]
struct CalibratedLayer {
    stats: VerbalizerLogitStats,
    memory: EpisodicManifoldMemory,
    lambda: f64,
}

#[derive(Clone, Debug)]
struct CalibratedReadout {
    layers: BTreeMap<usize, CalibratedLayer>,
    gamma: f64,
}

/// Per hook layer numbers of a calibrated fit.
#[derive(Clone, Debug, Serialize)]
pub struct CalibratedLayerReport {
    pub layer: usize,
    pub probe_logit_mean: Vec<f64>,
    pub probe_logit_std: Vec<f64>,
    pub memory_lambda: f64,
    pub memory_gate: LambdaReport,
    /// Of the leave-one-out standardised (and memory-mixed) readout.
    pub loo_accuracy: f64,
    pub loo_log_likelihood: f64,
    /// `loo_accuracy + 0.1 * loo_log_likelihood`.
    pub utility: f64,
    pub weight: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CalibratedFitReport {
    pub anchors: usize,
    pub hook_layers: Vec<usize>,
    pub min_exit_layer: usize,
    pub tau: f64,
    pub gamma: f64,
    pub layers: Vec<CalibratedLayerReport>,
}

/// Outcome of [`InterLayerMctsRunner::score_candidates_mcts_pruned`].
#[derive(Clone, Debug, Serialize)]
pub struct PrunedCandidateResult {
    /// Survivor with the highest deep probability of the target label.
    pub best: usize,
    /// Candidates kept by the MCTS, best first.
    pub survivors: Vec<usize>,
    /// Target-label probability of every candidate at the cut layer.
    pub shallow_values: Vec<f64>,
    /// Target-label probability of every survivor at the last layer.
    pub deep_values: BTreeMap<usize, f64>,
    /// Decoder layers really run: `shallow + deep`.
    pub layers_executed: usize,
    pub shallow_layers_executed: usize,
    /// Survivor passes start again at layer 0 (see the method docs), so
    /// they run every layer.
    pub deep_layers_executed: usize,
    /// `K * num_layers`: every candidate through every layer.
    pub full_layers: usize,
    pub mcts_visits: Vec<u64>,
}

impl InterLayerMctsRunner {
    fn probe_logits(&self, normed: &[f64]) -> Result<Vec<f64>, ModelError> {
        group_logits(&self.label_rows, &self.group_sizes, normed)
    }

    /// One full pass over `ids` that keeps the last-position raw and
    /// final-norm states at `layers`.
    pub fn capture_hook_states(
        model: &QwenModel,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<HookStates, ModelError> {
        let num_layers = model.config().num_layers;
        let set: BTreeSet<usize> = layers.iter().copied().collect();
        if set.is_empty() || set.iter().any(|&l| l >= num_layers) {
            return Err(ModelError::QwenInput(format!(
                "capture layers {layers:?} must be nonempty and below {num_layers}"
            )));
        }
        let mut raw = BTreeMap::new();
        let mut normed = BTreeMap::new();
        let out = model.forward_with_hook(&[ids.to_vec()], &mut |obs| {
            if set.contains(&obs.layer) {
                raw.insert(obs.layer, residual_last(obs.residual)?);
                normed.insert(obs.layer, to_f64(model.final_norm_last(obs.residual)?));
            }
            Ok(HookAction::Continue)
        })?;
        if out.exited_at.is_some() || out.layers_executed != num_layers {
            return Err(ModelError::Inference("capture pass exited early".into()));
        }
        Ok(HookStates {
            raw,
            normed,
            ffn: out.ffn,
            tokens: ids.len(),
        })
    }

    /// Fit the calibrated readout from anchor states (one full pass each,
    /// see [`Self::capture_hook_states`]): per hook layer the verbalizer
    /// logit stats, an [`EpisodicManifoldMemory`] over the raw residuals
    /// centred on their anchor mean, its weight from
    /// [`select_memory_lambda`] against the leave-one-out class prior, and
    /// (optionally) MCTS layer weights on the leave-one-out utilities.
    /// `label_groups[c]` are the token ids of label `c`.
    pub fn fit_calibrated(
        model: &QwenModel,
        anchors: &[HookStates],
        labels: &[usize],
        label_groups: &[Vec<u32>],
        config: &CalibratedConfig,
    ) -> Result<(Self, CalibratedFitReport), ModelError> {
        config.validate()?;
        let num_layers = model.config().num_layers;
        let k = label_groups.len();
        let n = anchors.len();
        if k < 2 || label_groups.iter().any(Vec::is_empty) || n != labels.len() || n < 3 {
            return Err(ModelError::QwenInput(
                "calibrated fit needs >= 2 nonempty label groups and >= 3 labelled anchors".into(),
            ));
        }
        if let Some(bad) = labels.iter().find(|&&y| y >= k) {
            return Err(ModelError::QwenInput(format!("anchor label {bad} >= {k}")));
        }
        let hook_layers = config
            .hook_layers
            .clone()
            .unwrap_or_else(|| default_hook_layers(num_layers));
        let ascending = hook_layers.windows(2).all(|w| w[0] < w[1]);
        if hook_layers.len() < 2 || !ascending || hook_layers.last() != Some(&(num_layers - 1)) {
            return Err(ModelError::QwenInput(format!(
                "hook layers {hook_layers:?} must ascend, hold >= 2 layers and end at {}",
                num_layers - 1
            )));
        }
        let min_exit_layer = config
            .min_exit_layer
            .unwrap_or_else(|| 2.max(num_layers / 2));
        if min_exit_layer >= num_layers {
            return Err(ModelError::QwenInput(
                "min_exit_layer must index a layer".into(),
            ));
        }
        let flat: Vec<u32> = label_groups.iter().flatten().copied().collect();
        let label_rows: Vec<Vec<f64>> =
            model.lm_head_rows(&flat)?.into_iter().map(to_f64).collect();
        let group_sizes: Vec<usize> = label_groups.iter().map(Vec::len).collect();
        let mut counts = vec![0.0; k];
        for &y in labels {
            counts[y] += 1.0;
        }
        let loo_prior: Vec<Vec<f64>> = labels
            .iter()
            .map(|&y| {
                (0..k)
                    .map(|c| (counts[c] - if c == y { 1.0 } else { 0.0 }) / (n - 1) as f64)
                    .collect()
            })
            .collect();
        let ids: Vec<usize> = (0..n).collect();
        let mut layers = BTreeMap::new();
        let mut reports = Vec::new();
        let mut utilities = BTreeMap::new();
        for &layer in &hook_layers {
            let mut raw = Vec::with_capacity(n);
            let mut z = Vec::with_capacity(n);
            for a in anchors {
                let (r, f) = a.layer(layer)?;
                raw.push(r.to_vec());
                z.push(group_logits(&label_rows, &group_sizes, f)?);
            }
            let dim = raw[0].len();
            if raw.iter().any(|r| r.len() != dim) {
                return Err(ModelError::QwenInput(format!(
                    "layer {layer}: ragged anchor states"
                )));
            }
            // The FIFO keeps the last `memory_capacity` anchors; centre on
            // those, the same centre the out-of-fold reads use.
            let kept = &raw[n.saturating_sub(config.memory_capacity)..];
            let center: Vec<f64> = (0..dim)
                .map(|j| kept.iter().map(|r| r[j]).sum::<f64>() / kept.len() as f64)
                .collect();
            let mut memory = EpisodicManifoldMemory::new(
                dim,
                k,
                config.memory_capacity,
                config.memory_beta,
                Some(center),
            )?;
            memory.write(&raw, labels, &ids)?;
            let stats = VerbalizerLogitStats::fit(&z)?;
            let base: Vec<Vec<f64>> = VerbalizerLogitStats::loo_standardize(&z)?
                .iter()
                .map(|r| softmax(r))
                .collect();
            let memory_loo = memory.loo_query_for_all_anchors(&raw, &ids)?;
            let (lambda, gate) = select_memory_lambda(
                &base,
                &memory_loo,
                labels,
                &DEFAULT_LAMBDA_GRID,
                MEMORY_SE_MULT,
                Some(&loo_prior),
            )?;
            let mut acc = 0.0;
            let mut ll = 0.0;
            for ((b, m), &y) in base.iter().zip(&memory_loo).zip(labels) {
                let p: Vec<f64> = b
                    .iter()
                    .zip(m)
                    .map(|(bv, mv)| (1.0 - lambda) * bv + lambda * mv)
                    .collect();
                if argmax(&p) == y {
                    acc += 1.0;
                }
                ll += p[y].max(PROB_FLOOR).ln();
            }
            let (acc, ll) = (acc / n as f64, ll / n as f64);
            let utility = acc + 0.1 * ll;
            check_finite("calibrated layer utility", &[utility])?;
            utilities.insert(layer, utility);
            reports.push(CalibratedLayerReport {
                layer,
                probe_logit_mean: stats.mean.clone(),
                probe_logit_std: stats.std.clone(),
                memory_lambda: lambda,
                memory_gate: gate,
                loo_accuracy: acc,
                loo_log_likelihood: ll,
                utility,
                weight: None,
            });
            layers.insert(
                layer,
                CalibratedLayer {
                    stats,
                    memory,
                    lambda,
                },
            );
        }
        let planner = LatentMctsPlanner {
            num_simulations: 32,
            temperature: 1.0,
            max_depth: 2,
            ..Default::default()
        };
        let layer_weights = if config.fit_layer_weights {
            mcts_layer_weights(
                &utilities,
                config.top_k_hooks,
                config.reward_scale,
                CALIBRATED_WEIGHT_TEMPERATURE,
                &planner,
            )?
        } else {
            BTreeMap::new()
        };
        for row in &mut reports {
            row.weight = layer_weights.get(&row.layer).copied();
        }
        let report = CalibratedFitReport {
            anchors: n,
            hook_layers: hook_layers.clone(),
            min_exit_layer,
            tau: config.tau,
            gamma: config.gamma,
            layers: reports,
        };
        Ok((
            Self {
                num_layers,
                hook_layers,
                label_rows,
                group_sizes,
                memories: BTreeMap::new(),
                calibrated: Some(CalibratedReadout {
                    layers,
                    gamma: config.gamma,
                }),
                layer_weights,
                tau: config.tau,
                min_exit_layer,
                mcts_margin_min: config.mcts_margin_min,
                planner,
            },
            report,
        ))
    }

    /// Calibrated label distribution at `layer` and its trajectory logits:
    /// `softmax((z - mean) / std)`, mixed with the memory read when its
    /// weight is positive (then the logits are the log of the mixture).
    fn calibrated_probs(
        &self,
        cal: &CalibratedReadout,
        layer: usize,
        raw: &[f64],
        normed: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
        let entry = cal.layers.get(&layer).ok_or_else(|| {
            ModelError::QwenInput(format!(
                "calibrated readout has no stats at layer {layer}; refit with it"
            ))
        })?;
        let z = entry.stats.standardize(&self.probe_logits(normed)?)?;
        let mut p = softmax(&z);
        let mut traj = z;
        if entry.lambda > 0.0 {
            let m = entry.memory.query(raw, None)?;
            p = p
                .iter()
                .zip(&m)
                .map(|(a, b)| (1.0 - entry.lambda) * a + entry.lambda * b)
                .collect();
            traj = p.iter().map(|v| v.max(PROB_FLOOR).ln()).collect();
        }
        check_finite("calibrated probabilities", &p)?;
        Ok((p, traj))
    }

    /// Label distribution at one layer from its raw and final-norm states,
    /// in whichever readout the runner was fitted with.
    pub fn layer_probs(
        &self,
        layer: usize,
        raw: &[f64],
        normed: &[f64],
    ) -> Result<Vec<f64>, ModelError> {
        if let Some(cal) = &self.calibrated {
            return Ok(self.calibrated_probs(cal, layer, raw, normed)?.0);
        }
        let z = match self.memories.get(&layer) {
            Some(m) => m.predict(normed)?,
            None => self.probe_logits(normed)?,
        };
        let p = softmax(&z.iter().map(|v| v / self.tau).collect::<Vec<_>>());
        check_finite("layer probabilities", &p)?;
        Ok(p)
    }

    /// The Python `predict`: weighted trajectory of the calibrated readout,
    /// exit at a checkpoint `>= min_exit_layer` once the top-1 label held
    /// over two effective checkpoints (weight `> WEIGHT_EPS`, see
    /// [`WeightedExitTracker`]), its gap beats the drift tail bound and the
    /// MCTS confirms it with a visit margin above `mcts_margin_min`.
    fn predict_calibrated(
        &self,
        model: &QwenModel,
        ids: &[u32],
        cal: &CalibratedReadout,
    ) -> Result<LayerExitDecision, ModelError> {
        let last = self.num_layers - 1;
        let hooks: BTreeSet<usize> = self.hook_layers.iter().copied().collect();
        let mut tracker =
            WeightedExitTracker::new(!self.layer_weights.is_empty(), cal.gamma, self.tau);
        let mut trace: Vec<CheckpointTrace> = Vec::new();
        let mut exit: Option<(f64, f64, f64, f64)> = None;
        let out = model.forward_with_hook(&[ids.to_vec()], &mut |obs| {
            let layer = obs.layer;
            if !hooks.contains(&layer) {
                return Ok(HookAction::Continue);
            }
            let raw = residual_last(obs.residual)?;
            let normed = to_f64(model.final_norm_last(obs.residual)?);
            let (instant, z) = self.calibrated_probs(cal, layer, &raw, &normed)?;
            let w = self.layer_weights.get(&layer).copied().unwrap_or(0.0);
            let eval = tracker.observe(w, &z, instant, layer < last)?;
            let mut row = CheckpointTrace {
                layer,
                probs: eval.probs.clone(),
                gap: eval.gap,
                bound: eval.bound,
                delta: eval.drift,
                gamma_hat: eval.gamma_hat,
                stable: eval.stable,
                mcts_margin: None,
            };
            let mut action = HookAction::Continue;
            if layer < last && layer >= self.min_exit_layer && eval.passes_bound() {
                let (best, margin, value) = self.mcts_margin(&eval.probs)?;
                row.mcts_margin = Some(margin);
                if best == argmax(&eval.probs) && margin > self.mcts_margin_min {
                    let (bound, drift) = (eval.bound.unwrap_or(0.0), eval.drift.unwrap_or(0.0));
                    exit = Some((bound, drift, margin, value));
                    action = HookAction::Exit;
                }
            }
            trace.push(row);
            Ok(action)
        })?;
        let final_row = trace
            .last()
            .ok_or_else(|| ModelError::Inference("no hook layer was read".into()))?;
        let early_exit = out.exited_at.is_some();
        if early_exit != exit.is_some()
            || (!early_exit && final_row.layer != last)
            || out.exited_at.is_some_and(|l| l != final_row.layer)
        {
            return Err(ModelError::Inference(
                "hooked pass and exit decision disagree".into(),
            ));
        }
        let (threshold, delta, mcts_margin, mcts_value) = match exit {
            Some((b, d, m, v)) => (Some(b), Some(d), Some(m), Some(v)),
            None => (None, None, None, None),
        };
        Ok(LayerExitDecision {
            label: argmax(&final_row.probs),
            probs: final_row.probs.clone(),
            exit_layer: final_row.layer,
            early_exit,
            layers_executed: out.layers_executed,
            num_layers: self.num_layers,
            margin: final_row.gap,
            threshold,
            lipschitz: None,
            delta,
            mcts_margin,
            mcts_value,
            ffn: out.ffn,
            active_hook_weights: self.layer_weights.clone(),
            checkpoints: trace,
        })
    }

    /// The calibrated weighted readout after the last layer from one capture
    /// pass, never exiting: what [`Self::predict`] returns when no
    /// checkpoint exits. `equal_weight` replaces the MCTS layer weights with
    /// `1 / |hooks|` on every hook layer (the no-MCTS ablation).
    pub fn full_depth_readout(
        &self,
        states: &HookStates,
        equal_weight: bool,
    ) -> Result<Vec<f64>, ModelError> {
        let cal = self.calibrated.as_ref().ok_or_else(|| {
            ModelError::QwenInput("full_depth_readout needs a calibrated runner".into())
        })?;
        let equal = 1.0 / self.hook_layers.len() as f64;
        let weighted = equal_weight || !self.layer_weights.is_empty();
        let mut tracker = WeightedExitTracker::new(weighted, cal.gamma, self.tau);
        let mut probs = None;
        for &layer in &self.hook_layers {
            let (raw, normed) = states.layer(layer)?;
            let (instant, z) = self.calibrated_probs(cal, layer, raw, normed)?;
            let w = if equal_weight {
                equal
            } else {
                self.layer_weights.get(&layer).copied().unwrap_or(0.0)
            };
            probs = Some(tracker.observe(w, &z, instant, false)?.probs);
        }
        probs.ok_or_else(|| ModelError::Inference("no hook layer was read".into()))
    }

    /// Rank `candidates` (token rows) by the probability of `target_label`.
    ///
    /// Every candidate runs layers `0..shallow_layers-1` only and is read at
    /// layer `shallow_layers - 1`. An MCTS over those values keeps `top_m`
    /// by root visits (ties by root value, then prior); only the survivors
    /// run the deep layers and are read at the last layer. The pruned
    /// candidates never run a layer past the cut.
    ///
    /// `QwenModel::forward_with_hook` always starts at the embeddings, so a
    /// survivor's deep pass recomputes its shallow layers too (the Python
    /// runner replays its cached residual instead). `layers_executed`
    /// counts those layers, so the pass saves layers only when
    /// `K * shallow + top_m * L < K * L`.
    pub fn score_candidates_mcts_pruned(
        &self,
        model: &QwenModel,
        candidates: &[Vec<u32>],
        target_label: usize,
        shallow_layers: usize,
        top_m: usize,
    ) -> Result<PrunedCandidateResult, ModelError> {
        let k = candidates.len();
        if model.config().num_layers != self.num_layers {
            return Err(ModelError::QwenInput(
                "runner was fitted on another depth".into(),
            ));
        }
        if !(1..=k).contains(&top_m) {
            return Err(ModelError::QwenInput(
                "top_m must be in [1, number of candidates]".into(),
            ));
        }
        if !(1..self.num_layers).contains(&shallow_layers) {
            return Err(ModelError::QwenInput(
                "shallow_layers must be in [1, num_layers)".into(),
            ));
        }
        if target_label >= self.group_sizes.len() {
            return Err(ModelError::QwenInput("target_label out of range".into()));
        }
        let cut = shallow_layers - 1;
        let last = self.num_layers - 1;
        let mut shallow = Vec::with_capacity(k);
        let mut shallow_layers_executed = 0;
        for ids in candidates {
            let mut state = None;
            let out = model.forward_with_hook(std::slice::from_ref(ids), &mut |obs| {
                if obs.layer == cut {
                    state = Some((
                        residual_last(obs.residual)?,
                        to_f64(model.final_norm_last(obs.residual)?),
                    ));
                    return Ok(HookAction::Exit);
                }
                Ok(HookAction::Continue)
            })?;
            let (raw, normed) = state.ok_or_else(|| {
                ModelError::Inference("shallow pass never reached the cut".into())
            })?;
            if out.exited_at != Some(cut) || out.layers_executed != shallow_layers {
                return Err(ModelError::Inference(
                    "shallow pass did not stop at the cut layer".into(),
                ));
            }
            shallow_layers_executed += out.layers_executed;
            shallow.push(self.layer_probs(cut, &raw, &normed)?[target_label]);
        }
        let z: Vec<f64> = shallow.iter().map(|p| p.max(PROB_FLOOR).ln()).collect();
        let mean = z.iter().sum::<f64>() / k as f64;
        let z: Vec<f64> = z.iter().map(|v| v - mean).collect();
        let planner = LatentMctsPlanner {
            num_simulations: (8 * k).max(self.planner.num_simulations),
            c_puct: self.planner.c_puct,
            temperature: 1.0,
            max_depth: 1,
            damping: self.planner.damping,
        };
        let plan = planner.plan(&z, &simplex_etf(k)?)?;
        let mut order: Vec<usize> = (0..k).collect();
        // Python `np.lexsort((priors, q, visits))[::-1]`: visits, then root
        // value, then prior, all descending; full ties put the later index first.
        order.sort_by(|&a, &b| {
            plan.root_visits[b]
                .cmp(&plan.root_visits[a])
                .then(plan.root_q[b].total_cmp(&plan.root_q[a]))
                .then(plan.root_priors[b].total_cmp(&plan.root_priors[a]))
                .then(b.cmp(&a))
        });
        let survivors: Vec<usize> = order[..top_m].to_vec();
        let mut deep = BTreeMap::new();
        let mut deep_layers_executed = 0;
        let mut best: Option<(usize, f64)> = None;
        for &i in &survivors {
            let mut state = None;
            let out =
                model.forward_with_hook(std::slice::from_ref(&candidates[i]), &mut |obs| {
                    if obs.layer == last {
                        state = Some((
                            residual_last(obs.residual)?,
                            to_f64(model.final_norm_last(obs.residual)?),
                        ));
                    }
                    Ok(HookAction::Continue)
                })?;
            let (raw, normed) = state.ok_or_else(|| {
                ModelError::Inference("deep pass never reached the last layer".into())
            })?;
            if out.exited_at.is_some() || out.layers_executed != self.num_layers {
                return Err(ModelError::Inference("deep pass exited early".into()));
            }
            deep_layers_executed += out.layers_executed;
            let value = self.layer_probs(last, &raw, &normed)?[target_label];
            deep.insert(i, value);
            if best.is_none_or(|(_, v)| value > v) {
                best = Some((i, value));
            }
        }
        let (best, _) = best.ok_or_else(|| ModelError::Inference("no survivor".into()))?;
        Ok(PrunedCandidateResult {
            best,
            survivors,
            shallow_values: shallow,
            deep_values: deep,
            layers_executed: shallow_layers_executed + deep_layers_executed,
            shallow_layers_executed,
            deep_layers_executed,
            full_layers: k * self.num_layers,
            mcts_visits: plan.root_visits.clone(),
        })
    }
}

/// BoolQ verbalizer strings; each must encode to one token.
pub const BOOLQ_YES_STRINGS: [&str; 4] = ["yes", "Yes", " yes", " Yes"];
pub const BOOLQ_NO_STRINGS: [&str; 4] = ["no", "No", " no", " No"];
const BOOLQ_PROMPT_SUFFIX: &str = "\nAnswer with yes or no based strictly on the passage.";
/// The system turn the Qwen2.5-Instruct chat template inserts when the
/// conversation has none.
const QWEN25_DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const QWEN15B_LAYERS: usize = 28;
const QWEN15B_HIDDEN: usize = 1536;
const QWEN15B_INTERMEDIATE: usize = 8960;

/// One user turn through the Qwen2.5-Instruct chat template with
/// `add_generation_prompt=True`, as `tokenizer.apply_chat_template` renders
/// it (the default system turn included).
pub fn qwen25_chat_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\n{QWEN25_DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Settings of [`run_boolq_benchmark`]; the CLI defaults are those of
/// `benchmarks/suites/benchmark_layerwise_mcts_qwen15b.py`.
#[derive(Clone, Debug)]
pub struct BoolqBenchOptions {
    pub model_path: PathBuf,
    /// Default: `tokenizer.json` beside the weights.
    pub tokenizer_path: Option<PathBuf>,
    pub data: PathBuf,
    pub n: usize,
    pub folds: usize,
    /// Below 1, the sparse arm keeps this fraction of the FFN channels of
    /// `ffn_layers`, chosen by activation importance on each fold's
    /// training rows only.
    pub ffn_keep_ratio: f64,
    /// Default [`crate::ffn_prune::default_upper_ffn_layers`] (22..28).
    pub ffn_layers: Option<Vec<usize>>,
    pub hook_layers: Option<Vec<usize>>,
    pub min_exit_layer: usize,
    pub tau: f64,
    pub gamma: f64,
    pub mcts_margin_min: f64,
    pub top_k_hooks: usize,
    /// Test rows per [`InterLayerMctsRunner::score_candidates_mcts_pruned`]
    /// group; 0 skips the candidate-pruning section.
    pub prune_group_size: usize,
    pub prune_shallow_layers: usize,
    pub prune_top_m: usize,
    /// FIFO size of each hook layer's episodic memory; may be below the
    /// training rows per fold (see
    /// [`EpisodicManifoldMemory::loo_query_for_all_anchors`]).
    pub memory_capacity: usize,
}

struct BoolqRow {
    id: String,
    label: usize,
    context: String,
}

/// First `n / 2` "yes" and `n / 2` "no" rows with distinct questions, in
/// file order (the Python `_rows`).
fn boolq_rows(path: &Path, n: usize) -> Result<Vec<BoolqRow>, ModelError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ModelError::QwenLoad(format!("read {}: {e}", path.display())))?;
    let mut want = [n / 2, n - n / 2];
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let row: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            ModelError::QwenInput(format!("{}:{}: {e}", path.display(), line_no + 1))
        })?;
        let field = |v: &serde_json::Value, what: &str| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                ModelError::QwenInput(format!(
                    "{}:{}: missing string {what}",
                    path.display(),
                    line_no + 1
                ))
            })
        };
        let label = match field(&row["ground_truth"], "ground_truth")?.as_str() {
            "yes" => 0,
            "no" => 1,
            other => {
                return Err(ModelError::QwenInput(format!(
                    "unsupported BoolQ label in {}: {other:?}",
                    path.display()
                )))
            }
        };
        let context = field(&row["context"], "context")?;
        let id = field(&row["id"], "id")?;
        if !context.starts_with("Passage: ") || !context.ends_with(BOOLQ_PROMPT_SUFFIX) {
            return Err(ModelError::QwenInput(format!(
                "unexpected BoolQ prompt structure in {}: {id}",
                path.display()
            )));
        }
        let question = field(&row["metadata"]["question"], "metadata.question")?
            .trim()
            .to_lowercase()
            .trim_end_matches('?')
            .to_owned();
        if !seen.contains(&question) && want[label] > 0 {
            want[label] -= 1;
            seen.insert(question);
            rows.push(BoolqRow { id, label, context });
        }
    }
    if want.iter().any(|&w| w > 0) {
        return Err(ModelError::QwenInput(format!(
            "BoolQ source lacks enough distinct balanced rows: missing yes {} no {}",
            want[0], want[1]
        )));
    }
    Ok(rows)
}

fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, ModelError> {
    tok.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| ModelError::QwenInput(format!("tokenize: {e}")))
}

/// One evaluated row of one arm.
#[derive(Clone, Debug)]
struct RowOutcome {
    label: usize,
    layers: usize,
    exit_layer: Option<usize>,
    ffn: FfnWork,
    tokens: usize,
    e2e_s: f64,
    infer_s: f64,
}

fn mcnemar_exact(wins: usize, losses: usize) -> f64 {
    let d = wins + losses;
    if d == 0 {
        return 1.0;
    }
    // Exact two-sided binomial tail at p = 1/2, in log space for large d.
    let ln_choose = |n: usize, k: usize| -> f64 {
        (1..=k)
            .map(|i| ((n - k + i) as f64).ln() - (i as f64).ln())
            .sum()
    };
    let tail: f64 = (0..=wins.min(losses))
        .map(|k| (ln_choose(d, k) - d as f64 * std::f64::consts::LN_2).exp())
        .sum();
    (2.0 * tail).min(1.0)
}

fn arm_summary(
    outcomes: &[RowOutcome],
    labels: &[usize],
    refs: &[(&str, &[RowOutcome])],
) -> Result<serde_json::Value, ModelError> {
    let n = outcomes.len();
    if n == 0 || n != labels.len() {
        return Err(ModelError::Inference("arm evaluated zero rows".into()));
    }
    let correct = outcomes
        .iter()
        .zip(labels)
        .filter(|(o, y)| o.label == **y)
        .count();
    let mean = |f: &dyn Fn(&RowOutcome) -> f64| outcomes.iter().map(f).sum::<f64>() / n as f64;
    let median = |f: &dyn Fn(&RowOutcome) -> f64| {
        let mut v: Vec<f64> = outcomes.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    };
    let mean_layers = mean(&|o| o.layers as f64);
    let mean_channels = mean(&|o| o.ffn.ffn_channels_executed as f64);
    let mean_macs_per_token = mean(&|o| o.ffn.ffn_macs_executed as f64 / o.tokens as f64);
    let dense_macs = (3 * QWEN15B_HIDDEN * QWEN15B_INTERMEDIATE * QWEN15B_LAYERS) as f64;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for o in outcomes {
        let key = o.exit_layer.map_or("full_28".to_owned(), |l| l.to_string());
        *hist.entry(key).or_default() += 1;
    }
    let mut out = serde_json::json!({
        "correct": correct,
        "n": n,
        "accuracy": correct as f64 / n as f64,
        "mean_layers_executed": mean_layers,
        "layer_reduction_vs_28": 1.0 - mean_layers / QWEN15B_LAYERS as f64,
        "early_exits": outcomes.iter().filter(|o| o.exit_layer.is_some()).count(),
        "exit_layer_histogram": hist,
        "mean_ffn_channels_executed_per_token": mean_channels,
        "mean_ffn_macs_executed_per_token": mean_macs_per_token,
        "ffn_channel_reduction_vs_dense28":
            1.0 - mean_channels / (QWEN15B_LAYERS * QWEN15B_INTERMEDIATE) as f64,
        "ffn_mac_reduction_vs_dense28": 1.0 - mean_macs_per_token / dense_macs,
        "e2e_seconds_total": outcomes.iter().map(|o| o.e2e_s).sum::<f64>(),
        "e2e_ms_per_row_mean": mean(&|o| o.e2e_s) * 1e3,
        "e2e_ms_per_row_median": median(&|o| o.e2e_s) * 1e3,
        "inference_seconds_total": outcomes.iter().map(|o| o.infer_s).sum::<f64>(),
        "inference_ms_per_row_mean": mean(&|o| o.infer_s) * 1e3,
        "inference_ms_per_row_median": median(&|o| o.infer_s) * 1e3,
        "predictions": outcomes.iter().map(|o| o.label).collect::<Vec<_>>(),
        "layers_executed": outcomes.iter().map(|o| o.layers).collect::<Vec<_>>(),
    });
    for (name, base) in refs {
        let wins = (0..n)
            .filter(|&i| outcomes[i].label == labels[i] && base[i].label != labels[i])
            .count();
        let losses = (0..n)
            .filter(|&i| outcomes[i].label != labels[i] && base[i].label == labels[i])
            .count();
        let agreement = (0..n)
            .filter(|&i| outcomes[i].label == base[i].label)
            .count();
        out[format!("vs_{name}")] = serde_json::json!({
            "agreement": agreement, "wins": wins, "losses": losses,
            "mcnemar_exact_p": mcnemar_exact(wins, losses),
        });
    }
    Ok(out)
}

/// Column L2 norms of a layer's `down [H, I]`, the `|down_c|` factor of the
/// activation importance.
fn down_column_norms(model: &QwenModel, layer: usize) -> Result<Vec<f64>, ModelError> {
    let (_, _, down) = model.ffn_matrices(layer)?;
    down.sqr()
        .and_then(|s| s.sum(0))
        .and_then(|s| s.to_vec1::<f32>())
        .map(|v| v.into_iter().map(|x| (x as f64).sqrt()).collect())
        .map_err(candle_err("down column norms"))
}

/// Meaning of the per-arm speed ratio, written into the BoolQ report.
pub const TIMING_BOUNDARY_NOTE: &str = "e2e_speedup_vs_full_28_uncalibrated is a compatibility name only; it equals capture_forward_vs_predict_timed_ratio = (sum of dense 28-layer capture forward seconds, no tokenizer) / (sum of the arm's per-row predict_timed seconds, chat template + tokenizer + runner.predict). It excludes model load, the per-fold fit and the sparse arm's pruned reload, so it is a local per-row timing ratio, not a full end-to-end speed-up.";

/// Meaning of the component ablation arms, written into the BoolQ report.
pub const ABLATION_NOTE: &str = "multihook_weighted_full28_no_early_exit uses fitted MCTS weights at full depth; multihook_no_mcts_margin_early_exit uses softmax base utility weights and the identical early-exit rule; multihook_equal_weight_full28 uses equal hook weights at full depth. The full-depth arms reuse the dense capture pass.";

fn predict_timed(
    runner: &InterLayerMctsRunner,
    model: &QwenModel,
    tok: &tokenizers::Tokenizer,
    context: &str,
) -> Result<RowOutcome, ModelError> {
    let t0 = Instant::now();
    let ids = encode(tok, &qwen25_chat_prompt(context))?;
    let t1 = Instant::now();
    let d = runner.predict(model, &ids)?;
    let t2 = Instant::now();
    check_finite("predict probabilities", &d.probs)?;
    Ok(RowOutcome {
        label: d.label,
        layers: d.layers_executed,
        exit_layer: d.early_exit.then_some(d.exit_layer),
        ffn: d.ffn,
        tokens: ids.len(),
        e2e_s: (t2 - t0).as_secs_f64(),
        infer_s: (t2 - t1).as_secs_f64(),
    })
}

fn check_qwen15b(model: &QwenModel) -> Result<(), ModelError> {
    let cfg = model.config();
    let width = model.ffn_dense_width(0)?;
    if (cfg.num_layers, cfg.hidden_size, width)
        != (QWEN15B_LAYERS, QWEN15B_HIDDEN, QWEN15B_INTERMEDIATE)
    {
        return Err(ModelError::QwenLoad(format!(
            "expected 28-layer Qwen2.5-1.5B, got {} layers, hidden {}, intermediate {width}",
            cfg.num_layers, cfg.hidden_size
        )));
    }
    Ok(())
}

/// Cross-fitted BoolQ evaluation of the calibrated MCTS early exit on the
/// 28-layer Qwen2.5-1.5B, the Rust counterpart of
/// `benchmarks/suites/benchmark_layerwise_mcts_qwen15b.py`.
///
/// Rows are split into `folds` stratified folds (row `j` of each label goes
/// to fold `j % folds`). Every row runs one full capture pass on the dense
/// model; each fold fits [`InterLayerMctsRunner::fit_calibrated`] on its
/// training rows only and runs the real early-exit
/// [`InterLayerMctsRunner::predict`] on its test rows. The sparse arm
/// reloads the model per fold and slices the FFN channels chosen by
/// activation importance summed over that fold's training rows. The
/// `full_28_*` arms read the last layer of the dense capture pass, which is
/// the same 28-layer forward a never-exiting `predict` would run.
///
/// `e2e_speedup_vs_full_28_uncalibrated` keeps its old name for
/// compatibility but is not an end-to-end speed-up: it is
/// `capture_forward_vs_predict_timed_ratio`, the dense 28-layer capture
/// forward time over the arm's per-row `predict_timed` time (template,
/// tokenizer and forward). Neither side counts model load, fold fitting or
/// the sparse arm's reload.
pub fn run_boolq_benchmark(opts: &BoolqBenchOptions) -> Result<serde_json::Value, ModelError> {
    let wall = Instant::now();
    if opts.n < 8 || opts.n % 2 == 1 || opts.folds < 2 || opts.n / 2 < opts.folds {
        return Err(ModelError::QwenInput(
            "--n must be even, >= 8, and give every fold both labels".into(),
        ));
    }
    if !(opts.ffn_keep_ratio > 0.0 && opts.ffn_keep_ratio <= 1.0) {
        return Err(ModelError::QwenInput(
            "--ffn-keep-ratio must be in (0, 1]".into(),
        ));
    }
    // Every config check that needs no model runs before the load and the
    // dense capture, so a bad setting cannot cost a full pass first.
    let probe_config = CalibratedConfig {
        tau: opts.tau,
        gamma: opts.gamma,
        mcts_margin_min: opts.mcts_margin_min,
        memory_capacity: opts.memory_capacity,
        ..Default::default()
    };
    probe_config.validate()?;
    let tokenizer_path = match &opts.tokenizer_path {
        Some(p) => p.clone(),
        None if opts.model_path.is_dir() => opts.model_path.join("tokenizer.json"),
        None => opts
            .model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json"),
    };
    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        ModelError::QwenLoad(format!("tokenizer {}: {e}", tokenizer_path.display()))
    })?;
    let t0 = Instant::now();
    let dense = QwenModel::load(&opts.model_path)?;
    let load_seconds = t0.elapsed().as_secs_f64();
    check_qwen15b(&dense)?;
    let weights = QwenModel::weights_path(&opts.model_path)?;
    let weights_sha256 = crate::qwen::sha256_file(&weights)?;
    let data_sha256 = crate::qwen::sha256_file(&opts.data)?;
    let num_layers = dense.config().num_layers;
    let last = num_layers - 1;

    let rows = boolq_rows(&opts.data, opts.n)?;
    let labels: Vec<usize> = rows.iter().map(|r| r.label).collect();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for strings in [BOOLQ_YES_STRINGS, BOOLQ_NO_STRINGS] {
        let mut group = Vec::new();
        for s in strings {
            let ids = encode(&tok, s)?;
            if ids.len() != 1 {
                return Err(ModelError::QwenInput(format!(
                    "verbalizer {s:?} is not a single token: {ids:?}"
                )));
            }
            group.push(ids[0]);
        }
        groups.push(group);
    }
    let t0 = Instant::now();
    let ids: Vec<Vec<u32>> = rows
        .iter()
        .map(|r| encode(&tok, &qwen25_chat_prompt(&r.context)))
        .collect::<Result<_, _>>()?;
    let tokenize_all_seconds = t0.elapsed().as_secs_f64();
    let mut fold_of = vec![0; rows.len()];
    for class in 0..2 {
        for (j, i) in (0..rows.len()).filter(|&i| labels[i] == class).enumerate() {
            fold_of[i] = j % opts.folds;
        }
    }
    let hook_layers = opts
        .hook_layers
        .clone()
        .unwrap_or_else(|| default_hook_layers(num_layers));
    let config = CalibratedConfig {
        hook_layers: Some(hook_layers.clone()),
        tau: opts.tau,
        gamma: opts.gamma,
        min_exit_layer: Some(opts.min_exit_layer),
        mcts_margin_min: opts.mcts_margin_min,
        top_k_hooks: opts.top_k_hooks,
        memory_capacity: opts.memory_capacity,
        ..Default::default()
    };
    if opts.prune_group_size > 0 && !hook_layers.contains(&(opts.prune_shallow_layers.max(1) - 1)) {
        return Err(ModelError::QwenInput(format!(
            "--prune-shallow-layers {} must cut at a hook layer of {hook_layers:?}",
            opts.prune_shallow_layers
        )));
    }

    // One full dense pass per row: the hook states every fold fits from.
    let t0 = Instant::now();
    let mut dense_states = Vec::with_capacity(rows.len());
    let mut capture_s = Vec::with_capacity(rows.len());
    for row_ids in &ids {
        let t = Instant::now();
        dense_states.push(InterLayerMctsRunner::capture_hook_states(
            &dense,
            row_ids,
            &hook_layers,
        )?);
        capture_s.push(t.elapsed().as_secs_f64());
    }
    let capture_seconds = t0.elapsed().as_secs_f64();
    eprintln!(
        "bench-boolq: dense capture of {} rows in {capture_seconds:.1}s",
        rows.len()
    );

    let sparse = opts.ffn_keep_ratio < 1.0;
    let ffn_layers: Vec<usize> = opts
        .ffn_layers
        .clone()
        .unwrap_or_else(|| crate::ffn_prune::default_upper_ffn_layers(num_layers).collect());
    let mut row_activation: Vec<BTreeMap<usize, (Vec<f64>, usize)>> = Vec::new();
    let mut down_norms = BTreeMap::new();
    let t0 = Instant::now();
    if sparse {
        let set: BTreeSet<usize> = ffn_layers.iter().copied().collect();
        if set.is_empty() || set.iter().any(|&l| l >= num_layers) {
            return Err(ModelError::QwenInput(format!(
                "FFN layers {ffn_layers:?} must index the {num_layers} layers"
            )));
        }
        for row_ids in &ids {
            let sums = dense.ffn_activation_sums(std::slice::from_ref(row_ids), &set)?;
            row_activation.push(
                sums.into_iter()
                    .map(|(l, acc)| (l, (acc.sums, acc.rows)))
                    .collect(),
            );
        }
        for &l in &set {
            down_norms.insert(l, down_column_norms(&dense, l)?);
        }
    }
    let activation_seconds = t0.elapsed().as_secs_f64();

    let uncal = |i: usize| -> Result<RowOutcome, ModelError> {
        let (_, normed) = dense_states[i].layer(last)?;
        let z: Vec<f64> = dense
            .lm_head_rows(&groups.concat())?
            .into_iter()
            .map(|r| dot(&to_f64(r), normed))
            .collect();
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        Ok(RowOutcome {
            label: argmax(&VerbalizerLogitStats::group_logits(&z, &sizes)?),
            layers: num_layers,
            exit_layer: None,
            ffn: dense_states[i].ffn,
            tokens: dense_states[i].tokens,
            e2e_s: capture_s[i],
            infer_s: capture_s[i],
        })
    };
    let mut arms: BTreeMap<&str, BTreeMap<usize, RowOutcome>> = BTreeMap::new();
    let mut fold_reports = Vec::new();
    let mut pruning_groups = Vec::new();
    let mut fit_seconds = 0.0;
    for fold in 0..opts.folds {
        let train: Vec<usize> = (0..rows.len()).filter(|&i| fold_of[i] != fold).collect();
        let test: Vec<usize> = (0..rows.len()).filter(|&i| fold_of[i] == fold).collect();
        if train.len() + test.len() != rows.len() || test.is_empty() {
            return Err(ModelError::Inference(
                "cross-fit partition is invalid".into(),
            ));
        }
        let train_y: Vec<usize> = train.iter().map(|&i| labels[i]).collect();
        let train_states: Vec<HookStates> =
            train.iter().map(|&i| dense_states[i].clone()).collect();
        let t = Instant::now();
        let (runner, fit) = InterLayerMctsRunner::fit_calibrated(
            &dense,
            &train_states,
            &train_y,
            &groups,
            &config,
        )?;
        let mut no_mcts_runner = runner.clone();
        let peak = fit
            .layers
            .iter()
            .map(|r| r.utility)
            .fold(f64::NEG_INFINITY, f64::max);
        let scores: Vec<f64> = fit
            .layers
            .iter()
            .map(|r| (config.reward_scale * (r.utility - peak)).exp())
            .collect();
        let total: f64 = scores.iter().sum();
        no_mcts_runner.layer_weights = fit
            .layers
            .iter()
            .zip(scores)
            .map(|(r, s)| (r.layer, s / total))
            .collect();
        let dense_fit_s = t.elapsed().as_secs_f64();
        fit_seconds += dense_fit_s;
        let mut fold_report = serde_json::json!({
            "fold": fold,
            "train": train.len(),
            "test": test.len(),
            "test_label_counts": [
                test.iter().filter(|&&i| labels[i] == 0).count(),
                test.iter().filter(|&&i| labels[i] == 1).count(),
            ],
            "mcts_multihook_dense_ffn": {"fit_seconds": dense_fit_s, "fit": fit},
        });
        for &i in &test {
            arms.entry("full_28_uncalibrated")
                .or_default()
                .insert(i, uncal(i)?);
            let (raw, normed) = dense_states[i].layer(last)?;
            let p = runner.layer_probs(last, raw, normed)?;
            arms.entry("full_28_calibrated").or_default().insert(
                i,
                RowOutcome {
                    label: argmax(&p),
                    ..uncal(i)?
                },
            );
            // Ablations on the same capture pass: the fitted weighted
            // runner forced to full depth, and equal hook weights (no MCTS).
            for (arm, equal_weight) in [
                ("multihook_weighted_full28_no_early_exit", false),
                ("multihook_equal_weight_full28", true),
            ] {
                let p = runner.full_depth_readout(&dense_states[i], equal_weight)?;
                arms.entry(arm).or_default().insert(
                    i,
                    RowOutcome {
                        label: argmax(&p),
                        ..uncal(i)?
                    },
                );
            }
            let o = predict_timed(&runner, &dense, &tok, &rows[i].context)?;
            arms.entry("mcts_multihook_dense_ffn")
                .or_default()
                .insert(i, o);
            let o = predict_timed(&no_mcts_runner, &dense, &tok, &rows[i].context)?;
            arms.entry("multihook_no_mcts_margin_early_exit")
                .or_default()
                .insert(i, o);
        }
        if opts.prune_group_size > 0 {
            for chunk in test.chunks(opts.prune_group_size) {
                if chunk.len() < 2 {
                    continue;
                }
                let cands: Vec<Vec<u32>> = chunk.iter().map(|&i| ids[i].clone()).collect();
                let t = Instant::now();
                let res = runner.score_candidates_mcts_pruned(
                    &dense,
                    &cands,
                    0,
                    opts.prune_shallow_layers,
                    opts.prune_top_m.min(chunk.len()),
                )?;
                let seconds = t.elapsed().as_secs_f64();
                // Reference: every candidate's full 28-layer calibrated yes
                // probability from its capture pass.
                let full: Vec<f64> = chunk
                    .iter()
                    .map(|&i| {
                        let (raw, normed) = dense_states[i].layer(last)?;
                        Ok(runner.layer_probs(last, raw, normed)?[0])
                    })
                    .collect::<Result<_, ModelError>>()?;
                let full_best = argmax(&full);
                pruning_groups.push(serde_json::json!({
                    "fold": fold,
                    "row_ids": chunk.iter().map(|&i| rows[i].id.clone()).collect::<Vec<_>>(),
                    "row_labels": chunk.iter().map(|&i| labels[i]).collect::<Vec<_>>(),
                    "pruned_best": res.best,
                    "pruned_best_is_yes": labels[chunk[res.best]] == 0,
                    "full_best": full_best,
                    "full_best_is_yes": labels[chunk[full_best]] == 0,
                    "agrees_with_full_ranking": res.best == full_best,
                    "full_yes_probs": full,
                    "result": res,
                    "seconds": seconds,
                }));
            }
        }
        if sparse {
            let mut importance: FfnImportance = BTreeMap::new();
            for (&layer, norms) in &down_norms {
                let mut sums = vec![0.0; norms.len()];
                let mut count = 0usize;
                for &i in &train {
                    let (s, r) = &row_activation[i][&layer];
                    for (a, b) in sums.iter_mut().zip(s) {
                        *a += b;
                    }
                    count += r;
                }
                if count == 0 {
                    return Err(ModelError::Inference(format!(
                        "no training activations at FFN layer {layer}"
                    )));
                }
                let imp: Vec<f64> = sums
                    .iter()
                    .zip(norms)
                    .map(|(s, n)| s / count as f64 * n)
                    .collect();
                check_finite("FFN channel importance", &imp)?;
                importance.insert(layer, imp);
            }
            let masks: FfnChannelMasks = select_top_ffn_channels(&importance, opts.ffn_keep_ratio)?;
            let t = Instant::now();
            let mut sparse_model = QwenModel::load(&opts.model_path)?;
            let prune = sparse_model.prune_ffn_channels(&masks)?;
            let mut sparse_train = Vec::with_capacity(train.len());
            for &i in &train {
                sparse_train.push(InterLayerMctsRunner::capture_hook_states(
                    &sparse_model,
                    &ids[i],
                    &hook_layers,
                )?);
            }
            let (sparse_runner, sparse_fit) = InterLayerMctsRunner::fit_calibrated(
                &sparse_model,
                &sparse_train,
                &train_y,
                &groups,
                &config,
            )?;
            let sparse_fit_s = t.elapsed().as_secs_f64();
            fit_seconds += sparse_fit_s;
            for &i in &test {
                let o = predict_timed(&sparse_runner, &sparse_model, &tok, &rows[i].context)?;
                arms.entry("mcts_multihook_sparse_ffn")
                    .or_default()
                    .insert(i, o);
            }
            fold_report["mcts_multihook_sparse_ffn"] = serde_json::json!({
                "fit_seconds_incl_reload_prune_capture": sparse_fit_s,
                "kept_channels": masks.iter().map(|(l, m)| (l.to_string(), m.len())).collect::<BTreeMap<_, _>>(),
                "prune": prune,
                "fit": sparse_fit,
            });
        }
        eprintln!(
            "bench-boolq: fold {fold} done at {:.1}s",
            wall.elapsed().as_secs_f64()
        );
        fold_reports.push(fold_report);
    }

    let evaluated: Vec<usize> = (0..rows.len()).collect();
    let y: Vec<usize> = evaluated.iter().map(|&i| labels[i]).collect();
    let ordered: BTreeMap<&str, Vec<RowOutcome>> = arms
        .iter()
        .map(|(name, m)| (*name, evaluated.iter().map(|&i| m[&i].clone()).collect()))
        .collect();
    let ref_names = [
        "full_28_uncalibrated",
        "full_28_calibrated",
        "mcts_multihook_dense_ffn",
        "multihook_weighted_full28_no_early_exit",
        "multihook_no_mcts_margin_early_exit",
    ];
    let mut arm_json = serde_json::Map::new();
    for (name, outcomes) in &ordered {
        let refs: Vec<(&str, &[RowOutcome])> = ref_names
            .iter()
            .filter(|r| *r != name)
            .map(|r| (*r, ordered[r].as_slice()))
            .collect();
        arm_json.insert(name.to_string(), arm_summary(outcomes, &y, &refs)?);
    }
    let headline = if sparse {
        "mcts_multihook_sparse_ffn"
    } else {
        "mcts_multihook_dense_ffn"
    };
    let base_e2e = arm_json["full_28_uncalibrated"]["e2e_seconds_total"]
        .as_f64()
        .unwrap_or(f64::NAN);
    for arm in arm_json.values_mut() {
        let total = arm["e2e_seconds_total"].as_f64().unwrap_or(f64::NAN);
        arm["e2e_speedup_vs_full_28_uncalibrated"] = serde_json::json!(base_e2e / total);
        arm["capture_forward_vs_predict_timed_ratio"] = serde_json::json!(base_e2e / total);
    }
    let pruned_layers: usize = pruning_groups
        .iter()
        .map(|g| g["result"]["layers_executed"].as_u64().unwrap_or(0) as usize)
        .sum();
    let full_layers: usize = pruning_groups
        .iter()
        .map(|g| g["result"]["full_layers"].as_u64().unwrap_or(0) as usize)
        .sum();
    let count = |key: &str| pruning_groups.iter().filter(|g| g[key] == true).count();
    let majority = labels
        .iter()
        .filter(|&&l| l == 0)
        .count()
        .max(labels.iter().filter(|&&l| l == 1).count());
    let prompt_tokens: Vec<usize> = ids.iter().map(Vec::len).collect();
    let setup = serde_json::json!({
        "suite": "gen-zero qwen bench-boolq (crates/gen-zero-model/src/inter_layer_mcts.rs::run_boolq_benchmark)",
        "reference_suite": "benchmarks/suites/benchmark_layerwise_mcts_qwen15b.py",
        "model": opts.model_path.display().to_string(),
        "weights": weights.display().to_string(),
        "weights_sha256": weights_sha256,
        "tokenizer": tokenizer_path.display().to_string(),
        "dtype": "float32",
        "device": "cpu",
        "host": std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_owned()).unwrap_or_default(),
        "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "data": opts.data.display().to_string(),
        "data_sha256": data_sha256,
        "n": rows.len(),
        "folds": opts.folds,
        "fold_rule": "stratified: the j-th row of each label (file order) goes to fold j % folds",
        "fold_label_counts": (0..opts.folds).map(|f| [0, 1].map(|c| (0..rows.len()).filter(|&i| fold_of[i] == f && labels[i] == c).count())).collect::<Vec<_>>(),
        "majority_baseline": majority as f64 / labels.len() as f64,
        "prompt": "<|im_start|>system\\n<default Qwen2.5 system>\\n<|im_end|>\\n<|im_start|>user\\n{context}<|im_end|>\\n<|im_start|>assistant\\n, add_special_tokens=false",
        "prompt_tokens_mean": prompt_tokens.iter().sum::<usize>() as f64 / prompt_tokens.len() as f64,
        "prompt_tokens": prompt_tokens,
        "verbalizer_strings": [BOOLQ_YES_STRINGS, BOOLQ_NO_STRINGS],
        "verbalizer_token_ids": groups,
        "labels": {"yes": 0, "no": 1},
        "total_layers": num_layers,
        "hook_layers": hook_layers,
        "min_exit_layer": opts.min_exit_layer,
        "tau": opts.tau,
        "gamma": opts.gamma,
        "mcts_margin_min": opts.mcts_margin_min,
        "top_k_hooks": opts.top_k_hooks,
        "memory_capacity": opts.memory_capacity,
        "ffn_keep_ratio": opts.ffn_keep_ratio,
        "ffn_layers": ffn_layers,
        "ffn_importance": "activation, summed over each fold's training rows on the dense model",
    });
    let run = serde_json::json!({
        "headline_arm": headline,
        "headline": arm_json[headline].clone(),
        "timing_boundary": "mcts arms: e2e = chat template + tokenizer + runner.predict, inference = runner.predict; full_28_* arms: the dense capture forward (no tokenizer); model load, capture, activation scoring and fitting are reported separately",
        "timing_boundary_note": TIMING_BOUNDARY_NOTE,
        "ablation_note": ABLATION_NOTE,
        "model_load_seconds": load_seconds,
        "tokenize_all_rows_seconds": tokenize_all_seconds,
        "dense_capture_seconds": capture_seconds,
        "activation_scoring_seconds": activation_seconds,
        "fit_seconds_total": fit_seconds,
        "fold_fits": fold_reports,
        "row_ids": rows.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
        "row_labels": y,
        "arms": arm_json,
        "candidate_pruning": {
            "target_label": "yes",
            "group_size": opts.prune_group_size,
            "shallow_layers": opts.prune_shallow_layers,
            "top_m": opts.prune_top_m,
            "model": "dense",
            "groups": pruning_groups.len(),
            "pruned_best_is_yes": count("pruned_best_is_yes"),
            "full_best_is_yes": count("full_best_is_yes"),
            "agrees_with_full_ranking": count("agrees_with_full_ranking"),
            "layers_executed": pruned_layers,
            "full_layers": full_layers,
            "note": "survivor deep passes restart at layer 0 (no resume entry in QwenModel), so layers_executed counts their shallow layers twice",
            "group_details": pruning_groups,
        },
    });
    let mut report = setup;
    for (key, value) in run.as_object().into_iter().flatten() {
        report[key] = value.clone();
    }
    report["total_wall_seconds"] = serde_json::json!(wall.elapsed().as_secs_f64());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn_prune::default_upper_ffn_layers;
    use crate::qwen::tests::tiny_model;

    #[test]
    fn hook_layers_for_qwen_1_5b_and_other_depths() {
        assert_eq!(default_hook_layers(28), vec![12, 16, 20, 22, 24, 26, 27]);
        let l24 = default_hook_layers(24);
        assert_eq!(l24.last(), Some(&23));
        assert!(l24.windows(2).all(|w| w[0] < w[1]), "{l24:?}");
        for n in 1..40 {
            let l = default_hook_layers(n);
            assert_eq!(l.last(), Some(&(n - 1)), "{n}");
            assert!(l.windows(2).all(|w| w[0] < w[1]), "{n}: {l:?}");
        }
        assert!(default_hook_layers(0).is_empty());
    }

    #[test]
    fn spd_inverse_inverts() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let inv = spd_inverse(&a).unwrap();
        for (i, row) in a.iter().enumerate() {
            for (j, _) in inv.iter().enumerate() {
                let v: f64 = (0..3).map(|k| row[k] * inv[k][j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert!(spd_inverse(&[vec![0.0]]).is_err());
        assert!((top_eigenvalue(&[vec![2.0, 1.0], vec![1.0, 2.0]]) - 3.0).abs() < 1e-9);
        assert!((top_eigenvalue(&[vec![9.0, -6.0], vec![-6.0, 4.0]]) - 13.0).abs() < 1e-9);
    }

    /// Exact LOO by refitting without each row, against the closed form.
    #[test]
    fn memory_loo_matches_brute_force_refit() {
        let features: Vec<Vec<f64>> = (0..7)
            .map(|i| {
                let t = i as f64;
                vec![t.sin(), (1.3 * t).cos(), 0.1 * t * t, (0.7 * t).sin() + 0.2]
            })
            .collect();
        let targets: Vec<Vec<f64>> = features
            .iter()
            .enumerate()
            .map(|(i, f)| vec![2.0 * f[0] - f[2] + 0.05 * (i as f64).cos(), f[1] + 0.5])
            .collect();
        let mem = InterLayerMemory::fit(3, &features, &targets).unwrap();
        let lambda_rel = mem.ridge_lambda;
        for i in 0..7 {
            let (fi, ti): (Vec<_>, Vec<_>) = (0..7)
                .filter(|&j| j != i)
                .map(|j| (features[j].clone(), targets[j].clone()))
                .unzip();
            // Refit at the selected absolute lambda: rebuild by hand.
            let n = fi.len();
            let mu: Vec<f64> = (0..4)
                .map(|d| fi.iter().map(|r| r[d]).sum::<f64>() / n as f64)
                .collect();
            let ym: Vec<f64> = (0..2)
                .map(|d| ti.iter().map(|r| r[d]).sum::<f64>() / n as f64)
                .collect();
            let xc: Vec<Vec<f64>> = fi
                .iter()
                .map(|r| r.iter().zip(&mu).map(|(a, b)| a - b).collect())
                .collect();
            let mut g: Vec<Vec<f64>> = xc
                .iter()
                .map(|a| xc.iter().map(|b| dot(a, b)).collect())
                .collect();
            for (j, row) in g.iter_mut().enumerate() {
                row[j] += lambda_rel;
            }
            let inv = spd_inverse(&g).unwrap();
            let x: Vec<f64> = features[i].iter().zip(&mu).map(|(a, b)| a - b).collect();
            let kv: Vec<f64> = xc.iter().map(|a| dot(a, &x)).collect();
            for c in 0..2 {
                let yc: Vec<f64> = ti.iter().map(|r| r[c] - ym[c]).collect();
                let alpha: Vec<f64> = (0..n)
                    .map(|a| (0..n).map(|b| inv[a][b] * yc[b]).sum())
                    .collect();
                let pred = ym[c] + dot(&kv, &alpha);
                assert!(
                    (pred - mem.loo_logits[i][c]).abs() < 1e-8,
                    "row {i} label {c}"
                );
            }
        }
    }

    #[test]
    fn memory_recovers_linear_map_and_its_lipschitz_constant() {
        let features: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                let t = i as f64;
                vec![t.sin(), (0.6 * t).cos(), (0.3 * t).sin() * 2.0]
            })
            .collect();
        // z = W^T f with W columns (3, 0, 0) and (0, 4, 0): ||W||_2 = 4.
        let targets: Vec<Vec<f64>> = features
            .iter()
            .map(|f| vec![3.0 * f[0], 4.0 * f[1]])
            .collect();
        let mem = InterLayerMemory::fit(0, &features, &targets).unwrap();
        assert!(mem.loo_mse < 1e-3, "{}", mem.loo_mse);
        assert!((mem.lipschitz - 4.0).abs() < 0.05, "{}", mem.lipschitz);
        let z = mem.predict(&[0.5, -0.25, 0.1]).unwrap();
        assert!(
            (z[0] - 1.5).abs() < 0.05 && (z[1] + 1.0).abs() < 0.05,
            "{z:?}"
        );
        assert!(mem.predict(&[1.0]).is_err());
        assert!(InterLayerMemory::fit(0, &features[..2], &targets[..2]).is_err());
        let same = vec![vec![1.0, 2.0, 3.0]; 4];
        assert!(InterLayerMemory::fit(0, &same, &targets[..4]).is_err());
    }

    #[test]
    fn layer_weights_favour_high_utility_layers_and_sum_to_one() {
        let utilities = BTreeMap::from([(2, 0.2), (4, 0.9), (6, 0.5), (7, 0.95)]);
        let w = fit_layer_weights(&utilities, 2, 10.0, &LatentMctsPlanner::default()).unwrap();
        assert!((w.values().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(w.values().filter(|v| **v > 0.0).count(), 2);
        assert!(w[&4] > 0.0 && w[&7] > 0.0, "{w:?}");
        assert_eq!(w[&2], 0.0);
        assert!(fit_layer_weights(&utilities, 0, 10.0, &LatentMctsPlanner::default()).is_err());
        assert!(
            fit_layer_weights(&BTreeMap::new(), 2, 10.0, &LatentMctsPlanner::default()).is_err()
        );
    }

    fn anchor_set() -> (Vec<Vec<u32>>, Vec<usize>) {
        let anchors: Vec<Vec<u32>> = (0..10u32)
            .map(|i| vec![1 + i % 7, 3 + (i * 5) % 11, 2 + (i * 3) % 13, 4 + i])
            .collect();
        let labels = (0..10).map(|i| i % 2).collect();
        (anchors, labels)
    }

    #[test]
    fn runner_full_pass_reads_the_model_label_logits() {
        let model = tiny_model(6);
        let (anchors, labels) = anchor_set();
        let config = InterLayerConfig {
            min_exit_layer: Some(5),
            fit_layer_weights: false,
            ..Default::default()
        };
        let (runner, report) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &config).unwrap();
        assert_eq!(report.hook_layers, default_hook_layers(6));
        assert_eq!(report.anchors, 10);
        assert!(runner.layer_weights().is_empty());
        let ids = vec![5, 9, 2, 8];
        let decision = runner.predict(&model, &ids).unwrap();
        // min_exit_layer 5 is the last layer: no exit, a full pass.
        assert!(!decision.early_exit);
        assert_eq!((decision.layers_executed, decision.exit_layer), (6, 5));
        assert_eq!(
            decision.ffn.ffn_channels_executed,
            decision.ffn.ffn_dense_channels
        );
        let final_trace = decision.checkpoints.last().unwrap();
        assert!(final_trace.bound.is_none());
        // The last checkpoint is softmax over the two label log-probabilities.
        let (hidden, _) = model.forward(std::slice::from_ref(&ids), None).unwrap();
        let last = hidden.narrow(1, 3, 1).unwrap().squeeze(1).unwrap();
        let lp = model.log_probs_of(&last, &[vec![7, 19]]).unwrap();
        for (a, b) in final_trace.probs.iter().zip(&softmax(&lp[0])) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        let bad = InterLayerConfig {
            hook_layers: Some(vec![1, 3]),
            ..Default::default()
        };
        assert!(InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &bad).is_err());
        assert!(
            InterLayerMctsRunner::fit(&model, &anchors[..2], &labels[..2], &[7, 19], &config)
                .is_err()
        );
        assert!(InterLayerMctsRunner::fit(&model, &anchors, &[2; 10], &[7, 19], &config).is_err());
    }

    #[test]
    fn runner_fits_sparse_mcts_layer_weights() {
        let model = tiny_model(6);
        let (anchors, labels) = anchor_set();
        let config = InterLayerConfig {
            top_k_hooks: 2,
            ..Default::default()
        };
        let (runner, report) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &config).unwrap();
        let w = runner.layer_weights();
        assert!((w.values().sum::<f64>() - 1.0).abs() < 1e-12);
        // Top-2 by visit share; a kept layer the search never visited keeps
        // weight 0 (as in the Python rule), so 1 or 2 layers carry weight.
        let active = w.values().filter(|v| **v > 0.0).count();
        assert!((1..=2).contains(&active), "{w:?}");
        assert_eq!(w.keys().copied().collect::<Vec<_>>(), report.hook_layers);
        for row in &report.layers {
            assert!(row.utility.is_finite());
            assert_eq!(row.weight, Some(w[&row.layer]));
            assert_eq!(row.ridge_lambda.is_some(), row.layer != 5);
        }
        let d = runner.predict(&model, &[1, 2, 3]).unwrap();
        assert_eq!(&d.active_hook_weights, w);
        assert!((d.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn weighted_full_pass_uses_real_last_layer_logits() {
        let model = tiny_model(6);
        let (anchors, labels) = anchor_set();
        let config = InterLayerConfig {
            min_exit_layer: Some(5),
            top_k_hooks: 2,
            ..Default::default()
        };
        let (runner, _) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &config).unwrap();
        assert!(!runner.layer_weights().is_empty());
        let ids = vec![5, 9, 2, 8];
        let d = runner.predict(&model, &ids).unwrap();
        assert!(!d.early_exit);
        assert_eq!(d.layers_executed, 6);
        let (hidden, _) = model.forward(std::slice::from_ref(&ids), None).unwrap();
        let last = hidden
            .narrow(1, ids.len() - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        let lp = model.log_probs_of(&last, &[vec![7, 19]]).unwrap();
        for (actual, expected) in d.probs.iter().zip(softmax(&lp[0])) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    /// Exits when the heuristic rule holds and never when the bound is huge.
    #[test]
    fn heuristic_exit_only_when_gap_beats_bound() {
        let model = tiny_model(8);
        let (anchors, labels) = anchor_set();
        let base = InterLayerConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            min_exit_layer: Some(2),
            mcts_margin_min: 0.0,
            fit_layer_weights: false,
            ..Default::default()
        };
        let (mut runner, _) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &base).unwrap();
        // Make the corrected readout at layers 4 and 5 a confident constant:
        // zero coefficients give L_f = 0 and logits = (8, 0), so the gap is
        // ~1 > 0 = bound and the label is stable from layer 4 to layer 5.
        for layer in [2, 4, 5] {
            let m = runner.memories.get_mut(&layer).unwrap();
            m.alpha
                .iter_mut()
                .for_each(|r| r.iter_mut().for_each(|v| *v = 0.0));
            m.target_mean = vec![8.0, 0.0];
            m.lipschitz = 0.0;
        }
        let ids = vec![3, 1, 4, 1, 5];
        let d = runner.predict(&model, &ids).unwrap();
        assert!(d.early_exit, "{d:?}");
        assert_eq!(d.exit_layer, 4);
        assert_eq!(d.layers_executed, 5);
        assert_eq!(d.label, 0);
        assert!(d.margin > d.threshold.unwrap());
        assert!(d.mcts_margin.unwrap() > 0.0);
        assert_eq!(
            d.checkpoints.iter().map(|c| c.layer).collect::<Vec<_>>(),
            vec![2, 4]
        );
        // A nonzero readout displacement makes the projected bound large;
        // changing L_f alone cannot do so when W^T df is exactly zero.
        for layer in [2, 4, 5] {
            let m = runner.memories.get_mut(&layer).unwrap();
            m.alpha[0][0] = 1e9;
            m.lipschitz = 1e9;
        }
        let d = runner.predict(&model, &ids).unwrap();
        assert!(!d.early_exit);
        assert_eq!(d.layers_executed, 8);
        assert!(d
            .checkpoints
            .iter()
            .all(|c| c.bound.is_none_or(|b| c.gap <= b)));
        // Below min_exit_layer nothing exits even with L_f = 0.
        for layer in [2, 4, 5] {
            runner.memories.get_mut(&layer).unwrap().lipschitz = 0.0;
        }
        runner.min_exit_layer = 6;
        assert!(!runner.predict(&model, &ids).unwrap().early_exit);
    }

    /// M06 on the heuristic readout: a zero-weight hook gets no step and no
    /// bound, and the next weighted hook measures its step from the last
    /// weighted hook, not from the zero-weight one.
    #[test]
    fn heuristic_exit_measures_steps_between_weighted_hooks() {
        let model = tiny_model(8);
        let (anchors, labels) = anchor_set();
        let base = InterLayerConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            min_exit_layer: Some(2),
            mcts_margin_min: 0.0,
            fit_layer_weights: false,
            ..Default::default()
        };
        let (mut runner, _) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &base).unwrap();
        // Confident constant readout with L_f = 0 (bound 0), but a nonzero
        // coefficient so the projected step depends on its reference state.
        for layer in [2, 4, 5] {
            let m = runner.memories.get_mut(&layer).unwrap();
            m.alpha
                .iter_mut()
                .for_each(|r| r.iter_mut().for_each(|v| *v = 0.0));
            m.alpha[0][0] = 1.0;
            m.target_mean = vec![8.0, 0.0];
            m.lipschitz = 0.0;
        }
        runner.layer_weights = BTreeMap::from([(2, 0.5), (4, 0.0), (5, 0.5), (7, 0.0)]);
        let ids = vec![3, 1, 4, 1, 5];
        let d = runner.predict(&model, &ids).unwrap();
        let read: Vec<usize> = d.checkpoints.iter().map(|c| c.layer).collect();
        assert_eq!(read, vec![2, 4, 5], "{d:?}");
        assert!(d.checkpoints[1].delta.is_none() && d.checkpoints[1].bound.is_none());
        assert!(!d.checkpoints[1].stable);
        assert_eq!(d.exit_layer, 5);
        let states = InterLayerMctsRunner::capture_hook_states(&model, &ids, &[2, 4, 5]).unwrap();
        let m5 = &runner.memories[&5];
        let (f2, f4, f5) = (&states.normed[&2], &states.normed[&4], &states.normed[&5]);
        let want = m5.projected_delta(f5, f2).unwrap();
        let stale = m5.projected_delta(f5, f4).unwrap();
        assert!((want - stale).abs() > 1e-9, "fixture too weak");
        assert!(
            (d.checkpoints[2].delta.unwrap() - want).abs() < 1e-9,
            "{d:?}"
        );
    }

    #[test]
    fn predict_counts_pruned_ffn_work_of_executed_layers() {
        let mut model = tiny_model(8);
        model.prune_upper_ffn_by_weight_norm(0.75).unwrap();
        let (anchors, labels) = anchor_set();
        let config = InterLayerConfig {
            hook_layers: Some(vec![3, 5, 7]),
            fit_layer_weights: false,
            ..Default::default()
        };
        let (runner, _) =
            InterLayerMctsRunner::fit(&model, &anchors, &labels, &[7, 19], &config).unwrap();
        let d = runner.predict(&model, &[4, 4, 2]).unwrap();
        let upper = default_upper_ffn_layers(8);
        let executed = 0..d.layers_executed;
        let want: u64 = executed
            .map(|l| if upper.contains(&l) { 24 } else { 32 })
            .sum();
        assert_eq!(d.ffn.ffn_channels_executed, want);
        assert_eq!(d.ffn.ffn_dense_channels, 32 * d.layers_executed as u64);
    }

    /// Deterministic pseudo-random numbers in [-1, 1).
    fn noise(seed: usize) -> f64 {
        let x = ((seed as f64) * 12.9898 + 78.233).sin() * 43_758.545_3;
        2.0 * (x - x.floor()) - 1.0
    }

    #[test]
    fn verbalizer_groups_and_closed_form_loo_stats() {
        let z = VerbalizerLogitStats::group_logits(&[1.0, 2.0, -1.0, 0.5, 0.5], &[2, 3]).unwrap();
        assert!((z[0] - (1f64.exp() + 2f64.exp()).ln()).abs() < 1e-12);
        assert!((z[1] - ((-1f64).exp() + 2.0 * 0.5f64.exp()).ln()).abs() < 1e-12);
        assert!(VerbalizerLogitStats::group_logits(&[1.0, 2.0], &[1, 2]).is_err());
        assert!(VerbalizerLogitStats::group_logits(&[1.0, f64::NAN], &[1, 1]).is_err());
        assert!(VerbalizerLogitStats::group_logits(&[1.0, 2.0], &[2, 0]).is_err());

        let rows: Vec<Vec<f64>> = (0..7)
            .map(|i| vec![noise(i) * 3.0 + 1.0, noise(i + 50) - 2.0])
            .collect();
        let mu = VerbalizerLogitStats::loo_mean(&rows).unwrap();
        let sd = VerbalizerLogitStats::loo_std(&rows).unwrap();
        for i in 0..7 {
            let others: Vec<Vec<f64>> = (0..7)
                .filter(|&j| j != i)
                .map(|j| rows[j].clone())
                .collect();
            let s = VerbalizerLogitStats::fit(&others).unwrap();
            for c in 0..2 {
                assert!((mu[i][c] - s.mean[c]).abs() < 1e-12, "row {i} label {c}");
                assert!((sd[i][c] - s.std[c]).abs() < 1e-9, "row {i} label {c}");
            }
        }
        let stats = VerbalizerLogitStats::fit(&rows).unwrap();
        let unit: Vec<f64> = (0..2).map(|c| stats.mean[c] + stats.std[c]).collect();
        for v in stats.standardize(&unit).unwrap() {
            assert!((v - 1.0).abs() < 1e-12, "{v}");
        }
        // Zero variance, in the full set or a leave-one-out set, fails closed.
        let flat_column = [vec![0.1, 1.0], vec![0.1, 2.0], vec![0.1, 3.0]];
        assert!(VerbalizerLogitStats::fit(&flat_column).is_err());
        let loo_flat = vec![vec![0.3, 1.0], vec![1.1, 2.0], vec![1.1, 3.0]];
        assert!(VerbalizerLogitStats::fit(&loo_flat).is_ok());
        assert!(VerbalizerLogitStats::loo_std(&loo_flat).is_err());
        assert!(VerbalizerLogitStats::fit(&[vec![1.0, f64::INFINITY], vec![2.0, 1.0]]).is_err());
        assert!(stats.standardize(&[1.0]).is_err());
    }

    /// Red check for the per-layer standardisation: an intermediate layer
    /// read through the final norm and the head carries a constant label
    /// offset that swamps the signal, so the raw argmax is always "yes".
    #[test]
    fn standardization_removes_layer_constant_offset() {
        let n = 80;
        let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let z: Vec<Vec<f64>> = labels
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let s = if y == 0 { 0.5 } else { -0.5 };
                vec![9.0 + s + 0.3 * noise(i), 1.0 - s + 0.3 * noise(i + 1000)]
            })
            .collect();
        let accuracy = |rows: &[Vec<f64>], ys: &[usize]| {
            rows.iter()
                .zip(ys)
                .filter(|(r, y)| argmax(r) == **y)
                .count() as f64
                / ys.len() as f64
        };
        assert_eq!(
            accuracy(&z, &labels),
            0.5,
            "the raw offset must swamp the signal"
        );
        let loo = VerbalizerLogitStats::loo_standardize(&z).unwrap();
        let loo_acc = accuracy(&loo, &labels);
        assert!(
            loo_acc >= 0.95,
            "leave-one-out standardised accuracy {loo_acc}"
        );
        // Stats from the first 60 rows standardise the 20 held-out rows.
        let stats = VerbalizerLogitStats::fit(&z[..60]).unwrap();
        let held: Vec<Vec<f64>> = z[60..]
            .iter()
            .map(|r| stats.standardize(r).unwrap())
            .collect();
        let held_acc = accuracy(&held, &labels[60..]);
        assert!(
            held_acc >= 0.95,
            "held-out standardised accuracy {held_acc}"
        );
    }

    #[test]
    fn episodic_memory_cosine_read_and_leave_one_out() {
        let labels = [0, 0, 1, 1];
        let centre = || Some(vec![10.0, 10.0, 0.0]);
        let mut mem = EpisodicManifoldMemory::new(3, 2, 4, 12.0, centre()).unwrap();
        let states = vec![
            vec![11.0, 10.0, 0.0],
            vec![11.0, 10.1, 0.0],
            vec![10.0, 11.0, 0.0],
            vec![10.1, 11.0, 0.0],
        ];
        mem.write(&states, &labels, &[0, 1, 2, 3]).unwrap();
        let key = mem.encode(&states[0]).unwrap();
        assert!((dot(&key, &key) - 1.0).abs() < 1e-12);
        let q = mem.query(&[12.0, 10.0, 0.0], None).unwrap();
        assert!(q[0] > 0.99, "{q:?}");
        // A retained FIFO always uses its own center, even if none was supplied.
        let mut flat = EpisodicManifoldMemory::new(3, 2, 4, 12.0, None).unwrap();
        flat.write(&states, &labels, &[0, 1, 2, 3]).unwrap();
        let fq = flat.query(&[12.0, 10.0, 0.0], None).unwrap();
        assert!((fq[0] - q[0]).abs() < 1e-12, "{fq:?}");
        let loo = mem.loo_query().unwrap();
        for (i, row) in loo.iter().enumerate() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(row[labels[i]] > 0.9, "entry {i}: {row:?}");
        }
        // The ring evicts the oldest entry; zero keys and an emptied LOO fail.
        mem.write(&[vec![9.0, 10.0, 1.0]], &[1], &[4]).unwrap();
        assert_eq!(mem.len(), 4);
        assert!(mem.encode(mem.center.as_ref().unwrap()).is_err());
        let mut one = EpisodicManifoldMemory::new(3, 2, 4, 12.0, None).unwrap();
        one.write(&[vec![1.0, 0.0, 0.0]], &[0], &[0]).unwrap();
        assert!(one.loo_query().is_err());
        assert!(EpisodicManifoldMemory::new(3, 2, 4, 0.0, None).is_err());
        assert!(mem.write(&[vec![1.0, 2.0, 3.0]], &[2], &[9]).is_err());
    }

    /// The out-of-fold read of entry i must equal a memory built from the
    /// other n - 1 states alone, centred on their own mean.
    #[test]
    fn loo_query_recomputes_leave_one_out_center() {
        let (n, h) = (9, 6);
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let ids: Vec<usize> = (0..n).map(|i| 100 + i).collect();
        // A shared offset plus a label direction plus noise; one far outlier
        // moves the full mean well away from every leave-one-out mean.
        let states: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..h)
                    .map(|j| {
                        let outlier = if i == 4 { 40.0 } else { 0.0 };
                        5.0 + outlier
                            + if j == labels[i] { 1.0 } else { 0.0 }
                            + noise(i * h + j + 3)
                    })
                    .collect()
            })
            .collect();
        let mut mem = EpisodicManifoldMemory::new(h, 3, 64, 4.0, None).unwrap();
        mem.write(&states, &labels, &ids).unwrap();
        let loo = mem.loo_query().unwrap();
        for i in 0..n {
            let keep: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            let kept: Vec<Vec<f64>> = keep.iter().map(|&j| states[j].clone()).collect();
            let mu: Vec<f64> = (0..h)
                .map(|j| kept.iter().map(|r| r[j]).sum::<f64>() / (n - 1) as f64)
                .collect();
            let kept_labels: Vec<usize> = keep.iter().map(|&j| labels[j]).collect();
            let kept_ids: Vec<usize> = keep.iter().map(|&j| ids[j]).collect();
            let mut other = EpisodicManifoldMemory::new(h, 3, 64, 4.0, Some(mu)).unwrap();
            other.write(&kept, &kept_labels, &kept_ids).unwrap();
            let direct = other.query(&states[i], None).unwrap();
            for (d, l) in direct.iter().zip(&loo[i]) {
                assert!(
                    (d - l).abs() < 1e-6,
                    "entry {i}: loo {:?} vs leave-one-out memory {direct:?}",
                    loo[i]
                );
            }
        }
    }

    /// Gate check in H = 1536 with n = 80: random keys carry no label
    /// information, yet a saturated base probe gains from mixing in anything.
    /// Only the comparison with the leave-one-out prior rejects that memory.
    #[test]
    fn memory_gate_rejects_high_dimensional_noise() {
        let (n, h) = (80, 1536);
        let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let ids: Vec<usize> = (0..n).collect();
        let states: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..h).map(|j| noise(i * h + j + 7)).collect())
            .collect();
        let mut mem = EpisodicManifoldMemory::new(h, 2, 256, 12.0, None).unwrap();
        mem.write(&states, &labels, &ids).unwrap();
        let memory_loo = mem.loo_query().unwrap();
        // Saturated probe: 0.999 on the true label for 70% of the rows,
        // 0.001 on the rest.
        let base: Vec<Vec<f64>> = labels
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let right = i % 10 < 7;
                let p = if right == (y == 0) { 0.999 } else { 0.001 };
                vec![p, 1.0 - p]
            })
            .collect();
        let prior: Vec<Vec<f64>> = labels
            .iter()
            .map(|&y| {
                (0..2)
                    .map(|c| (40.0 - if c == y { 1.0 } else { 0.0 }) / 79.0)
                    .collect()
            })
            .collect();
        let (lam, report) = select_memory_lambda(
            &base,
            &memory_loo,
            &labels,
            &DEFAULT_LAMBDA_GRID,
            MEMORY_SE_MULT,
            Some(&prior),
        )
        .unwrap();
        assert!(
            report
                .candidates
                .iter()
                .any(|c| c.mean_gain > MEMORY_SE_MULT * c.se),
            "the probe must be saturated enough to gain from any mixture: {report:?}"
        );
        assert_eq!(lam, 0.0, "noise memory accepted: {report:?}");
        assert!(report.candidates.iter().all(|c| !c.passed));

        // Positive control: keys clustered by label pass both tests.
        let clustered: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..h)
                    .map(|j| {
                        let sign = if (j % 2 == 0) == (labels[i] == 0) {
                            1.0
                        } else {
                            -1.0
                        };
                        sign + 0.5 * noise(i * h + j + 99)
                    })
                    .collect()
            })
            .collect();
        let mut good = EpisodicManifoldMemory::new(h, 2, 256, 12.0, None).unwrap();
        good.write(&clustered, &labels, &ids).unwrap();
        let flat = vec![vec![0.5, 0.5]; n];
        let (lam, _) = select_memory_lambda(
            &flat,
            &good.loo_query().unwrap(),
            &labels,
            &DEFAULT_LAMBDA_GRID,
            MEMORY_SE_MULT,
            Some(&prior),
        )
        .unwrap();
        assert!(lam > 0.0, "an informative memory was rejected");
    }

    /// A positive mean gain below 2 standard errors is not enough.
    #[test]
    fn memory_gate_requires_two_standard_errors() {
        let labels = vec![0; 8];
        let base = vec![vec![0.5, 0.5]; 8];
        let memory: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                if i % 2 == 0 {
                    vec![0.6, 0.4]
                } else {
                    vec![0.45, 0.55]
                }
            })
            .collect();
        let grid = DEFAULT_LAMBDA_GRID;
        let (lam, report) =
            select_memory_lambda(&base, &memory, &labels, &grid, 2.0, None).unwrap();
        assert!(report
            .candidates
            .iter()
            .all(|c| c.mean_gain > 0.0 && c.mean_gain < 2.0 * c.se));
        assert_eq!(lam, 0.0, "{report:?}");
        let (lam, _) = select_memory_lambda(&base, &memory, &labels, &grid, 0.5, None).unwrap();
        assert!(lam > 0.0);
        assert!(select_memory_lambda(&base, &memory, &labels[..1], &[0.5], 2.0, None).is_err());
        assert!(select_memory_lambda(&base, &memory, &labels, &[1.5], 2.0, None).is_err());
    }

    fn calibrated_fixture(
        model: &QwenModel,
        config: &CalibratedConfig,
    ) -> (InterLayerMctsRunner, CalibratedFitReport, Vec<usize>) {
        let (anchors, labels) = anchor_set();
        let layers = config.hook_layers.clone().unwrap();
        let states: Vec<HookStates> = anchors
            .iter()
            .map(|ids| InterLayerMctsRunner::capture_hook_states(model, ids, &layers).unwrap())
            .collect();
        let groups = [vec![7, 3], vec![19, 11]];
        let (runner, report) =
            InterLayerMctsRunner::fit_calibrated(model, &states, &labels, &groups, config).unwrap();
        (runner, report, layers)
    }

    #[test]
    fn calibrated_runner_full_pass_matches_its_readout() {
        let model = tiny_model(8);
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            min_exit_layer: Some(7),
            fit_layer_weights: false,
            ..Default::default()
        };
        let (runner, report, layers) = calibrated_fixture(&model, &config);
        assert_eq!(report.hook_layers, vec![2, 4, 5, 7]);
        assert_eq!(report.anchors, 10);
        for row in &report.layers {
            assert!(row.probe_logit_std.iter().all(|s| *s > 0.0));
            assert!((0.0..=1.0).contains(&row.memory_lambda));
        }
        let ids = vec![5, 9, 2, 8];
        let d = runner.predict(&model, &ids).unwrap();
        assert!(!d.early_exit);
        assert_eq!(d.layers_executed, 8);
        let states = InterLayerMctsRunner::capture_hook_states(&model, &ids, &layers).unwrap();
        let (raw, normed) = states.layer(7).unwrap();
        let want = runner.layer_probs(7, raw, normed).unwrap();
        for (a, b) in d.probs.iter().zip(&want) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        // The probe is the logsumexp of each label's lm_head rows on the
        // final-norm state, standardised by the anchor stats.
        let entry = &runner.calibrated.as_ref().unwrap().layers[&7];
        let rows: Vec<Vec<f64>> = model
            .lm_head_rows(&[7, 3, 19, 11])
            .unwrap()
            .into_iter()
            .map(to_f64)
            .collect();
        let lse = |a: f64, b: f64| (a.exp() + b.exp()).ln();
        let z = [
            lse(dot(&rows[0], normed), dot(&rows[1], normed)),
            lse(dot(&rows[2], normed), dot(&rows[3], normed)),
        ];
        let mut expected = softmax(&entry.stats.standardize(&z).unwrap());
        if entry.lambda > 0.0 {
            let m = entry.memory.query(raw, None).unwrap();
            for (e, v) in expected.iter_mut().zip(m) {
                *e = (1.0 - entry.lambda) * *e + entry.lambda * v;
            }
        }
        for (a, b) in d.probs.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        let (anchors, labels) = anchor_set();
        let short: Vec<HookStates> = anchors
            .iter()
            .map(|a| InterLayerMctsRunner::capture_hook_states(&model, a, &[2, 4, 7]).unwrap())
            .collect();
        let groups = [vec![7, 3], vec![19, 11]];
        let fit = |s: &[HookStates], y: &[usize], g: &[Vec<u32>]| {
            InterLayerMctsRunner::fit_calibrated(&model, s, y, g, &config)
        };
        // Missing hook layer 5, too few anchors, an empty token group.
        assert!(fit(&short, &labels, &groups).is_err());
        assert!(fit(&short[..2], &labels[..2], &groups).is_err());
        assert!(fit(&short, &labels, &[vec![7], vec![]]).is_err());
        assert!(InterLayerMctsRunner::capture_hook_states(&model, &ids, &[8]).is_err());
    }

    #[test]
    fn calibrated_runner_exits_on_a_confident_stable_readout() {
        let model = tiny_model(8);
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            tau: 0.25,
            min_exit_layer: Some(2),
            mcts_margin_min: 0.0,
            top_k_hooks: 2,
            ..Default::default()
        };
        let (mut runner, report, _) = calibrated_fixture(&model, &config);
        let w = runner.layer_weights().clone();
        assert!((w.values().sum::<f64>() - 1.0).abs() < 1e-12, "{w:?}");
        assert!(w.values().filter(|v| **v > 0.0).count() <= 2);
        assert!(report.layers.iter().all(|r| r.weight.is_some()));
        // Every checkpoint reads "label 0" with near certainty: drift ~0
        // gives a ~0 bound, the label is stable and the MCTS confirms it.
        runner.layer_weights.clear();
        for entry in runner.calibrated.as_mut().unwrap().layers.values_mut() {
            entry.stats.mean = vec![-40.0, 0.0];
            entry.stats.std = vec![1.0, 1e6];
            entry.lambda = 0.0;
        }
        let ids = vec![3, 1, 4, 1, 5];
        let d = runner.predict(&model, &ids).unwrap();
        assert!(d.early_exit, "{d:?}");
        assert_eq!((d.exit_layer, d.layers_executed, d.label), (4, 5, 0));
        assert!(d.margin > d.threshold.unwrap());
        let read: Vec<usize> = d.checkpoints.iter().map(|c| c.layer).collect();
        assert_eq!(read, vec![2, 4]);
        assert!(d.checkpoints[1].gamma_hat.is_some() && d.checkpoints[1].stable);
        assert_eq!(d.ffn.ffn_dense_channels, 32 * 5);
        // Below min_exit_layer nothing exits: the pass runs all 8 layers.
        runner.min_exit_layer = 6;
        let d = runner.predict(&model, &ids).unwrap();
        assert!(!d.early_exit);
        assert_eq!(d.layers_executed, 8);
        // Weights with no mass on any checkpoint forbid an exit.
        runner.min_exit_layer = 2;
        runner.layer_weights = BTreeMap::from([(2, 0.0), (4, 0.0), (5, 0.0), (7, 1.0)]);
        assert!(!runner.predict(&model, &ids).unwrap().early_exit);
    }

    #[test]
    fn mcts_pruned_candidates_never_run_deep_layers() {
        let model = tiny_model(8);
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            fit_layer_weights: false,
            ..Default::default()
        };
        let (runner, _, layers) = calibrated_fixture(&model, &config);
        let candidates: Vec<Vec<u32>> = (0..5u32)
            .map(|i| vec![2 + i, 9, 3 * i % 7 + 1, 5])
            .collect();
        let res = runner
            .score_candidates_mcts_pruned(&model, &candidates, 0, 5, 2)
            .unwrap();
        assert_eq!(res.survivors.len(), 2);
        assert_eq!(res.shallow_layers_executed, 5 * 5);
        assert_eq!(res.deep_layers_executed, 2 * 8);
        assert_eq!(res.layers_executed, 41);
        assert_eq!(res.full_layers, 40);
        assert!(res.survivors.contains(&res.best));
        let top_visits = *res.mcts_visits.iter().max().unwrap();
        assert_eq!(res.mcts_visits[res.survivors[0]], top_visits);
        assert_eq!(res.deep_values.len(), 2);
        for (i, ids) in candidates.iter().enumerate() {
            let s = InterLayerMctsRunner::capture_hook_states(&model, ids, &layers).unwrap();
            let (raw, normed) = s.layer(4).unwrap();
            let shallow = runner.layer_probs(4, raw, normed).unwrap()[0];
            assert!((shallow - res.shallow_values[i]).abs() < 1e-9);
            if let Some(v) = res.deep_values.get(&i) {
                let (raw, normed) = s.layer(7).unwrap();
                assert!((runner.layer_probs(7, raw, normed).unwrap()[0] - v).abs() < 1e-9);
            }
        }
        let best_deep = res.deep_values.values().copied().fold(f64::MIN, f64::max);
        assert_eq!(res.deep_values[&res.best], best_deep);
        let score = |target, shallow, top_m| {
            runner.score_candidates_mcts_pruned(&model, &candidates, target, shallow, top_m)
        };
        assert!(score(0, 5, 0).is_err());
        assert!(score(0, 8, 2).is_err());
        assert!(score(2, 5, 2).is_err());
        // Cut at layer 3, which has no calibrated stats: refuse, never read raw.
        assert!(score(0, 4, 2).is_err());
    }

    #[test]
    fn boolq_rows_prompt_and_mcnemar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("boolq.jsonl");
        let row = |id: &str, label: &str, q: &str| {
            serde_json::json!({
                "id": id, "ground_truth": label,
                "context": format!("Passage: p {id}{BOOLQ_PROMPT_SUFFIX}"),
                "metadata": {"question": q},
            })
            .to_string()
        };
        let lines = [
            row("a", "yes", "Is it?"),
            row("b", "no", " is it"),
            row("c", "no", "other q"),
            row("d", "yes", "third"),
            row("e", "no", "fifth"),
            row("f", "yes", "sixth"),
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();
        let rows = boolq_rows(&path, 4).unwrap();
        // "b" repeats "Is it?" after trimming, lower-casing and dropping "?".
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["a", "c", "d", "e"]);
        assert_eq!(
            rows.iter().map(|r| r.label).collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert!(boolq_rows(&path, 6).is_err());
        std::fs::write(&path, row("x", "maybe", "q")).unwrap();
        assert!(boolq_rows(&path, 2).is_err());
        assert_eq!(
            qwen25_chat_prompt("hi"),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful \
             assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert!((mcnemar_exact(5, 0) - 0.0625).abs() < 1e-12);
        assert!((mcnemar_exact(3, 1) - 0.625).abs() < 1e-12);
    }

    /// Drive the calibrated exit rule over hand-written checkpoints, with
    /// the MCTS confirmation assumed: index of the first exit, if any, and
    /// the readout the pass returns. Trajectory logits are `ln p`, as for a
    /// memory-mixed readout.
    fn run_exit_rule(weights: &[f64], dists: &[[f64; 2]]) -> (Option<usize>, Vec<f64>) {
        let mut tracker = WeightedExitTracker::new(true, 0.5, 0.25);
        let last = dists.len() - 1;
        let mut probs = Vec::new();
        for (i, (w, p)) in weights.iter().zip(dists).enumerate() {
            let traj: Vec<f64> = p.iter().map(|v| v.ln()).collect();
            let eval = tracker.observe(*w, &traj, p.to_vec(), i < last).unwrap();
            if i < last && eval.passes_bound() {
                assert!(tracker.effective_steps >= 2, "exit on one observation");
                return (Some(i), eval.probs);
            }
            probs = eval.probs;
        }
        (None, probs)
    }

    /// M06 (WebGPT round 1): hooks [12, 16, 27], weights [0.5, 0, 0.5].
    /// Hook 16 carries no weight, so the running mean does not move and
    /// its drift and bound are 0; counting it as an observation exited
    /// there with label 0. Skipping it gives the full weighted readout,
    /// label 1 at about [0.0866878, 0.9133122].
    #[test]
    fn zero_weight_checkpoint_is_no_evidence_of_convergence() {
        let (sharp0, sharp1) = ([0.9, 0.1], [0.001, 0.999]);
        let want = [0.086_687_8, 0.913_312_2];
        let close = |p: &[f64]| p.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6);

        let (exit, p) = run_exit_rule(&[0.5, 0.0, 0.5], &[sharp0, sharp1, sharp1]);
        assert_eq!(exit, None, "exited at a zero-weight checkpoint: {p:?}");
        assert_eq!(argmax(&p), 1);
        assert!(close(&p), "{p:?}");

        // Two zero-weight checkpoints in a row.
        let (exit, p) = run_exit_rule(&[0.5, 0.0, 0.0, 0.5], &[sharp0, sharp1, sharp1, sharp1]);
        assert_eq!(exit, None, "{p:?}");
        assert!(close(&p), "{p:?}");

        // Only the last layer has weight: no checkpoint is an observation.
        let (exit, p) = run_exit_rule(&[0.0, 0.0, 1.0], &[sharp0, sharp0, sharp1]);
        assert_eq!(exit, None, "{p:?}");
        assert!((p[1] - 0.999).abs() < 1e-9, "{p:?}");

        // Positive control: two weighted checkpoints that agree do exit.
        let (exit, _) = run_exit_rule(&[0.5, 0.5, 0.0], &[sharp0, sharp0, sharp1]);
        assert_eq!(exit, Some(1));
    }

    /// The same rule inside `predict`: confident readouts everywhere, but
    /// only hook 2 and the last layer carry weight, so no checkpoint has a
    /// second observation to compare with and the pass runs to the end.
    #[test]
    fn calibrated_runner_never_exits_on_zero_weight_checkpoints() {
        let model = tiny_model(8);
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            tau: 0.25,
            min_exit_layer: Some(2),
            mcts_margin_min: 0.0,
            fit_layer_weights: false,
            ..Default::default()
        };
        let (mut runner, _, _) = calibrated_fixture(&model, &config);
        for entry in runner.calibrated.as_mut().unwrap().layers.values_mut() {
            entry.stats.mean = vec![-40.0, 0.0];
            entry.stats.std = vec![1.0, 1e6];
            entry.lambda = 0.0;
        }
        let ids = vec![3, 1, 4, 1, 5];
        runner.layer_weights = BTreeMap::from([(2, 0.5), (4, 0.0), (5, 0.0), (7, 0.5)]);
        let d = runner.predict(&model, &ids).unwrap();
        assert!(!d.early_exit, "{d:?}");
        assert_eq!(d.layers_executed, 8);
        assert!(d.checkpoints[1..3]
            .iter()
            .all(|c| c.bound.is_none() && !c.stable));
        // Positive control: weight on hooks 2 and 4 exits at 4.
        runner.layer_weights = BTreeMap::from([(2, 0.5), (4, 0.5), (5, 0.0), (7, 0.0)]);
        let d = runner.predict(&model, &ids).unwrap();
        assert!(d.early_exit, "{d:?}");
        assert_eq!(d.exit_layer, 4);
    }

    /// The no-exit ablation arm equals a `predict` that cannot exit, and the
    /// equal-weight arm equals the runner with weights `1 / |hooks|`.
    #[test]
    fn full_depth_readout_matches_a_never_exiting_predict() {
        let model = tiny_model(8);
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 4, 5, 7]),
            min_exit_layer: Some(7),
            top_k_hooks: 2,
            ..Default::default()
        };
        let (mut runner, _, layers) = calibrated_fixture(&model, &config);
        assert!(!runner.layer_weights().is_empty());
        for ids in [vec![5, 9, 2, 8], vec![3, 1, 4, 1, 5]] {
            let states = InterLayerMctsRunner::capture_hook_states(&model, &ids, &layers).unwrap();
            let d = runner.predict(&model, &ids).unwrap();
            assert!(!d.early_exit);
            let full = runner.full_depth_readout(&states, false).unwrap();
            for (a, b) in d.probs.iter().zip(&full) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
            let equal = runner.full_depth_readout(&states, true).unwrap();
            let saved = std::mem::replace(
                &mut runner.layer_weights,
                layers.iter().map(|&l| (l, 0.25)).collect(),
            );
            let d = runner.predict(&model, &ids).unwrap();
            runner.layer_weights = saved;
            for (a, b) in d.probs.iter().zip(&equal) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
        }
    }

    /// M07: anchors the FIFO dropped still get an out-of-fold row (all
    /// retained entries, retained-mean centre); retained anchors get their
    /// `loo_query` row. Brute force: a memory holding exactly the entries
    /// the read may see.
    #[test]
    fn loo_query_for_all_anchors_keeps_one_row_per_anchor() {
        let (n, cap, h) = (9, 5, 6);
        let states: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..h)
                    .map(|j| noise(i * h + j + 3) + (i % 3) as f64)
                    .collect()
            })
            .collect();
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let ids: Vec<usize> = (0..n).collect();
        let kept = &states[n - cap..];
        let mean = |rows: &[Vec<f64>]| -> Vec<f64> {
            (0..h)
                .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64)
                .collect()
        };
        let mut mem = EpisodicManifoldMemory::new(h, 3, cap, 4.0, Some(mean(kept))).unwrap();
        mem.write(&states, &labels, &ids).unwrap();
        assert_eq!(mem.len(), cap);
        let rows = mem.loo_query_for_all_anchors(&states, &ids).unwrap();
        assert_eq!(rows.len(), n);
        let loo = mem.loo_query().unwrap();
        for i in 0..n {
            let want = if i >= n - cap {
                loo[i - (n - cap)].clone()
            } else {
                let mut seen =
                    EpisodicManifoldMemory::new(h, 3, cap, 4.0, Some(mean(kept))).unwrap();
                seen.write(kept, &labels[n - cap..], &ids[n - cap..])
                    .unwrap();
                seen.query(&states[i], None).unwrap()
            };
            for (a, b) in rows[i].iter().zip(&want) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "anchor {i}: {:?} vs {want:?}",
                    rows[i]
                );
            }
        }
        assert!(mem
            .loo_query_for_all_anchors(&states[..2], &ids[..1])
            .is_err());
    }

    /// M07: more training anchors than memory slots (256) must fit, with the
    /// lambda gate scored on every anchor. 258 and 800 are the training rows
    /// of a 5-fold split of 322 and 1000.
    #[test]
    fn calibrated_fit_beyond_memory_capacity_keeps_rows_aligned() {
        let model = tiny_model(8);
        let layers = vec![2, 4, 7];
        let (anchors, base_labels) = anchor_set();
        let base: Vec<HookStates> = anchors
            .iter()
            .map(|ids| InterLayerMctsRunner::capture_hook_states(&model, ids, &layers).unwrap())
            .collect();
        let jitter = |s: &HookStates, i: usize| {
            let mut s = s.clone();
            for (l, v) in s.raw.iter_mut().chain(s.normed.iter_mut()) {
                for (j, x) in v.iter_mut().enumerate() {
                    *x += 0.05 * noise(i * 7919 + l * 131 + j);
                }
            }
            s
        };
        let config = CalibratedConfig {
            hook_layers: Some(layers),
            fit_layer_weights: false,
            ..Default::default()
        };
        let groups = [vec![7, 3], vec![19, 11]];
        let cap = config.memory_capacity;
        assert_eq!(cap, 256);
        for n in [cap, cap + 1, 258, 800] {
            let states: Vec<HookStates> = (0..n).map(|i| jitter(&base[i % 10], i)).collect();
            let labels: Vec<usize> = (0..n).map(|i| base_labels[i % 10]).collect();
            let (runner, report) =
                InterLayerMctsRunner::fit_calibrated(&model, &states, &labels, &groups, &config)
                    .unwrap_or_else(|e| panic!("n = {n}: {e}"));
            assert_eq!(report.anchors, n);
            for row in &report.layers {
                assert_eq!(row.memory_gate.n, n, "layer {}", row.layer);
            }
            for (layer, entry) in &runner.calibrated.as_ref().unwrap().layers {
                assert_eq!(entry.memory.len(), n.min(cap));
                // Centred on the retained anchors, as the out-of-fold reads.
                let kept = &states[n.saturating_sub(cap)..];
                let center = entry.memory.center.as_ref().unwrap();
                for (j, c) in center.iter().enumerate() {
                    let mean =
                        kept.iter().map(|s| s.raw[layer][j]).sum::<f64>() / kept.len() as f64;
                    assert!((c - mean).abs() < 1e-9, "n = {n}, layer {layer}");
                }
            }
        }
    }

    /// A memory capacity below 2 fails before any forward pass, in the fit
    /// and in the benchmark (whose model path does not exist here).
    #[test]
    fn bad_memory_capacity_fails_before_any_forward() {
        for cap in [0, 1, 2] {
            let config = CalibratedConfig {
                memory_capacity: cap,
                ..Default::default()
            };
            assert!(config.validate().is_err());
            let opts = BoolqBenchOptions {
                model_path: PathBuf::from("/nonexistent/qwen"),
                tokenizer_path: None,
                data: PathBuf::from("/nonexistent/boolq.jsonl"),
                n: 100,
                folds: 5,
                ffn_keep_ratio: 1.0,
                ffn_layers: None,
                hook_layers: None,
                min_exit_layer: 22,
                tau: 0.25,
                gamma: 0.5,
                mcts_margin_min: 0.05,
                top_k_hooks: 4,
                prune_group_size: 0,
                prune_shallow_layers: 21,
                prune_top_m: 2,
                memory_capacity: cap,
            };
            let err = run_boolq_benchmark(&opts).unwrap_err().to_string();
            assert!(err.contains("memory_capacity"), "{err}");
        }
        let model = tiny_model(8);
        let (anchors, labels) = anchor_set();
        let states: Vec<HookStates> = anchors
            .iter()
            .map(|ids| InterLayerMctsRunner::capture_hook_states(&model, ids, &[2, 7]).unwrap())
            .collect();
        let config = CalibratedConfig {
            hook_layers: Some(vec![2, 7]),
            memory_capacity: 0,
            ..Default::default()
        };
        let groups = [vec![7, 3], vec![19, 11]];
        let err = InterLayerMctsRunner::fit_calibrated(&model, &states, &labels, &groups, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("memory_capacity"), "{err}");
    }

    #[test]
    fn fifo_center_loo_and_duplicate_ids_regression() {
        let states: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![i as f64, (i * i % 7) as f64, (i * 3 % 5) as f64])
            .collect();
        let labels: Vec<usize> = (0..8).map(|i| i % 2).collect();
        let ids: Vec<usize> = (0..8).collect();
        let mut memory = EpisodicManifoldMemory::new(3, 2, 4, 3.0, Some(vec![0.0; 3])).unwrap();
        memory.write(&states, &labels, &ids).unwrap();
        let center = memory.center.as_ref().unwrap().clone();
        for j in 0..3 {
            let expected = states[4..].iter().map(|h| h[j]).sum::<f64>() / 4.0;
            assert!((center[j] - expected).abs() < 1e-12);
        }
        let loo = memory.loo_query_for_all_anchors(&states, &ids).unwrap();
        let deployed = memory.query(&states[0], None).unwrap();
        for c in 0..2 {
            assert!((loo[0][c] - deployed[c]).abs() < 1e-12);
        }
        for i in 4..8 {
            let kept: Vec<usize> = (4..8).filter(|j| *j != i).collect();
            let raw: Vec<Vec<f64>> = kept.iter().map(|&j| states[j].clone()).collect();
            let ys: Vec<usize> = kept.iter().map(|&j| labels[j]).collect();
            let mut refit = EpisodicManifoldMemory::new(3, 2, 4, 3.0, Some(vec![0.0; 3])).unwrap();
            refit.write(&raw, &ys, &kept).unwrap();
            let expected = refit.query(&states[i], None).unwrap();
            for c in 0..2 {
                assert!((loo[i][c] - expected[c]).abs() < 1e-12);
                assert!(
                    (0.4 * loo[i][c] + 0.6 * 0.5 - (0.4 * expected[c] + 0.6 * 0.5)).abs() < 1e-12
                );
            }
        }
        assert!(memory.write(&[states[0].clone()], &[0], &[4]).is_err());
        assert!(memory.write(&states[..2], &labels[..2], &[9, 9]).is_err());
        assert!(memory
            .loo_query_for_all_anchors(&states[..2], &[0, 0])
            .is_err());
        assert!(memory.query(&center, None).is_err());
        let mut same = EpisodicManifoldMemory::new(3, 2, 3, 3.0, Some(vec![0.0; 3])).unwrap();
        assert!(same
            .write(&vec![vec![1.0; 3]; 3], &[0, 1, 0], &[0, 1, 2])
            .is_err());
        assert!(EpisodicManifoldMemory::new(3, 2, 2, 3.0, None).is_err());
    }

    /// WebGPT round 1, section 4: on 80 balanced rows the leave-one-out
    /// prior gives the true label 39/79 < 1/2, so a constant [0.5, 0.5]
    /// memory beats it (and the saturated base) by many standard errors
    /// while carrying no information. Against the uniform row its gain is
    /// exactly 0, so the gate keeps lambda = 0.
    #[test]
    fn memory_gate_rejects_a_constant_uniform_memory() {
        let n = 80;
        let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let base: Vec<Vec<f64>> = labels
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let right = i % 10 < 7;
                let p = if right == (y == 0) { 0.999 } else { 0.001 };
                vec![p, 1.0 - p]
            })
            .collect();
        let prior: Vec<Vec<f64>> = labels
            .iter()
            .map(|&y| {
                (0..2)
                    .map(|c| (40.0 - if c == y { 1.0 } else { 0.0 }) / 79.0)
                    .collect()
            })
            .collect();
        let flat = vec![vec![0.5, 0.5]; n];
        let (lam, report) = select_memory_lambda(
            &base,
            &flat,
            &labels,
            &DEFAULT_LAMBDA_GRID,
            MEMORY_SE_MULT,
            Some(&prior),
        )
        .unwrap();
        // The old two-test gate (base and LOO prior) would accept it.
        assert!(
            report.candidates.iter().any(|c| {
                c.mean_gain > MEMORY_SE_MULT * c.se
                    && c.gain_over_null.unwrap() > MEMORY_SE_MULT * c.gain_over_null_se.unwrap()
            }),
            "fixture too weak: {report:?}"
        );
        assert!(report.candidates.iter().all(|c| c.gain_over_uniform == 0.0));
        assert!(report
            .candidates
            .iter()
            .all(|c| c.gain_over_laplace.is_some()));
        assert_eq!(lam, 0.0, "uninformative memory accepted: {report:?}");
        assert!(report.candidates.iter().all(|c| !c.passed));
    }
}
