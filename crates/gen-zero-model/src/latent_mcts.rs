//! Zero-token latent MCTS over simplex-ETF candidate embeddings.
//!
//! A port of the default path of `LatentMctsPlanner.plan` in
//! `python/gen_zero/causal/latent_mcts.py`: plain PUCT over the full budget,
//! the contractive spherical transition without fitted Hamiltonian
//! generators, every candidate legal in every state. The opt-in Python
//! extras (Sequential Halving, early stop, transposition cache, legal masks)
//! are not used by the inter-layer runner and are not ported.
//!
//! Why a local planner instead of `gen-zero-planner`: that crate searches
//! action sequences over a world model, not label or layer candidates in a
//! latent simplex, and `gen-zero-model` cannot depend on it without adding a
//! dependency edge (and a `Cargo.lock` change) for one small search.

use crate::error::ModelError;

const EPS: f64 = 1e-12;
const TRANSITION_MAX_ANGLE: f64 = 0.3;

/// `k` unit rows of a simplex equiangular tight frame in `k` dimensions.
pub fn simplex_etf(k: usize) -> Result<Vec<Vec<f64>>, ModelError> {
    match k {
        0 => Err(ModelError::QwenInput("simplex ETF needs k >= 1".into())),
        1 => Ok(vec![vec![1.0]]),
        _ => {
            let scale = (k as f64 / (k as f64 - 1.0)).sqrt();
            Ok((0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| scale * (if i == j { 1.0 } else { 0.0 } - 1.0 / k as f64))
                        .collect()
                })
                .collect())
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn unit(z: &[f64]) -> Vec<f64> {
    let scale = z.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return vec![0.0; z.len()];
    }
    // Scaling first avoids overflow or underflow for extreme observations.
    let scaled: Vec<f64> = z.iter().map(|v| v / scale).collect();
    let norm = dot(&scaled, &scaled).sqrt();
    scaled.iter().map(|v| v / norm).collect()
}

pub(crate) fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Confidence of a prior, discounted by Lyapunov evidence (not correctness).
fn leaf_value(priors: &[f64], stability: f64) -> f64 {
    let k = priors.len();
    if k == 1 {
        return stability;
    }
    let mut sorted = priors.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let margin = sorted[k - 1] - sorted[k - 2];
    let entropy: f64 = -priors.iter().map(|p| p * (p + EPS).ln()).sum::<f64>();
    let certainty = 1.0 - entropy / (k as f64).ln();
    (0.5 * margin + 0.5 * certainty.clamp(0.0, 1.0)) * stability
}

/// One step of the spherical flow toward target `a` with `V(z) = 1 - e_a . z`
/// (`ContractiveHamiltonianTransition` without generators).
fn transition(targets: &[Vec<f64>], gamma: f64, z: &[f64], a: usize) -> Vec<f64> {
    let target = &targets[a];
    let u = unit(z);
    if u.iter().all(|v| *v == 0.0) {
        return target.clone();
    }
    let cosine = dot(target, &u).clamp(-1.0, 1.0);
    let tangent: Vec<f64> = target.iter().zip(&u).map(|(t, x)| t - cosine * x).collect();
    let sine = dot(&tangent, &tangent).sqrt();
    if sine < EPS {
        return u;
    }
    let theta = sine.atan2(cosine);
    let next_theta = 2.0 * ((-gamma).exp() * (theta / 2.0).sin()).atan2((theta / 2.0).cos());
    let angle = TRANSITION_MAX_ANGLE.min((theta - next_theta).max(0.0));
    let moved: Vec<f64> = u
        .iter()
        .zip(&tangent)
        .map(|(x, t)| angle.cos() * x + angle.sin() * t / sine)
        .collect();
    unit(&moved)
}

struct Node {
    z: Vec<f64>,
    unit_z: Vec<f64>,
    priors: Vec<f64>,
    visits: Vec<u64>,
    q: Vec<f64>,
    total_visits: u64,
    children: Vec<Option<usize>>,
    value: f64,
    evidence: f64,
}

impl Node {
    fn new(z: Vec<f64>, priors: Vec<f64>, evidence: f64) -> Self {
        let k = priors.len();
        Self {
            unit_z: unit(&z),
            value: leaf_value(&priors, evidence),
            z,
            priors,
            visits: vec![0; k],
            q: vec![0.0; k],
            total_visits: 0,
            children: vec![None; k],
            evidence,
        }
    }

    /// PUCT: `argmax_a Q + c * P * sqrt(max(N_total, 1)) / (1 + N)`, first max wins.
    fn select(&self, c_puct: f64) -> usize {
        let root = (self.total_visits.max(1) as f64).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for a in 0..self.priors.len() {
            let score = c_puct * self.priors[a] * root / (1.0 + self.visits[a] as f64) + self.q[a];
            if score > best_score {
                best = a;
                best_score = score;
            }
        }
        best
    }
}

/// Root statistics of one search.
#[derive(Clone, Debug, PartialEq)]
pub struct MctsPlan {
    pub best: usize,
    /// Root visit shares.
    pub policy: Vec<f64>,
    pub root_q: Vec<f64>,
    pub root_visits: Vec<u64>,
    pub root_priors: Vec<f64>,
}

/// PUCT planner; defaults match the Python constructor.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentMctsPlanner {
    pub num_simulations: usize,
    pub c_puct: f64,
    pub temperature: f64,
    pub max_depth: usize,
    pub damping: f64,
}

impl Default for LatentMctsPlanner {
    fn default() -> Self {
        Self {
            num_simulations: 100,
            c_puct: 1.414,
            temperature: 0.1,
            max_depth: 6,
            damping: 0.3,
        }
    }
}

impl LatentMctsPlanner {
    fn validate(&self) -> Result<(), ModelError> {
        let ok = self.num_simulations >= 1
            && self.temperature.is_finite()
            && self.temperature > 0.0
            && self.c_puct.is_finite()
            && self.c_puct >= 0.0
            && self.max_depth >= 1
            && self.damping > 0.0
            && self.damping <= 1.0;
        if ok {
            Ok(())
        } else {
            Err(ModelError::QwenInput(format!(
                "invalid MCTS planner {self:?}"
            )))
        }
    }

    fn project(&self, z: &[f64], emb: &[Vec<f64>]) -> Result<Vec<f64>, ModelError> {
        let logits: Vec<f64> = emb.iter().map(|e| dot(e, z) / self.temperature).collect();
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(ModelError::Inference(
                "non-finite MCTS logits or priors".into(),
            ));
        }
        let priors = softmax(&logits);
        if !priors.iter().all(|x| x.is_finite()) {
            return Err(ModelError::Inference(
                "non-finite MCTS logits or priors".into(),
            ));
        }
        Ok(priors)
    }

    /// Search from `initial_state` over `embeddings` (one row per candidate).
    pub fn plan(
        &self,
        initial_state: &[f64],
        embeddings: &[Vec<f64>],
    ) -> Result<MctsPlan, ModelError> {
        self.validate()?;
        let k = embeddings.len();
        let d = initial_state.len();
        if k == 0 || embeddings.iter().any(|e| e.len() != d) {
            return Err(ModelError::QwenInput(format!(
                "MCTS needs candidate embeddings of shape ({k}, {d})"
            )));
        }
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if !finite(initial_state) || !embeddings.iter().all(|e| finite(e)) {
            return Err(ModelError::QwenInput(
                "MCTS states and embeddings must be finite".into(),
            ));
        }
        if embeddings.iter().any(|e| e.iter().all(|x| *x == 0.0)) {
            return Err(ModelError::QwenInput(
                "MCTS candidate embeddings must be nonzero".into(),
            ));
        }
        let targets: Vec<Vec<f64>> = embeddings.iter().map(|e| unit(e)).collect();
        let root_unit = unit(initial_state);
        let mut nodes = vec![Node::new(
            initial_state.to_vec(),
            self.project(initial_state, embeddings)?,
            1.0,
        )];
        for _ in 0..self.num_simulations {
            let mut node = 0;
            let mut path = Vec::new();
            let mut value = nodes[0].value;
            for _ in 0..self.max_depth {
                let a = nodes[node].select(self.c_puct);
                path.push((node, a));
                match nodes[node].children[a] {
                    Some(child) => {
                        node = child;
                        value = nodes[child].value;
                    }
                    None => {
                        let parent = &nodes[node];
                        let z_next = transition(&targets, self.damping, &parent.z, a);
                        let child_unit = unit(&z_next);
                        let before = dot(&targets[a], &parent.unit_z);
                        let after = dot(&targets[a], &child_unit);
                        let stability = (-(before - after).max(0.0)).exp();
                        let gap: Vec<f64> = child_unit
                            .iter()
                            .zip(&root_unit)
                            .map(|(c, r)| c - r)
                            .collect();
                        let relational = (-0.5 * dot(&gap, &gap).sqrt()).exp();
                        let evidence = parent.evidence.min(stability * relational);
                        let priors = self.project(&z_next, embeddings)?;
                        let child = Node::new(z_next, priors, evidence);
                        value = child.value;
                        nodes.push(child);
                        let id = nodes.len() - 1;
                        nodes[node].children[a] = Some(id);
                        break;
                    }
                }
            }
            for (parent, a) in path {
                let p = &mut nodes[parent];
                p.visits[a] += 1;
                p.total_visits += 1;
                p.q[a] += (value - p.q[a]) / p.visits[a] as f64;
            }
        }
        let root = &nodes[0];
        let total = root.visits.iter().sum::<u64>() as f64;
        // Most visits, then Q, then prior; a full tie goes to the later index
        // (numpy lexsort's last element).
        let mut best = 0;
        for a in 1..k {
            let order = root.visits[a]
                .cmp(&root.visits[best])
                .then(root.q[a].total_cmp(&root.q[best]))
                .then(root.priors[a].total_cmp(&root.priors[best]));
            if order != std::cmp::Ordering::Less {
                best = a;
            }
        }
        let plan = MctsPlan {
            best,
            policy: root.visits.iter().map(|&v| v as f64 / total).collect(),
            root_q: root.q.clone(),
            root_visits: root.visits.clone(),
            root_priors: root.priors.clone(),
        };
        if !finite(&plan.policy) || !finite(&plan.root_q) || !finite(&plan.root_priors) {
            return Err(ModelError::Inference(
                "non-finite MCTS logits or priors".into(),
            ));
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etf_rows_are_unit_and_equiangular() {
        for k in 2..6 {
            let e = simplex_etf(k).unwrap();
            for i in 0..k {
                assert!((dot(&e[i], &e[i]) - 1.0).abs() < 1e-12);
                for j in 0..k {
                    if i != j {
                        assert!((dot(&e[i], &e[j]) + 1.0 / (k as f64 - 1.0)).abs() < 1e-12);
                    }
                }
            }
        }
        assert_eq!(simplex_etf(1).unwrap(), vec![vec![1.0]]);
        assert!(simplex_etf(0).is_err());
    }

    #[test]
    fn single_simulation_equals_map_argmax() {
        let planner = LatentMctsPlanner {
            num_simulations: 1,
            temperature: 1.0,
            max_depth: 2,
            ..Default::default()
        };
        let emb = simplex_etf(4).unwrap();
        let plan = planner.plan(&[0.1, 2.0, -0.5, 0.3], &emb).unwrap();
        assert_eq!(plan.best, 1);
        assert_eq!(plan.root_visits.iter().sum::<u64>(), 1);
    }

    #[test]
    fn search_concentrates_visits_on_dominant_candidate() {
        let planner = LatentMctsPlanner {
            num_simulations: 64,
            temperature: 0.25,
            max_depth: 2,
            ..Default::default()
        };
        let emb = simplex_etf(3).unwrap();
        let plan = planner.plan(&[3.0, -1.0, -2.0], &emb).unwrap();
        assert_eq!(plan.best, 0);
        assert_eq!(plan.root_visits.iter().sum::<u64>(), 64);
        assert!((plan.policy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(plan.policy[0] > plan.policy[1] && plan.policy[0] > plan.policy[2]);
    }

    #[test]
    fn transition_moves_toward_target_on_the_sphere() {
        let targets = vec![unit(&[1.0, 0.0, 0.0]), unit(&[0.0, 1.0, 0.0])];
        let z = unit(&[0.2, 1.0, 0.1]);
        let next = transition(&targets, 0.3, &z, 0);
        assert!((dot(&next, &next) - 1.0).abs() < 1e-12);
        assert!(dot(&next, &targets[0]) > dot(&z, &targets[0]));
    }

    #[test]
    fn rejects_overflowed_temperature_logits() {
        let planner = LatentMctsPlanner {
            temperature: 1e-308,
            num_simulations: 2,
            ..Default::default()
        };
        assert!(planner
            .plan(&[10.0, -10.0], &simplex_etf(2).unwrap())
            .is_err());
    }

    #[test]
    fn rejects_bad_inputs() {
        let p = LatentMctsPlanner::default();
        assert!(p.plan(&[1.0], &[]).is_err());
        assert!(p.plan(&[1.0, f64::NAN], &simplex_etf(2).unwrap()).is_err());
        assert!(p.plan(&[1.0, 0.0], &[vec![0.0, 0.0]]).is_err());
        let bad = LatentMctsPlanner {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(bad.plan(&[1.0, 0.0], &simplex_etf(2).unwrap()).is_err());
    }
}
