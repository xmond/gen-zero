//! Multi-step semantic lookahead for the `imagine` verb.
//!
//! PUCT Monte Carlo tree search over action sequences of length `horizon`.
//! A node is a history of actions already taken. Its priors come from a
//! [`PriorOracle`]: in production the semantic bridge, which asks the Zero
//! backbone for `P(next action | scenario, history)`.
//!
//! Value of a simulated path = geometric mean of the step probabilities along
//! it, `exp(mean log p)`, in (0, 1]. It is the `sequence_likelihood`: how
//! strongly the scenario text supports that whole sequence under the language
//! model. It is NOT an environment reward and NOT observed feedback: no
//! environment simulator exists for free-text scenarios, so nothing here
//! measures what would really happen in the world.
//!
//! Root exploration. A confident root prior with a confident continuation
//! used to starve the other root actions (visits `[0, 8, 0]`): the favourite's
//! value plus its prior bonus beat the optimistic start of the others at every
//! step. Two AlphaZero-style measures prevent that:
//! 1. every feasible root action is expanded once before PUCT selection runs,
//!    so each branch gets a real lookahead value to be compared on;
//! 2. the root prior is mixed with Dirichlet noise,
//!    `P' = (1 - eps) P + eps Dir(alpha)`, from a seeded generator, so results
//!    are reproducible and the seed is reported.

use crate::bridge::{AskInput, BridgeError, SemanticBridgeClient};
use gen_zero_core::NormalizedEntropy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub const MAX_HORIZON: usize = 8;
pub const MAX_SIMULATIONS: usize = 128;

pub type PriorFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<f64>, BridgeError>> + Send + 'a>>;

/// Source of next-action probabilities for a history of action indices.
pub trait PriorOracle: Sync {
    fn priors<'a>(&'a self, history: &'a [usize]) -> PriorFuture<'a>;
}

/// Oracle backed by the Python semantic scorer.
pub struct BridgeOracle<'a> {
    pub client: &'a SemanticBridgeClient,
    pub scenario: &'a str,
    pub state: Option<&'a Value>,
    pub candidates: &'a [String],
}

impl PriorOracle for BridgeOracle<'_> {
    fn priors<'a>(&'a self, history: &'a [usize]) -> PriorFuture<'a> {
        Box::pin(async move {
            let names: Vec<String> = history
                .iter()
                .map(|&i| self.candidates[i].clone())
                .collect();
            let resp = self
                .client
                .semantic_ask(&AskInput {
                    context: self.scenario,
                    candidates: self.candidates,
                    state: self.state,
                    history: &names,
                    return_embedding: false,
                })
                .await?;
            // The bridge validated that every candidate is present exactly once.
            Ok(self
                .candidates
                .iter()
                .map(|c| {
                    resp.candidates
                        .iter()
                        .find(|s| &s.name == c)
                        .map(|s| s.probability)
                        .unwrap_or(0.0)
                })
                .collect())
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LookaheadConfig {
    pub horizon: usize,
    pub simulations: usize,
    pub c_puct: f64,
    pub root_noise: RootNoise,
}

/// Dirichlet exploration noise mixed into the root prior only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootNoise {
    pub alpha: f64,
    pub epsilon: f64,
    pub seed: u64,
}

impl Default for RootNoise {
    /// AlphaZero uses eps = 0.25 and alpha ~ 10 / (typical branching); the
    /// candidate lists here are short, so alpha = 0.3 (its chess value).
    fn default() -> Self {
        Self {
            alpha: 0.3,
            epsilon: 0.25,
            seed: 0,
        }
    }
}

/// Default exploration constant (AlphaZero's c_init 1.25).
pub const DEFAULT_C_PUCT: f64 = 1.25;

/// Gamma(shape, 1) by Marsaglia-Tsang; shape < 1 uses the boost
/// `Gamma(a) = Gamma(a + 1) * U^(1/a)`.
fn gamma(rng: &mut StdRng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal by Box-Muller.
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen();
        let x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// One sample from a symmetric Dirichlet(alpha) over `k` outcomes.
fn dirichlet(rng: &mut StdRng, alpha: f64, k: usize) -> Vec<f64> {
    let draws: Vec<f64> = (0..k).map(|_| gamma(rng, alpha)).collect();
    let total: f64 = draws.iter().sum();
    if total > 0.0 && total.is_finite() {
        draws.iter().map(|g| g / total).collect()
    } else {
        vec![1.0 / k as f64; k]
    }
}

#[derive(Clone, Debug)]
pub struct LookaheadStep {
    pub action: usize,
    pub probability: f64,
    pub entropy: NormalizedEntropy,
}

#[derive(Clone, Debug)]
pub struct LookaheadResult {
    pub best_action: usize,
    pub principal_variation: Vec<LookaheadStep>,
    pub root_visits: Vec<u32>,
    pub root_values: Vec<f64>,
    /// Root prior after the Dirichlet mix (the one PUCT selected with).
    pub root_noisy_priors: Vec<f64>,
    /// Mean geometric step likelihood of the best action's simulated paths.
    /// A language-model likelihood, not an environment reward.
    pub sequence_likelihood: f64,
    /// Share of root visits that went to the best action.
    pub confidence: f64,
    pub oracle_calls: usize,
    pub simulations: usize,
}

struct Node {
    priors: Vec<f64>,
    entropy: NormalizedEntropy,
    children: Vec<Option<usize>>,
    visits: Vec<u32>,
    value_sum: Vec<f64>,
}

impl Node {
    fn new(priors: Vec<f64>) -> Self {
        let k = priors.len();
        let p32: Vec<f32> = priors.iter().map(|&p| p as f32).collect();
        Self {
            entropy: NormalizedEntropy::from_probabilities(&p32),
            priors,
            children: vec![None; k],
            visits: vec![0; k],
            value_sum: vec![0.0; k],
        }
    }

    fn q(&self, a: usize) -> f64 {
        if self.visits[a] == 0 {
            0.0
        } else {
            self.value_sum[a] / self.visits[a] as f64
        }
    }

    /// Selection value. Unvisited actions start at the top of the value range
    /// (1.0), so every feasible action is tried once before the prior and the
    /// observed values take over. A pessimistic start (0.0) makes the search
    /// greedy: one visited action with value 0.9 would never be challenged.
    fn q_select(&self, a: usize) -> f64 {
        if self.visits[a] == 0 {
            1.0
        } else {
            self.q(a)
        }
    }

    /// PUCT selection with `priors` (the node's own, or the noisy root prior).
    fn select(&self, priors: &[f64], c_puct: f64, allowed: &[bool]) -> usize {
        let total: u32 = self.visits.iter().sum();
        let sqrt_total = ((total + 1) as f64).sqrt();
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for a in (0..priors.len()).filter(|&a| allowed[a]) {
            let u = c_puct * priors[a] * sqrt_total / (1.0 + self.visits[a] as f64);
            let score = self.q_select(a) + u;
            if score > best_score {
                best_score = score;
                best = Some(a);
            }
        }
        best.unwrap_or(0)
    }

    fn most_visited(&self, allowed: &[bool]) -> usize {
        (0..self.priors.len())
            .filter(|&a| allowed[a])
            .max_by(|&a, &b| {
                self.visits[a]
                    .cmp(&self.visits[b])
                    .then(self.q(a).total_cmp(&self.q(b)))
            })
            .unwrap_or(0)
    }
}

async fn expand<O: PriorOracle>(
    oracle: &O,
    history: &[usize],
    k: usize,
) -> Result<Node, BridgeError> {
    let priors = oracle.priors(history).await?;
    if priors.len() != k || priors.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(BridgeError::InvalidResponse(format!(
            "oracle returned {} priors for {} actions",
            priors.len(),
            k
        )));
    }
    Ok(Node::new(priors))
}

/// Run the search. `allowed[a] == false` removes action `a` from every node
/// (formally infeasible actions never enter a plan).
pub async fn run_lookahead<O: PriorOracle>(
    oracle: &O,
    allowed: &[bool],
    cfg: LookaheadConfig,
) -> Result<LookaheadResult, BridgeError> {
    let k = allowed.len();
    if k == 0 || !allowed.iter().any(|&a| a) {
        return Err(BridgeError::InvalidResponse(
            "no feasible action to plan over".into(),
        ));
    }
    let horizon = cfg.horizon.clamp(1, MAX_HORIZON);
    let simulations = cfg.simulations.clamp(1, MAX_SIMULATIONS);
    let floor = 1e-12_f64;

    let mut nodes = vec![expand(oracle, &[], k).await?];
    let mut oracle_calls = 1;

    let mut rng = StdRng::seed_from_u64(cfg.root_noise.seed);
    let eps = cfg.root_noise.epsilon.clamp(0.0, 1.0);
    let noise = dirichlet(&mut rng, cfg.root_noise.alpha.max(1e-3), k);
    // Noise only over feasible actions, renormalized: an infeasible action
    // must not absorb exploration mass it can never use.
    let feasible_mass: f64 = (0..k).filter(|&a| allowed[a]).map(|a| noise[a]).sum();
    let root_noisy_priors: Vec<f64> = (0..k)
        .map(|a| {
            let n = if allowed[a] && feasible_mass > 0.0 {
                noise[a] / feasible_mass
            } else {
                0.0
            };
            (1.0 - eps) * nodes[0].priors[a] + eps * n
        })
        .collect();

    // Unvisited feasible root actions, most probable first: each gets one
    // simulation before PUCT takes over.
    let mut forced: Vec<usize> = (0..k).filter(|&a| allowed[a]).collect();
    forced.sort_by(|&a, &b| root_noisy_priors[b].total_cmp(&root_noisy_priors[a]));
    let mut forced = forced.into_iter();

    for _ in 0..simulations {
        let mut node = 0;
        let mut history: Vec<usize> = Vec::with_capacity(horizon);
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(horizon);
        let mut log_p = 0.0;
        loop {
            let a = if node == 0 {
                match forced.next() {
                    Some(a) => a,
                    None => nodes[0].select(&root_noisy_priors, cfg.c_puct, allowed),
                }
            } else {
                nodes[node].select(&nodes[node].priors, cfg.c_puct, allowed)
            };
            log_p += nodes[node].priors[a].max(floor).ln();
            history.push(a);
            path.push((node, a));
            if history.len() == horizon {
                break;
            }
            match nodes[node].children[a] {
                Some(child) => node = child,
                None => {
                    let child = expand(oracle, &history, k).await?;
                    oracle_calls += 1;
                    nodes.push(child);
                    let idx = nodes.len() - 1;
                    nodes[node].children[a] = Some(idx);
                    break;
                }
            }
        }
        let value = (log_p / history.len() as f64).exp();
        for (n, a) in path {
            nodes[n].visits[a] += 1;
            nodes[n].value_sum[a] += value;
        }
    }

    let root = &nodes[0];
    let best_action = root.most_visited(allowed);
    let mut principal_variation = Vec::with_capacity(horizon);
    let mut node = Some(0);
    while let Some(n) = node {
        if principal_variation.len() == horizon {
            break;
        }
        let a = nodes[n].most_visited(allowed);
        if nodes[n].visits[a] == 0 {
            break;
        }
        principal_variation.push(LookaheadStep {
            action: a,
            probability: nodes[n].priors[a],
            entropy: nodes[n].entropy,
        });
        node = nodes[n].children[a];
    }

    let total_visits: u32 = root.visits.iter().sum();
    Ok(LookaheadResult {
        best_action,
        sequence_likelihood: root.q(best_action),
        confidence: if total_visits == 0 {
            0.0
        } else {
            root.visits[best_action] as f64 / total_visits as f64
        },
        root_noisy_priors,
        root_visits: root.visits.clone(),
        root_values: (0..k).map(|a| root.q(a)).collect(),
        principal_variation,
        oracle_calls,
        simulations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Table oracle for testing the search algorithm itself: the prior depends
    /// on the last action, like a first-order transition matrix.
    struct TableOracle {
        root: Vec<f64>,
        after: Vec<Vec<f64>>,
        calls: AtomicUsize,
    }

    impl PriorOracle for TableOracle {
        fn priors<'a>(&'a self, history: &'a [usize]) -> PriorFuture<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let p = match history.last() {
                None => self.root.clone(),
                Some(&last) => self.after[last].clone(),
            };
            Box::pin(async move { Ok(p) })
        }
    }

    fn cfg(horizon: usize, simulations: usize) -> LookaheadConfig {
        LookaheadConfig {
            horizon,
            simulations,
            c_puct: DEFAULT_C_PUCT,
            root_noise: RootNoise::default(),
        }
    }

    #[tokio::test]
    async fn lookahead_prefers_the_sequence_with_higher_joint_likelihood() {
        // Action 0 looks best one step ahead (0.55) but leads nowhere certain;
        // action 1 (0.45) is followed by a near-certain continuation.
        let oracle = TableOracle {
            root: vec![0.55, 0.45],
            after: vec![vec![0.5, 0.5], vec![0.02, 0.98]],
            calls: AtomicUsize::new(0),
        };
        let greedy = run_lookahead(&oracle, &[true, true], cfg(1, 64))
            .await
            .unwrap();
        assert_eq!(greedy.best_action, 0);

        let deep = run_lookahead(&oracle, &[true, true], cfg(3, 96))
            .await
            .unwrap();
        assert_eq!(deep.best_action, 1, "{:?}", deep.root_values);
        assert_eq!(deep.principal_variation.len(), 3);
        assert!(deep
            .principal_variation
            .iter()
            .skip(1)
            .all(|s| s.action == 1));
        assert!(deep.sequence_likelihood > greedy.sequence_likelihood);
        assert!(deep.oracle_calls <= 1 + deep.simulations);
    }

    #[tokio::test]
    async fn search_explores_every_feasible_root_action() {
        // A confident prior must not starve the alternatives of visits.
        let oracle = TableOracle {
            root: vec![0.05, 0.9, 0.05],
            after: vec![vec![0.34, 0.33, 0.33]; 3],
            calls: AtomicUsize::new(0),
        };
        let r = run_lookahead(&oracle, &[true, true, true], cfg(2, 12))
            .await
            .unwrap();
        assert!(r.root_visits.iter().all(|&v| v >= 1), "{:?}", r.root_visits);
        assert_eq!(r.best_action, 1);
        assert_eq!(r.oracle_calls, oracle.calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn confident_prior_with_confident_continuation_does_not_starve_the_root() {
        // Reviewer repro: prior 0.9 on one action and a near-certain
        // continuation gave root visits [0, 8, 0] with the optimistic-start
        // selection alone. Every feasible root action must be looked ahead.
        let oracle = TableOracle {
            root: vec![0.05, 0.9, 0.05],
            after: vec![vec![0.02, 0.96, 0.02]; 3],
            calls: AtomicUsize::new(0),
        };
        let r = run_lookahead(&oracle, &[true, true, true], cfg(2, 8))
            .await
            .unwrap();
        assert!(r.root_visits.iter().all(|&v| v >= 1), "{:?}", r.root_visits);
        // Each root branch was expanded, i.e. its continuation was evaluated.
        assert!(
            r.oracle_calls >= 4,
            "root + 3 children, got {}",
            r.oracle_calls
        );
        assert_eq!(r.best_action, 1);
    }

    #[tokio::test]
    async fn root_noise_is_seeded_and_reported() {
        let oracle = TableOracle {
            root: vec![0.2, 0.5, 0.3],
            after: vec![vec![0.3, 0.4, 0.3]; 3],
            calls: AtomicUsize::new(0),
        };
        let a = run_lookahead(&oracle, &[true, true, true], cfg(3, 32))
            .await
            .unwrap();
        let b = run_lookahead(&oracle, &[true, true, true], cfg(3, 32))
            .await
            .unwrap();
        assert_eq!(a.root_visits, b.root_visits, "same seed, same search");
        let noisy = &a.root_noisy_priors;
        assert!((noisy.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_ne!(noisy, &vec![0.2, 0.5, 0.3], "noise was mixed into the root");
    }

    #[test]
    fn dirichlet_sample_is_a_distribution() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for alpha in [0.03, 0.3, 1.0, 3.0] {
            let d = dirichlet(&mut rng, alpha, 5);
            assert_eq!(d.len(), 5);
            assert!(d.iter().all(|&x| (0.0..=1.0).contains(&x)));
            assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{alpha}: {d:?}");
        }
    }

    #[tokio::test]
    async fn infeasible_actions_never_enter_the_plan() {
        let oracle = TableOracle {
            root: vec![0.9, 0.1],
            after: vec![vec![0.9, 0.1], vec![0.9, 0.1]],
            calls: AtomicUsize::new(0),
        };
        let r = run_lookahead(&oracle, &[false, true], cfg(3, 16))
            .await
            .unwrap();
        assert_eq!(r.best_action, 1);
        assert!(r.principal_variation.iter().all(|s| s.action == 1));
        assert_eq!(r.root_visits[0], 0);
        assert!(run_lookahead(&oracle, &[false, false], cfg(3, 16))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn oracle_failure_aborts_the_search() {
        struct Failing;
        impl PriorOracle for Failing {
            fn priors<'a>(&'a self, _: &'a [usize]) -> PriorFuture<'a> {
                Box::pin(async { Err(BridgeError::Transport("down".into())) })
            }
        }
        let err = run_lookahead(&Failing, &[true, true], cfg(2, 4))
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::Transport(_)));
    }
}
