//! Two-player normal-form regret matching (one information set per player).
//! Average strategies approach Nash in zero-sum games; general-sum no-regret
//! learning does not in general produce a Nash equilibrium of the marginals.
use crate::PlannerError;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct NormalFormGame {
    row: Vec<Vec<f64>>,
    column: Vec<Vec<f64>>,
    scale: f64,
    zero_sum: bool,
}

impl NormalFormGame {
    /// Both matrices use [row action][column action]. Finite utilities are
    /// scaled by a common positive constant for numerical stability.
    pub fn new(row: Vec<Vec<f64>>, column: Vec<Vec<f64>>) -> Result<Self, PlannerError> {
        let cols = row.first().map_or(0, Vec::len);
        if row.is_empty()
            || cols == 0
            || row.len() != column.len()
            || row.iter().chain(&column).any(|r| r.len() != cols)
        {
            return Err(PlannerError::InvalidInput(
                "nonempty rectangular matching payoff matrices required".into(),
            ));
        }
        if row.iter().chain(&column).flatten().any(|v| !v.is_finite()) {
            return Err(PlannerError::InvalidInput("payoffs must be finite".into()));
        }
        let scale = row
            .iter()
            .chain(&column)
            .flatten()
            .map(|v| v.abs())
            .fold(0.0, f64::max)
            .max(1.0);
        let zero_sum = row
            .iter()
            .flatten()
            .zip(column.iter().flatten())
            .all(|(a, b)| *a == -*b);
        let normalize = |matrix: Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>, PlannerError> {
            matrix
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .map(|v| {
                            let normalized = v / scale;
                            if v != 0.0 && normalized == 0.0 {
                                Err(PlannerError::InvalidInput(
                                    "payoff normalization underflow".into(),
                                ))
                            } else {
                                Ok(normalized)
                            }
                        })
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            row: normalize(row)?,
            column: normalize(column)?,
            scale,
            zero_sum,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GameSolution {
    pub row_strategy: Vec<f64>,
    pub column_strategy: Vec<f64>,
    /// Best-response duality gap in ORIGINAL payoff units, only for zero-sum.
    /// An iteration budget is not a convergence certificate: inspect this gap.
    pub zero_sum_gap: Option<f64>,
    pub iterations: usize,
}

impl GameSolution {
    pub fn sample_row(&self, rng: &mut impl Rng) -> usize {
        sample(&self.row_strategy, rng)
    }
    pub fn sample_column(&self, rng: &mut impl Rng) -> usize {
        sample(&self.column_strategy, rng)
    }
}

pub(crate) fn sample(probabilities: &[f64], rng: &mut impl Rng) -> usize {
    let u = rng.gen::<f64>();
    let mut cumulative = 0.0;
    for (i, p) in probabilities.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    // Floating-point summation residue must never select a zero-mass branch.
    probabilities
        .iter()
        .rposition(|p| *p > 0.0)
        .expect("validated probability distribution")
}

fn regret_strategy(regrets: &[f64]) -> Vec<f64> {
    let sum: f64 = regrets.iter().map(|r| r.max(0.0)).sum();
    regrets
        .iter()
        .map(|r| {
            if sum > 0.0 {
                r.max(0.0) / sum
            } else {
                1.0 / regrets.len() as f64
            }
        })
        .collect()
}

pub(crate) fn solve(
    game: &NormalFormGame,
    iterations: usize,
) -> Result<GameSolution, PlannerError> {
    solve_until(game, iterations, &crate::engine::SearchBudget::default())
}

pub(crate) fn solve_until(
    game: &NormalFormGame,
    iterations: usize,
    budget: &crate::engine::SearchBudget<'_>,
) -> Result<GameSolution, PlannerError> {
    budget.check()?;
    if iterations == 0 {
        return Err(PlannerError::InvalidInput(
            "regret matching needs at least one iteration".into(),
        ));
    }
    let rows = game.row.len();
    let cols = game.row[0].len();
    let mut row_regrets = vec![0.0; rows];
    let mut col_regrets = vec![0.0; cols];
    let mut row_sum = vec![0.0; rows];
    let mut col_sum = vec![0.0; cols];
    for _ in 0..iterations {
        budget.check()?;
        // Simultaneous updates against the same pair of current strategies.
        let p = regret_strategy(&row_regrets);
        let q = regret_strategy(&col_regrets);
        let row_values: Vec<f64> = game
            .row
            .iter()
            .map(|r| r.iter().zip(&q).map(|(a, b)| a * b).sum())
            .collect();
        let col_values: Vec<f64> = (0..cols)
            .map(|j| (0..rows).map(|i| p[i] * game.column[i][j]).sum())
            .collect();
        let row_value: f64 = p.iter().zip(&row_values).map(|(a, b)| a * b).sum();
        let col_value: f64 = q.iter().zip(&col_values).map(|(a, b)| a * b).sum();
        for i in 0..rows {
            row_regrets[i] += row_values[i] - row_value;
            row_sum[i] += p[i];
        }
        for j in 0..cols {
            col_regrets[j] += col_values[j] - col_value;
            col_sum[j] += q[j];
        }
    }
    let normalize = |mut values: Vec<f64>| {
        let sum: f64 = values.iter().sum();
        for v in &mut values {
            *v /= sum;
        }
        values
    };
    let p = normalize(row_sum);
    let q = normalize(col_sum);
    let zero_sum_gap = if game.zero_sum {
        let upper = game
            .row
            .iter()
            .map(|r| r.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max);
        let lower = (0..cols)
            .map(|j| (0..rows).map(|i| p[i] * game.row[i][j]).sum::<f64>())
            .fold(f64::INFINITY, f64::min);
        let gap = (upper - lower).max(0.0) * game.scale;
        if !gap.is_finite() {
            return Err(PlannerError::DivergentState(
                "equilibrium gap overflow".into(),
            ));
        }
        Some(gap)
    } else {
        None
    };
    Ok(GameSolution {
        row_strategy: p,
        column_strategy: q,
        zero_sum_gap,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_game_updates_both_players_and_preserves_payoff_units() {
        let row = vec![vec![3.0, 1.0, 2.0], vec![0.0, -2.0, -1.0]];
        let column = row.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
        let game = NormalFormGame::new(row, column).unwrap();
        let solution = solve(&game, 10_000).unwrap();
        assert!(solution.row_strategy[0] > 0.999);
        assert!(solution.column_strategy[1] > 0.999);
        assert!(solution.zero_sum_gap.unwrap() < 0.001);

        // One round is not an equilibrium certificate. Gap remains expressed
        // in original units even when normalization scales very large payoffs.
        for scale in [1.0, 1e100] {
            let row = vec![vec![2.0 * scale, -scale], vec![-scale, scale]];
            let column = row.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
            let game = NormalFormGame::new(row, column).unwrap();
            let solution = solve(&game, 1).unwrap();
            assert!((solution.zero_sum_gap.unwrap() / scale - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn normalization_and_gap_overflow_fail_closed() {
        assert!(NormalFormGame::new(
            vec![vec![f64::MAX, f64::from_bits(1)]],
            vec![vec![0.0, 0.0]],
        )
        .is_err());
        let game = NormalFormGame::new(
            vec![vec![f64::MAX, f64::MAX], vec![-f64::MAX, -f64::MAX]],
            vec![vec![-f64::MAX, -f64::MAX], vec![f64::MAX, f64::MAX]],
        )
        .unwrap();
        // Even near the numeric limit, finite representable gaps are retained.
        assert_eq!(solve(&game, 1).unwrap().zero_sum_gap, Some(f64::MAX));
        let row = vec![
            vec![f64::MAX; 3],
            vec![-f64::MAX, f64::MAX, f64::MAX],
            vec![-f64::MAX, f64::MAX, f64::MAX],
        ];
        let column = row.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
        let game = NormalFormGame::new(row, column).unwrap();
        assert!(matches!(
            solve(&game, 1),
            Err(PlannerError::DivergentState(_))
        ));
    }
}
