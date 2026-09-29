use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::{CemDistribution, MpcCemEngine, PlannerConfig};

struct SequenceModel {
    divergent: bool,
}
impl WorldModelDynamics for SequenceModel {
    type Error = CoreError;
    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let t = state.values[0] as usize;
        let mut next = state.clone();
        next.values[0] += 1.0;
        if action.0 as usize != t {
            next.values[1] = 1.0;
        }
        if self.divergent && t == 1 && action == ActionId(3) {
            next.values[2] = f32::NAN;
            return Ok((next, 1e6, false));
        }
        let reward = if t == 2 && next.values[1] == 0.0 {
            1.0
        } else {
            0.0
        };
        Ok((next, reward, t == 2))
    }
    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        for i in 0..states.len() {
            (next[i], rewards[i], dones[i]) = self.step(&states[i], actions[i])?;
        }
        Ok(())
    }
}
fn engine() -> MpcCemEngine {
    MpcCemEngine {
        num_samples: 2048,
        num_elites: 16,
        horizon: 3,
        num_iterations: 8,
        gamma: 0.5,
    }
}

#[test]
fn learns_start_exec_finish_with_independent_time_distributions() {
    let actions = [ActionId(0), ActionId(1), ActionId(2)];
    let frame = LocalActionFrame::new(&["START", "EXEC", "FINISH"], &actions).unwrap();
    let result = engine()
        .optimize(
            &FullLatent::zeros(),
            &frame,
            &SequenceModel { divergent: false },
            &PolicyGate::default(),
        )
        .unwrap();
    assert_eq!(result.actions, actions);
    assert_eq!(result.score, 0.25); // gamma^2, not a fixed gamma multiplier
    let CemDistribution::Categorical { probabilities } = result.distribution else {
        panic!()
    };
    for (t, row) in probabilities.iter().enumerate() {
        assert!(row[t] > 0.999, "{probabilities:?}");
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    // Executable shared-distribution baseline: pooling perfect elites still
    // produces [1/3,1/3,1/3] at EVERY time. It can sample the sequence but cannot
    // represent it with certainty. AM-GM bounds p(START)p(EXEC)p(FINISH) <= 1/27.
    let mut shared = [0.0f32; 3];
    for action in result.actions {
        shared[action.0 as usize] += 1.0 / 3.0;
    }
    assert!(shared.iter().product::<f32>() <= 1.0 / 27.0 + 1e-7);
    assert!(
        probabilities
            .iter()
            .enumerate()
            .map(|(t, row)| row[t])
            .product::<f32>()
            > 0.999
    );
}

#[test]
fn excludes_divergent_and_gate_blocked_paths_and_handles_16_actions() {
    let actions: Vec<_> = (0..16).map(ActionId).collect();
    let names: Vec<_> = (0..16).map(|_| "action").collect();
    let frame = LocalActionFrame::new(&names, &actions).unwrap();
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(1),
        "blocked",
        ActionId(4),
    ));
    let mut cem = engine();
    cem.num_samples = 16384;
    cem.num_elites = 1;
    assert!(matches!(
        cem.optimize(
            &FullLatent::zeros(),
            &frame,
            &SequenceModel { divergent: true },
            &gate
        ),
        Err(gen_zero_planner::PlannerError::DivergentState(_))
    ));
    let result = cem
        .optimize(
            &FullLatent::zeros(),
            &frame,
            &SequenceModel { divergent: false },
            &gate,
        )
        .unwrap();
    assert!(!result.actions.contains(&ActionId(4)));
    let CemDistribution::Categorical { probabilities } = result.distribution else {
        panic!()
    };
    assert!(probabilities.iter().all(|row| row[4] < 0.001));
}

#[test]
fn gaussian_shape_and_discount_are_validated() {
    assert!(
        CemDistribution::diagonal_gaussian(vec![vec![0.0; 2]; 3], vec![vec![1.0; 2]; 3]).is_ok()
    );
    for variance in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(CemDistribution::diagonal_gaussian(vec![vec![0.0]], vec![vec![variance]]).is_err());
    }
    assert!(CemDistribution::diagonal_gaussian(vec![], vec![]).is_err());
    assert!(CemDistribution::diagonal_gaussian(vec![vec![0.0]], vec![vec![1.0, 1.0]]).is_err());
    for gamma in [-0.1, 1.1, f32::NAN] {
        assert!(PlannerConfig {
            cem_gamma: gamma,
            ..Default::default()
        }
        .validate()
        .is_err());
        let frame = LocalActionFrame::new(&["START"], &[ActionId(0)]).unwrap();
        assert!(MpcCemEngine {
            gamma,
            ..Default::default()
        }
        .optimize(
            &FullLatent::zeros(),
            &frame,
            &SequenceModel { divergent: false },
            &PolicyGate::default()
        )
        .is_err());
    }
}
