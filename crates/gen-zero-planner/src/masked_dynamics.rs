//! State-dependent preconditions shared by search and direct model execution.
use gen_zero_core::{ActionId, CoreError, FullLatent, SafetyEstimate, WorldModelDynamics};
use std::sync::Arc;

/// A deterministic predicate over action IDs (not positions in the frame).
/// Return a unique subset of the supplied candidates. Legality must depend only
/// on state and action, not on candidate ordering or the presence of other IDs.
pub trait StateActionMask: Send + Sync {
    fn allowed_actions(&self, state: &[f32], candidate_actions: &[u32]) -> Vec<u32>;
}

/// Production state-action mask that enforces finite numerical state representation (no NaN or Inf).
#[derive(Clone, Copy, Debug, Default)]
pub struct FiniteStateActionMask;

impl StateActionMask for FiniteStateActionMask {
    fn allowed_actions(&self, state: &[f32], candidate_actions: &[u32]) -> Vec<u32> {
        if state.iter().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        candidate_actions.to_vec()
    }
}


pub struct MaskedDynamics<D: WorldModelDynamics> {
    inner: D,
    mask: Arc<dyn StateActionMask>,
}

impl<D: WorldModelDynamics> MaskedDynamics<D> {
    pub fn new(inner: D, mask: Arc<dyn StateActionMask>) -> Self {
        Self { inner, mask }
    }
}

impl<D: WorldModelDynamics<Error = CoreError>> WorldModelDynamics for MaskedDynamics<D> {
    type Error = CoreError;

    fn allowed_actions(
        &self,
        state: &FullLatent,
        candidates: &[ActionId],
    ) -> Result<Vec<ActionId>, CoreError> {
        let candidates = self.inner.allowed_actions(state, candidates)?;
        let ids: Vec<_> = candidates.iter().map(|a| a.0).collect();
        let allowed = self.mask.allowed_actions(state.as_slice(), &ids);
        for (i, id) in allowed.iter().enumerate() {
            if !ids.contains(id) || allowed[..i].contains(id) {
                return Err(CoreError::WorldModel(format!(
                    "invalid action mask output: {id}"
                )));
            }
        }
        Ok(candidates
            .into_iter()
            .filter(|a| allowed.contains(&a.0))
            .collect())
    }

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        if self.allowed_actions(state, &[action])?.is_empty() {
            return Err(CoreError::WorldModel(format!(
                "action {} rejected by state mask",
                action.0
            )));
        }
        self.inner.step(state, action)
    }

    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        if [actions.len(), next_states.len(), rewards.len(), dones.len()]
            .iter()
            .any(|&n| n != states.len())
        {
            return Err(CoreError::WorldModel(
                "masked dynamics batch length mismatch".into(),
            ));
        }
        // Validate the complete batch before invoking the inner kernel.
        for (state, &action) in states.iter().zip(actions) {
            if self.allowed_actions(state, &[action])?.is_empty() {
                return Err(CoreError::WorldModel(format!(
                    "action {} rejected by state mask",
                    action.0
                )));
            }
        }
        self.inner
            .step_batch(states, actions, next_states, rewards, dones)
    }

    fn safety_estimate(
        &self,
        state: &FullLatent,
        reward: f32,
        done: bool,
    ) -> Option<SafetyEstimate> {
        self.inner.safety_estimate(state, reward, done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AStarEngine, AStarGoal, MctsEngine, MpcCemEngine, PlannerError, PlanningEngine};
    use gen_zero_core::LocalActionFrame;
    use gen_zero_gate::PolicyGate;
    use std::sync::Mutex;

    // Analytic two-step system: state[0] is elapsed time, action 0 then 1
    // satisfies its preconditions. The unmasked kernel deliberately accepts both
    // actions so the recorded calls expose illegal expansions without masking them.
    #[derive(Default)]
    struct Dynamics {
        calls: Mutex<Vec<(usize, ActionId)>>,
    }
    impl WorldModelDynamics for Dynamics {
        type Error = CoreError;
        fn step(&self, s: &FullLatent, a: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
            self.calls
                .lock()
                .unwrap()
                .push((s.as_slice()[0] as usize, a));
            let mut next = s.clone();
            next.as_mut_slice()[0] += 1.0;
            Ok((next, 1.0, false))
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
    struct Mask {
        empty: bool,
    }
    impl StateActionMask for Mask {
        fn allowed_actions(&self, state: &[f32], candidates: &[u32]) -> Vec<u32> {
            candidates
                .iter()
                .copied()
                .filter(|&a| !self.empty && state[0] < 2.0 && a as f32 == state[0])
                .collect()
        }
    }
    struct All;
    impl StateActionMask for All {
        fn allowed_actions(&self, _: &[f32], candidates: &[u32]) -> Vec<u32> {
            candidates.to_vec()
        }
    }
    fn engines() -> Vec<Box<dyn PlanningEngine>> {
        vec![
            Box::new(MctsEngine::default()),
            Box::new(AStarEngine {
                goal: Some(AStarGoal::Predicate(|s| s.as_slice()[0] == 2.0)),
                ..Default::default()
            }),
            Box::new(MpcCemEngine::default()),
        ]
    }
    #[test]
    fn all_searches_mask_successors_and_stop_at_dead_ends() {
        let acts = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["first", "second"], &acts).unwrap();
        for engine in engines() {
            let model = MaskedDynamics::new(Dynamics::default(), Arc::new(Mask { empty: false }));
            let report = engine
                .plan_report(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
                .unwrap();
            assert_eq!(report.action, ActionId(0), "{}", engine.name());
            if engine.name() != "AStarEngine" {
                assert!(report.has_dead_end, "{}", engine.name());
            }
            let calls = model.inner.calls.lock().unwrap();
            assert!(
                calls.contains(&(1, ActionId(1))),
                "{} never reached successor",
                engine.name()
            );
            assert!(
                calls
                    .iter()
                    .all(|&(depth, action)| depth < 2 && action.0 as usize == depth),
                "{}: {calls:?}",
                engine.name()
            );
        }
    }
    #[test]
    fn empty_root_fails_closed_without_any_transition() {
        let acts = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &acts).unwrap();
        for engine in engines() {
            let model = MaskedDynamics::new(Dynamics::default(), Arc::new(Mask { empty: true }));
            assert_eq!(
                engine
                    .plan(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
                    .unwrap_err(),
                PlannerError::NoFeasibleAction
            );
            assert!(model.inner.calls.lock().unwrap().is_empty());
        }
    }
    #[test]
    fn unmasked_and_allow_all_have_identical_results_and_transition_traces() {
        let acts = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &acts).unwrap();
        for engine in engines() {
            let plain = Dynamics::default();
            let decorated = MaskedDynamics::new(Dynamics::default(), Arc::new(All));
            let expected = engine
                .plan(&FullLatent::zeros(), &frame, &plain, &PolicyGate::default())
                .unwrap();
            let actual = engine
                .plan(
                    &FullLatent::zeros(),
                    &frame,
                    &decorated,
                    &PolicyGate::default(),
                )
                .unwrap();
            assert_eq!(actual, expected, "{}", engine.name());
            assert_eq!(
                *plain.calls.lock().unwrap(),
                *decorated.inner.calls.lock().unwrap()
            );
        }
    }
    #[test]
    fn direct_and_batch_calls_cannot_bypass_mask() {
        let model = MaskedDynamics::new(Dynamics::default(), Arc::new(Mask { empty: false }));
        assert!(model.step(&FullLatent::zeros(), ActionId(1)).is_err());
        let states = [FullLatent::zeros(), FullLatent::zeros()];
        assert!(model
            .step_batch(
                &states,
                &[ActionId(0), ActionId(1)],
                &mut states.clone(),
                &mut [0.0; 2],
                &mut [false; 2]
            )
            .is_err());
        assert!(model.inner.calls.lock().unwrap().is_empty());
        assert!(model
            .step_batch(&states, &[], &mut [], &mut [], &mut [])
            .is_err());
    }
    #[test]
    fn malformed_masks_fail_closed() {
        struct Invalid(Vec<u32>);
        impl StateActionMask for Invalid {
            fn allowed_actions(&self, _: &[f32], _: &[u32]) -> Vec<u32> {
                self.0.clone()
            }
        }
        for output in [vec![999], vec![0, 0]] {
            let model = MaskedDynamics::new(Dynamics::default(), Arc::new(Invalid(output)));
            assert!(model
                .allowed_actions(&FullLatent::zeros(), &[ActionId(0)])
                .is_err());
            assert!(model.inner.calls.lock().unwrap().is_empty());
        }
    }
    #[test]
    fn astar_dead_end_is_not_a_goal() {
        let model = MaskedDynamics::new(Dynamics::default(), Arc::new(Mask { empty: false }));
        let engine = AStarEngine {
            goal: Some(AStarGoal::Predicate(|s| s.as_slice()[0] == 3.0)),
            ..Default::default()
        };
        let acts = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &acts).unwrap();
        assert_eq!(
            engine
                .plan(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
                .unwrap_err(),
            PlannerError::SearchGoalUnreachable
        );
    }
}
