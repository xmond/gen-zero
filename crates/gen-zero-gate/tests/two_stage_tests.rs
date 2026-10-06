//! Two-stage dual-track gateway: fast pass, stage 2 routing, flips and
//! fail-closed paths. The verifier here is a test double that returns
//! projections with exact, known cosines; the real Qwen verifier lives in
//! `gen-zero-model` and is exercised by its own tests and the CLI.

use gen_zero_gate::{
    CausalVerifier, GateError, TeacherProjectionPair, TeacherWeights, TriTeacherProjections,
    TwoStageConfig, TwoStageDualTrackGateway, VerifierType,
};
use std::cell::{Cell, RefCell};

/// Unit vectors at a given cosine: z1 = e0, z2 = cos*e0 + sin*e1.
fn pair_at(cos: f32) -> TeacherProjectionPair {
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    TeacherProjectionPair {
        z1: vec![3.0, 0.0, 0.0],
        z2: vec![2.0 * cos, 2.0 * sin, 0.0],
    }
}

enum Behaviour {
    Cosines(f32, f32, f32),
    Raw(TriTeacherProjections),
    Fail,
}

struct FakeVerifier {
    behaviour: Behaviour,
    calls: Cell<usize>,
    last_pair: RefCell<Option<(String, String)>>,
}

impl FakeVerifier {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            calls: Cell::new(0),
            last_pair: RefCell::new(None),
        }
    }

    fn uniform(cos: f32) -> Self {
        Self::new(Behaviour::Cosines(cos, cos, cos))
    }
}

impl CausalVerifier for FakeVerifier {
    fn verifier_id(&self) -> String {
        "fake-tri-teacher".into()
    }

    fn project_pair(&self, s1: &str, s2: &str) -> Result<TriTeacherProjections, GateError> {
        self.calls.set(self.calls.get() + 1);
        *self.last_pair.borrow_mut() = Some((s1.to_string(), s2.to_string()));
        match &self.behaviour {
            Behaviour::Cosines(a, b, c) => Ok(TriTeacherProjections {
                proj_405b: pair_at(*a),
                proj_q72b: pair_at(*b),
                proj_llama70b: pair_at(*c),
            }),
            Behaviour::Raw(p) => Ok(p.clone()),
            Behaviour::Fail => Err(GateError::VerifierFailure("backbone OOM".into())),
        }
    }
}

const CONTEXT: &str = "Marie Curie won the Nobel Prize in Physics in 1903.";
const QUESTION: &str = "When did Marie Curie win the Nobel Prize in Physics?";

fn gateway() -> TwoStageDualTrackGateway {
    TwoStageDualTrackGateway::new(TwoStageConfig::default()).expect("default config is valid")
}

/// Scores with `null - best == diff`.
fn scores(diff: f32) -> (f32, f32) {
    (5.0, 5.0 + diff)
}

#[test]
fn default_config_matches_spec() {
    let c = TwoStageConfig::default();
    assert_eq!(c.ambiguity_low, -1.5);
    assert_eq!(c.ambiguity_high, 0.5);
    assert_eq!(c.tri_teacher_threshold, 0.91);
    assert_eq!(c.verifier_type, VerifierType::TriTeacher);
    assert_eq!(c.logistic_slope, 15.0);
    assert_eq!(
        c.teacher_weights,
        TeacherWeights {
            w_405b: 0.5,
            w_q72b: 0.3,
            w_llama70b: 0.2
        }
    );
}

#[test]
fn fast_pass_confident_answer_skips_stage2() {
    let v = FakeVerifier::uniform(0.0);
    let (best, null) = scores(-3.0);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.fast_pass);
    assert!(!ev.stage2_triggered);
    assert!(!ev.decision_flipped);
    assert_eq!(ev.score_diff, -3.0);
    assert!(ev.is_answerable);
    assert_eq!(ev.final_answer.as_deref(), Some("1903"));
    assert_eq!(ev.tri_sim, None);
    assert_eq!(ev.p_same_meaning, None);
    assert_eq!(v.calls.get(), 0, "stage 2 must not run outside the band");
}

#[test]
fn fast_pass_confident_no_answer_skips_stage2() {
    let v = FakeVerifier::uniform(1.0);
    let (best, null) = scores(2.0);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.fast_pass);
    assert!(!ev.stage2_triggered);
    assert!(!ev.is_answerable);
    assert_eq!(ev.final_answer, None);
    assert_eq!(v.calls.get(), 0);
}

#[test]
fn fast_pass_no_answer_accepts_empty_candidate() {
    let v = FakeVerifier::uniform(1.0);
    let (best, null) = scores(2.0);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "", best, null, &v)
        .unwrap();
    assert!(!ev.is_answerable);
    assert_eq!(v.calls.get(), 0);
}

#[test]
fn ambiguity_band_triggers_stage2_with_spec_sentence_pair() {
    let v = FakeVerifier::uniform(0.95);
    let (best, null) = scores(-0.5);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, " 1903 ", best, null, &v)
        .unwrap();
    assert!(!ev.fast_pass);
    assert!(ev.stage2_triggered);
    assert_eq!(v.calls.get(), 1);
    let (s1, s2) = v.last_pair.borrow().clone().unwrap();
    assert_eq!(s1, CONTEXT);
    assert_eq!(s2, format!("{QUESTION} 1903"));
    assert_eq!(ev.sentence_1.as_deref(), Some(CONTEXT));
    assert_eq!(ev.verifier_id.as_deref(), Some("fake-tri-teacher"));
}

#[test]
fn band_edges_are_inclusive() {
    let g = gateway();
    for diff in [-1.5f32, 0.5] {
        let v = FakeVerifier::uniform(0.95);
        let (best, null) = scores(diff);
        let ev = g
            .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
            .unwrap();
        assert!(
            ev.stage2_triggered,
            "score_diff {diff} must route to stage 2"
        );
        assert_eq!(v.calls.get(), 1);
    }
    for diff in [-1.5001f32, 0.5001] {
        let v = FakeVerifier::uniform(0.95);
        let (best, null) = scores(diff);
        let ev = g
            .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
            .unwrap();
        assert!(ev.fast_pass, "score_diff {diff} must fast-pass");
        assert_eq!(v.calls.get(), 0);
    }
}

#[test]
fn counterfactual_candidate_is_flipped_to_no_answer() {
    // tri_sim = 0.5*0.80 + 0.3*0.85 + 0.2*0.70 = 0.795 < 0.91
    let v = FakeVerifier::new(Behaviour::Cosines(0.80, 0.85, 0.70));
    let (best, null) = scores(-0.5);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1911", best, null, &v)
        .unwrap();
    assert!(ev.stage2_triggered);
    assert!(!ev.is_answerable);
    assert_eq!(ev.final_answer, None);
    assert!(
        ev.decision_flipped,
        "stage 1 said answerable (diff<0), stage 2 rejects"
    );
    let tri = ev.tri_sim.unwrap();
    assert!((tri - 0.795).abs() < 1e-5, "tri_sim={tri}");
    assert!((ev.sim_405b.unwrap() - 0.80).abs() < 1e-5);
    assert!((ev.sim_q72b.unwrap() - 0.85).abs() < 1e-5);
    assert!((ev.sim_llama70b.unwrap() - 0.70).abs() < 1e-5);
    let p = ev.p_same_meaning.unwrap();
    let expected = 1.0 / (1.0 + (-15.0f32 * (0.795 - 0.91)).exp());
    assert!((p - expected).abs() < 1e-5, "p={p} expected={expected}");
    assert!(p < 0.5);
}

#[test]
fn supported_candidate_is_confirmed() {
    // tri_sim = 0.5*0.97 + 0.3*0.93 + 0.2*0.90 = 0.944 >= 0.91
    let v = FakeVerifier::new(Behaviour::Cosines(0.97, 0.93, 0.90));
    let (best, null) = scores(-0.5);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.stage2_triggered);
    assert!(ev.is_answerable);
    assert_eq!(ev.final_answer.as_deref(), Some("1903"));
    assert!(!ev.decision_flipped);
    assert!((ev.tri_sim.unwrap() - 0.944).abs() < 1e-5);
    assert!(ev.p_same_meaning.unwrap() > 0.5);
}

#[test]
fn stage2_reject_on_stage1_no_answer_side_is_not_a_flip() {
    let v = FakeVerifier::uniform(0.5);
    let (best, null) = scores(0.3);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1911", best, null, &v)
        .unwrap();
    assert!(ev.stage2_triggered);
    assert!(!ev.is_answerable);
    assert!(
        !ev.decision_flipped,
        "stage 1 already said no answer (diff>0)"
    );
}

#[test]
fn stage2_accept_on_stage1_no_answer_side_is_a_flip() {
    let v = FakeVerifier::uniform(0.99);
    let (best, null) = scores(0.3);
    let ev = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.is_answerable);
    assert!(ev.decision_flipped);
}

#[test]
fn tri_sim_exactly_at_threshold_is_accepted() {
    // All three cosines equal the threshold, so tri_sim == threshold (up to
    // rounding of the weighted sum). Use a threshold exactly representable
    // by the weighted sum: 0.5.
    let cfg = TwoStageConfig {
        tri_teacher_threshold: 0.5,
        ..TwoStageConfig::default()
    };
    let g = TwoStageDualTrackGateway::new(cfg).unwrap();
    let v = FakeVerifier::uniform(0.5);
    let (best, null) = scores(-0.5);
    let ev = g
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    let tri = ev.tri_sim.unwrap();
    assert_eq!(ev.is_answerable, tri >= 0.5, "tri_sim={tri}");
    assert!((ev.p_same_meaning.unwrap() - 0.5).abs() < 1e-3);
}

#[test]
fn verifier_error_fails_closed() {
    let v = FakeVerifier::new(Behaviour::Fail);
    let (best, null) = scores(-0.5);
    let err = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap_err();
    assert!(matches!(err, GateError::VerifierFailure(_)), "{err:?}");
}

#[test]
fn nan_projection_fails_closed() {
    let mut p = pair_at(0.9);
    p.z2[1] = f32::NAN;
    let v = FakeVerifier::new(Behaviour::Raw(TriTeacherProjections {
        proj_405b: pair_at(0.9),
        proj_q72b: p,
        proj_llama70b: pair_at(0.9),
    }));
    let (best, null) = scores(-0.5);
    let err = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap_err();
    assert!(
        matches!(err, GateError::NumericalFault(ref m) if m.contains("q72b")),
        "{err:?}"
    );
}

#[test]
fn zero_norm_projection_fails_closed() {
    let v = FakeVerifier::new(Behaviour::Raw(TriTeacherProjections {
        proj_405b: TeacherProjectionPair {
            z1: vec![0.0; 3],
            z2: vec![1.0, 0.0, 0.0],
        },
        proj_q72b: pair_at(0.9),
        proj_llama70b: pair_at(0.9),
    }));
    let (best, null) = scores(-0.5);
    let err = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap_err();
    assert!(
        matches!(err, GateError::NumericalFault(ref m) if m.contains("zero norm")),
        "{err:?}"
    );
}

#[test]
fn mismatched_or_empty_projection_fails_closed() {
    let (best, null) = scores(-0.5);
    for bad in [
        TeacherProjectionPair {
            z1: vec![1.0, 0.0],
            z2: vec![1.0, 0.0, 0.0],
        },
        TeacherProjectionPair {
            z1: vec![],
            z2: vec![],
        },
    ] {
        let v = FakeVerifier::new(Behaviour::Raw(TriTeacherProjections {
            proj_405b: pair_at(0.9),
            proj_q72b: pair_at(0.9),
            proj_llama70b: bad,
        }));
        let err = gateway()
            .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
            .unwrap_err();
        assert!(
            matches!(err, GateError::NumericalFault(ref m) if m.contains("llama70b")),
            "{err:?}"
        );
    }
}

#[test]
fn non_finite_stage1_scores_are_rejected_before_routing() {
    let g = gateway();
    for (best, null) in [
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, 1.0),
        (1.0, f32::NEG_INFINITY),
        (-f32::MAX, f32::MAX),
    ] {
        let v = FakeVerifier::uniform(0.99);
        let err = g
            .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
            .unwrap_err();
        assert!(
            matches!(err, GateError::InvalidInput(_)),
            "({best}, {null}) -> {err:?}"
        );
        assert_eq!(v.calls.get(), 0);
    }
}

#[test]
fn empty_candidate_in_band_fails_closed() {
    let v = FakeVerifier::uniform(0.99);
    let (best, null) = scores(-0.5);
    let err = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "   ", best, null, &v)
        .unwrap_err();
    assert!(
        matches!(err, GateError::InvalidInput(ref m) if m.contains("ambiguity band")),
        "{err:?}"
    );
    assert_eq!(v.calls.get(), 0);
}

#[test]
fn empty_candidate_on_confident_answer_fails_closed() {
    let v = FakeVerifier::uniform(0.99);
    let (best, null) = scores(-3.0);
    let err = gateway()
        .decide_with_scores(CONTEXT, QUESTION, "", best, null, &v)
        .unwrap_err();
    assert!(matches!(err, GateError::InvalidInput(_)), "{err:?}");
}

#[test]
fn empty_context_in_band_fails_closed() {
    let v = FakeVerifier::uniform(0.99);
    let (best, null) = scores(-0.5);
    let err = gateway()
        .decide_with_scores(" ", QUESTION, "1903", best, null, &v)
        .unwrap_err();
    assert!(matches!(err, GateError::InvalidInput(_)), "{err:?}");
    assert_eq!(v.calls.get(), 0);
}

#[test]
fn invalid_configs_are_rejected() {
    let base = TwoStageConfig::default();
    let cases = [
        TwoStageConfig {
            ambiguity_low: 1.0,
            ambiguity_high: 0.0,
            ..base.clone()
        },
        TwoStageConfig {
            ambiguity_low: f32::NAN,
            ..base.clone()
        },
        TwoStageConfig {
            tri_teacher_threshold: 1.0,
            ..base.clone()
        },
        TwoStageConfig {
            tri_teacher_threshold: f32::NAN,
            ..base.clone()
        },
        TwoStageConfig {
            logistic_slope: 0.0,
            ..base.clone()
        },
        TwoStageConfig {
            logistic_slope: f32::INFINITY,
            ..base.clone()
        },
        TwoStageConfig {
            teacher_weights: TeacherWeights {
                w_405b: 0.5,
                w_q72b: 0.5,
                w_llama70b: 0.5,
            },
            ..base.clone()
        },
        TwoStageConfig {
            teacher_weights: TeacherWeights {
                w_405b: 1.2,
                w_q72b: -0.2,
                w_llama70b: 0.0,
            },
            ..base.clone()
        },
    ];
    for cfg in cases {
        let err = TwoStageDualTrackGateway::new(cfg.clone()).unwrap_err();
        assert!(
            matches!(err, GateError::InvalidConfig(_)),
            "{cfg:?} -> {err:?}"
        );
    }
}

#[test]
fn unknown_verifier_type_is_rejected() {
    assert_eq!(
        "tri_teacher".parse::<VerifierType>().unwrap(),
        VerifierType::TriTeacher
    );
    assert_eq!(
        serde_json::to_string(&VerifierType::TriTeacher).unwrap(),
        "\"tri_teacher\""
    );
    let err = "single_teacher".parse::<VerifierType>().unwrap_err();
    assert!(matches!(err, GateError::InvalidConfig(_)));
}
