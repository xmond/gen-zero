//! Refusal trace sink hook: stage 2 / flip events are handed to an optional
//! durable sink, fail-closed on a sink error, and never invoked on the
//! fast-pass path. The verifier here is the same test double style as
//! `two_stage_tests.rs`.

use gen_zero_gate::{
    CausalVerifier, GateError, RefusalTraceEvent, RefusalTraceSink, TeacherProjectionPair,
    TriTeacherProjections, TwoStageConfig, TwoStageDualTrackGateway,
};
use std::cell::Cell;
use std::sync::{Arc, Mutex};

/// Unit vectors at a given cosine: z1 = e0, z2 = cos*e0 + sin*e1.
fn pair_at(cos: f32) -> TeacherProjectionPair {
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    TeacherProjectionPair {
        z1: vec![3.0, 0.0, 0.0],
        z2: vec![2.0 * cos, 2.0 * sin, 0.0],
    }
}

struct FakeVerifier {
    cos: (f32, f32, f32),
    calls: Cell<usize>,
}

impl FakeVerifier {
    fn uniform(cos: f32) -> Self {
        Self {
            cos: (cos, cos, cos),
            calls: Cell::new(0),
        }
    }
}

impl CausalVerifier for FakeVerifier {
    fn verifier_id(&self) -> String {
        "fake-tri-teacher".into()
    }

    fn project_pair(&self, _s1: &str, _s2: &str) -> Result<TriTeacherProjections, GateError> {
        self.calls.set(self.calls.get() + 1);
        let (a, b, c) = self.cos;
        Ok(TriTeacherProjections {
            proj_405b: pair_at(a),
            proj_q72b: pair_at(b),
            proj_llama70b: pair_at(c),
        })
    }
}

const CONTEXT: &str = "Marie Curie won the Nobel Prize in Physics in 1903.";
const QUESTION: &str = "When did Marie Curie win the Nobel Prize in Physics?";

/// Scores with `null - best == diff`.
fn scores(diff: f32) -> (f32, f32) {
    (5.0, 5.0 + diff)
}

/// Records every event it is given; never fails.
#[derive(Default)]
struct RecordingSink {
    events: Mutex<Vec<RefusalTraceEvent>>,
}

impl RefusalTraceSink for RecordingSink {
    fn record_refusal(&self, event: RefusalTraceEvent) -> Result<(), String> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }
}

/// Always fails, simulating a durable-store write error.
struct FailingSink {
    message: String,
}

impl RefusalTraceSink for FailingSink {
    fn record_refusal(&self, _event: RefusalTraceEvent) -> Result<(), String> {
        Err(self.message.clone())
    }
}

#[test]
fn stage2_event_is_recorded_with_matching_fields() {
    // tri_sim = 0.5*0.97 + 0.3*0.93 + 0.2*0.90 = 0.944 >= 0.91 (threshold)
    let v = FakeVerifier::uniform(0.0);
    let sink = Arc::new(RecordingSink::default());
    let gw = TwoStageDualTrackGateway::new(TwoStageConfig::default())
        .unwrap()
        .with_refusal_sink(sink.clone());

    let (best, null) = scores(-0.5); // inside the ambiguity band [-1.5, 0.5]
    let ev = gw
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();

    let recorded = sink.events.lock().unwrap();
    assert_eq!(recorded.len(), 1, "exactly one event must be recorded");
    let e = &recorded[0];
    assert_eq!(e.context, CONTEXT);
    assert_eq!(e.question, QUESTION);
    assert_eq!(e.candidate, "1903");
    assert_eq!(e.best_span_score, best);
    assert_eq!(e.null_score, null);
    assert_eq!(e.score_diff, ev.score_diff);
    assert_eq!(e.stage2_triggered, ev.stage2_triggered);
    assert!(e.stage2_triggered);
    assert_eq!(e.decision_flipped, ev.decision_flipped);
    assert_eq!(e.is_answerable, ev.is_answerable);
    assert_eq!(e.tri_sim, ev.tri_sim);
    assert_eq!(e.p_same_meaning, ev.p_same_meaning);
    assert_eq!(e.verifier_id, ev.verifier_id);
    assert_eq!(e.verifier_type, ev.verifier_type);
}

#[test]
fn no_sink_attached_behaves_as_before() {
    let v_with = FakeVerifier::uniform(0.0);
    let v_without = FakeVerifier::uniform(0.0);
    let sink = Arc::new(RecordingSink::default());

    let gw_with_sink = TwoStageDualTrackGateway::new(TwoStageConfig::default())
        .unwrap()
        .with_refusal_sink(sink.clone());
    let gw_without_sink = TwoStageDualTrackGateway::new(TwoStageConfig::default()).unwrap();

    let (best, null) = scores(-0.5); // ambiguity band
    let ev_with_sink = gw_with_sink
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v_with)
        .unwrap();
    let ev_without_sink = gw_without_sink
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v_without)
        .unwrap();

    assert_eq!(ev_with_sink, ev_without_sink);
}

#[test]
fn failing_sink_fails_the_decision_closed() {
    let v = FakeVerifier::uniform(0.0);
    let sink = Arc::new(FailingSink {
        message: "durable store unreachable".into(),
    });
    let gw = TwoStageDualTrackGateway::new(TwoStageConfig::default())
        .unwrap()
        .with_refusal_sink(sink);

    let (best, null) = scores(-0.5); // ambiguity band, triggers stage 2
    let err = gw
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap_err();

    assert!(
        matches!(err, GateError::RefusalSinkFailure(ref m) if m == "durable store unreachable"),
        "{err:?}"
    );
}

#[test]
fn fast_pass_never_calls_the_sink() {
    let v = FakeVerifier::uniform(0.0);
    let sink = Arc::new(RecordingSink::default());
    let gw = TwoStageDualTrackGateway::new(TwoStageConfig::default())
        .unwrap()
        .with_refusal_sink(sink.clone());

    // Clearly outside the ambiguity band [-1.5, 0.5] on the confident side.
    let (best, null) = scores(-3.0);
    let ev = gw
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.fast_pass);
    assert!(sink.events.lock().unwrap().is_empty());

    // And clearly outside the band on the confident no-answer side.
    let (best, null) = scores(5.0);
    let ev = gw
        .decide_with_scores(CONTEXT, QUESTION, "1903", best, null, &v)
        .unwrap();
    assert!(ev.fast_pass);
    assert!(sink.events.lock().unwrap().is_empty());
}
