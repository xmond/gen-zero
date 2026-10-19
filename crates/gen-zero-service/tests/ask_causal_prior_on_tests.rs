//! `ask` with the policy prior ON (`GENZERO_CAUSAL_PRIOR=on`).
//!
//! Its own test binary on purpose: the variable is process-global, and
//! `worldmodel_verbs_tests` relies on nobody in its process turning it on.
//!
//! A local stub scorer answers with canned probabilities. This checks the
//! plumbing (scorer -> policy prior -> confirmation gate), not a model.

#[path = "support/stub_scorer.rs"]
mod support;

use gen_zero_service::ServiceError;
use serde_json::json;
use support::{assert_raw_rejected, engine_with, milk_ask, milk_imagine, BAD_RAW};

/// WebGPT R3 M01.A: the hazard is only in `state`. The scorer's prior sees
/// context + state and penalises both candidates by -16, so the ranking
/// does not move and "drink it" stays chosen. The gate must see the same
/// text and stop the ask; it used to check `context` alone and say ok.
#[tokio::test]
async fn ask_gate_sees_the_state_the_policy_prior_scored() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    let engine = engine_with(vec![0.99, 0.01], None).await;

    let hazard = json!({
        "action": "ask", "context": "select storage",
        "state": {"food": "The milk is expired"},
        "candidates": ["drink it", "eat it"],
    });
    match engine.execute(&hazard).await {
        Err(ServiceError::ConfirmationRequired(msg)) => {
            assert!(msg.contains("causal utility prior"), "{msg}");
            assert!(msg.contains("drink it"), "{msg}");
        }
        Ok(out) => panic!("expected ConfirmationRequired, got ok: {}", out.meta),
        Err(e) => panic!("expected ConfirmationRequired, got {e:?}"),
    }

    // Control: the same request without the hazardous state proceeds.
    let safe = json!({
        "action": "ask", "context": "select storage",
        "candidates": ["drink it", "eat it"],
    });
    let out = engine.execute(&safe).await.expect("no hazard, no veto");
    assert!(!out.is_error, "{}", out.meta);
    assert_eq!(out.meta["chosen_action"], "drink it", "{}", out.meta);
}

/// WebGPT R3 M01.C: with the prior on, `imagine` reports the model's raw
/// step probabilities and raw plan likelihood next to the reweighted ones,
/// and its summary does not call the reweighted score a model likelihood.
#[tokio::test]
async fn imagine_reports_raw_likelihood_beside_the_reweighted_score() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    let engine = engine_with(vec![0.99, 0.01], None).await;
    let out = engine
        .execute(&json!({
            "action": "imagine", "scenario": "The milk is expired",
            "candidate_actions": ["drink it", "throw it away"],
            "horizon": 2, "simulations": 8,
        }))
        .await
        .expect("imagine runs");
    let meta = &out.meta;
    assert_eq!(meta["policy_prior_enabled"], true, "{meta}");
    assert_eq!(meta["likelihood_source"], "policy_prior_reweighted");
    assert_eq!(meta["best_action"], "throw it away", "{meta}");

    let plan = meta["plan"].as_array().expect("plan");
    assert!(!plan.is_empty(), "{meta}");
    let (mut raw_log, mut log) = (0.0, 0.0);
    for step in plan {
        let raw = step["raw_probability"].as_f64().expect("raw_probability");
        let p = step["probability"].as_f64().unwrap();
        // The stub's raw score for "throw it away" is 0.01; the prior lifts it.
        assert!((raw - 0.01).abs() < 1e-9, "{step}");
        assert!(p > 0.99, "{step}");
        raw_log += raw.max(1e-12).ln();
        log += p.max(1e-12).ln();
    }
    let num = |k: &str| meta[k].as_f64().unwrap_or_else(|| panic!("{k}: {meta}"));
    assert!((num("raw_sequence_log_likelihood") - raw_log).abs() < 1e-9);
    assert!((num("raw_sequence_likelihood") - raw_log.exp()).abs() < 1e-12);
    assert!((num("plan_sequence_log_likelihood") - log).abs() < 1e-9);

    let text: String = out.content.iter().map(|c| c.text.as_str()).collect();
    assert!(
        text.contains("policy-prior-reweighted sequence score"),
        "{text}"
    );
    assert!(
        text.contains("raw language-model likelihood of the plan"),
        "{text}"
    );
    assert!(!text.contains("a language-model likelihood"), "{text}");
}

/// WebGPT R4 M11: a valid `probability: [0.99, 0.01]` next to an explicit
/// invalid `raw_probability` used to pass, and the prior re-ranked on the
/// raw value (`[1e100, 2.0]` or `[0.01, 0.99]` flipped expired milk to
/// "drink it"). Both `ask` and `imagine` must reject it.
#[tokio::test]
async fn invalid_raw_probability_is_rejected_with_the_prior_on() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    for raw in BAD_RAW {
        let engine = engine_with(vec![0.99, 0.01], Some(raw.to_vec())).await;
        let out = engine.execute(&milk_ask()).await.expect("ask outcome");
        assert_raw_rejected(&out, &format!("ask {raw:?}"));
        assert!(out.meta.get("chosen_action").is_none(), "{}", out.meta);

        let engine = engine_with(vec![0.99, 0.01], Some(raw.to_vec())).await;
        let out = engine
            .execute(&milk_imagine())
            .await
            .expect("imagine outcome");
        assert_raw_rejected(&out, &format!("imagine {raw:?}"));
        assert!(out.meta.get("best_action").is_none(), "{}", out.meta);
    }
}

/// Control: the legacy bridge omits `raw_probability`, or sends one equal
/// to `probability`. The prior still moves expired milk to "throw it away".
#[tokio::test]
async fn omitted_or_matching_raw_probability_still_works_with_the_prior_on() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    for raw in [None, Some(vec![0.99, 0.01])] {
        let engine = engine_with(vec![0.99, 0.01], raw.clone()).await;
        let out = engine.execute(&milk_ask()).await.expect("ask runs");
        assert!(!out.is_error, "{raw:?}: {}", out.meta);
        assert_eq!(out.meta["chosen_action"], "throw it away", "{}", out.meta);

        let engine = engine_with(vec![0.99, 0.01], raw.clone()).await;
        let out = engine.execute(&milk_imagine()).await.expect("imagine runs");
        assert!(!out.is_error, "{raw:?}: {}", out.meta);
        assert_eq!(out.meta["best_action"], "throw it away", "{}", out.meta);
        let raw_likelihood = out.meta["raw_sequence_likelihood"].as_f64().unwrap();
        assert!((raw_likelihood - 1e-4).abs() < 1e-9, "{}", out.meta);
    }
}
