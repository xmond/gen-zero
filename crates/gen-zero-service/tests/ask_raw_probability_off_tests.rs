//! WebGPT R4 M11 with the policy prior OFF (`GENZERO_CAUSAL_PRIOR=off`).
//!
//! Its own test binary on purpose: the variable is process-global, and
//! `ask_causal_prior_on_tests` turns it on for its whole process.

#[path = "support/stub_scorer.rs"]
mod support;

use support::{assert_raw_rejected, engine_with, milk_ask, milk_imagine, BAD_RAW};

/// A valid `probability: [0.99, 0.01]` next to an explicit invalid
/// `raw_probability` is rejected even when no prior would use it.
#[tokio::test]
async fn invalid_raw_probability_is_rejected_with_the_prior_off() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "off");
    for raw in BAD_RAW {
        let engine = engine_with(vec![0.99, 0.01], Some(raw.to_vec())).await;
        let out = engine.execute(&milk_ask()).await.expect("ask outcome");
        assert_raw_rejected(&out, &format!("ask {raw:?}"));

        let engine = engine_with(vec![0.99, 0.01], Some(raw.to_vec())).await;
        let out = engine
            .execute(&milk_imagine())
            .await
            .expect("imagine outcome");
        assert_raw_rejected(&out, &format!("imagine {raw:?}"));
    }
}

/// Control: with `raw_probability` omitted (the legacy Python bridge) or
/// equal to `probability`, the model's own pick stands.
#[tokio::test]
async fn omitted_or_matching_raw_probability_still_works_with_the_prior_off() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "off");
    for raw in [None, Some(vec![0.99, 0.01])] {
        let engine = engine_with(vec![0.99, 0.01], raw.clone()).await;
        let out = engine.execute(&milk_ask()).await.expect("ask runs");
        assert!(!out.is_error, "{raw:?}: {}", out.meta);
        assert_eq!(out.meta["chosen_action"], "drink it", "{}", out.meta);

        let engine = engine_with(vec![0.99, 0.01], raw.clone()).await;
        let out = engine.execute(&milk_imagine()).await.expect("imagine runs");
        assert!(!out.is_error, "{raw:?}: {}", out.meta);
        assert_eq!(out.meta["best_action"], "drink it", "{}", out.meta);
        assert_eq!(out.meta["policy_prior_enabled"], false, "{}", out.meta);
    }
}
