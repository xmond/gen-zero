//! Weight-backed checks of the native Qwen scorer. They need real Qwen2.5-0.5B
//! weights, so they are ignored by default and run explicitly:
//!
//! ```text
//! GENZERO_QWEN_TEST_GGUF=/path/Qwen2.5-0.5B.Q8_0.gguf \
//! GENZERO_QWEN_TEST_TOKENIZER=/path/tokenizer.json \
//! GENZERO_QWEN_TEST_DIR=/path/hf-snapshot-dir \
//! cargo test --release -p gen-zero-model --test qwen_native_parity -- --ignored --nocapture
//! ```
//!
//! A missing variable is a test failure, never a silent pass.

use candle_core::IndexOp;
use gen_zero_model::semantic_qwen::{
    candidate_text, compose_prompt, ASK_FRAME, RISK_ESCALATE_THRESHOLD, RISK_HARD_STOP_THRESHOLD,
};
use gen_zero_model::QwenSemanticScorer;
use std::path::PathBuf;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} (see file header)")))
}

fn gguf_scorer() -> QwenSemanticScorer {
    let tok = env_path("GENZERO_QWEN_TEST_TOKENIZER");
    QwenSemanticScorer::load(&env_path("GENZERO_QWEN_TEST_GGUF"), Some(&tok)).expect("load gguf")
}

fn dir_scorer() -> QwenSemanticScorer {
    QwenSemanticScorer::load(&env_path("GENZERO_QWEN_TEST_DIR"), None).expect("load safetensors")
}

/// Batched right-padded continuations must equal scoring each full sequence alone.
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn batched_continuations_equal_one_by_one_sequences() {
    let scorer = gguf_scorer();
    let prompt = compose_prompt("The build is broken after the last merge.", ASK_FRAME, &[]);
    let names = [
        "revert_the_merge",
        "go_home",
        "read the compiler error output carefully",
    ];
    let texts: Vec<String> = names.iter().map(|n| candidate_text(n)).collect();
    let (batched, _) = scorer
        .continuation_log_likelihoods(&prompt, &texts)
        .unwrap();

    let model = scorer.model();
    for (text, got) in texts.iter().zip(&batched) {
        let p = scorer.encode(&prompt).unwrap();
        let c = scorer.encode(text).unwrap();
        let full: Vec<u32> = p.iter().chain(&c).copied().collect();
        let (hidden, _) = model.forward(std::slice::from_ref(&full), None).unwrap();
        let mut total = 0.0;
        for (j, &tok) in c.iter().enumerate() {
            let pos = p.len() - 1 + j;
            let row = hidden.i((0, pos)).unwrap().unsqueeze(0).unwrap();
            total += model.log_probs_of(&row, &[vec![tok]]).unwrap()[0][0];
        }
        println!("{text:?}: batched {got:.5} single {total:.5}");
        assert!((got - total).abs() < 1e-3, "{text:?}: {got} vs {total}");
    }
}

#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn score_candidates_is_a_normalized_non_uniform_distribution() {
    let scorer = gguf_scorer();
    let candidates: Vec<String> = [
        "water the plants",
        "file the quarterly tax return",
        "reboot the router",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let probs = scorer
        .score_candidates(
            "The leaves of my tomato plants are dry and drooping.\nThe first action to take is:",
            &candidates,
        )
        .unwrap();
    println!("score_candidates = {probs:?}");
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
    assert!(probs.iter().all(|p| p.is_finite() && *p > 0.0));
    let spread =
        probs.iter().cloned().fold(0.0, f64::max) - probs.iter().cloned().fold(1.0, f64::min);
    assert!(spread > 0.05, "distribution is nearly uniform: {probs:?}");
    assert_eq!(
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0,
        0,
        "the plant context should favour watering: {probs:?}"
    );

    let pmi = scorer
        .score_pmi(
            "The leaves of my tomato plants are dry and drooping.",
            &candidates,
            ASK_FRAME,
            &[],
            None,
        )
        .unwrap();
    println!("score_pmi = {:?}", pmi.candidates);
    assert_eq!(pmi.chosen_index, 0);
    let sum: f64 = pmi.candidates.iter().map(|c| c.probability).sum();
    assert!((sum - 1.0).abs() < 1e-9);
    assert!(pmi.entropy < 1.0);
}

struct Row {
    text: String,
    label: u8,
    python_p: f64,
}

fn report_rows() -> Vec<(String, Row)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../python/gen_zero/service/risk_data/report.json"
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut rows = Vec::new();
    for set in ["calibration", "heldout"] {
        for r in report[set]["rows"].as_array().expect("rows") {
            rows.push((
                set.to_string(),
                Row {
                    text: r["text"].as_str().unwrap().to_string(),
                    label: r["label"].as_u64().unwrap() as u8,
                    python_p: r["p_dangerous"].as_f64().unwrap(),
                },
            ));
        }
    }
    rows
}

fn tier(p: f64) -> &'static str {
    if p >= RISK_HARD_STOP_THRESHOLD {
        "HardStop"
    } else if p >= RISK_ESCALATE_THRESHOLD {
        "Escalate"
    } else {
        "Proceed"
    }
}

/// Per-row paired comparison against the Python fp32 scores in report.json.
/// Prints every row whose tier differs; returns (max |diff|, mean |diff|, tier changes).
fn risk_parity(scorer: &QwenSemanticScorer, label: &str) -> (f64, f64, usize, usize) {
    let rows = report_rows();
    let mut diffs = Vec::new();
    let mut changes = 0;
    let mut dangerous_proceed = 0;
    for (set, row) in &rows {
        let ours = scorer.assess_risk_detailed(&row.text).unwrap().p_dangerous;
        let d = (ours - row.python_p).abs();
        diffs.push(d);
        if tier(ours) != tier(row.python_p) {
            changes += 1;
            println!(
                "[{label}] TIER CHANGE {set} label={} python={:.4} ({}) native={:.4} ({}) {:?}",
                row.label,
                row.python_p,
                tier(row.python_p),
                ours,
                tier(ours),
                row.text
            );
        }
        if row.label == 1 && tier(ours) == "Proceed" {
            dangerous_proceed += 1;
        }
    }
    let max = diffs.iter().cloned().fold(0.0, f64::max);
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    println!(
        "[{label}] rows={} max|dp|={max:.5} mean|dp|={mean:.5} tier_changes={changes} dangerous_proceed={dangerous_proceed}",
        rows.len()
    );
    (max, mean, changes, dangerous_proceed)
}

#[test]
#[ignore = "needs Qwen2.5-0.5B safetensors weights"]
fn risk_parity_with_python_report_safetensors_f32() {
    let (max, _, changes, dangerous_proceed) = risk_parity(&dir_scorer(), "safetensors-f32");
    // Same fp32 weights as the Python reference: only summation order differs.
    assert!(max < 1e-3, "max |dp| {max}");
    assert_eq!(changes, 0);
    assert_eq!(dangerous_proceed, 0);
}

#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn risk_parity_with_python_report_gguf() {
    let (_, _, _, dangerous_proceed) = risk_parity(&gguf_scorer(), "gguf");
    // Quantization moves scores; the thresholds were calibrated on fp32. The
    // property the gate relies on is recall: no dangerous row may proceed.
    assert_eq!(
        dangerous_proceed, 0,
        "a dangerous report row proceeds under GGUF"
    );
}

/// Causality: the hidden state at a position must not depend on how many
/// tokens follow it, beyond f32 rounding (gemm picks its blocking by matrix
/// size, so the last bits move with the sequence length).
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn hidden_state_does_not_depend_on_later_tokens() {
    let scorer = gguf_scorer();
    let model = scorer.model();
    let ids = scorer
        .encode(
            "The build is broken after the last merge. Somebody pushed a change to the \
             parser that nobody reviewed, and now every integration test fails on startup.",
        )
        .unwrap();
    assert!(ids.len() >= 20, "{} tokens", ids.len());
    let (reference, _) = model.forward(std::slice::from_ref(&ids), None).unwrap();
    let mut worst = (0f32, 0usize);
    for len in 1..=ids.len() {
        let (h, _) = model.forward(&[ids[..len].to_vec()], None).unwrap();
        let diff = (h.i((0, len - 1)).unwrap() - reference.i((0, len - 1)).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        if diff > worst.0 {
            worst = (diff, len);
        }
    }
    println!(
        "max hidden drift {} (at length {}) over {} lengths",
        worst.0,
        worst.1,
        ids.len()
    );
    assert!(
        worst.0 < 1e-3,
        "position {} drifts by {}",
        worst.1 - 1,
        worst.0
    );
}

/// Ask and route PMI scores vs the Python fp32 scorer, candidate by candidate
/// (fixture written by `python/gen_zero/scripts/dump_semantic_pmi_reference.py`).
/// Returns the largest |Δ log-likelihood| and |Δ probability| and whether every
/// case picked the same candidate.
fn pmi_parity(scorer: &QwenSemanticScorer, label: &str) -> (f64, f64, bool) {
    use gen_zero_model::semantic_qwen::{
        select_ask_frame, state_text, tool_continuation, ROUTE_FRAME,
    };
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/python_pmi_reference.json")).unwrap();
    let strings = |v: &serde_json::Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    };
    let (mut max_ll, mut max_p, mut same_choice) = (0f64, 0f64, true);
    for case in fixture["cases"].as_array().unwrap() {
        let result = if case["kind"] == "ask" {
            let names = strings(&case["candidates"]);
            let history = strings(&case["history"]);
            let context = state_text(case["context"].as_str().unwrap(), case.get("state"));
            let (frame, _) = select_ask_frame(&context, &names, &history);
            assert_eq!(frame, case["frame"].as_str().unwrap());
            scorer
                .score_pmi(&context, &names, frame, &history, None)
                .unwrap()
        } else {
            let tools = case["tools"].as_array().unwrap();
            let names: Vec<String> = tools
                .iter()
                .map(|t| t["name"].as_str().unwrap().to_string())
                .collect();
            let texts: Vec<String> = tools
                .iter()
                .map(|t| tool_continuation(t["name"].as_str().unwrap(), t["description"].as_str()))
                .collect();
            scorer
                .score_pmi(
                    case["context"].as_str().unwrap(),
                    &names,
                    ROUTE_FRAME,
                    &[],
                    Some(&texts),
                )
                .unwrap()
        };
        let python = case["python"].as_array().unwrap();
        let py_best = python
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1["probability"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b.1["probability"].as_f64().unwrap())
            })
            .unwrap()
            .0;
        same_choice &= py_best == result.chosen_index;
        for (ours, py) in result.candidates.iter().zip(python) {
            assert_eq!(ours.name, py["name"].as_str().unwrap());
            let dll = (ours.log_likelihood - py["log_likelihood"].as_f64().unwrap()).abs();
            let dbl = (ours.baseline_log_likelihood
                - py["baseline_log_likelihood"].as_f64().unwrap())
            .abs();
            let dp = (ours.probability - py["probability"].as_f64().unwrap()).abs();
            println!(
                "[{label}] {:<44} p native={:.5} python={:.5} |dll|={dll:.5} |dbase|={dbl:.5}",
                ours.name,
                ours.probability,
                py["probability"].as_f64().unwrap()
            );
            max_ll = max_ll.max(dll).max(dbl);
            max_p = max_p.max(dp);
        }
    }
    println!(
        "[{label}] PMI parity: max|dll|={max_ll:.5} max|dp|={max_p:.5} same_choice={same_choice}"
    );
    (max_ll, max_p, same_choice)
}

#[test]
#[ignore = "needs Qwen2.5-0.5B safetensors weights"]
fn pmi_parity_with_python_safetensors_f32() {
    let (max_ll, max_p, same_choice) = pmi_parity(&dir_scorer(), "safetensors-f32");
    assert!(same_choice);
    assert!(max_ll < 1e-2, "max |d log-likelihood| {max_ll}");
    assert!(max_p < 5e-3, "max |d probability| {max_p}");
}

#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn pmi_parity_with_python_gguf() {
    let (_, _, same_choice) = pmi_parity(&gguf_scorer(), "gguf");
    assert!(
        same_choice,
        "GGUF picks a different candidate than the fp32 reference"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `embed` gives one finite unit vector of the hidden width per text,
/// deterministic across calls, and refuses text with no token.
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn embed_returns_finite_unit_vectors_of_hidden_width() {
    let scorer = gguf_scorer();
    assert_eq!(scorer.info().hidden_size, 896);
    let v = scorer
        .embed("Restart the payment service after the outage.")
        .unwrap();
    assert_eq!(v.len(), 896);
    assert!(v.iter().all(|x| x.is_finite()));
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    assert_eq!(
        v,
        scorer
            .embed("Restart the payment service after the outage.")
            .unwrap()
    );
    assert!(scorer.embed("").is_err());
    // Longer than one window: every token still pooled, no truncation error.
    let long = "The quarterly audit found no anomalies in the ledger. ".repeat(120);
    assert!(scorer.encode(&long).unwrap().len() > gen_zero_model::qwen::MAX_SEQ_LEN);
    let v = scorer.embed(&long).unwrap();
    assert_eq!(v.len(), 896);
    assert!(v.iter().all(|x| x.is_finite()));
}

/// Paraphrases with no shared content word must sit closer than unrelated
/// texts. Prints the cosine matrix so the spread (anisotropy) is on record.
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn embed_puts_paraphrases_closer_than_unrelated_texts() {
    let scorer = gguf_scorer();
    let texts = [
        "The physician prescribed antibiotics for the infection.",
        "A doctor gave the patient medicine to treat bacteria.",
        "The stock market fell sharply after the interest rate hike.",
        "Shares dropped when the central bank raised borrowing costs.",
        "She planted tomatoes in the garden this spring.",
        "The compiler rejected the program because of a type error.",
    ];
    let vecs: Vec<Vec<f32>> = texts.iter().map(|t| scorer.embed(t).unwrap()).collect();
    for (i, a) in vecs.iter().enumerate() {
        let row: Vec<String> = vecs
            .iter()
            .map(|b| format!("{:.3}", cosine(a, b)))
            .collect();
        println!("cos[{i}] = {}", row.join(" "));
    }
    for (a, b) in [(0usize, 1usize), (2, 3)] {
        let pair = cosine(&vecs[a], &vecs[b]);
        for other in (0..texts.len()).filter(|&o| o != a && o != b) {
            assert!(
                pair > cosine(&vecs[a], &vecs[other]) && pair > cosine(&vecs[b], &vecs[other]),
                "pair ({a},{b}) cos {pair} not above text {other}"
            );
        }
    }
}
