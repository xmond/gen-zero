//! Discrete topology presets sealed at mount time.
//!
//! Every request enters through an exposed entry (HTTP `/v1/mounts`,
//! `/message`, the MCP frame handler). The checks:
//! 1. each of the four presets mounts and gets a deterministic SHA-256 seal;
//! 2. names outside the whitelist and mis-sized dynamics are refused at mount;
//! 3. a point or event whose width is not the sealed `dim()` is
//!    `FiberMismatch` (HTTP 400), with no verdict;
//! 4. the Busemann test runs end to end on 64, 128 and 256 coordinates.
//!
//! The CLI exit-code check lives in `crates/gen-zero-cli/tests/
//! discrete_presets_cli.rs`: only the crate that owns the binary gets
//! `CARGO_BIN_EXE_gen-zero`.
//!
//! Points are hand-built and the geometry parameters are presets; nothing
//! here is trained or measures accuracy on any dataset.

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use gen_zero_lod::TopologyPreset;
use gen_zero_service::{McpServer, PolymorphicZeroEngine};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use tower::util::ServiceExt;

const TOKEN: &str = "gz_test_discrete_presets";

fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_semantic(None))
}

async fn send(engine: &Arc<PolymorphicZeroEngine>, path: &str, body: Value) -> (StatusCode, Value) {
    let app = McpServer::build_router(Arc::clone(engine), Some(TOKEN.to_string()));
    let req = Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"));
    let resp = app
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let shown = serde_json::to_string(&v).unwrap();
    eprintln!("{path} -> {status}: {}", &shown[..shown.len().min(2000)]);
    (status, v)
}

/// The scheme 1 fixture with its topology replaced.
fn assets(topology: &str) -> Value {
    let mut a: Value = serde_json::from_str(include_str!(
        "fixtures/cognitive_assets_boolq_entailment.json"
    ))
    .unwrap();
    a["entailment"]["topology"] = json!(topology);
    a
}

/// `rows x cols` diagonal matrix with `d` on the diagonal.
fn diag(rows: usize, cols: usize, d: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| if i == j { d } else { 0.0 }).collect())
        .collect()
}

/// Assets with `entailment.dynamics` `A = 0`, `B = I`, both `n x n`.
fn fiber_assets(topology: &str, n: usize) -> Value {
    let mut a = assets(topology);
    a["entailment"]["dynamics"] = json!({"a": diag(n, n, 0.0), "b": diag(n, n, 1.0)});
    a
}

async fn publish(engine: &Arc<PolymorphicZeroEngine>, assets: Value) -> (StatusCode, Value) {
    send(
        engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": assets, "reason": "discrete preset fixture"}),
    )
    .await
}

async fn mount(assets: Value) -> Arc<PolymorphicZeroEngine> {
    let engine = engine();
    let (status, body) = publish(&engine, assets).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    engine
}

/// Hyperbolic radius `r` at angle `theta` from axis 0; sphere at its pole.
/// The pole index comes from the preset layout, never a fixed offset.
fn point(preset: TopologyPreset, r: f64, theta: f64) -> Vec<f64> {
    let l = preset.layout();
    let mut v = vec![0.0; preset.dim()];
    v[0] = r * theta.cos();
    v[1] = r * theta.sin();
    v[l.s_range().start] = 1.0;
    v
}

fn entail_req(passage: &[f64], question: &[f64]) -> Value {
    json!({"action": "entail", "entailment": {"passage": passage, "question": question}})
}

fn is_sha256_hex(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

async fn expect_refused(engine: &Arc<PolymorphicZeroEngine>, body: Value, code: &str) -> Value {
    let (status, body) = send(engine, "/message", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["result"]["is_error"], true, "{body}");
    assert_eq!(body["result"]["rejection"]["code"], code, "{body}");
    assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");
    body
}

// ---------------------------------------------------------------- mount seal

#[tokio::test]
async fn all_four_presets_mount_with_a_deterministic_sha256_seal() {
    let mut geometry_seals = HashSet::new();
    let mut mount_digests = HashSet::new();
    for preset in TopologyPreset::ALL {
        let name = preset.as_str();
        // Two independent engines, same assets: the same seal.
        let (s1, b1) = publish(&engine(), assets(name)).await;
        let (s2, b2) = publish(&engine(), assets(name)).await;
        assert_eq!(s1, StatusCode::OK, "{b1}");
        assert_eq!(s2, StatusCode::OK, "{b2}");
        assert_eq!(b1["published"]["to"], 2);
        assert_eq!(b1["assets"]["entailment"], name);
        assert_eq!(b1["assets"]["entailment_dim"], preset.dim());
        for field in ["model", "geometry", "policy"] {
            assert!(is_sha256_hex(&b1["sealed"][field]), "{field}: {b1}");
            assert_eq!(b1["sealed"][field], b2["sealed"][field], "{field}");
        }
        assert!(is_sha256_hex(&b1["published"]["digest"]), "{b1}");
        assert_eq!(b1["published"]["digest"], b2["published"]["digest"]);
        assert_eq!(b1["mount"]["digest"], b2["mount"]["digest"]);
        assert_eq!(b1["mount"]["has_cognitive_assets"], true);
        assert!(
            geometry_seals.insert(b1["sealed"]["geometry"].clone()),
            "{name}"
        );
        assert!(
            mount_digests.insert(b1["published"]["digest"].clone()),
            "{name}"
        );
    }
    // Four presets, four geometry seals and four mount digests.
    assert_eq!(geometry_seals.len(), 4);
    assert_eq!(mount_digests.len(), 4);
}

#[tokio::test]
async fn deep_128d_alias_seals_as_boolq_128d() {
    let (_, canonical) = publish(&engine(), assets("boolq_128d")).await;
    let (status, alias) = publish(&engine(), assets("deep_128d")).await;
    assert_eq!(status, StatusCode::OK, "{alias}");
    assert_eq!(alias["assets"]["entailment"], "boolq_128d");
    assert_eq!(alias["sealed"]["geometry"], canonical["sealed"]["geometry"]);
    assert_eq!(
        alias["published"]["digest"],
        canonical["published"]["digest"]
    );
}

#[tokio::test]
async fn unlisted_topology_names_are_refused_at_mount() {
    let engine = engine();
    for bad in [
        "dynamic_100d",
        "random_512d",
        "extended_512d",
        "BOOLQ_128D",
        "boolq_128d ",
        "",
    ] {
        let (status, body) = publish(&engine, assets(bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad:?}: {body}");
        assert_eq!(body["error"]["code"], "InvalidParams", "{body}");
        assert_eq!(body["error"]["stage"], "assets", "{body}");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("unsupported topology preset")
                && msg.contains("compact_64d, balanced_128d, boolq_128d, extended_256d"),
            "{msg}"
        );
    }
    // None of the refusals advanced the mount: base_version 1 still wins CAS.
    let (status, body) = publish(&engine, assets("compact_64d")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["published"]["from"], 1);
    assert_eq!(body["published"]["to"], 2);
}

#[tokio::test]
async fn dynamics_must_be_exactly_preset_dim_square() {
    let engine = engine();
    let cases = [
        ("compact_64d", 128, 128, 128, 128),
        ("compact_64d", 64, 64, 64, 128),
        ("balanced_128d", 64, 64, 64, 64),
        ("boolq_128d", 256, 256, 256, 256),
        ("extended_256d", 128, 128, 128, 128),
        ("extended_256d", 256, 255, 256, 256),
    ];
    for (name, ar, ac, br, bc) in cases {
        let mut a = assets(name);
        a["entailment"]["dynamics"] = json!({"a": diag(ar, ac, 0.0), "b": diag(br, bc, 1.0)});
        let (status, body) = publish(&engine, a).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{name} {ar}x{ac}/{br}x{bc}: {body}"
        );
        assert_eq!(body["error"]["code"], "InvalidParams", "{body}");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains(&format!("for topology {name}")), "{msg}");
    }
    // Correct widths mount, one fresh engine each.
    for preset in TopologyPreset::ALL {
        let (status, body) =
            publish(&self::engine(), fiber_assets(preset.as_str(), preset.dim())).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["assets"]["entailment_dynamics"], true);
    }
}

// ---------------------------------------------------------------- run time

#[tokio::test]
async fn runtime_width_mismatch_is_fiber_mismatch() {
    use TopologyPreset::*;
    // (mounted, sent passage width, sent question width)
    let cases = [
        (Compact64d, Boolq128d, Boolq128d),
        (Compact64d, Compact64d, Extended256d),
        (Balanced128d, Compact64d, Compact64d),
        (Boolq128d, Compact64d, Boolq128d),
        (Extended256d, Boolq128d, Boolq128d),
        (Extended256d, Extended256d, Balanced128d),
    ];
    for (mounted, p_as, q_as) in cases {
        let engine = mount(assets(mounted.as_str())).await;
        let p = point(p_as, 0.3, 0.0);
        let q = point(q_as, 0.8, 0.02);
        let body = expect_refused(&engine, entail_req(&p, &q), "FiberMismatch").await;
        let bad = if p.len() != mounted.dim() {
            p.len()
        } else {
            q.len()
        };
        assert_eq!(
            body["result"]["rejection"]["detail"],
            format!(
                "passage/question dimension mismatch: expected {}, got {bad}",
                mounted.dim()
            ),
            "{body}"
        );
        assert_eq!(body["result"]["rejection"]["stage"], "entailment");
    }
}

#[tokio::test]
async fn event_width_mismatch_is_fiber_mismatch() {
    let preset = TopologyPreset::Compact64d;
    let engine = mount(fiber_assets(preset.as_str(), preset.dim())).await;
    let p = point(preset, 0.3, 0.0);
    let ok = json!([{"time_ns": 1_000_000_000u64, "input": vec![0.0; 64]}]);
    for width in [63, 128, 256] {
        let wide = json!([{"time_ns": 1_000_000_000u64, "input": vec![0.0; width]}]);
        for (pe, qe) in [(ok.clone(), wide.clone()), (wide.clone(), ok.clone())] {
            let req = json!({"action": "entail", "entailment": {
                "passage": p, "question": p,
                "passage_events": pe, "question_events": qe, "window_start_ns": 0,
            }});
            let body = expect_refused(&engine, req, "FiberMismatch").await;
            let detail = body["result"]["rejection"]["detail"].as_str().unwrap();
            assert!(
                detail.contains(&format!("dimension mismatch: expected 64, got {width}")),
                "{detail}"
            );
        }
    }
}

// ---------------------------------------------------------------- verdicts

#[tokio::test]
async fn busemann_entailment_runs_end_to_end_on_every_preset() {
    let mut frames = HashSet::new();
    for preset in TopologyPreset::ALL {
        let engine = mount(assets(preset.as_str())).await;
        let l = preset.layout();
        let general = point(preset, 0.3, 0.0);
        let specific = point(preset, 0.8, 0.02);

        let (status, body) = send(&engine, "/message", entail_req(&general, &specific)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["result"]["is_error"], false, "{body}");
        let e = &body["result"]["meta"]["entailment"];
        assert_eq!(e["is_entailed"], true, "{e}");
        assert_eq!(e["mode"], "direct");
        assert_eq!(e["factors"]["deeper"], true);
        assert!(
            (e["cone_angle"].as_f64().unwrap() - 0.02).abs() < 1e-12,
            "{e}"
        );
        assert_eq!(e["topology"], preset.as_str());
        assert_eq!(e["dim"], preset.dim());
        assert_eq!(e["hyperbolic_dim"], l.h());
        assert_eq!(e["euclidean_dim"], l.e());
        assert_eq!(e["spherical_dim"], l.s_ambient());
        assert_eq!(
            l.h() + l.e() + l.s_ambient(),
            preset.dim(),
            "layout must fill the width"
        );
        assert_eq!(e["geometry"]["topology"], preset.as_str());
        assert_eq!(e["mount_version"], 2);
        assert!(frames.insert(e["geometry"]["frame"].clone()), "{preset}");

        // Reverse direction: a false verdict, not an error.
        let (status, body) = send(&engine, "/message", entail_req(&specific, &general)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let e = &body["result"]["meta"]["entailment"];
        assert_eq!(e["is_entailed"], false, "{e}");
        assert_eq!(e["factors"]["deeper"], false);
        assert_eq!(e["confidence"], 0.0);
    }
    assert_eq!(frames.len(), 4, "every preset has its own geometry frame");
}

#[tokio::test]
async fn mcp_tools_call_follows_the_mounted_preset() {
    for preset in [TopologyPreset::Compact64d, TopologyPreset::Extended256d] {
        let engine = mount(assets(preset.as_str())).await;
        let server = McpServer {
            engine: Arc::clone(&engine),
            auth_token: None,
            bridge_required: false,
            closed_loop: None,
        };
        let call = |args: Value| {
            let mut buf = json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "zero", "arguments": args},
            })
            .to_string()
            .into_bytes();
            buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
            buf
        };
        let mut ok = call(entail_req(
            &point(preset, 0.3, 0.0),
            &point(preset, 0.8, 0.02),
        ));
        let v: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut ok).await).unwrap();
        assert_eq!(v["result"]["isError"], false, "{v}");
        let e = &v["result"]["_meta"]["entailment"];
        assert_eq!(e["is_entailed"], true, "{v}");
        assert_eq!(e["topology"], preset.as_str());
        assert_eq!(e["dim"], preset.dim());

        // The 128-wide BoolQ point is the wrong width on both mounts.
        let boolq = TopologyPreset::Boolq128d;
        let mut bad = call(entail_req(
            &point(boolq, 0.3, 0.0),
            &point(boolq, 0.8, 0.02),
        ));
        let v: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut bad).await).unwrap();
        assert_eq!(v["result"]["isError"], true, "{v}");
        assert_eq!(
            v["result"]["_meta"]["reject"]["code"], "FiberMismatch",
            "{v}"
        );
    }
}

#[tokio::test]
async fn fiber_cross_diff_path_runs_on_64d_and_256d() {
    for preset in [TopologyPreset::Compact64d, TopologyPreset::Extended256d] {
        let n = preset.dim();
        let engine = mount(fiber_assets(preset.as_str(), n)).await;
        let p = point(preset, 0.3, 0.0);
        let zero = vec![0.0; n];
        let mut radial = vec![0.0; n];
        radial[0] = 0.5;
        // p == q: the direct test fails (not deeper). With `A = 0`, `B = I`
        // the question stream moves q outward along axis 0, so q' is deeper.
        let req = json!({"action": "entail", "entailment": {
            "passage": p, "question": p,
            "passage_events": [{"time_ns": 1_000_000_000u64, "input": zero}],
            "question_events": [{"time_ns": 1_000_000_000u64, "input": radial}],
            "window_start_ns": 0,
        }});
        let (status, body) = send(&engine, "/message", req).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let e = &body["result"]["meta"]["entailment"];
        assert_eq!(e["mode"], "fiber_cross_diff_ssm", "{e}");
        assert_eq!(e["is_entailed"], true, "{e}");
        assert_eq!(e["fiber_ssm"]["direct"]["is_entailed"], false, "{e}");
        assert_eq!(e["fiber_ssm"]["scan"]["steps"], 1);
        assert_eq!(
            e["fiber_ssm"]["evidence_question"]
                .as_array()
                .unwrap()
                .len(),
            n
        );
        assert_eq!(e["dim"], n);
    }
}

// ---------------------------------------------------------------- weights

/// Reviewer reproduction: each `alpha` is finite and > 0, but their sum
/// overflows to `+inf`, so the confidence average was `inf / inf = NaN`,
/// served as `"confidence": null` under HTTP 200. The mount must refuse it.
#[tokio::test]
async fn overflow_weights_fail_closed_with_domain_violation() {
    let engine = engine();
    for preset in TopologyPreset::ALL {
        let mut a = assets(preset.as_str());
        for k in ["alpha_h", "alpha_e", "alpha_s"] {
            a["entailment"][k] = json!(1e308);
        }
        let (status, body) = publish(&engine, a).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{preset}: {body}");
        assert_eq!(body["error"]["code"], "InvalidParams", "{body}");
        assert_eq!(body["error"]["stage"], "assets", "{body}");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("alpha_h + alpha_e + alpha_s must be finite"),
            "{msg}"
        );
    }
    // No refused mount was installed: an entail request still has no geometry,
    // and base_version 1 still wins CAS.
    let p = point(TopologyPreset::Compact64d, 0.3, 0.0);
    let q = point(TopologyPreset::Compact64d, 0.8, 0.02);
    let (status, body) = send(&engine, "/message", entail_req(&p, &q)).await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["is_error"], true, "{body}");
    assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");

    // One huge weight with a finite sum mounts, and the verdict carries a
    // finite numeric confidence, never `null`.
    let mut a = assets("compact_64d");
    a["entailment"]["alpha_h"] = json!(1e308);
    let (status, body) = publish(&engine, a).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["published"]["from"], 1);
    let (status, body) = send(&engine, "/message", entail_req(&p, &q)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let e = &body["result"]["meta"]["entailment"];
    assert_eq!(e["is_entailed"], true, "{e}");
    let c = e["confidence"]
        .as_f64()
        .expect("confidence must be a number");
    assert!(c.is_finite() && (0.0..=1.0).contains(&c), "{e}");
}
