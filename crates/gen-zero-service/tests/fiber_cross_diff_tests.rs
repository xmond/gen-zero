//! Scheme 2: two-channel fiber cross-difference and SSM scan.
//!
//! Part 1 drives `FiberCrossDiff` through its public API (typed algebra,
//! fail-closed checks, the zero-difference ground state). Part 2 sends real
//! `entail` requests through HTTP `/message` and the MCP frame handler and
//! checks that the verdict comes from the fiber path: gauge from the sealed
//! geometry, cross-difference, parallel scan, moved question point, Busemann.
//!
//! The dynamics are hand-written (`A = 0` or `A = -I`, `B = I`); nothing here
//! is trained, and nothing here measures accuracy on any dataset.

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use gen_zero_service::tangent_ssm::{
    Backend, Epochs, Event, FiberCrossDiff, FiberId, FrozenContext, Matrix, ParallelTangentSsm,
    Reject, ScanBudget, Tangent, Version, ZohTangentSsm,
};
use gen_zero_service::{McpServer, PolymorphicZeroEngine};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

// ---------------------------------------------------------------- part 1

fn fid(tag: u8) -> FiberId {
    FiberId {
        patch: 0,
        base: [tag; 32],
        frame: [7; 32],
        path: [tag; 32],
        epochs: Epochs {
            version: Version(1),
            model: [1; 32],
            geometry: [2; 32],
            atlas: [3; 32],
            graph: [4; 32],
            policy: [5; 32],
        },
    }
}

fn ev(i: u64, time_ns: u64, input: &[f64]) -> Event {
    Event {
        id: [i as u8; 32],
        time_ns,
        input: input.to_vec().into_boxed_slice(),
    }
}

fn budget() -> ScanBudget {
    ScanBudget {
        max_steps: 64,
        max_growth: 1e6,
        max_state_norm: 1e6,
    }
}

fn neg_identity(n: usize) -> Matrix {
    let mut rows = vec![vec![0.0; n]; n];
    for (i, r) in rows.iter_mut().enumerate() {
        r[i] = -1.0;
    }
    Matrix::from_rows(&rows).unwrap()
}

fn rot90() -> Vec<Vec<f64>> {
    // e0 -> e1, e1 -> -e0, e2 fixed.
    vec![
        vec![0.0, -1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]
}

#[test]
fn identical_channels_under_identity_gauge_give_zero_diff_and_ground_state_scan() {
    let f = fid(1);
    let op = FiberCrossDiff::new(f.clone(), f.clone(), Matrix::identity(3)).unwrap();
    let stream = vec![
        ev(1, 100_000_000, &[0.4, -1.5, 2.0]),
        ev(2, 250_000_000, &[3.0, 0.25, -0.75]),
        ev(3, 400_000_000, &[-2.0, 1.0, 0.5]),
    ];
    let ctx =
        FrozenContext::new(f.clone(), neg_identity(3), Matrix::identity(3), 0, budget()).unwrap();
    for backend in [Backend::CpuSerial, Backend::CpuParallel { threads: 2 }] {
        let run = op
            .scan(&ZohTangentSsm::new(), &ctx, &stream, &stream, backend)
            .unwrap();
        assert_eq!(run.diffs.len(), 3);
        for d in &run.diffs {
            assert_eq!(d.fiber(), &f);
            assert!(d.coords().iter().all(|x| *x == 0.0), "{:?}", d.coords());
        }
        assert_eq!(run.output.states.len(), 3);
        for h in &run.output.states {
            assert!(h.coords().iter().all(|x| *x == 0.0), "{:?}", h.coords());
        }
        assert_eq!(run.output.evidence.backend, backend);
    }
}

#[test]
fn gauge_moves_the_passage_vector_before_the_difference() {
    let (p, q) = (fid(1), fid(2));
    let op = FiberCrossDiff::from_rows(p.clone(), q.clone(), &rot90()).unwrap();
    let v_p = Tangent::new(p.clone(), vec![1.0, 0.0, 0.0]).unwrap();
    let moved = op.transport(&v_p).unwrap();
    assert_eq!(moved.fiber(), &q);
    assert_eq!(moved.coords(), &[0.0, 1.0, 0.0]);

    // The transported passage vector cancels exactly.
    let same = Tangent::new(q.clone(), vec![0.0, 1.0, 0.0]).unwrap();
    assert_eq!(
        op.cross_diff(&v_p, &same).unwrap().coords(),
        &[0.0, 0.0, 0.0]
    );
    // Without transport the naive difference would be (-1, 1, 0); with it
    // the question's own e0 survives and the moved e1 is removed.
    let own = Tangent::new(q.clone(), vec![1.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        op.cross_diff(&v_p, &own).unwrap().coords(),
        &[1.0, -1.0, 0.0]
    );
}

#[test]
fn cross_diff_scan_equals_a_scan_of_the_differenced_events() {
    let (p, q) = (fid(1), fid(2));
    let op = FiberCrossDiff::from_rows(p, q.clone(), &rot90()).unwrap();
    let passage = vec![
        ev(1, 1_000_000, &[1.0, 2.0, 3.0]),
        ev(2, 3_000_000, &[0.5, 0.0, -1.0]),
    ];
    let question = vec![
        ev(3, 1_000_000, &[0.0, 0.0, 1.0]),
        ev(4, 3_000_000, &[2.0, 2.0, 2.0]),
    ];
    let ssm = ZohTangentSsm::new();
    let ctx =
        FrozenContext::new(q.clone(), neg_identity(3), Matrix::identity(3), 0, budget()).unwrap();
    let run = op
        .scan(&ssm, &ctx, &passage, &question, Backend::CpuSerial)
        .unwrap();
    // v_delta = v_q - R v_p with R e0 = e1, R e1 = -e0.
    assert_eq!(run.diffs[0].coords(), &[2.0, -1.0, -2.0]);
    assert_eq!(run.diffs[1].coords(), &[2.0, 1.5, 3.0]);

    let manual: Vec<Event> = run
        .diffs
        .iter()
        .zip(&passage)
        .enumerate()
        .map(|(i, (d, e))| ev(10 + i as u64, e.time_ns, d.coords()))
        .collect();
    let window = ssm.prepare(&manual, &ctx).unwrap();
    let h0 = Tangent::new(q, vec![0.0; 3]).unwrap();
    let reference = ssm.scan(&window, &h0, Backend::CpuSerial).unwrap();
    for (a, b) in run.output.states.iter().zip(&reference.states) {
        assert_eq!(a.coords(), b.coords());
    }
    let h = run.output.states[1].coords();
    assert!(h.iter().any(|x| x.abs() > 1e-4), "{h:?}");
}

#[test]
fn non_finite_gauges_are_domain_violations() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut rows = rot90();
        rows[1][2] = bad;
        let err = FiberCrossDiff::from_rows(fid(1), fid(2), &rows).unwrap_err();
        assert_eq!(err, Reject::DomainViolation, "gauge entry {bad}");
    }
}

#[test]
fn dimension_and_fiber_mismatches_are_refused() {
    let (p, q) = (fid(1), fid(2));
    // Ragged or empty gauge.
    let ragged = vec![vec![1.0, 0.0], vec![0.0]];
    assert_eq!(
        FiberCrossDiff::from_rows(p.clone(), q.clone(), &ragged).unwrap_err(),
        Reject::FiberMismatch
    );
    assert_eq!(
        FiberCrossDiff::from_rows(p.clone(), q.clone(), &[]).unwrap_err(),
        Reject::FiberMismatch
    );

    let op = FiberCrossDiff::from_rows(p.clone(), q.clone(), &rot90()).unwrap();
    let v2 = Tangent::new(p.clone(), vec![1.0, 0.0]).unwrap();
    assert_eq!(op.transport(&v2).unwrap_err(), Reject::FiberMismatch);
    // A passage vector tagged with the question fiber is not in the source.
    let wrong = Tangent::new(q.clone(), vec![1.0, 0.0, 0.0]).unwrap();
    assert_eq!(op.transport(&wrong).unwrap_err(), Reject::FiberMismatch);
    let v_p = Tangent::new(p.clone(), vec![1.0, 0.0, 0.0]).unwrap();
    let q4 = Tangent::new(q.clone(), vec![0.0; 4]).unwrap();
    assert_eq!(op.cross_diff(&v_p, &q4).unwrap_err(), Reject::FiberMismatch);
    let q_in_p = Tangent::new(p.clone(), vec![0.0; 3]).unwrap();
    assert_eq!(
        op.cross_diff(&v_p, &q_in_p).unwrap_err(),
        Reject::FiberMismatch
    );

    // Stream pairing: unequal length, shifted time, wrong width.
    let a = vec![ev(1, 10, &[1.0, 0.0, 0.0]), ev(2, 20, &[1.0, 0.0, 0.0])];
    let b = vec![ev(3, 10, &[1.0, 0.0, 0.0])];
    assert_eq!(op.diff_events(&a, &b).unwrap_err(), Reject::FiberMismatch);
    let shifted = vec![ev(3, 10, &[0.0; 3]), ev(4, 21, &[0.0; 3])];
    assert_eq!(
        op.diff_events(&a, &shifted).unwrap_err(),
        Reject::DomainViolation
    );
    let narrow = vec![ev(3, 10, &[0.0; 2]), ev(4, 20, &[0.0; 2])];
    assert_eq!(
        op.diff_events(&a, &narrow).unwrap_err(),
        Reject::FiberMismatch
    );
    assert_eq!(
        op.diff_events(&[], &[]).unwrap_err(),
        Reject::DomainViolation
    );
    // A NaN input is refused before any arithmetic.
    let nan = vec![ev(3, 10, &[f64::NAN, 0.0, 0.0]), ev(4, 20, &[0.0; 3])];
    assert_eq!(
        op.diff_events(&a, &nan).unwrap_err(),
        Reject::NonFiniteState
    );

    // The SSM context must model the target fiber at the target width.
    let ssm = ZohTangentSsm::new();
    let on_source =
        FrozenContext::new(p, neg_identity(3), Matrix::identity(3), 0, budget()).unwrap();
    assert_eq!(
        op.scan(&ssm, &on_source, &a, &a, Backend::CpuSerial)
            .unwrap_err(),
        Reject::FiberMismatch
    );
    let narrow_ctx = FrozenContext::new(
        q,
        neg_identity(3),
        Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]]).unwrap(),
        0,
        budget(),
    )
    .unwrap();
    assert_eq!(
        op.scan(&ssm, &narrow_ctx, &a, &a, Backend::CpuSerial)
            .unwrap_err(),
        Reject::FiberMismatch
    );
}

// ---------------------------------------------------------------- part 2

const TOKEN: &str = "gz_test_fiber_cross_diff";
const DIM: usize = 128;
/// First Euclidean (topic) coordinate of the BoolQ layout `[H^80 | R^24 | S]`.
const E0: usize = 80;

fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_bridge(None))
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
    eprintln!("{path} -> {status}: {}", &shown[..shown.len().min(4000)]);
    (status, v)
}

fn plain_assets() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/cognitive_assets_boolq_entailment.json"
    ))
    .unwrap()
}

/// The scheme 1 fixture plus `entailment.dynamics` `dh/dt = a_diag h + v_delta`.
fn fiber_assets(a_diag: f64) -> Value {
    let square = |d: f64| -> Vec<Vec<f64>> {
        (0..DIM)
            .map(|i| (0..DIM).map(|j| if i == j { d } else { 0.0 }).collect())
            .collect()
    };
    let mut assets = plain_assets();
    assets["entailment"]["dynamics"] = json!({"a": square(a_diag), "b": square(1.0)});
    assets
}

async fn mount(assets: Value) -> Arc<PolymorphicZeroEngine> {
    let engine = engine();
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": assets, "reason": "fiber fixture"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    engine
}

/// Hyperbolic radius `r` at angle `theta` from axis 0; sphere at its pole.
fn point(r: f64, theta: f64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    v[0] = r * theta.cos();
    v[1] = r * theta.sin();
    v[104] = 1.0;
    v
}

/// A tangent vector with the given `(index, value)` entries. Sphere entries
/// stay zero except where named, so it is tangent at `point(..)` as long as
/// index 104 (the sphere pole) is not used.
fn tangent(entries: &[(usize, f64)]) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for (i, x) in entries {
        v[*i] = *x;
    }
    v
}

/// One event per input, 1 s apart from `t = 0`, so with `A = 0` the final
/// state is the plain sum of the cross-differences.
fn events(inputs: &[Vec<f64>]) -> Value {
    Value::Array(
        inputs
            .iter()
            .enumerate()
            .map(|(i, u)| json!({"time_ns": (i as u64 + 1) * 1_000_000_000, "input": u}))
            .collect(),
    )
}

fn fiber_req(p: &[f64], q: &[f64], pe: Value, qe: Value) -> Value {
    json!({"action": "entail", "entailment": {
        "passage": p, "question": q,
        "passage_events": pe, "question_events": qe,
        "window_start_ns": 0,
    }})
}

async fn entail_ok(engine: &Arc<PolymorphicZeroEngine>, body: Value) -> Value {
    let (status, body) = send(engine, "/message", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["is_error"], false, "{body}");
    body["result"]["meta"]["entailment"].clone()
}

async fn entail_refused(engine: &Arc<PolymorphicZeroEngine>, body: Value, code: &str) -> Value {
    let (status, body) = send(engine, "/message", body).await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["rejection"]["code"], code, "{body}");
    assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");
    body
}

#[tokio::test]
async fn http_zero_difference_leaves_the_question_and_the_verdict_unchanged() {
    let engine = mount(fiber_assets(0.0)).await;
    let (p, q) = (point(0.3, 0.0), point(0.8, 0.02));
    let direct = entail_ok(
        &engine,
        json!({"action": "entail", "entailment": {"passage": p, "question": q}}),
    )
    .await;
    assert_eq!(direct["mode"], "direct");
    assert!(direct.get("fiber_ssm").is_none());

    // p == q, identity transport, identical streams: v_delta = 0, h_T = 0.
    let same = events(&[tangent(&[(0, 0.2), (E0, 1.0)]), tangent(&[(3, -0.4)])]);
    let e = entail_ok(&engine, fiber_req(&p, &p, same.clone(), same)).await;
    let f = &e["fiber_ssm"];
    assert_eq!(e["mode"], "fiber_cross_diff_ssm", "{e}");
    assert_eq!(f["trivial_path"], true);
    assert_eq!(f["arc_length"], 0.0);
    assert_eq!(f["diff_norms_inf"], json!([0.0, 0.0]));
    assert_eq!(f["final_state_norm_inf"], 0.0);
    assert_eq!(f["question_shift"], 0.0);
    assert_eq!(f["evidence_question"], json!(p));
    assert_eq!(f["scan"]["steps"], 2);
    assert!(f["scan"]["prefix_digest"].as_str().unwrap().len() == 64);
    assert_eq!(e["is_entailed"], f["direct"]["is_entailed"]);
    assert_eq!(
        e["is_entailed"], false,
        "a point does not strictly contain itself"
    );

    // Zero streams on a real pair: q' = q, so the fiber verdict is the
    // direct verdict, field for field.
    let zero = events(&[tangent(&[])]);
    let e = entail_ok(&engine, fiber_req(&p, &q, zero.clone(), zero)).await;
    assert_eq!(e["fiber_ssm"]["trivial_path"], false);
    assert!(e["fiber_ssm"]["arc_length"].as_f64().unwrap() > 0.0);
    assert_eq!(e["fiber_ssm"]["question_shift"], 0.0);
    for k in [
        "is_entailed",
        "confidence",
        "cone_angle",
        "busemann_depth_gain",
        "topic_shift",
    ] {
        assert_eq!(e[k], direct[k], "{k}");
    }
}

#[tokio::test]
async fn http_question_evidence_decides_and_shared_background_cancels() {
    let engine = mount(fiber_assets(0.0)).await;
    let p = point(0.3, 0.0);
    let radial = tangent(&[(0, 0.5)]);
    let background = tangent(&[(E0, 1.5), (E0 + 3, -0.5)]);
    let none = tangent(&[]);
    let plus = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };

    // Question-only radial evidence pushes q' deeper along p's ray: entailed,
    // where the unfiltered pair (p, p) is not.
    let clean = entail_ok(
        &engine,
        fiber_req(
            &p,
            &p,
            events(std::slice::from_ref(&none)),
            events(std::slice::from_ref(&radial)),
        ),
    )
    .await;
    assert_eq!(clean["mode"], "fiber_cross_diff_ssm");
    assert_eq!(clean["is_entailed"], true, "{clean}");
    assert_eq!(clean["fiber_ssm"]["direct"]["is_entailed"], false);
    assert!(clean["busemann_depth_gain"].as_f64().unwrap() > 0.0);

    // The same evidence riding on a background both channels share: the
    // background cancels, the moved question point is bit-identical.
    let noisy = entail_ok(
        &engine,
        fiber_req(
            &p,
            &p,
            events(std::slice::from_ref(&background)),
            events(&[plus(&radial, &background)]),
        ),
    )
    .await;
    assert_eq!(noisy["is_entailed"], true, "{noisy}");
    assert_eq!(
        noisy["fiber_ssm"]["evidence_question"],
        clean["fiber_ssm"]["evidence_question"]
    );
    assert_eq!(noisy["confidence"], clean["confidence"]);

    // Control: the background only in the question channel is not cancelled
    // and pushes the topic past its tolerance: not entailed.
    let leaked = entail_ok(
        &engine,
        fiber_req(
            &p,
            &p,
            events(&[none]),
            events(&[plus(&radial, &background)]),
        ),
    )
    .await;
    assert_eq!(leaked["is_entailed"], false, "{leaked}");
    assert_eq!(leaked["factors"]["topic_aligned"], false);
}

#[tokio::test]
async fn http_transport_between_distinct_points_cancels_the_euclidean_background() {
    // p != q: the Euclidean factor is flat, so a shared topic vector cancels
    // exactly after transport; the hyperbolic part is rescaled by the
    // conformal ratio and does not.
    let engine = mount(fiber_assets(0.0)).await;
    let (p, q) = (point(0.3, 0.0), point(0.5, 0.01));
    let bg = tangent(&[(E0 + 1, 0.9)]);
    let e = entail_ok(
        &engine,
        fiber_req(&p, &q, events(std::slice::from_ref(&bg)), events(&[bg])),
    )
    .await;
    assert_eq!(e["fiber_ssm"]["diff_norms_inf"], json!([0.0]), "{e}");
    assert_eq!(e["fiber_ssm"]["evidence_question"], json!(q));

    let h = tangent(&[(1, 0.1)]);
    let e = entail_ok(
        &engine,
        fiber_req(&p, &q, events(std::slice::from_ref(&h)), events(&[h])),
    )
    .await;
    let d = e["fiber_ssm"]["diff_norms_inf"][0].as_f64().unwrap();
    assert!(d > 0.0, "hyperbolic transport rescales: {e}");
}

#[tokio::test]
async fn http_decaying_dynamics_run_the_matrix_exponential_branch() {
    // A = -I: every step takes the scaling-and-squaring branch. One event of
    // dt = 1 ms and input u gives h_1 = (1 - e^{-dt}) u.
    let engine = mount(fiber_assets(-1.0)).await;
    let p = point(0.3, 0.0);
    let u = tangent(&[(0, 0.5)]);
    let req = json!({"action": "entail", "entailment": {
        "passage": p, "question": p,
        "passage_events": [{"time_ns": 1_000_000, "input": tangent(&[])}],
        "question_events": [{"time_ns": 1_000_000, "input": u}],
        "window_start_ns": 0, "backend": "cpu_serial",
    }});
    let e = entail_ok(&engine, req).await;
    let f = &e["fiber_ssm"];
    assert_eq!(f["scan"]["backend"], "cpu_serial");
    let h = f["final_state_norm_inf"].as_f64().unwrap();
    let expect = 0.5 * (-(-1e-3f64).exp_m1());
    assert!((h - expect).abs() < 1e-12, "h = {h}, expect {expect}");
    assert!(f["question_shift"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn http_fiber_requests_fail_closed() {
    let p = point(0.3, 0.0);
    let one = events(&[tangent(&[(0, 0.1)])]);

    // Streams on a mount without dynamics: refused, never run as scheme 1.
    let plain = mount(plain_assets()).await;
    entail_refused(
        &plain,
        fiber_req(&p, &p, one.clone(), one.clone()),
        "BackendUnavailable",
    )
    .await;

    let engine = mount(fiber_assets(0.0)).await;
    let only_passage = json!({"action": "entail", "entailment": {
        "passage": p, "question": p, "passage_events": one, "window_start_ns": 0,
    }});
    entail_refused(&engine, only_passage, "InvalidParams").await;
    let no_start = json!({"action": "entail", "entailment": {
        "passage": p, "question": p, "passage_events": one, "question_events": one,
    }});
    entail_refused(&engine, no_start, "InvalidParams").await;
    let dangling = json!({"action": "entail", "entailment": {
        "passage": p, "question": p, "window_start_ns": 0,
    }});
    entail_refused(&engine, dangling, "InvalidParams").await;

    // Unequal lengths and a narrow input: fiber mismatch.
    let two = events(&[tangent(&[]), tangent(&[])]);
    entail_refused(
        &engine,
        fiber_req(&p, &p, one.clone(), two),
        "FiberMismatch",
    )
    .await;
    let narrow = json!([{"time_ns": 1_000_000_000u64, "input": vec![0.0; 127]}]);
    entail_refused(
        &engine,
        fiber_req(&p, &p, one.clone(), narrow),
        "FiberMismatch",
    )
    .await;
    // Shifted timestamps: no re-alignment.
    let late = json!([{"time_ns": 1_000_000_001u64, "input": tangent(&[])}]);
    entail_refused(
        &engine,
        fiber_req(&p, &p, one.clone(), late),
        "DomainViolation",
    )
    .await;
    // Past the fiber window budget: refused before any exponential runs.
    let long: Vec<Vec<f64>> = (0..257).map(|_| tangent(&[])).collect();
    entail_refused(
        &engine,
        fiber_req(&p, &p, events(&long), events(&long)),
        "BudgetExceeded",
    )
    .await;
    // A sphere-normal input is not a tangent vector at the point.
    let normal = events(&[tangent(&[(104, 0.3)])]);
    entail_refused(
        &engine,
        fiber_req(&p, &p, one.clone(), normal),
        "DomainViolation",
    )
    .await;
    // Evidence that throws q' out of the Poincare ball: no verdict.
    let huge = events(&[tangent(&[(0, 1e6)])]);
    let tiny = events(&[tangent(&[])]);
    let body = entail_refused_any(&engine, fiber_req(&p, &p, tiny, huge)).await;
    let code = body["result"]["rejection"]["code"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        ["DomainViolation", "NonFiniteState"].contains(&code.as_str()),
        "{body}"
    );

    // Assets whose dynamics are not 128 x 128 do not mount.
    let mut bad = plain_assets();
    bad["entailment"]["dynamics"] = json!({"a": [[0.0]], "b": [[1.0]]});
    let fresh = Arc::new(PolymorphicZeroEngine::new().with_bridge(None));
    let (status, body) = send(
        &fresh,
        "/v1/mounts",
        json!({"base_version": 1, "assets": bad, "reason": "bad dynamics"}),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
}

async fn entail_refused_any(engine: &Arc<PolymorphicZeroEngine>, body: Value) -> Value {
    let (status, body) = send(engine, "/message", body).await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");
    body
}

#[tokio::test]
async fn mcp_frames_reach_the_fiber_path() {
    let engine = mount(fiber_assets(0.0)).await;
    let server = McpServer {
        engine: Arc::clone(&engine),
        auth_token: None,
        bridge_required: false,
    };
    let p = point(0.3, 0.0);
    let args = fiber_req(
        &p,
        &p,
        events(&[tangent(&[])]),
        events(&[tangent(&[(0, 0.5)])]),
    );
    let mut buf = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "zero", "arguments": args},
    })
    .to_string()
    .into_bytes();
    buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
    let v: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut buf).await).unwrap();
    assert_eq!(v["result"]["isError"], false, "{v}");
    let e = &v["result"]["_meta"]["entailment"];
    assert_eq!(e["mode"], "fiber_cross_diff_ssm", "{v}");
    assert_eq!(e["is_entailed"], true, "{v}");
    assert_eq!(e["fiber_ssm"]["scan"]["steps"], 1);
}

// ---------------------------------------------------------------- review probes

#[test]
fn review_empty_gauge_must_be_refused() {
    let result = FiberCrossDiff::new(fid(1), fid(2), Matrix::identity(0));
    assert_eq!(result.unwrap_err(), Reject::FiberMismatch);
}

#[test]
fn review_rectangular_transport_must_be_refused() {
    // Parallel transport cannot map a 3-wide fiber onto a 2-wide one.
    let wide = [vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    assert_eq!(
        FiberCrossDiff::from_rows(fid(1), fid(2), &wide).unwrap_err(),
        Reject::FiberMismatch
    );
    let tall = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]]).unwrap();
    assert_eq!(
        FiberCrossDiff::new(fid(1), fid(2), tall).unwrap_err(),
        Reject::FiberMismatch
    );
}

#[tokio::test]
async fn review_intermediate_normal_state_must_be_refused() {
    // Sphere pole is e104; e105 is tangent. B sends tangent e105 into normal e104.
    let mut assets = fiber_assets(0.0);
    assets["entailment"]["dynamics"]["b"][104][105] = json!(1.0);
    assets["entailment"]["dynamics"]["b"][105][105] = json!(0.0);
    let engine = mount(assets).await;
    let p = point(0.3, 0.0);
    entail_refused(
        &engine,
        fiber_req(
            &p,
            &p,
            events(&[tangent(&[])]),
            events(&[tangent(&[(105, 0.25)])]),
        ),
        "DomainViolation",
    )
    .await;
    // h_1 = 0.25 e104 (normal), h_2 = 0.1 e0 (tangent): the final state alone
    // would pass, so the normal intermediate state must refuse the request.
    let req = fiber_req(
        &p,
        &p,
        events(&[tangent(&[]), tangent(&[])]),
        events(&[tangent(&[(105, 0.25)]), tangent(&[(105, -0.25), (0, 0.1)])]),
    );
    entail_refused(&engine, req, "DomainViolation").await;
}

#[tokio::test]
async fn review_gpu_backend_is_explicitly_refused() {
    let engine = mount(fiber_assets(0.0)).await;
    let p = point(0.3, 0.0);
    let mut req = fiber_req(
        &p,
        &p,
        events(&[tangent(&[])]),
        events(&[tangent(&[(0, 0.1)])]),
    );
    req["entailment"]["backend"] = json!("gpu");
    entail_refused(&engine, req, "BackendUnavailable").await;
}
