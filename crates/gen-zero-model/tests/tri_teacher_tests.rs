//! Tri-teacher adapter and pair decider.
//!
//! * `#[ignore]`d, real artifact: the trained PAWS adapter
//!   (`adapter_paws_tri.safetensors`, ~117 MB, not shipped in this repository)
//!   at `$GENZERO_TRI_TEACHER_ADAPTER` loads with the shapes its metadata
//!   declares, and `decide_embeddings` on the Python decider's own pooled
//!   states reproduces its similarities
//!   (`tests/fixtures/tri_teacher_paws_golden.json`, fp32 CPU reference).
//! * Synthetic artifacts written here: the math (LayerNorm+Linear heads,
//!   normalization, 0.5/0.3/0.2 mix, sigmoid), every fail-closed path of the
//!   loader and decider, and the full text path (tiny Qwen2 checkpoint +
//!   word-level tokenizer) including the LoRA merge and the gate verifier.
//! * `#[ignore]`d: end-to-end `decide` on the real Qwen2.5-0.5B weights
//!   against the Python golden; needs `GENZERO_QWEN_BASE_DIR` and
//!   `GENZERO_TRI_TEACHER_ADAPTER`.

use gen_zero_gate::{
    CausalVerifier, GateError, TeacherProjectionPair, TriTeacherProjections, TwoStageConfig,
    TwoStageDualTrackGateway,
};
use gen_zero_model::tri_teacher::{projection_cosine, MIN_PROJECTION_NORM};
use gen_zero_model::{
    calibrated_probability, decide_projections, ModelError, ProjectionHead, QwenModel, TeacherSlot,
    TeacherWeights, TriTeacherLoRAAdapter, TriTeacherPairDecider, TriTeacherProjector,
    DEFAULT_TRI_TEACHER_THRESHOLD, TRI_TEACHER_ADAPTER_FORMAT,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The trained adapter is too large to ship, so the tests that need it are
/// `#[ignore]`d and read its path from the same variable the service uses.
fn real_adapter() -> PathBuf {
    PathBuf::from(std::env::var("GENZERO_TRI_TEACHER_ADAPTER").expect(
        "GENZERO_TRI_TEACHER_ADAPTER must point at the trained adapter_paws_tri.safetensors",
    ))
}

fn golden() -> Value {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tri_teacher_paws_golden.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("golden fixture")).expect("golden json")
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

// ---------------------------------------------------------------- safetensors writer

/// One tensor to write: shape, dtype tag and raw little-endian bytes.
struct Raw {
    shape: Vec<usize>,
    dtype: &'static str,
    bytes: Vec<u8>,
}

fn raw_f32(shape: &[usize], values: &[f32]) -> Raw {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    Raw {
        shape: shape.to_vec(),
        dtype: "F32",
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn write_safetensors(
    path: &Path,
    meta: &BTreeMap<String, String>,
    tensors: &BTreeMap<String, Raw>,
) {
    let mut header = serde_json::Map::new();
    if !meta.is_empty() {
        header.insert("__metadata__".into(), json!(meta));
    }
    let mut offset = 0usize;
    let mut data = Vec::new();
    for (name, t) in tensors {
        header.insert(
            name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.len()]}),
        );
        offset += t.bytes.len();
        data.extend_from_slice(&t.bytes);
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// Deterministic, varied values in roughly [-0.8, 0.8].
fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 11) % 17) as f32 - 8.0) / 10.0)
        .collect()
}

// ---------------------------------------------------------------- synthetic adapter

struct Synth {
    config: Value,
    format: String,
    tensors: BTreeMap<String, Raw>,
}

const T_DIMS: [usize; 3] = [3, 2, 2];

/// Adapter for a student of width `d` whose v_proj outputs `kv` dims.
fn synth(d: usize, kv: usize, layers: usize) -> Synth {
    let rank = 2;
    let config = json!({
        "model_id_or_path": "synthetic",
        "rank": rank,
        "alpha": 4.0,
        "student_dim": d,
        "teacher_dims": {"405b": T_DIMS[0], "q72b": T_DIMS[1], "llama70b": T_DIMS[2]},
        "weights": {"405b": 0.5, "q72b": 0.3, "llama70b": 0.2},
        "lam": 0.0,
        "num_classes": 2,
        "tau": 0.07,
        "target_modules": ["q_proj", "v_proj"],
        "variants": {"405b": "a", "q72b": "b", "llama70b": "c"},
    });
    let mut t = BTreeMap::new();
    for (i, (slot, td)) in ["405b", "q72b", "llama70b"].iter().zip(T_DIMS).enumerate() {
        t.insert(
            format!("proj_{slot}.0.weight"),
            raw_f32(
                &[d],
                &pattern(d, i + 1)
                    .iter()
                    .map(|x| 1.0 + x)
                    .collect::<Vec<_>>(),
            ),
        );
        t.insert(
            format!("proj_{slot}.0.bias"),
            raw_f32(&[d], &pattern(d, i + 2)),
        );
        t.insert(
            format!("proj_{slot}.1.weight"),
            raw_f32(&[td, d], &pattern(td * d, i + 3)),
        );
        t.insert(
            format!("proj_{slot}.1.bias"),
            raw_f32(&[td], &pattern(td, i + 4)),
        );
        t.insert(
            format!("teacher_mean_{slot}"),
            raw_f32(&[td], &vec![0.0; td]),
        );
        t.insert(
            format!("teacher_std_{slot}"),
            raw_f32(&[td], &vec![1.0; td]),
        );
    }
    t.insert("task_head.0.weight".into(), raw_f32(&[d], &vec![1.0; d]));
    t.insert("task_head.0.bias".into(), raw_f32(&[d], &vec![0.0; d]));
    t.insert(
        "task_head.1.weight".into(),
        raw_f32(&[2, d], &pattern(2 * d, 9)),
    );
    t.insert("task_head.1.bias".into(), raw_f32(&[2], &[0.0, 0.0]));
    for l in 0..layers {
        for (target, out) in [("q_proj", d), ("v_proj", kv)] {
            let p = format!("backbone.layers.{l}.self_attn.{target}");
            t.insert(
                format!("{p}.lora_A"),
                raw_f32(&[rank, d], &pattern(rank * d, l + 5)),
            );
            t.insert(
                format!("{p}.lora_B"),
                raw_f32(&[out, rank], &pattern(out * rank, l + 7)),
            );
        }
    }
    Synth {
        config,
        format: TRI_TEACHER_ADAPTER_FORMAT.into(),
        tensors: t,
    }
}

impl Synth {
    fn write(&self, dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let meta = BTreeMap::from([
            ("format".to_string(), self.format.clone()),
            ("adapter_config".to_string(), self.config.to_string()),
        ]);
        write_safetensors(&path, &meta, &self.tensors);
        path
    }

    fn load(&self) -> Result<TriTeacherLoRAAdapter, ModelError> {
        let dir = tempfile::tempdir().unwrap();
        TriTeacherLoRAAdapter::load(&self.write(dir.path(), "a.safetensors"))
    }
}

fn artifact_err(r: Result<TriTeacherLoRAAdapter, ModelError>, needle: &str) {
    match r {
        Err(ModelError::TriTeacherArtifact(m)) => {
            assert!(m.contains(needle), "error {m:?} lacks {needle:?}")
        }
        Err(e) => panic!("wrong error kind: {e:?}"),
        Ok(_) => panic!("adapter accepted, expected an error containing {needle:?}"),
    }
}

// ---------------------------------------------------------------- real artifact

#[test]
#[ignore = "needs the trained adapter at $GENZERO_TRI_TEACHER_ADAPTER"]
fn real_adapter_loads_with_metadata_shapes() {
    let a = TriTeacherLoRAAdapter::load(&real_adapter()).expect("load real adapter");
    let c = a.config();
    assert_eq!((c.rank, c.alpha, c.student_dim), (16, 32.0, 896));
    assert_eq!(c.lora_scaling(), 2.0);
    assert_eq!(c.teacher_dims().unwrap(), [16384, 8192, 8192]);
    assert_eq!(c.target_modules, vec!["q_proj", "v_proj"]);
    let w = c.teacher_weights().unwrap();
    assert_eq!(w.as_array(), [0.5, 0.3, 0.2]);
    assert_eq!(a.num_layers(), 24);
    assert_eq!(a.lora().len(), 48);
    for l in a.lora() {
        assert_eq!(l.a.dims(), [16, 896]);
        let out = if l.target == gen_zero_model::LoraTarget::QProj {
            896
        } else {
            128
        };
        assert_eq!(l.b.dims(), [out, 16]);
    }
    for (slot, dim) in TeacherSlot::ALL.iter().zip([16384, 8192, 8192]) {
        let h = a.projector().head(*slot);
        assert_eq!((h.in_dim(), h.out_dim()), (896, dim), "{slot:?}");
    }
    let p = a.provenance();
    assert_eq!(p.sha256, golden()["adapter_sha256"].as_str().unwrap());
    assert_eq!(p.lora_pairs, 48);
    assert_eq!(
        p.dropped_training_tensors.len(),
        10,
        "{:?}",
        p.dropped_training_tensors
    );
    assert!(p
        .dropped_training_tensors
        .iter()
        .all(|n| n.starts_with("task_head.") || n.starts_with("teacher_")));
}

#[test]
#[ignore = "needs the trained adapter at $GENZERO_TRI_TEACHER_ADAPTER"]
fn real_adapter_reproduces_python_decisions_from_pooled_states() {
    let g = golden();
    let decider = TriTeacherPairDecider::from_adapter(
        TriTeacherLoRAAdapter::load(&real_adapter()).unwrap(),
        g["threshold"].as_f64().unwrap(),
    )
    .unwrap();
    assert!(!decider.has_encoder());
    let pairs = g["pairs"].as_array().unwrap();
    assert!(pairs.len() >= 6);
    let mut worst = 0f64;
    for p in pairs {
        let d = decider
            .decide_embeddings(&f32s(&p["u"]), &f32s(&p["v"]))
            .unwrap();
        for (k, got) in [
            ("sim_405b", d.sim_405b),
            ("sim_q72b", d.sim_q72b),
            ("sim_llama70b", d.sim_llama70b),
            ("tri_sim", d.tri_sim),
            ("p_same_meaning", d.p_same_meaning),
        ] {
            let want = p[k].as_f64().unwrap();
            let gap = (got - want).abs();
            worst = worst.max(gap);
            assert!(gap < 1e-5, "{} {k}: rust {got} python {want}", p["id"]);
        }
        assert_eq!(
            d.same_meaning,
            p["same_meaning"].as_bool().unwrap(),
            "{}",
            p["id"]
        );
    }
    println!("max |rust - python| over {} pairs: {worst:e}", pairs.len());
}

// ---------------------------------------------------------------- math

fn head(in_dim: usize, out_dim: usize, seed: usize) -> ProjectionHead {
    ProjectionHead::new(
        in_dim,
        out_dim,
        vec![1.0; in_dim],
        vec![0.0; in_dim],
        pattern(in_dim * out_dim, seed),
        pattern(out_dim, seed + 1),
    )
    .unwrap()
}

fn projector(weights: [f64; 3]) -> TriTeacherProjector {
    TriTeacherProjector::new(
        4,
        [head(4, 3, 1), head(4, 2, 2), head(4, 2, 3)],
        TeacherWeights::new(weights[0], weights[1], weights[2]).unwrap(),
    )
    .unwrap()
}

#[test]
fn projection_head_is_layernorm_then_linear() {
    let ln_w = vec![1.5, 0.5, 2.0, 1.0];
    let ln_b = vec![0.1, -0.2, 0.0, 0.3];
    let w = vec![1.0, 2.0, -1.0, 0.5, /* row 2 */ 0.0, -1.0, 1.0, 2.0];
    let b = vec![0.25, -0.5];
    let h = ProjectionHead::new(4, 2, ln_w.clone(), ln_b.clone(), w.clone(), b.clone()).unwrap();
    let p = TriTeacherProjector::new(
        4,
        [h.clone(), h.clone(), h],
        TeacherWeights::new(0.5, 0.3, 0.2).unwrap(),
    )
    .unwrap();
    let u = [1.0f32, 2.0, 3.0, 6.0];
    let out = p.project(&u, &u).unwrap();
    // Reference: torch LayerNorm (biased variance, eps 1e-5) then x W^T + b.
    let mean = u.iter().sum::<f32>() / 4.0;
    let var = u.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    let n: Vec<f32> = (0..4)
        .map(|i| (u[i] - mean) / (var + 1e-5).sqrt() * ln_w[i] + ln_b[i])
        .collect();
    let want: Vec<f32> = (0..2)
        .map(|r| (0..4).map(|c| n[c] * w[r * 4 + c]).sum::<f32>() + b[r])
        .collect();
    for slot in [&out.proj_405b, &out.proj_q72b, &out.proj_llama70b] {
        for (g, w) in slot.z1.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        assert_eq!(slot.z1, slot.z2);
    }
}

#[test]
fn cosine_is_scale_invariant_and_l2_normalized() {
    let a = [3.0f32, 4.0, 0.0];
    assert!((projection_cosine(&a, &a.map(|x| 7.5 * x), "t").unwrap() - 1.0).abs() < 1e-12);
    assert!((projection_cosine(&a, &a.map(|x| -0.01 * x), "t").unwrap() + 1.0).abs() < 1e-12);
    assert!(projection_cosine(&a, &[0.0, 0.0, 2.0], "t").unwrap().abs() < 1e-12);
    // <a/|a|, b/|b|> with |a| = 5, |b| = 13.
    let c = projection_cosine(&a, &[5.0, 12.0, 0.0], "t").unwrap();
    assert!((c - (15.0 + 48.0) / 65.0).abs() < 1e-12, "{c}");
}

fn pair(z1: &[f32], z2: &[f32]) -> TeacherProjectionPair {
    TeacherProjectionPair {
        z1: z1.to_vec(),
        z2: z2.to_vec(),
    }
}

/// Projections whose per-teacher cosines are exactly `c` (2-d unit vectors).
fn with_cosines(c: [f64; 3]) -> TriTeacherProjections {
    let p = |c: f64| pair(&[1.0, 0.0], &[c as f32, (1.0 - c * c).sqrt() as f32]);
    TriTeacherProjections {
        proj_405b: p(c[0]),
        proj_q72b: p(c[1]),
        proj_llama70b: p(c[2]),
    }
}

#[test]
fn tri_sim_is_the_05_03_02_weighted_sum() {
    let w = TeacherWeights::new(0.5, 0.3, 0.2).unwrap();
    for c in [
        [0.9, 0.8, 0.7],
        [1.0, -1.0, 0.0],
        [0.25, 0.5, 0.75],
        [-0.3, 0.95, 0.6],
    ] {
        let d = decide_projections(&with_cosines(c), w, 0.91).unwrap();
        let sims = [d.sim_405b, d.sim_q72b, d.sim_llama70b];
        for (g, want) in sims.iter().zip(c) {
            assert!((g - want).abs() < 1e-6, "{g} vs {want}");
        }
        let want = 0.5 * sims[0] + 0.3 * sims[1] + 0.2 * sims[2];
        assert!((d.tri_sim - want).abs() < 1e-15, "{} vs {want}", d.tri_sim);
    }
    // Each weight acts on its own teacher: only 405b agrees -> tri_sim = 0.5.
    let d = decide_projections(&with_cosines([1.0, 0.0, 0.0]), w, 0.91).unwrap();
    assert!((d.tri_sim - 0.5).abs() < 1e-7);
    let d = decide_projections(&with_cosines([0.0, 1.0, 0.0]), w, 0.91).unwrap();
    assert!((d.tri_sim - 0.3).abs() < 1e-7);
    let d = decide_projections(&with_cosines([0.0, 0.0, 1.0]), w, 0.91).unwrap();
    assert!((d.tri_sim - 0.2).abs() < 1e-7);
}

#[test]
fn weights_are_used_as_given_never_renormalized() {
    let w = TeacherWeights::new(1.0, 1.0, 1.0).unwrap();
    let d = decide_projections(&with_cosines([1.0, 1.0, 1.0]), w, 0.91).unwrap();
    assert!((d.tri_sim - 3.0).abs() < 1e-6, "{}", d.tri_sim);
}

#[test]
fn sigmoid_calibration_and_threshold_boundary() {
    let t = DEFAULT_TRI_TEACHER_THRESHOLD;
    assert_eq!(t, 0.91);
    assert_eq!(calibrated_probability(t, t), 0.5);
    let want = 1.0 / (1.0 + (-1.5f64).exp()); // 15 * 0.1
    assert!((calibrated_probability(t + 0.1, t) - want).abs() < 1e-15);
    assert!((calibrated_probability(t - 0.1, t) - (1.0 - want)).abs() < 1e-15);
    assert!(calibrated_probability(1.0, -1.0) > 0.999_999);
    assert!(calibrated_probability(-1.0, 1.0) < 1e-6);

    // Exactly at the threshold: same meaning, p = 0.5. Use a threshold equal
    // to the computed tri_sim so the boundary is hit bit-exactly.
    let w = TeacherWeights::new(0.5, 0.3, 0.2).unwrap();
    let probe = decide_projections(&with_cosines([0.8, 0.6, 0.4]), w, 0.0).unwrap();
    let at = decide_projections(&with_cosines([0.8, 0.6, 0.4]), w, probe.tri_sim).unwrap();
    assert!(at.same_meaning);
    assert_eq!(at.p_same_meaning, 0.5);
    let above =
        decide_projections(&with_cosines([0.8, 0.6, 0.4]), w, probe.tri_sim + 1e-9).unwrap();
    assert!(!above.same_meaning);
    assert!(above.p_same_meaning < 0.5);
    let below =
        decide_projections(&with_cosines([0.8, 0.6, 0.4]), w, probe.tri_sim - 1e-9).unwrap();
    assert!(below.same_meaning && below.p_same_meaning > 0.5);
}

// ---------------------------------------------------------------- fail closed: decider

fn input_err<T: std::fmt::Debug>(r: Result<T, ModelError>, needle: &str) {
    match r {
        Err(ModelError::TriTeacherInput(m)) => {
            assert!(m.contains(needle), "{m:?} lacks {needle:?}")
        }
        other => panic!("expected TriTeacherInput({needle}), got {other:?}"),
    }
}

#[test]
fn decider_rejects_bad_embeddings() {
    let d = TriTeacherPairDecider::new(projector([0.5, 0.3, 0.2]), 0.91).unwrap();
    let ok = [0.1f32, -0.2, 0.3, 0.4];
    assert!(d.decide_embeddings(&ok, &ok).is_ok());
    input_err(d.decide_embeddings(&ok[..3], &ok), "u has 3 dims");
    input_err(d.decide_embeddings(&ok, &[0.0; 5]), "v has 5 dims");
    input_err(d.decide_embeddings(&[0.1, f32::NAN, 0.3, 0.4], &ok), "u[1]");
    input_err(
        d.decide_embeddings(&ok, &[0.1, 0.2, f32::INFINITY, 0.4]),
        "v[2]",
    );
    input_err(
        d.decide_embeddings(&ok, &[0.1, 0.2, 0.3, f32::NEG_INFINITY]),
        "v[3]",
    );
    input_err(d.decide_embeddings(&[], &[]), "u has 0 dims");
}

#[test]
fn decider_rejects_bad_thresholds() {
    for t in [f64::NAN, f64::INFINITY, 1.5, -1.01] {
        input_err(
            TriTeacherPairDecider::new(projector([0.5, 0.3, 0.2]), t).map(|_| ()),
            "threshold",
        );
        input_err(
            decide_projections(
                &with_cosines([0.5; 3]),
                TeacherWeights::new(0.5, 0.3, 0.2).unwrap(),
                t,
            ),
            "threshold",
        );
    }
}

#[test]
fn bad_weights_are_refused() {
    for w in [
        [f64::NAN, 0.3, 0.2],
        [0.5, -0.1, 0.2],
        [0.0, 0.0, 0.0],
        [0.5, f64::INFINITY, 0.2],
    ] {
        assert!(
            matches!(
                TeacherWeights::new(w[0], w[1], w[2]),
                Err(ModelError::TriTeacherArtifact(_))
            ),
            "{w:?}"
        );
    }
}

#[test]
fn zero_or_non_finite_projection_fails_closed() {
    // A head with zero weight and zero bias maps everything to the origin.
    let zero =
        ProjectionHead::new(4, 2, vec![1.0; 4], vec![0.0; 4], vec![0.0; 8], vec![0.0; 2]).unwrap();
    let p = TriTeacherProjector::new(
        4,
        [head(4, 3, 1), zero, head(4, 2, 3)],
        TeacherWeights::new(0.5, 0.3, 0.2).unwrap(),
    )
    .unwrap();
    let d = TriTeacherPairDecider::new(p, 0.91).unwrap();
    match d.decide_embeddings(&[0.1, 0.2, 0.3, 0.4], &[0.4, 0.3, 0.2, 0.1]) {
        Err(ModelError::NumericalInstability(m)) => {
            assert!(m.contains("proj_q72b") && m.contains("below"), "{m}")
        }
        other => panic!("expected NumericalInstability, got {other:?}"),
    }
    let w = TeacherWeights::new(0.5, 0.3, 0.2).unwrap();
    let mut bad = with_cosines([0.5; 3]);
    bad.proj_llama70b.z2[0] = f32::NAN;
    assert!(matches!(
        decide_projections(&bad, w, 0.91),
        Err(ModelError::NumericalInstability(_))
    ));
    let mut tiny = with_cosines([0.5; 3]);
    tiny.proj_405b.z1 = vec![(MIN_PROJECTION_NORM / 10.0) as f32, 0.0];
    assert!(matches!(
        decide_projections(&tiny, w, 0.91),
        Err(ModelError::NumericalInstability(_))
    ));
    let mut ragged = with_cosines([0.5; 3]);
    ragged.proj_405b.z2.push(1.0);
    input_err(decide_projections(&ragged, w, 0.91), "proj_405b");
}

#[test]
fn projection_only_decider_refuses_text() {
    let d = TriTeacherPairDecider::new(projector([0.5, 0.3, 0.2]), 0.91).unwrap();
    assert!(matches!(
        d.decide("a", "b"),
        Err(ModelError::TriTeacherEncoderMissing(_))
    ));
    assert!(matches!(
        d.encode("a"),
        Err(ModelError::TriTeacherEncoderMissing(_))
    ));
    assert!(matches!(
        d.project_pair("a", "b"),
        Err(GateError::VerifierFailure(_))
    ));
}

#[test]
fn head_constructor_validates() {
    assert!(
        ProjectionHead::new(4, 2, vec![1.0; 3], vec![0.0; 4], vec![0.0; 8], vec![0.0; 2]).is_err()
    );
    assert!(
        ProjectionHead::new(4, 2, vec![1.0; 4], vec![0.0; 4], vec![0.0; 7], vec![0.0; 2]).is_err()
    );
    assert!(ProjectionHead::new(
        4,
        2,
        vec![1.0; 4],
        vec![0.0; 4],
        vec![f32::NAN; 8],
        vec![0.0; 2]
    )
    .is_err());
    assert!(ProjectionHead::new(0, 2, vec![], vec![], vec![], vec![0.0; 2]).is_err());
    // Heads that read a different width than the student are refused.
    assert!(TriTeacherProjector::new(
        5,
        [head(4, 3, 1), head(4, 2, 2), head(4, 2, 3)],
        TeacherWeights::new(0.5, 0.3, 0.2).unwrap()
    )
    .is_err());
}

// ---------------------------------------------------------------- fail closed: loader

#[test]
fn synthetic_adapter_loads_and_decides() {
    let a = synth(4, 2, 2).load().expect("synthetic adapter");
    assert_eq!(a.num_layers(), 2);
    assert_eq!(a.lora().len(), 4);
    assert_eq!(a.config().lora_scaling(), 2.0);
    let d = TriTeacherPairDecider::from_adapter(a, 0.91).unwrap();
    let r = d
        .decide_embeddings(&[0.1, 0.2, 0.3, 0.4], &[0.1, 0.2, 0.3, 0.4])
        .unwrap();
    assert!((r.tri_sim - 1.0).abs() < 1e-9 && r.same_meaning);
    assert_eq!(d.info().teacher_dims, T_DIMS);
}

#[test]
fn loader_refuses_wrong_format_and_config() {
    let mut s = synth(4, 2, 2);
    s.format = "gen_zero.causal_lora.v1".into();
    artifact_err(s.load(), "format");

    let mut s = synth(4, 2, 2);
    s.config["surprise"] = json!(1);
    artifact_err(s.load(), "unknown field");

    let mut s = synth(4, 2, 2);
    s.config["target_modules"] = json!(["q_proj", "k_proj"]);
    artifact_err(s.load(), "k_proj");

    let mut s = synth(4, 2, 2);
    s.config["weights"] = json!({"405b": 0.5, "q72b": 0.3});
    artifact_err(s.load(), "weights");

    let mut s = synth(4, 2, 2);
    s.config["rank"] = json!(0);
    artifact_err(s.load(), "rank");

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("junk.safetensors");
    std::fs::write(&p, b"\xff\xff\xff\xff\xff\xff\xff\x7fnope").unwrap();
    artifact_err(TriTeacherLoRAAdapter::load(&p), "implausible");
    std::fs::write(&p, b"abc").unwrap();
    artifact_err(TriTeacherLoRAAdapter::load(&p), "shorter");
    artifact_err(
        TriTeacherLoRAAdapter::load(&dir.path().join("missing")),
        "read",
    );
}

#[test]
fn loader_refuses_missing_extra_and_misshaped_tensors() {
    let mut s = synth(4, 2, 2);
    s.tensors.remove("proj_q72b.1.bias");
    artifact_err(s.load(), "missing");

    let mut s = synth(4, 2, 2);
    s.tensors
        .insert("proj_extra.0.weight".into(), raw_f32(&[4], &[1.0; 4]));
    artifact_err(s.load(), "unexpected");

    let mut s = synth(4, 2, 2);
    s.tensors
        .insert("proj_405b.1.weight".into(), raw_f32(&[2, 4], &[0.1; 8]));
    artifact_err(s.load(), "proj_405b.1.weight");

    let mut s = synth(4, 2, 2);
    s.tensors.insert(
        "backbone.layers.0.self_attn.q_proj.lora_A".into(),
        raw_f32(&[3, 4], &[0.1; 12]),
    );
    artifact_err(s.load(), "lora_A");

    // A layer gap (layers 0 and 2, no 1).
    let mut s = synth(4, 2, 3);
    s.tensors
        .retain(|k, _| !k.starts_with("backbone.layers.1."));
    artifact_err(s.load(), "0..N");

    // A layer that lacks one target.
    let mut s = synth(4, 2, 2);
    s.tensors
        .remove("backbone.layers.1.self_attn.v_proj.lora_A");
    artifact_err(s.load(), "missing");
}

#[test]
fn loader_refuses_non_finite_and_non_f32_values() {
    let mut s = synth(4, 2, 2);
    s.tensors.insert(
        "proj_llama70b.1.weight".into(),
        raw_f32(&[2, 4], &[0.1, 0.2, f32::NAN, 0.4, 0.5, 0.6, 0.7, 0.8]),
    );
    artifact_err(s.load(), "non-finite");

    let mut s = synth(4, 2, 2);
    s.tensors.insert(
        "backbone.layers.1.self_attn.v_proj.lora_B".into(),
        raw_f32(&[2, 2], &[0.1, f32::INFINITY, 0.0, 0.0]),
    );
    artifact_err(s.load(), "non-finite");

    // Training-only tensors are validated before they are dropped.
    let mut s = synth(4, 2, 2);
    s.tensors
        .insert("teacher_mean_q72b".into(), raw_f32(&[2], &[f32::NAN, 0.0]));
    artifact_err(s.load(), "teacher_mean_q72b");
    let mut s = synth(4, 2, 2);
    s.tensors
        .insert("teacher_std_405b".into(), raw_f32(&[3], &[1.0, 0.0, 1.0]));
    artifact_err(s.load(), "teacher_std_405b");

    let mut s = synth(4, 2, 2);
    s.tensors.insert(
        "proj_405b.0.bias".into(),
        Raw {
            shape: vec![4],
            dtype: "F16",
            bytes: vec![0; 8],
        },
    );
    artifact_err(s.load(), "expected F32");
}

// ---------------------------------------------------------------- full text path (tiny model)

const H: usize = 8;
const KV: usize = 4;
const LAYERS: usize = 2;
const VOCAB: usize = 32;
const WORDS: [&str; 8] = ["the", "dog", "chased", "cat", "a", "bird", "saw", "tree"];

/// A Qwen2-shaped checkpoint with hidden 8, 2 heads, 1 kv head. `patch`
/// replaces named tensors (used to write a pre-merged copy).
fn tiny_qwen(dir: &Path, patch: &BTreeMap<String, Vec<f32>>) {
    std::fs::create_dir_all(dir).unwrap();
    let config = json!({
        "model_type": "qwen2", "hidden_size": H, "num_attention_heads": 2,
        "num_hidden_layers": LAYERS, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "vocab_size": VOCAB, "intermediate_size": 16,
        "tie_word_embeddings": true,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut t = BTreeMap::new();
    let mut put = |name: String, shape: &[usize], seed: usize, scale: f32| {
        let n = shape.iter().product();
        let v = patch
            .get(&name)
            .cloned()
            .unwrap_or_else(|| pattern(n, seed).iter().map(|x| x * scale).collect());
        t.insert(name, raw_f32(shape, &v));
    };
    put("model.embed_tokens.weight".into(), &[VOCAB, H], 1, 1.0);
    put("model.norm.weight".into(), &[H], 2, 0.0);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let s = 10 * (l + 1);
        put(format!("{p}.input_layernorm.weight"), &[H], s, 0.0);
        put(
            format!("{p}.post_attention_layernorm.weight"),
            &[H],
            s + 1,
            0.0,
        );
        put(format!("{p}.self_attn.q_proj.weight"), &[H, H], s + 2, 0.5);
        put(format!("{p}.self_attn.k_proj.weight"), &[KV, H], s + 3, 0.5);
        put(format!("{p}.self_attn.v_proj.weight"), &[KV, H], s + 4, 0.5);
        put(format!("{p}.self_attn.q_proj.bias"), &[H], s + 5, 0.1);
        put(format!("{p}.self_attn.k_proj.bias"), &[KV], s + 6, 0.1);
        put(format!("{p}.self_attn.v_proj.bias"), &[KV], s + 7, 0.1);
        put(format!("{p}.self_attn.o_proj.weight"), &[H, H], s + 8, 0.5);
        put(format!("{p}.mlp.gate_proj.weight"), &[16, H], s + 9, 0.5);
        put(format!("{p}.mlp.up_proj.weight"), &[16, H], s + 10, 0.5);
        put(format!("{p}.mlp.down_proj.weight"), &[H, 16], s + 11, 0.5);
    }
    // RMSNorm weights near 1, not 0.
    for (name, raw) in t.iter_mut() {
        if (name.ends_with("layernorm.weight") || name == "model.norm.weight")
            && !patch.contains_key(name)
        {
            *raw = raw_f32(&[H], &[1.0; H]);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &BTreeMap::new(), &t);
    let vocab: serde_json::Map<String, Value> = WORDS
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), json!(i + 1)))
        .chain([("<unk>".to_string(), json!(0))])
        .collect();
    let tok = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    });
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
}

fn get_f32(s: &Synth, name: &str) -> Vec<f32> {
    s.tensors[name]
        .bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `W + (alpha / rank) * B @ A` computed by hand.
fn merged(
    w: &[f32],
    b: &[f32],
    a: &[f32],
    out: usize,
    rank: usize,
    inp: usize,
    scale: f32,
) -> Vec<f32> {
    let mut m = w.to_vec();
    for o in 0..out {
        for i in 0..inp {
            let d: f32 = (0..rank).map(|r| b[o * rank + r] * a[r * inp + i]).sum();
            m[o * inp + i] += scale * d;
        }
    }
    m
}

fn last_hidden(model: &QwenModel, ids: &[u32]) -> Vec<f32> {
    let (h, _) = model.forward(&[ids.to_vec()], None).unwrap();
    h.narrow(1, ids.len() - 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn lora_merge_equals_a_pre_merged_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let s = synth(H, KV, LAYERS);
    let adapter_path = s.write(dir.path(), "adapter.safetensors");
    let base = dir.path().join("base");
    tiny_qwen(&base, &BTreeMap::new());

    // Pre-merged copy, built independently of the code under test.
    let base_model = QwenModel::from_safetensors_dir(&base).unwrap();
    let mut patch = BTreeMap::new();
    for l in 0..LAYERS {
        for (target, out, seed) in [
            ("q_proj", H, 10 * (l + 1) + 2),
            ("v_proj", KV, 10 * (l + 1) + 4),
        ] {
            let p = format!("backbone.layers.{l}.self_attn.{target}");
            let w: Vec<f32> = pattern(out * H, seed).iter().map(|x| x * 0.5).collect();
            let m = merged(
                &w,
                &get_f32(&s, &format!("{p}.lora_B")),
                &get_f32(&s, &format!("{p}.lora_A")),
                out,
                2,
                H,
                2.0,
            );
            patch.insert(format!("model.layers.{l}.self_attn.{target}.weight"), m);
        }
    }
    let premerged = dir.path().join("premerged");
    tiny_qwen(&premerged, &patch);
    let premerged = QwenModel::from_safetensors_dir(&premerged).unwrap();

    let mut ours = QwenModel::from_safetensors_dir(&base).unwrap();
    TriTeacherLoRAAdapter::load(&adapter_path)
        .unwrap()
        .merge_into(&mut ours)
        .unwrap();

    let ids = [1u32, 2, 3, 1, 4];
    let (a, b, c) = (
        last_hidden(&ours, &ids),
        last_hidden(&premerged, &ids),
        last_hidden(&base_model, &ids),
    );
    let gap = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let moved = a
        .iter()
        .zip(&c)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(gap < 1e-5, "merged vs pre-merged gap {gap}");
    assert!(
        moved > 1e-3,
        "LoRA did not change the model (moved {moved})"
    );
}

#[test]
fn merge_refuses_a_mismatched_base() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    tiny_qwen(&base, &BTreeMap::new());
    let mut model = QwenModel::from_safetensors_dir(&base).unwrap();
    // Three adapter layers for a two-layer model.
    let a = synth(H, KV, 3).load().unwrap();
    artifact_err(a.merge_into(&mut model).map(|_| a.clone()), "layers");
    // Wrong v_proj width: rejected before any layer is touched.
    let before = last_hidden(&model, &[1, 2, 3]);
    let a = synth(H, KV + 1, LAYERS).load().unwrap();
    artifact_err(a.merge_into(&mut model).map(|_| a.clone()), "v_proj");
    assert_eq!(
        last_hidden(&model, &[1, 2, 3]),
        before,
        "a refused merge changed the model"
    );
}

#[test]
fn full_decider_runs_text_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = synth(H, KV, LAYERS).write(dir.path(), "adapter.safetensors");
    let base = dir.path().join("base");
    tiny_qwen(&base, &BTreeMap::new());
    let d = TriTeacherPairDecider::load(&adapter, &base, None, 0.91).unwrap();
    assert!(d.has_encoder());
    let info = d.info();
    let enc = info.encoder.as_ref().unwrap();
    assert_eq!(enc.pooling, "last_token_final_norm");
    assert_eq!(enc.tokenizer, base.join("tokenizer.json"));
    assert_eq!(info.adapter.as_ref().unwrap().lora_pairs, 4);

    // encode = last-token hidden state of the merged model.
    let mut merged_model = QwenModel::from_safetensors_dir(&base).unwrap();
    TriTeacherLoRAAdapter::load(&adapter)
        .unwrap()
        .merge_into(&mut merged_model)
        .unwrap();
    let u = d.encode("the dog chased the cat").unwrap();
    assert_eq!(u, last_hidden(&merged_model, &[1, 2, 3, 1, 4]));

    // decide(text) = decide_embeddings(encode, encode).
    let (s1, s2) = ("the dog chased the cat", "a bird saw the tree");
    let by_text = d.decide(s1, s2).unwrap();
    let by_emb = d
        .decide_embeddings(&d.encode(s1).unwrap(), &d.encode(s2).unwrap())
        .unwrap();
    assert_eq!(by_text, by_emb);
    let same = d.decide(s1, s1).unwrap();
    assert!((same.tri_sim - 1.0).abs() < 1e-9 && same.same_meaning);

    // Empty text is refused, not pooled from nothing.
    input_err(d.encode(""), "tokenizes to nothing");

    // Through the two-stage gateway: in the ambiguity band the gate calls this
    // verifier and must reach the same tri_sim (it computes in f32).
    let gate = TwoStageDualTrackGateway::new(TwoStageConfig::default()).unwrap();
    let ev = gate
        .decide_with_scores(
            "the dog chased the cat",
            "a bird saw",
            "the tree",
            1.0,
            0.5,
            &d,
        )
        .unwrap();
    assert!(ev.stage2_triggered);
    let direct = d
        .decide("the dog chased the cat", "a bird saw the tree")
        .unwrap();
    assert!((ev.tri_sim.unwrap() as f64 - direct.tri_sim).abs() < 1e-5);
    assert_eq!(ev.is_answerable, direct.same_meaning);
    assert!(ev
        .verifier_id
        .unwrap()
        .contains("tri-teacher-lora:adapter="));
}

#[test]
fn full_decider_refuses_gguf_and_missing_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = synth(H, KV, LAYERS).write(dir.path(), "adapter.safetensors");
    let gguf = dir.path().join("model.gguf");
    std::fs::write(&gguf, b"GGUF").unwrap();
    match TriTeacherPairDecider::load(&adapter, &gguf, None, 0.91) {
        Err(ModelError::TriTeacherArtifact(m)) => assert!(m.contains("GGUF"), "{m}"),
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!("GGUF base accepted"),
    }
    let base = dir.path().join("base");
    tiny_qwen(&base, &BTreeMap::new());
    std::fs::remove_file(base.join("tokenizer.json")).unwrap();
    assert!(matches!(
        TriTeacherPairDecider::load(&adapter, &base, None, 0.91),
        Err(ModelError::QwenLoad(_))
    ));
}

// ---------------------------------------------------------------- real weights (ignored)

/// End-to-end on the real Qwen2.5-0.5B HF checkpoint against the Python
/// golden. Run with `GENZERO_QWEN_BASE_DIR=<snapshot dir>
/// GENZERO_TRI_TEACHER_ADAPTER=<adapter> cargo test -p
/// gen-zero-model --test tri_teacher_tests -- --ignored --nocapture`.
#[test]
#[ignore = "needs Qwen2.5-0.5B HF weights at $GENZERO_QWEN_BASE_DIR and the adapter at $GENZERO_TRI_TEACHER_ADAPTER"]
fn real_weights_match_python_decider_end_to_end() {
    let base = std::env::var("GENZERO_QWEN_BASE_DIR")
        .expect("GENZERO_QWEN_BASE_DIR must point at the Qwen2.5-0.5B snapshot directory");
    let g = golden();
    let d = TriTeacherPairDecider::load(
        &real_adapter(),
        Path::new(&base),
        None,
        g["threshold"].as_f64().unwrap(),
    )
    .expect("load real decider");
    let info = d.info();
    assert_eq!(
        info.encoder.as_ref().unwrap().base_sha256,
        g["base_model_sha256"].as_str().unwrap()
    );
    let (mut worst_state, mut worst_sim) = (0f32, 0f64);
    for p in g["pairs"].as_array().unwrap() {
        for (key, text) in [("u", "sentence_1"), ("v", "sentence_2")] {
            let got = d.encode(p[text].as_str().unwrap()).unwrap();
            let want = f32s(&p[key]);
            let gap = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            worst_state = worst_state.max(gap);
        }
        let r = d
            .decide(
                p["sentence_1"].as_str().unwrap(),
                p["sentence_2"].as_str().unwrap(),
            )
            .unwrap();
        for (k, got) in [
            ("sim_405b", r.sim_405b),
            ("sim_q72b", r.sim_q72b),
            ("sim_llama70b", r.sim_llama70b),
            ("tri_sim", r.tri_sim),
        ] {
            let gap = (got - p[k].as_f64().unwrap()).abs();
            worst_sim = worst_sim.max(gap);
            assert!(gap < 1e-3, "{} {k}: rust {got} python {}", p["id"], p[k]);
        }
        assert_eq!(
            r.same_meaning,
            p["same_meaning"].as_bool().unwrap(),
            "{}",
            p["id"]
        );
        println!(
            "{} tri_sim rust {:.6} python {:.6}",
            p["id"],
            r.tri_sim,
            p["tri_sim"].as_f64().unwrap()
        );
    }
    println!("max pooled-state gap {worst_state:e}, max similarity gap {worst_sim:e}");
}

#[test]
fn demo_adapter_file_loads_and_verifies_embeddings() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let demo_path = repo_root.join("examples/weights/tri_teacher_demo.safetensors");
    if !demo_path.exists() {
        eprintln!(
            "skipping demo adapter test: file not present at {}",
            demo_path.display()
        );
        return;
    }
    let adapter = TriTeacherLoRAAdapter::load(&demo_path).expect("load demo adapter");
    assert_eq!(adapter.num_layers(), 24);
    assert_eq!(adapter.lora().len(), 48);
    let decider = TriTeacherPairDecider::from_adapter(adapter, 0.91).expect("create decider");
    let u = vec![0.1f32; 896];
    let v = vec![0.1f32; 896];
    let res = decider
        .decide_embeddings(&u, &v)
        .expect("decide embeddings");
    assert!((res.tri_sim - 1.0).abs() < 1e-5);
    assert!(res.same_meaning);
}
