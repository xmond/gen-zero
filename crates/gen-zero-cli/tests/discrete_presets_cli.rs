//! `gen-zero entail` follows the preset the mount seals, and a width that is
//! not the sealed `dim()` exits 1 with a `FiberMismatch` rejection on stdout.

use serde_json::Value;
use std::process::Command;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../gen-zero-service/tests/fixtures/cognitive_assets_boolq_entailment.json"
);

/// The scheme 1 fixture with `topology` replaced, in a per-process temp file.
fn assets_file(topology: &str) -> std::path::PathBuf {
    let mut assets: Value =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    assets["entailment"]["topology"] = topology.into();
    let path = std::env::temp_dir().join(format!(
        "gen-zero-preset-{topology}-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, serde_json::to_vec(&assets).unwrap()).unwrap();
    path
}

/// Width `dim`, hyperbolic radius `r` at angle `theta`, sphere at the pole
/// `sphere_start` (the first sphere coordinate of the layout).
fn point(dim: usize, sphere_start: usize, r: f64, theta: f64) -> String {
    let mut v = vec![0.0; dim];
    v[0] = r * theta.cos();
    v[1] = r * theta.sin();
    v[sphere_start] = 1.0;
    serde_json::to_string(&v).unwrap()
}

fn entail(passage: &str, question: &str, assets: &std::path::Path) -> (Option<i32>, Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(["entail", "--passage", passage, "--question", question])
        .arg("--mount-assets")
        .arg(assets)
        .env("GENZERO_PYTHON_ENDPOINT", "off")
        .env_remove("GENZERO_MOUNT_ASSETS")
        .output()
        .expect("run gen-zero");
    eprintln!(
        "exit={:?}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    (out.status.code(), json)
}

#[test]
fn compact_64d_mount_answers_64_wide_and_refuses_128_wide() {
    let assets = assets_file("compact_64d");
    // compact_64d: H^32 x R^16 x S^15, sphere starts at 48.
    let (code, out) = entail(&point(64, 48, 0.3, 0.0), &point(64, 48, 0.8, 0.02), &assets);
    assert_eq!(code, Some(0), "{out}");
    let e = &out["meta"]["entailment"];
    assert_eq!(e["is_entailed"], true, "{out}");
    assert_eq!(e["topology"], "compact_64d");
    assert_eq!(e["dim"], 64);
    assert_eq!(
        (
            &e["hyperbolic_dim"],
            &e["euclidean_dim"],
            &e["spherical_dim"]
        ),
        (&32.into(), &16.into(), &16.into())
    );

    // A BoolQ 128-wide pair on the 64D mount: exit 1, standard rejection JSON.
    let (code, out) = entail(
        &point(128, 104, 0.3, 0.0),
        &point(128, 104, 0.8, 0.02),
        &assets,
    );
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["is_error"], true, "{out}");
    assert_eq!(out["rejection"]["code"], "FiberMismatch", "{out}");
    assert_eq!(out["rejection"]["http_status"], 400, "{out}");
    assert_eq!(
        out["rejection"]["detail"],
        "passage/question dimension mismatch: expected 64, got 128"
    );
    assert!(out["meta"].get("entailment").is_none(), "{out}");
    let _ = std::fs::remove_file(assets);
}

#[test]
fn extended_256d_mount_refuses_64_wide_and_unlisted_preset_fails_startup() {
    let assets = assets_file("extended_256d");
    // extended_256d: H^160 x R^48 x S^47, sphere starts at 208.
    let (code, out) = entail(
        &point(256, 208, 0.3, 0.0),
        &point(256, 208, 0.8, 0.02),
        &assets,
    );
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(out["meta"]["entailment"]["dim"], 256);
    let (code, out) = entail(&point(64, 48, 0.3, 0.0), &point(64, 48, 0.8, 0.02), &assets);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(out["rejection"]["code"], "FiberMismatch", "{out}");
    let _ = std::fs::remove_file(assets);

    // An unlisted preset never mounts: the CLI stops before any request.
    let bad = assets_file("dynamic_100d");
    let (code, out) = entail(&point(64, 48, 0.3, 0.0), &point(64, 48, 0.8, 0.02), &bad);
    assert_ne!(code, Some(0), "{out}");
    assert!(out.is_null(), "no outcome may be printed: {out}");
    let _ = std::fs::remove_file(bad);
}

#[test]
fn overflowing_alpha_weights_fail_startup_instead_of_printing_null() {
    // Each weight is finite; their sum overflows to `+inf`. Own file name:
    // `assets_file("compact_64d")` is shared with a test running in parallel.
    let mut assets: Value =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    assets["entailment"]["topology"] = "compact_64d".into();
    for k in ["alpha_h", "alpha_e", "alpha_s"] {
        assets["entailment"][k] = 1e308.into();
    }
    let bad = std::env::temp_dir().join(format!(
        "gen-zero-preset-overflow-alpha-{}.json",
        std::process::id()
    ));
    std::fs::write(&bad, serde_json::to_vec(&assets).unwrap()).unwrap();
    let (code, out) = entail(&point(64, 48, 0.3, 0.0), &point(64, 48, 0.8, 0.02), &bad);
    assert_ne!(code, Some(0), "{out}");
    assert!(out.is_null(), "no outcome may be printed: {out}");
    let _ = std::fs::remove_file(bad);
}
