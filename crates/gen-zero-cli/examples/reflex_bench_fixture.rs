//! Write a seeded, random-weight reflex plugin archive for latency benchmarks.
//!
//! The weights are NOT trained: this fixture exists only so `gen-zero
//! reflex-bench` and the `reflex-patch-*` commands can be exercised without a
//! trained checkpoint. Predict latency depends on the shapes (input_dim, rank,
//! steps, candidates), not on the weight values, so the timings are
//! representative of that shape; the decisions are not meaningful.
//!
//! Usage:
//!   cargo run --release -p gen-zero-cli --example reflex_bench_fixture -- \
//!     --out /tmp/base.gzr --seed 1 [--task bench] [--input-dim 1024] \
//!     [--rank 16] [--steps 8] [--candidates 4]
//!
//! Two archives written with the same shape and different seeds are valid
//! `reflex-patch-create --base/--target` inputs.

use anyhow::{bail, Context};
use gen_zero_model::{ReflexHead, ReflexOperator, ReflexOperatorConfig, ReflexPlugin};
use std::path::PathBuf;

struct Args {
    out: PathBuf,
    seed: u32,
    task: String,
    input_dim: usize,
    rank: usize,
    steps: usize,
    candidates: usize,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut out = None;
    let mut args = Args {
        out: PathBuf::new(),
        seed: 1,
        task: "bench".into(),
        input_dim: 1024,
        rank: 16,
        steps: 8,
        candidates: 4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--seed" => args.seed = value.parse().context("--seed")?,
            "--task" => args.task = value,
            "--input-dim" => args.input_dim = value.parse().context("--input-dim")?,
            "--rank" => args.rank = value.parse().context("--rank")?,
            "--steps" => args.steps = value.parse().context("--steps")?,
            "--candidates" => args.candidates = value.parse().context("--candidates")?,
            other => bail!("unknown flag {other}"),
        }
    }
    args.out = out.context("--out is required")?;
    if args.seed == 0 {
        // xorshift32 has a fixed point at 0: every weight would be identical.
        bail!("--seed must be non-zero");
    }
    Ok(args)
}

struct Xorshift32(u32);

impl Xorshift32 {
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                ((x as f32) / (u32::MAX as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

fn main() -> anyhow::Result<()> {
    let a = parse_args()?;
    let (d, r) = (a.input_dim, a.rank);
    let mut rng = Xorshift32(a.seed);
    // Fan-in scaled uniform init keeps the recurrence numerically tame.
    let s_d = 1.0 / (d as f32).sqrt();
    let s_r = 1.0 / (r as f32).sqrt();
    let config = ReflexOperatorConfig {
        input_dim: d,
        lora_rank: r,
        alpha: 0.5,
        steps: a.steps,
        epsilon: 1e-6,
    };
    let operator = ReflexOperator::new(
        config,
        rng.vec(d * r, s_d),
        rng.vec(d * r, s_d),
        rng.vec(2 * r * r, s_r),
        rng.vec(r, s_r),
        rng.vec(2 * r * r, s_r),
        rng.vec(r, s_r),
        rng.vec(r * d, s_r),
    )?;
    let names: Vec<String> = (0..a.candidates).map(|k| format!("action_{k}")).collect();
    let head = ReflexHead::new(
        "main",
        d,
        rng.vec(a.candidates * d, s_d),
        rng.vec(a.candidates, s_d),
        names,
    )?;
    let plugin = ReflexPlugin::new(a.task, operator, vec![head], Some("main".into()))?;
    let bytes = plugin.to_bytes()?;
    std::fs::write(&a.out, &bytes).with_context(|| format!("write {}", a.out.display()))?;
    println!(
        "{}",
        serde_json::json!({
            "out": a.out,
            "bytes": bytes.len(),
            "input_dim": d,
            "lora_rank": r,
            "steps": a.steps,
            "candidates": a.candidates,
            "seed": a.seed,
            "trained": false,
        })
    );
    Ok(())
}
