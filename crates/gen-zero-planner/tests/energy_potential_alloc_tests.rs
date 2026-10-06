//! Zero heap allocation and per-call latency of the causal potential.
//!
//! This binary installs a counting global allocator (it affects this test
//! binary only). The counter is thread-local, so the test harness's own
//! threads do not disturb a measurement.
//!
//! The sub-microsecond bound is asserted in optimised builds only
//! (`cargo test --release`). A debug build asserts a loose bound instead and
//! says so on stdout; it never skips silently.

use gen_zero_core::ActionId;
use gen_zero_lod::{CausalPotential, MAX_CAUSAL_ATOMS};
use gen_zero_planner::triad::{CausalDag, CausalEdge, CausalNode};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

// SAFETY: forwards every call to the system allocator unchanged; the only
// extra work is a thread-local counter bump, which does not allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

/// The largest graph of the shared golden fixture (23 nodes, 16-node goal cone).
fn largest_fixture() -> (CausalDag, Vec<u64>) {
    let g: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/energy_potential_golden.json")).unwrap();
    let entry = g["dags"]
        .as_array()
        .unwrap()
        .iter()
        .max_by_key(|e| {
            (
                e["cone"].as_array().unwrap().len(),
                e["dag"]["n"].as_u64().unwrap(),
            )
        })
        .unwrap();
    let d = &entry["dag"];
    let n = d["n"].as_u64().unwrap() as usize;
    let nodes: Vec<CausalNode> = (0..n)
        .map(|i| CausalNode {
            action: ActionId(i as u32),
            cost: d["cost"][i].as_u64().unwrap() as u32,
            value: d["value"][i].as_f64().unwrap(),
            is_or: d["is_or"][i].as_bool().unwrap(),
        })
        .collect();
    let mut edges = Vec::new();
    for c in 0..n {
        for p in d["parents"][c].as_array().unwrap() {
            edges.push(CausalEdge {
                parent: ActionId(p.as_u64().unwrap() as u32),
                child: ActionId(c as u32),
            });
        }
    }
    let target = ActionId(d["target"].as_u64().unwrap() as u32);
    let dag = CausalDag::new(nodes, &edges, target, d["budget"].as_u64().unwrap() as u32).unwrap();
    let done: Vec<u64> = entry["closure"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p[0].as_u64().unwrap())
        .collect();
    assert_eq!(entry["cone"].as_array().unwrap().len(), 16);
    (dag, done)
}

#[test]
fn potential_construction_closure_and_bias_never_allocate() {
    let (dag, done) = largest_fixture();
    let all = (1_u64 << dag.len()) - 1;
    let mut out = [0.0_f64; MAX_CAUSAL_ATOMS];
    let before = allocs();
    let pot = CausalPotential::new(&dag).unwrap();
    let built = allocs();
    let mut acc = 0.0;
    for &m in &done {
        acc += pot.closure(m);
        acc += pot.delta_phi(m, (m.count_ones() as usize) % dag.len());
        pot.bias_into(m, all & !m, 2.0, &mut out).unwrap();
        acc += out[0];
    }
    let after = allocs();
    black_box(acc);
    assert_eq!(
        built - before,
        0,
        "construction: {} heap allocations",
        built - before
    );
    assert_eq!(
        after - built,
        0,
        "closure/bias: {} heap allocations",
        after - built
    );
    // The same cone the DAG computes itself.
    assert_eq!(pot.cone(), dag.goal_cone());
    // The counter itself works: a Vec allocation is seen.
    let v = black_box(vec![1_u8; 64]);
    assert!(allocs() > after);
    drop(v);
}

#[test]
fn closure_is_sub_microsecond_in_release() {
    let (dag, done) = largest_fixture();
    let pot = CausalPotential::new(&dag).unwrap();
    let calls = 200_000_usize;
    // Warm-up, then the timed loop over the fixture's 1500 done-sets.
    let mut acc = 0.0;
    for &m in done.iter().cycle().take(10_000) {
        acc += pot.closure(black_box(m));
    }
    let t0 = Instant::now();
    for &m in done.iter().cycle().take(calls) {
        acc += pot.closure(black_box(m));
    }
    let ns = t0.elapsed().as_nanos() as f64 / calls as f64;
    black_box(acc);
    let release = !cfg!(debug_assertions);
    println!(
        "[energy-potential] closure on a 16-node goal cone: {ns:.1} ns/call over {calls} calls ({} build)",
        if release { "release" } else { "debug" }
    );
    if release {
        assert!(ns < 1_000.0, "closure took {ns:.1} ns, bound is 1000 ns");
    } else {
        println!("[energy-potential] debug build: asserting the loose 50 us bound; run --release for the 1 us bound");
        assert!(ns < 50_000.0, "closure took {ns:.1} ns in debug");
    }
}
