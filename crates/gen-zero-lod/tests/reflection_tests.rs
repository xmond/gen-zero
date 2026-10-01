use gen_zero_core::GraphFactProvider;
use gen_zero_lod::{
    EdgeType, EpistemicStatus, LodBand, LodError, LodGraph, LodNode, MixedCurvatureCoord,
    ReflectionRevocation,
};

fn node(entity: u64) -> LodNode {
    LodNode::new(
        0,
        LodBand::Lod0Atomic,
        MixedCurvatureCoord::origin(),
        "action",
        entity,
    )
    .with_prior(0.9)
}

#[test]
fn reflection_is_idempotent_and_atomic_across_concurrent_deposits() {
    let graph = std::sync::Arc::new(LodGraph::new());
    graph.add_node(node(7)).unwrap();
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let graph = graph.clone();
            std::thread::spawn(move || {
                graph
                    .reflect_failure(7, "observed action 7 terminate at step 1", i)
                    .unwrap()
                    .evidence
            })
        })
        .collect();
    let ids: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert!(ids.iter().all(|id| *id == ids[0]));
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.pending_edge_count(), 1);
    assert!(graph.is_revoked(7));
    assert!(graph.get_node(0).unwrap().confidence < 0.3);
}

/// The loop a <-> b with b also falsifying a is not a contraction at the
/// reflection's parameters (row a weighs 1 + 1, q = 1.7). Admission refuses the
/// edge that closes it, so the reflection afterwards evolves a contractive
/// graph, commits its evidence and revokes the action.
#[test]
fn admission_refuses_the_divergent_loop_and_reflection_then_succeeds() {
    let graph = LodGraph::new();
    let a = graph.add_node(node(7)).unwrap();
    let b = graph.add_node(node(8)).unwrap();
    graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
    graph.add_edge(b, a, EdgeType::DependsOn, 1.0).unwrap();
    assert!(matches!(
        graph.add_edge(b, a, EdgeType::Falsifies, 1.0),
        Err(LodError::FixedPointNotContractive { block_size: 2, .. })
    ));
    assert_eq!(graph.pending_edge_count(), 2);
    assert!(!graph.is_revoked(7) && !graph.is_revoked(8));

    let reflection = graph
        .reflect_failure(7, "model terminal diagnostic", 1)
        .unwrap();
    assert_eq!(reflection.target, a);
    assert_eq!(reflection.revocation, ReflectionRevocation::Evolution);
    assert!(reflection.evolution.adapted_blocks.is_empty());
    assert!(reflection.evolution.contraction < 1.0);
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.pending_edge_count(), 3);
    assert!(graph.is_revoked(7));
    assert!(graph.get_node(a).unwrap().confidence < 0.3);
}

/// Support from an axiom and a second falsifier that dilutes the observation to
/// a tenth of the action's `P-`: the evolution leaves the action at 0.9. The
/// reflection still commits its evidence and revokes the action, by an
/// explicit quarantine that it reports.
#[test]
fn reflection_that_evolution_cannot_revoke_quarantines_explicitly() {
    let graph = LodGraph::new();
    let a = graph.add_node(node(7)).unwrap();
    let s = graph
        .add_node(node(1).with_status(EpistemicStatus::Axiomatic))
        .unwrap();
    let f = graph.add_node(node(2).with_prior(0.0)).unwrap();
    graph.add_edge(s, a, EdgeType::DependsOn, 1.0).unwrap();
    graph.add_edge(f, a, EdgeType::Falsifies, 9.0).unwrap();

    let reflection = graph
        .reflect_failure(7, "model terminal diagnostic", 1)
        .unwrap();
    assert_eq!(reflection.revocation, ReflectionRevocation::Quarantine);
    // a = 0.15 * 0.9 + 0.85 * (1 - 1/10 * 1 - 9/10 * 0) = 0.9.
    assert!((reflection.target_confidence - 0.9).abs() < 1e-5);
    assert_eq!(graph.node_count(), 4);
    assert_eq!(
        graph
            .get_node(reflection.evidence)
            .unwrap()
            .payload
            .as_deref(),
        Some("model terminal diagnostic")
    );
    assert!(graph.is_revoked(7));
    // The same observation again reuses its evidence and is still a quarantine:
    // the label follows the action's status, not the revocation set it is in.
    let again = graph
        .reflect_failure(7, "model terminal diagnostic", 2)
        .unwrap();
    assert_eq!(again.evidence, reflection.evidence);
    assert_eq!(again.revocation, ReflectionRevocation::Quarantine);
    assert_eq!(graph.node_count(), 4);
    // A quarantine is a manual revocation: a later evolution does not lift it.
    graph
        .evolve_epistemic_fixed_point(0.85, 1e-6, 0.3, 0.6)
        .unwrap();
    assert!(graph.is_revoked(7));
    assert!(graph.planning_prior(7).is_err());
}

#[test]
fn axiom_conflict_is_an_error_not_a_fabricated_success() {
    let graph = LodGraph::new();
    graph
        .add_node(node(7).with_status(EpistemicStatus::Axiomatic))
        .unwrap();
    assert!(graph
        .reflect_failure(7, "model terminal diagnostic", 1)
        .is_err());
    assert!(graph.is_revoked(7));
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.get_node(0).unwrap().confidence, 1.0);
}

/// Two pairs that support each other, each node falsified from the other pair
/// with 0.1764588 of its `P-` (the rest from a node at 0). The difference mode
/// of the pairs decays at `q = beta (1 + 0.1764588) = 1 - 1e-5`: a contraction,
/// but from these priors an evolution at tolerance 1e-6 does not finish inside
/// `MAX_FIXED_POINT_STEPS`. Before the step check, admission let this graph in
/// and every later reflection, on any action, failed with `FixedPointDiverged`.
fn slow_pairs(graph: &LodGraph) -> Vec<(u32, u32, EdgeType, f32)> {
    let a = graph.add_node(node(1).with_prior(1.0)).unwrap();
    let b = graph.add_node(node(2).with_prior(1.0)).unwrap();
    let c = graph.add_node(node(3).with_prior(1.0 - 6.0e-5)).unwrap();
    let d = graph.add_node(node(4).with_prior(1.0 - 6.0e-5)).unwrap();
    let zero = graph.add_node(node(5).with_prior(0.0)).unwrap();
    let mut edges = vec![
        (b, a, EdgeType::DependsOn, 1.0),
        (a, b, EdgeType::DependsOn, 1.0),
        (d, c, EdgeType::DependsOn, 1.0),
        (c, d, EdgeType::DependsOn, 1.0),
    ];
    for (source, target) in [(c, a), (d, b), (a, c), (b, d)] {
        edges.push((zero, target, EdgeType::Falsifies, 8235412.0));
        edges.push((source, target, EdgeType::Falsifies, 1764588.0));
    }
    edges
}

#[test]
fn admission_refuses_a_cycle_too_slow_for_the_step_budget() {
    let graph = LodGraph::new();
    let edges = slow_pairs(&graph);
    match graph.add_edges(&edges) {
        Err(LodError::FixedPointTooSlow {
            block_size,
            contraction,
            k_max,
            max_steps,
        }) => {
            assert_eq!(block_size, 4);
            assert!(contraction < 1.0 && contraction > 0.9999);
            assert!(k_max > max_steps, "{k_max} vs {max_steps}");
        }
        other => panic!("expected FixedPointTooSlow, got {other:?}"),
    }
    assert_eq!(graph.pending_edge_count(), 0);
    graph.add_node(node(99)).unwrap();
    let reflection = graph
        .reflect_failure(99, "model terminal diagnostic", 1)
        .unwrap();
    assert_eq!(reflection.revocation, ReflectionRevocation::Evolution);
    assert!(graph.is_revoked(99));
}
