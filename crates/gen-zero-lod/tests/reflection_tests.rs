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
