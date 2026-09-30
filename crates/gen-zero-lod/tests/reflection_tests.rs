use gen_zero_core::GraphFactProvider;
use gen_zero_lod::{EdgeType, EpistemicStatus, LodBand, LodGraph, LodNode, MixedCurvatureCoord};

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
                    .0
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

#[test]
fn noncontractive_evolution_returns_error_and_quarantines_without_partial_evidence() {
    let graph = LodGraph::new();
    let a = graph.add_node(node(7)).unwrap();
    let b = graph.add_node(node(8)).unwrap();
    graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
    graph.add_edge(b, a, EdgeType::DependsOn, 1.0).unwrap();
    graph.add_edge(b, a, EdgeType::Falsifies, 1.0).unwrap();
    let result = graph.reflect_failure(7, "model terminal diagnostic", 1);
    assert!(result.is_err());
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.pending_edge_count(), 3);
    assert!(graph.is_revoked(7));
    assert!(!graph.is_revoked(8));
    assert_eq!(graph.get_node(a).unwrap().confidence, 0.9);
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
