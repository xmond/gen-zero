use gen_zero_core::GraphFactProvider;
use gen_zero_lod::{
    EdgeType, LodBand, LodError, LodGraph, LodNode, MixedCurvatureCoord, MAX_GRAPH_NODES,
};

fn node(entity: u64) -> LodNode {
    LodNode::new(
        0,
        LodBand::Lod0Atomic,
        MixedCurvatureCoord::origin(),
        "",
        entity,
    )
}

#[test]
fn reflection_refuses_real_capacity_without_partial_insertion() {
    let graph = LodGraph::new();
    // Real public inserts up to the production limit: no test-only smaller cap
    // and no fabricated node count.
    for entity in 0..(MAX_GRAPH_NODES - 1) as u64 {
        graph.add_node(node(entity)).unwrap();
    }
    let absent = MAX_GRAPH_NODES as u32;
    assert!(matches!(
        graph.reflect_failure(absent, "needs target and evidence", 1),
        Err(LodError::GraphCapacityExceeded { additional: 2, .. })
    ));
    assert_eq!(graph.node_count(), MAX_GRAPH_NODES - 1);
    assert_eq!(graph.pending_edge_count(), 0);
    assert!(graph.is_revoked(u64::from(absent)));
    graph.add_node(node((MAX_GRAPH_NODES - 1) as u64)).unwrap();
    for payload in ["first observation", "different observation"] {
        assert!(matches!(
            graph.reflect_failure(7, payload, 2),
            Err(LodError::GraphCapacityExceeded {
                current: MAX_GRAPH_NODES,
                additional: 1,
                max: MAX_GRAPH_NODES
            })
        ));
        assert_eq!(graph.node_count(), MAX_GRAPH_NODES);
        assert_eq!(graph.pending_edge_count(), 0);
        assert!(graph.is_revoked(7));
        assert_eq!(graph.get_node(7).unwrap().confidence, 0.5);
    }
    assert!(matches!(
        graph.add_node(node(u64::MAX)),
        Err(LodError::GraphCapacityExceeded { .. })
    ));
}

#[test]
fn reflection_evidence_is_not_a_recall_or_diffusion_hit() {
    let graph = LodGraph::new();
    let knowledge = graph.add_node(node(1)).unwrap();
    let report = graph.reflect_failure(7, "internal diagnostic", 1).unwrap();
    let dir = tempfile::tempdir().unwrap();
    graph.save_to_dir(dir.path()).unwrap();
    let graph = LodGraph::load_from_dir(dir.path(), graph.geometry()).unwrap();
    let evidence = graph.get_node(report.evidence).unwrap();
    assert!(evidence.is_internal_evidence());
    // Force an incoming retrieval path as well: filtering only anchors is not enough.
    graph
        .add_edge(knowledge, report.evidence, EdgeType::Semantic, 1.0)
        .unwrap();
    graph.flush_edges_to_csr().unwrap();
    let result = graph
        .hybrid_rag_search(
            &evidence.coord,
            &evidence.hdc_fingerprint,
            10,
            1.0,
            0.15,
            200,
        )
        .unwrap();
    assert_eq!(result.searchable_nodes, 1);
    assert_eq!(result.anchors.len(), 1);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].node_id, knowledge);
    assert_eq!(
        graph.get_node(report.evidence).unwrap().payload.as_deref(),
        Some("internal diagnostic")
    );
    assert!(graph.is_revoked(7));
}

#[test]
fn hybrid_ranking_is_invariant_to_text_track_distance_units() {
    let query_text = "valve pressure measurement";
    let query_vector: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0).sin()).collect();
    let build = |scale: f32| {
        let graph = LodGraph::new();
        let (coord, hdc) = graph.project_text(query_text).unwrap();
        for (i, offset) in [2.0, 4.0, 8.0].into_iter().enumerate() {
            let mut chart = coord;
            chart.euclidean[0] += offset * scale;
            let embedding = (0..16).map(|j| ((j + i * 3) as f32 + 0.5).cos()).collect();
            graph
                .add_node(
                    LodNode::new(0, LodBand::Lod0Atomic, chart, "candidate", i as u64)
                        .with_hdc_fingerprint(hdc)
                        .with_embedding(embedding),
                )
                .unwrap();
        }
        graph.flush_edges_to_csr().unwrap();
        graph
    };
    let base = build(1.0);
    let scaled = build(100.0);
    let search = |g: &LodGraph, text, vector| {
        g.hybrid_rag_search_query(text, vector, 3, 0.0, 0.15, 200)
            .unwrap()
    };
    let text = search(&base, Some(query_text), None);
    let vector = search(&base, None, Some(query_vector.as_slice()));
    let first = search(&base, Some(query_text), Some(query_vector.as_slice()));
    let second = search(&scaled, Some(query_text), Some(query_vector.as_slice()));
    assert_eq!(first.anchors.len(), 3); // duplicate nodes are fused
    assert_eq!(
        first.hits.iter().map(|h| h.node_id).collect::<Vec<_>>(),
        second.hits.iter().map(|h| h.node_id).collect::<Vec<_>>()
    );
    for (a, b) in first.hits.iter().zip(&second.hits) {
        assert!((a.ppr_score - b.ppr_score).abs() < 1e-6);
        assert!((a.anchor_distance.unwrap() - b.anchor_distance.unwrap()).abs() < 1e-6);
        let track_distance = |r: &gen_zero_lod::HybridRagResult| {
            r.anchors.iter().find(|x| x.0 == a.node_id).unwrap().1
        };
        let expected = track_distance(&text).min(track_distance(&vector));
        assert!((a.anchor_distance.unwrap() - expected).abs() < 1e-6);
    }
    // All three chart offsets are nonzero; the maximum sets the unit.
    for ((_, d), expected) in text.anchors.iter().zip([0.25, 0.5, 1.0]) {
        assert!((d - expected).abs() < 1e-6);
    }
    let total: f32 = first.anchors.iter().map(|a| 1.0 / (1.0 + a.1)).sum();
    for hit in &first.hits {
        let expected = 1.0 / (1.0 + hit.anchor_distance.unwrap()) / total;
        assert!((hit.ppr_score - expected).abs() < 1e-6);
    }
}
