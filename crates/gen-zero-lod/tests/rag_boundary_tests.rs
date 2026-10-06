use gen_zero_core::GraphFactProvider;
use gen_zero_lod::{
    ConflictDirection, ConflictEdge, DiffusionQuality, EdgeType, LodBand, LodError, LodGraph,
    LodNode, MixedCurvatureCoord, MAX_GRAPH_NODES,
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
        g.hybrid_rag_search_query(text, vector, None, 3, 0.0, 0.15, 200)
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

#[test]
fn rag_hits_carry_falsifies_edges_between_hits_in_both_directions() {
    let graph = LodGraph::new();
    let (coord, hdc) = graph.project_text("disputed claim").unwrap();
    let at_query = |label: &str, entity: u64| {
        LodNode::new(0, LodBand::Lod0Atomic, coord, label, entity).with_hdc_fingerprint(hdc)
    };
    let a = graph.add_node(at_query("claim A", 11)).unwrap();
    let b = graph.add_node(at_query("claim B", 12)).unwrap();
    let c = graph.add_node(at_query("claim C", 13)).unwrap();
    let (far_coord, far_hdc) = graph
        .project_text("tidal schedule of a distant harbour")
        .unwrap();
    let d = graph
        .add_node(
            LodNode::new(0, LodBand::Lod0Atomic, far_coord, "claim D", 14)
                .with_hdc_fingerprint(far_hdc),
        )
        .unwrap();
    // b falsifies a (committed); a falsifies c (still pending); c only
    // validates b, which is no conflict; d falsifies a but is never a hit.
    graph.add_edge(b, a, EdgeType::Falsifies, 0.7).unwrap();
    graph.add_edge(c, b, EdgeType::Validates, 1.0).unwrap();
    graph.add_edge(d, a, EdgeType::Falsifies, 0.9).unwrap();
    graph.flush_edges_to_csr().unwrap();
    graph.add_edge(a, c, EdgeType::Falsifies, 0.3).unwrap();
    assert_eq!(graph.pending_edge_count(), 1);

    let result = graph
        .hybrid_rag_search(&coord, &hdc, 3, 0.0, 0.15, 200)
        .unwrap();
    let hit = |id: u32| {
        result
            .hits
            .iter()
            .find(|h| h.node_id == id)
            .unwrap_or_else(|| panic!("node {id} missing from {:?}", result.hits))
    };
    assert!(
        result.hits.iter().all(|h| h.node_id != d),
        "{:?}",
        result.hits
    );
    let edge = |counterpart: u32, entity: u64, direction, weight| ConflictEdge {
        counterpart_node_id: counterpart,
        counterpart_entity_id: entity,
        direction,
        weight,
    };
    let mut a_edges = hit(a).conflict_edges.clone();
    a_edges.sort_by_key(|e| e.counterpart_node_id);
    assert_eq!(
        a_edges,
        vec![
            edge(b, 12, ConflictDirection::FalsifiedBy, 0.7),
            edge(c, 13, ConflictDirection::Falsifies, 0.3),
        ]
    );
    assert_eq!(
        hit(b).conflict_edges,
        vec![edge(a, 11, ConflictDirection::Falsifies, 0.7)]
    );
    assert_eq!(
        hit(c).conflict_edges,
        vec![edge(a, 11, ConflictDirection::FalsifiedBy, 0.3)]
    );
}

#[test]
fn diffusion_quality_is_converged_only_for_a_finite_residual_below_tolerance() {
    assert_eq!(
        DiffusionQuality::assess(true, 1e-7, 1e-6),
        DiffusionQuality::Converged
    );
    assert_eq!(
        DiffusionQuality::assess(false, 1e-7, 1e-6),
        DiffusionQuality::Degraded
    );
    // A `converged` flag that contradicts its own residual is not trusted.
    for residual in [1e-6, 0.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            DiffusionQuality::assess(true, residual, 1e-6),
            DiffusionQuality::Degraded,
            "{residual}"
        );
    }
}

#[test]
fn hybrid_rag_search_reports_a_capped_diffusion_as_degraded() {
    let graph = LodGraph::new();
    let (coord, hdc) = graph.project_text("pump").unwrap();
    let a = graph
        .add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "pump", 1).with_hdc_fingerprint(hdc))
        .unwrap();
    let b = graph.add_node(node(2)).unwrap();
    graph.add_edge(a, b, EdgeType::Semantic, 1.0).unwrap();
    graph.add_edge(b, a, EdgeType::Semantic, 1.0).unwrap();
    graph.flush_edges_to_csr().unwrap();

    let capped = graph
        .hybrid_rag_search(&coord, &hdc, 1, 0.0, 0.15, 1)
        .unwrap()
        .diffusion
        .unwrap();
    assert!(!capped.converged, "{capped:?}");
    assert_eq!(capped.quality, DiffusionQuality::Degraded);

    let full = graph
        .hybrid_rag_search(&coord, &hdc, 1, 0.0, 0.15, 500)
        .unwrap()
        .diffusion
        .unwrap();
    assert!(full.converged, "{full:?}");
    assert_eq!(full.quality, DiffusionQuality::Converged);
}

#[test]
fn hybrid_rag_search_does_not_filter_its_own_falsifies_conflicts() {
    // hybrid_rag_search is a boundary primitive: it ranks by PPR only. Keeping
    // or dropping a contradicted hit is `graph_rag`'s job one layer up (tested
    // against the full response shape in gen-zero-service's graph_verbs_tests);
    // this only pins down that the primitive keeps both sides and only marks
    // the conflict in `conflict_edges`, so a caller that ignores the mark fails
    // loudly in its own tests rather than finding the data pre-filtered here.
    let graph = LodGraph::new();
    let (coord, hdc) = graph.project_text("disputed claim").unwrap();
    let a = graph
        .add_node(
            LodNode::new(0, LodBand::Lod0Atomic, coord, "claim A", 1).with_hdc_fingerprint(hdc),
        )
        .unwrap();
    let b = graph
        .add_node(
            LodNode::new(0, LodBand::Lod0Atomic, coord, "claim B", 2).with_hdc_fingerprint(hdc),
        )
        .unwrap();
    graph.add_edge(b, a, EdgeType::Falsifies, 1.0).unwrap();
    graph.flush_edges_to_csr().unwrap();

    let result = graph
        .hybrid_rag_search(&coord, &hdc, 2, 0.0, 0.15, 200)
        .unwrap();
    let ids: Vec<u32> = result.hits.iter().map(|h| h.node_id).collect();
    assert!(ids.contains(&a) && ids.contains(&b), "{ids:?}");
    let hit_b = result.hits.iter().find(|h| h.node_id == b).unwrap();
    assert_eq!(hit_b.conflict_edges.len(), 1, "{hit_b:?}");
    assert_eq!(hit_b.conflict_edges[0].counterpart_node_id, a);
    assert_eq!(
        hit_b.conflict_edges[0].direction,
        ConflictDirection::Falsifies
    );
}
