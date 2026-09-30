# gen-zero-lod

Multi-scale Level of Detail (Lod) graph fusion: dynamic graph topology, the
epistemic lifecycle state machine, Spec 25 mixed-curvature product geometry,
Personalized PageRank flow, and Banach fixed-point confidence evolution.

## Architecture

- `error`: crate error type (`LodError`).
- `graph`: dynamic graph topology, graph geometry and epistemic lifecycle.
  Lock-free CSR snapshots (`ArcSwap<CsrGraph>`), a ticket-ordered pending edge
  buffer that each flush merges into the snapshot and drains, 2-stage HDC +
  manifold recall, and atomic checkpoints (`create_checkpoint` /
  `rollback_checkpoint` / `transact`). No MCTS virtual loss is involved: the
  planner's MCTS is sequential.
  - Geometry: a graph is built with `GeometryParams` (`LodGraph::with_geometry`;
    `LodGraph::new` is the unit geometry). Curvature and sphere radius fix the
    domain of every node coordinate; `alpha_h`, `alpha_e`, `alpha_s` weigh the
    recall distance. Node coordinates (`MixedCurvatureCoord`) are validated as
    points of the graph's `ProductManifold` (`H^4 x R^8 x S^3`).
  - Confidence: `evolve_epistemic_fixed_point(beta, tolerance, theta_lo,
    theta_hi)` iterates `c = (1 - beta) prior + beta P c`, where `P` is the
    row-normalized weight of the `DependsOn` / `CausalTransition` /
    `CoarseGrain` edges into each node. Axioms are held at 1, nodes refuted by evidence
    (`falsify_node`) at 0. The map contracts the max norm by `beta`, so graphs
    with cycles converge to one fixed point within
    `k_max = ceil(ln(tolerance (1 - beta) / ||c^1 - c^0||) / ln beta)` steps; a
    run that does not is `LodError::FixedPointDiverged` and commits nothing.
    Hysteresis moves statuses: below `theta_lo` is `Falsified` (entity revoked),
    above `theta_hi` is `Validated`, in between the status is kept.
    `retract_falsification` withdraws evidence; the next evolution returns the
    earlier confidences exactly.
  - Limits: the effect of a refutation on a dependent shrinks with the
    dependent's prior, its other dependencies and its distance from the refuted
    node. It is not a whole-subtree cascade. `beta` and the thresholds are
    caller parameters and are not calibrated on any data.
  - Coarse-graining: `coarse_grain_cluster(members, entity, coord, hdc)` inserts
    a summary node on the band its coordinate implies, which must be strictly
    coarser than every member, links each member to it with a `CoarseGrain`
    edge, sets the members' parent and flushes. `add_edge` refuses a
    `CoarseGrain` edge that does not go from a finer band to a coarser one.
    `zoom_node(id, In | Out)` moves one band through `LodBand::zoom_in` /
    `zoom_out`; `migrate_band_to_coord(id)` steps to the band the coordinate
    implies. Both keep the coarse-grain order. Checkpoints restore bands and
    parents.
- `manifold`: Spec 25 mixed-curvature product manifolds,
  `M = H_{-c}^{d_h} x R^{d_e} x S_R^{d_s}` with product metric
  `g = alpha_h g_H + alpha_e g_E + alpha_s g_S`.
- `node`: node representation and epistemic state machine. 4-tier Lod bands
  (0..3), 256-bit HDC binary fingerprints, and a strict epistemic verification
  status machine. `band_from_scale(t)` splits the coarse-graining scale
  `t >= 0` into four intervals of width `rho_max / 4`;
  `LodNode::derive_band_from_coord` takes `t = rho_max - rho`, where
  `rho = 2 artanh(sqrt(c) ||x_H||)` is the normalized hyperbolic depth and
  `rho_max ~ 10.597` the deepest the chart's boundary floor allows. The origin
  is `Lod3Systemic`, the boundary `Lod0Atomic`: a convention (general concepts
  near the origin), not a learned fact.
- `ppr`: Personalized PageRank sparse flow engine. Lock-free, CSR-based sparse
  power iteration for context diffusion, `p_{t+1} = (1-alpha) * W_semantic * p_t + alpha * e_seed`.
- `semiring`: learned relation semiring (Spec 24 §8.6.2). `P(R)` under
  `⊕ = ∪` is the additive structure; `⊗_g` is a partial operation lifted from a
  learned table, with a missing entry contributing `∅`. The soft extension:
  `SoftResultSet` is a distribution over relations with an explicit unclosed
  bucket, `compose_soft` is `(a ⊗_g b)(r) = Σ a(r1) b(r2) T_soft(r1, r2, g → r)`,
  and `fold_chain_soft(chain, genders, tau_h)` left-folds and refuses when the
  entropy exceeds `tau_h`, the unclosed mass is at least the top relation's, or
  the top two tie. The product is multilinear, so differentiable in its inputs,
  but no gradient is computed and `tau_h` is not calibrated.
- `weighted`: weighted chart fold: energy-scored relation composition
  (`E = -ln p`, lower is better, composition is addition) over the semiring
  table.

## Key exports

- `LodError`: crate error type.
- `BufferedEdge`, `CsrGraph`, `EdgeType`, `FlushReport`, `GraphCheckpoint`,
  `LodGraph`, `PprRanking`, `FixedPointReport`, `StatusTransition`,
  `MAX_FIXED_POINT_STEPS`: graph topology, transactions and confidence evolution.
- `ContainmentCriteria`, `ContainmentScore`, `Digest`, `Epochs`, `FiberId`,
  `GeometryParams`, `Layout`, `MixedCurvatureCoord`, `Point`, `ProductGeometry`,
  `ProductManifold`, `Reject`, `GeometryResult`, `Tangent`, `TopologyPreset`,
  `Version`, `MAX_PRESET_DIM`: mixed-curvature product geometry.
- `hdc_hamming_distance_256`, `EpistemicStatus`, `LodBand`, `LodNode`,
  `ZoomDirection`, `band_from_scale`, `band_scale_width`, `max_chart_depth`,
  `normalized_depth`, `scale_from_depth`: node representation and the
  scale-to-band map.
- `compute_ppr_csr`, `PprScores`: Personalized PageRank over a CSR graph; bad
  inputs are errors, never clamped.
- `AssociativityReport`, `AssociativityViolation`, `FoldOutcome`, `Gender`,
  `RelId`, `RelationKey`, `RelationSemiring`, `ResultSet`, `SoftResultSet`,
  `SoftSetError`, `BUDGET_EXCEEDED_REASON`, `DEFAULT_CHART_STEP_BUDGET`,
  `SOFT_MASS_TOLERANCE`, `SOFT_TIE_TOLERANCE`: learned relation semiring.
- `AxiomWeightError`, `AxiomWeights`, `LogProbSemiring`, `TropicalSemiring`,
  `WeightedCandidate`, `WeightedFoldOutcome`, `WeightedSemiring`: weighted
  chart fold.

## Dependencies

- `gen-zero-core`: base types and traits.
