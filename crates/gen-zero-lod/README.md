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
    row-normalized weight of the `DependsOn` / `CausalTransition` edges into
    each node. Axioms are held at 1, nodes refuted by evidence
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
- `manifold`: Spec 25 mixed-curvature product manifolds,
  `M = H_{-c}^{d_h} x R^{d_e} x S_R^{d_s}` with product metric
  `g = alpha_h g_H + alpha_e g_E + alpha_s g_S`.
- `node`: node representation and epistemic state machine. 4-tier Lod bands
  (0..3), 256-bit HDC binary fingerprints, and a strict epistemic verification
  status machine.
- `ppr`: Personalized PageRank sparse flow engine. Lock-free, CSR-based sparse
  power iteration for context diffusion, `p_{t+1} = (1-alpha) * W_semantic * p_t + alpha * e_seed`.
- `semiring`: learned relation semiring (Spec 24 §8.6.2). `P(R)` under
  `⊕ = ∪` is the additive structure; `⊗_g` is a partial operation lifted from a
  learned table, with a missing entry contributing `∅`.
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
- `hdc_hamming_distance_256`, `EpistemicStatus`, `LodBand`, `LodNode`: node
  representation.
- `compute_ppr_csr`, `PprScores`: Personalized PageRank over a CSR graph; bad
  inputs are errors, never clamped.
- `AssociativityReport`, `AssociativityViolation`, `FoldOutcome`, `Gender`,
  `RelId`, `RelationKey`, `RelationSemiring`, `ResultSet`,
  `BUDGET_EXCEEDED_REASON`, `DEFAULT_CHART_STEP_BUDGET`: learned relation
  semiring.
- `AxiomWeightError`, `AxiomWeights`, `LogProbSemiring`, `TropicalSemiring`,
  `WeightedCandidate`, `WeightedFoldOutcome`, `WeightedSemiring`: weighted
  chart fold.

## Dependencies

- `gen-zero-core`: base types and traits.
