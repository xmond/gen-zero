# gen-zero-lod

Multi-scale Level of Detail (Lod) graph fusion: dynamic graph topology, the
epistemic lifecycle state machine, Spec 25 mixed-curvature product geometry,
Personalized PageRank flow, and Pearl causal cascading pruning.

## Architecture

- `error`: crate error type (`LodError`).
- `graph`: dynamic graph topology, epistemic lifecycle, and Pearl causal
  cascading pruning. Lock-free CSR snapshots (`ArcSwap<CsrGraph>`), a
  ticket-ordered pending edge buffer that each flush merges into the snapshot
  and drains, 2-stage HDC + manifold recall, causal cascade pruning, and atomic
  checkpoints (`create_checkpoint` / `rollback_checkpoint` / `transact`). No
  MCTS virtual loss is involved: the planner's MCTS is sequential.
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
  `LodGraph`, `PprRanking`, `PruneOutcome`: graph topology and transactions.
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
