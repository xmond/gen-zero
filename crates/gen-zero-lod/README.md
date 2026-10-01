# gen-zero-lod

Multi-scale Level of Detail (Lod) graph fusion: dynamic graph topology, the
epistemic lifecycle state machine, Spec 25 mixed-curvature product geometry,
Personalized PageRank flow, Banach fixed-point confidence evolution, and
text retrieval over payload-carrying nodes.

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
  - Confidence: `evolve_signed_epistemic_fixed_point_within(beta, gamma,
    tolerance, theta_lo, theta_hi, max_steps)` solves
    `c = (1 - beta) prior + beta max(0, P+ c - gamma P- c)`, clamped to [0, 1].
    `P+` is the row-normalized weight of the `DependsOn` / `CausalTransition` /
    `CoarseGrain` edges into each node (a node without them is supported by its
    own prior), `P-` the same over its `Falsifies` edges; `gamma = 0` leaves
    them out and gives the unsigned map. `evolve_epistemic_fixed_point` uses
    `gamma = 1` and the full step budget. Axioms are held at 1, nodes refuted by
    evidence (`falsify_node`) at 0.
  - Blocks: the dependency graph is split into strongly connected components
    (iterative Tarjan) and solved in topological order, sources first. A node
    outside every cycle takes one evaluation. A cycle is iterated alone, inputs
    held, and must contract: its Lipschitz bound `q = beta * max row sum of
    (P+ + gamma P-)` inside it must be below 1, else
    `LodError::FixedPointNotContractive` (`beta < 1 / (1 + gamma)` always
    suffices). It stops within
    `k_max = ceil(ln(tolerance (1 - q) / ||x^1 - x^0||) / ln q)` steps; a block
    that does not is `LodError::FixedPointDiverged`. Either error commits
    nothing. The report carries `scc_count`, `trivial_scc_count`,
    `cyclic_scc_count`, `max_scc_size`, `node_updates` and an error bound
    propagated from block to block.
    Hysteresis moves statuses: below `theta_lo` is `Falsified` (entity revoked),
    above `theta_hi` is `Validated`, in between the status is kept.
    `retract_falsification` withdraws evidence; the next evolution returns the
    earlier confidences exactly.
  - Admission: `add_edge` / `add_edges` (and every internal edge write)
    refuse, with `FixedPointNotContractive`, an edge that closes a cycle whose
    bound is not below 1 at `ADMISSION_BETA = 0.85`, `ADMISSION_GAMMA = 1`, and
    with `FixedPointTooSlow` one whose bound is below 1 but so close (about
    `q > 0.9998`) that `k_max` at the reflection tolerance passes
    `MAX_FIXED_POINT_STEPS`. A single edge that closes no cycle only dilutes
    its target's row and passes after a reachability walk; a batch takes one
    `O(V + E)` pass. `retract_falsification` is refused the same way when
    unpinning the node would close such a cycle.
  - Reflection: `reflect_failure` evolves at the admission parameters, so on an
    admitted graph every cycle contracts. If a cycle does not (the invariant
    broken some other way, including one too slow to finish), the falsifier
    gain inside that cycle alone is
    lowered to reach `q* = (1 + beta a_max) / 2` and the block is listed in
    `FixedPointReport::adapted_blocks`. The observation's own `Falsifies` edge
    is never inside a cycle and keeps full gain. When the evolution leaves the
    action above the threshold, the action is quarantined (manual revocation)
    and `ReflectionReport::revocation` says so.
  - Limits: the effect of a refutation on a dependent shrinks with the
    dependent's prior, its other dependencies and its distance from the refuted
    node. It is not a whole-subtree cascade. `beta`, `gamma` and the thresholds
    are caller parameters and are not calibrated on any data.
  - Coarse-graining: `coarse_grain_cluster(members, entity, coord, hdc)` inserts
    a summary node on the band its coordinate implies, which must be strictly
    coarser than every member, links each member to it with a `CoarseGrain`
    edge, sets the members' parent and flushes. `add_edge` refuses a
    `CoarseGrain` edge that does not go from a finer band to a coarser one.
    `zoom_node(id, In | Out)` moves one band through `LodBand::zoom_in` /
    `zoom_out`; `migrate_band_to_coord(id)` steps to the band the coordinate
    implies. Both keep the coarse-grain order. Checkpoints restore bands and
    parents.
  - Retrieval: `hybrid_rag_search(coord, hdc, top_k, crag_margin, ppr_alpha,
    ppr_iters)` runs three stages under one read lock: Hamming prefilter to
    `4 * top_k` live nodes, product-geodesic rerank (with the CRAG neighbor
    expansion) to `top_k` anchors, then PPR from the anchors seeded
    `1 / (1 + distance)`. Hits are every anchor plus up to `top_k` nodes reached
    only by diffusion, ordered by PPR score, each with its confidence, payload,
    `source_uri`, `timestamp_ns` and digest. `hybrid_rag_search_text` projects
    the query with the graph's own projector first (`LodGraph::project_text`).
    The Hamming scan is a linear `count_ones` loop; it is not benchmarked and
    uses a hardware POPCNT only when the build enables that target feature.
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
  near the origin), not a learned fact. A node may carry a `payload` (knowledge
  text, at most 64 KiB) with `source_uri`, `timestamp_ns` and its BLAKE3
  `payload_digest`; `LodNode::with_payload` computes the digest and the graph
  re-checks it at insert. No payload means an all-zero digest.
- `projection`: `TextEmbeddingProjector`, a deterministic lexical projection of
  text to a chart coordinate and a 256-bit SimHash fingerprint. Features are
  word unigrams, word bigrams and character trigrams (keyed BLAKE3, weight
  `1 + ln tf`, no IDF). The fingerprint is Charikar SimHash; the coordinate is a
  16-row random projection: `H^4` by stereographic projection from the
  hyperboloid into the ball, `S^3` as a unit direction, `R^8` through `tanh`.
  Texts are close when they share n-grams; it is not a semantic embedding, and
  the hyperbolic depth of a projected point carries no hierarchy. Blank text or
  text with no alphanumeric token is `LodError::EmptyInput`.
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
- `HybridRagResult`, `RagHit`, `RagDiffusion`, `HYBRID_PPR_TOLERANCE`: hybrid
  retrieval output.
- `TextEmbeddingProjector`, `PROJECTOR_VERSION`, `HDC_BITS`: text projection.
- `ContainmentCriteria`, `ContainmentScore`, `Digest`, `Epochs`, `FiberId`,
  `GeometryParams`, `Layout`, `MixedCurvatureCoord`, `Point`, `ProductGeometry`,
  `ProductManifold`, `Reject`, `GeometryResult`, `Tangent`, `TopologyPreset`,
  `Version`, `MAX_PRESET_DIM`: mixed-curvature product geometry.
- `hdc_hamming_distance_256`, `EpistemicStatus`, `LodBand`, `LodNode`,
  `ZoomDirection`, `band_from_scale`, `band_scale_width`, `max_chart_depth`,
  `normalized_depth`, `scale_from_depth`: node representation and the
  scale-to-band map. `payload_digest`, `MAX_PAYLOAD_BYTES`,
  `MAX_SOURCE_URI_BYTES`: node payloads.
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
