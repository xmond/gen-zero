# B1001J T2 RAG hardening evidence

Workspace: `worktree-rag-hardening`
Branch: `feat/b1001j-t2-rag`
Base: `71167b7db1fb1758254ac7aadf4caf7cfd303598`

## Implemented

- `crates/gen-zero-lod/src/graph.rs:1303,1591,2973`: shared production cap
  `MAX_GRAPH_NODES = 1 << 20`, checked at every insertion and before reflection
  copies state. Missing action and missing evidence both count. Refusal returns
  `GraphCapacityExceeded`, publishes no staged nodes/edges, and quarantines the
  action through the existing error path. Snapshot loading checks the same cap
  (`src/graph/persistence.rs:198`). No LRU eviction or silent fallback.
- `crates/gen-zero-lod/src/node.rs:296`: the existing persisted reserved
  `source_uri = pipeline:failure-observation` is the internal evidence marker.
  It is excluded before recall/CRAG and from RAG diffusion hits
  (`src/graph.rs:2177,2407,2438`). Evidence remains present for confidence evolution.
  No serialization format change or heuristic matching of payloads was added.
- `crates/gen-zero-lod/src/graph.rs:1316,2390`: divide distances by the maximum
  among each track's recalled anchors, then deduplicate by minimum normalized
  distance and seed PPR with `1/(1+normalized_distance)`. Empty/all-zero tracks
  are defined; invalid distances return an error. Returned anchor distances
  are now dimensionless, including single-track searches. Service reports
  `distance_normalization: per_track_max` (`graph_verb.rs:845`).
- `crates/gen-zero-lod/src/projection.rs:12` and service `server.rs:577`:
  accurate deterministic random-sign/SimHash and real-valued chart-sketch
  descriptions; no learned manifold alignment, isometry or ranking guarantee.
  Actual API is `TextEmbeddingProjector::project_dense`; no separate
  `DenseEmbeddingProjector` type was introduced.
- `crates/gen-zero-lod/tests/rag_boundary_tests.rs:17,54,90`: real public inserts
  to the full 1,048,576-node cap; missing-target atomic refusal; repeated distinct
  observations at capacity; reserved evidence survives snapshot round-trip and
  is absent from recall/diffusion hits; text-track distances scaled by 100 leave
  hybrid ranking, deduplication and PPR weights unchanged.

## Commands and raw evidence

All command statuses below are captured from `$?`, without piping Cargo into
`head`/`tail`. Logs are original stdout/stderr, not reconstructed summaries.

```bash
cargo test -p gen-zero-lod -p gen-zero-service > regression-final.log 2>&1
result=$?
printf '%s\n' "$result" > regression-final.exit
exit "$result"
```

- `regression-initial.log` / `.exit`: first intermediate run, exit **101**;
  two assertions still expected raw distances. Corrected them to independently
  normalize the expected recall distances and compare against an independent
  PPR call. This was not an accepted final tree.
- `regression-second.log` / `.exit`: exit **0**, 537 passed, 0 failed, 7 ignored.
- `regression-final.log` / `.exit`: final source-tree regression, exit **0**, 537 passed, 0 failed, 7 ignored.

```bash
cargo test -p gen-zero-lod projection::tests::dense_ -- --nocapture
```

`projection-metrics.log` / `.exit`: exit **0**, four tests passed. Existing
measurement test bodies, trial counts and assertion thresholds were preserved;
only misleading names/comments changed. A worker's intermediate deletion of
measurement coverage was rejected and restored before accepted regression.
For 400 synthetic 256-dimensional triples per comparison:

| Input cosine comparison | Hamming ordering | Chart geodesic ordering |
| --- | ---: | ---: |
| 0.9 vs 0.3 | 1.000 | 0.998 |
| 0.8 vs 0.7 | 0.885 | 0.745 |

These are the measured fixed-fixture rates, not retrieval-quality benchmarks.
The supplied “barely above 50%” description is not supported by this fixture;
neither these rates nor the ideal random-hyperplane expectation establish
isometry or a per-query ordering guarantee.

Luna handled bounded documentation/comment edits in projection.rs, the LOD
README and server.rs. Main agent inspected the diff, rejected test removal,
implemented graph/service changes, and ran integrated verification.

## Unverified

- Seven existing ignored tests need Qwen model weights (one) or a live Python
  semantic scorer (six). They were not enabled or claimed as passed.
- No deployment/live production acceptance, long-duration workload memory
  benchmark, or real-corpus retrieval-quality evaluation was performed.
- Per-track maximum normalization removes multiplicative distance units only.
  It depends on the candidate set/top_k, maps a singleton nonzero distance to 1,
  and does not calibrate semantic relevance or cross-query scores.
- The node cap is not a total-memory bound for payloads, edges or other state.
  Reflection still copies/evolves the graph for accepted observations.

## Incomplete

No requested implementation remains pending. The changes and this evidence
are submitted together as a local commit on the requested branch. External
push/deployment is outside this task.
