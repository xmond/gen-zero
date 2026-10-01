//! Incremental, content-addressed snapshots. CURRENT.sha256 is the sole commit
//! point; immutable blocks are synced before its atomic replacement. A failed
//! commit poisons a mounted graph because a post-rename fsync failure is ambiguous.
use super::*;
use bincode::Options;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

const VERSION: u32 = 1;
const CURRENT: &str = "CURRENT.sha256";
const NODE_CHUNK: usize = 128;

pub(super) struct Persistence {
    dir: PathBuf,
    _lock: File,
    failed: AtomicBool,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    geometry: GeometryParams,
    nodes: Vec<String>,
    state: String,
    csr: String,
    next_ticket: u64,
}

fn fail(e: impl std::fmt::Display) -> LodError {
    LodError::Persistence(e.to_string())
}
fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, LodError> {
    bincode::serialize(value).map_err(fail)
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, LodError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(bytes.len() as u64)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(fail)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn lock_dir(dir: &Path) -> Result<File, LodError> {
    // Sync every newly created ancestor, not only the final graph directory.
    // Otherwise a durable manifest could live below an unsynced parent entry.
    let mut created = Vec::new();
    let mut path = dir;
    while !path.try_exists().map_err(fail)? {
        created.push(path.to_path_buf());
        path = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
    }
    fs::create_dir_all(dir).map_err(fail)?;
    for path in created.iter().rev() {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        sync_dir(parent)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("LOCK"))
        .map_err(fail)?;
    fs2::FileExt::try_lock_exclusive(&file).map_err(fail)?;
    Ok(file)
}
fn sync_dir(dir: &Path) -> Result<(), LodError> {
    File::open(dir).and_then(|f| f.sync_all()).map_err(fail)
}
fn replace(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), LodError> {
    // All callers own LOCK, including crash recovery; a leftover temp file has
    // never been committed and may safely be overwritten.
    let tmp = dir.join("WRITE.tmp");
    let mut f = File::create(&tmp).map_err(fail)?;
    f.write_all(bytes).map_err(fail)?;
    f.sync_all().map_err(fail)?;
    fs::rename(&tmp, dir.join(name)).map_err(fail)?;
    sync_dir(dir)
}
fn read_block(dir: &Path, hash: &str) -> Result<Vec<u8>, LodError> {
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(fail("invalid block digest"));
    }
    let bytes = fs::read(dir.join(format!("{hash}.block"))).map_err(fail)?;
    if digest(&bytes) != hash {
        return Err(fail("block SHA-256 mismatch"));
    }
    Ok(bytes)
}
fn block<T: Serialize>(dir: &Path, value: &T) -> Result<String, LodError> {
    let bytes = encode(value)?;
    let hash = digest(&bytes);
    let path = dir.join(format!("{hash}.block"));
    if path.try_exists().map_err(fail)? {
        read_block(dir, &hash)?; // Never silently reuse damaged content.
    } else {
        replace(dir, &format!("{hash}.block"), &bytes)?;
    }
    Ok(hash)
}

impl LodGraph {
    /// Save one consistent snapshot; only changed content blocks are written.
    /// Serialization and hashing still visit the whole graph (not an O(delta) WAL).
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), LodError> {
        let _txn = self.txn_lock.lock();
        self.check_persistence()?;
        if let Some(p) = &self.persistence {
            if p.dir == dir {
                return self.persist_commit();
            }
        }
        let _lock = lock_dir(dir)?;
        self.write_snapshot(dir)
    }

    fn write_snapshot(&self, dir: &Path) -> Result<(), LodError> {
        let _flush = self.flush_lock.lock();
        let st = self.state.read();
        let csr = self.csr_snapshot.load_full();
        let mut metadata = st.clone();
        metadata.nodes.clear();
        // Checkpoint identities are process-local; they cannot cross restarts.
        metadata.discarded.clear();
        metadata.generation = 0;
        let nodes = st
            .nodes
            .chunks(NODE_CHUNK)
            .map(|chunk| block(dir, &chunk))
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = Manifest {
            version: VERSION,
            geometry: self.geometry(),
            nodes,
            state: block(dir, &metadata)?,
            csr: block(dir, csr.as_ref())?,
            next_ticket: self.ticket_counter.load(Ordering::Relaxed),
        };
        let bytes = encode(&manifest)?;
        let envelope = encode(&(digest(&bytes), bytes))?;
        replace(dir, CURRENT, &envelope)?;
        // After the commit is durable, old/uncommitted blocks are unreachable.
        let retained: HashSet<_> = manifest
            .nodes
            .iter()
            .chain([&manifest.state, &manifest.csr])
            .map(|h| format!("{h}.block"))
            .collect();
        for entry in fs::read_dir(dir).map_err(fail)? {
            let entry = entry.map_err(fail)?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".block") && !retained.contains(name.as_ref()) {
                fs::remove_file(entry.path()).map_err(fail)?;
            }
        }
        sync_dir(dir)
    }

    /// Load and validate every referenced block. No empty-graph fallback exists.
    pub fn load_from_dir(dir: &Path, geometry: GeometryParams) -> Result<Self, LodError> {
        let _lock = lock_dir(dir)?;
        Self::read_snapshot(dir, geometry)
    }

    fn read_snapshot(dir: &Path, geometry: GeometryParams) -> Result<Self, LodError> {
        let envelope = fs::read(dir.join(CURRENT)).map_err(fail)?;
        let (hash, bytes): (String, Vec<u8>) = decode(&envelope)?;
        if digest(&bytes) != hash {
            return Err(fail("manifest SHA-256 mismatch"));
        }
        let m: Manifest = decode(&bytes)?;
        if m.version != VERSION {
            return Err(fail("unsupported snapshot version"));
        }
        let graph = Self::with_geometry(geometry)?;
        if graph.geometry() != m.geometry {
            return Err(fail("snapshot geometry mismatch"));
        }
        let mut st: GraphState = decode(&read_block(dir, &m.state)?)?;
        if !st.nodes.is_empty() {
            return Err(fail("nodes in metadata block"));
        }
        for hash in &m.nodes {
            let nodes: Vec<LodNode> = decode(&read_block(dir, hash)?)?;
            st.nodes.extend(nodes);
        }
        let csr: CsrGraph = decode(&read_block(dir, &m.csr)?)?;
        if csr.num_nodes > st.nodes.len()
            || st.entity_index.len() != st.nodes.len()
            || !st.manual_revocations.is_subset(&st.revocations)
        {
            return Err(fail("snapshot state invariant"));
        }
        csr.validate()?;
        for (id, node) in st.nodes.iter().enumerate() {
            node.coord.to_point(&graph.manifold)?;
            node.validate_payload()?;
            if node.id as usize != id
                || st.entity_index.get(&node.entity_id) != Some(&node.id)
                || !node.prior.is_finite()
                || !(0.0..=1.0).contains(&node.prior)
                || !node.confidence.is_finite()
                || !(0.0..=1.0).contains(&node.confidence)
                || (node.status.is_falsified() && !st.revocations.contains(&node.entity_id))
                || (node.refuted && (!node.status.is_falsified() || node.confidence != 0.0))
                || (node.status == EpistemicStatus::Axiomatic && node.confidence != 1.0)
                || node
                    .parent_id
                    .is_some_and(|p| p as usize >= st.nodes.len() || p == node.id)
            {
                return Err(fail(format!("invalid restored node {id}")));
            }
        }
        let mut previous = 0;
        for edge in &st.edge_buffer {
            check_edge(st.nodes.len(), edge.source, edge.target, edge.weight)?;
            if edge.ticket <= previous || edge.ticket >= m.next_ticket {
                return Err(fail("invalid edge ticket sequence"));
            }
            previous = edge.ticket;
        }
        if m.next_ticket == 0 {
            return Err(fail("invalid next ticket"));
        }
        for &(a, b) in &st.validated_deps {
            if !st.entity_index.contains_key(&a) || !st.entity_index.contains_key(&b) {
                return Err(fail("invalid validated dependency"));
            }
        }
        *graph.state.write() = st;
        graph.csr_snapshot.store(Arc::new(csr));
        graph.ticket_counter.store(m.next_ticket, Ordering::Relaxed);
        Ok(graph)
    }

    /// Own a durable directory for the lifetime of this graph. An existing
    /// directory without a commit is refused unless it contains only LOCK.
    pub fn open_persistent(dir: &Path, geometry: GeometryParams) -> Result<Self, LodError> {
        Self::open_persistent_with(dir, geometry, |_| Ok(()))
    }

    /// Run the initializer only for a fresh directory, while holding its writer
    /// lock, before publishing the first snapshot. Failed seeds cannot leave a
    /// valid empty snapshot that bypasses the seed on the next startup.
    pub fn open_persistent_with(
        dir: &Path,
        geometry: GeometryParams,
        initialize: impl FnOnce(&LodGraph) -> Result<(), LodError>,
    ) -> Result<Self, LodError> {
        let lock = lock_dir(dir)?;
        let mut graph = if dir.join(CURRENT).try_exists().map_err(fail)? {
            Self::read_snapshot(dir, geometry)?
        } else {
            for entry in fs::read_dir(dir).map_err(fail)? {
                if entry.map_err(fail)?.file_name() != "LOCK" {
                    return Err(fail("missing manifest in nonempty persistence directory"));
                }
            }
            let graph = Self::with_geometry(geometry)?;
            initialize(&graph)?;
            graph.write_snapshot(dir)?;
            // Persist directory creation in its parent as well.
            if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
                sync_dir(parent)?;
            }
            graph
        };
        graph.persistence = Some(Persistence {
            dir: dir.to_path_buf(),
            _lock: lock,
            failed: AtomicBool::new(false),
        });
        Ok(graph)
    }

    /// Once durability is uncertain, callers must stop serving this instance.
    pub fn check_persistence(&self) -> Result<(), LodError> {
        if self
            .persistence
            .as_ref()
            .is_some_and(|p| p.failed.load(Ordering::Acquire))
        {
            Err(fail(
                "previous durable commit failed; restart and inspect snapshot",
            ))
        } else {
            Ok(())
        }
    }
    pub fn is_persistent(&self) -> bool {
        self.persistence.is_some()
    }
    pub(super) fn persist_commit(&self) -> Result<(), LodError> {
        if let Some(p) = &self.persistence {
            if let Err(e) = self.write_snapshot(&p.dir) {
                p.failed.store(true, Ordering::Release);
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn node(entity: u64) -> LodNode {
        LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            "fact",
            entity,
        )
        .with_payload(
            format!("knowledge {entity}"),
            Some("test:source".into()),
            123,
        )
        .unwrap()
    }
    fn same(a: &LodGraph, b: &LodGraph) {
        let x = a.state.read();
        let y = b.state.read();
        // Binary encoding compares every floating-point bit, HDC bit and payload byte.
        assert_eq!(encode(&x.nodes).unwrap(), encode(&y.nodes).unwrap());
        assert_eq!(
            encode(a.csr_snapshot.load().as_ref()).unwrap(),
            encode(b.csr_snapshot.load().as_ref()).unwrap()
        );
        assert_eq!(x.entity_index, y.entity_index);
        assert_eq!(x.edge_buffer, y.edge_buffer);
        assert_eq!(x.revocations, y.revocations);
        assert_eq!(x.manual_revocations, y.manual_revocations);
        assert_eq!(x.privileges, y.privileges);
        assert_eq!(x.validated_deps, y.validated_deps);
        assert_eq!(
            a.ticket_counter.load(Ordering::Relaxed),
            b.ticket_counter.load(Ordering::Relaxed)
        );
    }
    #[test]
    fn exact_snapshot_roundtrip_and_incremental_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::new();
        for i in 0..1024 {
            graph.add_node(node(i)).unwrap();
        }
        graph.add_edge(0, 1, EdgeType::DependsOn, 0.75).unwrap();
        graph.flush_edges_to_csr().unwrap();
        graph.add_edge(1, 2, EdgeType::Semantic, 0.125).unwrap();
        graph.falsify_node(0).unwrap();
        graph
            .evolve_epistemic_fixed_point(0.85, 1e-6, 0.2, 0.8)
            .unwrap();
        graph.revoke_entity(9999);
        graph.add_privilege(42, 8);
        graph.save_to_dir(dir.path()).unwrap();
        let files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| {
                let p = e.unwrap().path();
                let t = fs::metadata(&p).unwrap().modified().unwrap();
                (p, t)
            })
            .filter(|(p, _)| p.extension().is_some_and(|x| x == "block"))
            .collect();
        graph.save_to_dir(dir.path()).unwrap();
        for (p, t) in files {
            assert_eq!(fs::metadata(p).unwrap().modified().unwrap(), t);
        }
        let start = Instant::now();
        let loaded = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap();
        let elapsed = start.elapsed();
        eprintln!(
            "restore sample: nodes=1024 csr_edges=1 pending_edges=1 elapsed_us={}",
            elapsed.as_micros()
        );
        same(&graph, &loaded);
        let (_, before): (String, Vec<u8>) =
            decode(&fs::read(dir.path().join(CURRENT)).unwrap()).unwrap();
        let before: Manifest = decode(&before).unwrap();
        graph.falsify_node(200).unwrap();
        graph.save_to_dir(dir.path()).unwrap();
        let (_, after): (String, Vec<u8>) =
            decode(&fs::read(dir.path().join(CURRENT)).unwrap()).unwrap();
        let after: Manifest = decode(&after).unwrap();
        assert_eq!(before.csr, after.csr);
        assert_eq!(
            before
                .nodes
                .iter()
                .zip(&after.nodes)
                .filter(|(a, b)| a != b)
                .count(),
            1
        );
        same(
            &graph,
            &LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap(),
        );
        loaded.add_edge(2, 3, EdgeType::Semantic, 0.25).unwrap();
        assert!(
            loaded.state.read().edge_buffer.last().unwrap().ticket
                > graph.state.read().edge_buffer.last().unwrap().ticket
        );
    }
    #[test]
    fn deserialization_cannot_construct_invalid_csr() {
        for csr in [
            CsrGraph {
                num_nodes: usize::MAX,
                row_offsets: vec![0],
                col_indices: vec![],
                edge_weights: vec![],
                edge_types: vec![],
            },
            CsrGraph {
                num_nodes: 1,
                row_offsets: vec![0, 1],
                col_indices: vec![1],
                edge_weights: vec![1.0],
                edge_types: vec![EdgeType::Semantic],
            },
            CsrGraph {
                num_nodes: 1,
                row_offsets: vec![0, 1],
                col_indices: vec![0],
                edge_weights: vec![f32::NAN],
                edge_types: vec![EdgeType::Semantic],
            },
        ] {
            assert!(decode::<CsrGraph>(&encode(&csr).unwrap()).is_err());
        }
    }
    #[test]
    fn corruption_missing_blocks_versions_and_geometry_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::new();
        graph.add_node(node(1)).unwrap();
        graph.save_to_dir(dir.path()).unwrap();
        let original = fs::read(dir.path().join(CURRENT)).unwrap();
        fs::write(dir.path().join(CURRENT), b"broken").unwrap();
        assert!(LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).is_err());
        let (_, bytes): (String, Vec<u8>) = decode(&original).unwrap();
        fs::write(
            dir.path().join(CURRENT),
            encode(&("0".repeat(64), bytes.clone())).unwrap(),
        )
        .unwrap();
        assert!(LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).is_err());
        let mut m: Manifest = decode(&bytes).unwrap();
        m.version += 1;
        let bad = encode(&m).unwrap();
        fs::write(
            dir.path().join(CURRENT),
            encode(&(digest(&bad), bad)).unwrap(),
        )
        .unwrap();
        assert!(LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).is_err());
        fs::write(dir.path().join(CURRENT), &original).unwrap();
        let mut geometry = GeometryParams::UNIT;
        geometry.curvature = 2.0;
        assert!(LodGraph::load_from_dir(dir.path(), geometry).is_err());
        let file = dir.path().join(format!("{}.block", m.nodes[0]));
        let bytes = fs::read(&file).unwrap();
        fs::write(&file, b"corrupt").unwrap();
        assert!(LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).is_err());
        fs::remove_file(&file).unwrap();
        assert!(LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).is_err());
        fs::write(file, bytes).unwrap();
        same(
            &graph,
            &LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap(),
        );
    }
    #[test]
    fn durable_transaction_failure_poison_and_writer_exclusion() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("data/lodgraph");
        let graph = LodGraph::open_persistent(&dir, GeometryParams::UNIT).unwrap();
        assert!(LodGraph::open_persistent(&dir, GeometryParams::UNIT).is_err());
        graph.transact(|g| g.add_node(node(1))).unwrap();
        // An actual filesystem failure before publication, not a mocked writer.
        fs::create_dir(dir.join("WRITE.tmp")).unwrap();
        assert!(graph.transact(|g| g.add_node(node(2))).is_err());
        assert_eq!(graph.node_count(), 1);
        assert!(graph.check_persistence().is_err());
        assert!(graph.transact(|g| g.add_node(node(3))).is_err());
        drop(graph);
        fs::remove_dir(dir.join("WRITE.tmp")).unwrap();
        let restored = LodGraph::open_persistent(&dir, GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 1);
    }
    #[test]
    fn reflection_and_failed_reflection_quarantine_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        graph.reflect_failure(7, "observed failure", 123).unwrap();
        graph
            .transact(|g| g.add_node(node(8).with_status(EpistemicStatus::Axiomatic)))
            .unwrap();
        assert!(graph.reflect_failure(8, "axiom conflict", 124).is_err());
        assert!(graph.state.read().manual_revocations.contains(&8));
        drop(graph);
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert!(graph.state.read().revocations.contains(&7));
        assert!(graph.state.read().manual_revocations.contains(&8));
        assert_eq!(graph.pending_edge_count(), 1);
    }
    #[test]
    fn uncommitted_files_never_replace_a_committed_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::new();
        graph.save_to_dir(dir.path()).unwrap();
        fs::write(dir.path().join("WRITE.tmp"), b"interrupted write").unwrap();
        fs::write(
            dir.path().join(format!("{}.block", "0".repeat(64))),
            b"orphan",
        )
        .unwrap();
        let restored = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 0);
        fs::remove_file(dir.path().join(CURRENT)).unwrap();
        assert!(LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).is_err());
    }
}
