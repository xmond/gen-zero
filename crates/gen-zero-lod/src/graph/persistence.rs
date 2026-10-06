//! Incremental, content-addressed snapshots. CURRENT.sha256 is the sole commit
//! point; immutable blocks are synced before its atomic replacement.
//!
//! A commit writes only what changed since the last one: node chunks and the
//! CSR snapshot are shared copy-on-write with the live graph, so a chunk whose
//! pointer is the one last committed reuses its recorded digest without being
//! serialized, hashed or read. Blocks this process wrote or verified are
//! trusted by digest; any other file under that name is overwritten, never
//! read back. The entity and alias indexes are not stored: restore rebuilds
//! them from the nodes.
//!
//! A failure before the CURRENT rename leaves the committed snapshot and the
//! live graph as they were, so the caller just gets the error. A failed rename
//! or post-rename directory sync is ambiguous and poisons the graph. Deleting
//! blocks the new manifest no longer references is cleanup: a failure there is
//! logged and retried after the next commit, it never fails or poisons one.
use super::*;
use bincode::Options;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// 4: adds `operator` to every node (3: `embedder_space` in the metadata
/// block; 2: metadata without the derived entity and alias indexes). Node
/// blocks are positional bincode, so an older snapshot is refused, not guessed.
const VERSION: u32 = 4;
const CURRENT: &str = "CURRENT.sha256";

pub(super) struct Persistence {
    dir: PathBuf,
    _lock: File,
    failed: AtomicBool,
    committed: Mutex<Committed>,
}

/// What the last durable commit wrote, and the blocks on disk.
#[derive(Default)]
struct Committed {
    /// Node chunk and block digest, per manifest entry.
    chunks: Vec<(Arc<Vec<LodNode>>, String)>,
    csr: Option<(Arc<CsrGraph>, String)>,
    /// Blocks of the committed manifest.
    retained: HashSet<String>,
    /// Blocks this process wrote or verified; their digests are trusted.
    known: HashSet<String>,
    /// Unreferenced blocks still to delete.
    garbage: HashSet<String>,
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

/// The persisted part of [`GraphState`] besides the nodes. Same field order
/// as [`MetaIn`], so both have one encoding.
#[derive(Serialize)]
struct MetaOut<'a> {
    edge_buffer: &'a [BufferedEdge],
    revocations: &'a HashSet<u64>,
    manual_revocations: &'a HashSet<u64>,
    privileges: &'a HashMap<u64, Vec<u32>>,
    validated_deps: &'a HashSet<(u64, u64)>,
    embedding_dim: Option<usize>,
    embedder_space: &'a Option<String>,
}

#[derive(Deserialize)]
struct MetaIn {
    edge_buffer: Vec<BufferedEdge>,
    revocations: HashSet<u64>,
    manual_revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
    embedding_dim: Option<usize>,
    embedder_space: Option<String>,
}

fn fail(e: impl std::fmt::Display) -> LodError {
    LodError::Persistence(e.to_string())
}
fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, LodError> {
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
fn block_name(hash: &str) -> String {
    format!("{hash}.block")
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
/// Write and sync `bytes` to the temp file. Nothing is committed yet.
fn write_tmp(dir: &Path, bytes: &[u8]) -> Result<PathBuf, LodError> {
    // All callers own LOCK, including crash recovery; a leftover temp file has
    // never been committed and may safely be overwritten.
    let tmp = dir.join("WRITE.tmp");
    let mut f = File::create(&tmp).map_err(fail)?;
    f.write_all(bytes).map_err(fail)?;
    f.sync_all().map_err(fail)?;
    Ok(tmp)
}
fn replace(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), LodError> {
    let tmp = write_tmp(dir, bytes)?;
    fs::rename(&tmp, dir.join(name)).map_err(fail)?;
    sync_dir(dir)
}
fn read_block(dir: &Path, hash: &str) -> Result<Vec<u8>, LodError> {
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(fail("invalid block digest"));
    }
    let bytes = fs::read(dir.join(block_name(hash))).map_err(fail)?;
    if digest(&bytes) != hash {
        return Err(fail("block SHA-256 mismatch"));
    }
    Ok(bytes)
}
/// Blocks in `dir` that `retained` does not name: crash leftovers to delete.
fn orphans(dir: &Path, retained: &HashSet<String>) -> Result<HashSet<String>, LodError> {
    let mut found = HashSet::new();
    for entry in fs::read_dir(dir).map_err(fail)? {
        let name = entry.map_err(fail)?.file_name();
        let name = name.to_string_lossy();
        if let Some(hash) = name.strip_suffix(".block") {
            if !retained.contains(hash) {
                found.insert(hash.to_owned());
            }
        }
    }
    Ok(found)
}

impl Committed {
    /// Write `bytes` as a block unless a trusted block has that digest.
    fn put(
        &self,
        dir: &Path,
        bytes: &[u8],
        fresh: &mut HashSet<String>,
        attempted: &mut HashSet<String>,
    ) -> Result<String, LodError> {
        let hash = digest(bytes);
        if !self.known.contains(&hash) && !fresh.contains(&hash) {
            // A failed write may still have renamed the block into place.
            attempted.insert(hash.clone());
            replace(dir, &block_name(&hash), bytes)?;
            fresh.insert(hash.clone());
        }
        Ok(hash)
    }

    /// Delete unreferenced blocks. Cleanup only: a failure is logged and the
    /// block stays queued for the next commit.
    fn collect_garbage(&mut self, dir: &Path) {
        let mut removed = false;
        for hash in std::mem::take(&mut self.garbage) {
            match fs::remove_file(dir.join(block_name(&hash))) {
                Ok(()) => removed = true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(
                        dir = %dir.display(),
                        block = %hash,
                        error = %e,
                        "lodgraph: cannot delete unreferenced block; retrying after the next commit"
                    );
                    // Never trust a file whose deletion failed half way.
                    self.known.remove(&hash);
                    self.garbage.insert(hash);
                    continue;
                }
            }
            self.known.remove(&hash);
        }
        if removed {
            if let Err(e) = sync_dir(dir) {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "lodgraph: block deletions not synced; a crash may leave orphans for the next open"
                );
            }
        }
    }
}

impl Persistence {
    fn new(dir: &Path, lock: File, committed: Committed) -> Self {
        Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            failed: AtomicBool::new(false),
            committed: Mutex::new(committed),
        }
    }

    /// Durably commit `st`. Only chunks, CSR and metadata that differ from the
    /// last commit are serialized and written.
    pub(super) fn commit(
        &self,
        st: &GraphState,
        csr: &Arc<CsrGraph>,
        next_ticket: u64,
        geometry: GeometryParams,
    ) -> Result<(), LodError> {
        let mut c = self.committed.lock();
        let dir = self.dir.as_path();
        let mut fresh = HashSet::new();
        let mut attempted = HashSet::new();
        let staged = (|| {
            let mut chunks = Vec::with_capacity(st.nodes.chunks().len());
            for (i, chunk) in st.nodes.chunks().iter().enumerate() {
                let hash = match c.chunks.get(i) {
                    Some((committed, hash)) if Arc::ptr_eq(committed, chunk) => hash.clone(),
                    _ => c.put(dir, &encode(chunk.as_slice())?, &mut fresh, &mut attempted)?,
                };
                chunks.push((Arc::clone(chunk), hash));
            }
            let csr_hash = match &c.csr {
                Some((committed, hash)) if Arc::ptr_eq(committed, csr) => hash.clone(),
                _ => c.put(dir, &encode(csr.as_ref())?, &mut fresh, &mut attempted)?,
            };
            let meta = MetaOut {
                edge_buffer: &st.edge_buffer,
                revocations: &st.revocations,
                manual_revocations: &st.manual_revocations,
                privileges: &st.privileges,
                validated_deps: &st.validated_deps,
                embedding_dim: st.embedding_dim,
                embedder_space: &st.embedder_space,
            };
            let state = c.put(dir, &encode(&meta)?, &mut fresh, &mut attempted)?;
            let manifest = Manifest {
                version: VERSION,
                geometry,
                nodes: chunks.iter().map(|(_, hash)| hash.clone()).collect(),
                state,
                csr: csr_hash.clone(),
                next_ticket,
            };
            let bytes = encode(&manifest)?;
            let tmp = write_tmp(dir, &encode(&(digest(&bytes), bytes))?)?;
            Ok::<_, LodError>((manifest, chunks, csr_hash, tmp))
        })();
        // Blocks written are valid whatever happens next; unreferenced ones
        // are deleted after a later commit.
        c.known.extend(fresh.iter().cloned());
        let (manifest, chunks, csr_hash, tmp) = match staged {
            Ok(staged) => staged,
            Err(e) => {
                // CURRENT is untouched: the committed snapshot stands.
                let unreferenced: Vec<_> = attempted.difference(&c.retained).cloned().collect();
                c.garbage.extend(unreferenced);
                return Err(e);
            }
        };
        if let Err(e) = fs::rename(&tmp, dir.join(CURRENT))
            .map_err(fail)
            .and_then(|()| sync_dir(dir))
        {
            // The new manifest may or may not be durable.
            self.failed.store(true, Ordering::Release);
            return Err(e);
        }
        let retained: HashSet<String> = manifest
            .nodes
            .iter()
            .chain([&manifest.state, &manifest.csr])
            .cloned()
            .collect();
        let dropped: Vec<_> = c.retained.difference(&retained).cloned().collect();
        c.garbage.extend(dropped);
        c.garbage.extend(fresh);
        c.garbage.retain(|hash| !retained.contains(hash));
        c.retained = retained;
        c.chunks = chunks;
        c.csr = Some((Arc::clone(csr), csr_hash));
        c.collect_garbage(dir);
        Ok(())
    }
}

impl LodGraph {
    /// Save one consistent snapshot. Into this graph's own directory it is an
    /// incremental commit; any other directory receives every block.
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), LodError> {
        let _txn = self.txn_lock.lock();
        self.check_persistence()?;
        let st = self.state.read();
        let csr = self.csr_snapshot.load_full();
        let next_ticket = self.ticket_counter.load(Ordering::Relaxed);
        if let Some(p) = self.persistence.as_ref().filter(|p| p.dir == dir) {
            return p.commit(&st, &csr, next_ticket, self.geometry());
        }
        let lock = lock_dir(dir)?;
        let export = Persistence::new(
            dir,
            lock,
            Committed {
                garbage: orphans(dir, &HashSet::new())?,
                ..Committed::default()
            },
        );
        export.commit(&st, &csr, next_ticket, self.geometry())
    }

    /// Load and validate every referenced block. No empty-graph fallback exists.
    pub fn load_from_dir(dir: &Path, geometry: GeometryParams) -> Result<Self, LodError> {
        let _lock = lock_dir(dir)?;
        Ok(Self::read_snapshot(dir, geometry)?.0)
    }

    fn read_snapshot(dir: &Path, geometry: GeometryParams) -> Result<(Self, Committed), LodError> {
        let envelope = fs::read(dir.join(CURRENT)).map_err(fail)?;
        let (hash, bytes): (String, Vec<u8>) = decode(&envelope)?;
        if digest(&bytes) != hash {
            return Err(fail("manifest SHA-256 mismatch"));
        }
        let m: Manifest = decode(&bytes)?;
        if m.version != VERSION {
            return Err(fail(format!(
                "unsupported snapshot version {} (this build reads version {VERSION})",
                m.version
            )));
        }
        let graph = Self::with_geometry(geometry)?;
        if graph.geometry() != m.geometry {
            return Err(fail("snapshot geometry mismatch"));
        }
        let meta: MetaIn = decode(&read_block(dir, &m.state)?)?;
        let mut chunks = Vec::with_capacity(m.nodes.len());
        let mut total_nodes = 0usize;
        for hash in &m.nodes {
            let chunk: Vec<LodNode> = decode(&read_block(dir, hash)?)?;
            super::check_node_capacity(total_nodes, chunk.len())?;
            total_nodes += chunk.len();
            chunks.push(Arc::new(chunk));
        }
        let nodes = NodeChunks::from_chunks(chunks)?;
        let csr = Arc::new(decode::<CsrGraph>(&read_block(dir, &m.csr)?)?);
        if csr.num_nodes > nodes.len() || !meta.manual_revocations.is_subset(&meta.revocations) {
            return Err(fail("snapshot state invariant"));
        }
        csr.validate()?;
        let mut entity_index = HashMap::with_capacity(nodes.len());
        let mut alias_index: HashMap<String, Vec<u32>> = HashMap::new();
        for (id, node) in nodes.iter().enumerate() {
            node.coord.to_point(&graph.manifold)?;
            node.validate_payload()?;
            if node.id as usize != id
                || entity_index.insert(node.entity_id, node.id).is_some()
                || !node.prior.is_finite()
                || !(0.0..=1.0).contains(&node.prior)
                || !node.confidence.is_finite()
                || !(0.0..=1.0).contains(&node.confidence)
                || (node.status.is_falsified() && !meta.revocations.contains(&node.entity_id))
                || (node.refuted && (!node.status.is_falsified() || node.confidence != 0.0))
                || (node.status == EpistemicStatus::Axiomatic && node.confidence != 1.0)
                || node
                    .parent_id
                    .is_some_and(|p| p as usize >= nodes.len() || p == node.id)
                || node
                    .embedding
                    .as_ref()
                    .is_some_and(|e| meta.embedding_dim != Some(e.len()))
                || node.embedder_space.is_some() && node.embedder_space != meta.embedder_space
            {
                return Err(fail(format!("invalid restored node {id}")));
            }
            // The insert-time alias rules, so the rebuilt index is one an
            // insert sequence could have produced.
            node.validate_aliases()?;
            let mut keys = HashSet::new();
            for alias in &node.aliases {
                let key = normalized(alias);
                let holders = alias_index.entry(key.clone()).or_default();
                if !keys.insert(key) || holders.len() >= MAX_ALIAS_HOLDERS {
                    return Err(fail(format!("invalid restored aliases on node {id}")));
                }
                holders.push(node.id);
            }
        }
        if meta.embedding_dim.is_some() && !nodes.iter().any(|n| n.embedding.is_some()) {
            return Err(fail("embedding dimension without an embedding"));
        }
        if meta.embedder_space.is_some()
            && !nodes
                .iter()
                .any(|n| n.embedder_space == meta.embedder_space)
        {
            return Err(fail("embedder space without a node declaring it"));
        }
        let mut previous = 0;
        for edge in &meta.edge_buffer {
            check_edge(nodes.len(), edge.source, edge.target, edge.weight)?;
            if edge.ticket <= previous || edge.ticket >= m.next_ticket {
                return Err(fail("invalid edge ticket sequence"));
            }
            previous = edge.ticket;
        }
        if m.next_ticket == 0 {
            return Err(fail("invalid next ticket"));
        }
        for &(a, b) in &meta.validated_deps {
            if !entity_index.contains_key(&a) || !entity_index.contains_key(&b) {
                return Err(fail("invalid validated dependency"));
            }
        }
        // Every write path admits only cycles that contract inside the step
        // budget; a restored graph must satisfy the same rule, or the first
        // evolution after startup could diverge.
        if let Some(refusal) = admission_refusal(&nodes, &csr, &meta.edge_buffer) {
            return Err(fail(format!(
                "restored graph holds an inadmissible cycle: {refusal}"
            )));
        }
        let retained: HashSet<String> = m.nodes.iter().chain([&m.state, &m.csr]).cloned().collect();
        let committed = Committed {
            chunks: nodes
                .chunks()
                .iter()
                .cloned()
                .zip(m.nodes.iter().cloned())
                .collect(),
            csr: Some((Arc::clone(&csr), m.csr.clone())),
            known: retained.clone(),
            garbage: orphans(dir, &retained)?,
            retained,
        };
        *graph.state.write() = GraphState {
            nodes,
            entity_index: Arc::new(entity_index),
            edge_buffer: meta.edge_buffer,
            revocations: meta.revocations,
            manual_revocations: meta.manual_revocations,
            privileges: meta.privileges,
            validated_deps: meta.validated_deps,
            alias_index: Arc::new(alias_index),
            embedding_dim: meta.embedding_dim,
            embedder_space: meta.embedder_space,
            generation: 0,
            discarded: Vec::new(),
        };
        graph.csr_snapshot.store(csr);
        graph.ticket_counter.store(m.next_ticket, Ordering::Relaxed);
        Ok((graph, committed))
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
        if dir.join(CURRENT).try_exists().map_err(fail)? {
            let (mut graph, committed) = Self::read_snapshot(dir, geometry)?;
            if !committed.garbage.is_empty() {
                tracing::info!(
                    dir = %dir.display(),
                    orphans = committed.garbage.len(),
                    "lodgraph: unreferenced blocks from an interrupted run are deleted after the next commit"
                );
            }
            graph.persistence = Some(Persistence::new(dir, lock, committed));
            return Ok(graph);
        }
        for entry in fs::read_dir(dir).map_err(fail)? {
            if entry.map_err(fail)?.file_name() != "LOCK" {
                return Err(fail("missing manifest in nonempty persistence directory"));
            }
        }
        let mut graph = Self::with_geometry(geometry)?;
        initialize(&graph)?;
        let p = Persistence::new(dir, lock, Committed::default());
        {
            let st = graph.state.read();
            let csr = graph.csr_snapshot.load_full();
            p.commit(
                &st,
                &csr,
                graph.ticket_counter.load(Ordering::Relaxed),
                graph.geometry(),
            )?;
        }
        // Persist directory creation in its parent as well.
        if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            sync_dir(parent)?;
        }
        graph.persistence = Some(p);
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

    /// Revoke `entity` on the live graph without a commit and poison the mount.
    /// Only for a quarantine the disk refused: a stricter live state than the
    /// disk is safe, and the poison keeps it from being served as durable.
    pub(super) fn revoke_uncommitted(&self, entity: u64) {
        if let Some(p) = &self.persistence {
            p.failed.store(true, Ordering::Release);
        }
        let _txn = self.txn_lock.lock();
        let _flush = self.flush_lock.lock();
        let mut st = self.state.write();
        st.revocations.insert(entity);
        st.manual_revocations.insert(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::time::{Instant, SystemTime};

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
        let nodes = |st: &GraphState| encode(&st.nodes.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(nodes(&x), nodes(&y));
        assert_eq!(
            encode(a.csr_snapshot.load().as_ref()).unwrap(),
            encode(b.csr_snapshot.load().as_ref()).unwrap()
        );
        assert_eq!(x.entity_index, y.entity_index);
        assert_eq!(x.alias_index, y.alias_index);
        assert_eq!(x.edge_buffer, y.edge_buffer);
        assert_eq!(x.revocations, y.revocations);
        assert_eq!(x.manual_revocations, y.manual_revocations);
        assert_eq!(x.privileges, y.privileges);
        assert_eq!(x.validated_deps, y.validated_deps);
        assert_eq!(x.embedding_dim, y.embedding_dim);
        assert_eq!(x.embedder_space, y.embedder_space);
        assert_eq!(
            a.ticket_counter.load(Ordering::Relaxed),
            b.ticket_counter.load(Ordering::Relaxed)
        );
    }
    fn manifest(dir: &Path) -> Manifest {
        let (_, bytes): (String, Vec<u8>) = decode(&fs::read(dir.join(CURRENT)).unwrap()).unwrap();
        decode(&bytes).unwrap()
    }
    /// Every `.block` file in `dir` with its modification time.
    fn blocks(dir: &Path) -> HashMap<String, SystemTime> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".block"))
            .map(|e| {
                let modified = e.metadata().unwrap().modified().unwrap();
                (e.file_name().to_string_lossy().into_owned(), modified)
            })
            .collect()
    }
    fn fill(graph: &LodGraph, count: u64) {
        graph
            .transact(|g| {
                for i in 0..count {
                    g.add_node(node(i))?;
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn exact_snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::new();
        for i in 0..1024 {
            graph.add_node(node(i)).unwrap();
        }
        graph.add_node(node(5000).with_aliases(["Alias"])).unwrap();
        graph.add_edge(0, 1, EdgeType::DependsOn, 0.75).unwrap();
        graph.flush_edges_to_csr().unwrap();
        graph.add_edge(1, 2, EdgeType::Semantic, 0.125).unwrap();
        graph.falsify_node(0).unwrap();
        graph
            .evolve_epistemic_fixed_point(0.85, 1e-6, 0.2, 0.8)
            .unwrap();
        graph.revoke_entity(9999).unwrap();
        graph.add_privilege(42, 8).unwrap();
        graph.save_to_dir(dir.path()).unwrap();
        let start = Instant::now();
        let loaded = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap();
        eprintln!(
            "restore sample: nodes=1025 csr_edges=1 pending_edges=1 elapsed_us={}",
            start.elapsed().as_micros()
        );
        // The indexes are rebuilt from the nodes, not stored.
        same(&graph, &loaded);
        loaded.add_edge(2, 3, EdgeType::Semantic, 0.25).unwrap();
        assert!(
            loaded.state.read().edge_buffer.last().unwrap().ticket
                > graph.state.read().edge_buffer.last().unwrap().ticket
        );
    }

    /// A commit writes the chunk it changed and the metadata, nothing else:
    /// every other block keeps its file and modification time, and the block it
    /// replaced is deleted.
    #[test]
    fn commit_writes_only_dirty_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        fill(&graph, 1024);
        graph
            .transact(|g| {
                g.add_edge(0, 1, EdgeType::DependsOn, 0.75)?;
                g.flush_edges_to_csr().map(|_| ())
            })
            .unwrap();
        let before = manifest(dir.path());
        let files = blocks(dir.path());
        assert_eq!(files.len(), 8 + 2);
        graph.transact(|g| g.falsify_node(200).map(|_| ())).unwrap();
        let after = manifest(dir.path());
        assert_eq!(before.csr, after.csr);
        let changed: Vec<usize> = (0..8)
            .filter(|&i| before.nodes[i] != after.nodes[i])
            .collect();
        assert_eq!(changed, vec![200 / NODE_CHUNK]);
        let now = blocks(dir.path());
        assert_eq!(now.len(), 8 + 2, "replaced blocks are deleted");
        for (name, modified) in &files {
            let replaced =
                *name == block_name(&before.nodes[1]) || *name == block_name(&before.state);
            assert_eq!(now.get(name), (!replaced).then_some(modified), "{name}");
        }
        drop(graph);
        let restored = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert!(restored.get_node(200).unwrap().status.is_falsified());
        assert_eq!(restored.node_count(), 1024);
    }

    /// A node's operator signature survives a commit and a reopen; the
    /// registered implementation does not, by design.
    #[test]
    fn operator_signature_round_trips() {
        use crate::operator::{OperatorKind, OperatorSignature};
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        let hard = OperatorSignature::new("fs.write", OperatorKind::HardDcm, "1");
        let soft =
            OperatorSignature::new("pcm.sum", OperatorKind::SoftPcm, "2").with_embedder_space("q");
        graph.add_node(node(1).with_operator(hard.clone())).unwrap();
        graph.add_node(node(2).with_operator(soft.clone())).unwrap();
        graph.add_node(node(3)).unwrap();
        drop(graph);
        let restored = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert_eq!(restored.get_node(0).unwrap().operator, Some(hard.clone()));
        assert_eq!(restored.get_node(1).unwrap().operator, Some(soft));
        assert_eq!(restored.get_node(2).unwrap().operator, None);
        assert_eq!(
            restored.operator(&hard).err().unwrap(),
            LodError::OperatorNotFound("fs.write".into())
        );
    }

    /// The write cost of a one-node transaction follows the change, not the
    /// graph: run with `--ignored --nocapture` in release for the 100k figure.
    #[test]
    #[ignore = "scale measurement; run in release"]
    fn single_node_commit_at_100k_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        let start = Instant::now();
        fill(&graph, 100_000);
        eprintln!(
            "bulk commit: nodes=100000 elapsed_ms={}",
            start.elapsed().as_millis()
        );
        for round in 0..5u64 {
            let files = blocks(dir.path());
            let start = Instant::now();
            graph
                .transact(|g| g.add_node(node(200_000 + round)).map(|_| ()))
                .unwrap();
            let elapsed = start.elapsed();
            let now = blocks(dir.path());
            let written = now
                .iter()
                .filter(|(n, t)| files.get(*n) != Some(*t))
                .count();
            eprintln!(
                "single-node commit: nodes={} elapsed_us={} blocks_written={written} blocks_total={}",
                graph.node_count(),
                elapsed.as_micros(),
                now.len()
            );
            assert!(written <= 2, "one chunk and the metadata, got {written}");
        }
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
        // 3 is the layout before nodes carried `operator`.
        for version in [1, 3, VERSION + 1] {
            m.version = version;
            let bad = encode(&m).unwrap();
            fs::write(
                dir.path().join(CURRENT),
                encode(&(digest(&bad), bad)).unwrap(),
            )
            .unwrap();
            let err = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT)
                .err()
                .unwrap();
            assert!(
                err.to_string().contains("unsupported snapshot version"),
                "{err}"
            );
        }
        fs::write(dir.path().join(CURRENT), &original).unwrap();
        let mut geometry = GeometryParams::UNIT;
        geometry.curvature = 2.0;
        assert!(LodGraph::load_from_dir(dir.path(), geometry).is_err());
        let file = dir.path().join(block_name(&m.nodes[0]));
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

    /// Every write path refuses a cycle that does not contract inside the step
    /// budget; a snapshot holding one is refused at startup, not served.
    #[test]
    fn restore_refuses_an_inadmissible_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::new();
        let a = graph.add_node(node(1)).unwrap();
        let b = graph.add_node(node(2)).unwrap();
        // Support and falsification both ways: row sum 2, bound 0.85 * 2 >= 1.
        // Pushed past admission, as only a damaged or foreign writer could.
        {
            let mut st = graph.state.write();
            for (source, target, edge_type) in [
                (a, b, EdgeType::DependsOn),
                (b, a, EdgeType::DependsOn),
                (a, b, EdgeType::Falsifies),
                (b, a, EdgeType::Falsifies),
            ] {
                let ticket = graph.ticket_counter.fetch_add(1, Ordering::Relaxed);
                st.edge_buffer.push(BufferedEdge {
                    source,
                    target,
                    edge_type,
                    weight: 1.0,
                    ticket,
                });
            }
        }
        graph.save_to_dir(dir.path()).unwrap();
        let err = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT)
            .err()
            .unwrap();
        assert!(err.to_string().contains("inadmissible cycle"), "{err}");
        let err = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT)
            .err()
            .unwrap();
        assert!(err.to_string().contains("inadmissible cycle"), "{err}");
    }

    /// A commit that fails before the manifest rename changes neither disk nor
    /// the live graph, and does not poison: the next commit succeeds.
    #[test]
    fn failed_commit_leaves_live_and_disk_untouched() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("data/lodgraph");
        let graph = LodGraph::open_persistent(&dir, GeometryParams::UNIT).unwrap();
        assert!(LodGraph::open_persistent(&dir, GeometryParams::UNIT).is_err());
        graph.transact(|g| g.add_node(node(1))).unwrap();
        let committed = fs::read(dir.join(CURRENT)).unwrap();
        // An actual filesystem failure before publication, not a mocked writer.
        fs::create_dir(dir.join("WRITE.tmp")).unwrap();
        assert!(graph.transact(|g| g.add_node(node(2))).is_err());
        assert!(graph.revoke_entity(1).is_err());
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node_for_entity(2), None);
        assert!(!graph.is_revoked(1));
        assert_eq!(fs::read(dir.join(CURRENT)).unwrap(), committed);
        graph.check_persistence().unwrap();
        fs::remove_dir(dir.join("WRITE.tmp")).unwrap();
        graph.transact(|g| g.add_node(node(3))).unwrap();
        drop(graph);
        let restored = LodGraph::open_persistent(&dir, GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.node_for_entity(2), None);
        assert_eq!(restored.node_for_entity(3), Some(1));
    }

    /// A failed manifest rename is ambiguous: the graph is poisoned and
    /// refuses every later write.
    #[test]
    fn failed_manifest_rename_poisons() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        fs::remove_file(dir.path().join(CURRENT)).unwrap();
        fs::create_dir(dir.path().join(CURRENT)).unwrap();
        fs::write(dir.path().join(CURRENT).join("occupied"), b"x").unwrap();
        assert!(graph.transact(|g| g.add_node(node(1))).is_err());
        assert_eq!(graph.node_count(), 0);
        assert!(graph.check_persistence().is_err());
        assert!(graph.transact(|g| g.add_node(node(2))).is_err());
    }

    /// Readers see only committed states: not the writes of a running
    /// transaction, not those of one that fails, not those of a dry run.
    #[test]
    fn readers_never_see_uncommitted_writes() {
        let dir = tempfile::tempdir().unwrap();
        let graph = Arc::new(LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap());
        graph.transact(|g| g.add_node(node(1)).map(|_| ())).unwrap();
        let committed = fs::read(dir.path().join(CURRENT)).unwrap();
        let inside = Arc::new(Barrier::new(2));
        let checked = Arc::new(Barrier::new(2));
        let writer = {
            let (graph, inside, checked) = (graph.clone(), inside.clone(), checked.clone());
            std::thread::spawn(move || {
                graph.transact(|g| {
                    g.add_node(node(2))?;
                    g.falsify_node(0)?;
                    inside.wait();
                    checked.wait();
                    Err::<(), _>(LodError::InvalidQuery("abort after the reads".into()))
                })
            })
        };
        inside.wait();
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.is_revoked(1));
        assert!(!graph.get_node(0).unwrap().status.is_falsified());
        checked.wait();
        assert!(writer.join().unwrap().is_err());
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.is_revoked(1));

        let report = graph
            .dry_run(|g| {
                g.add_node(node(3))?;
                g.falsify_node(0)?;
                assert!(!graph.is_revoked(1), "a dry run is private");
                Ok(g.node_count())
            })
            .unwrap();
        assert_eq!(report, 2);
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.is_revoked(1));
        assert_eq!(fs::read(dir.path().join(CURRENT)).unwrap(), committed);
        graph.check_persistence().unwrap();
    }

    /// Deleting an unreferenced block is cleanup: its failure is logged and
    /// retried, the commit stands and the graph keeps serving.
    #[test]
    fn block_cleanup_failure_is_only_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        fill(&graph, 2 * NODE_CHUNK as u64);
        let old = block_name(&manifest(dir.path()).nodes[0]);
        // Deleting a directory with `remove_file` fails on every platform.
        fs::remove_file(dir.path().join(&old)).unwrap();
        fs::create_dir(dir.path().join(&old)).unwrap();
        fs::write(dir.path().join(&old).join("pinned"), b"x").unwrap();
        graph.transact(|g| g.falsify_node(0).map(|_| ())).unwrap();
        graph.check_persistence().unwrap();
        assert!(dir.path().join(&old).is_dir());
        assert_ne!(block_name(&manifest(dir.path()).nodes[0]), old);
        graph
            .transact(|g| g.add_node(node(9999)).map(|_| ()))
            .unwrap();
        fs::remove_dir_all(dir.path().join(&old)).unwrap();
        graph
            .transact(|g| g.add_node(node(10_000)).map(|_| ()))
            .unwrap();
        drop(graph);
        let restored = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 2 * NODE_CHUNK + 2);
        assert!(restored.get_node(0).unwrap().status.is_falsified());
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
        let orphan = dir.path().join(block_name(&"0".repeat(64)));
        fs::write(&orphan, b"orphan").unwrap();
        let restored = LodGraph::load_from_dir(dir.path(), GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 0);
        drop(restored);
        // An orphan left by a crash is deleted after the next commit.
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert!(orphan.exists());
        graph.add_node(node(1)).unwrap();
        assert!(!orphan.exists());
        drop(graph);
        fs::remove_file(dir.path().join(CURRENT)).unwrap();
        assert!(LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).is_err());
    }

    /// A reflection the disk refuses still revokes the action in memory and
    /// poisons the mount: it never fails open, and a disk fault is not
    /// recorded as a durable quarantine.
    #[test]
    fn refused_quarantine_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let graph = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        fs::create_dir(dir.path().join("WRITE.tmp")).unwrap();
        assert!(graph.reflect_failure(7, "observed failure", 1).is_err());
        assert!(graph.is_revoked(7));
        assert!(graph.check_persistence().is_err());
        drop(graph);
        fs::remove_dir(dir.path().join("WRITE.tmp")).unwrap();
        let restored = LodGraph::open_persistent(dir.path(), GeometryParams::UNIT).unwrap();
        assert_eq!(restored.node_count(), 0, "nothing uncommitted reached disk");
        assert!(
            !restored.is_revoked(7),
            "a disk fault is not a durable quarantine"
        );
    }

    /// Candidates share the counters: a checkpoint or ticket issued inside a
    /// running transaction or dry run is never issued again outside it.
    #[test]
    fn candidates_never_reissue_counter_values() {
        let graph = LodGraph::new();
        graph.add_node(node(1)).unwrap();
        graph.add_node(node(2)).unwrap();
        let (inside, ticket) = graph
            .dry_run(|g| {
                Ok((
                    g.create_checkpoint().seq,
                    g.add_edge(0, 1, EdgeType::Semantic, 1.0)?,
                ))
            })
            .unwrap();
        let outside = graph.create_checkpoint().seq;
        assert_ne!(inside, outside);
        assert!(graph.add_edge(0, 1, EdgeType::Semantic, 1.0).unwrap() > ticket);
        let inside = graph.transact(|g| Ok(g.create_checkpoint().seq)).unwrap();
        assert!(graph.create_checkpoint().seq > inside);
    }

    /// A checkpoint taken inside a dropped candidate describes a state that
    /// was never published; no rollback may restore it.
    #[test]
    fn checkpoints_of_dropped_candidates_cannot_be_restored() {
        let graph = LodGraph::new();
        graph.add_node(node(1)).unwrap();
        let mut stash = None;
        let _ = graph.transact(|g| {
            g.falsify_node(0)?;
            stash = Some(g.create_checkpoint());
            Err::<(), _>(LodError::InvalidQuery("abort".into()))
        });
        assert!(graph.rollback_checkpoint(&stash.unwrap()).is_err());
        let dry = graph
            .dry_run(|g| {
                g.falsify_node(0)?;
                Ok(g.create_checkpoint())
            })
            .unwrap();
        assert!(graph.rollback_checkpoint(&dry).is_err());
        assert!(!graph.get_node(0).unwrap().status.is_falsified());
        // A checkpoint of the live graph itself still restores.
        let live = graph.create_checkpoint();
        graph.falsify_node(0).unwrap();
        graph.rollback_checkpoint(&live).unwrap();
        assert!(!graph.get_node(0).unwrap().status.is_falsified());
    }
}
