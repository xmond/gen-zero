//! Merkle Mountain Range (MMR) ledger with Keyed BLAKE3 cryptographic guarantees.
//!
//! Features:
//! - Monotonically append-only tree structure
//! - Amortized O(1) append; the root is updated incrementally and read in O(1)
//! - O(log N) inclusion proofs and verification for the most recent `window` leaves
//! - Bounded memory: about `2 * window + 128` node hashes, whatever the history length
//! - Keyed BLAKE3 hashing for entry and tree-node authentication
//!
//! The root commits to every leaf ever appended. Only the proof material (entries and the
//! subtree hashes their proofs need) is limited to the window; a proof request for an older
//! leaf returns [`ProvenanceError::LeafPruned`] instead of a wrong answer.

use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use crate::entry::DecisionAuditEntry;
use crate::error::ProvenanceError;
use serde::{Deserialize, Serialize};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// Sibling proof element specifying left or right positioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofSibling {
    Left([u8; 32]),
    Right([u8; 32]),
}

/// An inclusion proof for a leaf in the MMR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmrInclusionProof {
    pub leaf_index: u64,
    pub leaf_count: u64,
    pub leaf_hash: [u8; 32],
    pub siblings: Vec<ProofSibling>,
    pub mmr_root: [u8; 32],
}

/// Default number of most recent leaves that stay provable.
pub const MAX_AUDIT_LEAVES: usize = 4096;

/// Complete subtree hashes at one height. Node `j` covers leaves `[j << h, (j + 1) << h)`.
/// Only nodes with index `>= first` are kept.
#[derive(Clone)]
struct Level {
    first: u64,
    nodes: VecDeque<[u8; 32]>,
}

/// Merkle Mountain Range Ledger.
///
/// The root is the right-to-left bagging of the mountain peaks. It equals the root of the
/// pairwise tree that carries an odd trailing node up one level.
#[derive(Clone)]
pub struct MmrLedger {
    key: [u8; 32],
    window: usize,
    leaf_count: u64,
    entries: VecDeque<DecisionAuditEntry>,
    levels: Vec<Level>,
    root: [u8; 32],
}

const SNAPSHOT_FORMAT_VERSION: u32 = 1;
// A 4096-leaf proof window plus all retained levels fits well below this bound. Checking the
// file length before JSON decoding keeps a corrupt path from allocating without limit.
const MAX_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024;
const MAX_SNAPSHOT_WINDOW: usize = 1 << 20;

/// Serializable form of one retained MMR level. The ledger keeps these nodes so that a
/// restart can preserve the root for the complete append history while retaining only the
/// bounded proof window in memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedLevel {
    first: u64,
    nodes: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerSnapshotPayload {
    format_version: u32,
    key: [u8; 32],
    window: usize,
    leaf_count: u64,
    entries: Vec<DecisionAuditEntry>,
    levels: Vec<PersistedLevel>,
    root: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerSnapshot {
    payload: LedgerSnapshotPayload,
    checksum: [u8; 32],
}

impl MmrLedger {
    /// Create a new MMR ledger with a secret Keyed BLAKE3 key and a proof window of
    /// [`MAX_AUDIT_LEAVES`] leaves.
    pub fn new(key: [u8; 32]) -> Self {
        Self::build(key, MAX_AUDIT_LEAVES)
    }

    /// Create a ledger that keeps the most recent `window` leaves provable.
    pub fn with_window(key: [u8; 32], window: usize) -> Result<Self, ProvenanceError> {
        if window == 0 {
            return Err(ProvenanceError::MmrError(
                "proof window must hold at least one leaf".into(),
            ));
        }
        Ok(Self::build(key, window))
    }

    /// The keyed hash used by this ledger. A persisted engine uses this key to recreate the
    /// capability arbiter and to verify proofs after a restart.
    #[inline]
    pub fn key(&self) -> [u8; 32] {
        self.key
    }

    fn build(key: [u8; 32], window: usize) -> Self {
        Self {
            key,
            window,
            leaf_count: 0,
            entries: VecDeque::new(),
            levels: Vec::new(),
            root: [0u8; 32],
        }
    }

    /// Total number of leaves ever appended (not only the retained ones).
    #[inline]
    pub fn len(&self) -> usize {
        self.leaf_count as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Number of most recent leaves that stay provable.
    #[inline]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Index of the oldest leaf that can still be proved.
    #[inline]
    pub fn oldest_retained_index(&self) -> u64 {
        self.leaf_count - self.entries.len() as u64
    }

    /// Number of node hashes held in memory. Bounded by about `2 * window + 2 * height`.
    pub fn stored_nodes(&self) -> usize {
        self.levels.iter().map(|l| l.nodes.len()).sum()
    }

    /// Append a new DecisionAuditEntry to the MMR ledger. Amortized O(1) hashing, and the
    /// root is refreshed from at most 64 peaks.
    pub fn append(&mut self, entry: DecisionAuditEntry) -> u64 {
        let index = self.leaf_count;
        let mut node = entry.hash_entry(&self.key);
        self.entries.push_back(entry);
        if self.entries.len() > self.window {
            self.entries.pop_front();
        }
        self.leaf_count += 1;

        // Carry up: a level with an even node count just completed a pair, which becomes a
        // node one level higher.
        let mut height = 0;
        loop {
            if height == self.levels.len() {
                self.levels.push(Level {
                    first: 0,
                    nodes: VecDeque::new(),
                });
            }
            let level = &mut self.levels[height];
            level.nodes.push_back(node);
            if (level.first + level.nodes.len() as u64) % 2 == 1 {
                break;
            }
            let left = level.nodes[level.nodes.len() - 2];
            node = self.combine_hashes(&left, &node);
            height += 1;
        }

        self.prune();
        self.root = self.bag_peaks(0..self.levels.len()).unwrap_or([0u8; 32]);
        index
    }

    /// Append a leaf and atomically persist the resulting ledger snapshot.
    ///
    /// The current ledger is left untouched if encoding, writing or syncing the snapshot
    /// fails. The candidate is published only after its durable snapshot has been committed.
    /// A directory-sync error after rename can still leave the candidate on disk; callers
    /// must reconcile the snapshot before retrying after an indeterminate persistence error.
    /// This synchronous API must run on a blocking worker when called from async code.
    pub fn append_durable<P: AsRef<Path>>(
        &mut self,
        entry: DecisionAuditEntry,
        path: P,
    ) -> Result<u64, ProvenanceError> {
        let mut candidate = self.clone();
        let index = candidate.append(entry);
        candidate.persist_snapshot(path)?;
        *self = candidate;
        Ok(index)
    }

    /// Encode the bounded ledger state, including the retained MMR nodes and its complete
    /// history count/root, into a checksummed snapshot.
    pub fn snapshot_bytes(&self) -> Result<Vec<u8>, ProvenanceError> {
        let payload = LedgerSnapshotPayload {
            format_version: SNAPSHOT_FORMAT_VERSION,
            key: self.key,
            window: self.window,
            leaf_count: self.leaf_count,
            entries: self.entries.iter().copied().collect(),
            levels: self
                .levels
                .iter()
                .map(|level| PersistedLevel {
                    first: level.first,
                    nodes: level.nodes.iter().copied().collect(),
                })
                .collect(),
            root: self.root,
        };
        let payload_bytes = serde_json::to_vec(&payload)
            .map_err(|e| ProvenanceError::PersistenceError(format!("encode snapshot: {e}")))?;
        let checksum = *blake3::hash(&payload_bytes).as_bytes();
        serde_json::to_vec(&LedgerSnapshot { payload, checksum })
            .map_err(|e| ProvenanceError::PersistenceError(format!("encode snapshot: {e}")))
    }

    /// Restore a ledger from a checksummed snapshot. The checksum detects corruption, not
    /// malicious replacement: the snapshot contains its own key and must come from trusted
    /// storage. `expected_window` is checked so a host
    /// cannot accidentally restart with a wider proof retention policy than configured.
    pub fn from_snapshot_bytes(
        bytes: &[u8],
        expected_window: Option<usize>,
    ) -> Result<Self, ProvenanceError> {
        if bytes.is_empty() || bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(ProvenanceError::PersistenceError(format!(
                "snapshot size {} is outside the supported range",
                bytes.len()
            )));
        }
        let snapshot: LedgerSnapshot = serde_json::from_slice(bytes)
            .map_err(|e| ProvenanceError::PersistenceError(format!("decode snapshot: {e}")))?;
        if snapshot.payload.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(ProvenanceError::PersistenceError(format!(
                "unsupported snapshot format version {}",
                snapshot.payload.format_version
            )));
        }
        if expected_window.is_some_and(|window| window != snapshot.payload.window) {
            return Err(ProvenanceError::PersistenceError(format!(
                "snapshot proof window {} does not match configured window {}",
                snapshot.payload.window,
                expected_window.expect("is_some checked")
            )));
        }
        let payload_bytes = serde_json::to_vec(&snapshot.payload)
            .map_err(|e| ProvenanceError::PersistenceError(format!("encode snapshot: {e}")))?;
        let expected_checksum = *blake3::hash(&payload_bytes).as_bytes();
        if snapshot.checksum != expected_checksum {
            return Err(ProvenanceError::PersistenceError(
                "snapshot checksum mismatch".into(),
            ));
        }

        let payload = snapshot.payload;
        if payload.window == 0 {
            return Err(ProvenanceError::MmrError(
                "proof window must hold at least one leaf".into(),
            ));
        }
        if payload.window > MAX_SNAPSHOT_WINDOW {
            return Err(ProvenanceError::PersistenceError(format!(
                "snapshot proof window {} exceeds supported limit {MAX_SNAPSHOT_WINDOW}",
                payload.window
            )));
        }
        let expected_entries = payload.leaf_count.min(payload.window as u64) as usize;
        if payload.entries.len() != expected_entries {
            return Err(ProvenanceError::PersistenceError(
                "snapshot retained entry count does not match its proof window and leaf count"
                    .into(),
            ));
        }
        let oldest = payload.leaf_count - payload.entries.len() as u64;
        for (offset, entry) in payload.entries.iter().enumerate() {
            let expected_index = oldest + offset as u64;
            if entry.leaf_index != expected_index {
                return Err(ProvenanceError::PersistenceError(format!(
                    "retained entry index {} does not match expected index {expected_index}",
                    entry.leaf_index
                )));
            }
        }

        let levels = payload
            .levels
            .into_iter()
            .map(|level| {
                if level.first.checked_add(level.nodes.len() as u64).is_none() {
                    return Err(ProvenanceError::PersistenceError(
                        "MMR level index overflow".into(),
                    ));
                }
                Ok(Level {
                    first: level.first,
                    nodes: level.nodes.into_iter().collect(),
                })
            })
            .collect::<Result<Vec<_>, ProvenanceError>>()?;
        let ledger = Self {
            key: payload.key,
            window: payload.window,
            leaf_count: payload.leaf_count,
            entries: payload.entries.into_iter().collect(),
            levels,
            root: payload.root,
        };
        ledger.validate_snapshot_state()
    }

    /// Load a checksummed snapshot from disk. A missing file is an error; callers that want to
    /// create a new ledger should persist an empty ledger explicitly first.
    pub fn load_snapshot<P: AsRef<Path>>(
        path: P,
        expected_window: Option<usize>,
    ) -> Result<Self, ProvenanceError> {
        let read_error =
            |e| ProvenanceError::PersistenceError(format!("read {}: {e}", path.as_ref().display()));
        let file = fs::File::open(path.as_ref()).map_err(read_error)?;
        let mut bytes = Vec::new();
        file.take(MAX_SNAPSHOT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(read_error)?;
        Self::from_snapshot_bytes(&bytes, expected_window)
    }

    /// Persist a checksummed snapshot using a temporary sibling and atomic rename. The file is
    /// synced before rename, and the parent directory is synced after rename where supported.
    /// Callers must serialize writers to the same path. A post-rename sync failure is an
    /// indeterminate disk commit, not a rollback. Run this synchronous I/O on a blocking worker.
    pub fn persist_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<(), ProvenanceError> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let file_name = path.file_name().ok_or_else(|| {
            ProvenanceError::PersistenceError(format!(
                "snapshot path {} has no file name",
                path.display()
            ))
        })?;
        let temp_name = format!(
            ".{}.tmp-{}-{}",
            file_name.to_string_lossy(),
            std::process::id(),
            rand::random::<u64>()
        );
        let temp_path = parent.join(temp_name);
        let bytes = self.snapshot_bytes()?;
        let result = (|| -> std::io::Result<()> {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut file = options.open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp_path, path)?;
            // On Unix this makes the rename durable across a directory metadata flush. Some
            // filesystems reject directory sync; reporting that failure keeps the contract
            // fail-closed instead of claiming a durable append.
            #[cfg(unix)]
            fs::File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&temp_path);
            return Err(ProvenanceError::PersistenceError(format!(
                "write {}: {error}",
                path.display()
            )));
        }
        Ok(())
    }

    /// Alias emphasizing that this is a complete state snapshot rather than a proof export.
    pub fn save_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<(), ProvenanceError> {
        self.persist_snapshot(path)
    }

    /// Validate all serialized MMR material before exposing it to callers. In particular,
    /// every retained leaf must still have a valid proof against the restored root.
    fn validate_snapshot_state(&self) -> Result<Self, ProvenanceError> {
        let required_levels = if self.leaf_count == 0 {
            0
        } else {
            (u64::BITS - self.leaf_count.leading_zeros()) as usize
        };
        if self.levels.len() != required_levels {
            return Err(ProvenanceError::PersistenceError(format!(
                "snapshot has {} MMR levels, expected {required_levels}",
                self.levels.len()
            )));
        }
        // Proofs do not visit every retained node. Validate the append frontier too:
        // a surplus node on a non-peak level can pass current proofs but corrupt the
        // next carry, and a missing node can make a later append panic.
        let oldest = self.oldest_retained_index();
        for (height, level) in self.levels.iter().enumerate() {
            let end = self.leaf_count >> height;
            let first = (oldest >> height).saturating_sub(1);
            if level.first != first || level.nodes.len() as u64 != end - first {
                return Err(ProvenanceError::PersistenceError(format!(
                    "snapshot MMR level {height} does not match its append frontier"
                )));
            }
        }
        let max_nodes = self.window.saturating_mul(2).saturating_add(128);
        if self.stored_nodes() > max_nodes {
            return Err(ProvenanceError::PersistenceError(format!(
                "snapshot stores {} MMR nodes, exceeding bounded limit {max_nodes}",
                self.stored_nodes()
            )));
        }
        let computed_root = self.bag_peaks(0..self.levels.len()).unwrap_or([0u8; 32]);
        if computed_root != self.root {
            return Err(ProvenanceError::PersistenceError(
                "snapshot MMR root does not match retained peaks".into(),
            ));
        }
        let oldest = self.oldest_retained_index();
        for index in oldest..self.leaf_count {
            let proof = self.generate_proof(index)?;
            let entry = self.get_entry(index as usize).ok_or_else(|| {
                ProvenanceError::PersistenceError(format!(
                    "snapshot omitted retained entry {index}"
                ))
            })?;
            let expected = entry.hash_entry(&self.key);
            if !Self::verify_against_root(&self.key, &proof, &self.root, &expected)? {
                return Err(ProvenanceError::PersistenceError(format!(
                    "snapshot proof verification failed for retained leaf {index}"
                )));
            }
        }
        Ok(self.clone())
    }

    /// Drop nodes that no proof for a retained leaf can reach. For a retained leaf `i`, the
    /// path at height `h` touches node `i >> h` and its sibling, so the lowest index needed
    /// is `(oldest >> h) - 1`. Mountain peaks are always the last node of their level.
    fn prune(&mut self) {
        let oldest = self.oldest_retained_index();
        for (height, level) in self.levels.iter_mut().enumerate() {
            let keep_from = (oldest >> height).saturating_sub(1);
            while level.first < keep_from && !level.nodes.is_empty() {
                level.nodes.pop_front();
                level.first += 1;
            }
        }
    }

    /// Peak at `height`, if the mountain range has one there.
    fn peak(&self, height: usize) -> Option<[u8; 32]> {
        if (self.leaf_count >> height) & 1 == 1 {
            self.levels[height].nodes.back().copied()
        } else {
            None
        }
    }

    /// Bag the peaks at the given heights, right to left: the lowest peak is the rightmost.
    fn bag_peaks(&self, heights: std::ops::Range<usize>) -> Option<[u8; 32]> {
        heights
            .filter_map(|h| self.peak(h))
            .fold(None, |acc, peak| match acc {
                None => Some(peak),
                Some(right) => Some(self.combine_hashes(&peak, &right)),
            })
    }

    fn node(&self, height: usize, index: u64) -> Result<[u8; 32], ProvenanceError> {
        let level = &self.levels[height];
        index
            .checked_sub(level.first)
            .and_then(|offset| level.nodes.get(offset as usize))
            .copied()
            .ok_or_else(|| {
                ProvenanceError::MmrError(format!(
                    "node {index} at height {height} is not retained"
                ))
            })
    }

    /// Combine two hashes using Keyed BLAKE3.
    #[inline]
    fn combine_hashes(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(left);
        hasher.update(right);
        *hasher.finalize().as_bytes()
    }

    /// Current MMR root. O(1): it is maintained by [`Self::append`].
    #[inline]
    pub fn get_root(&self) -> [u8; 32] {
        self.root
    }

    /// Generate an O(log N) inclusion proof for an entry at `leaf_index`.
    ///
    /// Fails with [`ProvenanceError::LeafPruned`] when the leaf is older than the window.
    pub fn generate_proof(&self, leaf_index: u64) -> Result<MmrInclusionProof, ProvenanceError> {
        if leaf_index >= self.leaf_count {
            return Err(ProvenanceError::InvalidInclusionProof { leaf_index });
        }
        let oldest_retained = self.oldest_retained_index();
        if leaf_index < oldest_retained {
            return Err(ProvenanceError::LeafPruned {
                leaf_index,
                oldest_retained,
            });
        }

        let leaf_hash = self.node(0, leaf_index)?;
        let mut siblings = Vec::new();

        // Climb the leaf's own mountain until its peak.
        let mut height = 0;
        let mut pos = leaf_index;
        loop {
            let count = self.leaf_count >> height;
            if pos % 2 == 0 && pos + 1 == count {
                break;
            }
            if pos % 2 == 0 {
                siblings.push(ProofSibling::Right(self.node(height, pos + 1)?));
            } else {
                siblings.push(ProofSibling::Left(self.node(height, pos - 1)?));
            }
            pos /= 2;
            height += 1;
        }

        // Lower peaks sit to the right and are bagged first; higher peaks join from the left.
        if let Some(right) = self.bag_peaks(0..height) {
            siblings.push(ProofSibling::Right(right));
        }
        for h in height + 1..self.levels.len() {
            if let Some(left) = self.peak(h) {
                siblings.push(ProofSibling::Left(left));
            }
        }

        Ok(MmrInclusionProof {
            leaf_index,
            leaf_count: self.leaf_count,
            leaf_hash,
            siblings,
            mmr_root: self.root,
        })
    }

    /// Verify with this ledger's signing key, a caller-pinned root and expected entry hash.
    pub fn verify_inclusion_against_root(
        &self,
        proof: &MmrInclusionProof,
        trusted_root: &[u8; 32],
        expected_leaf_hash: &[u8; 32],
    ) -> bool {
        Self::verify_against_root(&self.key, proof, trusted_root, expected_leaf_hash)
            .unwrap_or(false)
    }

    /// Verify against a root and entry hash obtained independently of the proof.
    pub fn verify_against_root(
        key: &[u8; 32],
        proof: &MmrInclusionProof,
        trusted_root: &[u8; 32],
        expected_leaf_hash: &[u8; 32],
    ) -> Result<bool, ProvenanceError> {
        if &proof.leaf_hash != expected_leaf_hash
            || proof.leaf_count == 0
            || proof.leaf_index >= proof.leaf_count
            || &proof.mmr_root != trusted_root
        {
            return Ok(false);
        }
        let mut current_hash = proof.leaf_hash;
        let mut index = proof.leaf_index;
        let mut width = proof.leaf_count;
        let mut siblings = proof.siblings.iter();

        while width > 1 {
            let expected_left = index % 2 == 1;
            let has_sibling = expected_left || index + 1 < width;
            if has_sibling {
                let Some(sibling) = siblings.next() else {
                    return Ok(false);
                };
                if matches!(
                    (expected_left, sibling),
                    (true, ProofSibling::Right(_)) | (false, ProofSibling::Left(_))
                ) {
                    return Ok(false);
                }
                let mut hasher = blake3::Hasher::new_keyed(key);
                match sibling {
                    ProofSibling::Right(h) => {
                        hasher.update(&current_hash);
                        hasher.update(h);
                    }
                    ProofSibling::Left(h) => {
                        hasher.update(h);
                        hasher.update(&current_hash);
                    }
                }
                current_hash = *hasher.finalize().as_bytes();
            }
            index /= 2;
            width = width.div_ceil(2);
        }
        Ok(siblings.next().is_none() && &current_hash == trusted_root)
    }

    /// Retrieve a retained entry by its global leaf index. Entries older than the window
    /// return `None`.
    pub fn get_entry(&self, index: usize) -> Option<&DecisionAuditEntry> {
        let offset = (index as u64).checked_sub(self.oldest_retained_index())?;
        self.entries.get(offset as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_core::ActionId;
    use std::fs;
    use std::path::PathBuf;

    fn entry(i: u64) -> DecisionAuditEntry {
        DecisionAuditEntry::new(
            i,
            [0; 32],
            [1; 32],
            [2; 32],
            [3; 32],
            ActionId(i as u32),
            0,
            5000 + i,
        )
    }

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gen-zero-mmr-{label}-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    fn combine(key: &[u8; 32], left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(key);
        hasher.update(left);
        hasher.update(right);
        *hasher.finalize().as_bytes()
    }

    fn next_level(key: &[u8; 32], level: &[[u8; 32]]) -> Vec<[u8; 32]> {
        level
            .chunks(2)
            .map(|pair| match pair {
                [l, r] => combine(key, l, r),
                [odd] => *odd,
                _ => unreachable!(),
            })
            .collect()
    }

    /// The previous full-rebuild algorithm, kept as the reference the incremental ledger must
    /// match byte for byte: pairwise levels, an odd trailing node carried up unchanged.
    fn reference_root_and_proof(
        key: &[u8; 32],
        leaves: &[[u8; 32]],
        idx: usize,
    ) -> ([u8; 32], Vec<ProofSibling>) {
        let mut level = leaves.to_vec();
        let mut pos = idx;
        let mut siblings = Vec::new();
        while level.len() > 1 {
            if pos % 2 == 1 {
                siblings.push(ProofSibling::Left(level[pos - 1]));
            } else if pos + 1 < level.len() {
                siblings.push(ProofSibling::Right(level[pos + 1]));
            }
            level = next_level(key, &level);
            pos /= 2;
        }
        (level[0], siblings)
    }

    #[test]
    fn incremental_root_and_proofs_match_full_rebuild() {
        let key = [0x31u8; 32];
        let mut mmr = MmrLedger::new(key);
        let mut leaves = Vec::new();
        for n in 1..=300u64 {
            let e = entry(n - 1);
            leaves.push(e.hash_entry(&key));
            mmr.append(e);
            let (root, _) = reference_root_and_proof(&key, &leaves, 0);
            assert_eq!(mmr.get_root(), root, "root mismatch at size {n}");
        }
        for idx in 0..leaves.len() {
            let (root, siblings) = reference_root_and_proof(&key, &leaves, idx);
            let proof = mmr.generate_proof(idx as u64).unwrap();
            assert_eq!(proof.siblings, siblings, "proof mismatch for leaf {idx}");
            assert_eq!(proof.mmr_root, root);
            assert!(MmrLedger::verify_against_root(
                &key,
                &proof,
                &mmr.get_root(),
                &mmr.get_entry(proof.leaf_index as usize)
                    .unwrap()
                    .hash_entry(&key)
            )
            .unwrap());
        }
    }

    #[test]
    fn window_bounds_memory_but_root_covers_full_history() {
        for window in [1, 2, 3, 5, 7, 16, 33] {
            check_window(window);
        }
    }

    fn check_window(window: usize) {
        let key = [0x77u8; 32];
        let total = 1000u64;
        let mut mmr = MmrLedger::with_window(key, window).unwrap();
        let mut leaves = Vec::new();
        let mut max_nodes = 0;
        for i in 0..total {
            let e = entry(i);
            leaves.push(e.hash_entry(&key));
            assert_eq!(mmr.append(e), i);
            max_nodes = max_nodes.max(mmr.stored_nodes());
        }
        assert_eq!(mmr.len(), total as usize);
        assert_eq!(mmr.oldest_retained_index(), total - window as u64);
        // Two nodes per leaf in the window plus at most two per level.
        assert!(max_nodes <= 2 * window + 2 * 64, "stored {max_nodes} nodes");

        let oldest = mmr.oldest_retained_index();
        for idx in oldest..total {
            let (root, siblings) = reference_root_and_proof(&key, &leaves, idx as usize);
            let proof = mmr.generate_proof(idx).unwrap();
            assert_eq!(proof.mmr_root, root);
            assert_eq!(proof.siblings, siblings);
            assert!(MmrLedger::verify_against_root(
                &key,
                &proof,
                &mmr.get_root(),
                &mmr.get_entry(proof.leaf_index as usize)
                    .unwrap()
                    .hash_entry(&key)
            )
            .unwrap());
            assert_eq!(mmr.get_entry(idx as usize).unwrap().leaf_index, idx);
        }
        assert!(matches!(
            mmr.generate_proof(oldest - 1),
            Err(ProvenanceError::LeafPruned { leaf_index, oldest_retained })
                if leaf_index == oldest - 1 && oldest_retained == oldest
        ));
        assert!(mmr.get_entry(0).is_none());
        assert!(matches!(
            mmr.generate_proof(total),
            Err(ProvenanceError::InvalidInclusionProof { .. })
        ));
    }

    #[test]
    fn rejects_root_disguised_as_single_leaf_and_wrong_entry() {
        let key = [19; 32];
        let mut ledger = MmrLedger::new(key);
        for i in 0..3 {
            ledger.append(entry(i));
        }
        let root = ledger.get_root();
        let expected = entry(0).hash_entry(&key);
        let proof = ledger.generate_proof(0).unwrap();
        assert!(MmrLedger::verify_against_root(&key, &proof, &root, &expected).unwrap());
        assert!(
            !MmrLedger::verify_against_root(&key, &proof, &root, &entry(1).hash_entry(&key))
                .unwrap()
        );
        let forged = MmrInclusionProof {
            leaf_index: 0,
            leaf_count: 1,
            leaf_hash: root,
            siblings: vec![],
            mmr_root: root,
        };
        assert!(!MmrLedger::verify_against_root(&key, &forged, &root, &expected).unwrap());
    }

    #[test]
    fn zero_window_is_rejected() {
        assert!(matches!(
            MmrLedger::with_window([0u8; 32], 0),
            Err(ProvenanceError::MmrError(_))
        ));
    }

    #[test]
    fn tampered_proof_fails_verification() {
        let key = [9u8; 32];
        let mut mmr = MmrLedger::new(key);
        for i in 0..11 {
            mmr.append(entry(i));
        }
        let mut proof = mmr.generate_proof(4).unwrap();
        proof.leaf_hash[0] ^= 1;
        assert!(!MmrLedger::verify_against_root(
            &key,
            &proof,
            &mmr.get_root(),
            &mmr.get_entry(proof.leaf_index as usize)
                .unwrap()
                .hash_entry(&key)
        )
        .unwrap());
    }

    #[test]
    fn test_mmr_append_root_and_proof() {
        let key = [42u8; 32];
        let mut mmr = MmrLedger::new(key);

        for i in 0..8 {
            let entry = DecisionAuditEntry::new(
                i,
                [0; 32],
                [1; 32],
                [2; 32],
                [3; 32],
                ActionId(i as u32),
                0,
                1000 + i,
            );
            mmr.append(entry);
        }

        assert_eq!(mmr.len(), 8);
        let root = mmr.get_root();
        assert_ne!(root, [0u8; 32]);

        // Generate and verify proof for leaf 3
        let proof = mmr.generate_proof(3).unwrap();
        assert_eq!(proof.leaf_index, 3);
        assert_eq!(proof.mmr_root, root);

        let valid = MmrLedger::verify_against_root(
            &key,
            &proof,
            &mmr.get_root(),
            &mmr.get_entry(proof.leaf_index as usize)
                .unwrap()
                .hash_entry(&key),
        )
        .unwrap();
        assert!(valid);
    }

    #[test]
    fn test_mmr_non_power_of_two_proofs() {
        let key = [0x88u8; 32];
        for size in [3, 5, 7, 13] {
            let mut mmr = MmrLedger::new(key);
            for i in 0..size {
                let entry = DecisionAuditEntry::new(
                    i,
                    [0; 32],
                    [1; 32],
                    [2; 32],
                    [3; 32],
                    ActionId(i as u32),
                    0,
                    2000 + i,
                );
                mmr.append(entry);
            }

            let root = mmr.get_root();
            // Verify proofs for all leaves
            for leaf_idx in 0..size {
                let proof = mmr.generate_proof(leaf_idx).unwrap();
                assert_eq!(proof.mmr_root, root);
                assert!(MmrLedger::verify_against_root(
                    &key,
                    &proof,
                    &mmr.get_root(),
                    &mmr.get_entry(proof.leaf_index as usize)
                        .unwrap()
                        .hash_entry(&key)
                )
                .unwrap());
            }
        }
    }

    #[test]
    fn rejects_untrusted_root_and_invalid_path() {
        let key = [7; 32];
        let mut ledger = MmrLedger::new(key);
        for i in 0..3 {
            ledger.append(DecisionAuditEntry::new(
                i,
                [0; 32],
                [1; 32],
                [2; 32],
                [3; 32],
                ActionId(i as u32),
                0,
                i,
            ));
        }
        let root = ledger.get_root();
        let expected = ledger.get_entry(2).unwrap().hash_entry(&key);
        let proof = ledger.generate_proof(2).unwrap();
        assert!(MmrLedger::verify_against_root(&key, &proof, &root, &expected).unwrap());
        assert!(!MmrLedger::verify_against_root(&key, &proof, &[9; 32], &expected).unwrap());
        let mut invalid = proof.clone();
        invalid.leaf_index = 3;
        assert!(!MmrLedger::verify_against_root(&key, &invalid, &root, &expected).unwrap());
        let mut invalid = proof;
        invalid.leaf_count = 4;
        assert!(!MmrLedger::verify_against_root(&key, &invalid, &root, &expected).unwrap());
    }

    #[test]
    fn snapshot_restart_preserves_key_root_and_proof_window() {
        let key = [0x41; 32];
        let path = temp_path("restart");
        let mut ledger = MmrLedger::with_window(key, 4).unwrap();
        for i in 0..10 {
            ledger.append(entry(i));
        }
        ledger.persist_snapshot(&path).unwrap();

        let mut restored = MmrLedger::load_snapshot(&path, Some(4)).unwrap();
        assert_eq!(restored.key(), key);
        assert_eq!(restored.get_root(), ledger.get_root());
        assert_eq!(restored.len(), ledger.len());
        assert_eq!(restored.window(), 4);
        assert!(restored.generate_proof(6).is_ok());
        assert!(matches!(
            restored.generate_proof(5),
            Err(ProvenanceError::LeafPruned { .. })
        ));
        for i in 10..15 {
            ledger.append(entry(i));
            restored.append(entry(i));
            assert_eq!(restored.get_root(), ledger.get_root());
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn corrupt_snapshot_is_rejected_before_restore() {
        let path = temp_path("corrupt");
        let mut ledger = MmrLedger::with_window([0x42; 32], 4).unwrap();
        ledger.append(entry(0));
        ledger.persist_snapshot(&path).unwrap();
        let mut snapshot: LedgerSnapshot =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        snapshot.checksum[0] ^= 1;
        fs::write(&path, serde_json::to_vec(&snapshot).unwrap()).unwrap();
        let error = MmrLedger::load_snapshot(&path, Some(4))
            .err()
            .expect("corruption must fail");
        assert!(error.to_string().contains("checksum"), "{error}");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checksummed_but_incomplete_snapshot_is_rejected() {
        let mut ledger = MmrLedger::with_window([0x44; 32], 4).unwrap();
        ledger.append(entry(0));
        for missing_entries in [true, false] {
            let mut snapshot: LedgerSnapshot =
                serde_json::from_slice(&ledger.snapshot_bytes().unwrap()).unwrap();
            if missing_entries {
                snapshot.payload.entries.clear();
            } else {
                snapshot.payload.levels.push(PersistedLevel {
                    first: 0,
                    nodes: vec![],
                });
            }
            snapshot.checksum =
                *blake3::hash(&serde_json::to_vec(&snapshot.payload).unwrap()).as_bytes();
            assert!(MmrLedger::from_snapshot_bytes(
                &serde_json::to_vec(&snapshot).unwrap(),
                Some(4)
            )
            .is_err());
        }
    }

    #[test]
    fn restored_frontiers_append_across_peak_boundaries() {
        for window in [1, 2, 7] {
            let mut ledger = MmrLedger::with_window([0x46; 32], window).unwrap();
            for i in 0..65 {
                let mut restored =
                    MmrLedger::from_snapshot_bytes(&ledger.snapshot_bytes().unwrap(), Some(window))
                        .unwrap();
                ledger.append(entry(i));
                restored.append(entry(i));
                assert_eq!(restored.get_root(), ledger.get_root());
                for index in restored.oldest_retained_index()..=i {
                    let proof = restored.generate_proof(index).unwrap();
                    assert!(restored.verify_inclusion_against_root(
                        &proof,
                        &ledger.get_root(),
                        &entry(index).hash_entry(&ledger.key)
                    ));
                }
            }
        }
    }

    #[test]
    fn snapshot_rejects_surplus_nodes_outside_current_proofs() {
        let mut ledger = MmrLedger::with_window([0x45; 32], 2).unwrap();
        for i in 0..4 {
            ledger.append(entry(i));
        }
        // Height zero is not a peak at size four; its extra trailing node is not
        // touched by any current proof, but changes the next append's carry.
        ledger.levels[0].nodes.push_back([9; 32]);
        for i in 2..4 {
            let proof = ledger.generate_proof(i).unwrap();
            assert!(ledger.verify_inclusion_against_root(
                &proof,
                &ledger.get_root(),
                &entry(i).hash_entry(&ledger.key)
            ));
        }
        assert!(
            MmrLedger::from_snapshot_bytes(&ledger.snapshot_bytes().unwrap(), Some(2)).is_err()
        );
    }

    #[test]
    fn durable_append_error_does_not_publish_candidate() {
        let path = temp_path("append-error");
        fs::create_dir(&path).unwrap();
        let mut ledger = MmrLedger::with_window([0x43; 32], 4).unwrap();
        let root = ledger.get_root();
        let error = ledger.append_durable(entry(0), &path).unwrap_err();
        assert!(error.to_string().contains("write"), "{error}");
        assert_eq!(ledger.len(), 0);
        assert_eq!(ledger.get_root(), root);
        fs::remove_dir(path).unwrap();
    }
}
