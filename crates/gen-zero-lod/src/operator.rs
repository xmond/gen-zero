//! Causal state-transition operators (RFC #1, Phase 0): one contract for both
//! tracks of execution.
//!
//! - [`OperatorKind::HardDcm`]: a deterministic hard tool (CLI, API, SQL,
//!   system call) with physical side effects. Never pure. Every run must hold
//!   a one-shot nonce; the graph refuses a missing or spent one.
//! - [`OperatorKind::SoftPcm`]: a cognitive soft operator (a local model or a
//!   manifold computation) that only reads state. Always pure. Takes no nonce.
//!
//! A [`crate::LodNode`] names its operator by [`OperatorSignature`]; the
//! signature is persisted with the node. The implementation is a runtime
//! object registered with [`crate::LodGraph::register_operator`] and never
//! persisted. [`crate::LodGraph::execute_operator`] runs the gate in order:
//! node and signature checks, nonce, [`CausalOperator::check_preconditions`],
//! [`CausalOperator::transit`], [`CausalOperator::verify_postconditions`].
//! Any failure is an error; no stage is skipped or softened.
//!
//! Phase 0 limits: `transit` gets a read-only [`GraphState`], so the graph
//! does not apply `state_delta`, and a failed postcondition after a hard
//! tool ran reports the violation but cannot undo the external side effect.

use crate::error::LodError;
use crate::graph::GraphState;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Longest operator name, in bytes.
pub const MAX_OPERATOR_NAME_BYTES: usize = 256;
/// Longest operator version, in bytes.
pub const MAX_OPERATOR_VERSION_BYTES: usize = 64;
/// Longest embedder space identity of a soft operator, in bytes.
pub const MAX_OPERATOR_SPACE_BYTES: usize = 256;
/// Most operators one graph registers.
pub const MAX_REGISTERED_OPERATORS: usize = 4096;
/// Most spent nonces one graph remembers. Past it, every hard run is refused
/// rather than forgetting an old nonce and so allowing its replay.
pub const MAX_SPENT_NONCES: usize = 1 << 20;

/// The two execution tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    /// Deterministic hard tool with physical side effects.
    HardDcm,
    /// Pure cognitive soft operator.
    SoftPcm,
}

impl OperatorKind {
    /// Whether every operator of this kind must be pure.
    pub fn is_pure(self) -> bool {
        matches!(self, Self::SoftPcm)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HardDcm => "hard_dcm",
            Self::SoftPcm => "soft_pcm",
        }
    }
}

/// Identity of an operator. Lookup compares the whole signature, so a node
/// bound to version `1` never runs an operator registered as version `2`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorSignature {
    pub name: String,
    pub operator_kind: OperatorKind,
    /// Model space a soft operator computes in. A hard tool has none.
    pub embedder_space: Option<String>,
    pub version: String,
    /// Must equal `operator_kind.is_pure()`; stated so a trace shows it.
    pub pure: bool,
}

impl OperatorSignature {
    /// A signature whose `pure` follows the kind and with no embedder space.
    pub fn new(name: impl Into<String>, kind: OperatorKind, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            operator_kind: kind,
            embedder_space: None,
            version: version.into(),
            pure: kind.is_pure(),
        }
    }

    pub fn with_embedder_space(mut self, space: impl Into<String>) -> Self {
        self.embedder_space = Some(space.into());
        self
    }

    /// Refuse an empty or oversized name, version or space, a name or version
    /// with surrounding whitespace or control characters, a `pure` flag that
    /// contradicts the kind, and an embedder space on a hard tool.
    pub fn validate(&self) -> Result<(), LodError> {
        check_text("name", &self.name, MAX_OPERATOR_NAME_BYTES)?;
        check_text("version", &self.version, MAX_OPERATOR_VERSION_BYTES)?;
        if self.pure != self.operator_kind.is_pure() {
            return Err(LodError::InvalidOperator(format!(
                "operator `{}` is {} but declares pure = {}",
                self.name,
                self.operator_kind.as_str(),
                self.pure
            )));
        }
        match (&self.embedder_space, self.operator_kind) {
            (Some(_), OperatorKind::HardDcm) => Err(LodError::InvalidOperator(format!(
                "hard operator `{}` declares an embedder space; only a soft operator \
                 computes in a model space",
                self.name
            ))),
            (Some(space), OperatorKind::SoftPcm) => {
                check_text("embedder_space", space, MAX_OPERATOR_SPACE_BYTES)
            }
            (None, _) => Ok(()),
        }
    }
}

fn check_text(field: &str, value: &str, max: usize) -> Result<(), LodError> {
    if value.is_empty() || value.len() > max {
        return Err(LodError::InvalidOperator(format!(
            "operator {field} must be 1..={max} bytes, got {}",
            value.len()
        )));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(LodError::InvalidOperator(format!(
            "operator {field} `{}` has surrounding whitespace or control characters",
            value.escape_debug()
        )));
    }
    Ok(())
}

/// One request to an operator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorInput {
    /// Bound execution target; must match the node passed to execute_operator.
    pub node_id: u32,
    pub parameters: serde_json::Value,
    /// Digest of the context the caller decided on, carried into the trace.
    pub context_digest: [u8; 32],
    /// One-shot permit. Required by a hard tool, refused by a soft operator.
    pub nonce: Option<[u8; 32]>,
}

/// What one transit produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorOutput {
    pub state_delta: serde_json::Value,
    /// Evidence references: paths, log ids, digests.
    pub artifacts: Vec<String>,
    /// Non-zero fails the run.
    pub exit_code: i32,
    /// Wall-clock time of `transit`. The graph measures it and overwrites
    /// whatever the operator put here.
    pub execution_time_ns: u64,
}

/// Postcondition verdict of an operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostconditionReport {
    /// Must equal `violations.is_empty()`.
    pub valid: bool,
    /// Must not be empty: a report that checked nothing proves nothing.
    pub invariants_checked: Vec<String>,
    pub violations: Vec<String>,
}

/// A causal state-transition operator: a hard tool or a soft operator.
pub trait CausalOperator: Send + Sync {
    fn signature(&self) -> &OperatorSignature;
    /// Refuse to run in `state`. An error blocks the transit.
    /// `transit` holds a graph read lock and must not re-enter graph mutation.
    fn check_preconditions(&self, state: &GraphState) -> Result<(), LodError>;
    fn transit(
        &self,
        state: &GraphState,
        input: &OperatorInput,
    ) -> Result<OperatorOutput, LodError>;
    /// Judge output against its original input and the execution node at completion.
    fn verify_postconditions(
        &self,
        output: &OperatorOutput,
        input: &OperatorInput,
        state_after: &crate::LodNode,
    ) -> Result<PostconditionReport, LodError>;
    /// Whether the operator is free of physical side effects.
    fn is_pure(&self) -> bool;
}

/// Trace of one successful run. `signature.operator_kind` names the track,
/// so a soft result can never pass for a hard one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorExecution {
    pub node_id: u32,
    pub entity_id: u64,
    pub signature: OperatorSignature,
    pub context_digest: [u8; 32],
    pub nonce: Option<[u8; 32]>,
    pub output: OperatorOutput,
    pub report: PostconditionReport,
}

/// Registered operators and spent nonces of one graph. Runtime only.
#[derive(Default)]
pub(crate) struct OperatorRegistry {
    operators: HashMap<String, Arc<dyn CausalOperator>>,
    spent: HashSet<(String, [u8; 32])>,
}

impl OperatorRegistry {
    /// Add `op`. `locked_space` is the graph's embedder space lock.
    pub(crate) fn register(
        &mut self,
        op: Arc<dyn CausalOperator>,
        locked_space: Option<&str>,
    ) -> Result<(), LodError> {
        let sig = op.signature();
        sig.validate()?;
        if op.is_pure() != sig.pure {
            return Err(LodError::InvalidOperator(format!(
                "operator `{}` signature says pure = {} but is_pure() is {}",
                sig.name,
                sig.pure,
                op.is_pure()
            )));
        }
        check_space(sig, locked_space)?;
        if self.operators.contains_key(&sig.name) {
            return Err(LodError::InvalidOperator(format!(
                "operator `{}` is already registered",
                sig.name
            )));
        }
        if self.operators.len() >= MAX_REGISTERED_OPERATORS {
            return Err(LodError::InvalidOperator(format!(
                "graph already holds {MAX_REGISTERED_OPERATORS} operators"
            )));
        }
        self.operators.insert(sig.name.clone(), op);
        Ok(())
    }

    /// The operator registered under `sig`. A name registered with any other
    /// field different is a mismatch, never a near match.
    pub(crate) fn lookup(
        &self,
        sig: &OperatorSignature,
    ) -> Result<Arc<dyn CausalOperator>, LodError> {
        let op = self
            .operators
            .get(&sig.name)
            .ok_or_else(|| LodError::OperatorNotFound(sig.name.clone()))?;
        if op.signature() != sig {
            return Err(LodError::InvalidOperator(format!(
                "operator `{}` is registered as {:?}, not {:?}",
                sig.name,
                op.signature(),
                sig
            )));
        }
        Ok(Arc::clone(op))
    }

    /// Check the nonce rule of `sig` and spend a hard tool's nonce.
    pub(crate) fn spend_nonce(
        &mut self,
        sig: &OperatorSignature,
        nonce: Option<[u8; 32]>,
    ) -> Result<(), LodError> {
        let refuse = |detail: String| LodError::OperatorNonceRejected {
            operator: sig.name.clone(),
            detail,
        };
        match (sig.operator_kind, nonce) {
            (OperatorKind::SoftPcm, None) => Ok(()),
            (OperatorKind::SoftPcm, Some(_)) => {
                Err(refuse("a pure soft operator takes no nonce".into()))
            }
            (OperatorKind::HardDcm, None) => {
                Err(refuse("a hard tool needs a one-shot nonce".into()))
            }
            (OperatorKind::HardDcm, Some(n)) => {
                if self.spent.contains(&(sig.name.clone(), n)) {
                    return Err(refuse(format!("nonce {n:?} is already spent")));
                }
                if self.spent.len() >= MAX_SPENT_NONCES {
                    return Err(refuse(format!(
                        "graph remembers {MAX_SPENT_NONCES} spent nonces, the most it can \
                         check against replay"
                    )));
                }
                self.spent.insert((sig.name.clone(), n));
                Ok(())
            }
        }
    }
}

/// Refuse a soft operator whose embedder space differs from the graph's lock.
pub(crate) fn check_space(
    sig: &OperatorSignature,
    locked_space: Option<&str>,
) -> Result<(), LodError> {
    match (sig.embedder_space.as_deref(), locked_space) {
        (Some(claimed), Some(locked)) if claimed != locked => {
            Err(LodError::InvalidOperator(format!(
                "operator `{}` computes in embedder space `{claimed}` but this graph's \
                 embeddings are from `{locked}`",
                sig.name
            )))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpistemicStatus, LodBand, LodGraph, LodNode, MixedCurvatureCoord};
    use parking_lot::Mutex;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const ACTION: u64 = 7;
    const READY: u64 = 8;

    fn sig_hard() -> OperatorSignature {
        OperatorSignature::new("fs.write", OperatorKind::HardDcm, "1")
    }

    fn sig_soft() -> OperatorSignature {
        OperatorSignature::new("pcm.sum", OperatorKind::SoftPcm, "1")
    }

    fn input(nonce: Option<u64>, parameters: serde_json::Value) -> OperatorInput {
        OperatorInput {
            node_id: 0,
            parameters,
            context_digest: [3; 32],
            nonce: nonce.map(|n| {
                let mut bytes = [0; 32];
                bytes[..8].copy_from_slice(&n.to_le_bytes());
                bytes
            }),
        }
    }

    /// Hard tool: appends `parameters.line` to a file it owns. Precondition:
    /// the `READY` entity is a validated node. Postcondition: the file holds
    /// exactly the lines written so far.
    struct FileAppend {
        sig: OperatorSignature,
        path: std::path::PathBuf,
        written: Mutex<Vec<String>>,
        transits: AtomicUsize,
        /// Report a violation after the transit, to test the postcondition gate.
        corrupt_after: bool,
    }

    impl FileAppend {
        fn new(dir: &std::path::Path) -> Self {
            Self {
                sig: sig_hard(),
                path: dir.join("out.log"),
                written: Mutex::new(Vec::new()),
                transits: AtomicUsize::new(0),
                corrupt_after: false,
            }
        }
    }

    impl CausalOperator for FileAppend {
        fn signature(&self) -> &OperatorSignature {
            &self.sig
        }
        fn check_preconditions(&self, state: &GraphState) -> Result<(), LodError> {
            match state.entity_node(READY) {
                Some(n) if n.status == EpistemicStatus::Validated => Ok(()),
                _ => Err(LodError::InvalidQuery("READY is not validated".into())),
            }
        }
        fn transit(
            &self,
            _: &GraphState,
            input: &OperatorInput,
        ) -> Result<OperatorOutput, LodError> {
            use std::io::Write;
            self.transits.fetch_add(1, Ordering::SeqCst);
            let line = input.parameters["line"]
                .as_str()
                .ok_or_else(|| LodError::InvalidQuery("missing line".into()))?;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(|e| LodError::InvalidQuery(e.to_string()))?;
            writeln!(f, "{line}").map_err(|e| LodError::InvalidQuery(e.to_string()))?;
            self.written.lock().push(line.to_string());
            Ok(OperatorOutput {
                state_delta: json!({"appended": line}),
                artifacts: vec![self.path.display().to_string()],
                exit_code: 0,
                execution_time_ns: u64::MAX,
            })
        }
        fn verify_postconditions(
            &self,
            _: &OperatorOutput,
            _: &OperatorInput,
            _: &LodNode,
        ) -> Result<PostconditionReport, LodError> {
            let on_disk = std::fs::read_to_string(&self.path)
                .map_err(|e| LodError::InvalidQuery(e.to_string()))?;
            let mut expected: String = self
                .written
                .lock()
                .iter()
                .map(|l| format!("{l}\n"))
                .collect();
            if self.corrupt_after {
                expected.push_str("missing\n");
            }
            let violations = if on_disk == expected {
                vec![]
            } else {
                vec![format!("file holds {on_disk:?}, expected {expected:?}")]
            };
            Ok(PostconditionReport {
                valid: violations.is_empty(),
                invariants_checked: vec!["file_matches_written_lines".into()],
                violations,
            })
        }
        fn is_pure(&self) -> bool {
            false
        }
    }

    /// Soft operator: sums the confidences of the requested entities. Pure.
    struct ConfidenceSum {
        sig: OperatorSignature,
        report: PostconditionReport,
    }

    impl ConfidenceSum {
        fn new() -> Self {
            Self {
                sig: sig_soft(),
                report: PostconditionReport {
                    valid: true,
                    invariants_checked: vec!["no_state_change".into()],
                    violations: vec![],
                },
            }
        }
    }

    impl CausalOperator for ConfidenceSum {
        fn signature(&self) -> &OperatorSignature {
            &self.sig
        }
        fn check_preconditions(&self, state: &GraphState) -> Result<(), LodError> {
            if state.node_count() == 0 {
                return Err(LodError::EmptyInput("empty graph".into()));
            }
            Ok(())
        }
        fn transit(
            &self,
            state: &GraphState,
            input: &OperatorInput,
        ) -> Result<OperatorOutput, LodError> {
            let entities = input.parameters["entities"]
                .as_array()
                .ok_or_else(|| LodError::InvalidQuery("missing entities".into()))?;
            let mut sum = 0.0f64;
            for e in entities {
                let e = e
                    .as_u64()
                    .ok_or_else(|| LodError::InvalidQuery("bad entity".into()))?;
                let node = state.entity_node(e).ok_or(LodError::EntityNotFound(e))?;
                sum += f64::from(node.confidence);
            }
            Ok(OperatorOutput {
                state_delta: json!({"confidence_sum": sum}),
                artifacts: vec![],
                exit_code: 0,
                execution_time_ns: 0,
            })
        }
        fn verify_postconditions(
            &self,
            _: &OperatorOutput,
            _: &OperatorInput,
            _: &LodNode,
        ) -> Result<PostconditionReport, LodError> {
            Ok(self.report.clone())
        }
        fn is_pure(&self) -> bool {
            true
        }
    }

    fn node(entity: u64, status: EpistemicStatus, op: Option<OperatorSignature>) -> LodNode {
        let mut n = LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            format!("n{entity}"),
            entity,
        );
        n.status = status;
        n.operator = op;
        n
    }

    /// Graph with an action node bound to `sig` and the `READY` node.
    fn graph(sig: OperatorSignature, ready: EpistemicStatus) -> (LodGraph, u32) {
        let g = LodGraph::new();
        let id = g
            .add_node(node(ACTION, EpistemicStatus::Hypothesized, Some(sig)))
            .unwrap();
        g.add_node(node(READY, ready, None).with_prior(0.8))
            .unwrap();
        (g, id)
    }

    #[test]
    fn hard_dcm_runs_with_nonce_and_records_trace() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        let op = Arc::new(FileAppend::new(dir.path()));
        g.register_operator(op.clone()).unwrap();
        let run = g
            .execute_operator(id, &input(Some(1), json!({"line": "hello"})))
            .unwrap();
        assert_eq!(run.signature.operator_kind, OperatorKind::HardDcm);
        assert!(!run.signature.pure);
        assert_eq!(
            (run.node_id, run.entity_id, run.nonce),
            (id, ACTION, input(Some(1), json!({})).nonce)
        );
        assert_eq!(run.context_digest, [3; 32]);
        assert_eq!(run.output.state_delta, json!({"appended": "hello"}));
        assert_ne!(
            run.output.execution_time_ns,
            u64::MAX,
            "graph measures time"
        );
        assert!(run.report.valid);
        assert_eq!(std::fs::read_to_string(&op.path).unwrap(), "hello\n");
    }

    #[test]
    fn hard_dcm_refuses_missing_and_replayed_nonce() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        let op = Arc::new(FileAppend::new(dir.path()));
        g.register_operator(op.clone()).unwrap();
        let err = g
            .execute_operator(id, &input(None, json!({"line": "a"})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorNonceRejected { .. }),
            "{err}"
        );
        g.execute_operator(id, &input(Some(9), json!({"line": "a"})))
            .unwrap();
        let err = g
            .execute_operator(id, &input(Some(9), json!({"line": "b"})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorNonceRejected { .. }),
            "{err}"
        );
        assert_eq!(
            op.transits.load(Ordering::SeqCst),
            1,
            "replay never reached the tool"
        );
        assert_eq!(std::fs::read_to_string(&op.path).unwrap(), "a\n");
    }

    #[test]
    fn precondition_failure_blocks_transit_and_spends_nonce() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Hypothesized);
        let op = Arc::new(FileAppend::new(dir.path()));
        g.register_operator(op.clone()).unwrap();
        let err = g
            .execute_operator(id, &input(Some(1), json!({"line": "x"})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorPreconditionFailed { .. }),
            "{err}"
        );
        assert_eq!(op.transits.load(Ordering::SeqCst), 0);
        assert!(!op.path.exists(), "no side effect behind a closed gate");
        let err = g
            .execute_operator(id, &input(Some(1), json!({"line": "x"})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorNonceRejected { .. }),
            "{err}"
        );
    }

    #[test]
    fn revoked_or_falsified_node_never_runs() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        let op = Arc::new(FileAppend::new(dir.path()));
        g.register_operator(op.clone()).unwrap();
        g.revoke_entity(ACTION).unwrap();
        let err = g
            .execute_operator(id, &input(Some(1), json!({"line": "x"})))
            .unwrap_err();
        assert!(matches!(err, LodError::OperatorNodeRevoked(_)), "{err}");

        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        g.register_operator(op.clone()).unwrap();
        g.falsify_node(id).unwrap();
        let err = g
            .execute_operator(id, &input(Some(2), json!({"line": "x"})))
            .unwrap_err();
        assert!(matches!(err, LodError::OperatorNodeRevoked(_)), "{err}");
        assert_eq!(op.transits.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn postcondition_violation_fails_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        let mut op = FileAppend::new(dir.path());
        op.corrupt_after = true;
        g.register_operator(Arc::new(op)).unwrap();
        let err = g
            .execute_operator(id, &input(Some(1), json!({"line": "x"})))
            .unwrap_err();
        match err {
            LodError::OperatorPostconditionFailed { violations, .. } => {
                assert_eq!(violations.len(), 1);
                assert!(violations[0].contains("expected"), "{violations:?}");
            }
            other => panic!("expected postcondition failure, got {other}"),
        }
    }

    #[test]
    fn transit_error_and_nonzero_exit_fail_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        g.register_operator(Arc::new(FileAppend::new(dir.path())))
            .unwrap();
        let err = g
            .execute_operator(id, &input(Some(1), json!({})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorTransitFailed { .. }),
            "{err}"
        );

        struct Exit3(OperatorSignature);
        impl CausalOperator for Exit3 {
            fn signature(&self) -> &OperatorSignature {
                &self.0
            }
            fn check_preconditions(&self, _: &GraphState) -> Result<(), LodError> {
                Ok(())
            }
            fn transit(
                &self,
                _: &GraphState,
                _: &OperatorInput,
            ) -> Result<OperatorOutput, LodError> {
                Ok(OperatorOutput {
                    state_delta: json!(null),
                    artifacts: vec![],
                    exit_code: 3,
                    execution_time_ns: 0,
                })
            }
            fn verify_postconditions(
                &self,
                _: &OperatorOutput,
                _: &OperatorInput,
                _: &LodNode,
            ) -> Result<PostconditionReport, LodError> {
                panic!("postconditions must not run after a failed transit")
            }
            fn is_pure(&self) -> bool {
                false
            }
        }
        let sig = OperatorSignature::new("exit3", OperatorKind::HardDcm, "1");
        let (g, id) = graph(sig.clone(), EpistemicStatus::Validated);
        g.register_operator(Arc::new(Exit3(sig))).unwrap();
        let err = g
            .execute_operator(id, &input(Some(1), json!({})))
            .unwrap_err();
        match err {
            LodError::OperatorTransitFailed { detail, .. } => {
                assert!(detail.contains("exit code 3"))
            }
            other => panic!("expected transit failure, got {other}"),
        }
    }

    #[test]
    fn soft_pcm_runs_pure_transition() {
        let (g, id) = graph(sig_soft(), EpistemicStatus::Validated);
        g.register_operator(Arc::new(ConfidenceSum::new())).unwrap();
        let before = g.get_node(id).unwrap();
        let run = g
            .execute_operator(id, &input(None, json!({"entities": [ACTION, READY]})))
            .unwrap();
        assert_eq!(run.signature.operator_kind, OperatorKind::SoftPcm);
        assert!(run.signature.pure);
        // ACTION prior 0.5, READY prior 0.8.
        let sum = run.output.state_delta["confidence_sum"].as_f64().unwrap();
        assert!((sum - 1.3).abs() < 1e-6, "{sum}");
        assert_eq!(run.nonce, None);
        let after = g.get_node(id).unwrap();
        assert_eq!(
            (before.confidence, before.status),
            (after.confidence, after.status),
            "a pure operator changes nothing"
        );
        let err = g
            .execute_operator(id, &input(Some(1), json!({"entities": []})))
            .unwrap_err();
        assert!(
            matches!(err, LodError::OperatorNonceRejected { .. }),
            "{err}"
        );
    }

    #[test]
    fn soft_pcm_transit_error_is_reported() {
        let (g, id) = graph(sig_soft(), EpistemicStatus::Validated);
        g.register_operator(Arc::new(ConfidenceSum::new())).unwrap();
        let err = g
            .execute_operator(id, &input(None, json!({"entities": [999]})))
            .unwrap_err();
        match err {
            LodError::OperatorTransitFailed { detail, .. } => assert!(detail.contains("999")),
            other => panic!("expected transit failure, got {other}"),
        }
    }

    #[test]
    fn inconsistent_or_empty_report_is_refused() {
        for (valid, checked, violations) in [
            (true, vec!["x".to_string()], vec!["bad".to_string()]),
            (false, vec!["x".to_string()], vec![]),
            (true, vec![], vec![]),
        ] {
            let (g, id) = graph(sig_soft(), EpistemicStatus::Validated);
            let mut op = ConfidenceSum::new();
            op.report = PostconditionReport {
                valid,
                invariants_checked: checked,
                violations,
            };
            g.register_operator(Arc::new(op)).unwrap();
            let err = g
                .execute_operator(id, &input(None, json!({"entities": []})))
                .unwrap_err();
            assert!(
                matches!(err, LodError::OperatorPostconditionFailed { .. }),
                "{err}"
            );
        }
    }

    #[test]
    fn registration_refuses_bad_signatures_and_duplicates() {
        let g = LodGraph::new();
        let mut op = ConfidenceSum::new();
        op.sig.pure = false;
        let err = g.register_operator(Arc::new(op)).unwrap_err();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");

        // Signature says hard, implementation says pure.
        let mut op = ConfidenceSum::new();
        op.sig = OperatorSignature::new("liar", OperatorKind::HardDcm, "1");
        assert!(g.register_operator(Arc::new(op)).is_err());

        let mut op = ConfidenceSum::new();
        op.sig.name = " padded".into();
        assert!(g.register_operator(Arc::new(op)).is_err());

        let dir = tempfile::tempdir().unwrap();
        let mut hard = FileAppend::new(dir.path());
        hard.sig = sig_hard().with_embedder_space("qwen");
        assert!(g.register_operator(Arc::new(hard)).is_err());

        g.register_operator(Arc::new(ConfidenceSum::new())).unwrap();
        let err = g
            .register_operator(Arc::new(ConfidenceSum::new()))
            .unwrap_err();
        assert!(err.to_string().contains("already registered"), "{err}");
    }

    #[test]
    fn soft_operator_space_must_match_graph_lock() {
        let g = LodGraph::new();
        g.add_node(
            node(1, EpistemicStatus::Hypothesized, None)
                .with_embedding(vec![0.5; 16])
                .with_embedder_space("qwen-a"),
        )
        .unwrap();
        let mut op = ConfidenceSum::new();
        op.sig = sig_soft().with_embedder_space("qwen-b");
        let err = g.register_operator(Arc::new(op)).unwrap_err();
        assert!(err.to_string().contains("qwen-b"), "{err}");
        let err = g
            .add_node(node(
                2,
                EpistemicStatus::Hypothesized,
                Some(sig_soft().with_embedder_space("qwen-b")),
            ))
            .unwrap_err();
        assert!(err.to_string().contains("qwen-b"), "{err}");
        let mut op = ConfidenceSum::new();
        op.sig = sig_soft().with_embedder_space("qwen-a");
        g.register_operator(Arc::new(op)).unwrap();

        // A node whose own embedding would set the lock is checked against it.
        let fresh = LodGraph::new();
        let err = fresh
            .add_node(
                node(
                    1,
                    EpistemicStatus::Hypothesized,
                    Some(sig_soft().with_embedder_space("qwen-b")),
                )
                .with_embedding(vec![0.5; 16])
                .with_embedder_space("qwen-a"),
            )
            .unwrap_err();
        assert!(err.to_string().contains("qwen-b"), "{err}");

        // Registered before any lock, run after a different lock appeared.
        let late = LodGraph::new();
        let mut op = ConfidenceSum::new();
        op.sig = sig_soft().with_embedder_space("qwen-b");
        let sig = op.sig.clone();
        late.register_operator(Arc::new(op)).unwrap();
        let id = late
            .add_node(node(1, EpistemicStatus::Hypothesized, Some(sig)))
            .unwrap();
        late.add_node(
            node(2, EpistemicStatus::Hypothesized, None)
                .with_embedding(vec![0.5; 16])
                .with_embedder_space("qwen-a"),
        )
        .unwrap();
        let err = late
            .execute_operator(id, &input(None, json!({"entities": []})))
            .unwrap_err();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");
    }

    #[test]
    fn lookup_is_by_whole_signature() {
        let g = LodGraph::new();
        g.register_operator(Arc::new(ConfidenceSum::new())).unwrap();
        assert!(g.operator(&sig_soft()).is_ok());
        let v2 = OperatorSignature::new("pcm.sum", OperatorKind::SoftPcm, "2");
        let err = g.operator(&v2).err().unwrap();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");
        let err = g.operator(&sig_hard()).err().unwrap();
        assert_eq!(err, LodError::OperatorNotFound("fs.write".into()));

        // A node bound to v2 does not run the v1 operator.
        let (g, id) = graph(v2, EpistemicStatus::Validated);
        g.register_operator(Arc::new(ConfidenceSum::new())).unwrap();
        let err = g
            .execute_operator(id, &input(None, json!({"entities": []})))
            .unwrap_err();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");
    }

    #[test]
    fn node_without_operator_or_unregistered_is_refused() {
        let (g, _) = graph(sig_soft(), EpistemicStatus::Validated);
        let ready = g.node_for_entity(READY).unwrap();
        let err = g
            .execute_operator(ready, &input(None, json!({})))
            .unwrap_err();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");
        let action = g.node_for_entity(ACTION).unwrap();
        let err = g
            .execute_operator(action, &input(None, json!({})))
            .unwrap_err();
        assert_eq!(err, LodError::OperatorNotFound("pcm.sum".into()));
        assert_eq!(
            g.execute_operator(99, &input(None, json!({}))).unwrap_err(),
            LodError::NodeNotFound(99)
        );
    }

    #[test]
    fn insert_refuses_invalid_node_signature() {
        let g = LodGraph::new();
        let mut sig = sig_hard();
        sig.pure = true;
        let err = g
            .add_node(node(1, EpistemicStatus::Hypothesized, Some(sig)))
            .unwrap_err();
        assert!(matches!(err, LodError::InvalidOperator(_)), "{err}");
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn signature_json_shape_and_lodnode_without_operator_key() {
        let sig = sig_soft().with_embedder_space("qwen");
        let v = serde_json::to_value(&sig).unwrap();
        assert_eq!(
            v,
            json!({"name": "pcm.sum", "operator_kind": "soft_pcm",
                   "embedder_space": "qwen", "version": "1", "pure": true})
        );
        assert_eq!(serde_json::from_value::<OperatorSignature>(v).unwrap(), sig);
        let bad = json!({"name": "a", "operator_kind": "soft_pcm", "embedder_space": null,
                         "version": "1", "pure": true, "extra": 1});
        assert!(serde_json::from_value::<OperatorSignature>(bad).is_err());

        let mut v = serde_json::to_value(node(1, EpistemicStatus::Hypothesized, None)).unwrap();
        v.as_object_mut().unwrap().remove("operator");
        let back: LodNode = serde_json::from_value(v).unwrap();
        assert_eq!(back.operator, None);
    }
    #[test]
    fn revocation_during_preconditions_blocks_transit() {
        struct RevokeDuringCheck {
            graph: std::sync::Weak<LodGraph>,
            signature: OperatorSignature,
        }
        impl CausalOperator for RevokeDuringCheck {
            fn signature(&self) -> &OperatorSignature {
                &self.signature
            }
            fn is_pure(&self) -> bool {
                false
            }
            fn check_preconditions(&self, _: &GraphState) -> Result<(), LodError> {
                self.graph.upgrade().unwrap().revoke_entity(ACTION)
            }
            fn transit(
                &self,
                _: &GraphState,
                _: &OperatorInput,
            ) -> Result<OperatorOutput, LodError> {
                panic!("revocation after snapshot must block transit")
            }
            fn verify_postconditions(
                &self,
                _: &OperatorOutput,
                _: &OperatorInput,
                _: &LodNode,
            ) -> Result<PostconditionReport, LodError> {
                panic!("no transit")
            }
        }
        let (g, id) = graph(sig_hard(), EpistemicStatus::Validated);
        let g = Arc::new(g);
        g.register_operator(Arc::new(RevokeDuringCheck {
            graph: Arc::downgrade(&g),
            signature: sig_hard(),
        }))
        .unwrap();
        assert!(
            matches!(g.execute_operator(id, &input(Some(19), json!({}))),
            Err(LodError::OperatorNodeRevoked(n)) if n == id)
        );
    }
}
