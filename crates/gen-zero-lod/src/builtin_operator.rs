//! Standard operators. DCM records an action in a bounded process-local audit;
//! PCM summarizes graph state without mutating it. Neither interprets a command
//! nor claims learned reasoning. Audit/nonce lifetime is the graph runtime.
use crate::{
    CausalOperator, GraphState, LodError, LodGraph, LodNode, OperatorInput, OperatorKind,
    OperatorOutput, OperatorSignature, PostconditionReport, MAX_SPENT_NONCES,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};

pub const DCM_NAME: &str = "builtin:dcm_executor";
pub const PCM_NAME: &str = "builtin:pcm_evaluator";
pub fn dcm_signature() -> OperatorSignature {
    OperatorSignature::new(DCM_NAME, OperatorKind::HardDcm, "1")
}
pub fn pcm_signature() -> OperatorSignature {
    OperatorSignature::new(PCM_NAME, OperatorKind::SoftPcm, "1")
}

type Audit = Arc<Mutex<HashMap<[u8; 32], Value>>>;
struct Builtin {
    signature: OperatorSignature,
    audit: Audit,
}

fn summary(node: &LodNode) -> Value {
    json!({"node_id": node.id, "entity_id": node.entity_id, "label": node.label,
        "confidence": node.confidence, "status": node.status})
}
fn record(node: &LodNode, input: &OperatorInput) -> Value {
    json!({"action": summary(node), "input": input.parameters,
        "nonce": input.nonce, "context_digest": input.context_digest})
}
impl CausalOperator for Builtin {
    fn signature(&self) -> &OperatorSignature {
        &self.signature
    }
    fn is_pure(&self) -> bool {
        self.signature.pure
    }
    fn check_preconditions(&self, _: &GraphState) -> Result<(), LodError> {
        Ok(())
    }
    fn transit(
        &self,
        state: &GraphState,
        input: &OperatorInput,
    ) -> Result<OperatorOutput, LodError> {
        let node = state
            .node(input.node_id)
            .ok_or(LodError::NodeNotFound(input.node_id))?;
        if node.operator.as_ref() != Some(&self.signature) {
            return Err(LodError::InvalidOperator(
                "builtin target signature mismatch".into(),
            ));
        }
        let delta = if self.is_pure() {
            json!({"evaluation": summary(node), "node_count": state.node_count(),
                "audit_records": self.audit.lock().len(), "effect": "none"})
        } else {
            let nonce = input
                .nonce
                .ok_or_else(|| LodError::InvalidQuery("missing nonce".into()))?;
            let mut audit = self.audit.lock();
            if audit.len() >= MAX_SPENT_NONCES || audit.contains_key(&nonce) {
                return Err(LodError::InvalidQuery(
                    "audit full or nonce already recorded".into(),
                ));
            }
            let entry = record(node, input);
            audit.insert(nonce, entry.clone());
            tracing::info!(
                node_id = node.id,
                "builtin DCM action recorded in process-local audit"
            );
            json!({"effect": "audit_append", "record": entry, "audit_records": audit.len(),
                "durability": "process_local"})
        };
        Ok(OperatorOutput {
            state_delta: delta,
            artifacts: vec![],
            exit_code: 0,
            execution_time_ns: 0,
        })
    }
    fn verify_postconditions(
        &self,
        output: &OperatorOutput,
        input: &OperatorInput,
        node: &LodNode,
    ) -> Result<PostconditionReport, LodError> {
        let mut violations = vec![];
        if input.node_id != node.id {
            violations.push("execution target differs from input".into());
        }
        if self.is_pure() {
            if output.state_delta["evaluation"] != summary(node)
                || output.state_delta["effect"] != "none"
            {
                violations.push("evaluation differs from state".into());
            }
        } else {
            let expected = record(node, input);
            let audit = self.audit.lock();
            if input.nonce.and_then(|n| audit.get(&n)) != Some(&expected)
                || output.state_delta["record"] != expected
            {
                violations.push("audit does not match input, output and state".into());
            }
        }
        Ok(PostconditionReport {
            valid: violations.is_empty(),
            invariants_checked: vec![
                "target binding".into(),
                "output matches state and execution evidence".into(),
            ],
            violations,
        })
    }
}
pub(crate) fn register(graph: &LodGraph) -> Result<(), LodError> {
    let audit = Audit::default();
    for signature in [dcm_signature(), pcm_signature()] {
        graph.register_operator(Arc::new(Builtin {
            signature,
            audit: audit.clone(),
        }))?;
    }
    Ok(())
}
