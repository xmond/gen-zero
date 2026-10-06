//! `TriTeacherPairDecider` as the stage 2 verifier of
//! `gen_zero_gate::TwoStageDualTrackGateway`.
//!
//! Only the raw head projections cross this boundary. On the gateway path the
//! gate owns the cosine, teacher weighting, threshold and every fail-closed
//! check. `TriTeacherPairDecider::decide` (`decide_projections` in
//! `tri_teacher.rs`) is a second implementation of the same formula that this
//! path never calls.

use crate::tri_teacher::TriTeacherPairDecider;
use gen_zero_gate::{CausalVerifier, GateError, TriTeacherProjections};

impl CausalVerifier for TriTeacherPairDecider {
    fn verifier_id(&self) -> String {
        let info = self.info();
        let adapter = info
            .adapter
            .as_ref()
            .map_or("no-adapter", |a| &a.sha256[..a.sha256.len().min(16)]);
        let encoder = info.encoder.as_ref().map_or("no-encoder", |e| {
            &e.base_sha256[..e.base_sha256.len().min(16)]
        });
        format!("qwen2-native+tri-teacher-lora:adapter={adapter}:base={encoder}")
    }

    fn project_pair(
        &self,
        sentence_1: &str,
        sentence_2: &str,
    ) -> Result<TriTeacherProjections, GateError> {
        self.project_texts(sentence_1, sentence_2)
            .map_err(|e| GateError::VerifierFailure(e.to_string()))
    }
}
