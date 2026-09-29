//! Capability-Level Permission Arbiter and Agent Sandbox Registry.
//!
//! Features:
//! - Granular capability-based access control (Execute, Query, Rollback, Admin)
//! - Per-agent capability tokens
//! - Zero-allocation bitflag permission checks

use crate::error::ProvenanceError;
use parking_lot::RwLock;
use rand::RngCore;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bitflag representations of agent capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityFlags(pub u32);

impl CapabilityFlags {
    pub const NONE: Self = Self(0);
    pub const QUERY_STATE: Self = Self(1 << 0);
    pub const EXECUTE_REFLEX: Self = Self(1 << 1);
    pub const EXECUTE_LOOKAHEAD: Self = Self(1 << 2);
    pub const TRIGGER_ROLLBACK: Self = Self(1 << 3);
    pub const MODIFY_GRAPH: Self = Self(1 << 4);
    pub const AUDIT_ADMIN: Self = Self(1 << 5);
    pub const ALL: Self = Self(0x3F);

    #[inline]
    pub fn contains(&self, required: Self) -> bool {
        (self.0 & required.0) == required.0
    }

    #[inline]
    pub fn add(&mut self, flag: Self) {
        self.0 |= flag.0;
    }
}

/// Token representing a cryptographically signed agent authorization session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityToken {
    pub agent_id: String,
    pub capabilities: CapabilityFlags,
    pub session_id: u64,
    pub epoch: u64,
    pub signature: [u8; 32], // Keyed BLAKE3 MAC over token payload
}

#[inline]
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Central Capability Arbiter coordinating sandboxed agent permissions.
pub struct CapabilityArbiter {
    key: [u8; 32],
    salt: [u8; 32],
    registered_agents: RwLock<HashMap<String, (CapabilityFlags, u64)>>,
    next_epoch: AtomicU64,
}

impl CapabilityArbiter {
    pub fn new(key: [u8; 32]) -> Self {
        let mut salt = [0; 32];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        Self {
            key,
            salt,
            registered_agents: RwLock::new(HashMap::new()),
            next_epoch: AtomicU64::new(1),
        }
    }

    /// Register or update capabilities for an agent.
    pub fn register_agent(&self, agent_id: &str, capabilities: CapabilityFlags) {
        let epoch = self.next_epoch.fetch_add(1, Ordering::SeqCst);
        assert!(epoch != u64::MAX, "capability epoch exhausted");
        self.registered_agents
            .write()
            .insert(agent_id.to_string(), (capabilities, epoch));
    }

    /// Revoke or deregister an agent.
    pub fn deregister_agent(&self, agent_id: &str) -> Option<CapabilityFlags> {
        self.registered_agents
            .write()
            .remove(agent_id)
            .map(|(caps, _)| caps)
    }

    /// Sign token fields using Keyed BLAKE3 with length prefixing to prevent delimiter ambiguity.
    fn sign_token(
        &self,
        agent_id: &str,
        capabilities: CapabilityFlags,
        session_id: u64,
        epoch: u64,
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(&self.salt);
        hasher.update(&(agent_id.len() as u64).to_le_bytes());
        hasher.update(agent_id.as_bytes());
        hasher.update(&capabilities.0.to_le_bytes());
        hasher.update(&session_id.to_le_bytes());
        hasher.update(&epoch.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Issue a signed capability token for an authorized agent.
    pub fn issue_token(
        &self,
        agent_id: &str,
        session_id: u64,
    ) -> Result<CapabilityToken, ProvenanceError> {
        let (caps, epoch) = self
            .registered_agents
            .read()
            .get(agent_id)
            .copied()
            .ok_or_else(|| ProvenanceError::PermissionDenied {
                agent_id: agent_id.to_string(),
                capability: "AGENT_NOT_REGISTERED".to_string(),
            })?;

        let signature = self.sign_token(agent_id, caps, session_id, epoch);

        Ok(CapabilityToken {
            agent_id: agent_id.to_string(),
            capabilities: caps,
            session_id,
            epoch,
            signature,
        })
    }

    /// Arbitrate whether a token has the requested capability, verifying HMAC signature and registry.
    pub fn authorize(
        &self,
        token: &CapabilityToken,
        required: CapabilityFlags,
    ) -> Result<(), ProvenanceError> {
        // 1. Verify cryptographic signature in constant time
        let expected_sig = self.sign_token(
            &token.agent_id,
            token.capabilities,
            token.session_id,
            token.epoch,
        );
        if !constant_time_eq(&token.signature, &expected_sig) {
            return Err(ProvenanceError::PermissionDenied {
                agent_id: token.agent_id.clone(),
                capability: "INVALID_TOKEN_SIGNATURE".to_string(),
            });
        }

        // 2. Verify agent is still active in registry with matching capabilities
        let (current_caps, current_epoch) = self
            .registered_agents
            .read()
            .get(&token.agent_id)
            .copied()
            .ok_or_else(|| ProvenanceError::PermissionDenied {
                agent_id: token.agent_id.clone(),
                capability: "AGENT_DEREGISTERED".to_string(),
            })?;

        if current_epoch != token.epoch
            || !current_caps.contains(required)
            || !token.capabilities.contains(required)
        {
            return Err(ProvenanceError::PermissionDenied {
                agent_id: token.agent_id.clone(),
                capability: format!("{:08b}", required.0),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_arbitration() {
        let key = [0x77u8; 32];
        let arbiter = CapabilityArbiter::new(key);
        let mut caps = CapabilityFlags::QUERY_STATE;
        caps.add(CapabilityFlags::EXECUTE_REFLEX);

        arbiter.register_agent("agent_007", caps);

        let token = arbiter.issue_token("agent_007", 101).unwrap();
        assert!(arbiter
            .authorize(&token, CapabilityFlags::QUERY_STATE)
            .is_ok());
        assert!(arbiter
            .authorize(&token, CapabilityFlags::EXECUTE_REFLEX)
            .is_ok());
        assert!(arbiter
            .authorize(&token, CapabilityFlags::TRIGGER_ROLLBACK)
            .is_err());

        // Test signature tampering rejection
        let mut forged_token = token.clone();
        forged_token.signature[0] ^= 0xFF;
        assert!(arbiter
            .authorize(&forged_token, CapabilityFlags::QUERY_STATE)
            .is_err());

        // Test revocation / deregistration
        arbiter.deregister_agent("agent_007");
        assert!(arbiter
            .authorize(&token, CapabilityFlags::QUERY_STATE)
            .is_err());
        arbiter.register_agent("agent_007", caps);
        assert!(arbiter
            .authorize(&token, CapabilityFlags::QUERY_STATE)
            .is_err());
        let renewed = arbiter.issue_token("agent_007", 101).unwrap();
        assert!(arbiter
            .authorize(&renewed, CapabilityFlags::QUERY_STATE)
            .is_ok());
        let restarted = CapabilityArbiter::new(key);
        restarted.register_agent("agent_007", caps);
        assert!(restarted
            .authorize(&renewed, CapabilityFlags::QUERY_STATE)
            .is_err());
    }
}
