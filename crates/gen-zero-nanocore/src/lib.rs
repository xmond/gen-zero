//! # gen-zero-nanocore
//!
//! Quantized micro-kernel fleet, Mixture of Vectors (MoV) vector fusion,
//! Fallback Watchdog, and bounded-RAM LRU Fleet Scheduler.

#![allow(deprecated)]

pub mod core_type;
pub mod error;
pub mod mov;
pub mod scheduler;

pub use core_type::{
    DomainId, NanoCoreInstance, DOMAIN_BROWSER, DOMAIN_CODE, DOMAIN_GENERAL, DOMAIN_SAFETY,
    DOMAIN_SQL, DOMAIN_TRADING, DOMAIN_VISION, MAX_OUT_DIM,
};
pub use error::NanoCoreError;
pub use mov::{FusedDecision, MoVFusionEngine, WatchdogConfig};
pub use scheduler::{NanoCoreFleetScheduler, DEFAULT_RAM_BUDGET_BYTES};
