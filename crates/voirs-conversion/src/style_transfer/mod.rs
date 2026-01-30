//! # Voice Style Transfer
//!
//! This module provides advanced voice style transfer capabilities, allowing transfer
//! of speaking styles, mannerisms, and vocal characteristics between voices while
//! preserving linguistic content.

// Module declarations
pub mod characteristics;
pub mod components;
pub mod config;
pub mod models;
pub mod system;
pub mod traits;

// Re-export main types and traits
pub use characteristics::*;
pub use components::*;
pub use config::*;
pub use models::*;
pub use system::{
    CachedStyleTransfer, StyleModelRepository, StyleTransferMetrics, StyleTransferSystem,
};
pub use traits::*;
