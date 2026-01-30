//! Auto-generated module structure

pub mod functions;
pub mod structs;

// TtsPipeline implementation modules (refactored from large types.rs)
pub mod g2p_backend;
pub mod pipeline_core;
pub mod stress_and_phonology;
pub mod text_processing;
pub mod unknown_word_handling;

// Re-export all types
pub use functions::*;
pub use structs::*;
