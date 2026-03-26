//! Pronunciation types module - split for maintainability

pub mod evaluator_core;
pub mod evaluator_impl2;
pub mod extended_types;

// Re-export all public types
pub use evaluator_core::*;
pub use extended_types::*;
