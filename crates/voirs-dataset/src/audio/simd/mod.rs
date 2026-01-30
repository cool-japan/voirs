//! SIMD module organization
//!
//! This module provides both low-level raw intrinsics for direct audio buffer
//! processing and high-level SciRS2-Core integrated operations.

pub mod enhanced;
pub mod intrinsics;

#[cfg(test)]
pub mod tests;

// Re-export both for convenience
pub use enhanced::EnhancedSimdProcessor;
pub use intrinsics::SimdAudioProcessor;
