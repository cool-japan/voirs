//! Comprehensive test suite for voirs-vocoder
//!
//! This test suite provides complete coverage of the vocoder implementation
//! including unit tests, integration tests, and quality assessment tests.

#[cfg(test)]
mod unit;

#[cfg(test)]
mod integration;

#[cfg(test)]
mod quality;

// Re-export test modules for convenience
pub use integration::*;
pub use quality::*;
pub use unit::*;
