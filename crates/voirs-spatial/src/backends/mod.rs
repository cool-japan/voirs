//! Backend implementations for spatial audio processing.
//!
//! This module provides optional backend implementations:
//! - ONNX-based neural HRTF synthesis via oxionnx

#[cfg(feature = "onnx")]
pub mod onnx;
