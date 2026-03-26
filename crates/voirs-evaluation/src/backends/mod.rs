//! Backend implementations for evaluation processing.
//!
//! This module provides optional backend implementations:
//! - ONNX-based MOS prediction via oxionnx

#[cfg(feature = "onnx")]
pub mod onnx;
