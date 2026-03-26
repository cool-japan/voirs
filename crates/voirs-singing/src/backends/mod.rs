//! Backend implementations for singing synthesis models.
//!
//! This module provides various backend implementations for singing synthesis,
//! including ONNX-based inference via oxionnx.

#[cfg(feature = "onnx")]
pub mod onnx;
