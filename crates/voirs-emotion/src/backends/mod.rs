//! Backend implementations for emotion processing
//!
//! This module provides abstraction layers for different inference backends
//! including ONNX Runtime for emotion classification.

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub use onnx::*;
