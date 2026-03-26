//! Backend implementations for voice conversion
//!
//! This module provides abstraction layers for different inference backends
//! including ONNX Runtime for voice conversion pipelines.

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub use onnx::*;
