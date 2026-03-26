//! Backend implementations for voice cloning
//!
//! This module provides abstraction layers for different inference backends
//! including ONNX Runtime for speaker encoding and voice cloning.

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub use onnx::*;
