//! Model runtime module for unified ONNX and model loading.
//!
//! Provides a high-level wrapper around OxiONNX sessions with configurable
//! optimization levels, profiling, GPU support, and format auto-detection.

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod loader;

#[cfg(feature = "onnx")]
pub use onnx::{OnnxSession, OnnxSessionConfig, ProfilingSummary};

pub use loader::{DetectedFormat, ModelFormatDetector};
