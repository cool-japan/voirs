//! FFI utility functions and audio processing utilities.
//!
//! This module has been refactored into separate submodules for better maintainability:
//! - `string_utils`: String conversion utilities for FFI
//! - `audio`: Audio processing and analysis utilities
//! - `performance`: Performance monitoring and optimization utilities
//! - `batch_ops`: Batch operation utilities for efficient processing
//! - `diagnostics`: Diagnostic utilities for troubleshooting

// Re-export string utilities at the top level
mod string_utils;
pub use string_utils::{
    c_str_to_str, create_string_array, free_string_array, string_to_owned_c_str,
};

// Re-export audio and performance modules
pub mod audio;
pub mod batch_ops;
pub mod diagnostics;
pub mod performance;
