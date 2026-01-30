//! Performance optimization modules for VoiRS FFI
//!
//! This module provides comprehensive performance optimization utilities including:
//! - FFI call overhead reduction through batching and caching
//! - Memory management optimizations with pool allocation
//! - Threading optimizations with work-stealing and NUMA awareness
//! - Language-specific optimizations for Python, C, and Node.js
//! - Inline hints and compiler optimization directives

pub mod c;
pub mod ffi;
pub mod inline_hints;
pub mod memory;
pub mod nodejs;
pub mod python;
pub mod threading;

// Re-export commonly used items
pub use inline_hints::{
    assume, black_box, cold_path, compiler_fence, fast_copy_aligned, fast_zero_aligned,
    likely, memory_fence, prefetch_read, prefetch_write, spin_loop_hint, unlikely,
};
