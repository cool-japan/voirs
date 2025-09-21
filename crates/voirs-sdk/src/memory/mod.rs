//! Memory management and optimization for VoiRS SDK
//!
//! This module provides efficient memory pool management, resource tracking,
//! and optimization strategies for high-performance audio synthesis.

pub mod optimization;
pub mod pools;
pub mod tracking;

pub use optimization::{MemoryLayout, MemoryOptimizer, OptimizationStrategy};
pub use pools::{AudioBufferPool, MemoryPool, PoolConfig, TensorPool};
pub use tracking::{MemoryTracker, ResourceStats, ResourceTracker};
