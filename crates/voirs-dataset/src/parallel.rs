//! Parallel processing utilities for dataset operations
//!
//! This module provides efficient parallel processing capabilities including:
//! - Worker thread management
//! - Memory pool allocation and reuse
//! - Progress tracking and ETA calculation
//! - Load balancing across workers

pub mod memory;
pub mod progress;
pub mod workers;

pub use memory::*;
pub use progress::*;
pub use workers::*;
