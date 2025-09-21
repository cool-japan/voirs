//! Research tools for dataset analysis and experimentation
//!
//! This module provides tools for researchers to:
//! - Track experiments and dataset versions
//! - Generate statistical analysis and visualizations
//! - Run standardized benchmarks and evaluations

pub mod analysis;
pub mod benchmarks;
pub mod experiments;

pub use analysis::*;
pub use benchmarks::*;
pub use experiments::*;
