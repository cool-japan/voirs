//! Statistical analysis modules for evaluation metrics
//!
//! This module provides comprehensive statistical analysis capabilities
//! for speech evaluation metrics, including correlation analysis,
//! regression testing, and experimental design.

pub mod ab_testing;
pub mod basic_tests;
pub mod causal_inference;
pub mod correlation;
pub mod experimental_design;
pub mod regression;
pub mod time_series;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use basic_tests::*;
pub use correlation::{
    CorrelationAnalyzer, CorrelationMatrix, CorrelationMethod, CorrelationResult,
    PartialCorrelationResult,
};
pub use regression::*;
pub use time_series::*;
pub use types::*;
pub use utils::*;
