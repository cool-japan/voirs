//! Statistical analysis modules for evaluation metrics
//!
//! This module provides comprehensive statistical analysis capabilities
//! for speech evaluation metrics, including correlation analysis,
//! regression testing, and experimental design.

pub mod ab_testing;
pub mod basic_tests;
pub mod bayesian;
pub mod causal_inference;
pub mod correlation;
pub mod experimental_design;
pub mod regression;
pub mod time_series;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use basic_tests::*;
pub use bayesian::{
    BayesianABTestResult, BayesianAnalyzer, BayesianEstimation, BayesianModelComparison,
    PriorParameters, PriorType,
};
pub use causal_inference::{
    covariate_balance, difference_in_differences, nearest_neighbor_matching, DidResult,
    MatchingResult,
};
pub use correlation::{
    CorrelationAnalyzer, CorrelationMatrix, CorrelationMethod, CorrelationResult,
    PartialCorrelationResult,
};
pub use experimental_design::{minimum_detectable_effect, required_sample_size};
pub use regression::*;
pub use time_series::*;
pub use types::*;
pub use utils::*;
