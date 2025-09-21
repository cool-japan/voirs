//! Analytics module
//!
//! Comprehensive analytics and data collection system for user feedback,
//! performance tracking, and system monitoring.

pub mod ab_testing;
pub mod core;
pub mod data;
pub mod memory_optimization;
pub mod metrics;
pub mod reports;
pub mod types;

// Re-export main types for backward compatibility
pub use core::*;
pub use data::*;
pub use metrics::*;
pub use reports::*;
pub use types::*;

// Memory optimization exports (namespaced to avoid conflicts)
pub use memory_optimization::{
    BoundedMetadata, CompactInteractionSummary, OptimizedDataCollector, OptimizedSessionData,
    OptimizedUserInteractionEvent,
};

// Enhanced core exports with memory optimization
pub use core::{
    AnalyticsManagerFactory, AnalyticsManagerTrait, ComprehensiveMemoryStats, MemoryCleanupResult,
    MemoryProfile, OptimizationMetrics, OptimizedAnalyticsManager,
};

// A/B testing framework exports
pub use ab_testing::{
    ABTestConfig, ABTestError, ABTestManager, Experiment, ExperimentMetrics, ExperimentStatus,
    StatisticalResult, SuccessCriteria, UserAssignment, Variant, VariantConfig,
};
