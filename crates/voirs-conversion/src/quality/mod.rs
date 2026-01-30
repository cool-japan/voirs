//! Quality assessment and artifact detection for voice conversion
//!
//! This module provides comprehensive quality analysis tools for voice conversion,
//! including artifact detection, perceptual optimization, objective metrics,
//! and adaptive quality control.

// Module declarations
pub mod adaptive_controller;
pub mod artifact_detection;
pub mod metrics;
pub mod perceptual;
pub mod targets;

// Re-export public types for backward compatibility

// From adaptive_controller
pub use adaptive_controller::{
    AdaptiveAdjustmentResult, AdaptiveQualityController, QualityStrategy,
    QualityStrategyAdjustment, QualityTrend, QualityTrigger, StrategyStats,
};

// From artifact_detection
pub use artifact_detection::{
    AdaptiveState, AdjustmentType, ArtifactDetector, ArtifactLocation, ArtifactThresholds,
    ArtifactType, ConfidenceStats, DetectedArtifacts, MemoryPool, ProductionMetrics,
    QualityAdjustment, QualityAssessment,
};

// From metrics
pub use metrics::{
    ObjectiveQualityMetrics, PerceptualParameters, QualityFeatures, QualityMetricsSystem,
};

// From perceptual
pub use perceptual::{
    CriticalBandAnalysis, CriticalBandAnalyzer, LoudnessAnalysis, LoudnessModel, MaskingAnalysis,
    MaskingCalculator, PerceptualOptimizationParams, PerceptualOptimizationResult,
    PerceptualOptimizer, PsychoacousticModel, SimultaneousMaskingParams, TemporalMaskingParams,
};

// From targets
pub use targets::{
    DetailedQualityMetrics, QualityTargetMeasurement, QualityTargetsAchievement,
    QualityTargetsConfig, QualityTargetsStatistics, QualityTargetsSystem,
};
