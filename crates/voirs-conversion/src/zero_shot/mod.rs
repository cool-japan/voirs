//! # Zero-shot Voice Conversion
//!
//! This module provides zero-shot voice conversion capabilities, allowing conversion
//! to target voices without requiring extensive training data or adaptation.

// Module declarations
pub mod config;
pub mod converter;
pub mod database;
pub mod metrics;
pub mod models;
pub mod quality;
pub mod style;

#[cfg(test)]
mod tests;

// Re-export main types for convenience
pub use config::{
    AdaptationSettings, PerformanceConstraints, QualityPreservationSettings, ZeroShotConfig,
    ZeroShotMethod,
};

pub use converter::ZeroShotConverter;

pub use database::{
    AudioSample, DatabaseMetadata, IndexStatistics, PhoneticAnalysis, ProsodicFeatures,
    QualityScores, ReferenceVoice, ReferenceVoiceDatabase, SearchPerformanceMetrics,
    SpeakerEmbedding, UsageStatistics, VoiceMetadata,
};

pub use metrics::{CachedConversion, PerformanceMetrics, ZeroShotMetrics};

pub use models::{
    AdaptedModel, BenchmarkResult, FeatureExtractor, ModelMetadata, ModelParameters, TrainingInfo,
    UniversalVoiceModel, VoiceGenerator,
};

pub use quality::{
    AssessmentFrequency, AssessmentMode, QualityAssessment, QualityAssessmentConfig,
    QualityAssessor, QualityClassification, QualityMetric, QualityThresholds,
};

pub use style::{
    ProsodicStyleFeatures, SpectralStyleFeatures, StyleAnalysis, StyleAnalysisConfig,
    StyleAnalyzer, StyleComparator, StyleExtractor, StyleFeatures, StyleSimilarity,
    TemporalStyleFeatures, VoiceQualityFeatures,
};
