//! Utility modules for vocoder operations
//!
//! This module contains various utility functions and helpers:
//! - Audio processing utilities
//! - Performance monitoring and benchmarking
//! - Quality assessment and validation

pub mod audio_processing;
pub mod helpers;
pub mod performance;
pub mod quality_assessment;

// Re-export audio processing functions
pub use audio_processing::*;

// Re-export helper functions
pub use helpers::{
    batch_vocode, concatenate_audio_buffers, create_mel_spectrogram_validated,
    normalize_mel_spectrogram, validate_mel_spectrogram, vocode_with_timing, MelValidation,
    ProcessingTiming,
};

// Re-export performance monitoring types
pub use performance::{
    PerformanceMetrics, PerformanceMonitor, PerformanceStatistics, PerformanceTier,
    PerformanceTimer,
};

// Re-export quality assessment types
pub use quality_assessment::{
    compare_quality, quick_quality_check, QualityAssessment, QualityIssue, QualityTier,
};
