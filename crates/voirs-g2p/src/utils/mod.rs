//! Utility modules for G2P processing.
//!
//! This module is organized into specialized sub-modules for better maintainability:
//! - `text_processing`: Text preprocessing and postprocessing utilities
//! - `phoneme_analysis`: Phoneme validation and analysis functions
//! - `phoneme_similarity`: Phoneme distance and similarity metrics
//! - `phoneme_simd`: SIMD-accelerated batch phoneme operations
//! - `phoneme_normalization`: Cross-language phoneme normalization utilities
//! - `phoneme_stats`: Statistical analysis and metrics for phoneme sequences
//! - `phoneme_validator`: Enhanced validation with detailed error messages and suggestions
//! - `diagnostics`: Error diagnostics and profiling utilities
//! - `quality`: Quality scoring and validation utilities

pub mod diagnostics;
pub mod phoneme_analysis;
pub mod phoneme_normalization;
pub mod phoneme_simd;
pub mod phoneme_similarity;
pub mod phoneme_stats;
pub mod phoneme_validator;
pub mod quality;
pub mod text_processing;

// Re-export commonly used items for backward compatibility
pub use text_processing::{
    detect_syllable_boundaries, postprocess_phonemes, preprocess_text, preprocess_text_simple,
    preprocess_text_with_config,
};

pub use phoneme_analysis::{
    analyze_phoneme_sequence, get_valid_phoneme_inventory_slice, is_consonant, is_vowel,
    validate_phonemes, PhonemeAnalysis,
};

pub use phoneme_similarity::{
    calculate_phoneme_error_rate, phoneme_feature_similarity, phoneme_levenshtein_distance,
    phoneme_sequence_similarity,
};

pub use phoneme_simd::{
    batch_count_phonemes, batch_distance_matrix, batch_phoneme_error_rate,
    batch_phoneme_similarity, batch_validate_phonemes, parallel_batch_process, PhonemeCount,
};

pub use diagnostics::{
    batch_process_phonemes, create_diagnostic_context, create_diagnostic_context_with_details,
    create_g2p_error_report, diagnose_conversion_error, extract_phonetic_features,
    DiagnosticReport, G2pProfiler, PerformanceReport, PerformanceStage,
};

pub use quality::{
    generate_conversion_debug_summary, score_phoneme_quality, segment_into_sentences,
    segment_text_for_streaming, validate_phoneme_sequence_advanced, ConversionDebugSummary,
    PhonemeQualityScore, TextSegment, ValidationError, ValidationReport,
};

pub use phoneme_normalization::{
    detect_phoneme_set, ipa_to_arpabet, normalize_phonemes, phoneme_equivalence,
    NormalizationConfig, PhonemeSet,
};

pub use phoneme_stats::PhonemeStatistics;

pub use phoneme_validator::{
    ErrorCategory, PhonemeValidationError, PhonemeValidationResult, PhonemeValidator,
    ValidationLevel,
};

// Additional re-exports for full compatibility
pub use phoneme_analysis::get_valid_phoneme_inventory;
