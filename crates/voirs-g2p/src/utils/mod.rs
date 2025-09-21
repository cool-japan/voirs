//! Utility modules for G2P processing.
//!
//! This module is organized into specialized sub-modules for better maintainability:
//! - `text_processing`: Text preprocessing and postprocessing utilities
//! - `phoneme_analysis`: Phoneme validation and analysis functions
//! - `diagnostics`: Error diagnostics and profiling utilities  
//! - `quality`: Quality scoring and validation utilities

pub mod diagnostics;
pub mod phoneme_analysis;
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

// Additional re-exports for full compatibility
pub use phoneme_analysis::get_valid_phoneme_inventory;
