//! Core data structures and type definitions for real-time feedback

use crate::adaptive::FeatureVector;
use crate::traits::{
    ExerciseType, FeedbackContext, FeedbackProvider, FeedbackResponse, FocusArea,
    LearningRecommendation, ProgressIndicators, ProgressSnapshot, SessionState, SuccessCriteria,
    TrainingExercise, UserFeedback, UserPreferences,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;
use voirs_sdk::AudioBuffer;

/// Phoneme difficulty mapping for common pronunciation issues
pub static PHONEME_DIFFICULTY: &[(char, f32)] = &[
    ('θ', 0.9), // th (thin)
    ('ð', 0.9), // th (this)
    ('ʃ', 0.8), // sh
    ('ʒ', 0.8), // zh (vision)
    ('ɹ', 0.7), // r
    ('l', 0.6), // l
    ('f', 0.5), // f
    ('v', 0.5), // v
    ('w', 0.4), // w
    ('j', 0.4), // y
];

// ============================================================================
// Phoneme Analysis Types
// ============================================================================

/// Phoneme-specific analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeAnalysis {
    /// Identified phonemes in the text
    pub phonemes: Vec<PhonemeInfo>,
    /// Overall phoneme difficulty score
    pub difficulty_score: f32,
    /// Problematic phonemes requiring attention
    pub problematic_phonemes: Vec<String>,
    /// Confidence in analysis
    pub confidence: f32,
}

/// Information about a specific phoneme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeInfo {
    /// Phoneme symbol
    pub symbol: String,
    /// Position in text (character index)
    pub position: usize,
    /// Difficulty level [0.0-1.0]
    pub difficulty: f32,
    /// Common mispronunciation patterns
    pub common_errors: Vec<String>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

// ============================================================================
// Prosody Analysis Types
// ============================================================================

/// Prosody analysis result for speech patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackProsodyAnalysis {
    /// Intonation patterns analysis
    pub intonation: FeedbackIntonationAnalysis,
    /// Stress patterns analysis
    pub stress: FeedbackStressAnalysis,
    /// Rhythm and timing analysis
    pub rhythm: FeedbackRhythmAnalysis,
    /// Overall prosody score [0.0-1.0]
    pub overall_score: f32,
    /// Confidence in analysis
    pub confidence: f32,
}

/// Intonation pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackIntonationAnalysis {
    /// Pitch range variation
    pub pitch_range: f32,
    /// Pitch contour smoothness
    pub contour_smoothness: f32,
    /// Appropriate rising/falling patterns
    pub pattern_appropriateness: f32,
    /// Detected intonation issues
    pub issues: Vec<IntonationIssue>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

/// Stress pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStressAnalysis {
    /// Syllable stress accuracy
    pub syllable_stress: f32,
    /// Word stress patterns
    pub word_stress: f32,
    /// Sentence stress appropriateness
    pub sentence_stress: f32,
    /// Detected stress issues
    pub issues: Vec<StressIssue>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

/// Rhythm and timing analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRhythmAnalysis {
    /// Speaking rate consistency
    pub rate_consistency: f32,
    /// Pause placement appropriateness
    pub pause_placement: f32,
    /// Syllable timing regularity
    pub timing_regularity: f32,
    /// Detected rhythm issues
    pub issues: Vec<RhythmIssue>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

// ============================================================================
// Issue Types
// ============================================================================

/// Specific intonation issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntonationIssue {
    /// Insufficient pitch variation in speech
    FlatIntonation,
    /// Too much pitch variation making speech unnatural
    ExcessivePitchVariation,
    /// Rising intonation used inappropriately
    InappropriateRisingPattern,
    /// Falling intonation used inappropriately
    InappropriateFallingPattern,
    /// Monotone speech delivery
    MonotoneDelivery,
    /// Abrupt pitch changes or breaks
    PitchBreaks,
}

/// Specific stress issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressIssue {
    /// Primary stress placed on wrong syllable
    IncorrectPrimaryStress,
    /// Secondary stress not properly emphasized
    MissingSecondaryStress,
    /// Excessive stress on syllables
    OverStressing,
    /// Insufficient stress on important syllables
    UnderStressing,
    /// All syllables stressed equally
    EqualStressPattern,
    /// Stress timing doesn't match natural rhythm
    StressTimingMismatch,
}

/// Specific rhythm issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RhythmIssue {
    /// Speaking rate varies too much
    InconsistentSpeakingRate,
    /// Pauses placed at inappropriate locations
    PoorPausePlacement,
    /// Syllables spoken too quickly
    RushedSyllables,
    /// Syllables elongated unnecessarily
    DrawnOutSyllables,
    /// Missing pauses at sentence boundaries
    MissingSentenceBoundaryPauses,
    /// Irregular stress-timed rhythm pattern
    IrregularStressTimedRhythm,
}

// ============================================================================
// Confidence and Filtering Types
// ============================================================================

/// Bayesian confidence estimation for feedback filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianConfidenceFilter {
    /// Prior confidence in different feedback types
    pub priors: ConfidencePriors,
    /// Evidence weights for different metrics
    pub evidence_weights: EvidenceWeights,
    /// Threshold for accepting feedback
    pub acceptance_threshold: f32,
    /// Historical accuracy for feedback types
    pub historical_accuracy: HistoricalAccuracy,
}

/// Prior confidence values for different feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidencePriors {
    /// Prior for phoneme-specific feedback
    pub phoneme_feedback: f32,
    /// Prior for prosody feedback
    pub prosody_feedback: f32,
    /// Prior for quality feedback
    pub quality_feedback: f32,
    /// Prior for pronunciation feedback
    pub pronunciation_feedback: f32,
}

/// Evidence weights for different metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceWeights {
    /// Weight for audio quality metrics
    pub quality_weight: f32,
    /// Weight for pronunciation accuracy
    pub pronunciation_weight: f32,
    /// Weight for prosody analysis
    pub prosody_weight: f32,
    /// Weight for phoneme analysis
    pub phoneme_weight: f32,
}

/// Historical accuracy for feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalAccuracy {
    /// Accuracy history for phoneme feedback
    pub phoneme_accuracy: VecDeque<f32>,
    /// Accuracy history for prosody feedback
    pub prosody_accuracy: VecDeque<f32>,
    /// Accuracy history for quality feedback
    pub quality_accuracy: VecDeque<f32>,
    /// Maximum history length
    pub max_history_length: usize,
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Real-time feedback system configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Maximum latency for feedback generation (milliseconds)
    pub max_latency_ms: u64,
    /// Stream timeout duration
    pub stream_timeout: Duration,
    /// Audio buffer size for processing
    pub audio_buffer_size: usize,
    /// Maximum number of concurrent streams
    pub max_concurrent_streams: usize,
    /// Enable detailed metrics collection
    pub enable_metrics: bool,
    /// Enable Bayesian confidence filtering
    pub enable_confidence_filtering: bool,
    /// Quality evaluation threshold
    pub quality_threshold: f32,
    /// Pronunciation accuracy threshold
    pub pronunciation_threshold: f32,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 100,
            stream_timeout: Duration::from_secs(300),
            audio_buffer_size: 1024,
            max_concurrent_streams: 10,
            enable_metrics: true,
            enable_confidence_filtering: true,
            quality_threshold: 0.7,
            pronunciation_threshold: 0.8,
        }
    }
}

/// Real-time system statistics
#[derive(Debug, Clone, Default)]
pub struct RealtimeStats {
    /// Number of currently active streams
    pub active_streams: usize,
    /// Total number of processed audio chunks
    pub total_chunks_processed: u64,
    /// Average processing latency in milliseconds
    pub average_latency_ms: f32,
    /// Total number of feedback items generated
    pub total_feedback_generated: u64,
    /// Average confidence score
    pub average_confidence: f32,
    /// System uptime
    pub uptime: Duration,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_realtime_config_default() {
        let config = RealtimeConfig::default();

        assert_eq!(config.max_latency_ms, 100);
        assert_eq!(config.stream_timeout, Duration::from_secs(300));
        assert_eq!(config.audio_buffer_size, 1024);
        assert_eq!(config.max_concurrent_streams, 10);
        assert!(config.enable_metrics);
        assert!(config.enable_confidence_filtering);
        assert_eq!(config.quality_threshold, 0.7);
        assert_eq!(config.pronunciation_threshold, 0.8);
    }

    #[test]
    fn test_realtime_config_custom() {
        let config = RealtimeConfig {
            max_latency_ms: 50,
            stream_timeout: Duration::from_secs(600),
            audio_buffer_size: 2048,
            max_concurrent_streams: 20,
            enable_metrics: false,
            enable_confidence_filtering: false,
            quality_threshold: 0.9,
            pronunciation_threshold: 0.95,
        };

        assert_eq!(config.max_latency_ms, 50);
        assert_eq!(config.stream_timeout, Duration::from_secs(600));
        assert_eq!(config.audio_buffer_size, 2048);
        assert_eq!(config.max_concurrent_streams, 20);
        assert!(!config.enable_metrics);
        assert!(!config.enable_confidence_filtering);
        assert_eq!(config.quality_threshold, 0.9);
        assert_eq!(config.pronunciation_threshold, 0.95);
    }

    #[test]
    fn test_realtime_stats_default() {
        let stats = RealtimeStats::default();

        assert_eq!(stats.active_streams, 0);
        assert_eq!(stats.total_chunks_processed, 0);
        assert_eq!(stats.average_latency_ms, 0.0);
        assert_eq!(stats.total_feedback_generated, 0);
        assert_eq!(stats.average_confidence, 0.0);
        assert_eq!(stats.uptime, Duration::from_secs(0));
        assert_eq!(stats.peak_memory_usage, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }

    #[test]
    fn test_realtime_stats_updates() {
        let mut stats = RealtimeStats::default();

        // Simulate some activity
        stats.active_streams = 5;
        stats.total_chunks_processed = 1000;
        stats.average_latency_ms = 45.5;
        stats.total_feedback_generated = 250;
        stats.average_confidence = 0.85;
        stats.uptime = Duration::from_secs(3600); // 1 hour
        stats.peak_memory_usage = 1024 * 1024; // 1MB
        stats.current_memory_usage = 512 * 1024; // 512KB

        assert_eq!(stats.active_streams, 5);
        assert_eq!(stats.total_chunks_processed, 1000);
        assert_eq!(stats.average_latency_ms, 45.5);
        assert_eq!(stats.total_feedback_generated, 250);
        assert_eq!(stats.average_confidence, 0.85);
        assert_eq!(stats.uptime, Duration::from_secs(3600));
        assert_eq!(stats.peak_memory_usage, 1024 * 1024);
        assert_eq!(stats.current_memory_usage, 512 * 1024);
    }

    #[test]
    fn test_phoneme_analysis_creation() {
        let analysis = PhonemeAnalysis {
            phonemes: vec![PhonemeInfo {
                symbol: "θ".to_string(),
                position: 0,
                difficulty: 0.9,
                common_errors: vec!["f".to_string(), "s".to_string()],
                suggestions: vec!["Place tongue between teeth".to_string()],
            }],
            difficulty_score: 0.8,
            problematic_phonemes: vec!["θ".to_string()],
            confidence: 0.95,
        };

        assert_eq!(analysis.phonemes.len(), 1);
        assert_eq!(analysis.phonemes[0].symbol, "θ");
        assert_eq!(analysis.phonemes[0].difficulty, 0.9);
        assert_eq!(analysis.difficulty_score, 0.8);
        assert!(analysis.problematic_phonemes.contains(&"θ".to_string()));
        assert_eq!(analysis.confidence, 0.95);
    }

    #[test]
    fn test_phoneme_info_details() {
        let phoneme = PhonemeInfo {
            symbol: "ɹ".to_string(),
            position: 5,
            difficulty: 0.7,
            common_errors: vec!["l".to_string(), "w".to_string()],
            suggestions: vec![
                "Curl tongue tip slightly".to_string(),
                "Don't touch tongue to teeth".to_string(),
            ],
        };

        assert_eq!(phoneme.symbol, "ɹ");
        assert_eq!(phoneme.position, 5);
        assert_eq!(phoneme.difficulty, 0.7);
        assert_eq!(phoneme.common_errors.len(), 2);
        assert_eq!(phoneme.suggestions.len(), 2);
        assert!(phoneme.common_errors.contains(&"l".to_string()));
        assert!(phoneme
            .suggestions
            .contains(&"Curl tongue tip slightly".to_string()));
    }

    #[test]
    fn test_prosody_analysis_creation() {
        let prosody = FeedbackProsodyAnalysis {
            intonation: FeedbackIntonationAnalysis {
                pitch_range: 0.8,
                contour_smoothness: 0.9,
                pattern_appropriateness: 0.7,
                issues: vec![],
                suggestions: vec!["Vary pitch more for questions".to_string()],
            },
            stress: FeedbackStressAnalysis {
                syllable_stress: 0.85,
                word_stress: 0.9,
                sentence_stress: 0.8,
                issues: vec![],
                suggestions: vec!["Emphasize stressed syllables more".to_string()],
            },
            rhythm: FeedbackRhythmAnalysis {
                rate_consistency: 0.75,
                pause_placement: 0.8,
                timing_regularity: 0.9,
                issues: vec![],
                suggestions: vec!["Slow down slightly".to_string()],
            },
            overall_score: 0.82,
            confidence: 0.88,
        };

        assert_eq!(prosody.intonation.pitch_range, 0.8);
        assert_eq!(prosody.stress.syllable_stress, 0.85);
        assert_eq!(prosody.rhythm.rate_consistency, 0.75);
        assert_eq!(prosody.overall_score, 0.82);
        assert_eq!(prosody.confidence, 0.88);
    }

    #[test]
    fn test_phoneme_difficulty_mapping() {
        // Test that the difficulty mapping contains expected phonemes
        let difficulty_map: HashMap<char, f32> = PHONEME_DIFFICULTY.iter().cloned().collect();

        assert!(difficulty_map.contains_key(&'θ'));
        assert!(difficulty_map.contains_key(&'ð'));
        assert!(difficulty_map.contains_key(&'ʃ'));
        assert!(difficulty_map.contains_key(&'ʒ'));
        assert!(difficulty_map.contains_key(&'ɹ'));

        // Test difficulty ordering
        assert!(difficulty_map[&'θ'] > difficulty_map[&'ɹ']);
        assert!(difficulty_map[&'ɹ'] > difficulty_map[&'w']);
        assert!(difficulty_map[&'ʃ'] > difficulty_map[&'f']);
    }

    #[test]
    fn test_config_threshold_bounds() {
        // Test extreme threshold values
        let config = RealtimeConfig {
            max_latency_ms: 1,
            stream_timeout: Duration::from_millis(1),
            audio_buffer_size: 1,
            max_concurrent_streams: 1,
            enable_metrics: true,
            enable_confidence_filtering: true,
            quality_threshold: 0.0,
            pronunciation_threshold: 1.0,
        };

        assert_eq!(config.max_latency_ms, 1);
        assert_eq!(config.quality_threshold, 0.0);
        assert_eq!(config.pronunciation_threshold, 1.0);
        assert_eq!(config.audio_buffer_size, 1);
        assert_eq!(config.max_concurrent_streams, 1);
    }

    #[test]
    fn test_stats_cloning() {
        let mut original_stats = RealtimeStats::default();
        original_stats.active_streams = 3;
        original_stats.average_latency_ms = 50.0;

        let cloned_stats = original_stats.clone();

        assert_eq!(original_stats.active_streams, cloned_stats.active_streams);
        assert_eq!(
            original_stats.average_latency_ms,
            cloned_stats.average_latency_ms
        );
    }

    #[test]
    fn test_config_debugging() {
        let config = RealtimeConfig::default();
        let debug_string = format!("{:?}", config);

        // Ensure debug formatting works
        assert!(debug_string.contains("RealtimeConfig"));
        assert!(debug_string.contains("max_latency_ms"));
        assert!(debug_string.contains("100"));
    }

    #[test]
    fn test_stats_debugging() {
        let stats = RealtimeStats::default();
        let debug_string = format!("{:?}", stats);

        // Ensure debug formatting works
        assert!(debug_string.contains("RealtimeStats"));
        assert!(debug_string.contains("active_streams"));
        assert!(debug_string.contains("0"));
    }

    #[test]
    fn test_serialization_compatibility() {
        // Test that our analysis types can be serialized/deserialized
        let analysis = PhonemeAnalysis {
            phonemes: vec![],
            difficulty_score: 0.5,
            problematic_phonemes: vec![],
            confidence: 0.8,
        };

        let serialized = serde_json::to_string(&analysis).expect("Failed to serialize");
        let deserialized: PhonemeAnalysis =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        assert_eq!(analysis.difficulty_score, deserialized.difficulty_score);
        assert_eq!(analysis.confidence, deserialized.confidence);
    }

    #[test]
    fn test_high_load_stats() {
        let mut stats = RealtimeStats::default();

        // Simulate high load conditions
        stats.active_streams = 1000;
        stats.total_chunks_processed = u64::MAX;
        stats.average_latency_ms = 999.9;
        stats.total_feedback_generated = u64::MAX;
        stats.average_confidence = 1.0;
        stats.peak_memory_usage = usize::MAX;
        stats.current_memory_usage = usize::MAX;

        assert_eq!(stats.active_streams, 1000);
        assert_eq!(stats.total_chunks_processed, u64::MAX);
        assert_eq!(stats.average_latency_ms, 999.9);
        assert_eq!(stats.total_feedback_generated, u64::MAX);
        assert_eq!(stats.average_confidence, 1.0);
        assert_eq!(stats.peak_memory_usage, usize::MAX);
        assert_eq!(stats.current_memory_usage, usize::MAX);
    }
}
