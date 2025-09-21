//! Visual speech alignment for multi-modal processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Visual speech alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Alignment method
    pub method: AlignmentMethod,
    /// Phoneme-viseme mapping
    pub phoneme_viseme_mapping: PhonemeVisemeMapping,
    /// Temporal alignment parameters
    pub temporal_alignment: TemporalAlignmentConfig,
    /// Quality metrics for alignment
    pub quality_metrics: AlignmentQualityMetrics,
}

/// Visual speech alignment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentMethod {
    /// Dynamic time warping
    DynamicTimeWarping,
    /// Hidden Markov Model alignment
    HiddenMarkovModel,
    /// Deep learning alignment
    DeepLearning,
    /// Forced alignment
    ForcedAlignment,
    /// Mutual information maximization
    MutualInformation,
}

/// Phoneme-viseme mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhonemeVisemeMapping {
    /// Mapping table
    pub mapping: HashMap<String, String>,
    /// Language-specific mappings
    pub language_mappings: HashMap<String, HashMap<String, String>>,
    /// Confidence scores for mappings
    pub confidence_scores: HashMap<String, f32>,
    /// Co-articulation modeling
    pub coarticulation: CoarticulationConfig,
}

/// Co-articulation modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarticulationConfig {
    /// Enable co-articulation modeling
    pub enabled: bool,
    /// Context window size
    pub context_window: usize,
    /// Influence weights
    pub influence_weights: Vec<f32>,
    /// Blending method
    pub blending_method: BlendingMethod,
}

/// Blending methods for co-articulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlendingMethod {
    /// Linear blending
    Linear,
    /// Weighted blending
    Weighted,
    /// Gaussian blending
    Gaussian,
    /// Exponential blending
    Exponential,
}

/// Temporal alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAlignmentConfig {
    /// Time resolution (seconds)
    pub time_resolution: f32,
    /// Alignment tolerance
    pub tolerance: f32,
    /// Smoothing parameters
    pub smoothing: AlignmentSmoothingConfig,
    /// Boundary constraints
    pub boundary_constraints: BoundaryConstraints,
}

/// Alignment smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentSmoothingConfig {
    /// Enable temporal smoothing
    pub enabled: bool,
    /// Smoothing window size
    pub window_size: usize,
    /// Smoothing strength
    pub strength: f32,
    /// Preserve boundaries
    pub preserve_boundaries: bool,
}

/// Boundary constraints for alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConstraints {
    /// Enforce word boundaries
    pub word_boundaries: bool,
    /// Enforce sentence boundaries
    pub sentence_boundaries: bool,
    /// Enforce silence boundaries
    pub silence_boundaries: bool,
    /// Minimum segment duration
    pub min_segment_duration: f32,
}

/// Alignment quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentQualityMetrics {
    /// Enabled metrics
    pub enabled_metrics: Vec<AlignmentMetric>,
    /// Confidence computation
    pub confidence_computation: ConfidenceComputation,
    /// Error detection
    pub error_detection: ErrorDetectionConfig,
}

/// Alignment quality metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentMetric {
    /// Temporal consistency
    TemporalConsistency,
    /// Spatial consistency
    SpatialConsistency,
    /// Lip synchronization accuracy
    LipSyncAccuracy,
    /// Phoneme-viseme correspondence
    PhonemeVisemeCorrespondence,
    /// Overall alignment quality
    OverallQuality,
}

/// Confidence computation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceComputation {
    /// Statistical confidence
    Statistical,
    /// Model-based confidence
    ModelBased,
    /// Ensemble confidence
    Ensemble,
    /// Bayesian confidence
    Bayesian,
}

/// Error detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetectionConfig {
    /// Detection threshold
    pub threshold: f32,
    /// Error types to detect
    pub error_types: Vec<AlignmentErrorType>,
    /// Correction strategies
    pub correction_strategies: Vec<CorrectionStrategy>,
}

/// Types of alignment errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentErrorType {
    /// Temporal misalignment
    TemporalMisalignment,
    /// Missing alignment
    MissingAlignment,
    /// Spurious alignment
    SpuriousAlignment,
    /// Inconsistent alignment
    InconsistentAlignment,
}

/// Correction strategies for alignment errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionStrategy {
    /// Automatic correction
    Automatic,
    /// Manual correction
    Manual,
    /// Hybrid correction
    Hybrid,
    /// No correction
    None,
}

/// Alignment result for a single segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentSegment {
    /// Start time (seconds)
    pub start_time: f32,
    /// End time (seconds)
    pub end_time: f32,
    /// Phoneme or word aligned
    pub text: String,
    /// Corresponding viseme
    pub viseme: Option<String>,
    /// Alignment confidence
    pub confidence: f32,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f32>,
}

/// Visual speech alignment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    /// Phoneme-viseme alignment
    pub phoneme_viseme_alignment: Vec<PhonemeVisemeAlignment>,
    /// Alignment confidence
    pub confidence: f32,
    /// Alignment quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Temporal boundaries
    pub temporal_boundaries: Vec<TemporalBoundary>,
}

/// Phoneme-viseme alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeVisemeAlignment {
    /// Phoneme
    pub phoneme: String,
    /// Corresponding viseme
    pub viseme: String,
    /// Alignment confidence
    pub confidence: f32,
    /// Temporal alignment
    pub temporal_alignment: TemporalAlignment,
}

/// Temporal alignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAlignment {
    /// Start time (seconds)
    pub start_time: f32,
    /// End time (seconds)
    pub end_time: f32,
    /// Duration (seconds)
    pub duration: f32,
    /// Alignment offset
    pub offset: f32,
}

/// Temporal boundary
pub type TemporalBoundary = (f32, f32, String); // (start, end, label)

/// Complete alignment for an utterance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceAlignment {
    /// Individual segment alignments
    pub segments: Vec<AlignmentSegment>,
    /// Overall alignment quality
    pub overall_quality: f32,
    /// Processing metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PhonemeVisemeMapping {
    /// Create new phoneme-viseme mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Add phoneme-viseme mapping
    pub fn add_mapping(&mut self, phoneme: String, viseme: String, confidence: f32) {
        self.mapping.insert(phoneme.clone(), viseme);
        self.confidence_scores.insert(phoneme, confidence);
    }

    /// Add language-specific mapping
    pub fn add_language_mapping(&mut self, language: String, phoneme: String, viseme: String) {
        self.language_mappings
            .entry(language)
            .or_default()
            .insert(phoneme, viseme);
    }

    /// Get viseme for phoneme
    pub fn get_viseme(&self, phoneme: &str, language: Option<&str>) -> Option<&String> {
        if let Some(lang) = language {
            if let Some(lang_mapping) = self.language_mappings.get(lang) {
                if let Some(viseme) = lang_mapping.get(phoneme) {
                    return Some(viseme);
                }
            }
        }
        self.mapping.get(phoneme)
    }

    /// Get confidence for phoneme mapping
    pub fn get_confidence(&self, phoneme: &str) -> f32 {
        self.confidence_scores.get(phoneme).copied().unwrap_or(0.0)
    }
}

impl AlignmentSegment {
    /// Create new alignment result
    pub fn new(start_time: f32, end_time: f32, text: String, confidence: f32) -> Self {
        Self {
            start_time,
            end_time,
            text,
            viseme: None,
            confidence,
            quality_metrics: HashMap::new(),
        }
    }

    /// Set viseme
    pub fn with_viseme(mut self, viseme: String) -> Self {
        self.viseme = Some(viseme);
        self
    }

    /// Add quality metric
    pub fn add_quality_metric(&mut self, name: String, value: f32) {
        self.quality_metrics.insert(name, value);
    }

    /// Get duration
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Check if alignment overlaps with another
    pub fn overlaps_with(&self, other: &AlignmentSegment) -> bool {
        self.start_time < other.end_time && other.start_time < self.end_time
    }
}

impl UtteranceAlignment {
    /// Create new utterance alignment
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            overall_quality: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Add segment alignment
    pub fn add_segment(&mut self, segment: AlignmentSegment) {
        self.segments.push(segment);
        self.update_overall_quality();
    }

    /// Update overall quality based on segment qualities
    fn update_overall_quality(&mut self) {
        if self.segments.is_empty() {
            self.overall_quality = 0.0;
            return;
        }

        let sum: f32 = self.segments.iter().map(|s| s.confidence).sum();
        self.overall_quality = sum / self.segments.len() as f32;
    }

    /// Get total duration
    pub fn total_duration(&self) -> f32 {
        if self.segments.is_empty() {
            return 0.0;
        }

        let start = self
            .segments
            .iter()
            .map(|s| s.start_time)
            .fold(f32::INFINITY, f32::min);
        let end = self
            .segments
            .iter()
            .map(|s| s.end_time)
            .fold(f32::NEG_INFINITY, f32::max);
        end - start
    }

    /// Find segments at specific time
    pub fn segments_at_time(&self, time: f32) -> Vec<&AlignmentSegment> {
        self.segments
            .iter()
            .filter(|s| s.start_time <= time && time <= s.end_time)
            .collect()
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            method: AlignmentMethod::DynamicTimeWarping,
            phoneme_viseme_mapping: PhonemeVisemeMapping::default(),
            temporal_alignment: TemporalAlignmentConfig::default(),
            quality_metrics: AlignmentQualityMetrics::default(),
        }
    }
}

impl Default for CoarticulationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            context_window: 3,
            influence_weights: vec![0.2, 0.6, 0.2],
            blending_method: BlendingMethod::Weighted,
        }
    }
}

impl Default for TemporalAlignmentConfig {
    fn default() -> Self {
        Self {
            time_resolution: 0.01,
            tolerance: 0.05,
            smoothing: AlignmentSmoothingConfig::default(),
            boundary_constraints: BoundaryConstraints::default(),
        }
    }
}

impl Default for AlignmentSmoothingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 5,
            strength: 0.3,
            preserve_boundaries: true,
        }
    }
}

impl Default for BoundaryConstraints {
    fn default() -> Self {
        Self {
            word_boundaries: true,
            sentence_boundaries: true,
            silence_boundaries: true,
            min_segment_duration: 0.02,
        }
    }
}

impl Default for AlignmentQualityMetrics {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                AlignmentMetric::TemporalConsistency,
                AlignmentMetric::LipSyncAccuracy,
                AlignmentMetric::OverallQuality,
            ],
            confidence_computation: ConfidenceComputation::Statistical,
            error_detection: ErrorDetectionConfig::default(),
        }
    }
}

impl Default for ErrorDetectionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            error_types: vec![
                AlignmentErrorType::TemporalMisalignment,
                AlignmentErrorType::MissingAlignment,
            ],
            correction_strategies: vec![CorrectionStrategy::Automatic],
        }
    }
}

impl Default for UtteranceAlignment {
    fn default() -> Self {
        Self::new()
    }
}
