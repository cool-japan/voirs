//! Quality assessment for multi-modal processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-modal quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalQualityConfig {
    /// Quality metrics
    pub metrics: Vec<MultiModalQualityMetric>,
    /// Assessment methods
    pub assessment_methods: Vec<QualityAssessmentMethod>,
    /// Aggregation strategy
    pub aggregation: QualityAggregationStrategy,
    /// Reporting configuration
    pub reporting: QualityReportingConfig,
}

/// Multi-modal quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiModalQualityMetric {
    /// Audio-visual synchronization quality
    AudioVisualSyncQuality,
    /// Lip synchronization accuracy
    LipSyncAccuracy,
    /// Gesture-speech coherence
    GestureSpeechCoherence,
    /// Overall multi-modal quality
    OverallMultiModalQuality,
    /// Temporal consistency
    TemporalConsistency,
    /// Spatial consistency
    SpatialConsistency,
}

/// Quality assessment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    /// Objective metrics
    Objective,
    /// Subjective evaluation
    Subjective,
    /// Hybrid assessment
    Hybrid,
    /// Machine learning-based
    MachineLearningBased,
}

/// Quality aggregation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAggregationStrategy {
    /// Weighted average
    WeightedAverage,
    /// Minimum quality
    MinimumQuality,
    /// Geometric mean
    GeometricMean,
    /// Harmonic mean
    HarmonicMean,
    /// Custom aggregation
    Custom,
}

/// Quality reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReportingConfig {
    /// Report format
    pub format: ReportFormat,
    /// Include visualizations
    pub include_visualizations: bool,
    /// Detail level
    pub detail_level: DetailLevel,
    /// Export options
    pub export_options: Vec<ExportOption>,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// XML format
    XML,
    /// HTML format
    HTML,
    /// PDF format
    PDF,
    /// CSV format
    CSV,
}

/// Detail levels for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Summary only
    Summary,
    /// Detailed analysis
    Detailed,
    /// Full analysis
    Full,
}

/// Export options for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportOption {
    /// Export raw data
    RawData,
    /// Export processed data
    ProcessedData,
    /// Export visualizations
    Visualizations,
    /// Export statistics
    Statistics,
}

/// Multi-modal quality results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalQualityResults {
    /// Individual metric scores
    pub metric_scores: HashMap<String, f32>,
    /// Overall quality score
    pub overall_score: f32,
    /// Quality assessment details
    pub assessment_details: QualityAssessmentDetails,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Processing metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Quality assessment details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentDetails {
    /// Synchronization analysis
    pub sync_analysis: SynchronizationAnalysis,
    /// Alignment analysis
    pub alignment_analysis: AlignmentAnalysis,
    /// Coherence analysis
    pub coherence_analysis: CoherenceAnalysis,
    /// Consistency analysis
    pub consistency_analysis: ConsistencyAnalysis,
}

/// Synchronization analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationAnalysis {
    /// Audio-visual offset
    pub av_offset: f32,
    /// Offset variance
    pub offset_variance: f32,
    /// Synchronization confidence
    pub sync_confidence: f32,
    /// Drift detection
    pub drift_detected: bool,
    /// Drift rate
    pub drift_rate: Option<f32>,
}

/// Alignment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentAnalysis {
    /// Phoneme-viseme alignment accuracy
    pub phoneme_viseme_accuracy: f32,
    /// Temporal alignment quality
    pub temporal_alignment_quality: f32,
    /// Spatial alignment quality
    pub spatial_alignment_quality: f32,
    /// Alignment consistency
    pub alignment_consistency: f32,
}

/// Coherence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceAnalysis {
    /// Gesture-speech coherence
    pub gesture_speech_coherence: f32,
    /// Semantic coherence
    pub semantic_coherence: f32,
    /// Prosodic coherence
    pub prosodic_coherence: f32,
    /// Multimodal coherence
    pub multimodal_coherence: f32,
}

/// Consistency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyAnalysis {
    /// Temporal consistency
    pub temporal_consistency: f32,
    /// Spatial consistency
    pub spatial_consistency: f32,
    /// Cross-modal consistency
    pub cross_modal_consistency: f32,
    /// Long-term consistency
    pub long_term_consistency: f32,
}

/// Quality threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable overall quality
    pub min_overall_quality: f32,
    /// Minimum synchronization quality
    pub min_sync_quality: f32,
    /// Minimum alignment quality
    pub min_alignment_quality: f32,
    /// Minimum coherence quality
    pub min_coherence_quality: f32,
    /// Minimum consistency quality
    pub min_consistency_quality: f32,
}

impl MultiModalQualityResults {
    /// Create new quality results
    pub fn new() -> Self {
        Self {
            metric_scores: HashMap::new(),
            overall_score: 0.0,
            assessment_details: QualityAssessmentDetails::new(),
            recommendations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add metric score
    pub fn add_metric_score(&mut self, metric: String, score: f32) {
        self.metric_scores.insert(metric, score);
        self.update_overall_score();
    }

    /// Update overall score based on metric scores
    fn update_overall_score(&mut self) {
        if self.metric_scores.is_empty() {
            self.overall_score = 0.0;
            return;
        }

        let sum: f32 = self.metric_scores.values().sum();
        self.overall_score = sum / self.metric_scores.len() as f32;
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Check if quality meets thresholds
    pub fn meets_thresholds(&self, thresholds: &QualityThresholds) -> bool {
        self.overall_score >= thresholds.min_overall_quality
            && self.assessment_details.sync_analysis.sync_confidence >= thresholds.min_sync_quality
            && self.assessment_details.alignment_analysis.temporal_alignment_quality >= thresholds.min_alignment_quality
            && self.assessment_details.coherence_analysis.multimodal_coherence >= thresholds.min_coherence_quality
            && self.assessment_details.consistency_analysis.temporal_consistency >= thresholds.min_consistency_quality
    }

    /// Generate quality recommendations
    pub fn generate_recommendations(&mut self, thresholds: &QualityThresholds) {
        self.recommendations.clear();

        if self.overall_score < thresholds.min_overall_quality {
            self.recommendations.push("Overall quality is below threshold. Consider improving synchronization and alignment.".to_string());
        }

        if self.assessment_details.sync_analysis.sync_confidence < thresholds.min_sync_quality {
            self.recommendations.push("Audio-visual synchronization needs improvement.".to_string());
        }

        if self.assessment_details.alignment_analysis.temporal_alignment_quality < thresholds.min_alignment_quality {
            self.recommendations.push("Temporal alignment accuracy needs improvement.".to_string());
        }

        if self.assessment_details.coherence_analysis.multimodal_coherence < thresholds.min_coherence_quality {
            self.recommendations.push("Multimodal coherence needs improvement.".to_string());
        }

        if self.assessment_details.consistency_analysis.temporal_consistency < thresholds.min_consistency_quality {
            self.recommendations.push("Temporal consistency needs improvement.".to_string());
        }

        if self.assessment_details.sync_analysis.drift_detected {
            self.recommendations.push("Synchronization drift detected. Consider drift correction.".to_string());
        }
    }
}

impl QualityAssessmentDetails {
    /// Create new quality assessment details
    pub fn new() -> Self {
        Self {
            sync_analysis: SynchronizationAnalysis::new(),
            alignment_analysis: AlignmentAnalysis::new(),
            coherence_analysis: CoherenceAnalysis::new(),
            consistency_analysis: ConsistencyAnalysis::new(),
        }
    }
}

impl SynchronizationAnalysis {
    /// Create new synchronization analysis
    pub fn new() -> Self {
        Self {
            av_offset: 0.0,
            offset_variance: 0.0,
            sync_confidence: 0.0,
            drift_detected: false,
            drift_rate: None,
        }
    }
}

impl AlignmentAnalysis {
    /// Create new alignment analysis
    pub fn new() -> Self {
        Self {
            phoneme_viseme_accuracy: 0.0,
            temporal_alignment_quality: 0.0,
            spatial_alignment_quality: 0.0,
            alignment_consistency: 0.0,
        }
    }
}

impl CoherenceAnalysis {
    /// Create new coherence analysis
    pub fn new() -> Self {
        Self {
            gesture_speech_coherence: 0.0,
            semantic_coherence: 0.0,
            prosodic_coherence: 0.0,
            multimodal_coherence: 0.0,
        }
    }
}

impl ConsistencyAnalysis {
    /// Create new consistency analysis
    pub fn new() -> Self {
        Self {
            temporal_consistency: 0.0,
            spatial_consistency: 0.0,
            cross_modal_consistency: 0.0,
            long_term_consistency: 0.0,
        }
    }
}

impl Default for MultiModalQualityConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                MultiModalQualityMetric::AudioVisualSyncQuality,
                MultiModalQualityMetric::LipSyncAccuracy,
                MultiModalQualityMetric::GestureSpeechCoherence,
                MultiModalQualityMetric::OverallMultiModalQuality,
            ],
            assessment_methods: vec![
                QualityAssessmentMethod::Objective,
                QualityAssessmentMethod::MachineLearningBased,
            ],
            aggregation: QualityAggregationStrategy::WeightedAverage,
            reporting: QualityReportingConfig::default(),
        }
    }
}

impl Default for QualityReportingConfig {
    fn default() -> Self {
        Self {
            format: ReportFormat::JSON,
            include_visualizations: true,
            detail_level: DetailLevel::Detailed,
            export_options: vec![
                ExportOption::ProcessedData,
                ExportOption::Statistics,
            ],
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_overall_quality: 0.7,
            min_sync_quality: 0.8,
            min_alignment_quality: 0.7,
            min_coherence_quality: 0.6,
            min_consistency_quality: 0.7,
        }
    }
}

impl Default for MultiModalQualityResults {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for QualityAssessmentDetails {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SynchronizationAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AlignmentAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CoherenceAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConsistencyAnalysis {
    fn default() -> Self {
        Self::new()
    }
}