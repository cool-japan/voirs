//! Report structures and quality assessment results

use crate::types::VoiceType;
use serde::{Deserialize, Serialize};

/// Comprehensive quality assessment report
///
/// Aggregates all quality evaluation results including naturalness, expression,
/// voice quality, and performance assessments with an overall quality score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveQualityReport {
    /// Naturalness evaluation results
    pub naturalness: NaturalnessReport,
    /// Musical expression evaluation results
    pub expression: ExpressionReport,
    /// Voice quality assessment results
    pub voice_quality: VoiceQualityReport,
    /// Performance evaluation results
    pub performance: PerformanceReport,
    /// Overall quality score (0.0-1.0), weighted average of all assessments
    pub overall_score: f64,
    /// UTC timestamp when evaluation was performed
    pub evaluation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Naturalness evaluation report
///
/// Provides detailed naturalness assessment scores for various aspects of
/// human-like singing quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalnessReport {
    /// Overall naturalness score (0.0-1.0), weighted average of all metrics
    pub overall_score: f64,
    /// Breath pattern naturalness score (0.0-1.0)
    pub breath_naturalness: f64,
    /// Vibrato naturalness score (0.0-1.0), evaluates rate and depth
    pub vibrato_naturalness: f64,
    /// Formant frequency naturalness score (0.0-1.0), compared to reference
    pub formant_naturalness: f64,
    /// Note transition smoothness score (0.0-1.0)
    pub transition_naturalness: f64,
    /// Timbre consistency throughout the audio (0.0-1.0)
    pub timbre_consistency: f64,
    /// Reference voice type used for comparison
    pub reference_voice_type: VoiceType,
    /// Recommendations for improving naturalness
    pub recommendations: Vec<String>,
}

/// Musical expression evaluation report
///
/// Evaluates the quality and effectiveness of musical expression in singing
/// synthesis including dynamics, pitch variation, timing, and emotional content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionReport {
    /// Overall expression score (0.0-1.0), weighted average of all metrics
    pub overall_score: f64,
    /// Dynamic range quality score (0.0-1.0), evaluates volume variation
    pub dynamic_range: f64,
    /// Pitch expression quality score (0.0-1.0), evaluates pitch variation
    pub pitch_expression: f64,
    /// Timing expression quality score (0.0-1.0), evaluates rhythmic variation
    pub timing_expression: f64,
    /// Emotional recognition accuracy score (0.0-1.0)
    pub emotional_recognition: f64,
    /// Detected emotions with confidence scores, format: (emotion_name, confidence)
    pub recognized_emotions: Vec<(String, f64)>,
    /// Expression consistency throughout the performance (0.0-1.0)
    pub expression_consistency: f64,
}

impl ExpressionReport {
    /// Create a neutral expression report
    pub fn neutral() -> Self {
        Self {
            overall_score: 0.5,
            dynamic_range: 0.5,
            pitch_expression: 0.5,
            timing_expression: 0.5,
            emotional_recognition: 0.5,
            recognized_emotions: vec![("neutral".to_string(), 0.8)],
            expression_consistency: 0.5,
        }
    }
}

/// Voice quality assessment report
///
/// Evaluates the fundamental quality of the synthesized voice characteristics
/// including timbre, pitch stability, harmonic content, and clarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityReport {
    /// Overall voice quality score (0.0-1.0), weighted average of all metrics
    pub overall_score: f64,
    /// Timbre quality score (0.0-1.0), evaluates tone color quality
    pub timbre_quality: f64,
    /// Pitch stability score (0.0-1.0), evaluates pitch consistency
    pub pitch_stability: f64,
    /// Harmonic richness score (0.0-1.0), evaluates harmonic content quality
    pub harmonic_richness: f64,
    /// Vocal clarity score (0.0-1.0), evaluates articulation and definition
    pub vocal_clarity: f64,
    /// Resonance quality score (0.0-1.0), evaluates vocal resonance characteristics
    pub resonance_quality: f64,
    /// Voice type match score (0.0-1.0), evaluates match to target voice type
    pub voice_type_match: f64,
    /// Recommendations for improving voice quality
    pub quality_recommendations: Vec<String>,
}

/// Performance evaluation report
///
/// Evaluates system performance characteristics during synthesis including
/// latency, stability, consistency, and resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Overall performance score (0.0-1.0), weighted average of all metrics
    pub overall_score: f64,
    /// Latency analysis score (0.0-1.0), evaluates response time
    pub latency_analysis: f64,
    /// Stability assessment score (0.0-1.0), evaluates system stability
    pub stability_assessment: f64,
    /// Consistency evaluation score (0.0-1.0), evaluates output consistency
    pub consistency_evaluation: f64,
    /// Resource efficiency score (0.0-1.0), evaluates CPU/memory usage
    pub resource_efficiency: f64,
    /// Recommendations for improving performance
    pub performance_recommendations: Vec<String>,
}

/// Reference profile for naturalness evaluation
///
/// Defines expected formant frequency ranges for different voice types,
/// used as reference for naturalness assessment.
#[derive(Debug)]
pub struct NaturalnessProfile {
    /// First formant frequency range in Hz (F1), format: (min, max)
    pub f1_range: (f32, f32),
    /// Second formant frequency range in Hz (F2), format: (min, max)
    pub f2_range: (f32, f32),
}

impl NaturalnessProfile {
    /// Default profile for soprano voice
    pub fn soprano_default() -> Self {
        Self {
            f1_range: (350.0, 900.0),
            f2_range: (1200.0, 2800.0),
        }
    }

    /// Default profile for alto voice
    pub fn alto_default() -> Self {
        Self {
            f1_range: (400.0, 1000.0),
            f2_range: (1000.0, 2300.0),
        }
    }

    /// Default profile for tenor voice
    pub fn tenor_default() -> Self {
        Self {
            f1_range: (300.0, 800.0),
            f2_range: (800.0, 2000.0),
        }
    }

    /// Default profile for bass voice
    pub fn bass_default() -> Self {
        Self {
            f1_range: (250.0, 700.0),
            f2_range: (600.0, 1800.0),
        }
    }
}

/// Voice quality reference profile
///
/// Comprehensive reference profile defining expected acoustic characteristics
/// for different voice types, used for voice quality assessment.
#[derive(Debug)]
pub struct VoiceQualityProfile {
    /// Expected spectral centroid frequency in Hz
    pub expected_centroid: f32,
    /// Expected spectral rolloff frequency in Hz
    pub expected_rolloff: f32,
    /// Expected spectral bandwidth in Hz
    pub expected_bandwidth: f32,
    /// First formant frequency range in Hz (F1), format: (min, max)
    pub f1_range: (f32, f32),
    /// Second formant frequency range in Hz (F2), format: (min, max)
    pub f2_range: (f32, f32),
    /// Third formant frequency range in Hz (F3), format: (min, max)
    pub f3_range: (f32, f32),
    /// Expected pitch range in Hz, format: (min, max)
    pub pitch_range: (f32, f32),
}

impl VoiceQualityProfile {
    /// Default profile for soprano voice
    pub fn soprano_default() -> Self {
        Self {
            expected_centroid: 1500.0,
            expected_rolloff: 6000.0,
            expected_bandwidth: 800.0,
            f1_range: (350.0, 900.0),
            f2_range: (1200.0, 2800.0),
            f3_range: (2500.0, 4000.0),
            pitch_range: (261.63, 1046.50), // C4 to C6
        }
    }

    /// Default profile for alto voice
    pub fn alto_default() -> Self {
        Self {
            expected_centroid: 1200.0,
            expected_rolloff: 5000.0,
            expected_bandwidth: 700.0,
            f1_range: (400.0, 1000.0),
            f2_range: (1000.0, 2300.0),
            f3_range: (2200.0, 3500.0),
            pitch_range: (196.00, 783.99), // G3 to G5
        }
    }

    /// Default profile for tenor voice
    pub fn tenor_default() -> Self {
        Self {
            expected_centroid: 1000.0,
            expected_rolloff: 4000.0,
            expected_bandwidth: 600.0,
            f1_range: (300.0, 800.0),
            f2_range: (800.0, 2000.0),
            f3_range: (1800.0, 3000.0),
            pitch_range: (130.81, 523.25), // C3 to C5
        }
    }

    /// Default profile for bass voice
    pub fn bass_default() -> Self {
        Self {
            expected_centroid: 800.0,
            expected_rolloff: 3000.0,
            expected_bandwidth: 500.0,
            f1_range: (250.0, 700.0),
            f2_range: (600.0, 1800.0),
            f3_range: (1500.0, 2500.0),
            pitch_range: (87.31, 349.23), // F2 to F4
        }
    }

    /// Get profile for specific voice type
    ///
    /// # Arguments
    ///
    /// * `voice_type` - Voice type to get profile for
    ///
    /// # Returns
    ///
    /// * Voice quality profile with appropriate characteristics for the voice type
    pub fn for_voice_type(voice_type: VoiceType) -> Self {
        match voice_type {
            VoiceType::Soprano => Self::soprano_default(),
            VoiceType::MezzoSoprano => Self::soprano_default(), // Use soprano as fallback
            VoiceType::Alto => Self::alto_default(),
            VoiceType::Tenor => Self::tenor_default(),
            VoiceType::Baritone => Self::bass_default(), // Use bass as fallback
            VoiceType::Bass => Self::bass_default(),
        }
    }
}

/// Quality assessment metrics
///
/// Flexible container for quality metrics with support for named metrics,
/// overall quality scores, and detailed analysis results in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0-1.0), computed as average of all metrics
    pub overall_quality: f64,
    /// Individual metric scores by name, each in range (0.0-1.0)
    pub metrics: std::collections::HashMap<String, f64>,
    /// Detailed analysis results as JSON values for additional context
    pub analysis_details: std::collections::HashMap<String, serde_json::Value>,
}

impl QualityMetrics {
    /// Create new quality metrics
    pub fn new() -> Self {
        Self {
            overall_quality: 0.0,
            metrics: std::collections::HashMap::new(),
            analysis_details: std::collections::HashMap::new(),
        }
    }

    /// Add a metric score
    pub fn add_metric(&mut self, name: String, score: f64) {
        self.metrics.insert(name, score.clamp(0.0, 1.0));
        self.update_overall_quality();
    }

    /// Update overall quality score based on individual metrics
    fn update_overall_quality(&mut self) {
        if self.metrics.is_empty() {
            self.overall_quality = 0.0;
        } else {
            self.overall_quality = self.metrics.values().sum::<f64>() / self.metrics.len() as f64;
        }
    }

    /// Get metric score by name
    ///
    /// # Arguments
    ///
    /// * `name` - Metric name to retrieve
    ///
    /// # Returns
    ///
    /// * `Some(score)` - Metric score (0.0-1.0) if metric exists
    /// * `None` - If metric name not found
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Check if quality meets threshold
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum acceptable quality score (0.0-1.0)
    ///
    /// # Returns
    ///
    /// * `true` - If overall quality score meets or exceeds threshold
    /// * `false` - If overall quality score is below threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_quality >= threshold
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}
