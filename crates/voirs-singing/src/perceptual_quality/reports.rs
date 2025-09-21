//! Report structures and quality assessment results

use crate::types::VoiceType;
use serde::{Deserialize, Serialize};

/// Comprehensive quality assessment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveQualityReport {
    pub naturalness: NaturalnessReport,
    pub expression: ExpressionReport,
    pub voice_quality: VoiceQualityReport,
    pub performance: PerformanceReport,
    pub overall_score: f64,
    pub evaluation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Naturalness evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalnessReport {
    pub overall_score: f64,
    pub breath_naturalness: f64,
    pub vibrato_naturalness: f64,
    pub formant_naturalness: f64,
    pub transition_naturalness: f64,
    pub timbre_consistency: f64,
    pub reference_voice_type: VoiceType,
    pub recommendations: Vec<String>,
}

/// Musical expression evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionReport {
    pub overall_score: f64,
    pub dynamic_range: f64,
    pub pitch_expression: f64,
    pub timing_expression: f64,
    pub emotional_recognition: f64,
    pub recognized_emotions: Vec<(String, f64)>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityReport {
    pub overall_score: f64,
    pub timbre_quality: f64,
    pub pitch_stability: f64,
    pub harmonic_richness: f64,
    pub vocal_clarity: f64,
    pub resonance_quality: f64,
    pub voice_type_match: f64,
    pub quality_recommendations: Vec<String>,
}

/// Performance evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub overall_score: f64,
    pub latency_analysis: f64,
    pub stability_assessment: f64,
    pub consistency_evaluation: f64,
    pub resource_efficiency: f64,
    pub performance_recommendations: Vec<String>,
}

/// Reference profile for naturalness evaluation
#[derive(Debug)]
pub struct NaturalnessProfile {
    pub f1_range: (f32, f32), // First formant range
    pub f2_range: (f32, f32), // Second formant range
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
#[derive(Debug)]
pub struct VoiceQualityProfile {
    pub expected_centroid: f32,
    pub expected_rolloff: f32,
    pub expected_bandwidth: f32,
    pub f1_range: (f32, f32),
    pub f2_range: (f32, f32),
    pub f3_range: (f32, f32),
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f64,
    /// Individual metric scores
    pub metrics: std::collections::HashMap<String, f64>,
    /// Detailed analysis results
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
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Check if quality meets threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_quality >= threshold
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}
