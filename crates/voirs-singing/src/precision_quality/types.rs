//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::types::Expression;

use super::types_4::ExpressionFeatures;

/// Pitch accuracy analysis report
#[derive(Debug, Clone)]
pub struct PitchAccuracyReport {
    /// Percentage of notes within 5 cents of target pitch
    pub accuracy_percentage: f32,
    /// Number of notes within 5 cents of target
    pub notes_within_5_cents: usize,
    /// Total number of notes analyzed
    pub total_notes: usize,
    /// Mean deviation in cents from target pitch
    pub mean_cent_deviation: f32,
    /// Maximum deviation in cents observed
    pub max_cent_deviation: f32,
    /// Pitch stability score (0.0-1.0)
    pub pitch_stability: f32,
    /// Individual cent deviations for each note
    pub cent_deviations: Vec<f32>,
}
impl Default for PitchAccuracyReport {
    fn default() -> Self {
        Self {
            accuracy_percentage: 85.0,
            notes_within_5_cents: 85,
            total_notes: 100,
            mean_cent_deviation: 0.0,
            max_cent_deviation: 0.0,
            pitch_stability: 1.0,
            cent_deviations: Vec::new(),
        }
    }
}
/// Detected musical expression with confidence
#[derive(Debug, Clone)]
pub struct DetectedExpression {
    /// Type of expression detected
    pub expression_type: Expression,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Extracted expression features
    pub features: ExpressionFeatures,
}
/// Expression recognition analysis report
#[derive(Debug, Clone)]
pub struct ExpressionRecognitionReport {
    /// Percentage of correctly recognized expressions
    pub recognition_rate_percentage: f32,
    /// Number of expressions correctly recognized
    pub expressions_correctly_recognized: usize,
    /// Total number of expressions analyzed
    pub total_expressions: usize,
    /// Mean recognition accuracy across all expressions
    pub mean_recognition_accuracy: f32,
    /// Individual recognition accuracies for each expression
    pub individual_accuracies: Vec<f32>,
    /// Detected expressions with confidence scores
    pub detected_expressions: Vec<DetectedExpression>,
}
impl Default for ExpressionRecognitionReport {
    fn default() -> Self {
        Self {
            recognition_rate_percentage: 65.0,
            expressions_correctly_recognized: 2,
            total_expressions: 3,
            mean_recognition_accuracy: 0.0,
            individual_accuracies: Vec::new(),
            detected_expressions: Vec::new(),
        }
    }
}
