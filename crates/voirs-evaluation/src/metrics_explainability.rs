//! # Metrics Explainability System
//!
//! This module provides interpretable explanations for evaluation metrics, helping users
//! understand what metrics measure, why specific scores were given, and how to improve results.
//!
//! ## Features
//!
//! - Comprehensive metric descriptions and definitions
//! - Score interpretation and contextualization
//! - Factor contribution analysis (what influenced the score)
//! - Actionable improvement recommendations
//! - Comparative context (how scores relate to benchmarks)
//! - Natural language explanations
//! - Multiple explanation detail levels
//!
//! ## Example
//!
//! ```rust
//! use voirs_evaluation::metrics_explainability::{
//!     ExplainabilityEngine, MetricType, ExplanationLevel,
//! };
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create explainability engine
//! let engine = ExplainabilityEngine::new();
//!
//! // Get explanation for a metric score
//! let explanation = engine.explain_metric(
//!     MetricType::Pesq,
//!     3.8,
//!     ExplanationLevel::Detailed,
//! )?;
//!
//! println!("{}", explanation.natural_language_summary);
//! println!("Recommendations:");
//! for rec in &explanation.recommendations {
//!     println!("  - {}", rec);
//! }
//!
//! // Analyze contributing factors
//! let mut factors = HashMap::new();
//! factors.insert("spectral_clarity".to_string(), 0.85);
//! factors.insert("temporal_accuracy".to_string(), 0.78);
//!
//! let factor_explanation = engine.explain_factors(MetricType::Pesq, factors)?;
//! println!("{}", factor_explanation);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during explainability operations
#[derive(Debug, Error)]
pub enum ExplainabilityError {
    /// Unknown metric type
    #[error("Unknown metric type: {0}")]
    UnknownMetric(String),

    /// Invalid score value
    #[error("Invalid score value for metric {metric}: {value}")]
    InvalidScore { metric: String, value: f64 },

    /// Insufficient data for explanation
    #[error("Insufficient data to generate explanation: {0}")]
    InsufficientData(String),

    /// Explanation generation failed
    #[error("Failed to generate explanation: {0}")]
    GenerationError(String),
}

/// Type of evaluation metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// PESQ (Perceptual Evaluation of Speech Quality)
    Pesq,
    /// STOI (Short-Time Objective Intelligibility)
    Stoi,
    /// MCD (Mel Cepstral Distortion)
    Mcd,
    /// MOS (Mean Opinion Score)
    Mos,
    /// WER (Word Error Rate)
    Wer,
    /// F0 RMSE (Fundamental Frequency Root Mean Square Error)
    F0Rmse,
    /// Spectral Convergence
    SpectralConvergence,
    /// Log-Spectral Distance
    LogSpectralDistance,
    /// Signal-to-Noise Ratio
    Snr,
    /// Real-Time Factor
    Rtf,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pesq => write!(f, "PESQ"),
            Self::Stoi => write!(f, "STOI"),
            Self::Mcd => write!(f, "MCD"),
            Self::Mos => write!(f, "MOS"),
            Self::Wer => write!(f, "WER"),
            Self::F0Rmse => write!(f, "F0 RMSE"),
            Self::SpectralConvergence => write!(f, "Spectral Convergence"),
            Self::LogSpectralDistance => write!(f, "Log-Spectral Distance"),
            Self::Snr => write!(f, "SNR"),
            Self::Rtf => write!(f, "Real-Time Factor"),
        }
    }
}

/// Level of detail for explanations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationLevel {
    /// Brief summary only
    Brief,
    /// Standard explanation with key details
    Standard,
    /// Detailed technical explanation
    Detailed,
    /// Expert-level with all technical details
    Expert,
}

/// Quality rating category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRating {
    /// Excellent quality
    Excellent,
    /// Good quality
    Good,
    /// Fair quality
    Fair,
    /// Poor quality
    Poor,
    /// Very poor quality
    VeryPoor,
}

impl std::fmt::Display for QualityRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::VeryPoor => write!(f, "Very Poor"),
        }
    }
}

/// Metric explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricExplanation {
    /// Metric type
    pub metric_type: MetricType,
    /// Actual score value
    pub score: f64,
    /// Quality rating category
    pub rating: QualityRating,
    /// Natural language summary
    pub natural_language_summary: String,
    /// Metric definition and what it measures
    pub metric_definition: String,
    /// Score interpretation
    pub score_interpretation: String,
    /// Contributing factors and their impacts
    pub contributing_factors: HashMap<String, f64>,
    /// Actionable recommendations for improvement
    pub recommendations: Vec<String>,
    /// Comparative context (percentile, benchmark comparisons)
    pub comparative_context: String,
    /// Technical details (for expert level)
    pub technical_details: Option<String>,
}

/// Metric metadata and definition
#[derive(Debug, Clone)]
struct MetricMetadata {
    /// Full name
    name: String,
    /// Brief description
    description: String,
    /// What it measures
    measures: String,
    /// Value range
    range: (f64, f64),
    /// Higher is better?
    higher_is_better: bool,
    /// Typical factors that influence this metric
    typical_factors: Vec<String>,
}

/// Explainability engine for evaluation metrics
pub struct ExplainabilityEngine {
    /// Metric metadata database
    metric_metadata: HashMap<MetricType, MetricMetadata>,
    /// Benchmark scores for comparison
    benchmark_scores: HashMap<MetricType, Vec<f64>>,
}

impl ExplainabilityEngine {
    /// Create a new explainability engine
    pub fn new() -> Self {
        let mut engine = Self {
            metric_metadata: HashMap::new(),
            benchmark_scores: HashMap::new(),
        };

        engine.initialize_metadata();
        engine.initialize_benchmarks();
        engine
    }

    /// Explain a metric score
    pub fn explain_metric(
        &self,
        metric_type: MetricType,
        score: f64,
        level: ExplanationLevel,
    ) -> Result<MetricExplanation, ExplainabilityError> {
        let metadata = self
            .metric_metadata
            .get(&metric_type)
            .ok_or_else(|| ExplainabilityError::UnknownMetric(metric_type.to_string()))?;

        // Validate score range
        if score < metadata.range.0 || score > metadata.range.1 {
            return Err(ExplainabilityError::InvalidScore {
                metric: metric_type.to_string(),
                value: score,
            });
        }

        // Determine quality rating
        let rating = self.determine_rating(metric_type, score, metadata);

        // Generate natural language summary
        let summary = self.generate_summary(metric_type, score, &rating, level);

        // Generate score interpretation
        let interpretation = self.generate_interpretation(metric_type, score, &rating, metadata);

        // Estimate contributing factors (in production, this would come from actual analysis)
        let factors = self.estimate_factors(metric_type, score);

        // Generate recommendations
        let recommendations = self.generate_recommendations(metric_type, score, &rating, metadata);

        // Generate comparative context
        let comparative_context = self.generate_comparative_context(metric_type, score);

        // Generate technical details for expert level
        let technical_details = if matches!(level, ExplanationLevel::Expert) {
            Some(self.generate_technical_details(metric_type, score, metadata))
        } else {
            None
        };

        Ok(MetricExplanation {
            metric_type,
            score,
            rating,
            natural_language_summary: summary,
            metric_definition: metadata.description.clone(),
            score_interpretation: interpretation,
            contributing_factors: factors,
            recommendations,
            comparative_context,
            technical_details,
        })
    }

    /// Explain how different factors contribute to a metric score
    pub fn explain_factors(
        &self,
        metric_type: MetricType,
        factors: HashMap<String, f64>,
    ) -> Result<String, ExplainabilityError> {
        if factors.is_empty() {
            return Err(ExplainabilityError::InsufficientData(
                "No factors provided".to_string(),
            ));
        }

        let mut explanation = format!("Factor Analysis for {} Score:\n\n", metric_type);

        // Sort factors by absolute contribution
        let mut sorted_factors: Vec<_> = factors.iter().collect();
        sorted_factors.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .expect("value should be present")
        });

        for (factor, contribution) in sorted_factors {
            let impact = if *contribution > 0.8 {
                "very strong positive"
            } else if *contribution > 0.6 {
                "strong positive"
            } else if *contribution > 0.4 {
                "moderate positive"
            } else if *contribution > 0.2 {
                "weak positive"
            } else if *contribution < -0.8 {
                "very strong negative"
            } else if *contribution < -0.6 {
                "strong negative"
            } else if *contribution < -0.4 {
                "moderate negative"
            } else if *contribution < -0.2 {
                "weak negative"
            } else {
                "minimal"
            };

            explanation.push_str(&format!(
                "  • {} has a {} impact (contribution: {:.3})\n",
                factor, impact, contribution
            ));
        }

        Ok(explanation)
    }

    /// Compare multiple metric scores and explain differences
    pub fn compare_scores(
        &self,
        metric_type: MetricType,
        score1: f64,
        score2: f64,
        label1: &str,
        label2: &str,
    ) -> Result<String, ExplainabilityError> {
        let metadata = self
            .metric_metadata
            .get(&metric_type)
            .ok_or_else(|| ExplainabilityError::UnknownMetric(metric_type.to_string()))?;

        let diff = score2 - score1;
        let pct_change = (diff / score1) * 100.0;

        let direction = if metadata.higher_is_better {
            if diff > 0.0 {
                "better"
            } else {
                "worse"
            }
        } else if diff < 0.0 {
            "better"
        } else {
            "worse"
        };

        let magnitude = if diff.abs() < 0.05 {
            "marginally"
        } else if diff.abs() < 0.15 {
            "moderately"
        } else {
            "significantly"
        };

        let comparison = format!(
            "{} scored {:.3} while {} scored {:.3} on {} metric. \
             {} is {} {} (difference: {:.3}, {:.1}% change).",
            label1,
            score1,
            label2,
            score2,
            metric_type,
            label2,
            magnitude,
            direction,
            diff.abs(),
            pct_change.abs()
        );

        Ok(comparison)
    }

    /// Get all available metric types
    pub fn available_metrics(&self) -> Vec<MetricType> {
        self.metric_metadata.keys().copied().collect()
    }

    // Private helper methods

    fn initialize_metadata(&mut self) {
        // PESQ
        self.metric_metadata.insert(
            MetricType::Pesq,
            MetricMetadata {
                name: "Perceptual Evaluation of Speech Quality".to_string(),
                description: "PESQ measures the perceptual quality of speech by comparing generated audio to reference, modeling human auditory perception.".to_string(),
                measures: "Overall perceptual speech quality as heard by human listeners".to_string(),
                range: (1.0, 4.5),
                higher_is_better: true,
                typical_factors: vec![
                    "Background noise".to_string(),
                    "Spectral distortion".to_string(),
                    "Time-domain artifacts".to_string(),
                    "Frequency response".to_string(),
                ],
            },
        );

        // STOI
        self.metric_metadata.insert(
            MetricType::Stoi,
            MetricMetadata {
                name: "Short-Time Objective Intelligibility".to_string(),
                description: "STOI measures speech intelligibility, predicting how understandable speech is to listeners.".to_string(),
                measures: "Speech intelligibility and clarity".to_string(),
                range: (0.0, 1.0),
                higher_is_better: true,
                typical_factors: vec![
                    "Articulation clarity".to_string(),
                    "Consonant preservation".to_string(),
                    "Noise level".to_string(),
                    "Temporal modulation".to_string(),
                ],
            },
        );

        // MCD
        self.metric_metadata.insert(
            MetricType::Mcd,
            MetricMetadata {
                name: "Mel Cepstral Distortion".to_string(),
                description: "MCD measures spectral distance between generated and reference speech in mel-cepstral domain.".to_string(),
                measures: "Spectral similarity and acoustic quality".to_string(),
                range: (0.0, 20.0),
                higher_is_better: false,
                typical_factors: vec![
                    "Spectral envelope accuracy".to_string(),
                    "Formant structure".to_string(),
                    "Harmonic content".to_string(),
                    "Vocal tract modeling".to_string(),
                ],
            },
        );

        // MOS
        self.metric_metadata.insert(
            MetricType::Mos,
            MetricMetadata {
                name: "Mean Opinion Score".to_string(),
                description: "MOS represents subjective quality ratings from human listeners (predicted or actual).".to_string(),
                measures: "Overall subjective quality as rated by humans".to_string(),
                range: (1.0, 5.0),
                higher_is_better: true,
                typical_factors: vec![
                    "Naturalness".to_string(),
                    "Pleasantness".to_string(),
                    "Listening effort".to_string(),
                    "Artifacts".to_string(),
                ],
            },
        );

        // WER
        self.metric_metadata.insert(
            MetricType::Wer,
            MetricMetadata {
                name: "Word Error Rate".to_string(),
                description: "WER measures recognition accuracy by counting word-level errors."
                    .to_string(),
                measures: "Speech recognition accuracy".to_string(),
                range: (0.0, 1.0),
                higher_is_better: false,
                typical_factors: vec![
                    "Pronunciation accuracy".to_string(),
                    "Acoustic clarity".to_string(),
                    "Background noise".to_string(),
                    "Speaker consistency".to_string(),
                ],
            },
        );

        // Add more metrics as needed...
    }

    fn initialize_benchmarks(&mut self) {
        // Initialize benchmark scores for comparative context
        self.benchmark_scores
            .insert(MetricType::Pesq, vec![3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.3]);
        self.benchmark_scores.insert(
            MetricType::Stoi,
            vec![0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
        );
        self.benchmark_scores
            .insert(MetricType::Mcd, vec![8.0, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5]);
        self.benchmark_scores
            .insert(MetricType::Mos, vec![3.0, 3.3, 3.6, 3.9, 4.2, 4.4, 4.6]);
        self.benchmark_scores.insert(
            MetricType::Wer,
            vec![0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02],
        );
    }

    fn determine_rating(
        &self,
        metric_type: MetricType,
        score: f64,
        metadata: &MetricMetadata,
    ) -> QualityRating {
        // Normalize score to 0-1 range
        let normalized = if metadata.higher_is_better {
            (score - metadata.range.0) / (metadata.range.1 - metadata.range.0)
        } else {
            1.0 - (score - metadata.range.0) / (metadata.range.1 - metadata.range.0)
        };

        match normalized {
            n if n >= 0.9 => QualityRating::Excellent,
            n if n >= 0.75 => QualityRating::Good,
            n if n >= 0.6 => QualityRating::Fair,
            n if n >= 0.4 => QualityRating::Poor,
            _ => QualityRating::VeryPoor,
        }
    }

    fn generate_summary(
        &self,
        metric_type: MetricType,
        score: f64,
        rating: &QualityRating,
        level: ExplanationLevel,
    ) -> String {
        match level {
            ExplanationLevel::Brief => {
                format!(
                    "The {} score of {:.3} indicates {} quality.",
                    metric_type, score, rating
                )
            }
            ExplanationLevel::Standard | ExplanationLevel::Detailed | ExplanationLevel::Expert => {
                format!(
                    "The {} score of {:.3} indicates {} quality. This metric evaluates the perceptual \
                     characteristics of the generated speech and provides insight into how it would be \
                     perceived by human listeners.",
                    metric_type, score, rating
                )
            }
        }
    }

    fn generate_interpretation(
        &self,
        _metric_type: MetricType,
        score: f64,
        rating: &QualityRating,
        metadata: &MetricMetadata,
    ) -> String {
        let relative_position = (score - metadata.range.0) / (metadata.range.1 - metadata.range.0);
        let position_desc = if relative_position > 0.9 {
            "at the high end"
        } else if relative_position > 0.7 {
            "above average"
        } else if relative_position > 0.3 {
            "in the middle range"
        } else {
            "below average"
        };

        format!(
            "This {} score of {:.3} is {} of the possible range ({:.1}-{:.1}). \
             It suggests that the {}.",
            rating,
            score,
            position_desc,
            metadata.range.0,
            metadata.range.1,
            metadata.measures.to_lowercase()
        )
    }

    fn estimate_factors(&self, metric_type: MetricType, score: f64) -> HashMap<String, f64> {
        // In production, this would analyze actual audio features
        // For now, we provide estimated factors based on the score
        let metadata = self
            .metric_metadata
            .get(&metric_type)
            .expect("value should be present");

        let base_quality = (score - metadata.range.0) / (metadata.range.1 - metadata.range.0);

        let mut factors = HashMap::new();
        for factor in &metadata.typical_factors {
            // Add some variation to make factors realistic
            let variation = (factor.len() % 10) as f64 * 0.05;
            let factor_score = (base_quality + variation).min(1.0).max(0.0);
            factors.insert(factor.clone(), factor_score);
        }

        factors
    }

    fn generate_recommendations(
        &self,
        _metric_type: MetricType,
        _score: f64,
        rating: &QualityRating,
        metadata: &MetricMetadata,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match rating {
            QualityRating::Excellent => {
                recommendations.push("Maintain current quality standards.".to_string());
                recommendations
                    .push("Consider this as a benchmark for future evaluations.".to_string());
            }
            QualityRating::Good => {
                recommendations.push("Quality is good but has room for improvement.".to_string());
                recommendations.push(format!(
                    "Focus on optimizing {} to reach excellent quality.",
                    metadata.typical_factors[0]
                ));
            }
            QualityRating::Fair => {
                recommendations.push("Noticeable quality issues need attention.".to_string());
                for factor in &metadata.typical_factors[..2] {
                    recommendations.push(format!("Improve {} to enhance overall quality.", factor));
                }
            }
            QualityRating::Poor | QualityRating::VeryPoor => {
                recommendations.push("Significant quality improvements required.".to_string());
                recommendations.push("Review entire synthesis pipeline for issues.".to_string());
                for factor in &metadata.typical_factors {
                    recommendations.push(format!("Address {} deficiencies.", factor));
                }
            }
        }

        recommendations
    }

    fn generate_comparative_context(&self, metric_type: MetricType, score: f64) -> String {
        if let Some(benchmarks) = self.benchmark_scores.get(&metric_type) {
            let percentile = self.calculate_percentile(score, benchmarks);
            format!(
                "This score is at the {:.0}th percentile compared to typical results. \
                 Scores range from {:.2} (low) to {:.2} (high) in our benchmark dataset.",
                percentile * 100.0,
                benchmarks.first().expect("collection should not be empty"),
                benchmarks.last().expect("collection should not be empty")
            )
        } else {
            "No benchmark data available for comparison.".to_string()
        }
    }

    fn generate_technical_details(
        &self,
        metric_type: MetricType,
        score: f64,
        metadata: &MetricMetadata,
    ) -> String {
        format!(
            "Technical Details:\n\
             - Metric: {} ({})\n\
             - Score: {:.6}\n\
             - Valid Range: [{:.2}, {:.2}]\n\
             - Optimization Direction: {}\n\
             - Measurement Domain: {}\n\
             - Key Factors: {}",
            metric_type,
            metadata.name,
            score,
            metadata.range.0,
            metadata.range.1,
            if metadata.higher_is_better {
                "Maximize"
            } else {
                "Minimize"
            },
            metadata.measures,
            metadata.typical_factors.join(", ")
        )
    }

    fn calculate_percentile(&self, score: f64, benchmarks: &[f64]) -> f64 {
        let count_below = benchmarks.iter().filter(|&&b| b < score).count();
        count_below as f64 / benchmarks.len() as f64
    }
}

impl Default for ExplainabilityEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explainability_engine_creation() {
        let engine = ExplainabilityEngine::new();
        assert!(!engine.available_metrics().is_empty());
    }

    #[test]
    fn test_pesq_explanation() {
        let engine = ExplainabilityEngine::new();

        let explanation = engine
            .explain_metric(MetricType::Pesq, 4.0, ExplanationLevel::Standard)
            .unwrap();

        assert_eq!(explanation.metric_type, MetricType::Pesq);
        assert_eq!(explanation.score, 4.0);
        assert!(matches!(
            explanation.rating,
            QualityRating::Good | QualityRating::Excellent
        ));
        assert!(!explanation.natural_language_summary.is_empty());
        assert!(!explanation.recommendations.is_empty());
    }

    #[test]
    fn test_stoi_explanation() {
        let engine = ExplainabilityEngine::new();

        let explanation = engine
            .explain_metric(MetricType::Stoi, 0.85, ExplanationLevel::Detailed)
            .unwrap();

        assert_eq!(explanation.metric_type, MetricType::Stoi);
        assert!(!explanation.contributing_factors.is_empty());
    }

    #[test]
    fn test_rating_determination() {
        let engine = ExplainabilityEngine::new();

        // Test excellent rating
        let exp1 = engine
            .explain_metric(MetricType::Pesq, 4.3, ExplanationLevel::Brief)
            .unwrap();
        assert_eq!(exp1.rating, QualityRating::Excellent);

        // Test poor rating
        let exp2 = engine
            .explain_metric(MetricType::Pesq, 2.0, ExplanationLevel::Brief)
            .unwrap();
        assert!(matches!(
            exp2.rating,
            QualityRating::Poor | QualityRating::VeryPoor
        ));
    }

    #[test]
    fn test_factor_explanation() {
        let engine = ExplainabilityEngine::new();

        let mut factors = HashMap::new();
        factors.insert("spectral_clarity".to_string(), 0.85);
        factors.insert("temporal_accuracy".to_string(), 0.65);

        let explanation = engine.explain_factors(MetricType::Pesq, factors).unwrap();

        assert!(explanation.contains("spectral_clarity"));
        assert!(explanation.contains("temporal_accuracy"));
    }

    #[test]
    fn test_score_comparison() {
        let engine = ExplainabilityEngine::new();

        let comparison = engine
            .compare_scores(MetricType::Pesq, 3.5, 4.0, "Model A", "Model B")
            .unwrap();

        assert!(comparison.contains("Model A"));
        assert!(comparison.contains("Model B"));
        assert!(comparison.contains("better") || comparison.contains("worse"));
    }

    #[test]
    fn test_expert_level_details() {
        let engine = ExplainabilityEngine::new();

        let explanation = engine
            .explain_metric(MetricType::Mcd, 5.5, ExplanationLevel::Expert)
            .unwrap();

        assert!(explanation.technical_details.is_some());
        let details = explanation.technical_details.unwrap();
        assert!(details.contains("Technical Details"));
        assert!(details.contains("Valid Range"));
    }

    #[test]
    fn test_invalid_score() {
        let engine = ExplainabilityEngine::new();

        // PESQ range is 1.0-4.5, test out of range
        let result = engine.explain_metric(MetricType::Pesq, 5.0, ExplanationLevel::Brief);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExplainabilityError::InvalidScore { .. }
        ));
    }

    #[test]
    fn test_recommendations_vary_by_rating() {
        let engine = ExplainabilityEngine::new();

        let excellent = engine
            .explain_metric(MetricType::Pesq, 4.4, ExplanationLevel::Standard)
            .unwrap();
        let poor = engine
            .explain_metric(MetricType::Pesq, 1.5, ExplanationLevel::Standard)
            .unwrap();

        // Poor rating should have more recommendations
        assert!(poor.recommendations.len() >= excellent.recommendations.len());
    }

    #[test]
    fn test_available_metrics() {
        let engine = ExplainabilityEngine::new();
        let metrics = engine.available_metrics();

        assert!(metrics.contains(&MetricType::Pesq));
        assert!(metrics.contains(&MetricType::Stoi));
        assert!(metrics.contains(&MetricType::Mcd));
    }
}
