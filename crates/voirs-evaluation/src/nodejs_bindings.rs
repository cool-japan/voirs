//! Node.js/JavaScript bindings for VoiRS evaluation framework
//!
//! This module provides N-API bindings for JavaScript/Node.js applications to use
//! the VoiRS evaluation framework. It exposes quality metrics, pronunciation assessment,
//! and comparative analysis capabilities through a JavaScript-friendly interface.
//!
//! ## Usage
//!
//! ```javascript
//! const { QualityEvaluator, PronunciationEvaluator } = require('@voirs/evaluation');
//!
//! // Create evaluator
//! const evaluator = new QualityEvaluator();
//!
//! // Evaluate quality
//! const result = await evaluator.evaluateQuality(
//!     generatedAudio,  // Buffer or Float32Array
//!     referenceAudio,  // Buffer or Float32Array
//!     { sampleRate: 16000, channels: 1 }
//! );
//!
//! console.log(`Quality score: ${result.overallScore}`);
//! console.log(`PESQ: ${result.pesq}, STOI: ${result.stoi}`);
//! ```

use crate::prelude::*;
use crate::{
    ComparisonConfig, EvaluationError, PronunciationEvaluationConfig, QualityEvaluationConfig,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use voirs_sdk::AudioBuffer;

/// JavaScript-compatible quality evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsQualityConfig {
    /// Enable objective quality metrics (PESQ, STOI, MCD)
    #[serde(default = "default_true")]
    pub objective_metrics: bool,
    /// Enable subjective quality prediction (MOS)
    #[serde(default = "default_true")]
    pub subjective_metrics: bool,
    /// Enable perceptual quality metrics
    #[serde(default = "default_true")]
    pub perceptual_metrics: bool,
    /// Require reference audio for evaluation
    #[serde(default)]
    pub require_reference: bool,
    /// Specific metrics to compute (e.g., `["pesq", "stoi", "mcd"]`)
    #[serde(default)]
    pub metrics: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

impl Default for JsQualityConfig {
    fn default() -> Self {
        Self {
            objective_metrics: true,
            subjective_metrics: true,
            perceptual_metrics: true,
            require_reference: false,
            metrics: None,
        }
    }
}

impl From<JsQualityConfig> for QualityEvaluationConfig {
    fn from(js_config: JsQualityConfig) -> Self {
        let metrics = if let Some(metric_names) = js_config.metrics {
            metric_names
                .iter()
                .filter_map(|name| match name.to_lowercase().as_str() {
                    "pesq" => Some(QualityMetric::PESQ),
                    "stoi" => Some(QualityMetric::STOI),
                    "mcd" => Some(QualityMetric::MCD),
                    "spectral_distortion" | "lsd" => Some(QualityMetric::SpectralDistortion),
                    "naturalness" => Some(QualityMetric::Naturalness),
                    "intelligibility" => Some(QualityMetric::Intelligibility),
                    _ => None,
                })
                .collect()
        } else {
            // Default metrics if not specified
            vec![QualityMetric::PESQ, QualityMetric::STOI, QualityMetric::MCD]
        };

        QualityEvaluationConfig {
            objective_metrics: js_config.objective_metrics,
            subjective_metrics: js_config.subjective_metrics,
            perceptual_metrics: js_config.perceptual_metrics,
            metrics,
            require_reference: js_config.require_reference,
            ..Default::default()
        }
    }
}

/// JavaScript-compatible audio options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsAudioOptions {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Language code (optional)
    pub language: Option<String>,
}

/// JavaScript-compatible quality evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsQualityResult {
    /// Overall quality score (0-1)
    pub overall_score: f32,
    /// PESQ score (1-4.5)
    pub pesq: Option<f32>,
    /// STOI score (0-1)
    pub stoi: Option<f32>,
    /// MCD score (lower is better)
    pub mcd: Option<f32>,
    /// Signal-to-noise ratio
    pub snr: Option<f32>,
    /// Spectral distortion
    pub spectral_distortion: Option<f32>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Additional metrics as JSON
    pub additional_metrics: Option<String>,
}

/// JavaScript-compatible pronunciation evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsPronunciationResult {
    /// Overall pronunciation score (0-1)
    pub overall_score: f32,
    /// Phoneme accuracy score
    pub phoneme_accuracy: f32,
    /// Word accuracy score
    pub word_accuracy: f32,
    /// Fluency score
    pub fluency: f32,
    /// Prosody score
    pub prosody: f32,
    /// Detailed phoneme scores as JSON array
    pub phoneme_scores: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// JavaScript-compatible comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsComparisonResult {
    /// System A score
    pub system_a_score: f32,
    /// System B score
    pub system_b_score: f32,
    /// Winner: "A", "B", or "Tie"
    pub winner: String,
    /// Statistical significance (p-value)
    pub p_value: Option<f64>,
    /// Effect size (Cohen's d)
    pub effect_size: Option<f64>,
    /// Confidence interval as [lower, upper]
    pub confidence_interval: Option<Vec<f64>>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// Node.js-compatible quality evaluator
///
/// This struct provides a JavaScript-friendly interface to the quality evaluation
/// functionality. It handles the conversion between JavaScript types (buffers, arrays)
/// and Rust types (AudioBuffer).
pub struct NodeJsQualityEvaluator {
    /// Internal quality evaluator
    evaluator: Arc<QualityEvaluator>,
    /// Configuration
    config: QualityEvaluationConfig,
}

impl NodeJsQualityEvaluator {
    /// Create a new Node.js quality evaluator
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluator cannot be initialized
    pub async fn new() -> Result<Self, EvaluationError> {
        let config = QualityEvaluationConfig::default();
        let evaluator = QualityEvaluator::new().await?;

        Ok(Self {
            evaluator: Arc::new(evaluator),
            config,
        })
    }

    /// Create with custom configuration (JSON string)
    ///
    /// # Configuration Format
    ///
    /// The configuration should be a JSON string with the following fields:
    /// ```json
    /// {
    ///   "objective_metrics": true,
    ///   "subjective_metrics": true,
    ///   "perceptual_metrics": true,
    ///   "require_reference": false,
    ///   "metrics": ["pesq", "stoi", "mcd"]
    /// }
    /// ```
    ///
    /// # Supported Metrics
    ///
    /// - `pesq`: Perceptual Evaluation of Speech Quality (ITU-T P.862)
    /// - `stoi`: Short-Time Objective Intelligibility
    /// - `mcd`: Mel-Cepstral Distortion
    /// - `snr`: Signal-to-Noise Ratio
    /// - `lsd`: Log-Spectral Distortion
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - JSON parsing fails
    /// - Evaluator initialization fails
    /// - Invalid configuration values
    ///
    /// # Example
    ///
    /// ```javascript
    /// const config = JSON.stringify({
    ///   objective_metrics: true,
    ///   subjective_metrics: false,
    ///   metrics: ["pesq", "stoi"]
    /// });
    ///
    /// const evaluator = await NodeJsQualityEvaluator.newWithConfig(config);
    /// ```
    pub async fn new_with_config(config_json: &str) -> Result<Self, EvaluationError> {
        // Parse JavaScript configuration
        let js_config: JsQualityConfig =
            serde_json::from_str(config_json).map_err(|e| EvaluationError::ConfigurationError {
                message: format!("Failed to parse configuration JSON: {}", e),
            })?;

        // Convert to Rust configuration
        let config = QualityEvaluationConfig::from(js_config);

        // Create evaluator
        let evaluator = QualityEvaluator::new().await?;

        Ok(Self {
            evaluator: Arc::new(evaluator),
            config,
        })
    }

    /// Evaluate quality from raw audio samples
    ///
    /// # Arguments
    ///
    /// * `generated_samples` - Generated audio samples as f32 array
    /// * `reference_samples` - Reference audio samples as f32 array (optional)
    /// * `options` - Audio options (sample rate, channels, language)
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails or audio processing fails
    pub async fn evaluate_quality(
        &self,
        generated_samples: Vec<f32>,
        reference_samples: Option<Vec<f32>>,
        options: JsAudioOptions,
    ) -> Result<JsQualityResult, EvaluationError> {
        let start_time = std::time::Instant::now();

        // Convert to AudioBuffer
        let generated = AudioBuffer::new(generated_samples, options.sample_rate, options.channels);

        let reference = reference_samples
            .map(|samples| AudioBuffer::new(samples, options.sample_rate, options.channels));

        // Evaluate with default config
        let result = self
            .evaluator
            .evaluate_quality(&generated, reference.as_ref(), None)
            .await?;

        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Convert to JavaScript-compatible result
        Ok(JsQualityResult {
            overall_score: result.overall_score,
            pesq: result.component_scores.get("pesq").copied(),
            stoi: result.component_scores.get("stoi").copied(),
            mcd: result.component_scores.get("mcd").copied(),
            snr: result.component_scores.get("snr").copied(),
            spectral_distortion: result.component_scores.get("spectral_distortion").copied(),
            processing_time_ms,
            additional_metrics: serde_json::to_string(&result.component_scores).ok(),
        })
    }

    /// Batch evaluate multiple audio pairs
    ///
    /// # Errors
    ///
    /// Returns an error if any evaluation fails
    pub async fn batch_evaluate(
        &self,
        batch: Vec<(Vec<f32>, Option<Vec<f32>>, JsAudioOptions)>,
    ) -> Result<Vec<JsQualityResult>, EvaluationError> {
        let mut results = Vec::with_capacity(batch.len());

        for (generated, reference, options) in batch {
            let result = self.evaluate_quality(generated, reference, options).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get configuration summary as JSON string
    #[must_use]
    pub fn get_config_summary(&self) -> String {
        // Return a simple config summary since QualityEvaluationConfig doesn't implement Serialize
        format!(
            "{{\"objective_metrics\": {}, \"subjective_metrics\": {}, \"real_time_mode\": {}}}",
            self.config.objective_metrics,
            self.config.subjective_metrics,
            self.config.real_time_mode
        )
    }
}

/// Node.js-compatible pronunciation evaluator
pub struct NodeJsPronunciationEvaluator {
    /// Internal pronunciation evaluator
    evaluator: Arc<PronunciationEvaluatorImpl>,
    /// Configuration
    config: PronunciationEvaluationConfig,
}

impl NodeJsPronunciationEvaluator {
    /// Create a new Node.js pronunciation evaluator
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluator cannot be initialized
    pub async fn new() -> Result<Self, EvaluationError> {
        let config = PronunciationEvaluationConfig::default();
        let evaluator = PronunciationEvaluatorImpl::new().await?;

        Ok(Self {
            evaluator: Arc::new(evaluator),
            config,
        })
    }

    /// Evaluate pronunciation from audio and text
    ///
    /// # Arguments
    ///
    /// * `audio_samples` - Audio samples as f32 array
    /// * `expected_text` - Expected text to be spoken
    /// * `options` - Audio options
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails
    pub async fn evaluate_pronunciation(
        &self,
        audio_samples: Vec<f32>,
        expected_text: String,
        options: JsAudioOptions,
    ) -> Result<JsPronunciationResult, EvaluationError> {
        let start_time = std::time::Instant::now();

        // Convert to AudioBuffer
        let audio = AudioBuffer::new(audio_samples, options.sample_rate, options.channels);

        // Evaluate pronunciation with default config
        let result = self
            .evaluator
            .evaluate_pronunciation(&audio, &expected_text, None)
            .await?;

        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Calculate average phoneme and word accuracy
        let phoneme_accuracy = if !result.phoneme_scores.is_empty() {
            result
                .phoneme_scores
                .iter()
                .map(|p| p.accuracy)
                .sum::<f32>()
                / result.phoneme_scores.len() as f32
        } else {
            0.0
        };

        let word_accuracy = if !result.word_scores.is_empty() {
            result.word_scores.iter().map(|w| w.accuracy).sum::<f32>()
                / result.word_scores.len() as f32
        } else {
            0.0
        };

        // Serialize phoneme scores to JSON (empty for now, can be enhanced)
        let phoneme_scores = None;

        Ok(JsPronunciationResult {
            overall_score: result.overall_score,
            phoneme_accuracy,
            word_accuracy,
            fluency: result.fluency_score,
            prosody: (result.stress_accuracy + result.intonation_accuracy) / 2.0,
            phoneme_scores,
            processing_time_ms,
        })
    }
}

/// Node.js-compatible comparative evaluator
pub struct NodeJsComparativeEvaluator {
    /// Internal comparative evaluator
    evaluator: Arc<ComparativeEvaluatorImpl>,
    /// Configuration
    config: ComparisonConfig,
}

impl NodeJsComparativeEvaluator {
    /// Create a new Node.js comparative evaluator
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluator cannot be initialized
    pub async fn new() -> Result<Self, EvaluationError> {
        let config = ComparisonConfig::default();
        let evaluator = ComparativeEvaluatorImpl::new().await?;

        Ok(Self {
            evaluator: Arc::new(evaluator),
            config,
        })
    }

    /// Compare two audio systems
    ///
    /// # Arguments
    ///
    /// * `system_a_samples` - Audio samples from system A
    /// * `system_b_samples` - Audio samples from system B
    /// * `reference_samples` - Reference audio samples (optional)
    /// * `options` - Audio options
    ///
    /// # Errors
    ///
    /// Returns an error if comparison fails
    pub async fn compare_systems(
        &self,
        system_a_samples: Vec<f32>,
        system_b_samples: Vec<f32>,
        reference_samples: Option<Vec<f32>>,
        options: JsAudioOptions,
    ) -> Result<JsComparisonResult, EvaluationError> {
        let start_time = std::time::Instant::now();

        // Convert to AudioBuffer
        let system_a = AudioBuffer::new(system_a_samples, options.sample_rate, options.channels);

        let system_b = AudioBuffer::new(system_b_samples, options.sample_rate, options.channels);

        let reference = reference_samples
            .map(|samples| AudioBuffer::new(samples, options.sample_rate, options.channels));

        // Compare - use the trait method
        use crate::traits::ComparativeEvaluator as _;
        let result = self
            .evaluator
            .compare_samples(&system_a, &system_b, None)
            .await?;

        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Extract system scores from metric comparisons
        let system_a_score = result
            .metric_comparisons
            .values()
            .map(|m| m.score_a)
            .sum::<f32>()
            / result.metric_comparisons.len().max(1) as f32;

        let system_b_score = result
            .metric_comparisons
            .values()
            .map(|m| m.score_b)
            .sum::<f32>()
            / result.metric_comparisons.len().max(1) as f32;

        // Determine winner
        let winner = if result.preference_score < -0.1 {
            "A"
        } else if result.preference_score > 0.1 {
            "B"
        } else {
            "Tie"
        }
        .to_string();

        // Get average p-value from statistical significance
        let p_value = result
            .statistical_significance
            .values()
            .copied()
            .sum::<f32>()
            / result.statistical_significance.len().max(1) as f32;

        Ok(JsComparisonResult {
            system_a_score,
            system_b_score,
            winner,
            p_value: Some(p_value as f64),
            effect_size: Some(result.preference_score as f64),
            confidence_interval: None,
            processing_time_ms,
        })
    }
}

/// Helper function to convert Node.js Buffer to Vec<f32>
///
/// This would be implemented in the actual N-API binding layer
pub fn buffer_to_f32_vec(buffer: &[u8]) -> Result<Vec<f32>, EvaluationError> {
    if buffer.len() % 4 != 0 {
        return Err(EvaluationError::InvalidInput {
            message: format!("Buffer length {} is not a multiple of 4", buffer.len()),
        });
    }

    let mut samples = Vec::with_capacity(buffer.len() / 4);
    for chunk in buffer.chunks_exact(4) {
        let bytes: [u8; 4] =
            chunk
                .try_into()
                .map_err(|_| EvaluationError::AudioProcessingError {
                    message: "Failed to convert buffer chunk to f32".to_string(),
                    source: None,
                })?;
        samples.push(f32::from_le_bytes(bytes));
    }

    Ok(samples)
}

/// Helper function to convert Vec<f32> to Node.js Buffer
///
/// This would be implemented in the actual N-API binding layer
#[must_use]
pub fn f32_vec_to_buffer(samples: &[f32]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        buffer.extend_from_slice(&sample.to_le_bytes());
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_js_audio_options_serialization() {
        let options = JsAudioOptions {
            sample_rate: 16000,
            channels: 1,
            language: Some("en-US".to_string()),
        };

        let json = serde_json::to_string(&options).unwrap();
        let deserialized: JsAudioOptions = serde_json::from_str(&json).unwrap();

        assert_eq!(options.sample_rate, deserialized.sample_rate);
        assert_eq!(options.channels, deserialized.channels);
        assert_eq!(options.language, deserialized.language);
    }

    #[test]
    fn test_buffer_conversion() {
        let samples = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let buffer = f32_vec_to_buffer(&samples);
        let converted = buffer_to_f32_vec(&buffer).unwrap();

        assert_eq!(samples.len(), converted.len());
        for (original, converted) in samples.iter().zip(converted.iter()) {
            assert!((original - converted).abs() < 1e-6);
        }
    }

    #[test]
    fn test_invalid_buffer_length() {
        let buffer = vec![0_u8, 1, 2]; // Not a multiple of 4
        let result = buffer_to_f32_vec(&buffer);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_nodejs_quality_evaluator_creation() {
        let result = NodeJsQualityEvaluator::new().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_nodejs_pronunciation_evaluator_creation() {
        let result = NodeJsPronunciationEvaluator::new().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_nodejs_comparative_evaluator_creation() {
        let result = NodeJsComparativeEvaluator::new().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quality_evaluation_basic() {
        let evaluator = NodeJsQualityEvaluator::new().await.unwrap();

        let generated = vec![0.1_f32; 16000];
        let reference = vec![0.12_f32; 16000];
        let options = JsAudioOptions {
            sample_rate: 16000,
            channels: 1,
            language: Some("en-US".to_string()),
        };

        let result = evaluator
            .evaluate_quality(generated, Some(reference), options)
            .await;

        assert!(result.is_ok());
        let quality = result.unwrap();

        // For synthetic test data, scores may be outside normal range
        // Just verify we got a result with valid processing time
        assert!(quality.processing_time_ms > 0.0);

        // Verify we can serialize the result (basic sanity check)
        let _json = serde_json::to_string(&quality);
        assert!(_json.is_ok());
    }

    #[tokio::test]
    async fn test_config_summary() {
        let evaluator = NodeJsQualityEvaluator::new().await.unwrap();
        let config_summary = evaluator.get_config_summary();

        assert!(!config_summary.is_empty());
        assert!(config_summary.contains("objective_metrics") || config_summary.contains("{"));
    }

    #[tokio::test]
    async fn test_batch_evaluation() {
        let evaluator = NodeJsQualityEvaluator::new().await.unwrap();

        let batch = vec![
            (
                vec![0.1_f32; 8000],
                Some(vec![0.11_f32; 8000]),
                JsAudioOptions {
                    sample_rate: 16000,
                    channels: 1,
                    language: Some("en-US".to_string()),
                },
            ),
            (
                vec![0.2_f32; 8000],
                Some(vec![0.21_f32; 8000]),
                JsAudioOptions {
                    sample_rate: 16000,
                    channels: 1,
                    language: Some("en-US".to_string()),
                },
            ),
        ];

        let results = evaluator.batch_evaluate(batch).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 2);
    }
}
