//! OxiONNX backend for MOS (Mean Opinion Score) prediction.
//!
//! This module provides an ONNX-based neural MOS predictor that estimates
//! the perceived quality of synthesized speech on a 1.0–5.0 scale.
//! The model takes raw audio waveform samples and outputs a single MOS score.

use crate::EvaluationError;
use oxionnx::{OptLevel, Session, Tensor};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Configuration for the ONNX MOS predictor backend.
#[derive(Debug, Clone)]
pub struct OnnxMosPredictorConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,

    /// Expected sample rate of input audio in Hz (default: 16000).
    pub sample_rate: u32,

    /// Maximum input length in samples (default: 160000, i.e. 10 seconds at 16 kHz).
    pub max_length: usize,

    /// ONNX graph optimization level (default: All).
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference.
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse.
    pub enable_memory_pool: bool,
}

impl Default for OnnxMosPredictorConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            sample_rate: 16000,
            max_length: 16000 * 10,
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
        }
    }
}

/// ONNX-based MOS (Mean Opinion Score) predictor.
///
/// Predicts subjective speech quality from raw audio waveform using a
/// pre-trained neural network exported to ONNX format.
///
/// ## Model I/O
///
/// - **Input**: `waveform` tensor of shape `[1, samples]` (float32, 16 kHz mono).
/// - **Output**: `mos_score` tensor of shape `[1]` (float32, range 1.0–5.0).
pub struct OnnxMosPredictor {
    /// The loaded ONNX session.
    session: Arc<RwLock<Session>>,

    /// Configuration snapshot.
    config: OnnxMosPredictorConfig,
}

impl OnnxMosPredictor {
    /// Create a new ONNX MOS predictor by loading the model from disk.
    pub fn new(config: OnnxMosPredictorConfig) -> Result<Self, EvaluationError> {
        info!(
            "Loading ONNX MOS predictor model from {:?}",
            config.model_path
        );

        let session = load_session(&config.model_path, &config)?;

        info!("ONNX MOS predictor model loaded successfully");
        debug!(
            "Config: sample_rate={}, max_length={}",
            config.sample_rate, config.max_length
        );

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
        })
    }

    /// Predict a MOS score for the given audio samples.
    ///
    /// # Arguments
    /// * `audio_samples` - Raw audio waveform at the configured sample rate (mono, float32).
    ///
    /// # Returns
    /// A MOS score in the range 1.0–5.0.
    pub fn predict_mos(&self, audio_samples: &[f32]) -> Result<f32, EvaluationError> {
        if audio_samples.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Audio samples must not be empty".to_string(),
            });
        }

        // Truncate to max_length if necessary
        let samples = if audio_samples.len() > self.config.max_length {
            warn!(
                "Audio length {} exceeds max_length {}, truncating",
                audio_samples.len(),
                self.config.max_length
            );
            &audio_samples[..self.config.max_length]
        } else {
            audio_samples
        };

        let num_samples = samples.len();

        // Build input tensor [1, samples]
        let input = Tensor::new(samples.to_vec(), vec![1, num_samples]);

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("waveform", input);

        // Run inference
        let session = self
            .session
            .read()
            .map_err(|e| EvaluationError::ModelError {
                message: format!("Session lock poisoned: {e}"),
                source: None,
            })?;

        let outputs = session
            .run(&inputs)
            .map_err(|e| EvaluationError::ModelError {
                message: format!("ONNX inference failed: {e}"),
                source: None,
            })?;

        // Extract MOS score
        let mos_tensor = outputs
            .get("mos_score")
            .ok_or_else(|| EvaluationError::ModelError {
                message: "Missing 'mos_score' output from ONNX model".to_string(),
                source: None,
            })?;

        let raw_score =
            mos_tensor
                .data
                .first()
                .copied()
                .ok_or_else(|| EvaluationError::ModelError {
                    message: "MOS score output tensor is empty".to_string(),
                    source: None,
                })?;

        // Clamp to valid MOS range [1.0, 5.0]
        let score = raw_score.clamp(1.0, 5.0);

        debug!(
            "MOS prediction: raw={:.4}, clamped={:.4}, samples={}",
            raw_score, score, num_samples
        );

        Ok(score)
    }

    /// Return the configured sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Return the configured maximum input length.
    pub fn max_length(&self) -> usize {
        self.config.max_length
    }
}

/// Load an ONNX session from disk with the supplied configuration.
fn load_session(path: &Path, config: &OnnxMosPredictorConfig) -> Result<Session, EvaluationError> {
    let mut builder = Session::builder()
        .with_optimization_level(config.opt_level)
        .with_memory_pool(config.enable_memory_pool);
    if config.enable_profiling {
        builder = builder.with_profiling();
    }
    builder.load(path).map_err(|e| EvaluationError::ModelError {
        message: format!("Failed to load ONNX model from {}: {e}", path.display()),
        source: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxMosPredictorConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.max_length, 160000);
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
    }
}
