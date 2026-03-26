//! OxiONNX backend for neural HRTF synthesis.
//!
//! This module provides an ONNX-based neural HRTF synthesizer that generates
//! personalized Head-Related Transfer Functions using a neural network model.
//! The model takes a source position (azimuth, elevation, distance) and produces
//! left and right ear HRTF filter coefficients.

use crate::{Error, HrtfError};
use oxionnx::{OptLevel, Session, Tensor};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Configuration for the ONNX neural HRTF backend.
#[derive(Debug, Clone)]
pub struct OnnxNeuralHrtfConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,

    /// Length of the HRTF filter in samples (default: 512).
    pub filter_length: usize,

    /// Sample rate in Hz (default: 48000).
    pub sample_rate: u32,

    /// ONNX graph optimization level (default: All).
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference.
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse.
    pub enable_memory_pool: bool,
}

impl Default for OnnxNeuralHrtfConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            filter_length: 512,
            sample_rate: 48000,
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
        }
    }
}

/// ONNX-based neural HRTF synthesizer.
///
/// Takes a source position described by azimuth, elevation, and distance,
/// and produces left and right ear HRTF filter coefficients via an ONNX model.
///
/// ## Model I/O
///
/// - **Input**: `source_position` tensor of shape `[1, 3]` containing
///   `(azimuth, elevation, distance)` in radians / meters.
/// - **Output**: `hrtf_left` tensor of shape `[1, filter_length]` and
///   `hrtf_right` tensor of shape `[1, filter_length]`.
pub struct OnnxNeuralHrtf {
    /// The loaded ONNX session.
    session: Arc<RwLock<Session>>,

    /// Configuration snapshot.
    config: OnnxNeuralHrtfConfig,
}

impl OnnxNeuralHrtf {
    /// Create a new ONNX neural HRTF synthesizer by loading the model from disk.
    pub fn new(config: OnnxNeuralHrtfConfig) -> Result<Self, Error> {
        info!(
            "Loading ONNX neural HRTF model from {:?}",
            config.model_path
        );

        let session = load_session(&config.model_path, &config)?;

        info!("ONNX neural HRTF model loaded successfully");
        debug!(
            "Config: filter_length={}, sample_rate={}",
            config.filter_length, config.sample_rate
        );

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
        })
    }

    /// Synthesize left and right HRTF filters for the given source position.
    ///
    /// # Arguments
    /// * `azimuth` - Horizontal angle in radians (0 = front, positive = right).
    /// * `elevation` - Vertical angle in radians (0 = horizon, positive = up).
    /// * `distance` - Distance from the listener in meters.
    ///
    /// # Returns
    /// A tuple of `(left_hrtf, right_hrtf)` filter coefficient vectors,
    /// each of length `filter_length`.
    pub fn synthesize_hrtf(
        &self,
        azimuth: f32,
        elevation: f32,
        distance: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        // Build input tensor [1, 3]
        let input = Tensor::new(vec![azimuth, elevation, distance], vec![1, 3]);

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("source_position", input);

        // Run inference
        let session = self
            .session
            .read()
            .map_err(|e| Error::LegacyProcessing(format!("Session lock poisoned: {e}")))?;

        let outputs = session.run(&inputs).map_err(|e| {
            Error::Hrtf(HrtfError::InterpolationFailed {
                reason: format!("ONNX inference failed: {e}"),
            })
        })?;

        // Extract left HRTF
        let hrtf_left_tensor = outputs.get("hrtf_left").ok_or_else(|| {
            Error::Hrtf(HrtfError::InterpolationFailed {
                reason: "Missing 'hrtf_left' output from ONNX model".to_string(),
            })
        })?;

        // Extract right HRTF
        let hrtf_right_tensor = outputs.get("hrtf_right").ok_or_else(|| {
            Error::Hrtf(HrtfError::InterpolationFailed {
                reason: "Missing 'hrtf_right' output from ONNX model".to_string(),
            })
        })?;

        let left = truncate_or_pad(&hrtf_left_tensor.data, self.config.filter_length);
        let right = truncate_or_pad(&hrtf_right_tensor.data, self.config.filter_length);

        debug!(
            "Synthesized HRTF: azimuth={:.3}, elevation={:.3}, distance={:.3}, filter_len={}",
            azimuth, elevation, distance, self.config.filter_length
        );

        Ok((left, right))
    }

    /// Return the configured filter length.
    pub fn filter_length(&self) -> usize {
        self.config.filter_length
    }

    /// Return the configured sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// Load an ONNX session from disk with the supplied configuration.
fn load_session(path: &Path, config: &OnnxNeuralHrtfConfig) -> Result<Session, Error> {
    let mut builder = Session::builder()
        .with_optimization_level(config.opt_level)
        .with_memory_pool(config.enable_memory_pool);
    if config.enable_profiling {
        builder = builder.with_profiling();
    }
    builder.load(path).map_err(|e| {
        Error::Hrtf(HrtfError::DatabaseLoadFailed {
            path: path.display().to_string(),
            reason: format!("Failed to load ONNX model: {e}"),
        })
    })
}

/// Truncate or zero-pad a slice to the target length.
fn truncate_or_pad(data: &[f32], target_len: usize) -> Vec<f32> {
    if data.len() >= target_len {
        data[..target_len].to_vec()
    } else {
        let mut out = data.to_vec();
        out.resize(target_len, 0.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_or_pad_exact() {
        let data = vec![1.0, 2.0, 3.0];
        let result = truncate_or_pad(&data, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_truncate_or_pad_shorter() {
        let data = vec![1.0, 2.0];
        let result = truncate_or_pad(&data, 4);
        assert_eq!(result, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_truncate_or_pad_longer() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = truncate_or_pad(&data, 2);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_default_config() {
        let config = OnnxNeuralHrtfConfig::default();
        assert_eq!(config.filter_length, 512);
        assert_eq!(config.sample_rate, 48000);
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
    }
}
