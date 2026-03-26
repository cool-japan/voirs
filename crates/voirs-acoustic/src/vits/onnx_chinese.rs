//! Chinese VITS ONNX model loader
//!
//! Specialized loader for Chinese VITS models (AISHELL3-style)
//! with multiple input parameters.

use std::path::Path;

#[cfg(feature = "onnx")]
use oxionnx::{OptLevel, Session, Tensor};

use crate::{AcousticError, Result};

/// Chinese VITS ONNX inference model
#[cfg(feature = "onnx")]
pub struct ChineseVitsOnnxInference {
    session: Session,
}

#[cfg(feature = "onnx")]
impl ChineseVitsOnnxInference {
    /// Load Chinese VITS model from ONNX file
    pub fn from_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .with_optimization_level(OptLevel::All)
            .load(model_path.as_ref())
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to load ONNX model: {}", e),
            })?;

        Ok(Self { session })
    }

    /// Synthesize audio from token IDs
    ///
    /// # Parameters
    /// - `token_ids`: Token IDs (phoneme sequence)
    /// - `noise_scale`: Noise scale (default: 0.667)
    /// - `length_scale`: Length scale (default: 1.0)
    /// - `noise_scale_w`: Noise scale W (default: 0.8)
    /// - `speaker_id`: Speaker ID (default: 0)
    pub fn synthesize_with_params(
        &self,
        token_ids: &[i64],
        noise_scale: f32,
        length_scale: f32,
        noise_scale_w: f32,
        speaker_id: i64,
    ) -> Result<Vec<f32>> {
        use std::collections::HashMap;

        // Create input tensors - convert i64 to f32 for oxionnx
        let x_data: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let x = Tensor::new(x_data, vec![1, token_ids.len()]);
        let x_length = Tensor::new(vec![token_ids.len() as f32], vec![1]);
        let noise_scale_tensor = Tensor::new(vec![noise_scale], vec![1]);
        let length_scale_tensor = Tensor::new(vec![length_scale], vec![1]);
        let noise_scale_w_tensor = Tensor::new(vec![noise_scale_w], vec![1]);
        let sid = Tensor::new(vec![speaker_id as f32], vec![1]);

        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert("x", x);
        inputs.insert("x_length", x_length);
        inputs.insert("noise_scale", noise_scale_tensor);
        inputs.insert("length_scale", length_scale_tensor);
        inputs.insert("noise_scale_w", noise_scale_w_tensor);
        inputs.insert("sid", sid);

        // Run inference
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Inference failed: {}", e),
            })?;

        // Extract output audio (first output)
        let audio_tensor = outputs
            .values()
            .next()
            .ok_or_else(|| AcousticError::ModelError {
                message: "No output from model".to_string(),
            })?;

        // Flatten the audio data
        Ok(audio_tensor.data.clone())
    }

    /// Synthesize audio with default parameters
    pub fn synthesize(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        self.synthesize_with_params(token_ids, 0.667, 1.0, 0.8, 0)
    }
}

#[cfg(not(feature = "onnx"))]
pub struct ChineseVitsOnnxInference;

#[cfg(not(feature = "onnx"))]
impl ChineseVitsOnnxInference {
    pub fn from_file<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
        Err(AcousticError::ModelError {
            message: "ONNX feature not enabled. Enable with --features onnx".to_string(),
        })
    }

    pub fn synthesize(&self, _token_ids: &[i64]) -> Result<Vec<f32>> {
        Err(AcousticError::ModelError {
            message: "ONNX feature not enabled".to_string(),
        })
    }

    pub fn synthesize_with_params(
        &self,
        _token_ids: &[i64],
        _noise_scale: f32,
        _length_scale: f32,
        _noise_scale_w: f32,
        _speaker_id: i64,
    ) -> Result<Vec<f32>> {
        Err(AcousticError::ModelError {
            message: "ONNX feature not enabled".to_string(),
        })
    }
}
