//! Chinese VITS ONNX model loader
//!
//! Specialized loader for Chinese VITS models (AISHELL3-style)
//! with multiple input parameters.

use std::path::Path;

#[cfg(feature = "onnx")]
use ort::{inputs, session::Session, value::Value};

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
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create session builder: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to load ONNX model: {}", e)))?;

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
        &mut self,
        token_ids: &[i64],
        noise_scale: f32,
        length_scale: f32,
        noise_scale_w: f32,
        speaker_id: i64,
    ) -> Result<Vec<f32>> {
        // Create input tensors
        let x = Value::from_array((vec![1, token_ids.len()], token_ids.to_vec()))
            .map_err(|e| AcousticError::ModelError(format!("Failed to create x tensor: {}", e)))?;

        let x_length = Value::from_array((vec![1], vec![token_ids.len() as i64])).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create x_length tensor: {}", e))
        })?;

        let noise_scale_tensor = Value::from_array((vec![1], vec![noise_scale])).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create noise_scale tensor: {}", e))
        })?;

        let length_scale_tensor =
            Value::from_array((vec![1], vec![length_scale])).map_err(|e| {
                AcousticError::ModelError(format!("Failed to create length_scale tensor: {}", e))
            })?;

        let noise_scale_w_tensor =
            Value::from_array((vec![1], vec![noise_scale_w])).map_err(|e| {
                AcousticError::ModelError(format!("Failed to create noise_scale_w tensor: {}", e))
            })?;

        let sid = Value::from_array((vec![1], vec![speaker_id])).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create sid tensor: {}", e))
        })?;

        // Run inference
        let inputs_vec = inputs![
            "x" => x,
            "x_length" => x_length,
            "noise_scale" => noise_scale_tensor,
            "length_scale" => length_scale_tensor,
            "noise_scale_w" => noise_scale_w_tensor,
            "sid" => sid
        ];

        let outputs = self
            .session
            .run(inputs_vec)
            .map_err(|e| AcousticError::ModelError(format!("Inference failed: {}", e)))?;

        // Extract output audio (first output)
        let audio_tensor = outputs
            .iter()
            .next()
            .ok_or_else(|| AcousticError::ModelError("No output from model".to_string()))?
            .1;

        // Convert to Vec<f32> - output is (1, 1, N) shape, flatten it
        let (_, audio_slice) = audio_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to extract audio: {}", e)))?;

        // Flatten the audio data
        let audio_data: Vec<f32> = audio_slice.to_vec();

        Ok(audio_data)
    }

    /// Synthesize audio with default parameters
    pub fn synthesize(&mut self, token_ids: &[i64]) -> Result<Vec<f32>> {
        self.synthesize_with_params(token_ids, 0.667, 1.0, 0.8, 0)
    }
}

#[cfg(not(feature = "onnx"))]
pub struct ChineseVitsOnnxInference;

#[cfg(not(feature = "onnx"))]
impl ChineseVitsOnnxInference {
    pub fn from_file<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
        Err(AcousticError::ModelError(
            "ONNX feature not enabled. Enable with --features onnx".to_string(),
        ))
    }

    pub fn synthesize(&self, _token_ids: &[i64]) -> Result<Vec<f32>> {
        Err(AcousticError::ModelError(
            "ONNX feature not enabled".to_string(),
        ))
    }

    pub fn synthesize_with_params(
        &self,
        _token_ids: &[i64],
        _noise_scale: f32,
        _length_scale: f32,
        _noise_scale_w: f32,
        _speaker_id: i64,
    ) -> Result<Vec<f32>> {
        Err(AcousticError::ModelError(
            "ONNX feature not enabled".to_string(),
        ))
    }
}
