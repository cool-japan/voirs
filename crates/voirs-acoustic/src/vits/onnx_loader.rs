//! VITS ONNX model loader and inference
//!
//! Uses ONNX Runtime to directly load and run VITS models

use std::path::Path;

#[cfg(feature = "onnx")]
use ort::{inputs, session::Session, value::Value};

use crate::{AcousticError, Result};

/// VITS ONNX inference model
#[cfg(feature = "onnx")]
pub struct VitsOnnxInference {
    session: Session,
}

#[cfg(feature = "onnx")]
impl VitsOnnxInference {
    /// Load VITS model from ONNX file
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
    pub fn synthesize(&mut self, token_ids: &[i64]) -> Result<Vec<f32>> {
        // Create input tensor: 1D array (no batch dimension)
        let input_shape = vec![token_ids.len()];
        let token_data = token_ids.to_vec();
        let input_tensor = Value::from_array((input_shape, token_data)).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create input tensor: {}", e))
        })?;

        // Run inference
        let inputs_vec = inputs!["text" => input_tensor];
        let outputs = self
            .session
            .run(inputs_vec)
            .map_err(|e| AcousticError::ModelError(format!("Inference failed: {}", e)))?;

        // Extract output audio - VITS outputs "wav" as first output
        let audio_tensor = outputs
            .get("wav")
            .ok_or_else(|| AcousticError::ModelError("No 'wav' output from model".to_string()))?;

        // Convert to Vec<f32>
        let (_, audio_slice) = audio_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to extract audio: {}", e)))?;

        let audio_data: Vec<f32> = audio_slice.to_vec();

        Ok(audio_data)
    }
}

#[cfg(not(feature = "onnx"))]
pub struct VitsOnnxInference;

#[cfg(not(feature = "onnx"))]
impl VitsOnnxInference {
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
}
