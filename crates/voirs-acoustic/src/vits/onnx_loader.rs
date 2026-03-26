//! VITS ONNX model loader and inference
//!
//! Uses ONNX Runtime to directly load and run VITS models

use std::path::Path;

#[cfg(feature = "onnx")]
use oxionnx::{OptLevel, Session, Tensor};

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
            .with_optimization_level(OptLevel::All)
            .load(model_path.as_ref())
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to load ONNX model: {}", e),
            })?;

        Ok(Self { session })
    }

    /// Synthesize audio from token IDs
    pub fn synthesize(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        use std::collections::HashMap;

        // Create input tensor: convert i64 to f32 (oxionnx uses f32 tensors)
        let token_data: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let input_tensor = Tensor::new(token_data, vec![token_ids.len()]);

        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert("text", input_tensor);

        // Run inference
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Inference failed: {}", e),
            })?;

        // Extract output audio - VITS outputs "wav" as first output
        let audio_tensor = outputs
            .get("wav")
            .ok_or_else(|| AcousticError::ModelError {
                message: "No 'wav' output from model".to_string(),
            })?;

        Ok(audio_tensor.data.clone())
    }
}

#[cfg(not(feature = "onnx"))]
pub struct VitsOnnxInference;

#[cfg(not(feature = "onnx"))]
impl VitsOnnxInference {
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
}
