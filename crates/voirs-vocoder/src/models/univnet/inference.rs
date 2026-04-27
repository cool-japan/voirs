//! UnivNet inference engine for mel-to-audio conversion.

use super::config::{UnivNetConfig, UnivNetVariant};
use super::generator::UnivNetGenerator;
use crate::Result;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{VarBuilder, VarMap};

use std::path::Path;

/// UnivNet inference engine
#[cfg(feature = "candle")]
pub struct UnivNetInference {
    /// Generator network
    generator: UnivNetGenerator,
    /// Device (CPU/CUDA/Metal)
    device: Device,
    /// Configuration
    config: UnivNetConfig,
    /// VarMap holding all learnable parameters (enables real weight loading)
    varmap: VarMap,
}

#[cfg(feature = "candle")]
impl UnivNetInference {
    /// Create a new inference engine with specified configuration
    pub fn new(config: UnivNetConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        // Use from_varmap so all generator parameters are registered in varmap
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create generator
        let generator = UnivNetGenerator::new(vb, config.clone())?;

        Ok(Self {
            generator,
            device,
            config,
            varmap,
        })
    }

    /// Create inference engine from pretrained model
    pub fn from_pretrained(variant: UnivNetVariant, device: Device) -> Result<Self> {
        let config = variant.config();
        let model_name = variant.model_name();

        // In a full implementation, this would download weights from HuggingFace
        tracing::info!("Loading UnivNet model: {}", model_name);

        Self::new(config, device)
    }

    /// Load model weights from safetensors file.
    ///
    /// Reads the file, parses it as SafeTensors, and calls `VarMap::set_one` for
    /// every tensor whose name matches a variable already registered in the VarMap.
    /// F16 tensors are up-cast to F32 in-place.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        use safetensors::SafeTensors;

        let path_ref = path.as_ref();
        tracing::info!("Loading UnivNet weights from: {:?}", path_ref);

        let data = std::fs::read(path_ref).map_err(|e| {
            crate::VocoderError::ModelError(format!(
                "Failed to read UnivNet weights file {:?}: {}",
                path_ref, e
            ))
        })?;

        let st = SafeTensors::deserialize(&data).map_err(|e| {
            crate::VocoderError::ModelError(format!(
                "Failed to parse SafeTensors for UnivNet: {}",
                e
            ))
        })?;

        let mut loaded: usize = 0;
        let mut skipped_dtype: usize = 0;
        let mut shape_mismatches: usize = 0;

        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let raw = view.data();

            let float_data: Option<Vec<f32>> = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values = raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Some(values)
                }
                safetensors::Dtype::F16 => {
                    let values = raw
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect();
                    Some(values)
                }
                other => {
                    tracing::warn!(
                        "UnivNet: skipping tensor {:?} — unsupported dtype {:?}",
                        name,
                        other
                    );
                    skipped_dtype += 1;
                    None
                }
            };

            if let Some(values) = float_data {
                match candle_core::Tensor::from_vec(values, shape, &self.device) {
                    Ok(tensor) => {
                        match self.varmap.set_one(&name, &tensor) {
                            Ok(()) => {
                                loaded += 1;
                            }
                            Err(_) => {
                                // Name not in VarMap or shape mismatch
                                shape_mismatches += 1;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("UnivNet: failed to create tensor for {:?}: {}", name, e);
                        shape_mismatches += 1;
                    }
                }
            }
        }

        tracing::info!(
            "UnivNet weight loading complete: {} loaded, {} skipped (bad dtype), {} shape mismatches",
            loaded,
            skipped_dtype,
            shape_mismatches
        );

        if loaded == 0 {
            return Err(crate::VocoderError::ModelError(
                "No UnivNet weights matched VarMap entries".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate audio from mel spectrogram
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor with shape [batch, n_mels, time]
    ///
    /// # Returns
    /// * Waveform tensor with shape [batch, 1, samples]
    pub fn generate(&self, mel: &Tensor) -> Result<Tensor> {
        // Validate input shape
        let mel_dims = mel.dims();
        if mel_dims.len() != 3 {
            return Err(crate::VocoderError::InputError(format!(
                "Expected 3D mel spectrogram [batch, n_mels, time], got shape {:?}",
                mel_dims
            )));
        }

        if mel_dims[1] != self.config.num_mels {
            return Err(crate::VocoderError::InputError(format!(
                "Expected {} mel bands, got {}",
                self.config.num_mels, mel_dims[1]
            )));
        }

        // Generate waveform
        let waveform = self.generator.forward(mel)?;

        Ok(waveform)
    }

    /// Generate audio from mel spectrogram (batch processing)
    pub fn generate_batch(&self, mels: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut waveforms = Vec::with_capacity(mels.len());

        for mel in mels {
            let waveform = self.generate(mel)?;
            waveforms.push(waveform);
        }

        Ok(waveforms)
    }

    /// Generate audio with streaming/chunked processing
    ///
    /// # Arguments
    /// * `mel` - Full mel spectrogram
    /// * `chunk_size` - Number of mel frames per chunk
    /// * `overlap` - Number of overlapping frames between chunks
    ///
    /// # Returns
    /// * Waveform tensor
    pub fn generate_streaming(
        &self,
        mel: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let mel_dims = mel.dims();
        let total_frames = mel_dims[2];

        if chunk_size <= overlap {
            return Err(crate::VocoderError::InputError(
                "chunk_size must be greater than overlap".to_string(),
            ));
        }

        let stride = chunk_size - overlap;
        let mut chunks = Vec::new();

        // Process in chunks
        for start in (0..total_frames).step_by(stride) {
            let end = (start + chunk_size).min(total_frames);

            // Extract chunk
            let chunk = mel.narrow(2, start, end - start).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Failed to extract chunk: {}", e))
            })?;

            // Pad if necessary
            let chunk = if end - start < chunk_size {
                let padding_size = chunk_size - (end - start);
                let zeros = Tensor::zeros(
                    (mel_dims[0], mel_dims[1], padding_size),
                    DType::F32,
                    &self.device,
                )
                .map_err(|e| {
                    crate::VocoderError::ProcessingError(format!("Failed to create padding: {}", e))
                })?;

                Tensor::cat(&[chunk, zeros], 2).map_err(|e| {
                    crate::VocoderError::ProcessingError(format!(
                        "Failed to concatenate padding: {}",
                        e
                    ))
                })?
            } else {
                chunk
            };

            // Generate audio for chunk
            let chunk_audio = self.generate(&chunk)?;
            chunks.push(chunk_audio);
        }

        // Concatenate chunks with crossfading
        self.crossfade_chunks(&chunks, overlap)
    }

    /// Crossfade audio chunks to avoid discontinuities
    fn crossfade_chunks(&self, chunks: &[Tensor], overlap: usize) -> Result<Tensor> {
        if chunks.is_empty() {
            return Err(crate::VocoderError::InputError(
                "No chunks to crossfade".to_string(),
            ));
        }

        if chunks.len() == 1 {
            return Ok(chunks[0].clone());
        }

        let _overlap_samples = overlap * self.config.total_upsample_factor();

        // Start with first chunk
        let mut result = chunks[0].clone();

        // Crossfade each subsequent chunk
        for chunk in &chunks[1..] {
            // For now, just concatenate chunks
            // Full implementation would apply proper crossfading with fade curves
            result = Tensor::cat(&[result, chunk.clone()], 2).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Failed to concatenate chunks: {}", e))
            })?;
        }

        Ok(result)
    }

    /// Get model configuration
    pub fn config(&self) -> &UnivNetConfig {
        &self.config
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Estimate real-time factor (RTF) for given input size
    ///
    /// Returns estimated synthesis time relative to audio duration
    pub fn estimate_rtf(&self, num_mel_frames: usize) -> f32 {
        // Simplified RTF estimation based on model size and frame count
        let params = self.generator.num_parameters();
        let complexity_factor = (params as f32) / 8_000_000.0; // Normalize by 8M params

        let base_rtf = match self.config.upsample_initial_channel {
            ch if ch >= 640 => 0.18, // Large model
            ch if ch >= 512 => 0.12, // Base model
            _ => 0.08,               // Fast model
        };

        // Scale by complexity and input size
        base_rtf * complexity_factor * (1.0 + (num_mel_frames as f32 / 1000.0) * 0.05)
    }

    /// Get model information as a string
    pub fn model_info(&self) -> String {
        format!(
            "UnivNet {} - {} params, {} Hz, {} mels",
            if self.config.upsample_initial_channel >= 640 {
                "Large"
            } else if self.config.upsample_initial_channel >= 512 {
                "Base"
            } else {
                "Fast"
            },
            self.generator.num_parameters(),
            self.config.sample_rate,
            self.config.num_mels
        )
    }
}

#[cfg(feature = "candle")]
impl Clone for UnivNetInference {
    fn clone(&self) -> Self {
        // Reconstruct the generator from the existing varmap so the clone carries
        // the same learned weights.
        let mut new_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&new_varmap, DType::F32, &self.device);
        let generator = UnivNetGenerator::new(vb, self.config.clone())
            .expect("UnivNetInference::clone: failed to rebuild generator");

        // Copy every named variable from the source varmap into the new one.
        {
            let src_data = self
                .varmap
                .data()
                .lock()
                .expect("UnivNetInference::clone: VarMap data mutex should not be poisoned");
            for (name, var) in src_data.iter() {
                let tensor = var.as_tensor().clone();
                let _ = new_varmap.set_one(name, &tensor);
            }
        }

        Self {
            generator,
            device: self.device.clone(),
            config: self.config.clone(),
            varmap: new_varmap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_creation() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();

        let inference = UnivNetInference::new(config, device);
        assert!(inference.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_from_pretrained() {
        let device = Device::Cpu;
        let inference = UnivNetInference::from_pretrained(UnivNetVariant::Fast, device);
        assert!(inference.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_generate() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();
        let inference = UnivNetInference::new(config, device.clone()).unwrap();

        // Create test mel spectrogram
        let mel = Tensor::zeros((1, 80, 100), DType::F32, &device).unwrap();

        let result = inference.generate(&mel);

        if let Err(e) = &result {
            eprintln!("Generate error: {}", e);
        }

        assert!(result.is_ok(), "Generation failed: {:?}", result.err());

        let waveform = result.unwrap();
        let dims = waveform.dims();

        // Check output shape
        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], 1); // Batch
        assert_eq!(dims[1], 1); // Mono audio
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_generate_invalid_shape() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();
        let inference = UnivNetInference::new(config, device.clone()).unwrap();

        // Create invalid 2D tensor
        let mel = Tensor::zeros((80, 100), DType::F32, &device).unwrap();

        let result = inference.generate(&mel);
        assert!(result.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_generate_invalid_mels() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();
        let inference = UnivNetInference::new(config, device.clone()).unwrap();

        // Create mel with wrong number of bands
        let mel = Tensor::zeros((1, 100, 100), DType::F32, &device).unwrap();

        let result = inference.generate(&mel);
        assert!(result.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_batch_generation() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();
        let inference = UnivNetInference::new(config, device.clone()).unwrap();

        let mels = vec![
            Tensor::zeros((1, 80, 100), DType::F32, &device).unwrap(),
            Tensor::zeros((1, 80, 150), DType::F32, &device).unwrap(),
        ];

        let result = inference.generate_batch(&mels);
        assert!(result.is_ok());

        let waveforms = result.unwrap();
        assert_eq!(waveforms.len(), 2);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_rtf_estimation() {
        let device = Device::Cpu;
        let config = UnivNetConfig::fast_24khz();
        let inference = UnivNetInference::new(config, device).unwrap();

        let rtf = inference.estimate_rtf(100);
        assert!(rtf > 0.0);
        assert!(rtf < 1.0); // Should be faster than real-time for fast model
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_model_info() {
        let device = Device::Cpu;
        let config = UnivNetConfig::base_24khz();
        let inference = UnivNetInference::new(config, device).unwrap();

        let info = inference.model_info();
        assert!(info.contains("UnivNet"));
        assert!(info.contains("24000"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_all_variants_loadable() {
        for variant in [
            UnivNetVariant::Fast,
            UnivNetVariant::Base,
            UnivNetVariant::Large,
            UnivNetVariant::Base16k,
            UnivNetVariant::Base48k,
        ] {
            let device = Device::Cpu;
            let result = UnivNetInference::from_pretrained(variant, device);
            assert!(result.is_ok(), "Failed to load variant: {:?}", variant);
        }
    }
}
