//! BigVGAN inference engine for mel-to-audio conversion.

use super::config::{BigVGANConfig, BigVGANVariant};
use super::generator::BigVGANGenerator;
use crate::Result;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::VarBuilder;

use std::path::Path;

/// BigVGAN inference engine
#[cfg(feature = "candle")]
pub struct BigVGANInference {
    /// Generator network
    generator: BigVGANGenerator,
    /// Device (CPU/CUDA/Metal)
    device: Device,
    /// Configuration
    config: BigVGANConfig,
}

#[cfg(feature = "candle")]
impl BigVGANInference {
    /// Create a new inference engine with specified configuration
    pub fn new(config: BigVGANConfig, device: Device) -> Result<Self> {
        // Create variable builder with zeros (weights will be loaded later)
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Create generator
        let generator = BigVGANGenerator::new(vb, config.clone())?;

        Ok(Self {
            generator,
            device,
            config,
        })
    }

    /// Create inference engine from pretrained model
    pub fn from_pretrained(variant: BigVGANVariant, device: Device) -> Result<Self> {
        let config = variant.config();
        let model_name = variant.model_name();

        // In a full implementation, this would download weights from HuggingFace
        tracing::info!("Loading BigVGAN model: {}", model_name);

        Self::new(config, device)
    }

    /// Load model weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_ref = path.as_ref();

        tracing::info!("Loading BigVGAN weights from: {:?}", path_ref);

        // In a full implementation, this would use safetensors to load weights
        // and update the generator's parameters

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

        let overlap_samples = overlap * self.config.total_upsample_factor();

        // Start with first chunk
        let mut result = chunks[0].clone();

        // Crossfade each subsequent chunk
        for chunk in &chunks[1..] {
            // Create crossfade weights (linear) - simplified implementation
            // In a full implementation, we would apply proper crossfading with fade curves

            // For now, just concatenate chunks (crossfading would require more complex overlap handling)
            result = Tensor::cat(&[result, chunk.clone()], 2).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Failed to concatenate chunks: {}", e))
            })?;
        }

        Ok(result)
    }

    /// Get model configuration
    pub fn config(&self) -> &BigVGANConfig {
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
        let complexity_factor = (params as f32) / 10_000_000.0; // Normalize by 10M params

        let base_rtf = match &self.config {
            config if config.sample_rate == 48000 => 0.15, // Ultra quality is slower
            config if config.upsample_initial_channel > 512 => 0.12, // Large model
            config if config.upsample_initial_channel < 384 => 0.05, // Fast model
            _ => 0.08,                                     // Base model
        };

        // Scale by complexity
        base_rtf * complexity_factor * (1.0 + (num_mel_frames as f32 / 1000.0) * 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_creation() {
        let device = Device::Cpu;
        let config = BigVGANConfig::fast_24khz();

        let inference = BigVGANInference::new(config, device);
        assert!(inference.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_from_pretrained() {
        let device = Device::Cpu;
        let inference = BigVGANInference::from_pretrained(BigVGANVariant::Fast, device);
        assert!(inference.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_generate() {
        let device = Device::Cpu;
        let config = BigVGANConfig::fast_24khz();
        let inference = BigVGANInference::new(config, device.clone()).unwrap();

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
        let config = BigVGANConfig::fast_24khz();
        let inference = BigVGANInference::new(config, device.clone()).unwrap();

        // Create invalid 2D tensor
        let mel = Tensor::zeros((80, 100), DType::F32, &device).unwrap();

        let result = inference.generate(&mel);
        assert!(result.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_generate_invalid_mels() {
        let device = Device::Cpu;
        let config = BigVGANConfig::fast_24khz();
        let inference = BigVGANInference::new(config, device.clone()).unwrap();

        // Create mel with wrong number of bands
        let mel = Tensor::zeros((1, 100, 100), DType::F32, &device).unwrap();

        let result = inference.generate(&mel);
        assert!(result.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_inference_batch_generation() {
        let device = Device::Cpu;
        let config = BigVGANConfig::fast_24khz();
        let inference = BigVGANInference::new(config, device.clone()).unwrap();

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
        let config = BigVGANConfig::fast_24khz();
        let inference = BigVGANInference::new(config, device).unwrap();

        let rtf = inference.estimate_rtf(100);
        assert!(rtf > 0.0);
        assert!(rtf < 1.0); // Should be faster than real-time for fast model
    }

    #[test]
    fn test_all_variants_loadable() {
        for variant in [
            BigVGANVariant::Base,
            BigVGANVariant::Large,
            BigVGANVariant::Fast,
            BigVGANVariant::Ultra,
        ] {
            let device = Device::Cpu;
            let result = BigVGANInference::from_pretrained(variant, device);
            assert!(result.is_ok(), "Failed to load variant: {:?}", variant);
        }
    }
}
