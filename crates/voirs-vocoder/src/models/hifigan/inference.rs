//! HiFi-GAN inference implementation for mel-to-audio conversion.

use crate::{Result, VocoderError, MelSpectrogram, AudioBuffer, SynthesisConfig};
use super::{HiFiGanConfig, generator::HiFiGanGenerator};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

/// HiFi-GAN inference engine
#[derive(Debug)]
pub struct HiFiGanInference {
    /// Generator model
    generator: HiFiGanGenerator,
    /// Configuration
    config: HiFiGanConfig,
    /// Preprocessing settings
    preprocess_config: PreprocessConfig,
    /// Postprocessing settings
    postprocess_config: PostprocessConfig,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// Mel spectrogram normalization
    pub normalize_mel: bool,
    /// Mel mean for normalization
    pub mel_mean: f32,
    /// Mel standard deviation for normalization
    pub mel_std: f32,
    /// Padding mode for input
    pub padding_mode: PaddingMode,
    /// Minimum input length
    pub min_length: u32,
}

/// Postprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessConfig {
    /// Apply audio normalization
    pub normalize_audio: bool,
    /// Target audio level (RMS)
    pub target_level: f32,
    /// Apply high-pass filter
    pub apply_highpass: bool,
    /// High-pass filter cutoff frequency
    pub highpass_cutoff: f32,
    /// Apply low-pass filter
    pub apply_lowpass: bool,
    /// Low-pass filter cutoff frequency
    pub lowpass_cutoff: f32,
    /// Apply DC offset removal
    pub remove_dc: bool,
}

/// Padding modes for input preprocessing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingMode {
    /// No padding
    None,
    /// Zero padding
    Zero,
    /// Reflection padding
    Reflect,
    /// Replication padding
    Replicate,
}

/// Inference statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Inference time in milliseconds
    pub inference_time_ms: f32,
    /// Real-time factor
    pub real_time_factor: f32,
    /// Input mel frames
    pub input_frames: u32,
    /// Output audio samples
    pub output_samples: u32,
    /// Sample rate
    pub sample_rate: u32,
    /// Peak audio level
    pub peak_level: f32,
    /// RMS audio level
    pub rms_level: f32,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            normalize_mel: true,
            mel_mean: -4.0,
            mel_std: 4.0,
            padding_mode: PaddingMode::Reflect,
            min_length: 32,
        }
    }
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            normalize_audio: true,
            target_level: -12.0, // dB
            apply_highpass: true,
            highpass_cutoff: 50.0, // Hz
            apply_lowpass: false,
            lowpass_cutoff: 8000.0, // Hz
            remove_dc: true,
        }
    }
}

impl HiFiGanInference {
    /// Create new inference engine
    #[cfg(feature = "candle")]
    pub fn new(
        generator: HiFiGanGenerator,
        config: HiFiGanConfig,
    ) -> Result<Self> {
        Ok(Self {
            generator,
            config,
            preprocess_config: PreprocessConfig::default(),
            postprocess_config: PostprocessConfig::default(),
        })
    }

    /// Create inference engine without Candle (for testing)
    #[cfg(not(feature = "candle"))]
    pub fn new(
        generator: HiFiGanGenerator,
        config: HiFiGanConfig,
    ) -> Result<Self> {
        Ok(Self {
            generator,
            config,
            preprocess_config: PreprocessConfig::default(),
            postprocess_config: PostprocessConfig::default(),
        })
    }

    /// Set preprocessing configuration
    pub fn with_preprocess_config(mut self, config: PreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    /// Set postprocessing configuration
    pub fn with_postprocess_config(mut self, config: PostprocessConfig) -> Self {
        self.postprocess_config = config;
        self
    }

    /// Convert mel spectrogram to audio
    pub async fn infer(
        &self,
        mel: &MelSpectrogram,
        synthesis_config: Option<&SynthesisConfig>,
    ) -> Result<(AudioBuffer, InferenceStats)> {
        let start_time = Instant::now();
        
        // Preprocess mel spectrogram
        let preprocessed_mel = self.preprocess_mel(mel)?;
        
        // Generate audio using the generator
        let raw_audio = self.generate_audio(&preprocessed_mel).await?;
        
        // Apply synthesis configuration
        let modified_audio = if let Some(config) = synthesis_config {
            self.apply_synthesis_config(&raw_audio, config)?
        } else {
            raw_audio
        };
        
        // Postprocess audio
        let final_audio = self.postprocess_audio(&modified_audio)?;
        
        // Calculate inference statistics
        let inference_time = start_time.elapsed().as_millis() as f32;
        let stats = self.calculate_stats(
            mel,
            &final_audio,
            inference_time,
        )?;
        
        Ok((final_audio, stats))
    }

    /// Preprocess mel spectrogram
    fn preprocess_mel(&self, mel: &MelSpectrogram) -> Result<ProcessedMel> {
        let mut data = mel.data.clone();
        
        // Normalize mel spectrogram
        if self.preprocess_config.normalize_mel {
            for frame in &mut data {
                for value in frame {
                    *value = (*value - self.preprocess_config.mel_mean) / self.preprocess_config.mel_std;
                }
            }
        }
        
        // Apply padding if needed
        if mel.n_frames < self.preprocess_config.min_length as usize {
            let padding_needed = self.preprocess_config.min_length as usize - mel.n_frames;
            
            match self.preprocess_config.padding_mode {
                PaddingMode::None => {
                    // No padding
                }
                PaddingMode::Zero => {
                    // Zero padding
                    for mel_channel in &mut data {
                        mel_channel.extend(vec![0.0; padding_needed]);
                    }
                }
                PaddingMode::Reflect => {
                    // Reflection padding
                    for mel_channel in &mut data {
                        let original_len = mel_channel.len();
                        for i in 0..padding_needed {
                            let reflect_idx = original_len - 1 - (i % original_len);
                            mel_channel.push(mel_channel[reflect_idx]);
                        }
                    }
                }
                PaddingMode::Replicate => {
                    // Replication padding
                    for mel_channel in &mut data {
                        let last_value = *mel_channel.last().unwrap_or(&0.0);
                        mel_channel.extend(vec![last_value; padding_needed]);
                    }
                }
            }
        }
        
        let n_frames = data.first().map_or(0, |row| row.len());
        
        Ok(ProcessedMel {
            data,
            n_mels: mel.n_mels,
            n_frames,
            sample_rate: mel.sample_rate,
            hop_length: mel.hop_length,
        })
    }

    /// Generate audio using the generator
    async fn generate_audio(&self, mel: &ProcessedMel) -> Result<AudioBuffer> {
        #[cfg(feature = "candle")]
        {
            // Convert mel to tensor
            let tensor = self.mel_to_tensor(mel)?;
            
            // Forward pass through generator
            let output_tensor = self.generator.forward(&tensor)?;
            
            // Convert tensor to audio buffer
            self.tensor_to_audio_buffer(&output_tensor)
        }
        
        #[cfg(not(feature = "candle"))]
        {
            // Generate dummy audio for testing
            let duration = (mel.n_frames as u32 * mel.hop_length) as f32 / mel.sample_rate as f32;
            let num_samples = (duration * mel.sample_rate as f32) as usize;
            let samples = vec![0.0; num_samples]; // Silence for now
            
            Ok(AudioBuffer::new(samples, mel.sample_rate, 1))
        }
    }

    /// Convert mel to tensor
    #[cfg(feature = "candle")]
    fn mel_to_tensor(&self, mel: &ProcessedMel) -> Result<Tensor> {
        let device = self.generator.device();
        
        // Convert Vec<Vec<f32>> to flattened Vec<f32>
        let mut flattened = Vec::new();
        for frame_idx in 0..mel.n_frames {
            for mel_idx in 0..mel.n_mels {
                flattened.push(mel.data[mel_idx][frame_idx]);
            }
        }
        
        // Create tensor with shape [1, n_mels, n_frames] (batch, channels, time)
        let tensor = Tensor::from_vec(flattened, (1, mel.n_mels, mel.n_frames), device)?;
        
        Ok(tensor)
    }

    /// Convert tensor to audio buffer
    #[cfg(feature = "candle")]
    fn tensor_to_audio_buffer(&self, tensor: &Tensor) -> Result<AudioBuffer> {
        // Squeeze dimensions to get [samples] from [batch, channels, samples]
        // Assume tensor shape is [1, 1, samples]
        let squeezed = tensor.squeeze(0)?.squeeze(0)?; // Remove batch and channel dimensions
        
        // Get tensor data
        let data = squeezed.to_vec1::<f32>()?;
        
        // Create audio buffer
        Ok(AudioBuffer::new(data, self.config.sample_rate, 1))
    }

    /// Apply synthesis configuration
    fn apply_synthesis_config(
        &self,
        audio: &AudioBuffer,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        let mut samples = audio.samples().to_vec();
        
        // Apply speed modification
        if config.speed != 1.0 {
            samples = self.apply_speed_change(&samples, config.speed)?;
        }
        
        // Apply pitch shift
        if config.pitch_shift != 0.0 {
            samples = self.apply_pitch_shift(&samples, config.pitch_shift)?;
        }
        
        // Apply energy scaling
        if config.energy != 1.0 {
            for sample in &mut samples {
                *sample *= config.energy;
            }
        }
        
        Ok(AudioBuffer::new(samples, audio.sample_rate(), 1))
    }

    /// Apply speed change (simplified implementation)
    fn apply_speed_change(&self, samples: &[f32], speed: f32) -> Result<Vec<f32>> {
        if speed <= 0.0 {
            return Err(VocoderError::InputError("Speed must be positive".to_string()));
        }
        
        let new_length = (samples.len() as f32 / speed) as usize;
        let mut result = Vec::with_capacity(new_length);
        
        for i in 0..new_length {
            let original_idx = (i as f32 * speed) as usize;
            if original_idx < samples.len() {
                result.push(samples[original_idx]);
            } else {
                result.push(0.0);
            }
        }
        
        Ok(result)
    }

    /// Apply pitch shift (simplified implementation)
    fn apply_pitch_shift(&self, samples: &[f32], _pitch_shift: f32) -> Result<Vec<f32>> {
        // TODO: Implement proper pitch shifting
        // For now, just return original samples
        Ok(samples.to_vec())
    }

    /// Postprocess audio
    fn postprocess_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let mut samples = audio.samples().to_vec();
        
        // Remove DC offset
        if self.postprocess_config.remove_dc {
            let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
            for sample in &mut samples {
                *sample -= dc_offset;
            }
        }
        
        // Apply high-pass filter (simplified)
        if self.postprocess_config.apply_highpass {
            // TODO: Implement proper high-pass filter
            // For now, just apply DC removal
        }
        
        // Apply low-pass filter (simplified)
        if self.postprocess_config.apply_lowpass {
            // TODO: Implement proper low-pass filter
        }
        
        // Normalize audio using peak normalization to preserve dynamic range
        if self.postprocess_config.normalize_audio {
            let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
            if peak > 0.0 {
                // Use peak normalization instead of RMS to preserve dynamic range
                // Target peak at -6dB to leave headroom and avoid clipping
                let target_peak = 0.5; // -6dB 
                let gain = target_peak / peak;
                for sample in &mut samples {
                    *sample *= gain;
                }
            }
        }
        
        // Clip samples to [-1, 1]
        for sample in &mut samples {
            *sample = sample.clamp(-1.0, 1.0);
        }
        
        Ok(AudioBuffer::new(samples, audio.sample_rate(), 1))
    }

    /// Calculate inference statistics
    fn calculate_stats(
        &self,
        mel: &MelSpectrogram,
        audio: &AudioBuffer,
        inference_time_ms: f32,
    ) -> Result<InferenceStats> {
        let audio_duration = audio.duration();
        let real_time_factor = inference_time_ms / 1000.0 / audio_duration;
        
        let samples = audio.samples();
        let peak_level = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
        let rms_level = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        
        Ok(InferenceStats {
            inference_time_ms,
            real_time_factor,
            input_frames: mel.n_frames as u32,
            output_samples: samples.len() as u32,
            sample_rate: audio.sample_rate(),
            peak_level,
            rms_level,
        })
    }
}

/// Processed mel spectrogram
#[derive(Debug, Clone)]
struct ProcessedMel {
    data: Vec<Vec<f32>>,
    n_mels: usize,
    n_frames: usize,
    sample_rate: u32,
    hop_length: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hifigan::HiFiGanVariants;

    #[test]
    fn test_preprocess_config() {
        let config = PreprocessConfig::default();
        
        assert!(config.normalize_mel);
        assert_eq!(config.mel_mean, -4.0);
        assert_eq!(config.mel_std, 4.0);
        assert_eq!(config.min_length, 32);
    }

    #[test]
    fn test_postprocess_config() {
        let config = PostprocessConfig::default();
        
        assert!(config.normalize_audio);
        assert_eq!(config.target_level, -12.0);
        assert!(config.apply_highpass);
        assert_eq!(config.highpass_cutoff, 50.0);
        assert!(config.remove_dc);
    }

    #[tokio::test]
    async fn test_inference_creation() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            
            let inference = HiFiGanInference::new(generator, hifigan_config);
            assert!(inference.is_ok());
        }
    }

    #[tokio::test]
    async fn test_mel_preprocessing() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            let inference = HiFiGanInference::new(generator, hifigan_config).unwrap();
            
            // Create test mel spectrogram
            let mel_data = vec![vec![1.0; 10]; 80]; // 80 mel channels, 10 frames
            let mel = MelSpectrogram::new(mel_data, 22050, 256);
            
            let processed = inference.preprocess_mel(&mel).unwrap();
            
            // Check that normalization was applied
            assert_eq!(processed.n_mels, 80);
            assert!(processed.n_frames >= 10); // May be padded
            
            // Check normalization (original value 1.0 should be normalized)
            let normalized_value = (1.0 - (-4.0)) / 4.0; // (value - mean) / std
            assert!((processed.data[0][0] - normalized_value).abs() < 0.001);
        }
    }

    #[tokio::test]
    async fn test_synthesis_config_application() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            let inference = HiFiGanInference::new(generator, hifigan_config).unwrap();
            
            let samples = vec![0.5; 1000];
            let audio = AudioBuffer::new(samples, 22050, 1);
            
            let mut config = SynthesisConfig::default();
            config.energy = 2.0;
            
            let modified = inference.apply_synthesis_config(&audio, &config).unwrap();
            
            // Check energy scaling
            assert!((modified.samples()[0] - 1.0).abs() < 0.001);
        }
    }

    #[tokio::test]
    async fn test_speed_change() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            let inference = HiFiGanInference::new(generator, hifigan_config).unwrap();
            
            let samples = vec![0.5; 1000];
            
            // Test speed increase (should shorten audio)
            let faster = inference.apply_speed_change(&samples, 2.0).unwrap();
            assert_eq!(faster.len(), 500);
            
            // Test speed decrease (should lengthen audio)
            let slower = inference.apply_speed_change(&samples, 0.5).unwrap();
            assert_eq!(slower.len(), 2000);
        }
    }

    #[tokio::test]
    async fn test_audio_postprocessing() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            let inference = HiFiGanInference::new(generator, hifigan_config).unwrap();
            
            // Create audio with DC offset
            let samples = vec![0.5; 1000];
            let audio = AudioBuffer::new(samples, 22050, 1);
            
            let processed = inference.postprocess_audio(&audio).unwrap();
            
            // Check that DC was removed
            let dc_offset = processed.samples().iter().sum::<f32>() / processed.samples().len() as f32;
            assert!(dc_offset.abs() < 0.001);
            
            // Check that samples are clipped to [-1, 1]
            for sample in processed.samples() {
                assert!(*sample >= -1.0 && *sample <= 1.0);
            }
        }
    }

    #[test]
    fn test_padding_modes() {
        #[cfg(not(feature = "candle"))]
        {
            let hifigan_config = HiFiGanVariants::v3();
            let generator = HiFiGanGenerator::new(hifigan_config.clone()).unwrap();
            let mut inference = HiFiGanInference::new(generator, hifigan_config).unwrap();
            
            // Test with short mel that needs padding
            let mel_data = vec![vec![1.0, 2.0, 3.0]; 80]; // 80 mel channels, 3 frames
            let mel = MelSpectrogram::new(mel_data, 22050, 256);
            
            // Test zero padding
            inference.preprocess_config.padding_mode = PaddingMode::Zero;
            let processed = inference.preprocess_mel(&mel).unwrap();
            assert_eq!(processed.n_frames, 32); // min_length
            
            // Test reflection padding
            inference.preprocess_config.padding_mode = PaddingMode::Reflect;
            let processed = inference.preprocess_mel(&mel).unwrap();
            assert_eq!(processed.n_frames, 32);
            
            // Test replication padding
            inference.preprocess_config.padding_mode = PaddingMode::Replicate;
            let processed = inference.preprocess_mel(&mel).unwrap();
            assert_eq!(processed.n_frames, 32);
        }
    }
}