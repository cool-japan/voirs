//! Core traits for acoustic models
//!
//! This module contains the fundamental traits that define the interface
//! for acoustic models in the VoiRS system.

use async_trait::async_trait;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::{Result, AcousticError, LanguageCode, Phoneme, MelSpectrogram, SynthesisConfig};

/// Features supported by acoustic models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcousticModelFeature {
    /// Multi-speaker support
    MultiSpeaker,
    /// Real-time streaming inference
    StreamingInference,
    /// Streaming synthesis (alias for StreamingInference)
    StreamingSynthesis,
    /// Batch processing
    BatchProcessing,
    /// Controllable prosody
    ProsodyControl,
    /// Style transfer
    StyleTransfer,
    /// GPU acceleration support
    GpuAcceleration,
    /// Voice cloning capability
    VoiceCloning,
    /// Real-time inference optimization
    RealTimeInference,
}

/// Acoustic model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture
    pub architecture: String,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of mel channels
    pub mel_channels: u32,
    /// Whether model supports multiple speakers
    pub is_multi_speaker: bool,
    /// Number of speakers (if multi-speaker)
    pub speaker_count: Option<u32>,
}

/// Main trait for acoustic models that convert phonemes to mel spectrograms
#[async_trait]
pub trait AcousticModel: Send + Sync {
    /// Synthesize mel spectrogram from phonemes
    ///
    /// # Arguments
    /// * `phonemes` - Input phoneme sequence
    /// * `config` - Optional synthesis configuration
    ///
    /// # Returns
    /// A mel spectrogram representing the synthesized speech
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram>;
    
    /// Batch synthesis for multiple phoneme sequences
    ///
    /// # Arguments
    /// * `inputs` - Array of phoneme sequences
    /// * `configs` - Optional array of synthesis configurations
    ///
    /// # Returns
    /// Vector of mel spectrograms for each input sequence
    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>>;
    
    /// Get model metadata
    fn metadata(&self) -> AcousticModelMetadata;
    
    /// Check if model supports a feature
    fn supports(&self, feature: AcousticModelFeature) -> bool;
    
    /// Set speaker for multi-speaker models
    ///
    /// # Arguments
    /// * `speaker_id` - Speaker ID to use, None for default speaker
    async fn set_speaker(&mut self, speaker_id: Option<u32>) -> Result<()> {
        let _ = speaker_id;
        Ok(())
    }
}

/// Trait for loading and managing acoustic models
pub trait ModelLoader: Send + Sync {
    /// Load model from a file path
    async fn load_from_file(&self, path: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// Load model from HuggingFace Hub
    async fn load_from_hub(&self, repo_id: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// List available models in a directory
    fn list_models(&self, directory: &str) -> Result<Vec<String>>;
}

/// Trait for backend implementations (Candle, ONNX, etc.)
pub trait Backend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;
    
    /// Check if GPU acceleration is available
    fn supports_gpu(&self) -> bool;
    
    /// Get available devices
    fn available_devices(&self) -> Vec<String>;
    
    /// Create model instance
    async fn create_model(&self, model_path: &str) -> Result<Box<dyn AcousticModel>>;
}

/// Trait for mel spectrogram computation
pub trait MelComputation: Send + Sync {
    /// Compute mel spectrogram from audio
    fn compute_mel(&self, audio: &[f32], sample_rate: u32) -> Result<MelSpectrogram>;
    
    /// Compute mel spectrogram with custom parameters
    fn compute_mel_with_params(
        &self,
        audio: &[f32],
        sample_rate: u32,
        n_mels: u32,
        hop_length: u32,
        win_length: u32,
    ) -> Result<MelSpectrogram>;
    
    /// Convert mel spectrogram back to audio (vocoder)
    fn mel_to_audio(&self, mel: &MelSpectrogram) -> Result<Vec<f32>>;
}

/// Trait for prosody control
pub trait ProsodyController: Send + Sync {
    /// Apply pitch modification
    fn modify_pitch(&self, mel: &mut MelSpectrogram, pitch_shift: f32) -> Result<()>;
    
    /// Apply duration modification
    fn modify_duration(&self, mel: &mut MelSpectrogram, speed_factor: f32) -> Result<()>;
    
    /// Apply energy modification
    fn modify_energy(&self, mel: &mut MelSpectrogram, energy_factor: f32) -> Result<()>;
    
    /// Apply comprehensive prosody control
    fn apply_prosody(
        &self,
        mel: &mut MelSpectrogram,
        config: &SynthesisConfig,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acoustic_model_feature() {
        assert_eq!(
            AcousticModelFeature::MultiSpeaker,
            AcousticModelFeature::MultiSpeaker
        );
        assert_ne!(
            AcousticModelFeature::MultiSpeaker,
            AcousticModelFeature::StreamingInference
        );
    }

    #[test]
    fn test_acoustic_model_metadata() {
        let metadata = AcousticModelMetadata {
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            architecture: "Test".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            sample_rate: 22050,
            mel_channels: 80,
            is_multi_speaker: false,
            speaker_count: None,
        };
        
        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.sample_rate, 22050);
        assert!(!metadata.is_multi_speaker);
    }
}