//! Core traits for VoiRS components.

use crate::{
    error::Result,
    types::{LanguageCode, MelSpectrogram, Phoneme, SynthesisConfig},
    AudioBuffer,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for Grapheme-to-Phoneme (G2P) conversion
#[async_trait]
pub trait G2p: Send + Sync {
    /// Convert text to phonemes for given language
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>>;
    
    /// Get list of supported language codes
    fn supported_languages(&self) -> Vec<LanguageCode>;
    
    /// Get backend metadata and capabilities
    fn metadata(&self) -> G2pMetadata;
    
    /// Preprocess text before phoneme conversion
    async fn preprocess(&self, text: &str, lang: Option<LanguageCode>) -> Result<String> {
        // Default implementation: return text as-is
        let _ = lang; // Suppress unused parameter warning
        Ok(text.to_string())
    }
    
    /// Detect language of input text
    async fn detect_language(&self, text: &str) -> Result<LanguageCode> {
        // Default implementation: return first supported language
        let _ = text; // Suppress unused parameter warning
        self.supported_languages()
            .first()
            .copied()
            .ok_or_else(|| crate::VoirsError::g2p_error("No supported languages"))
    }
}

/// G2P backend metadata
#[derive(Debug, Clone)]
pub struct G2pMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub supported_languages: Vec<LanguageCode>,
    pub accuracy_scores: HashMap<LanguageCode, f32>,
}

/// Trait for acoustic models (text/phonemes to mel spectrogram)
#[async_trait]
pub trait AcousticModel: Send + Sync {
    /// Generate mel spectrogram from phonemes
    async fn synthesize(
        &self, 
        phonemes: &[Phoneme], 
        config: Option<&SynthesisConfig>
    ) -> Result<MelSpectrogram>;
    
    /// Batch synthesis for multiple inputs
    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>
    ) -> Result<Vec<MelSpectrogram>>;
    
    /// Get model metadata and capabilities
    fn metadata(&self) -> AcousticModelMetadata;
    
    /// Check if model supports specific features
    fn supports(&self, feature: AcousticModelFeature) -> bool;
    
    /// Set speaker ID for multi-speaker models
    async fn set_speaker(&mut self, speaker_id: Option<u32>) -> Result<()> {
        let _ = speaker_id; // Suppress unused parameter warning
        Ok(()) // Default: no-op for single speaker models
    }
}

/// Acoustic model metadata
#[derive(Debug, Clone)]
pub struct AcousticModelMetadata {
    pub name: String,
    pub version: String,
    pub architecture: String, // e.g., "VITS", "FastSpeech2"
    pub supported_languages: Vec<LanguageCode>,
    pub sample_rate: u32,
    pub mel_channels: u32,
    pub is_multi_speaker: bool,
    pub speaker_count: Option<u32>,
}

/// Acoustic model features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticModelFeature {
    MultiSpeaker,
    EmotionControl,
    ProsodyControl,
    StreamingInference,
    BatchProcessing,
    GpuAcceleration,
}

/// Trait for vocoders (mel spectrogram to audio)
#[async_trait]
pub trait Vocoder: Send + Sync {
    /// Convert mel spectrogram to audio
    async fn vocode(
        &self, 
        mel: &MelSpectrogram, 
        config: Option<&SynthesisConfig>
    ) -> Result<AudioBuffer>;
    
    /// Stream-based vocoding for real-time synthesis
    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn futures::Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>
    ) -> Result<Box<dyn futures::Stream<Item = Result<AudioBuffer>> + Send + Unpin>>;
    
    /// Batch vocoding for multiple inputs
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>
    ) -> Result<Vec<AudioBuffer>>;
    
    /// Get vocoder metadata and capabilities
    fn metadata(&self) -> VocoderMetadata;
    
    /// Check if vocoder supports specific features
    fn supports(&self, feature: VocoderFeature) -> bool;
}

/// Vocoder metadata
#[derive(Debug, Clone)]
pub struct VocoderMetadata {
    pub name: String,
    pub version: String,
    pub architecture: String, // e.g., "HiFi-GAN", "DiffWave"
    pub sample_rate: u32,
    pub mel_channels: u32,
    pub latency_ms: f32,
    pub quality_score: f32, // MOS or similar metric
}

/// Vocoder features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocoderFeature {
    StreamingInference,
    BatchProcessing,
    GpuAcceleration,
    MultiSampleRate,
    EnhancementFilters,
    RealtimeProcessing,
}

/// Trait for text preprocessing
#[async_trait]
pub trait TextProcessor: Send + Sync {
    /// Preprocess text before G2P conversion
    async fn process(&self, text: &str, lang: LanguageCode) -> Result<String>;
    
    /// Normalize text (unicode, case, etc.)
    fn normalize(&self, text: &str) -> Result<String>;
    
    /// Expand abbreviations and numbers
    fn expand(&self, text: &str, lang: LanguageCode) -> Result<String>;
    
    /// Clean and validate text
    fn clean(&self, text: &str) -> Result<String>;
}

/// Trait for audio post-processing
#[async_trait]
pub trait AudioProcessor: Send + Sync {
    /// Process audio buffer with effects
    async fn process(&self, audio: &AudioBuffer) -> Result<AudioBuffer>;
    
    /// Get processor metadata
    fn metadata(&self) -> AudioProcessorMetadata;
}

/// Audio processor metadata
#[derive(Debug, Clone)]
pub struct AudioProcessorMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub processing_time_ms: f32,
}

/// Trait for voice management
#[async_trait]
pub trait VoiceManager: Send + Sync {
    /// List available voices
    async fn list_voices(&self) -> Result<Vec<crate::types::VoiceConfig>>;
    
    /// Get voice by ID
    async fn get_voice(&self, voice_id: &str) -> Result<Option<crate::types::VoiceConfig>>;
    
    /// Download voice if not available
    async fn download_voice(&self, voice_id: &str) -> Result<()>;
    
    /// Check if voice is available locally
    fn is_voice_available(&self, voice_id: &str) -> bool;
    
    /// Get default voice for language
    fn default_voice_for_language(&self, lang: LanguageCode) -> Option<String>;
}

/// Trait for model caching
#[async_trait]
pub trait ModelCache: Send + Sync {
    /// Get cached model as Any
    async fn get_any(&self, key: &str) -> Result<Option<Box<dyn std::any::Any + Send + Sync>>>;
    
    /// Store model in cache as Any
    async fn put_any(&self, key: &str, value: Box<dyn std::any::Any + Send + Sync>) -> Result<()>;
    
    /// Remove model from cache
    async fn remove(&self, key: &str) -> Result<()>;
    
    /// Clear entire cache
    async fn clear(&self) -> Result<()>;
    
    /// Get cache statistics
    fn stats(&self) -> CacheStats;
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub total_entries: usize,
    pub memory_usage_bytes: usize,
    pub hit_rate: f32,
    pub miss_rate: f32,
}

/// Trait for plugin system
pub trait Plugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;
    
    /// Get plugin version
    fn version(&self) -> &str;
    
    /// Initialize plugin
    fn initialize(&self, _config: &crate::plugins::PluginConfig) -> Result<()> {
        Ok(()) // Default: no initialization needed
    }
    
    /// Shutdown plugin
    fn shutdown(&self) -> Result<()> {
        Ok(()) // Default: no cleanup needed
    }
}

/// Trait for audio effects plugins
#[async_trait]
pub trait AudioEffectPlugin: Plugin {
    /// Process audio with effect
    async fn process(&self, audio: &AudioBuffer) -> Result<AudioBuffer>;
    
    /// Get effect parameters
    fn parameters(&self) -> HashMap<String, f32>;
    
    /// Set effect parameter
    fn set_parameter(&mut self, name: &str, value: f32) -> Result<()>;
}