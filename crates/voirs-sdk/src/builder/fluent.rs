//! Fluent API methods for pipeline builder.

use crate::{
    config::PipelineConfig,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, QualityLevel, SynthesisConfig},
    voice::DefaultVoiceManager,
};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::RwLock;

use super::{PresetProfile};
use super::builder_impl::VoirsPipelineBuilder;

impl VoirsPipelineBuilder {
    /// Set the voice to use for synthesis
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice_id = Some(voice.into());
        self
    }

    /// Set the language (will auto-select appropriate voice)
    pub fn with_language(mut self, language: LanguageCode) -> Self {
        // Store language preference - will be used during build to select voice
        self.config.default_synthesis.sample_rate = match language {
            LanguageCode::JaJp => 22050, // Japanese typically uses 22kHz
            _ => 16000, // Other languages default to 16kHz
        };
        self.config.default_synthesis.language = language;
        self
    }

    /// Set synthesis quality level
    pub fn with_quality(mut self, quality: QualityLevel) -> Self {
        self.config.default_synthesis.quality = quality;
        self
    }

    /// Enable or disable GPU acceleration
    pub fn with_gpu_acceleration(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        if enabled && self.config.device == "cpu" {
            self.config.device = "cuda".to_string();
        } else if !enabled {
            self.config.device = "cpu".to_string();
        }
        self
    }

    /// Set specific device for computation
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Set number of CPU threads to use
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    /// Set custom cache directory
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.cache_dir = Some(path.into());
        self
    }

    /// Set maximum cache size in MB
    pub fn with_cache_size(mut self, size_mb: u32) -> Self {
        self.config.max_cache_size_mb = size_mb;
        self
    }

    /// Set speaking rate (0.5 - 2.0)
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.config.default_synthesis.speaking_rate = rate;
        self
    }

    /// Set pitch shift in semitones (-12.0 - 12.0)
    pub fn with_pitch_shift(mut self, semitones: f32) -> Self {
        self.config.default_synthesis.pitch_shift = semitones;
        self
    }

    /// Set volume gain in dB (-20.0 - 20.0)
    pub fn with_volume_gain(mut self, gain_db: f32) -> Self {
        self.config.default_synthesis.volume_gain = gain_db;
        self
    }

    /// Enable or disable audio enhancement
    pub fn with_enhancement(mut self, enabled: bool) -> Self {
        self.config.default_synthesis.enable_enhancement = enabled;
        self
    }

    /// Set output sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.default_synthesis.sample_rate = sample_rate;
        self
    }

    /// Set output audio format
    pub fn with_audio_format(mut self, format: crate::types::AudioFormat) -> Self {
        self.config.default_synthesis.output_format = format;
        self
    }

    /// Use custom G2P component
    pub fn with_g2p(mut self, g2p: Arc<dyn G2p>) -> Self {
        self.custom_g2p = Some(g2p);
        self
    }

    /// Use custom acoustic model
    pub fn with_acoustic_model(mut self, acoustic: Arc<dyn AcousticModel>) -> Self {
        self.custom_acoustic = Some(acoustic);
        self
    }

    /// Use custom vocoder
    pub fn with_vocoder(mut self, vocoder: Arc<dyn Vocoder>) -> Self {
        self.custom_vocoder = Some(vocoder);
        self
    }

    /// Use custom voice manager
    pub fn with_voice_manager(mut self, manager: Arc<RwLock<DefaultVoiceManager>>) -> Self {
        self.voice_manager = Some(manager);
        self
    }

    /// Enable or disable validation during build
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }

    /// Enable or disable automatic model downloading
    pub fn with_auto_download(mut self, enabled: bool) -> Self {
        self.auto_download = enabled;
        self.config.model_loading.auto_download = enabled;
        self
    }

    /// Load configuration from file
    pub fn with_config_file(mut self, path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        self.config = PipelineConfig::from_file(path)?;
        Ok(self)
    }

    /// Merge with existing configuration
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Apply configuration overrides
    pub fn with_config_overrides<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut PipelineConfig),
    {
        f(&mut self.config);
        self
    }

    /// Set synthesis configuration overrides
    pub fn with_synthesis_config(mut self, synthesis_config: SynthesisConfig) -> Self {
        self.config.default_synthesis = synthesis_config;
        self
    }

    /// Enable preset configuration profiles
    pub fn with_preset(mut self, preset: PresetProfile) -> Self {
        match preset {
            PresetProfile::HighQuality => {
                self.config.default_synthesis.quality = QualityLevel::Ultra;
                self.config.use_gpu = true;
                self.config.default_synthesis.enable_enhancement = true;
                self.config.default_synthesis.sample_rate = 48000;
            }
            PresetProfile::FastSynthesis => {
                self.config.default_synthesis.quality = QualityLevel::Medium;
                self.config.use_gpu = true;
                self.config.default_synthesis.enable_enhancement = false;
                self.config.default_synthesis.sample_rate = 16000;
            }
            PresetProfile::LowMemory => {
                self.config.default_synthesis.quality = QualityLevel::Low;
                self.config.use_gpu = false;
                self.config.max_cache_size_mb = 256;
                self.config.default_synthesis.sample_rate = 16000;
            }
            PresetProfile::Streaming => {
                self.config.default_synthesis.quality = QualityLevel::Medium;
                self.config.use_gpu = true;
                self.config.audio_processing.buffer_size = 2048;
                self.config.default_synthesis.enable_enhancement = false;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioFormat;
    use crate::builder::builder_impl::VoirsPipelineBuilder;

    #[test]
    fn test_fluent_api_chaining() {
        let builder = VoirsPipelineBuilder::new()
            .with_voice("test-voice")
            .with_quality(QualityLevel::High)
            .with_gpu_acceleration(true)
            .with_threads(4)
            .with_speaking_rate(1.2)
            .with_pitch_shift(2.0)
            .with_volume_gain(3.0)
            .with_enhancement(true)
            .with_sample_rate(22050)
            .with_audio_format(AudioFormat::Wav);

        assert_eq!(builder.voice_id, Some("test-voice".to_string()));
        assert_eq!(builder.config.default_synthesis.quality, QualityLevel::High);
        assert!(builder.config.use_gpu);
        assert_eq!(builder.config.num_threads, Some(4));
        assert_eq!(builder.config.default_synthesis.speaking_rate, 1.2);
        assert_eq!(builder.config.default_synthesis.pitch_shift, 2.0);
        assert_eq!(builder.config.default_synthesis.volume_gain, 3.0);
        assert!(builder.config.default_synthesis.enable_enhancement);
        assert_eq!(builder.config.default_synthesis.sample_rate, 22050);
        assert_eq!(builder.config.default_synthesis.output_format, AudioFormat::Wav);
    }

    #[test]
    fn test_preset_configurations() {
        let high_quality = VoirsPipelineBuilder::new().with_preset(PresetProfile::HighQuality);
        assert_eq!(high_quality.config.default_synthesis.quality, QualityLevel::Ultra);
        assert!(high_quality.config.use_gpu);
        assert!(high_quality.config.default_synthesis.enable_enhancement);
        assert_eq!(high_quality.config.default_synthesis.sample_rate, 48000);

        let fast_synthesis = VoirsPipelineBuilder::new().with_preset(PresetProfile::FastSynthesis);
        assert_eq!(fast_synthesis.config.default_synthesis.quality, QualityLevel::Medium);
        assert!(fast_synthesis.config.use_gpu);
        assert!(!fast_synthesis.config.default_synthesis.enable_enhancement);
        assert_eq!(fast_synthesis.config.default_synthesis.sample_rate, 16000);

        let low_memory = VoirsPipelineBuilder::new().with_preset(PresetProfile::LowMemory);
        assert_eq!(low_memory.config.default_synthesis.quality, QualityLevel::Low);
        assert!(!low_memory.config.use_gpu);
        assert_eq!(low_memory.config.max_cache_size_mb, 256);
        assert_eq!(low_memory.config.default_synthesis.sample_rate, 16000);
    }

    #[test]
    fn test_device_configuration() {
        let cpu_builder = VoirsPipelineBuilder::new().with_gpu_acceleration(false);
        assert!(!cpu_builder.config.use_gpu);
        assert_eq!(cpu_builder.config.device, "cpu");

        let gpu_builder = VoirsPipelineBuilder::new().with_gpu_acceleration(true);
        assert!(gpu_builder.config.use_gpu);
        assert_eq!(gpu_builder.config.device, "cuda");

        let custom_device = VoirsPipelineBuilder::new().with_device("custom-device");
        assert_eq!(custom_device.config.device, "custom-device");
    }

    #[test]
    fn test_language_configuration() {
        let japanese = VoirsPipelineBuilder::new().with_language(LanguageCode::JaJp);
        assert_eq!(japanese.config.default_synthesis.sample_rate, 22050);
        assert_eq!(japanese.config.default_synthesis.language, LanguageCode::JaJp);

        let english = VoirsPipelineBuilder::new().with_language(LanguageCode::EnUs);
        assert_eq!(english.config.default_synthesis.sample_rate, 16000);
        assert_eq!(english.config.default_synthesis.language, LanguageCode::EnUs);
    }
}