//! Acoustic model definitions and management.

use crate::{
    traits::{AcousticModel, AcousticModelFeature, AcousticModelMetadata},
    AcousticError, LanguageCode, MelSpectrogram, Phoneme, Result, SynthesisConfig,
};
use async_trait::async_trait;

/// Dummy acoustic model for testing and development
///
/// This model generates synthetic mel spectrograms from phoneme sequences
/// to enable end-to-end testing of the VoiRS pipeline without requiring
/// trained neural network models.
pub struct DummyAcousticModel {
    /// Model configuration
    config: DummyAcousticConfig,
    /// Random seed for reproducible output
    seed: Option<u64>,
}

/// Configuration for dummy acoustic model
#[derive(Debug, Clone)]
pub struct DummyAcousticConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of mel channels
    pub mel_channels: u32,
    /// Hop length in samples
    pub hop_length: u32,
    /// Average duration per phoneme in milliseconds
    pub phoneme_duration_ms: f32,
}

impl Default for DummyAcousticConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            mel_channels: 80,
            hop_length: 256,
            phoneme_duration_ms: 100.0, // 100ms per phoneme
        }
    }
}

impl DummyAcousticModel {
    /// Create new dummy acoustic model
    pub fn new() -> Self {
        Self {
            config: DummyAcousticConfig::default(),
            seed: None,
        }
    }

    /// Create dummy model with custom configuration
    pub fn with_config(config: DummyAcousticConfig) -> Self {
        Self { config, seed: None }
    }

    /// Set random seed for reproducible output
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate mel spectrogram data based on phoneme count and speed factor
    fn generate_mel_data(&self, phoneme_count: usize, speed_factor: f32) -> Vec<Vec<f32>> {
        let frames_per_phoneme = (self.config.phoneme_duration_ms / speed_factor / 1000.0
            * self.config.sample_rate as f32
            / self.config.hop_length as f32) as usize;

        let total_frames = std::cmp::max(1, phoneme_count * frames_per_phoneme);
        let mut data = vec![vec![0.0; total_frames]; self.config.mel_channels as usize];

        // Generate synthetic mel features
        for (mel_idx, channel) in data.iter_mut().enumerate() {
            for (frame_idx, frame_value) in channel.iter_mut().enumerate() {
                // Create some variation based on mel channel and frame position
                let base_value = (mel_idx as f32 / self.config.mel_channels as f32) * 2.0 - 1.0;
                let time_variation = (frame_idx as f32 / total_frames as f32) * 0.5;
                let noise = (fastrand::f32() - 0.5) * 0.2; // Small random component

                *frame_value = base_value + time_variation + noise;
            }
        }

        data
    }
}

impl Default for DummyAcousticModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AcousticModel for DummyAcousticModel {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        // Return error for empty phoneme sequences
        if phonemes.is_empty() {
            return Err(AcousticError::InputError(
                "Cannot synthesize empty phoneme sequence".to_string(),
            ));
        }

        let phoneme_count = phonemes.len();

        // Apply synthesis config if provided
        let speed_factor = config.map(|c| c.speed).unwrap_or(1.0);

        // Generate mel spectrogram data with speed factor applied
        let data = self.generate_mel_data(phoneme_count, speed_factor);

        tracing::debug!(
            "DummyAcousticModel: Generated mel spectrogram for {} phonemes ({} frames, {} mels)",
            phonemes.len(),
            data.first().map(|row| row.len()).unwrap_or(0),
            data.len()
        );

        Ok(MelSpectrogram::new(
            data,
            self.config.sample_rate,
            self.config.hop_length,
        ))
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        let mut results = Vec::with_capacity(inputs.len());

        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            let mel = self.synthesize(phonemes, config).await?;
            results.push(mel);
        }

        Ok(results)
    }

    fn metadata(&self) -> AcousticModelMetadata {
        AcousticModelMetadata {
            name: "Dummy Acoustic Model".to_string(),
            version: "0.1.0".to_string(),
            architecture: "Dummy".to_string(),
            supported_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
            ],
            sample_rate: self.config.sample_rate,
            mel_channels: self.config.mel_channels,
            is_multi_speaker: false,
            speaker_count: None,
        }
    }

    fn supports(&self, feature: AcousticModelFeature) -> bool {
        matches!(
            feature,
            AcousticModelFeature::BatchProcessing | AcousticModelFeature::ProsodyControl
        )
    }
}

/// Model loading functionality
pub struct ModelLoader {
    /// Base directory for model files
    #[allow(dead_code)]
    model_dir: String,
}

impl ModelLoader {
    /// Create new model loader
    pub fn new(model_dir: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
        }
    }

    /// Load dummy model (always succeeds)
    pub fn load_dummy(&self) -> Result<DummyAcousticModel> {
        Ok(DummyAcousticModel::new())
    }

    /// List available models in directory
    pub fn list_models(&self) -> Result<Vec<String>> {
        // For now, just return the dummy model
        Ok(vec!["dummy".to_string()])
    }
}
