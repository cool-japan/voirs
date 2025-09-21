//! Automatic Speech Recognition (ASR) implementations
//!
//! This module provides various ASR model implementations including:
//! - Whisper (`OpenAI`)
//! - `DeepSpeech` (Mozilla)
//! - `Wav2Vec2` (Facebook)
//! - Conformer (Convolution-augmented Transformer)
//!
//! Each implementation follows the common [`ASRModel`] trait interface
//! for consistent usage across different models.

use crate::traits::{ASRModel, RecognitionResult};
use crate::RecognitionError;
use std::sync::Arc;
use voirs_sdk::LanguageCode;

// Model implementations
#[cfg(feature = "whisper")]
pub mod whisper;
// #[cfg(feature = "whisper")]
// pub use whisper::WhisperModel;

// Pure Rust Whisper implementation (no Python dependencies)
#[cfg(feature = "whisper-pure")]
pub mod whisper_pure;
#[cfg(feature = "whisper-pure")]
pub use whisper_pure::PureRustWhisper;

#[cfg(feature = "deepspeech")]
pub mod deepspeech;

// Advanced model optimization
pub mod advanced_optimization;
pub mod attention_optimizations;
pub mod memory_optimizations;
pub mod ondevice_optimizations;
pub mod optimization_integration;
#[cfg(feature = "deepspeech")]
pub use deepspeech::DeepSpeechModel;

#[cfg(feature = "wav2vec2")]
pub mod wav2vec2;
#[cfg(feature = "wav2vec2")]
pub use wav2vec2::Wav2Vec2Model;

// Intelligent fallback mechanism
pub mod intelligent_fallback;
pub use intelligent_fallback::{
    FallbackConfig, FallbackReason, FallbackResult, FallbackStats, IntelligentASRFallback,
    ModelMetrics,
};

// Enhanced intelligent model management
pub mod intelligent_model_manager;
pub use intelligent_model_manager::{
    AudioQualityLevel, IntelligentModelConfig, IntelligentModelManager, ModelContext,
    ProcessingPriority, ResourceStatus, UsageStatistics,
};

// Comprehensive benchmarking suite
pub mod benchmarking_suite;
pub use benchmarking_suite::{
    ASRBenchmarkingSuite, AccuracyRequirement, AccuracyValidationReport, AccuracyValidationResult,
    AccuracyValidator, BenchmarkReport, BenchmarkingConfig, PerformanceBenchmark, WERResult,
};

// Transformer-based end-to-end ASR
#[cfg(feature = "transformer")]
pub mod transformer;
#[cfg(feature = "transformer")]
pub use transformer::{
    create_transformer_asr, create_transformer_asr_with_config, TransformerASR, TransformerConfig,
};

// Conformer: Convolution-augmented Transformer for Speech Recognition
#[cfg(feature = "conformer")]
pub mod conformer;
#[cfg(feature = "conformer")]
pub use conformer::{
    create_conformer_asr, create_conformer_asr_with_config, ConformerConfig, ConformerModel,
};

/// ASR backend enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ASRBackend {
    /// `OpenAI` Whisper
    Whisper {
        /// Model size variant
        model_size: WhisperModelSize,
        /// Model path (optional, will download if not provided)
        model_path: Option<String>,
    },
    /// Mozilla `DeepSpeech`
    DeepSpeech {
        /// Model path
        model_path: String,
        /// Scorer path (optional)
        scorer_path: Option<String>,
    },
    /// Facebook `Wav2Vec2`
    Wav2Vec2 {
        /// Model identifier
        model_id: String,
        /// Model path (optional, will download if not provided)
        model_path: Option<String>,
    },
    /// Transformer-based end-to-end ASR
    #[cfg(feature = "transformer")]
    Transformer {
        /// Model configuration
        config: Option<TransformerConfig>,
    },
    /// Conformer: Convolution-augmented Transformer
    #[cfg(feature = "conformer")]
    Conformer {
        /// Model configuration
        config: Option<ConformerConfig>,
    },
}

impl ASRBackend {
    /// Create a default Whisper backend
    pub fn default_whisper() -> Self {
        Self::Whisper {
            model_size: WhisperModelSize::Base,
            model_path: None,
        }
    }

    /// Create a Whisper backend with specific model size
    pub fn whisper(model_size: WhisperModelSize) -> Self {
        Self::Whisper {
            model_size,
            model_path: None,
        }
    }

    /// Create a DeepSpeech backend
    pub fn deepspeech(model_path: String) -> Self {
        Self::DeepSpeech {
            model_path,
            scorer_path: None,
        }
    }

    /// Create a Wav2Vec2 backend
    pub fn wav2vec2(model_id: String) -> Self {
        Self::Wav2Vec2 {
            model_id,
            model_path: None,
        }
    }

    /// Create a Transformer backend with default configuration
    #[cfg(feature = "transformer")]
    pub fn transformer() -> Self {
        Self::Transformer { config: None }
    }

    /// Create a Transformer backend with custom configuration
    #[cfg(feature = "transformer")]
    pub fn transformer_with_config(config: TransformerConfig) -> Self {
        Self::Transformer {
            config: Some(config),
        }
    }

    /// Create a Conformer backend with default configuration
    #[cfg(feature = "conformer")]
    pub fn conformer() -> Self {
        Self::Conformer { config: None }
    }

    /// Create a Conformer backend with custom configuration
    #[cfg(feature = "conformer")]
    pub fn conformer_with_config(config: ConformerConfig) -> Self {
        Self::Conformer {
            config: Some(config),
        }
    }
}

/// Whisper model size variants
#[derive(Debug, Clone, PartialEq)]
pub enum WhisperModelSize {
    /// Tiny model (~39 MB)
    Tiny,
    /// Base model (~74 MB)
    Base,
    /// Small model (~244 MB)
    Small,
    /// Medium model (~769 MB)
    Medium,
    /// Large model (~1550 MB)
    Large,
    /// Large-v2 model (~1550 MB)
    LargeV2,
    /// Large-v3 model (~1550 MB)
    LargeV3,
}

impl WhisperModelSize {
    /// Get the model name string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            WhisperModelSize::Tiny => "tiny",
            WhisperModelSize::Base => "base",
            WhisperModelSize::Small => "small",
            WhisperModelSize::Medium => "medium",
            WhisperModelSize::Large => "large",
            WhisperModelSize::LargeV2 => "large-v2",
            WhisperModelSize::LargeV3 => "large-v3",
        }
    }

    /// Get approximate model size in MB
    #[must_use]
    pub fn size_mb(&self) -> f32 {
        match self {
            WhisperModelSize::Tiny => 39.0,
            WhisperModelSize::Base => 74.0,
            WhisperModelSize::Small => 244.0,
            WhisperModelSize::Medium => 769.0,
            WhisperModelSize::Large => 1550.0,
            WhisperModelSize::LargeV2 => 1550.0,
            WhisperModelSize::LargeV3 => 1550.0,
        }
    }
}

/// Factory function to create ASR models
pub async fn create_asr_model(backend: ASRBackend) -> RecognitionResult<Arc<dyn ASRModel>> {
    match backend {
        #[cfg(feature = "whisper-pure")]
        ASRBackend::Whisper {
            model_size,
            model_path: _,
        } => {
            let model = PureRustWhisper::new_from_model_size(model_size).await?;
            Ok(Arc::new(model))
        }
        #[cfg(not(feature = "whisper-pure"))]
        ASRBackend::Whisper { .. } => Err(RecognitionError::FeatureNotSupported {
            feature: "whisper-pure".to_string(),
        }
        .into()),

        #[cfg(feature = "deepspeech")]
        ASRBackend::DeepSpeech {
            model_path,
            scorer_path,
        } => {
            let model = DeepSpeechModel::new(model_path, scorer_path).await?;
            Ok(Arc::new(model))
        }
        #[cfg(not(feature = "deepspeech"))]
        ASRBackend::DeepSpeech { .. } => Err(RecognitionError::FeatureNotSupported {
            feature: "deepspeech".to_string(),
        }
        .into()),

        #[cfg(feature = "wav2vec2")]
        ASRBackend::Wav2Vec2 {
            model_id,
            model_path,
        } => {
            let model = Wav2Vec2Model::new(model_id, model_path).await?;
            Ok(Arc::new(model))
        }
        #[cfg(not(feature = "wav2vec2"))]
        ASRBackend::Wav2Vec2 { .. } => Err(RecognitionError::FeatureNotSupported {
            feature: "wav2vec2".to_string(),
        }
        .into()),

        #[cfg(feature = "transformer")]
        ASRBackend::Transformer { config } => {
            let model = if let Some(config) = config {
                create_transformer_asr_with_config(config).await?
            } else {
                create_transformer_asr().await?
            };
            Ok(model)
        }

        #[cfg(feature = "conformer")]
        ASRBackend::Conformer { config } => {
            let model = if let Some(config) = config {
                create_conformer_asr_with_config(config).await?
            } else {
                create_conformer_asr().await?
            };
            Ok(model)
        }
    }
}

/// Get recommended ASR backend for a given language
#[must_use]
pub fn recommended_backend_for_language(language: LanguageCode) -> ASRBackend {
    match language {
        LanguageCode::EnUs | LanguageCode::EnGb => {
            // Whisper works well for English
            ASRBackend::Whisper {
                model_size: WhisperModelSize::Base,
                model_path: None,
            }
        }
        LanguageCode::JaJp | LanguageCode::ZhCn | LanguageCode::KoKr => {
            // Whisper has good multilingual support
            ASRBackend::Whisper {
                model_size: WhisperModelSize::Small,
                model_path: None,
            }
        }
        _ => {
            // Default to Whisper for other languages
            ASRBackend::Whisper {
                model_size: WhisperModelSize::Base,
                model_path: None,
            }
        }
    }
}

/// ASR model comparison utility
pub struct ASRModelComparison {
    /// Model name
    pub name: String,
    /// Supported languages
    pub languages: Vec<LanguageCode>,
    /// Model size in MB
    pub size_mb: f32,
    /// Relative inference speed (1.0 = real-time)
    pub inference_speed: f32,
    /// Average WER across languages
    pub average_wer: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
}

/// Compare different ASR models
#[must_use]
pub fn compare_asr_models() -> Vec<ASRModelComparison> {
    vec![
        ASRModelComparison {
            name: "Whisper Tiny".to_string(),
            languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            size_mb: 39.0,
            inference_speed: 3.0,
            average_wer: 0.08,
            memory_usage_mb: 200.0,
        },
        ASRModelComparison {
            name: "Whisper Base".to_string(),
            languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            size_mb: 74.0,
            inference_speed: 2.0,
            average_wer: 0.06,
            memory_usage_mb: 350.0,
        },
        ASRModelComparison {
            name: "Whisper Small".to_string(),
            languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            size_mb: 244.0,
            inference_speed: 1.2,
            average_wer: 0.04,
            memory_usage_mb: 800.0,
        },
        ASRModelComparison {
            name: "Whisper Medium".to_string(),
            languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            size_mb: 769.0,
            inference_speed: 0.8,
            average_wer: 0.03,
            memory_usage_mb: 1500.0,
        },
        ASRModelComparison {
            name: "Whisper Large".to_string(),
            languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            size_mb: 1550.0,
            inference_speed: 0.5,
            average_wer: 0.025,
            memory_usage_mb: 2500.0,
        },
    ]
}

/// Utility functions for ASR processing
pub mod utils {
    use super::RecognitionResult;
    use voirs_sdk::AudioBuffer;

    /// Preprocess audio for ASR
    pub fn preprocess_audio(audio: &AudioBuffer) -> RecognitionResult<AudioBuffer> {
        // Convert to mono if stereo
        let samples = if audio.channels() > 1 {
            // Simple downmix to mono
            let mut mono_samples = Vec::new();
            let samples = audio.samples();
            for chunk in samples.chunks(audio.channels() as usize) {
                let mono_sample = chunk.iter().sum::<f32>() / chunk.len() as f32;
                mono_samples.push(mono_sample);
            }
            mono_samples
        } else {
            audio.samples().to_vec()
        };

        // Resample to 16kHz if needed
        let target_sample_rate = 16000;
        let resampled_samples = if audio.sample_rate() == target_sample_rate {
            samples
        } else {
            // Simple linear interpolation resampling
            let ratio = audio.sample_rate() as f32 / target_sample_rate as f32;
            let new_length = (samples.len() as f32 / ratio) as usize;
            let mut resampled = Vec::with_capacity(new_length);

            for i in 0..new_length {
                let src_index = i as f32 * ratio;
                let src_index_floor = src_index.floor() as usize;
                let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);

                let frac = src_index - src_index_floor as f32;
                let sample =
                    samples[src_index_floor] * (1.0 - frac) + samples[src_index_ceil] * frac;
                resampled.push(sample);
            }
            resampled
        };

        Ok(AudioBuffer::new(resampled_samples, target_sample_rate, 1))
    }

    /// Normalize audio volume
    pub fn normalize_audio(audio: &AudioBuffer) -> RecognitionResult<AudioBuffer> {
        let samples = audio.samples();
        let max_amplitude = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if max_amplitude > 0.0 {
            let scale = 0.9 / max_amplitude;
            let normalized: Vec<f32> = samples.iter().map(|s| s * scale).collect();
            Ok(AudioBuffer::new(
                normalized,
                audio.sample_rate(),
                audio.channels(),
            ))
        } else {
            Ok(audio.clone())
        }
    }

    /// Apply noise reduction
    pub fn reduce_noise(audio: &AudioBuffer) -> RecognitionResult<AudioBuffer> {
        // Simple spectral subtraction noise reduction
        // This is a basic implementation - more sophisticated methods could be used
        let samples = audio.samples();
        let mut processed = samples.to_vec();

        // Apply a simple high-pass filter to reduce low-frequency noise
        let mut prev_sample = 0.0;
        for sample in &mut processed {
            let filtered = *sample - prev_sample * 0.95;
            prev_sample = *sample;
            *sample = filtered;
        }

        Ok(AudioBuffer::new(
            processed,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Split audio into chunks for processing
    #[must_use]
    pub fn split_audio(audio: &AudioBuffer, chunk_duration_seconds: f32) -> Vec<AudioBuffer> {
        let samples_per_chunk = (audio.sample_rate() as f32 * chunk_duration_seconds) as usize;
        let samples = audio.samples();

        samples
            .chunks(samples_per_chunk)
            .map(|chunk| AudioBuffer::new(chunk.to_vec(), audio.sample_rate(), audio.channels()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_size() {
        assert_eq!(WhisperModelSize::Tiny.as_str(), "tiny");
        assert_eq!(WhisperModelSize::Base.as_str(), "base");
        assert_eq!(WhisperModelSize::Small.as_str(), "small");
        assert_eq!(WhisperModelSize::Medium.as_str(), "medium");
        assert_eq!(WhisperModelSize::Large.as_str(), "large");
        assert_eq!(WhisperModelSize::LargeV2.as_str(), "large-v2");
        assert_eq!(WhisperModelSize::LargeV3.as_str(), "large-v3");
    }

    #[test]
    fn test_model_size_mb() {
        assert_eq!(WhisperModelSize::Tiny.size_mb(), 39.0);
        assert_eq!(WhisperModelSize::Base.size_mb(), 74.0);
        assert_eq!(WhisperModelSize::Small.size_mb(), 244.0);
        assert_eq!(WhisperModelSize::Medium.size_mb(), 769.0);
        assert_eq!(WhisperModelSize::Large.size_mb(), 1550.0);
        assert_eq!(WhisperModelSize::LargeV2.size_mb(), 1550.0);
        assert_eq!(WhisperModelSize::LargeV3.size_mb(), 1550.0);
    }

    #[test]
    fn test_recommended_backend() {
        let backend = recommended_backend_for_language(LanguageCode::EnUs);
        match backend {
            ASRBackend::Whisper { model_size, .. } => {
                assert_eq!(model_size, WhisperModelSize::Base);
            }
            _ => assert!(false, "Expected Whisper backend"),
        }
    }

    #[test]
    fn test_compare_models() {
        let comparisons = compare_asr_models();
        assert!(!comparisons.is_empty());

        for comparison in &comparisons {
            assert!(!comparison.name.is_empty());
            assert!(!comparison.languages.is_empty());
            assert!(comparison.size_mb > 0.0);
            assert!(comparison.inference_speed > 0.0);
            assert!(comparison.average_wer >= 0.0);
            assert!(comparison.memory_usage_mb > 0.0);
        }
    }

    #[test]
    fn test_audio_preprocessing() {
        use voirs_sdk::AudioBuffer;

        // Test stereo to mono conversion
        let stereo_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let stereo_audio = AudioBuffer::new(stereo_samples, 16000, 2);

        let mono_audio = utils::preprocess_audio(&stereo_audio).unwrap();
        assert_eq!(mono_audio.channels(), 1);
        assert_eq!(mono_audio.sample_rate(), 16000);

        // Test normalization
        let loud_samples = vec![2.0, -2.0, 1.5, -1.5];
        let loud_audio = AudioBuffer::new(loud_samples, 16000, 1);

        let normalized = utils::normalize_audio(&loud_audio).unwrap();
        let max_amplitude = normalized
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_amplitude <= 1.0);
    }
}
