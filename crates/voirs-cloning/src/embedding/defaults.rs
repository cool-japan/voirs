//! Default implementations for embedding-related configuration types
//!
//! This module provides sensible default values for all configuration structures
//! used in speaker embedding extraction. These defaults are optimized for general
//! use cases and can be customized as needed.

use super::types::*;

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            window_size: 1024,
            hop_size: 512,
            fft_size: 1024,
            num_mel_filters: 80,
            network_architecture: NetworkArchitecture::CNN {
                conv_layers: vec![(32, 3), (64, 3), (128, 3)],
                fc_layers: vec![256, 128],
            },
            feature_method: FeatureExtractionMethod::MelSpectrogram,
            preprocessing: PreprocessingConfig::default(),
            batch_size: 8,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            vad_enabled: true,
            noise_reduction: false,
            normalization: NormalizationMethod::ZScore,
            augmentation: None,
        }
    }
}

impl Default for EmbeddingMetadata {
    fn default() -> Self {
        Self {
            gender: None,
            age_estimate: None,
            language: None,
            emotion: None,
            voice_quality: VoiceQuality::default(),
            extraction_time: None,
        }
    }
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            f0_mean: 0.0,
            f0_std: 0.0,
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            jitter: 0.0,
            shimmer: 0.0,
            energy_mean: 0.0,
            energy_std: 0.0,
        }
    }
}

impl Default for SpeakerEmbeddingExtractor {
    fn default() -> Self {
        Self::new(EmbeddingConfig::default())
            .expect("Failed to create default SpeakerEmbeddingExtractor")
    }
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.01,
            decay_factor: 0.99,
            min_learning_rate: 0.001,
            convergence_threshold: 0.95,
            max_steps: 100,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window_duration: 1.0, // 1 second windows
            hop_duration: 0.5,    // 0.5 second hops (50% overlap)
            temporal_smoothing: true,
            smoothing_factor: 0.3,
            aggregation_method: AggregationMethod::Weighted,
            min_confidence_threshold: 0.5,
        }
    }
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            base_refinement_weight: 0.1,
            quality_amplification: 2.0,
            max_refinement_weight: 0.5,
            convergence_threshold: 0.001,
            convergence_window: 3,
        }
    }
}
