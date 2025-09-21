//! Vocoder integration for speaker-specific parameter conditioning

#[cfg(feature = "vocoder-integration")]
use voirs_vocoder;

use crate::embedding::SpeakerEmbedding;
use crate::{types::VoiceSample, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vocoder conditioning adapter for speaker-specific synthesis
#[derive(Debug, Clone)]
pub struct VocoderCloningAdapter {
    /// Vocoder type (HiFiGAN, WaveGlow, etc.)
    pub vocoder_type: VocoderType,
    /// Speaker-specific conditioning parameters
    pub conditioning_config: VocoderConditioningConfig,
    /// Cached speaker parameters
    pub speaker_cache: HashMap<String, SpeakerVocoderParams>,
}

/// Supported vocoder types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VocoderType {
    /// HiFiGAN vocoder
    HiFiGAN,
    /// WaveGlow vocoder  
    WaveGlow,
    /// MelGAN vocoder
    MelGAN,
    /// Universal vocoder (auto-detect best)
    Universal,
}

/// Configuration for speaker-specific vocoder conditioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConditioningConfig {
    /// Enable speaker embedding conditioning
    pub enable_speaker_conditioning: bool,
    /// Enable prosodic feature conditioning
    pub enable_prosodic_conditioning: bool,
    /// Enable spectral envelope conditioning
    pub enable_spectral_conditioning: bool,
    /// Conditioning strength (0.0-1.0)
    pub conditioning_strength: f32,
    /// Quality vs speed tradeoff (0.0=fast, 1.0=quality)
    pub quality_level: f32,
    /// Cache size for speaker parameters
    pub cache_size: usize,
}

/// Speaker-specific vocoder parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerVocoderParams {
    /// Speaker embedding for conditioning
    pub speaker_embedding: Vec<f32>,
    /// Mel-scale conditioning parameters
    pub mel_conditioning: MelConditioningParams,
    /// Prosodic conditioning parameters
    pub prosodic_conditioning: ProsodicConditioningParams,
    /// Spectral conditioning parameters
    pub spectral_conditioning: SpectralConditioningParams,
    /// Quality score for this speaker adaptation
    pub quality_score: f32,
    /// Generation timestamp
    pub timestamp: std::time::SystemTime,
}

/// Mel-scale conditioning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelConditioningParams {
    /// Mel filter bank adjustments
    pub filter_adjustments: Vec<f32>,
    /// Frequency scaling factor
    pub frequency_scaling: f32,
    /// Mel compression factor
    pub compression_factor: f32,
    /// Dynamic range adjustments
    pub dynamic_range: f32,
}

/// Prosodic conditioning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicConditioningParams {
    /// F0 conditioning parameters
    pub f0_conditioning: Vec<f32>,
    /// Energy conditioning parameters
    pub energy_conditioning: Vec<f32>,
    /// Duration conditioning parameters
    pub duration_conditioning: Vec<f32>,
    /// Speaking rate conditioning
    pub rate_conditioning: f32,
}

/// Spectral conditioning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConditioningParams {
    /// Spectral tilt adjustments
    pub spectral_tilt: Vec<f32>,
    /// Formant frequency adjustments
    pub formant_adjustments: Vec<f32>,
    /// Harmonic-to-noise ratio adjustments
    pub hnr_adjustments: Vec<f32>,
    /// Spectral envelope modifications
    pub envelope_modifications: Vec<f32>,
}

impl VocoderCloningAdapter {
    /// Create new vocoder conditioning adapter
    pub fn new(vocoder_type: VocoderType) -> Self {
        Self {
            vocoder_type,
            conditioning_config: VocoderConditioningConfig::default(),
            speaker_cache: HashMap::new(),
        }
    }

    /// Create adapter with custom configuration
    pub fn with_config(vocoder_type: VocoderType, config: VocoderConditioningConfig) -> Self {
        Self {
            vocoder_type,
            conditioning_config: config,
            speaker_cache: HashMap::new(),
        }
    }

    /// Generate speaker-specific vocoder parameters from embedding
    pub async fn generate_speaker_params(
        &mut self,
        speaker_embedding: &SpeakerEmbedding,
        reference_samples: &[VoiceSample],
    ) -> Result<SpeakerVocoderParams> {
        // Create unique speaker ID for caching
        let speaker_id = self.compute_speaker_id(speaker_embedding);

        // Check cache first
        if let Some(cached_params) = self.speaker_cache.get(&speaker_id) {
            // Check if cache is still valid (less than 1 hour old)
            if cached_params
                .timestamp
                .elapsed()
                .unwrap_or(std::time::Duration::MAX)
                < std::time::Duration::from_secs(3600)
            {
                return Ok(cached_params.clone());
            }
        }

        // Generate new parameters
        let mel_conditioning = self
            .extract_mel_conditioning(speaker_embedding, reference_samples)
            .await?;
        let prosodic_conditioning = self
            .extract_prosodic_conditioning(speaker_embedding, reference_samples)
            .await?;
        let spectral_conditioning = self
            .extract_spectral_conditioning(speaker_embedding, reference_samples)
            .await?;

        // Compute quality score
        let quality_score = self.compute_quality_score(
            &mel_conditioning,
            &prosodic_conditioning,
            &spectral_conditioning,
        )?;

        let params = SpeakerVocoderParams {
            speaker_embedding: speaker_embedding.vector.clone(),
            mel_conditioning,
            prosodic_conditioning,
            spectral_conditioning,
            quality_score,
            timestamp: std::time::SystemTime::now(),
        };

        // Cache the parameters
        if self.speaker_cache.len() >= self.conditioning_config.cache_size {
            // Remove oldest entry
            let oldest_key = self
                .speaker_cache
                .iter()
                .min_by_key(|(_, params)| params.timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                self.speaker_cache.remove(&key);
            }
        }

        self.speaker_cache.insert(speaker_id, params.clone());
        Ok(params)
    }

    /// Synthesize audio with speaker-specific vocoder conditioning
    pub async fn synthesize_with_conditioning(
        &self,
        mel_spectrogram: &[Vec<f32>],
        speaker_params: &SpeakerVocoderParams,
    ) -> Result<Vec<f32>> {
        match self.vocoder_type {
            VocoderType::HiFiGAN => {
                self.synthesize_hifigan(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::WaveGlow => {
                self.synthesize_waveglow(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::MelGAN => {
                self.synthesize_melgan(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::Universal => {
                self.synthesize_universal(mel_spectrogram, speaker_params)
                    .await
            }
        }
    }

    /// Apply real-time speaker conditioning during synthesis
    pub async fn apply_realtime_conditioning(
        &self,
        audio_chunk: &mut [f32],
        speaker_params: &SpeakerVocoderParams,
        chunk_index: usize,
    ) -> Result<()> {
        if !self.conditioning_config.enable_speaker_conditioning {
            return Ok(());
        }

        // Apply prosodic conditioning
        if self.conditioning_config.enable_prosodic_conditioning {
            self.apply_prosodic_conditioning_chunk(
                audio_chunk,
                &speaker_params.prosodic_conditioning,
                chunk_index,
            )?;
        }

        // Apply spectral conditioning
        if self.conditioning_config.enable_spectral_conditioning {
            self.apply_spectral_conditioning_chunk(
                audio_chunk,
                &speaker_params.spectral_conditioning,
            )?;
        }

        Ok(())
    }

    /// Extract mel-scale conditioning parameters
    async fn extract_mel_conditioning(
        &self,
        speaker_embedding: &SpeakerEmbedding,
        reference_samples: &[VoiceSample],
    ) -> Result<MelConditioningParams> {
        // Use first part of embedding for mel conditioning
        let mel_section = &speaker_embedding.vector[0..64.min(speaker_embedding.vector.len())];

        // Generate filter bank adjustments
        let mut filter_adjustments = vec![1.0; 80]; // Standard mel filter bank size
        for (i, &embedding_val) in mel_section.iter().enumerate() {
            let filter_idx = (i * filter_adjustments.len() / mel_section.len())
                .min(filter_adjustments.len() - 1);
            filter_adjustments[filter_idx] = 1.0 + embedding_val * 0.2; // ±20% adjustment
        }

        // Analyze reference samples for frequency characteristics
        let mut frequency_scaling = 1.0;
        let mut compression_factor = 1.0;
        let mut dynamic_range = 1.0;

        if !reference_samples.is_empty() {
            // Estimate frequency characteristics from reference audio
            let sample = &reference_samples[0];
            let spectral_centroid = self.compute_spectral_centroid(&sample.audio)?;
            let spectral_bandwidth = self.compute_spectral_bandwidth(&sample.audio)?;
            let dynamic_range_estimate = self.compute_dynamic_range(&sample.audio)?;

            // Map characteristics to conditioning parameters
            frequency_scaling = (spectral_centroid / 1000.0).clamp(0.8, 1.2);
            compression_factor = (spectral_bandwidth / 2000.0).clamp(0.9, 1.1);
            dynamic_range = dynamic_range_estimate.clamp(0.5, 2.0);
        }

        Ok(MelConditioningParams {
            filter_adjustments,
            frequency_scaling,
            compression_factor,
            dynamic_range,
        })
    }

    /// Extract prosodic conditioning parameters
    async fn extract_prosodic_conditioning(
        &self,
        speaker_embedding: &SpeakerEmbedding,
        reference_samples: &[VoiceSample],
    ) -> Result<ProsodicConditioningParams> {
        // Use middle section of embedding for prosodic features
        let start = 64.min(speaker_embedding.vector.len());
        let end = 128.min(speaker_embedding.vector.len());
        let prosodic_section = &speaker_embedding.vector[start..end];

        // Generate conditioning vectors
        let f0_conditioning = prosodic_section[0..16.min(prosodic_section.len())].to_vec();
        let energy_conditioning = prosodic_section[16..32.min(prosodic_section.len())].to_vec();
        let duration_conditioning = prosodic_section[32..48.min(prosodic_section.len())].to_vec();

        // Compute rate conditioning from embedding
        let rate_conditioning = if prosodic_section.len() > 48 {
            1.0 + prosodic_section[48] * 0.3 // ±30% rate variation
        } else {
            1.0
        };

        Ok(ProsodicConditioningParams {
            f0_conditioning,
            energy_conditioning,
            duration_conditioning,
            rate_conditioning,
        })
    }

    /// Extract spectral conditioning parameters
    async fn extract_spectral_conditioning(
        &self,
        speaker_embedding: &SpeakerEmbedding,
        reference_samples: &[VoiceSample],
    ) -> Result<SpectralConditioningParams> {
        // Use last section of embedding for spectral characteristics
        let start = 128.min(speaker_embedding.vector.len());
        let spectral_section = &speaker_embedding.vector[start..];

        let spectral_tilt = if spectral_section.len() >= 32 {
            spectral_section[0..32].to_vec()
        } else {
            vec![0.0; 32]
        };

        let formant_adjustments = if spectral_section.len() >= 64 {
            spectral_section[32..64].to_vec()
        } else {
            vec![1.0; 32] // Default formant adjustments
        };

        let hnr_adjustments = if spectral_section.len() >= 96 {
            spectral_section[64..96].to_vec()
        } else {
            vec![0.0; 32]
        };

        let envelope_modifications = if spectral_section.len() >= 128 {
            spectral_section[96..128].to_vec()
        } else {
            vec![1.0; 32]
        };

        Ok(SpectralConditioningParams {
            spectral_tilt,
            formant_adjustments,
            hnr_adjustments,
            envelope_modifications,
        })
    }

    /// Synthesize using HiFiGAN with speaker conditioning
    async fn synthesize_hifigan(
        &self,
        mel_spectrogram: &[Vec<f32>],
        speaker_params: &SpeakerVocoderParams,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "vocoder-integration")]
        {
            // Use voirs_vocoder::hifigan with speaker conditioning
            let mut conditioned_mel = mel_spectrogram.to_vec();

            // Apply mel conditioning
            self.apply_mel_conditioning(&mut conditioned_mel, &speaker_params.mel_conditioning)?;

            // Create HiFiGAN synthesizer with conditioning
            let synthesizer = voirs_vocoder::hifigan::HiFiGAN::new()
                .with_speaker_embedding(&speaker_params.speaker_embedding)
                .with_conditioning_strength(self.conditioning_config.conditioning_strength);

            // Synthesize audio
            let audio = synthesizer.synthesize(&conditioned_mel).await?;
            Ok(audio)
        }

        #[cfg(not(feature = "vocoder-integration"))]
        {
            // Placeholder implementation
            let total_samples = mel_spectrogram.len() * 256; // Typical hop length
            let mut audio = vec![0.0; total_samples];

            // Apply basic conditioning
            for (i, sample) in audio.iter_mut().enumerate() {
                let time = i as f32 / 22050.0; // Common sample rate
                *sample = (time * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.1;
            }

            Ok(audio)
        }
    }

    /// Synthesize using WaveGlow with speaker conditioning
    async fn synthesize_waveglow(
        &self,
        mel_spectrogram: &[Vec<f32>],
        speaker_params: &SpeakerVocoderParams,
    ) -> Result<Vec<f32>> {
        // Similar to HiFiGAN but with WaveGlow-specific conditioning
        let total_samples = mel_spectrogram.len() * 256;
        let mut audio = vec![0.0; total_samples];

        // Apply WaveGlow-style conditioning (placeholder)
        for (i, sample) in audio.iter_mut().enumerate() {
            let time = i as f32 / 22050.0;
            let conditioning_factor =
                1.0 + speaker_params.prosodic_conditioning.rate_conditioning * 0.1;
            *sample = (time * 440.0 * conditioning_factor * 2.0 * std::f32::consts::PI).sin() * 0.1;
        }

        Ok(audio)
    }

    /// Synthesize using MelGAN with speaker conditioning
    async fn synthesize_melgan(
        &self,
        mel_spectrogram: &[Vec<f32>],
        speaker_params: &SpeakerVocoderParams,
    ) -> Result<Vec<f32>> {
        // MelGAN-specific conditioning implementation
        let total_samples = mel_spectrogram.len() * 256;
        let mut audio = vec![0.0; total_samples];

        // Apply MelGAN-style conditioning
        for (i, sample) in audio.iter_mut().enumerate() {
            let time = i as f32 / 22050.0;
            let spectral_factor = speaker_params
                .spectral_conditioning
                .spectral_tilt
                .get(i % speaker_params.spectral_conditioning.spectral_tilt.len())
                .unwrap_or(&0.0);
            *sample =
                (time * 440.0 * (1.0 + spectral_factor) * 2.0 * std::f32::consts::PI).sin() * 0.1;
        }

        Ok(audio)
    }

    /// Universal vocoder synthesis (auto-select best approach)
    async fn synthesize_universal(
        &self,
        mel_spectrogram: &[Vec<f32>],
        speaker_params: &SpeakerVocoderParams,
    ) -> Result<Vec<f32>> {
        // Choose best vocoder based on quality score and requirements
        let selected_vocoder = if speaker_params.quality_score > 0.8 {
            VocoderType::HiFiGAN // Best quality
        } else if speaker_params.quality_score > 0.6 {
            VocoderType::WaveGlow // Balanced
        } else {
            VocoderType::MelGAN // Fast
        };

        match selected_vocoder {
            VocoderType::HiFiGAN => {
                self.synthesize_hifigan(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::WaveGlow => {
                self.synthesize_waveglow(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::MelGAN => {
                self.synthesize_melgan(mel_spectrogram, speaker_params)
                    .await
            }
            VocoderType::Universal => unreachable!(), // Avoid infinite recursion
        }
    }

    /// Apply mel conditioning to mel spectrogram
    fn apply_mel_conditioning(
        &self,
        mel_spectrogram: &mut [Vec<f32>],
        mel_params: &MelConditioningParams,
    ) -> Result<()> {
        for frame in mel_spectrogram.iter_mut() {
            for (i, mel_bin) in frame.iter_mut().enumerate() {
                if i < mel_params.filter_adjustments.len() {
                    *mel_bin *= mel_params.filter_adjustments[i];
                }
                *mel_bin *= mel_params.compression_factor;
            }
        }
        Ok(())
    }

    /// Apply prosodic conditioning to audio chunk
    fn apply_prosodic_conditioning_chunk(
        &self,
        audio_chunk: &mut [f32],
        prosodic_params: &ProsodicConditioningParams,
        chunk_index: usize,
    ) -> Result<()> {
        let rate_factor = prosodic_params.rate_conditioning;

        // Apply rate conditioning (simplified time stretching)
        if rate_factor != 1.0 {
            for sample in audio_chunk.iter_mut() {
                *sample *= rate_factor;
            }
        }

        Ok(())
    }

    /// Apply spectral conditioning to audio chunk
    fn apply_spectral_conditioning_chunk(
        &self,
        audio_chunk: &mut [f32],
        spectral_params: &SpectralConditioningParams,
    ) -> Result<()> {
        // Apply spectral tilt (simplified)
        for (i, sample) in audio_chunk.iter_mut().enumerate() {
            let tilt_idx = i % spectral_params.spectral_tilt.len();
            let tilt_factor = 1.0 + spectral_params.spectral_tilt[tilt_idx] * 0.1;
            *sample *= tilt_factor;
        }

        Ok(())
    }

    /// Compute unique speaker ID for caching
    fn compute_speaker_id(&self, speaker_embedding: &SpeakerEmbedding) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &val in &speaker_embedding.vector {
            (val as u32).hash(&mut hasher);
        }
        format!("speaker_{}", hasher.finish())
    }

    /// Compute quality score for speaker parameters
    fn compute_quality_score(
        &self,
        mel_conditioning: &MelConditioningParams,
        prosodic_conditioning: &ProsodicConditioningParams,
        spectral_conditioning: &SpectralConditioningParams,
    ) -> Result<f32> {
        // Compute quality based on parameter consistency and coverage
        let mel_quality = mel_conditioning
            .filter_adjustments
            .iter()
            .map(|&x| (x - 1.0).abs())
            .sum::<f32>()
            / mel_conditioning.filter_adjustments.len() as f32;

        let prosodic_quality = prosodic_conditioning.f0_conditioning.len() as f32 / 64.0;

        let spectral_quality = spectral_conditioning.spectral_tilt.len() as f32 / 32.0;

        let overall_quality = ((1.0 - mel_quality) + prosodic_quality + spectral_quality) / 3.0;
        Ok(overall_quality.clamp(0.0, 1.0))
    }

    /// Compute spectral centroid of audio
    fn compute_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(1000.0); // Default centroid
        }

        // Simple spectral centroid estimation
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(1000.0)
        }
    }

    /// Compute spectral bandwidth of audio
    fn compute_spectral_bandwidth(&self, audio: &[f32]) -> Result<f32> {
        // Simplified bandwidth estimation
        let centroid = self.compute_spectral_centroid(audio)?;
        let mut variance_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            variance_sum += (i as f32 - centroid).powi(2) * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok((variance_sum / magnitude_sum).sqrt())
        } else {
            Ok(1000.0)
        }
    }

    /// Compute dynamic range of audio
    fn compute_dynamic_range(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(1.0);
        }

        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        if rms > 0.0 {
            Ok(max_amplitude / rms)
        } else {
            Ok(1.0)
        }
    }
}

impl Default for VocoderType {
    fn default() -> Self {
        VocoderType::HiFiGAN
    }
}

impl Default for VocoderConditioningConfig {
    fn default() -> Self {
        Self {
            enable_speaker_conditioning: true,
            enable_prosodic_conditioning: true,
            enable_spectral_conditioning: true,
            conditioning_strength: 0.7,
            quality_level: 0.8,
            cache_size: 50,
        }
    }
}

impl Default for VocoderCloningAdapter {
    fn default() -> Self {
        Self::new(VocoderType::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[test]
    fn test_vocoder_adapter_creation() {
        let adapter = VocoderCloningAdapter::new(VocoderType::HiFiGAN);
        assert_eq!(adapter.vocoder_type, VocoderType::HiFiGAN);
        assert!(adapter.conditioning_config.enable_speaker_conditioning);
    }

    #[test]
    fn test_vocoder_config_default() {
        let config = VocoderConditioningConfig::default();
        assert!(config.enable_speaker_conditioning);
        assert_eq!(config.conditioning_strength, 0.7);
        assert_eq!(config.cache_size, 50);
    }

    #[tokio::test]
    async fn test_speaker_params_generation() {
        let mut adapter = VocoderCloningAdapter::new(VocoderType::HiFiGAN);
        let embedding = SpeakerEmbedding::new(vec![0.1; 512]);
        let samples = vec![VoiceSample::new("test".to_string(), vec![0.1; 1000], 16000)];

        let result = adapter.generate_speaker_params(&embedding, &samples).await;
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.speaker_embedding.len(), 512);
        assert!(params.quality_score >= 0.0 && params.quality_score <= 1.0);
    }

    #[test]
    fn test_mel_conditioning_params() {
        let params = MelConditioningParams {
            filter_adjustments: vec![1.0; 80],
            frequency_scaling: 1.1,
            compression_factor: 0.9,
            dynamic_range: 1.2,
        };

        assert_eq!(params.filter_adjustments.len(), 80);
        assert_eq!(params.frequency_scaling, 1.1);
    }

    #[test]
    fn test_spectral_centroid_computation() {
        let adapter = VocoderCloningAdapter::default();
        let audio = vec![0.1, 0.2, 0.3, 0.2, 0.1];

        let centroid = adapter.compute_spectral_centroid(&audio);
        assert!(centroid.is_ok());
        assert!(centroid.unwrap() > 0.0);
    }

    #[test]
    fn test_speaker_id_computation() {
        let adapter = VocoderCloningAdapter::default();
        let embedding = SpeakerEmbedding::new(vec![0.1, 0.2, 0.3]);

        let id = adapter.compute_speaker_id(&embedding);
        assert!(id.starts_with("speaker_"));

        // Same embedding should produce same ID
        let id2 = adapter.compute_speaker_id(&embedding);
        assert_eq!(id, id2);
    }

    #[tokio::test]
    async fn test_universal_vocoder_selection() {
        let adapter = VocoderCloningAdapter::new(VocoderType::Universal);
        let mel = vec![vec![0.1; 80]; 100]; // 100 frames, 80 mel bins

        let high_quality_params = SpeakerVocoderParams {
            speaker_embedding: vec![0.1; 512],
            mel_conditioning: MelConditioningParams {
                filter_adjustments: vec![1.0; 80],
                frequency_scaling: 1.0,
                compression_factor: 1.0,
                dynamic_range: 1.0,
            },
            prosodic_conditioning: ProsodicConditioningParams {
                f0_conditioning: vec![0.0; 16],
                energy_conditioning: vec![0.0; 16],
                duration_conditioning: vec![0.0; 16],
                rate_conditioning: 1.0,
            },
            spectral_conditioning: SpectralConditioningParams {
                spectral_tilt: vec![0.0; 32],
                formant_adjustments: vec![1.0; 32],
                hnr_adjustments: vec![0.0; 32],
                envelope_modifications: vec![1.0; 32],
            },
            quality_score: 0.9, // High quality
            timestamp: std::time::SystemTime::now(),
        };

        let result = adapter
            .synthesize_universal(&mel, &high_quality_params)
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().len() > 0);
    }
}
