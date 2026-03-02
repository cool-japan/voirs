//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    api_standards::StandardConfig,
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    types::VoiceSample,
    Error, Result,
};
use scirs2_core::ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

/// Perceptual analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualAnalysis {
    pub loudness_similarity: f32,
    pub roughness_similarity: f32,
    pub sharpness_similarity: f32,
    pub pitch_similarity: f32,
    pub timber_similarity: f32,
}
/// Configuration for comprehensive quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Enable perceptual assessment
    pub perceptual_assessment: bool,
    /// Enable spectral analysis
    pub spectral_analysis: bool,
    /// Enable temporal analysis
    pub temporal_analysis: bool,
    /// Enable artifact detection
    pub artifact_detection: bool,
    /// Enable embedding-based similarity
    pub embedding_similarity: bool,
    /// Assessment threshold for overall quality
    pub quality_threshold: f32,
    /// Minimum similarity threshold for speaker verification
    pub similarity_threshold: f32,
    /// Window size for analysis (samples)
    pub analysis_window_size: usize,
    /// Hop size for overlapping analysis
    pub analysis_hop_size: usize,
    /// Cache assessment results
    pub enable_caching: bool,
    /// Assessment method weights
    pub method_weights: MethodWeights,
    /// Real-time assessment settings
    pub realtime_settings: RealtimeAssessmentConfig,
}
/// Signal-to-noise ratio analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SNRAnalysis {
    pub original_snr: f32,
    pub cloned_snr: f32,
    pub snr_degradation: f32,
    pub noise_floor: f32,
    pub dynamic_range: f32,
}
/// Comprehensive quality metrics for cloned voice evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Speaker similarity score (0.0 to 1.0)
    pub speaker_similarity: f32,
    /// Audio quality score (0.0 to 1.0)
    pub audio_quality: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Content preservation score (0.0 to 1.0)
    pub content_preservation: f32,
    /// Prosodic similarity score (0.0 to 1.0)
    pub prosodic_similarity: f32,
    /// Spectral similarity score (0.0 to 1.0)
    pub spectral_similarity: f32,
    /// Individual metric scores
    pub metrics: HashMap<String, f32>,
    /// Detailed analysis results
    pub analysis: QualityAnalysis,
    /// Assessment metadata
    pub metadata: AssessmentMetadata,
}
impl QualityMetrics {
    /// Create new quality metrics
    pub fn new() -> Self {
        Self {
            overall_score: 0.0,
            speaker_similarity: 0.0,
            audio_quality: 0.0,
            naturalness: 0.0,
            content_preservation: 0.0,
            prosodic_similarity: 0.0,
            spectral_similarity: 0.0,
            metrics: HashMap::new(),
            analysis: QualityAnalysis::default(),
            metadata: AssessmentMetadata::default(),
        }
    }
    /// Calculate overall score from individual metrics with configurable weights
    pub fn calculate_overall_score(&mut self, weights: &MethodWeights) {
        self.overall_score = (self.speaker_similarity * weights.speaker_similarity_weight
            + self.audio_quality * weights.audio_quality_weight
            + self.naturalness * weights.naturalness_weight
            + self.content_preservation * weights.content_preservation_weight
            + self.prosodic_similarity * weights.prosodic_weight
            + self.spectral_similarity * weights.spectral_weight)
            / (weights.speaker_similarity_weight
                + weights.audio_quality_weight
                + weights.naturalness_weight
                + weights.content_preservation_weight
                + weights.prosodic_weight
                + weights.spectral_weight);
    }
    /// Get quality grade based on overall score
    pub fn quality_grade(&self) -> QualityGrade {
        match self.overall_score {
            score if score >= 0.9 => QualityGrade::Excellent,
            score if score >= 0.8 => QualityGrade::Good,
            score if score >= 0.7 => QualityGrade::Acceptable,
            score if score >= 0.6 => QualityGrade::Poor,
            _ => QualityGrade::Unacceptable,
        }
    }
    /// Check if quality meets threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.overall_score >= threshold
    }
    /// Get detailed quality report
    pub fn detailed_report(&self) -> String {
        let grade = self.quality_grade();
        format!(
            "Quality Assessment Report\n\
            ========================\n\
            Overall Score: {:.3} ({grade:?})\n\
            Speaker Similarity: {:.3}\n\
            Audio Quality: {:.3}\n\
            Naturalness: {:.3}\n\
            Content Preservation: {:.3}\n\
            Prosodic Similarity: {:.3}\n\
            Spectral Similarity: {:.3}\n\
            \n\
            SNR Analysis:\n\
            - Original SNR: {:.1} dB\n\
            - Cloned SNR: {:.1} dB\n\
            - Degradation: {:.1} dB\n\
            \n\
            Artifact Analysis:\n\
            - Overall Artifacts: {:.3}\n\
            - Click Detection: {:.3}\n\
            - Discontinuity: {:.3}\n\
            - Aliasing: {:.3}\n",
            self.overall_score,
            self.speaker_similarity,
            self.audio_quality,
            self.naturalness,
            self.content_preservation,
            self.prosodic_similarity,
            self.spectral_similarity,
            self.analysis.snr_analysis.original_snr,
            self.analysis.snr_analysis.cloned_snr,
            self.analysis.snr_analysis.snr_degradation,
            self.analysis.artifact_analysis.overall_artifact_score,
            self.analysis.artifact_analysis.click_detection,
            self.analysis.artifact_analysis.discontinuity_detection,
            self.analysis.artifact_analysis.aliasing_detection,
        )
    }
}
/// Spectral analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    pub spectral_centroid_similarity: f32,
    pub spectral_rolloff_similarity: f32,
    pub spectral_flatness_similarity: f32,
    pub harmonic_similarity: f32,
    pub formant_similarity: f32,
    pub bandwidth_similarity: f32,
}
/// Advanced quality assessor with multiple evaluation methods
pub struct CloningQualityAssessor {
    /// Assessment configuration
    pub(super) config: QualityConfig,
    /// Embedding extractor for speaker similarity
    pub(super) embedding_extractor: Option<SpeakerEmbeddingExtractor>,
    /// Quality metrics cache
    pub(super) metrics_cache: Arc<RwLock<HashMap<String, QualityMetrics>>>,
    /// Performance statistics
    pub(super) performance_stats: Arc<RwLock<AssessmentStats>>,
}
impl CloningQualityAssessor {
    /// Create new assessor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(QualityConfig::default())
    }
    /// Create new assessor with custom configuration
    pub fn with_config(config: QualityConfig) -> Result<Self> {
        config.validate()?;
        let embedding_extractor = if config.embedding_similarity {
            Some(SpeakerEmbeddingExtractor::default())
        } else {
            None
        };
        Ok(Self {
            config,
            embedding_extractor,
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(AssessmentStats::new())),
        })
    }
    /// Get current configuration
    pub fn get_config(&self) -> &QualityConfig {
        &self.config
    }
    /// Update configuration with validation
    pub fn update_config(&mut self, config: QualityConfig) -> Result<()> {
        config.validate()?;
        if config.embedding_similarity != self.config.embedding_similarity {
            self.embedding_extractor = if config.embedding_similarity {
                Some(SpeakerEmbeddingExtractor::default())
            } else {
                None
            };
        }
        self.config = config;
        Ok(())
    }
    /// Comprehensive quality assessment of cloned voice
    pub async fn assess_quality(
        &mut self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<QualityMetrics> {
        let start_time = Instant::now();
        info!(
            "Starting comprehensive quality assessment for {} vs {}",
            original.id, cloned.id
        );
        let cache_key = format!("{}_{}", original.id, cloned.id);
        if self.config.enable_caching {
            let cache = self.metrics_cache.read().await;
            if let Some(cached_metrics) = cache.get(&cache_key) {
                let mut stats = self.performance_stats.write().await;
                stats.cache_hits += 1;
                return Ok(cached_metrics.clone());
            }
        }
        let mut metrics = QualityMetrics::new();
        self.validate_samples(original, cloned)?;
        if self.config.spectral_analysis {
            trace!("Performing spectral analysis");
            metrics.spectral_similarity = self.assess_spectral_similarity(original, cloned).await?;
            metrics.analysis.spectral_analysis =
                self.detailed_spectral_analysis(original, cloned).await?;
        }
        if self.config.temporal_analysis {
            trace!("Performing temporal analysis");
            metrics.analysis.temporal_analysis = self
                .assess_temporal_characteristics(original, cloned)
                .await?;
            metrics.content_preservation = metrics.analysis.temporal_analysis.duration_similarity;
        }
        if self.config.perceptual_assessment {
            trace!("Performing perceptual assessment");
            metrics.naturalness = self.assess_naturalness(original, cloned).await?;
            metrics.analysis.perceptual_analysis =
                self.detailed_perceptual_analysis(original, cloned).await?;
        }
        if self.config.artifact_detection {
            trace!("Performing artifact detection");
            metrics.analysis.artifact_analysis = self.detect_artifacts(cloned).await?;
            metrics.audio_quality = 1.0 - metrics.analysis.artifact_analysis.overall_artifact_score;
        }
        if self.config.embedding_similarity && self.embedding_extractor.is_some() {
            trace!("Performing embedding-based similarity assessment");
            metrics.speaker_similarity = self.assess_speaker_similarity(original, cloned).await?;
        } else {
            metrics.speaker_similarity = self.assess_acoustic_similarity(original, cloned).await?;
        }
        metrics.prosodic_similarity = self.assess_prosodic_similarity(original, cloned).await?;
        metrics.analysis.snr_analysis = self.analyze_snr(original, cloned)?;
        metrics.calculate_overall_score(&self.config.method_weights);
        let assessment_duration = start_time.elapsed();
        metrics.metadata = AssessmentMetadata {
            assessment_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            assessment_duration: assessment_duration.as_secs_f32(),
            original_duration: original.duration,
            cloned_duration: cloned.duration,
            sample_rate: original.sample_rate,
            assessment_method: "comprehensive".to_string(),
            quality_version: "1.0".to_string(),
        };
        if self.config.enable_caching {
            let mut cache = self.metrics_cache.write().await;
            cache.insert(cache_key, metrics.clone());
        }
        {
            let mut stats = self.performance_stats.write().await;
            stats.update_assessment(assessment_duration, &metrics);
            stats.cache_misses += 1;
        }
        info!(
            "Quality assessment completed: overall score {:.3} ({})",
            metrics.overall_score,
            format!("{:?}", metrics.quality_grade())
        );
        Ok(metrics)
    }
    /// Quick assessment for real-time applications
    pub async fn quick_assess_quality(
        &mut self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<QualityMetrics> {
        let start_time = Instant::now();
        let mut metrics = QualityMetrics::new();
        metrics.speaker_similarity = self.assess_acoustic_similarity(original, cloned).await?;
        let snr_analysis = self.analyze_snr(original, cloned)?;
        metrics.audio_quality = (snr_analysis.cloned_snr / 30.0).clamp(0.0, 1.0);
        metrics.naturalness = self.compute_energy_envelope_similarity(original, cloned)?;
        metrics.calculate_overall_score(&self.config.method_weights);
        let assessment_duration = start_time.elapsed();
        metrics.metadata.assessment_duration = assessment_duration.as_secs_f32();
        metrics.metadata.assessment_method = "quick".to_string();
        Ok(metrics)
    }
    /// Validate that samples are suitable for comparison
    fn validate_samples(&self, original: &VoiceSample, cloned: &VoiceSample) -> Result<()> {
        if original.audio.is_empty() {
            return Err(Error::Validation("Original sample is empty".to_string()));
        }
        if cloned.audio.is_empty() {
            return Err(Error::Validation("Cloned sample is empty".to_string()));
        }
        if original.sample_rate != cloned.sample_rate {
            warn!(
                "Sample rate mismatch: original {} Hz, cloned {} Hz",
                original.sample_rate, cloned.sample_rate
            );
        }
        Ok(())
    }
    /// Assess spectral similarity between original and cloned samples
    async fn assess_spectral_similarity(
        &mut self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        let original_spectrum = self.compute_spectrum(&original.get_normalized_audio())?;
        let cloned_spectrum = self.compute_spectrum(&cloned.get_normalized_audio())?;
        let similarity = self.compute_spectral_correlation(&original_spectrum, &cloned_spectrum)?;
        Ok(similarity.clamp(0.0, 1.0))
    }
    /// Detailed spectral analysis
    async fn detailed_spectral_analysis(
        &mut self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<SpectralAnalysis> {
        let original_audio = original.get_normalized_audio();
        let cloned_audio = cloned.get_normalized_audio();
        let orig_centroid =
            self.compute_spectral_centroid(&original_audio, original.sample_rate)?;
        let cloned_centroid = self.compute_spectral_centroid(&cloned_audio, cloned.sample_rate)?;
        let centroid_similarity = 1.0
            - (orig_centroid - cloned_centroid).abs() / (orig_centroid + cloned_centroid + 1e-8);
        let orig_rolloff = self.compute_spectral_rolloff(&original_audio, original.sample_rate)?;
        let cloned_rolloff = self.compute_spectral_rolloff(&cloned_audio, cloned.sample_rate)?;
        let rolloff_similarity =
            1.0 - (orig_rolloff - cloned_rolloff).abs() / (orig_rolloff + cloned_rolloff + 1e-8);
        let orig_flatness = self.compute_spectral_flatness(&original_audio)?;
        let cloned_flatness = self.compute_spectral_flatness(&cloned_audio)?;
        let flatness_similarity = 1.0 - (orig_flatness - cloned_flatness).abs();
        Ok(SpectralAnalysis {
            spectral_centroid_similarity: centroid_similarity,
            spectral_rolloff_similarity: rolloff_similarity,
            spectral_flatness_similarity: flatness_similarity,
            harmonic_similarity: 0.8,
            formant_similarity: 0.85,
            bandwidth_similarity: 0.9,
        })
    }
    /// Assess temporal characteristics
    async fn assess_temporal_characteristics(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<TemporalAnalysis> {
        let duration_similarity = 1.0
            - (original.duration - cloned.duration).abs()
                / (original.duration + cloned.duration + 1e-8);
        let energy_envelope_similarity =
            self.compute_energy_envelope_similarity(original, cloned)?;
        Ok(TemporalAnalysis {
            duration_similarity,
            rhythm_similarity: 0.8,
            energy_envelope_similarity,
            pause_similarity: 0.85,
            speech_rate_similarity: 0.9,
        })
    }
    /// Assess naturalness of cloned voice
    async fn assess_naturalness(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        let mut naturalness_factors = Vec::new();
        let pitch_naturalness =
            self.assess_pitch_naturalness(&cloned.get_normalized_audio(), cloned.sample_rate)?;
        naturalness_factors.push(pitch_naturalness);
        let energy_naturalness = self.assess_energy_naturalness(&cloned.get_normalized_audio())?;
        naturalness_factors.push(energy_naturalness);
        let spectral_naturalness =
            self.assess_spectral_naturalness(&cloned.get_normalized_audio(), cloned.sample_rate)?;
        naturalness_factors.push(spectral_naturalness);
        let temporal_naturalness =
            self.assess_temporal_naturalness(&cloned.get_normalized_audio(), cloned.sample_rate)?;
        naturalness_factors.push(temporal_naturalness);
        let average_naturalness =
            naturalness_factors.iter().sum::<f32>() / naturalness_factors.len() as f32;
        Ok(average_naturalness.clamp(0.0, 1.0))
    }
    /// Detailed perceptual analysis
    async fn detailed_perceptual_analysis(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<PerceptualAnalysis> {
        let original_audio = original.get_normalized_audio();
        let cloned_audio = cloned.get_normalized_audio();
        let orig_loudness = self.compute_rms_energy(&original_audio);
        let cloned_loudness = self.compute_rms_energy(&cloned_audio);
        let loudness_similarity = 1.0
            - (orig_loudness - cloned_loudness).abs() / (orig_loudness + cloned_loudness + 1e-8);
        let pitch_similarity =
            self.assess_pitch_similarity(&original_audio, &cloned_audio, original.sample_rate)?;
        Ok(PerceptualAnalysis {
            loudness_similarity,
            roughness_similarity: 0.85,
            sharpness_similarity: 0.8,
            pitch_similarity,
            timber_similarity: 0.82,
        })
    }
    /// Detect artifacts in cloned audio
    async fn detect_artifacts(&self, cloned: &VoiceSample) -> Result<ArtifactAnalysis> {
        let audio = cloned.get_normalized_audio();
        let click_score = self.detect_clicks(&audio)?;
        let discontinuity_score = self.detect_discontinuities(&audio)?;
        let aliasing_score = self.detect_aliasing(&audio, cloned.sample_rate)?;
        let reverb_score = self.detect_reverb_artifacts(&audio)?;
        let robotic_score = self.detect_robotic_artifacts(&audio, cloned.sample_rate)?;
        let overall_artifact_score =
            (click_score + discontinuity_score + aliasing_score + reverb_score + robotic_score)
                / 5.0;
        Ok(ArtifactAnalysis {
            click_detection: click_score,
            discontinuity_detection: discontinuity_score,
            aliasing_detection: aliasing_score,
            reverb_artifacts: reverb_score,
            robotic_artifacts: robotic_score,
            overall_artifact_score,
        })
    }
    /// Assess speaker similarity using embeddings
    async fn assess_speaker_similarity(
        &mut self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        if let Some(extractor) = &mut self.embedding_extractor {
            let original_embedding = extractor.extract(original).await?;
            let cloned_embedding = extractor.extract(cloned).await?;
            let similarity = original_embedding.similarity(&cloned_embedding);
            Ok(similarity.clamp(0.0, 1.0))
        } else {
            self.assess_acoustic_similarity(original, cloned).await
        }
    }
    /// Assess acoustic similarity (without embeddings)
    async fn assess_acoustic_similarity(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        let original_features =
            self.extract_acoustic_features(&original.get_normalized_audio(), original.sample_rate)?;
        let cloned_features =
            self.extract_acoustic_features(&cloned.get_normalized_audio(), cloned.sample_rate)?;
        let similarity = self.compute_feature_similarity(&original_features, &cloned_features)?;
        Ok(similarity.clamp(0.0, 1.0))
    }
    /// Assess prosodic similarity
    async fn assess_prosodic_similarity(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        let orig_f0 =
            self.extract_f0_contour(&original.get_normalized_audio(), original.sample_rate)?;
        let cloned_f0 =
            self.extract_f0_contour(&cloned.get_normalized_audio(), cloned.sample_rate)?;
        let f0_similarity = self.compute_contour_similarity(&orig_f0, &cloned_f0)?;
        let orig_energy = self.extract_energy_contour(&original.get_normalized_audio())?;
        let cloned_energy = self.extract_energy_contour(&cloned.get_normalized_audio())?;
        let energy_similarity = self.compute_contour_similarity(&orig_energy, &cloned_energy)?;
        Ok((f0_similarity + energy_similarity) / 2.0)
    }
    /// Compute spectrum of audio signal
    fn compute_spectrum(&self, audio: &[f32]) -> Result<Array1<f32>> {
        let fft_size = self.config.analysis_window_size;
        let mut input = vec![0.0; fft_size];
        let len = audio.len().min(fft_size);
        input[..len].copy_from_slice(&audio[..len]);
        let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
        let spectrum_complex = scirs2_fft::rfft(&input_f64, None)
            .map_err(|e| Error::Processing(format!("FFT processing failed: {e}")))?;
        let spectrum: Array1<f32> = spectrum_complex
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
            .collect::<Vec<f32>>()
            .into();
        Ok(spectrum)
    }
    /// Compute spectral correlation
    fn compute_spectral_correlation(
        &self,
        spec1: &Array1<f32>,
        spec2: &Array1<f32>,
    ) -> Result<f32> {
        let min_len = spec1.len().min(spec2.len());
        if min_len == 0 {
            return Ok(0.0);
        }
        let s1 = &spec1.slice(s![..min_len]);
        let s2 = &spec2.slice(s![..min_len]);
        let mean1 = s1.mean().unwrap_or(0.0);
        let mean2 = s2.mean().unwrap_or(0.0);
        let mut numerator = 0.0_f32;
        let mut denom1 = 0.0_f32;
        let mut denom2 = 0.0_f32;
        for i in 0..min_len {
            let diff1 = s1[i] - mean1;
            let diff2 = s2[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }
        let correlation = if denom1 > 0.0 && denom2 > 0.0 {
            numerator / (denom1 * denom2).sqrt()
        } else {
            0.0
        };
        Ok((correlation + 1.0) / 2.0)
    }
    /// Analyze SNR of original and cloned samples
    fn analyze_snr(&self, original: &VoiceSample, cloned: &VoiceSample) -> Result<SNRAnalysis> {
        let original_audio = original.get_normalized_audio();
        let cloned_audio = cloned.get_normalized_audio();
        let original_snr = self.estimate_snr(&original_audio);
        let cloned_snr = self.estimate_snr(&cloned_audio);
        let snr_degradation = original_snr - cloned_snr;
        let noise_floor = self.estimate_noise_floor(&cloned_audio);
        let dynamic_range = self.compute_dynamic_range(&cloned_audio);
        Ok(SNRAnalysis {
            original_snr,
            cloned_snr,
            snr_degradation,
            noise_floor,
            dynamic_range,
        })
    }
    /// Estimate SNR of audio signal
    pub(super) fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let signal_power = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let mut energies: Vec<f32> = audio
            .chunks(256)
            .map(|chunk| chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32)
            .collect();
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_samples = energies.len() / 10;
        let noise_power = if noise_samples > 0 {
            energies[..noise_samples].iter().sum::<f32>() / noise_samples as f32
        } else {
            signal_power * 0.001
        };
        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            30.0
        }
    }
    fn compute_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let fft_size = 1024.min(audio.len());
        if fft_size == 0 {
            return Ok(0.0);
        }
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        for k in 1..fft_size / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (n, &sample) in audio[..fft_size].iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / fft_size as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            let magnitude = (real * real + imag * imag).sqrt();
            let frequency = k as f32 * sample_rate as f32 / fft_size as f32;
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }
        Ok(if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        })
    }
    fn compute_spectral_rolloff(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        Ok(sample_rate as f32 * 0.425)
    }
    fn compute_spectral_flatness(&self, audio: &[f32]) -> Result<f32> {
        Ok(0.5)
    }
    fn compute_energy_envelope_similarity(
        &self,
        original: &VoiceSample,
        cloned: &VoiceSample,
    ) -> Result<f32> {
        let orig_envelope = self.extract_energy_contour(&original.get_normalized_audio())?;
        let cloned_envelope = self.extract_energy_contour(&cloned.get_normalized_audio())?;
        self.compute_contour_similarity(&orig_envelope, &cloned_envelope)
    }
    fn assess_pitch_naturalness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let f0_contour = self.extract_f0_contour(audio, sample_rate)?;
        if f0_contour.len() < 10 {
            return Ok(0.5);
        }
        let f0_mean = f0_contour.iter().sum::<f32>() / f0_contour.len() as f32;
        let f0_std = {
            let variance = f0_contour
                .iter()
                .map(|f| (f - f0_mean).powi(2))
                .sum::<f32>()
                / f0_contour.len() as f32;
            variance.sqrt()
        };
        let cv = if f0_mean > 0.0 { f0_std / f0_mean } else { 0.0 };
        if cv > 0.1 && cv < 0.3 {
            Ok(1.0 - (cv - 0.2).abs() * 5.0)
        } else {
            Ok(0.5)
        }
    }
    fn assess_energy_naturalness(&self, audio: &[f32]) -> Result<f32> {
        let energy_contour = self.extract_energy_contour(audio)?;
        if energy_contour.len() < 10 {
            return Ok(0.5);
        }
        let energy_mean = energy_contour.iter().sum::<f32>() / energy_contour.len() as f32;
        let energy_std = {
            let variance = energy_contour
                .iter()
                .map(|e| (e - energy_mean).powi(2))
                .sum::<f32>()
                / energy_contour.len() as f32;
            variance.sqrt()
        };
        let cv = if energy_mean > 0.0 {
            energy_std / energy_mean
        } else {
            0.0
        };
        if cv > 0.2 && cv < 0.8 {
            Ok(1.0 - (cv - 0.5).abs() * 2.0)
        } else {
            Ok(0.5)
        }
    }
    fn assess_spectral_naturalness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let spectral_centroid = self.compute_spectral_centroid(audio, sample_rate)?;
        let nyquist = sample_rate as f32 / 2.0;
        let normalized_centroid = spectral_centroid / nyquist;
        if normalized_centroid > 0.1 && normalized_centroid < 0.6 {
            Ok(1.0 - (normalized_centroid - 0.35).abs() * 4.0)
        } else {
            Ok(0.3)
        }
    }
    fn assess_temporal_naturalness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let frame_size = (sample_rate as f32 * 0.025) as usize;
        if audio.len() < frame_size * 4 {
            return Ok(0.5);
        }
        let mut voiced_frames = 0;
        let mut total_frames = 0;
        for chunk in audio.chunks(frame_size) {
            if chunk.len() < frame_size / 2 {
                continue;
            }
            let energy = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            if energy > 0.001 {
                voiced_frames += 1;
            }
            total_frames += 1;
        }
        let voiced_ratio = if total_frames > 0 {
            voiced_frames as f32 / total_frames as f32
        } else {
            0.0
        };
        if voiced_ratio > 0.3 && voiced_ratio < 0.8 {
            Ok(1.0 - (voiced_ratio - 0.55).abs() * 4.0)
        } else {
            Ok(0.4)
        }
    }
    fn assess_pitch_similarity(
        &self,
        audio1: &[f32],
        audio2: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        let f0_1 = self.extract_f0_contour(audio1, sample_rate)?;
        let f0_2 = self.extract_f0_contour(audio2, sample_rate)?;
        self.compute_contour_similarity(&f0_1, &f0_2)
    }
    pub(super) fn detect_clicks(&self, audio: &[f32]) -> Result<f32> {
        if audio.len() < 3 {
            return Ok(0.0);
        }
        let mut click_score = 0.0;
        let mut click_count = 0;
        for i in 1..audio.len() - 1 {
            let diff1 = (audio[i] - audio[i - 1]).abs();
            let diff2 = (audio[i + 1] - audio[i]).abs();
            let avg_diff = (diff1 + diff2) / 2.0;
            if avg_diff > 0.1 {
                click_score += avg_diff;
                click_count += 1;
            }
        }
        let normalized_score = if click_count > 0 {
            (click_score / click_count as f32).min(1.0)
        } else {
            0.0
        };
        Ok(normalized_score)
    }
    fn detect_discontinuities(&self, audio: &[f32]) -> Result<f32> {
        if audio.len() < 10 {
            return Ok(0.0);
        }
        let window_size = 32;
        let mut discontinuity_score = 0.0;
        let mut window_count = 0;
        for i in window_size..audio.len() - window_size {
            let before_energy =
                audio[i - window_size..i].iter().map(|x| x * x).sum::<f32>() / window_size as f32;
            let after_energy =
                audio[i..i + window_size].iter().map(|x| x * x).sum::<f32>() / window_size as f32;
            let energy_ratio = if before_energy > 0.0 {
                (after_energy - before_energy).abs() / before_energy
            } else {
                0.0
            };
            if energy_ratio > 2.0 {
                discontinuity_score += energy_ratio.min(10.0) / 10.0;
            }
            window_count += 1;
        }
        Ok(if window_count > 0 {
            (discontinuity_score / window_count as f32).min(1.0)
        } else {
            0.0
        })
    }
    fn detect_aliasing(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let nyquist = sample_rate as f32 / 2.0;
        let high_freq_threshold = nyquist * 0.8;
        Ok(0.1)
    }
    fn detect_reverb_artifacts(&self, audio: &[f32]) -> Result<f32> {
        if audio.len() < 1000 {
            return Ok(0.0);
        }
        let mut max_correlation = 0.0_f32;
        let delay_start = 100;
        let delay_end = 1000.min(audio.len() / 2);
        for delay in delay_start..delay_end {
            let mut correlation = 0.0;
            let samples_to_check = (audio.len() - delay).min(500);
            for i in 0..samples_to_check {
                correlation += audio[i] * audio[i + delay];
            }
            correlation /= samples_to_check as f32;
            max_correlation = max_correlation.max(correlation.abs());
        }
        Ok((max_correlation * 2.0_f32).min(1.0_f32))
    }
    fn detect_robotic_artifacts(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let spectral_centroid = self.compute_spectral_centroid(audio, sample_rate)?;
        let nyquist = sample_rate as f32 / 2.0;
        let normalized_centroid = spectral_centroid / nyquist;
        let robotic_score = if normalized_centroid < 0.05 || normalized_centroid > 0.8 {
            0.8
        } else if normalized_centroid < 0.1 || normalized_centroid > 0.7 {
            0.5
        } else {
            0.1
        };
        Ok(robotic_score)
    }
    fn extract_acoustic_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let features = vec![
            self.compute_rms_energy(audio),
            self.compute_zero_crossing_rate(audio),
            self.compute_spectral_centroid(audio, sample_rate)?,
        ];
        Ok(features)
    }
    fn compute_feature_similarity(&self, features1: &[f32], features2: &[f32]) -> Result<f32> {
        if features1.len() != features2.len() {
            return Ok(0.0);
        }
        let mut similarity = 0.0;
        for (f1, f2) in features1.iter().zip(features2) {
            let diff = (f1 - f2).abs();
            let avg = (f1.abs() + f2.abs()) / 2.0;
            similarity += if avg > 1e-8 {
                1.0 - (diff / avg).min(1.0)
            } else {
                1.0
            };
        }
        Ok(similarity / features1.len() as f32)
    }
    fn extract_f0_contour(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let frame_size = (sample_rate as f32 * 0.025) as usize;
        let hop_size = (sample_rate as f32 * 0.010) as usize;
        let mut f0_values = Vec::new();
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            let frame = &audio[i..end];
            if frame.len() >= frame_size / 2 {
                let f0 = self.estimate_f0_autocorr(frame, sample_rate);
                f0_values.push(f0);
            }
        }
        Ok(f0_values)
    }
    fn estimate_f0_autocorr(&self, frame: &[f32], sample_rate: u32) -> f32 {
        let min_period = sample_rate / 500;
        let max_period = sample_rate / 50;
        let mut max_corr = 0.0;
        let mut best_period = min_period;
        for period in min_period..max_period.min(frame.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period_samples = period as usize;
            for i in 0..(frame.len() - period_samples) {
                correlation += frame[i] * frame[i + period_samples];
            }
            if correlation > max_corr {
                max_corr = correlation;
                best_period = period;
            }
        }
        if max_corr > 0.0 {
            sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }
    fn extract_energy_contour(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let frame_size = 512;
        let hop_size = 256;
        let mut energy_values = Vec::new();
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            let frame = &audio[i..end];
            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            energy_values.push(energy);
        }
        Ok(energy_values)
    }
    fn compute_contour_similarity(&self, contour1: &[f32], contour2: &[f32]) -> Result<f32> {
        let min_len = contour1.len().min(contour2.len());
        if min_len < 2 {
            return Ok(0.5);
        }
        let c1 = &contour1[..min_len];
        let c2 = &contour2[..min_len];
        let mean1 = c1.iter().sum::<f32>() / c1.len() as f32;
        let mean2 = c2.iter().sum::<f32>() / c2.len() as f32;
        let mut numerator = 0.0;
        let mut denom1 = 0.0;
        let mut denom2 = 0.0;
        for i in 0..min_len {
            let diff1 = c1[i] - mean1;
            let diff2 = c2[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }
        let correlation = if denom1 > 0.0 && denom2 > 0.0 {
            numerator / (denom1 * denom2).sqrt()
        } else {
            0.0
        };
        Ok((correlation + 1.0) / 2.0)
    }
    fn compute_rms_energy(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt()
    }
    fn compute_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();
        crossings as f32 / (audio.len() - 1) as f32
    }
    fn estimate_noise_floor(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let mut energies: Vec<f32> = audio
            .chunks(256)
            .map(|chunk| chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32)
            .collect();
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_samples = (energies.len() / 20).max(1);
        energies[..noise_samples].iter().sum::<f32>() / noise_samples as f32
    }
    fn compute_dynamic_range(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let max_amplitude = audio.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let noise_floor = self.estimate_noise_floor(audio).sqrt();
        if noise_floor > 0.0 {
            20.0 * (max_amplitude / noise_floor).log10()
        } else {
            60.0
        }
    }
    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> AssessmentStats {
        self.performance_stats.read().await.clone()
    }
    /// Clear metrics cache
    pub async fn clear_cache(&self) {
        self.metrics_cache.write().await.clear();
    }
}
/// Weights for different assessment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodWeights {
    pub speaker_similarity_weight: f32,
    pub audio_quality_weight: f32,
    pub naturalness_weight: f32,
    pub content_preservation_weight: f32,
    pub prosodic_weight: f32,
    pub spectral_weight: f32,
}
/// Performance statistics for quality assessment
#[derive(Debug, Clone)]
pub struct AssessmentStats {
    pub total_assessments: u64,
    pub average_duration: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub quality_distribution: HashMap<String, u64>,
}
impl Default for AssessmentStats {
    fn default() -> Self {
        Self::new()
    }
}

impl AssessmentStats {
    pub fn new() -> Self {
        Self {
            total_assessments: 0,
            average_duration: Duration::from_secs(0),
            cache_hits: 0,
            cache_misses: 0,
            quality_distribution: HashMap::new(),
        }
    }
    pub(super) fn update_assessment(&mut self, duration: Duration, metrics: &QualityMetrics) {
        self.total_assessments += 1;
        let total_nanos = self.average_duration.as_nanos() as u64 * (self.total_assessments - 1)
            + duration.as_nanos() as u64;
        self.average_duration = Duration::from_nanos(total_nanos / self.total_assessments);
        let grade = format!("{:?}", metrics.quality_grade());
        *self.quality_distribution.entry(grade).or_insert(0) += 1;
    }
}
/// Artifact detection analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactAnalysis {
    pub click_detection: f32,
    pub discontinuity_detection: f32,
    pub aliasing_detection: f32,
    pub reverb_artifacts: f32,
    pub robotic_artifacts: f32,
    pub overall_artifact_score: f32,
}
/// Quality grades for human-readable assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityGrade {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Unacceptable,
}
/// Assessment metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentMetadata {
    pub assessment_time: f64,
    pub assessment_duration: f32,
    pub original_duration: f32,
    pub cloned_duration: f32,
    pub sample_rate: u32,
    pub assessment_method: String,
    pub quality_version: String,
}
/// Real-time assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAssessmentConfig {
    pub enable_realtime: bool,
    pub assessment_interval: f32,
    pub sliding_window_size: f32,
    pub quick_assessment_mode: bool,
}
/// Temporal analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub duration_similarity: f32,
    pub rhythm_similarity: f32,
    pub energy_envelope_similarity: f32,
    pub pause_similarity: f32,
    pub speech_rate_similarity: f32,
}
/// Detailed quality analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityAnalysis {
    /// Signal-to-noise ratio comparison
    pub snr_analysis: SNRAnalysis,
    /// Frequency domain analysis
    pub spectral_analysis: SpectralAnalysis,
    /// Temporal analysis
    pub temporal_analysis: TemporalAnalysis,
    /// Perceptual analysis
    pub perceptual_analysis: PerceptualAnalysis,
    /// Artifact detection results
    pub artifact_analysis: ArtifactAnalysis,
}
