//! Zero-shot singing voice synthesis
//!
//! This module provides capabilities for generating singing voices for new speakers
//! with minimal or no training data, using pre-trained models and few-shot learning techniques.

use crate::ai::StyleEmbedding;
use crate::core::SingingEngine;
use crate::models::{ModelType, SingingModel, SingingModelBuilder};
use crate::precision_quality::functions::detect_f0_autocorr_frame;
use crate::types::{SingingRequest, SingingResponse, VoiceCharacteristics, VoiceType};
use crate::voice_conversion::{SpeakerEmbedding, VoiceQualityMetrics};
use crate::Error;
use candle_core::{Device, Tensor};
use scirs2_core::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Zero-shot singing synthesis system
pub struct ZeroShotSynthesizer {
    /// Pre-trained base model
    base_model: Box<dyn SingingModel>,
    /// Voice adaptation engine
    adaptation_engine: VoiceAdaptationEngine,
    /// Reference voice database
    reference_voices: HashMap<String, ReferenceVoice>,
    /// Synthesis configuration
    config: ZeroShotConfig,
}

/// Voice adaptation engine for zero-shot synthesis
pub struct VoiceAdaptationEngine {
    /// Device for computation
    device: Device,
    /// Adaptation method
    method: AdaptationMethod,
    /// Speaker encoder for extracting voice features
    speaker_encoder: SpeakerEncoder,
    /// Voice cloning capabilities
    voice_cloner: VoiceCloner,
}

/// Speaker encoder for extracting voice representations
pub struct SpeakerEncoder {
    /// Embedding dimension
    embedding_dim: usize,
    /// Feature extraction layers
    feature_extractors: Vec<FeatureExtractor>,
    /// Normalization parameters
    normalization: NormalizationParams,
}

/// Voice cloning system for zero-shot synthesis
pub struct VoiceCloner {
    /// Cloning strategy
    strategy: CloningStrategy,
    /// Quality thresholds
    quality_thresholds: QualityThresholds,
    /// Adaptation parameters
    adaptation_params: AdaptationParams,
}

/// Reference voice for zero-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceVoice {
    /// Voice identifier
    pub voice_id: String,
    /// Voice name or description
    pub voice_name: String,
    /// Audio samples for this voice
    pub audio_samples: Vec<AudioSample>,
    /// Extracted voice embedding
    pub voice_embedding: Vec<f32>,
    /// Voice characteristics
    pub characteristics: VoiceCharacteristics,
    /// Quality metrics
    pub quality_metrics: VoiceQualityMetrics,
    /// Supported languages
    pub languages: Vec<String>,
    /// Vocal range information
    pub vocal_range: VocalRange,
}

/// Audio sample with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSample {
    /// Audio data
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Transcription if available
    pub transcription: Option<String>,
    /// Phoneme sequence if available
    pub phonemes: Option<Vec<String>>,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
}

/// Vocal range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalRange {
    /// Lowest comfortable note (MIDI)
    pub lowest_note: u8,
    /// Highest comfortable note (MIDI)
    pub highest_note: u8,
    /// Optimal range start (MIDI)
    pub optimal_start: u8,
    /// Optimal range end (MIDI)
    pub optimal_end: u8,
    /// Break points between registers
    pub register_breaks: Vec<u8>,
}

/// Zero-shot synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotConfig {
    /// Adaptation method to use
    pub adaptation_method: AdaptationMethod,
    /// Quality vs speed tradeoff
    pub quality_mode: QualityMode,
    /// Number of reference samples to use
    pub num_reference_samples: usize,
    /// Adaptation learning rate
    pub adaptation_lr: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Enable voice similarity preservation
    pub preserve_similarity: bool,
    /// Enable prosody adaptation
    pub adapt_prosody: bool,
    /// Enable timbre adaptation
    pub adapt_timbre: bool,
}

/// Adaptation methods for zero-shot synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationMethod {
    /// Direct embedding interpolation
    EmbeddingInterpolation,
    /// Few-shot fine-tuning
    FewShotFineTuning,
    /// Meta-learning approach
    MetaLearning,
    /// Speaker adaptation layers
    SpeakerAdaptation,
    /// Hybrid approach
    Hybrid,
}

/// Quality vs speed modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMode {
    /// Fast synthesis with basic quality
    Fast,
    /// Balanced quality and speed
    Balanced,
    /// High quality with longer processing time
    HighQuality,
    /// Ultra-high quality for studio use
    Studio,
}

/// Feature extraction types
#[derive(Clone)]
pub enum FeatureExtractor {
    /// Mel-frequency cepstral coefficients
    MFCC {
        /// Number of MFCC coefficients to extract
        num_coeffs: usize,
    },
    /// Mel-spectrogram features
    MelSpectrogram {
        /// Number of mel frequency bands
        num_mels: usize,
    },
    /// Fundamental frequency tracking
    F0Tracking,
    /// Spectral centroid
    SpectralCentroid,
    /// Harmonic-to-noise ratio
    HNR,
    /// Voice quality features
    VoiceQuality,
}

/// Cloning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloningStrategy {
    /// Direct voice transfer
    DirectTransfer,
    /// Gradual adaptation
    GradualAdaptation,
    /// Multi-stage cloning
    MultiStage,
    /// Ensemble-based cloning
    Ensemble,
}

/// Quality thresholds for voice cloning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum speaker similarity
    pub min_speaker_similarity: f32,
    /// Minimum audio quality
    pub min_audio_quality: f32,
    /// Minimum naturalness score
    pub min_naturalness: f32,
    /// Maximum distortion allowed
    pub max_distortion: f32,
}

/// Adaptation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParams {
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Regularization strength
    pub regularization: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Dropout rate during adaptation
    pub dropout_rate: f32,
}

/// Normalization parameters for features
#[derive(Clone)]
pub struct NormalizationParams {
    /// Mean values for normalization
    pub mean: Vec<f32>,
    /// Standard deviation for normalization
    pub std: Vec<f32>,
    /// Min-max normalization bounds
    pub min_max: Option<(f32, f32)>,
}

/// Zero-shot synthesis request
#[derive(Debug, Clone)]
pub struct ZeroShotRequest {
    /// Target voice specification
    pub target_voice: TargetVoiceSpec,
    /// Musical content to synthesize
    pub content: SingingRequest,
    /// Synthesis configuration
    pub config: ZeroShotConfig,
    /// Additional parameters
    pub parameters: HashMap<String, f32>,
}

/// Target voice specification for zero-shot synthesis
#[derive(Debug, Clone)]
pub enum TargetVoiceSpec {
    /// Reference audio samples
    AudioSamples {
        /// Audio samples containing the target voice
        samples: Vec<AudioSample>,
        /// Optional textual description of the voice
        voice_description: Option<String>,
    },
    /// Voice characteristics description
    VoiceDescription {
        /// Explicit voice characteristics to use
        characteristics: VoiceCharacteristics,
        /// Preferred singing styles
        style_preferences: Vec<String>,
    },
    /// Existing reference voice
    ReferenceVoice {
        /// ID of the reference voice to use
        voice_id: String,
        /// Strength of adaptation (0.0-1.0+)
        adaptation_strength: f32,
    },
    /// Voice interpolation between multiple references
    VoiceInterpolation {
        /// IDs of voices to interpolate between
        voice_ids: Vec<String>,
        /// Interpolation weights for each voice
        weights: Vec<f32>,
    },
}

/// Zero-shot synthesis result
#[derive(Debug, Clone)]
pub struct ZeroShotResult {
    /// Synthesized audio
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Adapted voice characteristics
    pub adapted_voice: VoiceCharacteristics,
    /// Adaptation quality metrics
    pub adaptation_metrics: AdaptationMetrics,
    /// Processing statistics
    pub processing_stats: ProcessingStats,
}

/// Adaptation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    /// Speaker similarity to target (0.0-1.0)
    pub speaker_similarity: f32,
    /// Voice quality score (0.0-1.0)
    pub voice_quality: f32,
    /// Adaptation convergence (0.0-1.0)
    pub convergence: f32,
    /// Stability score (0.0-1.0)
    pub stability: f32,
    /// Naturalness preservation (0.0-1.0)
    pub naturalness: f32,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Adaptation time in milliseconds
    pub adaptation_time_ms: f64,
    /// Synthesis time in milliseconds
    pub synthesis_time_ms: f64,
    /// Number of adaptation iterations
    pub adaptation_iterations: usize,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
}

impl ZeroShotSynthesizer {
    /// Create a new zero-shot synthesizer
    pub fn new(device: Device) -> Result<Self, Error> {
        let base_model = SingingModelBuilder::new("zero-shot-base".to_string())
            .model_type(ModelType::Basic)
            .build()?;

        let adaptation_engine = VoiceAdaptationEngine::new(device)?;
        let config = ZeroShotConfig::default();

        Ok(Self {
            base_model,
            adaptation_engine,
            reference_voices: HashMap::new(),
            config,
        })
    }

    /// Add a reference voice to the database
    pub fn add_reference_voice(&mut self, voice: ReferenceVoice) -> Result<(), Error> {
        if voice.audio_samples.is_empty() {
            return Err(Error::Voice(
                "Reference voice must have at least one audio sample".to_string(),
            ));
        }

        // Validate voice embedding dimension
        if voice.voice_embedding.len() != 512 {
            return Err(Error::Voice(
                "Voice embedding must have 512 dimensions".to_string(),
            ));
        }

        self.reference_voices.insert(voice.voice_id.clone(), voice);
        Ok(())
    }

    /// Remove a reference voice from the database
    ///
    /// # Arguments
    ///
    /// * `voice_id` - The unique identifier of the voice to remove
    ///
    /// # Returns
    ///
    /// Returns `Some(ReferenceVoice)` if the voice was found and removed, `None` otherwise
    pub fn remove_reference_voice(&mut self, voice_id: &str) -> Option<ReferenceVoice> {
        self.reference_voices.remove(voice_id)
    }

    /// List all available reference voice identifiers in the database
    ///
    /// # Returns
    ///
    /// Returns a vector of voice ID strings for all registered reference voices
    pub fn list_reference_voices(&self) -> Vec<&str> {
        self.reference_voices.keys().map(|s| s.as_str()).collect()
    }

    /// Perform zero-shot singing synthesis with voice adaptation
    ///
    /// This method adapts the base model to the target voice specification and
    /// synthesizes singing audio with the adapted voice characteristics.
    ///
    /// # Arguments
    ///
    /// * `request` - Zero-shot synthesis request containing target voice, content, and configuration
    ///
    /// # Returns
    ///
    /// Returns a `ZeroShotResult` containing synthesized audio, adapted voice characteristics,
    /// and quality metrics
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Voice adaptation fails
    /// - Synthesis fails
    /// - Target voice specification is invalid
    pub async fn synthesize_zero_shot(
        &self,
        request: ZeroShotRequest,
    ) -> Result<ZeroShotResult, Error> {
        let start_time = std::time::Instant::now();

        // Extract target voice features
        let adaptation_start = std::time::Instant::now();
        let adapted_voice = self
            .adapt_voice(&request.target_voice, &request.config)
            .await?;
        let adaptation_time = adaptation_start.elapsed().as_millis() as f64;

        // Synthesize with adapted voice
        let synthesis_start = std::time::Instant::now();
        let mut synthesis_request = request.content.clone();
        synthesis_request.voice = adapted_voice.clone();

        // Use the base model for synthesis (in a real implementation, this would use the adapted model)
        let response = self
            .synthesize_with_adapted_voice(synthesis_request)
            .await?;
        let synthesis_time = synthesis_start.elapsed().as_millis() as f64;

        // Calculate adaptation metrics
        let adaptation_metrics =
            self.calculate_adaptation_metrics(&adapted_voice, &request.target_voice)?;

        let total_time = start_time.elapsed().as_millis() as f64;

        Ok(ZeroShotResult {
            audio: response.audio,
            sample_rate: response.sample_rate,
            adapted_voice,
            adaptation_metrics,
            processing_stats: ProcessingStats {
                total_time_ms: total_time,
                adaptation_time_ms: adaptation_time,
                synthesis_time_ms: synthesis_time,
                adaptation_iterations: request.config.adaptation_steps,
                memory_usage_mb: 256.0, // Placeholder
            },
        })
    }

    /// Adapt voice characteristics to target
    async fn adapt_voice(
        &self,
        target_spec: &TargetVoiceSpec,
        config: &ZeroShotConfig,
    ) -> Result<VoiceCharacteristics, Error> {
        match target_spec {
            TargetVoiceSpec::AudioSamples { samples, .. } => {
                self.adaptation_engine
                    .adapt_from_audio(samples, config)
                    .await
            }
            TargetVoiceSpec::VoiceDescription {
                characteristics, ..
            } => {
                // Direct adaptation from characteristics
                Ok(characteristics.clone())
            }
            TargetVoiceSpec::ReferenceVoice {
                voice_id,
                adaptation_strength,
            } => {
                let reference = self.reference_voices.get(voice_id).ok_or_else(|| {
                    Error::Voice(format!("Reference voice not found: {}", voice_id))
                })?;

                self.adaptation_engine
                    .adapt_from_reference(reference, *adaptation_strength, config)
                    .await
            }
            TargetVoiceSpec::VoiceInterpolation { voice_ids, weights } => {
                self.adaptation_engine
                    .interpolate_voices(voice_ids, weights, &self.reference_voices, config)
                    .await
            }
        }
    }

    /// Synthesize with adapted voice characteristics
    async fn synthesize_with_adapted_voice(
        &self,
        request: SingingRequest,
    ) -> Result<SingingResponse, Error> {
        // Placeholder implementation - in reality, this would use the adapted model
        // For now, use a simple synthesis approach

        let sample_rate = request.sample_rate;
        let duration_samples = (request
            .target_duration
            .unwrap_or(std::time::Duration::from_secs(3))
            .as_secs_f32()
            * sample_rate as f32) as usize;

        // Generate simple sine wave placeholder
        let mut audio = Vec::with_capacity(duration_samples);
        for i in 0..duration_samples {
            let t = i as f32 / sample_rate as f32;
            let frequency = 440.0; // A4
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
            audio.push(sample);
        }

        Ok(SingingResponse {
            audio,
            sample_rate,
            duration: std::time::Duration::from_secs_f32(
                duration_samples as f32 / sample_rate as f32,
            ),
            voice: request.voice,
            technique: request.technique,
            stats: crate::types::SingingStats::default(),
            metadata: HashMap::new(),
        })
    }

    /// Calculate adaptation quality metrics
    fn calculate_adaptation_metrics(
        &self,
        adapted_voice: &VoiceCharacteristics,
        target_spec: &TargetVoiceSpec,
    ) -> Result<AdaptationMetrics, Error> {
        // Placeholder implementation for adaptation metrics
        // In reality, this would compare the adapted voice with the target

        let speaker_similarity = match target_spec {
            TargetVoiceSpec::AudioSamples { .. } => 0.85,
            TargetVoiceSpec::VoiceDescription { .. } => 0.90,
            TargetVoiceSpec::ReferenceVoice { .. } => 0.88,
            TargetVoiceSpec::VoiceInterpolation { .. } => 0.82,
        };

        Ok(AdaptationMetrics {
            speaker_similarity,
            voice_quality: 0.87,
            convergence: 0.92,
            stability: 0.89,
            naturalness: 0.86,
        })
    }

    /// Create a reference voice from audio samples
    ///
    /// This method analyzes audio samples to create a reference voice that can be used
    /// for zero-shot synthesis. It extracts voice embeddings, analyzes vocal range,
    /// and calculates quality metrics.
    ///
    /// # Arguments
    ///
    /// * `voice_id` - Unique identifier for the voice
    /// * `voice_name` - Human-readable name for the voice
    /// * `audio_samples` - Vector of audio samples containing the voice
    /// * `characteristics` - Voice characteristics (timbre, range, etc.)
    ///
    /// # Returns
    ///
    /// Returns a `ReferenceVoice` that can be added to the synthesizer's database
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No audio samples are provided
    /// - Voice embedding extraction fails
    /// - Vocal range analysis fails
    pub fn create_reference_voice(
        voice_id: String,
        voice_name: String,
        audio_samples: Vec<AudioSample>,
        characteristics: VoiceCharacteristics,
    ) -> Result<ReferenceVoice, Error> {
        if audio_samples.is_empty() {
            return Err(Error::Voice(
                "At least one audio sample is required".to_string(),
            ));
        }

        // Extract voice embedding from audio samples
        let voice_embedding = Self::extract_voice_embedding(&audio_samples)?;

        // Analyze vocal range
        let vocal_range = Self::analyze_vocal_range(&audio_samples)?;

        // Calculate quality metrics
        let quality_metrics = Self::calculate_voice_quality(&audio_samples)?;

        Ok(ReferenceVoice {
            voice_id,
            voice_name,
            audio_samples,
            voice_embedding,
            characteristics,
            quality_metrics,
            languages: vec!["en".to_string()], // Default to English
            vocal_range,
        })
    }

    /// Extract a 512-dimensional voice embedding from audio samples.
    ///
    /// Computes per-frame MFCCs (Hann window, FFT, mel filterbank, log, DCT-II) across
    /// all frames of all samples, aggregates their mean, standard deviation and mean
    /// delta, tiles the aggregate into a 512-dimensional vector and L2-normalizes it.
    /// The result is fully deterministic for a given input.
    fn extract_voice_embedding(samples: &[AudioSample]) -> Result<Vec<f32>, Error> {
        Ok(voice_embedding_from_samples(samples))
    }

    /// Analyze the vocal range from audio samples using autocorrelation F0 detection.
    ///
    /// Estimates per-frame F0 over voiced frames, converts to MIDI notes, derives the
    /// comfortable range from robust percentiles, the optimal range from the densest
    /// contiguous region, and register breaks from histogram valleys.
    fn analyze_vocal_range(samples: &[AudioSample]) -> Result<VocalRange, Error> {
        Ok(vocal_range_from_samples(samples))
    }

    /// Calculate voice quality metrics from audio samples using real DSP.
    ///
    /// Derives vibrato rate/depth from the FFT of the detrended voiced-F0 contour,
    /// breathiness from the harmonic-to-noise ratio, roughness from F0 jitter,
    /// brightness from the spectral centroid, and vocal range from the analyzed span.
    fn calculate_voice_quality(samples: &[AudioSample]) -> Result<VoiceQualityMetrics, Error> {
        Ok(voice_quality_from_samples(samples))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Voice-analysis DSP helpers
//
// These free functions implement the real signal processing used by
// `ZeroShotSynthesizer`. They reuse the crate-wide autocorrelation F0 detector
// (`detect_f0_autocorr_frame`) and the SciRS2 FFT/DCT primitives.
// ──────────────────────────────────────────────────────────────────────────

/// Number of MFCC coefficients retained per frame.
const MFCC_COEFFS: usize = 13;
/// Number of triangular mel filterbank channels.
const MEL_FILTERS: usize = 26;
/// Analysis frame length (samples) for spectral features (MFCC, centroid).
const SPEC_FRAME_LEN: usize = 1024;
/// Hop length (samples) for spectral features.
const SPEC_HOP: usize = 256;
/// Analysis frame length (samples) for autocorrelation F0 estimation.
const F0_FRAME_LEN: usize = 2048;
/// Hop length (samples) for autocorrelation F0 estimation.
const F0_HOP: usize = 512;
/// Dimensionality of the voice embedding vector.
const EMBEDDING_DIM: usize = 512;

/// Peak-normalize a frame so its maximum absolute value is 1.0.
///
/// This makes the amplitude-dependent raw autocorrelation threshold inside
/// [`detect_f0_autocorr_frame`] robust to the input gain.
fn peak_normalize(frame: &[f32]) -> Vec<f32> {
    let peak = frame.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak > 1e-9 {
        frame.iter().map(|&x| x / peak).collect()
    } else {
        frame.to_vec()
    }
}

/// Normalized autocorrelation coefficient of a frame at a given lag, in `[0, 1]`.
///
/// Used as a harmonic-to-noise ratio (HNR) proxy at the detected pitch period.
fn normalized_autocorr_at_lag(frame: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag >= frame.len() {
        return 0.0;
    }
    let mean = frame.iter().sum::<f32>() / frame.len() as f32;
    let n_ov = frame.len() - lag;
    let mut cross = 0.0f32;
    let mut left_sq = 0.0f32;
    let mut right_sq = 0.0f32;
    for i in 0..n_ov {
        let a = frame[i] - mean;
        let b = frame[i + lag] - mean;
        cross += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    let denom = (left_sq * right_sq).sqrt();
    if denom > 1e-12 {
        (cross / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Analyze a single frame, returning `(f0_hz, hnr)` if the frame is voiced.
///
/// [`detect_f0_autocorr_frame`] supplies the coarse integer-lag pitch and the
/// voicing decision. Because it maximizes a *raw* autocorrelation it can lock
/// onto an integer multiple of the true period (an octave-down error) that
/// varies with frame phase; this is corrected by preferring a sub-period whose
/// *normalized* autocorrelation is comparable. The result is then refined to
/// sub-sample precision by parabolic interpolation, removing the pitch jitter
/// that would otherwise masquerade as vibrato on steady tones.
fn analyze_voiced_frame(frame: &[f32], sample_rate: u32) -> Option<(f32, f32)> {
    let normalized = peak_normalize(frame);
    let coarse_f0 = detect_f0_autocorr_frame(&normalized, sample_rate as f32);
    if coarse_f0 <= 0.0 {
        return None;
    }
    let mut period = (sample_rate as f32 / coarse_f0).round() as usize;
    let min_lag = ((sample_rate as f32 / 800.0) as usize).max(2);

    // Octave-error correction: prefer the fundamental when a sub-period (a higher
    // F0) correlates comparably. Larger divisors (higher F0) are tried first.
    let base_r = normalized_autocorr_at_lag(&normalized, period);
    for div in [4usize, 3, 2] {
        let candidate = period / div;
        if candidate >= min_lag
            && normalized_autocorr_at_lag(&normalized, candidate) >= 0.85 * base_r.max(1e-6)
        {
            period = candidate;
            break;
        }
    }

    let hnr = normalized_autocorr_at_lag(&normalized, period);

    // Sub-sample refinement via parabolic interpolation of the normalized peak.
    let refined_period = if period >= 2 && period + 1 < normalized.len() {
        let r_minus = normalized_autocorr_at_lag(&normalized, period - 1);
        let r_zero = normalized_autocorr_at_lag(&normalized, period);
        let r_plus = normalized_autocorr_at_lag(&normalized, period + 1);
        let denom = r_minus - 2.0 * r_zero + r_plus;
        let delta = if denom < -1e-6 {
            (0.5 * (r_minus - r_plus) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        period as f32 + delta
    } else {
        period as f32
    };

    Some((sample_rate as f32 / refined_period, hnr))
}

/// Compute `(f0_hz, hnr)` pairs for all voiced frames of an audio buffer.
fn voiced_frames(audio: &[f32], sample_rate: u32) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    if audio.len() < F0_FRAME_LEN {
        if audio.len() >= 64 {
            if let Some(v) = analyze_voiced_frame(audio, sample_rate) {
                out.push(v);
            }
        }
        return out;
    }
    let mut pos = 0;
    while pos + F0_FRAME_LEN <= audio.len() {
        if let Some(v) = analyze_voiced_frame(&audio[pos..pos + F0_FRAME_LEN], sample_rate) {
            out.push(v);
        }
        pos += F0_HOP;
    }
    out
}

/// Compute mel-frequency cepstral coefficients for a single frame.
///
/// Mirrors the established crate MFCC pipeline: Hann window, FFT power spectrum,
/// triangular mel filterbank, log compression and an orthonormal DCT-II.
fn mfcc_frame(frame: &[f32], sample_rate: u32) -> [f32; MFCC_COEFFS] {
    let len = frame.len();
    if len < 2 {
        return [0.0f32; MFCC_COEFFS];
    }

    let windowed: Vec<f64> = frame
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (len - 1) as f64).cos());
            s as f64 * w
        })
        .collect();

    let complex_in: Vec<Complex<f64>> = windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let fft_out = match scirs2_fft::fft(&complex_in, None) {
        Ok(v) => v,
        Err(_) => return [0.0f32; MFCC_COEFFS],
    };

    let n_bins = len / 2 + 1;
    let power: Vec<f64> = fft_out[..n_bins]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).max(1e-30))
        .collect();

    let nyquist = sample_rate as f64 / 2.0;
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(nyquist);
    let mel_pts: Vec<f64> = (0..=MEL_FILTERS + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (MEL_FILTERS + 1) as f64)
        .collect();
    let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_pts: Vec<usize> = hz_pts
        .iter()
        .map(|&hz| ((hz / nyquist) * (n_bins - 1) as f64).round() as usize)
        .collect();

    let mut filt = [0.0f64; MEL_FILTERS];
    for m in 0..MEL_FILTERS {
        let start = bin_pts[m];
        let center = bin_pts[m + 1];
        let end = bin_pts[m + 2];
        for k in start..center {
            if k < power.len() && center > start {
                filt[m] += power[k] * (k - start) as f64 / (center - start) as f64;
            }
        }
        for k in center..end {
            if k < power.len() && end > center {
                filt[m] += power[k] * (end - k) as f64 / (end - center) as f64;
            }
        }
        filt[m] = filt[m].max(1e-30).ln();
    }

    let dct_out = match scirs2_fft::dct(&filt, None, Some("ortho")) {
        Ok(v) => v,
        Err(_) => return [0.0f32; MFCC_COEFFS],
    };
    let mut mfcc = [0.0f32; MFCC_COEFFS];
    for (i, m) in mfcc.iter_mut().enumerate() {
        *m = dct_out.get(i).copied().unwrap_or(0.0) as f32;
    }
    mfcc
}

/// Build a deterministic 512-dimensional voice embedding from audio samples.
fn voice_embedding_from_samples(samples: &[AudioSample]) -> Vec<f32> {
    let mut frames: Vec<[f32; MFCC_COEFFS]> = Vec::new();
    for sample in samples {
        let audio = &sample.audio;
        if audio.len() < SPEC_FRAME_LEN {
            if !audio.is_empty() {
                let mut padded = audio.clone();
                padded.resize(SPEC_FRAME_LEN, 0.0);
                frames.push(mfcc_frame(&padded, sample.sample_rate));
            }
            continue;
        }
        let mut pos = 0;
        while pos + SPEC_FRAME_LEN <= audio.len() {
            frames.push(mfcc_frame(
                &audio[pos..pos + SPEC_FRAME_LEN],
                sample.sample_rate,
            ));
            pos += SPEC_HOP;
        }
    }

    if frames.is_empty() {
        return vec![0.0f32; EMBEDDING_DIM];
    }

    let frame_count = frames.len() as f32;

    let mut mean = [0.0f32; MFCC_COEFFS];
    for f in &frames {
        for (acc, &v) in mean.iter_mut().zip(f.iter()) {
            *acc += v;
        }
    }
    for acc in mean.iter_mut() {
        *acc /= frame_count;
    }

    let mut var = [0.0f32; MFCC_COEFFS];
    for f in &frames {
        for ((acc, &v), &mu) in var.iter_mut().zip(f.iter()).zip(mean.iter()) {
            let d = v - mu;
            *acc += d * d;
        }
    }
    let std_dev: [f32; MFCC_COEFFS] = std::array::from_fn(|i| (var[i] / frame_count).sqrt());

    let mut delta = [0.0f32; MFCC_COEFFS];
    if frames.len() >= 2 {
        for pair in frames.windows(2) {
            for ((acc, &next), &prev) in delta.iter_mut().zip(pair[1].iter()).zip(pair[0].iter()) {
                *acc += next - prev;
            }
        }
        let delta_count = (frames.len() - 1) as f32;
        for acc in delta.iter_mut() {
            *acc /= delta_count;
        }
    }

    let mut base = Vec::with_capacity(3 * MFCC_COEFFS);
    base.extend_from_slice(&mean);
    base.extend_from_slice(&std_dev);
    base.extend_from_slice(&delta);

    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    for (i, e) in embedding.iter_mut().enumerate() {
        *e = base[i % base.len()];
    }

    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for e in embedding.iter_mut() {
            *e /= norm;
        }
    }
    embedding
}

/// Convert a fundamental frequency in Hz to the nearest MIDI note number.
fn hz_to_midi(f0: f32) -> Option<u8> {
    if f0 <= 0.0 {
        return None;
    }
    let midi = (69.0 + 12.0 * (f0 / 440.0).log2()).round();
    if midi.is_finite() && (0.0..=127.0).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

/// Return the value at the given percentile (0–100) of a sorted slice.
fn percentile(sorted: &[u8], pct: f32) -> u8 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Detect register-break MIDI notes as histogram valleys between populated regions.
fn detect_register_breaks(hist: &[u32; 128], lowest: u8, highest: u8, mode_count: u32) -> Vec<u8> {
    let mut breaks = Vec::new();
    if highest <= lowest + 2 {
        return breaks;
    }
    let valley_thr = ((mode_count as f32) * 0.35) as u32;
    for n in (lowest + 1)..highest {
        let c = hist[n as usize];
        let prev = hist[(n - 1) as usize];
        let next = hist[(n + 1) as usize];
        let is_valley = c <= prev && c <= next && c < valley_thr && (prev > 0 || next > 0);
        let spaced = breaks.last().is_none_or(|&b: &u8| n > b + 1);
        if is_valley && spaced {
            breaks.push(n);
            if breaks.len() >= 3 {
                break;
            }
        }
    }
    breaks
}

/// Default vocal range used when too few voiced frames are available.
fn default_vocal_range() -> VocalRange {
    VocalRange {
        lowest_note: 48,
        highest_note: 84,
        optimal_start: 60,
        optimal_end: 72,
        register_breaks: vec![60, 67],
    }
}

/// Analyze vocal range from audio samples via autocorrelation F0 detection.
fn vocal_range_from_samples(samples: &[AudioSample]) -> VocalRange {
    let mut midi_notes: Vec<u8> = Vec::new();
    for sample in samples {
        for (f0, _) in voiced_frames(&sample.audio, sample.sample_rate) {
            if let Some(m) = hz_to_midi(f0) {
                midi_notes.push(m);
            }
        }
    }

    if midi_notes.len() < 8 {
        return default_vocal_range();
    }

    let mut sorted = midi_notes.clone();
    sorted.sort_unstable();
    let lowest = percentile(&sorted, 5.0);
    let highest = percentile(&sorted, 95.0).max(lowest);
    let lowest = lowest.min(highest);

    let mut hist = [0u32; 128];
    for &m in &midi_notes {
        hist[m as usize] += 1;
    }

    let mode = (lowest..=highest)
        .max_by_key(|&n| hist[n as usize])
        .unwrap_or(lowest);
    let mode_count = hist[mode as usize].max(1);

    // Densest contiguous region: expand from the mode while bins stay populated.
    let thr = ((mode_count as f32) * 0.25).ceil() as u32;
    let mut start = mode;
    while start > lowest && hist[(start - 1) as usize] >= thr {
        start -= 1;
    }
    let mut end = mode;
    while end < highest && hist[(end + 1) as usize] >= thr {
        end += 1;
    }

    let register_breaks = detect_register_breaks(&hist, lowest, highest, mode_count);
    let register_breaks = if register_breaks.is_empty() {
        vec![(lowest as u16 + (highest as u16 - lowest as u16) / 2) as u8]
    } else {
        register_breaks
    };

    VocalRange {
        lowest_note: lowest,
        highest_note: highest,
        optimal_start: start,
        optimal_end: end.max(start),
        register_breaks,
    }
}

/// FFT-based spectral centroid (Hz) of a single Hann-windowed frame.
///
/// Uses the real FFT ([`scirs2_fft::rfft`]), which returns the `len / 2 + 1`
/// non-redundant bins directly.
fn frame_spectral_centroid(frame: &[f32], sample_rate: u32) -> Option<f32> {
    let len = frame.len();
    if len < 2 {
        return None;
    }
    let windowed: Vec<f32> = frame
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
            s * w
        })
        .collect();
    let spectrum = scirs2_fft::rfft(&windowed, None).ok()?;
    let mut weighted = 0.0f64;
    let mut magnitude = 0.0f64;
    for (k, c) in spectrum.iter().enumerate() {
        let freq = k as f64 * sample_rate as f64 / len as f64;
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        weighted += freq * mag;
        magnitude += mag;
    }
    if magnitude > 1e-12 {
        Some((weighted / magnitude) as f32)
    } else {
        None
    }
}

/// Mean normalized spectral centroid (centroid / Nyquist) across all frames.
fn mean_brightness(samples: &[AudioSample]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for sample in samples {
        let audio = &sample.audio;
        if audio.len() < SPEC_FRAME_LEN {
            continue;
        }
        let nyquist = sample.sample_rate as f32 / 2.0;
        let mut pos = 0;
        while pos + SPEC_FRAME_LEN <= audio.len() {
            if let Some(centroid) =
                frame_spectral_centroid(&audio[pos..pos + SPEC_FRAME_LEN], sample.sample_rate)
            {
                sum += (centroid / nyquist).clamp(0.0, 1.0);
                count += 1;
            }
            pos += SPEC_HOP;
        }
    }
    if count > 0 {
        (sum / count as f32).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Map a mean fundamental frequency (Hz) onto the conventional vocal type.
///
/// Thresholds follow the standard vocal-fundamental ranges (see
/// [`VoiceAdaptationEngine::estimate_voice_type`] for the table).
fn classify_voice_by_f0(mean_f0: f32) -> VoiceType {
    if mean_f0 < 130.0 {
        VoiceType::Bass
    } else if mean_f0 < 165.0 {
        VoiceType::Baritone
    } else if mean_f0 < 220.0 {
        VoiceType::Tenor
    } else if mean_f0 < 260.0 {
        VoiceType::Alto
    } else if mean_f0 < 350.0 {
        VoiceType::MezzoSoprano
    } else {
        VoiceType::Soprano
    }
}

/// Estimate `(vibrato_rate_hz, vibrato_depth_cents)` from a voiced-F0 contour.
///
/// The contour is linearly detrended, transformed via FFT, and the spectral peak
/// within the 4–8 Hz vibrato band selects the rate; its magnitude yields the depth.
fn estimate_vibrato(f0s: &[f32], frame_rate: f32) -> (f32, f32) {
    let n = f0s.len();
    if n < 10 || frame_rate <= 0.0 {
        return (5.0, 0.0);
    }
    let mean_f0 = f0s.iter().sum::<f32>() / n as f32;

    // Least-squares linear detrend.
    let x_mean = (n - 1) as f32 / 2.0;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &f) in f0s.iter().enumerate() {
        let xi = i as f32 - x_mean;
        num += xi * (f - mean_f0);
        den += xi * xi;
    }
    let slope = if den > 1e-12 { num / den } else { 0.0 };
    let intercept = mean_f0 - slope * x_mean;

    let detrended: Vec<Complex<f64>> = f0s
        .iter()
        .enumerate()
        .map(|(i, &f)| Complex::new((f - (slope * i as f32 + intercept)) as f64, 0.0))
        .collect();
    let spectrum = match scirs2_fft::fft(&detrended, Some(n)) {
        Ok(v) => v,
        Err(_) => return (5.0, 0.0),
    };

    let half = n / 2;
    let k_lo = ((4.0 * n as f32 / frame_rate).floor() as usize).max(1);
    let k_hi = ((8.0 * n as f32 / frame_rate).ceil() as usize).min(half);
    if k_lo > k_hi {
        return (5.0, 0.0);
    }

    let mut peak_mag = 0.0f64;
    let mut peak_k = k_lo;
    for k in k_lo..=k_hi {
        let mag = spectrum[k].norm();
        if mag > peak_mag {
            peak_mag = mag;
            peak_k = k;
        }
    }

    let rate = (peak_k as f32 * frame_rate / n as f32).clamp(4.0, 8.0);
    let peak_amp = (peak_mag / (n as f64 / 2.0).max(1.0)) as f32;
    let depth = if mean_f0 > 1.0 {
        (1200.0 * ((mean_f0 + peak_amp) / mean_f0).log2()).clamp(0.0, 200.0)
    } else {
        0.0
    };
    (rate, depth)
}

/// Calculate voice quality metrics from audio samples using real DSP.
fn voice_quality_from_samples(samples: &[AudioSample]) -> VoiceQualityMetrics {
    let sample_rate = samples.first().map(|s| s.sample_rate).unwrap_or(22050);

    let mut f0s: Vec<f32> = Vec::new();
    let mut hnrs: Vec<f32> = Vec::new();
    for sample in samples {
        for (f0, hnr) in voiced_frames(&sample.audio, sample.sample_rate) {
            f0s.push(f0);
            hnrs.push(hnr);
        }
    }

    if f0s.is_empty() {
        return VoiceQualityMetrics::default();
    }

    let range = vocal_range_from_samples(samples);
    let vocal_range = (range.highest_note as f32 - range.lowest_note as f32).max(0.0);

    let frame_rate = sample_rate as f32 / F0_HOP as f32;
    let (vibrato_rate, vibrato_depth) = estimate_vibrato(&f0s, frame_rate);

    let mean_hnr = hnrs.iter().sum::<f32>() / hnrs.len() as f32;
    let breathiness = (1.0 - mean_hnr).clamp(0.0, 1.0);

    let roughness = if f0s.len() < 2 {
        0.0
    } else {
        let jitter = f0s
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() / w[0].max(1.0))
            .sum::<f32>()
            / (f0s.len() - 1) as f32;
        jitter.clamp(0.0, 1.0)
    };

    let brightness = mean_brightness(samples);

    VoiceQualityMetrics {
        vocal_range,
        vibrato_rate,
        vibrato_depth,
        breathiness,
        roughness,
        brightness,
    }
}

impl VoiceAdaptationEngine {
    /// Create a new voice adaptation engine
    pub fn new(device: Device) -> Result<Self, Error> {
        let speaker_encoder = SpeakerEncoder::new(512)?;
        let voice_cloner = VoiceCloner::new()?;

        Ok(Self {
            device,
            method: AdaptationMethod::Hybrid,
            speaker_encoder,
            voice_cloner,
        })
    }

    /// Adapt voice characteristics from raw audio samples
    ///
    /// Extracts features from audio samples and adapts voice characteristics
    /// including voice type, pitch range, and vibrato parameters.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to analyze
    /// * `config` - Zero-shot configuration for adaptation
    ///
    /// # Returns
    ///
    /// Returns adapted `VoiceCharacteristics` extracted from the audio
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction or voice estimation fails
    pub async fn adapt_from_audio(
        &self,
        samples: &[AudioSample],
        config: &ZeroShotConfig,
    ) -> Result<VoiceCharacteristics, Error> {
        // Extract a speaker-feature vector (used by the vibrato estimator).
        let features = self.speaker_encoder.extract_features(samples)?;

        // Adapt voice characteristics based on real acoustic analysis.
        let mut adapted_voice = VoiceCharacteristics::default();

        // Classify voice type from the measured F0 / spectral content.
        adapted_voice.voice_type = self.estimate_voice_type(samples)?;

        // Adapt other characteristics.
        adapted_voice.range = self.estimate_pitch_range(samples)?;
        adapted_voice.vibrato_frequency = self.estimate_vibrato_rate(&features)?;

        Ok(adapted_voice)
    }

    /// Adapt voice characteristics from a reference voice
    ///
    /// Uses an existing reference voice to generate adapted voice characteristics,
    /// applying the specified adaptation strength to control how much the characteristics
    /// are modified.
    ///
    /// # Arguments
    ///
    /// * `reference` - Reference voice to adapt from
    /// * `adaptation_strength` - Strength of adaptation (0.0-1.0+)
    /// * `config` - Zero-shot configuration for adaptation
    ///
    /// # Returns
    ///
    /// Returns adapted `VoiceCharacteristics` based on the reference voice
    ///
    /// # Errors
    ///
    /// Returns an error if adaptation fails
    pub async fn adapt_from_reference(
        &self,
        reference: &ReferenceVoice,
        adaptation_strength: f32,
        config: &ZeroShotConfig,
    ) -> Result<VoiceCharacteristics, Error> {
        // Blend reference characteristics with adaptation strength
        let mut adapted = reference.characteristics.clone();

        // Apply adaptation strength
        adapted.range.0 *= adaptation_strength;
        adapted.range.1 *= adaptation_strength;
        adapted.vibrato_frequency *= adaptation_strength;

        Ok(adapted)
    }

    /// Interpolate voice characteristics between multiple reference voices
    ///
    /// Creates a blended voice by combining characteristics from multiple reference voices
    /// using weighted interpolation. This allows creating new voices that blend properties
    /// from existing voices.
    ///
    /// # Arguments
    ///
    /// * `voice_ids` - IDs of reference voices to interpolate
    /// * `weights` - Interpolation weights for each voice (should sum to 1.0, but will be normalized)
    /// * `reference_voices` - Database of available reference voices
    /// * `config` - Zero-shot configuration for adaptation
    ///
    /// # Returns
    ///
    /// Returns interpolated `VoiceCharacteristics` combining the reference voices
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Voice IDs and weights have different lengths
    /// - Total weight is zero or negative
    /// - Any reference voice ID is not found
    pub async fn interpolate_voices(
        &self,
        voice_ids: &[String],
        weights: &[f32],
        reference_voices: &HashMap<String, ReferenceVoice>,
        config: &ZeroShotConfig,
    ) -> Result<VoiceCharacteristics, Error> {
        if voice_ids.len() != weights.len() {
            return Err(Error::Voice(
                "Voice IDs and weights must have the same length".to_string(),
            ));
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(Error::Voice("Total weight must be positive".to_string()));
        }

        let mut interpolated = VoiceCharacteristics::default();
        let mut accumulated_pitch_min = 0.0;
        let mut accumulated_pitch_max = 0.0;
        let mut accumulated_vibrato = 0.0;

        for (voice_id, &weight) in voice_ids.iter().zip(weights.iter()) {
            let reference = reference_voices
                .get(voice_id)
                .ok_or_else(|| Error::Voice(format!("Reference voice not found: {}", voice_id)))?;

            let normalized_weight = weight / total_weight;
            accumulated_pitch_min += reference.characteristics.range.0 * normalized_weight;
            accumulated_pitch_max += reference.characteristics.range.1 * normalized_weight;
            accumulated_vibrato += reference.characteristics.vibrato_frequency * normalized_weight;
        }

        interpolated.range = (accumulated_pitch_min, accumulated_pitch_max);
        interpolated.vibrato_frequency = accumulated_vibrato;

        Ok(interpolated)
    }

    /// Classify the voice type from real acoustic measurements.
    ///
    /// The decision is driven primarily by the mean fundamental frequency over
    /// all voiced frames (autocorrelation F0 via [`voiced_frames`]), mapped onto
    /// the conventional vocal-fundamental ranges:
    ///
    /// | mean F0 (Hz) | voice type   |
    /// |--------------|--------------|
    /// | `< 130`      | Bass         |
    /// | `130 – 165`  | Baritone     |
    /// | `165 – 220`  | Tenor        |
    /// | `220 – 260`  | Alto         |
    /// | `260 – 350`  | MezzoSoprano |
    /// | `>= 350`     | Soprano      |
    ///
    /// When no voiced frames are found (silent or noise-only input) the mean
    /// normalized spectral centroid (brightness, via [`mean_brightness`]) is
    /// mapped onto a pseudo mean-F0 as a fallback, so a bright spectrum yields a
    /// high voice and a dark spectrum a low voice.
    fn estimate_voice_type(&self, samples: &[AudioSample]) -> Result<VoiceType, Error> {
        let mut f0_sum = 0.0f32;
        let mut f0_count = 0usize;
        for sample in samples {
            for (f0, _) in voiced_frames(&sample.audio, sample.sample_rate) {
                f0_sum += f0;
                f0_count += 1;
            }
        }

        let voice_type = if f0_count > 0 {
            classify_voice_by_f0(f0_sum / f0_count as f32)
        } else {
            // Brightness fallback: map normalized centroid [0, 1] -> pseudo F0.
            classify_voice_by_f0(90.0 + mean_brightness(samples) * 360.0)
        };
        Ok(voice_type)
    }

    /// Estimate the pitch range (Hz) from real F0 analysis.
    ///
    /// Runs the autocorrelation F0 detector over every frame of every sample
    /// ([`voiced_frames`]) and returns the robust `(low, high)` bounds as the
    /// 5th and 95th percentiles of the voiced-F0 distribution. Percentiles reject
    /// octave-jump and noise outliers that raw min/max would capture. Falls back
    /// to a default mid-voice range when no voiced frames are detected.
    fn estimate_pitch_range(&self, samples: &[AudioSample]) -> Result<(f32, f32), Error> {
        let mut f0s: Vec<f32> = Vec::new();
        for sample in samples {
            for (f0, _) in voiced_frames(&sample.audio, sample.sample_rate) {
                if f0 > 0.0 {
                    f0s.push(f0);
                }
            }
        }

        if f0s.is_empty() {
            // Default comfortable mid-voice range (C3 – C5).
            return Ok((130.81, 523.25));
        }

        f0s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |p: f32| -> f32 {
            let idx = ((p / 100.0) * (f0s.len() - 1) as f32).round() as usize;
            f0s[idx.min(f0s.len() - 1)]
        };
        let low = pct(5.0);
        let high = pct(95.0);
        if high > low {
            Ok((low, high))
        } else {
            // Degenerate (near-constant F0): widen slightly around the value.
            Ok((low * 0.97, high * 1.03))
        }
    }

    /// Estimate vibrato rate from features
    fn estimate_vibrato_rate(&self, features: &[f32]) -> Result<f32, Error> {
        // Placeholder implementation
        let rate = features.iter().take(10).sum::<f32>() * 10.0;
        Ok(rate.clamp(3.0, 8.0))
    }
}

impl SpeakerEncoder {
    /// Create a new speaker encoder
    pub fn new(embedding_dim: usize) -> Result<Self, Error> {
        let feature_extractors = vec![
            FeatureExtractor::MFCC { num_coeffs: 13 },
            FeatureExtractor::MelSpectrogram { num_mels: 80 },
            FeatureExtractor::F0Tracking,
            FeatureExtractor::SpectralCentroid,
        ];

        let normalization = NormalizationParams {
            mean: vec![0.0; embedding_dim],
            std: vec![1.0; embedding_dim],
            min_max: Some((-1.0, 1.0)),
        };

        Ok(Self {
            embedding_dim,
            feature_extractors,
            normalization,
        })
    }

    /// Extract voice features from audio samples
    ///
    /// Processes audio samples through configured feature extractors to generate
    /// a fixed-dimensional feature vector suitable for speaker encoding.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to extract features from (uses up to first 5 samples)
    ///
    /// # Returns
    ///
    /// Returns a feature vector with dimension matching `embedding_dim`
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction fails
    pub fn extract_features(&self, samples: &[AudioSample]) -> Result<Vec<f32>, Error> {
        let mut features = Vec::new();

        for sample in samples.iter().take(5) {
            // Limit to first 5 samples
            let sample_features =
                self.extract_sample_features(&sample.audio, sample.sample_rate)?;
            features.extend(sample_features);
        }

        // Pad or truncate to embedding dimension
        features.resize(self.embedding_dim, 0.0);

        Ok(features)
    }

    /// Extract features from a single audio sample
    fn extract_sample_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>, Error> {
        let mut features = Vec::new();

        // Basic audio statistics
        let mean = audio.iter().sum::<f32>() / audio.len() as f32;
        let variance = audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / audio.len() as f32;
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        features.push(mean);
        features.push(variance);
        features.push(rms);

        // Zero crossing rate
        let zero_crossings = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f32;
        features.push(zero_crossings / audio.len() as f32);

        // Spectral centroid (simplified)
        let spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate)?;
        features.push(spectral_centroid);

        Ok(features)
    }

    /// Calculate the magnitude-weighted spectral centroid (in Hz).
    ///
    /// Splits the signal into Hann-windowed frames, transforms each with the real
    /// FFT ([`scirs2_fft::rfft`], via [`frame_spectral_centroid`]) and averages the
    /// per-frame centroids. Returns `0.0` for silent or sub-frame-length input.
    fn calculate_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> Result<f32, Error> {
        if audio.is_empty() || sample_rate == 0 {
            return Ok(0.0);
        }

        let frame_len = SPEC_FRAME_LEN.min(audio.len());
        if frame_len < 4 {
            return Ok(0.0);
        }

        let mut centroid_sum = 0.0f64;
        let mut frame_count = 0usize;
        let mut pos = 0;
        while pos + frame_len <= audio.len() {
            if let Some(c) = frame_spectral_centroid(&audio[pos..pos + frame_len], sample_rate) {
                centroid_sum += c as f64;
                frame_count += 1;
            }
            // For inputs shorter than a full hop, analyze the single frame only.
            if frame_len == audio.len() {
                break;
            }
            pos += SPEC_HOP;
        }

        if frame_count == 0 {
            return Ok(0.0);
        }
        Ok((centroid_sum / frame_count as f64) as f32)
    }
}

impl VoiceCloner {
    /// Create a new voice cloner
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            strategy: CloningStrategy::MultiStage,
            quality_thresholds: QualityThresholds::default(),
            adaptation_params: AdaptationParams::default(),
        })
    }
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            adaptation_method: AdaptationMethod::Hybrid,
            quality_mode: QualityMode::Balanced,
            num_reference_samples: 3,
            adaptation_lr: 0.001,
            adaptation_steps: 100,
            preserve_similarity: true,
            adapt_prosody: true,
            adapt_timbre: true,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_speaker_similarity: 0.7,
            min_audio_quality: 0.8,
            min_naturalness: 0.75,
            max_distortion: 0.2,
        }
    }
}

impl Default for AdaptationParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            regularization: 0.01,
            temperature: 1.0,
            dropout_rate: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_zero_shot_synthesizer_creation() {
        let device = Device::Cpu;
        let synthesizer = ZeroShotSynthesizer::new(device);
        assert!(synthesizer.is_ok());
    }

    #[test]
    fn test_reference_voice_creation() {
        let audio_samples = vec![AudioSample {
            audio: vec![0.0; 44100],
            sample_rate: 44100,
            duration: 1.0,
            transcription: Some("test".to_string()),
            phonemes: Some(vec![
                "t".to_string(),
                "e".to_string(),
                "s".to_string(),
                "t".to_string(),
            ]),
            quality_score: 0.9,
        }];

        let characteristics = VoiceCharacteristics::for_voice_type(VoiceType::Soprano);

        let reference = ZeroShotSynthesizer::create_reference_voice(
            "test_voice".to_string(),
            "Test Voice".to_string(),
            audio_samples,
            characteristics,
        );

        assert!(reference.is_ok());
        let reference = reference.unwrap();
        assert_eq!(reference.voice_id, "test_voice");
        assert_eq!(reference.voice_embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_voice_adaptation_engine() {
        let device = Device::Cpu;
        let engine = VoiceAdaptationEngine::new(device);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        let samples = vec![AudioSample {
            audio: vec![0.0; 1000],
            sample_rate: 44100,
            duration: 0.023,
            transcription: None,
            phonemes: None,
            quality_score: 0.8,
        }];

        let config = ZeroShotConfig::default();
        let result = engine.adapt_from_audio(&samples, &config).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_speaker_encoder() {
        let encoder = SpeakerEncoder::new(256);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        let samples = vec![AudioSample {
            audio: vec![0.1, -0.1, 0.2, -0.2],
            sample_rate: 44100,
            duration: 0.0001,
            transcription: None,
            phonemes: None,
            quality_score: 0.9,
        }];

        let features = encoder.extract_features(&samples);
        assert!(features.is_ok());
        assert_eq!(features.unwrap().len(), 256);
    }

    #[test]
    fn test_zero_shot_config_default() {
        let config = ZeroShotConfig::default();
        assert!(matches!(config.adaptation_method, AdaptationMethod::Hybrid));
        assert!(matches!(config.quality_mode, QualityMode::Balanced));
        assert_eq!(config.num_reference_samples, 3);
        assert!(config.preserve_similarity);
    }

    #[test]
    fn test_vocal_range() {
        let range = VocalRange {
            lowest_note: 48,
            highest_note: 84,
            optimal_start: 60,
            optimal_end: 72,
            register_breaks: vec![60, 67],
        };

        assert_eq!(range.highest_note - range.lowest_note, 36); // 3 octaves
        assert!(range.optimal_start >= range.lowest_note);
        assert!(range.optimal_end <= range.highest_note);
    }

    /// Generate an exponential frequency sweep (linear in MIDI) from `f0` to `f1`.
    fn make_exp_sweep(f0: f32, f1: f32, sample_rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        let log_ratio = (f1 / f0).ln();
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / n as f32;
            let f = f0 * (log_ratio * t).exp();
            phase += 2.0 * std::f32::consts::PI * f / sample_rate as f32;
            out.push(amp * phase.sin());
        }
        out
    }

    /// Generate a tone whose F0 is sinusoidally modulated at `vib_hz`.
    fn make_vibrato_tone(
        center: f32,
        vib_hz: f32,
        depth: f32,
        sample_rate: u32,
        secs: f32,
        amp: f32,
    ) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let f = center * (1.0 + depth * (2.0 * std::f32::consts::PI * vib_hz * t).sin());
            phase += 2.0 * std::f32::consts::PI * f / sample_rate as f32;
            out.push(amp * phase.sin());
        }
        out
    }

    /// Generate a steady tone summing the given partials.
    fn make_tone(freqs: &[f32], sample_rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amp * freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * t).sin())
                    .sum::<f32>()
                    / freqs.len() as f32
            })
            .collect()
    }

    fn sample_from(audio: Vec<f32>, sample_rate: u32) -> AudioSample {
        let duration = audio.len() as f32 / sample_rate as f32;
        AudioSample {
            audio,
            sample_rate,
            duration,
            transcription: None,
            phonemes: None,
            quality_score: 1.0,
        }
    }

    #[test]
    fn test_analyze_vocal_range_glissando() {
        let sample_rate = 44100;
        // Exponential sweep C3 (130.81 Hz, MIDI 48) -> C5 (523.25 Hz, MIDI 72).
        let audio = make_exp_sweep(130.81, 523.25, sample_rate, 3.0, 0.8);
        let samples = vec![sample_from(audio, sample_rate)];

        let range = vocal_range_from_samples(&samples);
        assert!(
            range.lowest_note < range.highest_note,
            "lowest {} highest {}",
            range.lowest_note,
            range.highest_note
        );
        assert!(
            range.highest_note - range.lowest_note >= 12,
            "span too small: {}..{}",
            range.lowest_note,
            range.highest_note
        );
        assert!(
            (44..=56).contains(&range.lowest_note),
            "lowest_note {}",
            range.lowest_note
        );
        assert!(
            (66..=78).contains(&range.highest_note),
            "highest_note {}",
            range.highest_note
        );
        assert!(range.optimal_start >= range.lowest_note);
        assert!(range.optimal_end <= range.highest_note);
    }

    #[test]
    fn test_vibrato_rate_detection() {
        let sample_rate = 44100;
        let audio = make_vibrato_tone(220.0, 5.5, 0.04, sample_rate, 3.0, 0.8);
        let samples = vec![sample_from(audio, sample_rate)];

        let metrics = voice_quality_from_samples(&samples);
        assert!(
            (metrics.vibrato_rate - 5.5).abs() < 0.7,
            "vibrato_rate {}",
            metrics.vibrato_rate
        );
        assert!(
            metrics.vibrato_depth > 5.0,
            "expected audible vibrato depth, got {}",
            metrics.vibrato_depth
        );
    }

    #[test]
    fn test_voice_embedding_unit_norm_deterministic() {
        let sample_rate = 44100;
        let audio = make_tone(&[220.0, 440.0, 660.0], sample_rate, 1.0, 0.8);
        let samples = vec![sample_from(audio, sample_rate)];

        let first = voice_embedding_from_samples(&samples);
        let second = voice_embedding_from_samples(&samples);

        assert_eq!(first.len(), 512);
        assert_eq!(first, second, "embedding must be deterministic");

        let norm = first.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "embedding norm {}", norm);
    }

    #[test]
    fn test_flat_tone_small_vibrato_depth() {
        let sample_rate = 44100;
        // 220.5 Hz has an exact 200-sample period at 44100 Hz -> stable F0 estimate.
        let audio = make_tone(&[220.5], sample_rate, 2.0, 0.8);
        let samples = vec![sample_from(audio, sample_rate)];

        let metrics = voice_quality_from_samples(&samples);
        assert!(
            metrics.vibrato_depth < 20.0,
            "flat tone vibrato_depth {}",
            metrics.vibrato_depth
        );
    }

    #[test]
    fn test_spectral_centroid_high_vs_low_tone() {
        let sample_rate = 44100;
        let encoder = SpeakerEncoder::new(64).unwrap();
        let low = make_tone(&[200.0], sample_rate, 1.0, 0.8);
        let high = make_tone(&[5000.0], sample_rate, 1.0, 0.8);

        let c_low = encoder
            .calculate_spectral_centroid(&low, sample_rate)
            .unwrap();
        let c_high = encoder
            .calculate_spectral_centroid(&high, sample_rate)
            .unwrap();

        assert!(
            c_high > c_low * 2.0,
            "high-tone centroid {} should greatly exceed low-tone centroid {}",
            c_high,
            c_low
        );
        assert!(c_low < 1000.0, "low-tone centroid too high: {}", c_low);
        assert!(c_high > 3000.0, "high-tone centroid too low: {}", c_high);
    }

    #[test]
    fn test_estimate_pitch_range_glissando() {
        let sample_rate = 44100;
        let engine = VoiceAdaptationEngine::new(Device::Cpu).unwrap();
        // Exponential sweep A3 (220 Hz) -> A5 (880 Hz).
        let audio = make_exp_sweep(220.0, 880.0, sample_rate, 3.0, 0.8);
        let samples = vec![sample_from(audio, sample_rate)];

        let (low, high) = engine.estimate_pitch_range(&samples).unwrap();
        assert!(high > low, "range {}..{}", low, high);
        assert!(high - low > 300.0, "range too narrow: {}..{}", low, high);
        assert!((180.0..=340.0).contains(&low), "low {}", low);
        assert!((700.0..=920.0).contains(&high), "high {}", high);
    }

    #[test]
    fn test_estimate_voice_type_low_vs_high() {
        let sample_rate = 44100;
        let engine = VoiceAdaptationEngine::new(Device::Cpu).unwrap();

        // ~98 Hz (G2) -> low voice; ~440 Hz (A4) -> high voice.
        let low_audio = make_tone(&[98.0], sample_rate, 1.5, 0.8);
        let high_audio = make_tone(&[440.0], sample_rate, 1.5, 0.8);
        let low_voice = engine
            .estimate_voice_type(&[sample_from(low_audio, sample_rate)])
            .unwrap();
        let high_voice = engine
            .estimate_voice_type(&[sample_from(high_audio, sample_rate)])
            .unwrap();

        assert!(
            matches!(low_voice, VoiceType::Bass | VoiceType::Baritone),
            "expected low voice, got {:?}",
            low_voice
        );
        assert!(
            matches!(high_voice, VoiceType::Soprano | VoiceType::MezzoSoprano),
            "expected high voice, got {:?}",
            high_voice
        );
    }
}
