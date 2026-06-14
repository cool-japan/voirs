//! # VoiRS Dataset Utilities
//!
//! Dataset loading, preprocessing, and management utilities for training
//! and evaluation of VoiRS speech synthesis models.

// Allow pedantic lints that are acceptable for audio/DSP processing code
#![allow(clippy::cast_precision_loss)] // Acceptable for audio sample conversions
#![allow(clippy::cast_possible_truncation)] // Controlled truncation in audio processing
#![allow(clippy::cast_sign_loss)] // Intentional in index calculations
#![allow(clippy::missing_errors_doc)] // Many internal functions with self-documenting error types
#![allow(clippy::missing_panics_doc)] // Panics are documented where relevant
#![allow(clippy::unused_self)] // Some trait implementations require &self for consistency
#![allow(clippy::must_use_candidate)] // Not all return values need must_use annotation
#![allow(clippy::doc_markdown)] // Technical terms don't all need backticks
#![allow(clippy::unnecessary_wraps)] // Result wrappers maintained for API consistency
#![allow(clippy::float_cmp)] // Exact float comparisons are intentional in some contexts
#![allow(clippy::match_same_arms)] // Pattern matching clarity sometimes requires duplication
#![allow(clippy::module_name_repetitions)] // Type names often repeat module names
#![allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
#![allow(clippy::too_many_lines)] // Some functions are inherently complex
#![allow(clippy::needless_pass_by_value)] // Some functions designed for ownership transfer
#![allow(clippy::similar_names)] // Many similar variable names in algorithms
#![allow(clippy::unused_async)] // Public API functions may need async for consistency
#![allow(clippy::needless_range_loop)] // Range loops sometimes clearer than iterators
#![allow(clippy::uninlined_format_args)] // Explicit argument names can improve clarity
#![allow(clippy::manual_clamp)] // Manual clamping sometimes clearer
#![allow(clippy::return_self_not_must_use)] // Not all builder methods need must_use
#![allow(clippy::cast_possible_wrap)] // Controlled wrapping in processing code
#![allow(clippy::cast_lossless)] // Explicit casts preferred for clarity
#![allow(clippy::wildcard_imports)] // Prelude imports are convenient and standard
#![allow(clippy::format_push_string)] // Sometimes more readable than alternative
#![allow(clippy::redundant_closure_for_method_calls)] // Closures sometimes needed for type inference

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for dataset operations
pub type Result<T> = std::result::Result<T, DatasetError>;

/// Dataset-specific error types
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Dataset loading failed: {0}")]
    LoadError(String),

    #[error("Invalid format: {0}")]
    FormatError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Audio processing error: {0}")]
    AudioError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Index out of bounds: {0}")]
    IndexError(usize),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Audio file error: {0}")]
    HoundError(#[from] hound::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Dataset split error: {0}")]
    SplitError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Cloud storage error: {0}")]
    CloudStorage(String),

    #[error("Git error: {0}")]
    Git(String),

    #[error("MLOps error: {0}")]
    MLOps(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Language codes supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (US)
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    Ja,
    /// Mandarin Chinese
    ZhCn,
    /// Korean
    Ko,
    /// German
    De,
    /// French
    Fr,
    /// Spanish
    Es,
}

impl LanguageCode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageCode::EnUs => "en-US",
            LanguageCode::EnGb => "en-GB",
            LanguageCode::Ja => "ja",
            LanguageCode::ZhCn => "zh-CN",
            LanguageCode::Ko => "ko",
            LanguageCode::De => "de",
            LanguageCode::Fr => "fr",
            LanguageCode::Es => "es",
        }
    }
}

/// A phoneme with its symbol and optional features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    /// Phoneme symbol (IPA or language-specific)
    pub symbol: String,
    /// Optional phoneme features
    pub features: Option<HashMap<String, String>>,
    /// Duration in seconds (if available)
    pub duration: Option<f32>,
}

impl Phoneme {
    /// Create new phoneme
    pub fn new<S: Into<String>>(symbol: S) -> Self {
        Self {
            symbol: symbol.into(),
            features: None,
            duration: None,
        }
    }
}

/// Audio data structure with efficient processing capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    /// Audio samples (interleaved for multi-channel)
    samples: Vec<f32>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of channels
    channels: u32,
    /// Optional metadata
    metadata: HashMap<String, String>,
}

/// Audio buffer for holding PCM audio data (legacy compatibility)
pub type AudioBuffer = AudioData;

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            metadata: HashMap::new(),
        }
    }

    /// Create silence
    pub fn silence(duration: f32, sample_rate: u32, channels: u32) -> Self {
        let num_samples = (duration * sample_rate as f32 * channels as f32) as usize;
        Self::new(vec![0.0; num_samples], sample_rate, channels)
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate * self.channels) as f32
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Get samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get mutable samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Resample audio to new sample rate using high-quality linear interpolation
    pub fn resample(&self, new_sample_rate: u32) -> Result<AudioData> {
        if new_sample_rate == self.sample_rate {
            return Ok(self.clone());
        }

        if self.samples.is_empty() {
            return Ok(AudioData::new(vec![], new_sample_rate, self.channels));
        }

        // High-quality linear interpolation resampling
        let ratio = self.sample_rate as f64 / new_sample_rate as f64;
        let new_length = (self.samples.len() as f64 / ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f64 * ratio;
            let index_floor = src_index.floor() as usize;
            let index_ceil = (index_floor + 1).min(self.samples.len() - 1);
            let fraction = src_index - index_floor as f64;

            if index_floor >= self.samples.len() {
                new_samples.push(0.0);
            } else if index_floor == index_ceil {
                // At the boundary, use the last sample
                new_samples.push(self.samples[index_floor]);
            } else {
                // Linear interpolation between adjacent samples
                let sample1 = self.samples[index_floor];
                let sample2 = self.samples[index_ceil];
                let interpolated = sample1 + (sample2 - sample1) * fraction as f32;
                new_samples.push(interpolated);
            }
        }

        Ok(AudioData::new(new_samples, new_sample_rate, self.channels))
    }

    /// High-quality resampling using windowed sinc interpolation
    pub fn resample_windowed_sinc(&self, new_sample_rate: u32) -> Result<AudioData> {
        if new_sample_rate == self.sample_rate {
            return Ok(self.clone());
        }

        if self.samples.is_empty() {
            return Ok(AudioData::new(vec![], new_sample_rate, self.channels));
        }

        // Windowed sinc resampling parameters
        const FILTER_LENGTH: usize = 128;
        const KAISER_BETA: f64 = 8.6;

        let ratio = new_sample_rate as f64 / self.sample_rate as f64;
        let new_length = (self.samples.len() as f64 * ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_length);

        // Precompute Kaiser window coefficients
        let kaiser_window = Self::kaiser_window(FILTER_LENGTH, KAISER_BETA);

        // Calculate filter cutoff frequency
        let cutoff = if ratio < 1.0 { ratio } else { 1.0 };

        for i in 0..new_length {
            let src_index = i as f64 / ratio;
            let mut sample = 0.0f64;

            // Apply windowed sinc filter
            for (j, &window_coeff) in kaiser_window.iter().enumerate().take(FILTER_LENGTH) {
                let filter_index = j as i32 - (FILTER_LENGTH as i32 / 2);
                let sample_index = src_index + filter_index as f64;

                if sample_index >= 0.0 && sample_index < self.samples.len() as f64 {
                    let t = sample_index - sample_index.floor();
                    let src_sample = if t == 0.0 {
                        self.samples[sample_index as usize] as f64
                    } else {
                        // Linear interpolation between samples
                        let idx = sample_index.floor() as usize;
                        let next_idx = (idx + 1).min(self.samples.len() - 1);
                        let s1 = self.samples[idx] as f64;
                        let s2 = self.samples[next_idx] as f64;
                        s1 + (s2 - s1) * t
                    };

                    // Windowed sinc coefficient
                    let x = (filter_index as f64 - (src_index - src_index.floor())) * cutoff;
                    let sinc_val = if x.abs() < 1e-10 {
                        cutoff
                    } else {
                        let pi_x = std::f64::consts::PI * x;
                        (pi_x.sin() / pi_x) * cutoff
                    };

                    sample += src_sample * sinc_val * window_coeff;
                }
            }

            new_samples.push(sample.clamp(-1.0, 1.0) as f32);
        }

        Ok(AudioData::new(new_samples, new_sample_rate, self.channels))
    }

    /// Generate Kaiser window coefficients
    fn kaiser_window(length: usize, beta: f64) -> Vec<f64> {
        let mut window = Vec::with_capacity(length);
        let alpha = (length - 1) as f64 / 2.0;
        let i0_beta = Self::modified_bessel_i0(beta);

        for i in 0..length {
            let x = (i as f64 - alpha) / alpha;
            let arg = beta * (1.0 - x * x).sqrt();
            window.push(Self::modified_bessel_i0(arg) / i0_beta);
        }

        window
    }

    /// Modified Bessel function of the first kind (I0)
    fn modified_bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_squared = x * x;

        for k in 1..=50 {
            term *= x_squared / (4.0 * k as f64 * k as f64);
            sum += term;
            if term < 1e-15 * sum {
                break;
            }
        }

        sum
    }

    /// Normalize audio amplitude
    pub fn normalize(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Ok(());
        }

        use crate::audio::simd::SimdAudioProcessor;
        let max_amplitude = SimdAudioProcessor::find_peak(&self.samples);

        if max_amplitude > 0.0 {
            let scale = 1.0 / max_amplitude;
            SimdAudioProcessor::apply_gain(&mut self.samples, scale);
        }

        Ok(())
    }

    /// Calculate RMS (Root Mean Square) of the audio
    pub fn rms(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }

        use crate::audio::simd::SimdAudioProcessor;
        let rms = SimdAudioProcessor::calculate_rms(&self.samples);
        Some(rms)
    }

    /// Calculate peak amplitude of the audio
    pub fn peak(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }

        use crate::audio::simd::SimdAudioProcessor;
        let peak = SimdAudioProcessor::find_peak(&self.samples);
        Some(peak)
    }

    /// Calculate LUFS (Loudness Units Full Scale) of the audio
    /// This is a perceptual loudness measurement following ITU-R BS.1770-4
    pub fn lufs(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }

        let loudness = self.calculate_integrated_loudness();
        Some(loudness)
    }

    /// Calculate integrated loudness following ITU-R BS.1770-4
    fn calculate_integrated_loudness(&self) -> f32 {
        // ITU-R BS.1770-4 integrated loudness: K-weighting (pre-filter + RLB
        // high-pass) followed by a gated mean square over 400 ms blocks.
        let filtered_samples = self.apply_k_weighting();

        // Calculate mean square with gating
        let mean_square = self.calculate_gated_mean_square(&filtered_samples);

        // Convert to LUFS
        if mean_square > 0.0 {
            -0.691 + 10.0 * mean_square.log10()
        } else {
            -70.0 // Minimum practical LUFS value
        }
    }

    /// Apply the ITU-R BS.1770-4 K-weighting filter (two cascaded biquad IIR
    /// stages) to the sample buffer.
    ///
    /// * **Stage 1 — pre-filter (high-shelf, ~1682 Hz, +4 dB, Q ≈ 0.707):**
    ///   models the acoustic effect of the head, gently boosting high
    ///   frequencies.
    /// * **Stage 2 — RLB high-pass (~38 Hz, Q ≈ 0.5 / second-order Butterworth):**
    ///   the revised low-frequency B-weighting that strongly attenuates DC and
    ///   sub-bass.
    ///
    /// Both stages are second-order sections whose coefficients are derived from
    /// the analogue prototypes of BS.1770-4 Annex 1 via the bilinear transform
    /// for the actual sample rate (so the response is correct at any rate, not
    /// just 48 kHz). The filtering is performed in `f64` for numerical
    /// stability. The buffer is treated as a single channel-agnostic stream,
    /// matching the integrated-loudness measurement that consumes it.
    fn apply_k_weighting(&self) -> Vec<f32> {
        let fs = f64::from(self.sample_rate);
        if fs <= 0.0 || self.samples.is_empty() {
            return self.samples.clone();
        }

        // ---- K-weighting biquad coefficients (BS.1770-4), derived for `fs` ----
        // See `k_weighting_pre_coeffs` / `k_weighting_rlb_coeffs` below for the
        // canonical bilinear-transform derivation (reproduces the published
        // 48 kHz reference coefficients exactly).
        let [b0_pre, b1_pre, b2_pre, a1_pre, a2_pre] = Self::k_weighting_pre_coeffs(fs);
        let [b0_rlb, b1_rlb, b2_rlb, a1_rlb, a2_rlb] = Self::k_weighting_rlb_coeffs(fs);

        // ---- Run the two biquad stages in series (direct form II) ----
        let mut w1 = [0.0f64; 2]; // state for stage 1
        let mut w2 = [0.0f64; 2]; // state for stage 2

        self.samples
            .iter()
            .map(|&x| {
                let xd = f64::from(x);

                // Stage 1
                let w1n = xd - a1_pre * w1[0] - a2_pre * w1[1];
                let y1 = b0_pre * w1n + b1_pre * w1[0] + b2_pre * w1[1];
                w1[1] = w1[0];
                w1[0] = w1n;

                // Stage 2
                let w2n = y1 - a1_rlb * w2[0] - a2_rlb * w2[1];
                let y2 = b0_rlb * w2n + b1_rlb * w2[0] + b2_rlb * w2[1];
                w2[1] = w2[0];
                w2[0] = w2n;

                y2 as f32
            })
            .collect()
    }

    /// BS.1770-4 K-weighting stage-1 "pre-filter" (high-shelf) biquad
    /// coefficients `[b0, b1, b2, a1, a2]` (normalised to a0 = 1), derived for
    /// `fs` (Hz) via the bilinear transform (canonical EBU R128 / De Man
    /// derivation). At 48 kHz this reproduces the published BS.1770-4 reference
    /// `b = [1.53512485958697, -2.69169618940638, 1.19839281085285]`,
    /// `a = [1, -1.69065929318241, 0.73248077421585]`.
    fn k_weighting_pre_coeffs(fs: f64) -> [f64; 5] {
        let f0 = 1_681.974_450_955_532_f64;
        let q = 0.707_175_236_955_419_3_f64;
        let gain_db = 3.999_843_853_973_347_f64;

        let k = (std::f64::consts::PI * f0 / fs).tan();
        let k2 = k * k;
        let vh = 10.0_f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6_f64);
        let denom = 1.0 + k / q + k2;

        [
            (vh + vb * k / q + k2) / denom,
            2.0 * (k2 - vh) / denom,
            (vh - vb * k / q + k2) / denom,
            2.0 * (k2 - 1.0) / denom,
            (1.0 - k / q + k2) / denom,
        ]
    }

    /// BS.1770-4 K-weighting stage-2 RLB (revised low-frequency B-weighting)
    /// high-pass biquad coefficients `[b0, b1, b2, a1, a2]`. The numerator is
    /// exactly `[1, -2, 1]`; the denominator is derived for `fs` (Hz) with
    /// Q ≈ 0.5003 (the RLB prototype, NOT a Butterworth Q = 0.707). At 48 kHz
    /// this reproduces the reference `a = [1, -1.99004745483398, 0.99007225036616]`.
    fn k_weighting_rlb_coeffs(fs: f64) -> [f64; 5] {
        let f0 = 38.135_470_876_139_82_f64;
        let q = 0.500_327_037_325_395_3_f64;

        let k = (std::f64::consts::PI * f0 / fs).tan();
        let k2 = k * k;
        let denom = 1.0 + k / q + k2;

        [
            1.0,
            -2.0,
            1.0,
            2.0 * (k2 - 1.0) / denom,
            (1.0 - k / q + k2) / denom,
        ]
    }

    /// Calculate gated mean square for LUFS measurement
    fn calculate_gated_mean_square(&self, samples: &[f32]) -> f32 {
        // Block size for gating (400ms at typical sample rates)
        let block_size = (0.4 * self.sample_rate as f32) as usize;
        if block_size == 0 || samples.len() < block_size {
            // Fallback for short audio
            return samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        }

        let mut block_powers = Vec::new();

        // Calculate power for each block
        for chunk in samples.chunks(block_size) {
            let power = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            block_powers.push(power);
        }

        // Apply relative gate (-70 LUFS)
        let relative_threshold = block_powers.iter().sum::<f32>() / block_powers.len() as f32 * 0.1; // -10dB relative

        let gated_powers: Vec<f32> = block_powers
            .into_iter()
            .filter(|&power| power >= relative_threshold)
            .collect();

        if gated_powers.is_empty() {
            relative_threshold
        } else {
            gated_powers.iter().sum::<f32>() / gated_powers.len() as f32
        }
    }

    /// Normalize audio to target RMS level
    pub fn normalize_rms(&mut self, target_rms: f32) -> Result<()> {
        if let Some(current_rms) = self.rms() {
            if current_rms > 0.0 {
                let scale = target_rms / current_rms;
                use crate::audio::simd::SimdAudioProcessor;
                SimdAudioProcessor::apply_gain(&mut self.samples, scale);
            }
        }
        Ok(())
    }

    /// Normalize audio to target peak level
    pub fn normalize_peak(&mut self, target_peak: f32) -> Result<()> {
        if let Some(current_peak) = self.peak() {
            if current_peak > 0.0 {
                let scale = target_peak / current_peak;
                use crate::audio::simd::SimdAudioProcessor;
                SimdAudioProcessor::apply_gain(&mut self.samples, scale);
            }
        }
        Ok(())
    }

    /// Normalize audio to target LUFS level
    pub fn normalize_lufs(&mut self, target_lufs: f32) -> Result<()> {
        if let Some(current_lufs) = self.lufs() {
            let lufs_difference = target_lufs - current_lufs;
            let scale = 10.0_f32.powf(lufs_difference / 20.0); // Convert dB to linear scale
            use crate::audio::simd::SimdAudioProcessor;
            SimdAudioProcessor::apply_gain(&mut self.samples, scale);
        }
        Ok(())
    }

    /// Comprehensive normalization with multiple options
    pub fn normalize_comprehensive(&mut self, config: NormalizationConfig) -> Result<()> {
        match config.method {
            NormalizationMethod::Peak => {
                self.normalize_peak(config.target_level)?;
            }
            NormalizationMethod::Rms => {
                self.normalize_rms(config.target_level)?;
            }
            NormalizationMethod::Lufs => {
                self.normalize_lufs(config.target_level)?;
            }
        }

        // Apply optional limiting to prevent clipping
        if config.apply_limiting {
            self.apply_soft_limiter(config.limiter_threshold)?;
        }

        Ok(())
    }

    /// Apply soft limiting to prevent clipping
    fn apply_soft_limiter(&mut self, threshold: f32) -> Result<()> {
        for sample in &mut self.samples {
            let abs_sample = sample.abs();
            if abs_sample > threshold {
                // Soft limiting using tanh compression
                let sign = sample.signum();
                let compressed = threshold * (abs_sample / threshold).tanh();
                *sample = sign * compressed;
            }
        }
        Ok(())
    }
}

/// Audio file formats supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV format
    Wav,
    /// FLAC format
    Flac,
    /// MP3 format
    Mp3,
    /// OGG Vorbis format
    Ogg,
    /// OPUS format
    Opus,
}

pub mod audio;
pub mod augmentation;
pub mod cache;
pub mod datasets;
pub mod error;
pub mod export;
pub mod integration;
pub mod metadata;
pub mod ml;
pub mod parallel;
pub mod performance;
pub mod processing;
pub mod profiling;
pub mod quality;
pub mod research;
pub mod sampling;
pub mod streaming;
pub mod traits;
pub mod utils;
pub mod versioning;

// Legacy modules for backward compatibility
pub mod formats;
pub mod loaders;
pub mod preprocessors;
pub mod splits;
pub mod validation;

// Re-export split types for convenience
pub use splits::{DatasetSplit, DatasetSplits, SplitConfig, SplitStatistics, SplitStrategy};

/// Pure-Rust TLS crypto provider installation.
pub mod tls {
    use std::sync::Once;

    static INSTALL_PROVIDER: Once = Once::new();

    /// Install the pure-Rust rustls [`CryptoProvider`](rustls::crypto::CryptoProvider)
    /// as the process-wide default.
    ///
    /// `reqwest` is built with `rustls-no-provider`, so a default crypto provider must
    /// be installed before any TLS handshake. This is `Once`-guarded and therefore
    /// safe (and cheap) to call from every network entry point.
    pub fn ensure_crypto_provider() {
        INSTALL_PROVIDER.call_once(|| {
            let provider = (*oxitls_adapter_rustls_rustcrypto::pure_provider()).clone();
            let _ = rustls::crypto::CryptoProvider::install_default(provider);
        });
    }
}

/// Speaker information structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker identifier
    pub id: String,
    /// Speaker name (if available)
    pub name: Option<String>,
    /// Speaker gender
    pub gender: Option<String>,
    /// Speaker age
    pub age: Option<u32>,
    /// Speaker accent/region
    pub accent: Option<String>,
    /// Additional speaker metadata
    pub metadata: HashMap<String, String>,
}

/// Quality metrics for audio samples
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr: Option<f32>,
    /// Clipping percentage
    pub clipping: Option<f32>,
    /// Dynamic range in dB
    pub dynamic_range: Option<f32>,
    /// Spectral quality score (0-1)
    pub spectral_quality: Option<f32>,
    /// Overall quality score (0-1)
    pub overall_quality: Option<f32>,
}

/// Audio normalization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Normalize to peak amplitude
    Peak,
    /// Normalize to RMS level
    Rms,
    /// Normalize to LUFS level (perceptual loudness)
    Lufs,
}

/// Audio normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Normalization method to use
    pub method: NormalizationMethod,
    /// Target level (peak: 0.0-1.0, RMS: 0.0-1.0, LUFS: -70.0 to 0.0 dB)
    pub target_level: f32,
    /// Apply soft limiting to prevent clipping
    pub apply_limiting: bool,
    /// Limiter threshold (0.0-1.0)
    pub limiter_threshold: f32,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::Peak,
            target_level: 0.9,
            apply_limiting: true,
            limiter_threshold: 0.95,
        }
    }
}

impl NormalizationConfig {
    /// Create configuration for peak normalization
    pub fn peak(target_level: f32) -> Self {
        Self {
            method: NormalizationMethod::Peak,
            target_level,
            apply_limiting: true,
            limiter_threshold: 0.95,
        }
    }

    /// Create configuration for RMS normalization
    pub fn rms(target_level: f32) -> Self {
        Self {
            method: NormalizationMethod::Rms,
            target_level,
            apply_limiting: true,
            limiter_threshold: 0.95,
        }
    }

    /// Create configuration for LUFS normalization
    pub fn lufs(target_lufs: f32) -> Self {
        Self {
            method: NormalizationMethod::Lufs,
            target_level: target_lufs,
            apply_limiting: true,
            limiter_threshold: 0.95,
        }
    }
}

/// Dataset sample with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSample {
    /// Unique identifier for this sample
    pub id: String,

    /// Original text
    pub text: String,

    /// Audio data
    pub audio: AudioData,

    /// Speaker information (if available)
    pub speaker: Option<SpeakerInfo>,

    /// Language of the text
    pub language: LanguageCode,

    /// Quality metrics
    pub quality: QualityMetrics,

    /// Phoneme sequence (if available)
    pub phonemes: Option<Vec<Phoneme>>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset item containing text, phonemes, and audio (legacy compatibility)
pub type DatasetItem = DatasetSample;

impl DatasetSample {
    /// Create new dataset sample
    pub fn new(id: String, text: String, audio: AudioData, language: LanguageCode) -> Self {
        Self {
            id,
            text,
            audio,
            speaker: None,
            language,
            quality: QualityMetrics {
                snr: None,
                clipping: None,
                dynamic_range: None,
                spectral_quality: None,
                overall_quality: None,
            },
            phonemes: None,
            metadata: HashMap::new(),
        }
    }

    /// Set phonemes
    pub fn with_phonemes(mut self, phonemes: Vec<Phoneme>) -> Self {
        self.phonemes = Some(phonemes);
        self
    }

    /// Set speaker information
    pub fn with_speaker(mut self, speaker: SpeakerInfo) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Set quality metrics
    pub fn with_quality(mut self, quality: QualityMetrics) -> Self {
        self.quality = quality;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.audio.duration()
    }

    /// Get speaker ID (for backward compatibility)
    pub fn speaker_id(&self) -> Option<&str> {
        self.speaker.as_ref().map(|s| s.id.as_str())
    }
}

/// Dataset trait for different dataset formats
pub trait Dataset {
    /// Get dataset name
    fn name(&self) -> &str;

    /// Get number of items in dataset
    fn len(&self) -> usize;

    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get item by index
    fn get_item(&self, index: usize) -> Result<DatasetItem>;

    /// Get all items (for small datasets)
    fn get_all_items(&self) -> Result<Vec<DatasetItem>> {
        (0..self.len()).map(|i| self.get_item(i)).collect()
    }

    /// Get dataset statistics
    fn statistics(&self) -> DatasetStatistics;

    /// Validate dataset
    fn validate(&self) -> Result<ValidationReport>;
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Total number of items
    pub total_items: usize,

    /// Total duration in seconds
    pub total_duration: f32,

    /// Average duration per item
    pub average_duration: f32,

    /// Language distribution
    pub language_distribution: std::collections::HashMap<LanguageCode, usize>,

    /// Speaker distribution (if applicable)
    pub speaker_distribution: std::collections::HashMap<String, usize>,

    /// Text length statistics
    pub text_length_stats: LengthStatistics,

    /// Audio duration statistics
    pub duration_stats: DurationStatistics,
}

/// Length statistics for text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthStatistics {
    pub min: usize,
    pub max: usize,
    pub mean: f32,
    pub median: usize,
    pub std_dev: f32,
}

/// Duration statistics for audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStatistics {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
}

/// Dataset validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the dataset is valid
    pub is_valid: bool,

    /// List of errors found
    pub errors: Vec<String>,

    /// List of warnings
    pub warnings: Vec<String>,

    /// Number of items validated
    pub items_validated: usize,
}

/// In-memory dataset implementation
pub struct MemoryDataset {
    name: String,
    items: Vec<DatasetItem>,
}

impl MemoryDataset {
    /// Create new in-memory dataset
    pub fn new(name: String) -> Self {
        Self {
            name,
            items: Vec::new(),
        }
    }

    /// Add item to dataset
    pub fn add_item(&mut self, item: DatasetItem) {
        self.items.push(item);
    }

    /// Add multiple items
    pub fn add_items(&mut self, items: Vec<DatasetItem>) {
        self.items.extend(items);
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl Dataset for MemoryDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get_item(&self, index: usize) -> Result<DatasetItem> {
        self.items.get(index).cloned().ok_or_else(|| {
            DatasetError::ConfigError(format!("Dataset index {index} out of bounds"))
        })
    }

    fn statistics(&self) -> DatasetStatistics {
        if self.items.is_empty() {
            return DatasetStatistics {
                total_items: 0,
                total_duration: 0.0,
                average_duration: 0.0,
                language_distribution: std::collections::HashMap::new(),
                speaker_distribution: std::collections::HashMap::new(),
                text_length_stats: LengthStatistics {
                    min: 0,
                    max: 0,
                    mean: 0.0,
                    median: 0,
                    std_dev: 0.0,
                },
                duration_stats: DurationStatistics {
                    min: 0.0,
                    max: 0.0,
                    mean: 0.0,
                    median: 0.0,
                    std_dev: 0.0,
                },
            };
        }

        let total_items = self.items.len();
        let total_duration: f32 = self.items.iter().map(DatasetSample::duration).sum();
        let average_duration = total_duration / total_items as f32;

        // Language distribution
        let mut language_distribution = std::collections::HashMap::new();
        for item in &self.items {
            *language_distribution.entry(item.language).or_insert(0) += 1;
        }

        // Speaker distribution
        let mut speaker_distribution = std::collections::HashMap::new();
        for item in &self.items {
            if let Some(speaker) = item.speaker_id() {
                *speaker_distribution.entry(speaker.to_string()).or_insert(0) += 1;
            }
        }

        // Text length statistics
        let text_lengths: Vec<usize> = self.items.iter().map(|item| item.text.len()).collect();
        let text_length_stats = calculate_length_stats(&text_lengths);

        // Duration statistics
        let durations: Vec<f32> = self.items.iter().map(DatasetSample::duration).collect();
        let duration_stats = calculate_duration_stats(&durations);

        DatasetStatistics {
            total_items,
            total_duration,
            average_duration,
            language_distribution,
            speaker_distribution,
            text_length_stats,
            duration_stats,
        }
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (i, item) in self.items.iter().enumerate() {
            // Check for empty text
            if item.text.trim().is_empty() {
                errors.push(format!("Item {i}: Empty text"));
            }

            // Check for very short audio
            if item.duration() < 0.1 {
                warnings.push(format!(
                    "Item {}: Very short audio ({:.3}s)",
                    i,
                    item.duration()
                ));
            }

            // Check for very long audio
            if item.duration() > 30.0 {
                warnings.push(format!(
                    "Item {}: Very long audio ({:.1}s)",
                    i,
                    item.duration()
                ));
            }

            // Check for empty audio
            if item.audio.is_empty() {
                errors.push(format!("Item {i}: Empty audio"));
            }
        }

        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            items_validated: self.items.len(),
        })
    }
}

/// Calculate length statistics
fn calculate_length_stats(values: &[usize]) -> LengthStatistics {
    if values.is_empty() {
        return LengthStatistics {
            min: 0,
            max: 0,
            mean: 0.0,
            median: 0,
            std_dev: 0.0,
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable();

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let sum: usize = values.iter().sum();
    let mean = sum as f32 / values.len() as f32;
    let median = sorted[sorted.len() / 2];

    let variance: f32 = values
        .iter()
        .map(|&x| (x as f32 - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    let std_dev = variance.sqrt();

    LengthStatistics {
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

/// Calculate duration statistics
fn calculate_duration_stats(values: &[f32]) -> DurationStatistics {
    if values.is_empty() {
        return DurationStatistics {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let sum: f32 = values.iter().sum();
    let mean = sum / values.len() as f32;
    let median = sorted[sorted.len() / 2];

    let variance: f32 =
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt();

    DurationStatistics {
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LanguageCode;

    #[test]
    fn test_dataset_item_creation() {
        let audio = AudioBuffer::silence(1.0, 22050, 1);
        let item = DatasetItem::new(
            "test-001".to_string(),
            "Hello, world!".to_string(),
            audio,
            LanguageCode::EnUs,
        );

        assert_eq!(item.id, "test-001");
        assert_eq!(item.text, "Hello, world!");
        assert_eq!(item.language, LanguageCode::EnUs);
        assert!(item.phonemes.is_none());
        assert!(item.speaker_id().is_none());
    }

    #[test]
    fn test_memory_dataset() {
        let mut dataset = MemoryDataset::new("test-dataset".to_string());

        // Add test items
        for i in 0..3 {
            let audio = AudioBuffer::silence(1.0, 22050, 1);
            let item = DatasetItem::new(
                format!("item-{i:03}"),
                format!("Text number {i}"),
                audio,
                LanguageCode::EnUs,
            );
            dataset.add_item(item);
        }

        assert_eq!(dataset.name(), "test-dataset");
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());

        // Test item retrieval
        let item = dataset.get_item(1).unwrap();
        assert_eq!(item.id, "item-001");
        assert_eq!(item.text, "Text number 1");

        // Test statistics
        let stats = dataset.statistics();
        assert_eq!(stats.total_items, 3);
        assert!(stats.total_duration > 0.0);
        assert_eq!(stats.language_distribution[&LanguageCode::EnUs], 3);

        // Test validation
        let report = dataset.validate().unwrap();
        assert!(report.is_valid);
        assert_eq!(report.items_validated, 3);
    }

    #[test]
    fn test_windowed_sinc_resampling() {
        // Test with sine wave to verify frequency preservation
        let sample_rate = 44100;
        let new_sample_rate = 22050;
        let frequency = 1000.0; // 1kHz test tone
        let duration = 0.1; // 100ms

        let mut samples = Vec::new();
        let num_samples = (sample_rate as f32 * duration) as usize;

        // Generate sine wave
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }

        let original_audio = AudioData::new(samples, sample_rate, 1);
        let resampled = original_audio
            .resample_windowed_sinc(new_sample_rate)
            .unwrap();

        // Check that the resampled audio has the correct sample rate and length
        assert_eq!(resampled.sample_rate(), new_sample_rate);
        let expected_length = (num_samples * new_sample_rate as usize) / sample_rate as usize;
        assert!((resampled.samples().len() as i32 - expected_length as i32).abs() <= 1);

        // Verify that the frequency content is preserved (basic check)
        let resampled_samples = resampled.samples();
        assert!(!resampled_samples.is_empty());

        // Check that the signal hasn't been completely distorted
        let original_rms = original_audio.rms().unwrap();
        let resampled_rms = resampled.rms().unwrap();
        assert!((original_rms - resampled_rms).abs() < 0.1);
    }

    #[test]
    fn test_windowed_sinc_resampling_same_rate() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let audio = AudioData::new(samples.clone(), 44100, 1);

        let result = audio.resample_windowed_sinc(44100).unwrap();

        assert_eq!(result.sample_rate(), 44100);
        assert_eq!(result.samples(), &samples);
    }

    #[test]
    fn test_windowed_sinc_resampling_empty() {
        let audio = AudioData::new(vec![], 44100, 1);
        let result = audio.resample_windowed_sinc(22050).unwrap();

        assert_eq!(result.sample_rate(), 22050);
        assert!(result.samples().is_empty());
    }

    #[test]
    fn test_windowed_sinc_upsampling() {
        let samples = vec![1.0, 0.0, -1.0, 0.0];
        let audio = AudioData::new(samples, 22050, 1);

        let result = audio.resample_windowed_sinc(44100).unwrap();

        assert_eq!(result.sample_rate(), 44100);
        assert_eq!(result.samples().len(), 8); // Double the length

        // Check that the signal quality is maintained
        let rms = result.rms().unwrap();
        assert!(rms > 0.0);
    }

    #[test]
    fn test_windowed_sinc_downsampling() {
        let mut samples = Vec::new();
        for i in 0..88 {
            samples.push((i as f32 / 88.0).sin());
        }
        let audio = AudioData::new(samples, 44100, 1);

        let result = audio.resample_windowed_sinc(22050).unwrap();

        assert_eq!(result.sample_rate(), 22050);
        assert_eq!(result.samples().len(), 44); // Half the length

        // Check that the signal quality is maintained
        let rms = result.rms().unwrap();
        assert!(rms > 0.0);
    }

    #[test]
    fn test_kaiser_window_properties() {
        let window = AudioData::kaiser_window(64, 8.6);

        // Kaiser window should have symmetric properties
        assert_eq!(window.len(), 64);
        assert!((window[0] - window[63]).abs() < 1e-10);
        assert!((window[16] - window[47]).abs() < 1e-10);

        // Maximum should be at the center
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b));
        assert!((window[31] - max_val).abs() < 1e-10);
    }

    #[test]
    fn test_modified_bessel_i0_known_values() {
        // Test known values of modified Bessel function I0
        assert!((AudioData::modified_bessel_i0(0.0) - 1.0).abs() < 1e-10);
        assert!((AudioData::modified_bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-10);
        assert!((AudioData::modified_bessel_i0(2.0) - 2.2795853023360673).abs() < 1e-10);
    }

    // ------- ITU-R BS.1770-4 K-weighting filter tests -------

    /// RMS of a slice (helper for the K-weighting assertions).
    fn slice_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        (sum_sq / samples.len() as f64).sqrt() as f32
    }

    /// `n` samples of a unit-amplitude sine at `freq` Hz for `sample_rate`.
    fn sine(freq: f64, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(sample_rate)).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_k_weighting_attenuates_dc() {
        let sample_rate = 48_000u32;
        // 0.5 s of DC (constant 1.0) — the RLB high-pass must remove it.
        let audio = AudioData::new(vec![1.0f32; 24_000], sample_rate, 1);
        let filtered = audio.apply_k_weighting();

        assert_eq!(filtered.len(), 24_000);
        assert!(filtered.iter().all(|s| s.is_finite()));

        // Once the filter settles, the steady-state response to DC is ~0.
        let tail = &filtered[filtered.len() - 4_000..];
        let tail_rms = slice_rms(tail);
        assert!(
            tail_rms < 0.01,
            "DC should be strongly attenuated, tail RMS = {tail_rms}"
        );
    }

    #[test]
    fn test_k_weighting_boosts_2khz_relative_to_100hz() {
        let sample_rate = 48_000u32;
        let n = 24_000usize; // 0.5 s

        let low = AudioData::new(sine(100.0, sample_rate, n), sample_rate, 1);
        let high = AudioData::new(sine(2_000.0, sample_rate, n), sample_rate, 1);

        let low_filt = low.apply_k_weighting();
        let high_filt = high.apply_k_weighting();

        // Use the second half to skip the startup transient. Both inputs are
        // unit sines, so a single reference input RMS (~1/sqrt(2)) applies.
        let half = n / 2;
        let in_rms = slice_rms(&sine(100.0, sample_rate, n)[half..]);
        let low_gain = slice_rms(&low_filt[half..]) / in_rms;
        let high_gain = slice_rms(&high_filt[half..]) / in_rms;

        // K-weighting gently boosts highs: 2 kHz gain clearly exceeds 100 Hz
        // gain and sits above unity, while 100 Hz stays near unity.
        assert!(
            high_gain > low_gain * 1.1,
            "2 kHz gain {high_gain} should exceed 100 Hz gain {low_gain}"
        );
        assert!(
            high_gain > 1.0,
            "2 kHz should be boosted above unity, got {high_gain}"
        );
    }

    #[test]
    fn test_k_weighting_empty_input() {
        let audio = AudioData::new(Vec::new(), 48_000, 1);
        assert!(audio.apply_k_weighting().is_empty());
    }

    /// The derived K-weighting biquad coefficients must reproduce the canonical
    /// ITU-R BS.1770-4 reference values at 48 kHz (guards against the spurious
    /// √2-factor derivation that does not match the standard).
    #[test]
    fn test_k_weighting_coeffs_match_bs1770_reference_48k() {
        let pre = AudioData::k_weighting_pre_coeffs(48_000.0);
        let pre_ref = [
            1.535_124_859_586_97_f64,
            -2.691_696_189_406_38,
            1.198_392_810_852_85,
            -1.690_659_293_182_41,
            0.732_480_774_215_85,
        ];
        for (got, want) in pre.into_iter().zip(pre_ref) {
            assert!(
                (got - want).abs() < 1e-6,
                "pre-filter coeff {got} != reference {want}"
            );
        }

        let rlb = AudioData::k_weighting_rlb_coeffs(48_000.0);
        let rlb_ref = [
            1.0_f64,
            -2.0,
            1.0,
            -1.990_047_454_833_98,
            0.990_072_250_366_16,
        ];
        for (got, want) in rlb.into_iter().zip(rlb_ref) {
            assert!(
                (got - want).abs() < 1e-6,
                "RLB coeff {got} != reference {want}"
            );
        }
    }
}
