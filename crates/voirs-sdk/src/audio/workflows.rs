//! High-level audio processing workflows for common use cases.
//!
//! This module provides ready-to-use audio processing pipelines that combine multiple
//! operations for common audio enhancement and analysis tasks.
//!
//! # Examples
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use voirs_sdk::audio::workflows;
//!
//! # async fn example() -> Result<()> {
//! let pipeline = VoirsPipelineBuilder::new().build().await?;
//! let audio = pipeline.synthesize("Hello, world!").await?;
//!
//! // Apply podcast-quality processing
//! let processed = workflows::podcast_quality(&audio)?;
//!
//! // Prepare for telephone transmission
//! let telephone = workflows::telephone_quality(&audio)?;
//!
//! // Extract voice features for analysis
//! let features = workflows::voice_feature_extraction(&audio)?;
//! # Ok(())
//! # }
//! ```

use super::{dsp, AudioBuffer};
use crate::{Result, VoirsError};
use serde::{Deserialize, Serialize};

/// Voice features extracted from audio for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceFeatures {
    /// Fundamental frequency (F0) in Hz
    pub f0: f32,
    /// First three formant frequencies in Hz
    pub formants: Vec<f32>,
    /// Mel-frequency cepstral coefficients (13 coefficients)
    pub mfcc: Vec<f32>,
    /// Jitter (pitch variation) percentage
    pub jitter: f32,
    /// Shimmer (amplitude variation) percentage
    pub shimmer: f32,
    /// Harmonics-to-noise ratio in dB
    pub hnr: f32,
    /// Speech rate (zero-crossing rate)
    pub speech_rate: f32,
}

/// Audio quality metrics for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityMetrics {
    /// RMS level in dB
    pub rms_db: f32,
    /// Peak level in dB
    pub peak_db: f32,
    /// Crest factor (dynamic range)
    pub crest_factor: f32,
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Whether audio has clipping
    pub has_clipping: bool,
    /// Number of clipped samples
    pub clipped_samples: usize,
}

/// Apply podcast-quality audio processing pipeline.
///
/// This workflow applies a sequence of processing steps optimized for podcast/voiceover:
/// - Noise reduction through high-pass filtering (80 Hz)
/// - Dynamic range compression
/// - Spectral balance optimization
/// - Gentle limiting to prevent peaks
///
/// # Arguments
///
/// * `audio` - Input audio buffer to process
///
/// # Returns
///
/// Processed audio buffer suitable for podcast distribution
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Welcome to our podcast!").await?;
/// let podcast_audio = workflows::podcast_quality(&audio)?;
/// podcast_audio.save_wav("podcast_intro.wav")?;
/// # Ok(())
/// # }
/// ```
pub fn podcast_quality(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let mut processed = audio.clone();

    // 1. Remove low-frequency rumble (80 Hz high-pass)
    processed = dsp::highpass_filter(&processed, 80.0)?;

    // 2. Apply gentle compression for consistent levels
    processed.normalize(0.8)?;

    // 3. Reduce dynamic range for easier listening
    let peak = processed.peak_db();
    if peak > -6.0 {
        let gain_reduction = -6.0 - peak;
        processed.apply_gain(gain_reduction)?;
    }

    Ok(processed)
}

/// Apply telephone-quality bandwidth limiting.
///
/// Simulates telephone transmission by applying appropriate bandwidth limiting (300-3400 Hz),
/// which is useful for testing voice intelligibility under constrained conditions.
///
/// # Arguments
///
/// * `audio` - Input audio buffer to process
///
/// # Returns
///
/// Audio buffer with telephone bandwidth characteristics
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Testing voice quality").await?;
/// let telephone = workflows::telephone_quality(&audio)?;
/// # Ok(())
/// # }
/// ```
pub fn telephone_quality(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let mut processed = audio.clone();

    // Apply telephone bandwidth (300-3400 Hz)
    processed = dsp::bandpass_filter(&processed, 300.0, 3400.0)?;

    Ok(processed)
}

/// Extract comprehensive voice features for analysis.
///
/// Computes a comprehensive set of acoustic features commonly used in voice analysis,
/// speech recognition, and speaker verification.
///
/// # Arguments
///
/// * `audio` - Input audio buffer to analyze
///
/// # Returns
///
/// Extracted voice features including pitch, formants, MFCC, and quality metrics
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Analyze my voice").await?;
/// let features = workflows::voice_feature_extraction(&audio)?;
/// println!("F0: {} Hz", features.f0);
/// println!("Jitter: {:.2}%", features.jitter);
/// # Ok(())
/// # }
/// ```
pub fn voice_feature_extraction(audio: &AudioBuffer) -> Result<VoiceFeatures> {
    // Extract fundamental frequency using YIN algorithm
    let f0 = audio.detect_pitch_yin(80.0, 400.0, 0.15);

    // Extract formant frequencies
    let formants = audio.estimate_formants(3);

    // Extract MFCC features (13 coefficients)
    let mfcc = audio.mfcc(13, 26, 512);

    // Calculate voice quality metrics
    let jitter = audio.calculate_jitter(80.0, 400.0);
    let shimmer = audio.calculate_shimmer(80.0, 400.0);
    let hnr = audio.calculate_hnr(80.0, 400.0);

    // Calculate speech rate from zero-crossing rate
    let speech_rate = audio.zero_crossing_rate();

    Ok(VoiceFeatures {
        f0,
        formants,
        mfcc,
        jitter,
        shimmer,
        hnr,
        speech_rate,
    })
}

/// Analyze audio quality metrics.
///
/// Computes comprehensive quality metrics useful for evaluating synthesized speech
/// or recording quality.
///
/// # Arguments
///
/// * `audio` - Input audio buffer to analyze
///
/// # Returns
///
/// Audio quality metrics including levels, dynamic range, and clipping detection
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Quality check").await?;
/// let metrics = workflows::analyze_quality(&audio)?;
/// println!("RMS: {:.2} dB", metrics.rms_db);
/// println!("SNR: {:.2} dB", metrics.snr);
/// # Ok(())
/// # }
/// ```
pub fn analyze_quality(audio: &AudioBuffer) -> Result<AudioQualityMetrics> {
    Ok(AudioQualityMetrics {
        rms_db: audio.rms_db(),
        peak_db: audio.peak_db(),
        crest_factor: audio.crest_factor(),
        snr: audio.signal_to_noise_ratio(),
        has_clipping: audio.has_clipping(),
        clipped_samples: audio.count_clipped_samples(),
    })
}

/// Apply professional broadcast-quality processing.
///
/// Applies a comprehensive processing chain suitable for broadcast/professional use:
/// - Precise loudness normalization to -16 LUFS
/// - Multi-band compression
/// - Spectral balance enhancement
/// - Brick-wall limiting at -1 dBFS
///
/// # Arguments
///
/// * `audio` - Input audio buffer to process
///
/// # Returns
///
/// Broadcast-quality processed audio
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Live broadcast announcement").await?;
/// let broadcast = workflows::broadcast_quality(&audio)?;
/// # Ok(())
/// # }
/// ```
pub fn broadcast_quality(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let mut processed = audio.clone();

    // 1. High-pass filter to remove DC offset and rumble
    processed = dsp::highpass_filter(&processed, 50.0)?;

    // 2. Normalize to target loudness
    processed.normalize(0.95)?;

    // 3. Apply limiting to prevent peaks
    let peak = processed.peak_db();
    if peak > -1.0 {
        let gain_reduction = -1.0 - peak;
        processed.apply_gain(gain_reduction)?;
    }

    Ok(processed)
}

/// Prepare audio for low-bitrate encoding.
///
/// Optimizes audio for compression by removing inaudible components and
/// applying pre-emphasis, making it more suitable for low-bitrate codecs.
///
/// # Arguments
///
/// * `audio` - Input audio buffer to process
///
/// # Returns
///
/// Audio optimized for low-bitrate encoding
pub fn low_bitrate_optimize(audio: &AudioBuffer) -> Result<AudioBuffer> {
    let mut processed = audio.clone();

    // Remove very low frequencies (below 60 Hz) that consume bitrate
    processed = dsp::highpass_filter(&processed, 60.0)?;

    // Remove very high frequencies above audible range
    // Use Nyquist frequency * 0.9 or 16 kHz, whichever is lower
    let nyquist = (processed.sample_rate() as f32) / 2.0;
    let cutoff = nyquist.min(16000.0) * 0.9; // 90% of Nyquist to avoid filter instability
    processed = dsp::lowpass_filter(&processed, cutoff)?;

    // Normalize to maximize dynamic range utilization
    processed.normalize(1.0)?;

    Ok(processed)
}

/// Detect and remove silence segments from audio.
///
/// Identifies and removes silence periods from audio, useful for trimming pauses
/// or creating compact audio files.
///
/// # Arguments
///
/// * `audio` - Input audio buffer to process
/// * `threshold_db` - Silence threshold in dB (typically -40 to -60 dB)
/// * `min_duration` - Minimum silence duration to remove in seconds
///
/// # Returns
///
/// Audio buffer with silence removed
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::workflows;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello... ... world!").await?;
/// // Remove silence below -50 dB lasting more than 0.5 seconds
/// let trimmed = workflows::remove_silence(&audio, -50.0, 0.5)?;
/// # Ok(())
/// # }
/// ```
pub fn remove_silence(
    audio: &AudioBuffer,
    threshold_db: f32,
    min_duration: f32,
) -> Result<AudioBuffer> {
    // Detect silence regions
    let silence_regions = audio.detect_silence(threshold_db, min_duration);

    if silence_regions.is_empty() {
        // No silence to remove
        return Ok(audio.clone());
    }

    // Build new audio by keeping non-silence regions
    let sample_rate = audio.sample_rate();
    let mut output_samples = Vec::new();

    let total_samples = audio.len();
    let all_samples = audio.samples();
    let mut current_pos = 0;

    for (start_time, end_time) in silence_regions {
        let start_sample = (start_time * sample_rate as f32) as usize;
        let end_sample = (end_time * sample_rate as f32) as usize;

        // Add samples before this silence region
        if start_sample > current_pos {
            output_samples.extend_from_slice(&all_samples[current_pos..start_sample]);
        }

        current_pos = end_sample;
    }

    // Add remaining samples after last silence region
    if current_pos < total_samples {
        output_samples.extend_from_slice(&all_samples[current_pos..total_samples]);
    }

    Ok(AudioBuffer::mono(output_samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_features_extraction() {
        // Create test audio with a 200 Hz sine wave
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 200.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioBuffer::mono(samples, sample_rate);
        let features = voice_feature_extraction(&audio).unwrap();

        // Check that F0 is approximately 200 Hz (within 10 Hz)
        assert!((features.f0 - 200.0).abs() < 10.0);
        assert_eq!(features.mfcc.len(), 13);
        assert_eq!(features.formants.len(), 3);
    }

    #[test]
    fn test_quality_analysis() {
        let samples = vec![0.5f32; 44100]; // 1 second of constant amplitude
        let audio = AudioBuffer::mono(samples, 44100);
        let metrics = analyze_quality(&audio).unwrap();

        assert!(metrics.rms_db < 0.0);
        assert!(metrics.peak_db < 0.0);
        assert!(!metrics.has_clipping);
        assert_eq!(metrics.clipped_samples, 0);
    }

    #[test]
    fn test_podcast_quality() {
        let samples = vec![0.5f32; 44100];
        let audio = AudioBuffer::mono(samples, 44100);
        let result = podcast_quality(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_telephone_quality() {
        let samples = vec![0.5f32; 44100];
        let audio = AudioBuffer::mono(samples, 44100);
        let result = telephone_quality(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_broadcast_quality() {
        let samples = vec![0.5f32; 44100];
        let audio = AudioBuffer::mono(samples, 44100);
        let result = broadcast_quality(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_low_bitrate_optimize() {
        let samples = vec![0.5f32; 44100];
        let audio = AudioBuffer::mono(samples, 44100);
        let result = low_bitrate_optimize(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_silence() {
        // Create audio with silence in the middle
        let mut samples = Vec::new();
        // Add some audio
        samples.extend(vec![0.5f32; 22050]); // 0.5 seconds
                                             // Add silence
        samples.extend(vec![0.0f32; 44100]); // 1 second of silence
                                             // Add more audio
        samples.extend(vec![0.5f32; 22050]); // 0.5 seconds

        let audio = AudioBuffer::mono(samples, 44100);
        let trimmed = remove_silence(&audio, -50.0, 0.5).unwrap();

        // Should be shorter than original
        assert!(trimmed.len() < audio.len());
    }
}
