//! Integrated Signal Processing for Emotion Expression
//!
//! This module provides a unified interface for comprehensive emotion-aware signal processing,
//! combining formant manipulation, spectral processing, and breath control into a cohesive
//! emotion synthesis pipeline.
//!
//! ## Features
//!
//! - **Unified Pipeline**: Single interface for all signal processing operations
//! - **Emotion-Aware Processing**: All operations respect emotional context
//! - **Real-time Capable**: Optimized for streaming audio synthesis
//! - **Modular Design**: Enable/disable processing stages as needed
//! - **Quality Presets**: Pre-configured settings for different use cases
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::signal_processing::{SignalProcessor, SignalProcessingConfig, ProcessingQuality};
//! use voirs_emotion::types::Emotion;
//!
//! // Create processor with high-quality settings
//! let config = SignalProcessingConfig::preset(ProcessingQuality::High);
//! let mut processor = SignalProcessor::new(config, 44100.0);
//!
//! // Process audio with emotion
//! let audio = vec![0.5; 88200]; // 2 seconds of audio
//! let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    breath::{BreathConfig, BreathPauseController, Pause, PauseType},
    formant::FormantAnalyzer,
    spectral::{SpectralConfig, SpectralProcessor},
    types::Emotion,
    Error, Result,
};
use scirs2_core::numeric::Complex;
use scirs2_fft::{irfft, rfft};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Quality preset for signal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingQuality {
    /// Low quality, optimized for speed and minimal CPU usage
    Low,
    /// Medium quality, balanced between quality and performance
    Medium,
    /// High quality, optimized for best audio quality
    High,
    /// Ultra quality, maximum quality regardless of performance
    Ultra,
}

/// Configuration for integrated signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingConfig {
    /// Enable formant processing
    pub enable_formant: bool,
    /// Enable spectral processing
    pub enable_spectral: bool,
    /// Enable breath and pause processing
    pub enable_breath: bool,
    /// Processing quality level
    pub quality: ProcessingQuality,
    /// FFT size for frequency domain processing
    pub fft_size: usize,
    /// Overlap factor for windowed processing (0.0-0.75)
    pub overlap_factor: f32,
    /// Enable adaptive processing based on emotion intensity
    pub adaptive_intensity: bool,
}

impl SignalProcessingConfig {
    /// Create configuration for a specific quality preset
    pub fn preset(quality: ProcessingQuality) -> Self {
        let (fft_size, overlap_factor) = match quality {
            ProcessingQuality::Low => (1024, 0.25),
            ProcessingQuality::Medium => (2048, 0.50),
            ProcessingQuality::High => (4096, 0.75),
            ProcessingQuality::Ultra => (8192, 0.75),
        };

        Self {
            enable_formant: true,
            enable_spectral: true,
            enable_breath: true,
            quality,
            fft_size,
            overlap_factor,
            adaptive_intensity: true,
        }
    }

    /// Create configuration with all features enabled at high quality
    pub fn full() -> Self {
        Self::preset(ProcessingQuality::High)
    }

    /// Create minimal configuration for real-time processing
    pub fn minimal() -> Self {
        Self {
            enable_formant: false,
            enable_spectral: true,
            enable_breath: false,
            quality: ProcessingQuality::Low,
            fft_size: 1024,
            overlap_factor: 0.25,
            adaptive_intensity: false,
        }
    }
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self::preset(ProcessingQuality::Medium)
    }
}

/// Integrated signal processor for emotion-aware audio synthesis
pub struct SignalProcessor {
    config: SignalProcessingConfig,
    sample_rate: f32,
    formant_analyzer: Option<FormantAnalyzer>,
    spectral_processor: Option<SpectralProcessor>,
    breath_controller: Option<BreathPauseController>,
}

impl SignalProcessor {
    /// Create a new signal processor with the given configuration
    pub fn new(config: SignalProcessingConfig, sample_rate: f32) -> Self {
        let formant_analyzer = if config.enable_formant {
            Some(FormantAnalyzer::new(sample_rate))
        } else {
            None
        };

        let spectral_processor = if config.enable_spectral {
            Some(SpectralProcessor::new(sample_rate, config.fft_size))
        } else {
            None
        };

        let breath_controller = if config.enable_breath {
            Some(BreathPauseController::new(
                BreathConfig::default(),
                sample_rate,
            ))
        } else {
            None
        };

        Self {
            config,
            sample_rate,
            formant_analyzer,
            spectral_processor,
            breath_controller,
        }
    }

    /// Process audio with emotion-aware signal processing
    ///
    /// # Arguments
    ///
    /// * `audio` - Input audio samples
    /// * `emotion` - Emotion to apply
    /// * `intensity` - Emotion intensity (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Processed audio with emotion-specific characteristics
    pub fn process_with_emotion(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let intensity = intensity.clamp(0.0, 1.0);
        let mut output = audio.to_vec();

        // Apply adaptive intensity scaling based on configuration
        let effective_intensity = if self.config.adaptive_intensity {
            self.calculate_adaptive_intensity(intensity, emotion)
        } else {
            intensity
        };

        // Stage 1: Spectral processing (frequency domain shaping)
        if self.config.enable_spectral {
            output = self.apply_spectral_processing(&output, emotion, effective_intensity)?;
        }

        // Stage 2: Breath and pause insertion (naturalness)
        if self.config.enable_breath {
            output = self.apply_breath_processing(&output, emotion, effective_intensity)?;
        }

        Ok(output)
    }

    /// Process text with emotion-aware breath and pause insertion
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to analyze
    /// * `audio` - Audio samples corresponding to the text
    /// * `emotion` - Emotion context for pause timing
    /// * `intensity` - Emotion intensity
    ///
    /// # Returns
    ///
    /// Audio with natural pauses and breath sounds inserted
    pub fn process_text_with_emotion(
        &mut self,
        text: &str,
        audio: &[f32],
        emotion: &Emotion,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        if !self.config.enable_breath {
            return Ok(audio.to_vec());
        }

        let controller = self
            .breath_controller
            .as_mut()
            .ok_or_else(|| Error::Processing("Breath controller not initialized".to_string()))?;

        // Analyze text for pause locations
        let pauses = controller.process_text(text, emotion);

        // Insert pauses and breath sounds
        Ok(controller.insert_pauses(audio, &pauses))
    }

    /// Extract basic emotion features from audio using signal analysis
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// Detected emotion and confidence score (simplified heuristic)
    pub fn analyze_emotion(&self, audio: &[f32]) -> Result<(Emotion, f32)> {
        if audio.is_empty() {
            return Ok((Emotion::Neutral, 0.0));
        }

        // Analyze spectral characteristics
        let spectral_features = self.analyze_spectral_features(audio)?;

        // Combine features to estimate emotion
        let emotion = self.estimate_emotion_from_features(&spectral_features);
        let confidence = self.calculate_confidence(&spectral_features);

        Ok((emotion, confidence))
    }

    /// Get current processing configuration
    pub fn config(&self) -> &SignalProcessingConfig {
        &self.config
    }

    /// Update processing configuration
    pub fn set_config(&mut self, config: SignalProcessingConfig) {
        self.config = config;
    }

    // Private helper methods

    fn calculate_adaptive_intensity(&self, intensity: f32, emotion: &Emotion) -> f32 {
        // Adjust intensity based on emotion type and quality setting
        let quality_factor = match self.config.quality {
            ProcessingQuality::Low => 0.7,
            ProcessingQuality::Medium => 0.85,
            ProcessingQuality::High => 1.0,
            ProcessingQuality::Ultra => 1.15,
        };

        let emotion_factor = match emotion {
            Emotion::Excited | Emotion::Angry => 1.2, // Boost high-energy emotions
            Emotion::Sad | Emotion::Calm => 0.9,      // Reduce low-energy emotions
            _ => 1.0,
        };

        (intensity * quality_factor * emotion_factor).clamp(0.0, 1.0)
    }

    /// Apply emotion-specific spectral shaping in the frequency domain.
    ///
    /// The audio is analyzed with a Hann-windowed Short-Time Fourier Transform
    /// (via [`scirs2_fft::rfft`]). For every frame the magnitude spectrum is
    /// transformed by the crate's [`SpectralProcessor`] (spectral tilt, harmonic
    /// enhancement, high-frequency emphasis and centroid shift) while the phase
    /// is preserved. The frame is then inverted with [`scirs2_fft::irfft`] and
    /// summed back with weighted overlap-add, normalizing by the accumulated
    /// squared analysis window so that constant-overlap-add (COLA) reconstruction
    /// holds for the 75% hop used here.
    fn apply_spectral_processing(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        // Build the emotion-specific, intensity-scaled configuration and load it
        // into the persistent spectral processor.
        let mut config = SpectralConfig::from_emotion(emotion.clone());
        config.tilt *= intensity;
        // Interpolate the multiplicative parameters towards their neutral value
        // of 1.0 by `intensity` so a zero intensity is a true pass-through.
        config.harmonic_boost = 1.0 + (config.harmonic_boost - 1.0) * intensity;
        config.hf_emphasis = 1.0 + (config.hf_emphasis - 1.0) * intensity;
        config.centroid_shift *= intensity;

        let fft_size = {
            let spectral = self.spectral_processor.as_mut().ok_or_else(|| {
                Error::Processing("Spectral processor not initialized".to_string())
            })?;
            spectral.set_config(config);
            spectral.fft_size()
        };

        // Frames shorter than one FFT window cannot be meaningfully analyzed in
        // the frequency domain; leave such audio untouched.
        if audio.len() < fft_size || fft_size == 0 {
            return Ok(audio.to_vec());
        }

        // 75% overlap (hop = N/4) gives smooth COLA reconstruction with a Hann
        // window. A coarse fundamental estimate feeds the harmonic enhancement.
        let hop = (fft_size / 4).max(1);
        let f0 = estimate_fundamental(audio, self.sample_rate);
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / fft_size as f32).cos()))
            .collect();

        let mut output = vec![0.0f32; audio.len()];
        let mut window_norm = vec![0.0f32; audio.len()];

        let mut pos = 0usize;
        while pos + fft_size <= audio.len() {
            // Analysis: window the frame and forward-transform it.
            let frame_f64: Vec<f64> = (0..fft_size)
                .map(|i| (audio[pos + i] * window[i]) as f64)
                .collect();

            let mut spectrum = match rfft(&frame_f64, Some(fft_size)) {
                Ok(spec) => spec,
                Err(e) => return Err(Error::Processing(format!("rfft failed: {e:?}"))),
            };

            // Extract magnitudes, transform them, then re-apply onto the
            // complex spectrum keeping the original phase of each bin.
            let mut magnitudes: Vec<f32> = spectrum
                .iter()
                .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
                .collect();

            {
                let spectral = self.spectral_processor.as_ref().ok_or_else(|| {
                    Error::Processing("Spectral processor not initialized".to_string())
                })?;
                spectral.process_spectrum(&mut magnitudes, f0);
            }

            for (bin, mag) in spectrum.iter_mut().zip(magnitudes.iter()) {
                let old_mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
                if old_mag > 1e-12 {
                    let gain = (*mag as f64) / old_mag;
                    *bin = Complex::new(bin.re * gain, bin.im * gain);
                } else {
                    // Reconstruct a (zero-phase) bin where the input had no energy.
                    *bin = Complex::new(*mag as f64, 0.0);
                }
            }

            // Synthesis: inverse transform and weighted overlap-add.
            let frame_out = match irfft(&spectrum, Some(fft_size)) {
                Ok(out) => out,
                Err(e) => return Err(Error::Processing(format!("irfft failed: {e:?}"))),
            };

            for i in 0..fft_size {
                let w = window[i];
                output[pos + i] += frame_out[i] as f32 * w;
                window_norm[pos + i] += w * w;
            }

            pos += hop;
        }

        // Normalize by the accumulated squared window. Samples never covered by
        // a full frame (the trailing tail) fall back to the dry signal.
        for i in 0..output.len() {
            if window_norm[i] > 1e-6 {
                output[i] /= window_norm[i];
            } else {
                output[i] = audio[i];
            }
        }

        Ok(output)
    }

    fn apply_breath_processing(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        let controller = self
            .breath_controller
            .as_mut()
            .ok_or_else(|| Error::Processing("Breath controller not initialized".to_string()))?;

        // Create simple pause pattern based on audio length
        let duration = audio.len() as f32 / self.sample_rate;
        let mut pauses = Vec::new();

        // Add pauses at natural intervals
        let config = BreathConfig::from_emotion(emotion.clone());
        let pause_interval = 60.0 / config.frequency; // frequency is breaths per minute

        let mut time = pause_interval;
        while time < duration {
            pauses.push(Pause {
                pause_type: PauseType::Breath,
                position: (time * self.sample_rate) as usize,
                duration: (config.duration * self.sample_rate) as usize,
                insert_breath: true,
            });
            time += pause_interval;
        }

        Ok(controller.insert_pauses(audio, &pauses))
    }

    fn analyze_spectral_features(&self, audio: &[f32]) -> Result<SpectralFeatures> {
        // Calculate basic spectral features
        let energy = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let energy = energy.sqrt();

        // Calculate spectral centroid (simplified)
        let centroid = self.calculate_spectral_centroid(audio)?;

        // Calculate spectral rolloff
        let rolloff = self.calculate_spectral_rolloff(audio)?;

        Ok(SpectralFeatures {
            energy,
            centroid,
            rolloff,
        })
    }

    /// Calculate the spectral centroid (brightness) of the audio in Hz.
    ///
    /// Uses a Hann-windowed real FFT ([`scirs2_fft::rfft`]) and computes
    /// `Σ(f_k·|X_k|) / Σ|X_k|` with `f_k = k · sample_rate / N`.
    fn calculate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }
        Ok(spectral_centroid_hz(audio, self.sample_rate))
    }

    fn calculate_spectral_rolloff(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let total_energy: f32 = audio.iter().map(|x| x * x).sum();
        let threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        for (i, &sample) in audio.iter().enumerate() {
            cumulative_energy += sample * sample;
            if cumulative_energy >= threshold {
                return Ok(i as f32 / audio.len() as f32);
            }
        }

        Ok(1.0)
    }

    fn estimate_emotion_from_features(&self, spectral: &SpectralFeatures) -> Emotion {
        // Simple heuristic-based emotion estimation. The centroid is expressed in
        // Hz, so it is normalized by the Nyquist frequency to a [0, 1] brightness.
        let nyquist = (self.sample_rate / 2.0).max(1.0);
        let brightness = (spectral.centroid / nyquist).clamp(0.0, 1.0);
        let high_energy = spectral.energy > 0.5;
        let high_centroid = brightness > 0.25;

        match (high_energy, high_centroid) {
            (true, true) => Emotion::Excited,
            (true, false) => Emotion::Angry,
            (false, true) => Emotion::Calm,
            (false, false) => Emotion::Sad,
        }
    }

    fn calculate_confidence(&self, spectral: &SpectralFeatures) -> f32 {
        // Calculate confidence based on feature strength. The centroid (Hz) is
        // normalized to a [0, 1] brightness via the Nyquist frequency.
        let nyquist = (self.sample_rate / 2.0).max(1.0);
        let brightness = (spectral.centroid / nyquist).clamp(0.0, 1.0);
        let energy_confidence = spectral.energy.clamp(0.0, 1.0);
        let spectral_confidence = (brightness * 4.0).clamp(0.0, 1.0);

        (energy_confidence + spectral_confidence) / 2.0
    }
}

/// Spectral features for emotion analysis
#[derive(Debug, Clone)]
struct SpectralFeatures {
    energy: f32,
    centroid: f32,
    rolloff: f32,
}

/// Compute the spectral centroid of `audio` in Hz using a real FFT.
///
/// A Hann window is applied (over the first power-of-two-friendly span up to
/// the whole signal) before the [`scirs2_fft::rfft`]. The centroid is the
/// magnitude-weighted mean frequency `Σ(f_k·|X_k|) / Σ|X_k|` with
/// `f_k = k · sample_rate / N`. Returns `0.0` for empty/silent input.
fn spectral_centroid_hz(audio: &[f32], sample_rate: f32) -> f32 {
    if audio.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }

    let n = audio.len();
    let windowed: Vec<f64> = audio
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
            (x * w) as f64
        })
        .collect();

    let spectrum = match rfft(&windowed, Some(n)) {
        Ok(spec) => spec,
        Err(_) => return 0.0,
    };

    let mut weighted = 0.0f64;
    let mut magnitude = 0.0f64;
    let bin_hz = sample_rate as f64 / n as f64;
    for (k, c) in spectrum.iter().enumerate() {
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        weighted += (k as f64 * bin_hz) * mag;
        magnitude += mag;
    }

    if magnitude > 1e-12 {
        (weighted / magnitude) as f32
    } else {
        0.0
    }
}

/// Estimate the fundamental frequency (Hz) of `audio` via autocorrelation.
///
/// Returns `Some(f0)` when a plausible periodicity is found within the typical
/// human voice range (60–500 Hz), otherwise `None` (harmonic enhancement is
/// then skipped by the spectral processor).
///
/// `pub(crate)` so other modules (e.g. [`crate::sdk_integration`]) can reuse
/// the same real F0 estimator instead of re-deriving pitch statistics
/// on their own.
pub(crate) fn estimate_fundamental(audio: &[f32], sample_rate: f32) -> Option<f32> {
    if audio.len() < 2 || sample_rate <= 0.0 {
        return None;
    }

    let min_f0 = 60.0f32;
    let max_f0 = 500.0f32;
    let min_lag = (sample_rate / max_f0).floor() as usize;
    let max_lag = ((sample_rate / min_f0).ceil() as usize).min(audio.len() - 1);
    if min_lag < 1 || max_lag <= min_lag {
        return None;
    }

    let energy: f32 = audio.iter().map(|&x| x * x).sum();
    if energy < 1e-9 {
        return None;
    }

    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut corr = 0.0f32;
        for i in 0..(audio.len() - lag) {
            corr += audio[i] * audio[i + lag];
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Require the peak autocorrelation to be a reasonable fraction of the
    // zero-lag energy to consider the signal voiced.
    if best_lag > 0 && best_corr > 0.3 * energy {
        Some(sample_rate / best_lag as f32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_processor_creation() {
        let config = SignalProcessingConfig::default();
        let processor = SignalProcessor::new(config, 44100.0);
        assert_eq!(processor.sample_rate, 44100.0);
    }

    #[test]
    fn test_processing_quality_presets() {
        let low = SignalProcessingConfig::preset(ProcessingQuality::Low);
        let high = SignalProcessingConfig::preset(ProcessingQuality::High);

        assert!(low.fft_size < high.fft_size);
        assert!(low.overlap_factor < high.overlap_factor);
    }

    #[test]
    fn test_process_with_emotion() -> Result<()> {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let audio = vec![0.5; 4410]; // 0.1 seconds
        let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;

        assert!(!result.is_empty());
        assert_eq!(result.len(), audio.len());

        Ok(())
    }

    #[test]
    fn test_analyze_emotion() {
        let config = SignalProcessingConfig::full();
        let processor = SignalProcessor::new(config, 44100.0);

        let audio = vec![0.5; 4410]; // 0.1 seconds
        let (emotion, confidence) = processor.analyze_emotion(&audio).unwrap();

        assert!(matches!(
            emotion,
            Emotion::Happy | Emotion::Sad | Emotion::Angry | Emotion::Calm | Emotion::Excited
        ));
        assert!((0.0..=1.0).contains(&confidence));
    }

    #[test]
    fn test_adaptive_intensity() {
        let config = SignalProcessingConfig::default();
        let processor = SignalProcessor::new(config, 44100.0);

        let intensity_excited = processor.calculate_adaptive_intensity(0.8, &Emotion::Excited);
        let intensity_calm = processor.calculate_adaptive_intensity(0.8, &Emotion::Calm);

        assert!(intensity_excited > intensity_calm);
    }

    #[test]
    fn test_empty_audio_handling() -> Result<()> {
        let config = SignalProcessingConfig::default();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let empty_audio: Vec<f32> = vec![];
        let result = processor.process_with_emotion(&empty_audio, &Emotion::Happy, 0.8)?;

        assert!(result.is_empty());

        Ok(())
    }

    #[test]
    fn test_config_updates() {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let new_config = SignalProcessingConfig::full();
        processor.set_config(new_config.clone());

        assert_eq!(processor.config().enable_formant, new_config.enable_formant);
        assert_eq!(
            processor.config().enable_spectral,
            new_config.enable_spectral
        );
        assert_eq!(processor.config().enable_breath, new_config.enable_breath);
    }

    #[test]
    fn test_spectral_features_calculation() {
        let config = SignalProcessingConfig::full();
        let processor = SignalProcessor::new(config, 44100.0);

        // Create test signal with known characteristics
        let audio: Vec<f32> = (0..4410)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
            .collect();

        let features = processor.analyze_spectral_features(&audio).unwrap();

        assert!(features.energy > 0.0);
        assert!(features.centroid >= 0.0);
        assert!((0.0..=1.0).contains(&features.rolloff));
    }

    /// Build a single-frequency sine tone.
    fn make_tone(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_spectral_centroid_matches_tone_frequency() {
        let sr = 44100.0;
        let audio = make_tone(2000.0, sr, 8192);
        let centroid = spectral_centroid_hz(&audio, sr);
        // A pure tone's centroid should sit close to its frequency.
        assert!(
            (centroid - 2000.0).abs() < 200.0,
            "centroid {centroid} not near 2000 Hz"
        );
    }

    #[test]
    fn test_spectral_centroid_hf_higher_than_lf() {
        let sr = 44100.0;
        let lf = make_tone(300.0, sr, 8192);
        let hf = make_tone(6000.0, sr, 8192);
        let lf_centroid = spectral_centroid_hz(&lf, sr);
        let hf_centroid = spectral_centroid_hz(&hf, sr);
        assert!(
            hf_centroid > lf_centroid,
            "hf {hf_centroid} should exceed lf {lf_centroid}"
        );
    }

    #[test]
    fn test_spectral_centroid_empty_is_zero() {
        assert_eq!(spectral_centroid_hz(&[], 44100.0), 0.0);
    }

    #[test]
    fn test_apply_spectral_processing_transforms_audio() {
        let sr = 44100.0;
        let config = SignalProcessingConfig::full();
        let mut processor = SignalProcessor::new(config, sr);

        // Broadband-ish signal: sum of two tones so HF emphasis has material to act on.
        let audio: Vec<f32> = (0..8192)
            .map(|i| {
                let t = i as f32 / sr;
                0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
            })
            .collect();

        let processed = processor
            .apply_spectral_processing(&audio, &Emotion::Happy, 1.0)
            .unwrap();

        assert_eq!(processed.len(), audio.len());
        // Output must genuinely differ from the dry input somewhere in the
        // covered region (Happy boosts highs / harmonics).
        let max_diff = audio
            .iter()
            .zip(processed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-3, "spectral processing did not alter audio");
    }

    #[test]
    fn test_estimate_fundamental_recovers_tone() {
        let sr = 16000.0;
        let audio = make_tone(150.0, sr, 4096);
        let f0 = estimate_fundamental(&audio, sr).expect("voiced tone should yield f0");
        assert!((f0 - 150.0).abs() < 10.0, "f0 {f0} not near 150 Hz");
    }
}
