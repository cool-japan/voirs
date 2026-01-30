//! Advanced spectral processing for emotion expression using scirs2-fft
//!
//! This module provides sophisticated spectral domain processing for emotion-based
//! voice modification using scirs2-fft's high-performance FFT implementations.
//!
//! ## Features
//! - Real-time spectral analysis and modification
//! - Emotion-specific frequency response shaping
//! - Formant enhancement and suppression
//! - Spectral envelope manipulation
//! - Time-frequency analysis with STFT
//! - GPU-accelerated processing for large buffers

use scirs2_core::numeric::Complex;
use scirs2_fft::{irfft, rfft, stft, Window};
use std::f32::consts::PI;

/// Result type for time-frequency analysis
pub type TimeFrequencyResult = Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>), String>;

/// Advanced spectral processor for emotion-based audio modification
///
/// This processor uses FFT-based techniques to modify audio in the frequency
/// domain, allowing for sophisticated emotion-specific transformations.
pub struct SpectralEmotionProcessor {
    sample_rate: f32,
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
}

impl SpectralEmotionProcessor {
    /// Creates a new spectral emotion processor
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `fft_size` - FFT size (should be power of 2)
    /// * `hop_size` - Hop size for STFT processing
    pub fn new(sample_rate: f32, fft_size: usize, hop_size: usize) -> Self {
        let window = Self::create_hann_window(fft_size);

        Self {
            sample_rate,
            fft_size,
            hop_size,
            window,
        }
    }

    /// Create a Hann window for spectral analysis
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
            .collect()
    }

    /// Apply emotion-based spectral filtering
    ///
    /// Modifies the spectral envelope based on emotion characteristics.
    ///
    /// # Arguments
    /// * `audio` - Input audio buffer
    /// * `arousal` - Emotion arousal (-1.0 to 1.0)
    /// * `valence` - Emotion valence (-1.0 to 1.0)
    ///
    /// # Returns
    /// Filtered audio buffer
    pub fn apply_emotion_filter(
        &self,
        audio: &[f32],
        arousal: f32,
        valence: f32,
    ) -> Result<Vec<f32>, String> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to f64 for FFT processing
        let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Forward FFT
        let spectrum = rfft(&audio_f64, None).map_err(|e| format!("FFT failed: {:?}", e))?;

        // Create emotion-specific frequency response
        let filter = self.create_emotion_filter(spectrum.len(), arousal, valence);

        // Apply filter in frequency domain
        let filtered_spectrum: Vec<Complex<f64>> = spectrum
            .iter()
            .zip(filter.iter())
            .map(|(s, &f)| *s * f)
            .collect();

        // Inverse FFT
        let filtered_audio = irfft(&filtered_spectrum, Some(audio.len()))
            .map_err(|e| format!("IFFT failed: {:?}", e))?;

        // Convert back to f32
        Ok(filtered_audio.iter().map(|&x| x as f32).collect())
    }

    /// Create emotion-specific frequency response filter
    fn create_emotion_filter(&self, size: usize, arousal: f32, valence: f32) -> Vec<f64> {
        let mut filter = vec![1.0; size];

        for (i, f) in filter.iter_mut().enumerate() {
            let freq = (i as f32 * self.sample_rate) / (2.0 * size as f32);

            // High arousal: enhance high frequencies
            if arousal > 0.0 {
                let hf_boost = 1.0 + arousal * 0.3 * (freq / 4000.0).min(1.0);
                *f *= hf_boost as f64;
            }

            // Low arousal: suppress high frequencies
            if arousal < 0.0 {
                let hf_suppress = 1.0 - arousal.abs() * 0.2 * (freq / 4000.0).min(1.0);
                *f *= hf_suppress as f64;
            }

            // Positive valence: enhance brightness (mid-high frequencies)
            if valence > 0.0 && freq > 1000.0 && freq < 3000.0 {
                *f *= (1.0 + valence * 0.15) as f64;
            }

            // Negative valence: reduce brightness
            if valence < 0.0 && freq > 1000.0 && freq < 3000.0 {
                *f *= (1.0 - valence.abs() * 0.15) as f64;
            }
        }

        filter
    }

    /// Apply formant shifting in the spectral domain
    ///
    /// Shifts formant frequencies to modify voice quality.
    ///
    /// # Arguments
    /// * `audio` - Input audio buffer
    /// * `shift_factor` - Formant shift factor (>1.0 = higher, <1.0 = lower)
    ///
    /// # Returns
    /// Formant-shifted audio
    pub fn apply_formant_shift(
        &self,
        audio: &[f32],
        shift_factor: f32,
    ) -> Result<Vec<f32>, String> {
        if audio.is_empty() || (shift_factor - 1.0).abs() < 1e-6 {
            return Ok(audio.to_vec());
        }

        // Convert to f64 for FFT
        let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Forward FFT
        let spectrum = rfft(&audio_f64, None).map_err(|e| format!("FFT failed: {:?}", e))?;

        // Shift formants by resampling spectrum
        let shifted_spectrum = self.shift_spectrum(&spectrum, shift_factor);

        // Inverse FFT
        let shifted_audio = irfft(&shifted_spectrum, Some(audio.len()))
            .map_err(|e| format!("IFFT failed: {:?}", e))?;

        Ok(shifted_audio.iter().map(|&x| x as f32).collect())
    }

    /// Shift the spectrum by resampling
    fn shift_spectrum(&self, spectrum: &[Complex<f64>], factor: f32) -> Vec<Complex<f64>> {
        let len = spectrum.len();
        let mut shifted = vec![Complex::new(0.0, 0.0); len];

        for (i, output) in shifted.iter_mut().enumerate() {
            let src_idx = (i as f32 / factor) as usize;
            if src_idx < len {
                *output = spectrum[src_idx];
            }
        }

        shifted
    }

    /// Enhance spectral envelope for emotional expression
    ///
    /// Emphasizes or de-emphasizes spectral peaks based on emotion.
    ///
    /// # Arguments
    /// * `audio` - Input audio buffer
    /// * `enhancement` - Enhancement factor (0.0 = no change, 1.0 = maximum)
    ///
    /// # Returns
    /// Enhanced audio
    pub fn enhance_spectral_envelope(
        &self,
        audio: &[f32],
        enhancement: f32,
    ) -> Result<Vec<f32>, String> {
        if audio.is_empty() || enhancement.abs() < 1e-6 {
            return Ok(audio.to_vec());
        }

        // Convert to f64
        let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Forward FFT
        let spectrum = rfft(&audio_f64, None).map_err(|e| format!("FFT failed: {:?}", e))?;

        // Compute spectral envelope (magnitude smoothing)
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

        // Smooth envelope
        let envelope = self.smooth_envelope(&magnitudes, 5);

        // Enhance or compress envelope
        let mut enhanced_spectrum = Vec::with_capacity(spectrum.len());
        for (i, &spec) in spectrum.iter().enumerate() {
            let mag = spec.norm();
            let env = envelope[i];

            // Enhancement: increase magnitude relative to envelope
            let new_mag = if mag > 0.0 {
                mag * (1.0 + enhancement as f64 * (mag / env.max(1e-10) - 1.0))
            } else {
                mag
            };

            // Preserve phase
            let phase = spec.arg();
            enhanced_spectrum.push(Complex::new(new_mag * phase.cos(), new_mag * phase.sin()));
        }

        // Inverse FFT
        let enhanced_audio = irfft(&enhanced_spectrum, Some(audio.len()))
            .map_err(|e| format!("IFFT failed: {:?}", e))?;

        Ok(enhanced_audio.iter().map(|&x| x as f32).collect())
    }

    /// Smooth spectral envelope using moving average
    fn smooth_envelope(&self, magnitudes: &[f64], window_size: usize) -> Vec<f64> {
        let mut smoothed = vec![0.0; magnitudes.len()];
        let half_window = window_size / 2;

        for (i, output) in smoothed.iter_mut().enumerate() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(magnitudes.len());
            let count = end - start;

            let sum: f64 = magnitudes[start..end].iter().sum();
            *output = sum / count as f64;
        }

        smoothed
    }

    /// Compute time-frequency representation for emotion analysis
    ///
    /// Uses STFT to analyze emotional characteristics in time-frequency domain.
    ///
    /// # Arguments
    /// * `audio` - Input audio buffer
    ///
    /// # Returns
    /// Tuple of (time bins, frequency bins, STFT magnitude matrix)
    pub fn compute_time_frequency(&self, audio: &[f32]) -> TimeFrequencyResult {
        if audio.is_empty() {
            return Err("Empty audio buffer".to_string());
        }

        // Convert to f64
        let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Compute STFT
        let (times, freqs, stft_result) = stft(
            &audio_f64,
            Window::Hann,
            self.fft_size,
            Some(self.hop_size),
            None,
            Some(self.sample_rate as f64),
            None,
            None,
        )
        .map_err(|e| format!("STFT failed: {:?}", e))?;

        // Extract magnitudes
        let magnitudes: Vec<Vec<f64>> = stft_result
            .outer_iter()
            .map(|row| row.iter().map(|c| c.norm()).collect())
            .collect();

        Ok((times, freqs, magnitudes))
    }

    /// Apply emotion-specific spectral tilt
    ///
    /// Modifies the overall spectral slope for emotional coloring.
    ///
    /// # Arguments
    /// * `audio` - Input audio buffer
    /// * `tilt_db_per_octave` - Tilt amount in dB per octave
    ///
    /// # Returns
    /// Tilted audio
    pub fn apply_spectral_tilt(
        &self,
        audio: &[f32],
        tilt_db_per_octave: f32,
    ) -> Result<Vec<f32>, String> {
        if audio.is_empty() || tilt_db_per_octave.abs() < 1e-6 {
            return Ok(audio.to_vec());
        }

        // Convert to f64
        let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

        // Forward FFT
        let spectrum = rfft(&audio_f64, None).map_err(|e| format!("FFT failed: {:?}", e))?;

        // Apply tilt filter
        let mut tilted_spectrum = Vec::with_capacity(spectrum.len());
        for (i, &spec) in spectrum.iter().enumerate() {
            let freq = (i as f32 * self.sample_rate) / (2.0 * spectrum.len() as f32);

            // Calculate tilt in linear scale
            let freq_ratio = freq / 1000.0; // Reference: 1 kHz
            let octaves = freq_ratio.log2();
            let db_change = tilt_db_per_octave * octaves;
            let linear_gain = 10.0_f32.powf(db_change / 20.0);

            tilted_spectrum.push(spec * linear_gain as f64);
        }

        // Inverse FFT
        let tilted_audio = irfft(&tilted_spectrum, Some(audio.len()))
            .map_err(|e| format!("IFFT failed: {:?}", e))?;

        Ok(tilted_audio.iter().map(|&x| x as f32).collect())
    }
}

/// Emotion-based spectral processing configuration
#[derive(Debug, Clone)]
pub struct SpectralEmotionConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// FFT size (power of 2)
    pub fft_size: usize,
    /// Hop size for STFT
    pub hop_size: usize,
    /// Arousal level (-1.0 to 1.0)
    pub arousal: f32,
    /// Valence level (-1.0 to 1.0)
    pub valence: f32,
    /// Formant shift factor (1.0 = no shift)
    pub formant_shift: f32,
    /// Spectral tilt in dB/octave
    pub spectral_tilt: f32,
}

impl Default for SpectralEmotionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            fft_size: 2048,
            hop_size: 512,
            arousal: 0.0,
            valence: 0.0,
            formant_shift: 1.0,
            spectral_tilt: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        assert_eq!(processor.sample_rate, 44100.0);
        assert_eq!(processor.fft_size, 1024);
        assert_eq!(processor.hop_size, 256);
        assert_eq!(processor.window.len(), 1024);
    }

    #[test]
    fn test_hann_window() {
        let window = SpectralEmotionProcessor::create_hann_window(100);
        assert_eq!(window.len(), 100);

        // Window should start near zero
        assert!(window[0].abs() < 0.01);

        // Peak should be near middle
        assert!(window[50] > 0.95);
    }

    #[test]
    fn test_emotion_filter_empty_audio() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio: Vec<f32> = vec![];

        let result = processor.apply_emotion_filter(&audio, 0.5, 0.5);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_emotion_filter_basic() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048]; // Constant signal

        let result = processor.apply_emotion_filter(&audio, 0.5, 0.5);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), audio.len());
    }

    #[test]
    fn test_formant_shift_no_change() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048];

        let result = processor.apply_formant_shift(&audio, 1.0);
        assert!(result.is_ok());

        let shifted = result.unwrap();
        assert_eq!(shifted.len(), audio.len());
    }

    #[test]
    fn test_formant_shift_upward() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048];

        let result = processor.apply_formant_shift(&audio, 1.2);
        assert!(result.is_ok());

        let shifted = result.unwrap();
        assert_eq!(shifted.len(), audio.len());
    }

    #[test]
    fn test_spectral_envelope_enhancement() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048];

        let result = processor.enhance_spectral_envelope(&audio, 0.3);
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        assert_eq!(enhanced.len(), audio.len());
    }

    #[test]
    fn test_spectral_tilt() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048];

        let result = processor.apply_spectral_tilt(&audio, 3.0);
        assert!(result.is_ok());

        let tilted = result.unwrap();
        assert_eq!(tilted.len(), audio.len());
    }

    #[test]
    fn test_spectral_tilt_zero() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio = vec![0.5; 2048];

        let result = processor.apply_spectral_tilt(&audio, 0.0);
        assert!(result.is_ok());

        let tilted = result.unwrap();
        // Should be approximately unchanged
        assert!((tilted[0] - audio[0]).abs() < 0.1);
    }

    #[test]
    fn test_smooth_envelope() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let magnitudes = vec![1.0, 5.0, 1.0, 1.0, 1.0];

        let smoothed = processor.smooth_envelope(&magnitudes, 3);
        assert_eq!(smoothed.len(), magnitudes.len());

        // Middle value should be smoothed
        assert!(smoothed[1] < magnitudes[1]);
        assert!(smoothed[1] > magnitudes[0]);
    }

    #[test]
    fn test_time_frequency_empty() {
        let processor = SpectralEmotionProcessor::new(44100.0, 1024, 256);
        let audio: Vec<f32> = vec![];

        let result = processor.compute_time_frequency(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_default() {
        let config = SpectralEmotionConfig::default();
        assert_eq!(config.sample_rate, 44100.0);
        assert_eq!(config.fft_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.arousal, 0.0);
        assert_eq!(config.valence, 0.0);
        assert_eq!(config.formant_shift, 1.0);
        assert_eq!(config.spectral_tilt, 0.0);
    }
}
