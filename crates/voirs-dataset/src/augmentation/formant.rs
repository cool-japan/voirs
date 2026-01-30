//! Formant-preserving pitch shifting augmentation
//!
//! This module provides advanced pitch shifting that preserves vocal tract characteristics
//! (formants) while changing the fundamental frequency. This is particularly important for
//! speech data augmentation as it maintains naturalness and speaker identity while varying pitch.
//!
//! The implementation uses a combination of:
//! - Phase vocoder for pitch shifting
//! - Time-stretching to compensate for formant shifts
//! - LPC (Linear Predictive Coding) envelope preservation

use crate::{AudioData, DatasetError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::{Complex, Complex64};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Formant-preserving pitch shift configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantConfig {
    /// Pitch shift values in semitones
    pub pitch_shifts: Vec<f32>,
    /// Formant preservation strength (0.0 = no preservation, 1.0 = full preservation)
    pub formant_preservation: f32,
    /// FFT window size
    pub window_size: usize,
    /// Hop size ratio
    pub hop_ratio: f32,
    /// LPC order for formant extraction
    pub lpc_order: usize,
    /// Use high-quality processing (slower but better)
    pub high_quality: bool,
}

impl Default for FormantConfig {
    fn default() -> Self {
        Self {
            pitch_shifts: vec![-2.0, -1.0, 1.0, 2.0],
            formant_preservation: 0.9,
            window_size: 2048,
            hop_ratio: 0.25,
            lpc_order: 16,
            high_quality: true,
        }
    }
}

impl FormantConfig {
    /// Create configuration for speech (high formant preservation)
    pub fn for_speech() -> Self {
        Self {
            pitch_shifts: vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            formant_preservation: 0.95,
            window_size: 1024,
            hop_ratio: 0.25,
            lpc_order: 16,
            high_quality: true,
        }
    }

    /// Create configuration for singing (moderate formant preservation)
    pub fn for_singing() -> Self {
        Self {
            pitch_shifts: vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            formant_preservation: 0.7,
            window_size: 2048,
            hop_ratio: 0.125,
            lpc_order: 20,
            high_quality: true,
        }
    }

    /// Create configuration for aggressive augmentation
    pub fn aggressive() -> Self {
        Self {
            pitch_shifts: vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
            formant_preservation: 0.85,
            window_size: 2048,
            hop_ratio: 0.25,
            lpc_order: 16,
            high_quality: false,
        }
    }
}

/// Formant-preserving pitch shift augmentor
pub struct FormantAugmentor {
    config: FormantConfig,
}

impl FormantAugmentor {
    /// Create new formant augmentor
    pub fn new(config: FormantConfig) -> Self {
        Self { config }
    }

    /// Apply formant-preserving pitch shift
    pub fn shift_pitch(&mut self, audio: &AudioData, semitones: f32) -> Result<AudioData> {
        if semitones == 0.0 {
            return Ok(audio.clone());
        }

        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(audio.clone());
        }

        // Calculate pitch shift factor
        let pitch_factor = 2.0f32.powf(semitones / 12.0);

        // Process each channel
        let num_channels = audio.channels() as usize;
        let mut shifted_channels = Vec::new();

        for ch in 0..num_channels {
            // Extract channel data
            let channel_data: Vec<f32> = if num_channels == 1 {
                samples.to_vec()
            } else {
                samples
                    .iter()
                    .skip(ch)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            };

            // Apply formant-preserving pitch shift
            let shifted = self.formant_preserving_shift(&channel_data, pitch_factor)?;
            shifted_channels.push(shifted);
        }

        // Interleave channels if multi-channel
        let output_samples = if num_channels == 1 {
            shifted_channels[0].clone()
        } else {
            let output_len = shifted_channels
                .iter()
                .map(|ch| ch.len())
                .min()
                .unwrap_or(0)
                * num_channels;
            let mut interleaved = vec![0.0; output_len];

            let samples_per_channel = output_len / num_channels;
            for i in 0..samples_per_channel {
                for (ch, channel_data) in shifted_channels.iter().enumerate() {
                    if i < channel_data.len() {
                        interleaved[i * num_channels + ch] = channel_data[i];
                    }
                }
            }
            interleaved
        };

        Ok(AudioData::new(
            output_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Formant-preserving pitch shift implementation
    fn formant_preserving_shift(&mut self, input: &[f32], pitch_factor: f32) -> Result<Vec<f32>> {
        // Step 1: Extract formant envelope using LPC
        let formant_envelope = if self.config.formant_preservation > 0.0 {
            Some(self.extract_formant_envelope(input)?)
        } else {
            None
        };

        // Step 2: Apply pitch shift using phase vocoder
        let pitch_shifted = self.pitch_shift_phase_vocoder(input, pitch_factor)?;

        // Step 3: Apply formant envelope correction
        if let Some(envelope) = formant_envelope {
            Ok(self.apply_formant_correction(&pitch_shifted, &envelope, pitch_factor)?)
        } else {
            Ok(pitch_shifted)
        }
    }

    /// Extract formant envelope using Linear Predictive Coding
    fn extract_formant_envelope(&self, signal: &[f32]) -> Result<Vec<f32>> {
        let window_size = self.config.window_size;
        let hop_size = (window_size as f32 * self.config.hop_ratio) as usize;
        let num_frames = (signal.len() - window_size) / hop_size + 1;

        let mut envelope = vec![0.0; signal.len()];

        // Process each frame
        for frame_idx in 0..num_frames {
            let pos = frame_idx * hop_size;

            // Extract windowed frame
            let mut frame = vec![0.0; window_size];
            for i in 0..window_size {
                if pos + i < signal.len() {
                    // Apply Hann window
                    let window_val =
                        0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size - 1) as f32).cos());
                    frame[i] = signal[pos + i] * window_val;
                }
            }

            // Calculate LPC coefficients
            let lpc_coeffs = self.calculate_lpc_coefficients(&frame, self.config.lpc_order);

            // Calculate spectral envelope from LPC
            let frame_envelope = self.lpc_to_spectral_envelope(&lpc_coeffs, window_size);

            // Overlap-add envelope
            for i in 0..window_size {
                if pos + i < envelope.len() {
                    envelope[pos + i] += frame_envelope[i];
                }
            }
        }

        // Normalize envelope
        let max_env = envelope.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        if max_env > 1e-8 {
            for val in &mut envelope {
                *val /= max_env;
            }
        }

        Ok(envelope)
    }

    /// Calculate LPC coefficients using Levinson-Durbin algorithm
    fn calculate_lpc_coefficients(&self, signal: &[f32], order: usize) -> Vec<f32> {
        // Calculate autocorrelation
        let mut autocorr = vec![0.0; order + 1];
        for lag in 0..=order {
            let mut sum = 0.0;
            for i in 0..(signal.len() - lag) {
                sum += signal[i] * signal[i + lag];
            }
            autocorr[lag] = sum;
        }

        // Levinson-Durbin recursion
        let mut lpc = vec![0.0; order + 1];
        lpc[0] = 1.0;

        if autocorr[0].abs() < 1e-8 {
            return lpc;
        }

        let mut error = autocorr[0];

        for i in 1..=order {
            let mut lambda = 0.0;
            for j in 1..i {
                lambda -= lpc[j] * autocorr[i - j];
            }
            lambda -= autocorr[i];
            lambda /= error;

            // Update coefficients
            let mut new_lpc = lpc.clone();
            new_lpc[i] = lambda;
            for j in 1..i {
                new_lpc[j] = lpc[j] + lambda * lpc[i - j];
            }
            lpc = new_lpc;

            error *= 1.0 - lambda * lambda;
            if error.abs() < 1e-8 {
                break;
            }
        }

        lpc
    }

    /// Convert LPC coefficients to spectral envelope
    #[allow(clippy::needless_range_loop)]
    fn lpc_to_spectral_envelope(&self, lpc: &[f32], size: usize) -> Vec<f32> {
        let mut envelope = vec![1.0; size];

        for i in 0..size {
            let freq = i as f32 / size as f32;
            let mut real = lpc[0];
            let mut imag = 0.0;

            for (k, &coeff) in lpc.iter().enumerate().skip(1) {
                let phase = -2.0 * PI * freq * k as f32;
                real += coeff * phase.cos();
                imag += coeff * phase.sin();
            }

            let magnitude = (real * real + imag * imag).sqrt();
            envelope[i] = if magnitude > 1e-8 {
                1.0 / magnitude
            } else {
                1.0
            };
        }

        envelope
    }

    /// Pitch shift using phase vocoder
    fn pitch_shift_phase_vocoder(&mut self, input: &[f32], pitch_factor: f32) -> Result<Vec<f32>> {
        // Time-stretch by 1/pitch_factor, then resample by pitch_factor
        // This preserves formants while changing pitch
        let time_stretch_factor = 1.0 / pitch_factor;

        let window_size = self.config.window_size;
        let hop_size = (window_size as f32 * self.config.hop_ratio) as usize;
        let synthesis_hop = (hop_size as f32 * time_stretch_factor) as usize;

        // Create Hann window
        let window: Vec<f32> = (0..window_size)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / (window_size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let num_frames = (input.len() - window_size) / hop_size + 1;
        let output_length = (num_frames - 1) * synthesis_hop + window_size;

        let mut output = vec![0.0; output_length];
        let mut norm_window = vec![0.0; output_length];

        let mut prev_phase = vec![0.0; window_size / 2 + 1];
        let mut phase_cumulative = vec![0.0; window_size / 2 + 1];

        for frame_idx in 0..num_frames {
            let input_pos = frame_idx * hop_size;
            let output_pos = frame_idx * synthesis_hop;

            // Extract and window frame
            let windowed_frame: Vec<f32> = (0..window_size)
                .map(|i| {
                    if input_pos + i < input.len() {
                        input[input_pos + i] * window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // FFT using scirs2-fft
            let spectrum64 = scirs2_fft::fft(&windowed_frame, Some(window_size))
                .map_err(|e| DatasetError::AudioError(format!("FFT error: {}", e)))?;

            // Convert to f32 Complex for processing
            let spectrum: Vec<Complex<f32>> = spectrum64
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();

            // Extract magnitude and phase
            if frame_idx > 0 {
                for k in 0..=window_size / 2 {
                    let phase = spectrum[k].arg();
                    let phase_diff = phase - prev_phase[k];
                    phase_cumulative[k] += phase_diff;
                    prev_phase[k] = phase;
                }
            } else {
                for k in 0..=window_size / 2 {
                    prev_phase[k] = spectrum[k].arg();
                    phase_cumulative[k] = prev_phase[k];
                }
            }

            // Reconstruct with modified phases
            let mut modified_spectrum = vec![Complex::zero(); window_size];
            for k in 0..=window_size / 2 {
                let mag = spectrum[k].norm();
                let phase = phase_cumulative[k];
                modified_spectrum[k] = Complex::new(mag * phase.cos(), mag * phase.sin());
            }

            // Conjugate symmetry
            for k in 1..window_size / 2 {
                modified_spectrum[window_size - k] = modified_spectrum[k].conj();
            }

            // Inverse FFT using scirs2-fft
            // Convert back to f64 for scirs2-fft
            let modified_spectrum64: Vec<Complex64> = modified_spectrum
                .iter()
                .map(|c| Complex64::new(c.re as f64, c.im as f64))
                .collect();

            let time_domain64 = scirs2_fft::ifft(&modified_spectrum64, Some(window_size))
                .map_err(|e| DatasetError::AudioError(format!("IFFT error: {}", e)))?;

            // Overlap-add
            for i in 0..window_size {
                if output_pos + i < output.len() {
                    let sample = time_domain64[i].re as f32;
                    output[output_pos + i] += sample * window[i];
                    norm_window[output_pos + i] += window[i] * window[i];
                }
            }
        }

        // Normalize
        for i in 0..output.len() {
            if norm_window[i] > 1e-8 {
                output[i] /= norm_window[i];
            }
        }

        // Resample to achieve pitch shift
        Ok(self.resample(&output, pitch_factor))
    }

    /// Simple linear interpolation resampling
    #[allow(clippy::needless_range_loop)]
    fn resample(&self, input: &[f32], factor: f32) -> Vec<f32> {
        let output_len = (input.len() as f32 / factor) as usize;
        let mut output = vec![0.0; output_len];

        for i in 0..output_len {
            let src_pos = i as f32 * factor;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f32;

            if src_idx + 1 < input.len() {
                output[i] = input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac;
            } else if src_idx < input.len() {
                output[i] = input[src_idx];
            }
        }

        output
    }

    /// Apply formant correction
    #[allow(clippy::needless_range_loop)]
    fn apply_formant_correction(
        &self,
        shifted: &[f32],
        original_envelope: &[f32],
        pitch_factor: f32,
    ) -> Result<Vec<f32>> {
        let strength = self.config.formant_preservation;
        if strength < 0.01 {
            return Ok(shifted.to_vec());
        }

        // Resample original envelope to match shifted signal length
        let target_len = shifted.len();
        let mut resampled_envelope = vec![0.0; target_len];

        for i in 0..target_len {
            let src_pos = i as f32 * original_envelope.len() as f32 / target_len as f32;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f32;

            if src_idx + 1 < original_envelope.len() {
                resampled_envelope[i] = original_envelope[src_idx] * (1.0 - frac)
                    + original_envelope[src_idx + 1] * frac;
            } else if src_idx < original_envelope.len() {
                resampled_envelope[i] = original_envelope[src_idx];
            }
        }

        // Apply envelope with strength parameter
        let mut corrected = shifted.to_vec();
        for i in 0..corrected.len() {
            let envelope_factor = 1.0 + (resampled_envelope[i] - 1.0) * strength;
            corrected[i] *= envelope_factor;
        }

        Ok(corrected)
    }

    /// Process multiple pitch shifts
    pub fn shift_multiple(&mut self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut results = Vec::new();
        let pitch_shifts = self.config.pitch_shifts.clone();

        for semitones in pitch_shifts {
            let shifted = self.shift_pitch(audio, semitones)?;
            results.push(shifted);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(sample_rate: u32, duration: f32) -> AudioData {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let fundamental = 200.0; // Low voice fundamental

        // Create harmonic signal (simplified voice)
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let mut sample = 0.0;
                // Add harmonics
                for harmonic in 1..=10 {
                    let freq = fundamental * harmonic as f32;
                    let amp = 1.0 / harmonic as f32;
                    sample += (2.0 * PI * freq * t).sin() * amp * 0.1;
                }
                sample
            })
            .collect();

        AudioData::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_config_default() {
        let config = FormantConfig::default();
        assert_eq!(config.pitch_shifts, vec![-2.0, -1.0, 1.0, 2.0]);
        assert_eq!(config.formant_preservation, 0.9);
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.lpc_order, 16);
        assert!(config.high_quality);
    }

    #[test]
    fn test_config_for_speech() {
        let config = FormantConfig::for_speech();
        assert_eq!(config.formant_preservation, 0.95);
        assert_eq!(config.lpc_order, 16);
    }

    #[test]
    fn test_config_for_singing() {
        let config = FormantConfig::for_singing();
        assert_eq!(config.formant_preservation, 0.7);
        assert_eq!(config.lpc_order, 20);
    }

    #[test]
    fn test_shift_zero_semitones() {
        let config = FormantConfig::default();
        let mut augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let shifted = augmentor.shift_pitch(&audio, 0.0).unwrap();

        // Should return identical audio
        assert_eq!(shifted.samples().len(), audio.samples().len());
        assert_eq!(shifted.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_shift_up() {
        let config = FormantConfig::default();
        let mut augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let shifted = augmentor.shift_pitch(&audio, 2.0).unwrap();

        // Should produce output
        assert!(!shifted.samples().is_empty());
        assert_eq!(shifted.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_shift_down() {
        let config = FormantConfig::default();
        let mut augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let shifted = augmentor.shift_pitch(&audio, -2.0).unwrap();

        // Should produce output
        assert!(!shifted.samples().is_empty());
        assert_eq!(shifted.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_shift_empty_audio() {
        let config = FormantConfig::default();
        let mut augmentor = FormantAugmentor::new(config);
        let audio = AudioData::new(vec![], 16000, 1);

        let shifted = augmentor.shift_pitch(&audio, 1.0).unwrap();

        assert!(shifted.samples().is_empty());
    }

    #[test]
    fn test_shift_multiple() {
        let config = FormantConfig {
            pitch_shifts: vec![-2.0, 0.0, 2.0],
            ..Default::default()
        };
        let mut augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let results = augmentor.shift_multiple(&audio).unwrap();

        assert_eq!(results.len(), 3);
        // All should have valid output
        for result in &results {
            assert!(!result.samples().is_empty());
        }
    }

    #[test]
    fn test_lpc_coefficients() {
        let config = FormantConfig::default();
        let augmentor = FormantAugmentor::new(config);

        // Create simple test signal
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        let lpc = augmentor.calculate_lpc_coefficients(&signal, 10);

        assert_eq!(lpc.len(), 11);
        assert_eq!(lpc[0], 1.0); // First coefficient should be 1.0
    }

    #[test]
    fn test_formant_envelope_extraction() {
        let config = FormantConfig::default();
        let augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let envelope = augmentor.extract_formant_envelope(audio.samples()).unwrap();

        assert_eq!(envelope.len(), audio.samples().len());
        // Envelope should be normalized
        let max_val = envelope.iter().copied().fold(0.0f32, f32::max);
        assert!(max_val <= 1.0 + 1e-6);
    }

    #[test]
    fn test_resampling() {
        let config = FormantConfig::default();
        let augmentor = FormantAugmentor::new(config);

        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Upsample by 2x
        let upsampled = augmentor.resample(&input, 0.5);
        assert_eq!(upsampled.len(), 200);

        // Downsample by 2x
        let downsampled = augmentor.resample(&input, 2.0);
        assert_eq!(downsampled.len(), 50);
    }

    #[test]
    fn test_no_formant_preservation() {
        let config = FormantConfig {
            formant_preservation: 0.0,
            ..Default::default()
        };
        let mut augmentor = FormantAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        // Should still work but skip formant correction
        let shifted = augmentor.shift_pitch(&audio, 2.0).unwrap();
        assert!(!shifted.samples().is_empty());
    }

    #[test]
    fn test_multi_channel_shift() {
        let config = FormantConfig::default();
        let mut augmentor = FormantAugmentor::new(config);

        // Create stereo test audio
        let num_samples = 8000;
        let mut stereo_samples = Vec::new();
        for i in 0..num_samples {
            let t = i as f32 / 16000.0;
            let sample = (2.0 * PI * 200.0 * t).sin() * 0.5;
            stereo_samples.push(sample);
            stereo_samples.push(sample);
        }

        let stereo_audio = AudioData::new(stereo_samples, 16000, 2);

        let shifted = augmentor.shift_pitch(&stereo_audio, 2.0).unwrap();

        // Should maintain stereo format
        assert_eq!(shifted.channels(), 2);
        assert!(!shifted.samples().is_empty());
    }
}
