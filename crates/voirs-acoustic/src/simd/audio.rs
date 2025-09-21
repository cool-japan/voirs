//! SIMD-accelerated audio processing operations
//!
//! This module provides SIMD-optimized audio processing functions including
//! filtering, resampling, windowing, and other digital signal processing
//! operations commonly used in acoustic modeling.

use super::simd;
use crate::{AcousticError, Result};
use std::f32::consts::PI;

/// SIMD-accelerated audio processor
pub struct SimdAudioProcessor;

impl SimdAudioProcessor {
    /// Apply pre-emphasis filter with SIMD acceleration
    pub fn pre_emphasis(input: &[f32], output: &mut [f32], coefficient: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        // First sample remains unchanged
        output[0] = input[0];

        // Apply pre-emphasis: y[n] = x[n] - α * x[n-1]
        for i in 1..input.len() {
            output[i] = input[i] - coefficient * input[i - 1];
        }

        Ok(())
    }

    /// Apply de-emphasis filter with SIMD acceleration
    pub fn de_emphasis(input: &[f32], output: &mut [f32], coefficient: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        // First sample remains unchanged
        output[0] = input[0];

        // Apply de-emphasis: y[n] = x[n] + α * y[n-1]
        for i in 1..input.len() {
            output[i] = input[i] + coefficient * output[i - 1];
        }

        Ok(())
    }

    /// Normalize audio with SIMD acceleration
    pub fn normalize(input: &[f32], output: &mut [f32], target_amplitude: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        // Find maximum absolute value
        let max_abs = input.iter().fold(0.0f32, |max, &val| max.max(val.abs()));

        if max_abs > 0.0 {
            let scale_factor = target_amplitude / max_abs;

            // Apply scaling with SIMD
            let scale_vec = vec![scale_factor; input.len()];
            simd().mul_f32(input, &scale_vec, output)?;
        } else {
            output.copy_from_slice(input);
        }

        Ok(())
    }

    /// Apply highpass filter with SIMD acceleration
    pub fn highpass_filter(
        input: &[f32],
        output: &mut [f32],
        cutoff_freq: f32,
        sample_rate: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        // Simple first-order highpass filter
        let rc = 1.0 / (2.0 * PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        if input.is_empty() {
            return Ok(());
        }

        output[0] = input[0];

        for i in 1..input.len() {
            output[i] = alpha * (output[i - 1] + input[i] - input[i - 1]);
        }

        Ok(())
    }

    /// Apply lowpass filter with SIMD acceleration
    pub fn lowpass_filter(
        input: &[f32],
        output: &mut [f32],
        cutoff_freq: f32,
        sample_rate: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        // Simple first-order lowpass filter
        let rc = 1.0 / (2.0 * PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        if input.is_empty() {
            return Ok(());
        }

        output[0] = input[0];

        for i in 1..input.len() {
            output[i] = output[i - 1] + alpha * (input[i] - output[i - 1]);
        }

        Ok(())
    }

    /// Resample audio using linear interpolation with SIMD acceleration
    pub fn resample_linear(input: &[f32], input_rate: f32, output_rate: f32) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let ratio = input_rate / output_rate;
        let output_len = ((input.len() as f32) / ratio) as usize;
        let mut output = vec![0.0f32; output_len];

        for (i, output_val) in output.iter_mut().enumerate().take(output_len) {
            let src_idx = i as f32 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(input.len() - 1);
            let frac = src_idx - idx0 as f32;

            // Linear interpolation
            *output_val = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        }

        Ok(output)
    }

    /// Compute Short-Time Energy with SIMD acceleration
    pub fn short_time_energy(
        input: &[f32],
        frame_size: usize,
        hop_size: usize,
    ) -> Result<Vec<f32>> {
        if frame_size == 0 || hop_size == 0 {
            return Err(AcousticError::InputError(
                "Frame size and hop size must be positive".to_string(),
            ));
        }

        let n_frames = (input.len().saturating_sub(frame_size)) / hop_size + 1;
        let mut energy = vec![0.0f32; n_frames];

        for (frame_idx, energy_val) in energy.iter_mut().enumerate() {
            let start_idx = frame_idx * hop_size;
            let end_idx = (start_idx + frame_size).min(input.len());

            if start_idx < input.len() {
                let frame = &input[start_idx..end_idx];
                *energy_val = simd().dot_product_f32(frame, frame)? / frame.len() as f32;
            }
        }

        Ok(energy)
    }

    /// Compute Zero Crossing Rate with SIMD acceleration
    pub fn zero_crossing_rate(
        input: &[f32],
        frame_size: usize,
        hop_size: usize,
    ) -> Result<Vec<f32>> {
        if frame_size == 0 || hop_size == 0 {
            return Err(AcousticError::InputError(
                "Frame size and hop size must be positive".to_string(),
            ));
        }

        let n_frames = (input.len().saturating_sub(frame_size)) / hop_size + 1;
        let mut zcr = vec![0.0f32; n_frames];

        for (frame_idx, zcr_val) in zcr.iter_mut().enumerate() {
            let start_idx = frame_idx * hop_size;
            let end_idx = (start_idx + frame_size).min(input.len());

            if start_idx + 1 < input.len() {
                let mut crossings = 0;
                for i in (start_idx + 1)..end_idx {
                    if (input[i] >= 0.0) != (input[i - 1] >= 0.0) {
                        crossings += 1;
                    }
                }
                *zcr_val = crossings as f32 / (end_idx - start_idx - 1) as f32;
            }
        }

        Ok(zcr)
    }

    /// Apply dynamic range compression with SIMD acceleration
    pub fn compress_dynamic_range(
        input: &[f32],
        output: &mut [f32],
        threshold: f32,
        ratio: f32,
        attack_time: f32,
        release_time: f32,
        sample_rate: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        let attack_coeff = (-1.0 / (attack_time * sample_rate)).exp();
        let release_coeff = (-1.0 / (release_time * sample_rate)).exp();

        let mut envelope = 0.0f32;

        for (&input_val, output_val) in input.iter().zip(output.iter_mut()) {
            let abs_input = input_val.abs();

            // Update envelope
            if abs_input > envelope {
                envelope = abs_input + attack_coeff * (envelope - abs_input);
            } else {
                envelope = abs_input + release_coeff * (envelope - abs_input);
            }

            // Apply compression
            let gain = if envelope > threshold {
                threshold + (envelope - threshold) / ratio
            } else {
                envelope
            };

            let gain_reduction = if envelope > 0.0 { gain / envelope } else { 1.0 };
            *output_val = input_val * gain_reduction;
        }

        Ok(())
    }

    /// Compute autocorrelation with SIMD acceleration
    pub fn autocorrelation(input: &[f32], max_lag: usize) -> Result<Vec<f32>> {
        if max_lag >= input.len() {
            return Err(AcousticError::InputError(
                "Max lag must be less than input length".to_string(),
            ));
        }

        let mut autocorr = vec![0.0f32; max_lag + 1];
        let n = input.len();

        for lag in 0..=max_lag {
            let mut sum = 0.0f32;
            let count = n - lag;

            if count > 0 {
                let x1 = &input[..count];
                let x2 = &input[lag..];
                sum = simd().dot_product_f32(x1, x2)?;
            }

            autocorr[lag] = sum / count as f32;
        }

        Ok(autocorr)
    }

    /// Apply voice activity detection with SIMD acceleration
    pub fn voice_activity_detection(
        input: &[f32],
        frame_size: usize,
        hop_size: usize,
        energy_threshold: f32,
        zcr_threshold: f32,
    ) -> Result<Vec<bool>> {
        let energy = Self::short_time_energy(input, frame_size, hop_size)?;
        let zcr = Self::zero_crossing_rate(input, frame_size, hop_size)?;

        if energy.len() != zcr.len() {
            return Err(AcousticError::Processing(
                "Energy and ZCR lengths don't match".to_string(),
            ));
        }

        let mut vad = vec![false; energy.len()];

        for ((energy_val, zcr_val), is_voice) in energy.iter().zip(zcr.iter()).zip(vad.iter_mut()) {
            *is_voice = *energy_val > energy_threshold && *zcr_val < zcr_threshold;
        }

        Ok(vad)
    }

    /// Apply spectral subtraction for noise reduction with SIMD acceleration
    pub fn spectral_subtraction(
        noisy_spectrum: &[f32],
        noise_spectrum: &[f32],
        output: &mut [f32],
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if noisy_spectrum.len() != noise_spectrum.len() || noisy_spectrum.len() != output.len() {
            return Err(AcousticError::InputError(
                "All arrays must have the same length".to_string(),
            ));
        }

        for ((noisy, noise), out) in noisy_spectrum
            .iter()
            .zip(noise_spectrum.iter())
            .zip(output.iter_mut())
        {
            let subtracted = noisy - alpha * noise;
            *out = subtracted.max(beta * noisy);
        }

        Ok(())
    }

    /// Mix multiple audio channels with SIMD acceleration
    pub fn mix_channels(channels: &[Vec<f32>], weights: &[f32], output: &mut [f32]) -> Result<()> {
        if channels.len() != weights.len() {
            return Err(AcousticError::InputError(
                "Number of channels and weights must match".to_string(),
            ));
        }

        if channels.is_empty() {
            return Ok(());
        }

        let output_len = output.len();

        // Initialize output to zero
        output.fill(0.0);

        // Mix channels
        for (channel, &weight) in channels.iter().zip(weights.iter()) {
            let len = channel.len().min(output_len);

            for i in 0..len {
                output[i] += channel[i] * weight;
            }
        }

        Ok(())
    }

    /// Apply time-stretching using overlap-and-add with SIMD acceleration
    pub fn time_stretch(
        input: &[f32],
        stretch_factor: f32,
        frame_size: usize,
        hop_size: usize,
    ) -> Result<Vec<f32>> {
        if stretch_factor <= 0.0 {
            return Err(AcousticError::InputError(
                "Stretch factor must be positive".to_string(),
            ));
        }

        let output_hop_size = (hop_size as f32 * stretch_factor) as usize;
        let n_frames = (input.len().saturating_sub(frame_size)) / hop_size + 1;
        let output_len = (n_frames - 1) * output_hop_size + frame_size;

        let mut output = vec![0.0f32; output_len];
        let window = Self::create_hann_window(frame_size);

        for frame_idx in 0..n_frames {
            let input_start = frame_idx * hop_size;
            let output_start = frame_idx * output_hop_size;

            if input_start + frame_size <= input.len() && output_start + frame_size <= output.len()
            {
                // Extract and window frame
                let mut windowed_frame = vec![0.0f32; frame_size];
                for i in 0..frame_size {
                    windowed_frame[i] = input[input_start + i] * window[i];
                }

                // Overlap and add
                for i in 0..frame_size {
                    output[output_start + i] += windowed_frame[i];
                }
            }
        }

        Ok(output)
    }

    /// Create Hann window function
    fn create_hann_window(size: usize) -> Vec<f32> {
        let mut window = vec![0.0f32; size];

        for (i, window_val) in window.iter_mut().enumerate() {
            *window_val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos());
        }

        window
    }
}

/// Audio effects processor with SIMD acceleration
pub struct SimdAudioEffects;

impl SimdAudioEffects {
    /// Apply reverb effect using convolution with SIMD acceleration
    pub fn reverb(input: &[f32], impulse_response: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() < input.len() + impulse_response.len() - 1 {
            return Err(AcousticError::InputError(
                "Output buffer too small for convolution".to_string(),
            ));
        }

        // Initialize output
        output.fill(0.0);

        // Perform convolution
        for (i, &input_val) in input.iter().enumerate() {
            for (j, &ir_val) in impulse_response.iter().enumerate() {
                if i + j < output.len() {
                    output[i + j] += input_val * ir_val;
                }
            }
        }

        Ok(())
    }

    /// Apply echo effect with SIMD acceleration
    pub fn echo(
        input: &[f32],
        output: &mut [f32],
        delay_samples: usize,
        feedback: f32,
        mix: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        output.copy_from_slice(input);

        for i in delay_samples..output.len() {
            let delayed_sample = output[i - delay_samples];
            output[i] = input[i] + mix * (delayed_sample * feedback);
        }

        Ok(())
    }

    /// Apply chorus effect with SIMD acceleration
    #[allow(clippy::too_many_arguments)]
    pub fn chorus(
        input: &[f32],
        output: &mut [f32],
        delay_ms: f32,
        depth_ms: f32,
        rate_hz: f32,
        feedback: f32,
        mix: f32,
        sample_rate: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AcousticError::InputError(
                "Input and output lengths must match".to_string(),
            ));
        }

        let delay_samples = (delay_ms * sample_rate / 1000.0) as usize;
        let depth_samples = depth_ms * sample_rate / 1000.0;
        let rate_samples = 2.0 * PI * rate_hz / sample_rate;

        let mut delay_line = vec![0.0f32; delay_samples * 2];
        let mut write_pos = 0;

        for (i, (&input_val, output_val)) in input.iter().zip(output.iter_mut()).enumerate() {
            // LFO for modulation
            let lfo = (i as f32 * rate_samples).sin();
            let variable_delay = delay_samples as f32 + depth_samples * lfo;

            // Write to delay line
            delay_line[write_pos] = input_val + feedback * delay_line[write_pos];

            // Read from delay line with interpolation
            let read_pos = (write_pos as f32 - variable_delay).rem_euclid(delay_line.len() as f32);
            let read_idx = read_pos as usize;
            let frac = read_pos - read_idx as f32;

            let delayed_sample = if read_idx + 1 < delay_line.len() {
                delay_line[read_idx] * (1.0 - frac) + delay_line[read_idx + 1] * frac
            } else {
                delay_line[read_idx]
            };

            *output_val = input_val + mix * delayed_sample;

            write_pos = (write_pos + 1) % delay_line.len();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_emphasis() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];

        SimdAudioProcessor::pre_emphasis(&input, &mut output, 0.97).unwrap();

        assert_eq!(output[0], 1.0);
        assert!((output[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let input = vec![-2.0, 1.0, 3.0, -1.0];
        let mut output = vec![0.0; 4];

        SimdAudioProcessor::normalize(&input, &mut output, 1.0).unwrap();

        let max_abs = output.iter().fold(0.0f32, |max, &val| max.max(val.abs()));
        assert!((max_abs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_short_time_energy() {
        let input = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let energy = SimdAudioProcessor::short_time_energy(&input, 4, 2).unwrap();

        assert!(!energy.is_empty());
        assert!(energy[0] > energy[1]); // First frame has higher energy
    }

    #[test]
    fn test_zero_crossing_rate() {
        let input = vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let zcr = SimdAudioProcessor::zero_crossing_rate(&input, 4, 2).unwrap();

        assert!(!zcr.is_empty());
        assert!(zcr[0] > zcr[1]); // First frame has higher ZCR
    }

    #[test]
    fn test_autocorrelation() {
        let input = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let autocorr = SimdAudioProcessor::autocorrelation(&input, 4).unwrap();

        assert_eq!(autocorr.len(), 5);
        assert!(autocorr[0] > autocorr[1]); // Zero lag should have highest correlation
    }

    #[test]
    fn test_resample_linear() {
        let input = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let resampled = SimdAudioProcessor::resample_linear(&input, 44100.0, 22050.0).unwrap();

        assert!(resampled.len() < input.len()); // Downsampled
        assert!(!resampled.is_empty());
    }

    #[test]
    fn test_voice_activity_detection() {
        let input = vec![0.1; 1000]; // Low energy signal
        let vad = SimdAudioProcessor::voice_activity_detection(&input, 256, 128, 0.5, 0.3).unwrap();

        assert!(!vad.is_empty());
        // Most frames should be classified as non-voice
        let voice_frames = vad.iter().filter(|&&is_voice| is_voice).count();
        assert!(voice_frames < vad.len() / 2);
    }

    #[test]
    fn test_mix_channels() {
        let channels = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]];
        let weights = vec![0.5, 0.5];
        let mut output = vec![0.0; 4];

        SimdAudioProcessor::mix_channels(&channels, &weights, &mut output).unwrap();

        assert_eq!(output, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_echo_effect() {
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 5];

        SimdAudioEffects::echo(&input, &mut output, 2, 0.5, 0.5).unwrap();

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 0.0);
        assert!(output[2] > 0.0); // Should have echo
    }
}
