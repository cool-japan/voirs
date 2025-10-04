//! Audio processing module for voirs-vocoder.
//!
//! This module provides comprehensive audio processing capabilities including:
//! - Audio buffer operations and format conversions
//! - Audio I/O (WAV, FLAC, MP3, etc.)
//! - Audio analysis and quality metrics
//! - Real-time audio processing utilities

pub mod analysis;
pub mod io;
pub mod ops;

pub use analysis::*;
pub use io::*;
pub use ops::*;

use crate::{AudioBuffer, Result};

/// Audio format types supported by the vocoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// 32-bit floating point
    F32,
    /// 16-bit signed integer
    I16,
    /// 24-bit signed integer
    I24,
    /// 32-bit signed integer
    I32,
}

/// Audio quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioQuality {
    /// Low quality (fast processing)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (slow processing)
    High,
    /// Ultra quality (very slow processing)
    Ultra,
}

/// Audio output configuration
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    /// Output sample rate
    pub sample_rate: u32,
    /// Number of output channels
    pub channels: u32,
    /// Audio format
    pub format: AudioFormat,
    /// Quality setting
    pub quality: AudioQuality,
    /// Enable real-time processing
    pub realtime: bool,
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            format: AudioFormat::F32,
            quality: AudioQuality::Medium,
            realtime: false,
        }
    }
}

/// Audio processing context
pub struct AudioProcessor {
    config: AudioOutputConfig,
}

impl AudioProcessor {
    /// Create new audio processor
    pub fn new(config: AudioOutputConfig) -> Self {
        Self { config }
    }

    /// Process audio buffer with current configuration
    pub fn process(&self, audio: &mut AudioBuffer) -> Result<()> {
        // Apply quality-based processing
        match self.config.quality {
            AudioQuality::Low => {
                // Basic processing only
            }
            AudioQuality::Medium => {
                // Standard processing
                post_process_audio(audio)?;
            }
            AudioQuality::High => {
                // Enhanced processing
                post_process_audio(audio)?;
                apply_enhancement_filters(audio)?;
            }
            AudioQuality::Ultra => {
                // Maximum quality processing
                post_process_audio(audio)?;
                apply_enhancement_filters(audio)?;
                apply_advanced_processing(audio)?;
            }
        }

        Ok(())
    }
}

/// Apply post-processing to audio
pub fn post_process_audio(audio: &mut AudioBuffer) -> Result<()> {
    // DC offset removal
    remove_dc_offset(audio);

    // Normalize audio levels
    normalize_audio(audio);

    Ok(())
}

/// Apply enhancement filters
pub fn apply_enhancement_filters(audio: &mut AudioBuffer) -> Result<()> {
    // High-frequency enhancement
    enhance_high_frequencies(audio);

    // Dynamic range optimization
    optimize_dynamic_range(audio);

    Ok(())
}

/// Apply advanced processing
pub fn apply_advanced_processing(audio: &mut AudioBuffer) -> Result<()> {
    // Spectral enhancement
    enhance_spectral_quality(audio);

    // Harmonic enhancement
    enhance_harmonics(audio);

    Ok(())
}

/// Remove DC offset from audio
pub fn remove_dc_offset(audio: &mut AudioBuffer) {
    let samples = audio.samples_mut();
    if samples.is_empty() {
        return;
    }

    // Calculate DC offset
    let dc_offset: f32 = samples.iter().sum::<f32>() / samples.len() as f32;

    // Remove DC offset
    for sample in samples.iter_mut() {
        *sample -= dc_offset;
    }
}

/// Normalize audio levels
pub fn normalize_audio(audio: &mut AudioBuffer) {
    let samples = audio.samples_mut();
    if samples.is_empty() {
        return;
    }

    // Find peak level
    let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    if peak > 0.0 && peak != 1.0 {
        // Normalize to prevent clipping (leave some headroom)
        let scale = 0.95 / peak;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Enhance high frequencies using a high-shelf filter
pub fn enhance_high_frequencies(audio: &mut AudioBuffer) {
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples_mut();

    // High-shelf filter parameters
    let cutoff_freq = 4000.0; // 4kHz cutoff
    let gain_db = 3.0; // 3dB boost
    let q = 0.7; // Q factor

    // Convert gain from dB to linear
    let gain = 10.0_f32.powf(gain_db / 20.0);

    // Calculate filter coefficients (high-shelf filter)
    let w = 2.0 * std::f32::consts::PI * cutoff_freq / sample_rate;
    let cos_w = w.cos();
    let sin_w = w.sin();
    let _alpha = sin_w / (2.0 * q);

    let _s = 1.0;
    let beta = gain.sqrt() / q;

    // High-shelf filter coefficients
    let a = gain + 1.0;
    let b = gain - 1.0;
    let c = 1.0 + gain;
    let d = 1.0 - gain;

    let b0 = gain * (a + b * cos_w + beta * sin_w);
    let b1 = -2.0 * gain * (b + a * cos_w);
    let b2 = gain * (a + b * cos_w - beta * sin_w);
    let a0 = c + d * cos_w + beta * sin_w;
    let a1 = -2.0 * (d + c * cos_w);
    let a2 = c + d * cos_w - beta * sin_w;

    // Normalize coefficients
    let norm = 1.0 / a0;
    let b0_norm = b0 * norm;
    let b1_norm = b1 * norm;
    let b2_norm = b2 * norm;
    let a1_norm = a1 * norm;
    let a2_norm = a2 * norm;

    // Apply filter (biquad implementation)
    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;

    for sample in samples.iter_mut() {
        let input = *sample;
        let output = b0_norm * input + b1_norm * x1 + b2_norm * x2 - a1_norm * y1 - a2_norm * y2;

        // Update delay line
        x2 = x1;
        x1 = input;
        y2 = y1;
        y1 = output;

        *sample = output;
    }
}

/// Optimize dynamic range using a soft compressor
pub fn optimize_dynamic_range(audio: &mut AudioBuffer) {
    let sample_rate = audio.sample_rate() as f32;
    let samples = audio.samples_mut();

    // Compressor parameters
    let threshold = 0.7; // Compression threshold (0-1)
    let ratio = 4.0; // Compression ratio (4:1)
    let attack_ms = 5.0; // Attack time in milliseconds
    let release_ms = 50.0; // Release time in milliseconds
    let knee_width = 0.1; // Soft knee width

    // Calculate envelope follower coefficients
    let attack_coeff = (-2.2 / (attack_ms * sample_rate * 0.001)).exp();
    let release_coeff = (-2.2 / (release_ms * sample_rate * 0.001)).exp();

    let mut envelope = 0.0;
    let mut gain_reduction = 1.0;

    for sample in samples.iter_mut() {
        let input = *sample;
        let input_level = input.abs();

        // Envelope follower
        let coeff = if input_level > envelope {
            attack_coeff
        } else {
            release_coeff
        };
        envelope = input_level + coeff * (envelope - input_level);

        // Calculate gain reduction
        let mut target_gain = 1.0;

        if envelope > threshold {
            // Soft knee implementation
            let over_threshold = envelope - threshold;
            let knee_ratio = (over_threshold / knee_width).min(1.0);

            // Apply soft knee
            let compressed_gain = if over_threshold <= knee_width {
                // Soft knee region
                let knee_factor = knee_ratio * knee_ratio;
                let compression_factor = 1.0 + knee_factor * (1.0 / ratio - 1.0);
                threshold + over_threshold * compression_factor
            } else {
                // Hard compression region
                threshold + over_threshold / ratio
            };

            target_gain = if envelope > 0.0 {
                compressed_gain / envelope
            } else {
                1.0
            };
        }

        // Smooth gain changes
        let gain_coeff = if target_gain < gain_reduction {
            attack_coeff
        } else {
            release_coeff
        };
        gain_reduction = target_gain + gain_coeff * (gain_reduction - target_gain);

        // Apply gain reduction
        *sample = input * gain_reduction;
    }

    // Apply makeup gain to compensate for overall level reduction
    let makeup_gain = 1.0 / (1.0 - (1.0 - 1.0 / ratio) * (threshold / 2.0));
    let makeup_gain_clamped = makeup_gain.min(2.0); // Limit makeup gain

    for sample in samples.iter_mut() {
        *sample *= makeup_gain_clamped;
        // Soft limiting to prevent clipping
        *sample = sample.clamp(-1.0, 1.0);
    }
}

/// Enhance spectral quality using FFT-based processing
pub fn enhance_spectral_quality(audio: &mut AudioBuffer) {
    use scirs2_core::Complex;

    let samples = audio.samples_mut();
    if samples.len() < 256 {
        return; // Too short for meaningful processing
    }

    let frame_size = 1024.min(samples.len().next_power_of_two());
    let hop_size = frame_size / 4;

    let mut output_samples = vec![0.0f32; samples.len()];

    // Hann window for overlapping frames
    let window: Vec<f32> = (0..frame_size)
        .map(|i| {
            0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (frame_size - 1) as f32).cos())
        })
        .collect();

    // Process overlapping frames
    let mut pos = 0;
    while pos + frame_size <= samples.len() {
        // Apply window and copy to input buffer
        let mut input_buffer = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            input_buffer[i] = samples[pos + i] * window[i];
        }

        // Forward FFT
        let mut spectrum = match scirs2_fft::rfft(&input_buffer, None) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Spectral enhancement: Boost mid-high frequencies selectively
        let spectrum_len = spectrum.len();
        for (i, bin) in spectrum.iter_mut().enumerate() {
            let freq_ratio = i as f32 / spectrum_len as f32;

            // Apply frequency-dependent enhancement
            let enhancement = if freq_ratio < 0.1 {
                1.0 // Low frequencies unchanged
            } else if freq_ratio < 0.4 {
                1.0 + 0.15 * ((freq_ratio - 0.1) / 0.3) // Gradual boost
            } else if freq_ratio < 0.7 {
                1.15 // Mid frequencies boosted
            } else {
                1.15 * (1.0 - (freq_ratio - 0.7) / 0.3) // High frequencies tapered
            };

            *bin *= enhancement as f64;
        }

        // Inverse FFT
        let time_output = match scirs2_fft::irfft(&spectrum, Some(frame_size)) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Overlap-add with windowing
        for i in 0..frame_size {
            if pos + i < output_samples.len() && i < time_output.len() {
                output_samples[pos + i] += (time_output[i] as f32) * window[i] / frame_size as f32;
            }
        }

        pos += hop_size;
    }

    // Copy enhanced audio back
    for (original, enhanced) in samples.iter_mut().zip(output_samples.iter()) {
        *original = enhanced.clamp(-1.0, 1.0);
    }
}

/// Enhance harmonics using advanced spectral analysis
pub fn enhance_harmonics(audio: &mut AudioBuffer) {
    use scirs2_core::Complex;

    let samples = audio.samples_mut();
    if samples.len() < 512 {
        return; // Too short for harmonic analysis
    }

    let frame_size = 2048.min(samples.len().next_power_of_two());
    let hop_size = frame_size / 8; // Smaller hop for better harmonic tracking

    let mut output_samples = vec![0.0f32; samples.len()];

    // Hann window
    let window: Vec<f32> = (0..frame_size)
        .map(|i| {
            0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (frame_size - 1) as f32).cos())
        })
        .collect();

    // Process overlapping frames
    let mut pos = 0;
    while pos + frame_size <= samples.len() {
        // Apply window and copy to input buffer
        let mut input_buffer = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            input_buffer[i] = samples[pos + i] * window[i];
        }

        // Forward FFT
        let mut spectrum = match scirs2_fft::rfft(&input_buffer, None) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Find fundamental frequency by looking for the strongest peak
        let mut fundamental_bin = 0;
        let mut max_magnitude = 0.0;

        // Search in the range 80Hz - 800Hz (assuming 44.1kHz sample rate)
        let min_bin = (80.0 * frame_size as f32 / 44100.0) as usize;
        let max_bin = (800.0 * frame_size as f32 / 44100.0) as usize;

        for (i, bin) in spectrum[min_bin..max_bin.min(spectrum.len())]
            .iter()
            .enumerate()
        {
            let magnitude = bin.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                fundamental_bin = min_bin + i;
            }
        }

        // Enhance harmonics if we found a fundamental
        if fundamental_bin > 0 && max_magnitude > 0.01 {
            // Enhance the first 6 harmonics
            for harmonic in 2..=6 {
                let harmonic_bin = fundamental_bin * harmonic;
                if harmonic_bin < spectrum.len() {
                    // Calculate enhancement factor based on harmonic number
                    let enhancement_factor = match harmonic {
                        2 => 1.4,  // Strong second harmonic
                        3 => 1.2,  // Moderate third harmonic
                        4 => 1.1,  // Mild fourth harmonic
                        5 => 1.05, // Slight fifth harmonic
                        _ => 1.02, // Very mild higher harmonics
                    };

                    // Apply enhancement with some spread to adjacent bins
                    for offset in -1i32..=1i32 {
                        let bin_index = (harmonic_bin as i32 + offset) as usize;
                        if bin_index < spectrum.len() {
                            spectrum[bin_index] *= enhancement_factor;
                        }
                    }
                }
            }

            // Also enhance the fundamental slightly
            spectrum[fundamental_bin] *= 1.1;
        }

        // Inverse FFT
        let time_output = match scirs2_fft::irfft(&spectrum, Some(frame_size)) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Overlap-add with windowing
        for i in 0..frame_size {
            if pos + i < output_samples.len() && i < time_output.len() {
                output_samples[pos + i] += (time_output[i] as f32) * window[i] / frame_size as f32;
            }
        }

        pos += hop_size;
    }

    // Copy enhanced audio back with gain compensation
    let gain_compensation = 0.8; // Prevent clipping from harmonic enhancement
    for (original, enhanced) in samples.iter_mut().zip(output_samples.iter()) {
        *original = (enhanced * gain_compensation).clamp(-1.0, 1.0);
    }
}

/// Extensions for AudioBuffer
impl AudioBuffer {
    /// Convert to different format
    pub fn convert_format(&self, format: AudioFormat) -> Result<Vec<u8>> {
        match format {
            AudioFormat::F32 => Ok(self.to_f32_bytes()),
            AudioFormat::I16 => Ok(self.to_i16_bytes()),
            AudioFormat::I24 => Ok(self.to_i24_bytes()),
            AudioFormat::I32 => Ok(self.to_i32_bytes()),
        }
    }

    /// Convert to F32 bytes
    pub fn to_f32_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .flat_map(|&x| x.to_le_bytes().to_vec())
            .collect()
    }

    /// Convert to I16 bytes
    pub fn to_i16_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|&x| (x * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect()
    }

    /// Convert to I24 bytes
    pub fn to_i24_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|&x| (x * 8388607.0).clamp(-8388608.0, 8388607.0) as i32)
            .flat_map(|x| x.to_le_bytes()[..3].to_vec())
            .collect()
    }

    /// Convert to I32 bytes
    pub fn to_i32_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|&x| (x * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as i32)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_processor() {
        let config = AudioOutputConfig::default();
        let processor = AudioProcessor::new(config);

        let mut audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 22050, 1);
        let result = processor.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_dc_offset() {
        let mut audio = AudioBuffer::new(vec![1.0, 1.0, 1.0, 1.0], 22050, 1);
        remove_dc_offset(&mut audio);

        // All samples should be 0 after DC removal
        for &sample in audio.samples() {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_normalize_audio() {
        let mut audio = AudioBuffer::new(vec![2.0, -2.0, 1.0, -1.0], 22050, 1);
        normalize_audio(&mut audio);

        // Peak should be around 0.95 (95% of max)
        let peak = audio
            .samples()
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_format_conversion() {
        let audio = AudioBuffer::new(vec![0.5, -0.5, 0.25, -0.25], 22050, 1);

        // Test I16 conversion
        let i16_bytes = audio.to_i16_bytes();
        assert_eq!(i16_bytes.len(), 8); // 4 samples * 2 bytes per sample

        // Test F32 conversion
        let f32_bytes = audio.to_f32_bytes();
        assert_eq!(f32_bytes.len(), 16); // 4 samples * 4 bytes per sample
    }
}
