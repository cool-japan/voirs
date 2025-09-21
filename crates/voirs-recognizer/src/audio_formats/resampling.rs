//! Audio resampling and sample rate conversion utilities
//!
//! Provides high-quality sample rate conversion using various algorithms
//! optimized for speech recognition applications.

use crate::RecognitionError;
use voirs_sdk::AudioBuffer;

/// Resampling quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingQuality {
    /// Fast resampling with lower quality (linear interpolation)
    Fast,
    /// Medium quality resampling (cubic interpolation)
    Medium,
    /// High quality resampling (sinc interpolation)
    High,
    /// Highest quality for critical applications (windowed sinc)
    Ultra,
}

impl ResamplingQuality {
    /// Get the window size for sinc interpolation
    #[must_use]
    pub fn window_size(&self) -> usize {
        match self {
            ResamplingQuality::Fast => 4,
            ResamplingQuality::Medium => 8,
            ResamplingQuality::High => 16,
            ResamplingQuality::Ultra => 32,
        }
    }

    /// Get the filter cutoff frequency relative to Nyquist
    #[must_use]
    pub fn cutoff_frequency(&self) -> f32 {
        match self {
            ResamplingQuality::Fast => 0.8,
            ResamplingQuality::Medium => 0.85,
            ResamplingQuality::High => 0.9,
            ResamplingQuality::Ultra => 0.95,
        }
    }
}

/// Audio resampler for sample rate conversion
pub struct AudioResampler {
    quality: ResamplingQuality,
}

impl AudioResampler {
    /// Create a new resampler with specified quality
    #[must_use]
    pub fn new(quality: ResamplingQuality) -> Self {
        Self { quality }
    }

    /// Create a fast resampler (good for real-time applications)
    #[must_use]
    pub fn fast() -> Self {
        Self::new(ResamplingQuality::Fast)
    }

    /// Create a high-quality resampler (good for offline processing)
    #[must_use]
    pub fn high_quality() -> Self {
        Self::new(ResamplingQuality::High)
    }

    /// Resample audio buffer to target sample rate
    pub fn resample(
        &self,
        audio: &AudioBuffer,
        target_sample_rate: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        let source_rate = audio.sample_rate();

        if source_rate == target_sample_rate {
            return Ok(audio.clone()); // No resampling needed
        }

        let samples = audio.samples();
        let channels = audio.channels();

        // Handle multi-channel audio by resampling each channel separately
        let mut resampled_channels = Vec::new();

        for channel in 0..channels {
            let channel_samples = extract_channel(samples, channels, channel);
            let resampled =
                self.resample_mono(&channel_samples, source_rate, target_sample_rate)?;
            resampled_channels.push(resampled);
        }

        // Interleave channels back together
        let resampled_samples = interleave_channels(&resampled_channels);

        Ok(AudioBuffer::new(
            resampled_samples,
            target_sample_rate,
            channels,
        ))
    }

    /// Resample mono audio samples
    fn resample_mono(
        &self,
        samples: &[f32],
        source_rate: u32,
        target_rate: u32,
    ) -> Result<Vec<f32>, RecognitionError> {
        let ratio = f64::from(target_rate) / f64::from(source_rate);
        let output_length = (samples.len() as f64 * ratio).round() as usize;

        match self.quality {
            ResamplingQuality::Fast => self.resample_linear(samples, ratio, output_length),
            ResamplingQuality::Medium => self.resample_cubic(samples, ratio, output_length),
            ResamplingQuality::High | ResamplingQuality::Ultra => {
                self.resample_sinc(samples, ratio, output_length)
            }
        }
    }

    /// Linear interpolation resampling (fastest)
    fn resample_linear(
        &self,
        samples: &[f32],
        ratio: f64,
        output_length: usize,
    ) -> Result<Vec<f32>, RecognitionError> {
        let mut output = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let source_pos = i as f64 / ratio;
            let index = source_pos.floor() as usize;
            let fraction = source_pos - index as f64;

            if index >= samples.len() {
                break;
            }

            let sample = if index + 1 < samples.len() {
                // Linear interpolation
                let s0 = f64::from(samples[index]);
                let s1 = f64::from(samples[index + 1]);
                s0 + fraction * (s1 - s0)
            } else {
                f64::from(samples[index])
            };

            output.push(sample as f32);
        }

        Ok(output)
    }

    /// Cubic interpolation resampling (better quality)
    fn resample_cubic(
        &self,
        samples: &[f32],
        ratio: f64,
        output_length: usize,
    ) -> Result<Vec<f32>, RecognitionError> {
        let mut output = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let source_pos = i as f64 / ratio;
            let index = source_pos.floor() as usize;
            let fraction = source_pos - index as f64;

            if index >= samples.len() {
                break;
            }

            // Get 4 points for cubic interpolation
            let y0 = if index > 0 {
                f64::from(samples[index - 1])
            } else {
                f64::from(samples[index])
            };
            let y1 = f64::from(samples[index]);
            let y2 = if index + 1 < samples.len() {
                f64::from(samples[index + 1])
            } else {
                y1
            };
            let y3 = if index + 2 < samples.len() {
                f64::from(samples[index + 2])
            } else {
                y2
            };

            // Cubic interpolation using Catmull-Rom spline
            let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
            let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
            let c = -0.5 * y0 + 0.5 * y2;
            let d = y1;

            let sample = a * fraction.powi(3) + b * fraction.powi(2) + c * fraction + d;
            output.push(sample as f32);
        }

        Ok(output)
    }

    /// Sinc interpolation resampling (highest quality)
    fn resample_sinc(
        &self,
        samples: &[f32],
        ratio: f64,
        output_length: usize,
    ) -> Result<Vec<f32>, RecognitionError> {
        let mut output = Vec::with_capacity(output_length);
        let window_size = self.quality.window_size();
        let cutoff = f64::from(self.quality.cutoff_frequency());

        for i in 0..output_length {
            let source_pos = i as f64 / ratio;
            let center = source_pos.round() as isize;
            let offset = source_pos - center as f64;

            let mut sample = 0.0;
            let mut weight_sum = 0.0;

            for j in -(window_size as isize / 2)..(window_size as isize / 2) {
                let sample_index = center + j;

                if sample_index >= 0 && sample_index < samples.len() as isize {
                    let x = offset - j as f64;
                    let weight = if x.abs() < 1e-10 {
                        1.0
                    } else {
                        let sinc_val =
                            (std::f64::consts::PI * x * cutoff).sin() / (std::f64::consts::PI * x);
                        let window_val = self.blackman_window(x, window_size);
                        sinc_val * window_val
                    };

                    sample += f64::from(samples[sample_index as usize]) * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 1e-10 {
                sample /= weight_sum;
            }

            output.push(sample as f32);
        }

        Ok(output)
    }

    /// Blackman window function for sinc interpolation
    fn blackman_window(&self, x: f64, window_size: usize) -> f64 {
        let n = window_size as f64;
        let abs_x = x.abs();

        if abs_x >= n / 2.0 {
            return 0.0;
        }

        let arg = 2.0 * std::f64::consts::PI * (abs_x + n / 2.0) / n;
        0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos()
    }
}

impl Default for AudioResampler {
    fn default() -> Self {
        Self::new(ResamplingQuality::Medium)
    }
}

/// Extract a single channel from interleaved audio samples
fn extract_channel(samples: &[f32], total_channels: u32, channel: u32) -> Vec<f32> {
    if total_channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks_exact(total_channels as usize)
        .map(|chunk| chunk[channel as usize])
        .collect()
}

/// Interleave multiple channels back into a single sample array
fn interleave_channels(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
    }

    if channels.len() == 1 {
        return channels[0].clone();
    }

    let num_channels = channels.len();
    let samples_per_channel = channels[0].len();
    let mut output = Vec::with_capacity(num_channels * samples_per_channel);

    for sample_idx in 0..samples_per_channel {
        for channel in channels {
            if sample_idx < channel.len() {
                output.push(channel[sample_idx]);
            }
        }
    }

    output
}

/// Convert multi-channel audio to mono by mixing channels
#[must_use]
pub fn mix_to_mono(samples: &[f32], channels: u32) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    let samples_per_channel = samples.len() / channels as usize;
    let mut mono_samples = Vec::with_capacity(samples_per_channel);

    for i in 0..samples_per_channel {
        let mut sum = 0.0;
        for ch in 0..channels {
            let sample_idx = i * channels as usize + ch as usize;
            if sample_idx < samples.len() {
                sum += samples[sample_idx];
            }
        }
        mono_samples.push(sum / channels as f32);
    }

    mono_samples
}

/// Normalize audio samples to [-1.0, 1.0] range
pub fn normalize_audio(samples: &mut [f32]) {
    let max_abs = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    if max_abs > 1e-10 {
        let scale = 1.0 / max_abs;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Remove DC offset from audio samples
pub fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples.iter_mut() {
        *sample -= mean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampling_quality_settings() {
        assert_eq!(ResamplingQuality::Fast.window_size(), 4);
        assert_eq!(ResamplingQuality::High.window_size(), 16);
        assert!(
            ResamplingQuality::Ultra.cutoff_frequency()
                > ResamplingQuality::Fast.cutoff_frequency()
        );
    }

    #[test]
    fn test_no_resampling_needed() {
        let resampler = AudioResampler::fast();
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioBuffer::new(samples.clone(), 16000, 1);

        let result = resampler.resample(&audio, 16000).unwrap();
        assert_eq!(result.samples(), &samples);
        assert_eq!(result.sample_rate(), 16000);
    }

    #[test]
    fn test_linear_resampling() {
        let resampler = AudioResampler::fast();
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let audio = AudioBuffer::new(samples, 4, 1);

        let result = resampler.resample(&audio, 8).unwrap();
        assert_eq!(result.sample_rate(), 8);
        assert_eq!(result.samples().len(), 8);
    }

    #[test]
    fn test_channel_extraction() {
        let stereo_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let left_channel = extract_channel(&stereo_samples, 2, 0);
        let right_channel = extract_channel(&stereo_samples, 2, 1);

        assert_eq!(left_channel, vec![0.1, 0.3, 0.5]);
        assert_eq!(right_channel, vec![0.2, 0.4, 0.6]);
    }

    #[test]
    fn test_channel_interleaving() {
        let channels = vec![vec![0.1, 0.3, 0.5], vec![0.2, 0.4, 0.6]];
        let interleaved = interleave_channels(&channels);
        assert_eq!(interleaved, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_mix_to_mono() {
        let stereo_samples = vec![0.1, 0.3, 0.2, 0.4];
        let mono_samples = mix_to_mono(&stereo_samples, 2);
        assert_eq!(mono_samples, vec![0.2, 0.3]); // (0.1+0.3)/2, (0.2+0.4)/2
    }

    #[test]
    fn test_normalize_audio() {
        let mut samples = vec![0.5, -1.0, 0.25, 0.75];
        normalize_audio(&mut samples);

        let max_abs = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_dc_offset() {
        let mut samples = vec![1.1, 1.2, 1.3, 1.4];
        remove_dc_offset(&mut samples);

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 1e-6);
    }
}
