//! Acoustic Processing Utilities
//!
//! Helper functions for common acoustic model operations including:
//! - Audio preprocessing and normalization
//! - Mel spectrogram manipulation
//! - Prosody smoothing and interpolation
//! - Quality-aware synthesis helpers

use crate::{AcousticError, Result};

/// Audio normalization utilities
pub mod audio {
    use super::*;

    /// Normalize audio to target RMS level
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `target_rms` - Target RMS level (default: 0.1)
    ///
    /// # Returns
    /// Normalized audio with specified RMS level
    pub fn normalize_rms(audio: &[f32], target_rms: f32) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // Calculate current RMS
        let sum_squares: f32 = audio.iter().map(|x| x * x).sum();
        let current_rms = (sum_squares / audio.len() as f32).sqrt();

        if current_rms < 1e-8 {
            // Audio is silent, return as-is
            return audio.to_vec();
        }

        // Calculate scaling factor
        let scale = target_rms / current_rms;

        // Apply normalization
        audio.iter().map(|x| x * scale).collect()
    }

    /// Normalize audio to peak level
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `target_peak` - Target peak level (default: 0.95)
    ///
    /// # Returns
    /// Normalized audio with specified peak level
    pub fn normalize_peak(audio: &[f32], target_peak: f32) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // Find peak amplitude
        let peak = audio
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        if peak < 1e-8 {
            // Audio is silent, return as-is
            return audio.to_vec();
        }

        // Calculate scaling factor
        let scale = target_peak / peak;

        // Apply normalization
        audio.iter().map(|x| x * scale).collect()
    }

    /// Apply fade-in to audio
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `fade_samples` - Number of samples for fade-in
    ///
    /// # Returns
    /// Audio with fade-in applied
    pub fn fade_in(audio: &[f32], fade_samples: usize) -> Vec<f32> {
        let mut result = audio.to_vec();
        let actual_fade = fade_samples.min(audio.len());

        for (i, sample) in result.iter_mut().enumerate().take(actual_fade) {
            let factor = i as f32 / actual_fade as f32;
            *sample *= factor;
        }

        result
    }

    /// Apply fade-out to audio
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `fade_samples` - Number of samples for fade-out
    ///
    /// # Returns
    /// Audio with fade-out applied
    pub fn fade_out(audio: &[f32], fade_samples: usize) -> Vec<f32> {
        let mut result = audio.to_vec();
        let len = audio.len();
        let actual_fade = fade_samples.min(len);
        let start_idx = len - actual_fade;

        for i in 0..actual_fade {
            let factor = 1.0 - (i as f32 / actual_fade as f32);
            result[start_idx + i] *= factor;
        }

        result
    }

    /// Cross-fade between two audio segments
    ///
    /// # Arguments
    /// * `audio1` - First audio segment
    /// * `audio2` - Second audio segment
    /// * `crossfade_samples` - Number of samples for cross-fade
    ///
    /// # Returns
    /// Cross-faded audio
    pub fn crossfade(audio1: &[f32], audio2: &[f32], crossfade_samples: usize) -> Vec<f32> {
        let len1 = audio1.len();
        let len2 = audio2.len();

        if len1 == 0 {
            return audio2.to_vec();
        }
        if len2 == 0 {
            return audio1.to_vec();
        }

        let actual_fade = crossfade_samples.min(len1).min(len2);
        let mut result = Vec::with_capacity(len1 + len2 - actual_fade);

        // First part (before crossfade)
        result.extend_from_slice(&audio1[..len1 - actual_fade]);

        // Crossfade region
        for i in 0..actual_fade {
            let factor = i as f32 / actual_fade as f32;
            let sample1 = audio1[len1 - actual_fade + i] * (1.0 - factor);
            let sample2 = audio2[i] * factor;
            result.push(sample1 + sample2);
        }

        // Second part (after crossfade)
        result.extend_from_slice(&audio2[actual_fade..]);

        result
    }

    /// Remove DC offset from audio
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    ///
    /// # Returns
    /// Audio with DC offset removed
    pub fn remove_dc_offset(audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // Calculate mean
        let mean: f32 = audio.iter().sum::<f32>() / audio.len() as f32;

        // Subtract mean
        audio.iter().map(|x| x - mean).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_normalize_rms() {
            let audio = vec![0.1, 0.2, 0.3, 0.4];
            let normalized = normalize_rms(&audio, 0.5);

            let rms: f32 = normalized.iter().map(|x| x * x).sum::<f32>() / normalized.len() as f32;
            let rms = rms.sqrt();

            assert!((rms - 0.5).abs() < 0.01);
        }

        #[test]
        fn test_normalize_peak() {
            let audio = vec![0.1, 0.5, 0.3, 0.2];
            let normalized = normalize_peak(&audio, 0.95);

            let peak = normalized.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            assert!((peak - 0.95).abs() < 0.01);
        }

        #[test]
        fn test_fade_in() {
            let audio = vec![1.0; 100];
            let faded = fade_in(&audio, 10);

            assert!(faded[0] < 0.1);
            assert!(faded[9] > 0.8);
            assert_eq!(faded[50], 1.0);
        }

        #[test]
        fn test_fade_out() {
            let audio = vec![1.0; 100];
            let faded = fade_out(&audio, 10);

            assert_eq!(faded[50], 1.0);
            assert!(faded[90] > 0.8);
            assert!(faded[99] < 0.15); // Last sample should be nearly silent
        }

        #[test]
        fn test_crossfade() {
            let audio1 = vec![1.0; 100];
            let audio2 = vec![0.5; 100];
            let crossfaded = crossfade(&audio1, &audio2, 20);

            assert_eq!(crossfaded.len(), 180);
            assert_eq!(crossfaded[0], 1.0);
            assert_eq!(crossfaded[179], 0.5);
        }

        #[test]
        fn test_remove_dc_offset() {
            let audio = vec![1.0, 1.1, 0.9, 1.0];
            let corrected = remove_dc_offset(&audio);

            let mean: f32 = corrected.iter().sum::<f32>() / corrected.len() as f32;
            assert!(mean.abs() < 1e-6);
        }
    }
}

/// Prosody smoothing and interpolation utilities
pub mod prosody {
    use super::*;

    /// Smooth prosody parameters using moving average
    ///
    /// # Arguments
    /// * `values` - Input prosody values (pitch, energy, etc.)
    /// * `window_size` - Smoothing window size (frames)
    ///
    /// # Returns
    /// Smoothed prosody values
    pub fn smooth_moving_average(values: &[f32], window_size: usize) -> Vec<f32> {
        if values.is_empty() || window_size == 0 {
            return values.to_vec();
        }

        let half_window = window_size / 2;
        let mut result = Vec::with_capacity(values.len());

        for i in 0..values.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(values.len());
            let sum: f32 = values[start..end].iter().sum();
            let count = (end - start) as f32;
            result.push(sum / count);
        }

        result
    }

    /// Interpolate prosody values linearly
    ///
    /// # Arguments
    /// * `start_value` - Starting value
    /// * `end_value` - Ending value
    /// * `num_frames` - Number of frames to interpolate
    ///
    /// # Returns
    /// Interpolated values
    pub fn interpolate_linear(start_value: f32, end_value: f32, num_frames: usize) -> Vec<f32> {
        if num_frames == 0 {
            return Vec::new();
        }

        if num_frames == 1 {
            return vec![start_value];
        }

        (0..num_frames)
            .map(|i| {
                let t = i as f32 / (num_frames - 1) as f32;
                start_value * (1.0 - t) + end_value * t
            })
            .collect()
    }

    /// Interpolate prosody values with cubic smoothing
    ///
    /// # Arguments
    /// * `start_value` - Starting value
    /// * `end_value` - Ending value
    /// * `num_frames` - Number of frames to interpolate
    ///
    /// # Returns
    /// Smoothly interpolated values using cubic easing
    pub fn interpolate_cubic(start_value: f32, end_value: f32, num_frames: usize) -> Vec<f32> {
        if num_frames == 0 {
            return Vec::new();
        }

        if num_frames == 1 {
            return vec![start_value];
        }

        (0..num_frames)
            .map(|i| {
                let t = i as f32 / (num_frames - 1) as f32;
                // Cubic ease-in-out
                let t = if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                };
                start_value * (1.0 - t) + end_value * t
            })
            .collect()
    }

    /// Detect outliers in prosody sequence and smooth them
    ///
    /// # Arguments
    /// * `values` - Input prosody values
    /// * `threshold` - Outlier detection threshold (standard deviations)
    ///
    /// # Returns
    /// Smoothed values with outliers corrected
    pub fn smooth_outliers(values: &[f32], threshold: f32) -> Vec<f32> {
        if values.len() < 3 {
            return values.to_vec();
        }

        // Calculate mean and std dev
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        let mut result = values.to_vec();

        for i in 0..values.len() {
            let deviation = (values[i] - mean).abs();
            if deviation > threshold * std_dev {
                // Replace outlier with interpolated value
                let prev = if i > 0 { result[i - 1] } else { mean };
                let next = if i < values.len() - 1 {
                    values[i + 1]
                } else {
                    mean
                };
                result[i] = (prev + next) / 2.0;
            }
        }

        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_smooth_moving_average() {
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let smoothed = smooth_moving_average(&values, 3);

            assert_eq!(smoothed.len(), 5);
            assert!((smoothed[2] - 3.0).abs() < 0.01);
        }

        #[test]
        fn test_interpolate_linear() {
            let interpolated = interpolate_linear(0.0, 10.0, 11);

            assert_eq!(interpolated.len(), 11);
            assert_eq!(interpolated[0], 0.0);
            assert_eq!(interpolated[10], 10.0);
            assert_eq!(interpolated[5], 5.0);
        }

        #[test]
        fn test_interpolate_cubic() {
            let interpolated = interpolate_cubic(0.0, 10.0, 11);

            assert_eq!(interpolated.len(), 11);
            assert_eq!(interpolated[0], 0.0);
            assert_eq!(interpolated[10], 10.0);
            // Middle value should be close to 5.0 with cubic easing
            assert!((interpolated[5] - 5.0).abs() < 1.0);
        }

        #[test]
        fn test_smooth_outliers() {
            let values = vec![1.0, 2.0, 10.0, 3.0, 2.0]; // 10.0 is an outlier relative to others
            let smoothed = smooth_outliers(&values, 1.5);

            // Outlier should be smoothed to average of neighbors
            // (2.0 + 3.0) / 2 = 2.5
            assert!((smoothed[2] - 2.5).abs() < 1.0); // Should be close to interpolated value
            assert!(smoothed[2] < values[2]); // Should be less than original outlier
        }
    }
}

/// Quality-aware synthesis utilities
pub mod quality {
    use super::*;

    /// Quality level for adaptive synthesis
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum QualityLevel {
        /// Maximum quality (slowest)
        Maximum,
        /// High quality
        High,
        /// Medium quality (balanced)
        Medium,
        /// Low quality (faster)
        Low,
        /// Minimum quality (fastest)
        Minimum,
    }

    impl QualityLevel {
        /// Get quality as a float between 0.0 and 1.0
        pub fn as_float(&self) -> f32 {
            match self {
                QualityLevel::Maximum => 1.0,
                QualityLevel::High => 0.8,
                QualityLevel::Medium => 0.6,
                QualityLevel::Low => 0.4,
                QualityLevel::Minimum => 0.2,
            }
        }

        /// Create from float value
        pub fn from_float(value: f32) -> Self {
            if value >= 0.9 {
                QualityLevel::Maximum
            } else if value >= 0.7 {
                QualityLevel::High
            } else if value >= 0.5 {
                QualityLevel::Medium
            } else if value >= 0.3 {
                QualityLevel::Low
            } else {
                QualityLevel::Minimum
            }
        }

        /// Get recommended chunk size for this quality level
        pub fn recommended_chunk_size(&self) -> usize {
            match self {
                QualityLevel::Maximum => 512,
                QualityLevel::High => 256,
                QualityLevel::Medium => 128,
                QualityLevel::Low => 64,
                QualityLevel::Minimum => 32,
            }
        }

        /// Get recommended sampling steps for diffusion models
        pub fn recommended_steps(&self) -> usize {
            match self {
                QualityLevel::Maximum => 50,
                QualityLevel::High => 30,
                QualityLevel::Medium => 20,
                QualityLevel::Low => 10,
                QualityLevel::Minimum => 5,
            }
        }
    }

    /// Synthesis parameters adjusted for quality level
    #[derive(Debug, Clone)]
    pub struct QualityAwareParams {
        /// Quality level
        pub quality: QualityLevel,
        /// Chunk size
        pub chunk_size: usize,
        /// Number of diffusion steps
        pub diffusion_steps: usize,
        /// Enable/disable expensive features
        pub use_expensive_features: bool,
    }

    impl QualityAwareParams {
        /// Create parameters for given quality level
        pub fn new(quality: QualityLevel) -> Self {
            Self {
                chunk_size: quality.recommended_chunk_size(),
                diffusion_steps: quality.recommended_steps(),
                use_expensive_features: matches!(
                    quality,
                    QualityLevel::Maximum | QualityLevel::High
                ),
                quality,
            }
        }

        /// Adapt parameters based on available resources
        pub fn adapt_to_resources(&mut self, available_memory_gb: f32, cpu_load: f32) {
            // Reduce quality if resources are constrained
            if available_memory_gb < 2.0 || cpu_load > 0.8 {
                self.quality = QualityLevel::from_float(self.quality.as_float() * 0.8);
                self.chunk_size = self.quality.recommended_chunk_size();
                self.diffusion_steps = self.quality.recommended_steps();
                self.use_expensive_features = false;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_quality_level_conversion() {
            assert_eq!(QualityLevel::Maximum.as_float(), 1.0);
            assert_eq!(QualityLevel::from_float(0.95), QualityLevel::Maximum);
            assert_eq!(QualityLevel::from_float(0.75), QualityLevel::High);
        }

        #[test]
        fn test_quality_aware_params() {
            let params = QualityAwareParams::new(QualityLevel::High);
            assert_eq!(params.chunk_size, 256);
            assert_eq!(params.diffusion_steps, 30);
            assert!(params.use_expensive_features);
        }

        #[test]
        fn test_adapt_to_resources() {
            let mut params = QualityAwareParams::new(QualityLevel::Maximum);

            // Simulate low resources
            params.adapt_to_resources(1.0, 0.9);

            // Quality should be reduced
            assert!(params.quality.as_float() < 1.0);
            assert!(!params.use_expensive_features);
        }
    }
}

/// Mel spectrogram manipulation utilities
pub mod mel {
    use super::*;

    /// Concatenate multiple mel spectrograms along time axis
    ///
    /// # Arguments
    /// * `mels` - Vector of mel spectrograms (each is [n_mels, time] flattened)
    /// * `n_mels` - Number of mel bands
    ///
    /// # Returns
    /// Concatenated mel spectrogram
    pub fn concatenate(mels: &[&[f32]], n_mels: usize) -> Result<Vec<f32>> {
        if mels.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate total time frames
        let total_frames: usize = mels.iter().map(|mel| mel.len() / n_mels).sum();

        let mut result = Vec::with_capacity(n_mels * total_frames);

        // Transpose and concatenate
        for mel in mels {
            if mel.len() % n_mels != 0 {
                return Err(AcousticError::ProcessingError {
                    message: format!(
                        "Mel spectrogram size {} not divisible by n_mels {}",
                        mel.len(),
                        n_mels
                    ),
                });
            }
            result.extend_from_slice(mel);
        }

        Ok(result)
    }

    /// Apply temporal smoothing to mel spectrogram
    ///
    /// # Arguments
    /// * `mel` - Input mel spectrogram (flattened [n_mels, time])
    /// * `n_mels` - Number of mel bands
    /// * `window_size` - Smoothing window size (frames)
    ///
    /// # Returns
    /// Smoothed mel spectrogram
    pub fn smooth_temporal(mel: &[f32], n_mels: usize, window_size: usize) -> Vec<f32> {
        if mel.is_empty() || window_size == 0 {
            return mel.to_vec();
        }

        let n_frames = mel.len() / n_mels;
        let mut result = vec![0.0; mel.len()];

        let half_window = window_size / 2;

        for t in 0..n_frames {
            let start = t.saturating_sub(half_window);
            let end = (t + half_window + 1).min(n_frames);
            let count = (end - start) as f32;

            for m in 0..n_mels {
                let mut sum = 0.0;
                for t2 in start..end {
                    sum += mel[t2 * n_mels + m];
                }
                result[t * n_mels + m] = sum / count;
            }
        }

        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_concatenate() {
            let mel1 = vec![1.0; 160]; // 80 mels x 2 frames
            let mel2 = vec![2.0; 240]; // 80 mels x 3 frames

            let result = concatenate(&[&mel1, &mel2], 80).unwrap();
            assert_eq!(result.len(), 400); // 80 mels x 5 frames
        }

        #[test]
        fn test_smooth_temporal() {
            let mel = vec![1.0; 160]; // 80 mels x 2 frames
            let smoothed = smooth_temporal(&mel, 80, 2);

            assert_eq!(smoothed.len(), 160);
        }
    }
}
