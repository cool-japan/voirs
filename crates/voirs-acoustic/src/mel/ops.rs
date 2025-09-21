//! Mel spectrogram tensor operations and manipulations
//!
//! This module provides efficient tensor operations for mel spectrograms
//! including normalization, transformations, and optimized computations.

use super::MelStats;
use crate::{AcousticError, MelSpectrogram, Result};

/// Mel spectrogram operations
pub struct MelOps;

impl MelOps {
    /// Normalize mel spectrogram using various methods
    pub fn normalize(mel: &mut MelSpectrogram, method: NormalizationMethod) -> Result<()> {
        match method {
            NormalizationMethod::MinMax => Self::normalize_min_max(mel),
            NormalizationMethod::ZScore => Self::normalize_z_score(mel),
            NormalizationMethod::RobustScale => Self::normalize_robust_scale(mel),
            NormalizationMethod::UnitNorm => Self::normalize_unit_norm(mel),
            NormalizationMethod::PerChannel => Self::normalize_per_channel(mel),
        }
    }

    /// Min-max normalization to [0, 1] range
    pub fn normalize_min_max(mel: &mut MelSpectrogram) -> Result<()> {
        let stats = MelStats::compute(mel)?;
        let global_min = stats.global.min;
        let global_max = stats.global.max;
        let range = global_max - global_min;

        if range == 0.0 {
            return Err(AcousticError::InputError(
                "Cannot normalize constant signal".to_string(),
            ));
        }

        for channel in &mut mel.data {
            for value in channel {
                *value = (*value - global_min) / range;
            }
        }

        Ok(())
    }

    /// Z-score normalization (zero mean, unit variance)
    pub fn normalize_z_score(mel: &mut MelSpectrogram) -> Result<()> {
        let stats = MelStats::compute(mel)?;
        let global_mean = stats.global.mean;
        let global_std = stats.global.std;

        if global_std == 0.0 {
            return Err(AcousticError::InputError(
                "Cannot normalize constant signal".to_string(),
            ));
        }

        for channel in &mut mel.data {
            for value in channel {
                *value = (*value - global_mean) / global_std;
            }
        }

        Ok(())
    }

    /// Robust scaling using median and IQR
    pub fn normalize_robust_scale(mel: &mut MelSpectrogram) -> Result<()> {
        let all_values: Vec<f32> = mel.data.iter().flatten().copied().collect();
        let (median, iqr) = Self::compute_median_iqr(&all_values)?;

        if iqr == 0.0 {
            return Err(AcousticError::InputError(
                "Cannot normalize with zero IQR".to_string(),
            ));
        }

        for channel in &mut mel.data {
            for value in channel {
                *value = (*value - median) / iqr;
            }
        }

        Ok(())
    }

    /// Unit norm normalization (L2 norm = 1)
    pub fn normalize_unit_norm(mel: &mut MelSpectrogram) -> Result<()> {
        let norm: f32 = mel
            .data
            .iter()
            .flatten()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        if norm == 0.0 {
            return Err(AcousticError::InputError(
                "Cannot normalize zero signal".to_string(),
            ));
        }

        for channel in &mut mel.data {
            for value in channel {
                *value /= norm;
            }
        }

        Ok(())
    }

    /// Per-channel normalization
    pub fn normalize_per_channel(mel: &mut MelSpectrogram) -> Result<()> {
        for channel in &mut mel.data {
            let mean: f32 = channel.iter().sum::<f32>() / channel.len() as f32;
            let variance: f32 =
                channel.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / channel.len() as f32;
            let std = variance.sqrt();

            if std == 0.0 {
                continue; // Skip channels with zero variance
            }

            for value in channel {
                *value = (*value - mean) / std;
            }
        }

        Ok(())
    }

    /// Compute median and interquartile range
    fn compute_median_iqr(values: &[f32]) -> Result<(f32, f32)> {
        if values.is_empty() {
            return Err(AcousticError::InputError(
                "Empty input for median/IQR computation".to_string(),
            ));
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        Ok((median, iqr))
    }

    /// Apply time stretching to mel spectrogram
    pub fn time_stretch(mel: &MelSpectrogram, factor: f32) -> Result<MelSpectrogram> {
        if factor <= 0.0 {
            return Err(AcousticError::InputError(
                "Time stretch factor must be > 0".to_string(),
            ));
        }

        let new_n_frames = ((mel.n_frames as f32) / factor).round() as usize;
        let mut new_data = vec![vec![0.0; new_n_frames]; mel.n_mels];

        for (mel_idx, channel) in mel.data.iter().enumerate() {
            for new_frame_idx in 0..new_n_frames {
                let original_frame = new_frame_idx as f32 * factor;
                let frame_low = original_frame.floor() as usize;
                let frame_high = (frame_low + 1).min(mel.n_frames - 1);
                let frac = original_frame - frame_low as f32;

                if frame_low < channel.len() && frame_high < channel.len() {
                    new_data[mel_idx][new_frame_idx] =
                        channel[frame_low] * (1.0 - frac) + channel[frame_high] * frac;
                }
            }
        }

        Ok(MelSpectrogram::new(
            new_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }

    /// Apply pitch shifting to mel spectrogram
    pub fn pitch_shift(mel: &MelSpectrogram, semitones: f32) -> Result<MelSpectrogram> {
        let shift_factor = 2.0_f32.powf(semitones / 12.0);
        let mut new_data = vec![vec![0.0; mel.n_frames]; mel.n_mels];

        for frame_idx in 0..mel.n_frames {
            for (mel_idx, new_channel) in new_data.iter_mut().enumerate() {
                let original_mel = (mel_idx as f32) / shift_factor;
                let mel_low = original_mel.floor() as usize;
                let mel_high = (mel_low + 1).min(mel.n_mels - 1);
                let frac = original_mel - mel_low as f32;

                if mel_low < mel.data.len() && mel_high < mel.data.len() {
                    new_channel[frame_idx] = mel.data[mel_low][frame_idx] * (1.0 - frac)
                        + mel.data[mel_high][frame_idx] * frac;
                }
            }
        }

        Ok(MelSpectrogram::new(
            new_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }

    /// Concatenate multiple mel spectrograms along time axis
    pub fn concatenate(mels: &[&MelSpectrogram]) -> Result<MelSpectrogram> {
        if mels.is_empty() {
            return Err(AcousticError::InputError(
                "Cannot concatenate empty list".to_string(),
            ));
        }

        let first = mels[0];
        let n_mels = first.n_mels;
        let sample_rate = first.sample_rate;
        let hop_length = first.hop_length;

        // Validate compatibility
        for mel in mels.iter().skip(1) {
            if mel.n_mels != n_mels {
                return Err(AcousticError::InputError(
                    "All mel spectrograms must have same n_mels".to_string(),
                ));
            }
            if mel.sample_rate != sample_rate {
                return Err(AcousticError::InputError(
                    "All mel spectrograms must have same sample_rate".to_string(),
                ));
            }
            if mel.hop_length != hop_length {
                return Err(AcousticError::InputError(
                    "All mel spectrograms must have same hop_length".to_string(),
                ));
            }
        }

        let total_frames: usize = mels.iter().map(|mel| mel.n_frames).sum();
        let mut concatenated_data = vec![vec![0.0; total_frames]; n_mels];

        let mut frame_offset = 0;
        for mel in mels {
            for (mel_idx, channel) in mel.data.iter().enumerate() {
                for (frame_idx, &value) in channel.iter().enumerate() {
                    concatenated_data[mel_idx][frame_offset + frame_idx] = value;
                }
            }
            frame_offset += mel.n_frames;
        }

        Ok(MelSpectrogram::new(
            concatenated_data,
            sample_rate,
            hop_length,
        ))
    }

    /// Extract a slice from mel spectrogram
    pub fn slice(
        mel: &MelSpectrogram,
        start_frame: usize,
        end_frame: usize,
    ) -> Result<MelSpectrogram> {
        if start_frame >= end_frame {
            return Err(AcousticError::InputError(
                "Start frame must be < end frame".to_string(),
            ));
        }
        if end_frame > mel.n_frames {
            return Err(AcousticError::InputError(
                "End frame exceeds mel spectrogram length".to_string(),
            ));
        }

        let slice_length = end_frame - start_frame;
        let mut sliced_data = vec![vec![0.0; slice_length]; mel.n_mels];

        for (mel_idx, channel) in mel.data.iter().enumerate() {
            for (slice_idx, frame_idx) in (start_frame..end_frame).enumerate() {
                if frame_idx < channel.len() {
                    sliced_data[mel_idx][slice_idx] = channel[frame_idx];
                }
            }
        }

        Ok(MelSpectrogram::new(
            sliced_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }

    /// Pad mel spectrogram with zeros or constant values
    pub fn pad(
        mel: &MelSpectrogram,
        pad_left: usize,
        pad_right: usize,
        mode: PaddingMode,
    ) -> Result<MelSpectrogram> {
        let new_length = mel.n_frames + pad_left + pad_right;
        let mut padded_data = vec![vec![0.0; new_length]; mel.n_mels];

        for (mel_idx, channel) in mel.data.iter().enumerate() {
            // Apply left padding
            match mode {
                PaddingMode::Zero => {
                    // Already initialized with zeros
                }
                PaddingMode::Constant(value) => {
                    for i in 0..pad_left {
                        padded_data[mel_idx][i] = value;
                    }
                }
                PaddingMode::Reflect => {
                    for i in 0..pad_left {
                        let source_idx = (pad_left - 1 - i).min(channel.len() - 1);
                        padded_data[mel_idx][i] = channel[source_idx];
                    }
                }
                PaddingMode::Edge => {
                    let edge_value = channel.first().copied().unwrap_or(0.0);
                    for i in 0..pad_left {
                        padded_data[mel_idx][i] = edge_value;
                    }
                }
            }

            // Copy original data
            for (i, &value) in channel.iter().enumerate() {
                padded_data[mel_idx][pad_left + i] = value;
            }

            // Apply right padding
            match mode {
                PaddingMode::Zero => {
                    // Already initialized with zeros
                }
                PaddingMode::Constant(value) => {
                    for i in 0..pad_right {
                        padded_data[mel_idx][pad_left + mel.n_frames + i] = value;
                    }
                }
                PaddingMode::Reflect => {
                    for i in 0..pad_right {
                        let source_idx = channel.len() - 1 - (i % channel.len());
                        padded_data[mel_idx][pad_left + mel.n_frames + i] = channel[source_idx];
                    }
                }
                PaddingMode::Edge => {
                    let edge_value = channel.last().copied().unwrap_or(0.0);
                    for i in 0..pad_right {
                        padded_data[mel_idx][pad_left + mel.n_frames + i] = edge_value;
                    }
                }
            }
        }

        Ok(MelSpectrogram::new(
            padded_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }

    /// Apply smoothing filter to mel spectrogram
    pub fn smooth(mel: &mut MelSpectrogram, kernel_size: usize) -> Result<()> {
        if kernel_size == 0 || kernel_size % 2 == 0 {
            return Err(AcousticError::InputError(
                "Kernel size must be odd and > 0".to_string(),
            ));
        }

        let half_kernel = kernel_size / 2;
        let _kernel_weight = 1.0 / kernel_size as f32;

        for channel in &mut mel.data {
            let original = channel.clone();

            #[allow(clippy::needless_range_loop)]
            for i in 0..channel.len() {
                let mut sum = 0.0;
                let mut count = 0;

                for j in 0..kernel_size {
                    let idx = i as i32 + j as i32 - half_kernel as i32;
                    if idx >= 0 && (idx as usize) < original.len() {
                        sum += original[idx as usize];
                        count += 1;
                    }
                }

                if count > 0 {
                    channel[i] = sum / count as f32;
                }
            }
        }

        Ok(())
    }

    /// Apply delta (first derivative) computation
    pub fn compute_delta(mel: &MelSpectrogram, window_size: usize) -> Result<MelSpectrogram> {
        if window_size == 0 {
            return Err(AcousticError::InputError(
                "Window size must be > 0".to_string(),
            ));
        }

        let mut delta_data = vec![vec![0.0; mel.n_frames]; mel.n_mels];

        for (mel_idx, channel) in mel.data.iter().enumerate() {
            for frame_idx in 0..mel.n_frames {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for n in 1..=window_size {
                    let coeff = n as f32;

                    if frame_idx + n < mel.n_frames {
                        numerator += coeff * channel[frame_idx + n];
                        denominator += coeff * coeff;
                    }

                    if frame_idx >= n {
                        numerator -= coeff * channel[frame_idx - n];
                        denominator += coeff * coeff;
                    }
                }

                if denominator > 0.0 {
                    delta_data[mel_idx][frame_idx] = numerator / denominator;
                }
            }
        }

        Ok(MelSpectrogram::new(
            delta_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }

    /// Apply energy-based voice activity detection
    pub fn voice_activity_detection(mel: &MelSpectrogram, threshold: f32) -> Result<Vec<bool>> {
        let mut vad_result = vec![false; mel.n_frames];

        #[allow(clippy::needless_range_loop)]
        for frame_idx in 0..mel.n_frames {
            let mut energy = 0.0;
            for mel_idx in 0..mel.n_mels {
                if frame_idx < mel.data[mel_idx].len() {
                    energy += mel.data[mel_idx][frame_idx].powi(2);
                }
            }

            let rms = (energy / mel.n_mels as f32).sqrt();
            vad_result[frame_idx] = rms > threshold;
        }

        Ok(vad_result)
    }

    /// Resize mel spectrogram to target dimensions
    pub fn resize(
        mel: &MelSpectrogram,
        target_n_mels: usize,
        target_n_frames: usize,
    ) -> Result<MelSpectrogram> {
        if target_n_mels == 0 || target_n_frames == 0 {
            return Err(AcousticError::InputError(
                "Target dimensions must be > 0".to_string(),
            ));
        }

        let mut resized_data = vec![vec![0.0; target_n_frames]; target_n_mels];

        #[allow(clippy::needless_range_loop)]
        for new_mel_idx in 0..target_n_mels {
            for new_frame_idx in 0..target_n_frames {
                // Bilinear interpolation
                let mel_coord =
                    (new_mel_idx as f32) * (mel.n_mels as f32 - 1.0) / (target_n_mels as f32 - 1.0);
                let frame_coord = (new_frame_idx as f32) * (mel.n_frames as f32 - 1.0)
                    / (target_n_frames as f32 - 1.0);

                let mel_low = mel_coord.floor() as usize;
                let mel_high = (mel_low + 1).min(mel.n_mels - 1);
                let frame_low = frame_coord.floor() as usize;
                let frame_high = (frame_low + 1).min(mel.n_frames - 1);

                let mel_frac = mel_coord - mel_low as f32;
                let frame_frac = frame_coord - frame_low as f32;

                if mel_low < mel.data.len()
                    && mel_high < mel.data.len()
                    && frame_low < mel.data[mel_low].len()
                    && frame_high < mel.data[mel_low].len()
                    && frame_low < mel.data[mel_high].len()
                    && frame_high < mel.data[mel_high].len()
                {
                    let v1 = mel.data[mel_low][frame_low] * (1.0 - frame_frac)
                        + mel.data[mel_low][frame_high] * frame_frac;
                    let v2 = mel.data[mel_high][frame_low] * (1.0 - frame_frac)
                        + mel.data[mel_high][frame_high] * frame_frac;

                    resized_data[new_mel_idx][new_frame_idx] =
                        v1 * (1.0 - mel_frac) + v2 * mel_frac;
                }
            }
        }

        Ok(MelSpectrogram::new(
            resized_data,
            mel.sample_rate,
            mel.hop_length,
        ))
    }
}

/// Normalization methods for mel spectrograms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (zero mean, unit variance)
    ZScore,
    /// Robust scaling using median and IQR
    RobustScale,
    /// Unit norm normalization (L2 norm = 1)
    UnitNorm,
    /// Per-channel normalization
    PerChannel,
}

/// Padding modes for mel spectrograms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// Pad with zeros
    Zero,
    /// Pad with constant value
    Constant(f32),
    /// Reflect padding
    Reflect,
    /// Edge padding (repeat edge values)
    Edge,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mel() -> MelSpectrogram {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![0.5, 1.0, 1.5, 2.0],
        ];
        MelSpectrogram::new(data, 22050, 256)
    }

    #[test]
    fn test_normalize_min_max() {
        let mut mel = create_test_mel();
        MelOps::normalize_min_max(&mut mel).unwrap();

        // Check that values are in [0, 1] range
        for channel in &mel.data {
            for &value in channel {
                assert!((0.0..=1.0).contains(&value));
            }
        }
    }

    #[test]
    fn test_normalize_z_score() {
        let mut mel = create_test_mel();
        MelOps::normalize_z_score(&mut mel).unwrap();

        // Check that global mean is approximately zero
        let global_mean: f32 =
            mel.data.iter().flatten().sum::<f32>() / (mel.n_mels * mel.n_frames) as f32;
        assert!((global_mean).abs() < 1e-6);
    }

    #[test]
    fn test_time_stretch() {
        let mel = create_test_mel();

        // Stretch by factor of 2 (slow down)
        let stretched = MelOps::time_stretch(&mel, 2.0).unwrap();
        assert_eq!(stretched.n_frames, 2); // 4 / 2 = 2
        assert_eq!(stretched.n_mels, mel.n_mels);

        // Compress by factor of 0.5 (speed up)
        let compressed = MelOps::time_stretch(&mel, 0.5).unwrap();
        assert_eq!(compressed.n_frames, 8); // 4 / 0.5 = 8
    }

    #[test]
    fn test_pitch_shift() {
        let mel = create_test_mel();
        let shifted = MelOps::pitch_shift(&mel, 12.0).unwrap(); // One octave up

        assert_eq!(shifted.n_frames, mel.n_frames);
        assert_eq!(shifted.n_mels, mel.n_mels);
    }

    #[test]
    fn test_concatenate() {
        let mel1 = create_test_mel();
        let mel2 = create_test_mel();

        let concatenated = MelOps::concatenate(&[&mel1, &mel2]).unwrap();

        assert_eq!(concatenated.n_frames, mel1.n_frames + mel2.n_frames);
        assert_eq!(concatenated.n_mels, mel1.n_mels);
        assert_eq!(concatenated.sample_rate, mel1.sample_rate);
    }

    #[test]
    fn test_slice() {
        let mel = create_test_mel();
        let sliced = MelOps::slice(&mel, 1, 3).unwrap();

        assert_eq!(sliced.n_frames, 2);
        assert_eq!(sliced.n_mels, mel.n_mels);
        assert_eq!(sliced.data[0][0], mel.data[0][1]);
        assert_eq!(sliced.data[0][1], mel.data[0][2]);
    }

    #[test]
    fn test_pad() {
        let mel = create_test_mel();
        let padded = MelOps::pad(&mel, 2, 3, PaddingMode::Zero).unwrap();

        assert_eq!(padded.n_frames, mel.n_frames + 5); // 2 + 4 + 3 = 9
        assert_eq!(padded.n_mels, mel.n_mels);

        // Check that padding is zero
        assert_eq!(padded.data[0][0], 0.0);
        assert_eq!(padded.data[0][1], 0.0);
        assert_eq!(padded.data[0][6], 0.0);
        assert_eq!(padded.data[0][7], 0.0);
        assert_eq!(padded.data[0][8], 0.0);

        // Check that original data is preserved
        assert_eq!(padded.data[0][2], mel.data[0][0]);
        assert_eq!(padded.data[0][3], mel.data[0][1]);
    }

    #[test]
    fn test_pad_constant() {
        let mel = create_test_mel();
        let padded = MelOps::pad(&mel, 1, 1, PaddingMode::Constant(5.0)).unwrap();

        assert_eq!(padded.data[0][0], 5.0);
        assert_eq!(padded.data[0][5], 5.0);
    }

    #[test]
    fn test_smooth() {
        let mut mel = create_test_mel();
        MelOps::smooth(&mut mel, 3).unwrap();

        // After smoothing, values should be averaged
        assert_eq!(mel.n_frames, 4);
        assert_eq!(mel.n_mels, 3);
    }

    #[test]
    fn test_compute_delta() {
        let mel = create_test_mel();
        let delta = MelOps::compute_delta(&mel, 1).unwrap();

        assert_eq!(delta.n_frames, mel.n_frames);
        assert_eq!(delta.n_mels, mel.n_mels);
    }

    #[test]
    fn test_voice_activity_detection() {
        let mel = create_test_mel();
        let vad = MelOps::voice_activity_detection(&mel, 1.0).unwrap();

        assert_eq!(vad.len(), mel.n_frames);
        assert!(vad.iter().any(|&x| x)); // At least some frames should be active
    }

    #[test]
    fn test_resize() {
        let mel = create_test_mel();
        let resized = MelOps::resize(&mel, 5, 6).unwrap();

        assert_eq!(resized.n_mels, 5);
        assert_eq!(resized.n_frames, 6);
    }

    #[test]
    fn test_error_cases() {
        let mel = create_test_mel();

        // Invalid time stretch factor
        assert!(MelOps::time_stretch(&mel, 0.0).is_err());
        assert!(MelOps::time_stretch(&mel, -1.0).is_err());

        // Invalid slice parameters
        assert!(MelOps::slice(&mel, 3, 2).is_err()); // start >= end
        assert!(MelOps::slice(&mel, 0, 10).is_err()); // end > length

        // Invalid smooth kernel size
        let mut mel_copy = mel.clone();
        assert!(MelOps::smooth(&mut mel_copy, 0).is_err()); // kernel size 0
        assert!(MelOps::smooth(&mut mel_copy, 2).is_err()); // even kernel size

        // Invalid resize dimensions
        assert!(MelOps::resize(&mel, 0, 5).is_err());
        assert!(MelOps::resize(&mel, 5, 0).is_err());

        // Empty concatenation
        assert!(MelOps::concatenate(&[]).is_err());
    }

    #[test]
    fn test_concatenate_incompatible() {
        let mel1 = create_test_mel();
        let mut mel2 = create_test_mel();
        mel2.sample_rate = 16000; // Different sample rate

        assert!(MelOps::concatenate(&[&mel1, &mel2]).is_err());
    }
}
