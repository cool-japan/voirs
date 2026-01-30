//! Property-based tests for mel spectrogram operations
//!
//! These tests use proptest to verify numerical properties across a wide range
//! of inputs, ensuring robustness and correctness of mel operations.

use proptest::prelude::*;
use voirs_acoustic::mel::ops::{MelOps, NormalizationMethod, PaddingMode};
use voirs_acoustic::MelSpectrogram;

/// Strategy for generating valid mel spectrogram dimensions
fn mel_dimensions() -> impl Strategy<Value = (usize, usize)> {
    (10usize..=128, 10usize..=500).prop_map(|(mels, frames)| (mels, frames))
}

/// Strategy for generating valid mel spectrogram data
fn mel_data_strategy() -> impl Strategy<Value = Vec<Vec<f32>>> {
    mel_dimensions().prop_flat_map(|(n_mels, n_frames)| {
        prop::collection::vec(
            prop::collection::vec(-100.0f32..100.0f32, n_frames..=n_frames),
            n_mels..=n_mels,
        )
    })
}

/// Strategy for generating positive-valued mel data (for certain normalizations)
fn positive_mel_data_strategy() -> impl Strategy<Value = Vec<Vec<f32>>> {
    mel_dimensions().prop_flat_map(|(n_mels, n_frames)| {
        prop::collection::vec(
            // Use a more conservative range to avoid proptest float sampler edge cases
            prop::collection::vec(0.1f32..10.0f32, n_frames..=n_frames),
            n_mels..=n_mels,
        )
    })
}

proptest! {
    /// Property: MinMax normalization should map all values to [0, 1]
    #[test]
    fn prop_minmax_normalization_bounds(data in mel_data_strategy()) {
        let mut mel = MelSpectrogram::new(data, 22050, 256);

        // Skip constant signals
        let stats = voirs_acoustic::mel::MelStats::compute(&mel).ok();
        if let Some(s) = stats {
            if (s.global.max - s.global.min).abs() < 1e-6 {
                return Ok(());
            }
        }

        MelOps::normalize_min_max(&mut mel).ok();

        // All values should be in [0, 1]
        for channel in &mel.data {
            for &value in channel {
                prop_assert!(value >= -0.01 && value <= 1.01,
                    "Value {} outside [0, 1] range", value);
            }
        }
    }

    /// Property: Z-score normalization should produce zero mean (approximately)
    #[test]
    fn prop_zscore_normalization_zero_mean(data in mel_data_strategy()) {
        let mut mel = MelSpectrogram::new(data, 22050, 256);

        // Skip constant signals
        let stats = voirs_acoustic::mel::MelStats::compute(&mel).ok();
        if let Some(s) = stats {
            if s.global.std < 1e-6 {
                return Ok(());
            }
        }

        MelOps::normalize_z_score(&mut mel).ok();

        // Compute global mean
        let sum: f32 = mel.data.iter().flatten().sum();
        let count = (mel.n_mels * mel.n_frames) as f32;
        let mean = sum / count;

        // Mean should be approximately zero
        prop_assert!((mean).abs() < 0.01,
            "Mean {} not close to zero", mean);
    }

    /// Property: Unit norm normalization should produce L2 norm ≈ 1
    #[test]
    fn prop_unit_norm_normalization(data in positive_mel_data_strategy()) {
        let mut mel = MelSpectrogram::new(data, 22050, 256);

        MelOps::normalize_unit_norm(&mut mel).ok();

        // Compute L2 norm
        let norm: f32 = mel.data.iter()
            .flatten()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        // Norm should be approximately 1.0
        prop_assert!((norm - 1.0).abs() < 0.01,
            "L2 norm {} not close to 1.0", norm);
    }

    /// Property: SIMD unit norm should match scalar unit norm
    #[test]
    fn prop_simd_unit_norm_equivalence(data in positive_mel_data_strategy()) {
        let mut mel_scalar = MelSpectrogram::new(data.clone(), 22050, 256);
        let mut mel_simd = MelSpectrogram::new(data, 22050, 256);

        MelOps::normalize_unit_norm(&mut mel_scalar).ok();
        MelOps::normalize_unit_norm_simd(&mut mel_simd).ok();

        // Results should be very close
        for (ch_scalar, ch_simd) in mel_scalar.data.iter().zip(mel_simd.data.iter()) {
            for (&v_scalar, &v_simd) in ch_scalar.iter().zip(ch_simd.iter()) {
                prop_assert!((v_scalar - v_simd).abs() < 1e-4,
                    "SIMD result {} differs from scalar {} by more than tolerance",
                    v_simd, v_scalar);
            }
        }
    }

    /// Property: Time stretching preserves number of mel bins
    #[test]
    fn prop_time_stretch_preserves_mels(
        data in mel_data_strategy(),
        factor in 0.5f32..2.0f32
    ) {
        let mel = MelSpectrogram::new(data, 22050, 256);
        let original_mels = mel.n_mels;

        let stretched = MelOps::time_stretch(&mel, factor).ok();

        if let Some(result) = stretched {
            prop_assert_eq!(result.n_mels, original_mels,
                "Number of mel bins changed after time stretching");
        }
    }

    /// Property: Pitch shifting preserves frame count
    #[test]
    fn prop_pitch_shift_preserves_frames(
        data in mel_data_strategy(),
        semitones in -12.0f32..12.0f32
    ) {
        let mel = MelSpectrogram::new(data, 22050, 256);
        let original_frames = mel.n_frames;

        let shifted = MelOps::pitch_shift(&mel, semitones).ok();

        if let Some(result) = shifted {
            prop_assert_eq!(result.n_frames, original_frames,
                "Number of frames changed after pitch shifting");
        }
    }

    /// Property: Smoothing preserves dimensions
    #[test]
    fn prop_smooth_preserves_dimensions(
        data in mel_data_strategy(),
        kernel_size in prop::sample::select(vec![3, 5, 7, 9])
    ) {
        let mut mel = MelSpectrogram::new(data, 22050, 256);
        let original_mels = mel.n_mels;
        let original_frames = mel.n_frames;

        MelOps::smooth(&mut mel, kernel_size).ok();

        prop_assert_eq!(mel.n_mels, original_mels);
        prop_assert_eq!(mel.n_frames, original_frames);
    }

    /// Property: SIMD smoothing matches scalar smoothing
    #[test]
    fn prop_simd_smooth_equivalence(
        data in mel_data_strategy(),
        kernel_size in prop::sample::select(vec![3, 5, 7])
    ) {
        let mut mel_scalar = MelSpectrogram::new(data.clone(), 22050, 256);
        let mut mel_simd = MelSpectrogram::new(data, 22050, 256);

        MelOps::smooth(&mut mel_scalar, kernel_size).ok();
        MelOps::smooth_simd(&mut mel_simd, kernel_size).ok();

        // Results should be very close
        for (ch_scalar, ch_simd) in mel_scalar.data.iter().zip(mel_simd.data.iter()) {
            for (&v_scalar, &v_simd) in ch_scalar.iter().zip(ch_simd.iter()) {
                prop_assert!((v_scalar - v_simd).abs() < 1e-4,
                    "SIMD smooth result differs from scalar by more than tolerance");
            }
        }
    }

    /// Property: Concatenation preserves total frame count
    #[test]
    fn prop_concatenate_preserves_frame_count(
        data1 in mel_data_strategy(),
        data2 in mel_data_strategy()
    ) {
        // Ensure same number of mel bins
        let n_mels = data1.len().min(data2.len());
        let data1_trimmed: Vec<Vec<f32>> = data1.into_iter().take(n_mels).collect();
        let data2_trimmed: Vec<Vec<f32>> = data2.into_iter().take(n_mels).collect();

        let mel1 = MelSpectrogram::new(data1_trimmed, 22050, 256);
        let mel2 = MelSpectrogram::new(data2_trimmed, 22050, 256);

        let frames1 = mel1.n_frames;
        let frames2 = mel2.n_frames;

        let concatenated = MelOps::concatenate(&[&mel1, &mel2]).ok();

        if let Some(result) = concatenated {
            prop_assert_eq!(result.n_frames, frames1 + frames2,
                "Concatenated frame count doesn't match sum");
        }
    }

    /// Property: Slicing preserves mel bins
    #[test]
    fn prop_slice_preserves_mels(
        data in mel_data_strategy(),
        start_frac in 0.0f32..0.5f32,
        end_frac in 0.5f32..1.0f32
    ) {
        let mel = MelSpectrogram::new(data, 22050, 256);
        let n_mels = mel.n_mels;

        let start = (mel.n_frames as f32 * start_frac) as usize;
        let end = (mel.n_frames as f32 * end_frac) as usize;

        if start < end && end <= mel.n_frames {
            let sliced = MelOps::slice(&mel, start, end).ok();

            if let Some(result) = sliced {
                prop_assert_eq!(result.n_mels, n_mels,
                    "Number of mel bins changed after slicing");
                prop_assert_eq!(result.n_frames, end - start,
                    "Sliced frame count doesn't match expected");
            }
        }
    }

    /// Property: Padding increases frame count correctly
    #[test]
    fn prop_padding_increases_frames(
        data in mel_data_strategy(),
        pad_left in 0usize..50,
        pad_right in 0usize..50
    ) {
        let mel = MelSpectrogram::new(data, 22050, 256);
        let original_frames = mel.n_frames;

        let padded = MelOps::pad(&mel, pad_left, pad_right, PaddingMode::Zero).ok();

        if let Some(result) = padded {
            prop_assert_eq!(result.n_frames, original_frames + pad_left + pad_right,
                "Padded frame count doesn't match expected");
        }
    }

    /// Property: Normalization is idempotent (normalizing twice = normalizing once)
    /// Note: Uses reduced case count to avoid proptest float sampler edge cases
    #[test]
    fn prop_normalization_idempotent(data in positive_mel_data_strategy().prop_perturb(|d, _| d).no_shrink()) {
        let mut mel1 = MelSpectrogram::new(data.clone(), 22050, 256);
        let mut mel2 = MelSpectrogram::new(data, 22050, 256);

        // Normalize once
        MelOps::normalize_unit_norm(&mut mel1).ok();

        // Normalize twice
        MelOps::normalize_unit_norm(&mut mel2).ok();
        MelOps::normalize_unit_norm(&mut mel2).ok();

        // Results should be very close
        for (ch1, ch2) in mel1.data.iter().zip(mel2.data.iter()) {
            for (&v1, &v2) in ch1.iter().zip(ch2.iter()) {
                prop_assert!((v1 - v2).abs() < 1e-4,
                    "Normalization not idempotent: {} vs {}", v1, v2);
            }
        }
    }
}
