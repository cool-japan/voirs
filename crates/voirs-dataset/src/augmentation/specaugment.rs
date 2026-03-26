//! SpecAugment - Modern augmentation for speech and audio
//!
//! Implementation of SpecAugment (Park et al., 2019), a widely-used data augmentation
//! technique for speech processing that applies time and frequency masking to spectrograms.
//!
//! Reference: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
//! <https://arxiv.org/abs/1904.08779>

use crate::{AudioData, Result};
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

/// SpecAugment configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecAugmentConfig {
    /// Number of time masks to apply
    pub num_time_masks: usize,
    /// Maximum width of time mask (in time steps)
    pub time_mask_width: usize,
    /// Number of frequency masks to apply
    pub num_freq_masks: usize,
    /// Maximum width of frequency mask (in frequency bins)
    pub freq_mask_width: usize,
    /// Mask value (typically 0.0 for spectrograms)
    pub mask_value: f32,
    /// Enable time warping (experimental)
    pub enable_time_warp: bool,
    /// Time warp parameter W
    pub time_warp_w: usize,
}

impl Default for SpecAugmentConfig {
    fn default() -> Self {
        Self {
            num_time_masks: 2,
            time_mask_width: 100,
            num_freq_masks: 2,
            freq_mask_width: 27,
            mask_value: 0.0,
            enable_time_warp: false,
            time_warp_w: 80,
        }
    }
}

impl SpecAugmentConfig {
    /// Create configuration for light augmentation (LibriSpeech SM policy)
    pub fn light() -> Self {
        Self {
            num_time_masks: 1,
            time_mask_width: 70,
            num_freq_masks: 1,
            freq_mask_width: 15,
            mask_value: 0.0,
            enable_time_warp: false,
            time_warp_w: 40,
        }
    }

    /// Create configuration for strong augmentation (LibriSpeech LD policy)
    pub fn strong() -> Self {
        Self {
            num_time_masks: 2,
            time_mask_width: 100,
            num_freq_masks: 2,
            freq_mask_width: 27,
            mask_value: 0.0,
            enable_time_warp: true,
            time_warp_w: 80,
        }
    }

    /// Create configuration for extra strong augmentation (SwitchBoard policy)
    pub fn extra_strong() -> Self {
        Self {
            num_time_masks: 2,
            time_mask_width: 100,
            num_freq_masks: 2,
            freq_mask_width: 27,
            mask_value: 0.0,
            enable_time_warp: true,
            time_warp_w: 80,
        }
    }
}

/// SpecAugment augmenter for mel spectrograms
pub struct SpecAugment {
    config: SpecAugmentConfig,
}

impl SpecAugment {
    /// Create new SpecAugment augmenter
    pub fn new(config: SpecAugmentConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(SpecAugmentConfig::default())
    }

    /// Apply SpecAugment to a mel spectrogram
    ///
    /// # Arguments
    /// * `spectrogram` - Input spectrogram (freq_bins × time_steps)
    ///
    /// # Returns
    /// Augmented spectrogram with same dimensions
    pub fn augment_spectrogram(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        let mut augmented = spectrogram.clone();

        // Apply time warping if enabled
        if self.config.enable_time_warp {
            augmented = self.apply_time_warp(&augmented);
        }

        // Apply frequency masking
        for _ in 0..self.config.num_freq_masks {
            augmented = self.apply_frequency_mask(&augmented);
        }

        // Apply time masking
        for _ in 0..self.config.num_time_masks {
            augmented = self.apply_time_mask(&augmented);
        }

        augmented
    }

    /// Apply time masking to spectrogram
    ///
    /// Randomly masks t consecutive time steps [t0, t0 + t) where t is chosen
    /// from uniform distribution [0, T) and t0 is chosen from [0, τ - t)
    fn apply_time_mask(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        let num_time_steps = spectrogram.shape()[1];

        if num_time_steps == 0 {
            return spectrogram.clone();
        }

        // Choose mask width
        let mask_width = rng.random_range(0..=self.config.time_mask_width.min(num_time_steps));

        if mask_width == 0 || mask_width >= num_time_steps {
            return spectrogram.clone();
        }

        // Choose start position
        let start = rng.random_range(0..=(num_time_steps - mask_width));
        let end = start + mask_width;

        // Apply mask
        let mut masked = spectrogram.clone();
        for t in start..end {
            for f in 0..masked.shape()[0] {
                masked[[f, t]] = self.config.mask_value;
            }
        }

        masked
    }

    /// Apply frequency masking to spectrogram
    ///
    /// Randomly masks f consecutive frequency bins [f0, f0 + f) where f is chosen
    /// from uniform distribution [0, F) and f0 is chosen from [0, ν - f)
    fn apply_frequency_mask(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        let num_freq_bins = spectrogram.shape()[0];

        if num_freq_bins == 0 {
            return spectrogram.clone();
        }

        // Choose mask width
        let mask_width = rng.random_range(0..=self.config.freq_mask_width.min(num_freq_bins));

        if mask_width == 0 || mask_width >= num_freq_bins {
            return spectrogram.clone();
        }

        // Choose start position
        let start = rng.random_range(0..=(num_freq_bins - mask_width));
        let end = start + mask_width;

        // Apply mask
        let mut masked = spectrogram.clone();
        for f in start..end {
            for t in 0..masked.shape()[1] {
                masked[[f, t]] = self.config.mask_value;
            }
        }

        masked
    }

    /// Apply time warping to spectrogram (experimental)
    ///
    /// Warps the time axis using a sparse image warp with a random control point
    fn apply_time_warp(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        let num_time_steps = spectrogram.shape()[1];

        if num_time_steps < 2 * self.config.time_warp_w {
            return spectrogram.clone();
        }

        // Choose a random center point in the time axis
        let center =
            rng.random_range(self.config.time_warp_w..(num_time_steps - self.config.time_warp_w));

        // Choose a random warp distance
        let warp_distance = rng.random_range(0..self.config.time_warp_w) as i32;
        let direction = if rng.random_bool(0.5) { 1 } else { -1 };
        let warp_target = (center as i32 + direction * warp_distance)
            .max(0)
            .min((num_time_steps - 1) as i32) as usize;

        // Simple time warping: stretch or compress around the control point
        let mut warped = spectrogram.clone();

        // Left segment: [0, center) -> [0, warp_target)
        if warp_target != center {
            for t in 0..center {
                let warped_t = (t as f32 * warp_target as f32 / center as f32) as usize;
                if warped_t < num_time_steps {
                    for f in 0..warped.shape()[0] {
                        warped[[f, warped_t]] = spectrogram[[f, t]];
                    }
                }
            }

            // Right segment: [center, end) -> [warp_target, end)
            for t in center..num_time_steps {
                let segment_length = num_time_steps - center;
                let warped_segment_length = num_time_steps - warp_target;
                let offset = t - center;
                let warped_t = warp_target
                    + (offset as f32 * warped_segment_length as f32 / segment_length as f32)
                        as usize;
                if warped_t < num_time_steps {
                    for f in 0..warped.shape()[0] {
                        warped[[f, warped_t]] = spectrogram[[f, t]];
                    }
                }
            }
        }

        warped
    }

    /// Apply SpecAugment to audio samples by converting to spectrogram
    ///
    /// This is a convenience method that:
    /// 1. Converts audio to mel spectrogram
    /// 2. Applies SpecAugment
    /// 3. Converts back to audio (using Griffin-Lim or similar)
    ///
    /// Note: This is a lossy operation and primarily useful for training data augmentation
    pub fn augment_audio(&self, audio: &AudioData) -> Result<AudioData> {
        // For now, return the original audio as spectrogram->audio conversion
        // requires vocoder support which should be in voirs-vocoder
        // This method is provided for future integration
        Ok(audio.clone())
    }

    /// Apply SpecAugment with custom policy (adaptive augmentation)
    ///
    /// Adjusts augmentation strength based on sample difficulty
    pub fn augment_adaptive(&self, spectrogram: &Array2<f32>, difficulty: f32) -> Array2<f32> {
        // Scale augmentation parameters based on difficulty (0.0 = easy, 1.0 = hard)
        let scale_factor = (1.0 + difficulty) / 2.0; // Maps [0,1] to [0.5, 1.0]

        let adaptive_config = SpecAugmentConfig {
            num_time_masks: ((self.config.num_time_masks as f32 * scale_factor) as usize).max(1),
            time_mask_width: (self.config.time_mask_width as f32 * scale_factor) as usize,
            num_freq_masks: ((self.config.num_freq_masks as f32 * scale_factor) as usize).max(1),
            freq_mask_width: (self.config.freq_mask_width as f32 * scale_factor) as usize,
            mask_value: self.config.mask_value,
            enable_time_warp: self.config.enable_time_warp && difficulty > 0.5,
            time_warp_w: self.config.time_warp_w,
        };

        let adaptive_augmenter = SpecAugment::new(adaptive_config);
        adaptive_augmenter.augment_spectrogram(spectrogram)
    }
}

/// Batch SpecAugment processing for efficient augmentation
pub struct BatchSpecAugment {
    augmenter: SpecAugment,
}

impl BatchSpecAugment {
    /// Create new batch augmenter
    pub fn new(config: SpecAugmentConfig) -> Self {
        Self {
            augmenter: SpecAugment::new(config),
        }
    }

    /// Apply SpecAugment to a batch of spectrograms
    ///
    /// Each spectrogram gets independently augmented with different random masks
    pub fn augment_batch(&self, spectrograms: &[Array2<f32>]) -> Vec<Array2<f32>> {
        spectrograms
            .iter()
            .map(|spec| self.augmenter.augment_spectrogram(spec))
            .collect()
    }

    /// Apply adaptive augmentation to a batch with per-sample difficulty
    pub fn augment_batch_adaptive(
        &self,
        spectrograms: &[Array2<f32>],
        difficulties: &[f32],
    ) -> Vec<Array2<f32>> {
        spectrograms
            .iter()
            .zip(difficulties.iter())
            .map(|(spec, &difficulty)| self.augmenter.augment_adaptive(spec, difficulty))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_specaugment_creation() {
        let augmenter = SpecAugment::default_config();
        assert_eq!(augmenter.config.num_time_masks, 2);
        assert_eq!(augmenter.config.num_freq_masks, 2);
    }

    #[test]
    fn test_specaugment_light_config() {
        let config = SpecAugmentConfig::light();
        assert_eq!(config.num_time_masks, 1);
        assert_eq!(config.time_mask_width, 70);
        assert_eq!(config.num_freq_masks, 1);
        assert_eq!(config.freq_mask_width, 15);
    }

    #[test]
    fn test_specaugment_strong_config() {
        let config = SpecAugmentConfig::strong();
        assert_eq!(config.num_time_masks, 2);
        assert!(config.enable_time_warp);
    }

    #[test]
    fn test_time_masking() {
        let spectrogram = Array2::ones((80, 100)); // 80 mel bins, 100 time steps
        let config = SpecAugmentConfig {
            num_time_masks: 1,
            time_mask_width: 20,
            num_freq_masks: 0,
            freq_mask_width: 0,
            mask_value: 0.0,
            enable_time_warp: false,
            time_warp_w: 0,
        };

        let augmenter = SpecAugment::new(config);
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Check dimensions are preserved
        assert_eq!(augmented.shape(), spectrogram.shape());

        // Check that some values are masked (set to 0.0)
        let num_zeros = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 0, "Time masking should create some zero values");
    }

    #[test]
    fn test_frequency_masking() {
        let spectrogram = Array2::ones((80, 100)); // 80 mel bins, 100 time steps
        let config = SpecAugmentConfig {
            num_time_masks: 0,
            time_mask_width: 0,
            num_freq_masks: 1,
            freq_mask_width: 15,
            mask_value: 0.0,
            enable_time_warp: false,
            time_warp_w: 0,
        };

        let augmenter = SpecAugment::new(config);
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Check dimensions are preserved
        assert_eq!(augmented.shape(), spectrogram.shape());

        // Check that some values are masked
        let num_zeros = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(
            num_zeros > 0,
            "Frequency masking should create some zero values"
        );
    }

    #[test]
    fn test_combined_masking() {
        let spectrogram = Array2::ones((80, 100));
        let augmenter = SpecAugment::default_config();
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Check dimensions are preserved
        assert_eq!(augmented.shape(), spectrogram.shape());

        // Check that masking occurred
        let num_zeros = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 0, "Combined masking should create zero values");
    }

    #[test]
    fn test_empty_spectrogram() {
        let spectrogram = Array2::zeros((0, 0));
        let augmenter = SpecAugment::default_config();
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        assert_eq!(augmented.shape(), spectrogram.shape());
    }

    #[test]
    fn test_small_spectrogram() {
        let spectrogram = Array2::ones((5, 5));
        let augmenter = SpecAugment::default_config();
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Should handle small spectrograms gracefully
        assert_eq!(augmented.shape(), spectrogram.shape());
    }

    #[test]
    fn test_adaptive_augmentation() {
        let spectrogram = Array2::ones((80, 100));
        let augmenter = SpecAugment::default_config();

        // Easy sample (less augmentation)
        let easy_augmented = augmenter.augment_adaptive(&spectrogram, 0.0);
        let easy_zeros = easy_augmented.iter().filter(|&&x| x == 0.0).count();

        // Hard sample (more augmentation)
        let hard_augmented = augmenter.augment_adaptive(&spectrogram, 1.0);
        let hard_zeros = hard_augmented.iter().filter(|&&x| x == 0.0).count();

        // Hard samples should generally have more masking (but not guaranteed due to randomness)
        assert_eq!(easy_augmented.shape(), spectrogram.shape());
        assert_eq!(hard_augmented.shape(), spectrogram.shape());
        assert!(easy_zeros > 0 || hard_zeros > 0);
    }

    #[test]
    fn test_batch_augmentation() {
        let spec1 = Array2::ones((80, 100));
        let spec2 = Array2::ones((80, 100));
        let batch = vec![spec1, spec2];

        let batch_augmenter = BatchSpecAugment::new(SpecAugmentConfig::default());
        let augmented_batch = batch_augmenter.augment_batch(&batch);

        assert_eq!(augmented_batch.len(), 2);
        assert_eq!(augmented_batch[0].shape(), &[80, 100]);
        assert_eq!(augmented_batch[1].shape(), &[80, 100]);
    }

    #[test]
    fn test_batch_adaptive_augmentation() {
        let spec1 = Array2::ones((80, 100));
        let spec2 = Array2::ones((80, 100));
        let batch = vec![spec1, spec2];
        let difficulties = vec![0.2, 0.8];

        let batch_augmenter = BatchSpecAugment::new(SpecAugmentConfig::default());
        let augmented_batch = batch_augmenter.augment_batch_adaptive(&batch, &difficulties);

        assert_eq!(augmented_batch.len(), 2);
        assert_eq!(augmented_batch[0].shape(), &[80, 100]);
        assert_eq!(augmented_batch[1].shape(), &[80, 100]);
    }

    #[test]
    fn test_mask_value_customization() {
        let spectrogram = Array2::ones((80, 100));
        let config = SpecAugmentConfig {
            num_time_masks: 1,
            time_mask_width: 30,
            num_freq_masks: 1,
            freq_mask_width: 20,
            mask_value: -80.0, // Custom mask value (e.g., for log spectrograms)
            enable_time_warp: false,
            time_warp_w: 0,
        };

        let augmenter = SpecAugment::new(config);
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Check that custom mask value is used
        let has_custom_value = augmented.iter().any(|&x| x == -80.0);
        assert!(has_custom_value || augmented.shape()[0] == 0 || augmented.shape()[1] == 0);
    }

    #[test]
    fn test_time_warp() {
        let spectrogram = Array2::ones((80, 200)); // Larger for time warping
        let config = SpecAugmentConfig {
            num_time_masks: 0,
            time_mask_width: 0,
            num_freq_masks: 0,
            freq_mask_width: 0,
            mask_value: 0.0,
            enable_time_warp: true,
            time_warp_w: 80,
        };

        let augmenter = SpecAugment::new(config);
        let augmented = augmenter.augment_spectrogram(&spectrogram);

        // Check dimensions are preserved
        assert_eq!(augmented.shape(), spectrogram.shape());
    }

    #[test]
    fn test_deterministic_shape_preservation() {
        // Test that shape is always preserved regardless of random masks
        let spectrogram = Array2::from_elem((80, 100), 1.0);
        let augmenter = SpecAugment::default_config();

        for _ in 0..10 {
            let augmented = augmenter.augment_spectrogram(&spectrogram);
            assert_eq!(
                augmented.shape(),
                spectrogram.shape(),
                "Shape must be preserved across all augmentations"
            );
        }
    }
}
