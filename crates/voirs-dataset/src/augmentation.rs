//! Data augmentation utilities for speech synthesis datasets
//!
//! This module provides audio augmentation techniques including speed perturbation,
//! pitch shifting, noise injection, and room simulation.

pub mod noise;
pub mod pitch;
pub mod room;
pub mod speed;

use crate::{DatasetSample, Result};
use serde::{Deserialize, Serialize};

/// Augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Enable speed perturbation
    pub speed_perturbation: bool,
    /// Speed factors to apply
    pub speed_factors: Vec<f32>,
    /// Enable pitch shifting
    pub pitch_shifting: bool,
    /// Pitch shift range in semitones
    pub pitch_shift_range: (f32, f32),
    /// Enable noise injection
    pub noise_injection: bool,
    /// SNR range for noise injection
    pub snr_range: (f32, f32),
    /// Enable room simulation
    pub room_simulation: bool,
    /// Room types to simulate
    pub room_types: Vec<String>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.0, 1.1],
            pitch_shifting: false,
            pitch_shift_range: (-2.0, 2.0),
            noise_injection: false,
            snr_range: (10.0, 30.0),
            room_simulation: false,
            room_types: vec!["small_room".to_string(), "large_room".to_string()],
        }
    }
}

/// Audio augmentation pipeline
pub struct AudioAugmentor {
    config: AugmentationConfig,
}

impl AudioAugmentor {
    /// Create new augmentor with configuration
    pub fn new(config: AugmentationConfig) -> Self {
        Self { config }
    }

    /// Apply augmentation to a sample
    pub fn augment_sample(&self, sample: &DatasetSample) -> Result<Vec<DatasetSample>> {
        let mut augmented_samples = Vec::new();

        // Original sample
        augmented_samples.push(sample.clone());

        // Speed perturbation
        if self.config.speed_perturbation {
            for &factor in &self.config.speed_factors {
                if factor != 1.0 {
                    let augmented = self.apply_speed_perturbation(sample, factor)?;
                    augmented_samples.push(augmented);
                }
            }
        }

        // Pitch shifting
        if self.config.pitch_shifting {
            let pitch_shifts = [-2.0, -1.0, 1.0, 2.0]; // Semitones
            for &shift in &pitch_shifts {
                if shift >= self.config.pitch_shift_range.0
                    && shift <= self.config.pitch_shift_range.1
                {
                    let augmented = self.apply_pitch_shift(sample, shift)?;
                    augmented_samples.push(augmented);
                }
            }
        }

        // Noise injection
        if self.config.noise_injection {
            let snr_levels = [10.0, 15.0, 20.0]; // dB
            for &snr in &snr_levels {
                if (self.config.snr_range.0..=self.config.snr_range.1).contains(&snr) {
                    let augmented = self.apply_noise_injection(sample, snr)?;
                    augmented_samples.push(augmented);
                }
            }
        }

        // Room simulation
        if self.config.room_simulation {
            for room_type in &self.config.room_types {
                let augmented = self.apply_room_simulation(sample, room_type)?;
                augmented_samples.push(augmented);
            }
        }

        Ok(augmented_samples)
    }

    /// Apply speed perturbation
    fn apply_speed_perturbation(
        &self,
        sample: &DatasetSample,
        factor: f32,
    ) -> Result<DatasetSample> {
        use crate::augmentation::speed::{SpeedAugmentor, SpeedConfig};

        let config = SpeedConfig {
            speed_factors: vec![factor],
            preserve_pitch: true,
            window_size: 1024,
            overlap_ratio: 0.5,
            high_quality: true,
        };

        let augmentor = SpeedAugmentor::new(config);
        let augmented_audio = augmentor.apply_speed_perturbation(&sample.audio, factor)?;

        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_speed_{:.1}x", sample.id, factor);

        Ok(augmented)
    }

    /// Apply pitch shifting
    fn apply_pitch_shift(&self, sample: &DatasetSample, semitones: f32) -> Result<DatasetSample> {
        use crate::augmentation::pitch::{PitchAugmentor, PitchConfig};

        let config = PitchConfig {
            pitch_shifts: vec![semitones],
            preserve_formants: true,
            window_size: 2048,
            overlap_ratio: 0.75,
            high_quality: true,
            formant_preservation: 0.8,
        };

        let augmentor = PitchAugmentor::new(config);
        let augmented_audio = augmentor.apply_pitch_shift(&sample.audio, semitones)?;

        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_pitch_{:+.1}st", sample.id, semitones);

        Ok(augmented)
    }

    /// Apply noise injection
    fn apply_noise_injection(&self, sample: &DatasetSample, snr_db: f32) -> Result<DatasetSample> {
        use crate::augmentation::noise::{NoiseAugmentor, NoiseConfig, NoiseType};

        let config = NoiseConfig {
            noise_types: vec![NoiseType::White],
            snr_levels: vec![snr_db],
            noise_color: 1.0,
            dynamic_snr: false,
            snr_variation: 0.0,
            preserve_statistics: true,
        };

        let mut augmentor = NoiseAugmentor::new(config);
        let augmented_audio =
            augmentor.apply_noise_injection(&sample.audio, NoiseType::White, snr_db)?;

        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_noise_{}dB", sample.id, snr_db as i32);

        Ok(augmented)
    }

    /// Apply room simulation
    fn apply_room_simulation(
        &self,
        sample: &DatasetSample,
        room_type: &str,
    ) -> Result<DatasetSample> {
        use crate::augmentation::room::{RoomAugmentor, RoomConfig, RoomType};

        let room_enum = match room_type.to_lowercase().as_str() {
            "small_room" => RoomType::SmallRoom,
            "medium_room" => RoomType::MediumRoom,
            "large_room" => RoomType::LargeRoom,
            "concert_hall" => RoomType::ConcertHall,
            "cathedral" => RoomType::Cathedral,
            "studio" => RoomType::Studio,
            "bathroom" => RoomType::Bathroom,
            "outdoor" => RoomType::Outdoor,
            _ => RoomType::MediumRoom,
        };

        let config = RoomConfig {
            room_types: vec![room_enum],
            reverb_time: 1.2,
            early_delay: 20.0,
            reverb_level: 0.3,
            damping: 0.5,
            room_size: 1.0,
            use_parametric: true,
            diffusion: 0.7,
        };

        let augmentor = RoomAugmentor::new(config, sample.audio.sample_rate());
        let augmented_audio = augmentor.apply_room_simulation(&sample.audio, room_enum)?;

        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_{}", sample.id, room_type);

        Ok(augmented)
    }
}

/// Augmentation statistics
#[derive(Debug, Clone)]
pub struct AugmentationStats {
    /// Original sample count
    pub original_count: usize,
    /// Augmented sample count
    pub augmented_count: usize,
    /// Augmentation factor
    pub augmentation_factor: f32,
    /// Processing time
    pub processing_time: std::time::Duration,
}

impl AugmentationStats {
    /// Create new stats
    pub fn new(original_count: usize) -> Self {
        Self {
            original_count,
            augmented_count: 0,
            augmentation_factor: 1.0,
            processing_time: std::time::Duration::from_secs(0),
        }
    }

    /// Update stats
    pub fn update(&mut self, augmented_count: usize, processing_time: std::time::Duration) {
        self.augmented_count = augmented_count;
        self.augmentation_factor = if self.original_count > 0 {
            self.augmented_count as f32 / self.original_count as f32
        } else {
            1.0
        };
        self.processing_time = processing_time;
    }
}

/// Batch augmentation processor
pub struct BatchAugmentor {
    augmentor: AudioAugmentor,
}

impl BatchAugmentor {
    /// Create new batch augmentor
    pub fn new(config: AugmentationConfig) -> Self {
        Self {
            augmentor: AudioAugmentor::new(config),
        }
    }

    /// Process multiple samples
    pub fn process_batch(
        &self,
        samples: &[DatasetSample],
    ) -> Result<(Vec<DatasetSample>, AugmentationStats)> {
        let start_time = std::time::Instant::now();
        let mut all_augmented = Vec::new();
        let mut stats = AugmentationStats::new(samples.len());

        for sample in samples {
            let augmented = self.augmentor.augment_sample(sample)?;
            all_augmented.extend(augmented);
        }

        let processing_time = start_time.elapsed();
        stats.update(all_augmented.len(), processing_time);

        Ok((all_augmented, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, DatasetSample};

    fn create_test_sample() -> DatasetSample {
        // Create simple sine wave test data
        let sample_rate = 16000;
        let duration_secs = 0.5;
        let frequency = 440.0; // A4 note
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioData::new(samples, sample_rate, 1);
        DatasetSample {
            id: "test_sample_001".to_string(),
            audio,
            text: "Test audio sample".to_string(),
            speaker: None,
            language: crate::LanguageCode::EnUs,
            quality: crate::QualityMetrics {
                snr: Some(30.0),
                clipping: Some(0.01),
                dynamic_range: Some(60.0),
                spectral_quality: Some(0.95),
                overall_quality: Some(0.9),
            },
            phonemes: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_augmentation_config_default() {
        let config = AugmentationConfig::default();
        assert!(config.speed_perturbation);
        assert_eq!(config.speed_factors, vec![0.9, 1.0, 1.1]);
        assert!(!config.pitch_shifting);
        assert_eq!(config.pitch_shift_range, (-2.0, 2.0));
        assert!(!config.noise_injection);
        assert_eq!(config.snr_range, (10.0, 30.0));
        assert!(!config.room_simulation);
        assert_eq!(
            config.room_types,
            vec!["small_room".to_string(), "large_room".to_string()]
        );
    }

    #[test]
    fn test_audio_augmentor_creation() {
        let config = AugmentationConfig::default();
        let augmentor = AudioAugmentor::new(config);
        assert!(augmentor.config.speed_perturbation);
    }

    #[test]
    fn test_augment_sample_speed_only() {
        let config = AugmentationConfig {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.0, 1.1],
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include original + 2 speed variants (excluding 1.0)
        assert_eq!(augmented.len(), 3); // original + 0.9x + 1.1x

        // Check that IDs are properly set
        assert_eq!(augmented[0].id, "test_sample_001");
        assert!(augmented[1].id.contains("speed"));
        assert!(augmented[2].id.contains("speed"));
    }

    #[test]
    fn test_augment_sample_multiple_features() {
        let config = AugmentationConfig {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.1],
            pitch_shifting: true,
            pitch_shift_range: (-1.0, 1.0),
            noise_injection: true,
            snr_range: (15.0, 25.0),
            room_simulation: true,
            room_types: vec!["small_room".to_string()],
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include multiple augmentation types
        assert!(augmented.len() > 1);

        // Check that different augmentation types are present
        let has_speed = augmented.iter().any(|s| s.id.contains("speed"));
        let has_pitch = augmented.iter().any(|s| s.id.contains("pitch"));
        let has_noise = augmented.iter().any(|s| s.id.contains("noise"));
        let has_room = augmented.iter().any(|s| s.id.contains("small_room"));

        assert!(has_speed);
        assert!(has_pitch);
        assert!(has_noise);
        assert!(has_room);
    }

    #[test]
    fn test_augment_sample_no_augmentation() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should only return the original sample
        assert_eq!(augmented.len(), 1);
        assert_eq!(augmented[0].id, sample.id);
    }

    #[test]
    fn test_augmentation_stats_creation() {
        let stats = AugmentationStats::new(10);
        assert_eq!(stats.original_count, 10);
        assert_eq!(stats.augmented_count, 0);
        assert_eq!(stats.augmentation_factor, 1.0);
    }

    #[test]
    fn test_augmentation_stats_update() {
        let mut stats = AugmentationStats::new(10);
        stats.update(30, std::time::Duration::from_millis(500));

        assert_eq!(stats.augmented_count, 30);
        assert_eq!(stats.augmentation_factor, 3.0);
        assert_eq!(stats.processing_time, std::time::Duration::from_millis(500));
    }

    #[test]
    fn test_batch_augmentor_creation() {
        let config = AugmentationConfig::default();
        let batch_augmentor = BatchAugmentor::new(config);
        assert!(batch_augmentor.augmentor.config.speed_perturbation);
    }

    #[test]
    fn test_batch_augmentor_process_batch() {
        let config = AugmentationConfig {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.1],
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            ..Default::default()
        };

        let batch_augmentor = BatchAugmentor::new(config);
        let samples = vec![create_test_sample(), create_test_sample()];

        let (augmented, stats) = batch_augmentor.process_batch(&samples).unwrap();

        // Each sample should produce multiple augmented versions
        assert!(augmented.len() > samples.len());
        assert_eq!(stats.original_count, 2);
        assert!(stats.augmented_count > 2);
        assert!(stats.augmentation_factor > 1.0);
        assert!(stats.processing_time.as_millis() > 0);
    }

    #[test]
    fn test_pitch_shift_range_filtering() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: true,
            pitch_shift_range: (-1.0, 1.0), // Limited range
            noise_injection: false,
            room_simulation: false,
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should filter out shifts outside the range
        for sample in &augmented {
            if sample.id.contains("pitch") {
                // Extract the semitone value from ID and check it's in range
                let parts: Vec<&str> = sample.id.split('_').collect();
                if let Some(pitch_part) = parts.iter().find(|&&p| p.ends_with("st")) {
                    let semitone_str = pitch_part.replace("st", "").replace("+", "");
                    if let Ok(semitones) = semitone_str.parse::<f32>() {
                        assert!((-1.0..=1.0).contains(&semitones));
                    }
                }
            }
        }
    }

    #[test]
    fn test_snr_range_filtering() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: true,
            snr_range: (15.0, 25.0), // Limited range
            room_simulation: false,
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should filter out SNR values outside the range
        for sample in &augmented {
            if sample.id.contains("noise") {
                // Extract the SNR value from ID and check it's in range
                let parts: Vec<&str> = sample.id.split('_').collect();
                if let Some(noise_part) = parts.iter().find(|&&p| p.ends_with("dB")) {
                    let snr_str = noise_part.replace("dB", "");
                    if let Ok(snr) = snr_str.parse::<f32>() {
                        assert!((15.0..=25.0).contains(&snr));
                    }
                }
            }
        }
    }

    #[test]
    fn test_room_type_mapping() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: true,
            room_types: vec!["small_room".to_string(), "large_room".to_string()],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should create room simulation variants
        let room_variants: Vec<_> = augmented
            .iter()
            .filter(|s| s.id.contains("small_room") || s.id.contains("large_room"))
            .collect();

        assert!(!room_variants.is_empty());
    }
}
