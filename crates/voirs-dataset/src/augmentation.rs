//! Data augmentation utilities for speech synthesis datasets
//!
//! This module provides audio augmentation techniques including speed perturbation,
//! pitch shifting, noise injection, room simulation, codec simulation, VTLP,
//! dynamic range compression, and modern techniques like SpecAugment and MixUp.

pub mod codec;
pub mod compression;
pub mod formant;
pub mod mixup;
pub mod noise;
pub mod pitch;
pub mod room;
pub mod specaugment;
pub mod speed;
pub mod timestretch;
pub mod vtlp;

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
    /// Enable VTLP (Vocal Tract Length Perturbation)
    pub vtlp: bool,
    /// VTLP warp factors (typically 0.8 to 1.2)
    pub vtlp_warp_factors: Vec<f32>,
    /// Enable dynamic range compression
    pub compression: bool,
    /// Compression presets to apply
    pub compression_presets: Vec<String>,
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
            vtlp: false,
            vtlp_warp_factors: vec![0.9, 1.0, 1.1],
            compression: false,
            compression_presets: vec!["light".to_string(), "medium".to_string()],
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

        // VTLP (Vocal Tract Length Perturbation)
        if self.config.vtlp {
            for &warp_factor in &self.config.vtlp_warp_factors {
                if (warp_factor - 1.0).abs() > 1e-5 {
                    // Skip identity transform
                    let augmented = self.apply_vtlp(sample, warp_factor)?;
                    augmented_samples.push(augmented);
                }
            }
        }

        // Dynamic Range Compression
        if self.config.compression {
            for preset in &self.config.compression_presets {
                let augmented = self.apply_compression(sample, preset)?;
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

    /// Apply VTLP (Vocal Tract Length Perturbation)
    fn apply_vtlp(&self, sample: &DatasetSample, warp_factor: f32) -> Result<DatasetSample> {
        use crate::augmentation::vtlp::{VtlpAugmentor, VtlpConfig};

        let config = VtlpConfig {
            warp_factors: vec![warp_factor],
            sample_rate: sample.audio.sample_rate(),
            window_size: 1024,
            hop_size: 256,
            lower_cutoff: 80.0, // Typical lower bound for speech
            upper_cutoff: 0.0,  // Use Nyquist frequency
        };

        let augmentor = VtlpAugmentor::new(config);
        let audio_samples = sample.audio.samples();
        let augmented_samples = augmentor.apply_vtlp(audio_samples, warp_factor)?;

        // Create new AudioData with augmented samples
        let augmented_audio = crate::AudioData::new(
            augmented_samples,
            sample.audio.sample_rate(),
            sample.audio.channels(),
        );

        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_vtlp_{:.2}", sample.id, warp_factor);

        Ok(augmented)
    }

    /// Apply dynamic range compression
    fn apply_compression(&self, sample: &DatasetSample, preset: &str) -> Result<DatasetSample> {
        use crate::augmentation::compression::{CompressionConfig, DynamicRangeCompressor};

        let config = match preset.to_lowercase().as_str() {
            "light" => CompressionConfig::light(),
            "medium" => CompressionConfig::medium(),
            "heavy" => CompressionConfig::heavy(),
            "limiter" => CompressionConfig::limiter(),
            _ => CompressionConfig::medium(),
        };

        // Update sample rate to match audio
        let mut config = config;
        config.sample_rate = sample.audio.sample_rate();

        let mut compressor = DynamicRangeCompressor::new(config);
        let audio_samples = sample.audio.samples();
        let compressed_samples = compressor.apply(audio_samples)?;

        // Create new AudioData with compressed samples
        let compressed_audio = crate::AudioData::new(
            compressed_samples,
            sample.audio.sample_rate(),
            sample.audio.channels(),
        );

        let mut augmented = sample.clone();
        augmented.audio = compressed_audio;
        augmented.id = format!("{}_comp_{}", sample.id, preset);

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
        assert!(!config.vtlp);
        assert_eq!(config.vtlp_warp_factors, vec![0.9, 1.0, 1.1]);
        assert!(!config.compression);
        assert_eq!(
            config.compression_presets,
            vec!["light".to_string(), "medium".to_string()]
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
            vtlp: false,
            vtlp_warp_factors: vec![0.9, 1.1],
            compression: false,
            compression_presets: vec!["light".to_string()],
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

    #[test]
    fn test_vtlp_augmentation() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: true,
            vtlp_warp_factors: vec![0.9, 1.1],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include original + 2 VTLP variants (0.9 and 1.1, excluding 1.0)
        assert_eq!(augmented.len(), 3);

        // Check that IDs are properly set
        assert_eq!(augmented[0].id, "test_sample_001");
        assert!(augmented[1].id.contains("vtlp"));
        assert!(augmented[2].id.contains("vtlp"));

        // Check that audio lengths are preserved
        for sample in &augmented {
            assert!(sample.audio.samples().len() > 0);
        }
    }

    #[test]
    fn test_vtlp_warp_factor_filtering() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: true,
            vtlp_warp_factors: vec![0.9, 1.0, 1.1], // 1.0 should be filtered out
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should only include original + 2 variants (1.0 is identity, so filtered)
        assert_eq!(augmented.len(), 3);

        // Verify no identity transform was applied
        let vtlp_samples: Vec<_> = augmented
            .iter()
            .filter(|s| s.id.contains("vtlp_1.00"))
            .collect();
        assert_eq!(vtlp_samples.len(), 0);
    }

    #[test]
    fn test_combined_augmentation_with_vtlp() {
        let config = AugmentationConfig {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.1],
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: true,
            vtlp_warp_factors: vec![0.9, 1.1],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include original + speed variants + VTLP variants
        assert!(augmented.len() >= 5); // original + 2 speed + 2 VTLP

        // Check that both augmentation types are present
        let has_speed = augmented.iter().any(|s| s.id.contains("speed"));
        let has_vtlp = augmented.iter().any(|s| s.id.contains("vtlp"));

        assert!(has_speed);
        assert!(has_vtlp);
    }

    #[test]
    fn test_compression_augmentation() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: false,
            compression: true,
            compression_presets: vec!["light".to_string(), "medium".to_string()],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include original + 2 compression variants
        assert_eq!(augmented.len(), 3);

        // Check that IDs are properly set
        assert_eq!(augmented[0].id, "test_sample_001");
        assert!(augmented[1].id.contains("comp_"));
        assert!(augmented[2].id.contains("comp_"));

        // Check that audio lengths are preserved
        for sample in &augmented {
            assert!(sample.audio.samples().len() > 0);
        }
    }

    #[test]
    fn test_compression_preset_matching() {
        let config = AugmentationConfig {
            speed_perturbation: false,
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: false,
            compression: true,
            compression_presets: vec![
                "light".to_string(),
                "heavy".to_string(),
                "limiter".to_string(),
            ],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should create compression variants
        let comp_variants: Vec<_> = augmented
            .iter()
            .filter(|s| {
                s.id.contains("comp_light")
                    || s.id.contains("comp_heavy")
                    || s.id.contains("comp_limiter")
            })
            .collect();

        assert_eq!(comp_variants.len(), 3);
    }

    #[test]
    fn test_combined_augmentation_with_compression() {
        let config = AugmentationConfig {
            speed_perturbation: true,
            speed_factors: vec![0.9, 1.1],
            pitch_shifting: false,
            noise_injection: false,
            room_simulation: false,
            vtlp: false,
            compression: true,
            compression_presets: vec!["medium".to_string()],
            ..Default::default()
        };

        let augmentor = AudioAugmentor::new(config);
        let sample = create_test_sample();
        let augmented = augmentor.augment_sample(&sample).unwrap();

        // Should include original + speed variants + compression variants
        assert!(augmented.len() >= 4); // original + 2 speed + 1 compression

        // Check that both augmentation types are present
        let has_speed = augmented.iter().any(|s| s.id.contains("speed"));
        let has_compression = augmented.iter().any(|s| s.id.contains("comp_"));

        assert!(has_speed);
        assert!(has_compression);
    }
}
