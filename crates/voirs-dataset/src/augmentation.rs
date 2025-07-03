//! Data augmentation utilities for speech synthesis datasets
//!
//! This module provides audio augmentation techniques including speed perturbation,
//! pitch shifting, noise injection, and room simulation.

pub mod speed;
pub mod pitch;
pub mod noise;
pub mod room;

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
                if shift >= self.config.pitch_shift_range.0 && shift <= self.config.pitch_shift_range.1 {
                    let augmented = self.apply_pitch_shift(sample, shift)?;
                    augmented_samples.push(augmented);
                }
            }
        }
        
        // Noise injection
        if self.config.noise_injection {
            let snr_levels = [10.0, 15.0, 20.0]; // dB
            for &snr in &snr_levels {
                if snr >= self.config.snr_range.0 && snr <= self.config.snr_range.1 {
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
    fn apply_speed_perturbation(&self, sample: &DatasetSample, factor: f32) -> Result<DatasetSample> {
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
        let augmented_audio = augmentor.apply_noise_injection(&sample.audio, NoiseType::White, snr_db)?;
        
        let mut augmented = sample.clone();
        augmented.audio = augmented_audio;
        augmented.id = format!("{}_noise_{}dB", sample.id, snr_db as i32);
        
        Ok(augmented)
    }
    
    /// Apply room simulation
    fn apply_room_simulation(&self, sample: &DatasetSample, room_type: &str) -> Result<DatasetSample> {
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
    pub fn process_batch(&self, samples: &[DatasetSample]) -> Result<(Vec<DatasetSample>, AugmentationStats)> {
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
