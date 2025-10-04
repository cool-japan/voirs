//! Main processor for spatial audio vocoder integration.

use crate::models::hifigan::HiFiGanConfig;
use crate::models::spatial::*;
use anyhow::Result;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Main spatial audio processor that integrates all spatial-specific features
pub struct SpatialProcessor {
    /// Spatial vocoder configuration
    pub config: SpatialVocoderConfig,
    /// Base HiFi-GAN configuration
    pub hifigan_config: HiFiGanConfig,
    /// Spatial vocoder instance
    pub vocoder: SpatialVocoder,
    /// Processing statistics
    pub stats: SpatialProcessingStats,
}

/// Statistics for spatial audio processing
#[derive(Debug, Clone, Default)]
pub struct SpatialProcessingStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Average localization accuracy
    pub average_localization_accuracy: f32,
    /// Average spatial impression
    pub average_spatial_impression: f32,
    /// Average immersion level
    pub average_immersion_level: f32,
    /// Average binaural quality
    pub average_binaural_quality: f32,
    /// Processing time statistics
    pub processing_time_stats: ProcessingTimeStats,
}

/// Processing time statistics
#[derive(Debug, Clone, Default)]
pub struct ProcessingTimeStats {
    /// Total processing time (seconds)
    pub total_time: f64,
    /// Average processing time per frame (milliseconds)
    pub avg_time_per_frame: f64,
    /// Minimum processing time per frame (milliseconds)
    pub min_time_per_frame: f64,
    /// Maximum processing time per frame (milliseconds)
    pub max_time_per_frame: f64,
}

/// Spatial audio processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialProcessingOptions {
    /// Enable HRTF processing
    pub enable_hrtf: bool,
    /// Enable binaural rendering
    pub enable_binaural: bool,
    /// Enable 3D positioning
    pub enable_positioning: bool,
    /// Enable room acoustics
    pub enable_acoustics: bool,
    /// Enable quality metrics calculation
    pub enable_quality_metrics: bool,
    /// Real-time processing mode
    pub real_time_mode: bool,
    /// Quality vs speed tradeoff (0.0-1.0, higher = better quality)
    pub quality_preference: f32,
}

impl SpatialProcessor {
    /// Create new spatial processor
    pub fn new(config: SpatialVocoderConfig, hifigan_config: HiFiGanConfig) -> Result<Self> {
        let vocoder = SpatialVocoder::new(config.clone())?;

        Ok(Self {
            config,
            hifigan_config,
            vocoder,
            stats: SpatialProcessingStats::default(),
        })
    }

    /// Process mel spectrogram with spatial audio
    pub fn process_mel_spectrogram_spatial(
        &mut self,
        mel_spectrogram: &Array2<f32>,
        position: &SpatialPosition,
    ) -> Result<SpatialAudioOutput> {
        let start_time = std::time::Instant::now();

        // Process with spatial vocoder
        let output = self
            .vocoder
            .process_mel_spectrogram_spatial(mel_spectrogram, position)?;

        // Update statistics
        self.update_processing_stats(start_time.elapsed().as_secs_f64(), &output)?;

        Ok(output)
    }

    /// Process mel spectrogram with options
    pub fn process_mel_spectrogram_spatial_with_options(
        &mut self,
        mel_spectrogram: &Array2<f32>,
        position: &SpatialPosition,
        options: &SpatialProcessingOptions,
    ) -> Result<SpatialAudioOutput> {
        let start_time = std::time::Instant::now();

        // Create temporary configuration based on options
        let temp_config = self.create_temp_config(options)?;
        let original_config = self.config.clone();

        // Update vocoder configuration
        self.vocoder.update_config(temp_config)?;

        // Process with spatial vocoder
        let output = self
            .vocoder
            .process_mel_spectrogram_spatial(mel_spectrogram, position)?;

        // Restore original configuration
        self.vocoder.update_config(original_config)?;

        // Update statistics
        self.update_processing_stats(start_time.elapsed().as_secs_f64(), &output)?;

        Ok(output)
    }

    /// Process audio in real-time chunks
    pub fn process_realtime_spatial_chunk(
        &mut self,
        audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<SpatialAudioOutput> {
        // Optimize for real-time processing
        let rt_config = self.create_realtime_config()?;
        let original_config = self.config.clone();

        // Update vocoder configuration for real-time
        self.vocoder.update_config(rt_config)?;

        // Process chunk
        let output = self.vocoder.process_spatial_audio(audio, position)?;

        // Restore original configuration
        self.vocoder.update_config(original_config)?;

        Ok(output)
    }

    /// Create temporary configuration based on options
    fn create_temp_config(
        &self,
        options: &SpatialProcessingOptions,
    ) -> Result<SpatialVocoderConfig> {
        let mut config = self.config.clone();

        // Update configurations based on options
        config.hrtf.enable_hrtf = options.enable_hrtf;
        config.binaural.enable_binaural = options.enable_binaural;
        config.positioning.enable_positioning = options.enable_positioning;
        config.acoustics.enable_acoustics = options.enable_acoustics;
        config.quality_metrics.enable_metrics = options.enable_quality_metrics;

        // Adjust quality based on preference
        if options.quality_preference > 0.7 {
            // High quality mode
            config.hrtf.interpolation_method =
                crate::models::spatial::config::HrtfInterpolation::Cubic;
            config.binaural.crossfeed_amount *= 1.1;
            config.acoustics.reverb_config.reverb_level *= 1.2;
        } else if options.quality_preference < 0.3 {
            // Speed mode
            config.hrtf.interpolation_method =
                crate::models::spatial::config::HrtfInterpolation::Nearest;
            config.binaural.crossfeed_amount *= 0.9;
            config.acoustics.reverb_config.reverb_level *= 0.8;
        }

        Ok(config)
    }

    /// Create real-time optimized configuration
    fn create_realtime_config(&self) -> Result<SpatialVocoderConfig> {
        let mut config = self.config.clone();

        // Optimize for real-time processing
        config.hrtf.interpolation_method =
            crate::models::spatial::config::HrtfInterpolation::Nearest;
        config.binaural.crossfeed_amount *= 0.8;
        config.acoustics.reverb_config.reverb_level *= 0.7;
        config.acoustics.early_reflections_config.reflection_count = 5; // Reduced reflections

        // Reduce quality metrics calculation for speed
        config.quality_metrics.calculate_localization_accuracy = false;
        config.quality_metrics.calculate_spatial_impression = false;

        Ok(config)
    }

    /// Update processing statistics
    fn update_processing_stats(
        &mut self,
        processing_time: f64,
        output: &SpatialAudioOutput,
    ) -> Result<()> {
        let frame_count = output.left_channel.len() / self.config.hop_size;
        let processing_time_ms = processing_time * 1000.0;
        let time_per_frame = processing_time_ms / frame_count as f64;

        // Update frame count
        self.stats.frames_processed += frame_count as u64;

        // Update processing time statistics
        self.stats.processing_time_stats.total_time += processing_time;

        if self.stats.processing_time_stats.avg_time_per_frame == 0.0 {
            self.stats.processing_time_stats.avg_time_per_frame = time_per_frame;
            self.stats.processing_time_stats.min_time_per_frame = time_per_frame;
            self.stats.processing_time_stats.max_time_per_frame = time_per_frame;
        } else {
            // Update average (simple moving average)
            let weight = 0.1;
            self.stats.processing_time_stats.avg_time_per_frame =
                self.stats.processing_time_stats.avg_time_per_frame * (1.0 - weight)
                    + time_per_frame * weight;

            // Update min/max
            if time_per_frame < self.stats.processing_time_stats.min_time_per_frame {
                self.stats.processing_time_stats.min_time_per_frame = time_per_frame;
            }
            if time_per_frame > self.stats.processing_time_stats.max_time_per_frame {
                self.stats.processing_time_stats.max_time_per_frame = time_per_frame;
            }
        }

        // Update quality metrics
        self.stats.average_localization_accuracy = output.quality_score;
        self.stats.average_spatial_impression = output.quality_score;
        self.stats.average_immersion_level = output.quality_score;
        self.stats.average_binaural_quality = output.quality_score;

        Ok(())
    }

    /// Get real-time performance metrics
    pub fn get_performance_metrics(&self) -> SpatialPerformanceMetrics {
        let real_time_factor = if self.stats.processing_time_stats.avg_time_per_frame > 0.0 {
            let frame_duration_ms =
                (self.config.hop_size as f64 / self.config.sample_rate as f64) * 1000.0;
            frame_duration_ms / self.stats.processing_time_stats.avg_time_per_frame
        } else {
            0.0
        };

        SpatialPerformanceMetrics {
            frames_processed: self.stats.frames_processed,
            total_processing_time: self.stats.processing_time_stats.total_time,
            avg_time_per_frame_ms: self.stats.processing_time_stats.avg_time_per_frame,
            min_time_per_frame_ms: self.stats.processing_time_stats.min_time_per_frame,
            max_time_per_frame_ms: self.stats.processing_time_stats.max_time_per_frame,
            real_time_factor,
            is_real_time_capable: real_time_factor > 1.0,
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SpatialVocoderConfig) -> Result<()> {
        self.config = config.clone();
        self.vocoder.update_config(config)?;
        Ok(())
    }

    /// Update HiFi-GAN configuration
    pub fn update_hifigan_config(&mut self, config: HiFiGanConfig) -> Result<()> {
        self.hifigan_config = config;
        // Note: In a real implementation, this would update the underlying HiFi-GAN model
        Ok(())
    }

    /// Reset processing statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpatialProcessingStats::default();
    }

    /// Get configuration
    pub fn get_config(&self) -> &SpatialVocoderConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &SpatialProcessingStats {
        &self.stats
    }
}

/// Performance metrics for spatial audio processing
#[derive(Debug, Clone)]
pub struct SpatialPerformanceMetrics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total processing time (seconds)
    pub total_processing_time: f64,
    /// Average processing time per frame (milliseconds)
    pub avg_time_per_frame_ms: f64,
    /// Minimum processing time per frame (milliseconds)
    pub min_time_per_frame_ms: f64,
    /// Maximum processing time per frame (milliseconds)
    pub max_time_per_frame_ms: f64,
    /// Real-time factor (>1.0 means real-time capable)
    pub real_time_factor: f64,
    /// Whether processor is real-time capable
    pub is_real_time_capable: bool,
}

impl Default for SpatialProcessingOptions {
    fn default() -> Self {
        Self {
            enable_hrtf: true,
            enable_binaural: true,
            enable_positioning: true,
            enable_acoustics: true,
            enable_quality_metrics: true,
            real_time_mode: false,
            quality_preference: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hifigan::HiFiGanVariant;

    #[test]
    fn test_spatial_processor_creation() {
        let spatial_config = SpatialVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();

        let processor = SpatialProcessor::new(spatial_config, hifigan_config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_spatial_processing_options_default() {
        let options = SpatialProcessingOptions::default();
        assert!(options.enable_hrtf);
        assert!(options.enable_binaural);
        assert!(options.enable_positioning);
        assert!(options.enable_acoustics);
        assert!(options.enable_quality_metrics);
        assert!(!options.real_time_mode);
        assert_eq!(options.quality_preference, 0.7);
    }
}
