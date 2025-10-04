//! Main processor for singing voice vocoder integration.

use crate::models::hifigan::HiFiGanConfig;
use crate::models::singing::*;
use anyhow::Result;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Main singing voice processor that integrates all singing-specific features
pub struct SingingProcessor {
    /// Singing vocoder configuration
    pub config: SingingVocoderConfig,
    /// Base HiFi-GAN configuration
    pub hifigan_config: HiFiGanConfig,
    /// Singing vocoder instance
    pub vocoder: SingingVocoder,
    /// Processing statistics
    pub stats: SingingProcessingStats,
}

/// Statistics for singing voice processing
#[derive(Debug, Clone, Default)]
pub struct SingingProcessingStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Average pitch accuracy
    pub average_pitch_accuracy: f32,
    /// Average harmonic clarity
    pub average_harmonic_clarity: f32,
    /// Average spectral stability
    pub average_spectral_stability: f32,
    /// Average singing quality
    pub average_singing_quality: f32,
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

/// Singing voice processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingProcessingOptions {
    /// Enable pitch stability correction
    pub enable_pitch_correction: bool,
    /// Enable vibrato enhancement
    pub enable_vibrato_enhancement: bool,
    /// Enable harmonic enhancement
    pub enable_harmonic_enhancement: bool,
    /// Enable breath sound processing
    pub enable_breath_processing: bool,
    /// Enable artifact reduction
    pub enable_artifact_reduction: bool,
    /// Enable quality metrics calculation
    pub enable_quality_metrics: bool,
    /// Real-time processing mode
    pub real_time_mode: bool,
    /// Quality vs speed tradeoff (0.0-1.0, higher = better quality)
    pub quality_preference: f32,
}

impl SingingProcessor {
    /// Create new singing processor
    pub fn new(config: SingingVocoderConfig, hifigan_config: HiFiGanConfig) -> Result<Self> {
        let vocoder = SingingVocoder::new(config.clone())?;

        Ok(Self {
            config,
            hifigan_config,
            vocoder,
            stats: SingingProcessingStats::default(),
        })
    }

    /// Process mel spectrogram with singing voice enhancements
    pub fn process_mel_spectrogram(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        // Process with singing vocoder
        let audio = self.vocoder.process_singing_voice(mel_spectrogram)?;

        // Update statistics
        self.update_processing_stats(start_time.elapsed().as_secs_f64(), &audio)?;

        Ok(audio)
    }

    /// Process mel spectrogram with options
    pub fn process_mel_spectrogram_with_options(
        &mut self,
        mel_spectrogram: &Array2<f32>,
        options: &SingingProcessingOptions,
    ) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        // Create temporary configuration based on options
        let temp_config = self.create_temp_config(options)?;
        let original_config = self.config.clone();

        // Update vocoder configuration
        self.vocoder.update_config(temp_config)?;

        // Process with singing vocoder
        let audio = self.vocoder.process_singing_voice(mel_spectrogram)?;

        // Restore original configuration
        self.vocoder.update_config(original_config)?;

        // Update statistics
        self.update_processing_stats(start_time.elapsed().as_secs_f64(), &audio)?;

        Ok(audio)
    }

    /// Process audio in real-time chunks
    pub fn process_realtime_chunk(&mut self, mel_chunk: &Array2<f32>) -> Result<Vec<f32>> {
        // Optimize for real-time processing
        let rt_config = self.create_realtime_config()?;
        let original_config = self.config.clone();

        // Update vocoder configuration for real-time
        self.vocoder.update_config(rt_config)?;

        // Process chunk
        let audio = self.vocoder.process_singing_voice(mel_chunk)?;

        // Restore original configuration
        self.vocoder.update_config(original_config)?;

        Ok(audio)
    }

    /// Create temporary configuration based on options
    fn create_temp_config(
        &self,
        options: &SingingProcessingOptions,
    ) -> Result<SingingVocoderConfig> {
        let mut config = self.config.clone();

        // Update configurations based on options
        config.pitch_stability.enable_correction = options.enable_pitch_correction;
        config.vibrato.enable_enhancement = options.enable_vibrato_enhancement;
        config.harmonic_enhancement.enable_enhancement = options.enable_harmonic_enhancement;
        config.breath_sound.enable_processing = options.enable_breath_processing;
        config.artifact_reduction.enable_reduction = options.enable_artifact_reduction;
        config.quality_metrics.enable_metrics = options.enable_quality_metrics;

        // Adjust quality based on preference
        if options.quality_preference > 0.7 {
            // High quality mode
            config.pitch_stability.correction_strength *= 1.2;
            config.vibrato.enhancement_strength *= 1.1;
            config.harmonic_enhancement.enhancement_strengths = config
                .harmonic_enhancement
                .enhancement_strengths
                .iter()
                .map(|x| x * 1.1)
                .collect();
            config.artifact_reduction.noise_reduction_strength *= 1.2;
        } else if options.quality_preference < 0.3 {
            // Speed mode
            config.pitch_stability.correction_strength *= 0.8;
            config.vibrato.enhancement_strength *= 0.9;
            config.harmonic_enhancement.enhancement_strengths = config
                .harmonic_enhancement
                .enhancement_strengths
                .iter()
                .map(|x| x * 0.9)
                .collect();
            config.artifact_reduction.noise_reduction_strength *= 0.8;
        }

        Ok(config)
    }

    /// Create real-time optimized configuration
    fn create_realtime_config(&self) -> Result<SingingVocoderConfig> {
        let mut config = self.config.clone();

        // Optimize for real-time processing
        config.pitch_stability.correction_strength *= 0.8;
        config.vibrato.enhancement_strength *= 0.9;
        config.harmonic_enhancement.enhancement_strengths = config
            .harmonic_enhancement
            .enhancement_strengths
            .iter()
            .map(|x| x * 0.9)
            .collect();
        config.artifact_reduction.noise_reduction_strength *= 0.7;
        config.artifact_reduction.temporal_artifact_reduction *= 0.8;

        // Reduce quality metrics calculation for speed
        config.quality_metrics.calculate_pitch_accuracy = false;
        config.quality_metrics.calculate_harmonic_clarity = false;
        config.quality_metrics.calculate_spectral_stability = false;

        Ok(config)
    }

    /// Update processing statistics
    fn update_processing_stats(&mut self, processing_time: f64, audio: &[f32]) -> Result<()> {
        let frame_count = audio.len() / self.config.hop_size;
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

        Ok(())
    }

    /// Get real-time performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let real_time_factor = if self.stats.processing_time_stats.avg_time_per_frame > 0.0 {
            let frame_duration_ms =
                (self.config.hop_size as f64 / self.config.sample_rate as f64) * 1000.0;
            frame_duration_ms / self.stats.processing_time_stats.avg_time_per_frame
        } else {
            0.0
        };

        PerformanceMetrics {
            frames_processed: self.stats.frames_processed,
            total_processing_time: self.stats.processing_time_stats.total_time,
            avg_time_per_frame_ms: self.stats.processing_time_stats.avg_time_per_frame,
            min_time_per_frame_ms: self.stats.processing_time_stats.min_time_per_frame,
            max_time_per_frame_ms: self.stats.processing_time_stats.max_time_per_frame,
            real_time_factor,
            is_real_time_capable: real_time_factor > 1.0,
        }
    }

    /// Get quality metrics from vocoder
    pub fn get_quality_metrics(&self) -> Result<QualityMetrics> {
        // This would typically be calculated during processing
        // For now, return default metrics
        Ok(QualityMetrics {
            pitch_accuracy: self.stats.average_pitch_accuracy,
            harmonic_clarity: self.stats.average_harmonic_clarity,
            spectral_stability: self.stats.average_spectral_stability,
            singing_quality: self.stats.average_singing_quality,
            additional_metrics: AdditionalMetrics {
                snr: 30.0,
                thd: 0.05,
                spectral_centroid: 2000.0,
                spectral_bandwidth: 1000.0,
                spectral_rolloff: 8000.0,
                zero_crossing_rate: 0.1,
            },
        })
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SingingVocoderConfig) -> Result<()> {
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
        self.stats = SingingProcessingStats::default();
    }

    /// Get configuration
    pub fn get_config(&self) -> &SingingVocoderConfig {
        &self.config
    }

    /// Get HiFi-GAN configuration
    pub fn get_hifigan_config(&self) -> &HiFiGanConfig {
        &self.hifigan_config
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &SingingProcessingStats {
        &self.stats
    }
}

/// Performance metrics for singing voice processing
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
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

impl Default for SingingProcessingOptions {
    fn default() -> Self {
        Self {
            enable_pitch_correction: true,
            enable_vibrato_enhancement: true,
            enable_harmonic_enhancement: true,
            enable_breath_processing: true,
            enable_artifact_reduction: true,
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
    fn test_singing_processor_creation() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();

        let processor = SingingProcessor::new(singing_config, hifigan_config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_mel_spectrogram_processing() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process_mel_spectrogram(&mel);
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_processing_with_options() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        let options = SingingProcessingOptions {
            enable_pitch_correction: false,
            enable_vibrato_enhancement: true,
            quality_preference: 0.8,
            ..Default::default()
        };

        let mel = Array2::ones((80, 100));
        let result = processor.process_mel_spectrogram_with_options(&mel, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_realtime_processing() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        // Create small chunk for real-time processing
        let mel_chunk = Array2::ones((80, 10));
        let result = processor.process_realtime_chunk(&mel_chunk);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_metrics() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        // Process some data to generate metrics
        let mel = Array2::ones((80, 100));
        let _ = processor.process_mel_spectrogram(&mel);

        let metrics = processor.get_performance_metrics();
        assert!(metrics.frames_processed > 0);
        assert!(metrics.total_processing_time > 0.0);
    }

    #[test]
    fn test_quality_metrics() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        let quality_metrics = processor.get_quality_metrics();
        assert!(quality_metrics.is_ok());

        let metrics = quality_metrics.unwrap();
        assert!(metrics.pitch_accuracy >= 0.0);
        assert!(metrics.harmonic_clarity >= 0.0);
        assert!(metrics.spectral_stability >= 0.0);
        assert!(metrics.singing_quality >= 0.0);
    }

    #[test]
    fn test_config_updates() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        let new_singing_config = SingingVocoderConfig {
            pitch_stability: PitchStabilityConfig {
                stability_threshold: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = processor.update_config(new_singing_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.pitch_stability.stability_threshold, 0.1);
    }

    #[test]
    fn test_stats_reset() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let mut processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        // Process some data
        let mel = Array2::ones((80, 100));
        let _ = processor.process_mel_spectrogram(&mel);

        assert!(processor.stats.frames_processed > 0);

        processor.reset_stats();
        assert_eq!(processor.stats.frames_processed, 0);
    }

    #[test]
    fn test_temp_config_creation() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        let options = SingingProcessingOptions {
            enable_pitch_correction: false,
            quality_preference: 0.9,
            ..Default::default()
        };

        let temp_config = processor.create_temp_config(&options);
        assert!(temp_config.is_ok());

        let config = temp_config.unwrap();
        assert!(!config.pitch_stability.enable_correction);
    }

    #[test]
    fn test_realtime_config_creation() {
        let singing_config = SingingVocoderConfig::default();
        let hifigan_config = HiFiGanVariant::V1.default_config();
        let processor = SingingProcessor::new(singing_config, hifigan_config).unwrap();

        let rt_config = processor.create_realtime_config();
        assert!(rt_config.is_ok());

        let config = rt_config.unwrap();
        assert!(!config.quality_metrics.calculate_pitch_accuracy);
    }

    #[test]
    fn test_processing_options_default() {
        let options = SingingProcessingOptions::default();
        assert!(options.enable_pitch_correction);
        assert!(options.enable_vibrato_enhancement);
        assert!(options.enable_harmonic_enhancement);
        assert!(options.enable_breath_processing);
        assert!(options.enable_artifact_reduction);
        assert!(options.enable_quality_metrics);
        assert!(!options.real_time_mode);
        assert_eq!(options.quality_preference, 0.7);
    }
}
