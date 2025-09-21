//! Quality metrics for spatial audio.

use crate::models::spatial::config::SpatialQualityMetricsConfig;
use crate::models::spatial::SpatialAudioOutput;
use anyhow::Result;

/// Quality metrics calculator for spatial audio
pub struct SpatialQualityMetrics {
    /// Configuration
    config: SpatialQualityMetricsConfig,
}

impl SpatialQualityMetrics {
    /// Create new quality metrics calculator
    pub fn new() -> Self {
        Self {
            config: SpatialQualityMetricsConfig::default(),
        }
    }

    /// Calculate quality metrics for spatial audio output
    pub fn calculate(&mut self, output: &SpatialAudioOutput) -> Result<f32> {
        if !self.config.enable_metrics {
            return Ok(0.8); // Default quality score
        }

        let mut total_score = 0.0;
        let mut metric_count = 0;

        if self.config.calculate_localization_accuracy {
            let localization_score = self.calculate_localization_accuracy(output)?;
            total_score += localization_score;
            metric_count += 1;
        }

        if self.config.calculate_spatial_impression {
            let spatial_impression = self.calculate_spatial_impression(output)?;
            total_score += spatial_impression;
            metric_count += 1;
        }

        if self.config.calculate_immersion_level {
            let immersion_level = self.calculate_immersion_level(output)?;
            total_score += immersion_level;
            metric_count += 1;
        }

        if self.config.calculate_binaural_quality {
            let binaural_quality = self.calculate_binaural_quality(output)?;
            total_score += binaural_quality;
            metric_count += 1;
        }

        let average_score = if metric_count > 0 {
            total_score / metric_count as f32
        } else {
            0.8
        };

        Ok(average_score.clamp(0.0, 1.0))
    }

    /// Calculate localization accuracy
    fn calculate_localization_accuracy(&self, output: &SpatialAudioOutput) -> Result<f32> {
        // Simple implementation based on channel difference
        let mut score = 0.0;
        let samples = output.left_channel.len().min(output.right_channel.len());

        for i in 0..samples {
            let left = output.left_channel[i];
            let right = output.right_channel[i];
            let difference = (left - right).abs();
            score += difference; // Higher difference suggests better localization
        }

        let normalized_score = (score / samples as f32).clamp(0.0, 1.0);
        Ok(normalized_score)
    }

    /// Calculate spatial impression
    fn calculate_spatial_impression(&self, output: &SpatialAudioOutput) -> Result<f32> {
        // Simple implementation based on correlation between channels
        let mut correlation = 0.0;
        let samples = output.left_channel.len().min(output.right_channel.len());

        for i in 0..samples {
            let left = output.left_channel[i];
            let right = output.right_channel[i];
            correlation += left * right;
        }

        let normalized_correlation = (correlation / samples as f32).abs();
        let spatial_impression = 1.0 - normalized_correlation.clamp(0.0, 1.0);
        Ok(spatial_impression)
    }

    /// Calculate immersion level
    fn calculate_immersion_level(&self, output: &SpatialAudioOutput) -> Result<f32> {
        // Simple implementation based on audio energy distribution
        let mut left_energy = 0.0;
        let mut right_energy = 0.0;

        for &sample in &output.left_channel {
            left_energy += sample * sample;
        }

        for &sample in &output.right_channel {
            right_energy += sample * sample;
        }

        let total_energy = left_energy + right_energy;
        let balance = if total_energy > 0.0 {
            1.0 - (left_energy - right_energy).abs() / total_energy
        } else {
            0.5
        };

        Ok(balance.clamp(0.0, 1.0))
    }

    /// Calculate binaural quality
    fn calculate_binaural_quality(&self, output: &SpatialAudioOutput) -> Result<f32> {
        // Simple implementation based on signal quality
        let mut left_quality = 0.0;
        let mut right_quality = 0.0;

        // Calculate RMS for quality estimation
        for &sample in &output.left_channel {
            left_quality += sample * sample;
        }

        for &sample in &output.right_channel {
            right_quality += sample * sample;
        }

        let left_rms = (left_quality / output.left_channel.len() as f32).sqrt();
        let right_rms = (right_quality / output.right_channel.len() as f32).sqrt();

        let quality = (left_rms + right_rms) / 2.0;
        Ok(quality.clamp(0.0, 1.0))
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SpatialQualityMetricsConfig) {
        self.config = config;
    }
}

impl Default for SpatialQualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spatial::{ProcessingInfo, SpatialAudioOutput, SpatialPosition};

    #[test]
    fn test_spatial_quality_metrics_creation() {
        let metrics = SpatialQualityMetrics::new();
        assert!(metrics.config.enable_metrics);
    }

    #[test]
    fn test_quality_calculation() {
        let mut metrics = SpatialQualityMetrics::new();

        let output = SpatialAudioOutput {
            left_channel: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            right_channel: vec![0.5, 0.4, 0.3, 0.2, 0.1],
            quality_score: 0.8,
            processing_info: ProcessingInfo {
                position: SpatialPosition::default(),
                reverb_level: 0.2,
                hrtf_applied: true,
                binaural_rendered: true,
            },
        };

        let result = metrics.calculate(&output);
        assert!(result.is_ok());

        let quality = result.unwrap();
        assert!((0.0..=1.0).contains(&quality));
    }

    #[test]
    fn test_metrics_disabled() {
        let config = SpatialQualityMetricsConfig {
            enable_metrics: false,
            ..Default::default()
        };

        let mut metrics = SpatialQualityMetrics::new();
        metrics.update_config(config);

        let output = SpatialAudioOutput {
            left_channel: vec![0.1, 0.2, 0.3],
            right_channel: vec![0.3, 0.2, 0.1],
            quality_score: 0.8,
            processing_info: ProcessingInfo {
                position: SpatialPosition::default(),
                reverb_level: 0.2,
                hrtf_applied: true,
                binaural_rendered: true,
            },
        };

        let result = metrics.calculate(&output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.8);
    }

    #[test]
    fn test_individual_metrics() {
        let metrics = SpatialQualityMetrics::new();

        let output = SpatialAudioOutput {
            left_channel: vec![0.1, 0.2, 0.3],
            right_channel: vec![0.3, 0.2, 0.1],
            quality_score: 0.8,
            processing_info: ProcessingInfo {
                position: SpatialPosition::default(),
                reverb_level: 0.2,
                hrtf_applied: true,
                binaural_rendered: true,
            },
        };

        let localization = metrics.calculate_localization_accuracy(&output);
        assert!(localization.is_ok());

        let spatial_impression = metrics.calculate_spatial_impression(&output);
        assert!(spatial_impression.is_ok());

        let immersion = metrics.calculate_immersion_level(&output);
        assert!(immersion.is_ok());

        let binaural = metrics.calculate_binaural_quality(&output);
        assert!(binaural.is_ok());
    }
}
