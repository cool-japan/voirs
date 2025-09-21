//! Bandwidth Extension Module
//!
//! Extends the frequency range of audio signals to improve recognition
//! accuracy for band-limited audio sources.

use crate::RecognitionError;
use voirs_sdk::AudioBuffer;

/// Configuration for bandwidth extension
#[derive(Debug, Clone)]
pub struct BandwidthExtensionConfig {
    /// Target bandwidth in Hz
    pub target_bandwidth: f32,
    /// Extension method
    pub method: ExtensionMethod,
    /// Quality level
    pub quality: QualityLevel,
    /// Enable spectral replication
    pub spectral_replication: bool,
    /// High frequency emphasis factor
    pub hf_emphasis: f32,
}

impl Default for BandwidthExtensionConfig {
    fn default() -> Self {
        Self {
            target_bandwidth: 8000.0,
            method: ExtensionMethod::SpectralReplication,
            quality: QualityLevel::Medium,
            spectral_replication: true,
            hf_emphasis: 1.2,
        }
    }
}

/// Bandwidth extension methods
#[derive(Debug, Clone, PartialEq)]
pub enum ExtensionMethod {
    /// Spectral replication from lower frequencies
    SpectralReplication,
    /// Linear prediction-based extension
    LinearPrediction,
    /// Neural network-based extension
    Neural,
    /// Harmonic extension
    Harmonic,
}

/// Quality levels for bandwidth extension
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    /// Low quality (fast processing)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (best results)
    High,
}

/// Statistics from bandwidth extension processing
#[derive(Debug, Clone, Default)]
pub struct BandwidthExtensionStats {
    /// Original bandwidth in Hz
    pub original_bandwidth: f32,
    /// Extended bandwidth in Hz
    pub extended_bandwidth: f32,
    /// Spectral centroid shift
    pub spectral_centroid_shift: f32,
    /// Energy increase in extended range
    pub extended_energy: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

/// Bandwidth extension processor
#[derive(Debug)]
pub struct BandwidthExtensionProcessor {
    config: BandwidthExtensionConfig,
    stats: BandwidthExtensionStats,
    filter_banks: Vec<Vec<f32>>,
}

impl BandwidthExtensionProcessor {
    /// Create a new bandwidth extension processor
    pub fn new(config: BandwidthExtensionConfig) -> Result<Self, RecognitionError> {
        // Initialize filter banks for spectral replication
        let filter_banks = Self::create_filter_banks(&config);

        Ok(Self {
            config,
            stats: BandwidthExtensionStats::default(),
            filter_banks,
        })
    }

    /// Create filter banks for bandwidth extension
    fn create_filter_banks(config: &BandwidthExtensionConfig) -> Vec<Vec<f32>> {
        let num_bands = match config.quality {
            QualityLevel::Low => 4,
            QualityLevel::Medium => 8,
            QualityLevel::High => 16,
        };

        // Create simple filter banks
        (0..num_bands)
            .map(|i| {
                let center_freq = (i + 1) as f32 * 1000.0;
                // Simplified filter coefficients
                vec![
                    0.1 * (center_freq / 1000.0).sin(),
                    0.2 * (center_freq / 1000.0).cos(),
                    0.1 * (center_freq / 2000.0).sin(),
                ]
            })
            .collect()
    }

    /// Process audio to extend bandwidth
    pub fn process(&mut self, audio: &AudioBuffer) -> Result<AudioBuffer, RecognitionError> {
        let start_time = std::time::Instant::now();

        let samples = audio.samples();
        let mut extended_samples = samples.to_vec();

        // Simple bandwidth extension using spectral replication
        if self.config.spectral_replication {
            self.apply_spectral_replication(&mut extended_samples, audio.sample_rate())?;
        }

        // Apply high-frequency emphasis
        if self.config.hf_emphasis != 1.0 {
            self.apply_hf_emphasis(&mut extended_samples, audio.sample_rate())?;
        }

        // Update statistics
        self.stats.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.stats.original_bandwidth = audio.sample_rate() as f32 / 2.0;
        self.stats.extended_bandwidth = self.config.target_bandwidth;

        Ok(AudioBuffer::new(
            extended_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Apply spectral replication
    fn apply_spectral_replication(
        &self,
        samples: &mut [f32],
        sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        // Simplified spectral replication
        let nyquist = sample_rate as f32 / 2.0;
        let extension_factor = self.config.target_bandwidth / nyquist;

        if extension_factor > 1.0 {
            // Apply simple high-frequency content generation
            let len = samples.len();
            for (i, sample) in samples.iter_mut().enumerate() {
                let original_sample = *sample;
                let freq_component =
                    (i as f32 * self.config.hf_emphasis / len as f32) * extension_factor;
                *sample += 0.1 * freq_component.sin() * original_sample.abs();
            }
        }

        Ok(())
    }

    /// Apply high-frequency emphasis
    fn apply_hf_emphasis(
        &self,
        samples: &mut [f32],
        _sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        // Simple high-frequency emphasis
        let emphasis = self.config.hf_emphasis;
        let mut previous_sample = 0.0;
        for (i, sample) in samples.iter_mut().enumerate() {
            let current_sample = *sample;
            if i > 0 {
                let diff = current_sample - previous_sample;
                *sample += diff * (emphasis - 1.0) * 0.1;
            }
            previous_sample = current_sample;
        }

        Ok(())
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &BandwidthExtensionStats {
        &self.stats
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BandwidthExtensionConfig) -> Result<(), RecognitionError> {
        self.filter_banks = Self::create_filter_banks(&config);
        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &BandwidthExtensionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_extension_config_default() {
        let config = BandwidthExtensionConfig::default();
        assert!((config.target_bandwidth - 8000.0).abs() < f32::EPSILON);
        assert_eq!(config.method, ExtensionMethod::SpectralReplication);
        assert_eq!(config.quality, QualityLevel::Medium);
        assert!(config.spectral_replication);
    }

    #[test]
    fn test_bandwidth_extension_processor_creation() {
        let config = BandwidthExtensionConfig::default();
        let processor = BandwidthExtensionProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_bandwidth_extension_processing() {
        let config = BandwidthExtensionConfig::default();
        let mut processor = BandwidthExtensionProcessor::new(config).unwrap();

        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = processor.process(&audio);
        assert!(result.is_ok());

        let extended = result.unwrap();
        assert_eq!(extended.sample_rate(), audio.sample_rate());
        assert_eq!(extended.channels(), audio.channels());
        assert_eq!(extended.samples().len(), audio.samples().len());
    }

    #[test]
    fn test_extension_methods() {
        let methods = vec![
            ExtensionMethod::SpectralReplication,
            ExtensionMethod::LinearPrediction,
            ExtensionMethod::Neural,
            ExtensionMethod::Harmonic,
        ];

        for method in methods {
            // Test that extension methods are properly comparable
            assert_eq!(method.clone(), method);
        }
    }

    #[test]
    fn test_quality_levels() {
        let levels = vec![QualityLevel::Low, QualityLevel::Medium, QualityLevel::High];

        for level in levels {
            // Test that quality levels are properly comparable
            assert_eq!(level.clone(), level);
        }
    }

    #[test]
    fn test_stats_default() {
        let stats = BandwidthExtensionStats::default();
        assert!((stats.original_bandwidth - 0.0).abs() < f32::EPSILON);
        assert!((stats.extended_bandwidth - 0.0).abs() < f32::EPSILON);
        assert!((stats.spectral_centroid_shift - 0.0).abs() < f32::EPSILON);
        assert!((stats.extended_energy - 0.0).abs() < f32::EPSILON);
        assert!((stats.processing_time_ms - 0.0).abs() < f32::EPSILON);
    }
}
