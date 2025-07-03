//! Quality control and metrics for speech synthesis datasets
//!
//! This module provides quality assessment, filtering, and review tools
//! for maintaining high-quality speech synthesis datasets.

pub mod metrics;
pub mod filters;
pub mod review;

use crate::{DatasetSample, QualityMetrics, Result};
use serde::{Deserialize, Serialize};

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum SNR threshold
    pub min_snr: f32,
    /// Maximum clipping percentage
    pub max_clipping: f32,
    /// Minimum dynamic range
    pub min_dynamic_range: f32,
    /// Minimum duration in seconds
    pub min_duration: f32,
    /// Maximum duration in seconds
    pub max_duration: f32,
    /// Minimum quality score
    pub min_quality_score: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_snr: 15.0,
            max_clipping: 0.01,
            min_dynamic_range: 20.0,
            min_duration: 0.5,
            max_duration: 15.0,
            min_quality_score: 0.5,
        }
    }
}

/// Quality assessor for dataset samples
pub struct QualityAssessor {
    config: QualityConfig,
}

impl QualityAssessor {
    /// Create new quality assessor
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }
    
    /// Assess quality of a sample
    pub fn assess_sample(&self, sample: &DatasetSample) -> Result<QualityMetrics> {
        let audio = &sample.audio;
        let samples = audio.samples();
        
        // Calculate SNR
        let snr = self.calculate_snr(samples);
        
        // Calculate clipping percentage
        let clipping = self.calculate_clipping(samples);
        
        // Calculate dynamic range
        let dynamic_range = self.calculate_dynamic_range(samples);
        
        // Calculate overall quality score
        let overall_quality = self.calculate_overall_quality(snr, clipping, dynamic_range, sample.duration());
        
        Ok(QualityMetrics {
            snr: Some(snr),
            clipping: Some(clipping),
            dynamic_range: Some(dynamic_range),
            spectral_quality: None, // TODO: Implement spectral quality
            overall_quality: Some(overall_quality),
        })
    }
    
    /// Check if sample meets quality standards
    pub fn meets_quality_standards(&self, sample: &DatasetSample) -> Result<bool> {
        let metrics = self.assess_sample(sample)?;
        
        // Check SNR
        if let Some(snr) = metrics.snr {
            if snr < self.config.min_snr {
                return Ok(false);
            }
        }
        
        // Check clipping
        if let Some(clipping) = metrics.clipping {
            if clipping > self.config.max_clipping {
                return Ok(false);
            }
        }
        
        // Check dynamic range
        if let Some(dynamic_range) = metrics.dynamic_range {
            if dynamic_range < self.config.min_dynamic_range {
                return Ok(false);
            }
        }
        
        // Check duration
        let duration = sample.duration();
        if duration < self.config.min_duration || duration > self.config.max_duration {
            return Ok(false);
        }
        
        // Check overall quality
        if let Some(quality) = metrics.overall_quality {
            if quality < self.config.min_quality_score {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        
        // Simple SNR calculation (placeholder)
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().fold(0.0f32, |max, &sample| max.max(sample.abs()));
        
        if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        }
    }
    
    /// Calculate clipping percentage
    fn calculate_clipping(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        
        let clipping_threshold = 0.99;
        let clipped_samples = samples.iter()
            .filter(|&&sample| sample.abs() >= clipping_threshold)
            .count();
        
        clipped_samples as f32 / samples.len() as f32
    }
    
    /// Calculate dynamic range
    fn calculate_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        
        let peak = samples.iter().fold(0.0f32, |max, &sample| max.max(sample.abs()));
        let noise_floor = samples.iter()
            .map(|&x| x.abs())
            .fold(1.0f32, |min, sample| min.min(sample.max(1e-6)));
        
        20.0 * (peak / noise_floor).log10()
    }
    
    /// Calculate overall quality score
    fn calculate_overall_quality(&self, snr: f32, clipping: f32, dynamic_range: f32, duration: f32) -> f32 {
        let mut score = 1.0;
        
        // SNR penalty
        if snr < self.config.min_snr {
            score *= snr / self.config.min_snr;
        }
        
        // Clipping penalty
        if clipping > self.config.max_clipping {
            score *= (self.config.max_clipping / clipping).min(1.0);
        }
        
        // Dynamic range penalty
        if dynamic_range < self.config.min_dynamic_range {
            score *= dynamic_range / self.config.min_dynamic_range;
        }
        
        // Duration penalty
        if duration < self.config.min_duration {
            score *= duration / self.config.min_duration;
        } else if duration > self.config.max_duration {
            score *= self.config.max_duration / duration;
        }
        
        score.max(0.0).min(1.0)
    }
}

/// Quality report for a dataset
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Total samples assessed
    pub total_samples: usize,
    /// Samples passing quality checks
    pub passing_samples: usize,
    /// Samples failing quality checks
    pub failing_samples: usize,
    /// Average quality score
    pub average_quality: f32,
    /// Quality distribution
    pub quality_distribution: std::collections::HashMap<String, usize>,
    /// Common quality issues
    pub common_issues: Vec<String>,
}

impl QualityReport {
    /// Create new quality report
    pub fn new(total_samples: usize) -> Self {
        Self {
            total_samples,
            passing_samples: 0,
            failing_samples: 0,
            average_quality: 0.0,
            quality_distribution: std::collections::HashMap::new(),
            common_issues: Vec::new(),
        }
    }
    
    /// Get pass rate
    pub fn pass_rate(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.passing_samples as f32 / self.total_samples as f32
    }
}

/// Batch quality processor
pub struct BatchQualityProcessor {
    assessor: QualityAssessor,
}

impl BatchQualityProcessor {
    /// Create new batch processor
    pub fn new(config: QualityConfig) -> Self {
        Self {
            assessor: QualityAssessor::new(config),
        }
    }
    
    /// Process multiple samples
    pub fn process_batch(&self, samples: &[DatasetSample]) -> Result<QualityReport> {
        let mut report = QualityReport::new(samples.len());
        let mut total_quality = 0.0;
        
        for sample in samples {
            let metrics = self.assessor.assess_sample(sample)?;
            let passes = self.assessor.meets_quality_standards(sample)?;
            
            if passes {
                report.passing_samples += 1;
            } else {
                report.failing_samples += 1;
            }
            
            if let Some(quality) = metrics.overall_quality {
                total_quality += quality;
            }
        }
        
        report.average_quality = if samples.len() > 0 {
            total_quality / samples.len() as f32
        } else {
            0.0
        };
        
        Ok(report)
    }
}
