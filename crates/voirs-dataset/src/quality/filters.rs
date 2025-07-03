//! Quality filters implementation
//!
//! This module provides automatic filtering capabilities for audio datasets
//! based on quality metrics, SNR thresholds, and other criteria.

use crate::{AudioData, DatasetSample, Result};
use crate::quality::metrics::{QualityMetrics, QualityMetricsCalculator, QualityConfig};
use std::collections::HashMap;

/// Filter criteria for quality assessment
pub struct FilterCriteria {
    /// Minimum SNR in dB
    pub min_snr: Option<f32>,
    /// Maximum SNR in dB (to filter out suspiciously clean audio)
    pub max_snr: Option<f32>,
    /// Minimum duration in seconds
    pub min_duration: Option<f32>,
    /// Maximum duration in seconds
    pub max_duration: Option<f32>,
    /// Maximum clipping percentage (0.0 to 1.0)
    pub max_clipping: Option<f32>,
    /// Minimum speech activity score
    pub min_speech_activity: Option<f32>,
    /// Minimum overall quality score
    pub min_quality_score: Option<f32>,
    /// Maximum THD+N ratio
    pub max_thd_n: Option<f32>,
    /// Minimum dynamic range in dB
    pub min_dynamic_range: Option<f32>,
    /// Maximum dynamic range in dB
    pub max_dynamic_range: Option<f32>,
    /// Required sample rates (empty = accept all)
    pub required_sample_rates: Vec<u32>,
    /// Required channel counts (empty = accept all)
    pub required_channels: Vec<u32>,
    /// Custom filter functions
    pub custom_filters: Vec<Box<dyn Fn(&QualityMetrics) -> bool + Send + Sync>>,
}

impl std::fmt::Debug for FilterCriteria {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterCriteria")
            .field("min_snr", &self.min_snr)
            .field("max_snr", &self.max_snr)
            .field("min_duration", &self.min_duration)
            .field("max_duration", &self.max_duration)
            .field("max_clipping", &self.max_clipping)
            .field("min_speech_activity", &self.min_speech_activity)
            .field("min_quality_score", &self.min_quality_score)
            .field("max_thd_n", &self.max_thd_n)
            .field("min_dynamic_range", &self.min_dynamic_range)
            .field("max_dynamic_range", &self.max_dynamic_range)
            .field("required_sample_rates", &self.required_sample_rates)
            .field("required_channels", &self.required_channels)
            .field("custom_filters", &format!("{} custom filters", self.custom_filters.len()))
            .finish()
    }
}

impl Clone for FilterCriteria {
    fn clone(&self) -> Self {
        Self {
            min_snr: self.min_snr,
            max_snr: self.max_snr,
            min_duration: self.min_duration,
            max_duration: self.max_duration,
            max_clipping: self.max_clipping,
            min_speech_activity: self.min_speech_activity,
            min_quality_score: self.min_quality_score,
            max_thd_n: self.max_thd_n,
            min_dynamic_range: self.min_dynamic_range,
            max_dynamic_range: self.max_dynamic_range,
            required_sample_rates: self.required_sample_rates.clone(),
            required_channels: self.required_channels.clone(),
            custom_filters: Vec::new(), // Can't clone function pointers, so start with empty vector
        }
    }
}

impl Default for FilterCriteria {
    fn default() -> Self {
        Self {
            min_snr: Some(10.0),
            max_snr: Some(60.0),
            min_duration: Some(0.5),
            max_duration: Some(30.0),
            max_clipping: Some(0.05),
            min_speech_activity: Some(0.3),
            min_quality_score: Some(0.4),
            max_thd_n: Some(0.1),
            min_dynamic_range: Some(20.0),
            max_dynamic_range: Some(100.0),
            required_sample_rates: vec![],
            required_channels: vec![],
            custom_filters: vec![],
        }
    }
}

impl FilterCriteria {
    /// Create lenient filter criteria (accepts more samples)
    pub fn lenient() -> Self {
        Self {
            min_snr: Some(5.0),
            max_snr: Some(80.0),
            min_duration: Some(0.1),
            max_duration: Some(60.0),
            max_clipping: Some(0.1),
            min_speech_activity: Some(0.1),
            min_quality_score: Some(0.2),
            max_thd_n: Some(0.2),
            min_dynamic_range: Some(10.0),
            max_dynamic_range: Some(120.0),
            required_sample_rates: vec![],
            required_channels: vec![],
            custom_filters: vec![],
        }
    }
    
    /// Create strict filter criteria (accepts fewer, higher quality samples)
    pub fn strict() -> Self {
        Self {
            min_snr: Some(20.0),
            max_snr: Some(50.0),
            min_duration: Some(1.0),
            max_duration: Some(15.0),
            max_clipping: Some(0.01),
            min_speech_activity: Some(0.7),
            min_quality_score: Some(0.8),
            max_thd_n: Some(0.05),
            min_dynamic_range: Some(30.0),
            max_dynamic_range: Some(80.0),
            required_sample_rates: vec![],
            required_channels: vec![],
            custom_filters: vec![],
        }
    }
    
    /// Add custom filter function
    pub fn add_custom_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&QualityMetrics) -> bool + Send + Sync + 'static,
    {
        self.custom_filters.push(Box::new(filter));
        self
    }
}

/// Filter result for a single sample
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the sample passed all filters
    pub passed: bool,
    /// Reasons why the sample was rejected (empty if passed)
    pub rejection_reasons: Vec<String>,
    /// Quality metrics for the sample
    pub metrics: QualityMetrics,
}

/// Filter statistics for a batch of samples
#[derive(Debug, Clone)]
pub struct FilterStats {
    /// Total samples processed
    pub total_samples: usize,
    /// Samples that passed all filters
    pub passed_samples: usize,
    /// Samples that failed filters
    pub failed_samples: usize,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f32,
    /// Rejection reasons and their counts
    pub rejection_counts: HashMap<String, usize>,
    /// Processing time
    pub processing_time: std::time::Duration,
}

impl FilterStats {
    /// Create new filter statistics
    pub fn new() -> Self {
        Self {
            total_samples: 0,
            passed_samples: 0,
            failed_samples: 0,
            pass_rate: 0.0,
            rejection_counts: HashMap::new(),
            processing_time: std::time::Duration::from_secs(0),
        }
    }
    
    /// Add result to statistics
    pub fn add_result(&mut self, result: &FilterResult) {
        self.total_samples += 1;
        
        if result.passed {
            self.passed_samples += 1;
        } else {
            self.failed_samples += 1;
            
            // Count rejection reasons
            for reason in &result.rejection_reasons {
                *self.rejection_counts.entry(reason.clone()).or_insert(0) += 1;
            }
        }
        
        self.pass_rate = if self.total_samples > 0 {
            self.passed_samples as f32 / self.total_samples as f32
        } else {
            0.0
        };
    }
    
    /// Set processing time
    pub fn set_processing_time(&mut self, duration: std::time::Duration) {
        self.processing_time = duration;
    }
    
    /// Get top rejection reasons
    pub fn top_rejection_reasons(&self, limit: usize) -> Vec<(String, usize)> {
        let mut reasons: Vec<(String, usize)> = self.rejection_counts
            .iter()
            .map(|(reason, count)| (reason.clone(), *count))
            .collect();
        
        reasons.sort_by(|a, b| b.1.cmp(&a.1));
        reasons.truncate(limit);
        reasons
    }
}

/// Quality filter for audio samples
pub struct QualityFilter {
    criteria: FilterCriteria,
    calculator: QualityMetricsCalculator,
}

impl QualityFilter {
    /// Create new quality filter
    pub fn new(criteria: FilterCriteria, quality_config: QualityConfig) -> Self {
        Self {
            criteria,
            calculator: QualityMetricsCalculator::new(quality_config),
        }
    }
    
    /// Create filter with default criteria and quality config
    pub fn default() -> Self {
        Self::new(FilterCriteria::default(), QualityConfig::default())
    }
    
    /// Create lenient filter
    pub fn lenient() -> Self {
        Self::new(FilterCriteria::lenient(), QualityConfig::default())
    }
    
    /// Create strict filter
    pub fn strict() -> Self {
        Self::new(FilterCriteria::strict(), QualityConfig::default())
    }
    
    /// Filter a single audio sample
    pub fn filter_audio(&self, audio: &AudioData) -> Result<FilterResult> {
        let metrics = self.calculator.calculate_metrics(audio)?;
        let (passed, reasons) = self.evaluate_metrics(&metrics);
        
        Ok(FilterResult {
            passed,
            rejection_reasons: reasons,
            metrics,
        })
    }
    
    /// Filter a single dataset sample
    pub fn filter_sample(&self, sample: &DatasetSample) -> Result<FilterResult> {
        self.filter_audio(&sample.audio)
    }
    
    /// Filter multiple audio samples
    pub fn filter_batch(&self, audio_files: &[AudioData]) -> Result<(Vec<FilterResult>, FilterStats)> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(audio_files.len());
        let mut stats = FilterStats::new();
        
        for audio in audio_files {
            let result = self.filter_audio(audio)?;
            stats.add_result(&result);
            results.push(result);
        }
        
        let processing_time = start_time.elapsed();
        stats.set_processing_time(processing_time);
        
        Ok((results, stats))
    }
    
    /// Filter multiple dataset samples
    pub fn filter_samples(&self, samples: &[DatasetSample]) -> Result<(Vec<FilterResult>, FilterStats)> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(samples.len());
        let mut stats = FilterStats::new();
        
        for sample in samples {
            let result = self.filter_sample(sample)?;
            stats.add_result(&result);
            results.push(result);
        }
        
        let processing_time = start_time.elapsed();
        stats.set_processing_time(processing_time);
        
        Ok((results, stats))
    }
    
    /// Get only the samples that pass the filter
    pub fn get_passing_samples(&self, samples: &[DatasetSample]) -> Result<Vec<DatasetSample>> {
        let (results, _) = self.filter_samples(samples)?;
        
        let passing_samples = samples.iter()
            .zip(results.iter())
            .filter(|(_, result)| result.passed)
            .map(|(sample, _)| sample.clone())
            .collect();
        
        Ok(passing_samples)
    }
    
    /// Evaluate metrics against filter criteria
    fn evaluate_metrics(&self, metrics: &QualityMetrics) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();
        
        // Check SNR
        if let Some(min_snr) = self.criteria.min_snr {
            if metrics.snr < min_snr {
                reasons.push(format!("SNR too low: {:.1} < {:.1} dB", metrics.snr, min_snr));
            }
        }
        
        if let Some(max_snr) = self.criteria.max_snr {
            if metrics.snr > max_snr {
                reasons.push(format!("SNR too high: {:.1} > {:.1} dB", metrics.snr, max_snr));
            }
        }
        
        // Check duration
        if let Some(min_duration) = self.criteria.min_duration {
            if metrics.duration < min_duration {
                reasons.push(format!("Duration too short: {:.2} < {:.2} seconds", metrics.duration, min_duration));
            }
        }
        
        if let Some(max_duration) = self.criteria.max_duration {
            if metrics.duration > max_duration {
                reasons.push(format!("Duration too long: {:.2} > {:.2} seconds", metrics.duration, max_duration));
            }
        }
        
        // Check clipping
        if let Some(max_clipping) = self.criteria.max_clipping {
            if metrics.clipping_percentage > max_clipping {
                reasons.push(format!("Too much clipping: {:.1}% > {:.1}%", 
                    metrics.clipping_percentage * 100.0, max_clipping * 100.0));
            }
        }
        
        // Check speech activity
        if let Some(min_speech_activity) = self.criteria.min_speech_activity {
            if metrics.speech_activity < min_speech_activity {
                reasons.push(format!("Low speech activity: {:.2} < {:.2}", 
                    metrics.speech_activity, min_speech_activity));
            }
        }
        
        // Check quality score
        if let Some(min_quality_score) = self.criteria.min_quality_score {
            if metrics.overall_score < min_quality_score {
                reasons.push(format!("Low quality score: {:.2} < {:.2}", 
                    metrics.overall_score, min_quality_score));
            }
        }
        
        // Check THD+N
        if let Some(max_thd_n) = self.criteria.max_thd_n {
            if metrics.thd_n > max_thd_n {
                reasons.push(format!("High distortion: {:.3} > {:.3}", metrics.thd_n, max_thd_n));
            }
        }
        
        // Check dynamic range
        if let Some(min_dynamic_range) = self.criteria.min_dynamic_range {
            if metrics.dynamic_range < min_dynamic_range {
                reasons.push(format!("Low dynamic range: {:.1} < {:.1} dB", 
                    metrics.dynamic_range, min_dynamic_range));
            }
        }
        
        if let Some(max_dynamic_range) = self.criteria.max_dynamic_range {
            if metrics.dynamic_range > max_dynamic_range {
                reasons.push(format!("High dynamic range: {:.1} > {:.1} dB", 
                    metrics.dynamic_range, max_dynamic_range));
            }
        }
        
        // Check sample rate
        if !self.criteria.required_sample_rates.is_empty() {
            if !self.criteria.required_sample_rates.contains(&metrics.sample_rate) {
                reasons.push(format!("Invalid sample rate: {} Hz", metrics.sample_rate));
            }
        }
        
        // Check channels
        if !self.criteria.required_channels.is_empty() {
            if !self.criteria.required_channels.contains(&metrics.channels) {
                reasons.push(format!("Invalid channel count: {}", metrics.channels));
            }
        }
        
        // Apply custom filters
        for (i, filter_fn) in self.criteria.custom_filters.iter().enumerate() {
            if !filter_fn(metrics) {
                reasons.push(format!("Failed custom filter #{}", i + 1));
            }
        }
        
        let passed = reasons.is_empty();
        (passed, reasons)
    }
}

/// Adaptive quality filter that adjusts criteria based on dataset characteristics
pub struct AdaptiveQualityFilter {
    base_criteria: FilterCriteria,
    calculator: QualityMetricsCalculator,
    adaptation_history: Vec<QualityMetrics>,
    max_history_size: usize,
}

impl AdaptiveQualityFilter {
    /// Create new adaptive filter
    pub fn new(base_criteria: FilterCriteria, quality_config: QualityConfig) -> Self {
        Self {
            base_criteria,
            calculator: QualityMetricsCalculator::new(quality_config),
            adaptation_history: Vec::new(),
            max_history_size: 1000,
        }
    }
    
    /// Add sample to adaptation history
    pub fn add_to_history(&mut self, audio: &AudioData) -> Result<()> {
        let metrics = self.calculator.calculate_metrics(audio)?;
        self.adaptation_history.push(metrics);
        
        // Keep only recent history
        if self.adaptation_history.len() > self.max_history_size {
            self.adaptation_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Get adapted filter criteria based on historical data
    pub fn get_adapted_criteria(&self) -> FilterCriteria {
        if self.adaptation_history.is_empty() {
            return self.base_criteria.clone();
        }
        
        let mut adapted = self.base_criteria.clone();
        
        // Calculate percentiles for adaptation
        let mut snr_values: Vec<f32> = self.adaptation_history.iter().map(|m| m.snr).collect();
        let mut quality_scores: Vec<f32> = self.adaptation_history.iter().map(|m| m.overall_score).collect();
        let mut durations: Vec<f32> = self.adaptation_history.iter().map(|m| m.duration).collect();
        
        snr_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        quality_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Adapt thresholds based on data distribution
        if let Some(p10) = self.percentile(&snr_values, 10.0) {
            adapted.min_snr = Some(p10.max(5.0)); // Don't go below 5 dB
        }
        
        if let Some(p10) = self.percentile(&quality_scores, 10.0) {
            adapted.min_quality_score = Some(p10.max(0.2)); // Don't go below 0.2
        }
        
        if let Some(p5) = self.percentile(&durations, 5.0) {
            adapted.min_duration = Some(p5.max(0.1)); // Don't go below 0.1 seconds
        }
        
        if let Some(p95) = self.percentile(&durations, 95.0) {
            adapted.max_duration = Some(p95.min(60.0)); // Don't go above 60 seconds
        }
        
        adapted
    }
    
    /// Calculate percentile value
    fn percentile(&self, sorted_values: &[f32], percentile: f32) -> Option<f32> {
        if sorted_values.is_empty() {
            return None;
        }
        
        let index = (percentile / 100.0 * (sorted_values.len() - 1) as f32).round() as usize;
        sorted_values.get(index).copied()
    }
    
    /// Filter with adapted criteria
    pub fn filter_adaptive(&mut self, samples: &[DatasetSample]) -> Result<(Vec<FilterResult>, FilterStats)> {
        // First pass: add samples to history
        for sample in samples {
            self.add_to_history(&sample.audio)?;
        }
        
        // Get adapted criteria
        let adapted_criteria = self.get_adapted_criteria();
        
        // Create temporary filter with adapted criteria
        let filter = QualityFilter::new(adapted_criteria, QualityConfig::default());
        
        // Filter samples
        filter.filter_samples(samples)
    }
}

/// Multi-stage quality filter for progressive filtering
pub struct MultiStageFilter {
    stages: Vec<(String, QualityFilter)>,
}

impl MultiStageFilter {
    /// Create new multi-stage filter
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
        }
    }
    
    /// Add filter stage
    pub fn add_stage(mut self, name: String, filter: QualityFilter) -> Self {
        self.stages.push((name, filter));
        self
    }
    
    /// Add lenient pre-filter stage
    pub fn with_prefilter(self) -> Self {
        self.add_stage("prefilter".to_string(), QualityFilter::lenient())
    }
    
    /// Add strict final filter stage
    pub fn with_final_filter(self) -> Self {
        self.add_stage("final".to_string(), QualityFilter::strict())
    }
    
    /// Process samples through all filter stages
    pub fn process_stages(&self, samples: &[DatasetSample]) -> Result<(Vec<DatasetSample>, Vec<(String, FilterStats)>)> {
        let mut current_samples = samples.to_vec();
        let mut stage_stats = Vec::new();
        
        for (stage_name, filter) in &self.stages {
            let (results, stats) = filter.filter_samples(&current_samples)?;
            
            // Keep only passing samples for next stage
            current_samples = current_samples.into_iter()
                .zip(results.iter())
                .filter(|(_, result)| result.passed)
                .map(|(sample, _)| sample)
                .collect();
            
            stage_stats.push((stage_name.clone(), stats));
        }
        
        Ok((current_samples, stage_stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, DatasetSample, LanguageCode, QualityMetrics};
    
    fn create_test_audio(samples: Vec<f32>, sample_rate: u32) -> AudioData {
        AudioData::new(samples, sample_rate, 1)
    }
    
    fn create_test_sample(audio: AudioData, id: &str) -> DatasetSample {
        DatasetSample {
            id: id.to_string(),
            text: "test audio".to_string(),
            audio,
            speaker: None,
            language: LanguageCode::EnUs,
            quality: QualityMetrics {
                snr: None,
                clipping: None,
                dynamic_range: None,
                spectral_quality: None,
                overall_quality: None,
            },
            phonemes: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    #[test]
    fn test_filter_criteria_default() {
        let criteria = FilterCriteria::default();
        assert!(criteria.min_snr.is_some());
        assert!(criteria.min_quality_score.is_some());
    }
    
    #[test]
    fn test_quality_filter_creation() {
        let filter = QualityFilter::default();
        // Should not panic
    }
    
    #[test]
    fn test_filter_result_creation() {
        let audio = create_test_audio(vec![0.1, 0.2, 0.3, 0.4], 22050);
        let sample = create_test_sample(audio, "test_001");
        
        let filter = QualityFilter::lenient();
        let result = filter.filter_sample(&sample);
        assert!(result.is_ok());
    }
}