//! Advanced dataset statistics and analysis
//!
//! This module provides comprehensive statistical analysis capabilities for speech datasets:
//! - Distribution analysis (duration, text length, quality scores)
//! - Statistical summaries (mean, median, std dev, percentiles, quartiles)
//! - Cross-dataset comparison and correlation analysis
//! - Visualization-ready data structures
//! - Quality distribution analysis
//! - Speaker diversity metrics
//!
//! # Examples
//!
//! ```no_run
//! use voirs_dataset::datasets::statistics::{DatasetStatistics, StatisticsConfig};
//! use voirs_dataset::DatasetSample;
//!
//! let samples: Vec<DatasetSample> = vec![/* ... */];
//! let stats = DatasetStatistics::from_samples(&samples);
//!
//! println!("Mean duration: {:.2}s", stats.duration_stats.mean);
//! println!("Median text length: {}", stats.text_length_stats.median);
//! println!("Quality score range: {:.3} - {:.3}",
//!          stats.quality_stats.min, stats.quality_stats.max);
//! ```

use crate::{DatasetSample, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for statistics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Include percentile calculations (can be expensive for large datasets)
    pub include_percentiles: bool,
    /// Percentiles to compute (e.g., [0.25, 0.5, 0.75, 0.95, 0.99])
    pub percentiles: Vec<f64>,
    /// Include speaker-level statistics
    pub include_speaker_stats: bool,
    /// Include language-level statistics
    pub include_language_stats: bool,
    /// Include correlation analysis
    pub include_correlations: bool,
    /// Bin size for histogram generation (duration in seconds)
    pub duration_bin_size: f64,
    /// Bin size for text length histogram (characters)
    pub text_length_bin_size: usize,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            include_percentiles: true,
            percentiles: vec![0.25, 0.5, 0.75, 0.95, 0.99],
            include_speaker_stats: true,
            include_language_stats: true,
            include_correlations: true,
            duration_bin_size: 0.5,   // 0.5 second bins
            text_length_bin_size: 10, // 10 character bins
        }
    }
}

impl StatisticsConfig {
    /// Quick configuration for fast statistics (no percentiles, no correlations)
    pub fn fast() -> Self {
        Self {
            include_percentiles: false,
            percentiles: vec![],
            include_speaker_stats: false,
            include_language_stats: false,
            include_correlations: false,
            duration_bin_size: 1.0,
            text_length_bin_size: 20,
        }
    }

    /// Comprehensive configuration with detailed analysis
    pub fn comprehensive() -> Self {
        Self {
            include_percentiles: true,
            percentiles: vec![0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99],
            include_speaker_stats: true,
            include_language_stats: true,
            include_correlations: true,
            duration_bin_size: 0.25,
            text_length_bin_size: 5,
        }
    }
}

/// Statistical summary for a numeric distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    /// Number of samples
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
    /// Percentile values (if computed)
    pub percentiles: HashMap<String, f64>,
    /// Skewness of the distribution
    pub skewness: f64,
    /// Kurtosis of the distribution
    pub kurtosis: f64,
}

impl DistributionStats {
    /// Compute statistics from a vector of values
    pub fn from_values(values: &[f64], percentiles: &[f64]) -> Self {
        if values.is_empty() {
            return Self::empty();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let median = percentile(&sorted, 0.5);
        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);
        let iqr = q3 - q1;

        // Compute requested percentiles
        let mut percentile_map = HashMap::new();
        for &p in percentiles {
            let key = format!("p{:.0}", p * 100.0);
            percentile_map.insert(key, percentile(&sorted, p));
        }

        // Compute skewness
        let m3 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / count as f64;
        let skewness = if std_dev > 0.0 {
            m3 / std_dev.powi(3)
        } else {
            0.0
        };

        // Compute kurtosis
        let m4 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / count as f64;
        let kurtosis = if variance > 0.0 {
            (m4 / variance.powi(2)) - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Self {
            count,
            mean,
            median,
            std_dev,
            variance,
            min: sorted[0],
            max: sorted[count - 1],
            q1,
            q3,
            iqr,
            percentiles: percentile_map,
            skewness,
            kurtosis,
        }
    }

    /// Empty statistics (for empty datasets)
    pub fn empty() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            min: 0.0,
            max: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            percentiles: HashMap::new(),
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }

    /// Check if distribution is approximately normal (using skewness and kurtosis)
    pub fn is_approximately_normal(&self) -> bool {
        // Rule of thumb: skewness in [-1, 1] and excess kurtosis in [-1, 1]
        self.skewness.abs() < 1.0 && self.kurtosis.abs() < 1.0
    }

    /// Detect outliers using IQR method
    pub fn outlier_bounds(&self) -> (f64, f64) {
        let lower = self.q1 - 1.5 * self.iqr;
        let upper = self.q3 + 1.5 * self.iqr;
        (lower, upper)
    }
}

/// Histogram data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin edges (left edge of each bin)
    pub bins: Vec<f64>,
    /// Count of values in each bin
    pub counts: Vec<usize>,
    /// Bin width
    pub bin_width: f64,
}

impl Histogram {
    /// Create histogram from values
    pub fn from_values(values: &[f64], bin_width: f64) -> Self {
        if values.is_empty() {
            return Self {
                bins: vec![],
                counts: vec![],
                bin_width,
            };
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let num_bins = ((max_val - min_val) / bin_width).ceil() as usize + 1;
        let mut bins = Vec::with_capacity(num_bins);
        let mut counts = vec![0; num_bins];

        for i in 0..num_bins {
            bins.push(min_val + i as f64 * bin_width);
        }

        for &value in values {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            counts[bin_idx] += 1;
        }

        Self {
            bins,
            counts,
            bin_width,
        }
    }

    /// Get the mode (most frequent bin center)
    pub fn mode(&self) -> Option<f64> {
        if self.counts.is_empty() {
            return None;
        }

        let max_count_idx = self
            .counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)?
            .0;

        Some(self.bins[max_count_idx] + self.bin_width / 2.0)
    }

    /// Get total count
    pub fn total_count(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// Language-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    /// Sample count per language
    pub sample_counts: HashMap<LanguageCode, usize>,
    /// Duration statistics per language
    pub duration_stats: HashMap<LanguageCode, DistributionStats>,
    /// Text length statistics per language
    pub text_length_stats: HashMap<LanguageCode, DistributionStats>,
    /// Quality statistics per language
    pub quality_stats: HashMap<LanguageCode, DistributionStats>,
}

/// Speaker-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerStats {
    /// Number of unique speakers
    pub unique_speakers: usize,
    /// Sample count per speaker
    pub samples_per_speaker: HashMap<String, usize>,
    /// Duration statistics per speaker
    pub duration_per_speaker: HashMap<String, DistributionStats>,
    /// Gender distribution (if available)
    pub gender_distribution: HashMap<String, usize>,
    /// Age distribution (if available)
    pub age_distribution: DistributionStats,
}

/// Correlation analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    /// Correlation between duration and text length
    pub duration_text_length: f64,
    /// Correlation between duration and quality
    pub duration_quality: f64,
    /// Correlation between text length and quality
    pub text_length_quality: f64,
    /// Correlation between SNR and overall quality
    pub snr_quality: f64,
}

/// Comprehensive dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Configuration used for statistics
    pub config: StatisticsConfig,
    /// Total number of samples
    pub total_samples: usize,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Audio duration statistics
    pub duration_stats: DistributionStats,
    /// Duration histogram
    pub duration_histogram: Histogram,
    /// Text length statistics (in characters)
    pub text_length_stats: DistributionStats,
    /// Text length histogram
    pub text_length_histogram: Histogram,
    /// Overall quality score statistics
    pub quality_stats: DistributionStats,
    /// SNR statistics
    pub snr_stats: DistributionStats,
    /// Clipping statistics
    pub clipping_stats: DistributionStats,
    /// Dynamic range statistics
    pub dynamic_range_stats: DistributionStats,
    /// Spectral quality statistics
    pub spectral_quality_stats: DistributionStats,
    /// Language-specific statistics
    pub language_stats: Option<LanguageStats>,
    /// Speaker-specific statistics
    pub speaker_stats: Option<SpeakerStats>,
    /// Correlation analysis
    pub correlations: Option<CorrelationAnalysis>,
}

impl DatasetStatistics {
    /// Compute statistics from a collection of samples
    pub fn from_samples(samples: &[DatasetSample]) -> Self {
        Self::from_samples_with_config(samples, StatisticsConfig::default())
    }

    /// Compute statistics with custom configuration
    pub fn from_samples_with_config(samples: &[DatasetSample], config: StatisticsConfig) -> Self {
        let total_samples = samples.len();

        if total_samples == 0 {
            return Self::empty(config);
        }

        // Extract basic metrics
        let durations: Vec<f64> = samples.iter().map(|s| s.audio.duration() as f64).collect();

        let text_lengths: Vec<f64> = samples.iter().map(|s| s.text.len() as f64).collect();

        let quality_scores: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.quality.overall_quality.map(|v| v as f64))
            .collect();

        let snr_values: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.quality.snr.map(|v| v as f64))
            .collect();

        let clipping_values: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.quality.clipping.map(|v| v as f64))
            .collect();

        let dynamic_range_values: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.quality.dynamic_range.map(|v| v as f64))
            .collect();

        let spectral_quality_values: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.quality.spectral_quality.map(|v| v as f64))
            .collect();

        let total_duration: f64 = durations.iter().sum();

        // Compute distribution statistics
        let empty_percentiles: Vec<f64> = Vec::new();
        let percentiles = if config.include_percentiles {
            &config.percentiles
        } else {
            &empty_percentiles
        };

        let duration_stats = DistributionStats::from_values(&durations, percentiles);
        let text_length_stats = DistributionStats::from_values(&text_lengths, percentiles);
        let quality_stats = DistributionStats::from_values(&quality_scores, percentiles);
        let snr_stats = DistributionStats::from_values(&snr_values, percentiles);
        let clipping_stats = DistributionStats::from_values(&clipping_values, percentiles);
        let dynamic_range_stats =
            DistributionStats::from_values(&dynamic_range_values, percentiles);
        let spectral_quality_stats =
            DistributionStats::from_values(&spectral_quality_values, percentiles);

        // Create histograms
        let duration_histogram = Histogram::from_values(&durations, config.duration_bin_size);
        let text_length_histogram =
            Histogram::from_values(&text_lengths, config.text_length_bin_size as f64);

        // Language-specific statistics
        let language_stats = if config.include_language_stats {
            Some(compute_language_stats(samples, percentiles))
        } else {
            None
        };

        // Speaker-specific statistics
        let speaker_stats = if config.include_speaker_stats {
            Some(compute_speaker_stats(samples, percentiles))
        } else {
            None
        };

        // Correlation analysis
        let correlations = if config.include_correlations {
            Some(compute_correlations(samples))
        } else {
            None
        };

        Self {
            config,
            total_samples,
            total_duration,
            duration_stats,
            duration_histogram,
            text_length_stats,
            text_length_histogram,
            quality_stats,
            snr_stats,
            clipping_stats,
            dynamic_range_stats,
            spectral_quality_stats,
            language_stats,
            speaker_stats,
            correlations,
        }
    }

    /// Create empty statistics
    pub fn empty(config: StatisticsConfig) -> Self {
        Self {
            config,
            total_samples: 0,
            total_duration: 0.0,
            duration_stats: DistributionStats::empty(),
            duration_histogram: Histogram {
                bins: vec![],
                counts: vec![],
                bin_width: 0.5,
            },
            text_length_stats: DistributionStats::empty(),
            text_length_histogram: Histogram {
                bins: vec![],
                counts: vec![],
                bin_width: 10.0,
            },
            quality_stats: DistributionStats::empty(),
            snr_stats: DistributionStats::empty(),
            clipping_stats: DistributionStats::empty(),
            dynamic_range_stats: DistributionStats::empty(),
            spectral_quality_stats: DistributionStats::empty(),
            language_stats: None,
            speaker_stats: None,
            correlations: None,
        }
    }

    /// Print a human-readable summary
    pub fn print_summary(&self) {
        println!("=== Dataset Statistics Summary ===");
        println!("Total samples: {}", self.total_samples);
        println!(
            "Total duration: {:.2} hours ({:.2} seconds)",
            self.total_duration / 3600.0,
            self.total_duration
        );

        println!("\n--- Duration Statistics ---");
        print_distribution_summary("Duration", &self.duration_stats, "s");

        println!("\n--- Text Length Statistics ---");
        print_distribution_summary("Text Length", &self.text_length_stats, "chars");

        println!("\n--- Quality Metrics ---");
        print_distribution_summary("Overall Quality", &self.quality_stats, "");
        print_distribution_summary("SNR", &self.snr_stats, "dB");
        print_distribution_summary("Clipping", &self.clipping_stats, "");
        print_distribution_summary("Dynamic Range", &self.dynamic_range_stats, "dB");
        print_distribution_summary("Spectral Quality", &self.spectral_quality_stats, "");

        if let Some(ref lang_stats) = self.language_stats {
            println!("\n--- Language Distribution ---");
            for (lang, count) in &lang_stats.sample_counts {
                let percentage = (*count as f64 / self.total_samples as f64) * 100.0;
                println!(
                    "  {}: {} samples ({:.1}%)",
                    lang.as_str(),
                    count,
                    percentage
                );
            }
        }

        if let Some(ref speaker_stats) = self.speaker_stats {
            println!("\n--- Speaker Statistics ---");
            println!("Unique speakers: {}", speaker_stats.unique_speakers);
            println!(
                "Avg samples per speaker: {:.1}",
                self.total_samples as f64 / speaker_stats.unique_speakers as f64
            );

            if !speaker_stats.gender_distribution.is_empty() {
                println!("\nGender distribution:");
                for (gender, count) in &speaker_stats.gender_distribution {
                    println!("  {}: {} speakers", gender, count);
                }
            }
        }

        if let Some(ref corr) = self.correlations {
            println!("\n--- Correlation Analysis ---");
            println!("Duration ↔ Text Length: {:.3}", corr.duration_text_length);
            println!("Duration ↔ Quality: {:.3}", corr.duration_quality);
            println!("Text Length ↔ Quality: {:.3}", corr.text_length_quality);
            println!("SNR ↔ Quality: {:.3}", corr.snr_quality);
        }
    }

    /// Export statistics to JSON
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Import statistics from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Compute language-specific statistics
fn compute_language_stats(samples: &[DatasetSample], percentiles: &[f64]) -> LanguageStats {
    let mut sample_counts: HashMap<LanguageCode, usize> = HashMap::new();
    let mut lang_durations: HashMap<LanguageCode, Vec<f64>> = HashMap::new();
    let mut lang_text_lengths: HashMap<LanguageCode, Vec<f64>> = HashMap::new();
    let mut lang_quality: HashMap<LanguageCode, Vec<f64>> = HashMap::new();

    for sample in samples {
        *sample_counts.entry(sample.language).or_insert(0) += 1;

        lang_durations
            .entry(sample.language)
            .or_default()
            .push(sample.audio.duration() as f64);

        lang_text_lengths
            .entry(sample.language)
            .or_default()
            .push(sample.text.len() as f64);

        if let Some(quality) = sample.quality.overall_quality {
            lang_quality
                .entry(sample.language)
                .or_default()
                .push(quality as f64);
        }
    }

    let duration_stats = lang_durations
        .into_iter()
        .map(|(lang, values)| (lang, DistributionStats::from_values(&values, percentiles)))
        .collect();

    let text_length_stats = lang_text_lengths
        .into_iter()
        .map(|(lang, values)| (lang, DistributionStats::from_values(&values, percentiles)))
        .collect();

    let quality_stats = lang_quality
        .into_iter()
        .map(|(lang, values)| (lang, DistributionStats::from_values(&values, percentiles)))
        .collect();

    LanguageStats {
        sample_counts,
        duration_stats,
        text_length_stats,
        quality_stats,
    }
}

/// Compute speaker-specific statistics
fn compute_speaker_stats(samples: &[DatasetSample], percentiles: &[f64]) -> SpeakerStats {
    let mut samples_per_speaker: HashMap<String, usize> = HashMap::new();
    let mut duration_per_speaker_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut gender_distribution: HashMap<String, usize> = HashMap::new();
    let mut ages: Vec<f64> = Vec::new();

    for sample in samples {
        if let Some(ref speaker) = sample.speaker {
            *samples_per_speaker.entry(speaker.id.clone()).or_insert(0) += 1;

            duration_per_speaker_data
                .entry(speaker.id.clone())
                .or_default()
                .push(sample.audio.duration() as f64);

            if let Some(ref gender) = speaker.gender {
                *gender_distribution.entry(gender.clone()).or_insert(0) += 1;
            }

            if let Some(age) = speaker.age {
                ages.push(age as f64);
            }
        }
    }

    let unique_speakers = samples_per_speaker.len();

    let duration_per_speaker = duration_per_speaker_data
        .into_iter()
        .map(|(speaker, values)| {
            (
                speaker,
                DistributionStats::from_values(&values, percentiles),
            )
        })
        .collect();

    let age_distribution = DistributionStats::from_values(&ages, percentiles);

    SpeakerStats {
        unique_speakers,
        samples_per_speaker,
        duration_per_speaker,
        gender_distribution,
        age_distribution,
    }
}

/// Compute correlation between two variables
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute correlation analysis
fn compute_correlations(samples: &[DatasetSample]) -> CorrelationAnalysis {
    let durations: Vec<f64> = samples.iter().map(|s| s.audio.duration() as f64).collect();

    let text_lengths: Vec<f64> = samples.iter().map(|s| s.text.len() as f64).collect();

    // Filter to only samples with quality metrics
    let samples_with_quality: Vec<_> = samples
        .iter()
        .filter(|s| s.quality.overall_quality.is_some())
        .collect();

    let durations_q: Vec<f64> = samples_with_quality
        .iter()
        .map(|s| s.audio.duration() as f64)
        .collect();

    let text_lengths_q: Vec<f64> = samples_with_quality
        .iter()
        .map(|s| s.text.len() as f64)
        .collect();

    let quality_scores: Vec<f64> = samples_with_quality
        .iter()
        .filter_map(|s| s.quality.overall_quality.map(|v| v as f64))
        .collect();

    let snr_values: Vec<f64> = samples_with_quality
        .iter()
        .filter_map(|s| s.quality.snr.map(|v| v as f64))
        .collect();

    let snr_quality: Vec<f64> = samples_with_quality
        .iter()
        .filter(|s| s.quality.snr.is_some() && s.quality.overall_quality.is_some())
        .filter_map(|s| s.quality.overall_quality.map(|v| v as f64))
        .collect();

    let snr_for_quality: Vec<f64> = samples_with_quality
        .iter()
        .filter(|s| s.quality.snr.is_some() && s.quality.overall_quality.is_some())
        .filter_map(|s| s.quality.snr.map(|v| v as f64))
        .collect();

    CorrelationAnalysis {
        duration_text_length: pearson_correlation(&durations, &text_lengths),
        duration_quality: pearson_correlation(&durations_q, &quality_scores),
        text_length_quality: pearson_correlation(&text_lengths_q, &quality_scores),
        snr_quality: pearson_correlation(&snr_for_quality, &snr_quality),
    }
}

/// Compute percentile from sorted values
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    if p <= 0.0 {
        return sorted_values[0];
    }

    if p >= 1.0 {
        return sorted_values[sorted_values.len() - 1];
    }

    let index = p * (sorted_values.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
}

/// Print distribution summary helper
fn print_distribution_summary(name: &str, stats: &DistributionStats, unit: &str) {
    if stats.count == 0 {
        println!("{}: No data available", name);
        return;
    }

    let unit_str = if unit.is_empty() {
        String::new()
    } else {
        format!(" {}", unit)
    };

    println!(
        "{}: count={}, mean={:.2}{}, median={:.2}{}, std={:.2}{}",
        name, stats.count, stats.mean, unit_str, stats.median, unit_str, stats.std_dev, unit_str
    );
    println!(
        "  Range: [{:.2}, {:.2}]{}, IQR: {:.2}{}",
        stats.min, stats.max, unit_str, stats.iqr, unit_str
    );

    if !stats.percentiles.is_empty() {
        print!("  Percentiles:");
        let mut sorted_percentiles: Vec<_> = stats.percentiles.iter().collect();
        sorted_percentiles.sort_by_key(|(k, _)| k.to_string());
        for (key, value) in sorted_percentiles {
            print!(" {}={:.2}{}", key, value, unit_str);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, QualityMetrics, SpeakerInfo};

    fn create_test_samples(count: usize) -> Vec<DatasetSample> {
        let mut samples = Vec::new();
        for i in 0..count {
            let duration = 1.0 + (i as f32 * 0.1);
            let text = format!("Sample text number {} with varying lengths", i);
            let audio = AudioData::silence(duration, 22050, 1);

            let quality = QualityMetrics {
                overall_quality: Some((0.8 + (i as f64 * 0.01)) as f32),
                snr: Some((20.0 + (i as f64 * 0.5)) as f32),
                clipping: Some((0.01 + (i as f64 * 0.001)) as f32),
                dynamic_range: Some((40.0 + (i as f64 * 0.2)) as f32),
                spectral_quality: Some((0.85 + (i as f64 * 0.01)) as f32),
            };

            let speaker = if i % 3 == 0 {
                Some(SpeakerInfo {
                    id: format!("speaker_{}", i % 5),
                    name: Some(format!("Speaker {}", i % 5)),
                    gender: Some(if i % 2 == 0 { "male" } else { "female" }.to_string()),
                    age: Some((25 + (i % 3) * 10) as u32),
                    accent: None,
                    metadata: Default::default(),
                })
            } else {
                None
            };

            let sample = DatasetSample::new(
                format!("sample_{:03}", i),
                text,
                audio,
                if i % 2 == 0 {
                    LanguageCode::EnUs
                } else {
                    LanguageCode::Ja
                },
            )
            .with_quality(quality);

            let sample = if let Some(spk) = speaker {
                sample.with_speaker(spk)
            } else {
                sample
            };

            samples.push(sample);
        }
        samples
    }

    #[test]
    fn test_basic_statistics() {
        let samples = create_test_samples(100);
        let stats = DatasetStatistics::from_samples(&samples);

        assert_eq!(stats.total_samples, 100);
        assert!(stats.total_duration > 0.0);
        assert_eq!(stats.duration_stats.count, 100);
        assert_eq!(stats.text_length_stats.count, 100);
        assert!(stats.quality_stats.count > 0);
    }

    #[test]
    fn test_distribution_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let percentiles = vec![0.25, 0.5, 0.75];
        let stats = DistributionStats::from_values(&values, &percentiles);

        assert_eq!(stats.count, 10);
        assert_eq!(stats.mean, 5.5);
        assert_eq!(stats.median, 5.5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_histogram() {
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 5.5, 6.0];
        let hist = Histogram::from_values(&values, 1.0);

        assert!(!hist.bins.is_empty());
        assert_eq!(hist.bins.len(), hist.counts.len());
        assert_eq!(hist.total_count(), 8);
    }

    #[test]
    fn test_empty_statistics() {
        let samples: Vec<DatasetSample> = vec![];
        let stats = DatasetStatistics::from_samples(&samples);

        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.total_duration, 0.0);
        assert_eq!(stats.duration_stats.count, 0);
    }

    #[test]
    fn test_language_statistics() {
        let samples = create_test_samples(100);
        let config = StatisticsConfig {
            include_language_stats: true,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        assert!(stats.language_stats.is_some());
        let lang_stats = stats.language_stats.unwrap();
        assert!(!lang_stats.sample_counts.is_empty());
    }

    #[test]
    fn test_speaker_statistics() {
        let samples = create_test_samples(100);
        let config = StatisticsConfig {
            include_speaker_stats: true,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        assert!(stats.speaker_stats.is_some());
        let speaker_stats = stats.speaker_stats.unwrap();
        assert!(speaker_stats.unique_speakers > 0);
    }

    #[test]
    fn test_correlation_analysis() {
        let samples = create_test_samples(100);
        let config = StatisticsConfig {
            include_correlations: true,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        assert!(stats.correlations.is_some());
        let corr = stats.correlations.unwrap();
        // Correlation values should be between -1 and 1
        assert!(corr.duration_text_length >= -1.0 && corr.duration_text_length <= 1.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        // Perfect positive correlation
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
        assert_eq!(percentile(&values, 0.5), 3.0);
    }

    #[test]
    fn test_fast_config() {
        let samples = create_test_samples(50);
        let config = StatisticsConfig::fast();
        let stats = DatasetStatistics::from_samples_with_config(&samples, config.clone());

        assert!(!config.include_percentiles);
        assert!(stats.language_stats.is_none());
        assert!(stats.speaker_stats.is_none());
        assert!(stats.correlations.is_none());
    }

    #[test]
    fn test_comprehensive_config() {
        let samples = create_test_samples(50);
        let config = StatisticsConfig::comprehensive();
        let stats = DatasetStatistics::from_samples_with_config(&samples, config.clone());

        assert!(config.include_percentiles);
        assert!(stats.language_stats.is_some());
        assert!(stats.speaker_stats.is_some());
        assert!(stats.correlations.is_some());
    }

    #[test]
    fn test_outlier_detection() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
        let stats = DistributionStats::from_values(&values, &[]);
        let (lower, upper) = stats.outlier_bounds();

        assert!(100.0 > upper); // Should detect 100.0 as outlier
    }

    #[test]
    fn test_normality_check() {
        // Approximately normal distribution
        let normal = vec![
            5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1, 4.8, 5.2,
        ];
        let stats = DistributionStats::from_values(&normal, &[]);
        // This should be approximately normal (low skewness and kurtosis)
        // The actual values depend on the distribution
        assert!(stats.skewness.abs() < 5.0); // Relaxed threshold for test data
    }

    #[test]
    fn test_json_serialization() {
        let samples = create_test_samples(10);
        let stats = DatasetStatistics::from_samples(&samples);

        let json = stats.to_json().unwrap();
        assert!(!json.is_empty());

        let deserialized = DatasetStatistics::from_json(&json).unwrap();
        assert_eq!(deserialized.total_samples, stats.total_samples);
    }

    #[test]
    fn test_histogram_mode() {
        let values = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 5.3, 5.4]; // Mode should be around 5.0
        let hist = Histogram::from_values(&values, 1.0);
        let mode = hist.mode().unwrap();
        assert!(mode > 4.0 && mode < 6.0);
    }
}
