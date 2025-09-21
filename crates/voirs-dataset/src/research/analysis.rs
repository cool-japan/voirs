//! Statistical analysis tools for datasets

use crate::{DatasetError, Result as DatasetResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Sample data for statistical analysis
#[derive(Debug, Clone)]
pub struct SampleAnalysisData<'a> {
    pub duration: f64,
    pub sample_rate: u32,
    pub snr: f64,
    pub thd: f64,
    pub quality_score: f64,
    pub text: &'a str,
    pub language: &'a str,
    pub speaker: &'a str,
    pub phonemes: &'a [String],
}

/// Comprehensive dataset statistics with advanced metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_samples: usize,
    pub duration_stats: DurationStatistics,
    pub quality_stats: QualityStatistics,
    pub sample_rate_distribution: HashMap<u32, usize>,
    pub language_distribution: HashMap<String, usize>,
    pub speaker_distribution: HashMap<String, usize>,
    pub text_length_stats: TextStatistics,
}

/// Duration-related statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: Vec<(f64, f64)>, // (percentile, value)
    pub total_duration_hours: f64,
}

/// Quality-related statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStatistics {
    pub mean_snr: f64,
    pub mean_thd: f64,
    pub mean_quality_score: f64,
    pub quality_distribution: HashMap<String, usize>, // "excellent", "good", "fair", "poor"
    pub outlier_count: usize,
    pub corruption_rate: f64,
}

/// Text-related statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    pub mean_char_count: f64,
    pub mean_word_count: f64,
    pub vocabulary_size: usize,
    pub character_distribution: HashMap<char, usize>,
    pub word_frequency: HashMap<String, usize>,
    pub phoneme_coverage: HashMap<String, usize>,
}

/// Data distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    pub metric_name: String,
    pub distribution_type: DistributionType,
    pub normality_test_p_value: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub outliers: Vec<f64>,
    pub recommendations: Vec<String>,
}

/// Types of statistical distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Exponential,
    Uniform,
    Bimodal,
    Unknown,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub output_format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
    pub color_scheme: ColorScheme,
    pub include_title: bool,
    pub include_grid: bool,
}

/// Supported image formats for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    PNG,
    SVG,
    PDF,
    HTML,
}

/// Color schemes for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Grayscale,
    Custom(Vec<String>),
}

/// Comprehensive report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub title: String,
    pub summary: String,
    pub dataset_overview: DatasetOverview,
    pub statistical_analysis: Vec<DistributionAnalysis>,
    pub quality_assessment: QualityAssessment,
    pub recommendations: Vec<String>,
    pub visualizations: Vec<VisualizationMetadata>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Dataset overview section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetOverview {
    pub sample_count: usize,
    pub total_duration: f64,
    pub languages: Vec<String>,
    pub speakers: usize,
    pub quality_summary: String,
}

/// Quality assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub overall_score: f64,
    pub quality_issues: Vec<QualityIssue>,
    pub data_completeness: f64,
    pub consistency_score: f64,
}

/// Individual quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    pub affected_samples: Vec<String>,
    pub recommendation: String,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Issue categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    AudioQuality,
    TextQuality,
    Alignment,
    Metadata,
    Format,
    Corruption,
}

/// Visualization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetadata {
    pub title: String,
    pub description: String,
    pub file_path: PathBuf,
    pub visualization_type: VisualizationType,
}

/// Types of visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    Histogram,
    BoxPlot,
    ScatterPlot,
    BarChart,
    Heatmap,
    TimeSeries,
    QQPlot,
    CorrelationMatrix,
}

impl Default for DatasetStats {
    fn default() -> Self {
        Self {
            total_samples: 0,
            duration_stats: DurationStatistics {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: Vec::new(),
                total_duration_hours: 0.0,
            },
            quality_stats: QualityStatistics {
                mean_snr: 0.0,
                mean_thd: 0.0,
                mean_quality_score: 0.0,
                quality_distribution: HashMap::new(),
                outlier_count: 0,
                corruption_rate: 0.0,
            },
            sample_rate_distribution: HashMap::new(),
            language_distribution: HashMap::new(),
            speaker_distribution: HashMap::new(),
            text_length_stats: TextStatistics {
                mean_char_count: 0.0,
                mean_word_count: 0.0,
                vocabulary_size: 0,
                character_distribution: HashMap::new(),
                word_frequency: HashMap::new(),
                phoneme_coverage: HashMap::new(),
            },
        }
    }
}

/// Advanced statistical analyzer with comprehensive analysis capabilities
#[derive(Debug, Default)]
pub struct StatisticalAnalyzer {
    durations: Vec<f64>,
    snr_values: Vec<f64>,
    thd_values: Vec<f64>,
    quality_scores: Vec<f64>,
    char_counts: Vec<usize>,
    word_counts: Vec<usize>,
    sample_rates: Vec<u32>,
    languages: Vec<String>,
    speakers: Vec<String>,
    words: Vec<String>,
    phonemes: Vec<String>,
}

impl StatisticalAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample for analysis
    pub fn analyze_sample(&mut self, data: &SampleAnalysisData) {
        self.durations.push(data.duration);
        self.snr_values.push(data.snr);
        self.thd_values.push(data.thd);
        self.quality_scores.push(data.quality_score);
        self.sample_rates.push(data.sample_rate);
        self.languages.push(data.language.to_string());
        self.speakers.push(data.speaker.to_string());

        self.char_counts.push(data.text.chars().count());
        let words: Vec<&str> = data.text.split_whitespace().collect();
        self.word_counts.push(words.len());

        for word in words {
            self.words.push(word.to_lowercase());
        }

        for phoneme in data.phonemes {
            self.phonemes.push(phoneme.clone());
        }
    }

    /// Get comprehensive statistics
    pub fn get_comprehensive_stats(&self) -> DatasetStats {
        let total_samples = self.durations.len();

        if total_samples == 0 {
            return DatasetStats::default();
        }

        // Duration statistics
        let duration_stats = self.calculate_duration_statistics();

        // Quality statistics
        let quality_stats = self.calculate_quality_statistics();

        // Sample rate distribution
        let sample_rate_distribution = self.calculate_frequency_distribution(&self.sample_rates);

        // Language distribution
        let language_distribution = self.calculate_string_frequency_distribution(&self.languages);

        // Speaker distribution
        let speaker_distribution = self.calculate_string_frequency_distribution(&self.speakers);

        // Text statistics
        let text_length_stats = self.calculate_text_statistics();

        DatasetStats {
            total_samples,
            duration_stats,
            quality_stats,
            sample_rate_distribution,
            language_distribution,
            speaker_distribution,
            text_length_stats,
        }
    }

    /// Calculate duration statistics with percentiles
    fn calculate_duration_statistics(&self) -> DurationStatistics {
        if self.durations.is_empty() {
            return DurationStatistics {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: Vec::new(),
                total_duration_hours: 0.0,
            };
        }

        let mut sorted_durations = self.durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = self.durations.iter().sum::<f64>() / self.durations.len() as f64;
        let median = self.percentile(&sorted_durations, 50.0);

        let variance = self
            .durations
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.durations.len() as f64;
        let std_dev = variance.sqrt();

        let min = sorted_durations[0];
        let max = sorted_durations[sorted_durations.len() - 1];

        let percentiles = vec![
            (5.0, self.percentile(&sorted_durations, 5.0)),
            (25.0, self.percentile(&sorted_durations, 25.0)),
            (50.0, median),
            (75.0, self.percentile(&sorted_durations, 75.0)),
            (95.0, self.percentile(&sorted_durations, 95.0)),
        ];

        let total_duration_hours = self.durations.iter().sum::<f64>() / 3600.0;

        DurationStatistics {
            mean,
            median,
            std_dev,
            min,
            max,
            percentiles,
            total_duration_hours,
        }
    }

    /// Calculate quality statistics
    fn calculate_quality_statistics(&self) -> QualityStatistics {
        if self.quality_scores.is_empty() {
            return QualityStatistics {
                mean_snr: 0.0,
                mean_thd: 0.0,
                mean_quality_score: 0.0,
                quality_distribution: HashMap::new(),
                outlier_count: 0,
                corruption_rate: 0.0,
            };
        }

        let mean_snr = self.snr_values.iter().sum::<f64>() / self.snr_values.len() as f64;
        let mean_thd = self.thd_values.iter().sum::<f64>() / self.thd_values.len() as f64;
        let mean_quality_score =
            self.quality_scores.iter().sum::<f64>() / self.quality_scores.len() as f64;

        // Quality distribution
        let mut quality_distribution = HashMap::new();
        for score in &self.quality_scores {
            let category = match score {
                s if *s >= 0.8 => "excellent",
                s if *s >= 0.6 => "good",
                s if *s >= 0.4 => "fair",
                _ => "poor",
            };
            *quality_distribution
                .entry(category.to_string())
                .or_insert(0) += 1;
        }

        // Outlier detection using IQR method
        let outlier_count = self.detect_outliers(&self.quality_scores).len();

        // Corruption rate (samples with very low quality)
        let corruption_count = self.quality_scores.iter().filter(|&&x| x < 0.2).count();
        let corruption_rate = corruption_count as f64 / self.quality_scores.len() as f64;

        QualityStatistics {
            mean_snr,
            mean_thd,
            mean_quality_score,
            quality_distribution,
            outlier_count,
            corruption_rate,
        }
    }

    /// Calculate text statistics
    fn calculate_text_statistics(&self) -> TextStatistics {
        if self.char_counts.is_empty() {
            return TextStatistics {
                mean_char_count: 0.0,
                mean_word_count: 0.0,
                vocabulary_size: 0,
                character_distribution: HashMap::new(),
                word_frequency: HashMap::new(),
                phoneme_coverage: HashMap::new(),
            };
        }

        let mean_char_count =
            self.char_counts.iter().sum::<usize>() as f64 / self.char_counts.len() as f64;
        let mean_word_count =
            self.word_counts.iter().sum::<usize>() as f64 / self.word_counts.len() as f64;

        // Character distribution
        let mut character_distribution = HashMap::new();
        for word in &self.words {
            for ch in word.chars() {
                *character_distribution.entry(ch).or_insert(0) += 1;
            }
        }

        // Word frequency
        let word_frequency = self.calculate_string_frequency_distribution(&self.words);
        let vocabulary_size = word_frequency.len();

        // Phoneme coverage
        let phoneme_coverage = self.calculate_string_frequency_distribution(&self.phonemes);

        TextStatistics {
            mean_char_count,
            mean_word_count,
            vocabulary_size,
            character_distribution,
            word_frequency,
            phoneme_coverage,
        }
    }

    /// Calculate percentile value
    fn percentile(&self, sorted_data: &[f64], percentile: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Detect outliers using IQR method
    fn detect_outliers(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < 4 {
            return Vec::new();
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1 = self.percentile(&sorted_data, 25.0);
        let q3 = self.percentile(&sorted_data, 75.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        data.iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .copied()
            .collect()
    }

    /// Calculate frequency distribution for numeric data
    fn calculate_frequency_distribution<T: std::hash::Hash + Eq + Clone>(
        &self,
        data: &[T],
    ) -> HashMap<T, usize> {
        let mut distribution = HashMap::new();
        for item in data {
            *distribution.entry(item.clone()).or_insert(0) += 1;
        }
        distribution
    }

    /// Calculate frequency distribution for string data
    fn calculate_string_frequency_distribution(&self, data: &[String]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for item in data {
            *distribution.entry(item.clone()).or_insert(0) += 1;
        }
        distribution
    }

    /// Analyze data distribution type
    pub fn analyze_distribution(&self, data: &[f64], metric_name: &str) -> DistributionAnalysis {
        if data.len() < 10 {
            return DistributionAnalysis {
                metric_name: metric_name.to_string(),
                distribution_type: DistributionType::Unknown,
                normality_test_p_value: 1.0,
                skewness: 0.0,
                kurtosis: 0.0,
                outliers: Vec::new(),
                recommendations: vec!["Insufficient data for distribution analysis".to_string()],
            };
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate skewness
        let skewness = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / data.len() as f64
        } else {
            0.0
        };

        // Calculate kurtosis
        let kurtosis = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / data.len() as f64
                - 3.0
        } else {
            0.0
        };

        // Simple normality test (Shapiro-Wilk approximation)
        let normality_test_p_value = self.approximate_normality_test(data);

        // Determine distribution type
        let distribution_type =
            self.classify_distribution(skewness, kurtosis, normality_test_p_value);

        // Detect outliers
        let outliers = self.detect_outliers(data);

        // Generate recommendations
        let recommendations = self.generate_distribution_recommendations(
            &distribution_type,
            skewness,
            kurtosis,
            outliers.len(),
        );

        DistributionAnalysis {
            metric_name: metric_name.to_string(),
            distribution_type,
            normality_test_p_value,
            skewness,
            kurtosis,
            outliers,
            recommendations,
        }
    }

    /// Approximate normality test
    fn approximate_normality_test(&self, data: &[f64]) -> f64 {
        // Simplified normality test based on skewness and kurtosis
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0; // Constant data is not normal
        }

        let skewness = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / data.len() as f64;
        let kurtosis = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / data.len() as f64
            - 3.0;

        // Approximate p-value based on deviation from normal distribution characteristics
        let deviation = (skewness.abs() + kurtosis.abs()) / 2.0;
        (1.0 - deviation).clamp(0.0, 1.0)
    }

    /// Classify distribution type
    fn classify_distribution(
        &self,
        skewness: f64,
        kurtosis: f64,
        normality_p: f64,
    ) -> DistributionType {
        if normality_p > 0.05 && skewness.abs() < 0.5 && kurtosis.abs() < 0.5 {
            DistributionType::Normal
        } else if skewness > 1.0 {
            DistributionType::LogNormal
        } else if kurtosis < -1.0 {
            DistributionType::Uniform
        } else if kurtosis > 1.0 {
            DistributionType::Bimodal
        } else {
            DistributionType::Unknown
        }
    }

    /// Generate recommendations based on distribution analysis
    fn generate_distribution_recommendations(
        &self,
        dist_type: &DistributionType,
        skewness: f64,
        _kurtosis: f64,
        outlier_count: usize,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match dist_type {
            DistributionType::Normal => {
                recommendations.push(
                    "Data follows normal distribution - parametric tests appropriate".to_string(),
                );
            }
            DistributionType::LogNormal => {
                recommendations.push("Consider log transformation before analysis".to_string());
                recommendations.push("Use non-parametric tests or transform data".to_string());
            }
            DistributionType::Uniform => {
                recommendations.push(
                    "Uniform distribution detected - check for artificial constraints".to_string(),
                );
            }
            DistributionType::Bimodal => {
                recommendations
                    .push("Bimodal distribution suggests multiple populations".to_string());
                recommendations.push("Consider clustering or stratified analysis".to_string());
            }
            _ => {
                recommendations.push("Use non-parametric statistical tests".to_string());
            }
        }

        if skewness.abs() > 1.0 {
            recommendations
                .push("High skewness detected - consider data transformation".to_string());
        }

        if outlier_count > 0 {
            recommendations.push(format!(
                "Found {outlier_count} outliers - investigate for data quality issues"
            ));
        }

        recommendations
    }

    /// Generate comprehensive analysis report
    pub fn generate_report(&self, title: &str) -> AnalysisReport {
        let stats = self.get_comprehensive_stats();

        // Dataset overview
        let dataset_overview = DatasetOverview {
            sample_count: stats.total_samples,
            total_duration: stats.duration_stats.total_duration_hours,
            languages: stats.language_distribution.keys().cloned().collect(),
            speakers: stats.speaker_distribution.len(),
            quality_summary: format!(
                "Mean quality: {:.2}, {} outliers detected",
                stats.quality_stats.mean_quality_score, stats.quality_stats.outlier_count
            ),
        };

        // Statistical analysis
        let statistical_analysis = vec![
            self.analyze_distribution(&self.durations, "Duration"),
            self.analyze_distribution(&self.quality_scores, "Quality Score"),
            self.analyze_distribution(&self.snr_values, "SNR"),
        ];

        // Quality assessment
        let quality_assessment = QualityAssessment {
            overall_score: stats.quality_stats.mean_quality_score,
            quality_issues: self.identify_quality_issues(&stats),
            data_completeness: self.calculate_data_completeness(),
            consistency_score: self.calculate_consistency_score(&stats),
        };

        // Generate recommendations
        let recommendations = self.generate_dataset_recommendations(&stats, &quality_assessment);

        // Summary
        let summary = format!(
            "Dataset contains {} samples with {:.1} hours of audio across {} languages. \
            Average quality score: {:.2}. {} critical issues identified.",
            stats.total_samples,
            stats.duration_stats.total_duration_hours,
            stats.language_distribution.len(),
            stats.quality_stats.mean_quality_score,
            quality_assessment
                .quality_issues
                .iter()
                .filter(|i| matches!(i.severity, IssueSeverity::Critical))
                .count()
        );

        AnalysisReport {
            title: title.to_string(),
            summary,
            dataset_overview,
            statistical_analysis,
            quality_assessment,
            recommendations,
            visualizations: Vec::new(), // Would be populated by visualization generator
            generated_at: chrono::Utc::now(),
        }
    }

    /// Identify quality issues in the dataset
    fn identify_quality_issues(&self, stats: &DatasetStats) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check for high corruption rate
        if stats.quality_stats.corruption_rate > 0.1 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Critical,
                category: IssueCategory::AudioQuality,
                description: format!(
                    "High corruption rate: {:.1}%",
                    stats.quality_stats.corruption_rate * 100.0
                ),
                affected_samples: Vec::new(), // Would contain actual sample IDs
                recommendation: "Review and filter low-quality samples".to_string(),
            });
        }

        // Check for excessive outliers
        if stats.quality_stats.outlier_count > stats.total_samples / 20 {
            // > 5%
            issues.push(QualityIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::AudioQuality,
                description: format!(
                    "{} outliers detected ({:.1}%)",
                    stats.quality_stats.outlier_count,
                    stats.quality_stats.outlier_count as f64 / stats.total_samples as f64 * 100.0
                ),
                affected_samples: Vec::new(),
                recommendation: "Investigate outlier samples for quality issues".to_string(),
            });
        }

        // Check for low vocabulary diversity
        if stats.text_length_stats.vocabulary_size < stats.total_samples / 10 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::TextQuality,
                description: "Low vocabulary diversity detected".to_string(),
                affected_samples: Vec::new(),
                recommendation: "Consider adding more diverse text content".to_string(),
            });
        }

        // Check for imbalanced language distribution
        let max_lang_samples = stats.language_distribution.values().max().unwrap_or(&0);
        let min_lang_samples = stats.language_distribution.values().min().unwrap_or(&0);
        if stats.language_distribution.len() > 1 && *max_lang_samples > *min_lang_samples * 10 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::Metadata,
                description: "Imbalanced language distribution detected".to_string(),
                affected_samples: Vec::new(),
                recommendation: "Balance dataset across languages or stratify analysis".to_string(),
            });
        }

        issues
    }

    /// Calculate data completeness score
    fn calculate_data_completeness(&self) -> f64 {
        let total_fields = 8; // duration, snr, thd, quality, text, language, speaker, phonemes
        let mut complete_fields = 0;

        if !self.durations.is_empty() {
            complete_fields += 1;
        }
        if !self.snr_values.is_empty() {
            complete_fields += 1;
        }
        if !self.thd_values.is_empty() {
            complete_fields += 1;
        }
        if !self.quality_scores.is_empty() {
            complete_fields += 1;
        }
        if !self.char_counts.is_empty() {
            complete_fields += 1;
        }
        if !self.languages.is_empty() {
            complete_fields += 1;
        }
        if !self.speakers.is_empty() {
            complete_fields += 1;
        }
        if !self.phonemes.is_empty() {
            complete_fields += 1;
        }

        complete_fields as f64 / total_fields as f64
    }

    /// Calculate consistency score
    fn calculate_consistency_score(&self, stats: &DatasetStats) -> f64 {
        let mut consistency_score: f64 = 1.0;

        // Penalize high standard deviation in durations
        if stats.duration_stats.std_dev > stats.duration_stats.mean {
            consistency_score -= 0.2;
        }

        // Penalize high quality variance
        if !self.quality_scores.is_empty() {
            let quality_variance = self
                .quality_scores
                .iter()
                .map(|x| (x - stats.quality_stats.mean_quality_score).powi(2))
                .sum::<f64>()
                / self.quality_scores.len() as f64;
            let quality_std = quality_variance.sqrt();

            if quality_std > 0.3 {
                // High variance in quality
                consistency_score -= 0.3;
            }
        }

        // Penalize uneven sample rate distribution
        let total_samples = stats.sample_rate_distribution.values().sum::<usize>();
        let max_sr_count = stats.sample_rate_distribution.values().max().unwrap_or(&0);
        if total_samples > 0 && *max_sr_count < total_samples * 8 / 10 {
            // Less than 80% at most common rate
            consistency_score -= 0.2;
        }

        consistency_score.max(0.0)
    }

    /// Generate dataset-level recommendations
    fn generate_dataset_recommendations(
        &self,
        stats: &DatasetStats,
        quality: &QualityAssessment,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Quality-based recommendations
        if quality.overall_score < 0.6 {
            recommendations.push(
                "Consider implementing quality filtering to remove low-quality samples".to_string(),
            );
        }

        if stats.quality_stats.corruption_rate > 0.05 {
            recommendations
                .push("High corruption rate detected - review data collection process".to_string());
        }

        // Duration-based recommendations
        if stats.duration_stats.std_dev > stats.duration_stats.mean {
            recommendations.push(
                "High variance in sample durations - consider length normalization".to_string(),
            );
        }

        // Text-based recommendations
        if stats.text_length_stats.vocabulary_size < 1000 {
            recommendations
                .push("Limited vocabulary - consider expanding text diversity".to_string());
        }

        // Distribution recommendations
        if stats.language_distribution.len() == 1 {
            recommendations.push("Single language dataset - consider multilingual expansion for broader applicability".to_string());
        }

        if stats.speaker_distribution.len() < 10 {
            recommendations.push(
                "Limited speaker diversity - consider adding more speakers for robustness"
                    .to_string(),
            );
        }

        // Data completeness recommendations
        if quality.data_completeness < 0.8 {
            recommendations
                .push("Incomplete metadata detected - ensure all fields are populated".to_string());
        }

        recommendations
    }
}

/// Report generator for creating publication-ready outputs
#[derive(Debug)]
pub struct ReportGenerator {
    #[allow(dead_code)]
    config: VisualizationConfig,
}

impl ReportGenerator {
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate HTML report
    pub fn generate_html_report(
        &self,
        report: &AnalysisReport,
        output_path: &Path,
    ) -> DatasetResult<()> {
        let html_content = self.format_html_report(report);

        std::fs::write(output_path, html_content).map_err(DatasetError::IoError)?;

        Ok(())
    }

    /// Format report as HTML
    fn format_html_report(&self, report: &AnalysisReport) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
        .issue-critical {{ color: red; font-weight: bold; }}
        .issue-high {{ color: orange; font-weight: bold; }}
        .issue-medium {{ color: #ff6600; }}
        .recommendation {{ background-color: #f0f8ff; padding: 10px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{}</h1>
        <p>Generated on: {}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p>{}</p>
    </div>
    
    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="metric">Samples: {}</div>
        <div class="metric">Duration: {:.1} hours</div>
        <div class="metric">Languages: {}</div>
        <div class="metric">Speakers: {}</div>
        <div class="metric">{}</div>
    </div>
    
    <div class="section">
        <h2>Quality Assessment</h2>
        <div class="metric">Overall Score: {:.2}</div>
        <div class="metric">Data Completeness: {:.1}%</div>
        <div class="metric">Consistency Score: {:.2}</div>
        
        <h3>Quality Issues</h3>
        {}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {}
    </div>
    
    <div class="section">
        <h2>Statistical Analysis</h2>
        {}
    </div>
</body>
</html>"#,
            report.title,
            report.title,
            report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
            report.summary,
            report.dataset_overview.sample_count,
            report.dataset_overview.total_duration,
            report.dataset_overview.languages.len(),
            report.dataset_overview.speakers,
            report.dataset_overview.quality_summary,
            report.quality_assessment.overall_score,
            report.quality_assessment.data_completeness * 100.0,
            report.quality_assessment.consistency_score,
            self.format_quality_issues(&report.quality_assessment.quality_issues),
            self.format_recommendations(&report.recommendations),
            self.format_statistical_analysis(&report.statistical_analysis)
        )
    }

    /// Format quality issues for HTML
    fn format_quality_issues(&self, issues: &[QualityIssue]) -> String {
        if issues.is_empty() {
            return "<p>No significant quality issues detected.</p>".to_string();
        }

        let issues_html: Vec<String> = issues
            .iter()
            .map(|issue| {
                let class = match issue.severity {
                    IssueSeverity::Critical => "issue-critical",
                    IssueSeverity::High => "issue-high",
                    IssueSeverity::Medium => "issue-medium",
                    _ => "",
                };

                format!(
                    "<div class=\"{}\"><strong>{:?}:</strong> {} - {}</div>",
                    class, issue.severity, issue.description, issue.recommendation
                )
            })
            .collect();

        issues_html.join("")
    }

    /// Format recommendations for HTML
    fn format_recommendations(&self, recommendations: &[String]) -> String {
        if recommendations.is_empty() {
            return "<p>No specific recommendations.</p>".to_string();
        }

        let rec_html: Vec<String> = recommendations
            .iter()
            .map(|rec| format!("<div class=\"recommendation\">• {rec}</div>"))
            .collect();

        rec_html.join("")
    }

    /// Format statistical analysis for HTML
    fn format_statistical_analysis(&self, analyses: &[DistributionAnalysis]) -> String {
        let analysis_html: Vec<String> = analyses
            .iter()
            .map(|analysis| {
                format!(
                    "<h4>{}</h4>
                <p><strong>Distribution:</strong> {:?}</p>
                <p><strong>Skewness:</strong> {:.3}</p>
                <p><strong>Kurtosis:</strong> {:.3}</p>
                <p><strong>Outliers:</strong> {}</p>
                <p><strong>Recommendations:</strong> {}</p>",
                    analysis.metric_name,
                    analysis.distribution_type,
                    analysis.skewness,
                    analysis.kurtosis,
                    analysis.outliers.len(),
                    analysis.recommendations.join(", ")
                )
            })
            .collect();

        analysis_html.join("")
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_format: ImageFormat::PNG,
            width: 800,
            height: 600,
            dpi: 150,
            color_scheme: ColorScheme::Default,
            include_title: true,
            include_grid: true,
        }
    }
}
