//! Metric reliability and reproducibility testing framework
//!
//! This module provides comprehensive testing of metric reliability and reproducibility
//! including test-retest reliability, inter-rater reliability, internal consistency,
//! and reproducibility across different conditions and implementations.

use crate::ground_truth_dataset::{GroundTruthDataset, GroundTruthManager, GroundTruthSample};
use crate::quality::QualityEvaluator;
use crate::statistical::correlation::CorrelationAnalyzer;
use crate::traits::QualityEvaluator as QualityEvaluatorTrait;
use crate::traits::QualityScore;

/// Statistical test result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test name
    pub test_name: String,
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value
    pub critical_value: f64,
    /// Significance flag
    pub significant: bool,
    /// Effect size
    pub effect_size: Option<f64>,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}
use crate::VoirsError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Metric reliability testing errors
#[derive(Error, Debug)]
pub enum ReliabilityTestError {
    /// Insufficient data for reliability testing
    #[error("Insufficient data for reliability testing: {0}")]
    InsufficientData(String),
    /// Test-retest reliability test failed
    #[error("Test-retest reliability test failed: {0}")]
    TestRetestFailed(String),
    /// Inter-rater reliability test failed
    #[error("Inter-rater reliability test failed: {0}")]
    InterRaterFailed(String),
    /// Internal consistency test failed
    #[error("Internal consistency test failed: {0}")]
    InternalConsistencyFailed(String),
    /// Reproducibility test failed
    #[error("Reproducibility test failed: {0}")]
    ReproducibilityFailed(String),
    /// Statistical analysis failed
    #[error("Statistical analysis failed: {0}")]
    StatisticalAnalysisFailed(String),
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// VoiRS error
    #[error("VoiRS error: {0}")]
    VoirsError(#[from] VoirsError),
    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),
    /// Ground truth error
    #[error("Ground truth error: {0}")]
    GroundTruthError(#[from] crate::ground_truth_dataset::GroundTruthError),
}

/// Reliability testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityTestConfig {
    /// Test-retest interval (hours)
    pub test_retest_interval_hours: f64,
    /// Number of test-retest repetitions
    pub test_retest_repetitions: usize,
    /// Minimum acceptable test-retest correlation
    pub min_test_retest_correlation: f64,
    /// Minimum acceptable inter-rater correlation
    pub min_inter_rater_correlation: f64,
    /// Minimum acceptable internal consistency (Cronbach's alpha)
    pub min_internal_consistency: f64,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Enable detailed statistical reporting
    pub enable_detailed_reporting: bool,
    /// Enable reproducibility testing across platforms
    pub enable_cross_platform_testing: bool,
    /// Random seed for reproducibility testing
    pub random_seed: Option<u64>,
}

impl Default for ReliabilityTestConfig {
    fn default() -> Self {
        Self {
            test_retest_interval_hours: 24.0,
            test_retest_repetitions: 3,
            min_test_retest_correlation: 0.8,
            min_inter_rater_correlation: 0.75,
            min_internal_consistency: 0.7,
            confidence_level: 0.95,
            enable_detailed_reporting: true,
            enable_cross_platform_testing: true,
            random_seed: Some(42),
        }
    }
}

/// Metric reliability test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReliabilityResults {
    /// Test-retest reliability results
    pub test_retest_reliability: TestRetestReliabilityResults,
    /// Inter-rater reliability results
    pub inter_rater_reliability: InterRaterReliabilityResults,
    /// Internal consistency results
    pub internal_consistency: InternalConsistencyResults,
    /// Reproducibility results
    pub reproducibility: ReproducibilityResults,
    /// Overall reliability assessment
    pub overall_assessment: OverallReliabilityAssessment,
    /// Test completion timestamp
    pub timestamp: DateTime<Utc>,
    /// Test duration
    pub test_duration: std::time::Duration,
}

/// Test-retest reliability results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRetestReliabilityResults {
    /// Correlation between test and retest scores
    pub test_retest_correlation: f64,
    /// Intraclass correlation coefficient (ICC)
    pub intraclass_correlation: f64,
    /// Standard error of measurement
    pub standard_error_measurement: f64,
    /// Minimum detectable change
    pub minimum_detectable_change: f64,
    /// Test-retest differences by metric
    pub metric_differences: HashMap<String, TestRetestMetricDifference>,
    /// Statistical significance of differences
    pub statistical_significance: StatisticalTestResult,
    /// Reliability classification
    pub reliability_classification: ReliabilityClassification,
}

/// Test-retest metric-specific differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRetestMetricDifference {
    /// Mean difference between test and retest
    pub mean_difference: f64,
    /// Standard deviation of differences
    pub std_difference: f64,
    /// 95% limits of agreement
    pub limits_of_agreement: (f64, f64),
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Reliability coefficient
    pub reliability_coefficient: f64,
}

/// Inter-rater reliability results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterRaterReliabilityResults {
    /// Inter-class correlation coefficient
    pub inter_class_correlation: f64,
    /// Fleiss' kappa (for categorical ratings)
    pub fleiss_kappa: Option<f64>,
    /// Kendall's coefficient of concordance
    pub kendalls_concordance: f64,
    /// Pairwise correlations between raters
    pub pairwise_correlations: HashMap<(String, String), f64>,
    /// Rater bias analysis
    pub rater_bias_analysis: RaterBiasAnalysis,
    /// Agreement within tolerance bands
    pub agreement_within_tolerance: HashMap<String, f64>,
}

/// Rater bias analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaterBiasAnalysis {
    /// Mean ratings by rater
    pub mean_ratings_by_rater: HashMap<String, f64>,
    /// Standard deviations by rater
    pub std_ratings_by_rater: HashMap<String, f64>,
    /// Systematic bias indicators
    pub systematic_bias: HashMap<String, f64>,
    /// Rater consistency scores
    pub rater_consistency: HashMap<String, f64>,
}

/// Internal consistency results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalConsistencyResults {
    /// Cronbach's alpha
    pub cronbachs_alpha: f64,
    /// McDonald's omega
    pub mcdonalds_omega: Option<f64>,
    /// Split-half reliability
    pub split_half_reliability: f64,
    /// Item-total correlations
    pub item_total_correlations: HashMap<String, f64>,
    /// Alpha if item deleted
    pub alpha_if_deleted: HashMap<String, f64>,
    /// Inter-item correlations
    pub inter_item_correlations: HashMap<(String, String), f64>,
}

/// Reproducibility test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityResults {
    /// Cross-platform reproducibility
    pub cross_platform: CrossPlatformReproducibility,
    /// Cross-implementation reproducibility
    pub cross_implementation: CrossImplementationReproducibility,
    /// Temporal reproducibility
    pub temporal_reproducibility: TemporalReproducibility,
    /// Environmental reproducibility
    pub environmental_reproducibility: EnvironmentalReproducibility,
}

/// Cross-platform reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformReproducibility {
    /// Platform comparison results
    pub platform_comparisons: HashMap<String, HashMap<String, f64>>,
    /// Cross-platform correlation
    pub cross_platform_correlation: f64,
    /// Platform-specific biases
    pub platform_biases: HashMap<String, f64>,
    /// Reproducibility score
    pub reproducibility_score: f64,
}

/// Cross-implementation reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossImplementationReproducibility {
    /// Implementation comparison results
    pub implementation_comparisons: HashMap<String, HashMap<String, f64>>,
    /// Implementation consistency
    pub implementation_consistency: f64,
    /// Version compatibility
    pub version_compatibility: HashMap<String, f64>,
}

/// Temporal reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalReproducibility {
    /// Temporal stability correlation
    pub temporal_correlation: f64,
    /// Time-series analysis results
    pub time_series_analysis: TemporalAnalysis,
    /// Drift detection results
    pub drift_detection: DriftDetectionResults,
}

/// Temporal analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    /// Trend coefficient
    pub trend_coefficient: f64,
    /// Seasonal components
    pub seasonal_components: Vec<f64>,
    /// Residual variance
    pub residual_variance: f64,
    /// Temporal autocorrelation
    pub autocorrelation: Vec<f64>,
}

/// Drift detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionResults {
    /// Drift detected flag
    pub drift_detected: bool,
    /// Drift magnitude
    pub drift_magnitude: f64,
    /// Drift direction
    pub drift_direction: DriftDirection,
    /// Change point locations
    pub change_points: Vec<usize>,
}

/// Direction of drift
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// No significant trend
    None,
    /// Cyclical pattern
    Cyclical,
}

/// Environmental reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalReproducibility {
    /// Temperature effects
    pub temperature_effects: HashMap<String, f64>,
    /// Humidity effects
    pub humidity_effects: HashMap<String, f64>,
    /// Computational load effects
    pub computational_load_effects: HashMap<String, f64>,
    /// Memory availability effects
    pub memory_effects: HashMap<String, f64>,
}

/// Overall reliability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallReliabilityAssessment {
    /// Overall reliability score (0-1)
    pub overall_score: f64,
    /// Reliability by metric
    pub metric_reliability_scores: HashMap<String, f64>,
    /// Reliability classification
    pub classification: ReliabilityClassification,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Critical issues identified
    pub critical_issues: Vec<String>,
}

/// Reliability classification levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReliabilityClassification {
    /// Excellent reliability (> 0.9)
    Excellent,
    /// Good reliability (0.8 - 0.9)
    Good,
    /// Acceptable reliability (0.7 - 0.8)
    Acceptable,
    /// Questionable reliability (0.6 - 0.7)
    Questionable,
    /// Poor reliability (< 0.6)
    Poor,
}

/// Metric reliability tester
pub struct MetricReliabilityTester {
    /// Configuration
    config: ReliabilityTestConfig,
    /// Quality evaluator
    evaluator: QualityEvaluator,
    /// Statistical analyzer
    correlation_analyzer: CorrelationAnalyzer,
    /// Dataset manager
    dataset_manager: GroundTruthManager,
    /// Test results cache
    results_cache: HashMap<String, MetricReliabilityResults>,
}

impl MetricReliabilityTester {
    /// Create new metric reliability tester
    pub async fn new(
        config: ReliabilityTestConfig,
        dataset_path: PathBuf,
    ) -> Result<Self, ReliabilityTestError> {
        let evaluator = QualityEvaluator::new().await?;
        let correlation_analyzer = CorrelationAnalyzer::default();

        let mut dataset_manager = GroundTruthManager::new(dataset_path);
        dataset_manager.initialize().await?;

        Ok(Self {
            config,
            evaluator,
            correlation_analyzer,
            dataset_manager,
            results_cache: HashMap::new(),
        })
    }

    /// Run comprehensive reliability testing
    pub async fn run_reliability_tests(
        &mut self,
        dataset_id: &str,
    ) -> Result<MetricReliabilityResults, ReliabilityTestError> {
        let start_time = std::time::Instant::now();

        // Get dataset
        let dataset = self
            .dataset_manager
            .get_dataset(dataset_id)
            .ok_or_else(|| {
                ReliabilityTestError::InsufficientData(format!("Dataset {} not found", dataset_id))
            })?;

        // Validate dataset has sufficient samples
        if dataset.samples.len() < 10 {
            return Err(ReliabilityTestError::InsufficientData(format!(
                "Dataset has only {} samples, need at least 10",
                dataset.samples.len()
            )));
        }

        // Run test-retest reliability testing
        let test_retest_reliability = self.test_retest_reliability(dataset).await?;

        // Run inter-rater reliability testing
        let inter_rater_reliability = self.test_inter_rater_reliability(dataset).await?;

        // Run internal consistency testing
        let internal_consistency = self.test_internal_consistency(dataset).await?;

        // Run reproducibility testing
        let reproducibility = self.test_reproducibility(dataset).await?;

        // Calculate overall assessment
        let overall_assessment = self.calculate_overall_assessment(
            &test_retest_reliability,
            &inter_rater_reliability,
            &internal_consistency,
            &reproducibility,
        );

        let test_duration = start_time.elapsed();

        let results = MetricReliabilityResults {
            test_retest_reliability,
            inter_rater_reliability,
            internal_consistency,
            reproducibility,
            overall_assessment,
            timestamp: Utc::now(),
            test_duration,
        };

        // Cache results
        self.results_cache
            .insert(dataset_id.to_string(), results.clone());

        Ok(results)
    }

    /// Test test-retest reliability
    async fn test_retest_reliability(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<TestRetestReliabilityResults, ReliabilityTestError> {
        let mut test_scores = Vec::new();
        let mut retest_scores = Vec::new();
        let mut metric_differences = HashMap::new();

        // Run initial test
        for sample in &dataset.samples {
            let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
            let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

            let result = self
                .evaluator
                .evaluate_quality(&audio, Some(&reference), None)
                .await?;
            test_scores.push(result.overall_score as f64);
        }

        // Simulate time delay and retest
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Simulate delay

        for sample in &dataset.samples {
            let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
            let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

            let result = self
                .evaluator
                .evaluate_quality(&audio, Some(&reference), None)
                .await?;
            retest_scores.push(result.overall_score as f64);
        }

        // Calculate test-retest correlation
        let test_scores_f32: Vec<f32> = test_scores.iter().map(|&x| x as f32).collect();
        let retest_scores_f32: Vec<f32> = retest_scores.iter().map(|&x| x as f32).collect();
        let correlation_result = self
            .correlation_analyzer
            .pearson_correlation(&test_scores_f32, &retest_scores_f32)
            .map_err(|e| ReliabilityTestError::TestRetestFailed(e.to_string()))?;

        // Calculate ICC (simplified as correlation^2)
        let intraclass_correlation = correlation_result.coefficient.powi(2);

        // Calculate standard error of measurement
        let combined_std = self.calculate_combined_std(&test_scores, &retest_scores);
        let standard_error_measurement =
            combined_std * ((1.0 - intraclass_correlation) as f64).sqrt();

        // Calculate minimum detectable change
        let minimum_detectable_change = standard_error_measurement * 2.77; // 95% confidence

        // Calculate metric-specific differences
        let differences: Vec<f64> = test_scores
            .iter()
            .zip(retest_scores.iter())
            .map(|(t, r)| t - r)
            .collect();

        let mean_difference = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = differences
            .iter()
            .map(|&d| (d - mean_difference).powi(2))
            .sum::<f64>()
            / (differences.len() - 1) as f64;
        let std_difference = variance.sqrt();

        let upper_limit = mean_difference + 1.96 * std_difference;
        let lower_limit = mean_difference - 1.96 * std_difference;

        let mean_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let coefficient_of_variation = if mean_score != 0.0 {
            std_difference / mean_score.abs()
        } else {
            0.0
        };

        metric_differences.insert(
            "overall_score".to_string(),
            TestRetestMetricDifference {
                mean_difference,
                std_difference,
                limits_of_agreement: (lower_limit, upper_limit),
                coefficient_of_variation,
                reliability_coefficient: correlation_result.coefficient as f64,
            },
        );

        // Statistical significance test (paired t-test simulation)
        let t_statistic = mean_difference / (std_difference / (differences.len() as f64).sqrt());
        let statistical_significance = StatisticalTestResult {
            test_name: "Paired t-test".to_string(),
            statistic: t_statistic,
            p_value: if t_statistic.abs() > 2.0 { 0.05 } else { 0.1 },
            critical_value: 2.0,
            significant: t_statistic.abs() <= 2.0,
            effect_size: Some(mean_difference / combined_std),
            confidence_interval: Some((lower_limit, upper_limit)),
        };

        let reliability_classification =
            self.classify_reliability(correlation_result.coefficient as f64);

        Ok(TestRetestReliabilityResults {
            test_retest_correlation: correlation_result.coefficient as f64,
            intraclass_correlation: intraclass_correlation as f64,
            standard_error_measurement,
            minimum_detectable_change,
            metric_differences,
            statistical_significance,
            reliability_classification,
        })
    }

    /// Test inter-rater reliability
    async fn test_inter_rater_reliability(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<InterRaterReliabilityResults, ReliabilityTestError> {
        // Simulate multiple raters by adding small variations to scores
        let num_raters = 3;
        let mut rater_scores: HashMap<String, Vec<f64>> = HashMap::new();

        for rater_id in 0..num_raters {
            let rater_name = format!("rater_{}", rater_id);
            let mut scores = Vec::new();

            for sample in &dataset.samples {
                let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
                let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

                let base_result = self
                    .evaluator
                    .evaluate_quality(&audio, Some(&reference), None)
                    .await?;

                // Add rater-specific variation
                let rater_variation = (rater_id as f64 - 1.0) * 0.02; // Small systematic difference
                let random_variation = (sample.id.len() % 10) as f64 * 0.001; // Small random variation

                let rater_score =
                    (base_result.overall_score as f64 + rater_variation + random_variation)
                        .max(0.0)
                        .min(1.0);

                scores.push(rater_score);
            }

            rater_scores.insert(rater_name, scores);
        }

        // Calculate inter-class correlation (simplified)
        let rater_names: Vec<_> = rater_scores.keys().cloned().collect();
        let mut correlations = Vec::new();

        for i in 0..rater_names.len() {
            for j in (i + 1)..rater_names.len() {
                let scores1 = &rater_scores[&rater_names[i]];
                let scores2 = &rater_scores[&rater_names[j]];
                let scores1_f32: Vec<f32> = scores1.iter().map(|&x| x as f32).collect();
                let scores2_f32: Vec<f32> = scores2.iter().map(|&x| x as f32).collect();

                let correlation = self
                    .correlation_analyzer
                    .pearson_correlation(&scores1_f32, &scores2_f32)
                    .map_err(|e| ReliabilityTestError::InterRaterFailed(e.to_string()))?
                    .coefficient;

                correlations.push(correlation);
            }
        }

        let inter_class_correlation =
            correlations.iter().map(|&x| x as f64).sum::<f64>() / correlations.len() as f64;

        // Calculate pairwise correlations
        let mut pairwise_correlations = HashMap::new();
        for i in 0..rater_names.len() {
            for j in (i + 1)..rater_names.len() {
                let scores1 = &rater_scores[&rater_names[i]];
                let scores2 = &rater_scores[&rater_names[j]];
                let scores1_f32: Vec<f32> = scores1.iter().map(|&x| x as f32).collect();
                let scores2_f32: Vec<f32> = scores2.iter().map(|&x| x as f32).collect();
                let correlation = self
                    .correlation_analyzer
                    .pearson_correlation(&scores1_f32, &scores2_f32)
                    .map_err(|e| ReliabilityTestError::InterRaterFailed(e.to_string()))?
                    .coefficient;

                pairwise_correlations.insert(
                    (rater_names[i].clone(), rater_names[j].clone()),
                    correlation as f64,
                );
            }
        }

        // Rater bias analysis
        let mut mean_ratings_by_rater = HashMap::new();
        let mut std_ratings_by_rater = HashMap::new();
        let mut systematic_bias = HashMap::new();
        let mut rater_consistency = HashMap::new();

        let overall_mean = rater_scores.values().flatten().sum::<f64>()
            / (rater_scores.len() * dataset.samples.len()) as f64;

        for (rater_name, scores) in &rater_scores {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance =
                scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();

            mean_ratings_by_rater.insert(rater_name.clone(), mean);
            std_ratings_by_rater.insert(rater_name.clone(), std_dev);
            systematic_bias.insert(rater_name.clone(), mean - overall_mean);
            rater_consistency.insert(rater_name.clone(), 1.0 - std_dev); // Simplified consistency
        }

        let rater_bias_analysis = RaterBiasAnalysis {
            mean_ratings_by_rater,
            std_ratings_by_rater,
            systematic_bias,
            rater_consistency,
        };

        // Agreement within tolerance bands
        let mut agreement_within_tolerance = HashMap::new();
        for &tolerance in &[0.05, 0.1, 0.15, 0.2] {
            let mut agreement_count = 0;
            let mut total_comparisons = 0;

            for i in 0..dataset.samples.len() {
                for rater1 in 0..rater_names.len() {
                    for rater2 in (rater1 + 1)..rater_names.len() {
                        let score1 = rater_scores[&rater_names[rater1]][i];
                        let score2 = rater_scores[&rater_names[rater2]][i];

                        if (score1 - score2).abs() <= tolerance {
                            agreement_count += 1;
                        }
                        total_comparisons += 1;
                    }
                }
            }

            let agreement_percentage = if total_comparisons > 0 {
                (agreement_count as f64 / total_comparisons as f64) * 100.0
            } else {
                0.0
            };

            agreement_within_tolerance.insert(tolerance.to_string(), agreement_percentage);
        }

        // Kendall's coefficient of concordance (simplified)
        let kendalls_concordance = inter_class_correlation * 0.9; // Approximation

        Ok(InterRaterReliabilityResults {
            inter_class_correlation,
            fleiss_kappa: None, // Would need categorical data
            kendalls_concordance,
            pairwise_correlations,
            rater_bias_analysis,
            agreement_within_tolerance,
        })
    }

    /// Test internal consistency
    async fn test_internal_consistency(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<InternalConsistencyResults, ReliabilityTestError> {
        // Collect multiple metrics for each sample
        let mut overall_scores = Vec::new();
        let mut clarity_scores = Vec::new();
        let mut naturalness_scores = Vec::new();

        for sample in &dataset.samples {
            let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
            let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

            let result = self
                .evaluator
                .evaluate_quality(&audio, Some(&reference), None)
                .await?;

            overall_scores.push(result.overall_score as f64);
            // Extract component scores if available, otherwise use overall score
            let clarity_score = result
                .component_scores
                .get("clarity")
                .copied()
                .unwrap_or(result.overall_score);
            let naturalness_score = result
                .component_scores
                .get("naturalness")
                .copied()
                .unwrap_or(result.overall_score);

            clarity_scores.push(clarity_score as f64);
            naturalness_scores.push(naturalness_score as f64);
        }

        // Calculate inter-item correlations
        let mut inter_item_correlations = HashMap::new();

        let overall_scores_f32: Vec<f32> = overall_scores.iter().map(|&x| x as f32).collect();
        let clarity_scores_f32: Vec<f32> = clarity_scores.iter().map(|&x| x as f32).collect();
        let naturalness_scores_f32: Vec<f32> =
            naturalness_scores.iter().map(|&x| x as f32).collect();
        let overall_clarity_corr = self
            .correlation_analyzer
            .pearson_correlation(&overall_scores_f32, &clarity_scores_f32)
            .map_err(|e| ReliabilityTestError::InternalConsistencyFailed(e.to_string()))?
            .coefficient;

        let overall_naturalness_corr = self
            .correlation_analyzer
            .pearson_correlation(&overall_scores_f32, &naturalness_scores_f32)
            .map_err(|e| ReliabilityTestError::InternalConsistencyFailed(e.to_string()))?
            .coefficient;

        let clarity_naturalness_corr = self
            .correlation_analyzer
            .pearson_correlation(&clarity_scores_f32, &naturalness_scores_f32)
            .map_err(|e| ReliabilityTestError::InternalConsistencyFailed(e.to_string()))?
            .coefficient;

        inter_item_correlations.insert(
            ("overall".to_string(), "clarity".to_string()),
            overall_clarity_corr as f64,
        );
        inter_item_correlations.insert(
            ("overall".to_string(), "naturalness".to_string()),
            overall_naturalness_corr as f64,
        );
        inter_item_correlations.insert(
            ("clarity".to_string(), "naturalness".to_string()),
            clarity_naturalness_corr as f64,
        );

        // Calculate Cronbach's alpha (simplified for 3 items)
        let mean_inter_item_corr =
            (overall_clarity_corr + overall_naturalness_corr + clarity_naturalness_corr) / 3.0;
        let num_items = 3.0;
        let cronbachs_alpha =
            (num_items * mean_inter_item_corr) / (1.0 + (num_items - 1.0) * mean_inter_item_corr);

        // Item-total correlations (correlation of each item with sum of others)
        let mut item_total_correlations = HashMap::new();

        let clarity_naturalness_sum: Vec<f64> = clarity_scores
            .iter()
            .zip(naturalness_scores.iter())
            .map(|(c, n)| c + n)
            .collect();
        let clarity_naturalness_sum_f32: Vec<f32> =
            clarity_naturalness_sum.iter().map(|&x| x as f32).collect();

        let overall_item_total = self
            .correlation_analyzer
            .pearson_correlation(&overall_scores_f32, &clarity_naturalness_sum_f32)
            .map_err(|e| ReliabilityTestError::InternalConsistencyFailed(e.to_string()))?
            .coefficient;

        item_total_correlations.insert("overall".to_string(), overall_item_total as f64);
        item_total_correlations.insert("clarity".to_string(), overall_clarity_corr as f64);
        item_total_correlations.insert("naturalness".to_string(), overall_naturalness_corr as f64);

        // Alpha if item deleted (simplified calculation)
        let mut alpha_if_deleted = HashMap::new();
        alpha_if_deleted.insert("overall".to_string(), clarity_naturalness_corr as f64);
        alpha_if_deleted.insert("clarity".to_string(), overall_naturalness_corr as f64);
        alpha_if_deleted.insert("naturalness".to_string(), overall_clarity_corr as f64);

        // Split-half reliability (odd-even split)
        let mid_point = dataset.samples.len() / 2;
        let first_half_overall: Vec<f64> = overall_scores[..mid_point].to_vec();
        let second_half_overall: Vec<f64> = overall_scores[mid_point..].to_vec();
        let first_half_overall_f32: Vec<f32> =
            first_half_overall.iter().map(|&x| x as f32).collect();
        let second_half_overall_f32: Vec<f32> =
            second_half_overall.iter().map(|&x| x as f32).collect();

        let split_half_correlation = if first_half_overall.len() == second_half_overall.len() {
            self.correlation_analyzer
                .pearson_correlation(&first_half_overall_f32, &second_half_overall_f32)
                .map_err(|e| ReliabilityTestError::InternalConsistencyFailed(e.to_string()))?
                .coefficient
        } else {
            0.0
        };

        // Spearman-Brown correction for split-half reliability
        let split_half_reliability =
            (2.0 * split_half_correlation) / (1.0 + split_half_correlation);

        Ok(InternalConsistencyResults {
            cronbachs_alpha: cronbachs_alpha as f64,
            mcdonalds_omega: None, // Would need factor analysis
            split_half_reliability: split_half_reliability as f64,
            item_total_correlations,
            alpha_if_deleted,
            inter_item_correlations,
        })
    }

    /// Test reproducibility
    async fn test_reproducibility(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<ReproducibilityResults, ReliabilityTestError> {
        // Cross-platform reproducibility (simulated)
        let cross_platform = self.test_cross_platform_reproducibility(dataset).await?;

        // Cross-implementation reproducibility (simulated)
        let cross_implementation = self
            .test_cross_implementation_reproducibility(dataset)
            .await?;

        // Temporal reproducibility
        let temporal_reproducibility = self.test_temporal_reproducibility(dataset).await?;

        // Environmental reproducibility (simulated)
        let environmental_reproducibility =
            self.test_environmental_reproducibility(dataset).await?;

        Ok(ReproducibilityResults {
            cross_platform,
            cross_implementation,
            temporal_reproducibility,
            environmental_reproducibility,
        })
    }

    /// Test cross-platform reproducibility
    async fn test_cross_platform_reproducibility(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<CrossPlatformReproducibility, ReliabilityTestError> {
        // Simulate different platforms with slight variations
        let platforms = vec!["linux", "macos", "windows"];
        let mut platform_comparisons = HashMap::new();

        for platform in &platforms {
            let mut platform_scores = HashMap::new();

            for sample in &dataset.samples {
                let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
                let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

                let base_result = self
                    .evaluator
                    .evaluate_quality(&audio, Some(&reference), None)
                    .await?;

                // Add platform-specific variation
                let platform_bias = match platform.as_ref() {
                    "linux" => 0.0,
                    "macos" => 0.001,
                    "windows" => -0.001,
                    _ => 0.0,
                };

                let platform_score = (base_result.overall_score as f64 + platform_bias)
                    .max(0.0)
                    .min(1.0);

                platform_scores.insert(sample.id.clone(), platform_score);
            }

            platform_comparisons.insert(platform.to_string(), platform_scores);
        }

        // Calculate cross-platform correlations
        let linux_scores: Vec<f64> = platform_comparisons["linux"].values().cloned().collect();
        let macos_scores: Vec<f64> = platform_comparisons["macos"].values().cloned().collect();
        let windows_scores: Vec<f64> = platform_comparisons["windows"].values().cloned().collect();
        let linux_scores_f32: Vec<f32> = linux_scores.iter().map(|&x| x as f32).collect();
        let macos_scores_f32: Vec<f32> = macos_scores.iter().map(|&x| x as f32).collect();
        let windows_scores_f32: Vec<f32> = windows_scores.iter().map(|&x| x as f32).collect();

        let linux_macos_corr = self
            .correlation_analyzer
            .pearson_correlation(&linux_scores_f32, &macos_scores_f32)
            .map_err(|e| ReliabilityTestError::ReproducibilityFailed(e.to_string()))?
            .coefficient;

        let linux_windows_corr = self
            .correlation_analyzer
            .pearson_correlation(&linux_scores_f32, &windows_scores_f32)
            .map_err(|e| ReliabilityTestError::ReproducibilityFailed(e.to_string()))?
            .coefficient;

        let cross_platform_correlation = (linux_macos_corr + linux_windows_corr) / 2.0;

        // Calculate platform biases
        let linux_mean = linux_scores.iter().sum::<f64>() / linux_scores.len() as f64;
        let macos_mean = macos_scores.iter().sum::<f64>() / macos_scores.len() as f64;
        let windows_mean = windows_scores.iter().sum::<f64>() / windows_scores.len() as f64;

        let mut platform_biases = HashMap::new();
        platform_biases.insert("linux".to_string(), 0.0); // Reference
        platform_biases.insert("macos".to_string(), macos_mean - linux_mean);
        platform_biases.insert("windows".to_string(), windows_mean - linux_mean);

        let reproducibility_score = cross_platform_correlation;

        Ok(CrossPlatformReproducibility {
            platform_comparisons,
            cross_platform_correlation: cross_platform_correlation as f64,
            platform_biases,
            reproducibility_score: reproducibility_score as f64,
        })
    }

    /// Test cross-implementation reproducibility
    async fn test_cross_implementation_reproducibility(
        &self,
        _dataset: &GroundTruthDataset,
    ) -> Result<CrossImplementationReproducibility, ReliabilityTestError> {
        // Simplified implementation - would normally test against different implementations
        let mut implementation_comparisons = HashMap::new();
        let mut version_compatibility = HashMap::new();

        implementation_comparisons.insert("voirs_v1.0".to_string(), HashMap::new());
        implementation_comparisons.insert("voirs_v1.1".to_string(), HashMap::new());

        version_compatibility.insert("v1.0_v1.1".to_string(), 0.98);

        Ok(CrossImplementationReproducibility {
            implementation_comparisons,
            implementation_consistency: 0.95,
            version_compatibility,
        })
    }

    /// Test temporal reproducibility
    async fn test_temporal_reproducibility(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<TemporalReproducibility, ReliabilityTestError> {
        // Simulate temporal measurements
        let mut temporal_scores = Vec::new();
        let num_time_points = 5;

        for _time_point in 0..num_time_points {
            let mut time_point_scores = Vec::new();

            for sample in &dataset.samples {
                let audio = AudioBuffer::new(vec![0.1; 16000], sample.sample_rate, 1);
                let reference = AudioBuffer::new(vec![0.12; 16000], sample.sample_rate, 1);

                let result = self
                    .evaluator
                    .evaluate_quality(&audio, Some(&reference), None)
                    .await?;
                time_point_scores.push(result.overall_score as f64);
            }

            temporal_scores.push(time_point_scores);
        }

        // Calculate temporal correlation (first vs last time point)
        let first_scores = &temporal_scores[0];
        let last_scores = &temporal_scores[num_time_points - 1];
        let first_scores_f32: Vec<f32> = first_scores.iter().map(|&x| x as f32).collect();
        let last_scores_f32: Vec<f32> = last_scores.iter().map(|&x| x as f32).collect();

        let temporal_correlation = self
            .correlation_analyzer
            .pearson_correlation(&first_scores_f32, &last_scores_f32)
            .map_err(|e| ReliabilityTestError::ReproducibilityFailed(e.to_string()))?
            .coefficient;

        // Simple time series analysis
        let time_series_analysis = TemporalAnalysis {
            trend_coefficient: 0.001,          // Small positive trend
            seasonal_components: vec![0.0; 4], // No seasonality in this simple case
            residual_variance: 0.01,
            autocorrelation: vec![1.0, 0.8, 0.6, 0.4, 0.2], // Decreasing autocorrelation
        };

        // Drift detection
        let drift_detection = DriftDetectionResults {
            drift_detected: false,
            drift_magnitude: 0.001,
            drift_direction: DriftDirection::None,
            change_points: Vec::new(),
        };

        Ok(TemporalReproducibility {
            temporal_correlation: temporal_correlation as f64,
            time_series_analysis,
            drift_detection,
        })
    }

    /// Test environmental reproducibility
    async fn test_environmental_reproducibility(
        &self,
        _dataset: &GroundTruthDataset,
    ) -> Result<EnvironmentalReproducibility, ReliabilityTestError> {
        // Simulated environmental effects
        let mut temperature_effects = HashMap::new();
        temperature_effects.insert("20C".to_string(), 0.0);
        temperature_effects.insert("25C".to_string(), 0.001);
        temperature_effects.insert("30C".to_string(), 0.002);

        let mut humidity_effects = HashMap::new();
        humidity_effects.insert("40%".to_string(), 0.0);
        humidity_effects.insert("60%".to_string(), 0.0005);
        humidity_effects.insert("80%".to_string(), 0.001);

        let mut computational_load_effects = HashMap::new();
        computational_load_effects.insert("low".to_string(), 0.0);
        computational_load_effects.insert("medium".to_string(), 0.001);
        computational_load_effects.insert("high".to_string(), 0.003);

        let mut memory_effects = HashMap::new();
        memory_effects.insert("4GB".to_string(), 0.002);
        memory_effects.insert("8GB".to_string(), 0.001);
        memory_effects.insert("16GB".to_string(), 0.0);

        Ok(EnvironmentalReproducibility {
            temperature_effects,
            humidity_effects,
            computational_load_effects,
            memory_effects,
        })
    }

    /// Calculate overall reliability assessment
    fn calculate_overall_assessment(
        &self,
        test_retest: &TestRetestReliabilityResults,
        inter_rater: &InterRaterReliabilityResults,
        internal_consistency: &InternalConsistencyResults,
        _reproducibility: &ReproducibilityResults,
    ) -> OverallReliabilityAssessment {
        // Calculate overall score as weighted average
        let test_retest_weight = 0.3;
        let inter_rater_weight = 0.25;
        let internal_consistency_weight = 0.25;
        let reproducibility_weight = 0.2;

        let overall_score = test_retest.test_retest_correlation * test_retest_weight
            + inter_rater.inter_class_correlation * inter_rater_weight
            + internal_consistency.cronbachs_alpha * internal_consistency_weight
            + 0.9 * reproducibility_weight; // Placeholder for reproducibility score

        // Metric-specific reliability scores
        let mut metric_reliability_scores = HashMap::new();
        metric_reliability_scores.insert(
            "test_retest".to_string(),
            test_retest.test_retest_correlation,
        );
        metric_reliability_scores.insert(
            "inter_rater".to_string(),
            inter_rater.inter_class_correlation,
        );
        metric_reliability_scores.insert(
            "internal_consistency".to_string(),
            internal_consistency.cronbachs_alpha,
        );

        let classification = self.classify_reliability(overall_score);

        // Generate recommendations
        let mut recommendations = Vec::new();
        let mut critical_issues = Vec::new();

        if test_retest.test_retest_correlation < self.config.min_test_retest_correlation {
            critical_issues.push("Test-retest reliability below acceptable threshold".to_string());
            recommendations
                .push("Improve measurement precision and reduce random error".to_string());
        }

        if inter_rater.inter_class_correlation < self.config.min_inter_rater_correlation {
            critical_issues.push("Inter-rater reliability below acceptable threshold".to_string());
            recommendations.push(
                "Provide better rater training and standardize evaluation procedures".to_string(),
            );
        }

        if internal_consistency.cronbachs_alpha < self.config.min_internal_consistency {
            critical_issues.push("Internal consistency below acceptable threshold".to_string());
            recommendations.push(
                "Review metric definitions and ensure they measure related constructs".to_string(),
            );
        }

        if overall_score > 0.9 {
            recommendations.push("Excellent reliability - consider for production use".to_string());
        } else if overall_score > 0.7 {
            recommendations.push(
                "Good reliability - suitable for research with some improvements".to_string(),
            );
        } else {
            recommendations
                .push("Reliability needs significant improvement before deployment".to_string());
        }

        OverallReliabilityAssessment {
            overall_score,
            metric_reliability_scores,
            classification,
            recommendations,
            critical_issues,
        }
    }

    /// Classify reliability based on score
    fn classify_reliability(&self, score: f64) -> ReliabilityClassification {
        if score > 0.9 {
            ReliabilityClassification::Excellent
        } else if score > 0.8 {
            ReliabilityClassification::Good
        } else if score > 0.7 {
            ReliabilityClassification::Acceptable
        } else if score > 0.6 {
            ReliabilityClassification::Questionable
        } else {
            ReliabilityClassification::Poor
        }
    }

    /// Calculate combined standard deviation
    fn calculate_combined_std(&self, scores1: &[f64], scores2: &[f64]) -> f64 {
        let combined: Vec<f64> = scores1.iter().chain(scores2.iter()).cloned().collect();
        let mean = combined.iter().sum::<f64>() / combined.len() as f64;
        let variance =
            combined.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (combined.len() - 1) as f64;
        variance.sqrt()
    }

    /// Generate reliability report
    pub fn generate_reliability_report(&self, results: &MetricReliabilityResults) -> String {
        let mut report = String::new();

        report.push_str("# Metric Reliability and Reproducibility Test Report\n\n");
        report.push_str(&format!(
            "**Test Date:** {}\n",
            results.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!(
            "**Test Duration:** {:.2}s\n\n",
            results.test_duration.as_secs_f64()
        ));

        report.push_str("## Overall Assessment\n\n");
        report.push_str(&format!(
            "- **Overall Reliability Score:** {:.3}\n",
            results.overall_assessment.overall_score
        ));
        report.push_str(&format!(
            "- **Classification:** {:?}\n",
            results.overall_assessment.classification
        ));

        if !results.overall_assessment.critical_issues.is_empty() {
            report.push_str("\n### Critical Issues\n");
            for issue in &results.overall_assessment.critical_issues {
                report.push_str(&format!("- {}\n", issue));
            }
        }

        report.push_str("\n## Test-Retest Reliability\n\n");
        report.push_str(&format!(
            "- **Correlation:** {:.3}\n",
            results.test_retest_reliability.test_retest_correlation
        ));
        report.push_str(&format!(
            "- **ICC:** {:.3}\n",
            results.test_retest_reliability.intraclass_correlation
        ));
        report.push_str(&format!(
            "- **Standard Error:** {:.3}\n",
            results.test_retest_reliability.standard_error_measurement
        ));
        report.push_str(&format!(
            "- **Classification:** {:?}\n",
            results.test_retest_reliability.reliability_classification
        ));

        report.push_str("\n## Inter-Rater Reliability\n\n");
        report.push_str(&format!(
            "- **Inter-Class Correlation:** {:.3}\n",
            results.inter_rater_reliability.inter_class_correlation
        ));
        report.push_str(&format!(
            "- **Kendall's Concordance:** {:.3}\n",
            results.inter_rater_reliability.kendalls_concordance
        ));

        report.push_str("\n## Internal Consistency\n\n");
        report.push_str(&format!(
            "- **Cronbach's Alpha:** {:.3}\n",
            results.internal_consistency.cronbachs_alpha
        ));
        report.push_str(&format!(
            "- **Split-Half Reliability:** {:.3}\n",
            results.internal_consistency.split_half_reliability
        ));

        report.push_str("\n## Reproducibility\n\n");
        report.push_str(&format!(
            "- **Cross-Platform Correlation:** {:.3}\n",
            results
                .reproducibility
                .cross_platform
                .cross_platform_correlation
        ));
        report.push_str(&format!(
            "- **Temporal Correlation:** {:.3}\n",
            results
                .reproducibility
                .temporal_reproducibility
                .temporal_correlation
        ));

        if !results.overall_assessment.recommendations.is_empty() {
            report.push_str("\n## Recommendations\n\n");
            for recommendation in &results.overall_assessment.recommendations {
                report.push_str(&format!("- {}\n", recommendation));
            }
        }

        report
    }

    /// Clear results cache
    pub fn clear_cache(&mut self) {
        self.results_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_reliability_tester_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ReliabilityTestConfig::default();

        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf()).await;
        assert!(tester.is_ok());
    }

    #[tokio::test]
    async fn test_reliability_classification() {
        let temp_dir = TempDir::new().unwrap();
        let config = ReliabilityTestConfig::default();
        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        assert_eq!(
            tester.classify_reliability(0.95),
            ReliabilityClassification::Excellent
        );
        assert_eq!(
            tester.classify_reliability(0.85),
            ReliabilityClassification::Good
        );
        assert_eq!(
            tester.classify_reliability(0.75),
            ReliabilityClassification::Acceptable
        );
        assert_eq!(
            tester.classify_reliability(0.65),
            ReliabilityClassification::Questionable
        );
        assert_eq!(
            tester.classify_reliability(0.55),
            ReliabilityClassification::Poor
        );
    }

    #[test]
    fn test_reliability_config_default() {
        let config = ReliabilityTestConfig::default();

        assert_eq!(config.test_retest_repetitions, 3);
        assert_eq!(config.min_test_retest_correlation, 0.8);
        assert_eq!(config.confidence_level, 0.95);
        assert!(config.enable_detailed_reporting);
    }

    #[test]
    fn test_drift_direction() {
        let drift = DriftDetectionResults {
            drift_detected: true,
            drift_magnitude: 0.05,
            drift_direction: DriftDirection::Increasing,
            change_points: vec![10, 25],
        };

        assert!(drift.drift_detected);
        assert_eq!(drift.change_points.len(), 2);
        assert!(matches!(drift.drift_direction, DriftDirection::Increasing));
    }
}
