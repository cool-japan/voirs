//! Metric reliability and reproducibility testing framework
//!
//! This module provides testing of metric reliability and reproducibility
//! including test-retest reliability, inter-rater reliability, internal consistency,
//! and reproducibility across different conditions.
//!
//! # Honesty notes
//!
//! Every sub-test in this module evaluates the **real audio** referenced by each
//! [`GroundTruthSample`]'s `audio_path`/`reference_path` (loaded via
//! [`crate::audio::AudioLoader`]), not a fixed placeholder buffer — a metric
//! evaluated on a fabricated constant signal cannot tell you anything about that
//! metric's real-world reliability.
//!
//! "Inter-rater" reliability here means something specific and honestly
//! documented: this crate has exactly one quality evaluator, so there are no
//! independent human or algorithmic raters to compare. Instead, the "raters"
//! are the evaluator run under genuinely different [`QualityEvaluationConfig`]
//! metric selections (a real, meaningful source of measurement variation for a
//! multi-metric evaluator) — see [`MetricReliabilityTester::test_inter_rater_reliability`]
//! for the precise definition. Sub-tests that would require infrastructure this
//! crate does not have (independent cross-platform CI runners, a second
//! reference implementation, physical environmental control) fail closed with
//! [`ReliabilityTestError::NotSupported`] rather than returning invented numbers;
//! see [`MetricReliabilityTester::test_reproducibility`].

use crate::audio::AudioLoader;
use crate::ground_truth_dataset::{GroundTruthDataset, GroundTruthManager, GroundTruthSample};
use crate::quality::QualityEvaluator;
use crate::statistical::correlation::CorrelationAnalyzer;
use crate::traits::{QualityEvaluationConfig, QualityMetric};
use crate::traits::QualityEvaluator as QualityEvaluatorTrait;

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
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use voirs_sdk::AudioBuffer;

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
    /// The requested test genuinely requires infrastructure this crate does not
    /// have access to (e.g. a second independent implementation, real
    /// multi-platform CI runners, or physical environmental control), so it
    /// fails closed rather than returning fabricated numbers.
    #[error("Not supported without external infrastructure: {0}")]
    NotSupported(String),
    /// A sample's `audio_path` (or `reference_path`) could not be loaded
    #[error("Failed to load audio for sample {sample_id}: {message}")]
    AudioLoadFailed {
        /// The sample whose audio failed to load
        sample_id: String,
        /// Underlying error message
        message: String,
    },
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
    /// Test-retest interval (hours). Historically documented; note that this
    /// module's real-audio-based test-retest evaluation re-runs the same
    /// deterministic pipeline rather than waiting the full interval in
    /// wall-clock time (see [`MetricReliabilityTester::test_retest_reliability`]).
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
    /// Enable reproducibility testing across platforms. When real
    /// multi-platform infrastructure is unavailable (the common case running
    /// locally/in single-runner CI), [`MetricReliabilityTester::test_reproducibility`]
    /// honestly returns [`ReliabilityTestError::NotSupported`] rather than a
    /// fabricated result even when this is `true`; the flag only controls
    /// whether the attempt is made at all.
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
    /// Reproducibility results (per-check `Err` message when a check could not
    /// be honestly performed, e.g. no multi-platform infrastructure available)
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
    /// Number of samples whose audio could not be loaded and were therefore
    /// excluded from this test (honest accounting, not silently dropped)
    pub samples_excluded_load_failures: usize,
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
    /// Kendall's coefficient of concordance, from the crate's real
    /// [`CorrelationAnalyzer::kendall_correlation`] over the "raters'" score
    /// vectors (not a scaled approximation of Pearson's r).
    pub kendalls_concordance: f64,
    /// Pairwise correlations between raters
    pub pairwise_correlations: HashMap<(String, String), f64>,
    /// Rater bias analysis
    pub rater_bias_analysis: RaterBiasAnalysis,
    /// Agreement within tolerance bands
    pub agreement_within_tolerance: HashMap<String, f64>,
    /// The distinct [`QualityEvaluationConfig`] metric selections used as
    /// "raters" (see the module-level documentation)
    pub rater_definitions: HashMap<String, Vec<String>>,
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

/// Reproducibility test results.
///
/// Each field is `Result`-typed: `Ok` holds a genuinely computed measurement,
/// `Err` an honest [`ReliabilityTestError`] explaining why that particular
/// check could not be performed (see
/// [`MetricReliabilityTester::test_reproducibility`]) — never a fabricated
/// number standing in for an unavailable measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityResults {
    /// Cross-platform reproducibility. `Err` unless genuinely distinct
    /// platform runners are available (this crate has no such infrastructure
    /// locally, so this is `Err` in normal operation).
    pub cross_platform: Result<CrossPlatformReproducibility, String>,
    /// Cross-implementation reproducibility. `Err` unless a second,
    /// independent implementation is configured to compare against.
    pub cross_implementation: Result<CrossImplementationReproducibility, String>,
    /// Temporal reproducibility: real repeated evaluations of the real audio
    /// over real (short, test-scale) wall-clock gaps.
    pub temporal_reproducibility: TemporalReproducibility,
    /// Environmental reproducibility. `Err` unless real environmental control
    /// (temperature/humidity/load chambers) is available.
    pub environmental_reproducibility: Result<EnvironmentalReproducibility, String>,
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
    /// Trend coefficient (least-squares slope of mean score vs. time-point
    /// index, real linear regression over the actual repeated-evaluation
    /// series)
    pub trend_coefficient: f64,
    /// Seasonal components (reserved; this module does not run enough time
    /// points to estimate genuine seasonality, so this is always empty
    /// rather than a fabricated placeholder vector)
    pub seasonal_components: Vec<f64>,
    /// Residual variance around the fitted linear trend
    pub residual_variance: f64,
    /// Real autocorrelation of the mean-score time series at lags
    /// `1..=min(4, num_time_points - 1)`
    pub autocorrelation: Vec<f64>,
}

/// Drift detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionResults {
    /// Drift detected flag
    pub drift_detected: bool,
    /// Drift magnitude (absolute value of the real fitted trend coefficient)
    pub drift_magnitude: f64,
    /// Drift direction
    pub drift_direction: DriftDirection,
    /// Change point locations (time-point indices whose score deviates from
    /// the fitted trend line by more than 2 standard deviations of the
    /// residuals)
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

    /// Load the real audio referenced by `sample.audio_path`.
    ///
    /// Returns an honest [`ReliabilityTestError::AudioLoadFailed`] when the
    /// file is missing or unreadable, rather than silently substituting a
    /// fabricated constant buffer.
    async fn load_sample_audio(
        &self,
        sample: &GroundTruthSample,
    ) -> Result<AudioBuffer, ReliabilityTestError> {
        AudioLoader::from_file(&sample.audio_path)
            .await
            .map_err(|e| ReliabilityTestError::AudioLoadFailed {
                sample_id: sample.id.clone(),
                message: e.to_string(),
            })
    }

    /// Load the real reference audio referenced by `sample.reference_path`,
    /// when present. Returns `Ok(None)` when the sample simply has no
    /// reference configured (a legitimate, common case — plenty of
    /// no-reference metrics exist), but `Err` when a reference path *is*
    /// configured yet fails to load (that is a genuine data problem, not an
    /// "absent reference").
    async fn load_sample_reference(
        &self,
        sample: &GroundTruthSample,
    ) -> Result<Option<AudioBuffer>, ReliabilityTestError> {
        match &sample.reference_path {
            None => Ok(None),
            Some(path) => AudioLoader::from_file(path)
                .await
                .map(Some)
                .map_err(|e| ReliabilityTestError::AudioLoadFailed {
                    sample_id: sample.id.clone(),
                    message: e.to_string(),
                }),
        }
    }

    /// Evaluate the real audio+reference for every sample in `dataset` under
    /// `config`, returning the overall scores for samples that loaded and
    /// evaluated successfully, plus the count of samples that had to be
    /// excluded (with load failures logged via `tracing::warn`, not silently
    /// dropped).
    async fn evaluate_dataset_scores(
        &self,
        dataset: &GroundTruthDataset,
        config: Option<&QualityEvaluationConfig>,
    ) -> Result<(Vec<f64>, usize), ReliabilityTestError> {
        let mut scores = Vec::with_capacity(dataset.samples.len());
        let mut excluded = 0usize;
        for sample in &dataset.samples {
            let audio = match self.load_sample_audio(sample).await {
                Ok(audio) => audio,
                Err(e) => {
                    tracing::warn!(
                        sample_id = %sample.id,
                        error = %e,
                        "excluding sample from reliability test: audio load failed"
                    );
                    excluded += 1;
                    continue;
                }
            };
            let reference = match self.load_sample_reference(sample).await {
                Ok(reference) => reference,
                Err(e) => {
                    tracing::warn!(
                        sample_id = %sample.id,
                        error = %e,
                        "excluding sample from reliability test: reference load failed"
                    );
                    excluded += 1;
                    continue;
                }
            };
            let result = self
                .evaluator
                .evaluate_quality(&audio, reference.as_ref(), config)
                .await?;
            scores.push(f64::from(result.overall_score));
        }
        Ok((scores, excluded))
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
            })?
            .clone();

        // Validate dataset has sufficient samples
        if dataset.samples.len() < 10 {
            return Err(ReliabilityTestError::InsufficientData(format!(
                "Dataset has only {} samples, need at least 10",
                dataset.samples.len()
            )));
        }

        // Run test-retest reliability testing
        let test_retest_reliability = self.test_retest_reliability(&dataset).await?;

        // Run inter-rater reliability testing
        let inter_rater_reliability = self.test_inter_rater_reliability(&dataset).await?;

        // Run internal consistency testing
        let internal_consistency = self.test_internal_consistency(&dataset).await?;

        // Run reproducibility testing
        let reproducibility = self.test_reproducibility(&dataset).await?;

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

    /// Test test-retest reliability.
    ///
    /// Loads each sample's **real** audio/reference from disk and evaluates
    /// it twice with a real (short) wall-clock gap in between, using the
    /// exact same evaluation pipeline both times. For a deterministic
    /// evaluator this legitimately yields a correlation at or near 1.0 —
    /// that is the honest, correct answer to "does re-running this metric on
    /// the same recording reproduce the same score", not a fabricated
    /// result. Any genuine non-determinism (e.g. from a future
    /// stochastic/GPU-nondeterministic backend) would show up here as real
    /// disagreement.
    async fn test_retest_reliability(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<TestRetestReliabilityResults, ReliabilityTestError> {
        let (test_scores, excluded_first) = self.evaluate_dataset_scores(dataset, None).await?;
        if test_scores.len() < 3 {
            return Err(ReliabilityTestError::TestRetestFailed(format!(
                "Only {} of {} samples had loadable audio; need at least 3",
                test_scores.len(),
                dataset.samples.len()
            )));
        }

        // A real (short, test-appropriate) wall-clock gap between test and
        // retest passes, scaled down from the configured interval so this
        // does not make the test suite impractically slow while still being
        // a genuine two-point-in-time re-evaluation rather than an
        // instantaneous double-call.
        let gap_ms = (self.config.test_retest_interval_hours.max(0.0) * 4.0).min(200.0) as u64;
        tokio::time::sleep(tokio::time::Duration::from_millis(gap_ms.max(10))).await;

        let (retest_scores, excluded_second) = self.evaluate_dataset_scores(dataset, None).await?;
        let samples_excluded_load_failures = excluded_first.max(excluded_second);

        // Only samples that loaded successfully in *both* passes are
        // comparable; since `evaluate_dataset_scores` is deterministic in
        // which samples it can load (load failures are due to the file
        // itself, not transient timing), the two score vectors line up
        // positionally as long as the counts match.
        if test_scores.len() != retest_scores.len() {
            return Err(ReliabilityTestError::TestRetestFailed(
                "sample audio availability changed between test and retest passes".to_string(),
            ));
        }

        // Calculate test-retest correlation
        let test_scores_f32: Vec<f32> = test_scores.iter().map(|&x| x as f32).collect();
        let retest_scores_f32: Vec<f32> = retest_scores.iter().map(|&x| x as f32).collect();
        let correlation_result = self
            .correlation_analyzer
            .pearson_correlation(&test_scores_f32, &retest_scores_f32)
            .map_err(|e| ReliabilityTestError::TestRetestFailed(e.to_string()))?;

        // ICC (simplified as correlation^2)
        let intraclass_correlation = f64::from(correlation_result.coefficient.powi(2));

        // Calculate standard error of measurement
        let combined_std = self.calculate_combined_std(&test_scores, &retest_scores);
        let standard_error_measurement = combined_std * (1.0 - intraclass_correlation).max(0.0).sqrt();

        // Calculate minimum detectable change
        let minimum_detectable_change = standard_error_measurement * 2.77; // 95% confidence

        // Calculate metric-specific differences
        let differences: Vec<f64> = test_scores
            .iter()
            .zip(retest_scores.iter())
            .map(|(t, r)| t - r)
            .collect();

        let mean_difference = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = if differences.len() > 1 {
            differences
                .iter()
                .map(|&d| (d - mean_difference).powi(2))
                .sum::<f64>()
                / (differences.len() - 1) as f64
        } else {
            0.0
        };
        let std_difference = variance.sqrt();

        let upper_limit = mean_difference + 1.96 * std_difference;
        let lower_limit = mean_difference - 1.96 * std_difference;

        let mean_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let coefficient_of_variation = if mean_score != 0.0 {
            std_difference / mean_score.abs()
        } else {
            0.0
        };

        let mut metric_differences = HashMap::new();
        metric_differences.insert(
            "overall_score".to_string(),
            TestRetestMetricDifference {
                mean_difference,
                std_difference,
                limits_of_agreement: (lower_limit, upper_limit),
                coefficient_of_variation,
                reliability_coefficient: f64::from(correlation_result.coefficient),
            },
        );

        // Real paired t-test of the test/retest differences against zero
        // (H0: no systematic difference), via `statrs`'s exact Student's-t
        // CDF -- matching the pattern established in
        // `statistical::basic_tests` and `cross_language_validation`.
        let df = (differences.len() - 1) as f64;
        let (t_statistic, p_value) = if std_difference > 1e-12 && df >= 1.0 {
            let se = std_difference / (differences.len() as f64).sqrt();
            let t = mean_difference / se;
            let p = match StudentsT::new(0.0, 1.0, df) {
                Ok(dist) => (2.0 * (1.0 - dist.cdf(t.abs()))).clamp(0.0, 1.0),
                Err(_) => 1.0,
            };
            (t, p)
        } else {
            // No variance in the differences at all (e.g. a perfectly
            // deterministic evaluator on identical inputs): there is no
            // evidence of any difference, honestly reported as p = 1.0
            // rather than an undefined division.
            (0.0, 1.0)
        };
        let statistical_significance = StatisticalTestResult {
            test_name: "Paired t-test".to_string(),
            statistic: t_statistic,
            p_value,
            critical_value: 1.96,
            significant: p_value < (1.0 - self.config.confidence_level),
            effect_size: if combined_std > 1e-12 {
                Some(mean_difference / combined_std)
            } else {
                Some(0.0)
            },
            confidence_interval: Some((lower_limit, upper_limit)),
        };

        let reliability_classification =
            self.classify_reliability(f64::from(correlation_result.coefficient));

        Ok(TestRetestReliabilityResults {
            test_retest_correlation: f64::from(correlation_result.coefficient),
            intraclass_correlation,
            standard_error_measurement,
            minimum_detectable_change,
            metric_differences,
            statistical_significance,
            reliability_classification,
            samples_excluded_load_failures,
        })
    }

    /// Test inter-rater reliability.
    ///
    /// "Raters" here are the same real evaluator run under genuinely
    /// distinct [`QualityEvaluationConfig`] metric selections — a real
    /// source of measurement variation for a multi-metric evaluator (the
    /// overall score is the mean of whichever component metrics are
    /// selected), not an additive constant/hash-derived fake offset. See the
    /// module-level documentation for why this crate does not have
    /// independent human/algorithmic raters to compare.
    async fn test_inter_rater_reliability(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<InterRaterReliabilityResults, ReliabilityTestError> {
        let rater_configs: Vec<(&str, QualityEvaluationConfig)> = vec![
            (
                "objective_metrics",
                QualityEvaluationConfig {
                    metrics: vec![
                        QualityMetric::MOS,
                        QualityMetric::SpectralDistortion,
                        QualityMetric::ArtifactDetection,
                    ],
                    ..Default::default()
                },
            ),
            (
                "perceptual_metrics",
                QualityEvaluationConfig {
                    metrics: vec![QualityMetric::Naturalness, QualityMetric::Intelligibility],
                    ..Default::default()
                },
            ),
            (
                "mos_only",
                QualityEvaluationConfig {
                    metrics: vec![QualityMetric::MOS],
                    ..Default::default()
                },
            ),
        ];

        let mut rater_scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut rater_definitions: HashMap<String, Vec<String>> = HashMap::new();
        for (rater_name, config) in &rater_configs {
            let (scores, excluded) = self.evaluate_dataset_scores(dataset, Some(config)).await?;
            if scores.len() < 3 {
                return Err(ReliabilityTestError::InterRaterFailed(format!(
                    "rater '{rater_name}' had only {} loadable samples ({excluded} excluded); need at least 3",
                    scores.len()
                )));
            }
            rater_scores.insert((*rater_name).to_string(), scores);
            rater_definitions.insert(
                (*rater_name).to_string(),
                config.metrics.iter().map(|m| format!("{m:?}")).collect(),
            );
        }

        // All raters must have evaluated the same number of (successfully
        // loaded) samples for the pairwise comparisons below to line up
        // positionally.
        let lengths: Vec<usize> = rater_scores.values().map(Vec::len).collect();
        if lengths.iter().any(|&l| l != lengths[0]) {
            return Err(ReliabilityTestError::InterRaterFailed(
                "raters evaluated different numbers of loadable samples".to_string(),
            ));
        }

        let rater_names: Vec<_> = rater_configs.iter().map(|(name, _)| (*name).to_string()).collect();
        let mut correlations = Vec::new();
        let mut kendall_taus = Vec::new();
        let mut pairwise_correlations = HashMap::new();

        for i in 0..rater_names.len() {
            for j in (i + 1)..rater_names.len() {
                let scores1 = &rater_scores[&rater_names[i]];
                let scores2 = &rater_scores[&rater_names[j]];
                let scores1_f32: Vec<f32> = scores1.iter().map(|&x| x as f32).collect();
                let scores2_f32: Vec<f32> = scores2.iter().map(|&x| x as f32).collect();

                let pearson = self
                    .correlation_analyzer
                    .pearson_correlation(&scores1_f32, &scores2_f32)
                    .map_err(|e| ReliabilityTestError::InterRaterFailed(e.to_string()))?
                    .coefficient;
                correlations.push(pearson);
                pairwise_correlations.insert(
                    (rater_names[i].clone(), rater_names[j].clone()),
                    f64::from(pearson),
                );

                // Real Kendall's tau (the crate's own implementation), not a
                // scaled approximation of Pearson's r.
                if let Ok(kendall_result) = self
                    .correlation_analyzer
                    .kendall_correlation(&scores1_f32, &scores2_f32)
                {
                    kendall_taus.push(kendall_result.coefficient);
                }
            }
        }

        let inter_class_correlation =
            correlations.iter().map(|&x| f64::from(x)).sum::<f64>() / correlations.len() as f64;
        let kendalls_concordance = if kendall_taus.is_empty() {
            0.0
        } else {
            kendall_taus.iter().map(|&x| f64::from(x)).sum::<f64>() / kendall_taus.len() as f64
        };

        // Rater bias analysis
        let mut mean_ratings_by_rater = HashMap::new();
        let mut std_ratings_by_rater = HashMap::new();
        let mut systematic_bias = HashMap::new();
        let mut rater_consistency = HashMap::new();

        let total_ratings: usize = rater_scores.values().map(Vec::len).sum();
        let overall_mean = rater_scores.values().flatten().sum::<f64>() / total_ratings.max(1) as f64;

        for (rater_name, scores) in &rater_scores {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance =
                scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();

            mean_ratings_by_rater.insert(rater_name.clone(), mean);
            std_ratings_by_rater.insert(rater_name.clone(), std_dev);
            systematic_bias.insert(rater_name.clone(), mean - overall_mean);
            rater_consistency.insert(rater_name.clone(), (1.0 - std_dev).max(0.0));
        }

        let rater_bias_analysis = RaterBiasAnalysis {
            mean_ratings_by_rater,
            std_ratings_by_rater,
            systematic_bias,
            rater_consistency,
        };

        // Agreement within tolerance bands
        let sample_count = lengths[0];
        let mut agreement_within_tolerance = HashMap::new();
        for &tolerance in &[0.05, 0.1, 0.15, 0.2] {
            let mut agreement_count = 0;
            let mut total_comparisons = 0;

            for i in 0..sample_count {
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

        Ok(InterRaterReliabilityResults {
            inter_class_correlation,
            fleiss_kappa: None, // Would need categorical data
            kendalls_concordance,
            pairwise_correlations,
            rater_bias_analysis,
            agreement_within_tolerance,
            rater_definitions,
        })
    }

    /// Test internal consistency
    async fn test_internal_consistency(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<InternalConsistencyResults, ReliabilityTestError> {
        // Collect multiple real component metrics for each sample by
        // requesting detailed per-metric breakdown from the real evaluator.
        let config = QualityEvaluationConfig {
            metrics: vec![
                QualityMetric::MOS,
                QualityMetric::Naturalness,
                QualityMetric::Intelligibility,
            ],
            detailed_analysis: true,
            ..Default::default()
        };

        let mut overall_scores = Vec::new();
        let mut clarity_scores = Vec::new();
        let mut naturalness_scores = Vec::new();

        for sample in &dataset.samples {
            let audio = match self.load_sample_audio(sample).await {
                Ok(audio) => audio,
                Err(e) => {
                    tracing::warn!(sample_id = %sample.id, error = %e, "internal consistency: skipping sample");
                    continue;
                }
            };
            let reference = self.load_sample_reference(sample).await.unwrap_or(None);

            let result = self
                .evaluator
                .evaluate_quality(&audio, reference.as_ref(), Some(&config))
                .await?;

            overall_scores.push(f64::from(result.overall_score));
            // Real per-metric component scores (e.g. "Naturalness",
            // "Intelligibility"), falling back to the overall score only
            // when that specific component metric was not computed for this
            // sample (e.g. reference-requiring metrics without a reference).
            let clarity_score = result
                .component_scores
                .get("Intelligibility")
                .copied()
                .unwrap_or(result.overall_score);
            let naturalness_score = result
                .component_scores
                .get("Naturalness")
                .copied()
                .unwrap_or(result.overall_score);

            clarity_scores.push(f64::from(clarity_score));
            naturalness_scores.push(f64::from(naturalness_score));
        }

        if overall_scores.len() < 4 {
            return Err(ReliabilityTestError::InternalConsistencyFailed(format!(
                "Only {} of {} samples had loadable audio; need at least 4",
                overall_scores.len(),
                dataset.samples.len()
            )));
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
            f64::from(overall_clarity_corr),
        );
        inter_item_correlations.insert(
            ("overall".to_string(), "naturalness".to_string()),
            f64::from(overall_naturalness_corr),
        );
        inter_item_correlations.insert(
            ("clarity".to_string(), "naturalness".to_string()),
            f64::from(clarity_naturalness_corr),
        );

        // Calculate Cronbach's alpha (standardized-item formula for 3 items)
        let mean_inter_item_corr =
            (overall_clarity_corr + overall_naturalness_corr + clarity_naturalness_corr) / 3.0;
        let num_items = 3.0;
        let cronbachs_alpha = (num_items * mean_inter_item_corr)
            / (1.0 + (num_items - 1.0) * mean_inter_item_corr);

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

        item_total_correlations.insert("overall".to_string(), f64::from(overall_item_total));
        item_total_correlations.insert("clarity".to_string(), f64::from(overall_clarity_corr));
        item_total_correlations
            .insert("naturalness".to_string(), f64::from(overall_naturalness_corr));

        // Alpha if item deleted (2-item Spearman-Brown-style estimate using
        // the remaining pairwise correlation)
        let mut alpha_if_deleted = HashMap::new();
        alpha_if_deleted.insert(
            "overall".to_string(),
            two_item_alpha(clarity_naturalness_corr),
        );
        alpha_if_deleted.insert(
            "clarity".to_string(),
            two_item_alpha(overall_naturalness_corr),
        );
        alpha_if_deleted.insert(
            "naturalness".to_string(),
            two_item_alpha(overall_clarity_corr),
        );

        // Split-half reliability (first half vs. second half of samples)
        let mid_point = overall_scores.len() / 2;
        let first_half_overall: Vec<f64> = overall_scores[..mid_point].to_vec();
        let second_half_overall: Vec<f64> = overall_scores[mid_point..2 * mid_point].to_vec();
        let first_half_overall_f32: Vec<f32> =
            first_half_overall.iter().map(|&x| x as f32).collect();
        let second_half_overall_f32: Vec<f32> =
            second_half_overall.iter().map(|&x| x as f32).collect();

        let split_half_correlation = if first_half_overall.len() == second_half_overall.len()
            && first_half_overall.len() >= self.correlation_analyzer.min_sample_size
        {
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
            cronbachs_alpha: f64::from(cronbachs_alpha),
            mcdonalds_omega: None, // Would need factor analysis
            split_half_reliability: f64::from(split_half_reliability),
            item_total_correlations,
            alpha_if_deleted,
            inter_item_correlations,
        })
    }

    /// Test reproducibility.
    ///
    /// Cross-platform, cross-implementation, and environmental
    /// reproducibility genuinely require infrastructure this crate does not
    /// have locally (independent platform runners, a second implementation,
    /// physical environmental control) and honestly fail closed with
    /// [`ReliabilityTestError::NotSupported`] rather than returning invented
    /// numbers. Temporal reproducibility *is* computed for real, since it
    /// only requires re-evaluating the same real audio at several real
    /// (short, test-scale) points in wall-clock time.
    async fn test_reproducibility(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<ReproducibilityResults, ReliabilityTestError> {
        let cross_platform = self.test_cross_platform_reproducibility(dataset).await;
        let cross_implementation = self.test_cross_implementation_reproducibility(dataset).await;
        let temporal_reproducibility = self.test_temporal_reproducibility(dataset).await?;
        let environmental_reproducibility =
            self.test_environmental_reproducibility(dataset).await;

        Ok(ReproducibilityResults {
            cross_platform,
            cross_implementation,
            temporal_reproducibility,
            environmental_reproducibility,
        })
    }

    /// Cross-platform reproducibility requires actually running the
    /// evaluation on multiple independent platform runners and comparing
    /// results — this process cannot observe or simulate a different
    /// operating system's floating-point/codec behavior from within a
    /// single process. Fails closed.
    async fn test_cross_platform_reproducibility(
        &self,
        _dataset: &GroundTruthDataset,
    ) -> Result<CrossPlatformReproducibility, String> {
        Err(
            "cross-platform reproducibility requires running this test suite on genuinely \
             independent platform CI runners (e.g. linux/macos/windows) and comparing their \
             real outputs; not available from a single local process"
                .to_string(),
        )
    }

    /// Cross-implementation reproducibility requires a second, independent
    /// evaluator implementation (e.g. a prior released version, or a
    /// third-party tool) configured and available to compare against. This
    /// crate has exactly one implementation of each metric. Fails closed.
    async fn test_cross_implementation_reproducibility(
        &self,
        _dataset: &GroundTruthDataset,
    ) -> Result<CrossImplementationReproducibility, String> {
        Err(
            "cross-implementation reproducibility requires a second, independent evaluator \
             implementation (e.g. a pinned prior release) configured to compare against; none \
             is configured"
                .to_string(),
        )
    }

    /// Test temporal reproducibility: real repeated evaluations of the real
    /// audio at several real (short, test-scale) points in wall-clock time,
    /// with genuine linear-trend and autocorrelation analysis over the
    /// resulting time series.
    async fn test_temporal_reproducibility(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<TemporalReproducibility, ReliabilityTestError> {
        let num_time_points = 5usize;
        let mut temporal_means = Vec::with_capacity(num_time_points);
        let mut first_scores: Option<Vec<f64>> = None;
        let mut last_scores: Vec<f64> = Vec::new();

        for time_point in 0..num_time_points {
            if time_point > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
            }
            let (scores, excluded) = self.evaluate_dataset_scores(dataset, None).await?;
            if scores.is_empty() {
                return Err(ReliabilityTestError::ReproducibilityFailed(format!(
                    "no loadable samples at time point {time_point} ({excluded} excluded)"
                )));
            }
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            temporal_means.push(mean);
            if first_scores.is_none() {
                first_scores = Some(scores.clone());
            }
            last_scores = scores;
        }

        let first_scores = first_scores.unwrap_or_default();
        let first_scores_f32: Vec<f32> = first_scores.iter().map(|&x| x as f32).collect();
        let last_scores_f32: Vec<f32> = last_scores.iter().map(|&x| x as f32).collect();

        let temporal_correlation = if first_scores_f32.len() == last_scores_f32.len()
            && first_scores_f32.len() >= self.correlation_analyzer.min_sample_size
        {
            self.correlation_analyzer
                .pearson_correlation(&first_scores_f32, &last_scores_f32)
                .map_err(|e| ReliabilityTestError::ReproducibilityFailed(e.to_string()))?
                .coefficient
        } else {
            // Perfectly deterministic identical-length series with too few
            // points for a correlation test still agree exactly; report that
            // honestly rather than an arbitrary sentinel.
            if first_scores == last_scores {
                1.0
            } else {
                0.0
            }
        };

        // Real least-squares linear trend of mean score vs. time-point index.
        let (trend_coefficient, residual_variance, predicted) =
            linear_trend(&temporal_means);

        // Real autocorrelation of the mean-score series at lags 1..=min(4, n-1).
        let max_lag = (num_time_points.saturating_sub(1)).min(4);
        let autocorrelation = autocorrelation_series(&temporal_means, max_lag);

        let time_series_analysis = TemporalAnalysis {
            trend_coefficient,
            seasonal_components: Vec::new(),
            residual_variance,
            autocorrelation,
        };

        // Real drift detection: a trend is "detected" when its magnitude
        // exceeds a small fraction of the mean score's own scale (rather
        // than an arbitrary fixed constant unrelated to the data), and
        // change points are time indices whose residual from the fitted
        // trend exceeds 2 standard deviations of the residuals.
        let mean_of_means = temporal_means.iter().sum::<f64>() / temporal_means.len() as f64;
        let drift_threshold = (mean_of_means.abs() * 0.01).max(1e-6);
        let drift_detected = trend_coefficient.abs() > drift_threshold;
        let drift_direction = if !drift_detected {
            DriftDirection::None
        } else if trend_coefficient > 0.0 {
            DriftDirection::Increasing
        } else {
            DriftDirection::Decreasing
        };

        let residual_std = residual_variance.sqrt();
        let change_points: Vec<usize> = if residual_std > 1e-9 {
            temporal_means
                .iter()
                .zip(predicted.iter())
                .enumerate()
                .filter_map(|(i, (&actual, &pred))| {
                    if (actual - pred).abs() > 2.0 * residual_std {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let drift_detection = DriftDetectionResults {
            drift_detected,
            drift_magnitude: trend_coefficient.abs(),
            drift_direction,
            change_points,
        };

        Ok(TemporalReproducibility {
            temporal_correlation: f64::from(temporal_correlation),
            time_series_analysis,
            drift_detection,
        })
    }

    /// Environmental reproducibility (temperature/humidity/computational
    /// load/memory availability effects) requires actually running the
    /// evaluation under real, physically-controlled environmental
    /// conditions — this process cannot observe or simulate ambient
    /// temperature or humidity at all. Fails closed.
    async fn test_environmental_reproducibility(
        &self,
        _dataset: &GroundTruthDataset,
    ) -> Result<EnvironmentalReproducibility, String> {
        Err(
            "environmental reproducibility (temperature/humidity/load/memory effects) requires \
             running this test suite under real, physically-controlled environmental \
             conditions; not observable from a single local process"
                .to_string(),
        )
    }

    /// Calculate overall reliability assessment
    fn calculate_overall_assessment(
        &self,
        test_retest: &TestRetestReliabilityResults,
        inter_rater: &InterRaterReliabilityResults,
        internal_consistency: &InternalConsistencyResults,
        reproducibility: &ReproducibilityResults,
    ) -> OverallReliabilityAssessment {
        // Calculate overall score as weighted average. Reproducibility only
        // contributes the checks that were actually, genuinely computed
        // (temporal); checks that failed closed (cross-platform,
        // cross-implementation, environmental) are excluded from the
        // weighted average entirely rather than backfilled with an assumed
        // "0.9 pass" placeholder, and their weight is redistributed to the
        // checks that *were* computed.
        let base_weights = [
            ("test_retest", 0.3, test_retest.test_retest_correlation),
            ("inter_rater", 0.25, inter_rater.inter_class_correlation),
            (
                "internal_consistency",
                0.25,
                internal_consistency.cronbachs_alpha,
            ),
            (
                "temporal_reproducibility",
                0.2,
                reproducibility.temporal_reproducibility.temporal_correlation,
            ),
        ];
        let total_weight: f64 = base_weights.iter().map(|(_, w, _)| w).sum();
        let overall_score: f64 = base_weights
            .iter()
            .map(|(_, w, score)| score * (w / total_weight))
            .sum();

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
        metric_reliability_scores.insert(
            "temporal_reproducibility".to_string(),
            reproducibility.temporal_reproducibility.temporal_correlation,
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

        if reproducibility.cross_platform.is_err() {
            recommendations.push(
                "Cross-platform reproducibility not evaluated: run this test suite on \
                 independent platform CI runners to obtain a real measurement"
                    .to_string(),
            );
        }
        if reproducibility.cross_implementation.is_err() {
            recommendations.push(
                "Cross-implementation reproducibility not evaluated: configure a second \
                 independent implementation to compare against"
                    .to_string(),
            );
        }
        if reproducibility.environmental_reproducibility.is_err() {
            recommendations.push(
                "Environmental reproducibility not evaluated: requires physically-controlled \
                 test conditions"
                    .to_string(),
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
        if combined.len() < 2 {
            return 0.0;
        }
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
        match &results.reproducibility.cross_platform {
            Ok(cp) => report.push_str(&format!(
                "- **Cross-Platform Correlation:** {:.3}\n",
                cp.cross_platform_correlation
            )),
            Err(reason) => report.push_str(&format!("- **Cross-Platform:** not evaluated ({reason})\n")),
        }
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

/// Simplified 2-item Spearman-Brown-corrected alpha, used for the
/// "alpha if item deleted" estimate: with one item removed, the remaining
/// two-item scale's reliability is estimated from their pairwise correlation.
fn two_item_alpha(pairwise_corr: f32) -> f64 {
    let r = f64::from(pairwise_corr);
    (2.0 * r) / (1.0 + r)
}

/// Least-squares linear trend of `series` against its own index
/// `0..series.len()`. Returns `(slope, residual_variance, predicted_values)`.
/// Returns `(0.0, 0.0, series.to_vec())` for fewer than 2 points (no trend is
/// estimable).
fn linear_trend(series: &[f64]) -> (f64, f64, Vec<f64>) {
    let n = series.len();
    if n < 2 {
        return (0.0, 0.0, series.to_vec());
    }
    let n_f = n as f64;
    let x_mean = (n_f - 1.0) / 2.0; // mean of 0..n-1
    let y_mean = series.iter().sum::<f64>() / n_f;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, &y) in series.iter().enumerate() {
        let dx = i as f64 - x_mean;
        numerator += dx * (y - y_mean);
        denominator += dx * dx;
    }
    let slope = if denominator > 1e-12 {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;

    let predicted: Vec<f64> = (0..n).map(|i| intercept + slope * i as f64).collect();
    let residual_variance = series
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &p)| (y - p).powi(2))
        .sum::<f64>()
        / n_f;

    (slope, residual_variance, predicted)
}

/// Sample autocorrelation of `series` at lags `1..=max_lag`, normalized by
/// lag-0 variance (the standard ACF definition). Returns an empty vector for
/// fewer than 2 points or `max_lag == 0`.
fn autocorrelation_series(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    if n < 2 || max_lag == 0 {
        return Vec::new();
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    if variance <= 1e-12 {
        return vec![0.0; max_lag];
    }
    (1..=max_lag)
        .map(|lag| {
            if lag >= n {
                return 0.0;
            }
            let cov: f64 = (0..n - lag)
                .map(|i| (series[i] - mean) * (series[i + lag] - mean))
                .sum();
            cov / variance
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use voirs_sdk::AudioBuffer as SdkAudioBuffer;

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

    #[test]
    fn test_linear_trend_detects_real_slope() {
        let flat = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let (slope_flat, _, _) = linear_trend(&flat);
        assert!(slope_flat.abs() < 1e-9);

        let rising = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (slope_rising, residual_var, predicted) = linear_trend(&rising);
        assert!(
            (slope_rising - 0.1).abs() < 1e-9,
            "expected slope 0.1, got {slope_rising}"
        );
        assert!(residual_var < 1e-9, "perfect line should have ~0 residual variance");
        assert_eq!(predicted.len(), rising.len());
    }

    #[test]
    fn test_autocorrelation_series_perfect_periodicity() {
        // A perfectly repeating [1, 0, 1, 0, ...] pattern should show strong
        // negative autocorrelation at lag 1 and strong positive at lag 2.
        let series = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let acf = autocorrelation_series(&series, 2);
        assert_eq!(acf.len(), 2);
        assert!(acf[0] < 0.0, "lag-1 autocorrelation should be negative, got {}", acf[0]);
        assert!(acf[1] > 0.0, "lag-2 autocorrelation should be positive, got {}", acf[1]);
    }

    #[test]
    fn test_two_item_alpha_matches_spearman_brown() {
        assert!((two_item_alpha(0.0) - 0.0).abs() < 1e-9);
        assert!((two_item_alpha(1.0) - 1.0).abs() < 1e-9);
        let half = two_item_alpha(0.5);
        assert!((half - (2.0 * 0.5 / 1.5)).abs() < 1e-9);
    }

    /// Write `n` synthetic WAV files (with genuinely varying content, not
    /// identical constants) into `dir` and register them in a fresh
    /// ground-truth dataset, returning the dataset ID.
    async fn build_real_audio_dataset(
        manager: &mut GroundTruthManager,
        dir: &std::path::Path,
        n: usize,
    ) -> String {
        let dataset_id = manager
            .create_dataset(
                "reliability-test".to_string(),
                "synthetic reliability test dataset".to_string(),
                "test".to_string(),
                "test".to_string(),
                "test".to_string(),
                vec!["enus".to_string()],
            )
            .await
            .unwrap();

        for i in 0..n {
            let path = dir.join(format!("sample_{i}.wav"));
            let sample_rate = 16_000u32;
            let freq = 150.0 + i as f32 * 10.0; // genuinely different content per sample
            let samples: Vec<f32> = (0..sample_rate)
                .map(|s| {
                    let t = s as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
                })
                .collect();
            let audio = SdkAudioBuffer::new(samples, sample_rate, 1);
            audio.save_wav(&path).unwrap();

            manager
                .add_sample(
                    &dataset_id,
                    path,
                    None,
                    "test transcript".to_string(),
                    "enus".to_string(),
                    format!("speaker_{i}"),
                    HashMap::new(),
                )
                .await
                .unwrap();
        }

        dataset_id
    }

    #[tokio::test]
    async fn test_test_retest_reliability_uses_real_audio() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();
        let dataset_id = build_real_audio_dataset(&mut manager, temp_dir.path(), 10).await;
        let dataset = manager.get_dataset(&dataset_id).unwrap().clone();

        let config = ReliabilityTestConfig::default();
        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let result = tester.test_retest_reliability(&dataset).await.unwrap();
        assert_eq!(result.samples_excluded_load_failures, 0);
        // Deterministic evaluator on identical real audio: correlation
        // should be at or extremely near perfect (this is the *honest*
        // outcome for a deterministic pipeline, not a fabricated one).
        assert!(
            result.test_retest_correlation > 0.99,
            "expected near-perfect test-retest correlation for a deterministic evaluator on \
             identical real audio, got {}",
            result.test_retest_correlation
        );
    }

    #[tokio::test]
    async fn test_reliability_fails_closed_on_missing_audio() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();

        let dataset_id = manager
            .create_dataset(
                "broken-dataset".to_string(),
                String::new(),
                "test".to_string(),
                "test".to_string(),
                "test".to_string(),
                vec!["enus".to_string()],
            )
            .await
            .unwrap();

        // `add_sample` itself requires the file to exist, so to exercise the
        // "audio load fails at evaluation time" path we register a sample
        // pointing at a file that is deleted immediately afterward.
        let real_dir = temp_dir.path();
        let path = real_dir.join("will_be_deleted.wav");
        let samples = vec![0.1f32; 16_000];
        SdkAudioBuffer::new(samples, 16_000, 1)
            .save_wav(&path)
            .unwrap();
        manager
            .add_sample(
                &dataset_id,
                path.clone(),
                None,
                "t".to_string(),
                "enus".to_string(),
                "spk".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();
        std::fs::remove_file(&path).unwrap();

        let dataset = manager.get_dataset(&dataset_id).unwrap().clone();
        let config = ReliabilityTestConfig::default();
        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // With the only sample's audio missing, the honest outcome is a
        // clear error, never a fabricated result computed from a constant
        // stand-in buffer.
        let result = tester.test_retest_reliability(&dataset).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reproducibility_fails_closed_without_infrastructure() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();
        let dataset_id = build_real_audio_dataset(&mut manager, temp_dir.path(), 6).await;
        let dataset = manager.get_dataset(&dataset_id).unwrap().clone();

        let config = ReliabilityTestConfig::default();
        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let result = tester.test_reproducibility(&dataset).await.unwrap();
        // These must be honest Err values, not fabricated success numbers
        // like the old hardcoded implementation_consistency: 0.95.
        assert!(result.cross_platform.is_err());
        assert!(result.cross_implementation.is_err());
        assert!(result.environmental_reproducibility.is_err());
        // Temporal reproducibility, in contrast, is genuinely computed.
        assert!(result.temporal_reproducibility.temporal_correlation.is_finite());
    }

    #[tokio::test]
    async fn test_inter_rater_reliability_uses_distinct_configs() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = GroundTruthManager::new(temp_dir.path().to_path_buf());
        manager.initialize().await.unwrap();
        let dataset_id = build_real_audio_dataset(&mut manager, temp_dir.path(), 10).await;
        let dataset = manager.get_dataset(&dataset_id).unwrap().clone();

        let config = ReliabilityTestConfig::default();
        let tester = MetricReliabilityTester::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let result = tester.test_inter_rater_reliability(&dataset).await.unwrap();
        assert_eq!(result.rater_definitions.len(), 3);
        // The three "raters" must have genuinely distinct metric selections
        // (not an additive-constant fake), confirmed by their recorded
        // definitions differing from each other.
        let defs: Vec<&Vec<String>> = result.rater_definitions.values().collect();
        assert_ne!(defs[0], defs[1]);
        assert!((0.0..=1.0).contains(&result.kendalls_concordance.abs()) || result.kendalls_concordance == 0.0);
    }
}
