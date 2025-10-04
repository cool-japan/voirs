//! Regression Testing Framework for VoiRS Evaluation
//!
//! This module provides comprehensive regression testing capabilities to ensure
//! that metric calculations remain stable across code changes and system updates.

use crate::traits::{PronunciationEvaluator, QualityEvaluator as QualityEvaluatorTrait};
use crate::{
    pronunciation::PronunciationEvaluatorImpl, quality::QualityEvaluator, EvaluationError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Regression test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestConfig {
    /// Test data directory
    pub test_data_dir: PathBuf,
    /// Tolerance for metric comparison (relative error)
    pub tolerance: f32,
    /// Whether to update baselines when tests fail
    pub update_baselines: bool,
    /// Specific tests to run (empty = all tests)
    pub test_filter: Vec<String>,
    /// Enable parallel test execution
    pub parallel_execution: bool,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        Self {
            test_data_dir: PathBuf::from("test_data/regression"),
            tolerance: 0.001, // 0.1% tolerance
            update_baselines: false,
            test_filter: Vec::new(),
            parallel_execution: true,
        }
    }
}

/// Baseline metric values for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBaseline {
    /// Test case identifier
    pub test_case: String,
    /// Audio file identifier
    pub audio_file: String,
    /// Quality metrics baseline
    pub quality_metrics: HashMap<String, f32>,
    /// Pronunciation metrics baseline  
    pub pronunciation_metrics: HashMap<String, f32>,
    /// Timestamp when baseline was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// VoiRS version when baseline was created
    pub version: String,
}

/// Regression test result
#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    /// Test case name
    pub test_case: String,
    /// Whether the test passed
    pub passed: bool,
    /// Metric comparison results
    pub metric_comparisons: Vec<MetricComparison>,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Individual metric comparison result
#[derive(Debug, Clone)]
pub struct MetricComparison {
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f32,
    /// Current value
    pub current_value: f32,
    /// Relative error
    pub relative_error: f32,
    /// Whether this metric passed the tolerance test
    pub passed: bool,
}

/// Regression test suite results
#[derive(Debug, Clone)]
pub struct RegressionTestSuite {
    /// Individual test results
    pub test_results: Vec<RegressionTestResult>,
    /// Overall pass rate
    pub pass_rate: f32,
    /// Total execution time
    pub total_execution_time_ms: u64,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
}

/// Regression testing framework
pub struct RegressionTester {
    config: RegressionTestConfig,
    quality_evaluator: QualityEvaluator,
    pronunciation_evaluator: PronunciationEvaluatorImpl,
}

impl RegressionTester {
    /// Create a new regression tester
    pub async fn new(config: RegressionTestConfig) -> Result<Self, EvaluationError> {
        let quality_evaluator = QualityEvaluator::new().await?;
        let pronunciation_evaluator = PronunciationEvaluatorImpl::new().await?;

        Ok(Self {
            config,
            quality_evaluator,
            pronunciation_evaluator,
        })
    }

    /// Run the complete regression test suite
    pub async fn run_regression_tests(&self) -> Result<RegressionTestSuite, EvaluationError> {
        let start_time = std::time::Instant::now();
        let mut test_results = Vec::new();

        // Load test cases
        let test_cases = self.load_test_cases().await?;

        if self.config.parallel_execution {
            // Run tests in parallel
            use futures::future::join_all;

            let futures: Vec<_> = test_cases
                .into_iter()
                .filter(|case| self.should_run_test(&case.test_case))
                .map(|case| self.run_single_test(case))
                .collect();

            let results = join_all(futures).await;
            for result in results {
                test_results.push(result?);
            }
        } else {
            // Run tests sequentially
            for test_case in test_cases {
                if self.should_run_test(&test_case.test_case) {
                    let result = self.run_single_test(test_case).await?;
                    test_results.push(result);
                }
            }
        }

        let total_execution_time_ms = start_time.elapsed().as_millis() as u64;
        let passed_tests = test_results.iter().filter(|r| r.passed).count();
        let failed_tests = test_results.len() - passed_tests;
        let pass_rate = if test_results.is_empty() {
            0.0
        } else {
            passed_tests as f32 / test_results.len() as f32
        };

        Ok(RegressionTestSuite {
            test_results,
            pass_rate,
            total_execution_time_ms,
            passed_tests,
            failed_tests,
        })
    }

    /// Run a single regression test
    async fn run_single_test(
        &self,
        baseline: MetricBaseline,
    ) -> Result<RegressionTestResult, EvaluationError> {
        let start_time = std::time::Instant::now();
        let mut metric_comparisons = Vec::new();
        let mut all_passed = true;

        // Load test audio
        let audio_path = self.config.test_data_dir.join(&baseline.audio_file);
        let audio_buffer = self.load_test_audio(&audio_path).await?;

        // Test quality metrics
        match self.test_quality_metrics(&audio_buffer, &baseline).await {
            Ok(comparisons) => {
                for comparison in comparisons {
                    if !comparison.passed {
                        all_passed = false;
                    }
                    metric_comparisons.push(comparison);
                }
            }
            Err(e) => {
                return Ok(RegressionTestResult {
                    test_case: baseline.test_case,
                    passed: false,
                    metric_comparisons,
                    error_message: Some(format!("Quality metrics test failed: {}", e)),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                });
            }
        }

        // Test pronunciation metrics
        match self
            .test_pronunciation_metrics(&audio_buffer, &baseline)
            .await
        {
            Ok(comparisons) => {
                for comparison in comparisons {
                    if !comparison.passed {
                        all_passed = false;
                    }
                    metric_comparisons.push(comparison);
                }
            }
            Err(e) => {
                return Ok(RegressionTestResult {
                    test_case: baseline.test_case,
                    passed: false,
                    metric_comparisons,
                    error_message: Some(format!("Pronunciation metrics test failed: {}", e)),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                });
            }
        }

        Ok(RegressionTestResult {
            test_case: baseline.test_case,
            passed: all_passed,
            metric_comparisons,
            error_message: None,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Test quality metrics against baseline
    async fn test_quality_metrics(
        &self,
        audio: &AudioBuffer,
        baseline: &MetricBaseline,
    ) -> Result<Vec<MetricComparison>, EvaluationError> {
        let mut comparisons = Vec::new();

        // Evaluate current quality metrics
        let quality_result = self
            .quality_evaluator
            .evaluate_quality(audio, None, None)
            .await?;

        // Compare each baseline quality metric
        for (metric_name, baseline_value) in &baseline.quality_metrics {
            let current_value = match metric_name.as_str() {
                "overall_score" => quality_result.overall_score,
                "confidence" => quality_result.confidence,
                _ => {
                    // Check component scores
                    if let Some(&score) = quality_result.component_scores.get(metric_name) {
                        score
                    } else {
                        continue; // Skip unknown metrics
                    }
                }
            };

            let relative_error = if *baseline_value != 0.0 {
                ((current_value - baseline_value) / baseline_value).abs()
            } else {
                current_value.abs()
            };

            let passed = relative_error <= self.config.tolerance;

            comparisons.push(MetricComparison {
                metric_name: metric_name.clone(),
                baseline_value: *baseline_value,
                current_value,
                relative_error,
                passed,
            });
        }

        Ok(comparisons)
    }

    /// Test pronunciation metrics against baseline
    async fn test_pronunciation_metrics(
        &self,
        audio: &AudioBuffer,
        baseline: &MetricBaseline,
    ) -> Result<Vec<MetricComparison>, EvaluationError> {
        let mut comparisons = Vec::new();

        // Create mock transcript for pronunciation evaluation
        let mock_transcript = "hello world test audio".to_string();

        // Evaluate current pronunciation metrics
        let pronunciation_result = self
            .pronunciation_evaluator
            .evaluate_pronunciation(audio, &mock_transcript, None)
            .await?;

        // Compare each baseline pronunciation metric
        for (metric_name, baseline_value) in &baseline.pronunciation_metrics {
            let current_value = match metric_name.as_str() {
                "overall_score" => pronunciation_result.overall_score,
                "fluency_score" => pronunciation_result.fluency_score,
                "rhythm_score" => pronunciation_result.rhythm_score,
                "stress_accuracy" => pronunciation_result.stress_accuracy,
                "intonation_accuracy" => pronunciation_result.intonation_accuracy,
                _ => continue, // Skip unknown metrics
            };

            let relative_error = if *baseline_value != 0.0 {
                ((current_value - baseline_value) / baseline_value).abs()
            } else {
                current_value.abs()
            };

            let passed = relative_error <= self.config.tolerance;

            comparisons.push(MetricComparison {
                metric_name: metric_name.clone(),
                baseline_value: *baseline_value,
                current_value,
                relative_error,
                passed,
            });
        }

        Ok(comparisons)
    }

    /// Load test cases from baseline files
    async fn load_test_cases(&self) -> Result<Vec<MetricBaseline>, EvaluationError> {
        let mut test_cases = Vec::new();

        // Create test data directory if it doesn't exist
        if !self.config.test_data_dir.exists() {
            fs::create_dir_all(&self.config.test_data_dir)
                .await
                .map_err(|e| EvaluationError::ProcessingError {
                    message: format!("Failed to create test data directory: {}", e),
                    source: Some(Box::new(e)),
                })?;

            // Generate default test cases
            return self.generate_default_test_cases().await;
        }

        // Read baseline files
        let mut entries = fs::read_dir(&self.config.test_data_dir)
            .await
            .map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to read test data directory: {}", e),
                source: Some(Box::new(e)),
            })?;

        while let Some(entry) =
            entries
                .next_entry()
                .await
                .map_err(|e| EvaluationError::ProcessingError {
                    message: format!("Failed to read directory entry: {}", e),
                    source: Some(Box::new(e)),
                })?
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(&path).await.map_err(|e| {
                    EvaluationError::ProcessingError {
                        message: format!("Failed to read baseline file: {}", e),
                        source: Some(Box::new(e)),
                    }
                })?;

                let baseline: MetricBaseline = serde_json::from_str(&content).map_err(|e| {
                    EvaluationError::ProcessingError {
                        message: format!("Failed to parse baseline file: {}", e),
                        source: Some(Box::new(e)),
                    }
                })?;

                test_cases.push(baseline);
            }
        }

        if test_cases.is_empty() {
            test_cases = self.generate_default_test_cases().await?;
        }

        Ok(test_cases)
    }

    /// Generate default test cases for regression testing
    async fn generate_default_test_cases(&self) -> Result<Vec<MetricBaseline>, EvaluationError> {
        let mut test_cases = Vec::new();

        // Generate synthetic test audio
        let test_audios = self.generate_test_audio().await?;

        for (i, audio) in test_audios.iter().enumerate() {
            let test_case_name = format!("synthetic_test_{}", i);
            let audio_file = format!("test_audio_{}.wav", i);

            // Evaluate metrics to create baseline
            let quality_result = self
                .quality_evaluator
                .evaluate_quality(audio, None, None)
                .await?;

            let pronunciation_result = self
                .pronunciation_evaluator
                .evaluate_pronunciation(audio, "test audio sample", None)
                .await?;

            // Create baseline
            let mut quality_metrics = HashMap::new();
            quality_metrics.insert("overall_score".to_string(), quality_result.overall_score);
            quality_metrics.insert("confidence".to_string(), quality_result.confidence);

            // Add component scores
            for (key, value) in &quality_result.component_scores {
                quality_metrics.insert(key.clone(), *value);
            }

            let mut pronunciation_metrics = HashMap::new();
            pronunciation_metrics.insert(
                "overall_score".to_string(),
                pronunciation_result.overall_score,
            );
            pronunciation_metrics.insert(
                "fluency_score".to_string(),
                pronunciation_result.fluency_score,
            );
            pronunciation_metrics.insert(
                "rhythm_score".to_string(),
                pronunciation_result.rhythm_score,
            );
            pronunciation_metrics.insert(
                "stress_accuracy".to_string(),
                pronunciation_result.stress_accuracy,
            );
            pronunciation_metrics.insert(
                "intonation_accuracy".to_string(),
                pronunciation_result.intonation_accuracy,
            );

            let baseline = MetricBaseline {
                test_case: test_case_name,
                audio_file,
                quality_metrics,
                pronunciation_metrics,
                created_at: chrono::Utc::now(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            };

            test_cases.push(baseline);
        }

        // Save baselines to files
        for baseline in &test_cases {
            let baseline_path = self
                .config
                .test_data_dir
                .join(format!("{}.json", baseline.test_case));
            let baseline_json = serde_json::to_string_pretty(baseline).map_err(|e| {
                EvaluationError::ProcessingError {
                    message: format!("Failed to serialize baseline: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

            fs::write(&baseline_path, baseline_json)
                .await
                .map_err(|e| EvaluationError::ProcessingError {
                    message: format!("Failed to write baseline file: {}", e),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(test_cases)
    }

    /// Generate synthetic test audio for regression testing
    async fn generate_test_audio(&self) -> Result<Vec<AudioBuffer>, EvaluationError> {
        let mut test_audios = Vec::new();
        let sample_rate = 16000;
        let duration_samples = sample_rate; // 1 second

        // Generate different types of test signals

        // 1. Pure sine wave
        let mut sine_samples = Vec::with_capacity(duration_samples);
        for i in 0..duration_samples {
            let t = i as f32 / sample_rate as f32;
            sine_samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        test_audios.push(AudioBuffer::new(sine_samples, sample_rate as u32, 1));

        // 2. White noise
        let noise_samples: Vec<f32> = (0..duration_samples)
            .map(|_| (scirs2_core::random::random::<f32>() - 0.5) * 0.3)
            .collect();
        test_audios.push(AudioBuffer::new(noise_samples, sample_rate as u32, 1));

        // 3. Mixed signal (sine + noise)
        let mut mixed_samples = Vec::with_capacity(duration_samples);
        for i in 0..duration_samples {
            let t = i as f32 / sample_rate as f32;
            let sine = 0.7 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            let noise = (scirs2_core::random::random::<f32>() - 0.5) * 0.1;
            mixed_samples.push(sine + noise);
        }
        test_audios.push(AudioBuffer::new(mixed_samples, sample_rate as u32, 1));

        Ok(test_audios)
    }

    /// Load test audio from file
    async fn load_test_audio(&self, path: &Path) -> Result<AudioBuffer, EvaluationError> {
        // For now, generate synthetic audio since we don't have actual files
        // In a real implementation, this would load the actual audio file
        let sample_rate = 16000;
        let duration_samples = sample_rate; // 1 second

        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        Ok(AudioBuffer::new(samples, sample_rate as u32, 1))
    }

    /// Check if a test should be run based on filter
    fn should_run_test(&self, test_case: &str) -> bool {
        if self.config.test_filter.is_empty() {
            return true;
        }

        self.config
            .test_filter
            .iter()
            .any(|filter| test_case.contains(filter))
    }

    /// Create a regression test report
    pub fn create_report(&self, suite: &RegressionTestSuite) -> String {
        let mut report = String::new();

        report.push_str("# VoiRS Evaluation Regression Test Report\n\n");
        report.push_str(&format!("**Overall Results:**\n"));
        report.push_str(&format!("- Pass Rate: {:.1}%\n", suite.pass_rate * 100.0));
        report.push_str(&format!("- Passed Tests: {}\n", suite.passed_tests));
        report.push_str(&format!("- Failed Tests: {}\n", suite.failed_tests));
        report.push_str(&format!(
            "- Total Execution Time: {}ms\n\n",
            suite.total_execution_time_ms
        ));

        if suite.failed_tests > 0 {
            report.push_str("## Failed Tests\n\n");
            for test_result in &suite.test_results {
                if !test_result.passed {
                    report.push_str(&format!("### {}\n", test_result.test_case));
                    if let Some(error) = &test_result.error_message {
                        report.push_str(&format!("**Error:** {}\n\n", error));
                    }

                    report.push_str("**Failed Metrics:**\n");
                    for comparison in &test_result.metric_comparisons {
                        if !comparison.passed {
                            report.push_str(&format!(
                                "- {}: baseline={:.4}, current={:.4}, error={:.2}%\n",
                                comparison.metric_name,
                                comparison.baseline_value,
                                comparison.current_value,
                                comparison.relative_error * 100.0
                            ));
                        }
                    }
                    report.push_str("\n");
                }
            }
        }

        report.push_str("## Test Summary\n\n");
        for test_result in &suite.test_results {
            let status = if test_result.passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            report.push_str(&format!(
                "- {} {} ({}ms)\n",
                status, test_result.test_case, test_result.execution_time_ms
            ));
        }

        report
    }
}

impl Default for RegressionTester {
    fn default() -> Self {
        panic!("RegressionTester cannot use Default::default() - use RegressionTester::new() instead due to async initialization requirements")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_regression_tester_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = RegressionTestConfig {
            test_data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let tester = RegressionTester::new(config).await.unwrap();
        assert_eq!(tester.config.tolerance, 0.001);
    }

    #[tokio::test]
    async fn test_generate_test_audio() {
        let temp_dir = TempDir::new().unwrap();
        let config = RegressionTestConfig {
            test_data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let tester = RegressionTester::new(config).await.unwrap();
        let test_audios = tester.generate_test_audio().await.unwrap();

        assert_eq!(test_audios.len(), 3);
        for audio in test_audios {
            assert_eq!(audio.sample_rate(), 16000);
            assert_eq!(audio.channels(), 1);
            assert_eq!(audio.samples().len(), 16000);
        }
    }

    #[tokio::test]
    async fn test_should_run_test() {
        let temp_dir = TempDir::new().unwrap();
        let config = RegressionTestConfig {
            test_data_dir: temp_dir.path().to_path_buf(),
            test_filter: vec!["synthetic".to_string()],
            ..Default::default()
        };

        let tester = RegressionTester::new(config).await.unwrap();

        assert!(tester.should_run_test("synthetic_test_1"));
        assert!(!tester.should_run_test("real_test_1"));
    }

    #[test]
    fn test_metric_comparison() {
        let comparison = MetricComparison {
            metric_name: "test_metric".to_string(),
            baseline_value: 1.0,
            current_value: 1.001,
            relative_error: 0.001,
            passed: true,
        };

        assert_eq!(comparison.metric_name, "test_metric");
        assert!(comparison.passed);
    }

    #[test]
    fn test_regression_test_suite() {
        let test_results = vec![
            RegressionTestResult {
                test_case: "test1".to_string(),
                passed: true,
                metric_comparisons: vec![],
                error_message: None,
                execution_time_ms: 100,
            },
            RegressionTestResult {
                test_case: "test2".to_string(),
                passed: false,
                metric_comparisons: vec![],
                error_message: Some("Test failed".to_string()),
                execution_time_ms: 150,
            },
        ];

        let suite = RegressionTestSuite {
            test_results,
            pass_rate: 0.5,
            total_execution_time_ms: 250,
            passed_tests: 1,
            failed_tests: 1,
        };

        assert_eq!(suite.pass_rate, 0.5);
        assert_eq!(suite.passed_tests, 1);
        assert_eq!(suite.failed_tests, 1);
    }
}
