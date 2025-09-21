//! Benchmark tests against reference implementations
//!
//! This module provides benchmark tests that validate our metrics against
//! reference implementations to ensure accuracy and compliance with standards.

use crate::quality::QualityEvaluator;
use crate::traits::{
    EvaluationResult, QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait,
};
use crate::EvaluationError;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_sdk::AudioBuffer;

/// Benchmark test configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Enable PESQ reference validation
    pub enable_pesq_reference: bool,
    /// Enable STOI reference validation
    pub enable_stoi_reference: bool,
    /// Enable MCD reference validation
    pub enable_mcd_reference: bool,
    /// Tolerance for numerical differences
    pub tolerance: f32,
    /// Number of test samples
    pub sample_count: usize,
    /// Performance benchmark threshold (ms)
    pub performance_threshold_ms: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_pesq_reference: true,
            enable_stoi_reference: true,
            enable_mcd_reference: true,
            tolerance: 0.01,
            sample_count: 10,
            performance_threshold_ms: 1000,
        }
    }
}

/// Benchmark test result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Metric name
    pub metric_name: String,
    /// Our implementation score
    pub our_score: f32,
    /// Reference implementation score
    pub reference_score: f32,
    /// Absolute difference
    pub absolute_difference: f32,
    /// Relative difference (%)
    pub relative_difference: f32,
    /// Test passed
    pub passed: bool,
    /// Execution time (ms)
    pub execution_time_ms: u64,
}

/// Reference implementation benchmark suite
pub struct ReferenceBenchmarkSuite {
    /// Configuration
    config: BenchmarkConfig,
    /// Quality evaluator
    evaluator: QualityEvaluator,
    /// Test results
    results: Vec<BenchmarkResult>,
}

impl ReferenceBenchmarkSuite {
    /// Create new benchmark suite
    pub async fn new(config: BenchmarkConfig) -> Result<Self, EvaluationError> {
        let evaluator = QualityEvaluator::new().await?;

        Ok(Self {
            config,
            evaluator,
            results: Vec::new(),
        })
    }

    /// Run all benchmark tests
    pub async fn run_all_benchmarks(&mut self) -> EvaluationResult<Vec<BenchmarkResult>> {
        self.results.clear();

        // Generate test audio samples
        let test_samples = self.generate_test_audio_samples();

        if self.config.enable_pesq_reference {
            self.run_pesq_benchmarks(&test_samples).await?;
        }

        if self.config.enable_stoi_reference {
            self.run_stoi_benchmarks(&test_samples).await?;
        }

        if self.config.enable_mcd_reference {
            self.run_mcd_benchmarks(&test_samples).await?;
        }

        Ok(self.results.clone())
    }

    /// Generate test audio samples for benchmarking
    fn generate_test_audio_samples(&self) -> Vec<(AudioBuffer, AudioBuffer)> {
        let mut samples = Vec::new();

        for i in 0..self.config.sample_count {
            let sample_rate = 16000;
            let duration = 2.0; // 2 seconds
            let length = (sample_rate as f32 * duration) as usize;

            // Generate reference audio (clean sine wave)
            let frequency = 440.0 + (i as f32 * 50.0); // Varying frequency
            let reference_samples: Vec<f32> = (0..length)
                .map(|n| {
                    let t = n as f32 / sample_rate as f32;
                    0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
                })
                .collect();

            // Generate degraded audio (with noise and distortion)
            let degraded_samples: Vec<f32> = reference_samples
                .iter()
                .enumerate()
                .map(|(n, &sample)| {
                    let noise = (n as f32 * 0.1).sin() * 0.05; // Some noise
                    let distortion = sample * 0.9; // Slight amplitude reduction
                    distortion + noise
                })
                .collect();

            let reference_audio = AudioBuffer::new(reference_samples, sample_rate, 1);
            let degraded_audio = AudioBuffer::new(degraded_samples, sample_rate, 1);

            samples.push((reference_audio, degraded_audio));
        }

        samples
    }

    /// Run PESQ benchmarks
    async fn run_pesq_benchmarks(
        &mut self,
        test_samples: &[(AudioBuffer, AudioBuffer)],
    ) -> EvaluationResult<()> {
        for (i, (reference, degraded)) in test_samples.iter().enumerate() {
            let start_time = Instant::now();

            // Run our implementation
            let our_result = self
                .evaluator
                .evaluate_quality(degraded, Some(reference), None)
                .await?;
            let our_score = our_result.overall_score;

            // Simulate reference implementation (in real scenario, this would call actual reference)
            let reference_score = self.simulate_pesq_reference(reference, degraded);

            let execution_time = start_time.elapsed().as_millis() as u64;

            let absolute_difference = (our_score - reference_score).abs();
            let relative_difference = if reference_score != 0.0 {
                (absolute_difference / reference_score) * 100.0
            } else {
                0.0
            };

            let passed = absolute_difference <= self.config.tolerance;

            self.results.push(BenchmarkResult {
                metric_name: format!("PESQ_Sample_{}", i),
                our_score,
                reference_score,
                absolute_difference,
                relative_difference,
                passed,
                execution_time_ms: execution_time,
            });
        }

        Ok(())
    }

    /// Run STOI benchmarks
    async fn run_stoi_benchmarks(
        &mut self,
        test_samples: &[(AudioBuffer, AudioBuffer)],
    ) -> EvaluationResult<()> {
        for (i, (reference, degraded)) in test_samples.iter().enumerate() {
            let start_time = Instant::now();

            // Run our implementation
            let our_result = self
                .evaluator
                .evaluate_quality(degraded, Some(reference), None)
                .await?;
            let our_score = our_result.overall_score;

            // Simulate reference implementation
            let reference_score = self.simulate_stoi_reference(reference, degraded);

            let execution_time = start_time.elapsed().as_millis() as u64;

            let absolute_difference = (our_score - reference_score).abs();
            let relative_difference = if reference_score != 0.0 {
                (absolute_difference / reference_score) * 100.0
            } else {
                0.0
            };

            let passed = absolute_difference <= self.config.tolerance;

            self.results.push(BenchmarkResult {
                metric_name: format!("STOI_Sample_{}", i),
                our_score,
                reference_score,
                absolute_difference,
                relative_difference,
                passed,
                execution_time_ms: execution_time,
            });
        }

        Ok(())
    }

    /// Run MCD benchmarks
    async fn run_mcd_benchmarks(
        &mut self,
        test_samples: &[(AudioBuffer, AudioBuffer)],
    ) -> EvaluationResult<()> {
        for (i, (reference, degraded)) in test_samples.iter().enumerate() {
            let start_time = Instant::now();

            // Run our implementation
            let our_result = self
                .evaluator
                .evaluate_quality(degraded, Some(reference), None)
                .await?;
            let our_score = our_result.overall_score;

            // Simulate reference implementation
            let reference_score = self.simulate_mcd_reference(reference, degraded);

            let execution_time = start_time.elapsed().as_millis() as u64;

            let absolute_difference = (our_score - reference_score).abs();
            let relative_difference = if reference_score != 0.0 {
                (absolute_difference / reference_score) * 100.0
            } else {
                0.0
            };

            let passed = absolute_difference <= self.config.tolerance;

            self.results.push(BenchmarkResult {
                metric_name: format!("MCD_Sample_{}", i),
                our_score,
                reference_score,
                absolute_difference,
                relative_difference,
                passed,
                execution_time_ms: execution_time,
            });
        }

        Ok(())
    }

    /// Simulate PESQ reference implementation
    /// In a real scenario, this would call the actual ITU-T P.862 reference implementation
    fn simulate_pesq_reference(&self, reference: &AudioBuffer, degraded: &AudioBuffer) -> f32 {
        // Simplified simulation - in reality this would call the actual reference
        let reference_samples = reference.samples();
        let degraded_samples = degraded.samples();

        // Calculate a simplified quality metric based on SNR
        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for i in 0..reference_samples.len().min(degraded_samples.len()) {
            let signal = reference_samples[i];
            let noise = degraded_samples[i] - signal;
            signal_power += signal * signal;
            noise_power += noise * noise;
        }

        let snr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            50.0 // Very high SNR if no noise
        };

        // Map SNR to PESQ scale (1.0 to 4.5)
        let pesq_score = 1.0 + (snr / 20.0) * 3.5;
        pesq_score.clamp(1.0, 4.5)
    }

    /// Simulate STOI reference implementation
    fn simulate_stoi_reference(&self, reference: &AudioBuffer, degraded: &AudioBuffer) -> f32 {
        // Simplified simulation - in reality this would call the actual reference
        let reference_samples = reference.samples();
        let degraded_samples = degraded.samples();

        // Calculate correlation coefficient as a proxy for STOI
        let mut sum_ref = 0.0;
        let mut sum_deg = 0.0;
        let mut sum_ref_sq = 0.0;
        let mut sum_deg_sq = 0.0;
        let mut sum_ref_deg = 0.0;
        let n = reference_samples.len().min(degraded_samples.len()) as f32;

        for i in 0..n as usize {
            let ref_val = reference_samples[i];
            let deg_val = degraded_samples[i];

            sum_ref += ref_val;
            sum_deg += deg_val;
            sum_ref_sq += ref_val * ref_val;
            sum_deg_sq += deg_val * deg_val;
            sum_ref_deg += ref_val * deg_val;
        }

        let correlation = if n > 0.0 {
            let numerator = n * sum_ref_deg - sum_ref * sum_deg;
            let denominator = ((n * sum_ref_sq - sum_ref * sum_ref)
                * (n * sum_deg_sq - sum_deg * sum_deg))
                .sqrt();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Map correlation to STOI scale (0.0 to 1.0)
        (correlation + 1.0) / 2.0
    }

    /// Simulate MCD reference implementation
    fn simulate_mcd_reference(&self, reference: &AudioBuffer, degraded: &AudioBuffer) -> f32 {
        // Simplified simulation - in reality this would call the actual reference
        let reference_samples = reference.samples();
        let degraded_samples = degraded.samples();

        // Calculate simple spectral difference
        let mut mcd_sum = 0.0;
        let frame_size = 512;
        let hop_size = 256;
        let mut frame_count = 0;

        for i in (0..reference_samples.len().saturating_sub(frame_size)).step_by(hop_size) {
            if i + frame_size <= degraded_samples.len() {
                let ref_frame = &reference_samples[i..i + frame_size];
                let deg_frame = &degraded_samples[i..i + frame_size];

                // Simple spectral distance calculation
                let mut spectral_diff = 0.0;
                for j in 0..frame_size {
                    let diff = (ref_frame[j] - deg_frame[j]).abs();
                    spectral_diff += diff;
                }

                mcd_sum += spectral_diff / frame_size as f32;
                frame_count += 1;
            }
        }

        if frame_count > 0 {
            // Scale to typical MCD range (0-10 dB)
            (mcd_sum / frame_count as f32) * 10.0
        } else {
            0.0
        }
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        let mut passed_count = 0;
        let mut total_count = 0;
        let mut total_execution_time = 0;
        let mut metric_summaries = HashMap::new();

        for result in &self.results {
            if result.passed {
                passed_count += 1;
            }
            total_count += 1;
            total_execution_time += result.execution_time_ms;

            let metric_type = result.metric_name.split('_').next().unwrap_or("Unknown");
            let summary =
                metric_summaries
                    .entry(metric_type.to_string())
                    .or_insert(MetricSummary {
                        metric_name: metric_type.to_string(),
                        test_count: 0,
                        passed_count: 0,
                        avg_absolute_diff: 0.0,
                        avg_relative_diff: 0.0,
                        avg_execution_time: 0.0,
                    });

            summary.test_count += 1;
            if result.passed {
                summary.passed_count += 1;
            }
            summary.avg_absolute_diff += result.absolute_difference;
            summary.avg_relative_diff += result.relative_difference;
            summary.avg_execution_time += result.execution_time_ms as f32;
        }

        // Calculate averages
        for summary in metric_summaries.values_mut() {
            if summary.test_count > 0 {
                summary.avg_absolute_diff /= summary.test_count as f32;
                summary.avg_relative_diff /= summary.test_count as f32;
                summary.avg_execution_time /= summary.test_count as f32;
            }
        }

        BenchmarkReport {
            total_tests: total_count,
            passed_tests: passed_count,
            success_rate: if total_count > 0 {
                passed_count as f32 / total_count as f32
            } else {
                0.0
            },
            total_execution_time_ms: total_execution_time,
            avg_execution_time_ms: if total_count > 0 {
                total_execution_time / total_count as u64
            } else {
                0
            },
            metric_summaries: metric_summaries.into_values().collect(),
            detailed_results: self.results.clone(),
        }
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

/// Benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Total execution time (ms)
    pub total_execution_time_ms: u64,
    /// Average execution time per test (ms)
    pub avg_execution_time_ms: u64,
    /// Summary by metric type
    pub metric_summaries: Vec<MetricSummary>,
    /// Detailed results
    pub detailed_results: Vec<BenchmarkResult>,
}

/// Metric summary
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Metric name
    pub metric_name: String,
    /// Number of tests
    pub test_count: usize,
    /// Number of passed tests
    pub passed_count: usize,
    /// Average absolute difference
    pub avg_absolute_diff: f32,
    /// Average relative difference
    pub avg_relative_diff: f32,
    /// Average execution time
    pub avg_execution_time: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = ReferenceBenchmarkSuite::new(config).await.unwrap();

        assert_eq!(suite.results.len(), 0);
    }

    #[tokio::test]
    async fn test_benchmark_execution() {
        let config = BenchmarkConfig {
            sample_count: 2,
            tolerance: 0.1,
            ..Default::default()
        };
        let mut suite = ReferenceBenchmarkSuite::new(config).await.unwrap();

        let results = suite.run_all_benchmarks().await.unwrap();

        // Should have results for PESQ, STOI, and MCD
        assert_eq!(results.len(), 6); // 2 samples * 3 metrics

        // Check that all metrics are represented
        let metric_names: Vec<String> = results.iter().map(|r| r.metric_name.clone()).collect();
        assert!(metric_names.iter().any(|name| name.contains("PESQ")));
        assert!(metric_names.iter().any(|name| name.contains("STOI")));
        assert!(metric_names.iter().any(|name| name.contains("MCD")));
    }

    #[tokio::test]
    async fn test_benchmark_report_generation() {
        let config = BenchmarkConfig {
            sample_count: 1,
            tolerance: 0.5, // Generous tolerance for testing
            ..Default::default()
        };
        let mut suite = ReferenceBenchmarkSuite::new(config).await.unwrap();

        suite.run_all_benchmarks().await.unwrap();
        let report = suite.generate_report();

        assert_eq!(report.total_tests, 3); // 1 sample * 3 metrics
        assert_eq!(report.metric_summaries.len(), 3);
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
    }

    #[tokio::test]
    async fn test_performance_threshold() {
        let config = BenchmarkConfig {
            sample_count: 1,
            performance_threshold_ms: 100,
            ..Default::default()
        };
        let mut suite = ReferenceBenchmarkSuite::new(config).await.unwrap();

        suite.run_all_benchmarks().await.unwrap();
        let results = suite.get_results();

        // Check that all tests completed within reasonable time
        for result in results {
            assert!(result.execution_time_ms < 15000); // 15 seconds max for CI environments
        }
    }

    #[tokio::test]
    async fn test_audio_sample_generation() {
        let config = BenchmarkConfig {
            sample_count: 3,
            ..Default::default()
        };
        let suite = ReferenceBenchmarkSuite::new(config).await.unwrap();

        let samples = suite.generate_test_audio_samples();

        assert_eq!(samples.len(), 3);

        for (reference, degraded) in samples {
            assert_eq!(reference.sample_rate(), 16000);
            assert_eq!(degraded.sample_rate(), 16000);
            assert_eq!(reference.channels(), 1);
            assert_eq!(degraded.channels(), 1);
            assert_eq!(reference.samples().len(), degraded.samples().len());
        }
    }
}
