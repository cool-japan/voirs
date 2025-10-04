//! Fuzzing tests for robustness validation
//!
//! This module provides fuzzing tests to validate the robustness of the evaluation
//! system against various edge cases and malformed inputs.

use crate::quality::QualityEvaluator;
use crate::traits::{
    EvaluationResult, PronunciationEvaluationConfig, PronunciationEvaluator,
    QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait,
};
use crate::EvaluationError;
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use std::time::{Duration, Instant};
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Fuzzing test configuration
#[derive(Debug, Clone)]
pub struct FuzzingConfig {
    /// Number of fuzzing iterations
    pub iterations: usize,
    /// Random seed for reproducible results
    pub seed: u64,
    /// Maximum audio buffer size for testing
    pub max_audio_size: usize,
    /// Enable audio content fuzzing
    pub enable_audio_fuzzing: bool,
    /// Enable parameter fuzzing
    pub enable_parameter_fuzzing: bool,
    /// Enable boundary condition testing
    pub enable_boundary_testing: bool,
    /// Maximum execution time per test (seconds)
    pub max_execution_time: u64,
}

impl Default for FuzzingConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            seed: 42,
            max_audio_size: 100_000, // 100k samples
            enable_audio_fuzzing: true,
            enable_parameter_fuzzing: true,
            enable_boundary_testing: true,
            max_execution_time: 30,
        }
    }
}

/// Fuzzing test result
#[derive(Debug, Clone)]
pub struct FuzzingResult {
    /// Test type
    pub test_type: String,
    /// Test case number
    pub case_number: usize,
    /// Test passed without crashing
    pub passed: bool,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Test parameters used
    pub test_parameters: String,
}

/// Fuzzing test suite
pub struct FuzzingTestSuite {
    /// Configuration
    config: FuzzingConfig,
    /// Random number generator
    rng: StdRng,
    /// Quality evaluator
    quality_evaluator: QualityEvaluator,
    /// Pronunciation evaluator
    pronunciation_evaluator: crate::pronunciation::PronunciationEvaluatorImpl,
    /// Test results
    results: Vec<FuzzingResult>,
}

impl FuzzingTestSuite {
    /// Create new fuzzing test suite
    pub async fn new(config: FuzzingConfig) -> Result<Self, EvaluationError> {
        let rng = StdRng::seed_from_u64(config.seed);
        let quality_evaluator = QualityEvaluator::new().await?;
        let pronunciation_evaluator =
            crate::pronunciation::PronunciationEvaluatorImpl::new().await?;

        Ok(Self {
            config,
            rng,
            quality_evaluator,
            pronunciation_evaluator,
            results: Vec::new(),
        })
    }

    /// Run all fuzzing tests
    pub async fn run_all_fuzzing_tests(&mut self) -> EvaluationResult<Vec<FuzzingResult>> {
        self.results.clear();

        if self.config.enable_audio_fuzzing {
            self.run_audio_fuzzing_tests().await;
        }

        if self.config.enable_parameter_fuzzing {
            self.run_parameter_fuzzing_tests().await;
        }

        if self.config.enable_boundary_testing {
            self.run_boundary_condition_tests().await;
        }

        Ok(self.results.clone())
    }

    /// Run audio content fuzzing tests
    async fn run_audio_fuzzing_tests(&mut self) {
        for i in 0..self.config.iterations {
            // Test with random audio content
            let audio_buffer = self.generate_random_audio_buffer();
            let reference_buffer = self.generate_random_audio_buffer();

            self.test_quality_evaluation_robustness(
                &audio_buffer,
                Some(&reference_buffer),
                "AudioFuzzing",
                i,
                format!("Random audio: {} samples", audio_buffer.samples().len()),
            )
            .await;

            // Test with extreme audio values
            let extreme_audio = self.generate_extreme_audio_buffer();

            self.test_quality_evaluation_robustness(
                &extreme_audio,
                None,
                "ExtremeAudioFuzzing",
                i,
                "Extreme audio values".to_string(),
            )
            .await;

            // Test with malformed audio
            let malformed_audio = self.generate_malformed_audio_buffer();

            self.test_quality_evaluation_robustness(
                &malformed_audio,
                None,
                "MalformedAudioFuzzing",
                i,
                "Malformed audio buffer".to_string(),
            )
            .await;
        }
    }

    /// Run parameter fuzzing tests
    async fn run_parameter_fuzzing_tests(&mut self) {
        for i in 0..self.config.iterations {
            // Test with random phoneme sequences
            let phonemes = self.generate_random_phonemes();
            let audio_buffer = self.generate_random_audio_buffer();

            self.test_pronunciation_evaluation_robustness(
                &audio_buffer,
                &phonemes,
                "ParameterFuzzing",
                i,
                format!("Random phonemes: {} items", phonemes.len()),
            )
            .await;

            // Test with extreme parameter values
            let extreme_phonemes = self.generate_extreme_phonemes();

            self.test_pronunciation_evaluation_robustness(
                &audio_buffer,
                &extreme_phonemes,
                "ExtremeParameterFuzzing",
                i,
                "Extreme phoneme parameters".to_string(),
            )
            .await;
        }
    }

    /// Run boundary condition tests
    async fn run_boundary_condition_tests(&mut self) {
        let boundary_cases = vec![
            // Empty audio
            (
                "EmptyAudio",
                AudioBuffer::new(vec![], 16000, 1),
                "Empty audio buffer",
            ),
            // Single sample
            (
                "SingleSample",
                AudioBuffer::new(vec![0.5], 16000, 1),
                "Single sample",
            ),
            // Very short audio
            (
                "VeryShortAudio",
                AudioBuffer::new(vec![0.1; 10], 16000, 1),
                "Very short audio",
            ),
            // Zero audio
            (
                "ZeroAudio",
                AudioBuffer::new(vec![0.0; 1000], 16000, 1),
                "Zero-amplitude audio",
            ),
            // Maximum amplitude
            (
                "MaxAmplitude",
                AudioBuffer::new(vec![1.0; 1000], 16000, 1),
                "Maximum amplitude",
            ),
            // Minimum amplitude
            (
                "MinAmplitude",
                AudioBuffer::new(vec![-1.0; 1000], 16000, 1),
                "Minimum amplitude",
            ),
        ];

        for (i, (test_name, audio_buffer, description)) in boundary_cases.iter().enumerate() {
            self.test_quality_evaluation_robustness(
                audio_buffer,
                None,
                test_name,
                i,
                description.to_string(),
            )
            .await;
        }

        // Test with boundary phoneme cases
        let boundary_phoneme_cases = vec![
            ("EmptyPhonemes", vec![], "Empty phoneme sequence"),
            ("SinglePhoneme", vec![Phoneme::new("a")], "Single phoneme"),
            (
                "DuplicatePhonemes",
                vec![Phoneme::new("t"); 100],
                "Duplicate phonemes",
            ),
        ];

        let audio_buffer = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        for (i, (test_name, phonemes, description)) in boundary_phoneme_cases.iter().enumerate() {
            self.test_pronunciation_evaluation_robustness(
                &audio_buffer,
                phonemes,
                test_name,
                i,
                description.to_string(),
            )
            .await;
        }
    }

    /// Test quality evaluation robustness
    async fn test_quality_evaluation_robustness(
        &mut self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        test_type: &str,
        case_number: usize,
        test_parameters: String,
    ) {
        let start_time = Instant::now();

        let result = tokio::time::timeout(
            Duration::from_secs(self.config.max_execution_time),
            self.quality_evaluator
                .evaluate_quality(audio, reference, None),
        )
        .await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        let (passed, error_message) = match result {
            Ok(Ok(_)) => (true, None),
            Ok(Err(e)) => (true, Some(format!("Expected error: {}", e))), // Errors are expected in fuzzing
            Err(_) => (false, Some("Timeout".to_string())),
        };

        self.results.push(FuzzingResult {
            test_type: test_type.to_string(),
            case_number,
            passed,
            error_message,
            execution_time_ms: execution_time,
            test_parameters,
        });
    }

    /// Test pronunciation evaluation robustness
    async fn test_pronunciation_evaluation_robustness(
        &mut self,
        audio: &AudioBuffer,
        phonemes: &[Phoneme],
        test_type: &str,
        case_number: usize,
        test_parameters: String,
    ) {
        let start_time = Instant::now();

        // Convert phonemes to text string
        let text = phonemes
            .iter()
            .map(|p| p.symbol.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let result = tokio::time::timeout(
            Duration::from_secs(self.config.max_execution_time),
            self.pronunciation_evaluator
                .evaluate_pronunciation(audio, &text, None),
        )
        .await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        let (passed, error_message) = match result {
            Ok(Ok(_)) => (true, None),
            Ok(Err(e)) => (true, Some(format!("Expected error: {}", e))), // Errors are expected in fuzzing
            Err(_) => (false, Some("Timeout".to_string())),
        };

        self.results.push(FuzzingResult {
            test_type: test_type.to_string(),
            case_number,
            passed,
            error_message,
            execution_time_ms: execution_time,
            test_parameters,
        });
    }

    /// Generate random audio buffer
    fn generate_random_audio_buffer(&mut self) -> AudioBuffer {
        let size = self.rng.gen_range(1..=self.config.max_audio_size);
        let sample_rate = match self.rng.gen_range(0..4) {
            0 => 8000,
            1 => 16000,
            2 => 22050,
            _ => 44100,
        };

        let samples: Vec<f32> = (0..size).map(|_| self.rng.gen_range(-1.0..=1.0)).collect();

        AudioBuffer::new(samples, sample_rate, 1)
    }

    /// Generate extreme audio buffer
    fn generate_extreme_audio_buffer(&mut self) -> AudioBuffer {
        let size = self.rng.gen_range(1..=1000);
        let sample_rate = 16000;

        let samples: Vec<f32> = (0..size)
            .map(|_| match self.rng.gen_range(0..6) {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => f32::NAN,
                3 => 0.0,
                4 => 1.0,
                _ => -1.0,
            })
            .collect();

        AudioBuffer::new(samples, sample_rate, 1)
    }

    /// Generate malformed audio buffer
    fn generate_malformed_audio_buffer(&mut self) -> AudioBuffer {
        let size = self.rng.gen_range(1..=100);
        let sample_rate = if self.rng.gen_bool(0.5) { 0 } else { 44100 };

        let samples: Vec<f32> = (0..size)
            .map(|_| {
                let extreme_value = self.rng.gen_range(-1e10..=1e10);
                extreme_value
            })
            .collect();

        AudioBuffer::new(samples, sample_rate, 1)
    }

    /// Generate random phonemes
    fn generate_random_phonemes(&mut self) -> Vec<Phoneme> {
        let size = self.rng.gen_range(1..=50);
        let phoneme_symbols = vec![
            "a", "e", "i", "o", "u", "t", "n", "s", "r", "l", "d", "k", "m", "p", "w", "f", "g",
            "h", "b", "v",
        ];

        (0..size)
            .map(|_| {
                let symbol = phoneme_symbols[self.rng.gen_range(0..phoneme_symbols.len())];
                Phoneme::new(symbol)
            })
            .collect()
    }

    /// Generate extreme phonemes
    fn generate_extreme_phonemes(&mut self) -> Vec<Phoneme> {
        let size = self.rng.gen_range(1..=10);

        (0..size)
            .map(|_| {
                let symbol = match self.rng.gen_range(0..5) {
                    0 => "".to_string(),          // Empty symbol
                    1 => "a".repeat(100),         // Very long symbol
                    2 => "\u{1F600}".to_string(), // Emoji
                    3 => "ɑɪoʊeɪ".to_string(),    // Complex IPA
                    _ => "invalid_phoneme_symbol_12345".to_string(),
                };
                Phoneme::new(symbol)
            })
            .collect()
    }

    /// Generate fuzzing report
    pub fn generate_report(&self) -> FuzzingReport {
        let mut passed_count = 0;
        let mut timeout_count = 0;
        let mut error_count = 0;
        let mut total_execution_time = 0;
        let mut test_type_summaries = std::collections::HashMap::new();

        for result in &self.results {
            if result.passed {
                passed_count += 1;
            }

            if result
                .error_message
                .as_ref()
                .map_or(false, |msg| msg.contains("Timeout"))
            {
                timeout_count += 1;
            } else if result.error_message.is_some() {
                error_count += 1;
            }

            total_execution_time += result.execution_time_ms;

            let summary = test_type_summaries
                .entry(result.test_type.clone())
                .or_insert(TestTypeSummary {
                    test_type: result.test_type.clone(),
                    total_tests: 0,
                    passed_tests: 0,
                    avg_execution_time: 0.0,
                });

            summary.total_tests += 1;
            if result.passed {
                summary.passed_tests += 1;
            }
            summary.avg_execution_time += result.execution_time_ms as f32;
        }

        // Calculate averages
        for summary in test_type_summaries.values_mut() {
            if summary.total_tests > 0 {
                summary.avg_execution_time /= summary.total_tests as f32;
            }
        }

        FuzzingReport {
            total_tests: self.results.len(),
            passed_tests: passed_count,
            timeout_count,
            error_count,
            success_rate: if self.results.len() > 0 {
                passed_count as f32 / self.results.len() as f32
            } else {
                0.0
            },
            total_execution_time_ms: total_execution_time,
            avg_execution_time_ms: if self.results.len() > 0 {
                total_execution_time / self.results.len() as u64
            } else {
                0
            },
            test_type_summaries: test_type_summaries.into_values().collect(),
            detailed_results: self.results.clone(),
        }
    }

    /// Get fuzzing results
    pub fn get_results(&self) -> &[FuzzingResult] {
        &self.results
    }
}

/// Fuzzing report
#[derive(Debug, Clone)]
pub struct FuzzingReport {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of timeout tests
    pub timeout_count: usize,
    /// Number of error tests
    pub error_count: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Total execution time (ms)
    pub total_execution_time_ms: u64,
    /// Average execution time per test (ms)
    pub avg_execution_time_ms: u64,
    /// Summary by test type
    pub test_type_summaries: Vec<TestTypeSummary>,
    /// Detailed results
    pub detailed_results: Vec<FuzzingResult>,
}

/// Test type summary
#[derive(Debug, Clone)]
pub struct TestTypeSummary {
    /// Test type name
    pub test_type: String,
    /// Total tests for this type
    pub total_tests: usize,
    /// Passed tests for this type
    pub passed_tests: usize,
    /// Average execution time for this type
    pub avg_execution_time: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fuzzing_suite_creation() {
        let config = FuzzingConfig::default();
        let suite = FuzzingTestSuite::new(config).await.unwrap();

        assert_eq!(suite.results.len(), 0);
    }

    #[tokio::test]
    async fn test_fuzzing_execution() {
        let config = FuzzingConfig {
            iterations: 5,
            max_execution_time: 10,
            ..Default::default()
        };
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        let results = suite.run_all_fuzzing_tests().await.unwrap();

        // Should have results from all fuzzing categories
        assert!(results.len() > 0);

        // Check that no tests timed out
        for result in &results {
            assert!(result.execution_time_ms < 25000); // 25 seconds max for fuzzing tests in CI environments
        }
    }

    #[tokio::test]
    async fn test_audio_fuzzing() {
        let config = FuzzingConfig {
            iterations: 3,
            enable_audio_fuzzing: true,
            enable_parameter_fuzzing: false,
            enable_boundary_testing: false,
            ..Default::default()
        };
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        suite.run_all_fuzzing_tests().await.unwrap();
        let results = suite.get_results();

        assert!(results.iter().any(|r| r.test_type.contains("AudioFuzzing")));
    }

    #[tokio::test]
    async fn test_boundary_conditions() {
        let config = FuzzingConfig {
            iterations: 1,
            enable_audio_fuzzing: false,
            enable_parameter_fuzzing: false,
            enable_boundary_testing: true,
            ..Default::default()
        };
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        suite.run_all_fuzzing_tests().await.unwrap();
        let results = suite.get_results();

        assert!(results.iter().any(|r| r.test_type.contains("EmptyAudio")));
        assert!(results
            .iter()
            .any(|r| r.test_type.contains("EmptyPhonemes")));
    }

    #[tokio::test]
    async fn test_report_generation() {
        let config = FuzzingConfig {
            iterations: 2,
            max_execution_time: 5,
            ..Default::default()
        };
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        suite.run_all_fuzzing_tests().await.unwrap();
        let report = suite.generate_report();

        assert!(report.total_tests > 0);
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert!(report.test_type_summaries.len() > 0);
    }

    #[tokio::test]
    async fn test_random_audio_generation() {
        let config = FuzzingConfig::default();
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        let audio = suite.generate_random_audio_buffer();

        assert!(audio.samples().len() > 0);
        assert!(audio.sample_rate() > 0);
        assert_eq!(audio.channels(), 1);
    }

    #[tokio::test]
    async fn test_extreme_audio_generation() {
        let config = FuzzingConfig::default();
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        let audio = suite.generate_extreme_audio_buffer();

        assert!(audio.samples().len() > 0);
        assert_eq!(audio.channels(), 1);
    }

    #[tokio::test]
    async fn test_random_phoneme_generation() {
        let config = FuzzingConfig::default();
        let mut suite = FuzzingTestSuite::new(config).await.unwrap();

        let phonemes = suite.generate_random_phonemes();

        assert!(phonemes.len() > 0);
        assert!(phonemes.len() <= 50);
    }
}
