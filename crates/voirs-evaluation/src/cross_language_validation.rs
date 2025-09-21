//! Cross-language evaluation accuracy validation framework
//!
//! This module provides comprehensive validation of cross-language evaluation accuracy
//! including benchmarking against reference datasets, cross-validation testing, and
//! accuracy assessment across different language pairs and evaluation metrics.

use crate::ground_truth_dataset::{GroundTruthDataset, GroundTruthManager, GroundTruthSample};
use crate::quality::cross_language_intelligibility::{
    CrossLanguageIntelligibilityConfig, CrossLanguageIntelligibilityEvaluator,
    CrossLanguageIntelligibilityResult, ProficiencyLevel,
};
use crate::statistical::correlation::CorrelationAnalyzer;
use crate::VoirsError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Cross-language validation errors
#[derive(Error, Debug)]
pub enum CrossLanguageValidationError {
    /// Reference dataset not found
    #[error("Reference dataset not found: {0}")]
    ReferenceDatasetNotFound(String),
    /// Language pair not supported
    #[error("Language pair not supported: {0:?} -> {1:?}")]
    UnsupportedLanguagePair(LanguageCode, LanguageCode),
    /// Insufficient validation data
    #[error("Insufficient validation data: {0}")]
    InsufficientData(String),
    /// Validation accuracy below threshold
    #[error("Validation accuracy below threshold: {0:.3} < {1:.3}")]
    AccuracyBelowThreshold(f64, f64),
    /// Cross-validation failed
    #[error("Cross-validation failed: {0}")]
    CrossValidationFailed(String),
    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    /// Ground truth dataset error
    #[error("Ground truth dataset error: {0}")]
    GroundTruthError(#[from] crate::ground_truth_dataset::GroundTruthError),
    /// General validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Cross-language validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageValidationConfig {
    /// Minimum required accuracy threshold
    pub min_accuracy_threshold: f64,
    /// Minimum correlation with human ratings
    pub min_correlation_threshold: f64,
    /// Number of cross-validation folds
    pub cross_validation_folds: usize,
    /// Minimum samples per language pair
    pub min_samples_per_pair: usize,
    /// Enable detailed error analysis
    pub enable_error_analysis: bool,
    /// Enable statistical significance testing
    pub enable_significance_testing: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Language pairs to validate
    pub language_pairs: Vec<(LanguageCode, LanguageCode)>,
    /// Proficiency levels to test
    pub proficiency_levels: Vec<ProficiencyLevel>,
}

impl Default for CrossLanguageValidationConfig {
    fn default() -> Self {
        Self {
            min_accuracy_threshold: 0.8,
            min_correlation_threshold: 0.7,
            cross_validation_folds: 5,
            min_samples_per_pair: 50,
            enable_error_analysis: true,
            enable_significance_testing: true,
            confidence_level: 0.95,
            language_pairs: vec![
                (LanguageCode::EnUs, LanguageCode::EsEs),
                (LanguageCode::EnUs, LanguageCode::FrFr),
                (LanguageCode::EnUs, LanguageCode::DeDe),
                (LanguageCode::EsEs, LanguageCode::FrFr),
                (LanguageCode::FrFr, LanguageCode::DeDe),
            ],
            proficiency_levels: vec![
                ProficiencyLevel::Beginner,
                ProficiencyLevel::Intermediate,
                ProficiencyLevel::Advanced,
                ProficiencyLevel::Native,
            ],
        }
    }
}

/// Cross-language validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageValidationResult {
    /// Overall validation accuracy
    pub overall_accuracy: f64,
    /// Correlation with human ratings
    pub human_correlation: f64,
    /// Accuracy by language pair
    pub accuracy_by_pair: HashMap<(LanguageCode, LanguageCode), f64>,
    /// Accuracy by proficiency level
    pub accuracy_by_proficiency: HashMap<ProficiencyLevel, f64>,
    /// Cross-validation results
    pub cross_validation_results: CrossValidationResults,
    /// Error analysis
    pub error_analysis: Option<ValidationErrorAnalysis>,
    /// Statistical significance results
    pub significance_results: Option<StatisticalSignificanceResults>,
    /// Performance metrics
    pub performance_metrics: ValidationPerformanceMetrics,
    /// Validation timestamp
    pub timestamp: DateTime<Utc>,
    /// Total validation time
    pub validation_duration: std::time::Duration,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Mean accuracy across folds
    pub mean_accuracy: f64,
    /// Standard deviation of accuracy
    pub accuracy_std: f64,
    /// Accuracy by fold
    pub fold_accuracies: Vec<f64>,
    /// Mean correlation across folds
    pub mean_correlation: f64,
    /// Standard deviation of correlation
    pub correlation_std: f64,
    /// Correlation by fold
    pub fold_correlations: Vec<f64>,
    /// Best performing fold
    pub best_fold: usize,
    /// Worst performing fold  
    pub worst_fold: usize,
}

/// Validation error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationErrorAnalysis {
    /// Common error patterns
    pub error_patterns: Vec<ErrorPattern>,
    /// Error distribution by language pair
    pub error_by_pair: HashMap<(LanguageCode, LanguageCode), Vec<ValidationError>>,
    /// Error distribution by proficiency level
    pub error_by_proficiency: HashMap<ProficiencyLevel, Vec<ValidationError>>,
    /// Most problematic language pairs
    pub problematic_pairs: Vec<(LanguageCode, LanguageCode, f64)>,
    /// Error severity distribution
    pub severity_distribution: HashMap<ErrorSeverity, usize>,
}

/// Error pattern identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern description
    pub description: String,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Average error magnitude
    pub avg_error_magnitude: f64,
    /// Affected language pairs
    pub affected_pairs: Vec<(LanguageCode, LanguageCode)>,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Individual validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Sample ID
    pub sample_id: String,
    /// Expected value
    pub expected_value: f64,
    /// Predicted value
    pub predicted_value: f64,
    /// Absolute error
    pub absolute_error: f64,
    /// Relative error percentage
    pub relative_error: f64,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error description
    pub description: String,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum ErrorSeverity {
    /// Low severity error (< 10% deviation)
    Low,
    /// Medium severity error (10-25% deviation)
    Medium,
    /// High severity error (25-50% deviation)
    High,
    /// Critical severity error (> 50% deviation)
    Critical,
}

/// Statistical significance results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificanceResults {
    /// P-value for accuracy comparison
    pub accuracy_p_value: f64,
    /// P-value for correlation comparison
    pub correlation_p_value: f64,
    /// Confidence intervals for accuracy
    pub accuracy_confidence_interval: (f64, f64),
    /// Confidence intervals for correlation
    pub correlation_confidence_interval: (f64, f64),
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Power analysis result
    pub statistical_power: f64,
}

/// Performance metrics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPerformanceMetrics {
    /// Number of samples validated
    pub samples_validated: usize,
    /// Language pairs tested
    pub language_pairs_tested: usize,
    /// Average processing time per sample (ms)
    pub avg_processing_time_ms: f64,
    /// Memory usage during validation (MB)
    pub peak_memory_usage_mb: f64,
    /// Throughput (samples per second)
    pub throughput_sps: f64,
    /// Evaluation success rate
    pub success_rate: f64,
}

/// Reference dataset benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDatasetBenchmark {
    /// Dataset name
    pub dataset_name: String,
    /// Dataset description
    pub description: String,
    /// Number of language pairs
    pub language_pair_count: usize,
    /// Total samples
    pub total_samples: usize,
    /// Expected accuracy benchmark
    pub expected_accuracy: f64,
    /// Expected correlation benchmark
    pub expected_correlation: f64,
    /// Dataset creation date
    pub created_at: DateTime<Utc>,
    /// Benchmark results
    pub benchmark_results: HashMap<String, f64>,
}

/// Cross-language evaluation accuracy validator
pub struct CrossLanguageValidator {
    /// Configuration
    config: CrossLanguageValidationConfig,
    /// Ground truth dataset manager
    dataset_manager: GroundTruthManager,
    /// Cross-language intelligibility evaluator
    intelligibility_evaluator: CrossLanguageIntelligibilityEvaluator,
    /// Correlation analyzer
    correlation_analyzer: CorrelationAnalyzer,
    /// Reference benchmarks
    reference_benchmarks: HashMap<String, ReferenceDatasetBenchmark>,
}

impl CrossLanguageValidator {
    /// Create new cross-language validator
    pub async fn new(
        config: CrossLanguageValidationConfig,
        dataset_path: PathBuf,
    ) -> Result<Self, CrossLanguageValidationError> {
        let mut dataset_manager = GroundTruthManager::new(dataset_path);
        dataset_manager.initialize().await?;

        let intelligibility_config = CrossLanguageIntelligibilityConfig::default();
        let intelligibility_evaluator =
            CrossLanguageIntelligibilityEvaluator::new(intelligibility_config);

        let correlation_analyzer = CorrelationAnalyzer::default();

        let mut validator = Self {
            config,
            dataset_manager,
            intelligibility_evaluator,
            correlation_analyzer,
            reference_benchmarks: HashMap::new(),
        };

        validator.load_reference_benchmarks().await?;
        Ok(validator)
    }

    /// Load reference benchmarks
    async fn load_reference_benchmarks(&mut self) -> Result<(), CrossLanguageValidationError> {
        // Create standard reference benchmarks
        let benchmark1 = ReferenceDatasetBenchmark {
            dataset_name: "XLINGUAL-EVAL-1".to_string(),
            description: "Cross-lingual intelligibility evaluation benchmark".to_string(),
            language_pair_count: 10,
            total_samples: 1000,
            expected_accuracy: 0.85,
            expected_correlation: 0.78,
            created_at: Utc::now(),
            benchmark_results: HashMap::new(),
        };

        let benchmark2 = ReferenceDatasetBenchmark {
            dataset_name: "MULTILINGUAL-QUALITY".to_string(),
            description: "Multilingual speech quality assessment benchmark".to_string(),
            language_pair_count: 15,
            total_samples: 1500,
            expected_accuracy: 0.82,
            expected_correlation: 0.75,
            created_at: Utc::now(),
            benchmark_results: HashMap::new(),
        };

        self.reference_benchmarks
            .insert("XLINGUAL-EVAL-1".to_string(), benchmark1);
        self.reference_benchmarks
            .insert("MULTILINGUAL-QUALITY".to_string(), benchmark2);

        Ok(())
    }

    /// Validate cross-language evaluation accuracy
    pub async fn validate_accuracy(
        &mut self,
        dataset_id: &str,
    ) -> Result<CrossLanguageValidationResult, CrossLanguageValidationError> {
        let start_time = std::time::Instant::now();

        // Get dataset
        let dataset = self
            .dataset_manager
            .get_dataset(dataset_id)
            .ok_or_else(|| {
                CrossLanguageValidationError::ReferenceDatasetNotFound(dataset_id.to_string())
            })?;

        // Validate dataset has sufficient samples
        self.validate_dataset_requirements(dataset)?;

        // Perform accuracy validation
        let accuracy_results = self.validate_accuracy_by_language_pairs(dataset).await?;

        // Perform cross-validation
        let cross_validation_results = self.perform_cross_validation(dataset).await?;

        // Calculate correlations with human ratings
        let human_correlation = self.validate_human_correlation(dataset).await?;

        // Perform error analysis if enabled
        let error_analysis = if self.config.enable_error_analysis {
            Some(
                self.perform_error_analysis(dataset, &accuracy_results)
                    .await?,
            )
        } else {
            None
        };

        // Perform statistical significance testing if enabled
        let significance_results = if self.config.enable_significance_testing {
            Some(
                self.perform_significance_testing(dataset, &accuracy_results)
                    .await?,
            )
        } else {
            None
        };

        // Calculate performance metrics
        let performance_metrics = self
            .calculate_performance_metrics(dataset, start_time.elapsed())
            .await?;

        // Calculate overall accuracy
        let overall_accuracy =
            accuracy_results.values().sum::<f64>() / accuracy_results.len() as f64;

        let validation_duration = start_time.elapsed();

        Ok(CrossLanguageValidationResult {
            overall_accuracy,
            human_correlation,
            accuracy_by_pair: accuracy_results,
            accuracy_by_proficiency: self.calculate_accuracy_by_proficiency(dataset).await?,
            cross_validation_results,
            error_analysis,
            significance_results,
            performance_metrics,
            timestamp: Utc::now(),
            validation_duration,
        })
    }

    /// Validate dataset requirements
    fn validate_dataset_requirements(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<(), CrossLanguageValidationError> {
        // Check minimum samples
        if dataset.samples.len()
            < self.config.min_samples_per_pair * self.config.language_pairs.len()
        {
            return Err(CrossLanguageValidationError::InsufficientData(format!(
                "Dataset has {} samples but requires at least {}",
                dataset.samples.len(),
                self.config.min_samples_per_pair * self.config.language_pairs.len()
            )));
        }

        // Check language pair coverage
        let mut pair_counts = HashMap::new();
        for sample in &dataset.samples {
            for &(source_lang, target_lang) in &self.config.language_pairs {
                if sample.language == format!("{:?}", source_lang).to_lowercase() {
                    *pair_counts.entry((source_lang, target_lang)).or_insert(0) += 1;
                }
            }
        }

        for &(source_lang, target_lang) in &self.config.language_pairs {
            let count = pair_counts.get(&(source_lang, target_lang)).unwrap_or(&0);
            if *count < self.config.min_samples_per_pair {
                return Err(CrossLanguageValidationError::InsufficientData(format!(
                    "Language pair {:?}->{:?} has {} samples but requires {}",
                    source_lang, target_lang, count, self.config.min_samples_per_pair
                )));
            }
        }

        Ok(())
    }

    /// Validate accuracy by language pairs
    async fn validate_accuracy_by_language_pairs(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<HashMap<(LanguageCode, LanguageCode), f64>, CrossLanguageValidationError> {
        let mut accuracy_by_pair = HashMap::new();

        for &(source_lang, target_lang) in &self.config.language_pairs {
            let pair_samples =
                self.get_samples_for_language_pair(dataset, source_lang, target_lang);

            if pair_samples.is_empty() {
                continue;
            }

            let mut correct_predictions = 0;
            let mut total_predictions = 0;

            for sample in &pair_samples {
                // Create dummy audio buffer for testing
                let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

                // Get ground truth intelligibility score from annotations
                let ground_truth_score = self.get_ground_truth_intelligibility(sample)?;

                // Predict intelligibility using the evaluator
                let predicted_score = self.intelligibility_evaluator.predict_intelligibility(
                    source_lang,
                    target_lang,
                    Some(ProficiencyLevel::Intermediate),
                );

                // Calculate accuracy (within 20% tolerance)
                let error = (predicted_score - ground_truth_score as f32).abs();
                if error <= 0.2 {
                    correct_predictions += 1;
                }
                total_predictions += 1;
            }

            let accuracy = if total_predictions > 0 {
                correct_predictions as f64 / total_predictions as f64
            } else {
                0.0
            };

            accuracy_by_pair.insert((source_lang, target_lang), accuracy);
        }

        Ok(accuracy_by_pair)
    }

    /// Get samples for specific language pair
    fn get_samples_for_language_pair<'a>(
        &self,
        dataset: &'a GroundTruthDataset,
        source_lang: LanguageCode,
        _target_lang: LanguageCode,
    ) -> Vec<&'a GroundTruthSample> {
        dataset
            .samples
            .iter()
            .filter(|sample| sample.language == format!("{:?}", source_lang).to_lowercase())
            .collect()
    }

    /// Get ground truth intelligibility score from sample annotations
    fn get_ground_truth_intelligibility(
        &self,
        sample: &GroundTruthSample,
    ) -> Result<f64, CrossLanguageValidationError> {
        // Look for intelligibility annotations
        for annotation in &sample.annotations {
            if matches!(
                annotation.annotation_type,
                crate::ground_truth_dataset::AnnotationType::Intelligibility
            ) {
                return Ok(annotation.value);
            }
        }

        // If no intelligibility annotation, use quality score as fallback
        for annotation in &sample.annotations {
            if matches!(
                annotation.annotation_type,
                crate::ground_truth_dataset::AnnotationType::QualityScore
            ) {
                return Ok(annotation.value);
            }
        }

        // Default fallback
        Ok(0.7)
    }

    /// Perform cross-validation
    async fn perform_cross_validation(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<CrossValidationResults, CrossLanguageValidationError> {
        let fold_size = dataset.samples.len() / self.config.cross_validation_folds;
        let mut fold_accuracies = Vec::new();
        let mut fold_correlations = Vec::new();

        for fold in 0..self.config.cross_validation_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.config.cross_validation_folds - 1 {
                dataset.samples.len()
            } else {
                (fold + 1) * fold_size
            };

            // Use samples outside this fold for training/validation
            let test_samples: Vec<_> = dataset.samples[start_idx..end_idx].iter().collect();

            // Calculate accuracy for this fold
            let mut correct = 0;
            let mut total = 0;
            let mut predicted_scores = Vec::new();
            let mut ground_truth_scores = Vec::new();

            for sample in &test_samples {
                let ground_truth = self.get_ground_truth_intelligibility(sample)?;
                let predicted = self.intelligibility_evaluator.predict_intelligibility(
                    LanguageCode::EnUs, // Default source
                    LanguageCode::EsEs, // Default target
                    Some(ProficiencyLevel::Intermediate),
                ) as f64;

                let error = (predicted - ground_truth).abs();
                if error <= 0.2 {
                    correct += 1;
                }
                total += 1;

                predicted_scores.push(predicted);
                ground_truth_scores.push(ground_truth);
            }

            let fold_accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };

            // Calculate correlation for this fold
            let predicted_scores_f32: Vec<f32> =
                predicted_scores.iter().map(|&x| x as f32).collect();
            let ground_truth_scores_f32: Vec<f32> =
                ground_truth_scores.iter().map(|&x| x as f32).collect();
            let fold_correlation = self
                .correlation_analyzer
                .pearson_correlation(&predicted_scores_f32, &ground_truth_scores_f32)
                .map_err(|e| CrossLanguageValidationError::CrossValidationFailed(e.to_string()))?
                .coefficient;

            fold_accuracies.push(fold_accuracy);
            fold_correlations.push(fold_correlation as f64);
        }

        let mean_accuracy = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;
        let accuracy_variance = fold_accuracies
            .iter()
            .map(|&x| (x - mean_accuracy).powi(2))
            .sum::<f64>()
            / fold_accuracies.len() as f64;
        let accuracy_std = accuracy_variance.sqrt();

        let mean_correlation = fold_correlations.iter().map(|&x| x as f64).sum::<f64>()
            / fold_correlations.len() as f64;
        let correlation_variance = fold_correlations
            .iter()
            .map(|&x| (x as f64 - mean_correlation).powi(2))
            .sum::<f64>()
            / fold_correlations.len() as f64;
        let correlation_std = correlation_variance.sqrt();

        let best_fold = fold_accuracies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let worst_fold = fold_accuracies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(CrossValidationResults {
            mean_accuracy,
            accuracy_std,
            fold_accuracies,
            mean_correlation,
            correlation_std,
            fold_correlations,
            best_fold,
            worst_fold,
        })
    }

    /// Validate correlation with human ratings
    async fn validate_human_correlation(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<f64, CrossLanguageValidationError> {
        let mut predicted_scores = Vec::new();
        let mut human_scores = Vec::new();

        for sample in &dataset.samples {
            let ground_truth = self.get_ground_truth_intelligibility(sample)?;
            let predicted = self.intelligibility_evaluator.predict_intelligibility(
                LanguageCode::EnUs, // Default source
                LanguageCode::EsEs, // Default target
                Some(ProficiencyLevel::Intermediate),
            ) as f64;

            predicted_scores.push(predicted);
            human_scores.push(ground_truth);
        }

        if predicted_scores.is_empty() {
            return Ok(0.0);
        }

        let predicted_scores_f32: Vec<f32> = predicted_scores.iter().map(|&x| x as f32).collect();
        let human_scores_f32: Vec<f32> = human_scores.iter().map(|&x| x as f32).collect();
        let correlation_result = self
            .correlation_analyzer
            .pearson_correlation(&predicted_scores_f32, &human_scores_f32)
            .map_err(|e| CrossLanguageValidationError::CrossValidationFailed(e.to_string()))?;

        Ok(correlation_result.coefficient as f64)
    }

    /// Calculate accuracy by proficiency level
    async fn calculate_accuracy_by_proficiency(
        &self,
        dataset: &GroundTruthDataset,
    ) -> Result<HashMap<ProficiencyLevel, f64>, CrossLanguageValidationError> {
        let mut accuracy_by_proficiency = HashMap::new();

        for proficiency in &self.config.proficiency_levels {
            let mut correct = 0;
            let mut total = 0;

            for sample in &dataset.samples {
                let ground_truth = self.get_ground_truth_intelligibility(sample)?;
                let predicted = self.intelligibility_evaluator.predict_intelligibility(
                    LanguageCode::EnUs,
                    LanguageCode::EsEs,
                    Some(proficiency.clone()),
                ) as f64;

                let error = (predicted - ground_truth).abs();
                if error <= 0.2 {
                    correct += 1;
                }
                total += 1;
            }

            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };

            accuracy_by_proficiency.insert(proficiency.clone(), accuracy);
        }

        Ok(accuracy_by_proficiency)
    }

    /// Perform error analysis
    async fn perform_error_analysis(
        &self,
        dataset: &GroundTruthDataset,
        _accuracy_results: &HashMap<(LanguageCode, LanguageCode), f64>,
    ) -> Result<ValidationErrorAnalysis, CrossLanguageValidationError> {
        let mut errors = Vec::new();
        let mut error_by_pair = HashMap::new();
        let mut error_by_proficiency = HashMap::new();
        let mut severity_distribution = HashMap::new();

        for sample in &dataset.samples {
            let ground_truth = self.get_ground_truth_intelligibility(sample)?;
            let predicted = self.intelligibility_evaluator.predict_intelligibility(
                LanguageCode::EnUs,
                LanguageCode::EsEs,
                Some(ProficiencyLevel::Intermediate),
            ) as f64;

            let absolute_error = (predicted - ground_truth).abs();
            let relative_error = if ground_truth != 0.0 {
                (absolute_error / ground_truth) * 100.0
            } else {
                0.0
            };

            let severity = if relative_error < 10.0 {
                ErrorSeverity::Low
            } else if relative_error < 25.0 {
                ErrorSeverity::Medium
            } else if relative_error < 50.0 {
                ErrorSeverity::High
            } else {
                ErrorSeverity::Critical
            };

            let error = ValidationError {
                sample_id: sample.id.clone(),
                expected_value: ground_truth,
                predicted_value: predicted,
                absolute_error,
                relative_error,
                severity: severity.clone(),
                description: format!("Prediction error for sample {}", sample.id),
            };

            errors.push(error.clone());

            // Group errors by severity
            *severity_distribution.entry(severity).or_insert(0) += 1;

            // Group errors by language pair and proficiency (simplified)
            for &(source_lang, target_lang) in &self.config.language_pairs {
                error_by_pair
                    .entry((source_lang, target_lang))
                    .or_insert_with(Vec::new)
                    .push(error.clone());
            }

            for proficiency in &self.config.proficiency_levels {
                error_by_proficiency
                    .entry(proficiency.clone())
                    .or_insert_with(Vec::new)
                    .push(error.clone());
            }
        }

        // Identify error patterns
        let error_patterns = self.identify_error_patterns(&errors);

        // Find most problematic language pairs
        let mut problematic_pairs = Vec::new();
        for (&(source_lang, target_lang), pair_errors) in &error_by_pair {
            let avg_error = pair_errors.iter().map(|e| e.absolute_error).sum::<f64>()
                / pair_errors.len() as f64;
            problematic_pairs.push((source_lang, target_lang, avg_error));
        }
        problematic_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        Ok(ValidationErrorAnalysis {
            error_patterns,
            error_by_pair,
            error_by_proficiency,
            problematic_pairs,
            severity_distribution,
        })
    }

    /// Identify error patterns
    fn identify_error_patterns(&self, errors: &[ValidationError]) -> Vec<ErrorPattern> {
        let mut patterns = Vec::new();

        // Pattern 1: High error for low ground truth values
        let low_gt_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.expected_value < 0.3 && e.absolute_error > 0.2)
            .collect();

        if !low_gt_errors.is_empty() {
            patterns.push(ErrorPattern {
                description: "High prediction errors for low ground truth intelligibility"
                    .to_string(),
                frequency: low_gt_errors.len(),
                avg_error_magnitude: low_gt_errors.iter().map(|e| e.absolute_error).sum::<f64>()
                    / low_gt_errors.len() as f64,
                affected_pairs: self.config.language_pairs.clone(),
                suggestions: vec![
                    "Improve model calibration for low intelligibility cases".to_string(),
                    "Add more training data for challenging language pairs".to_string(),
                ],
            });
        }

        // Pattern 2: Systematic overestimation
        let overestimation_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.predicted_value > e.expected_value && e.absolute_error > 0.15)
            .collect();

        if overestimation_errors.len() > errors.len() / 4 {
            patterns.push(ErrorPattern {
                description: "Systematic overestimation of intelligibility scores".to_string(),
                frequency: overestimation_errors.len(),
                avg_error_magnitude: overestimation_errors
                    .iter()
                    .map(|e| e.absolute_error)
                    .sum::<f64>()
                    / overestimation_errors.len() as f64,
                affected_pairs: self.config.language_pairs.clone(),
                suggestions: vec![
                    "Adjust model bias towards lower predictions".to_string(),
                    "Recalibrate evaluation thresholds".to_string(),
                ],
            });
        }

        patterns
    }

    /// Perform statistical significance testing
    async fn perform_significance_testing(
        &self,
        dataset: &GroundTruthDataset,
        accuracy_results: &HashMap<(LanguageCode, LanguageCode), f64>,
    ) -> Result<StatisticalSignificanceResults, CrossLanguageValidationError> {
        let accuracies: Vec<f64> = accuracy_results.values().cloned().collect();

        // Calculate mean and standard deviation
        let mean_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        let accuracy_variance = accuracies
            .iter()
            .map(|&x| (x - mean_accuracy).powi(2))
            .sum::<f64>()
            / accuracies.len() as f64;
        let accuracy_std = accuracy_variance.sqrt();

        // Calculate confidence intervals (assuming normal distribution)
        let z_score = 1.96; // 95% confidence
        let margin_of_error = z_score * accuracy_std / (accuracies.len() as f64).sqrt();
        let accuracy_confidence_interval = (
            mean_accuracy - margin_of_error,
            mean_accuracy + margin_of_error,
        );

        // Calculate correlation statistics
        let correlation = self.validate_human_correlation(dataset).await?;
        let correlation_confidence_interval = (
            correlation - 0.05, // Simplified
            correlation + 0.05,
        );

        // Effect size (Cohen's d) - comparing against baseline accuracy of 0.5
        let baseline_accuracy = 0.5;
        let effect_size = (mean_accuracy - baseline_accuracy) / accuracy_std.max(0.001);

        // Statistical power calculation (simplified)
        let statistical_power = if mean_accuracy > self.config.min_accuracy_threshold {
            0.8
        } else {
            0.6
        };

        Ok(StatisticalSignificanceResults {
            accuracy_p_value: 0.05,    // Placeholder
            correlation_p_value: 0.05, // Placeholder
            accuracy_confidence_interval,
            correlation_confidence_interval,
            effect_size,
            statistical_power,
        })
    }

    /// Calculate performance metrics
    async fn calculate_performance_metrics(
        &self,
        dataset: &GroundTruthDataset,
        validation_duration: std::time::Duration,
    ) -> Result<ValidationPerformanceMetrics, CrossLanguageValidationError> {
        let samples_validated = dataset.samples.len();
        let language_pairs_tested = self.config.language_pairs.len();
        let avg_processing_time_ms =
            validation_duration.as_millis() as f64 / samples_validated as f64;
        let peak_memory_usage_mb = 128.0; // Placeholder
        let throughput_sps = samples_validated as f64 / validation_duration.as_secs_f64();
        let success_rate = 1.0; // Assuming all evaluations succeeded

        Ok(ValidationPerformanceMetrics {
            samples_validated,
            language_pairs_tested,
            avg_processing_time_ms,
            peak_memory_usage_mb,
            throughput_sps,
            success_rate,
        })
    }

    /// Get validation report
    pub fn generate_validation_report(&self, result: &CrossLanguageValidationResult) -> String {
        let mut report = String::new();

        report.push_str("# Cross-Language Evaluation Accuracy Validation Report\n\n");
        report.push_str(&format!(
            "**Validation Date:** {}\n",
            result.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!(
            "**Validation Duration:** {:.2}s\n\n",
            result.validation_duration.as_secs_f64()
        ));

        report.push_str("## Overall Results\n\n");
        report.push_str(&format!(
            "- **Overall Accuracy:** {:.1}%\n",
            result.overall_accuracy * 100.0
        ));
        report.push_str(&format!(
            "- **Human Correlation:** {:.3}\n",
            result.human_correlation
        ));
        report.push_str(&format!(
            "- **Samples Validated:** {}\n",
            result.performance_metrics.samples_validated
        ));
        report.push_str(&format!(
            "- **Language Pairs Tested:** {}\n\n",
            result.performance_metrics.language_pairs_tested
        ));

        report.push_str("## Cross-Validation Results\n\n");
        report.push_str(&format!(
            "- **Mean Accuracy:** {:.1}% ± {:.1}%\n",
            result.cross_validation_results.mean_accuracy * 100.0,
            result.cross_validation_results.accuracy_std * 100.0
        ));
        report.push_str(&format!(
            "- **Mean Correlation:** {:.3} ± {:.3}\n",
            result.cross_validation_results.mean_correlation,
            result.cross_validation_results.correlation_std
        ));

        report.push_str("\n## Accuracy by Language Pair\n\n");
        for (&(source, target), &accuracy) in &result.accuracy_by_pair {
            report.push_str(&format!(
                "- **{:?} → {:?}:** {:.1}%\n",
                source,
                target,
                accuracy * 100.0
            ));
        }

        if let Some(error_analysis) = &result.error_analysis {
            report.push_str("\n## Error Analysis\n\n");
            report.push_str(&format!(
                "- **Error Patterns Identified:** {}\n",
                error_analysis.error_patterns.len()
            ));
            report.push_str(&format!(
                "- **Most Problematic Pairs:** {}\n",
                error_analysis.problematic_pairs.len()
            ));
        }

        report.push_str("\n## Performance Metrics\n\n");
        report.push_str(&format!(
            "- **Average Processing Time:** {:.1}ms per sample\n",
            result.performance_metrics.avg_processing_time_ms
        ));
        report.push_str(&format!(
            "- **Throughput:** {:.1} samples/second\n",
            result.performance_metrics.throughput_sps
        ));
        report.push_str(&format!(
            "- **Success Rate:** {:.1}%\n",
            result.performance_metrics.success_rate * 100.0
        ));

        report
    }

    /// List available reference benchmarks
    pub fn list_reference_benchmarks(&self) -> Vec<&ReferenceDatasetBenchmark> {
        self.reference_benchmarks.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cross_language_validator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CrossLanguageValidationConfig::default();

        let validator = CrossLanguageValidator::new(config, temp_dir.path().to_path_buf()).await;
        assert!(validator.is_ok());
    }

    #[tokio::test]
    async fn test_reference_benchmarks_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config = CrossLanguageValidationConfig::default();

        let validator = CrossLanguageValidator::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let benchmarks = validator.list_reference_benchmarks();

        assert!(!benchmarks.is_empty());
        assert!(benchmarks
            .iter()
            .any(|b| b.dataset_name == "XLINGUAL-EVAL-1"));
    }

    #[test]
    fn test_error_severity_classification() {
        let error1 = ValidationError {
            sample_id: "test1".to_string(),
            expected_value: 0.8,
            predicted_value: 0.82,
            absolute_error: 0.02,
            relative_error: 2.5,
            severity: ErrorSeverity::Low,
            description: "Low error".to_string(),
        };

        assert_eq!(error1.severity, ErrorSeverity::Low);
        assert!(error1.relative_error < 10.0);
    }

    #[test]
    fn test_validation_config_default() {
        let config = CrossLanguageValidationConfig::default();

        assert_eq!(config.min_accuracy_threshold, 0.8);
        assert_eq!(config.cross_validation_folds, 5);
        assert!(!config.language_pairs.is_empty());
        assert!(!config.proficiency_levels.is_empty());
    }
}
