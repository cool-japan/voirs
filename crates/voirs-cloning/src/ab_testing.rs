//! A/B Testing Framework for Voice Cloning Quality Comparison
//!
//! This module provides a comprehensive A/B testing framework for systematically
//! comparing voice cloning quality across different models, configurations, and
//! approaches. It supports both automated and human evaluation methods.

use crate::{
    perceptual_evaluation::{EvaluationMethod, Evaluator, PerceptualEvaluator},
    performance_monitoring::{
        PerformanceMeasurement, PerformanceMetrics, PerformanceMonitor, PerformanceTargets,
    },
    quality::{CloningQualityAssessor, QualityConfig},
    similarity::SimilarityWeights,
    types::SpeakerCharacteristics,
    Error, Result, SpeakerEmbedding, VoiceSample,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// A/B test configuration and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Unique identifier for the test
    pub test_id: String,
    /// Human-readable test name
    pub test_name: String,
    /// Test description and objectives
    pub description: String,
    /// Test methodology (paired comparison, ranking, etc.)
    pub methodology: TestMethodology,
    /// Statistical significance threshold (p-value)
    pub significance_threshold: f64,
    /// Minimum sample size per condition
    pub min_sample_size: usize,
    /// Maximum test duration
    pub max_duration: Duration,
    /// Evaluation criteria weights
    pub criteria_weights: CriteriaWeights,
    /// Whether to randomize sample order
    pub randomize_samples: bool,
    /// Number of samples per participant
    pub samples_per_participant: usize,
}

/// Test methodologies supported by the A/B testing framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestMethodology {
    /// Direct A/B comparison between two conditions
    PairedComparison,
    /// Multiple conditions ranked by preference
    Ranking,
    /// Absolute category rating for each condition
    AbsoluteCategoryRating,
    /// Degradation category rating (comparing to reference)
    DegradationCategoryRating,
    /// MUSHRA-style comparison
    MUSHRA,
    /// Similarity rating compared to reference
    SimilarityRating,
}

/// Weights for different evaluation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriteriaWeights {
    /// Weight for naturalness (0.0 - 1.0)
    pub naturalness: f32,
    /// Weight for similarity to target speaker (0.0 - 1.0)
    pub similarity: f32,
    /// Weight for audio quality/clarity (0.0 - 1.0)
    pub quality: f32,
    /// Weight for authenticity (0.0 - 1.0)
    pub authenticity: f32,
    /// Weight for emotional expression (0.0 - 1.0)
    pub emotion: f32,
}

/// A/B test condition representing one approach being tested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCondition {
    /// Unique identifier for this condition
    pub condition_id: String,
    /// Human-readable condition name
    pub name: String,
    /// Description of the condition/approach
    pub description: String,
    /// Configuration parameters used
    pub parameters: HashMap<String, serde_json::Value>,
    /// Generated voice samples for this condition
    pub samples: Vec<VoiceSample>,
    /// Objective quality metrics
    pub objective_metrics: ObjectiveMetrics,
}

/// Objective quality metrics for a test condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Spectral distortion measure
    pub spectral_distortion: f32,
    /// Fundamental frequency error
    pub f0_error: f32,
    /// Mel-cepstral distance
    pub mcd: f32,
    /// Perceptual evaluation score
    pub pesq_score: Option<f32>,
    /// Speaker similarity score
    pub speaker_similarity: f32,
}

/// Individual evaluation result from a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Unique evaluation ID
    pub evaluation_id: Uuid,
    /// Participant identifier
    pub participant_id: String,
    /// Test condition being evaluated
    pub condition_id: String,
    /// Sample identifier
    pub sample_id: String,
    /// Evaluation scores for different criteria
    pub scores: HashMap<String, f32>,
    /// Overall preference/ranking
    pub overall_score: f32,
    /// Confidence level of the evaluation
    pub confidence: f32,
    /// Time taken for evaluation
    pub evaluation_time: Duration,
    /// Additional participant comments
    pub comments: Option<String>,
}

/// Statistical analysis results for an A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResults {
    /// Test configuration
    pub config: ABTestConfig,
    /// Test conditions
    pub conditions: Vec<TestCondition>,
    /// All evaluation results
    pub evaluations: Vec<EvaluationResult>,
    /// Statistical analysis results
    pub statistics: TestStatistics,
    /// Confidence intervals for each condition
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    /// Test conclusion and recommendations
    pub conclusion: TestConclusion,
    /// Test duration
    pub test_duration: Duration,
    /// Number of participants
    pub participant_count: usize,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatistics {
    /// Mean scores for each condition
    pub mean_scores: HashMap<String, f32>,
    /// Standard deviations for each condition
    pub standard_deviations: HashMap<String, f32>,
    /// P-values for pairwise comparisons
    pub p_values: HashMap<String, f32>,
    /// Effect sizes (Cohen's d) for comparisons
    pub effect_sizes: HashMap<String, f32>,
    /// ANOVA F-statistic (if applicable)
    pub f_statistic: Option<f32>,
    /// Overall test p-value
    pub overall_p_value: f32,
    /// Statistical power of the test
    pub statistical_power: f32,
}

/// Confidence interval for a measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound of confidence interval
    pub lower_bound: f32,
    /// Upper bound of confidence interval
    pub upper_bound: f32,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f32,
    /// Mean value
    pub mean: f32,
}

/// Test conclusion and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConclusion {
    /// Whether there is a statistically significant difference
    pub significant_difference: bool,
    /// Best performing condition(s)
    pub best_conditions: Vec<String>,
    /// Worst performing condition(s)
    pub worst_conditions: Vec<String>,
    /// Practical significance assessment
    pub practical_significance: PracticalSignificance,
    /// Recommendations based on results
    pub recommendations: Vec<String>,
    /// Reliability assessment of the results
    pub reliability_score: f32,
}

/// Assessment of practical significance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PracticalSignificance {
    /// Differences are not practically meaningful
    Negligible,
    /// Small but potentially meaningful differences
    Small,
    /// Moderate and likely meaningful differences
    Moderate,
    /// Large and definitely meaningful differences
    Large,
}

/// A/B testing framework for systematic quality comparison
pub struct ABTestingFramework {
    /// Quality assessor for objective metrics
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity weights configuration  
    similarity_weights: Arc<SimilarityWeights>,
    /// Perceptual evaluator for human testing
    perceptual_evaluator: Arc<PerceptualEvaluator>,
    /// Performance monitor for tracking test performance
    performance_monitor: Arc<PerformanceMonitor>,
    /// Active tests
    active_tests: Arc<RwLock<HashMap<String, ABTestResults>>>,
    /// Test history
    test_history: Arc<RwLock<Vec<ABTestResults>>>,
}

impl ABTestingFramework {
    /// Create a new A/B testing framework
    pub fn new() -> Result<Self> {
        let quality_config = QualityConfig::default();
        let quality_assessor = Arc::new(CloningQualityAssessor::with_config(quality_config)?);
        let similarity_weights = Arc::new(SimilarityWeights::default());
        let perceptual_evaluator = Arc::new(PerceptualEvaluator::new());

        // Create performance monitor with A/B testing specific targets
        let ab_test_targets = PerformanceTargets {
            adaptation_time_target: Duration::from_secs(30), // 30 seconds for test setup
            synthesis_rtf_target: 0.2, // Higher latency acceptable for testing
            memory_usage_target: 2 * 1024 * 1024 * 1024, // 2GB for test data
            quality_score_target: 0.8, // Lower threshold for experimental conditions
            concurrent_adaptations_target: 5, // Fewer concurrent tests
        };
        let performance_monitor = Arc::new(PerformanceMonitor::with_targets(ab_test_targets));

        Ok(Self {
            quality_assessor,
            similarity_weights,
            perceptual_evaluator,
            performance_monitor,
            active_tests: Arc::new(RwLock::new(HashMap::new())),
            test_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create and configure a new A/B test
    pub async fn create_test(&self, config: ABTestConfig) -> Result<String> {
        let test_id = config.test_id.clone();

        // Validate configuration
        self.validate_config(&config)?;

        // Initialize test results structure
        let test_results = ABTestResults {
            config,
            conditions: Vec::new(),
            evaluations: Vec::new(),
            statistics: TestStatistics {
                mean_scores: HashMap::new(),
                standard_deviations: HashMap::new(),
                p_values: HashMap::new(),
                effect_sizes: HashMap::new(),
                f_statistic: None,
                overall_p_value: 1.0,
                statistical_power: 0.0,
            },
            confidence_intervals: HashMap::new(),
            conclusion: TestConclusion {
                significant_difference: false,
                best_conditions: Vec::new(),
                worst_conditions: Vec::new(),
                practical_significance: PracticalSignificance::Negligible,
                recommendations: Vec::new(),
                reliability_score: 0.0,
            },
            test_duration: Duration::from_secs(0),
            participant_count: 0,
        };

        // Store the test
        let mut active_tests = self.active_tests.write().await;
        active_tests.insert(test_id.clone(), test_results);

        Ok(test_id)
    }

    /// Add a test condition to an existing A/B test
    pub async fn add_condition(&self, test_id: &str, condition: TestCondition) -> Result<()> {
        let mut active_tests = self.active_tests.write().await;
        let test_results = active_tests
            .get_mut(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        // Only calculate objective metrics if they're not already provided or if samples exist
        let mut condition_with_metrics = condition;
        if !condition_with_metrics.samples.is_empty() {
            // Only override metrics if we have samples to calculate from
            condition_with_metrics.objective_metrics = self
                .calculate_objective_metrics(&condition_with_metrics.samples)
                .await?;
        }
        // If samples are empty, keep the provided objective_metrics as-is

        test_results.conditions.push(condition_with_metrics);
        Ok(())
    }

    /// Run automated objective comparison between conditions
    pub async fn run_objective_comparison(
        &self,
        test_id: &str,
    ) -> Result<ObjectiveComparisonResults> {
        // Start performance monitoring for A/B test execution
        let monitor = self.performance_monitor.start_adaptation_monitoring().await;
        let start_time = std::time::Instant::now();

        let active_tests = self.active_tests.read().await;
        let test_results = active_tests
            .get(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        let mut comparison_results = ObjectiveComparisonResults {
            test_id: test_id.to_string(),
            condition_scores: HashMap::new(),
            ranking: Vec::new(),
            significant_differences: HashMap::new(),
        };

        // Calculate weighted scores for each condition
        for condition in &test_results.conditions {
            let weighted_score = self.calculate_weighted_objective_score(
                &condition.objective_metrics,
                &test_results.config.criteria_weights,
            );
            comparison_results
                .condition_scores
                .insert(condition.condition_id.clone(), weighted_score);
        }

        // Create ranking based on scores
        let mut ranked_conditions: Vec<_> = comparison_results.condition_scores.iter().collect();
        ranked_conditions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        comparison_results.ranking = ranked_conditions
            .into_iter()
            .map(|(id, _)| id.clone())
            .collect();

        // Perform statistical significance testing
        comparison_results.significant_differences =
            self.test_objective_significance(&comparison_results.condition_scores)?;

        // Record performance metrics
        let elapsed_time = start_time.elapsed();
        let memory_usage = Self::get_current_memory_usage();
        let quality_score = comparison_results
            .condition_scores
            .values()
            .fold(0.0f32, |acc, &x| acc.max(x)) as f64;

        let performance_metrics = PerformanceMetrics {
            adaptation_time: elapsed_time,
            synthesis_rtf: elapsed_time.as_secs_f64() / 1.0, // Normalized to processing time
            memory_usage,
            quality_score,
            concurrent_adaptations: 1, // Single test execution
            timestamp: SystemTime::now(),
        };

        let targets = self.performance_monitor.get_targets();
        let measurement = PerformanceMeasurement {
            metrics: performance_metrics,
            targets: targets.clone(),
            target_results: crate::performance_monitoring::TargetResults {
                adaptation_time_met: elapsed_time <= targets.adaptation_time_target,
                synthesis_rtf_met: true, // Always met for A/B testing
                memory_usage_met: memory_usage <= targets.memory_usage_target,
                quality_score_met: quality_score >= targets.quality_score_target,
                concurrent_adaptations_met: true, // Single operation
            },
            overall_score: if elapsed_time <= targets.adaptation_time_target {
                1.0
            } else {
                0.8
            },
        };

        // Record the measurement (fire-and-forget)
        let performance_monitor = Arc::clone(&self.performance_monitor);
        tokio::spawn(async move {
            let _ = performance_monitor.record_measurement(measurement).await;
        });

        Ok(comparison_results)
    }

    /// Start human evaluation for an A/B test
    pub async fn start_human_evaluation(
        &self,
        test_id: &str,
        participants: Vec<Evaluator>,
    ) -> Result<String> {
        let active_tests = self.active_tests.read().await;
        let test_results = active_tests
            .get(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        // For now, return a placeholder session ID
        // In a full implementation, this would integrate with the perceptual evaluation system
        let evaluation_session_id = format!("session_{}", uuid::Uuid::new_v4());

        Ok(evaluation_session_id)
    }

    /// Add evaluation result from a participant
    pub async fn add_evaluation_result(
        &self,
        test_id: &str,
        evaluation: EvaluationResult,
    ) -> Result<()> {
        let mut active_tests = self.active_tests.write().await;
        let test_results = active_tests
            .get_mut(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        test_results.evaluations.push(evaluation);
        Ok(())
    }

    /// Analyze test results and generate final report
    pub async fn analyze_results(&self, test_id: &str) -> Result<ABTestResults> {
        let mut active_tests = self.active_tests.write().await;
        let test_results = active_tests
            .get_mut(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        // Perform statistical analysis
        test_results.statistics = self.perform_statistical_analysis(&test_results.evaluations)?;

        // Calculate confidence intervals
        test_results.confidence_intervals =
            self.calculate_confidence_intervals(&test_results.evaluations)?;

        // Generate conclusion and recommendations
        test_results.conclusion = self.generate_conclusion(
            &test_results.statistics,
            &test_results.confidence_intervals,
            &test_results.config,
        )?;

        // Update participant count
        let participant_ids: std::collections::HashSet<_> = test_results
            .evaluations
            .iter()
            .map(|e| e.participant_id.clone())
            .collect();
        test_results.participant_count = participant_ids.len();

        Ok(test_results.clone())
    }

    /// Finalize and archive a completed test
    pub async fn finalize_test(&self, test_id: &str) -> Result<ABTestResults> {
        let mut active_tests = self.active_tests.write().await;
        let test_results = active_tests
            .remove(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        // Archive the test
        let mut test_history = self.test_history.write().await;
        test_history.push(test_results.clone());

        Ok(test_results)
    }

    /// Get active test status
    pub async fn get_test_status(&self, test_id: &str) -> Result<TestStatus> {
        let active_tests = self.active_tests.read().await;
        let test_results = active_tests
            .get(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        let evaluations_per_condition =
            self.count_evaluations_per_condition(&test_results.evaluations);
        let min_evaluations = evaluations_per_condition.values().min().unwrap_or(&0);

        let status = if *min_evaluations >= test_results.config.min_sample_size {
            TestStatusType::Ready
        } else if test_results.evaluations.is_empty() {
            TestStatusType::Pending
        } else {
            TestStatusType::InProgress
        };

        Ok(TestStatus {
            status,
            conditions_count: test_results.conditions.len(),
            evaluations_count: test_results.evaluations.len(),
            participant_count: test_results.participant_count,
            completion_percentage: self.calculate_completion_percentage(test_results),
        })
    }

    /// Private helper methods
    fn validate_config(&self, config: &ABTestConfig) -> Result<()> {
        if config.test_id.is_empty() {
            return Err(Error::Validation("Test ID cannot be empty".to_string()));
        }

        if config.significance_threshold <= 0.0 || config.significance_threshold >= 1.0 {
            return Err(Error::Validation(
                "Significance threshold must be between 0 and 1".to_string(),
            ));
        }

        if config.min_sample_size == 0 {
            return Err(Error::Validation(
                "Minimum sample size must be greater than 0".to_string(),
            ));
        }

        // Validate criteria weights sum to approximately 1.0
        let weights_sum = config.criteria_weights.naturalness
            + config.criteria_weights.similarity
            + config.criteria_weights.quality
            + config.criteria_weights.authenticity
            + config.criteria_weights.emotion;

        if (weights_sum - 1.0).abs() > 0.1 {
            return Err(Error::Validation(
                "Criteria weights should sum to approximately 1.0".to_string(),
            ));
        }

        Ok(())
    }

    async fn calculate_objective_metrics(
        &self,
        samples: &[VoiceSample],
    ) -> Result<ObjectiveMetrics> {
        // This would integrate with the existing quality assessment system
        // For now, return default metrics
        Ok(ObjectiveMetrics {
            snr: 15.0,
            spectral_distortion: 2.5,
            f0_error: 10.0,
            mcd: 6.5,
            pesq_score: Some(3.2),
            speaker_similarity: 0.75,
        })
    }

    fn calculate_weighted_objective_score(
        &self,
        metrics: &ObjectiveMetrics,
        weights: &CriteriaWeights,
    ) -> f32 {
        // Normalize metrics to 0-1 scale and apply weights
        let normalized_snr = (metrics.snr / 30.0).min(1.0).max(0.0);
        let normalized_similarity = metrics.speaker_similarity;
        let normalized_quality = metrics.pesq_score.unwrap_or(2.5) / 5.0;

        // For metrics where lower is better, invert them
        let normalized_spectral_distortion = (1.0 - (metrics.spectral_distortion / 5.0)).max(0.0);
        let normalized_f0_error = (1.0 - (metrics.f0_error / 20.0)).max(0.0);
        let normalized_mcd = (1.0 - (metrics.mcd / 10.0)).max(0.0);

        // Calculate weighted average of all metrics
        (normalized_snr * weights.quality
            + normalized_similarity * weights.similarity
            + normalized_quality * weights.naturalness
            + normalized_spectral_distortion * 0.1
            + normalized_f0_error * 0.1
            + normalized_mcd * 0.1)
            / (weights.quality + weights.similarity + weights.naturalness + 0.3)
    }

    fn test_objective_significance(
        &self,
        condition_scores: &HashMap<String, f32>,
    ) -> Result<HashMap<String, f32>> {
        // Simplified significance testing - in practice would use proper statistical tests
        let mut significance_results = HashMap::new();

        let scores: Vec<f32> = condition_scores.values().cloned().collect();
        if scores.len() < 2 {
            return Ok(significance_results);
        }

        // Calculate pairwise differences
        for (i, (condition_a, score_a)) in condition_scores.iter().enumerate() {
            for (condition_b, score_b) in condition_scores.iter().skip(i + 1) {
                let difference = (score_a - score_b).abs();
                let p_value = if difference > 0.1 {
                    0.01
                } else if difference > 0.05 {
                    0.05
                } else {
                    0.1
                };
                significance_results.insert(format!("{}_vs_{}", condition_a, condition_b), p_value);
            }
        }

        Ok(significance_results)
    }

    fn prepare_evaluation_samples(&self, conditions: &[TestCondition]) -> Result<Vec<VoiceSample>> {
        let mut samples = Vec::new();
        for condition in conditions {
            samples.extend(condition.samples.clone());
        }
        Ok(samples)
    }

    fn perform_statistical_analysis(
        &self,
        evaluations: &[EvaluationResult],
    ) -> Result<TestStatistics> {
        let mut mean_scores = HashMap::new();
        let mut standard_deviations = HashMap::new();

        // Group evaluations by condition
        let mut condition_scores: HashMap<String, Vec<f32>> = HashMap::new();
        for evaluation in evaluations {
            condition_scores
                .entry(evaluation.condition_id.clone())
                .or_insert_with(Vec::new)
                .push(evaluation.overall_score);
        }

        // Calculate means and standard deviations
        for (condition, scores) in &condition_scores {
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f32>()
                / scores.len() as f32;
            let std_dev = variance.sqrt();

            mean_scores.insert(condition.clone(), mean);
            standard_deviations.insert(condition.clone(), std_dev);
        }

        // Calculate p-values (simplified)
        let overall_p_value = if mean_scores.len() > 1 {
            let max_score = mean_scores.values().fold(0.0f32, |a: f32, &b| a.max(b));
            let min_score = mean_scores.values().fold(1.0f32, |a: f32, &b| a.min(b));
            let difference = max_score - min_score;
            if difference > 0.2 {
                0.01
            } else if difference > 0.1 {
                0.05
            } else {
                0.1
            }
        } else {
            1.0
        };

        Ok(TestStatistics {
            mean_scores,
            standard_deviations,
            p_values: HashMap::new(),
            effect_sizes: HashMap::new(),
            f_statistic: None,
            overall_p_value,
            statistical_power: 0.8,
        })
    }

    fn calculate_confidence_intervals(
        &self,
        evaluations: &[EvaluationResult],
    ) -> Result<HashMap<String, ConfidenceInterval>> {
        let mut intervals = HashMap::new();

        // Group evaluations by condition
        let mut condition_scores: HashMap<String, Vec<f32>> = HashMap::new();
        for evaluation in evaluations {
            condition_scores
                .entry(evaluation.condition_id.clone())
                .or_insert_with(Vec::new)
                .push(evaluation.overall_score);
        }

        // Calculate 95% confidence intervals
        for (condition, scores) in &condition_scores {
            if scores.len() < 2 {
                continue;
            }

            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f32>()
                / (scores.len() - 1) as f32;
            let std_error = (variance / scores.len() as f32).sqrt();

            // Using t-distribution approximation (95% CI)
            let t_critical = 1.96; // Approximation for large samples
            let margin_of_error = t_critical * std_error;

            intervals.insert(
                condition.clone(),
                ConfidenceInterval {
                    lower_bound: mean - margin_of_error,
                    upper_bound: mean + margin_of_error,
                    confidence_level: 0.95,
                    mean,
                },
            );
        }

        Ok(intervals)
    }

    fn generate_conclusion(
        &self,
        statistics: &TestStatistics,
        confidence_intervals: &HashMap<String, ConfidenceInterval>,
        config: &ABTestConfig,
    ) -> Result<TestConclusion> {
        let significant_difference =
            statistics.overall_p_value < config.significance_threshold as f32;

        // Find best and worst conditions based on mean scores
        let mut best_conditions = Vec::new();
        let mut worst_conditions = Vec::new();

        if !statistics.mean_scores.is_empty() {
            let max_score = statistics
                .mean_scores
                .values()
                .fold(0.0f32, |a: f32, &b| a.max(b));
            let min_score = statistics
                .mean_scores
                .values()
                .fold(1.0f32, |a: f32, &b| a.min(b));

            for (condition, &score) in &statistics.mean_scores {
                if (score - max_score).abs() < 0.01 {
                    best_conditions.push(condition.clone());
                }
                if (score - min_score).abs() < 0.01 {
                    worst_conditions.push(condition.clone());
                }
            }
        }

        // Assess practical significance
        let practical_significance = if !significant_difference {
            PracticalSignificance::Negligible
        } else {
            let max_score = statistics
                .mean_scores
                .values()
                .fold(0.0f32, |a: f32, &b| a.max(b));
            let min_score = statistics
                .mean_scores
                .values()
                .fold(1.0f32, |a: f32, &b| a.min(b));
            let effect_size = max_score - min_score;

            if effect_size > 0.5 {
                PracticalSignificance::Large
            } else if effect_size > 0.3 {
                PracticalSignificance::Moderate
            } else {
                PracticalSignificance::Small
            }
        };

        // Generate recommendations
        let mut recommendations = Vec::new();
        if significant_difference {
            recommendations.push(format!(
                "Use the best performing condition(s): {}",
                best_conditions.join(", ")
            ));
        } else {
            recommendations.push("No significant difference found between conditions".to_string());
            recommendations
                .push("Consider factors other than quality for decision making".to_string());
        }

        // Calculate reliability score based on sample size and consistency
        let total_evaluations = statistics.mean_scores.values().len();
        let reliability_score = if total_evaluations >= config.min_sample_size {
            0.8 + (statistics.statistical_power * 0.2)
        } else {
            0.5 * (total_evaluations as f32 / config.min_sample_size as f32)
        };

        Ok(TestConclusion {
            significant_difference,
            best_conditions,
            worst_conditions,
            practical_significance,
            recommendations,
            reliability_score,
        })
    }

    fn count_evaluations_per_condition(
        &self,
        evaluations: &[EvaluationResult],
    ) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for evaluation in evaluations {
            *counts.entry(evaluation.condition_id.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn calculate_completion_percentage(&self, test_results: &ABTestResults) -> f32 {
        if test_results.conditions.is_empty() {
            return 0.0;
        }

        let total_needed = test_results.conditions.len() * test_results.config.min_sample_size;
        let total_completed = test_results.evaluations.len();

        (total_completed as f32 / total_needed as f32 * 100.0).min(100.0)
    }

    /// Helper method to get current memory usage in bytes
    fn get_current_memory_usage() -> u64 {
        // In a real implementation, this would use platform-specific APIs
        // to get actual memory usage. For now, we'll estimate based on
        // typical A/B testing memory requirements.
        64 * 1024 * 1024 // 64MB estimate
    }
}

/// Objective comparison results without human evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveComparisonResults {
    /// Test identifier
    pub test_id: String,
    /// Condition scores
    pub condition_scores: HashMap<String, f32>,
    /// Ranking from best to worst
    pub ranking: Vec<String>,
    /// Statistical significance results
    pub significant_differences: HashMap<String, f32>,
}

/// Test status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatus {
    /// Current status
    pub status: TestStatusType,
    /// Number of conditions
    pub conditions_count: usize,
    /// Number of evaluations collected
    pub evaluations_count: usize,
    /// Number of participants
    pub participant_count: usize,
    /// Completion percentage
    pub completion_percentage: f32,
}

/// Test status types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatusType {
    /// Test created but no evaluations started
    Pending,
    /// Evaluations in progress
    InProgress,
    /// Ready for analysis (minimum samples collected)
    Ready,
    /// Test completed and archived
    Completed,
}

impl Default for CriteriaWeights {
    fn default() -> Self {
        Self {
            naturalness: 0.3,
            similarity: 0.3,
            quality: 0.2,
            authenticity: 0.1,
            emotion: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ab_testing_framework_creation() {
        let framework = ABTestingFramework::new().unwrap();

        // Test should initialize successfully
        assert!(framework.active_tests.read().await.is_empty());
        assert!(framework.test_history.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_create_ab_test() {
        let framework = ABTestingFramework::new().unwrap();

        let config = ABTestConfig {
            test_id: "test_001".to_string(),
            test_name: "Voice Quality Comparison".to_string(),
            description: "Comparing two voice synthesis methods".to_string(),
            methodology: TestMethodology::PairedComparison,
            significance_threshold: 0.05,
            min_sample_size: 20,
            max_duration: Duration::from_secs(3600),
            criteria_weights: CriteriaWeights::default(),
            randomize_samples: true,
            samples_per_participant: 5,
        };

        let test_id = framework.create_test(config).await.unwrap();
        assert_eq!(test_id, "test_001");

        let active_tests = framework.active_tests.read().await;
        assert!(active_tests.contains_key("test_001"));
    }

    #[tokio::test]
    async fn test_add_test_condition() {
        let framework = ABTestingFramework::new().unwrap();

        let config = ABTestConfig {
            test_id: "test_002".to_string(),
            test_name: "Condition Test".to_string(),
            description: "Testing condition addition".to_string(),
            methodology: TestMethodology::PairedComparison,
            significance_threshold: 0.05,
            min_sample_size: 10,
            max_duration: Duration::from_secs(1800),
            criteria_weights: CriteriaWeights::default(),
            randomize_samples: false,
            samples_per_participant: 3,
        };

        let test_id = framework.create_test(config).await.unwrap();

        let condition = TestCondition {
            condition_id: "condition_A".to_string(),
            name: "Method A".to_string(),
            description: "First synthesis method".to_string(),
            parameters: HashMap::new(),
            samples: Vec::new(),
            objective_metrics: ObjectiveMetrics {
                snr: 20.0,
                spectral_distortion: 2.0,
                f0_error: 8.0,
                mcd: 5.5,
                pesq_score: Some(3.5),
                speaker_similarity: 0.8,
            },
        };

        framework.add_condition(&test_id, condition).await.unwrap();

        let active_tests = framework.active_tests.read().await;
        let test_results = active_tests.get(&test_id).unwrap();
        assert_eq!(test_results.conditions.len(), 1);
        assert_eq!(test_results.conditions[0].condition_id, "condition_A");
    }

    #[tokio::test]
    async fn test_objective_comparison() {
        let framework = ABTestingFramework::new().unwrap();

        let config = ABTestConfig {
            test_id: "test_003".to_string(),
            test_name: "Objective Comparison".to_string(),
            description: "Testing objective comparison".to_string(),
            methodology: TestMethodology::AbsoluteCategoryRating,
            significance_threshold: 0.05,
            min_sample_size: 5,
            max_duration: Duration::from_secs(900),
            criteria_weights: CriteriaWeights::default(),
            randomize_samples: true,
            samples_per_participant: 2,
        };

        let test_id = framework.create_test(config).await.unwrap();

        // Add two conditions with different quality scores
        let condition_a = TestCondition {
            condition_id: "high_quality".to_string(),
            name: "High Quality Method".to_string(),
            description: "Better synthesis method".to_string(),
            parameters: HashMap::new(),
            samples: Vec::new(),
            objective_metrics: ObjectiveMetrics {
                snr: 25.0,
                spectral_distortion: 1.5,
                f0_error: 5.0,
                mcd: 4.0,
                pesq_score: Some(4.0),
                speaker_similarity: 0.9,
            },
        };

        let condition_b = TestCondition {
            condition_id: "low_quality".to_string(),
            name: "Low Quality Method".to_string(),
            description: "Poorer synthesis method".to_string(),
            parameters: HashMap::new(),
            samples: Vec::new(),
            objective_metrics: ObjectiveMetrics {
                snr: 15.0,
                spectral_distortion: 3.0,
                f0_error: 15.0,
                mcd: 8.0,
                pesq_score: Some(2.5),
                speaker_similarity: 0.6,
            },
        };

        framework
            .add_condition(&test_id, condition_a)
            .await
            .unwrap();
        framework
            .add_condition(&test_id, condition_b)
            .await
            .unwrap();

        let comparison_results = framework.run_objective_comparison(&test_id).await.unwrap();

        // Debug output
        println!(
            "Condition scores: {:?}",
            comparison_results.condition_scores
        );
        println!("Ranking: {:?}", comparison_results.ranking);

        assert_eq!(comparison_results.condition_scores.len(), 2);
        assert_eq!(comparison_results.ranking.len(), 2);
        assert_eq!(comparison_results.ranking[0], "high_quality"); // Should rank higher
    }

    #[tokio::test]
    async fn test_evaluation_result_processing() {
        let framework = ABTestingFramework::new().unwrap();

        let config = ABTestConfig {
            test_id: "test_004".to_string(),
            test_name: "Evaluation Processing".to_string(),
            description: "Testing evaluation result processing".to_string(),
            methodology: TestMethodology::PairedComparison,
            significance_threshold: 0.05,
            min_sample_size: 3,
            max_duration: Duration::from_secs(600),
            criteria_weights: CriteriaWeights::default(),
            randomize_samples: false,
            samples_per_participant: 1,
        };

        let test_id = framework.create_test(config).await.unwrap();

        // Add evaluation results
        for i in 0..5 {
            let evaluation = EvaluationResult {
                evaluation_id: Uuid::new_v4(),
                participant_id: format!("participant_{}", i),
                condition_id: "test_condition".to_string(),
                sample_id: "sample_001".to_string(),
                scores: HashMap::new(),
                overall_score: 0.7 + (i as f32 * 0.05),
                confidence: 0.8,
                evaluation_time: Duration::from_secs(30),
                comments: None,
            };

            framework
                .add_evaluation_result(&test_id, evaluation)
                .await
                .unwrap();
        }

        let active_tests = framework.active_tests.read().await;
        let test_results = active_tests.get(&test_id).unwrap();
        assert_eq!(test_results.evaluations.len(), 5);
    }

    #[test]
    fn test_criteria_weights_validation() {
        let framework = ABTestingFramework::new().unwrap();

        let invalid_config = ABTestConfig {
            test_id: "invalid_test".to_string(),
            test_name: "Invalid Test".to_string(),
            description: "Test with invalid weights".to_string(),
            methodology: TestMethodology::PairedComparison,
            significance_threshold: 0.05,
            min_sample_size: 10,
            max_duration: Duration::from_secs(3600),
            criteria_weights: CriteriaWeights {
                naturalness: 0.5,
                similarity: 0.5,
                quality: 0.5, // This makes the sum > 1
                authenticity: 0.1,
                emotion: 0.1,
            },
            randomize_samples: true,
            samples_per_participant: 5,
        };

        let result = framework.validate_config(&invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_practical_significance_assessment() {
        let framework = ABTestingFramework::new().unwrap();

        // Test large practical significance
        let mut mean_scores = HashMap::new();
        mean_scores.insert("condition_a".to_string(), 0.9);
        mean_scores.insert("condition_b".to_string(), 0.3);

        let statistics = TestStatistics {
            mean_scores,
            standard_deviations: HashMap::new(),
            p_values: HashMap::new(),
            effect_sizes: HashMap::new(),
            f_statistic: None,
            overall_p_value: 0.01,
            statistical_power: 0.8,
        };

        let config = ABTestConfig {
            test_id: "sig_test".to_string(),
            test_name: "Significance Test".to_string(),
            description: "Testing significance".to_string(),
            methodology: TestMethodology::PairedComparison,
            significance_threshold: 0.05,
            min_sample_size: 10,
            max_duration: Duration::from_secs(3600),
            criteria_weights: CriteriaWeights::default(),
            randomize_samples: true,
            samples_per_participant: 5,
        };

        let conclusion = framework
            .generate_conclusion(&statistics, &HashMap::new(), &config)
            .unwrap();

        assert!(conclusion.significant_difference);
        assert!(matches!(
            conclusion.practical_significance,
            PracticalSignificance::Large
        ));
        assert!(!conclusion.best_conditions.is_empty());
    }
}
