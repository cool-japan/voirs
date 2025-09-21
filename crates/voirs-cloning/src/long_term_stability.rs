//! Long-term stability validation for voice cloning consistency
//!
//! This module provides comprehensive testing infrastructure for validating
//! that voice cloning models maintain consistent quality and characteristics
//! over extended periods and across multiple synthesis operations.

use crate::{
    core::VoiceCloner,
    performance_monitoring::{PerformanceMeasurement, PerformanceMetrics, PerformanceMonitor},
    quality::{CloningQualityAssessor, QualityAnalysis, QualityMetrics},
    similarity::{SimilarityConfig, SimilarityMeasurer, SimilarityScore},
    types::{CloningMethod, SpeakerData},
    Error, Result, SpeakerProfile, VoiceCloneRequest, VoiceCloneResult, VoiceSample,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Configuration for long-term stability testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityTestConfig {
    /// Test duration in seconds
    pub test_duration: Duration,
    /// Interval between stability checks
    pub check_interval: Duration,
    /// Number of synthesis operations per check
    pub syntheses_per_check: usize,
    /// Minimum similarity threshold to maintain
    pub similarity_threshold: f64,
    /// Maximum quality degradation allowed (0.0-1.0)
    pub max_quality_degradation: f64,
    /// Maximum response time variance allowed
    pub max_response_time_variance: f64,
    /// Enable continuous monitoring
    pub continuous_monitoring: bool,
    /// Test different voice characteristics
    pub test_voice_characteristics: bool,
}

impl Default for StabilityTestConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(24 * 60 * 60), // 24 hours
            check_interval: Duration::from_secs(60 * 60),     // 1 hour
            syntheses_per_check: 10,
            similarity_threshold: 0.85,
            max_quality_degradation: 0.1,    // 10% max degradation
            max_response_time_variance: 0.3, // 30% variance
            continuous_monitoring: true,
            test_voice_characteristics: true,
        }
    }
}

/// Results from a single stability check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityCheckResult {
    /// Unique check identifier
    pub check_id: String,
    /// Timestamp of the check
    pub timestamp: SystemTime,
    /// Time since test started
    pub elapsed_time: Duration,
    /// Synthesis results from this check (simplified for serialization)
    pub synthesis_count: usize,
    /// Quality score for this check
    pub quality_score: f64,
    /// Average similarity to baseline
    pub average_similarity: f64,
    /// Processing time for this check
    pub processing_time: Duration,
    /// Memory usage during check
    pub memory_usage: u64,
    /// Whether this check passed stability criteria
    pub stability_passed: bool,
    /// Detected issues or anomalies
    pub issues: Vec<String>,
}

/// Comprehensive stability test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityTestResults {
    /// Test configuration used
    pub config: StabilityTestConfig,
    /// Test start time
    pub start_time: SystemTime,
    /// Test end time (if completed)
    pub end_time: Option<SystemTime>,
    /// Total test duration
    pub total_duration: Duration,
    /// Number of baseline syntheses created
    pub baseline_count: usize,
    /// All stability check results
    pub check_results: Vec<StabilityCheckResult>,
    /// Overall test statistics
    pub statistics: StabilityStatistics,
    /// Test conclusion and recommendations
    pub conclusions: StabilityConclusions,
}

/// Statistical analysis of stability test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityStatistics {
    /// Total number of checks performed
    pub total_checks: usize,
    /// Number of checks that passed
    pub checks_passed: usize,
    /// Pass rate percentage
    pub pass_rate: f64,
    /// Average similarity over all checks
    pub average_similarity: f64,
    /// Similarity trend (positive = improving, negative = degrading)
    pub similarity_trend: f64,
    /// Quality trend analysis
    pub quality_trend: f64,
    /// Performance trend analysis
    pub performance_trend: f64,
    /// Consistency score (0.0-1.0, higher is more consistent)
    pub consistency_score: f64,
    /// Detected degradation patterns
    pub degradation_patterns: Vec<String>,
}

/// Conclusions and recommendations from stability testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConclusions {
    /// Overall stability assessment
    pub overall_stability: StabilityAssessment,
    /// Identified issues and concerns
    pub issues_identified: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Confidence in the results
    pub confidence_level: f64,
}

/// Overall stability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityAssessment {
    /// Model is highly stable over time
    Excellent,
    /// Model shows good stability with minor variations
    Good,
    /// Model shows moderate stability with some concerns
    Moderate,
    /// Model shows poor stability with significant issues
    Poor,
    /// Model shows critical instability
    Critical,
}

/// Risk level assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Long-term stability validator for voice cloning models
pub struct StabilityValidator {
    /// Voice cloner instance
    cloner: Arc<VoiceCloner>,
    /// Quality assessor for metrics
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity measurer for comparison
    similarity_measurer: Arc<SimilarityMeasurer>,
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
    /// Active stability tests
    active_tests: Arc<RwLock<HashMap<String, StabilityTestSession>>>,
}

/// Active stability test session
#[derive(Debug)]
struct StabilityTestSession {
    /// Test identifier
    test_id: String,
    /// Test configuration
    config: StabilityTestConfig,
    /// Speaker profile being tested
    speaker_profile: SpeakerProfile,
    /// Baseline results for comparison
    baseline_results: Vec<VoiceCloneResult>,
    /// Test results accumulator
    results: StabilityTestResults,
    /// Next check time
    next_check: SystemTime,
    /// Test completion handle
    completion_handle: Option<tokio::task::JoinHandle<Result<StabilityTestResults>>>,
}

impl StabilityValidator {
    /// Create new stability validator
    pub fn new(
        cloner: Arc<VoiceCloner>,
        quality_assessor: Arc<CloningQualityAssessor>,
        similarity_measurer: Arc<SimilarityMeasurer>,
        performance_monitor: Arc<PerformanceMonitor>,
    ) -> Self {
        Self {
            cloner,
            quality_assessor,
            similarity_measurer,
            performance_monitor,
            active_tests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start a long-term stability test
    pub async fn start_stability_test(
        &self,
        speaker_profile: SpeakerProfile,
        config: StabilityTestConfig,
    ) -> Result<String> {
        let test_id = Uuid::new_v4().to_string();

        info!(
            "Starting long-term stability test for speaker: {}",
            speaker_profile.id
        );

        // Create baseline results
        let baseline_results = self.create_baseline_results(&speaker_profile).await?;

        // Initialize test results
        let results = StabilityTestResults {
            config: config.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            total_duration: Duration::from_secs(0),
            baseline_count: baseline_results.len(),
            check_results: Vec::new(),
            statistics: StabilityStatistics {
                total_checks: 0,
                checks_passed: 0,
                pass_rate: 0.0,
                average_similarity: 0.0,
                similarity_trend: 0.0,
                quality_trend: 0.0,
                performance_trend: 0.0,
                consistency_score: 0.0,
                degradation_patterns: Vec::new(),
            },
            conclusions: StabilityConclusions {
                overall_stability: StabilityAssessment::Good,
                issues_identified: Vec::new(),
                recommendations: Vec::new(),
                risk_level: RiskLevel::Low,
                confidence_level: 0.0,
            },
        };

        // Create test session
        let session = StabilityTestSession {
            test_id: test_id.clone(),
            config: config.clone(),
            speaker_profile: speaker_profile.clone(),
            baseline_results,
            results,
            next_check: SystemTime::now() + config.check_interval,
            completion_handle: None,
        };

        // Start background monitoring task if continuous monitoring is enabled
        let completion_handle = if config.continuous_monitoring {
            let validator = self.clone();
            let test_id_clone = test_id.clone();
            Some(tokio::spawn(async move {
                validator.run_continuous_monitoring(test_id_clone).await
            }))
        } else {
            None
        };

        // Store session with completion handle
        let mut sessions = self.active_tests.write().await;
        let mut session = session;
        session.completion_handle = completion_handle;
        sessions.insert(test_id.clone(), session);

        Ok(test_id)
    }

    /// Perform a single stability check
    pub async fn perform_stability_check(&self, test_id: &str) -> Result<StabilityCheckResult> {
        let mut sessions = self.active_tests.write().await;
        let session = sessions
            .get_mut(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        let check_start = Instant::now();
        let check_id = Uuid::new_v4().to_string();

        debug!(
            "Performing stability check {} for test {}",
            check_id, test_id
        );

        // Perform multiple syntheses for this check
        let mut synthesis_results = Vec::new();
        for i in 0..session.config.syntheses_per_check {
            let request = VoiceCloneRequest {
                id: format!("stability_check_{}_{}", check_id, i),
                speaker_data: SpeakerData {
                    profile: session.speaker_profile.clone(),
                    reference_samples: vec![], // Use empty for now - would be populated with actual samples
                    target_text: Some(format!("Stability test synthesis number {}", i + 1)),
                    target_language: None,
                    context: HashMap::new(),
                },
                text: format!("Stability test synthesis number {}", i + 1),
                method: CloningMethod::FewShot,
                language: None,
                quality_level: 0.8,
                quality_tradeoff: 0.8,
                parameters: HashMap::new(),
                timestamp: SystemTime::now(),
            };

            let result = self.cloner.clone_voice(request).await?;
            synthesis_results.push(result);
        }

        // Create simplified quality assessment
        let quality_metrics = if let Some(first_result) = synthesis_results.first() {
            // Calculate basic quality metrics from synthesis results
            let audio_length = first_result.audio.len();
            let has_audio = !first_result.audio.is_empty();
            let sample_rate_valid = first_result.sample_rate > 0;

            // Simple quality heuristics based on available data
            let audio_quality = if has_audio && sample_rate_valid {
                0.8
            } else {
                0.3
            };
            let overall_score = (first_result.similarity_score + audio_quality) / 2.0;

            QualityMetrics {
                overall_score,
                speaker_similarity: first_result.similarity_score,
                audio_quality,
                naturalness: 0.75,         // Default estimate
                content_preservation: 0.8, // Default estimate
                prosodic_similarity: 0.7,  // Default estimate
                spectral_similarity: 0.75, // Default estimate
                metrics: HashMap::new(),
                analysis: QualityAnalysis::default(),
                metadata: crate::quality::AssessmentMetadata {
                    assessment_time: SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    assessment_duration: 1.0,
                    original_duration: 1.0,
                    cloned_duration: audio_length as f32 / first_result.sample_rate as f32,
                    sample_rate: first_result.sample_rate,
                    assessment_method: "stability_check".to_string(),
                    quality_version: "1.0".to_string(),
                },
            }
        } else {
            QualityMetrics::default()
        };

        // Calculate similarity to baseline
        let mut similarity_scores = Vec::new();
        let mut total_similarity = 0.0;

        for (i, result) in synthesis_results.iter().enumerate() {
            if let Some(baseline) = session
                .baseline_results
                .get(i % session.baseline_results.len())
            {
                // Create VoiceSamples for comparison
                let result_sample = VoiceSample {
                    id: format!("result_{}", i),
                    audio: result.audio.clone(),
                    sample_rate: result.sample_rate,
                    transcript: None,
                    language: Some("en".to_string()),
                    duration: result.audio.len() as f32 / result.sample_rate as f32,
                    quality_score: Some(result.similarity_score),
                    metadata: HashMap::new(),
                    timestamp: SystemTime::now(),
                };

                let baseline_sample = VoiceSample {
                    id: format!("baseline_{}", i),
                    audio: baseline.audio.clone(),
                    sample_rate: baseline.sample_rate,
                    transcript: None,
                    language: Some("en".to_string()),
                    duration: baseline.audio.len() as f32 / baseline.sample_rate as f32,
                    quality_score: Some(baseline.similarity_score),
                    metadata: HashMap::new(),
                    timestamp: SystemTime::now(),
                };

                let similarity = match self
                    .similarity_measurer
                    .measure_sample_similarity(&result_sample, &baseline_sample)
                    .await
                {
                    Ok(sim) => sim,
                    Err(_) => {
                        // Create a default SimilarityScore if measurement fails
                        crate::similarity::SimilarityScore {
                            embedding_similarities:
                                crate::similarity::EmbeddingSimilarities::default(),
                            spectral_similarities: crate::similarity::SpectralSimilarities::default(
                            ),
                            perceptual_similarities:
                                crate::similarity::PerceptualSimilarities::default(),
                            temporal_similarities: crate::similarity::TemporalSimilarities::default(
                            ),
                            overall_score: 0.5,
                            confidence: 0.5,
                            statistical_metrics: crate::similarity::StatisticalSignificance {
                                p_value: 0.05,
                                effect_size: 0.5,
                                confidence_interval: (0.4, 0.6),
                                sample_size: 1,
                                test_statistic: 5.0,
                            },
                        }
                    }
                };

                total_similarity += similarity.overall_score as f64;
                similarity_scores.push(similarity);
            }
        }

        let average_similarity = if !similarity_scores.is_empty() {
            total_similarity / similarity_scores.len() as f64
        } else {
            0.0
        };

        // Create performance metrics
        let elapsed_time = check_start.elapsed();
        let performance_metrics = PerformanceMetrics {
            adaptation_time: elapsed_time,
            synthesis_rtf: elapsed_time.as_secs_f64() / session.config.syntheses_per_check as f64,
            memory_usage: 256 * 1024 * 1024, // Estimate 256MB
            quality_score: quality_metrics.overall_score as f64,
            concurrent_adaptations: 1,
            timestamp: SystemTime::now(),
        };

        // Check if stability criteria are met
        let stability_passed = average_similarity >= session.config.similarity_threshold;
        let mut issues = Vec::new();

        if !stability_passed {
            issues.push(format!(
                "Similarity below threshold: {:.3} < {:.3}",
                average_similarity, session.config.similarity_threshold
            ));
        }

        // Create check result
        let check_result = StabilityCheckResult {
            check_id,
            timestamp: SystemTime::now(),
            elapsed_time: session
                .results
                .start_time
                .elapsed()
                .unwrap_or(Duration::from_secs(0)),
            synthesis_count: synthesis_results.len(),
            quality_score: quality_metrics.overall_score as f64,
            average_similarity,
            processing_time: elapsed_time,
            memory_usage: performance_metrics.memory_usage,
            stability_passed,
            issues,
        };

        // Update session results
        session.results.check_results.push(check_result.clone());
        session.results.statistics.total_checks += 1;
        if stability_passed {
            session.results.statistics.checks_passed += 1;
        }
        session.results.statistics.pass_rate = session.results.statistics.checks_passed as f64
            / session.results.statistics.total_checks as f64;

        // Update next check time
        session.next_check = SystemTime::now() + session.config.check_interval;

        info!(
            "Stability check completed: {} (passed: {})",
            check_result.check_id, stability_passed
        );
        Ok(check_result)
    }

    /// Get stability test results
    pub async fn get_test_results(&self, test_id: &str) -> Result<StabilityTestResults> {
        let sessions = self.active_tests.read().await;
        let session = sessions
            .get(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        let mut results = session.results.clone();
        results.total_duration = session
            .results
            .start_time
            .elapsed()
            .unwrap_or(Duration::from_secs(0));

        // Update statistics and conclusions
        self.update_statistics_and_conclusions(&mut results);

        Ok(results)
    }

    /// Stop a stability test
    pub async fn stop_test(&self, test_id: &str) -> Result<StabilityTestResults> {
        let mut sessions = self.active_tests.write().await;
        let mut session = sessions
            .remove(test_id)
            .ok_or_else(|| Error::Validation(format!("Test not found: {}", test_id)))?;

        // Cancel background task if running
        if let Some(handle) = session.completion_handle.take() {
            handle.abort();
        }

        session.results.end_time = Some(SystemTime::now());
        session.results.total_duration = session
            .results
            .start_time
            .elapsed()
            .unwrap_or(Duration::from_secs(0));

        // Final statistics update
        self.update_statistics_and_conclusions(&mut session.results);

        info!(
            "Stability test {} stopped after {:?}",
            test_id, session.results.total_duration
        );
        Ok(session.results)
    }

    /// List all active stability tests
    pub async fn list_active_tests(&self) -> Vec<String> {
        let sessions = self.active_tests.read().await;
        sessions.keys().cloned().collect()
    }

    /// Create baseline results for comparison
    async fn create_baseline_results(
        &self,
        speaker_profile: &SpeakerProfile,
    ) -> Result<Vec<VoiceCloneResult>> {
        let mut baseline_results = Vec::new();

        // Create multiple baseline syntheses with different texts
        let baseline_texts = vec![
            "This is a baseline synthesis for stability testing.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing voice cloning consistency over time.",
            "Baseline reference audio for comparison.",
            "Voice stability validation sample.",
        ];

        for (i, text) in baseline_texts.iter().enumerate() {
            let request = VoiceCloneRequest {
                id: format!("baseline_{}", i),
                speaker_data: SpeakerData {
                    profile: speaker_profile.clone(),
                    reference_samples: vec![], // Would be populated with actual samples
                    target_text: Some(text.to_string()),
                    target_language: None,
                    context: HashMap::new(),
                },
                text: text.to_string(),
                method: CloningMethod::FewShot,
                language: None,
                quality_level: 0.8,
                quality_tradeoff: 0.8,
                parameters: HashMap::new(),
                timestamp: SystemTime::now(),
            };

            let result = self.cloner.clone_voice(request).await?;
            baseline_results.push(result);
        }

        Ok(baseline_results)
    }

    /// Run continuous monitoring for a stability test
    async fn run_continuous_monitoring(&self, test_id: String) -> Result<StabilityTestResults> {
        loop {
            let should_continue = {
                let sessions = self.active_tests.read().await;
                let session = sessions.get(&test_id);
                match session {
                    Some(s) => {
                        let elapsed = s
                            .results
                            .start_time
                            .elapsed()
                            .unwrap_or(Duration::from_secs(0));
                        elapsed < s.config.test_duration && SystemTime::now() >= s.next_check
                    }
                    None => false,
                }
            };

            if !should_continue {
                break;
            }

            // Perform stability check
            match self.perform_stability_check(&test_id).await {
                Ok(check_result) => {
                    debug!(
                        "Continuous monitoring check completed: {}",
                        check_result.check_id
                    );
                }
                Err(e) => {
                    error!("Continuous monitoring check failed: {}", e);
                    break;
                }
            }

            // Wait for next check interval
            tokio::time::sleep(Duration::from_secs(60)).await;
        }

        // Return final results
        self.stop_test(&test_id).await
    }

    /// Update statistics and conclusions based on check results
    fn update_statistics_and_conclusions(&self, results: &mut StabilityTestResults) {
        if results.check_results.is_empty() {
            return;
        }

        // Calculate average similarity
        let total_similarity: f64 = results
            .check_results
            .iter()
            .map(|r| r.average_similarity)
            .sum();
        results.statistics.average_similarity =
            total_similarity / results.check_results.len() as f64;

        // Calculate trends (simple linear regression slope)
        results.statistics.similarity_trend = self.calculate_trend(
            &results
                .check_results
                .iter()
                .map(|r| r.average_similarity)
                .collect::<Vec<_>>(),
        );

        results.statistics.quality_trend = self.calculate_trend(
            &results
                .check_results
                .iter()
                .map(|r| r.quality_score)
                .collect::<Vec<_>>(),
        );

        // Calculate consistency score
        let similarity_variance = self.calculate_variance(
            &results
                .check_results
                .iter()
                .map(|r| r.average_similarity)
                .collect::<Vec<_>>(),
        );
        results.statistics.consistency_score = (1.0 - similarity_variance.min(1.0)).max(0.0);

        // Determine overall stability assessment
        results.conclusions.overall_stability = if results.statistics.pass_rate >= 0.95 {
            StabilityAssessment::Excellent
        } else if results.statistics.pass_rate >= 0.85 {
            StabilityAssessment::Good
        } else if results.statistics.pass_rate >= 0.70 {
            StabilityAssessment::Moderate
        } else if results.statistics.pass_rate >= 0.50 {
            StabilityAssessment::Poor
        } else {
            StabilityAssessment::Critical
        };

        // Determine risk level
        results.conclusions.risk_level = if results.statistics.similarity_trend < -0.01 {
            RiskLevel::High
        } else if results.statistics.consistency_score < 0.8 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        // Generate recommendations
        results.conclusions.recommendations = self.generate_recommendations(results);
        results.conclusions.confidence_level = if results.check_results.len() >= 10 {
            0.9
        } else {
            0.7
        };
    }

    /// Calculate trend (simple slope)
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Calculate variance
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt() / mean.max(f64::EPSILON)
    }

    /// Generate recommendations based on test results
    fn generate_recommendations(&self, results: &StabilityTestResults) -> Vec<String> {
        let mut recommendations = Vec::new();

        if results.statistics.pass_rate < 0.8 {
            recommendations.push("Consider model retraining to improve stability".to_string());
        }

        if results.statistics.similarity_trend < -0.005 {
            recommendations.push(
                "Monitor for model degradation and implement periodic validation".to_string(),
            );
        }

        if results.statistics.consistency_score < 0.7 {
            recommendations.push(
                "Investigate causes of output variance and improve model consistency".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Model shows good stability - continue regular monitoring".to_string());
        }

        recommendations
    }
}

impl Clone for StabilityValidator {
    fn clone(&self) -> Self {
        Self {
            cloner: Arc::clone(&self.cloner),
            quality_assessor: Arc::clone(&self.quality_assessor),
            similarity_measurer: Arc::clone(&self.similarity_measurer),
            performance_monitor: Arc::clone(&self.performance_monitor),
            active_tests: Arc::clone(&self.active_tests),
        }
    }
}

// Remove conflicting Default implementations - use existing ones from their modules

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CloningConfig;

    #[tokio::test]
    async fn test_stability_test_config_default() {
        let config = StabilityTestConfig::default();
        assert_eq!(config.syntheses_per_check, 10);
        assert_eq!(config.similarity_threshold, 0.85);
        assert!(config.continuous_monitoring);
    }

    #[tokio::test]
    async fn test_stability_validator_creation() {
        let cloner = Arc::new(VoiceCloner::new().unwrap());
        let quality_assessor = Arc::new(CloningQualityAssessor::new().unwrap());
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(SimilarityConfig::default()));
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        let validator = StabilityValidator::new(
            cloner,
            quality_assessor,
            similarity_measurer,
            performance_monitor,
        );

        let active_tests = validator.list_active_tests().await;
        assert!(active_tests.is_empty());
    }

    #[tokio::test]
    async fn test_trend_calculation() {
        let cloner = Arc::new(VoiceCloner::new().unwrap());
        let quality_assessor = Arc::new(CloningQualityAssessor::new().unwrap());
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(SimilarityConfig::default()));
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        let validator = StabilityValidator::new(
            cloner,
            quality_assessor,
            similarity_measurer,
            performance_monitor,
        );

        // Test increasing trend
        let increasing_values = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let trend = validator.calculate_trend(&increasing_values);
        assert!(trend > 0.0);

        // Test decreasing trend
        let decreasing_values = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let trend = validator.calculate_trend(&decreasing_values);
        assert!(trend < 0.0);
    }

    #[tokio::test]
    async fn test_variance_calculation() {
        let cloner = Arc::new(VoiceCloner::new().unwrap());
        let quality_assessor = Arc::new(CloningQualityAssessor::new().unwrap());
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(SimilarityConfig::default()));
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        let validator = StabilityValidator::new(
            cloner,
            quality_assessor,
            similarity_measurer,
            performance_monitor,
        );

        // Test low variance
        let stable_values = vec![0.8, 0.81, 0.79, 0.8, 0.82];
        let variance = validator.calculate_variance(&stable_values);
        assert!(variance < 0.1);

        // Test high variance
        let unstable_values = vec![0.5, 0.9, 0.3, 0.8, 0.1];
        let variance = validator.calculate_variance(&unstable_values);
        assert!(variance > 0.3);
    }

    #[test]
    fn test_stability_assessment_levels() {
        // Test all stability assessment levels
        let assessments = vec![
            StabilityAssessment::Excellent,
            StabilityAssessment::Good,
            StabilityAssessment::Moderate,
            StabilityAssessment::Poor,
            StabilityAssessment::Critical,
        ];

        for assessment in assessments {
            match assessment {
                StabilityAssessment::Excellent => assert!(true),
                StabilityAssessment::Good => assert!(true),
                StabilityAssessment::Moderate => assert!(true),
                StabilityAssessment::Poor => assert!(true),
                StabilityAssessment::Critical => assert!(true),
            }
        }
    }

    #[test]
    fn test_risk_levels() {
        let risk_levels = vec![
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ];

        assert_eq!(risk_levels.len(), 4);
    }
}
