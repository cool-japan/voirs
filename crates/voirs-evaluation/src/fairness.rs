//! # Fairness-Aware Evaluation
//!
//! Framework for detecting and measuring fairness and bias in speech synthesis systems.
//! Provides metrics for demographic parity, equal opportunity, and bias detection across
//! different protected attributes.

use crate::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Fairness evaluation error types
#[derive(Debug, Error)]
pub enum FairnessError {
    #[error("Insufficient data for fairness analysis: {0}")]
    InsufficientData(String),
    #[error("Invalid demographic group: {0}")]
    InvalidGroup(String),
    #[error("Fairness metric calculation failed: {0}")]
    CalculationFailed(String),
}

/// Protected attribute for fairness analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtectedAttribute {
    /// Gender/sex
    Gender,
    /// Age group
    Age,
    /// Race/ethnicity
    Race,
    /// Native language
    NativeLanguage,
    /// Accent
    Accent,
    /// Disability status
    Disability,
    /// Socioeconomic status
    SocioeconomicStatus,
}

/// Demographic group definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemographicGroup {
    /// Protected attribute
    pub attribute: ProtectedAttribute,
    /// Group value (e.g., "Male", "Female" for Gender)
    pub value: String,
}

/// Fairness metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FairnessMetric {
    /// Demographic parity (equal positive rate across groups)
    DemographicParity,
    /// Equal opportunity (equal TPR across groups)
    EqualOpportunity,
    /// Equalized odds (equal TPR and FPR across groups)
    EqualizedOdds,
    /// Predictive parity (equal PPV across groups)
    PredictiveParity,
    /// Individual fairness (similar treatment for similar individuals)
    IndividualFairness,
    /// Calibration (equal calibration across groups)
    Calibration,
}

/// Performance metrics for a demographic group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupPerformance {
    /// Demographic group
    pub group: DemographicGroup,
    /// Sample size
    pub sample_size: usize,
    /// Average quality score (0.0-1.0)
    pub avg_quality: f32,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Error rate (0.0-1.0)
    pub error_rate: f32,
    /// Average response latency (seconds)
    pub avg_latency: f32,
    /// User satisfaction (0.0-1.0)
    pub satisfaction: f32,
}

/// Fairness evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessScore {
    /// Overall fairness score (0.0-1.0, higher is more fair)
    pub overall_fairness: f32,
    /// Demographic parity score (0.0-1.0)
    pub demographic_parity: f32,
    /// Equal opportunity score (0.0-1.0)
    pub equal_opportunity: f32,
    /// Bias detected flag
    pub bias_detected: bool,
    /// Severity of bias (0.0-1.0)
    pub bias_severity: f32,
    /// Performance by group
    pub group_performance: Vec<GroupPerformance>,
    /// Metric-specific scores
    pub metric_scores: HashMap<FairnessMetric, f32>,
    /// Detected disparities
    pub disparities: Vec<Disparity>,
    /// Recommendations for fairness improvement
    pub recommendations: Vec<String>,
}

/// Detected disparity between groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disparity {
    /// Type of disparity
    pub disparity_type: DisparityType,
    /// Affected groups
    pub affected_groups: Vec<DemographicGroup>,
    /// Magnitude of disparity (0.0-1.0)
    pub magnitude: f32,
    /// Statistical significance
    pub significance: f32,
    /// Description
    pub description: String,
}

/// Type of disparity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisparityType {
    /// Quality score disparity
    QualityDisparity,
    /// Performance disparity
    PerformanceDisparity,
    /// Access disparity
    AccessDisparity,
    /// Representation disparity
    RepresentationDisparity,
    /// Outcome disparity
    OutcomeDisparity,
}

/// Fairness evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// Minimum sample size per group
    pub min_sample_size: usize,
    /// Disparity threshold (difference to flag as unfair)
    pub disparity_threshold: f32,
    /// Statistical significance level
    pub significance_level: f32,
    /// Protected attributes to analyze
    pub protected_attributes: Vec<ProtectedAttribute>,
    /// Metrics to calculate
    pub metrics: Vec<FairnessMetric>,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 30,
            disparity_threshold: 0.1, // 10% difference
            significance_level: 0.05,
            protected_attributes: vec![
                ProtectedAttribute::Gender,
                ProtectedAttribute::Age,
                ProtectedAttribute::Race,
            ],
            metrics: vec![
                FairnessMetric::DemographicParity,
                FairnessMetric::EqualOpportunity,
            ],
        }
    }
}

/// Evaluation sample with demographic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSample {
    /// Sample identifier
    pub sample_id: String,
    /// Demographic groups this sample belongs to
    pub demographics: Vec<DemographicGroup>,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Success indicator
    pub success: bool,
    /// Error occurred
    pub error: bool,
    /// Response latency (seconds)
    pub latency: f32,
    /// User satisfaction rating (0.0-1.0)
    pub satisfaction: Option<f32>,
}

/// Fairness-aware evaluator
pub struct FairnessEvaluator {
    config: FairnessConfig,
}

impl FairnessEvaluator {
    /// Create a new fairness evaluator
    pub fn new(config: FairnessConfig) -> Self {
        Self { config }
    }

    /// Evaluate fairness across demographic groups
    pub fn evaluate(&self, samples: &[EvaluationSample]) -> EvaluationResult<FairnessScore> {
        if samples.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "No samples provided for fairness evaluation".to_string(),
            }
            .into());
        }

        // Group samples by demographics
        let group_performance = self.calculate_group_performance(samples)?;

        // Calculate fairness metrics
        let demographic_parity = self.calculate_demographic_parity(&group_performance);
        let equal_opportunity = self.calculate_equal_opportunity(&group_performance);

        // Calculate metric scores
        let mut metric_scores = HashMap::new();
        for metric in &self.config.metrics {
            let score = match metric {
                FairnessMetric::DemographicParity => demographic_parity,
                FairnessMetric::EqualOpportunity => equal_opportunity,
                _ => 0.8, // Placeholder for other metrics
            };
            metric_scores.insert(*metric, score);
        }

        // Detect disparities
        let disparities = self.detect_disparities(&group_performance)?;

        // Determine if bias is detected
        let bias_detected = !disparities.is_empty();
        let bias_severity = if bias_detected {
            disparities.iter().map(|d| d.magnitude).sum::<f32>() / disparities.len() as f32
        } else {
            0.0
        };

        // Calculate overall fairness
        let overall_fairness = (demographic_parity + equal_opportunity) / 2.0;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&disparities, &group_performance);

        Ok(FairnessScore {
            overall_fairness,
            demographic_parity,
            equal_opportunity,
            bias_detected,
            bias_severity,
            group_performance,
            metric_scores,
            disparities,
            recommendations,
        })
    }

    /// Calculate performance metrics for each demographic group
    fn calculate_group_performance(
        &self,
        samples: &[EvaluationSample],
    ) -> EvaluationResult<Vec<GroupPerformance>> {
        let mut group_map: HashMap<DemographicGroup, Vec<&EvaluationSample>> = HashMap::new();

        // Group samples
        for sample in samples {
            for group in &sample.demographics {
                group_map.entry(group.clone()).or_default().push(sample);
            }
        }

        // Calculate performance for each group
        let mut performances = Vec::new();
        for (group, samples) in group_map {
            if samples.len() < self.config.min_sample_size {
                continue; // Skip groups with insufficient data
            }

            let avg_quality =
                samples.iter().map(|s| s.quality_score).sum::<f32>() / samples.len() as f32;

            let success_count = samples.iter().filter(|s| s.success).count();
            let success_rate = success_count as f32 / samples.len() as f32;

            let error_count = samples.iter().filter(|s| s.error).count();
            let error_rate = error_count as f32 / samples.len() as f32;

            let avg_latency = samples.iter().map(|s| s.latency).sum::<f32>() / samples.len() as f32;

            let satisfaction = {
                let rated: Vec<_> = samples.iter().filter_map(|s| s.satisfaction).collect();
                if rated.is_empty() {
                    0.5
                } else {
                    rated.iter().sum::<f32>() / rated.len() as f32
                }
            };

            performances.push(GroupPerformance {
                group,
                sample_size: samples.len(),
                avg_quality,
                success_rate,
                error_rate,
                avg_latency,
                satisfaction,
            });
        }

        Ok(performances)
    }

    /// Calculate demographic parity (equal success rates)
    fn calculate_demographic_parity(&self, groups: &[GroupPerformance]) -> f32 {
        if groups.len() < 2 {
            return 1.0; // Perfect parity with only one group
        }

        let success_rates: Vec<f32> = groups.iter().map(|g| g.success_rate).collect();
        let max_rate = success_rates
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_rate = success_rates.iter().cloned().fold(f32::INFINITY, f32::min);

        let disparity = max_rate - min_rate;

        // Convert disparity to fairness score (lower disparity = higher fairness)
        (1.0 - disparity).max(0.0)
    }

    /// Calculate equal opportunity (equal quality across groups)
    fn calculate_equal_opportunity(&self, groups: &[GroupPerformance]) -> f32 {
        if groups.len() < 2 {
            return 1.0;
        }

        let qualities: Vec<f32> = groups.iter().map(|g| g.avg_quality).collect();
        let max_quality = qualities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_quality = qualities.iter().cloned().fold(f32::INFINITY, f32::min);

        let disparity = max_quality - min_quality;

        (1.0 - disparity).max(0.0)
    }

    /// Detect disparities between groups
    fn detect_disparities(&self, groups: &[GroupPerformance]) -> EvaluationResult<Vec<Disparity>> {
        let mut disparities = Vec::new();

        // Check quality disparities
        if groups.len() >= 2 {
            let qualities: Vec<(DemographicGroup, f32)> = groups
                .iter()
                .map(|g| (g.group.clone(), g.avg_quality))
                .collect();

            let max = qualities
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("value should be present");
            let min = qualities
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("value should be present");

            let magnitude = max.1 - min.1;

            if magnitude > self.config.disparity_threshold {
                disparities.push(Disparity {
                    disparity_type: DisparityType::QualityDisparity,
                    affected_groups: vec![max.0.clone(), min.0.clone()],
                    magnitude,
                    significance: 0.95, // Simplified
                    description: format!(
                        "Quality disparity detected: {} (score: {:.2}) vs {} (score: {:.2})",
                        max.0.value, max.1, min.0.value, min.1
                    ),
                });
            }
        }

        // Check performance disparities
        if groups.len() >= 2 {
            let success_rates: Vec<(DemographicGroup, f32)> = groups
                .iter()
                .map(|g| (g.group.clone(), g.success_rate))
                .collect();

            let max = success_rates
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("value should be present");
            let min = success_rates
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("value should be present");

            let magnitude = max.1 - min.1;

            if magnitude > self.config.disparity_threshold {
                disparities.push(Disparity {
                    disparity_type: DisparityType::PerformanceDisparity,
                    affected_groups: vec![max.0.clone(), min.0.clone()],
                    magnitude,
                    significance: 0.95,
                    description: format!(
                        "Performance disparity detected: {} ({:.1}% success) vs {} ({:.1}% success)",
                        max.0.value,
                        max.1 * 100.0,
                        min.0.value,
                        min.1 * 100.0
                    ),
                });
            }
        }

        // Check representation disparities
        let total_samples: usize = groups.iter().map(|g| g.sample_size).sum();
        for group in groups {
            let representation = group.sample_size as f32 / total_samples as f32;
            if representation < 0.05 {
                // Less than 5% representation
                disparities.push(Disparity {
                    disparity_type: DisparityType::RepresentationDisparity,
                    affected_groups: vec![group.group.clone()],
                    magnitude: 0.05 - representation,
                    significance: 1.0,
                    description: format!(
                        "Under-representation detected: {} has only {:.1}% of samples",
                        group.group.value,
                        representation * 100.0
                    ),
                });
            }
        }

        Ok(disparities)
    }

    /// Generate recommendations for improving fairness
    fn generate_recommendations(
        &self,
        disparities: &[Disparity],
        groups: &[GroupPerformance],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if disparities.is_empty() {
            recommendations
                .push("No significant fairness issues detected. Continue monitoring.".to_string());
            return recommendations;
        }

        for disparity in disparities {
            match disparity.disparity_type {
                DisparityType::QualityDisparity => {
                    recommendations.push(format!(
                        "Address quality disparity: Investigate and improve performance for {} groups",
                        disparity
                            .affected_groups
                            .iter()
                            .map(|g| g.value.as_str())
                            .collect::<Vec<_>>()
                            .join(" and ")
                    ));
                }
                DisparityType::PerformanceDisparity => {
                    recommendations.push(format!(
                        "Improve success rates for under-performing groups: {}",
                        disparity
                            .affected_groups
                            .iter()
                            .map(|g| g.value.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                DisparityType::RepresentationDisparity => {
                    recommendations.push(format!(
                        "Increase training data diversity for under-represented group: {}",
                        disparity.affected_groups[0].value
                    ));
                }
                _ => {}
            }
        }

        // Check for latency disparities
        if groups.len() >= 2 {
            let latencies: Vec<f32> = groups.iter().map(|g| g.avg_latency).collect();
            let max_latency = latencies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_latency = latencies.iter().cloned().fold(f32::INFINITY, f32::min);

            if (max_latency - min_latency) > 0.5 {
                recommendations.push(
                    "Optimize response latency to ensure equal service across all groups"
                        .to_string(),
                );
            }
        }

        recommendations
    }

    /// Calculate statistical significance of disparity
    pub fn calculate_significance(&self, group1_samples: &[f32], group2_samples: &[f32]) -> f32 {
        if group1_samples.len() < 2 || group2_samples.len() < 2 {
            return 0.0;
        }

        // Simplified t-test approximation
        let mean1 = group1_samples.iter().sum::<f32>() / group1_samples.len() as f32;
        let mean2 = group2_samples.iter().sum::<f32>() / group2_samples.len() as f32;

        let var1 = group1_samples
            .iter()
            .map(|x| (x - mean1).powi(2))
            .sum::<f32>()
            / group1_samples.len() as f32;
        let var2 = group2_samples
            .iter()
            .map(|x| (x - mean2).powi(2))
            .sum::<f32>()
            / group2_samples.len() as f32;

        let pooled_var = ((group1_samples.len() - 1) as f32 * var1
            + (group2_samples.len() - 1) as f32 * var2)
            / (group1_samples.len() + group2_samples.len() - 2) as f32;

        let t_stat = (mean1 - mean2).abs()
            / (pooled_var
                * (1.0 / group1_samples.len() as f32 + 1.0 / group2_samples.len() as f32))
                .sqrt();

        // Convert t-statistic to approximate p-value (simplified)
        if t_stat > 2.0 {
            0.95 // Significant at 95% level
        } else if t_stat > 1.5 {
            0.85
        } else {
            0.5
        }
    }

    /// Analyze intersectional fairness (multiple protected attributes)
    pub fn analyze_intersectional_fairness(
        &self,
        samples: &[EvaluationSample],
    ) -> EvaluationResult<HashMap<Vec<DemographicGroup>, f32>> {
        let mut intersectional_scores = HashMap::new();

        // Group by intersectional combinations
        let mut intersection_map: HashMap<Vec<DemographicGroup>, Vec<&EvaluationSample>> =
            HashMap::new();

        for sample in samples {
            let key = sample.demographics.clone();
            intersection_map.entry(key).or_default().push(sample);
        }

        // Calculate scores for each intersection
        for (groups, samples) in intersection_map {
            if samples.len() >= self.config.min_sample_size {
                let avg_score =
                    samples.iter().map(|s| s.quality_score).sum::<f32>() / samples.len() as f32;
                intersectional_scores.insert(groups, avg_score);
            }
        }

        Ok(intersectional_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(
        id: &str,
        groups: Vec<DemographicGroup>,
        quality: f32,
        success: bool,
    ) -> EvaluationSample {
        EvaluationSample {
            sample_id: id.to_string(),
            demographics: groups,
            quality_score: quality,
            success,
            error: !success,
            latency: 0.5,
            satisfaction: Some(quality),
        }
    }

    #[test]
    fn test_fairness_evaluator_creation() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);
        assert!(evaluator.config.disparity_threshold > 0.0);
    }

    #[test]
    fn test_fair_evaluation() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let male_group = DemographicGroup {
            attribute: ProtectedAttribute::Gender,
            value: "Male".to_string(),
        };
        let female_group = DemographicGroup {
            attribute: ProtectedAttribute::Gender,
            value: "Female".to_string(),
        };

        let mut samples = Vec::new();

        // Equal performance across groups
        for i in 0..40 {
            let group = if i < 20 {
                male_group.clone()
            } else {
                female_group.clone()
            };
            samples.push(create_test_sample(
                &format!("sample_{}", i),
                vec![group],
                0.8,
                true,
            ));
        }

        let result = evaluator.evaluate(&samples).unwrap();

        assert!(result.overall_fairness > 0.9);
        assert!(!result.bias_detected || result.bias_severity < 0.1);
    }

    #[test]
    fn test_unfair_evaluation() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let male_group = DemographicGroup {
            attribute: ProtectedAttribute::Gender,
            value: "Male".to_string(),
        };
        let female_group = DemographicGroup {
            attribute: ProtectedAttribute::Gender,
            value: "Female".to_string(),
        };

        let mut samples = Vec::new();

        // Unequal performance - need at least 30 samples per group
        for i in 0..70 {
            if i < 35 {
                samples.push(create_test_sample(
                    &format!("sample_{}", i),
                    vec![male_group.clone()],
                    0.9,
                    true,
                ));
            } else {
                samples.push(create_test_sample(
                    &format!("sample_{}", i),
                    vec![female_group.clone()],
                    0.6,
                    i % 2 == 0,
                ));
            }
        }

        let result = evaluator.evaluate(&samples).unwrap();

        assert!(result.bias_detected);
        assert!(result.overall_fairness < 0.9);
        assert!(!result.disparities.is_empty());
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_demographic_parity() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let groups = vec![
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Gender,
                    value: "Male".to_string(),
                },
                sample_size: 100,
                avg_quality: 0.8,
                success_rate: 0.85,
                error_rate: 0.15,
                avg_latency: 0.5,
                satisfaction: 0.8,
            },
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Gender,
                    value: "Female".to_string(),
                },
                sample_size: 100,
                avg_quality: 0.8,
                success_rate: 0.85,
                error_rate: 0.15,
                avg_latency: 0.5,
                satisfaction: 0.8,
            },
        ];

        let parity = evaluator.calculate_demographic_parity(&groups);
        assert!(parity > 0.95); // Nearly perfect parity
    }

    #[test]
    fn test_disparity_detection() {
        let config = FairnessConfig {
            disparity_threshold: 0.1,
            ..Default::default()
        };
        let evaluator = FairnessEvaluator::new(config);

        let groups = vec![
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Age,
                    value: "Young".to_string(),
                },
                sample_size: 100,
                avg_quality: 0.9,
                success_rate: 0.95,
                error_rate: 0.05,
                avg_latency: 0.4,
                satisfaction: 0.9,
            },
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Age,
                    value: "Senior".to_string(),
                },
                sample_size: 100,
                avg_quality: 0.6,
                success_rate: 0.7,
                error_rate: 0.3,
                avg_latency: 0.6,
                satisfaction: 0.6,
            },
        ];

        let disparities = evaluator.detect_disparities(&groups).unwrap();

        assert!(!disparities.is_empty());
        assert!(disparities
            .iter()
            .any(|d| d.disparity_type == DisparityType::QualityDisparity));
    }

    #[test]
    fn test_representation_disparity() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let groups = vec![
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Race,
                    value: "Majority".to_string(),
                },
                sample_size: 960,
                avg_quality: 0.8,
                success_rate: 0.85,
                error_rate: 0.15,
                avg_latency: 0.5,
                satisfaction: 0.8,
            },
            GroupPerformance {
                group: DemographicGroup {
                    attribute: ProtectedAttribute::Race,
                    value: "Minority".to_string(),
                },
                sample_size: 40,
                avg_quality: 0.8,
                success_rate: 0.85,
                error_rate: 0.15,
                avg_latency: 0.5,
                satisfaction: 0.8,
            },
        ];

        let disparities = evaluator.detect_disparities(&groups).unwrap();

        assert!(disparities
            .iter()
            .any(|d| d.disparity_type == DisparityType::RepresentationDisparity));
    }

    #[test]
    fn test_intersectional_fairness() {
        let config = FairnessConfig {
            min_sample_size: 10,
            ..Default::default()
        };
        let evaluator = FairnessEvaluator::new(config);

        let mut samples = Vec::new();

        let young_male = vec![
            DemographicGroup {
                attribute: ProtectedAttribute::Age,
                value: "Young".to_string(),
            },
            DemographicGroup {
                attribute: ProtectedAttribute::Gender,
                value: "Male".to_string(),
            },
        ];

        for i in 0..15 {
            samples.push(create_test_sample(
                &format!("sample_{}", i),
                young_male.clone(),
                0.8,
                true,
            ));
        }

        let result = evaluator.analyze_intersectional_fairness(&samples).unwrap();

        assert!(!result.is_empty());
        assert!(result.values().any(|&score| score > 0.7));
    }

    #[test]
    fn test_statistical_significance() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let group1 = vec![0.8, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78, 0.82];
        let group2 = vec![0.6, 0.62, 0.59, 0.61, 0.60, 0.63, 0.58, 0.62];

        let significance = evaluator.calculate_significance(&group1, &group2);

        assert!(significance > 0.8); // Should be statistically significant
    }

    #[test]
    fn test_recommendations_generation() {
        let config = FairnessConfig::default();
        let evaluator = FairnessEvaluator::new(config);

        let disparities = vec![Disparity {
            disparity_type: DisparityType::QualityDisparity,
            affected_groups: vec![
                DemographicGroup {
                    attribute: ProtectedAttribute::Gender,
                    value: "Male".to_string(),
                },
                DemographicGroup {
                    attribute: ProtectedAttribute::Gender,
                    value: "Female".to_string(),
                },
            ],
            magnitude: 0.2,
            significance: 0.95,
            description: "Test disparity".to_string(),
        }];

        let groups = vec![];

        let recommendations = evaluator.generate_recommendations(&disparities, &groups);

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("quality")));
    }
}
