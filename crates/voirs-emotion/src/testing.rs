//! A/B Testing Framework for Systematic Emotion Quality Comparison
//!
//! This module provides comprehensive A/B testing capabilities for comparing
//! different emotion processing implementations, configurations, and parameters.

use crate::core::EmotionProcessor;
use crate::types::{Emotion, EmotionIntensity, EmotionParameters};
use crate::validation::{EvaluationCriteria, PerceptualEvaluation};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Configuration for A/B testing studies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Test name/description
    pub test_name: String,
    /// Minimum number of comparisons required
    pub min_comparisons: usize,
    /// Target confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Statistical power (e.g., 0.8 for 80%)
    pub power: f64,
    /// Effect size to detect
    pub effect_size: f64,
    /// Test emotions to compare
    pub test_emotions: Vec<Emotion>,
    /// Randomization seed for reproducibility
    pub randomization_seed: Option<u64>,
    /// Maximum test duration
    pub max_duration: Duration,
    /// Balance allocation between variants
    pub balanced_allocation: bool,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            test_name: "Emotion A/B Test".to_string(),
            min_comparisons: 30,
            confidence_level: 0.95,
            power: 0.8,
            effect_size: 0.3, // Medium effect size
            test_emotions: vec![
                Emotion::Happy,
                Emotion::Sad,
                Emotion::Angry,
                Emotion::Fear,
                Emotion::Neutral,
            ],
            randomization_seed: None,
            max_duration: Duration::from_secs(7200), // 2 hours
            balanced_allocation: true,
        }
    }
}

/// A/B test variant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestVariant {
    /// Variant identifier
    pub variant_id: String,
    /// Variant name/description
    pub variant_name: String,
    /// Variant configuration parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Whether this is the control variant
    pub is_control: bool,
    /// Allocation weight (for unbalanced tests)
    pub allocation_weight: f64,
}

impl ABTestVariant {
    /// Create new test variant
    pub fn new(variant_id: String, variant_name: String, is_control: bool) -> Self {
        Self {
            variant_id,
            variant_name,
            parameters: HashMap::new(),
            is_control,
            allocation_weight: 1.0,
        }
    }

    /// Add parameter to variant
    pub fn with_parameter<T: serde::Serialize>(
        mut self,
        key: String,
        value: T,
    ) -> crate::Result<Self> {
        let json_value = serde_json::to_value(value).map_err(|e| Error::Serialization(e))?;
        self.parameters.insert(key, json_value);
        Ok(self)
    }

    /// Set allocation weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.allocation_weight = weight.max(0.0);
        self
    }
}

/// Individual comparison result in A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABComparison {
    /// Comparison ID
    pub comparison_id: Uuid,
    /// Evaluator ID
    pub evaluator_id: String,
    /// Test emotion
    pub test_emotion: Emotion,
    /// Emotion intensity
    pub emotion_intensity: EmotionIntensity,
    /// Variant A details
    pub variant_a: ABTestVariant,
    /// Variant B details
    pub variant_b: ABTestVariant,
    /// Evaluation of variant A
    pub evaluation_a: EvaluationCriteria,
    /// Evaluation of variant B
    pub evaluation_b: EvaluationCriteria,
    /// Preferred variant ('A' or 'B')
    pub preference: String,
    /// Confidence in preference (1-10)
    pub confidence: u8,
    /// Comparison timestamp
    pub timestamp: SystemTime,
    /// Time taken for comparison
    pub comparison_duration: Duration,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ABComparison {
    /// Create new comparison
    pub fn new(
        evaluator_id: String,
        test_emotion: Emotion,
        emotion_intensity: EmotionIntensity,
        variant_a: ABTestVariant,
        variant_b: ABTestVariant,
        evaluation_a: EvaluationCriteria,
        evaluation_b: EvaluationCriteria,
        preference: String,
        confidence: u8,
        comparison_duration: Duration,
    ) -> crate::Result<Self> {
        // Validate inputs
        if preference != "A" && preference != "B" {
            return Err(Error::Validation(
                "Preference must be 'A' or 'B'".to_string(),
            ));
        }

        if confidence < 1 || confidence > 10 {
            return Err(Error::Validation(
                "Confidence must be between 1-10".to_string(),
            ));
        }

        evaluation_a.validate()?;
        evaluation_b.validate()?;

        Ok(Self {
            comparison_id: Uuid::new_v4(),
            evaluator_id,
            test_emotion,
            emotion_intensity,
            variant_a,
            variant_b,
            evaluation_a,
            evaluation_b,
            preference,
            confidence,
            timestamp: SystemTime::now(),
            comparison_duration,
            metadata: HashMap::new(),
        })
    }

    /// Get quality score difference (B - A)
    pub fn quality_difference(&self) -> f32 {
        self.evaluation_b.composite_score() - self.evaluation_a.composite_score()
    }

    /// Get winner variant ID
    pub fn winner_variant_id(&self) -> &str {
        if self.preference == "A" {
            &self.variant_a.variant_id
        } else {
            &self.variant_b.variant_id
        }
    }
}

/// Statistical analysis results for A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestStatistics {
    /// Total number of comparisons
    pub total_comparisons: usize,
    /// Number of unique evaluators
    pub unique_evaluators: usize,
    /// Win rate for each variant
    pub win_rates: HashMap<String, f64>,
    /// Average quality scores
    pub average_scores: HashMap<String, f64>,
    /// Standard deviations
    pub score_std_devs: HashMap<String, f64>,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Confidence interval for difference
    pub confidence_interval: (f64, f64),
    /// Is result statistically significant?
    pub is_significant: bool,
    /// Test power achieved
    pub achieved_power: f64,
    /// Recommended action
    pub recommendation: String,
}

/// A/B test manager for running emotion quality comparisons
#[derive(Debug)]
pub struct ABTestManager {
    /// Test configuration
    config: ABTestConfig,
    /// Test variants
    variants: Vec<ABTestVariant>,
    /// Collected comparisons
    comparisons: Vec<ABComparison>,
    /// Test start time
    start_time: SystemTime,
    /// Test ID
    test_id: Uuid,
    /// Random number generator
    rng: fastrand::Rng,
}

impl ABTestManager {
    /// Create new A/B test
    pub fn new(config: ABTestConfig) -> Self {
        let rng = if let Some(seed) = config.randomization_seed {
            fastrand::Rng::with_seed(seed)
        } else {
            fastrand::Rng::new()
        };

        Self {
            config,
            variants: Vec::new(),
            comparisons: Vec::new(),
            start_time: SystemTime::now(),
            test_id: Uuid::new_v4(),
            rng,
        }
    }

    /// Add test variant
    pub fn add_variant(&mut self, variant: ABTestVariant) -> crate::Result<()> {
        // Check for duplicate variant IDs
        if self
            .variants
            .iter()
            .any(|v| v.variant_id == variant.variant_id)
        {
            return Err(Error::Validation(format!(
                "Variant ID '{}' already exists",
                variant.variant_id
            )));
        }

        self.variants.push(variant);
        Ok(())
    }

    /// Add comparison result
    pub fn add_comparison(&mut self, comparison: ABComparison) -> crate::Result<()> {
        // Validate that variants exist
        let variant_ids: Vec<&str> = self
            .variants
            .iter()
            .map(|v| v.variant_id.as_str())
            .collect();

        if !variant_ids.contains(&comparison.variant_a.variant_id.as_str())
            || !variant_ids.contains(&comparison.variant_b.variant_id.as_str())
        {
            return Err(Error::Validation(
                "Unknown variant in comparison".to_string(),
            ));
        }

        // Check test duration
        if self.start_time.elapsed().unwrap_or(Duration::ZERO) > self.config.max_duration {
            return Err(Error::Validation("Test duration exceeded".to_string()));
        }

        self.comparisons.push(comparison);
        Ok(())
    }

    /// Generate next comparison pair
    pub fn next_comparison_pair(&mut self) -> crate::Result<(ABTestVariant, ABTestVariant)> {
        if self.variants.len() < 2 {
            return Err(Error::Validation(
                "At least 2 variants required for comparison".to_string(),
            ));
        }

        // For simplicity, randomly select 2 different variants
        let mut indices: Vec<usize> = (0..self.variants.len()).collect();

        // Shuffle indices
        for i in 0..indices.len() {
            let j = self.rng.usize(0..indices.len());
            indices.swap(i, j);
        }

        let variant_a = self.variants[indices[0]].clone();
        let variant_b = self.variants[indices[1]].clone();

        Ok((variant_a, variant_b))
    }

    /// Calculate test statistics
    pub fn calculate_statistics(&self) -> crate::Result<ABTestStatistics> {
        if self.comparisons.is_empty() {
            return Err(Error::Validation(
                "No comparisons available for statistics".to_string(),
            ));
        }

        let total_comparisons = self.comparisons.len();
        let unique_evaluators = self
            .comparisons
            .iter()
            .map(|c| c.evaluator_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Calculate win rates
        let mut win_counts: HashMap<String, usize> = HashMap::new();
        let mut variant_scores: HashMap<String, Vec<f32>> = HashMap::new();

        for comparison in &self.comparisons {
            // Count wins
            let winner_id = comparison.winner_variant_id();
            *win_counts.entry(winner_id.to_string()).or_insert(0) += 1;

            // Collect scores
            variant_scores
                .entry(comparison.variant_a.variant_id.clone())
                .or_default()
                .push(comparison.evaluation_a.composite_score());

            variant_scores
                .entry(comparison.variant_b.variant_id.clone())
                .or_default()
                .push(comparison.evaluation_b.composite_score());
        }

        // Calculate win rates and average scores
        let mut win_rates = HashMap::new();
        let mut average_scores = HashMap::new();
        let mut score_std_devs = HashMap::new();

        for variant in &self.variants {
            let wins = win_counts.get(&variant.variant_id).unwrap_or(&0);
            let comparisons_with_variant = self
                .comparisons
                .iter()
                .filter(|c| {
                    c.variant_a.variant_id == variant.variant_id
                        || c.variant_b.variant_id == variant.variant_id
                })
                .count();

            let win_rate = if comparisons_with_variant > 0 {
                *wins as f64 / comparisons_with_variant as f64
            } else {
                0.0
            };

            win_rates.insert(variant.variant_id.clone(), win_rate);

            // Calculate average scores and standard deviation
            if let Some(scores) = variant_scores.get(&variant.variant_id) {
                let avg = scores.iter().sum::<f32>() / scores.len() as f32;
                average_scores.insert(variant.variant_id.clone(), avg as f64);

                let variance = scores
                    .iter()
                    .map(|score| {
                        let diff = *score - avg;
                        diff * diff
                    })
                    .sum::<f32>()
                    / scores.len() as f32;
                score_std_devs.insert(variant.variant_id.clone(), variance.sqrt() as f64);
            }
        }

        // Simple statistical significance calculation (t-test approximation)
        let (p_value, effect_size, confidence_interval, is_significant) =
            self.calculate_significance(&variant_scores)?;

        // Calculate achieved power (simplified)
        let achieved_power = if total_comparisons >= self.config.min_comparisons {
            self.config.power
        } else {
            (total_comparisons as f64 / self.config.min_comparisons as f64) * self.config.power
        };

        // Generate recommendation
        let recommendation = self.generate_recommendation(
            &win_rates,
            &average_scores,
            is_significant,
            achieved_power,
        );

        Ok(ABTestStatistics {
            total_comparisons,
            unique_evaluators,
            win_rates,
            average_scores,
            score_std_devs,
            p_value,
            effect_size,
            confidence_interval,
            is_significant,
            achieved_power,
            recommendation,
        })
    }

    /// Calculate statistical significance (simplified t-test)
    fn calculate_significance(
        &self,
        variant_scores: &HashMap<String, Vec<f32>>,
    ) -> crate::Result<(f64, f64, (f64, f64), bool)> {
        if variant_scores.len() < 2 {
            return Ok((1.0, 0.0, (0.0, 0.0), false));
        }

        let variant_ids: Vec<&String> = variant_scores.keys().collect();
        if variant_ids.len() < 2 {
            return Ok((1.0, 0.0, (0.0, 0.0), false));
        }

        // Take first two variants for comparison
        let scores1 = &variant_scores[variant_ids[0]];
        let scores2 = &variant_scores[variant_ids[1]];

        if scores1.is_empty() || scores2.is_empty() {
            return Ok((1.0, 0.0, (0.0, 0.0), false));
        }

        // Calculate means
        let mean1 = scores1.iter().sum::<f32>() / scores1.len() as f32;
        let mean2 = scores2.iter().sum::<f32>() / scores2.len() as f32;

        // Calculate standard deviations
        let var1 =
            scores1.iter().map(|x| (x - mean1).powi(2)).sum::<f32>() / (scores1.len() as f32 - 1.0);
        let var2 =
            scores2.iter().map(|x| (x - mean2).powi(2)).sum::<f32>() / (scores2.len() as f32 - 1.0);

        let std1 = var1.sqrt();
        let std2 = var2.sqrt();

        // Effect size (Cohen's d)
        let pooled_std = ((var1 + var2) / 2.0).sqrt();
        let effect_size = if pooled_std > 0.0 {
            ((mean2 - mean1) / pooled_std).abs() as f64
        } else {
            0.0
        };

        // T-statistic
        let se = (var1 / scores1.len() as f32 + var2 / scores2.len() as f32).sqrt();
        let t_stat = if se > 0.0 {
            (mean2 - mean1).abs() / se
        } else {
            0.0
        };

        // Degrees of freedom
        let df = scores1.len() + scores2.len() - 2;

        // Simplified p-value approximation (assumes normal distribution)
        let p_value = if t_stat > 2.0 {
            0.05 // Roughly significant
        } else if t_stat > 1.0 {
            0.2
        } else {
            0.5
        };

        let is_significant = p_value < (1.0 - self.config.confidence_level);

        // Confidence interval (simplified)
        let margin_of_error = 1.96 * se as f64; // 95% CI
        let diff = (mean2 - mean1) as f64;
        let confidence_interval = (diff - margin_of_error, diff + margin_of_error);

        Ok((p_value, effect_size, confidence_interval, is_significant))
    }

    /// Generate recommendation based on test results
    fn generate_recommendation(
        &self,
        win_rates: &HashMap<String, f64>,
        average_scores: &HashMap<String, f64>,
        is_significant: bool,
        achieved_power: f64,
    ) -> String {
        if self.comparisons.len() < self.config.min_comparisons {
            return format!(
                "Continue testing: Need {} more comparisons to reach minimum sample size",
                self.config.min_comparisons - self.comparisons.len()
            );
        }

        if !is_significant {
            return "No significant difference detected. Consider longer test or larger effect size.".to_string();
        }

        // Find best performing variant
        let best_variant = win_rates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((variant_id, win_rate)) = best_variant {
            format!(
                "Significant difference detected! Variant '{}' performs best with {:.1}% win rate. Recommend implementing this variant.",
                variant_id, win_rate * 100.0
            )
        } else {
            "Test complete but unable to determine clear winner.".to_string()
        }
    }

    /// Check if test is complete
    pub fn is_test_complete(&self) -> bool {
        self.comparisons.len() >= self.config.min_comparisons
            || self.start_time.elapsed().unwrap_or(Duration::ZERO) >= self.config.max_duration
    }

    /// Export test results
    pub fn export_results(&self) -> crate::Result<String> {
        let statistics = self.calculate_statistics()?;

        let export_data = serde_json::json!({
            "test_id": self.test_id,
            "config": self.config,
            "variants": self.variants,
            "statistics": statistics,
            "comparisons": self.comparisons,
            "is_complete": self.is_test_complete(),
            "export_timestamp": SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
        });

        serde_json::to_string_pretty(&export_data).map_err(|e| Error::Serialization(e))
    }

    /// Get test progress summary
    pub fn get_progress_summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();

        summary.insert(
            "total_comparisons".to_string(),
            serde_json::Value::from(self.comparisons.len()),
        );
        summary.insert(
            "required_comparisons".to_string(),
            serde_json::Value::from(self.config.min_comparisons),
        );
        summary.insert(
            "progress_percent".to_string(),
            serde_json::Value::from(
                (self.comparisons.len() as f32 / self.config.min_comparisons as f32 * 100.0)
                    .min(100.0),
            ),
        );
        summary.insert(
            "is_complete".to_string(),
            serde_json::Value::from(self.is_test_complete()),
        );

        let unique_evaluators = self
            .comparisons
            .iter()
            .map(|c| c.evaluator_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();
        summary.insert(
            "unique_evaluators".to_string(),
            serde_json::Value::from(unique_evaluators),
        );

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionDimensions, EmotionIntensity, EmotionParameters};

    #[test]
    fn test_ab_test_variant_creation() {
        let variant =
            ABTestVariant::new("control".to_string(), "Control Variant".to_string(), true)
                .with_parameter("intensity_scaling".to_string(), 1.0)
                .unwrap()
                .with_weight(0.5);

        assert_eq!(variant.variant_id, "control");
        assert!(variant.is_control);
        assert_eq!(variant.allocation_weight, 0.5);
        assert!(variant.parameters.contains_key("intensity_scaling"));
    }

    #[test]
    fn test_ab_comparison_creation() {
        let variant_a = ABTestVariant::new("control".to_string(), "Control".to_string(), true);

        let variant_b = ABTestVariant::new("treatment".to_string(), "Treatment".to_string(), false);

        let criteria_a = EvaluationCriteria {
            naturalness: 7,
            appropriateness: 8,
            perceived_intensity: 6,
            overall_quality: 7,
            correct_emotion: true,
            comments: None,
        };

        let criteria_b = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 8,
            perceived_intensity: 7,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let comparison = ABComparison::new(
            "evaluator_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            variant_a,
            variant_b,
            criteria_a,
            criteria_b,
            "B".to_string(),
            8,
            Duration::from_secs(30),
        );

        assert!(comparison.is_ok());
        let comp = comparison.unwrap();
        assert_eq!(comp.preference, "B");
        assert_eq!(comp.confidence, 8);
        assert_eq!(comp.winner_variant_id(), "treatment");
        assert!(comp.quality_difference() > 0.0); // B should score higher than A
    }

    #[test]
    fn test_ab_test_manager() {
        let config = ABTestConfig {
            min_comparisons: 2,
            ..Default::default()
        };

        let mut manager = ABTestManager::new(config);

        // Add variants
        let control =
            ABTestVariant::new("control".to_string(), "Control Variant".to_string(), true);

        let treatment = ABTestVariant::new(
            "treatment".to_string(),
            "Treatment Variant".to_string(),
            false,
        );

        assert!(manager.add_variant(control.clone()).is_ok());
        assert!(manager.add_variant(treatment.clone()).is_ok());

        // Test comparison pair generation
        let (var_a, var_b) = manager.next_comparison_pair().unwrap();
        assert_ne!(var_a.variant_id, var_b.variant_id);

        // Add some comparisons
        let criteria_a = EvaluationCriteria {
            naturalness: 7,
            appropriateness: 8,
            perceived_intensity: 6,
            overall_quality: 7,
            correct_emotion: true,
            comments: None,
        };

        let criteria_b = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 8,
            perceived_intensity: 7,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let comparison1 = ABComparison::new(
            "eval_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            control.clone(),
            treatment.clone(),
            criteria_a.clone(),
            criteria_b.clone(),
            "B".to_string(),
            8,
            Duration::from_secs(30),
        )
        .unwrap();

        let comparison2 = ABComparison::new(
            "eval_002".to_string(),
            Emotion::Sad,
            EmotionIntensity::new(0.6),
            control,
            treatment,
            criteria_a,
            criteria_b,
            "B".to_string(),
            7,
            Duration::from_secs(35),
        )
        .unwrap();

        assert!(manager.add_comparison(comparison1).is_ok());
        assert!(manager.add_comparison(comparison2).is_ok());
        assert!(manager.is_test_complete());

        let stats = manager.calculate_statistics().unwrap();
        assert_eq!(stats.total_comparisons, 2);
        assert_eq!(stats.unique_evaluators, 2);

        // Treatment should have 100% win rate
        assert_eq!(stats.win_rates.get("treatment").unwrap_or(&0.0), &1.0);
        assert_eq!(stats.win_rates.get("control").unwrap_or(&1.0), &0.0);
    }

    #[test]
    fn test_statistical_calculations() {
        let config = ABTestConfig::default();
        let mut manager = ABTestManager::new(config);

        let control = ABTestVariant::new("control".to_string(), "Control".to_string(), true);
        let treatment = ABTestVariant::new("treatment".to_string(), "Treatment".to_string(), false);

        manager.add_variant(control.clone()).unwrap();
        manager.add_variant(treatment.clone()).unwrap();

        // Add multiple comparisons to test statistics
        for i in 0..10 {
            let criteria_a = EvaluationCriteria {
                naturalness: 6 + (i % 3) as u8,
                appropriateness: 7,
                perceived_intensity: 6,
                overall_quality: 6 + (i % 2) as u8,
                correct_emotion: true,
                comments: None,
            };

            let criteria_b = EvaluationCriteria {
                naturalness: 8,
                appropriateness: 8,
                perceived_intensity: 7,
                overall_quality: 8,
                correct_emotion: true,
                comments: None,
            };

            let preference = if i < 7 { "B" } else { "A" };

            let comparison = ABComparison::new(
                format!("eval_{:03}", i),
                Emotion::Happy,
                EmotionIntensity::new(0.8),
                control.clone(),
                treatment.clone(),
                criteria_a,
                criteria_b,
                preference.to_string(),
                8,
                Duration::from_secs(30),
            )
            .unwrap();

            manager.add_comparison(comparison).unwrap();
        }

        let stats = manager.calculate_statistics().unwrap();
        assert_eq!(stats.total_comparisons, 10);

        // Treatment should win most comparisons (7 out of 10)
        assert_eq!(stats.win_rates.get("treatment").unwrap_or(&0.0), &0.7);
        assert!(stats.average_scores.contains_key("control"));
        assert!(stats.average_scores.contains_key("treatment"));
    }
}
