//! Adaptive quality controller for intelligent quality adjustment and strategy management

use crate::{Error, Result};
use crate::quality::artifact_detection::{DetectedArtifacts, ArtifactType, AdjustmentType};
use crate::quality::metrics::ObjectiveQualityMetrics;
use std::collections::HashMap;
use tracing::{debug, info};

/// Adaptive quality adjustment system
#[derive(Debug, Clone)]
pub struct AdaptiveQualityController {
    /// Current quality target
    quality_target: f32,
    /// Adaptation sensitivity
    adaptation_rate: f32,
    /// Quality history for trend analysis
    quality_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
    /// Quality improvement strategies
    strategies: Vec<QualityStrategy>,
}

/// Quality improvement strategy
#[derive(Debug, Clone)]
pub struct QualityStrategy {
    /// Strategy name
    pub name: String,
    /// Trigger condition
    pub trigger: QualityTrigger,
    /// Adjustment parameters
    pub adjustment: QualityStrategyAdjustment,
    /// Expected effectiveness (0.0 to 1.0)
    pub effectiveness: f32,
    /// Usage count for learning
    pub usage_count: usize,
    /// Success rate for adaptive learning
    pub success_rate: f32,
}

/// Quality trigger conditions
#[derive(Debug, Clone)]
pub enum QualityTrigger {
    /// Triggered when overall quality is below threshold
    OverallQualityBelow(f32),
    /// Triggered when artifact score is above threshold
    ArtifactScoreAbove(f32),
    /// Triggered when specific artifact type exceeds threshold
    SpecificArtifact(ArtifactType, f32),
    /// Triggered when naturalness is below threshold
    NaturalnessBelow(f32),
    /// Triggered when SNR is below threshold
    SnrBelow(f32),
    /// Triggered by multiple conditions (all must be true)
    Combined(Vec<QualityTrigger>),
}

/// Quality strategy adjustment parameters
#[derive(Debug, Clone)]
pub struct QualityStrategyAdjustment {
    /// Adjustment type
    pub adjustment_type: AdjustmentType,
    /// Parameter adjustments
    pub parameter_changes: HashMap<String, f32>,
    /// Processing mode changes
    pub processing_mode_change: Option<String>,
    /// Model selection preference
    pub preferred_model: Option<String>,
}

impl AdaptiveQualityController {
    /// Create new adaptive quality controller
    pub fn new(quality_target: f32) -> Self {
        let mut strategies = Vec::new();

        // Add default quality improvement strategies
        strategies.push(QualityStrategy {
            name: "reduce_conversion_strength".to_string(),
            trigger: QualityTrigger::OverallQualityBelow(0.6),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::ReduceConversion,
                parameter_changes: [("conversion_strength".to_string(), -0.2)].into(),
                processing_mode_change: None,
                preferred_model: None,
            },
            effectiveness: 0.7,
            usage_count: 0,
            success_rate: 0.7,
        });

        strategies.push(QualityStrategy {
            name: "enable_noise_reduction".to_string(),
            trigger: QualityTrigger::SpecificArtifact(ArtifactType::Buzzing, 0.2),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::NoiseReduction,
                parameter_changes: [("noise_reduction_strength".to_string(), 0.8)].into(),
                processing_mode_change: Some("high_quality".to_string()),
                preferred_model: None,
            },
            effectiveness: 0.8,
            usage_count: 0,
            success_rate: 0.75,
        });

        strategies.push(QualityStrategy {
            name: "spectral_smoothing".to_string(),
            trigger: QualityTrigger::SpecificArtifact(ArtifactType::SpectralDiscontinuity, 0.15),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::SpectralSmoothing,
                parameter_changes: [("smoothing_factor".to_string(), 0.6)].into(),
                processing_mode_change: None,
                preferred_model: None,
            },
            effectiveness: 0.6,
            usage_count: 0,
            success_rate: 0.65,
        });

        strategies.push(QualityStrategy {
            name: "pitch_stabilization".to_string(),
            trigger: QualityTrigger::SpecificArtifact(ArtifactType::PitchVariation, 0.25),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::PitchStabilization,
                parameter_changes: [("pitch_smoothing".to_string(), 0.7)].into(),
                processing_mode_change: None,
                preferred_model: None,
            },
            effectiveness: 0.75,
            usage_count: 0,
            success_rate: 0.8,
        });

        strategies.push(QualityStrategy {
            name: "formant_preservation".to_string(),
            trigger: QualityTrigger::SpecificArtifact(ArtifactType::Metallic, 0.2),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::FormantPreservation,
                parameter_changes: [("formant_preservation".to_string(), 0.9)].into(),
                processing_mode_change: None,
                preferred_model: None,
            },
            effectiveness: 0.65,
            usage_count: 0,
            success_rate: 0.7,
        });

        strategies.push(QualityStrategy {
            name: "low_latency_fallback".to_string(),
            trigger: QualityTrigger::Combined(vec![
                QualityTrigger::OverallQualityBelow(0.4),
                QualityTrigger::ArtifactScoreAbove(0.8),
            ]),
            adjustment: QualityStrategyAdjustment {
                adjustment_type: AdjustmentType::ReduceConversion,
                parameter_changes: [
                    ("conversion_strength".to_string(), -0.4),
                    ("processing_quality".to_string(), -0.3),
                ]
                .into(),
                processing_mode_change: Some("low_latency".to_string()),
                preferred_model: Some("lightweight".to_string()),
            },
            effectiveness: 0.5,
            usage_count: 0,
            success_rate: 0.6,
        });

        Self {
            quality_target: quality_target.clamp(0.0, 1.0),
            adaptation_rate: 0.1,
            quality_history: Vec::new(),
            max_history: 10,
            strategies,
        }
    }

    /// Analyze quality and suggest adjustments
    pub fn analyze_and_adjust(
        &mut self,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
        current_params: &HashMap<String, f32>,
    ) -> Result<AdaptiveAdjustmentResult> {
        debug!("Analyzing quality for adaptive adjustments");

        // Update quality history
        self.update_quality_history(objective_quality.overall_score);

        // Evaluate current quality against target
        let quality_gap = self.quality_target - objective_quality.overall_score;

        // Find applicable strategies
        let applicable_strategies = self.find_applicable_strategies(artifacts, objective_quality);

        if applicable_strategies.is_empty() {
            return Ok(AdaptiveAdjustmentResult {
                should_adjust: false,
                selected_strategy: None,
                parameter_adjustments: HashMap::new(),
                processing_mode_change: None,
                preferred_model: None,
                expected_improvement: 0.0,
                confidence: 0.0,
            });
        }

        // Select best strategy based on effectiveness and success rate
        let selected_strategy = self.select_best_strategy(&applicable_strategies);

        // Calculate expected improvement
        let expected_improvement = selected_strategy.effectiveness * quality_gap.abs();

        // Prepare adjustment result
        let adjustment_result = AdaptiveAdjustmentResult {
            should_adjust: quality_gap > 0.05, // Only adjust if significant quality gap
            selected_strategy: Some(selected_strategy.name.clone()),
            parameter_adjustments: self.calculate_parameter_adjustments(
                &selected_strategy.adjustment,
                current_params,
                quality_gap,
            ),
            processing_mode_change: selected_strategy.adjustment.processing_mode_change.clone(),
            preferred_model: selected_strategy.adjustment.preferred_model.clone(),
            expected_improvement,
            confidence: selected_strategy.success_rate,
        };

        info!(
            "Adaptive quality analysis complete: should_adjust={}, expected_improvement={:.3}",
            adjustment_result.should_adjust, adjustment_result.expected_improvement
        );

        Ok(adjustment_result)
    }

    /// Update quality history for trend analysis
    fn update_quality_history(&mut self, quality_score: f32) {
        self.quality_history.push(quality_score);

        // Keep only recent history
        if self.quality_history.len() > self.max_history {
            self.quality_history.remove(0);
        }
    }

    /// Find strategies applicable to current quality issues
    fn find_applicable_strategies(
        &self,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
    ) -> Vec<&QualityStrategy> {
        self.strategies
            .iter()
            .filter(|strategy| {
                self.evaluate_trigger(&strategy.trigger, artifacts, objective_quality)
            })
            .collect()
    }

    /// Evaluate if a trigger condition is met
    fn evaluate_trigger(
        &self,
        trigger: &QualityTrigger,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
    ) -> bool {
        match trigger {
            QualityTrigger::OverallQualityBelow(threshold) => {
                objective_quality.overall_score < *threshold
            }
            QualityTrigger::ArtifactScoreAbove(threshold) => artifacts.overall_score > *threshold,
            QualityTrigger::SpecificArtifact(artifact_type, threshold) => {
                if let Some(&score) = artifacts.artifact_types.get(artifact_type) {
                    score > *threshold
                } else {
                    false
                }
            }
            QualityTrigger::NaturalnessBelow(threshold) => {
                objective_quality.naturalness < *threshold
            }
            QualityTrigger::SnrBelow(threshold) => objective_quality.snr_estimate < *threshold,
            QualityTrigger::Combined(triggers) => triggers
                .iter()
                .all(|t| self.evaluate_trigger(t, artifacts, objective_quality)),
        }
    }

    /// Select the best strategy from applicable ones
    fn select_best_strategy<'a>(&self, strategies: &[&'a QualityStrategy]) -> &'a QualityStrategy {
        strategies
            .iter()
            .max_by(|a, b| {
                let score_a = a.effectiveness * a.success_rate;
                let score_b = b.effectiveness * b.success_rate;
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap()
    }

    /// Calculate parameter adjustments based on strategy and quality gap
    fn calculate_parameter_adjustments(
        &self,
        adjustment: &QualityStrategyAdjustment,
        current_params: &HashMap<String, f32>,
        quality_gap: f32,
    ) -> HashMap<String, f32> {
        let mut adjusted_params = HashMap::new();
        let intensity_factor = (quality_gap.abs() / 0.3).min(1.0); // Scale by quality gap

        for (param_name, base_change) in &adjustment.parameter_changes {
            let current_value = current_params.get(param_name).copied().unwrap_or(1.0);
            let scaled_change = base_change * intensity_factor;
            let new_value = (current_value + scaled_change).clamp(0.0, 2.0);

            adjusted_params.insert(param_name.clone(), new_value);
        }

        adjusted_params
    }

    /// Update strategy effectiveness based on results
    pub fn update_strategy_effectiveness(
        &mut self,
        strategy_name: &str,
        quality_before: f32,
        quality_after: f32,
    ) {
        if let Some(strategy) = self.strategies.iter_mut().find(|s| s.name == strategy_name) {
            strategy.usage_count += 1;

            let improvement = quality_after - quality_before;
            let success = improvement > 0.05; // Consider successful if quality improved by > 5%

            // Update success rate using exponential moving average
            let alpha = 0.1;
            strategy.success_rate =
                strategy.success_rate * (1.0 - alpha) + if success { 1.0 } else { 0.0 } * alpha;

            // Update effectiveness estimate
            if success {
                let actual_effectiveness = improvement.abs().min(1.0);
                strategy.effectiveness =
                    strategy.effectiveness * (1.0 - alpha) + actual_effectiveness * alpha;
            }

            info!(
                "Updated strategy '{}': usage_count={}, success_rate={:.3}, effectiveness={:.3}",
                strategy_name, strategy.usage_count, strategy.success_rate, strategy.effectiveness
            );
        }
    }

    /// Get quality trend from history
    pub fn get_quality_trend(&self) -> QualityTrend {
        if self.quality_history.len() < 3 {
            return QualityTrend::Stable;
        }

        let recent = &self.quality_history[self.quality_history.len() - 3..];
        let trend_slope = (recent[2] - recent[0]) / 2.0;

        if trend_slope > 0.02 {
            QualityTrend::Improving
        } else if trend_slope < -0.02 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }

    /// Set quality target
    pub fn set_quality_target(&mut self, target: f32) {
        self.quality_target = target.clamp(0.0, 1.0);
    }

    /// Get current quality target
    pub fn quality_target(&self) -> f32 {
        self.quality_target
    }

    /// Get strategy statistics
    pub fn get_strategy_stats(&self) -> Vec<StrategyStats> {
        self.strategies
            .iter()
            .map(|s| StrategyStats {
                name: s.name.clone(),
                usage_count: s.usage_count,
                success_rate: s.success_rate,
                effectiveness: s.effectiveness,
            })
            .collect()
    }
}

/// Result of adaptive quality analysis
#[derive(Debug, Clone)]
pub struct AdaptiveAdjustmentResult {
    /// Whether adjustment is recommended
    pub should_adjust: bool,
    /// Selected strategy name
    pub selected_strategy: Option<String>,
    /// Parameter adjustments to apply
    pub parameter_adjustments: HashMap<String, f32>,
    /// Processing mode change recommendation
    pub processing_mode_change: Option<String>,
    /// Preferred model selection
    pub preferred_model: Option<String>,
    /// Expected quality improvement
    pub expected_improvement: f32,
    /// Confidence in the adjustment
    pub confidence: f32,
}

/// Quality trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTrend {
    /// Quality is improving over time
    Improving,
    /// Quality is stable
    Stable,
    /// Quality is degrading over time
    Degrading,
}

/// Strategy statistics for monitoring
#[derive(Debug, Clone)]
pub struct StrategyStats {
    /// Strategy name
    pub name: String,
    /// Number of times used
    pub usage_count: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Effectiveness estimate (0.0 to 1.0)
    pub effectiveness: f32,
}

impl Default for AdaptiveQualityController {
    fn default() -> Self {
        Self::new(0.8) // Default quality target of 80%
    }
}