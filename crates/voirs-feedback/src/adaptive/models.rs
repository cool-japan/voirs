//! Data models for adaptive feedback system
//!
//! This module contains the core data structures used for user modeling,
//! learning algorithms, and adaptive state management.

use super::types::*;
use crate::progress::TrendDirection;
use crate::traits::{AdaptiveState, FocusArea, PerformanceData, UserInteraction};
use crate::FeedbackError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// User model for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserModel {
    /// User ID
    pub user_id: String,
    /// Overall skill level [0.0, 1.0]
    pub skill_level: f32,
    /// Learning rate [0.0, 1.0]
    pub learning_rate: f32,
    /// Consistency score [0.0, 1.0]
    pub consistency_score: f32,
    /// Skill breakdown by area
    pub skill_breakdown: HashMap<FocusArea, f32>,
    /// Interaction history
    pub interaction_history: VecDeque<UserInteraction>,
    /// Performance history
    pub performance_history: VecDeque<PerformanceData>,
    /// Adaptive state
    pub adaptive_state: AdaptiveState,
    /// Model confidence [0.0, 1.0]
    pub confidence: f32,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Learning algorithm configuration and state
#[derive(Debug, Clone)]
pub struct LearningAlgorithm {
    /// Algorithm type identifier
    pub algorithm_type: String,
    /// Learning parameters
    pub parameters: HashMap<String, f32>,
    /// Algorithm state
    pub state: HashMap<String, f32>,
    /// Performance metrics
    pub metrics: HashMap<String, f32>,
}

/// Feature vector for machine learning
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Feature values
    pub features: Vec<f32>,
    /// Feature names
    pub feature_names: Vec<String>,
}

/// Feedback strategy configuration
#[derive(Debug, Clone)]
pub struct FeedbackStrategy {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Feedback tone
    pub tone: FeedbackTone,
    /// Detail level [0.0, 1.0]
    pub detail_level: f32,
    /// Personalization factors
    pub personalization_factors: Vec<String>,
}

/// Personalized recommendation
#[derive(Debug, Clone)]
pub struct PersonalizedRecommendation {
    /// Focus area
    pub focus_area: FocusArea,
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Priority level [0.0, 1.0]
    pub priority: f32,
    /// Estimated impact [0.0, 1.0]
    pub estimated_impact: f32,
    /// Confidence in recommendation [0.0, 1.0]
    pub confidence: f32,
    /// Explanation for user
    pub explanation: String,
    /// Recommended exercises
    pub exercises: Vec<String>,
}

/// Adaptive system statistics
#[derive(Debug, Clone)]
pub struct AdaptiveSystemStats {
    /// Total users
    pub total_users: usize,
    /// Currently active users
    pub active_users: usize,
    /// Total interactions
    pub total_interactions: usize,
    /// Total adaptations
    pub total_adaptations: usize,
    /// Total feedback generated
    pub total_feedback_count: usize,
    /// Average model confidence
    pub average_model_confidence: f32,
}

/// Time range for temporal analysis
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start timestamp
    pub start: DateTime<Utc>,
    /// End timestamp
    pub end: DateTime<Utc>,
}

/// Pattern of user fatigue during learning sessions
#[derive(Debug, Clone)]
pub struct FatiguePattern {
    /// Time until fatigue onset
    pub onset_time: Duration,
    /// Fatigue severity [0.0, 1.0]
    pub severity: f32,
    /// Time required for recovery
    pub recovery_time: Duration,
}

/// Overload phase for progressive training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverloadPhase {
    /// Unique identifier for this phase
    pub phase_id: String,
    /// Duration of this overload phase
    pub duration: Duration,
    /// Intensity increase factor
    pub intensity_increase: f32,
    /// Target metrics for this phase
    pub target_metrics: Vec<String>,
}

/// Success metric for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetric {
    /// Unique identifier for this metric
    pub metric_id: String,
    /// Type of metric being measured
    pub metric_type: MetricType,
    /// Target value to achieve
    pub target_value: f32,
    /// Current measured value
    pub current_value: f32,
    /// How frequently to measure this metric
    pub measurement_frequency: Duration,
}

/// Plateau detection threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauThreshold {
    /// Focus area being monitored
    pub focus_area: FocusArea,
    /// Minimum improvement required to avoid plateau detection
    pub minimum_improvement: f32,
    /// Time window for plateau detection
    pub time_window: Duration,
    /// Confidence threshold for plateau detection
    pub confidence_threshold: f32,
}

/// State representation for reinforcement learning
#[derive(Debug, Clone)]
pub struct RLState {
    /// Current skill level [0.0, 1.0]
    pub current_skill_level: f32,
    /// Recent performance score [0.0, 1.0]
    pub recent_performance: f32,
    /// Learning velocity
    pub learning_velocity: f32,
    /// Consistency score [0.0, 1.0]
    pub consistency: f32,
    /// Normalized session count
    pub session_count: f32,
    /// Time since last plateau (normalized)
    pub time_since_last_plateau: f32,
    /// Motivation level [0.0, 1.0]
    pub motivation_level: f32,
}

impl RLState {
    /// Convert state to string key for Q-table
    #[must_use]
    pub fn to_key(&self) -> String {
        // Discretize continuous values for Q-table
        let skill_bucket = (self.current_skill_level * 10.0) as u32;
        let performance_bucket = (self.recent_performance * 10.0) as u32;
        let velocity_bucket = (self.learning_velocity * 10.0) as u32;
        let consistency_bucket = (self.consistency * 10.0) as u32;

        format!("{skill_bucket}-{performance_bucket}-{velocity_bucket}-{consistency_bucket}")
    }
}

/// Result of difficulty adjustment
#[derive(Debug, Clone)]
pub struct DifficultyAdjustment {
    /// New difficulty level [0.0, 1.0]
    pub new_difficulty_level: f32,
    /// Magnitude of adjustment
    pub adjustment_magnitude: f32,
    /// Direction of adjustment
    pub adjustment_direction: AdjustmentDirection,
    /// Reasoning for adjustment
    pub reasoning: String,
    /// Confidence in adjustment [0.0, 1.0]
    pub confidence: f32,
    /// Expected impact on learning
    pub expected_impact: f32,
}

/// Plateau intervention recommendation
#[derive(Debug, Clone)]
pub struct PlateauIntervention {
    /// Whether plateau was detected
    pub plateau_detected: bool,
    /// Estimated duration of plateau
    pub plateau_duration_estimate: Duration,
    /// Recommended intervention strategies
    pub intervention_strategies: Vec<InterventionStrategy>,
    /// Confidence in intervention [0.0, 1.0]
    pub confidence: f32,
    /// Priority of intervention [0.0, 1.0]
    pub priority: f32,
}

impl PlateauIntervention {
    /// Create intervention indicating no plateau
    #[must_use]
    pub fn none() -> Self {
        Self {
            plateau_detected: false,
            plateau_duration_estimate: Duration::from_secs(0),
            intervention_strategies: vec![],
            confidence: 1.0,
            priority: 0.0,
        }
    }
}

/// Skill transfer optimization result
#[derive(Debug, Clone)]
pub struct SkillTransferOptimization {
    /// Source skill for transfer
    pub source_skill: FocusArea,
    /// Target skill for transfer
    pub target_skill: FocusArea,
    /// Potential for successful transfer [0.0, 1.0]
    pub transfer_potential: f32,
    /// Recommended transfer strategy
    pub optimization_strategy: TransferStrategy,
    /// Recommended exercises for transfer
    pub recommended_exercises: Vec<String>,
    /// Activities to bridge skills
    pub bridging_activities: Vec<String>,
    /// Expected time for transfer
    pub expected_transfer_time: Duration,
    /// Confidence in optimization [0.0, 1.0]
    pub confidence: f32,
}

/// Adaptive system metrics (internal)
#[derive(Debug, Default)]
pub(crate) struct AdaptiveMetrics {
    /// Total users in system
    pub total_users: usize,
    /// Total adaptations performed
    pub total_adaptations: usize,
    /// Total feedback generated
    pub total_feedback_count: usize,
}

impl LearningAlgorithm {
    /// Create a new learning algorithm
    pub fn new(config: &crate::traits::AdaptiveConfig) -> Result<Self, FeedbackError> {
        // Implementation would go here
        Ok(Self {
            algorithm_type: "default".to_string(),
            parameters: HashMap::new(),
            state: HashMap::new(),
            metrics: HashMap::new(),
        })
    }
}

impl UserModel {
    /// Create a new user model with default values
    pub fn new(user_id: String) -> Self {
        let mut skill_breakdown = HashMap::new();
        skill_breakdown.insert(FocusArea::Pronunciation, 0.5);
        skill_breakdown.insert(FocusArea::Intonation, 0.5);
        skill_breakdown.insert(FocusArea::Rhythm, 0.5);
        skill_breakdown.insert(FocusArea::Fluency, 0.5);

        Self {
            user_id,
            skill_level: 0.5,
            learning_rate: 0.1,
            consistency_score: 0.5,
            skill_breakdown,
            interaction_history: VecDeque::new(),
            performance_history: VecDeque::new(),
            adaptive_state: AdaptiveState::default(),
            confidence: 0.1,
            last_updated: Utc::now(),
        }
    }
}
