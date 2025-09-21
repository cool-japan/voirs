//! Type definitions for adaptive feedback system
//!
//! This module contains core enums and type definitions used throughout
//! the adaptive feedback system.

use crate::traits::FocusArea;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Strategy type for feedback delivery
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyType {
    /// Encouraging and supportive
    Encouraging,
    /// Direct and to-the-point
    Direct,
    /// Technical and detailed
    Technical,
    /// Adaptive based on context
    Adaptive,
}

/// Feedback tone
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackTone {
    /// Positive and encouraging
    Positive,
    /// Neutral and informative
    Neutral,
    /// Serious and focused
    Serious,
}

/// Learning style inference
#[derive(Debug, Clone, PartialEq)]
pub enum LearningStyle {
    /// Prefers structured exercises
    Structured,
    /// Prefers experimental approach
    Experimental,
    /// Responds well to feedback
    FeedbackDriven,
    /// Balanced approach
    Balanced,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Practice-based recommendation
    Practice,
    /// Exercise-based recommendation
    Exercise,
    /// Feedback-focused recommendation
    Feedback,
    /// Mixed approach
    Mixed,
    /// Focus area specific recommendation
    FocusArea,
    /// Consistency improvement recommendation
    Consistency,
    /// Engagement enhancement recommendation
    Engagement,
}

/// Type of metric being tracked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Overall skill level measurement
    SkillLevel,
    /// Audio quality score
    QualityScore,
    /// Consistency score across sessions
    ConsistencyScore,
    /// Rate of improvement over time
    ImprovementRate,
}

/// Primary drivers of user motivation
#[derive(Debug, Clone)]
pub enum MotivationDriver {
    /// Achievement-oriented motivation
    Achievement,
    /// Mastery-focused motivation
    Mastery,
    /// Social interaction motivation
    Social,
    /// Competitive motivation
    Competition,
    /// Progress-driven motivation
    Progress,
    /// Recognition-seeking motivation
    Recognition,
}

/// Actions available to the RL agent
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RLAction {
    /// Increase difficulty significantly
    IncreaseSignificantly,
    /// Increase difficulty slightly
    IncreaseSlightly,
    /// Maintain current difficulty
    Maintain,
    /// Decrease difficulty slightly
    DecreaseSlightly,
    /// Decrease difficulty significantly
    DecreaseSignificantly,
}

impl RLAction {
    /// Generate random action for exploration
    #[must_use]
    pub fn random() -> Self {
        match rand::random::<u32>() % 5 {
            0 => RLAction::IncreaseSignificantly,
            1 => RLAction::IncreaseSlightly,
            2 => RLAction::Maintain,
            3 => RLAction::DecreaseSlightly,
            _ => RLAction::DecreaseSignificantly,
        }
    }
}

/// Direction of difficulty adjustment
#[derive(Debug, Clone, PartialEq)]
pub enum AdjustmentDirection {
    /// Increase difficulty
    Increase,
    /// Maintain current level
    Maintain,
    /// Decrease difficulty
    Decrease,
}

/// Intervention strategies for breaking plateaus
#[derive(Debug, Clone)]
pub enum InterventionStrategy {
    /// Focus on improving consistency
    FocusOnConsistency,
    /// Target the weakest skill area
    TargetWeakestSkill(FocusArea),
    /// Change learning approach or method
    ChangeApproach,
    /// Provide motivational boost
    MotivationalBoost,
    /// Introduce new challenge type
    IntroduceNewChallenge,
    /// Take a strategic break
    StrategicBreak,
    /// Seek peer interaction
    PeerInteraction,
}

/// Strategy for skill transfer
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// Direct transfer (high similarity)
    DirectTransfer,
    /// Transfer with bridging activities
    BridgedTransfer,
    /// Learn independently (low similarity)
    IndependentLearning,
    /// Sequential transfer (one after another)
    SequentialTransfer,
}

/// Type of learning curve
#[derive(Debug, Clone, PartialEq)]
pub enum LearningCurveType {
    /// Linear progression
    Linear,
    /// Exponential growth
    Exponential,
    /// Power law progression
    PowerLaw,
    /// S-curve (sigmoid)
    Sigmoid,
}

/// Types of error patterns that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorPatternType {
    /// Inconsistency in performance
    ConsistencyIssues,
    /// Degradation in quality over time
    QualityDegradation,
    /// Pronunciation-specific errors
    PronunciationErrors,
    /// Temporal inconsistency (timing, rhythm)
    TemporalInconsistency,
    /// Context-dependent difficulties
    ContextualDifficulties,
    /// Fatigue-related performance drops
    FatigueRelated,
    /// Skill plateau indicators
    PlateauIndicators,
}

/// Type of assessment to perform
#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentType {
    /// Quick progress check
    Quick,
    /// Comprehensive evaluation
    Comprehensive,
    /// Diagnostic assessment
    Diagnostic,
    /// Peer comparison
    PeerComparison,
}

/// Performance metric types for forecasting
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceMetric {
    /// Overall quality score
    QualityScore,
    /// Pronunciation accuracy
    PronunciationAccuracy,
    /// Consistency measure
    Consistency,
    /// Learning velocity
    LearningVelocity,
    /// Engagement level
    EngagementLevel,
    /// Skill mastery level
    SkillMastery,
}

/// Prediction methodology used for forecasting
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionMethodology {
    /// Linear regression
    LinearRegression,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA time series
    ARIMA,
    /// Neural network
    NeuralNetwork,
    /// Ensemble method
    Ensemble,
}

/// Types of performance risks
#[derive(Debug, Clone, PartialEq)]
pub enum RiskType {
    /// Performance degradation
    PerformanceDegradation,
    /// Learning plateau
    LearningPlateau,
    /// Engagement decline
    EngagementDecline,
    /// Burnout risk
    BurnoutRisk,
    /// Skill transfer failure
    SkillTransferFailure,
    /// Motivation loss
    MotivationLoss,
    /// Confidence decline
    ConfidenceDecline,
}

/// Types of interventions
#[derive(Debug, Clone, PartialEq)]
pub enum InterventionType {
    /// Difficulty adjustment
    DifficultyAdjustment,
    /// Content modification
    ContentModification,
    /// Practice schedule change
    PracticeScheduleChange,
    /// Motivational support
    MotivationalSupport,
    /// Skill remediation
    SkillRemediation,
    /// Break recommendation
    BreakRecommendation,
    /// Feedback intervention
    Feedback,
}

/// Types of long-term outcomes
#[derive(Debug, Clone, PartialEq)]
pub enum OutcomeType {
    /// Skill mastery achievement
    SkillMastery,
    /// Target proficiency level
    ProficiencyLevel,
    /// Learning goal completion
    LearningGoalCompletion,
    /// Performance plateau
    PerformancePlateau,
    /// Skill transfer success
    SkillTransferSuccess,
}

/// Neural network activation functions
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Leaky ReLU
    LeakyReLU,
    /// Softmax (for output layer)
    Softmax,
}

/// Types of sequence patterns
#[derive(Debug, Clone, PartialEq)]
pub enum SequencePatternType {
    /// Engagement level sequences
    EngagementSequence,
    /// Performance progression sequences
    PerformanceSequence,
    /// Difficulty adaptation sequences
    DifficultySequence,
    /// Temporal usage patterns
    TemporalSequence,
    /// Error recovery patterns
    ErrorRecoverySequence,
    /// Motivation cycles
    MotivationCycle,
    /// Learning velocity patterns
    LearningVelocityPattern,
}

/// Types of learning anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Statistical outlier in performance
    StatisticalOutlier,
    /// Unusual behavioral pattern
    BehavioralAnomaly,
    /// Performance drift from norm
    PerformanceDrift,
    /// Engagement spike or drop
    EngagementAnomaly,
    /// Consistency breakdown
    ConsistencyBreakdown,
    /// Rapid improvement or decline
    VelocityAnomaly,
}

/// Types of model updates
#[derive(Debug, Clone, PartialEq)]
pub enum ModelUpdateType {
    /// Parameter weight adjustment
    ParameterAdjustment,
    /// Architecture modification
    ArchitectureModification,
    /// Hyperparameter tuning
    HyperparameterTuning,
    /// Training data augmentation
    DataAugmentation,
    /// Feature engineering update
    FeatureEngineering,
}

/// Priority levels for model updates
#[derive(Debug, Clone, PartialEq)]
pub enum UpdatePriority {
    /// Critical update needed immediately
    Critical,
    /// High priority update
    High,
    /// Medium priority update
    Medium,
    /// Low priority update
    Low,
}
