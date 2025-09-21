//! Skill taxonomy and progress tracking
//!
//! This module provides comprehensive skill tracking, hierarchical skill taxonomy,
//! and detailed progress analytics for individual skills and skill categories.

use crate::traits::FocusArea;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;
use uuid::Uuid;

/// Hierarchical skill taxonomy with detailed breakdown
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillTaxonomy {
    /// Primary skill category
    pub primary_skill: FocusArea,
    /// Detailed sub-skills within the primary category
    pub sub_skills: Vec<SubSkill>,
    /// Skill dependencies (prerequisites)
    pub dependencies: Vec<FocusArea>,
    /// Correlation weights with other skills
    pub correlation_weights: HashMap<FocusArea, f32>,
}

/// Sub-skill within a primary skill category
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubSkill {
    /// Unique identifier for the sub-skill
    pub id: String,
    /// Display name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Difficulty level (0.0 = beginner, 1.0 = expert)
    pub difficulty: f32,
    /// Prerequisites within the same primary skill
    pub prerequisites: Vec<String>,
    /// Weight contribution to primary skill (0.0 to 1.0)
    pub weight: f32,
}

/// Granular skill progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranularSkillProgress {
    /// Primary skill breakdown (existing)
    pub primary_skills: HashMap<FocusArea, f32>,
    /// Sub-skill detailed progress
    pub sub_skills: HashMap<String, SubSkillProgress>,
    /// Skill mastery certifications
    pub certifications: Vec<SkillCertification>,
    /// Cross-skill correlation analysis
    pub skill_correlations: HashMap<(FocusArea, FocusArea), f32>,
    /// Skill dependency completion status
    pub dependency_completion: HashMap<FocusArea, Vec<String>>,
    /// Fine-grained metrics per sub-skill
    pub sub_skill_metrics: HashMap<String, SubSkillMetrics>,
}

/// Progress tracking for individual sub-skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSkillProgress {
    /// Current proficiency level (0.0 to 1.0)
    pub proficiency: f32,
    /// Number of practice sessions for this sub-skill
    pub practice_sessions: u32,
    /// Total time spent practicing (in seconds)
    pub total_practice_time: u64,
    /// Recent performance trend
    pub performance_trend: Vec<f32>,
    /// Last practice timestamp
    pub last_practiced: DateTime<Utc>,
    /// Mastery status
    pub mastery_status: MasteryStatus,
}

/// Mastery status for skills
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MasteryStatus {
    /// Not yet started
    NotStarted,
    /// In progress (< 60% proficiency)
    InProgress,
    /// Proficient (60-85% proficiency)
    Proficient,
    /// Advanced (85-95% proficiency)
    Advanced,
    /// Mastered (> 95% proficiency)
    Mastered,
}

/// Skill certification for mastery achievement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCertification {
    /// Certification ID
    pub id: String,
    /// Skill area certified
    pub skill_area: FocusArea,
    /// Sub-skills included in certification
    pub sub_skills: Vec<String>,
    /// Certification level
    pub level: CertificationLevel,
    /// Date achieved
    pub achieved_at: DateTime<Utc>,
    /// Score achieved (0.0 to 1.0)
    pub score: f32,
    /// Validity period (for certifications that expire)
    pub valid_until: Option<DateTime<Utc>>,
}

/// Certification levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CertificationLevel {
    /// Bronze certification (60-75% proficiency)
    Bronze,
    /// Silver certification (75-90% proficiency)
    Silver,
    /// Gold certification (90-100% proficiency)
    Gold,
    /// Platinum certification (perfect performance)
    Platinum,
}

/// Fine-grained metrics for sub-skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSkillMetrics {
    /// Accuracy percentage
    pub accuracy: f32,
    /// Consistency score
    pub consistency: f32,
    /// Improvement rate per session
    pub improvement_rate: f32,
    /// Error patterns (most common errors)
    pub error_patterns: Vec<String>,
    /// Performance under different conditions
    pub condition_performance: HashMap<String, f32>,
    /// Retention rate (how well skill is maintained)
    pub retention_rate: f32,
    /// Transfer effectiveness to related skills
    pub transfer_effectiveness: HashMap<String, f32>,
}

/// Hierarchical skill taxonomy for fine-grained progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSkillTaxonomy {
    /// Root skill categories
    pub root_categories: Vec<SkillCategory>,
    /// Skill dependencies graph
    pub dependencies: SkillDependencyGraph,
    /// Cross-skill correlation matrix
    pub correlation_matrix: CrossSkillCorrelationMatrix,
    /// Mastery certification criteria
    pub certification_criteria: MasteryCertificationCriteria,
}

/// Skill category in the hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCategory {
    /// Unique category identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Parent category (if any)
    pub parent_id: Option<String>,
    /// Child categories
    pub children: Vec<SkillCategory>,
    /// Individual skills in this category
    pub skills: Vec<IndividualSkill>,
    /// Weight in overall assessment
    pub weight: f32,
    /// Minimum required level for category mastery
    pub mastery_threshold: f32,
}

/// Individual skill within a category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualSkill {
    /// Unique skill identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Skill type classification
    pub skill_type: SkillType,
    /// Difficulty level (1-10)
    pub difficulty_level: u8,
    /// Prerequisites (other skill IDs)
    pub prerequisites: Vec<String>,
    /// Weight within category
    pub weight: f32,
    /// Measurement metrics
    pub metrics: Vec<SkillMetric>,
    /// Learning objectives
    pub learning_objectives: Vec<String>,
}

/// Types of skills in the taxonomy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkillType {
    /// Fundamental phoneme production
    Phonetic,
    /// Prosodic elements (rhythm, stress, intonation)
    Prosodic,
    /// Overall audio quality
    Quality,
    /// Speaking fluency and flow
    Fluency,
    /// Emotional expression
    Expression,
    /// Technical/professional communication
    Technical,
    /// Cultural and contextual appropriateness
    Cultural,
}

/// Metrics for measuring individual skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetric {
    /// Metric name
    pub name: String,
    /// Measurement type
    pub metric_type: SkillMetricType,
    /// Target value for mastery
    pub target_value: f32,
    /// Current user value
    pub current_value: f32,
    /// Historical values
    pub history: Vec<MetricDataPoint>,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Types of skill metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkillMetricType {
    /// Accuracy percentage (0-100)
    Accuracy,
    /// Consistency score (0-1)
    Consistency,
    /// Error rate (lower is better)
    ErrorRate,
    /// Improvement velocity
    ImprovementRate,
    /// Retention score
    Retention,
    /// Custom metric
    Custom(String),
}

/// Data point for metric history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Measured value
    pub value: f32,
    /// Session context
    pub session_id: Option<Uuid>,
    /// Confidence in measurement
    pub confidence: f32,
}

/// Skill dependency graph for prerequisite modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDependencyGraph {
    /// Adjacency list representation
    pub adjacency_list: HashMap<String, Vec<SkillDependency>>,
    /// Dependency strength matrix
    pub strength_matrix: BTreeMap<String, BTreeMap<String, f32>>,
    /// Learning path recommendations
    pub optimal_paths: HashMap<String, Vec<String>>,
}

/// Individual skill dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDependency {
    /// Target skill ID
    pub target_skill_id: String,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Strength of dependency (0-1)
    pub strength: f32,
    /// Minimum required level in prerequisite
    pub required_level: f32,
}

/// Types of skill dependencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Hard prerequisite (must be learned first)
    Prerequisite,
    /// Soft dependency (helpful but not required)
    Supportive,
    /// Synergistic (skills improve together)
    Synergistic,
    /// Foundational (required for advanced skills)
    Foundational,
}

/// Cross-skill correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSkillCorrelationMatrix {
    /// Correlation coefficients between skills
    pub correlations: BTreeMap<String, BTreeMap<String, CorrelationData>>,
    /// Last analysis timestamp
    pub last_updated: DateTime<Utc>,
    /// Statistical significance data
    pub significance_data: HashMap<String, StatisticalSignificance>,
}

/// Correlation data between two skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationData {
    /// Correlation coefficient (-1 to 1)
    pub coefficient: f32,
    /// Statistical significance
    pub p_value: f32,
    /// Sample size
    pub sample_size: usize,
    /// Confidence interval
    pub confidence_interval: (f32, f32),
}

/// Statistical significance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    /// Is correlation statistically significant
    pub is_significant: bool,
    /// Significance level used
    pub alpha_level: f32,
    /// Effect size
    pub effect_size: f32,
    /// Power of the statistical test
    pub statistical_power: f32,
}

/// Mastery certification criteria and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasteryCertificationCriteria {
    /// Certification levels available
    pub certification_levels: Vec<CertificationLevel>,
    /// Requirements for each skill category
    pub category_requirements: HashMap<String, CategoryMasteryRequirement>,
    /// Overall mastery requirements
    pub overall_requirements: OverallMasteryRequirement,
}

/// Mastery requirements for a skill category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMasteryRequirement {
    /// Category ID
    pub category_id: String,
    /// Minimum average score required
    pub min_average_score: f32,
    /// Minimum individual skill levels
    pub min_individual_levels: HashMap<String, f32>,
    /// Required consistency duration
    pub consistency_duration: Duration,
    /// Maximum allowed variance
    pub max_variance: f32,
}

/// Overall mastery requirements across all skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMasteryRequirement {
    /// Minimum overall skill level
    pub min_overall_level: f32,
    /// Required categories at mastery level
    pub required_mastery_categories: Vec<String>,
    /// Minimum practice time
    pub min_practice_time: Duration,
    /// Assessment criteria
    pub assessment_criteria: Vec<AssessmentCriterion>,
}

/// Individual assessment criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentCriterion {
    /// Criterion identifier
    pub id: String,
    /// Criterion name
    pub name: String,
    /// Required value or threshold
    pub threshold: f32,
    /// Weight in overall assessment
    pub weight: f32,
    /// Measurement method
    pub measurement_method: String,
}

/// Progress tracking for individual skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillProgress {
    /// Skill identifier
    pub skill_id: String,
    /// Current level (0-1)
    pub current_level: f32,
    /// Progress history
    pub progress_history: Vec<SkillProgressSnapshot>,
    /// Mastery status
    pub mastery_status: MasteryStatus,
    /// Last assessment date
    pub last_assessed: DateTime<Utc>,
    /// Practice statistics
    pub practice_stats: SkillPracticeStats,
    /// Improvement trajectory
    pub improvement_trajectory: ImprovementTrajectory,
}

/// Snapshot of skill progress at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillProgressSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Skill level at this time
    pub level: f32,
    /// Session that generated this measurement
    pub session_id: Option<Uuid>,
    /// Confidence in measurement
    pub confidence: f32,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Practice statistics for a skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPracticeStats {
    /// Total practice time for this skill
    pub total_practice_time: Duration,
    /// Number of practice sessions
    pub practice_sessions: usize,
    /// Average session duration
    pub avg_session_duration: Duration,
    /// Success rate in exercises
    pub success_rate: f32,
    /// Most recent practice date
    pub last_practice: Option<DateTime<Utc>>,
    /// Practice frequency (sessions per week)
    pub practice_frequency: f32,
}

/// Improvement trajectory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementTrajectory {
    /// Linear regression slope
    pub slope: f32,
    /// R-squared correlation coefficient
    pub r_squared: f32,
    /// Predicted future level (1 month)
    pub predicted_level_1m: f32,
    /// Predicted future level (3 months)
    pub predicted_level_3m: f32,
    /// Confidence in predictions
    pub prediction_confidence: f32,
    /// Learning velocity (level change per hour)
    pub learning_velocity: f32,
}

/// Configuration for skill tracking
#[derive(Debug, Clone)]
pub struct SkillTrackingConfig {
    /// Enable detailed metric tracking
    pub enable_detailed_metrics: bool,
    /// Minimum confidence threshold for measurements
    pub min_confidence_threshold: f32,
    /// History retention period
    pub history_retention_days: u32,
    /// Update frequency for correlations
    pub correlation_update_interval: Duration,
    /// Statistical significance level
    pub significance_level: f32,
}

/// Compressed aggregated statistics for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSkillStats {
    /// Statistical summary instead of raw data
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value  
    pub max: f32,
    /// Sample count
    pub count: u32,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl CompressedSkillStats {
    /// Create from a collection of values
    pub fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
                last_updated: Utc::now(),
            };
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            mean,
            std_dev,
            min,
            max,
            count: values.len() as u32,
            last_updated: Utc::now(),
        }
    }

    /// Update statistics with new value using incremental formulas
    pub fn update(&mut self, new_value: f32) {
        let old_count = self.count as f32;
        let new_count = old_count + 1.0;

        // Update min/max
        self.min = self.min.min(new_value);
        self.max = self.max.max(new_value);

        // Update mean incrementally
        let old_mean = self.mean;
        self.mean = (old_mean * old_count + new_value) / new_count;

        // Update standard deviation incrementally
        if self.count == 0 {
            self.std_dev = 0.0;
        } else {
            let old_variance = self.std_dev.powi(2);
            let new_variance = (old_variance * old_count
                + (new_value - old_mean) * (new_value - self.mean))
                / new_count;
            self.std_dev = new_variance.sqrt();
        }

        self.count += 1;
        self.last_updated = Utc::now();
    }

    /// Get memory usage in bytes
    pub fn memory_usage() -> usize {
        std::mem::size_of::<Self>()
    }
}
