//! Long-term adaptation system for continuous learning from user feedback
//!
//! This module provides functionality for improving voice cloning quality over time
//! through continuous learning from user feedback, usage patterns, and quality assessments.

use crate::{CloningMethod, Result, SpeakerData, VoiceCloneRequest, VoiceSample};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Types of feedback that can be provided for long-term adaptation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Quality rating (1-5 stars)
    QualityRating(u8),
    /// Similarity rating (1-5 stars)
    SimilarityRating(u8),
    /// Naturalness rating (1-5 stars)
    NaturalnessRating(u8),
    /// Binary thumbs up/down feedback
    BinaryFeedback(bool),
    /// Preference comparison between two samples
    PreferenceComparison {
        preferred_sample_id: String,
        comparison_sample_id: String,
    },
    /// Text feedback with specific comments
    TextualFeedback(String),
    /// Audio correction - user provides corrected audio
    AudioCorrection {
        original_id: String,
        corrected_audio: VoiceSample,
    },
    /// Feature-specific feedback (e.g., "too robotic", "wrong accent")
    FeatureFeedback {
        feature: String,
        rating: f32,
        comment: Option<String>,
    },
}

impl FeedbackType {
    /// Convert feedback to numerical score for processing
    pub fn to_numerical_score(&self) -> f32 {
        match self {
            FeedbackType::QualityRating(rating) => *rating as f32 / 5.0,
            FeedbackType::SimilarityRating(rating) => *rating as f32 / 5.0,
            FeedbackType::NaturalnessRating(rating) => *rating as f32 / 5.0,
            FeedbackType::BinaryFeedback(positive) => {
                if *positive {
                    1.0
                } else {
                    0.0
                }
            }
            FeedbackType::PreferenceComparison { .. } => 0.5, // Neutral baseline
            FeedbackType::TextualFeedback(_) => 0.5,          // Requires text analysis
            FeedbackType::AudioCorrection { .. } => 0.8,      // High value for corrections
            FeedbackType::FeatureFeedback { rating, .. } => *rating,
        }
    }

    /// Get feedback category for processing
    pub fn get_category(&self) -> FeedbackCategory {
        match self {
            FeedbackType::QualityRating(_) => FeedbackCategory::Quality,
            FeedbackType::SimilarityRating(_) => FeedbackCategory::Similarity,
            FeedbackType::NaturalnessRating(_) => FeedbackCategory::Naturalness,
            FeedbackType::BinaryFeedback(_) => FeedbackCategory::Overall,
            FeedbackType::PreferenceComparison { .. } => FeedbackCategory::Preference,
            FeedbackType::TextualFeedback(_) => FeedbackCategory::Textual,
            FeedbackType::AudioCorrection { .. } => FeedbackCategory::Correction,
            FeedbackType::FeatureFeedback { .. } => FeedbackCategory::Feature,
        }
    }
}

/// Categories of feedback for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackCategory {
    Quality,
    Similarity,
    Naturalness,
    Overall,
    Preference,
    Textual,
    Correction,
    Feature,
}

/// User feedback record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    /// Unique feedback ID
    pub feedback_id: String,
    /// User ID (anonymized)
    pub user_id: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// ID of the voice sample being rated
    pub sample_id: String,
    /// Speaker ID
    pub speaker_id: String,
    /// Cloning request that generated the sample
    pub request_metadata: RequestMetadata,
    /// Type and content of feedback
    pub feedback_type: FeedbackType,
    /// Additional context
    pub context: FeedbackContext,
    /// Timestamp when feedback was provided
    pub timestamp: SystemTime,
    /// Confidence score for this feedback (0.0 to 1.0)
    pub confidence: f32,
    /// Whether this feedback has been processed for adaptation
    pub processed: bool,
}

/// Metadata about the original cloning request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Cloning method used
    pub method: CloningMethod,
    /// Text that was synthesized
    pub text: String,
    /// Language
    pub language: Option<String>,
    /// Quality level requested
    pub quality_level: f32,
    /// Processing parameters
    pub parameters: HashMap<String, f32>,
    /// Processing time
    pub processing_time: Duration,
}

/// Context information for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackContext {
    /// Use case (e.g., "audiobook", "announcement", "conversation")
    pub use_case: Option<String>,
    /// Listening environment (e.g., "quiet", "noisy", "mobile")
    pub environment: Option<String>,
    /// User's audio expertise level
    pub user_expertise: Option<ExpertiseLevel>,
    /// Device/platform used
    pub platform: Option<String>,
    /// Additional context tags
    pub tags: Vec<String>,
}

/// User expertise levels for weighting feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Novice,
    Intermediate,
    Expert,
    Professional,
}

impl ExpertiseLevel {
    /// Get weight for feedback based on expertise
    pub fn feedback_weight(&self) -> f32 {
        match self {
            ExpertiseLevel::Novice => 0.8,       // Slightly lower weight
            ExpertiseLevel::Intermediate => 1.0, // Normal weight
            ExpertiseLevel::Expert => 1.2,       // Higher weight
            ExpertiseLevel::Professional => 1.5, // Highest weight
        }
    }
}

/// Adaptation strategy for long-term learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Conservative - small incremental changes
    Conservative,
    /// Moderate - balanced adaptation rate
    Moderate,
    /// Aggressive - rapid adaptation
    Aggressive,
    /// Custom - user-defined parameters
    Custom {
        learning_rate: f32,
        adaptation_threshold: f32,
        feedback_weight: f32,
    },
}

impl AdaptationStrategy {
    /// Get learning rate for this strategy
    pub fn learning_rate(&self) -> f32 {
        match self {
            AdaptationStrategy::Conservative => 0.001,
            AdaptationStrategy::Moderate => 0.01,
            AdaptationStrategy::Aggressive => 0.1,
            AdaptationStrategy::Custom { learning_rate, .. } => *learning_rate,
        }
    }

    /// Get adaptation threshold for this strategy
    pub fn adaptation_threshold(&self) -> f32 {
        match self {
            AdaptationStrategy::Conservative => 0.8,
            AdaptationStrategy::Moderate => 0.6,
            AdaptationStrategy::Aggressive => 0.4,
            AdaptationStrategy::Custom {
                adaptation_threshold,
                ..
            } => *adaptation_threshold,
        }
    }

    /// Get feedback weight multiplier
    pub fn feedback_weight(&self) -> f32 {
        match self {
            AdaptationStrategy::Conservative => 0.5,
            AdaptationStrategy::Moderate => 1.0,
            AdaptationStrategy::Aggressive => 2.0,
            AdaptationStrategy::Custom {
                feedback_weight, ..
            } => *feedback_weight,
        }
    }
}

/// Configuration for long-term adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermAdaptationConfig {
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Maximum number of feedback samples to keep in memory
    pub max_feedback_history: usize,
    /// Minimum feedback samples required before adaptation
    pub min_feedback_for_adaptation: usize,
    /// Time window for feedback aggregation (in hours)
    pub feedback_window_hours: u64,
    /// Enable automatic adaptation without user confirmation
    pub enable_auto_adaptation: bool,
    /// Quality threshold for considering feedback
    pub feedback_quality_threshold: f32,
    /// Adaptation frequency (hours between adaptation cycles)
    pub adaptation_frequency_hours: u64,
    /// Enable speaker-specific adaptation
    pub enable_speaker_specific: bool,
    /// Enable cross-speaker learning
    pub enable_cross_speaker_learning: bool,
    /// Maximum number of adaptation iterations per cycle
    pub max_iterations_per_cycle: usize,
    /// Convergence threshold for stopping adaptation
    pub convergence_threshold: f32,
}

impl Default for LongTermAdaptationConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptationStrategy::Moderate,
            max_feedback_history: 10000,
            min_feedback_for_adaptation: 10,
            feedback_window_hours: 24,
            enable_auto_adaptation: false,
            feedback_quality_threshold: 0.6,
            adaptation_frequency_hours: 6,
            enable_speaker_specific: true,
            enable_cross_speaker_learning: false,
            max_iterations_per_cycle: 100,
            convergence_threshold: 0.001,
        }
    }
}

/// Adaptation result and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Adaptation session ID
    pub session_id: String,
    /// Number of feedback samples processed
    pub feedback_count: usize,
    /// Quality improvement achieved
    pub quality_improvement: f32,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Time taken for adaptation
    pub processing_time: Duration,
    /// Speakers affected by adaptation
    pub affected_speakers: Vec<String>,
    /// Adaptation statistics
    pub statistics: AdaptationStatistics,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Statistics from adaptation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStatistics {
    /// Average feedback score before adaptation
    pub avg_feedback_before: f32,
    /// Average feedback score after adaptation (estimated)
    pub avg_feedback_after: f32,
    /// Feedback distribution by category
    pub feedback_by_category: HashMap<FeedbackCategory, usize>,
    /// Feedback distribution by type
    pub feedback_by_type: HashMap<String, usize>,
    /// Speaker adaptation counts
    pub speaker_adaptations: HashMap<String, usize>,
    /// Quality metrics improvements
    pub quality_improvements: HashMap<String, f32>,
    /// Processing efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Efficiency metrics for adaptation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Memory usage during adaptation (MB)
    pub memory_usage_mb: f32,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Samples processed per second
    pub samples_per_second: f32,
    /// Convergence rate
    pub convergence_rate: f32,
}

/// Long-term adaptation engine
pub struct LongTermAdaptationEngine {
    /// Configuration
    config: LongTermAdaptationConfig,
    /// Feedback storage
    feedback_store: Arc<RwLock<VecDeque<UserFeedback>>>,
    /// Speaker data storage
    speaker_store: Arc<RwLock<HashMap<String, SpeakerData>>>,
    /// Adaptation history
    adaptation_history: Arc<RwLock<VecDeque<AdaptationResult>>>,
    /// Processing statistics
    statistics: Arc<RwLock<ProcessingStatistics>>,
    /// Last adaptation timestamp
    last_adaptation: Arc<RwLock<SystemTime>>,
}

/// Overall processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Total feedback received
    pub total_feedback: usize,
    /// Total adaptations performed
    pub total_adaptations: usize,
    /// Average quality improvement per adaptation
    pub avg_quality_improvement: f32,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Feedback processing rate
    pub feedback_processing_rate: f32,
    /// Success rate of adaptations
    pub adaptation_success_rate: f32,
}

impl LongTermAdaptationEngine {
    /// Create new adaptation engine
    pub fn new(config: LongTermAdaptationConfig) -> Self {
        Self {
            config,
            feedback_store: Arc::new(RwLock::new(VecDeque::new())),
            speaker_store: Arc::new(RwLock::new(HashMap::new())),
            adaptation_history: Arc::new(RwLock::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(ProcessingStatistics::default())),
            last_adaptation: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(LongTermAdaptationConfig::default())
    }

    /// Submit user feedback for processing
    pub fn submit_feedback(&self, feedback: UserFeedback) -> Result<()> {
        let mut store = self.feedback_store.write().unwrap();

        // Maintain size limit
        if store.len() >= self.config.max_feedback_history {
            store.pop_front(); // Remove oldest feedback
        }

        store.push_back(feedback);

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_feedback += 1;
        }

        // Check if adaptation should be triggered
        if self.should_trigger_adaptation() {
            self.trigger_adaptation_cycle()?;
        }

        Ok(())
    }

    /// Check if adaptation should be triggered
    fn should_trigger_adaptation(&self) -> bool {
        let store = self.feedback_store.read().unwrap();
        let last_adaptation = *self.last_adaptation.read().unwrap();

        // Check minimum feedback count
        if store.len() < self.config.min_feedback_for_adaptation {
            return false;
        }

        // Check time since last adaptation
        let time_since_adaptation = SystemTime::now()
            .duration_since(last_adaptation)
            .unwrap_or(Duration::from_secs(0));

        let adaptation_interval =
            Duration::from_secs(self.config.adaptation_frequency_hours * 3600);

        if time_since_adaptation < adaptation_interval {
            return false;
        }

        // Check if auto-adaptation is enabled
        if !self.config.enable_auto_adaptation {
            return false;
        }

        true
    }

    /// Trigger adaptation cycle
    pub fn trigger_adaptation_cycle(&self) -> Result<AdaptationResult> {
        let start_time = SystemTime::now();
        let session_id = uuid::Uuid::new_v4().to_string();

        // Collect feedback for processing
        let feedback_samples = self.collect_feedback_for_adaptation()?;

        if feedback_samples.is_empty() {
            return Err(crate::Error::InvalidInput(
                "No valid feedback for adaptation".to_string(),
            ));
        }

        // Group feedback by speaker
        let speaker_feedback = self.group_feedback_by_speaker(&feedback_samples);

        // Perform adaptation for each speaker
        let mut affected_speakers = Vec::new();
        let mut total_improvement = 0.0;
        let mut iterations = 0;

        for (speaker_id, feedback_list) in speaker_feedback {
            if let Ok(improvement) = self.adapt_speaker(&speaker_id, &feedback_list) {
                affected_speakers.push(speaker_id);
                total_improvement += improvement;
                iterations += 1;
            }
        }

        // Calculate statistics
        let processing_time = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));
        let statistics =
            self.calculate_adaptation_statistics(&feedback_samples, total_improvement)?;

        // Create result
        let result = AdaptationResult {
            session_id: session_id.clone(),
            feedback_count: feedback_samples.len(),
            quality_improvement: total_improvement / iterations.max(1) as f32,
            converged: total_improvement < self.config.convergence_threshold,
            iterations,
            processing_time,
            affected_speakers,
            statistics,
            timestamp: SystemTime::now(),
        };

        // Store result
        {
            let mut history = self.adaptation_history.write().unwrap();
            history.push_back(result.clone());

            // Maintain history size
            if history.len() > 100 {
                history.pop_front();
            }
        }

        // Update last adaptation time
        {
            let mut last_adaptation = self.last_adaptation.write().unwrap();
            *last_adaptation = SystemTime::now();
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_adaptations += 1;
            stats.avg_quality_improvement = (stats.avg_quality_improvement
                * (stats.total_adaptations - 1) as f32
                + result.quality_improvement)
                / stats.total_adaptations as f32;
            stats.total_processing_time += processing_time;
            stats.adaptation_success_rate = (stats.adaptation_success_rate
                * (stats.total_adaptations - 1) as f32
                + if result.converged { 1.0 } else { 0.0 })
                / stats.total_adaptations as f32;
        }

        Ok(result)
    }

    /// Collect feedback samples for adaptation
    fn collect_feedback_for_adaptation(&self) -> Result<Vec<UserFeedback>> {
        let store = self.feedback_store.read().unwrap();
        let cutoff_time =
            SystemTime::now() - Duration::from_secs(self.config.feedback_window_hours * 3600);

        let feedback_samples: Vec<UserFeedback> = store
            .iter()
            .filter(|feedback| {
                // Filter by time window
                feedback.timestamp > cutoff_time
                    // Filter by quality threshold
                    && feedback.confidence >= self.config.feedback_quality_threshold
                    // Filter unprocessed feedback
                    && !feedback.processed
            })
            .cloned()
            .collect();

        Ok(feedback_samples)
    }

    /// Group feedback by speaker
    fn group_feedback_by_speaker(
        &self,
        feedback_samples: &[UserFeedback],
    ) -> HashMap<String, Vec<UserFeedback>> {
        let mut grouped = HashMap::new();

        for feedback in feedback_samples {
            grouped
                .entry(feedback.speaker_id.clone())
                .or_insert_with(Vec::new)
                .push(feedback.clone());
        }

        grouped
    }

    /// Adapt speaker based on feedback
    fn adapt_speaker(&self, speaker_id: &str, feedback_list: &[UserFeedback]) -> Result<f32> {
        let mut speaker_store = self.speaker_store.write().unwrap();

        // Get or create speaker data
        let speaker_data = speaker_store
            .entry(speaker_id.to_string())
            .or_insert_with(|| SpeakerData::default());

        // Calculate feedback-based adjustments
        let adjustments = self.calculate_speaker_adjustments(feedback_list)?;

        // Apply adjustments to speaker data
        let improvement = self.apply_adjustments_to_speaker(speaker_data, &adjustments)?;

        Ok(improvement)
    }

    /// Calculate adjustments based on feedback
    fn calculate_speaker_adjustments(
        &self,
        feedback_list: &[UserFeedback],
    ) -> Result<SpeakerAdjustments> {
        let mut quality_adjustments = HashMap::new();
        let mut feature_adjustments = HashMap::new();
        let mut overall_score = 0.0;
        let mut weight_sum = 0.0;

        for feedback in feedback_list {
            let score = feedback.feedback_type.to_numerical_score();
            let weight = feedback
                .context
                .user_expertise
                .as_ref()
                .map(|e| e.feedback_weight())
                .unwrap_or(1.0)
                * feedback.confidence;

            weight_sum += weight;
            overall_score += score * weight;

            // Process specific feedback types
            match &feedback.feedback_type {
                FeedbackType::FeatureFeedback {
                    feature, rating, ..
                } => {
                    feature_adjustments.insert(feature.clone(), *rating);
                }
                FeedbackType::QualityRating(rating) => {
                    quality_adjustments.insert("overall_quality".to_string(), *rating as f32 / 5.0);
                }
                FeedbackType::SimilarityRating(rating) => {
                    quality_adjustments.insert("similarity".to_string(), *rating as f32 / 5.0);
                }
                FeedbackType::NaturalnessRating(rating) => {
                    quality_adjustments.insert("naturalness".to_string(), *rating as f32 / 5.0);
                }
                _ => {
                    // Generic adjustment
                    quality_adjustments.insert("generic".to_string(), score);
                }
            }
        }

        if weight_sum > 0.0 {
            overall_score /= weight_sum;
        }

        Ok(SpeakerAdjustments {
            overall_score,
            quality_adjustments,
            feature_adjustments,
            learning_rate: self.config.strategy.learning_rate(),
            confidence: weight_sum / feedback_list.len() as f32,
        })
    }

    /// Apply adjustments to speaker data
    fn apply_adjustments_to_speaker(
        &self,
        speaker_data: &mut SpeakerData,
        adjustments: &SpeakerAdjustments,
    ) -> Result<f32> {
        let mut improvement = 0.0;

        // Apply quality adjustments
        let characteristics = &mut speaker_data.profile.characteristics.adaptive_features;
        for (feature, target_value) in &adjustments.quality_adjustments {
            if let Some(current_value) = characteristics.get_mut(feature) {
                let delta = (target_value - *current_value) * adjustments.learning_rate;
                *current_value += delta;
                improvement += delta.abs();
            } else {
                characteristics.insert(feature.clone(), *target_value);
                improvement += target_value.abs();
            }
        }

        // Apply feature-specific adjustments
        for (feature, adjustment) in &adjustments.feature_adjustments {
            let key = format!("adaptive_{}", feature);
            if let Some(current_value) = characteristics.get_mut(&key) {
                let delta = (adjustment - *current_value) * adjustments.learning_rate;
                *current_value += delta;
                improvement += delta.abs();
            } else {
                characteristics.insert(key, *adjustment);
                improvement += adjustment.abs();
            }
        }

        // Update speaker metadata
        speaker_data.profile.metadata.insert(
            "last_adaptation".to_string(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
                .to_string(),
        );
        speaker_data.profile.metadata.insert(
            "adaptation_score".to_string(),
            adjustments.overall_score.to_string(),
        );
        speaker_data.profile.metadata.insert(
            "adaptation_confidence".to_string(),
            adjustments.confidence.to_string(),
        );

        Ok(improvement)
    }

    /// Calculate adaptation statistics
    fn calculate_adaptation_statistics(
        &self,
        feedback_samples: &[UserFeedback],
        quality_improvement: f32,
    ) -> Result<AdaptationStatistics> {
        let mut feedback_by_category = HashMap::new();
        let mut feedback_by_type = HashMap::new();
        let mut speaker_adaptations = HashMap::new();
        let mut quality_sum_before = 0.0;

        for feedback in feedback_samples {
            let category = feedback.feedback_type.get_category();
            *feedback_by_category.entry(category).or_insert(0) += 1;

            let type_name = match feedback.feedback_type {
                FeedbackType::QualityRating(_) => "quality_rating",
                FeedbackType::SimilarityRating(_) => "similarity_rating",
                FeedbackType::NaturalnessRating(_) => "naturalness_rating",
                FeedbackType::BinaryFeedback(_) => "binary_feedback",
                FeedbackType::PreferenceComparison { .. } => "preference_comparison",
                FeedbackType::TextualFeedback(_) => "textual_feedback",
                FeedbackType::AudioCorrection { .. } => "audio_correction",
                FeedbackType::FeatureFeedback { .. } => "feature_feedback",
            };
            *feedback_by_type.entry(type_name.to_string()).or_insert(0) += 1;

            *speaker_adaptations
                .entry(feedback.speaker_id.clone())
                .or_insert(0) += 1;
            quality_sum_before += feedback.feedback_type.to_numerical_score();
        }

        let avg_feedback_before = if !feedback_samples.is_empty() {
            quality_sum_before / feedback_samples.len() as f32
        } else {
            0.0
        };

        let avg_feedback_after = avg_feedback_before + quality_improvement;

        let mut quality_improvements = HashMap::new();
        quality_improvements.insert("overall".to_string(), quality_improvement);
        quality_improvements.insert("similarity".to_string(), quality_improvement * 0.8);
        quality_improvements.insert("naturalness".to_string(), quality_improvement * 0.9);
        quality_improvements.insert("quality".to_string(), quality_improvement * 1.1);

        let efficiency_metrics = EfficiencyMetrics {
            memory_usage_mb: 50.0,                                    // Mock value
            cpu_utilization: 65.0,                                    // Mock value
            samples_per_second: feedback_samples.len() as f32 / 10.0, // Mock calculation
            convergence_rate: 0.85,                                   // Mock value
        };

        Ok(AdaptationStatistics {
            avg_feedback_before,
            avg_feedback_after,
            feedback_by_category,
            feedback_by_type,
            speaker_adaptations,
            quality_improvements,
            efficiency_metrics,
        })
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> ProcessingStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> Vec<AdaptationResult> {
        self.adaptation_history
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get recent feedback
    pub fn get_recent_feedback(&self, limit: usize) -> Vec<UserFeedback> {
        let store = self.feedback_store.read().unwrap();
        store.iter().rev().take(limit).cloned().collect()
    }

    /// Manual adaptation trigger (for testing or admin control)
    pub fn force_adaptation(&self) -> Result<AdaptationResult> {
        self.trigger_adaptation_cycle()
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: LongTermAdaptationConfig) {
        self.config = new_config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &LongTermAdaptationConfig {
        &self.config
    }
}

/// Internal structure for speaker adjustments
#[derive(Debug, Clone)]
struct SpeakerAdjustments {
    overall_score: f32,
    quality_adjustments: HashMap<String, f32>,
    feature_adjustments: HashMap<String, f32>,
    learning_rate: f32,
    confidence: f32,
}

impl UserFeedback {
    /// Create new user feedback
    pub fn new(
        feedback_id: String,
        sample_id: String,
        speaker_id: String,
        feedback_type: FeedbackType,
        request_metadata: RequestMetadata,
    ) -> Self {
        Self {
            feedback_id,
            user_id: None,
            session_id: None,
            sample_id,
            speaker_id,
            request_metadata,
            feedback_type,
            context: FeedbackContext::default(),
            timestamp: SystemTime::now(),
            confidence: 1.0,
            processed: false,
        }
    }

    /// Set user context
    pub fn with_user_context(mut self, user_id: String, session_id: Option<String>) -> Self {
        self.user_id = Some(user_id);
        self.session_id = session_id;
        self
    }

    /// Set feedback context
    pub fn with_context(mut self, context: FeedbackContext) -> Self {
        self.context = context;
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Mark as processed
    pub fn mark_processed(&mut self) {
        self.processed = true;
    }
}

impl Default for FeedbackContext {
    fn default() -> Self {
        Self {
            use_case: None,
            environment: None,
            user_expertise: None,
            platform: None,
            tags: Vec::new(),
        }
    }
}

impl RequestMetadata {
    /// Create from voice clone request
    pub fn from_request(request: &VoiceCloneRequest, processing_time: Duration) -> Self {
        Self {
            method: request.method,
            text: request.text.clone(),
            language: request.language.clone(),
            quality_level: request.quality_level,
            parameters: request.parameters.clone(),
            processing_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_term_adaptation_engine_creation() {
        let engine = LongTermAdaptationEngine::default();
        assert_eq!(engine.config.strategy.learning_rate(), 0.01); // Moderate strategy
        assert!(!engine.config.enable_auto_adaptation);
        assert_eq!(engine.config.min_feedback_for_adaptation, 10);
    }

    #[test]
    fn test_feedback_type_numerical_conversion() {
        assert_eq!(FeedbackType::QualityRating(5).to_numerical_score(), 1.0);
        assert_eq!(FeedbackType::QualityRating(3).to_numerical_score(), 0.6);
        assert_eq!(FeedbackType::BinaryFeedback(true).to_numerical_score(), 1.0);
        assert_eq!(
            FeedbackType::BinaryFeedback(false).to_numerical_score(),
            0.0
        );
    }

    #[test]
    fn test_expertise_level_weights() {
        assert_eq!(ExpertiseLevel::Novice.feedback_weight(), 0.8);
        assert_eq!(ExpertiseLevel::Professional.feedback_weight(), 1.5);
    }

    #[test]
    fn test_adaptation_strategy_parameters() {
        let conservative = AdaptationStrategy::Conservative;
        assert_eq!(conservative.learning_rate(), 0.001);
        assert_eq!(conservative.adaptation_threshold(), 0.8);

        let aggressive = AdaptationStrategy::Aggressive;
        assert_eq!(aggressive.learning_rate(), 0.1);
        assert_eq!(aggressive.adaptation_threshold(), 0.4);
    }

    #[tokio::test]
    async fn test_feedback_submission() {
        let engine = LongTermAdaptationEngine::default();

        let feedback = UserFeedback::new(
            "test_feedback_1".to_string(),
            "sample_1".to_string(),
            "speaker_1".to_string(),
            FeedbackType::QualityRating(4),
            RequestMetadata {
                method: CloningMethod::FewShot,
                text: "Test text".to_string(),
                language: Some("en".to_string()),
                quality_level: 0.8,
                parameters: HashMap::new(),
                processing_time: Duration::from_secs(5),
            },
        );

        let result = engine.submit_feedback(feedback);
        assert!(result.is_ok());

        let stats = engine.get_statistics();
        assert_eq!(stats.total_feedback, 1);
    }

    #[test]
    fn test_feedback_context_creation() {
        let context = FeedbackContext {
            use_case: Some("audiobook".to_string()),
            environment: Some("quiet".to_string()),
            user_expertise: Some(ExpertiseLevel::Expert),
            platform: Some("web".to_string()),
            tags: vec!["high-quality".to_string(), "professional".to_string()],
        };

        assert_eq!(context.use_case.as_ref().unwrap(), "audiobook");
        assert_eq!(
            context.user_expertise.as_ref().unwrap().feedback_weight(),
            1.2
        );
        assert_eq!(context.tags.len(), 2);
    }

    #[test]
    fn test_feedback_categories() {
        assert_eq!(
            FeedbackType::QualityRating(5).get_category(),
            FeedbackCategory::Quality
        );
        assert_eq!(
            FeedbackType::SimilarityRating(4).get_category(),
            FeedbackCategory::Similarity
        );
        assert_eq!(
            FeedbackType::NaturalnessRating(3).get_category(),
            FeedbackCategory::Naturalness
        );
        assert_eq!(
            FeedbackType::BinaryFeedback(true).get_category(),
            FeedbackCategory::Overall
        );
    }

    #[test]
    fn test_custom_adaptation_strategy() {
        let custom = AdaptationStrategy::Custom {
            learning_rate: 0.05,
            adaptation_threshold: 0.7,
            feedback_weight: 1.5,
        };

        assert_eq!(custom.learning_rate(), 0.05);
        assert_eq!(custom.adaptation_threshold(), 0.7);
        assert_eq!(custom.feedback_weight(), 1.5);
    }

    #[test]
    fn test_configuration_defaults() {
        let config = LongTermAdaptationConfig::default();
        assert_eq!(config.max_feedback_history, 10000);
        assert_eq!(config.min_feedback_for_adaptation, 10);
        assert_eq!(config.feedback_window_hours, 24);
        assert!(!config.enable_auto_adaptation);
        assert!(config.enable_speaker_specific);
        assert!(!config.enable_cross_speaker_learning);
    }

    #[test]
    fn test_request_metadata_creation() {
        let request = VoiceCloneRequest::new(
            "test_request".to_string(),
            SpeakerData::default(),
            CloningMethod::FewShot,
            "Test synthesis".to_string(),
        );

        let metadata = RequestMetadata::from_request(&request, Duration::from_secs(10));
        assert_eq!(metadata.method, CloningMethod::FewShot);
        assert_eq!(metadata.text, "Test synthesis");
        assert_eq!(metadata.processing_time, Duration::from_secs(10));
    }
}
