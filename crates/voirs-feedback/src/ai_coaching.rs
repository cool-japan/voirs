//! AI-powered coaching system for personalized pronunciation guidance
//!
//! This module provides virtual pronunciation coaches, personalized learning companions,
//! automated skill assessment, intelligent intervention strategies, and emotional support
//! for enhanced learning experiences.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::adaptive::models::UserModel;
use crate::traits::{FeedbackContext, FocusArea, SessionScores};

/// Virtual coach personality types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoachPersonality {
    /// Encouraging and supportive coach
    Supportive {
        /// Level of encouragement (0.0-1.0)
        encouragement_level: f32,
        /// Patience level (0.0-1.0)
        patience_level: f32,
    },
    /// Direct and analytical coach
    Analytical {
        /// Level of detail (0.0-1.0)
        detail_level: f32,
        /// Precision focus level (0.0-1.0)
        precision_focus: f32,
    },
    /// Motivational and energetic coach
    Motivational {
        /// Energy level (0.0-1.0)
        energy_level: f32,
        /// Goal orientation level (0.0-1.0)
        goal_orientation: f32,
    },
    /// Gentle and understanding coach
    Nurturing {
        /// Empathy level (0.0-1.0)
        empathy_level: f32,
        /// Reassurance frequency (0.0-1.0)
        reassurance_frequency: f32,
    },
    /// Professional and structured coach
    Professional {
        /// Formality level (0.0-1.0)
        formality_level: f32,
        /// Structure emphasis level (0.0-1.0)
        structure_emphasis: f32,
    },
}

/// Coach communication style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Tone of communication
    pub tone: CoachTone,
    /// Level of verbosity
    pub verbosity: VerbosityLevel,
    /// Frequency of feedback delivery
    pub feedback_frequency: FeedbackFrequency,
    /// Style of encouragement
    pub encouragement_style: EncouragementStyle,
    /// Approach to corrections
    pub correction_approach: CorrectionApproach,
}

/// Coach tone options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoachTone {
    /// Formal tone
    Formal,
    /// Casual tone
    Casual,
    /// Friendly tone
    Friendly,
    /// Professional tone
    Professional,
    /// Encouraging tone
    Encouraging,
    /// Neutral tone
    Neutral,
}

/// Verbosity level for coach feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerbosityLevel {
    /// Concise feedback
    Concise,
    /// Moderate detail
    Moderate,
    /// Detailed feedback
    Detailed,
    /// Comprehensive feedback
    Comprehensive,
}

/// Frequency of feedback delivery
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackFrequency {
    /// Continuous real-time feedback
    Continuous,
    /// Feedback after each word
    AfterEachWord,
    /// Feedback after sentences
    AfterSentences,
    /// Feedback after sessions
    AfterSessions,
    /// Feedback only on request
    OnRequest,
}

/// Encouragement delivery style
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncouragementStyle {
    /// Frequent encouragement
    Frequent,
    /// Encouragement at milestones
    Milestone,
    /// Encouragement for achievements
    Achievement,
    /// Encouragement for improvements
    Improvement,
    /// Minimal encouragement
    Minimal,
}

/// Approach to delivering corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionApproach {
    /// Immediate correction
    Immediate,
    /// Gentle correction
    Gentle,
    /// Constructive correction
    Constructive,
    /// Detailed correction
    Detailed,
    /// Summary correction
    Summary,
}

/// Virtual pronunciation coach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualCoach {
    /// Unique coach identifier
    pub coach_id: Uuid,
    /// Coach display name
    pub name: String,
    /// Coach personality type
    pub personality: CoachPersonality,
    /// Communication style preferences
    pub communication_style: CommunicationStyle,
    /// Areas of specialization
    pub specializations: Vec<CoachSpecialization>,
    /// Experience level
    pub experience_level: ExperienceLevel,
    /// Supported languages
    pub languages: Vec<String>,
    /// Average user rating
    pub user_ratings: f32,
    /// Number of coaching interactions
    pub interaction_count: u32,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Coach areas of specialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoachSpecialization {
    /// Pronunciation accuracy
    PronunciationAccuracy,
    /// Accent reduction
    AccentReduction,
    /// Fluency improvement
    FluencyImprovement,
    /// Conversational skills
    ConversationalSkills,
    /// Business communication
    BusinessCommunication,
    /// Public speaking
    PublicSpeaking,
    /// Language learning
    LanguageLearning,
    /// Children education
    ChildrenEducation,
    /// Special needs support
    SpecialNeeds,
    /// Cultural adaptation
    CulturalAdaptation,
}

/// Coach experience level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// Novice level
    Novice,
    /// Intermediate level
    Intermediate,
    /// Experienced level
    Experienced,
    /// Expert level
    Expert,
    /// Master level
    Master,
}

/// Coaching session context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoachingSession {
    /// Unique session identifier
    pub session_id: Uuid,
    /// User identifier
    pub user_id: Uuid,
    /// Coach identifier
    pub coach_id: Uuid,
    /// Type of session
    pub session_type: SessionType,
    /// Learning objectives for the session
    pub learning_objectives: Vec<LearningObjective>,
    /// Current area of focus
    pub current_focus: FocusArea,
    /// Current session state
    pub session_state: SessionState,
    /// Session start time
    pub start_time: SystemTime,
    /// Session duration
    pub duration: Duration,
    /// History of interventions
    pub intervention_history: Vec<CoachIntervention>,
}

/// Type of coaching session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    /// Assessment session
    AssessmentSession,
    /// Practice session
    PracticeSession,
    /// Targeted improvement session
    TargetedImprovement,
    /// Conversation practice session
    ConversationPractice,
    /// Skill building session
    SkillBuilding,
    /// Progress review session
    ProgressReview,
}

/// Learning objectives for the session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningObjective {
    /// Unique objective identifier
    pub objective_id: Uuid,
    /// Objective description
    pub description: String,
    /// Target metric name
    pub target_metric: String,
    /// Target value to achieve
    pub target_value: f32,
    /// Current progress toward target
    pub progress: f32,
    /// Whether objective is completed
    pub completed: bool,
}

/// Current state of the coaching session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    /// Initializing session
    Initializing,
    /// Warm-up phase
    WarmUp,
    /// Main practice phase
    MainPractice,
    /// Targeted work phase
    TargetedWork,
    /// Review phase
    Review,
    /// Cool-down phase
    CoolDown,
    /// Session completed
    Completed,
}

/// Coach intervention during session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoachIntervention {
    /// Unique intervention identifier
    pub intervention_id: Uuid,
    /// Intervention timestamp
    pub timestamp: SystemTime,
    /// Type of intervention
    pub intervention_type: InterventionType,
    /// Reason that triggered the intervention
    pub trigger_reason: TriggerReason,
    /// Intervention message
    pub message: String,
    /// Optional guidance
    pub guidance: Option<String>,
    /// Optional emotional support
    pub emotional_support: Option<EmotionalSupport>,
    /// Effectiveness rating
    pub effectiveness: Option<f32>,
}

/// Type of coach intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionType {
    /// Encouragement intervention
    Encouragement,
    /// Correction intervention
    Correction,
    /// Guidance intervention
    Guidance,
    /// Technique intervention
    Technique,
    /// Motivation intervention
    Motivation,
    /// Break suggestion
    BreakSuggestion,
    /// Progress update
    ProgressUpdate,
    /// Skill reminder
    SkillReminder,
    /// Emotional support
    EmotionalSupport,
}

/// Reason that triggered an intervention
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerReason {
    /// Low performance detected
    LowPerformance,
    /// Frustration detected
    Frustration,
    /// Performance plateau
    Plateau,
    /// Confusion detected
    Confusion,
    /// Drop in motivation
    MotivationDrop,
    /// Technical difficulty
    TechnicalDifficulty,
    /// Improvement opportunity
    ImprovementOpportunity,
    /// Milestone reached
    MilestoneReached,
    /// Session progression
    SessionProgression,
}

/// Emotional support provided by the coach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSupport {
    /// Type of support
    pub support_type: SupportType,
    /// Intensity level
    pub intensity: SupportIntensity,
    /// Personalized message
    pub personalized_message: String,
    /// Coping strategies
    pub coping_strategies: Vec<CopingStrategy>,
}

/// Type of emotional support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportType {
    /// Encouragement support
    Encouragement,
    /// Reassurance support
    Reassurance,
    /// Motivation support
    Motivation,
    /// Comfort zone support
    ComfortZone,
    /// Confidence building
    ConfidenceBuilding,
    /// Stress reduction
    StressReduction,
}

/// Intensity level of support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportIntensity {
    /// Gentle support
    Gentle,
    /// Moderate support
    Moderate,
    /// Strong support
    Strong,
    /// Intensive support
    Intensive,
}

/// Coping strategies for learning challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopingStrategy {
    /// Strategy name
    pub strategy_name: String,
    /// Strategy description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    /// Effectiveness rating
    pub effectiveness_rating: f32,
}

/// User's coaching preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoachingPreferences {
    /// User identifier
    pub user_id: Uuid,
    /// Preferred coach personality
    pub preferred_coach_personality: CoachPersonality,
    /// Communication style preferences
    pub communication_preferences: CommunicationStyle,
    /// Intervention tolerance level
    pub intervention_tolerance: InterventionTolerance,
    /// Desired emotional support level
    pub emotional_support_level: EmotionalSupportLevel,
    /// Feedback detail preference
    pub feedback_detail_preference: FeedbackDetailLevel,
    /// Session structure preference
    pub session_structure_preference: SessionStructure,
}

/// Tolerance level for interventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionTolerance {
    /// Minimal interventions
    Minimal,
    /// Low intervention frequency
    Low,
    /// Moderate intervention frequency
    Moderate,
    /// High intervention frequency
    High,
    /// Maximum interventions
    Maximum,
}

/// Level of emotional support desired
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalSupportLevel {
    /// No emotional support
    None,
    /// Light emotional support
    Light,
    /// Moderate emotional support
    Moderate,
    /// Comprehensive emotional support
    Comprehensive,
    /// Intensive emotional support
    Intensive,
}

/// Detail level for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackDetailLevel {
    /// Summary feedback only
    Summary,
    /// Standard detail level
    Standard,
    /// Detailed feedback
    Detailed,
    /// Comprehensive feedback
    Comprehensive,
    /// Technical feedback
    Technical,
}

/// Session structure preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStructure {
    /// Flexible structure
    Flexible,
    /// Loose structure
    LooseStructure,
    /// Structured sessions
    Structured,
    /// Highly structured sessions
    HighlyStructured,
    /// Adaptive structure
    Adaptive,
}

/// Automated skill assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillAssessment {
    /// Unique assessment identifier
    pub assessment_id: Uuid,
    /// User identifier
    pub user_id: Uuid,
    /// Assessment date
    pub assessment_date: SystemTime,
    /// Overall assessment score
    pub overall_score: f32,
    /// Skill breakdown by focus area
    pub skill_breakdown: HashMap<FocusArea, SkillMetrics>,
    /// Identified improvement areas
    pub improvement_areas: Vec<ImprovementArea>,
    /// Identified strengths
    pub strengths: Vec<SkillStrength>,
    /// Recommended focus area
    pub recommended_focus: FocusArea,
    /// Confidence interval for assessment
    pub confidence_interval: f32,
}

/// Detailed metrics for specific skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetrics {
    /// Accuracy score
    pub accuracy: f32,
    /// Consistency score
    pub consistency: f32,
    /// Improvement rate
    pub improvement_rate: f32,
    /// Confidence level
    pub confidence_level: f32,
    /// Practice frequency
    pub practice_frequency: f32,
    /// Number of historical performance snapshots that carried real data
    /// for this focus area. `0` means every field above is a prior/fallback
    /// derived from the user's overall model state rather than area-specific
    /// evidence -- kept explicit so "insufficient data" is distinguishable
    /// from a genuinely measured value. `#[serde(default)]` so older
    /// serialized assessments without this field still deserialize.
    #[serde(default)]
    pub data_points: u32,
    /// Last assessment timestamp
    pub last_assessment: SystemTime,
}

/// Areas identified for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementArea {
    /// Focus area for improvement
    pub area: FocusArea,
    /// Current skill level
    pub current_level: f32,
    /// Target skill level
    pub target_level: f32,
    /// Improvement priority
    pub priority: ImprovementPriority,
    /// Estimated time to improve
    pub estimated_time_to_improve: Duration,
    /// Recommended exercises
    pub recommended_exercises: Vec<String>,
}

/// Priority level for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementPriority {
    /// Critical priority
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
    /// Optional improvement
    Optional,
}

/// User strengths identified in assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStrength {
    /// Focus area of strength
    pub area: FocusArea,
    /// Strength level
    pub strength_level: f32,
    /// Consistency of strength
    pub consistency: f32,
    /// Whether user has mentoring potential
    pub potential_for_mentoring: bool,
}

/// AI-powered coaching system
pub struct AICoachingSystem {
    /// Available virtual coaches
    coaches: RwLock<HashMap<Uuid, VirtualCoach>>,
    /// Active coaching sessions
    active_sessions: RwLock<HashMap<Uuid, CoachingSession>>,
    /// User coaching preferences
    user_preferences: RwLock<HashMap<Uuid, CoachingPreferences>>,
    /// Historical skill assessments
    assessment_history: RwLock<HashMap<Uuid, Vec<SkillAssessment>>>,
    /// Intervention strategies by trigger reason
    intervention_strategies: RwLock<HashMap<TriggerReason, Vec<InterventionStrategy>>>,
}

/// Strategy for coach interventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionStrategy {
    /// Unique strategy identifier
    pub strategy_id: Uuid,
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Trigger conditions for this strategy
    pub trigger_conditions: Vec<TriggerCondition>,
    /// Intervention template
    pub intervention_template: InterventionTemplate,
    /// Strategy effectiveness score
    pub effectiveness_score: f32,
    /// Number of times used
    pub usage_count: u32,
}

/// Condition that triggers an intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCondition {
    /// Metric to monitor
    pub metric: String,
    /// Threshold value
    pub threshold: f32,
    /// Comparison operator
    pub comparison: ComparisonOperator,
    /// Optional duration constraint
    pub duration: Option<Duration>,
}

/// Comparison operator for trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Less than threshold
    LessThan,
    /// Greater than threshold
    GreaterThan,
    /// Equals threshold
    Equals,
    /// Between two values (inclusive)
    BetweenInclusive(f32, f32),
    /// Between two values (exclusive)
    BetweenExclusive(f32, f32),
}

/// Template for intervention messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionTemplate {
    /// Message template
    pub message_template: String,
    /// Optional guidance template
    pub guidance_template: Option<String>,
    /// Optional emotional support template
    pub emotional_support_template: Option<EmotionalSupportTemplate>,
    /// Variables for personalization
    pub personalization_variables: Vec<String>,
}

/// Template for emotional support messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSupportTemplate {
    /// Type of support
    pub support_type: SupportType,
    /// Message template
    pub message_template: String,
    /// Coping strategies to suggest
    pub coping_strategies: Vec<CopingStrategy>,
}

impl Default for AICoachingSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AICoachingSystem {
    /// Create a new AI coaching system
    #[must_use]
    pub fn new() -> Self {
        Self {
            coaches: RwLock::new(HashMap::new()),
            active_sessions: RwLock::new(HashMap::new()),
            user_preferences: RwLock::new(HashMap::new()),
            assessment_history: RwLock::new(HashMap::new()),
            intervention_strategies: RwLock::new(Self::initialize_intervention_strategies()),
        }
    }

    /// Initialize default intervention strategies
    fn initialize_intervention_strategies() -> HashMap<TriggerReason, Vec<InterventionStrategy>> {
        let mut strategies = HashMap::new();

        // Low performance intervention
        strategies.insert(TriggerReason::LowPerformance, vec![
            InterventionStrategy {
                strategy_id: Uuid::new_v4(),
                name: "Gentle Encouragement".to_string(),
                description: "Provide supportive feedback when performance drops".to_string(),
                trigger_conditions: vec![
                    TriggerCondition {
                        metric: "accuracy".to_string(),
                        threshold: 0.6,
                        comparison: ComparisonOperator::LessThan,
                        duration: Some(Duration::from_secs(60)),
                    }
                ],
                intervention_template: InterventionTemplate {
                    message_template: "Don't worry, {user_name}! Learning pronunciation takes time. Let's focus on {focus_area} and take it step by step.".to_string(),
                    guidance_template: Some("Try slowing down and focusing on mouth position. Remember, quality over speed!".to_string()),
                    emotional_support_template: Some(EmotionalSupportTemplate {
                        support_type: SupportType::Encouragement,
                        message_template: "Every expert was once a beginner. You're making progress!".to_string(),
                        coping_strategies: vec![
                            CopingStrategy {
                                strategy_name: "Deep Breathing".to_string(),
                                description: "Take three deep breaths to relax".to_string(),
                                implementation_steps: vec![
                                    "Inhale slowly for 4 counts".to_string(),
                                    "Hold for 4 counts".to_string(),
                                    "Exhale slowly for 4 counts".to_string(),
                                ],
                                effectiveness_rating: 0.8,
                            }
                        ],
                    }),
                    personalization_variables: vec!["user_name".to_string(), "focus_area".to_string()],
                },
                effectiveness_score: 0.75,
                usage_count: 0,
            }
        ]);

        // Frustration intervention
        strategies.insert(TriggerReason::Frustration, vec![
            InterventionStrategy {
                strategy_id: Uuid::new_v4(),
                name: "Break and Reassure".to_string(),
                description: "Suggest a break and provide emotional support".to_string(),
                trigger_conditions: vec![
                    TriggerCondition {
                        metric: "error_rate".to_string(),
                        threshold: 0.7,
                        comparison: ComparisonOperator::GreaterThan,
                        duration: Some(Duration::from_secs(120)),
                    }
                ],
                intervention_template: InterventionTemplate {
                    message_template: "{user_name}, I can see you're working hard. How about we take a short break?".to_string(),
                    guidance_template: Some("Sometimes stepping away for a moment helps our brain process what we've learned.".to_string()),
                    emotional_support_template: Some(EmotionalSupportTemplate {
                        support_type: SupportType::ComfortZone,
                        message_template: "Frustration is normal in language learning. You're pushing your boundaries, which is great!".to_string(),
                        coping_strategies: vec![
                            CopingStrategy {
                                strategy_name: "Progressive Muscle Relaxation".to_string(),
                                description: "Relax muscle tension from stress".to_string(),
                                implementation_steps: vec![
                                    "Tense shoulders for 5 seconds, then release".to_string(),
                                    "Tense jaw muscles for 5 seconds, then release".to_string(),
                                    "Take 3 deep breaths".to_string(),
                                ],
                                effectiveness_rating: 0.85,
                            }
                        ],
                    }),
                    personalization_variables: vec!["user_name".to_string()],
                },
                effectiveness_score: 0.82,
                usage_count: 0,
            }
        ]);

        strategies
    }

    /// Register a new virtual coach
    pub async fn register_coach(&self, coach: VirtualCoach) -> Result<(), CoachingError> {
        let mut coaches = self.coaches.write().await;
        coaches.insert(coach.coach_id, coach);
        Ok(())
    }

    /// Get best coach for a user based on preferences and needs
    pub async fn get_recommended_coach(
        &self,
        user_id: Uuid,
    ) -> Result<VirtualCoach, CoachingError> {
        let coaches = self.coaches.read().await;
        let preferences = self.user_preferences.read().await;

        let user_prefs = preferences.get(&user_id);

        // Find best matching coach
        let mut best_match = None;
        let mut best_score = 0.0;

        for coach in coaches.values() {
            let compatibility_score = self.calculate_coach_compatibility(coach, user_prefs);
            if compatibility_score > best_score {
                best_score = compatibility_score;
                best_match = Some(coach.clone());
            }
        }

        best_match.ok_or(CoachingError::NoCoachAvailable)
    }

    /// Calculate compatibility between coach and user preferences
    fn calculate_coach_compatibility(
        &self,
        coach: &VirtualCoach,
        user_prefs: Option<&CoachingPreferences>,
    ) -> f32 {
        let Some(prefs) = user_prefs else {
            return coach.user_ratings / 5.0; // Fallback to rating
        };

        let mut score = 0.0;

        // Personality match (40% weight)
        let personality_score =
            self.match_personality(&coach.personality, &prefs.preferred_coach_personality);
        score += personality_score * 0.4;

        // Communication style match (30% weight)
        let comm_score = self.match_communication_style(
            &coach.communication_style,
            &prefs.communication_preferences,
        );
        score += comm_score * 0.3;

        // User rating (20% weight)
        score += (coach.user_ratings / 5.0) * 0.2;

        // Experience level (10% weight)
        let exp_score = match coach.experience_level {
            ExperienceLevel::Master => 1.0,
            ExperienceLevel::Expert => 0.9,
            ExperienceLevel::Experienced => 0.7,
            ExperienceLevel::Intermediate => 0.5,
            ExperienceLevel::Novice => 0.3,
        };
        score += exp_score * 0.1;

        score
    }

    /// Match coach personality with user preferences
    fn match_personality(
        &self,
        coach_personality: &CoachPersonality,
        user_preference: &CoachPersonality,
    ) -> f32 {
        match (coach_personality, user_preference) {
            (CoachPersonality::Supportive { .. }, CoachPersonality::Supportive { .. }) => 1.0,
            (CoachPersonality::Analytical { .. }, CoachPersonality::Analytical { .. }) => 1.0,
            (CoachPersonality::Motivational { .. }, CoachPersonality::Motivational { .. }) => 1.0,
            (CoachPersonality::Nurturing { .. }, CoachPersonality::Nurturing { .. }) => 1.0,
            (CoachPersonality::Professional { .. }, CoachPersonality::Professional { .. }) => 1.0,
            // Compatible combinations
            (CoachPersonality::Supportive { .. }, CoachPersonality::Nurturing { .. }) => 0.8,
            (CoachPersonality::Nurturing { .. }, CoachPersonality::Supportive { .. }) => 0.8,
            (CoachPersonality::Analytical { .. }, CoachPersonality::Professional { .. }) => 0.7,
            (CoachPersonality::Professional { .. }, CoachPersonality::Analytical { .. }) => 0.7,
            // Moderate compatibility
            _ => 0.5,
        }
    }

    /// Match communication styles
    fn match_communication_style(
        &self,
        coach_style: &CommunicationStyle,
        user_style: &CommunicationStyle,
    ) -> f32 {
        let mut score = 0.0;

        // Tone match (25% weight)
        score += if coach_style.tone == user_style.tone {
            1.0
        } else {
            0.5
        } * 0.25;

        // Verbosity match (25% weight)
        score += if coach_style.verbosity == user_style.verbosity {
            1.0
        } else {
            0.6
        } * 0.25;

        // Feedback frequency match (25% weight)
        score += if coach_style.feedback_frequency == user_style.feedback_frequency {
            1.0
        } else {
            0.7
        } * 0.25;

        // Encouragement style match (25% weight)
        score += if coach_style.encouragement_style == user_style.encouragement_style {
            1.0
        } else {
            0.6
        } * 0.25;

        score
    }

    /// Start a new coaching session
    pub async fn start_coaching_session(
        &self,
        user_id: Uuid,
        coach_id: Uuid,
        session_type: SessionType,
        learning_objectives: Vec<LearningObjective>,
    ) -> Result<Uuid, CoachingError> {
        let session_id = Uuid::new_v4();

        let session = CoachingSession {
            session_id,
            user_id,
            coach_id,
            session_type,
            learning_objectives,
            current_focus: FocusArea::Pronunciation, // Default, will be adjusted
            session_state: SessionState::Initializing,
            start_time: SystemTime::now(),
            duration: Duration::from_secs(0),
            intervention_history: Vec::new(),
        };

        let mut sessions = self.active_sessions.write().await;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Process user performance and provide coaching interventions
    pub async fn process_performance(
        &self,
        session_id: Uuid,
        performance_data: &SessionScores,
        context: &FeedbackContext,
    ) -> Result<Vec<CoachIntervention>, CoachingError> {
        let mut sessions = self.active_sessions.write().await;
        let session = sessions
            .get_mut(&session_id)
            .ok_or(CoachingError::SessionNotFound(session_id))?;

        let interventions = self
            .analyze_and_intervene(session, performance_data, context)
            .await?;

        // Add interventions to session history
        session.intervention_history.extend(interventions.clone());

        Ok(interventions)
    }

    /// Analyze performance and generate appropriate interventions
    async fn analyze_and_intervene(
        &self,
        session: &mut CoachingSession,
        performance: &SessionScores,
        context: &FeedbackContext,
    ) -> Result<Vec<CoachIntervention>, CoachingError> {
        let mut interventions = Vec::new();

        // Check for intervention triggers
        let triggers = self.identify_triggers(performance, session).await;

        for trigger in triggers {
            if let Some(intervention) = self
                .generate_intervention(session, trigger, performance)
                .await?
            {
                interventions.push(intervention);
            }
        }

        Ok(interventions)
    }

    /// Identify triggers for coach intervention
    async fn identify_triggers(
        &self,
        performance: &SessionScores,
        session: &CoachingSession,
    ) -> Vec<TriggerReason> {
        let mut triggers = Vec::new();

        // Low performance check
        if performance.overall_score < 0.6 {
            triggers.push(TriggerReason::LowPerformance);
        }

        // Check for frustration indicators (multiple low scores in a row)
        let recent_low_scores = session
            .intervention_history
            .iter()
            .rev()
            .take(3)
            .filter(|i| matches!(i.trigger_reason, TriggerReason::LowPerformance))
            .count();

        if recent_low_scores >= 2 {
            triggers.push(TriggerReason::Frustration);
        }

        // Progress milestone check
        if performance.overall_score > 0.85 {
            triggers.push(TriggerReason::MilestoneReached);
        }

        triggers
    }

    /// Generate appropriate intervention for trigger
    async fn generate_intervention(
        &self,
        session: &CoachingSession,
        trigger: TriggerReason,
        performance: &SessionScores,
    ) -> Result<Option<CoachIntervention>, CoachingError> {
        let strategies = self.intervention_strategies.read().await;
        let trigger_strategies = strategies.get(&trigger);

        let Some(strategies_list) = trigger_strategies else {
            return Ok(None);
        };

        // Select best strategy (for now, use the first one)
        let strategy = &strategies_list[0];

        // Get coach for personalization
        let coaches = self.coaches.read().await;
        let coach = coaches
            .get(&session.coach_id)
            .ok_or(CoachingError::CoachNotFound(session.coach_id))?;

        // Generate personalized message
        let message = self.personalize_message(
            &strategy.intervention_template.message_template,
            session,
            performance,
        );
        let guidance = strategy
            .intervention_template
            .guidance_template
            .as_ref()
            .map(|template| self.personalize_message(template, session, performance));

        // Generate emotional support if needed
        let emotional_support = strategy
            .intervention_template
            .emotional_support_template
            .as_ref()
            .map(|template| EmotionalSupport {
                support_type: template.support_type.clone(),
                intensity: SupportIntensity::Moderate,
                personalized_message: self.personalize_message(
                    &template.message_template,
                    session,
                    performance,
                ),
                coping_strategies: template.coping_strategies.clone(),
            });

        Ok(Some(CoachIntervention {
            intervention_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            intervention_type: self.map_trigger_to_intervention_type(&trigger),
            trigger_reason: trigger,
            message,
            guidance,
            emotional_support,
            effectiveness: None, // Will be filled by user feedback
        }))
    }

    /// Personalize message templates with context
    fn personalize_message(
        &self,
        template: &str,
        session: &CoachingSession,
        performance: &SessionScores,
    ) -> String {
        template
            .replace("{user_name}", "friend") // Would be actual user name
            .replace("{focus_area}", &format!("{:?}", session.current_focus))
            .replace(
                "{score}",
                &format!("{:.1}%", performance.overall_score * 100.0),
            )
    }

    /// Map trigger reason to intervention type
    fn map_trigger_to_intervention_type(&self, trigger: &TriggerReason) -> InterventionType {
        match trigger {
            TriggerReason::LowPerformance => InterventionType::Encouragement,
            TriggerReason::Frustration => InterventionType::EmotionalSupport,
            TriggerReason::Plateau => InterventionType::Technique,
            TriggerReason::Confusion => InterventionType::Guidance,
            TriggerReason::MotivationDrop => InterventionType::Motivation,
            TriggerReason::TechnicalDifficulty => InterventionType::Technique,
            TriggerReason::ImprovementOpportunity => InterventionType::Guidance,
            TriggerReason::MilestoneReached => InterventionType::Encouragement,
            TriggerReason::SessionProgression => InterventionType::ProgressUpdate,
        }
    }

    /// Conduct automated skill assessment
    ///
    /// Every metric is computed deterministically from the real
    /// `user_model` supplied by the caller -- no random sampling. When a
    /// focus area has no area-specific history yet, this is reflected
    /// honestly via a lower `confidence_level` and `data_points == 0`
    /// (see [`SkillMetrics::data_points`]) rather than a fabricated
    /// reading; running this twice on the same `user_model` always
    /// produces the same result.
    pub async fn conduct_skill_assessment(
        &self,
        user_id: Uuid,
        user_model: &UserModel,
    ) -> Result<SkillAssessment, CoachingError> {
        let assessment_id = Uuid::new_v4();

        // Analyze the user's real historical performance across focus areas.
        let focus_areas = [
            FocusArea::Pronunciation,
            FocusArea::Intonation,
            FocusArea::Rhythm,
            FocusArea::Fluency,
        ];

        let mut skill_breakdown = HashMap::new();
        for focus_area in &focus_areas {
            let metrics = Self::compute_skill_metrics(user_model, focus_area);
            skill_breakdown.insert(focus_area.clone(), metrics);
        }

        // Identify improvement areas. Iterates the fixed `focus_areas` order
        // (rather than the HashMap directly) so the result is deterministic
        // given the same `user_model`, not dependent on hash-map iteration
        // order.
        let mut improvement_areas = Vec::new();
        for area in &focus_areas {
            let metrics = &skill_breakdown[area];
            if metrics.accuracy < 0.8 {
                improvement_areas.push(ImprovementArea {
                    area: area.clone(),
                    current_level: metrics.accuracy,
                    target_level: 0.9,
                    priority: if metrics.accuracy < 0.6 {
                        ImprovementPriority::Critical
                    } else {
                        ImprovementPriority::High
                    },
                    estimated_time_to_improve: Duration::from_secs(3600 * 24 * 7), // 1 week
                    recommended_exercises: vec![
                        format!("Targeted {:?} practice", area),
                        "Slow-motion articulation".to_string(),
                        "Audio comparison exercises".to_string(),
                    ],
                });
            }
        }

        // Identify strengths (same deterministic ordering rationale).
        let mut strengths = Vec::new();
        for area in &focus_areas {
            let metrics = &skill_breakdown[area];
            if metrics.accuracy > 0.85 {
                strengths.push(SkillStrength {
                    area: area.clone(),
                    strength_level: metrics.accuracy,
                    consistency: metrics.consistency,
                    potential_for_mentoring: metrics.accuracy > 0.9 && metrics.consistency > 0.85,
                });
            }
        }

        // Calculate overall score
        let overall_score = skill_breakdown
            .values()
            .map(|metrics| metrics.accuracy)
            .sum::<f32>()
            / skill_breakdown.len() as f32;

        // Recommend focus area (lowest performing area). Iterates the fixed
        // `focus_areas` order with `total_cmp` so ties break deterministically
        // (first in canonical order) instead of depending on HashMap
        // iteration order or risking a NaN panic from `partial_cmp`.
        let recommended_focus = focus_areas
            .iter()
            .min_by(|a, b| {
                skill_breakdown[*a]
                    .accuracy
                    .total_cmp(&skill_breakdown[*b].accuracy)
            })
            .cloned()
            .unwrap_or(FocusArea::Pronunciation);

        // Confidence in the assessment as a whole: the average of the
        // per-area confidence levels, each of which already reflects how
        // much real evidence backs that area (see
        // `Self::compute_skill_metrics`).
        let confidence_interval = (skill_breakdown
            .values()
            .map(|metrics| metrics.confidence_level)
            .sum::<f32>()
            / skill_breakdown.len() as f32)
            .clamp(0.0, 1.0);

        let assessment = SkillAssessment {
            assessment_id,
            user_id,
            assessment_date: SystemTime::now(),
            overall_score,
            skill_breakdown,
            improvement_areas,
            strengths,
            recommended_focus,
            confidence_interval,
        };

        // Store assessment in history
        let mut history = self.assessment_history.write().await;
        history
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(assessment.clone());

        Ok(assessment)
    }

    /// Compute real, deterministic [`SkillMetrics`] for one focus area from
    /// the user's tracked model state. Never draws from a random source --
    /// the same `user_model` always yields the same result.
    fn compute_skill_metrics(user_model: &UserModel, focus_area: &FocusArea) -> SkillMetrics {
        // Per-area history: the trend value each historical performance
        // snapshot recorded for this specific focus area, oldest first
        // (`performance_history` is push_back/pop_front -- see
        // `AdaptiveFeedbackEngine::update_user_model`).
        let series: Vec<f32> = user_model
            .performance_history
            .iter()
            .filter_map(|snapshot| snapshot.improvement_trends.get(focus_area).copied())
            .collect();

        let has_direct_data = user_model.skill_breakdown.contains_key(focus_area);
        let accuracy = user_model
            .skill_breakdown
            .get(focus_area)
            .copied()
            .unwrap_or(user_model.skill_level)
            .clamp(0.0, 1.0);

        // Consistency: blend the user's tracked overall consistency with the
        // real variance of this area's historical trend values (low
        // variance => high consistency, via a bounded 1/(1+stddev) map).
        // With fewer than 2 area-specific samples there is nothing to
        // measure variance over, so fall back to the tracked overall
        // consistency score alone.
        let consistency = if series.len() >= 2 {
            let local_stability = 1.0 / (1.0 + crate::utils::standard_deviation(&series));
            ((user_model.consistency_score + local_stability) / 2.0).clamp(0.0, 1.0)
        } else {
            user_model.consistency_score.clamp(0.0, 1.0)
        };

        // Improvement rate: real linear-regression slope of this area's
        // historical trend values over session order (requires at least 2
        // points; otherwise there is honestly no measurable trend yet).
        let improvement_rate = if series.len() >= 2 {
            let x: Vec<f32> = (0..series.len()).map(|i| i as f32).collect();
            crate::statistical_helpers::linear_regression(&x, &series)
                .map_or(0.0, |(slope, _intercept)| slope)
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Confidence: the model's own tracked confidence (grows with
        // recorded interactions), halved when this specific area has no
        // direct entry in `skill_breakdown` -- `accuracy` above is then an
        // extrapolation from the general skill level, not area-specific
        // evidence.
        let confidence_level = if has_direct_data {
            user_model.confidence
        } else {
            user_model.confidence * 0.5
        }
        .clamp(0.0, 1.0);

        // Practice frequency: fraction of recorded historical sessions that
        // actually carried data for this specific focus area.
        let practice_frequency = if user_model.performance_history.is_empty() {
            0.0
        } else {
            series.len() as f32 / user_model.performance_history.len() as f32
        };

        SkillMetrics {
            accuracy,
            consistency,
            improvement_rate,
            confidence_level,
            practice_frequency,
            data_points: series.len() as u32,
            last_assessment: SystemTime::now(),
        }
    }

    /// Set user coaching preferences
    pub async fn set_coaching_preferences(
        &self,
        preferences: CoachingPreferences,
    ) -> Result<(), CoachingError> {
        let mut prefs = self.user_preferences.write().await;
        prefs.insert(preferences.user_id, preferences);
        Ok(())
    }

    /// Get coaching session history for a user
    pub async fn get_session_history(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<CoachingSession>, CoachingError> {
        let sessions = self.active_sessions.read().await;
        let user_sessions = sessions
            .values()
            .filter(|session| session.user_id == user_id)
            .cloned()
            .collect();

        Ok(user_sessions)
    }
}

/// Errors that can occur in AI coaching operations
#[derive(Debug, thiserror::Error)]
pub enum CoachingError {
    /// Coach with given ID not found
    #[error("Coach not found: {0}")]
    CoachNotFound(Uuid),

    /// Session with given ID not found
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    /// No suitable coach available
    #[error("No coach available")]
    NoCoachAvailable,

    /// Assessment operation failed
    #[error("Assessment failed: {0}")]
    AssessmentFailed(String),

    /// Invalid user preferences
    #[error("Invalid preferences: {0}")]
    InvalidPreferences(String),

    /// System-level error
    #[error("System error: {0}")]
    SystemError(String),
}

/// Result type for coaching operations
pub type CoachingResult<T> = Result<T, CoachingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coach_registration() {
        let system = AICoachingSystem::new();

        let coach = VirtualCoach {
            coach_id: Uuid::new_v4(),
            name: "Coach Sarah".to_string(),
            personality: CoachPersonality::Supportive {
                encouragement_level: 0.8,
                patience_level: 0.9,
            },
            communication_style: CommunicationStyle {
                tone: CoachTone::Friendly,
                verbosity: VerbosityLevel::Moderate,
                feedback_frequency: FeedbackFrequency::AfterEachWord,
                encouragement_style: EncouragementStyle::Frequent,
                correction_approach: CorrectionApproach::Gentle,
            },
            specializations: vec![CoachSpecialization::PronunciationAccuracy],
            experience_level: ExperienceLevel::Expert,
            languages: vec!["en".to_string(), "es".to_string()],
            user_ratings: 4.5,
            interaction_count: 100,
            created_at: SystemTime::now(),
        };

        assert!(system.register_coach(coach).await.is_ok());
    }

    #[tokio::test]
    async fn test_skill_assessment() {
        let system = AICoachingSystem::new();
        let user_id = Uuid::new_v4();

        let user_model = UserModel::default();
        let assessment = system
            .conduct_skill_assessment(user_id, &user_model)
            .await
            .unwrap();

        assert_eq!(assessment.user_id, user_id);
        assert!(!assessment.skill_breakdown.is_empty());
        assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 1.0);
    }

    /// A higher tracked `skill_breakdown` value for a focus area must
    /// produce a proportionally higher reported accuracy for that same
    /// area -- proving the result is a real function of `user_model`, not
    /// a random draw.
    #[tokio::test]
    async fn test_skill_assessment_reflects_skill_breakdown() {
        let system = AICoachingSystem::new();

        let mut high_model = UserModel::default();
        high_model
            .skill_breakdown
            .insert(FocusArea::Pronunciation, 0.95);
        high_model.confidence = 0.9;

        let mut low_model = UserModel::default();
        low_model
            .skill_breakdown
            .insert(FocusArea::Pronunciation, 0.25);
        low_model.confidence = 0.9;

        let high_assessment = system
            .conduct_skill_assessment(Uuid::new_v4(), &high_model)
            .await
            .unwrap();
        let low_assessment = system
            .conduct_skill_assessment(Uuid::new_v4(), &low_model)
            .await
            .unwrap();

        let high_accuracy = high_assessment.skill_breakdown[&FocusArea::Pronunciation].accuracy;
        let low_accuracy = low_assessment.skill_breakdown[&FocusArea::Pronunciation].accuracy;

        assert!((high_accuracy - 0.95).abs() < 1e-6);
        assert!((low_accuracy - 0.25).abs() < 1e-6);
        assert!(high_accuracy > low_accuracy);
        assert!(high_assessment.overall_score > low_assessment.overall_score);
    }

    /// A genuinely rising real performance-history trend must produce a
    /// positive `improvement_rate`, and a genuinely falling one a negative
    /// `improvement_rate` -- the sign is not fixed/random.
    #[tokio::test]
    async fn test_skill_assessment_improvement_rate_follows_trend_direction() {
        let system = AICoachingSystem::new();

        let mut improving_model = UserModel::default();
        for value in [0.1_f32, 0.2, 0.3, 0.4, 0.5] {
            let mut trends = HashMap::new();
            trends.insert(FocusArea::Pronunciation, value);
            improving_model
                .performance_history
                .push_back(crate::traits::PerformanceData {
                    quality_scores: vec![],
                    pronunciation_scores: vec![],
                    improvement_trends: trends,
                    learning_velocity: 0.0,
                    consistency: 0.0,
                });
        }

        let mut declining_model = UserModel::default();
        for value in [0.5_f32, 0.4, 0.3, 0.2, 0.1] {
            let mut trends = HashMap::new();
            trends.insert(FocusArea::Pronunciation, value);
            declining_model
                .performance_history
                .push_back(crate::traits::PerformanceData {
                    quality_scores: vec![],
                    pronunciation_scores: vec![],
                    improvement_trends: trends,
                    learning_velocity: 0.0,
                    consistency: 0.0,
                });
        }

        let improving_assessment = system
            .conduct_skill_assessment(Uuid::new_v4(), &improving_model)
            .await
            .unwrap();
        let declining_assessment = system
            .conduct_skill_assessment(Uuid::new_v4(), &declining_model)
            .await
            .unwrap();

        let improving_rate =
            improving_assessment.skill_breakdown[&FocusArea::Pronunciation].improvement_rate;
        let declining_rate =
            declining_assessment.skill_breakdown[&FocusArea::Pronunciation].improvement_rate;

        assert!(improving_rate > 0.0, "improving_rate = {improving_rate}");
        assert!(declining_rate < 0.0, "declining_rate = {declining_rate}");
        assert_eq!(
            improving_assessment.skill_breakdown[&FocusArea::Pronunciation].data_points,
            5
        );
    }

    /// Running the assessment twice on an identical `UserModel` must
    /// produce identical metrics -- the crux of "not `random()`".
    #[tokio::test]
    async fn test_skill_assessment_is_deterministic() {
        let system = AICoachingSystem::new();
        let mut model = UserModel::default();
        model.skill_breakdown.insert(FocusArea::Pronunciation, 0.6);
        model.confidence = 0.4;
        model.consistency_score = 0.5;

        let user_id = Uuid::new_v4();
        let first = system
            .conduct_skill_assessment(user_id, &model)
            .await
            .unwrap();
        let second = system
            .conduct_skill_assessment(user_id, &model)
            .await
            .unwrap();

        for area in [
            FocusArea::Pronunciation,
            FocusArea::Intonation,
            FocusArea::Rhythm,
            FocusArea::Fluency,
        ] {
            let a = &first.skill_breakdown[&area];
            let b = &second.skill_breakdown[&area];
            assert_eq!(a.accuracy, b.accuracy);
            assert_eq!(a.consistency, b.consistency);
            assert_eq!(a.improvement_rate, b.improvement_rate);
            assert_eq!(a.confidence_level, b.confidence_level);
            assert_eq!(a.practice_frequency, b.practice_frequency);
        }
        assert_eq!(first.overall_score, second.overall_score);
        assert_eq!(first.recommended_focus, second.recommended_focus);
    }

    /// A focus area absent from `skill_breakdown` and with no performance
    /// history must be marked honestly (`data_points == 0`, halved
    /// confidence) rather than silently presented as a fully-measured
    /// value.
    #[tokio::test]
    async fn test_skill_assessment_marks_insufficient_data_honestly() {
        let system = AICoachingSystem::new();
        let mut model = UserModel::default();
        // Pronunciation deliberately absent from skill_breakdown, and no
        // performance_history at all.
        model.skill_breakdown.insert(FocusArea::Intonation, 0.5);
        model.confidence = 0.8;

        let assessment = system
            .conduct_skill_assessment(Uuid::new_v4(), &model)
            .await
            .unwrap();

        let pronunciation = &assessment.skill_breakdown[&FocusArea::Pronunciation];
        assert_eq!(pronunciation.data_points, 0);
        // No direct skill_breakdown entry => confidence is halved relative
        // to the tracked model confidence.
        assert!((pronunciation.confidence_level - 0.4).abs() < 1e-6);

        let intonation = &assessment.skill_breakdown[&FocusArea::Intonation];
        assert!((intonation.confidence_level - 0.8).abs() < 1e-6);
    }
}
