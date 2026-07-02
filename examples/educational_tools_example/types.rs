//! Core data structures shared across the educational tools example.
//!
//! This module holds the plain-data types (lessons, learner profiles,
//! pronunciation analysis results, preferences, ...) that are produced and
//! consumed by the various subsystems implemented in sibling modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

// Core data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub language_code: String,
    pub language_name: String,
    pub phonetic_system: PhoneticSystem,
    pub grammar_rules: GrammarRules,
    pub vocabulary_database: VocabularyDatabase,
    pub cultural_context: CulturalContext,
    pub difficulty_levels: Vec<DifficultyLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMethod {
    pub method_name: String,
    pub description: String,
    pub target_skills: Vec<LanguageSkill>,
    pub effectiveness_rating: f32,
    pub recommended_duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LanguageSkill {
    Listening,
    Speaking,
    Reading,
    Writing,
    Pronunciation,
    Grammar,
    Vocabulary,
    Conversation,
    Culture,
}

#[derive(Debug, Clone)]
pub struct Lesson {
    pub lesson_id: Uuid,
    pub title: String,
    pub language: String,
    pub level: DifficultyLevel,
    pub skills_targeted: Vec<LanguageSkill>,
    pub content: LessonContent,
    pub activities: Vec<LearningActivity>,
    pub estimated_duration: Duration,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LessonContent {
    pub introduction: String,
    pub main_content: Vec<ContentSection>,
    pub examples: Vec<Example>,
    pub exercises: Vec<Exercise>,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct ContentSection {
    pub section_id: String,
    pub title: String,
    pub content_type: ContentType,
    pub text_content: String,
    pub audio_content: Option<AudioContent>,
    pub visual_aids: Vec<VisualAid>,
    pub interaction_points: Vec<InteractionPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Explanation,
    Dialogue,
    Pronunciation,
    Vocabulary,
    Grammar,
    Culture,
    Exercise,
}

#[derive(Debug, Clone)]
pub struct LearningActivity {
    pub activity_id: Uuid,
    pub activity_type: ActivityType,
    pub title: String,
    pub instructions: String,
    pub content: ActivityContent,
    pub scoring_criteria: ScoringCriteria,
    pub time_limit: Option<Duration>,
    pub hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    Pronunciation,
    Conversation,
    Vocabulary,
    Grammar,
    Listening,
    Translation,
    RolePlay,
    Storytelling,
}

#[derive(Debug, Clone)]
pub struct LearnerProfile {
    pub learner_id: String,
    pub name: String,
    pub native_language: String,
    pub target_languages: Vec<String>,
    pub learning_goals: Vec<LearningGoal>,
    pub skill_levels: HashMap<String, SkillLevel>,
    pub learning_preferences: LearningPreferences,
    pub performance_history: PerformanceHistory,
    pub accessibility_needs: AccessibilityNeeds,
}

#[derive(Debug, Clone)]
pub struct LearningGoal {
    pub goal_id: Uuid,
    pub description: String,
    pub target_skill: LanguageSkill,
    pub target_level: SkillLevel,
    pub deadline: Option<SystemTime>,
    pub priority: Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillLevel {
    Beginner,
    Elementary,
    Intermediate,
    UpperIntermediate,
    Advanced,
    Proficient,
}

#[derive(Debug, Clone)]
pub struct LearningPreferences {
    pub preferred_learning_style: LearningStyle,
    pub session_duration: Duration,
    pub difficulty_preference: DifficultyPreference,
    pub audio_preferences: AudioPreferences,
    pub visual_preferences: VisualPreferences,
    pub interaction_preferences: InteractionPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningStyle {
    Visual,
    Auditory,
    Kinesthetic,
    ReadingWriting,
    Multimodal,
}

#[derive(Debug, Clone)]
pub struct PronunciationAssessment {
    pub assessment_id: Uuid,
    pub learner_id: String,
    pub target_text: String,
    pub spoken_audio: AudioData,
    pub phonetic_analysis: PhoneticAnalysis,
    pub pronunciation_score: PronunciationScore,
    pub feedback: PronunciationFeedback,
    pub improvement_suggestions: Vec<ImprovementSuggestion>,
}

#[derive(Debug, Clone)]
pub struct PhoneticAnalysis {
    pub target_phonemes: Vec<Phoneme>,
    pub spoken_phonemes: Vec<Phoneme>,
    pub phoneme_accuracy: HashMap<String, f32>,
    pub rhythm_analysis: RhythmAnalysis,
    pub intonation_analysis: IntonationAnalysis,
    pub stress_pattern_analysis: StressPatternAnalysis,
}

#[derive(Debug, Clone)]
pub struct Phoneme {
    pub symbol: String,
    pub position: Duration,
    pub duration: Duration,
    pub frequency_data: FrequencyData,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct PronunciationScore {
    pub overall_score: f32,
    pub phoneme_accuracy: f32,
    pub rhythm_score: f32,
    pub intonation_score: f32,
    pub stress_score: f32,
    pub fluency_score: f32,
    pub confidence_level: f32,
}

#[derive(Debug, Clone)]
pub struct PronunciationFeedback {
    pub overall_feedback: String,
    pub specific_phoneme_feedback: HashMap<String, String>,
    pub rhythm_feedback: String,
    pub intonation_feedback: String,
    pub encouragement: String,
    pub next_steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImprovementSuggestion {
    pub suggestion_id: Uuid,
    pub target_area: String,
    pub description: String,
    pub practice_exercises: Vec<String>,
    pub priority: Priority,
    pub estimated_practice_time: Duration,
}

#[derive(Debug, Clone)]
pub struct LearningSession {
    pub session_id: Uuid,
    pub learner_id: String,
    pub lesson: Lesson,
    pub start_time: SystemTime,
    pub current_activity: Option<LearningActivity>,
    pub completed_activities: Vec<CompletedActivity>,
    pub session_state: SessionState,
    pub performance_metrics: SessionMetrics,
}

#[derive(Debug, Clone)]
pub struct CompletedActivity {
    pub activity: LearningActivity,
    pub completion_time: Duration,
    pub score: f32,
    pub attempts: u32,
    pub feedback_received: String,
    pub areas_for_improvement: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    Starting,
    InProgress,
    Paused,
    Review,
    Completed,
    Abandoned,
}

#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub engagement_score: f32,
    pub completion_rate: f32,
    pub average_score: f32,
    pub time_on_task: Duration,
    pub help_requests: u32,
    pub mistakes_made: u32,
    pub improvements_shown: f32,
}

// Supporting structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticSystem {
    pub phoneme_inventory: Vec<PhonemeInfo>,
    pub syllable_structure: SyllableStructure,
    pub stress_patterns: Vec<StressPattern>,
    pub intonation_patterns: Vec<IntonationPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarRules {
    pub syntax_rules: Vec<SyntaxRule>,
    pub morphology_rules: Vec<MorphologyRule>,
    pub exception_patterns: Vec<ExceptionPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyDatabase {
    pub words: HashMap<String, WordEntry>,
    pub phrases: HashMap<String, PhraseEntry>,
    pub frequency_lists: HashMap<String, Vec<String>>,
    pub semantic_networks: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalContext {
    pub cultural_notes: HashMap<String, String>,
    pub social_conventions: Vec<SocialConvention>,
    pub regional_variations: HashMap<String, RegionalVariation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyLevel {
    pub level_name: String,
    pub level_number: u8,
    pub description: String,
    pub prerequisites: Vec<String>,
    pub target_skills: Vec<LanguageSkill>,
}

#[derive(Debug, Clone)]
pub struct AudioContent {
    pub audio_id: Uuid,
    pub text: String,
    pub language: String,
    pub speaker_profile: SpeakerProfile,
    pub synthesis_parameters: SynthesisParameters,
    pub audio_path: Option<String>,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct VisualAid {
    pub visual_id: Uuid,
    pub visual_type: VisualType,
    pub content: String,
    pub description: String,
    pub cultural_relevance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualType {
    Image,
    Diagram,
    Chart,
    Animation,
    Video,
    Text,
}

#[derive(Debug, Clone)]
pub struct InteractionPoint {
    pub interaction_id: Uuid,
    pub interaction_type: InteractionType,
    pub trigger_condition: String,
    pub response_options: Vec<ResponseOption>,
    pub feedback_mechanism: FeedbackMechanism,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Question,
    Pronunciation,
    Translation,
    Correction,
    Explanation,
    Practice,
}

#[derive(Debug, Clone)]
pub struct ResponseOption {
    pub option_text: String,
    pub is_correct: bool,
    pub feedback: String,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackMechanism {
    Immediate,
    Delayed,
    OnDemand,
    Progressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyPreference {
    Challenging,
    Balanced,
    Comfortable,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct AudioPreferences {
    pub preferred_accent: String,
    pub speech_rate: f32,
    pub pitch_preference: f32,
    pub background_music: bool,
    pub sound_effects: bool,
}

#[derive(Debug, Clone)]
pub struct VisualPreferences {
    pub color_scheme: String,
    pub font_size: u16,
    pub animation_level: AnimationLevel,
    pub image_preference: ImagePreference,
}

#[derive(Debug, Clone)]
pub struct InteractionPreferences {
    pub preferred_input_method: InputMethod,
    pub feedback_frequency: FeedbackFrequency,
    pub help_system_usage: HelpSystemUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationLevel {
    None,
    Minimal,
    Standard,
    Rich,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImagePreference {
    Realistic,
    Illustrated,
    Minimal,
    Cultural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputMethod {
    Voice,
    Text,
    Touch,
    Gesture,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackFrequency {
    Immediate,
    AfterEachActivity,
    EndOfSession,
    Weekly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HelpSystemUsage {
    Frequent,
    Occasional,
    Minimal,
    OnDemand,
}

// Additional supporting types (simplified for demonstration)
#[derive(Debug, Clone)]
pub struct PhoneticAnalyzer;
#[derive(Debug, Clone)]
pub struct PronunciationScoringSystem;
#[derive(Debug, Clone)]
pub struct FeedbackGenerator;
#[derive(Debug, Clone)]
pub struct NativeSpeakerModel;
#[derive(Debug, Clone)]
pub struct LearnerProfiler;
#[derive(Debug, Clone)]
pub struct DifficultyAdjuster;
#[derive(Debug, Clone)]
pub struct LearningPathOptimizer;
#[derive(Debug, Clone)]
pub struct PerformancePredictor;
#[derive(Debug, Clone)]
pub struct LanguageVoiceSet;
#[derive(Debug, Clone)]
pub struct AccentProcessor;
#[derive(Debug, Clone)]
pub struct CodeSwitcher;
#[derive(Debug, Clone)]
pub struct CulturalAdapter;
#[derive(Debug, Clone)]
pub struct LessonDatabase;
#[derive(Debug, Clone)]
pub struct ActivityEngine;
#[derive(Debug, Clone)]
pub struct InteractionHandler;
#[derive(Debug, Clone)]
pub struct LessonPersonalizer;
#[derive(Debug, Clone)]
pub struct AnalyticsEngine;
#[derive(Debug, Clone)]
pub struct AchievementSystem;
#[derive(Debug, Clone)]
pub struct VisualizationEngine;
#[derive(Debug, Clone)]
pub struct ProgressStore;
#[derive(Debug, Clone)]
pub struct RewardSystem;
#[derive(Debug, Clone)]
pub struct ChallengeGenerator;
#[derive(Debug, Clone)]
pub struct SocialFeatures;
#[derive(Debug, Clone)]
pub struct MotivationSystem;
#[derive(Debug, Clone)]
pub struct CurriculumManager;
#[derive(Debug, Clone)]
pub struct AssessmentEngine;
#[derive(Debug, Clone)]
pub struct ActivityContent;
#[derive(Debug, Clone)]
pub struct ScoringCriteria;
#[derive(Debug, Clone)]
pub struct PerformanceHistory;
#[derive(Debug, Clone)]
pub struct AccessibilityNeeds;
#[derive(Debug, Clone)]
pub struct AudioData;
#[derive(Debug, Clone)]
pub struct RhythmAnalysis;
#[derive(Debug, Clone)]
pub struct IntonationAnalysis;
#[derive(Debug, Clone)]
pub struct StressPatternAnalysis;
#[derive(Debug, Clone)]
pub struct FrequencyData;
#[derive(Debug, Clone)]
pub struct Exercise;
#[derive(Debug, Clone)]
pub struct Example;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeInfo;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableStructure;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPattern;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationPattern;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxRule;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyRule;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceptionPattern;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordEntry;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseEntry;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialConvention;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalVariation;
#[derive(Debug, Clone)]
pub struct SpeakerProfile;
#[derive(Debug, Clone)]
pub struct SynthesisParameters;

// Extension traits for enum names
impl ActivityType {
    pub(crate) fn name(&self) -> &str {
        match self {
            ActivityType::Pronunciation => "Pronunciation",
            ActivityType::Conversation => "Conversation",
            ActivityType::Vocabulary => "Vocabulary",
            ActivityType::Grammar => "Grammar",
            ActivityType::Listening => "Listening",
            ActivityType::Translation => "Translation",
            ActivityType::RolePlay => "Role Play",
            ActivityType::Storytelling => "Storytelling",
        }
    }
}
