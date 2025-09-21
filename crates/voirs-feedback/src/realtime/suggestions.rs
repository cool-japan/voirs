//! Contextual suggestion engine

use super::types::*;
use crate::{
    FeedbackError, FeedbackResponse, FeedbackType, ProgressIndicators, UserFeedback,
    UserPreferences,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Contextual suggestion engine for personalized feedback
#[derive(Debug, Clone)]
pub struct SuggestionEngine {
    config: SuggestionConfig,
    user_patterns: Arc<RwLock<HashMap<String, UserLearningPattern>>>,
    suggestion_cache: Arc<RwLock<HashMap<String, CachedSuggestion>>>,
    feedback_history: Arc<RwLock<Vec<FeedbackHistoryEntry>>>,
}

/// Configuration for suggestion engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionConfig {
    pub max_suggestions_per_session: usize,
    pub suggestion_cooldown_ms: u64,
    pub personalization_enabled: bool,
    pub adaptive_difficulty: bool,
    pub context_window_size: usize,
    pub min_pattern_observations: usize,
    pub suggestion_categories: Vec<SuggestionCategory>,
}

/// Categories of suggestions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuggestionCategory {
    Pronunciation,
    Pacing,
    Clarity,
    Intonation,
    Breathing,
    Articulation,
    Rhythm,
    Emphasis,
    Volume,
    GeneralImprovement,
}

/// User learning pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLearningPattern {
    pub user_id: String,
    pub skill_level: SkillLevel,
    pub problem_areas: Vec<ProblemArea>,
    pub learning_velocity: f32,
    pub preferred_feedback_style: FeedbackStyle,
    pub success_patterns: Vec<SuccessPattern>,
    pub difficulty_progression: DifficultyProgression,
    pub session_preferences: SessionPreferences,
}

/// Skill level assessment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Identified problem areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemArea {
    pub category: SuggestionCategory,
    pub frequency: f32,
    pub severity: f32,
    pub improvement_rate: f32,
    pub last_occurrence: SystemTime,
}

/// Feedback style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackStyle {
    Detailed,
    Concise,
    Encouraging,
    Technical,
    Visual,
    Auditory,
}

/// Success pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessPattern {
    pub trigger_conditions: Vec<String>,
    pub successful_actions: Vec<String>,
    pub improvement_metrics: Vec<f32>,
    pub context: String,
}

/// Difficulty progression tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyProgression {
    pub current_level: f32,
    pub optimal_challenge_level: f32,
    pub plateau_threshold: f32,
    pub advancement_readiness: f32,
}

/// Session preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPreferences {
    pub preferred_session_length: Duration,
    pub break_frequency: Duration,
    pub focus_areas: Vec<SuggestionCategory>,
    pub avoid_areas: Vec<SuggestionCategory>,
}

/// Cached suggestion to avoid repetition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSuggestion {
    pub suggestion: Suggestion,
    pub created_at: SystemTime,
    pub usage_count: u32,
    pub effectiveness_score: f32,
}

/// Generated suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub id: String,
    pub category: SuggestionCategory,
    pub title: String,
    pub description: String,
    pub action_steps: Vec<ActionStep>,
    pub priority: SuggestionPriority,
    pub estimated_impact: f32,
    pub difficulty_level: f32,
    pub personalization_score: f32,
    pub context: SuggestionContext,
}

/// Action step for implementing suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    pub step_number: u32,
    pub instruction: String,
    pub expected_outcome: String,
    pub verification_method: String,
    pub estimated_time_seconds: u32,
}

/// Suggestion priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuggestionPriority {
    Critical,
    High,
    Medium,
    Low,
    Optional,
}

/// Context for the suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionContext {
    pub current_exercise: Option<String>,
    pub recent_errors: Vec<String>,
    pub session_progress: f32,
    pub user_energy_level: f32,
    pub environmental_factors: Vec<String>,
}

/// Feedback history entry for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackHistoryEntry {
    pub timestamp: SystemTime,
    pub user_id: String,
    pub feedback_type: String,
    pub accuracy_score: f32,
    pub areas_for_improvement: Vec<SuggestionCategory>,
    pub user_response: Option<UserResponseType>,
}

/// User response to feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserResponseType {
    Positive,
    Negative,
    Neutral,
    Ignored,
}

/// Result of suggestion generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionResult {
    pub suggestions: Vec<Suggestion>,
    pub personalization_confidence: f32,
    pub generation_time: Duration,
    pub cache_hit_ratio: f32,
}

impl SuggestionEngine {
    /// Create a new suggestion engine
    pub fn new(config: SuggestionConfig) -> Self {
        Self {
            config,
            user_patterns: Arc::new(RwLock::new(HashMap::new())),
            suggestion_cache: Arc::new(RwLock::new(HashMap::new())),
            feedback_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Generate contextual suggestions based on feedback and user patterns
    pub async fn generate_suggestions(
        &self,
        feedback: &FeedbackResponse,
        user_preferences: &UserPreferences,
        session_context: &SuggestionContext,
    ) -> Result<SuggestionResult, FeedbackError> {
        let start_time = SystemTime::now();
        let mut cache_hits = 0;
        let mut total_requests = 0;

        // Get or create user learning pattern
        let user_pattern = self
            .get_or_create_user_pattern(&user_preferences.user_id)
            .await;

        // Analyze current context
        let context = self
            .analyze_context(feedback, session_context, &user_pattern)
            .await;

        // Generate suggestions based on feedback and context
        let mut suggestions = Vec::new();

        // Check cache first
        for category in &self.config.suggestion_categories {
            total_requests += 1;
            let cache_key = format!(
                "{}_{:?}_{}",
                user_preferences.user_id,
                category,
                self.generate_context_hash(&context)
            );

            if let Some(cached) = self.get_cached_suggestion(&cache_key).await {
                if self.is_cache_valid(&cached) {
                    suggestions.push(cached.suggestion);
                    cache_hits += 1;
                    continue;
                }
            }

            // Generate new suggestion for this category
            if let Some(suggestion) = self
                .generate_category_suggestion(category, feedback, &context, &user_pattern)
                .await?
            {
                // Cache the new suggestion
                self.cache_suggestion(cache_key, &suggestion).await;
                suggestions.push(suggestion);
            }
        }

        // Sort suggestions by priority and personalization score
        suggestions.sort_by(|a, b| match a.priority.cmp(&b.priority) {
            std::cmp::Ordering::Equal => b
                .personalization_score
                .partial_cmp(&a.personalization_score)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        });

        // Limit number of suggestions
        suggestions.truncate(self.config.max_suggestions_per_session);

        // Calculate personalization confidence
        let personalization_confidence = if suggestions.is_empty() {
            0.0
        } else {
            suggestions
                .iter()
                .map(|s| s.personalization_score)
                .sum::<f32>()
                / suggestions.len() as f32
        };

        // Update user pattern based on generated suggestions
        self.update_user_pattern(&user_preferences.user_id, &suggestions, feedback)
            .await?;

        // Record feedback history
        self.record_feedback_history(&user_preferences.user_id, feedback)
            .await;

        let cache_hit_ratio = if total_requests > 0 {
            cache_hits as f32 / total_requests as f32
        } else {
            0.0
        };

        Ok(SuggestionResult {
            suggestions,
            personalization_confidence,
            generation_time: start_time.elapsed().unwrap_or(Duration::from_millis(0)),
            cache_hit_ratio,
        })
    }

    /// Get or create user learning pattern
    async fn get_or_create_user_pattern(&self, user_id: &str) -> UserLearningPattern {
        let patterns = self.user_patterns.read().await;
        if let Some(pattern) = patterns.get(user_id) {
            pattern.clone()
        } else {
            drop(patterns);
            let new_pattern = UserLearningPattern {
                user_id: user_id.to_string(),
                skill_level: SkillLevel::Beginner,
                problem_areas: Vec::new(),
                learning_velocity: 0.5,
                preferred_feedback_style: FeedbackStyle::Encouraging,
                success_patterns: Vec::new(),
                difficulty_progression: DifficultyProgression {
                    current_level: 0.3,
                    optimal_challenge_level: 0.4,
                    plateau_threshold: 0.1,
                    advancement_readiness: 0.2,
                },
                session_preferences: SessionPreferences {
                    preferred_session_length: Duration::from_secs(600), // 10 minutes
                    break_frequency: Duration::from_secs(120),          // 2 minutes
                    focus_areas: vec![
                        SuggestionCategory::Pronunciation,
                        SuggestionCategory::Clarity,
                    ],
                    avoid_areas: Vec::new(),
                },
            };

            let mut patterns = self.user_patterns.write().await;
            patterns.insert(user_id.to_string(), new_pattern.clone());
            new_pattern
        }
    }

    /// Analyze current context for suggestion generation
    async fn analyze_context(
        &self,
        feedback: &FeedbackResponse,
        session_context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> SuggestionContext {
        // Analyze recent errors from feedback
        let recent_errors = if feedback.overall_score < 0.7 {
            feedback
                .feedback_items
                .iter()
                .map(|item| item.message.clone())
                .collect()
        } else {
            Vec::new()
        };

        // Calculate session progress (simplified)
        let session_progress = session_context.session_progress;

        // Estimate user energy level based on session duration and performance
        let user_energy_level = if session_progress < 0.5 {
            1.0 - session_progress * 0.2 // Slight decrease
        } else {
            0.9 - (session_progress - 0.5) * 0.8 // More significant decrease
        };

        SuggestionContext {
            current_exercise: session_context.current_exercise.clone(),
            recent_errors,
            session_progress,
            user_energy_level,
            environmental_factors: vec![], // Could be expanded with noise level, etc.
        }
    }

    /// Generate suggestion for a specific category
    async fn generate_category_suggestion(
        &self,
        category: &SuggestionCategory,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Result<Option<Suggestion>, FeedbackError> {
        // Check if this category is relevant to current feedback
        if !self.is_category_relevant(category, feedback, context) {
            return Ok(None);
        }

        // Generate suggestion based on category
        let suggestion = match category {
            SuggestionCategory::Pronunciation => {
                self.generate_pronunciation_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Pacing => {
                self.generate_pacing_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Clarity => {
                self.generate_clarity_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Intonation => {
                self.generate_intonation_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Breathing => {
                self.generate_breathing_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Articulation => {
                self.generate_articulation_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Rhythm => {
                self.generate_rhythm_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Emphasis => {
                self.generate_emphasis_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::Volume => {
                self.generate_volume_suggestion(feedback, context, user_pattern)
                    .await
            }
            SuggestionCategory::GeneralImprovement => {
                self.generate_general_suggestion(feedback, context, user_pattern)
                    .await
            }
        };

        Ok(Some(suggestion))
    }

    /// Check if category is relevant to current situation
    fn is_category_relevant(
        &self,
        category: &SuggestionCategory,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
    ) -> bool {
        // Basic relevance check based on feedback score and recent errors
        let score = feedback.overall_score;

        match category {
            SuggestionCategory::Pronunciation => {
                score < 0.8
                    || context
                        .recent_errors
                        .iter()
                        .any(|e| e.contains("pronunciation"))
            }
            SuggestionCategory::Clarity => {
                score < 0.7 || context.recent_errors.iter().any(|e| e.contains("clarity"))
            }
            SuggestionCategory::Pacing => context
                .recent_errors
                .iter()
                .any(|e| e.contains("pace") || e.contains("speed")),
            SuggestionCategory::Volume => context
                .recent_errors
                .iter()
                .any(|e| e.contains("volume") || e.contains("loud")),
            _ => score < 0.8, // General categories are relevant when performance is sub-optimal
        }
    }

    /// Generate pronunciation-specific suggestion
    async fn generate_pronunciation_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Pronunciation);

        Suggestion {
            id: format!("pronunciation_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Pronunciation,
            title: "Improve Pronunciation Accuracy".to_string(),
            description: "Focus on articulating sounds more clearly and precisely".to_string(),
            action_steps: vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Slow down your speech to focus on each sound".to_string(),
                    expected_outcome: "Clearer articulation of individual phonemes".to_string(),
                    verification_method: "Record yourself speaking slowly".to_string(),
                    estimated_time_seconds: 30,
                },
                ActionStep {
                    step_number: 2,
                    instruction: "Practice problematic sounds in isolation".to_string(),
                    expected_outcome: "Improved accuracy on difficult phonemes".to_string(),
                    verification_method: "Repeat until consistent pronunciation".to_string(),
                    estimated_time_seconds: 60,
                },
            ],
            priority: if feedback.overall_score < 0.6 {
                SuggestionPriority::High
            } else {
                SuggestionPriority::Medium
            },
            estimated_impact: 0.8,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Pronunciation),
            context: context.clone(),
        }
    }

    /// Generate pacing-specific suggestion
    async fn generate_pacing_suggestion(
        &self,
        _feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Pacing);

        Suggestion {
            id: format!("pacing_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Pacing,
            title: "Adjust Speaking Pace".to_string(),
            description: "Find an optimal speaking rhythm for clarity and engagement".to_string(),
            action_steps: vec![ActionStep {
                step_number: 1,
                instruction: "Count silently while speaking to maintain steady rhythm".to_string(),
                expected_outcome: "More consistent speaking pace".to_string(),
                verification_method: "Monitor speaking rate consistency".to_string(),
                estimated_time_seconds: 45,
            }],
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.6,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Pacing),
            context: context.clone(),
        }
    }

    /// Generate clarity-specific suggestion
    async fn generate_clarity_suggestion(
        &self,
        _feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Clarity);

        Suggestion {
            id: format!("clarity_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Clarity,
            title: "Enhance Speech Clarity".to_string(),
            description: "Make your speech more understandable and crisp".to_string(),
            action_steps: vec![ActionStep {
                step_number: 1,
                instruction: "Open your mouth wider for vowel sounds".to_string(),
                expected_outcome: "Clearer vowel pronunciation".to_string(),
                verification_method: "Practice with mirror feedback".to_string(),
                estimated_time_seconds: 30,
            }],
            priority: SuggestionPriority::High,
            estimated_impact: 0.9,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Clarity),
            context: context.clone(),
        }
    }

    /// Generate intonation-specific suggestion
    async fn generate_intonation_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Intonation);
        let intonation_issues = self.analyze_intonation_issues(feedback);

        let action_steps = if intonation_issues.contains(&"monotone".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice with exaggerated pitch variations on individual words"
                        .to_string(),
                    expected_outcome: "More natural pitch contours".to_string(),
                    verification_method: "Record and compare pitch patterns".to_string(),
                    estimated_time_seconds: 120,
                },
                ActionStep {
                    step_number: 2,
                    instruction:
                        "Use questions and statements to practice rising and falling intonation"
                            .to_string(),
                    expected_outcome: "Clear distinction between question and statement patterns"
                        .to_string(),
                    verification_method: "Practice with feedback on pitch direction".to_string(),
                    estimated_time_seconds: 90,
                },
            ]
        } else if intonation_issues.contains(&"excessive".to_string()) {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice with smaller pitch variations while maintaining expressiveness"
                        .to_string(),
                expected_outcome: "More controlled and natural pitch patterns".to_string(),
                verification_method: "Monitor pitch range in real-time feedback".to_string(),
                estimated_time_seconds: 100,
            }]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice sentence-level intonation patterns with focus on natural melody"
                        .to_string(),
                expected_outcome: "Improved overall intonation naturalness".to_string(),
                verification_method: "Comparative analysis with native speaker patterns"
                    .to_string(),
                estimated_time_seconds: 90,
            }]
        };

        Suggestion {
            id: format!("intonation_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Intonation,
            title: "Enhance Intonation Patterns".to_string(),
            description: "Develop natural pitch variations and melody in speech".to_string(),
            action_steps,
            priority: SuggestionPriority::High,
            estimated_impact: 0.8,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Intonation),
            context: context.clone(),
        }
    }

    /// Generate breathing-specific suggestion
    async fn generate_breathing_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Breathing);
        let breathing_issues = self.analyze_breathing_issues(feedback);

        let action_steps = if breathing_issues.contains(&"breath_control".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice diaphragmatic breathing: Place hand on chest, another on abdomen. Breathe so only bottom hand moves".to_string(),
                    expected_outcome: "Improved breath support and control".to_string(),
                    verification_method: "Feel steady airflow during long vowel sounds".to_string(),
                    estimated_time_seconds: 180,
                },
                ActionStep {
                    step_number: 2,
                    instruction: "Practice breath phrases: Read sentences in one breath, gradually increasing length".to_string(),
                    expected_outcome: "Better breath phrase management".to_string(),
                    verification_method: "Complete sentences without gasping or strain".to_string(),
                    estimated_time_seconds: 120,
                },
            ]
        } else if breathing_issues.contains(&"shallow".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice deep breathing exercises: Inhale for 4 counts, hold for 4, exhale for 6".to_string(),
                    expected_outcome: "Deeper, more efficient breathing patterns".to_string(),
                    verification_method: "Longer phrases without breathlessness".to_string(),
                    estimated_time_seconds: 150,
                },
            ]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Establish natural breathing rhythm: Take comfortable breaths between phrases"
                        .to_string(),
                expected_outcome: "Smooth, natural breathing during speech".to_string(),
                verification_method: "Consistent voice quality throughout phrases".to_string(),
                estimated_time_seconds: 100,
            }]
        };

        Suggestion {
            id: format!("breathing_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Breathing,
            title: "Optimize Breathing Technique".to_string(),
            description: "Develop efficient breathing patterns for speech support".to_string(),
            action_steps,
            priority: SuggestionPriority::High,
            estimated_impact: 0.9,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Breathing),
            context: context.clone(),
        }
    }

    /// Generate articulation-specific suggestion
    async fn generate_articulation_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Articulation);
        let articulation_issues = self.analyze_articulation_issues(feedback);

        let action_steps = if articulation_issues.contains(&"consonant_clusters".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice consonant clusters slowly: 'str-', 'spr-', 'scr-' with exaggerated mouth movements".to_string(),
                    expected_outcome: "Clearer consonant cluster pronunciation".to_string(),
                    verification_method: "Listen for distinct consonant sounds in clusters".to_string(),
                    estimated_time_seconds: 120,
                },
                ActionStep {
                    step_number: 2,
                    instruction: "Build up speed gradually while maintaining clarity in cluster pronunciation".to_string(),
                    expected_outcome: "Natural speed with clear articulation".to_string(),
                    verification_method: "Record and compare clarity at different speeds".to_string(),
                    estimated_time_seconds: 90,
                },
            ]
        } else if articulation_issues.contains(&"final_consonants".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice word endings: Hold final consonants longer than normal 'bat-t', 'dog-g', 'has-s'".to_string(),
                    expected_outcome: "Clearer word endings and boundaries".to_string(),
                    verification_method: "Compare word pairs that differ only in final consonant".to_string(),
                    estimated_time_seconds: 100,
                },
            ]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction: "Practice tongue twisters and consonant drills to improve precision"
                    .to_string(),
                expected_outcome: "Enhanced overall articulation clarity".to_string(),
                verification_method: "Record and analyze speech clarity improvements".to_string(),
                estimated_time_seconds: 120,
            }]
        };

        Suggestion {
            id: format!("articulation_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Articulation,
            title: "Enhance Articulation Precision".to_string(),
            description: "Develop clearer and more precise consonant and vowel production"
                .to_string(),
            action_steps,
            priority: SuggestionPriority::High,
            estimated_impact: 0.85,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Articulation),
            context: context.clone(),
        }
    }

    /// Generate rhythm-specific suggestion
    async fn generate_rhythm_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Rhythm);
        let rhythm_issues = self.analyze_rhythm_issues(feedback);

        let action_steps = if rhythm_issues.contains(&"too_fast".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction:
                        "Practice with a metronome or slow backing track to establish steady rhythm"
                            .to_string(),
                    expected_outcome: "More controlled and natural speaking pace".to_string(),
                    verification_method: "Compare recorded speech with appropriate tempo standards"
                        .to_string(),
                    estimated_time_seconds: 150,
                },
                ActionStep {
                    step_number: 2,
                    instruction:
                        "Add deliberate pauses between phrases and sentences for better pacing"
                            .to_string(),
                    expected_outcome: "Natural rhythm with appropriate pauses".to_string(),
                    verification_method: "Listen for balanced phrase timing".to_string(),
                    estimated_time_seconds: 120,
                },
            ]
        } else if rhythm_issues.contains(&"too_slow".to_string()) {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice gradually increasing speaking speed while maintaining clarity"
                        .to_string(),
                expected_outcome: "More natural and engaging pace".to_string(),
                verification_method:
                    "Target conversational speaking rate (150-200 words per minute)".to_string(),
                estimated_time_seconds: 140,
            }]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction: "Practice with varied sentence lengths and natural stress patterns"
                    .to_string(),
                expected_outcome: "More natural and engaging speech rhythm".to_string(),
                verification_method: "Record and compare rhythm patterns with natural speech"
                    .to_string(),
                estimated_time_seconds: 110,
            }]
        };

        Suggestion {
            id: format!("rhythm_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Rhythm,
            title: "Optimize Speech Rhythm".to_string(),
            description: "Develop natural timing and flow patterns in speech".to_string(),
            action_steps,
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.75,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Rhythm),
            context: context.clone(),
        }
    }

    /// Generate emphasis-specific suggestion
    async fn generate_emphasis_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Emphasis);
        let emphasis_issues = self.analyze_emphasis_issues(feedback);

        let action_steps = if emphasis_issues.contains(&"weak_stress".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction: "Practice exaggerated stress on key words: 'THIS is important', 'I REALLY mean it'".to_string(),
                    expected_outcome: "Clearer word emphasis and meaning communication".to_string(),
                    verification_method: "Listen for distinct stress patterns on important words".to_string(),
                    estimated_time_seconds: 120,
                },
                ActionStep {
                    step_number: 2,
                    instruction: "Use pitch, volume, and duration changes together for natural emphasis".to_string(),
                    expected_outcome: "More natural and effective emphasis patterns".to_string(),
                    verification_method: "Practice with different emphasis techniques".to_string(),
                    estimated_time_seconds: 100,
                },
            ]
        } else if emphasis_issues.contains(&"over_emphasis".to_string()) {
            vec![ActionStep {
                step_number: 1,
                instruction: "Practice subtle emphasis: Use smaller changes in pitch and volume"
                    .to_string(),
                expected_outcome: "More natural and appropriate emphasis levels".to_string(),
                verification_method: "Compare with conversational emphasis patterns".to_string(),
                estimated_time_seconds: 110,
            }]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice selective emphasis: Choose 1-2 key words per sentence for stress"
                        .to_string(),
                expected_outcome: "Clear communication of intended meaning".to_string(),
                verification_method: "Verify that emphasis supports sentence meaning".to_string(),
                estimated_time_seconds: 90,
            }]
        };

        Suggestion {
            id: format!("emphasis_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Emphasis,
            title: "Enhance Speech Emphasis".to_string(),
            description:
                "Develop effective stress and emphasis techniques for clearer communication"
                    .to_string(),
            action_steps,
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.7,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Emphasis),
            context: context.clone(),
        }
    }

    /// Generate volume-specific suggestion
    async fn generate_volume_suggestion(
        &self,
        feedback: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        let difficulty =
            self.calculate_suggestion_difficulty(user_pattern, &SuggestionCategory::Volume);
        let volume_issues = self.analyze_volume_issues(feedback);

        let action_steps = if volume_issues.contains(&"too_quiet".to_string()) {
            vec![
                ActionStep {
                    step_number: 1,
                    instruction:
                        "Practice diaphragmatic breathing to support louder volume without strain"
                            .to_string(),
                    expected_outcome: "Increased volume with proper breath support".to_string(),
                    verification_method: "Maintain clear voice quality while increasing volume"
                        .to_string(),
                    estimated_time_seconds: 150,
                },
                ActionStep {
                    step_number: 2,
                    instruction: "Practice projection: Imagine speaking to someone across the room"
                        .to_string(),
                    expected_outcome: "Natural volume increase without shouting".to_string(),
                    verification_method: "Voice should carry without strain or distortion"
                        .to_string(),
                    estimated_time_seconds: 120,
                },
            ]
        } else if volume_issues.contains(&"too_loud".to_string()) {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice speaking at conversational volume: Imagine talking to someone nearby"
                        .to_string(),
                expected_outcome: "More appropriate and comfortable volume levels".to_string(),
                verification_method: "Volume should feel natural and not strain listeners"
                    .to_string(),
                estimated_time_seconds: 100,
            }]
        } else {
            vec![ActionStep {
                step_number: 1,
                instruction:
                    "Practice volume control: Vary volume for different situations and emphasis"
                        .to_string(),
                expected_outcome: "Flexible volume control for different contexts".to_string(),
                verification_method: "Appropriate volume adjustments for content and audience"
                    .to_string(),
                estimated_time_seconds: 110,
            }]
        };

        Suggestion {
            id: format!("volume_{}", uuid::Uuid::new_v4()),
            category: SuggestionCategory::Volume,
            title: "Optimize Volume Control".to_string(),
            description: "Develop appropriate volume levels for different speaking contexts"
                .to_string(),
            action_steps,
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.65,
            difficulty_level: difficulty,
            personalization_score: self
                .calculate_personalization_score(user_pattern, &SuggestionCategory::Volume),
            context: context.clone(),
        }
    }

    async fn generate_general_suggestion(
        &self,
        _: &FeedbackResponse,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        self.create_generic_suggestion(
            SuggestionCategory::GeneralImprovement,
            "Overall Improvement",
            "General practice recommendations",
            context,
            user_pattern,
        )
    }

    /// Create a generic suggestion for categories without specific implementation
    fn create_generic_suggestion(
        &self,
        category: SuggestionCategory,
        title: &str,
        description: &str,
        context: &SuggestionContext,
        user_pattern: &UserLearningPattern,
    ) -> Suggestion {
        Suggestion {
            id: format!("{:?}_{}", category, uuid::Uuid::new_v4()),
            category: category.clone(),
            title: title.to_string(),
            description: description.to_string(),
            action_steps: vec![ActionStep {
                step_number: 1,
                instruction: "Practice with focus and attention".to_string(),
                expected_outcome: "Gradual improvement in this area".to_string(),
                verification_method: "Self-assessment and feedback".to_string(),
                estimated_time_seconds: 60,
            }],
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.5,
            difficulty_level: self.calculate_suggestion_difficulty(user_pattern, &category),
            personalization_score: self.calculate_personalization_score(user_pattern, &category),
            context: context.clone(),
        }
    }

    /// Calculate suggestion difficulty based on user pattern
    fn calculate_suggestion_difficulty(
        &self,
        user_pattern: &UserLearningPattern,
        category: &SuggestionCategory,
    ) -> f32 {
        let base_difficulty = match user_pattern.skill_level {
            SkillLevel::Beginner => 0.3f32,
            SkillLevel::Intermediate => 0.5f32,
            SkillLevel::Advanced => 0.7f32,
            SkillLevel::Expert => 0.9f32,
        };

        // Adjust based on problem areas
        let problem_adjustment = if user_pattern
            .problem_areas
            .iter()
            .any(|p| &p.category == category)
        {
            -0.1f32 // Make it easier if it's a known problem area
        } else {
            0.0f32
        };

        (base_difficulty + problem_adjustment)
            .max(0.1f32)
            .min(0.9f32)
    }

    /// Calculate personalization score
    fn calculate_personalization_score(
        &self,
        user_pattern: &UserLearningPattern,
        category: &SuggestionCategory,
    ) -> f32 {
        let mut score = 0.5; // Base score

        // Higher score for focus areas
        if user_pattern
            .session_preferences
            .focus_areas
            .contains(category)
        {
            score += 0.3;
        }

        // Lower score for avoid areas
        if user_pattern
            .session_preferences
            .avoid_areas
            .contains(category)
        {
            score -= 0.4;
        }

        // Adjust based on problem frequency
        if let Some(problem) = user_pattern
            .problem_areas
            .iter()
            .find(|p| &p.category == category)
        {
            score += problem.frequency * 0.2;
        }

        score.max(0.0).min(1.0)
    }

    /// Analyze intonation issues from feedback
    fn analyze_intonation_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for monotone patterns
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("monotone")
                || item.message.to_lowercase().contains("flat")
                || item.message.to_lowercase().contains("no variation")
        }) {
            issues.push("monotone".to_string());
        }

        // Check for excessive pitch variation
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("excessive")
                || item.message.to_lowercase().contains("too much variation")
                || item.message.to_lowercase().contains("over-expressive")
        }) {
            issues.push("excessive".to_string());
        }

        // Check for inappropriate pitch patterns
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("pitch")
                || item.message.to_lowercase().contains("intonation")
                || item.message.to_lowercase().contains("melody")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Analyze breathing issues from feedback
    fn analyze_breathing_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for breath control issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("breath")
                || item.message.to_lowercase().contains("running out")
                || item.message.to_lowercase().contains("gasping")
        }) {
            issues.push("breath_control".to_string());
        }

        // Check for shallow breathing
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("shallow")
                || item.message.to_lowercase().contains("weak")
                || item.message.to_lowercase().contains("insufficient")
        }) {
            issues.push("shallow".to_string());
        }

        // Check for general breathing issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("support")
                || item.message.to_lowercase().contains("airflow")
                || item.message.to_lowercase().contains("breathing")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Analyze articulation issues from feedback
    fn analyze_articulation_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for consonant cluster issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("cluster")
                || item.message.to_lowercase().contains("consonant group")
                || item.message.to_lowercase().contains("str")
                || item.message.to_lowercase().contains("spr")
        }) {
            issues.push("consonant_clusters".to_string());
        }

        // Check for final consonant issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("final")
                || item.message.to_lowercase().contains("ending")
                || item.message.to_lowercase().contains("word-final")
        }) {
            issues.push("final_consonants".to_string());
        }

        // Check for general articulation issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("articulation")
                || item.message.to_lowercase().contains("unclear")
                || item.message.to_lowercase().contains("mumbling")
                || item.message.to_lowercase().contains("consonant")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Analyze rhythm issues from feedback
    fn analyze_rhythm_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for speaking too fast
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("too fast")
                || item.message.to_lowercase().contains("rushed")
                || item.message.to_lowercase().contains("speed")
        }) {
            issues.push("too_fast".to_string());
        }

        // Check for speaking too slow
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("too slow")
                || item.message.to_lowercase().contains("sluggish")
                || item.message.to_lowercase().contains("drag")
        }) {
            issues.push("too_slow".to_string());
        }

        // Check for general rhythm issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("rhythm")
                || item.message.to_lowercase().contains("timing")
                || item.message.to_lowercase().contains("pacing")
                || item.message.to_lowercase().contains("flow")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Analyze emphasis issues from feedback
    fn analyze_emphasis_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for weak stress
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("weak stress")
                || item.message.to_lowercase().contains("no emphasis")
                || item.message.to_lowercase().contains("flat delivery")
        }) {
            issues.push("weak_stress".to_string());
        }

        // Check for over emphasis
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("over emphasis")
                || item.message.to_lowercase().contains("too much stress")
                || item.message.to_lowercase().contains("exaggerated")
        }) {
            issues.push("over_emphasis".to_string());
        }

        // Check for general emphasis issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("emphasis")
                || item.message.to_lowercase().contains("stress")
                || item.message.to_lowercase().contains("accent")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Analyze volume issues from feedback
    fn analyze_volume_issues(&self, feedback: &FeedbackResponse) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for too quiet
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("too quiet")
                || item.message.to_lowercase().contains("too soft")
                || item.message.to_lowercase().contains("hard to hear")
        }) {
            issues.push("too_quiet".to_string());
        }

        // Check for too loud
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("too loud")
                || item.message.to_lowercase().contains("shouting")
                || item.message.to_lowercase().contains("aggressive")
        }) {
            issues.push("too_loud".to_string());
        }

        // Check for general volume issues
        if feedback.feedback_items.iter().any(|item| {
            item.message.to_lowercase().contains("volume")
                || item.message.to_lowercase().contains("loudness")
                || item.message.to_lowercase().contains("projection")
        }) {
            issues.push("general".to_string());
        }

        issues
    }

    /// Generate context hash for caching
    fn generate_context_hash(&self, context: &SuggestionContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.current_exercise.hash(&mut hasher);
        ((context.session_progress * 100.0) as u32).hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached suggestion
    async fn get_cached_suggestion(&self, key: &str) -> Option<CachedSuggestion> {
        let cache = self.suggestion_cache.read().await;
        cache.get(key).cloned()
    }

    /// Check if cached suggestion is still valid
    fn is_cache_valid(&self, cached: &CachedSuggestion) -> bool {
        SystemTime::now()
            .duration_since(cached.created_at)
            .unwrap_or(Duration::from_secs(0))
            < Duration::from_secs(300) // 5 minutes
    }

    /// Cache a new suggestion
    async fn cache_suggestion(&self, key: String, suggestion: &Suggestion) {
        let mut cache = self.suggestion_cache.write().await;
        cache.insert(
            key,
            CachedSuggestion {
                suggestion: suggestion.clone(),
                created_at: SystemTime::now(),
                usage_count: 0,
                effectiveness_score: 0.5,
            },
        );

        // Limit cache size
        if cache.len() > 1000 {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
    }

    /// Update user pattern based on suggestions and feedback
    async fn update_user_pattern(
        &self,
        user_id: &str,
        suggestions: &[Suggestion],
        feedback: &FeedbackResponse,
    ) -> Result<(), FeedbackError> {
        let mut patterns = self.user_patterns.write().await;
        if let Some(pattern) = patterns.get_mut(user_id) {
            // Update learning velocity based on feedback score
            if feedback.overall_score < 1.0 {
                pattern.learning_velocity = (pattern.learning_velocity * 0.9
                    + feedback.overall_score * 0.1)
                    .max(0.1)
                    .min(1.0);
            }

            // Update problem areas based on low-scoring suggestions
            for suggestion in suggestions {
                if suggestion.priority == SuggestionPriority::High
                    || suggestion.priority == SuggestionPriority::Critical
                {
                    let existing_problem = pattern
                        .problem_areas
                        .iter_mut()
                        .find(|p| p.category == suggestion.category);

                    if let Some(problem) = existing_problem {
                        problem.frequency = (problem.frequency + 0.1).min(1.0);
                        problem.last_occurrence = SystemTime::now();
                    } else {
                        pattern.problem_areas.push(ProblemArea {
                            category: suggestion.category.clone(),
                            frequency: 0.1,
                            severity: suggestion.estimated_impact,
                            improvement_rate: 0.0,
                            last_occurrence: SystemTime::now(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Record feedback in history for pattern analysis
    async fn record_feedback_history(&self, user_id: &str, feedback: &FeedbackResponse) {
        let entry = FeedbackHistoryEntry {
            timestamp: SystemTime::now(),
            user_id: user_id.to_string(),
            feedback_type: format!("{:?}", feedback.feedback_type),
            accuracy_score: feedback.overall_score,
            areas_for_improvement: vec![], // Would be populated based on feedback analysis
            user_response: None,
        };

        let mut history = self.feedback_history.write().await;
        history.push(entry);

        // Keep only recent history
        if history.len() > 10000 {
            let excess = history.len() - 10000;
            history.drain(0..excess);
        }
    }

    /// Get user learning pattern
    pub async fn get_user_pattern(&self, user_id: &str) -> Option<UserLearningPattern> {
        let patterns = self.user_patterns.read().await;
        patterns.get(user_id).cloned()
    }

    /// Clear suggestion cache
    pub async fn clear_cache(&self) -> Result<(), FeedbackError> {
        let mut cache = self.suggestion_cache.write().await;
        cache.clear();
        Ok(())
    }
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions_per_session: 5,
            suggestion_cooldown_ms: 5000,
            personalization_enabled: true,
            adaptive_difficulty: true,
            context_window_size: 10,
            min_pattern_observations: 3,
            suggestion_categories: vec![
                SuggestionCategory::Pronunciation,
                SuggestionCategory::Clarity,
                SuggestionCategory::Pacing,
                SuggestionCategory::GeneralImprovement,
            ],
        }
    }
}

// Simple UUID generation for testing
mod uuid {
    use std::cell::Cell;

    thread_local! {
        static COUNTER: Cell<u64> = Cell::new(0);
    }

    pub struct Uuid(u64);

    impl Uuid {
        pub fn new_v4() -> Self {
            COUNTER.with(|c| {
                let val = c.get();
                c.set(val.wrapping_add(1));
                Uuid(val)
            })
        }
    }

    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:016x}", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeedbackType;

    #[tokio::test]
    async fn test_suggestion_engine_creation() {
        let engine = SuggestionEngine::new(SuggestionConfig::default());

        let patterns = engine.user_patterns.read().await;
        assert!(patterns.is_empty());
    }

    #[tokio::test]
    async fn test_suggestion_generation() {
        let engine = SuggestionEngine::new(SuggestionConfig::default());

        let feedback = FeedbackResponse {
            feedback_type: FeedbackType::Pronunciation,
            feedback_items: vec![UserFeedback {
                message: "Pronunciation needs improvement".to_string(),
                suggestion: Some("Practice vowel sounds".to_string()),
                confidence: 0.8,
                score: 0.6,
                priority: 0.7,
                metadata: HashMap::new(),
            }],
            overall_score: 0.6,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: Vec::new(),
                stable_areas: Vec::new(),
                overall_trend: 0.0,
                completion_percentage: 60.0,
            },
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(100),
        };

        let preferences = UserPreferences {
            user_id: "test_user".to_string(),
            ..Default::default()
        };

        let session_context = SuggestionContext {
            current_exercise: Some("vowel_practice".to_string()),
            recent_errors: Vec::new(),
            session_progress: 0.5,
            user_energy_level: 0.8,
            environmental_factors: Vec::new(),
        };

        let result = engine
            .generate_suggestions(&feedback, &preferences, &session_context)
            .await
            .unwrap();

        assert!(!result.suggestions.is_empty());
        assert!(result.personalization_confidence >= 0.0);
        assert!(result.generation_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_user_pattern_creation() {
        let engine = SuggestionEngine::new(SuggestionConfig::default());

        let pattern = engine.get_or_create_user_pattern("test_user").await;
        assert_eq!(pattern.user_id, "test_user");
        assert_eq!(pattern.skill_level, SkillLevel::Beginner);

        // Verify pattern is cached
        let cached_pattern = engine.get_user_pattern("test_user").await;
        assert!(cached_pattern.is_some());
    }

    #[test]
    fn test_category_relevance() {
        let engine = SuggestionEngine::new(SuggestionConfig::default());

        let feedback = FeedbackResponse {
            feedback_type: FeedbackType::Technical,
            feedback_items: vec![UserFeedback {
                message: "Clarity issues detected".to_string(),
                suggestion: Some("Speak more clearly".to_string()),
                confidence: 0.8,
                score: 0.5,
                priority: 0.6,
                metadata: HashMap::new(),
            }],
            overall_score: 0.5,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: Vec::new(),
                stable_areas: Vec::new(),
                overall_trend: 0.0,
                completion_percentage: 50.0,
            },
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(100),
        };

        let context = SuggestionContext {
            current_exercise: None,
            recent_errors: vec!["clarity problem".to_string()],
            session_progress: 0.3,
            user_energy_level: 0.8,
            environmental_factors: vec![],
        };

        assert!(engine.is_category_relevant(&SuggestionCategory::Clarity, &feedback, &context));
        assert!(!engine.is_category_relevant(&SuggestionCategory::Volume, &feedback, &context));
    }

    #[tokio::test]
    async fn test_suggestion_caching() {
        let engine = SuggestionEngine::new(SuggestionConfig::default());

        let suggestion = Suggestion {
            id: "test_suggestion".to_string(),
            category: SuggestionCategory::Pronunciation,
            title: "Test".to_string(),
            description: "Test suggestion".to_string(),
            action_steps: vec![],
            priority: SuggestionPriority::Medium,
            estimated_impact: 0.5,
            difficulty_level: 0.5,
            personalization_score: 0.5,
            context: SuggestionContext {
                current_exercise: None,
                recent_errors: vec![],
                session_progress: 0.0,
                user_energy_level: 1.0,
                environmental_factors: vec![],
            },
        };

        let cache_key = "test_key".to_string();
        engine
            .cache_suggestion(cache_key.clone(), &suggestion)
            .await;

        let cached = engine.get_cached_suggestion(&cache_key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().suggestion.id, "test_suggestion");
    }
}
