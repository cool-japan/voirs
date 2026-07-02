//! Tests for the educational tools example.

use crate::system::EducationalVoiceSystem;
use crate::types::*;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

#[tokio::test]
async fn test_educational_system_creation() {
    let _system = EducationalVoiceSystem::new();
    // Test that system components are initialized
    // This would normally verify that all engines are properly set up
}

#[tokio::test]
async fn test_lesson_creation() {
    let system = EducationalVoiceSystem::new();
    let lesson = system
        .create_lesson("Spanish", "Greetings", SkillLevel::Beginner)
        .await;

    assert!(lesson.is_ok());
    let lesson = lesson.unwrap();
    assert_eq!(lesson.language, "Spanish");
    assert!(!lesson.activities.is_empty());
    assert!(lesson.estimated_duration > Duration::from_secs(0));
}

#[tokio::test]
async fn test_learning_session_start() {
    let system = EducationalVoiceSystem::new();
    let lesson = system
        .create_lesson("French", "Colors", SkillLevel::Elementary)
        .await
        .unwrap();
    let session = system.start_learning_session("test_learner", lesson).await;

    assert!(session.is_ok());
    let session = session.unwrap();
    assert_eq!(session.learner_id, "test_learner");
    assert!(matches!(session.session_state, SessionState::Starting));
}

#[tokio::test]
async fn test_pronunciation_assessment() {
    let system = EducationalVoiceSystem::new();
    let mock_audio = AudioData;
    let assessment = system
        .assess_pronunciation("test_learner", "Hello world", mock_audio, "English")
        .await;

    assert!(assessment.is_ok());
    let assessment = assessment.unwrap();
    assert_eq!(assessment.target_text, "Hello world");
    assert!(assessment.pronunciation_score.overall_score >= 0.0);
    assert!(assessment.pronunciation_score.overall_score <= 1.0);
    assert!(!assessment.feedback.overall_feedback.is_empty());
}

#[tokio::test]
async fn test_skill_level_conversion() {
    let system = EducationalVoiceSystem::new();

    assert_eq!(system.skill_level_to_number(&SkillLevel::Beginner), 1);
    assert_eq!(system.skill_level_to_number(&SkillLevel::Elementary), 2);
    assert_eq!(system.skill_level_to_number(&SkillLevel::Intermediate), 3);
    assert_eq!(system.skill_level_to_number(&SkillLevel::Advanced), 5);

    assert!(matches!(
        system.number_to_skill_level(1),
        SkillLevel::Beginner
    ));
    assert!(matches!(
        system.number_to_skill_level(3),
        SkillLevel::Intermediate
    ));
    assert!(matches!(
        system.number_to_skill_level(6),
        SkillLevel::Proficient
    ));
}

#[tokio::test]
async fn test_target_skills_determination() {
    let system = EducationalVoiceSystem::new();

    let pronunciation_skills = system.determine_target_skills("pronunciation practice");
    assert!(pronunciation_skills.contains(&LanguageSkill::Pronunciation));
    assert!(pronunciation_skills.contains(&LanguageSkill::Speaking));

    let vocabulary_skills = system.determine_target_skills("vocabulary building");
    assert!(vocabulary_skills.contains(&LanguageSkill::Vocabulary));

    let conversation_skills = system.determine_target_skills("conversation practice");
    assert!(conversation_skills.contains(&LanguageSkill::Conversation));
    assert!(conversation_skills.contains(&LanguageSkill::Listening));
}

#[tokio::test]
async fn test_pronunciation_score_calculation() {
    let system = EducationalVoiceSystem::new();

    // Create mock phonetic analysis
    let analysis = PhoneticAnalysis {
        target_phonemes: vec![],
        spoken_phonemes: vec![],
        phoneme_accuracy: HashMap::from([
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.7),
        ]),
        rhythm_analysis: RhythmAnalysis,
        intonation_analysis: IntonationAnalysis,
        stress_pattern_analysis: StressPatternAnalysis,
    };

    let score = system.calculate_pronunciation_score(&analysis).await;
    assert!(score.is_ok());

    let score = score.unwrap();
    assert!(score.overall_score >= 0.0);
    assert!(score.overall_score <= 1.0);
    assert!(score.phoneme_accuracy >= 0.0);
    assert!(score.phoneme_accuracy <= 1.0);
}

#[tokio::test]
async fn test_adaptive_content_generation() {
    let system = EducationalVoiceSystem::new();

    let learner_profile = LearnerProfile {
        learner_id: "test_learner".to_string(),
        name: "Test Student".to_string(),
        native_language: "English".to_string(),
        target_languages: vec!["Spanish".to_string()],
        learning_goals: vec![LearningGoal {
            goal_id: Uuid::new_v4(),
            description: "Basic Spanish conversation".to_string(),
            target_skill: LanguageSkill::Conversation,
            target_level: SkillLevel::Elementary,
            deadline: None,
            priority: Priority::High,
        }],
        skill_levels: HashMap::from([("Spanish_Conversation".to_string(), SkillLevel::Beginner)]),
        learning_preferences: LearningPreferences {
            preferred_learning_style: LearningStyle::Multimodal,
            session_duration: Duration::from_secs(15 * 60),
            difficulty_preference: DifficultyPreference::Balanced,
            audio_preferences: AudioPreferences {
                preferred_accent: "neutral".to_string(),
                speech_rate: 1.0,
                pitch_preference: 1.0,
                background_music: false,
                sound_effects: false,
            },
            visual_preferences: VisualPreferences {
                color_scheme: "default".to_string(),
                font_size: 12,
                animation_level: AnimationLevel::Standard,
                image_preference: ImagePreference::Realistic,
            },
            interaction_preferences: InteractionPreferences {
                preferred_input_method: InputMethod::Mixed,
                feedback_frequency: FeedbackFrequency::Immediate,
                help_system_usage: HelpSystemUsage::Occasional,
            },
        },
        performance_history: PerformanceHistory,
        accessibility_needs: AccessibilityNeeds,
    };

    let adaptive_lessons = system.generate_adaptive_content(&learner_profile).await;
    assert!(adaptive_lessons.is_ok());

    let lessons = adaptive_lessons.unwrap();
    assert!(!lessons.is_empty());
}

#[tokio::test]
async fn test_vocabulary_activities_generation() {
    let system = EducationalVoiceSystem::new();
    let activities = system
        .generate_vocabulary_activities("hola", "Spanish")
        .await;

    assert!(activities.is_ok());
    let activities = activities.unwrap();
    assert!(!activities.is_empty());

    // Check that we have different types of activities
    let activity_types: Vec<_> = activities.iter().map(|a| &a.activity_type).collect();
    assert!(activity_types.len() > 1); // Should have multiple activity types
}

#[tokio::test]
async fn test_pronunciation_guide_synthesis() {
    let system = EducationalVoiceSystem::new();
    let audio = system
        .synthesize_pronunciation_guide("bonjour", "French")
        .await;

    assert!(audio.is_ok());
    let audio = audio.unwrap();
    assert_eq!(audio.text, "bonjour");
    assert_eq!(audio.language, "French");
    assert!(audio.duration > Duration::from_secs(0));
}

#[tokio::test]
async fn test_learning_challenges_generation() {
    let system = EducationalVoiceSystem::new();

    let mock_profile = LearnerProfile {
        learner_id: "test".to_string(),
        name: "Test".to_string(),
        native_language: "English".to_string(),
        target_languages: vec!["Spanish".to_string()],
        learning_goals: vec![],
        skill_levels: HashMap::new(),
        learning_preferences: LearningPreferences {
            preferred_learning_style: LearningStyle::Visual,
            session_duration: Duration::from_secs(15 * 60),
            difficulty_preference: DifficultyPreference::Balanced,
            audio_preferences: AudioPreferences {
                preferred_accent: "neutral".to_string(),
                speech_rate: 1.0,
                pitch_preference: 1.0,
                background_music: false,
                sound_effects: false,
            },
            visual_preferences: VisualPreferences {
                color_scheme: "default".to_string(),
                font_size: 12,
                animation_level: AnimationLevel::Standard,
                image_preference: ImagePreference::Realistic,
            },
            interaction_preferences: InteractionPreferences {
                preferred_input_method: InputMethod::Touch,
                feedback_frequency: FeedbackFrequency::Immediate,
                help_system_usage: HelpSystemUsage::Frequent,
            },
        },
        performance_history: PerformanceHistory,
        accessibility_needs: AccessibilityNeeds,
    };

    let challenges = system.generate_learning_challenges(&mock_profile).await;
    assert!(challenges.is_ok());

    let challenges = challenges.unwrap();
    assert!(!challenges.is_empty());

    for challenge in &challenges {
        assert!(!challenge.title.is_empty());
        assert!(!challenge.description.is_empty());
        assert!(challenge.reward_points > 0);
    }
}
