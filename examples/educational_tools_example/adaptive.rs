//! Adaptive-difficulty content generation: builds bridging lessons between a
//! learner's current skill level and their learning goals, and personalizes
//! lessons to individual learner preferences.

use crate::errors::EducationalVoiceError;
use crate::system::EducationalVoiceSystem;
use crate::types::*;

impl EducationalVoiceSystem {
    /// Generate adaptive learning content based on learner progress
    pub async fn generate_adaptive_content(
        &self,
        learner_profile: &LearnerProfile,
    ) -> Result<Vec<Lesson>, EducationalVoiceError> {
        println!(
            "🧠 Generating adaptive content for learner: {}",
            learner_profile.name
        );

        let mut adaptive_lessons = Vec::new();

        // Analyze learner's current skill levels and learning goals
        for target_language in &learner_profile.target_languages {
            for goal in &learner_profile.learning_goals {
                // Get current skill level for this goal
                let current_level = learner_profile
                    .skill_levels
                    .get(&format!("{}_{:?}", target_language, goal.target_skill))
                    .cloned()
                    .unwrap_or(SkillLevel::Beginner);

                // Generate lessons to bridge the gap to target level
                let bridging_lessons = self
                    .generate_bridging_lessons(
                        target_language,
                        &goal.target_skill,
                        &current_level,
                        &goal.target_level,
                    )
                    .await?;

                adaptive_lessons.extend(bridging_lessons);
            }
        }

        // Personalize lessons based on learning preferences
        for lesson in &mut adaptive_lessons {
            self.personalize_lesson(lesson, learner_profile).await?;
        }

        println!("✅ Generated {} adaptive lessons", adaptive_lessons.len());
        Ok(adaptive_lessons)
    }

    async fn generate_bridging_lessons(
        &self,
        language: &str,
        skill: &LanguageSkill,
        current_level: &SkillLevel,
        target_level: &SkillLevel,
    ) -> Result<Vec<Lesson>, EducationalVoiceError> {
        let mut lessons = Vec::new();

        let current_num = self.skill_level_to_number(current_level);
        let target_num = self.skill_level_to_number(target_level);

        for level_num in (current_num + 1)..=target_num {
            let level = self.number_to_skill_level(level_num);
            let topic = format!("{:?} Skills Development", skill);

            let lesson = self.create_lesson(language, &topic, level).await?;
            lessons.push(lesson);
        }

        Ok(lessons)
    }

    async fn personalize_lesson(
        &self,
        _lesson: &mut Lesson,
        _learner_profile: &LearnerProfile,
    ) -> Result<(), EducationalVoiceError> {
        // Simulate lesson personalization based on learner preferences
        // This would modify the lesson content, activities, and presentation style
        Ok(())
    }
}
