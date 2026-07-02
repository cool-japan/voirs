//! Interactive vocabulary-building activities and demonstration.

use crate::errors::EducationalVoiceError;
use crate::system::EducationalVoiceSystem;
use crate::types::*;
use std::time::Duration;
use uuid::Uuid;

impl EducationalVoiceSystem {
    /// Demonstrate interactive vocabulary learning
    pub async fn demonstrate_vocabulary_learning(&self) -> Result<(), EducationalVoiceError> {
        println!("\n📖 === Interactive Vocabulary Learning Demo ===\n");

        let vocabulary_sets = vec![
            (
                "Spanish",
                "Family Members",
                vec!["padre", "madre", "hermano", "hermana", "abuelo", "abuela"],
            ),
            (
                "French",
                "Food Items",
                vec!["pomme", "pain", "fromage", "lait", "viande", "légume"],
            ),
            (
                "German",
                "Colors",
                vec!["rot", "blau", "grün", "gelb", "schwarz", "weiß"],
            ),
            (
                "Italian",
                "Weather",
                vec!["sole", "pioggia", "neve", "vento", "nuvola", "tempesta"],
            ),
        ];

        for (language, category, words) in vocabulary_sets {
            println!("🌍 Language: {} - Category: {}", language, category);

            for (i, word) in words.iter().enumerate() {
                println!("  {}. 🔤 Word: {}", i + 1, word);

                // Simulate vocabulary learning activities
                let activities = self.generate_vocabulary_activities(word, language).await?;

                println!("     📚 Learning Activities:");
                for activity in activities.iter().take(2) {
                    println!(
                        "       • {}: {}",
                        activity.activity_type.name(),
                        activity.title
                    );
                }

                // Simulate pronunciation synthesis
                let pronunciation_audio =
                    self.synthesize_pronunciation_guide(word, language).await?;
                println!(
                    "     🎵 Pronunciation Guide: {:?} duration",
                    pronunciation_audio.duration
                );

                // Simulate usage examples
                let examples = self.generate_usage_examples(word, language).await?;
                println!("     📝 Usage Examples: {} provided", examples.len());
            }

            println!();
        }

        Ok(())
    }

    pub(crate) async fn generate_vocabulary_activities(
        &self,
        word: &str,
        _language: &str,
    ) -> Result<Vec<LearningActivity>, EducationalVoiceError> {
        Ok(vec![
            LearningActivity {
                activity_id: Uuid::new_v4(),
                activity_type: ActivityType::Pronunciation,
                title: format!("Pronounce '{}'", word),
                instructions: format!("Listen and repeat the pronunciation of '{}'", word),
                content: ActivityContent,
                scoring_criteria: ScoringCriteria,
                time_limit: Some(Duration::from_secs(2 * 60)),
                hints: vec!["Listen carefully to the native speaker model".to_string()],
            },
            LearningActivity {
                activity_id: Uuid::new_v4(),
                activity_type: ActivityType::Vocabulary,
                title: format!("Use '{}' in context", word),
                instructions: format!("Create a sentence using the word '{}'", word),
                content: ActivityContent,
                scoring_criteria: ScoringCriteria,
                time_limit: Some(Duration::from_secs(3 * 60)),
                hints: vec!["Think about when you might use this word".to_string()],
            },
        ])
    }

    pub(crate) async fn synthesize_pronunciation_guide(
        &self,
        word: &str,
        language: &str,
    ) -> Result<AudioContent, EducationalVoiceError> {
        // Simulate pronunciation guide synthesis
        tokio::time::sleep(Duration::from_millis(50)).await;

        Ok(AudioContent {
            audio_id: Uuid::new_v4(),
            text: word.to_string(),
            language: language.to_string(),
            speaker_profile: SpeakerProfile,
            synthesis_parameters: SynthesisParameters,
            audio_path: Some(format!("{}_pronunciation.wav", word)),
            duration: Duration::from_secs(2),
        })
    }

    async fn generate_usage_examples(
        &self,
        _word: &str,
        _language: &str,
    ) -> Result<Vec<Example>, EducationalVoiceError> {
        // Simulate generation of usage examples
        Ok(vec![Example, Example, Example]) // Mock examples
    }
}
