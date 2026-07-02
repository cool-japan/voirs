//! Pronunciation assessment: phonetic analysis, scoring, feedback
//! generation, and the pronunciation-training demonstration.

use crate::errors::EducationalVoiceError;
use crate::system::EducationalVoiceSystem;
use crate::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

impl EducationalVoiceSystem {
    /// Assess pronunciation and provide feedback
    pub async fn assess_pronunciation(
        &self,
        learner_id: &str,
        target_text: &str,
        spoken_audio: AudioData,
        language: &str,
    ) -> Result<PronunciationAssessment, EducationalVoiceError> {
        println!("🎤 Assessing pronunciation for: '{}'", target_text);

        let start_time = Instant::now();

        // Simulate phonetic analysis
        let phonetic_analysis = self
            .analyze_pronunciation_phonetics(target_text, &spoken_audio, language)
            .await?;

        // Calculate pronunciation scores
        let pronunciation_score = self
            .calculate_pronunciation_score(&phonetic_analysis)
            .await?;

        // Generate detailed feedback
        let feedback = self
            .generate_pronunciation_feedback(&phonetic_analysis, &pronunciation_score, target_text)
            .await?;

        // Create improvement suggestions
        let improvement_suggestions = self
            .generate_improvement_suggestions(&phonetic_analysis, &pronunciation_score)
            .await?;

        let assessment = PronunciationAssessment {
            assessment_id: Uuid::new_v4(),
            learner_id: learner_id.to_string(),
            target_text: target_text.to_string(),
            spoken_audio,
            phonetic_analysis,
            pronunciation_score,
            feedback,
            improvement_suggestions,
        };

        let assessment_time = start_time.elapsed();
        println!(
            "⚡ Pronunciation assessed in {:?} - Overall score: {:.1}/10",
            assessment_time,
            assessment.pronunciation_score.overall_score * 10.0
        );

        Ok(assessment)
    }

    /// Demonstrate comprehensive pronunciation training
    pub async fn demonstrate_pronunciation_training(&self) -> Result<(), EducationalVoiceError> {
        println!("\n🎤 === Pronunciation Training Demonstration ===\n");

        let training_examples = vec![
            (
                "English",
                "The quick brown fox jumps over the lazy dog",
                "beginner",
            ),
            (
                "Spanish",
                "La rápida zorra marrón salta sobre el perro perezoso",
                "intermediate",
            ),
            (
                "French",
                "Le renard brun et rapide saute par-dessus le chien paresseux",
                "advanced",
            ),
            (
                "German",
                "Der schnelle braune Fuchs springt über den faulen Hund",
                "intermediate",
            ),
            (
                "Japanese",
                "素早い茶色の狐が怠け者の犬を飛び越える",
                "advanced",
            ),
        ];

        for (language, text, level) in training_examples {
            println!("📍 Language: {} (Level: {})", language, level);
            println!("📝 Target Text: {}", text);

            // Simulate audio input
            let mock_audio = AudioData; // Mock audio data

            // Assess pronunciation
            let assessment = self
                .assess_pronunciation("demo_learner", text, mock_audio, language)
                .await?;

            // Display results
            println!("📊 Pronunciation Scores:");
            println!(
                "  • Overall: {:.1}/10",
                assessment.pronunciation_score.overall_score * 10.0
            );
            println!(
                "  • Phoneme Accuracy: {:.1}/10",
                assessment.pronunciation_score.phoneme_accuracy * 10.0
            );
            println!(
                "  • Rhythm: {:.1}/10",
                assessment.pronunciation_score.rhythm_score * 10.0
            );
            println!(
                "  • Intonation: {:.1}/10",
                assessment.pronunciation_score.intonation_score * 10.0
            );
            println!(
                "  • Fluency: {:.1}/10",
                assessment.pronunciation_score.fluency_score * 10.0
            );

            println!("💬 Feedback: {}", assessment.feedback.overall_feedback);
            println!(
                "🎯 Improvement Areas: {}",
                assessment.improvement_suggestions.len()
            );

            for suggestion in assessment.improvement_suggestions.iter().take(2) {
                println!("  • {}: {}", suggestion.target_area, suggestion.description);
            }

            println!();
        }

        Ok(())
    }

    async fn analyze_pronunciation_phonetics(
        &self,
        target_text: &str,
        _spoken_audio: &AudioData,
        language: &str,
    ) -> Result<PhoneticAnalysis, EducationalVoiceError> {
        // Simulate phonetic analysis
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Mock phonetic analysis based on text length and complexity
        let _word_count = target_text.split_whitespace().count();
        let target_phonemes = self.text_to_phonemes(target_text, language);
        let spoken_phonemes = self.simulate_spoken_phonemes(&target_phonemes);

        let mut phoneme_accuracy = HashMap::new();
        for phoneme in &target_phonemes {
            // Simulate accuracy based on phoneme difficulty
            let accuracy = match phoneme.symbol.as_str() {
                "r" | "rr" => 0.6, // Difficult sounds
                "th" | "ð" => 0.7,
                _ => 0.85,
            };
            phoneme_accuracy.insert(phoneme.symbol.clone(), accuracy);
        }

        Ok(PhoneticAnalysis {
            target_phonemes,
            spoken_phonemes,
            phoneme_accuracy,
            rhythm_analysis: RhythmAnalysis,
            intonation_analysis: IntonationAnalysis,
            stress_pattern_analysis: StressPatternAnalysis,
        })
    }

    pub(crate) async fn calculate_pronunciation_score(
        &self,
        analysis: &PhoneticAnalysis,
    ) -> Result<PronunciationScore, EducationalVoiceError> {
        // Calculate overall pronunciation score based on various factors
        let phoneme_accuracy: f32 = analysis.phoneme_accuracy.values().sum::<f32>()
            / analysis.phoneme_accuracy.len() as f32;
        let rhythm_score = 0.8; // Mock rhythm score
        let intonation_score = 0.75; // Mock intonation score
        let stress_score = 0.82; // Mock stress score
        let fluency_score = (phoneme_accuracy + rhythm_score) / 2.0;

        let overall_score = phoneme_accuracy * 0.4
            + rhythm_score * 0.2
            + intonation_score * 0.2
            + stress_score * 0.1
            + fluency_score * 0.1;

        Ok(PronunciationScore {
            overall_score,
            phoneme_accuracy,
            rhythm_score,
            intonation_score,
            stress_score,
            fluency_score,
            confidence_level: 0.85,
        })
    }

    async fn generate_pronunciation_feedback(
        &self,
        analysis: &PhoneticAnalysis,
        score: &PronunciationScore,
        target_text: &str,
    ) -> Result<PronunciationFeedback, EducationalVoiceError> {
        let overall_feedback = if score.overall_score >= 0.9 {
            format!(
                "Excellent pronunciation of '{}'! Your articulation is very clear and natural.",
                target_text
            )
        } else if score.overall_score >= 0.8 {
            format!(
                "Good pronunciation of '{}'! You're doing well with most sounds.",
                target_text
            )
        } else if score.overall_score >= 0.7 {
            format!(
                "Fair pronunciation of '{}'! There are some areas we can improve together.",
                target_text
            )
        } else {
            format!(
                "Keep practicing '{}'! Every attempt is progress toward better pronunciation.",
                target_text
            )
        };

        let mut specific_phoneme_feedback = HashMap::new();
        for (phoneme, accuracy) in &analysis.phoneme_accuracy {
            if *accuracy < 0.7 {
                specific_phoneme_feedback.insert(
                    phoneme.clone(),
                    format!(
                        "The '{}' sound needs more practice. Try focusing on tongue position.",
                        phoneme
                    ),
                );
            }
        }

        let rhythm_feedback = if score.rhythm_score >= 0.8 {
            "Your rhythm and timing are quite natural!".to_string()
        } else {
            "Try to pay attention to the natural rhythm and stress patterns.".to_string()
        };

        let intonation_feedback = if score.intonation_score >= 0.8 {
            "Your intonation sounds natural and expressive!".to_string()
        } else {
            "Work on the rise and fall of your voice to sound more natural.".to_string()
        };

        Ok(PronunciationFeedback {
            overall_feedback,
            specific_phoneme_feedback,
            rhythm_feedback,
            intonation_feedback,
            encouragement:
                "Remember, pronunciation improves with practice. You're making great progress!"
                    .to_string(),
            next_steps: vec![
                "Practice the challenging sounds daily".to_string(),
                "Listen to native speakers and imitate their pronunciation".to_string(),
                "Record yourself and compare with the target pronunciation".to_string(),
            ],
        })
    }

    async fn generate_improvement_suggestions(
        &self,
        _analysis: &PhoneticAnalysis,
        score: &PronunciationScore,
    ) -> Result<Vec<ImprovementSuggestion>, EducationalVoiceError> {
        let mut suggestions = Vec::new();

        // Generate suggestions based on pronunciation weaknesses
        if score.phoneme_accuracy < 0.8 {
            suggestions.push(ImprovementSuggestion {
                suggestion_id: Uuid::new_v4(),
                target_area: "Phoneme Accuracy".to_string(),
                description: "Focus on articulating individual sounds more clearly".to_string(),
                practice_exercises: vec![
                    "Phoneme isolation drills".to_string(),
                    "Minimal pair practice".to_string(),
                    "Sound repetition exercises".to_string(),
                ],
                priority: Priority::High,
                estimated_practice_time: Duration::from_secs(15 * 60),
            });
        }

        if score.rhythm_score < 0.8 {
            suggestions.push(ImprovementSuggestion {
                suggestion_id: Uuid::new_v4(),
                target_area: "Rhythm and Timing".to_string(),
                description: "Practice natural speech rhythm and stress patterns".to_string(),
                practice_exercises: vec![
                    "Metronome-based speaking practice".to_string(),
                    "Sentence stress exercises".to_string(),
                    "Rhythm shadowing activities".to_string(),
                ],
                priority: Priority::Medium,
                estimated_practice_time: Duration::from_secs(10 * 60),
            });
        }

        if score.intonation_score < 0.8 {
            suggestions.push(ImprovementSuggestion {
                suggestion_id: Uuid::new_v4(),
                target_area: "Intonation Patterns".to_string(),
                description: "Work on the melody and pitch patterns of speech".to_string(),
                practice_exercises: vec![
                    "Intonation contour practice".to_string(),
                    "Question vs. statement intonation".to_string(),
                    "Emotional expression through intonation".to_string(),
                ],
                priority: Priority::Medium,
                estimated_practice_time: Duration::from_secs(12 * 60),
            });
        }

        Ok(suggestions)
    }

    // Helper methods for phonetic processing
    fn text_to_phonemes(&self, text: &str, _language: &str) -> Vec<Phoneme> {
        // Simplified phoneme extraction
        text.chars()
            .enumerate()
            .map(|(i, c)| Phoneme {
                symbol: c.to_string(),
                position: Duration::from_millis(i as u64 * 100),
                duration: Duration::from_millis(100),
                frequency_data: FrequencyData,
                confidence: 0.9,
            })
            .collect()
    }

    fn simulate_spoken_phonemes(&self, target_phonemes: &[Phoneme]) -> Vec<Phoneme> {
        // Simulate slightly different phonemes representing learner's pronunciation
        target_phonemes
            .iter()
            .map(|p| {
                Phoneme {
                    symbol: p.symbol.clone(),
                    position: p.position,
                    duration: p.duration + Duration::from_millis(10), // Slightly longer
                    frequency_data: FrequencyData,
                    confidence: p.confidence * 0.9, // Slightly less confident
                }
            })
            .collect()
    }
}
