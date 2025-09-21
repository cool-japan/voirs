//! Children's Speech Evaluation Protocols
//!
//! This module provides specialized evaluation metrics designed for children's speech
//! synthesis systems. It includes age-appropriate models, developmental milestone
//! tracking, and child-specific intelligibility and naturalness assessments.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::traits::*;
use crate::{AudioBuffer, EvaluationError, LanguageCode};

/// Age group categories for children's speech development
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    /// Early childhood (2-4 years)
    EarlyChildhood,
    /// Preschool (4-6 years)
    Preschool,
    /// Early elementary (6-8 years)
    EarlyElementary,
    /// Late elementary (8-12 years)
    LateElementary,
    /// Early adolescent (12-15 years)
    EarlyAdolescent,
    /// Late adolescent (15-18 years)
    LateAdolescent,
}

impl AgeGroup {
    /// Get typical age range for the group
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            AgeGroup::EarlyChildhood => (2, 4),
            AgeGroup::Preschool => (4, 6),
            AgeGroup::EarlyElementary => (6, 8),
            AgeGroup::LateElementary => (8, 12),
            AgeGroup::EarlyAdolescent => (12, 15),
            AgeGroup::LateAdolescent => (15, 18),
        }
    }

    /// Get expected speech characteristics for age group
    pub fn speech_characteristics(&self) -> SpeechCharacteristics {
        match self {
            AgeGroup::EarlyChildhood => SpeechCharacteristics {
                fundamental_frequency_range: (250.0, 400.0),
                articulation_accuracy: 0.6,
                vocabulary_complexity: 0.3,
                fluency_expectations: 0.4,
                grammatical_complexity: 0.3,
                prosody_development: 0.4,
                voice_quality_stability: 0.5,
            },
            AgeGroup::Preschool => SpeechCharacteristics {
                fundamental_frequency_range: (230.0, 380.0),
                articulation_accuracy: 0.75,
                vocabulary_complexity: 0.5,
                fluency_expectations: 0.6,
                grammatical_complexity: 0.5,
                prosody_development: 0.6,
                voice_quality_stability: 0.65,
            },
            AgeGroup::EarlyElementary => SpeechCharacteristics {
                fundamental_frequency_range: (220.0, 350.0),
                articulation_accuracy: 0.85,
                vocabulary_complexity: 0.7,
                fluency_expectations: 0.75,
                grammatical_complexity: 0.7,
                prosody_development: 0.75,
                voice_quality_stability: 0.8,
            },
            AgeGroup::LateElementary => SpeechCharacteristics {
                fundamental_frequency_range: (200.0, 320.0),
                articulation_accuracy: 0.9,
                vocabulary_complexity: 0.8,
                fluency_expectations: 0.85,
                grammatical_complexity: 0.8,
                prosody_development: 0.85,
                voice_quality_stability: 0.85,
            },
            AgeGroup::EarlyAdolescent => SpeechCharacteristics {
                fundamental_frequency_range: (180.0, 300.0),
                articulation_accuracy: 0.95,
                vocabulary_complexity: 0.9,
                fluency_expectations: 0.9,
                grammatical_complexity: 0.9,
                prosody_development: 0.9,
                voice_quality_stability: 0.8, // Lower due to voice changes
            },
            AgeGroup::LateAdolescent => SpeechCharacteristics {
                fundamental_frequency_range: (160.0, 280.0),
                articulation_accuracy: 0.98,
                vocabulary_complexity: 0.95,
                fluency_expectations: 0.95,
                grammatical_complexity: 0.95,
                prosody_development: 0.95,
                voice_quality_stability: 0.9,
            },
        }
    }
}

/// Expected speech characteristics for age groups
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechCharacteristics {
    /// Expected fundamental frequency range (Hz)
    pub fundamental_frequency_range: (f32, f32),
    /// Expected articulation accuracy (0.0-1.0)
    pub articulation_accuracy: f32,
    /// Expected vocabulary complexity (0.0-1.0)
    pub vocabulary_complexity: f32,
    /// Expected fluency level (0.0-1.0)
    pub fluency_expectations: f32,
    /// Expected grammatical complexity (0.0-1.0)
    pub grammatical_complexity: f32,
    /// Expected prosody development (0.0-1.0)
    pub prosody_development: f32,
    /// Expected voice quality stability (0.0-1.0)
    pub voice_quality_stability: f32,
}

/// Developmental milestone categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevelopmentalMilestone {
    /// Phoneme acquisition milestones
    PhonemeAcquisition,
    /// Grammatical development milestones
    GrammaticalDevelopment,
    /// Vocabulary growth milestones
    VocabularyGrowth,
    /// Prosody development milestones
    ProsodyDevelopment,
    /// Fluency development milestones
    FluencyDevelopment,
    /// Voice quality milestones
    VoiceQuality,
}

/// Phoneme acquisition stages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeAcquisition {
    /// Early acquired phonemes (by age 3)
    pub early_phonemes: Vec<String>,
    /// Middle acquired phonemes (by age 5)
    pub middle_phonemes: Vec<String>,
    /// Late acquired phonemes (by age 8)
    pub late_phonemes: Vec<String>,
    /// Phoneme accuracy by age group
    pub age_accuracy_map: HashMap<AgeGroup, f32>,
}

impl Default for PhonemeAcquisition {
    fn default() -> Self {
        let mut age_accuracy_map = HashMap::new();
        age_accuracy_map.insert(AgeGroup::EarlyChildhood, 0.6);
        age_accuracy_map.insert(AgeGroup::Preschool, 0.75);
        age_accuracy_map.insert(AgeGroup::EarlyElementary, 0.85);
        age_accuracy_map.insert(AgeGroup::LateElementary, 0.9);
        age_accuracy_map.insert(AgeGroup::EarlyAdolescent, 0.95);
        age_accuracy_map.insert(AgeGroup::LateAdolescent, 0.98);

        Self {
            early_phonemes: vec![
                "m".to_string(),
                "n".to_string(),
                "p".to_string(),
                "b".to_string(),
                "t".to_string(),
                "d".to_string(),
                "k".to_string(),
                "g".to_string(),
                "f".to_string(),
                "w".to_string(),
                "h".to_string(),
                "j".to_string(),
            ],
            middle_phonemes: vec![
                "l".to_string(),
                "s".to_string(),
                "z".to_string(),
                "ʃ".to_string(),
                "ʒ".to_string(),
                "ʧ".to_string(),
                "ʤ".to_string(),
                "v".to_string(),
            ],
            late_phonemes: vec!["r".to_string(), "θ".to_string(), "ð".to_string()],
            age_accuracy_map,
        }
    }
}

/// Child-specific intelligibility assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildIntelligibilityResult {
    /// Overall intelligibility score adjusted for age
    pub age_adjusted_intelligibility: f32,
    /// Raw intelligibility score
    pub raw_intelligibility: f32,
    /// Phoneme-level intelligibility
    pub phoneme_intelligibility: HashMap<String, f32>,
    /// Word-level intelligibility
    pub word_intelligibility: f32,
    /// Sentence-level intelligibility
    pub sentence_intelligibility: f32,
    /// Context effect on intelligibility
    pub context_effect: f32,
    /// Listener familiarity adjustment
    pub familiarity_adjustment: f32,
}

/// Child-specific naturalness assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildNaturalnessResult {
    /// Age-appropriate voice characteristics
    pub age_appropriateness: f32,
    /// Voice quality naturalness
    pub voice_quality: f32,
    /// Prosodic naturalness for age
    pub prosodic_naturalness: f32,
    /// Emotional expression appropriateness
    pub emotional_appropriateness: f32,
    /// Developmental stage alignment
    pub developmental_alignment: f32,
    /// Overall naturalness score
    pub overall_naturalness: f32,
}

/// Developmental milestone tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevelopmentalAssessment {
    /// Current developmental stage
    pub current_stage: AgeGroup,
    /// Milestone achievements
    pub milestone_achievements: HashMap<DevelopmentalMilestone, f32>,
    /// Areas needing development
    pub development_areas: Vec<String>,
    /// Strengths identified
    pub strengths: Vec<String>,
    /// Overall developmental score
    pub developmental_score: f32,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Educational progress tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EducationalProgressResult {
    /// Reading level assessment
    pub reading_level: f32,
    /// Vocabulary level assessment
    pub vocabulary_level: f32,
    /// Comprehension assessment
    pub comprehension_level: f32,
    /// Communication effectiveness
    pub communication_effectiveness: f32,
    /// Learning objective alignment
    pub learning_alignment: f32,
    /// Educational value score
    pub educational_value: f32,
}

/// Configuration for children's speech evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildrenEvaluationConfig {
    /// Target age group for evaluation
    pub target_age_group: AgeGroup,
    /// Language being evaluated
    pub language: LanguageCode,
    /// Whether to apply age-specific adjustments
    pub age_adjusted_scoring: bool,
    /// Whether to assess developmental milestones
    pub assess_developmental_milestones: bool,
    /// Whether to track educational progress
    pub track_educational_progress: bool,
    /// Whether to evaluate voice appropriateness
    pub evaluate_voice_appropriateness: bool,
    /// Custom phoneme acquisition model
    pub phoneme_acquisition: Option<PhonemeAcquisition>,
    /// Expected vocabulary level
    pub expected_vocabulary_level: Option<f32>,
    /// Educational context (classroom, therapy, etc.)
    pub educational_context: Option<String>,
    /// Listener familiarity level (parent, teacher, stranger)
    pub listener_familiarity: ListenerFamiliarity,
}

/// Listener familiarity with child's speech
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ListenerFamiliarity {
    /// Very familiar (parent, close family)
    VeryFamiliar,
    /// Familiar (teacher, frequent contact)
    Familiar,
    /// Somewhat familiar (occasional contact)
    SomewhatFamiliar,
    /// Unfamiliar (stranger)
    Unfamiliar,
}

impl ListenerFamiliarity {
    /// Get intelligibility adjustment factor
    pub fn intelligibility_adjustment(&self) -> f32 {
        match self {
            ListenerFamiliarity::VeryFamiliar => 1.2,
            ListenerFamiliarity::Familiar => 1.1,
            ListenerFamiliarity::SomewhatFamiliar => 1.0,
            ListenerFamiliarity::Unfamiliar => 0.9,
        }
    }
}

impl Default for ChildrenEvaluationConfig {
    fn default() -> Self {
        Self {
            target_age_group: AgeGroup::EarlyElementary,
            language: LanguageCode::EnUs,
            age_adjusted_scoring: true,
            assess_developmental_milestones: true,
            track_educational_progress: true,
            evaluate_voice_appropriateness: true,
            phoneme_acquisition: None,
            expected_vocabulary_level: None,
            educational_context: None,
            listener_familiarity: ListenerFamiliarity::Familiar,
        }
    }
}

/// Comprehensive children's speech evaluation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildrenEvaluationResult {
    /// Overall evaluation score adjusted for age
    pub overall_score: f32,
    /// Child-specific intelligibility assessment
    pub intelligibility: ChildIntelligibilityResult,
    /// Child-specific naturalness assessment
    pub naturalness: ChildNaturalnessResult,
    /// Developmental milestone assessment
    pub developmental_assessment: Option<DevelopmentalAssessment>,
    /// Educational progress evaluation
    pub educational_progress: Option<EducationalProgressResult>,
    /// Voice appropriateness for age
    pub voice_appropriateness: f32,
    /// Age group compatibility score
    pub age_compatibility: f32,
    /// Communication effectiveness score
    pub communication_effectiveness: f32,
    /// Developmental recommendations
    pub recommendations: Vec<String>,
    /// Confidence in the evaluation
    pub confidence: f32,
}

/// Children's speech evaluator implementation
pub struct ChildrenSpeechEvaluator {
    config: ChildrenEvaluationConfig,
}

impl ChildrenSpeechEvaluator {
    /// Create a new children's speech evaluator
    pub async fn new() -> Result<Self, EvaluationError> {
        Ok(Self {
            config: ChildrenEvaluationConfig::default(),
        })
    }

    /// Create evaluator with custom configuration
    pub async fn with_config(config: ChildrenEvaluationConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Update evaluator configuration
    pub fn set_config(&mut self, config: ChildrenEvaluationConfig) {
        self.config = config;
    }

    /// Evaluate children's speech synthesis quality
    pub async fn evaluate_children_speech(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        target_text: Option<&str>,
    ) -> Result<ChildrenEvaluationResult, EvaluationError> {
        // Assess child-specific intelligibility
        let intelligibility = self
            .assess_child_intelligibility(generated_audio, reference_audio, target_text)
            .await?;

        // Assess child-specific naturalness
        let naturalness = self
            .assess_child_naturalness(generated_audio, reference_audio)
            .await?;

        // Assess developmental milestones if enabled
        let developmental_assessment = if self.config.assess_developmental_milestones {
            Some(
                self.assess_developmental_milestones(generated_audio, target_text)
                    .await?,
            )
        } else {
            None
        };

        // Track educational progress if enabled
        let educational_progress = if self.config.track_educational_progress {
            Some(
                self.assess_educational_progress(generated_audio, target_text)
                    .await?,
            )
        } else {
            None
        };

        // Evaluate voice appropriateness
        let voice_appropriateness = self.evaluate_voice_appropriateness(generated_audio).await?;

        // Calculate age compatibility
        let age_compatibility = self.calculate_age_compatibility(generated_audio).await?;

        // Calculate communication effectiveness
        let communication_effectiveness = self
            .calculate_communication_effectiveness(&intelligibility, &naturalness)
            .await?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &intelligibility,
            &naturalness,
            developmental_assessment.as_ref(),
        );

        // Calculate overall score with age adjustments
        let overall_score = self
            .calculate_overall_score(
                &intelligibility,
                &naturalness,
                voice_appropriateness,
                age_compatibility,
                communication_effectiveness,
            )
            .await?;

        // Calculate evaluation confidence
        let confidence =
            self.calculate_evaluation_confidence(generated_audio, reference_audio, target_text);

        Ok(ChildrenEvaluationResult {
            overall_score,
            intelligibility,
            naturalness,
            developmental_assessment,
            educational_progress,
            voice_appropriateness,
            age_compatibility,
            communication_effectiveness,
            recommendations,
            confidence,
        })
    }

    /// Assess child-specific intelligibility
    async fn assess_child_intelligibility(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        target_text: Option<&str>,
    ) -> Result<ChildIntelligibilityResult, EvaluationError> {
        // Basic intelligibility assessment (simplified)
        let raw_intelligibility = self
            .calculate_raw_intelligibility(generated_audio, reference_audio)
            .await?;

        // Apply age-specific adjustments
        let age_characteristics = self.config.target_age_group.speech_characteristics();
        let age_adjustment =
            self.calculate_age_adjustment(raw_intelligibility, &age_characteristics);
        let age_adjusted_intelligibility = (raw_intelligibility * age_adjustment).min(1.0);

        // Assess phoneme-level intelligibility
        let phoneme_intelligibility = self
            .assess_phoneme_intelligibility(generated_audio, target_text)
            .await?;

        // Assess word and sentence level intelligibility
        let word_intelligibility = self
            .assess_word_intelligibility(generated_audio, target_text)
            .await?;
        let sentence_intelligibility = self
            .assess_sentence_intelligibility(generated_audio, target_text)
            .await?;

        // Calculate context effect
        let context_effect = self.calculate_context_effect(target_text);

        // Apply listener familiarity adjustment
        let familiarity_adjustment = self
            .config
            .listener_familiarity
            .intelligibility_adjustment();

        Ok(ChildIntelligibilityResult {
            age_adjusted_intelligibility,
            raw_intelligibility,
            phoneme_intelligibility,
            word_intelligibility,
            sentence_intelligibility,
            context_effect,
            familiarity_adjustment,
        })
    }

    /// Assess child-specific naturalness
    async fn assess_child_naturalness(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
    ) -> Result<ChildNaturalnessResult, EvaluationError> {
        // Assess age appropriateness of voice characteristics
        let age_appropriateness = self.assess_age_appropriateness(generated_audio).await?;

        // Assess voice quality naturalness
        let voice_quality = self
            .assess_voice_quality_naturalness(generated_audio, reference_audio)
            .await?;

        // Assess prosodic naturalness for age
        let prosodic_naturalness = self.assess_prosodic_naturalness(generated_audio).await?;

        // Assess emotional expression appropriateness
        let emotional_appropriateness = self
            .assess_emotional_appropriateness(generated_audio)
            .await?;

        // Assess developmental stage alignment
        let developmental_alignment = self.assess_developmental_alignment(generated_audio).await?;

        // Calculate overall naturalness
        let overall_naturalness = (age_appropriateness * 0.25
            + voice_quality * 0.25
            + prosodic_naturalness * 0.2
            + emotional_appropriateness * 0.15
            + developmental_alignment * 0.15)
            .min(1.0);

        Ok(ChildNaturalnessResult {
            age_appropriateness,
            voice_quality,
            prosodic_naturalness,
            emotional_appropriateness,
            developmental_alignment,
            overall_naturalness,
        })
    }

    /// Assess developmental milestones
    async fn assess_developmental_milestones(
        &self,
        generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<DevelopmentalAssessment, EvaluationError> {
        let mut milestone_achievements = HashMap::new();

        // Assess phoneme acquisition milestone
        let phoneme_score = self
            .assess_phoneme_acquisition_milestone(generated_audio, target_text)
            .await?;
        milestone_achievements.insert(DevelopmentalMilestone::PhonemeAcquisition, phoneme_score);

        // Assess other milestones (simplified for demonstration)
        milestone_achievements.insert(DevelopmentalMilestone::GrammaticalDevelopment, 0.8);
        milestone_achievements.insert(DevelopmentalMilestone::VocabularyGrowth, 0.75);
        milestone_achievements.insert(DevelopmentalMilestone::ProsodyDevelopment, 0.7);
        milestone_achievements.insert(DevelopmentalMilestone::FluencyDevelopment, 0.8);
        milestone_achievements.insert(DevelopmentalMilestone::VoiceQuality, 0.85);

        // Identify development areas and strengths
        let development_areas = self.identify_development_areas(&milestone_achievements);
        let strengths = self.identify_strengths(&milestone_achievements);

        // Calculate overall developmental score
        let developmental_score =
            milestone_achievements.values().sum::<f32>() / milestone_achievements.len() as f32;

        // Generate recommendations
        let recommendations = self.generate_developmental_recommendations(&milestone_achievements);

        Ok(DevelopmentalAssessment {
            current_stage: self.config.target_age_group,
            milestone_achievements,
            development_areas,
            strengths,
            developmental_score,
            recommendations,
        })
    }

    /// Assess educational progress
    async fn assess_educational_progress(
        &self,
        generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<EducationalProgressResult, EvaluationError> {
        // Assess reading level (simplified)
        let reading_level = self.assess_reading_level(target_text);

        // Assess vocabulary level
        let vocabulary_level = self.assess_vocabulary_level(target_text);

        // Assess comprehension level
        let comprehension_level = self
            .assess_comprehension_level(generated_audio, target_text)
            .await?;

        // Assess communication effectiveness
        let communication_effectiveness = 0.8; // Simplified

        // Assess learning objective alignment
        let learning_alignment = 0.75; // Simplified

        // Calculate educational value
        let educational_value = (reading_level
            + vocabulary_level
            + comprehension_level
            + communication_effectiveness
            + learning_alignment)
            / 5.0;

        Ok(EducationalProgressResult {
            reading_level,
            vocabulary_level,
            comprehension_level,
            communication_effectiveness,
            learning_alignment,
            educational_value,
        })
    }

    /// Calculate raw intelligibility score
    async fn calculate_raw_intelligibility(
        &self,
        generated_audio: &AudioBuffer,
        _reference_audio: Option<&AudioBuffer>,
    ) -> Result<f32, EvaluationError> {
        // Simplified intelligibility calculation based on audio characteristics
        let samples = generated_audio.samples();
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();

        // Basic intelligibility estimate based on signal characteristics
        let signal_clarity = (rms * 2.0).min(1.0);
        let noise_level = self.estimate_noise_level(samples);
        let snr_score = ((rms / noise_level.max(0.001)) / 10.0).min(1.0);

        Ok((signal_clarity + snr_score) / 2.0)
    }

    /// Calculate age-specific adjustment factor
    fn calculate_age_adjustment(
        &self,
        raw_score: f32,
        characteristics: &SpeechCharacteristics,
    ) -> f32 {
        // Adjust expectations based on age group
        let expectation_factor = characteristics.articulation_accuracy;

        // If raw score exceeds age expectations, bonus
        if raw_score > expectation_factor {
            1.0 + (raw_score - expectation_factor) * 0.5
        } else {
            // If below expectations, more lenient scoring
            1.0 - (expectation_factor - raw_score) * 0.3
        }
    }

    /// Assess phoneme-level intelligibility
    async fn assess_phoneme_intelligibility(
        &self,
        _generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<HashMap<String, f32>, EvaluationError> {
        let mut phoneme_scores = HashMap::new();

        if let Some(text) = target_text {
            // Simplified phoneme assessment
            let default_acquisition = PhonemeAcquisition::default();
            let phoneme_acquisition = self
                .config
                .phoneme_acquisition
                .as_ref()
                .unwrap_or(&default_acquisition);

            // Assess early phonemes
            for phoneme in &phoneme_acquisition.early_phonemes {
                if text.contains(phoneme) {
                    phoneme_scores.insert(phoneme.clone(), 0.9); // High score for early phonemes
                }
            }

            // Assess middle phonemes
            for phoneme in &phoneme_acquisition.middle_phonemes {
                if text.contains(phoneme) {
                    phoneme_scores.insert(phoneme.clone(), 0.8); // Good score for middle phonemes
                }
            }

            // Assess late phonemes
            for phoneme in &phoneme_acquisition.late_phonemes {
                if text.contains(phoneme) {
                    phoneme_scores.insert(phoneme.clone(), 0.7); // Lower score for late phonemes
                }
            }
        }

        Ok(phoneme_scores)
    }

    /// Assess word-level intelligibility
    async fn assess_word_intelligibility(
        &self,
        generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        if let Some(text) = target_text {
            let word_count = text.split_whitespace().count();
            let complexity_factor = (word_count as f32).log10() / 2.0; // Logarithmic complexity

            // Base intelligibility adjusted for word complexity
            let base_score = self
                .calculate_raw_intelligibility(generated_audio, None)
                .await?;
            Ok((base_score * (1.0 - complexity_factor * 0.1)).max(0.0))
        } else {
            self.calculate_raw_intelligibility(generated_audio, None)
                .await
        }
    }

    /// Assess sentence-level intelligibility
    async fn assess_sentence_intelligibility(
        &self,
        generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        if let Some(text) = target_text {
            let sentence_count =
                text.matches('.').count() + text.matches('!').count() + text.matches('?').count();
            let sentence_factor = if sentence_count > 0 {
                (sentence_count as f32).log10() / 3.0
            } else {
                0.0
            };

            // Base intelligibility adjusted for sentence complexity
            let base_score = self
                .calculate_raw_intelligibility(generated_audio, None)
                .await?;
            Ok((base_score * (1.0 - sentence_factor * 0.05)).max(0.0))
        } else {
            self.calculate_raw_intelligibility(generated_audio, None)
                .await
        }
    }

    /// Calculate context effect on intelligibility
    fn calculate_context_effect(&self, target_text: Option<&str>) -> f32 {
        if let Some(text) = target_text {
            // Higher context (more words, familiar vocabulary) improves intelligibility
            let word_count = text.split_whitespace().count();
            let context_boost = (word_count as f32 / 20.0).min(0.2); // Up to 20% boost for context
            0.8 + context_boost // Base context effect + boost
        } else {
            0.5 // No context available
        }
    }

    /// Assess age appropriateness of voice characteristics
    async fn assess_age_appropriateness(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();

        // Estimate fundamental frequency
        let estimated_f0 =
            self.estimate_fundamental_frequency(samples, generated_audio.sample_rate() as f32)?;

        // Get expected F0 range for age group
        let expected_range = self
            .config
            .target_age_group
            .speech_characteristics()
            .fundamental_frequency_range;

        // Calculate appropriateness based on F0 match
        let f0_appropriateness =
            if estimated_f0 >= expected_range.0 && estimated_f0 <= expected_range.1 {
                1.0
            } else {
                let distance = if estimated_f0 < expected_range.0 {
                    expected_range.0 - estimated_f0
                } else {
                    estimated_f0 - expected_range.1
                };
                (1.0 - distance / 100.0).max(0.0) // Penalty for being outside range
            };

        Ok(f0_appropriateness)
    }

    /// Estimate fundamental frequency
    fn estimate_fundamental_frequency(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> Result<f32, EvaluationError> {
        if samples.len() < 512 {
            return Ok(200.0); // Default F0 for children
        }

        let frame_size = 512;
        let frame = &samples[0..frame_size];

        // Simple autocorrelation-based F0 estimation
        let min_period = (sample_rate / 500.0) as usize; // Max F0
        let max_period = (sample_rate / 100.0) as usize; // Min F0

        if max_period >= frame.len() {
            return Ok(200.0);
        }

        let mut max_correlation = 0.0;
        let mut best_period = min_period;

        for period in min_period..=max_period.min(frame.len() - 1) {
            let mut correlation = 0.0;
            for i in 0..(frame.len() - period) {
                correlation += frame[i] * frame[i + period];
            }

            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }

        Ok(sample_rate / best_period as f32)
    }

    /// Assess voice quality naturalness
    async fn assess_voice_quality_naturalness(
        &self,
        generated_audio: &AudioBuffer,
        _reference_audio: Option<&AudioBuffer>,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();

        // Assess smoothness (lack of artifacts)
        let smoothness = self.assess_signal_smoothness(samples);

        // Assess harmonic content
        let harmonic_quality = self.assess_harmonic_quality(samples);

        // Assess consistency
        let consistency = self.assess_signal_consistency(samples);

        Ok((smoothness + harmonic_quality + consistency) / 3.0)
    }

    /// Assess prosodic naturalness for children
    async fn assess_prosodic_naturalness(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();

        // Children's prosody is typically more variable and expressive
        let rhythm_variability = self.assess_rhythm_variability(samples);
        let intonation_expressiveness = self.assess_intonation_expressiveness(samples);
        let timing_naturalness = self.assess_timing_naturalness(samples);

        // Children's prosody should be more expressive than adult speech
        let age_adjusted_score =
            (rhythm_variability + intonation_expressiveness + timing_naturalness) / 3.0;

        Ok(age_adjusted_score)
    }

    /// Assess emotional expression appropriateness
    async fn assess_emotional_appropriateness(
        &self,
        _generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Children's speech should have appropriate emotional range
        // This would involve more complex analysis in practice
        Ok(0.8) // Simplified score
    }

    /// Assess developmental stage alignment
    async fn assess_developmental_alignment(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let characteristics = self.config.target_age_group.speech_characteristics();

        // Assess multiple aspects of developmental alignment
        let f0_alignment = self.assess_age_appropriateness(generated_audio).await?;
        let complexity_alignment = characteristics.grammatical_complexity;
        let fluency_alignment = characteristics.fluency_expectations;

        Ok((f0_alignment + complexity_alignment + fluency_alignment) / 3.0)
    }

    /// Helper functions for audio analysis
    fn estimate_noise_level(&self, samples: &[f32]) -> f32 {
        // Estimate noise floor as minimum energy in quiet segments
        let mut min_energy = f32::INFINITY;
        let window_size = 512;

        for window in samples.chunks(window_size) {
            let energy = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;
            if energy < min_energy {
                min_energy = energy;
            }
        }

        min_energy.sqrt()
    }

    fn assess_signal_smoothness(&self, samples: &[f32]) -> f32 {
        // Assess signal smoothness by measuring derivative variations
        let mut variations = 0.0;
        for window in samples.windows(2) {
            variations += (window[1] - window[0]).abs();
        }

        let avg_variation = variations / (samples.len() - 1) as f32;
        (1.0 - avg_variation.min(1.0)).max(0.0)
    }

    fn assess_harmonic_quality(&self, _samples: &[f32]) -> f32 {
        // Simplified harmonic quality assessment
        0.8
    }

    fn assess_signal_consistency(&self, samples: &[f32]) -> f32 {
        // Assess energy consistency across the signal
        let window_size = 512;
        let mut energies = Vec::new();

        for window in samples.chunks(window_size) {
            let energy = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;
            energies.push(energy.sqrt());
        }

        if energies.is_empty() {
            return 0.0;
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        (1.0 - variance.sqrt().min(1.0)).max(0.0)
    }

    fn assess_rhythm_variability(&self, _samples: &[f32]) -> f32 {
        // Simplified rhythm variability assessment
        0.75
    }

    fn assess_intonation_expressiveness(&self, _samples: &[f32]) -> f32 {
        // Simplified intonation expressiveness assessment
        0.8
    }

    fn assess_timing_naturalness(&self, _samples: &[f32]) -> f32 {
        // Simplified timing naturalness assessment
        0.85
    }

    /// Assessment helper functions
    async fn assess_phoneme_acquisition_milestone(
        &self,
        _generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        if let Some(text) = target_text {
            let default_acquisition = PhonemeAcquisition::default();
            let phoneme_acquisition = self
                .config
                .phoneme_acquisition
                .as_ref()
                .unwrap_or(&default_acquisition);

            // Count phonemes by acquisition stage
            let mut early_count = 0;
            let mut middle_count = 0;
            let mut late_count = 0;

            for phoneme in &phoneme_acquisition.early_phonemes {
                if text.contains(phoneme) {
                    early_count += 1;
                }
            }
            for phoneme in &phoneme_acquisition.middle_phonemes {
                if text.contains(phoneme) {
                    middle_count += 1;
                }
            }
            for phoneme in &phoneme_acquisition.late_phonemes {
                if text.contains(phoneme) {
                    late_count += 1;
                }
            }

            // Calculate age-appropriate phoneme score
            let total_phonemes = early_count + middle_count + late_count;
            if total_phonemes == 0 {
                return Ok(0.8); // Default score
            }

            let age_characteristics = self.config.target_age_group.speech_characteristics();
            let expected_accuracy = age_characteristics.articulation_accuracy;

            Ok(expected_accuracy)
        } else {
            Ok(0.7) // Default score without text
        }
    }

    fn identify_development_areas(
        &self,
        achievements: &HashMap<DevelopmentalMilestone, f32>,
    ) -> Vec<String> {
        let mut areas = Vec::new();

        for (milestone, &score) in achievements {
            if score < 0.7 {
                areas.push(format!("{:?}", milestone));
            }
        }

        areas
    }

    fn identify_strengths(
        &self,
        achievements: &HashMap<DevelopmentalMilestone, f32>,
    ) -> Vec<String> {
        let mut strengths = Vec::new();

        for (milestone, &score) in achievements {
            if score > 0.85 {
                strengths.push(format!("{:?}", milestone));
            }
        }

        strengths
    }

    fn generate_developmental_recommendations(
        &self,
        _achievements: &HashMap<DevelopmentalMilestone, f32>,
    ) -> Vec<String> {
        // Generate specific recommendations based on milestone achievements
        vec![
            "Continue practicing phoneme articulation".to_string(),
            "Focus on prosodic expression development".to_string(),
            "Encourage vocabulary expansion".to_string(),
        ]
    }

    fn assess_reading_level(&self, target_text: Option<&str>) -> f32 {
        if let Some(text) = target_text {
            let word_count = text.split_whitespace().count();
            let sentence_count = text.matches('.').count().max(1);
            let avg_words_per_sentence = word_count as f32 / sentence_count as f32;

            // Simple reading level estimation
            (avg_words_per_sentence / 15.0).min(1.0)
        } else {
            0.5
        }
    }

    fn assess_vocabulary_level(&self, target_text: Option<&str>) -> f32 {
        if let Some(text) = target_text {
            let word_count = text.split_whitespace().count();
            let unique_words = text
                .split_whitespace()
                .collect::<std::collections::HashSet<_>>()
                .len();
            let vocabulary_diversity = unique_words as f32 / word_count as f32;

            vocabulary_diversity.min(1.0)
        } else {
            0.5
        }
    }

    async fn assess_comprehension_level(
        &self,
        _generated_audio: &AudioBuffer,
        _target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        // Simplified comprehension assessment
        Ok(0.8)
    }

    async fn evaluate_voice_appropriateness(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Comprehensive voice appropriateness assessment
        let age_appropriateness = self.assess_age_appropriateness(generated_audio).await?;
        let voice_quality = self
            .assess_voice_quality_naturalness(generated_audio, None)
            .await?;

        Ok((age_appropriateness + voice_quality) / 2.0)
    }

    async fn calculate_age_compatibility(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let characteristics = self.config.target_age_group.speech_characteristics();

        // Assess multiple compatibility factors
        let f0_compatibility = self.assess_age_appropriateness(generated_audio).await?;
        let voice_stability = characteristics.voice_quality_stability;

        Ok((f0_compatibility + voice_stability) / 2.0)
    }

    async fn calculate_communication_effectiveness(
        &self,
        intelligibility: &ChildIntelligibilityResult,
        naturalness: &ChildNaturalnessResult,
    ) -> Result<f32, EvaluationError> {
        // Communication effectiveness combines intelligibility and naturalness
        let effectiveness = (intelligibility.age_adjusted_intelligibility * 0.6
            + naturalness.overall_naturalness * 0.4)
            .min(1.0);

        Ok(effectiveness)
    }

    fn generate_recommendations(
        &self,
        intelligibility: &ChildIntelligibilityResult,
        naturalness: &ChildNaturalnessResult,
        developmental: Option<&DevelopmentalAssessment>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Intelligibility recommendations
        if intelligibility.age_adjusted_intelligibility < 0.7 {
            recommendations.push(
                "Focus on improving articulation clarity for better intelligibility".to_string(),
            );
        }

        // Naturalness recommendations
        if naturalness.age_appropriateness < 0.7 {
            recommendations
                .push("Adjust voice characteristics to be more age-appropriate".to_string());
        }

        // Developmental recommendations
        if let Some(dev) = developmental {
            if dev.developmental_score < 0.7 {
                recommendations
                    .push("Address identified developmental areas for improvement".to_string());
            }
        }

        // General recommendations
        recommendations.push("Continue practicing with age-appropriate content".to_string());

        recommendations
    }

    async fn calculate_overall_score(
        &self,
        intelligibility: &ChildIntelligibilityResult,
        naturalness: &ChildNaturalnessResult,
        voice_appropriateness: f32,
        age_compatibility: f32,
        communication_effectiveness: f32,
    ) -> Result<f32, EvaluationError> {
        // Weighted scoring for children's speech evaluation
        let overall_score = (intelligibility.age_adjusted_intelligibility * 0.3
            + naturalness.overall_naturalness * 0.25
            + voice_appropriateness * 0.2
            + age_compatibility * 0.15
            + communication_effectiveness * 0.1)
            .min(1.0);

        Ok(overall_score)
    }

    fn calculate_evaluation_confidence(
        &self,
        _generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        target_text: Option<&str>,
    ) -> f32 {
        let mut confidence = 0.6_f32; // Base confidence

        // Reference audio increases confidence
        if reference_audio.is_some() {
            confidence += 0.2;
        }

        // Target text increases confidence
        if target_text.is_some() {
            confidence += 0.15;
        }

        // Age-specific configuration increases confidence
        confidence += 0.05;

        confidence.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_children_evaluator_creation() {
        let evaluator = ChildrenSpeechEvaluator::new().await.unwrap();
        assert_eq!(evaluator.config.target_age_group, AgeGroup::EarlyElementary);
    }

    #[tokio::test]
    async fn test_children_evaluation() {
        let evaluator = ChildrenSpeechEvaluator::new().await.unwrap();

        // Create test audio
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 250.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = evaluator
            .evaluate_children_speech(&audio, None, Some("Hello world"))
            .await
            .unwrap();

        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(result.intelligibility.age_adjusted_intelligibility >= 0.0);
        assert!(result.naturalness.overall_naturalness >= 0.0);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_age_group_characteristics() {
        let early_childhood = AgeGroup::EarlyChildhood;
        let characteristics = early_childhood.speech_characteristics();

        assert_eq!(early_childhood.age_range(), (2, 4));
        assert!(characteristics.fundamental_frequency_range.0 > 200.0);
        assert!(characteristics.articulation_accuracy < 0.8);
    }

    #[test]
    fn test_listener_familiarity() {
        assert!(ListenerFamiliarity::VeryFamiliar.intelligibility_adjustment() > 1.0);
        assert!(ListenerFamiliarity::Unfamiliar.intelligibility_adjustment() < 1.0);
    }

    #[test]
    fn test_phoneme_acquisition_default() {
        let acquisition = PhonemeAcquisition::default();
        assert!(!acquisition.early_phonemes.is_empty());
        assert!(!acquisition.middle_phonemes.is_empty());
        assert!(!acquisition.late_phonemes.is_empty());
    }

    #[test]
    fn test_config_default() {
        let config = ChildrenEvaluationConfig::default();
        assert_eq!(config.target_age_group, AgeGroup::EarlyElementary);
        assert!(config.age_adjusted_scoring);
        assert!(config.assess_developmental_milestones);
    }

    #[tokio::test]
    async fn test_fundamental_frequency_estimation() {
        let evaluator = ChildrenSpeechEvaluator::new().await.unwrap();

        // Create test signal with known frequency
        let sample_rate = 16000.0;
        let frequency = 250.0;
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin())
            .collect();

        let estimated_f0 = evaluator
            .estimate_fundamental_frequency(&samples, sample_rate)
            .unwrap();

        // Should be reasonably close to the input frequency
        assert!((estimated_f0 - frequency).abs() < 50.0);
    }

    #[tokio::test]
    async fn test_age_appropriateness_assessment() {
        let mut config = ChildrenEvaluationConfig::default();
        config.target_age_group = AgeGroup::EarlyChildhood;
        let evaluator = ChildrenSpeechEvaluator::with_config(config).await.unwrap();

        // Create audio with child-appropriate F0 (300 Hz)
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let appropriateness = evaluator.assess_age_appropriateness(&audio).await.unwrap();
        assert!(appropriateness > 0.5); // Should be reasonably appropriate
    }
}
