//! Elderly and Pathological Speech Assessment
//!
//! This module provides specialized evaluation metrics designed for elderly speakers
//! and pathological speech conditions. It includes age-related voice change analysis,
//! communication disorder assessment, and assistive technology evaluation.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::traits::*;
use crate::{AudioBuffer, EvaluationError, LanguageCode};

/// Age categories for elderly speech assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElderlyAgeGroup {
    /// Young elderly (65-74 years)
    YoungElderly,
    /// Old elderly (75-84 years)
    OldElderly,
    /// Oldest elderly (85+ years)
    OldestElderly,
}

impl ElderlyAgeGroup {
    /// Get typical age range for the group
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            ElderlyAgeGroup::YoungElderly => (65, 74),
            ElderlyAgeGroup::OldElderly => (75, 84),
            ElderlyAgeGroup::OldestElderly => (85, 120),
        }
    }

    /// Get expected speech changes for age group
    pub fn expected_changes(&self) -> AgeRelatedChanges {
        match self {
            ElderlyAgeGroup::YoungElderly => AgeRelatedChanges {
                fundamental_frequency_change: 0.1,
                voice_tremor_likelihood: 0.2,
                articulation_precision_decline: 0.1,
                speaking_rate_change: 0.05,
                volume_control_issues: 0.1,
                breath_support_decline: 0.15,
                cognitive_load_sensitivity: 0.1,
            },
            ElderlyAgeGroup::OldElderly => AgeRelatedChanges {
                fundamental_frequency_change: 0.2,
                voice_tremor_likelihood: 0.4,
                articulation_precision_decline: 0.2,
                speaking_rate_change: 0.15,
                volume_control_issues: 0.2,
                breath_support_decline: 0.3,
                cognitive_load_sensitivity: 0.25,
            },
            ElderlyAgeGroup::OldestElderly => AgeRelatedChanges {
                fundamental_frequency_change: 0.3,
                voice_tremor_likelihood: 0.6,
                articulation_precision_decline: 0.3,
                speaking_rate_change: 0.25,
                volume_control_issues: 0.35,
                breath_support_decline: 0.45,
                cognitive_load_sensitivity: 0.4,
            },
        }
    }
}

/// Age-related speech changes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgeRelatedChanges {
    /// Fundamental frequency change factor (0.0-1.0)
    pub fundamental_frequency_change: f32,
    /// Likelihood of voice tremor (0.0-1.0)
    pub voice_tremor_likelihood: f32,
    /// Articulation precision decline (0.0-1.0)
    pub articulation_precision_decline: f32,
    /// Speaking rate change factor (0.0-1.0)
    pub speaking_rate_change: f32,
    /// Volume control issues (0.0-1.0)
    pub volume_control_issues: f32,
    /// Breath support decline (0.0-1.0)
    pub breath_support_decline: f32,
    /// Cognitive load sensitivity (0.0-1.0)
    pub cognitive_load_sensitivity: f32,
}

/// Types of pathological speech conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathologicalCondition {
    /// Parkinson's disease speech characteristics
    Parkinsons,
    /// Stroke-related speech impairments
    Stroke,
    /// Alzheimer's and dementia-related changes
    Dementia,
    /// Vocal cord paralysis or weakness
    VocalCordParalysis,
    /// Laryngeal pathologies
    LaryngealPathology,
    /// Respiratory conditions affecting speech
    RespiratoryConditions,
    /// Hearing loss affecting speech production
    HearingLoss,
    /// General age-related voice changes
    Presbyphonia,
    /// Motor speech disorders
    MotorSpeechDisorder,
    /// Cognitive-communication disorders
    CognitiveCommunicationDisorder,
}

impl PathologicalCondition {
    /// Get characteristic features of the condition
    pub fn characteristic_features(&self) -> PathologicalFeatures {
        match self {
            PathologicalCondition::Parkinsons => PathologicalFeatures {
                reduced_loudness: 0.8,
                monotone_speech: 0.7,
                rapid_speech_rate: 0.6,
                voice_tremor: 0.5,
                articulation_imprecision: 0.6,
                reduced_stress: 0.7,
                breathy_voice: 0.4,
                communication_impact: 0.6,
            },
            PathologicalCondition::Stroke => PathologicalFeatures {
                reduced_loudness: 0.5,
                monotone_speech: 0.6,
                rapid_speech_rate: 0.3,
                voice_tremor: 0.2,
                articulation_imprecision: 0.8,
                reduced_stress: 0.6,
                breathy_voice: 0.3,
                communication_impact: 0.7,
            },
            PathologicalCondition::Dementia => PathologicalFeatures {
                reduced_loudness: 0.4,
                monotone_speech: 0.5,
                rapid_speech_rate: 0.2,
                voice_tremor: 0.3,
                articulation_imprecision: 0.5,
                reduced_stress: 0.6,
                breathy_voice: 0.3,
                communication_impact: 0.8,
            },
            PathologicalCondition::VocalCordParalysis => PathologicalFeatures {
                reduced_loudness: 0.9,
                monotone_speech: 0.6,
                rapid_speech_rate: 0.1,
                voice_tremor: 0.1,
                articulation_imprecision: 0.3,
                reduced_stress: 0.5,
                breathy_voice: 0.9,
                communication_impact: 0.7,
            },
            PathologicalCondition::LaryngealPathology => PathologicalFeatures {
                reduced_loudness: 0.7,
                monotone_speech: 0.5,
                rapid_speech_rate: 0.2,
                voice_tremor: 0.4,
                articulation_imprecision: 0.3,
                reduced_stress: 0.4,
                breathy_voice: 0.8,
                communication_impact: 0.6,
            },
            PathologicalCondition::RespiratoryConditions => PathologicalFeatures {
                reduced_loudness: 0.6,
                monotone_speech: 0.4,
                rapid_speech_rate: 0.1,
                voice_tremor: 0.2,
                articulation_imprecision: 0.4,
                reduced_stress: 0.5,
                breathy_voice: 0.6,
                communication_impact: 0.5,
            },
            PathologicalCondition::HearingLoss => PathologicalFeatures {
                reduced_loudness: 0.3,
                monotone_speech: 0.6,
                rapid_speech_rate: 0.2,
                voice_tremor: 0.1,
                articulation_imprecision: 0.5,
                reduced_stress: 0.5,
                breathy_voice: 0.2,
                communication_impact: 0.6,
            },
            PathologicalCondition::Presbyphonia => PathologicalFeatures {
                reduced_loudness: 0.5,
                monotone_speech: 0.4,
                rapid_speech_rate: 0.2,
                voice_tremor: 0.6,
                articulation_imprecision: 0.3,
                reduced_stress: 0.4,
                breathy_voice: 0.7,
                communication_impact: 0.4,
            },
            PathologicalCondition::MotorSpeechDisorder => PathologicalFeatures {
                reduced_loudness: 0.6,
                monotone_speech: 0.7,
                rapid_speech_rate: 0.4,
                voice_tremor: 0.3,
                articulation_imprecision: 0.8,
                reduced_stress: 0.7,
                breathy_voice: 0.4,
                communication_impact: 0.8,
            },
            PathologicalCondition::CognitiveCommunicationDisorder => PathologicalFeatures {
                reduced_loudness: 0.3,
                monotone_speech: 0.5,
                rapid_speech_rate: 0.3,
                voice_tremor: 0.2,
                articulation_imprecision: 0.4,
                reduced_stress: 0.6,
                breathy_voice: 0.3,
                communication_impact: 0.9,
            },
        }
    }
}

/// Pathological speech features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathologicalFeatures {
    /// Reduced loudness severity (0.0-1.0)
    pub reduced_loudness: f32,
    /// Monotone speech severity (0.0-1.0)
    pub monotone_speech: f32,
    /// Rapid speech rate issues (0.0-1.0)
    pub rapid_speech_rate: f32,
    /// Voice tremor presence (0.0-1.0)
    pub voice_tremor: f32,
    /// Articulation imprecision (0.0-1.0)
    pub articulation_imprecision: f32,
    /// Reduced stress patterns (0.0-1.0)
    pub reduced_stress: f32,
    /// Breathy voice quality (0.0-1.0)
    pub breathy_voice: f32,
    /// Overall communication impact (0.0-1.0)
    pub communication_impact: f32,
}

/// Severity levels for speech impairments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Normal or minimal impairment
    Normal,
    /// Mild impairment
    Mild,
    /// Moderate impairment
    Moderate,
    /// Severe impairment
    Severe,
    /// Profound impairment
    Profound,
}

impl SeverityLevel {
    /// Convert severity score to level
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s >= 0.9 => SeverityLevel::Normal,
            s if s >= 0.7 => SeverityLevel::Mild,
            s if s >= 0.5 => SeverityLevel::Moderate,
            s if s >= 0.3 => SeverityLevel::Severe,
            _ => SeverityLevel::Profound,
        }
    }

    /// Get score range for severity level
    pub fn score_range(&self) -> (f32, f32) {
        match self {
            SeverityLevel::Normal => (0.9, 1.0),
            SeverityLevel::Mild => (0.7, 0.9),
            SeverityLevel::Moderate => (0.5, 0.7),
            SeverityLevel::Severe => (0.3, 0.5),
            SeverityLevel::Profound => (0.0, 0.3),
        }
    }
}

/// Communication effectiveness assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunicationEffectivenessResult {
    /// Overall communication effectiveness score
    pub overall_effectiveness: f32,
    /// Intelligibility in quiet conditions
    pub quiet_intelligibility: f32,
    /// Intelligibility in noise
    pub noise_intelligibility: f32,
    /// Listener burden (effort required to understand)
    pub listener_burden: f32,
    /// Communication efficiency
    pub communication_efficiency: f32,
    /// Functional communication level
    pub functional_level: SeverityLevel,
    /// Specific communication strengths
    pub strengths: Vec<String>,
    /// Areas needing support
    pub support_areas: Vec<String>,
}

/// Assistive technology evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistiveTechnologyResult {
    /// Voice amplification effectiveness
    pub amplification_effectiveness: f32,
    /// Speech clarity enhancement
    pub clarity_enhancement: f32,
    /// Technology adaptation score
    pub technology_adaptation: f32,
    /// User acceptance likelihood
    pub user_acceptance: f32,
    /// Recommended assistive technologies
    pub recommended_technologies: Vec<String>,
    /// Technology configuration suggestions
    pub configuration_suggestions: Vec<String>,
}

/// Clinical assessment metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClinicalAssessmentResult {
    /// Dysarthria severity assessment
    pub dysarthria_severity: Option<SeverityLevel>,
    /// Voice quality assessment
    pub voice_quality: VoiceQualityMetrics,
    /// Respiratory support assessment
    pub respiratory_support: f32,
    /// Motor speech control
    pub motor_speech_control: f32,
    /// Cognitive-linguistic function
    pub cognitive_linguistic_function: f32,
    /// Clinical recommendations
    pub clinical_recommendations: Vec<String>,
    /// Therapy goals suggestions
    pub therapy_goals: Vec<String>,
}

/// Voice quality metrics for clinical assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Fundamental frequency stability
    pub f0_stability: f32,
    /// Jitter (frequency perturbation)
    pub jitter: f32,
    /// Shimmer (amplitude perturbation)
    pub shimmer: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_noise_ratio: f32,
    /// Voice breaks or interruptions
    pub voice_breaks: f32,
    /// Breathiness rating
    pub breathiness: f32,
    /// Roughness rating
    pub roughness: f32,
    /// Strain rating
    pub strain: f32,
}

/// Configuration for elderly and pathological speech evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElderlyPathologicalConfig {
    /// Target age group (if elderly)
    pub target_age_group: Option<ElderlyAgeGroup>,
    /// Known pathological conditions
    pub pathological_conditions: Vec<PathologicalCondition>,
    /// Language being evaluated
    pub language: LanguageCode,
    /// Whether to perform clinical assessment
    pub perform_clinical_assessment: bool,
    /// Whether to evaluate assistive technology needs
    pub evaluate_assistive_technology: bool,
    /// Whether to assess communication effectiveness
    pub assess_communication_effectiveness: bool,
    /// Expected severity level (if known)
    pub expected_severity: Option<SeverityLevel>,
    /// Listener familiarity level
    pub listener_familiarity: ListenerFamiliarity,
    /// Communication context (clinical, home, etc.)
    pub communication_context: CommunicationContext,
    /// Whether to apply age-adjusted scoring
    pub age_adjusted_scoring: bool,
}

/// Listener familiarity with speaker's condition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ListenerFamiliarity {
    /// Very familiar (family, caregivers)
    VeryFamiliar,
    /// Familiar (healthcare workers, friends)
    Familiar,
    /// Somewhat familiar (occasional contact)
    SomewhatFamiliar,
    /// Unfamiliar (strangers, general public)
    Unfamiliar,
}

impl ListenerFamiliarity {
    /// Get intelligibility adjustment factor
    pub fn intelligibility_adjustment(&self) -> f32 {
        match self {
            ListenerFamiliarity::VeryFamiliar => 1.3,
            ListenerFamiliarity::Familiar => 1.15,
            ListenerFamiliarity::SomewhatFamiliar => 1.0,
            ListenerFamiliarity::Unfamiliar => 0.8,
        }
    }
}

/// Communication context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunicationContext {
    /// Clinical or therapeutic setting
    Clinical,
    /// Home environment
    Home,
    /// Community or public setting
    Community,
    /// Workplace
    Workplace,
    /// Educational setting
    Educational,
    /// Telehealth or remote
    Telehealth,
}

impl CommunicationContext {
    /// Get context-specific expectations
    pub fn expectations(&self) -> ContextExpectations {
        match self {
            CommunicationContext::Clinical => ContextExpectations {
                intelligibility_requirement: 0.9,
                listener_patience: 0.9,
                time_pressure: 0.3,
                background_noise: 0.1,
                communication_importance: 0.9,
            },
            CommunicationContext::Home => ContextExpectations {
                intelligibility_requirement: 0.8,
                listener_patience: 0.8,
                time_pressure: 0.2,
                background_noise: 0.3,
                communication_importance: 0.8,
            },
            CommunicationContext::Community => ContextExpectations {
                intelligibility_requirement: 0.7,
                listener_patience: 0.4,
                time_pressure: 0.7,
                background_noise: 0.6,
                communication_importance: 0.7,
            },
            CommunicationContext::Workplace => ContextExpectations {
                intelligibility_requirement: 0.8,
                listener_patience: 0.5,
                time_pressure: 0.8,
                background_noise: 0.4,
                communication_importance: 0.9,
            },
            CommunicationContext::Educational => ContextExpectations {
                intelligibility_requirement: 0.85,
                listener_patience: 0.7,
                time_pressure: 0.5,
                background_noise: 0.5,
                communication_importance: 0.9,
            },
            CommunicationContext::Telehealth => ContextExpectations {
                intelligibility_requirement: 0.9,
                listener_patience: 0.8,
                time_pressure: 0.4,
                background_noise: 0.2,
                communication_importance: 0.95,
            },
        }
    }
}

/// Context-specific expectations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextExpectations {
    /// Required intelligibility level (0.0-1.0)
    pub intelligibility_requirement: f32,
    /// Listener patience level (0.0-1.0)
    pub listener_patience: f32,
    /// Time pressure level (0.0-1.0)
    pub time_pressure: f32,
    /// Background noise level (0.0-1.0)
    pub background_noise: f32,
    /// Communication importance (0.0-1.0)
    pub communication_importance: f32,
}

impl Default for ElderlyPathologicalConfig {
    fn default() -> Self {
        Self {
            target_age_group: Some(ElderlyAgeGroup::YoungElderly),
            pathological_conditions: vec![],
            language: LanguageCode::EnUs,
            perform_clinical_assessment: true,
            evaluate_assistive_technology: true,
            assess_communication_effectiveness: true,
            expected_severity: None,
            listener_familiarity: ListenerFamiliarity::Familiar,
            communication_context: CommunicationContext::Clinical,
            age_adjusted_scoring: true,
        }
    }
}

/// Comprehensive elderly and pathological speech evaluation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElderlyPathologicalResult {
    /// Overall evaluation score
    pub overall_score: f32,
    /// Communication effectiveness assessment
    pub communication_effectiveness: CommunicationEffectivenessResult,
    /// Clinical assessment results
    pub clinical_assessment: Option<ClinicalAssessmentResult>,
    /// Assistive technology evaluation
    pub assistive_technology: Option<AssistiveTechnologyResult>,
    /// Detected pathological features
    pub pathological_features: HashMap<PathologicalCondition, f32>,
    /// Age-related changes assessment
    pub age_related_changes: Option<AgeRelatedChanges>,
    /// Overall severity level
    pub severity_level: SeverityLevel,
    /// Context-adjusted scores
    pub context_adjusted_scores: HashMap<String, f32>,
    /// Recommendations for support
    pub support_recommendations: Vec<String>,
    /// Quality of life impact assessment
    pub quality_of_life_impact: f32,
    /// Confidence in the evaluation
    pub confidence: f32,
}

/// Elderly and pathological speech evaluator implementation
pub struct ElderlyPathologicalEvaluator {
    config: ElderlyPathologicalConfig,
}

impl ElderlyPathologicalEvaluator {
    /// Create a new elderly/pathological speech evaluator
    pub async fn new() -> Result<Self, EvaluationError> {
        Ok(Self {
            config: ElderlyPathologicalConfig::default(),
        })
    }

    /// Create evaluator with custom configuration
    pub async fn with_config(config: ElderlyPathologicalConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Update evaluator configuration
    pub fn set_config(&mut self, config: ElderlyPathologicalConfig) {
        self.config = config;
    }

    /// Evaluate elderly/pathological speech synthesis quality
    pub async fn evaluate_elderly_pathological_speech(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        target_text: Option<&str>,
    ) -> Result<ElderlyPathologicalResult, EvaluationError> {
        // Assess communication effectiveness
        let communication_effectiveness = self
            .assess_communication_effectiveness(generated_audio, reference_audio, target_text)
            .await?;

        // Perform clinical assessment if enabled
        let clinical_assessment = if self.config.perform_clinical_assessment {
            Some(
                self.perform_clinical_assessment(generated_audio, reference_audio)
                    .await?,
            )
        } else {
            None
        };

        // Evaluate assistive technology needs if enabled
        let assistive_technology = if self.config.evaluate_assistive_technology {
            Some(
                self.evaluate_assistive_technology_needs(generated_audio)
                    .await?,
            )
        } else {
            None
        };

        // Detect pathological features
        let pathological_features = self.detect_pathological_features(generated_audio).await?;

        // Assess age-related changes if applicable
        let age_related_changes = if let Some(age_group) = self.config.target_age_group {
            Some(
                self.assess_age_related_changes(generated_audio, age_group)
                    .await?,
            )
        } else {
            None
        };

        // Determine overall severity level
        let severity_level =
            self.determine_severity_level(&communication_effectiveness, &pathological_features);

        // Calculate context-adjusted scores
        let context_adjusted_scores = self.calculate_context_adjusted_scores(
            &communication_effectiveness,
            &pathological_features,
        );

        // Generate support recommendations
        let support_recommendations = self.generate_support_recommendations(
            &communication_effectiveness,
            &pathological_features,
            clinical_assessment.as_ref(),
        );

        // Assess quality of life impact
        let quality_of_life_impact = self.assess_quality_of_life_impact(
            &communication_effectiveness,
            &pathological_features,
            severity_level,
        );

        // Calculate overall score with appropriate adjustments
        let overall_score = self
            .calculate_overall_score(
                &communication_effectiveness,
                &pathological_features,
                severity_level,
                &context_adjusted_scores,
            )
            .await?;

        // Calculate evaluation confidence
        let confidence =
            self.calculate_evaluation_confidence(generated_audio, reference_audio, target_text);

        Ok(ElderlyPathologicalResult {
            overall_score,
            communication_effectiveness,
            clinical_assessment,
            assistive_technology,
            pathological_features,
            age_related_changes,
            severity_level,
            context_adjusted_scores,
            support_recommendations,
            quality_of_life_impact,
            confidence,
        })
    }

    /// Assess communication effectiveness
    async fn assess_communication_effectiveness(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        target_text: Option<&str>,
    ) -> Result<CommunicationEffectivenessResult, EvaluationError> {
        // Basic intelligibility assessment
        let quiet_intelligibility = self
            .assess_intelligibility_quiet(generated_audio, reference_audio)
            .await?;

        // Simulate noise conditions
        let noise_intelligibility = self
            .assess_intelligibility_noise(generated_audio, reference_audio)
            .await?;

        // Calculate listener burden
        let listener_burden = self
            .calculate_listener_burden(generated_audio, target_text)
            .await?;

        // Assess communication efficiency
        let communication_efficiency = self
            .assess_communication_efficiency(generated_audio, target_text)
            .await?;

        // Apply familiarity adjustment
        let familiarity_adjustment = self
            .config
            .listener_familiarity
            .intelligibility_adjustment();
        let adjusted_quiet = (quiet_intelligibility * familiarity_adjustment).min(1.0);
        let adjusted_noise = (noise_intelligibility * familiarity_adjustment).min(1.0);

        // Calculate overall effectiveness
        let overall_effectiveness = (adjusted_quiet * 0.4
            + adjusted_noise * 0.3
            + (1.0 - listener_burden) * 0.2
            + communication_efficiency * 0.1)
            .min(1.0);

        // Determine functional level
        let functional_level = SeverityLevel::from_score(overall_effectiveness);

        // Identify strengths and support areas
        let strengths = self.identify_communication_strengths(
            adjusted_quiet,
            adjusted_noise,
            communication_efficiency,
        );
        let support_areas = self.identify_support_areas(
            adjusted_quiet,
            adjusted_noise,
            listener_burden,
            communication_efficiency,
        );

        Ok(CommunicationEffectivenessResult {
            overall_effectiveness,
            quiet_intelligibility: adjusted_quiet,
            noise_intelligibility: adjusted_noise,
            listener_burden,
            communication_efficiency,
            functional_level,
            strengths,
            support_areas,
        })
    }

    /// Perform clinical assessment
    async fn perform_clinical_assessment(
        &self,
        generated_audio: &AudioBuffer,
        _reference_audio: Option<&AudioBuffer>,
    ) -> Result<ClinicalAssessmentResult, EvaluationError> {
        // Assess voice quality metrics
        let voice_quality = self.assess_voice_quality_metrics(generated_audio).await?;

        // Assess dysarthria severity
        let dysarthria_severity = self
            .assess_dysarthria_severity(generated_audio, &voice_quality)
            .await?;

        // Assess respiratory support
        let respiratory_support = self.assess_respiratory_support(generated_audio).await?;

        // Assess motor speech control
        let motor_speech_control = self.assess_motor_speech_control(generated_audio).await?;

        // Assess cognitive-linguistic function
        let cognitive_linguistic_function = self
            .assess_cognitive_linguistic_function(generated_audio)
            .await?;

        // Generate clinical recommendations
        let clinical_recommendations = self.generate_clinical_recommendations(
            &voice_quality,
            dysarthria_severity,
            respiratory_support,
            motor_speech_control,
        );

        // Generate therapy goals
        let therapy_goals =
            self.generate_therapy_goals(&voice_quality, dysarthria_severity, respiratory_support);

        Ok(ClinicalAssessmentResult {
            dysarthria_severity,
            voice_quality,
            respiratory_support,
            motor_speech_control,
            cognitive_linguistic_function,
            clinical_recommendations,
            therapy_goals,
        })
    }

    /// Evaluate assistive technology needs
    async fn evaluate_assistive_technology_needs(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<AssistiveTechnologyResult, EvaluationError> {
        // Assess need for voice amplification
        let amplification_effectiveness = self.assess_amplification_needs(generated_audio).await?;

        // Assess need for clarity enhancement
        let clarity_enhancement = self
            .assess_clarity_enhancement_needs(generated_audio)
            .await?;

        // Assess technology adaptation capability
        let technology_adaptation = self.assess_technology_adaptation(generated_audio).await?;

        // Estimate user acceptance likelihood
        let user_acceptance = self.estimate_user_acceptance(
            amplification_effectiveness,
            clarity_enhancement,
            technology_adaptation,
        );

        // Recommend specific technologies
        let recommended_technologies = self.recommend_assistive_technologies(
            amplification_effectiveness,
            clarity_enhancement,
            technology_adaptation,
        );

        // Suggest technology configurations
        let configuration_suggestions = self.suggest_technology_configurations(
            &recommended_technologies,
            amplification_effectiveness,
            clarity_enhancement,
        );

        Ok(AssistiveTechnologyResult {
            amplification_effectiveness,
            clarity_enhancement,
            technology_adaptation,
            user_acceptance,
            recommended_technologies,
            configuration_suggestions,
        })
    }

    /// Assess voice quality metrics
    async fn assess_voice_quality_metrics(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<VoiceQualityMetrics, EvaluationError> {
        let samples = generated_audio.samples();
        let sample_rate = generated_audio.sample_rate() as f32;

        // Assess F0 stability
        let f0_stability = self.calculate_f0_stability(samples, sample_rate)?;

        // Calculate jitter (simplified)
        let jitter = self.calculate_jitter(samples, sample_rate)?;

        // Calculate shimmer (simplified)
        let shimmer = self.calculate_shimmer(samples)?;

        // Estimate harmonic-to-noise ratio
        let harmonic_noise_ratio = self.calculate_harmonic_noise_ratio(samples)?;

        // Detect voice breaks
        let voice_breaks = self.detect_voice_breaks(samples)?;

        // Assess breathiness
        let breathiness = self.assess_breathiness(samples)?;

        // Assess roughness
        let roughness = self.assess_roughness(samples, sample_rate)?;

        // Assess strain
        let strain = self.assess_strain(samples)?;

        Ok(VoiceQualityMetrics {
            f0_stability,
            jitter,
            shimmer,
            harmonic_noise_ratio,
            voice_breaks,
            breathiness,
            roughness,
            strain,
        })
    }

    /// Detect pathological features
    async fn detect_pathological_features(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<HashMap<PathologicalCondition, f32>, EvaluationError> {
        let mut features = HashMap::new();
        let samples = generated_audio.samples();

        // Basic acoustic analysis
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();

        // Assess features for each known condition
        for &condition in &self.config.pathological_conditions {
            let severity = self.assess_condition_severity(samples, condition).await?;
            features.insert(condition, severity);
        }

        // If no specific conditions specified, perform general screening
        if self.config.pathological_conditions.is_empty() {
            // Screen for common pathological features
            features.insert(
                PathologicalCondition::Presbyphonia,
                self.screen_presbyphonia(samples).await?,
            );
            features.insert(
                PathologicalCondition::VocalCordParalysis,
                self.screen_vocal_cord_issues(samples).await?,
            );
            features.insert(
                PathologicalCondition::Parkinsons,
                self.screen_parkinsons_features(samples).await?,
            );
        }

        Ok(features)
    }

    /// Assessment helper methods
    async fn assess_intelligibility_quiet(
        &self,
        generated_audio: &AudioBuffer,
        _reference_audio: Option<&AudioBuffer>,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();

        // Basic intelligibility estimate based on signal clarity
        let signal_clarity = (rms * 2.0).min(1.0);
        let noise_level = self.estimate_noise_level(samples);
        let snr = (rms / noise_level.max(0.001)).log10() / 2.0; // Normalize to 0-1 range

        Ok((signal_clarity * 0.6 + snr.min(1.0) * 0.4).min(1.0))
    }

    async fn assess_intelligibility_noise(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
    ) -> Result<f32, EvaluationError> {
        // Simulate noise degradation
        let quiet_score = self
            .assess_intelligibility_quiet(generated_audio, reference_audio)
            .await?;
        let noise_degradation = 0.3; // Typical degradation in noise

        Ok((quiet_score * (1.0 - noise_degradation)).max(0.0))
    }

    async fn calculate_listener_burden(
        &self,
        generated_audio: &AudioBuffer,
        _target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();

        // Assess signal variability (higher variability = higher burden)
        let mean_energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let variance = samples
            .iter()
            .map(|&x| (x * x - mean_energy).powi(2))
            .sum::<f32>()
            / samples.len() as f32;

        let variability = variance.sqrt() / mean_energy.sqrt().max(0.001);

        // Higher variability suggests more effort needed
        Ok(variability.min(1.0))
    }

    async fn assess_communication_efficiency(
        &self,
        generated_audio: &AudioBuffer,
        target_text: Option<&str>,
    ) -> Result<f32, EvaluationError> {
        let duration = generated_audio.duration();

        if let Some(text) = target_text {
            let word_count = text.split_whitespace().count();
            let words_per_second = word_count as f32 / duration;

            // Optimal speaking rate for elderly/pathological: 140-180 words/minute (2.3-3.0 words/second)
            let efficiency = if words_per_second >= 2.3 && words_per_second <= 3.0 {
                1.0
            } else if words_per_second < 2.3 {
                words_per_second / 2.3
            } else {
                3.0 / words_per_second
            };

            Ok(efficiency.min(1.0))
        } else {
            // Estimate based on temporal characteristics
            Ok(0.7) // Default moderate efficiency
        }
    }

    /// Voice quality assessment methods
    fn calculate_f0_stability(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> Result<f32, EvaluationError> {
        // Simplified F0 stability calculation
        let frame_size = (sample_rate * 0.025) as usize; // 25ms frames
        let hop_size = (sample_rate * 0.010) as usize; // 10ms hop

        let mut f0_values = Vec::new();

        for i in (0..samples.len()).step_by(hop_size) {
            if i + frame_size > samples.len() {
                break;
            }

            let frame = &samples[i..i + frame_size];
            let f0 = self.estimate_frame_f0(frame, sample_rate)?;
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            return Ok(0.0);
        }

        // Calculate coefficient of variation
        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let variance = f0_values
            .iter()
            .map(|&f0| (f0 - mean_f0).powi(2))
            .sum::<f32>()
            / f0_values.len() as f32;
        let std_dev = variance.sqrt();

        let cv = std_dev / mean_f0;
        let stability = (1.0 - cv.min(1.0)).max(0.0);

        Ok(stability)
    }

    fn estimate_frame_f0(&self, frame: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
        // Simple autocorrelation-based F0 estimation
        let min_period = (sample_rate / 500.0) as usize; // Max F0
        let max_period = (sample_rate / 50.0) as usize; // Min F0

        if max_period >= frame.len() {
            return Ok(0.0);
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

        if max_correlation > 0.3 {
            Ok(sample_rate / best_period as f32)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_jitter(&self, samples: &[f32], sample_rate: f32) -> Result<f32, EvaluationError> {
        // Simplified jitter calculation as F0 period variation
        let frame_size = (sample_rate * 0.025) as usize;
        let hop_size = (sample_rate * 0.010) as usize;

        let mut periods = Vec::new();

        for i in (0..samples.len()).step_by(hop_size) {
            if i + frame_size > samples.len() {
                break;
            }

            let frame = &samples[i..i + frame_size];
            let f0 = self.estimate_frame_f0(frame, sample_rate)?;
            if f0 > 0.0 {
                periods.push(sample_rate / f0);
            }
        }

        if periods.len() < 2 {
            return Ok(0.0);
        }

        // Calculate period-to-period variation
        let mut jitter_sum = 0.0;
        for window in periods.windows(2) {
            jitter_sum += (window[1] - window[0]).abs();
        }

        let mean_period = periods.iter().sum::<f32>() / periods.len() as f32;
        let jitter = (jitter_sum / (periods.len() - 1) as f32) / mean_period;

        Ok(jitter.min(1.0))
    }

    fn calculate_shimmer(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Simplified shimmer calculation as amplitude variation
        let frame_size = 1024;
        let hop_size = 512;

        let mut amplitudes = Vec::new();

        for i in (0..samples.len()).step_by(hop_size) {
            if i + frame_size > samples.len() {
                break;
            }

            let frame = &samples[i..i + frame_size];
            let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            amplitudes.push(rms);
        }

        if amplitudes.len() < 2 {
            return Ok(0.0);
        }

        // Calculate amplitude-to-amplitude variation
        let mut shimmer_sum = 0.0;
        for window in amplitudes.windows(2) {
            if window[0] > 0.0 && window[1] > 0.0 {
                shimmer_sum += ((window[1] - window[0]).abs() / window[0]).min(2.0);
            }
        }

        let shimmer = shimmer_sum / (amplitudes.len() - 1) as f32;
        Ok(shimmer.min(1.0))
    }

    fn calculate_harmonic_noise_ratio(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Simplified HNR calculation
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();

        // Estimate noise level
        let noise_level = self.estimate_noise_level(samples);

        // Calculate HNR in dB, then normalize to 0-1
        let hnr_db = 20.0 * (rms / noise_level.max(0.001)).log10();
        let normalized_hnr = (hnr_db / 30.0).min(1.0).max(0.0); // Normalize to 0-30 dB range

        Ok(normalized_hnr)
    }

    fn detect_voice_breaks(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Detect sudden amplitude drops that might indicate voice breaks
        let threshold = 0.1; // Amplitude threshold
        let min_break_duration = 0.05; // 50ms minimum break
        let sample_rate = 16000.0; // Assume 16kHz
        let min_break_samples = (min_break_duration * sample_rate) as usize;

        let mut in_break = false;
        let mut break_count = 0;
        let mut break_samples = 0;

        for &sample in samples {
            if sample.abs() < threshold {
                if !in_break {
                    in_break = true;
                    break_samples = 1;
                } else {
                    break_samples += 1;
                }
            } else {
                if in_break && break_samples >= min_break_samples {
                    break_count += 1;
                }
                in_break = false;
            }
        }

        // Normalize by audio duration
        let total_duration = samples.len() as f32 / sample_rate;
        let breaks_per_second = break_count as f32 / total_duration;

        Ok((breaks_per_second / 2.0).min(1.0)) // Normalize to 0-1 (2 breaks/sec = 1.0)
    }

    fn assess_breathiness(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Estimate breathiness based on spectral characteristics
        // Higher frequency content relative to low frequencies suggests breathiness
        let window_size = 1024;
        let mut total_breathiness = 0.0;
        let mut window_count = 0;

        for chunk in samples.chunks(window_size) {
            if chunk.len() == window_size {
                let high_freq_energy = chunk
                    .iter()
                    .skip(window_size / 2)
                    .map(|&x| x * x)
                    .sum::<f32>();
                let low_freq_energy = chunk
                    .iter()
                    .take(window_size / 2)
                    .map(|&x| x * x)
                    .sum::<f32>();

                if low_freq_energy > 0.0 {
                    let ratio = high_freq_energy / low_freq_energy;
                    total_breathiness += ratio.min(2.0); // Cap at 2.0
                    window_count += 1;
                }
            }
        }

        if window_count > 0 {
            Ok((total_breathiness / window_count as f32 / 2.0).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_roughness(&self, samples: &[f32], _sample_rate: f32) -> Result<f32, EvaluationError> {
        // Estimate roughness based on signal irregularity
        let mut roughness_sum = 0.0;
        let window_size = 256;

        for chunk in samples.chunks(window_size) {
            if chunk.len() == window_size {
                // Calculate signal variability within window
                let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
                let variance =
                    chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / chunk.len() as f32;

                roughness_sum += variance.sqrt();
            }
        }

        let num_windows = samples.len() / window_size;
        if num_windows > 0 {
            Ok((roughness_sum / num_windows as f32).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_strain(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Estimate strain based on signal compression and energy concentration
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let peak_energy = samples.iter().map(|&x| x * x).fold(0.0, f32::max);

        // High peak-to-average ratio might indicate strain
        if energy > 0.0 {
            let crest_factor = peak_energy.sqrt() / energy.sqrt();
            Ok((crest_factor / 10.0).min(1.0)) // Normalize assuming max crest factor of 10
        } else {
            Ok(0.0)
        }
    }

    fn estimate_noise_level(&self, samples: &[f32]) -> f32 {
        // Estimate noise floor as minimum energy in quiet segments
        let window_size = 512;
        let mut min_energy = f32::INFINITY;

        for chunk in samples.chunks(window_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            if energy < min_energy {
                min_energy = energy;
            }
        }

        min_energy.sqrt()
    }

    /// Additional assessment methods (simplified implementations)
    async fn assess_condition_severity(
        &self,
        _samples: &[f32],
        condition: PathologicalCondition,
    ) -> Result<f32, EvaluationError> {
        // Simplified condition-specific assessment
        let features = condition.characteristic_features();
        Ok(features.communication_impact * 0.8) // Simplified scoring
    }

    async fn screen_presbyphonia(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Screen for age-related voice changes
        let voice_quality = self
            .assess_voice_quality_metrics(&AudioBuffer::new(samples.to_vec(), 16000, 1))
            .await?;
        Ok((voice_quality.breathiness + voice_quality.voice_breaks) / 2.0)
    }

    async fn screen_vocal_cord_issues(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Screen for vocal cord paralysis/weakness
        let voice_quality = self
            .assess_voice_quality_metrics(&AudioBuffer::new(samples.to_vec(), 16000, 1))
            .await?;
        Ok((voice_quality.breathiness + (1.0 - voice_quality.f0_stability)) / 2.0)
    }

    async fn screen_parkinsons_features(&self, samples: &[f32]) -> Result<f32, EvaluationError> {
        // Screen for Parkinson's-related features
        let voice_quality = self
            .assess_voice_quality_metrics(&AudioBuffer::new(samples.to_vec(), 16000, 1))
            .await?;
        Ok((voice_quality.voice_breaks + voice_quality.jitter + voice_quality.shimmer) / 3.0)
    }

    async fn assess_age_related_changes(
        &self,
        _generated_audio: &AudioBuffer,
        age_group: ElderlyAgeGroup,
    ) -> Result<AgeRelatedChanges, EvaluationError> {
        // Return expected changes for age group (could be enhanced with actual analysis)
        Ok(age_group.expected_changes())
    }

    async fn assess_dysarthria_severity(
        &self,
        _generated_audio: &AudioBuffer,
        voice_quality: &VoiceQualityMetrics,
    ) -> Result<Option<SeverityLevel>, EvaluationError> {
        // Assess dysarthria severity based on voice quality metrics
        let severity_score = (voice_quality.f0_stability
            + (1.0 - voice_quality.jitter)
            + (1.0 - voice_quality.shimmer)
            + voice_quality.harmonic_noise_ratio / 30.0)
            / 4.0;

        Ok(Some(SeverityLevel::from_score(severity_score)))
    }

    async fn assess_respiratory_support(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Assess respiratory support based on volume consistency and breath patterns
        let samples = generated_audio.samples();
        let consistency = self.assess_signal_consistency(samples);
        Ok(consistency)
    }

    async fn assess_motor_speech_control(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Assess motor control based on speech precision and coordination
        let voice_quality = self.assess_voice_quality_metrics(generated_audio).await?;
        let control_score = (voice_quality.f0_stability + (1.0 - voice_quality.jitter)) / 2.0;
        Ok(control_score)
    }

    async fn assess_cognitive_linguistic_function(
        &self,
        _generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Simplified cognitive-linguistic assessment
        Ok(0.8) // Would require more complex analysis in practice
    }

    async fn assess_amplification_needs(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let samples = generated_audio.samples();
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        // Lower RMS suggests higher need for amplification
        let amplification_need = 1.0 - (rms * 5.0).min(1.0);
        Ok(amplification_need)
    }

    async fn assess_clarity_enhancement_needs(
        &self,
        generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        let voice_quality = self.assess_voice_quality_metrics(generated_audio).await?;

        // Higher jitter/shimmer suggests higher need for clarity enhancement
        let clarity_need = (voice_quality.jitter + voice_quality.shimmer) / 2.0;
        Ok(clarity_need)
    }

    async fn assess_technology_adaptation(
        &self,
        _generated_audio: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Simplified technology adaptation assessment
        // Would involve cognitive assessment in practice
        if let Some(age_group) = self.config.target_age_group {
            let changes = age_group.expected_changes();
            Ok(1.0 - changes.cognitive_load_sensitivity)
        } else {
            Ok(0.7) // Default moderate adaptation
        }
    }

    fn assess_signal_consistency(&self, samples: &[f32]) -> f32 {
        let window_size = 512;
        let mut energies = Vec::new();

        for chunk in samples.chunks(window_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
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

    /// Helper methods for generating recommendations and assessments
    fn determine_severity_level(
        &self,
        communication_effectiveness: &CommunicationEffectivenessResult,
        _pathological_features: &HashMap<PathologicalCondition, f32>,
    ) -> SeverityLevel {
        SeverityLevel::from_score(communication_effectiveness.overall_effectiveness)
    }

    fn calculate_context_adjusted_scores(
        &self,
        communication_effectiveness: &CommunicationEffectivenessResult,
        pathological_features: &HashMap<PathologicalCondition, f32>,
    ) -> HashMap<String, f32> {
        let mut scores = HashMap::new();
        let context_expectations = self.config.communication_context.expectations();

        // Adjust intelligibility for context
        let context_adjusted_intelligibility = communication_effectiveness.quiet_intelligibility
            * context_expectations.intelligibility_requirement;
        scores.insert(
            "context_adjusted_intelligibility".to_string(),
            context_adjusted_intelligibility,
        );

        // Adjust for time pressure
        let time_pressure_adjustment = 1.0 - context_expectations.time_pressure * 0.2;
        scores.insert(
            "time_pressure_adjusted".to_string(),
            communication_effectiveness.communication_efficiency * time_pressure_adjustment,
        );

        // Adjust for background noise
        let noise_adjusted = communication_effectiveness.noise_intelligibility
            * (1.0 - context_expectations.background_noise * 0.3);
        scores.insert("noise_adjusted".to_string(), noise_adjusted);

        scores
    }

    fn generate_support_recommendations(
        &self,
        communication_effectiveness: &CommunicationEffectivenessResult,
        pathological_features: &HashMap<PathologicalCondition, f32>,
        clinical_assessment: Option<&ClinicalAssessmentResult>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Based on communication effectiveness
        if communication_effectiveness.overall_effectiveness < 0.7 {
            recommendations.push(
                "Consider communication partner training for improved understanding".to_string(),
            );
        }

        if communication_effectiveness.listener_burden > 0.6 {
            recommendations
                .push("Implement communication strategies to reduce listener effort".to_string());
        }

        // Based on pathological features
        for (condition, &severity) in pathological_features {
            if severity > 0.5 {
                match condition {
                    PathologicalCondition::Parkinsons => {
                        recommendations
                            .push("LSVT LOUD therapy may improve voice loudness".to_string());
                    }
                    PathologicalCondition::VocalCordParalysis => {
                        recommendations.push(
                            "Voice therapy focusing on breath support and coordination".to_string(),
                        );
                    }
                    PathologicalCondition::Presbyphonia => {
                        recommendations.push(
                            "Voice exercises to maintain vocal strength and flexibility"
                                .to_string(),
                        );
                    }
                    _ => {
                        recommendations.push("Condition-specific therapy recommended".to_string());
                    }
                }
            }
        }

        // Based on clinical assessment
        if let Some(clinical) = clinical_assessment {
            if clinical.respiratory_support < 0.6 {
                recommendations
                    .push("Respiratory muscle training to improve breath support".to_string());
            }
        }

        recommendations
    }

    fn assess_quality_of_life_impact(
        &self,
        communication_effectiveness: &CommunicationEffectivenessResult,
        pathological_features: &HashMap<PathologicalCondition, f32>,
        severity_level: SeverityLevel,
    ) -> f32 {
        let communication_impact = 1.0 - communication_effectiveness.overall_effectiveness;

        let pathological_impact = pathological_features
            .values()
            .map(|&severity| severity * 0.1)
            .sum::<f32>()
            .min(0.5);

        let severity_impact = match severity_level {
            SeverityLevel::Normal => 0.0,
            SeverityLevel::Mild => 0.1,
            SeverityLevel::Moderate => 0.3,
            SeverityLevel::Severe => 0.6,
            SeverityLevel::Profound => 0.9,
        };

        (communication_impact + pathological_impact + severity_impact).min(1.0)
    }

    fn identify_communication_strengths(
        &self,
        quiet_intelligibility: f32,
        noise_intelligibility: f32,
        communication_efficiency: f32,
    ) -> Vec<String> {
        let mut strengths = Vec::new();

        if quiet_intelligibility > 0.8 {
            strengths.push("Good intelligibility in quiet conditions".to_string());
        }

        if noise_intelligibility > 0.7 {
            strengths.push("Maintains intelligibility in background noise".to_string());
        }

        if communication_efficiency > 0.8 {
            strengths.push("Efficient communication rate".to_string());
        }

        if strengths.is_empty() {
            strengths.push("Maintains functional communication".to_string());
        }

        strengths
    }

    fn identify_support_areas(
        &self,
        quiet_intelligibility: f32,
        noise_intelligibility: f32,
        listener_burden: f32,
        communication_efficiency: f32,
    ) -> Vec<String> {
        let mut areas = Vec::new();

        if quiet_intelligibility < 0.7 {
            areas.push("Speech clarity in quiet conditions".to_string());
        }

        if noise_intelligibility < 0.6 {
            areas.push("Communication in noisy environments".to_string());
        }

        if listener_burden > 0.6 {
            areas.push("Reducing listener effort and concentration".to_string());
        }

        if communication_efficiency < 0.6 {
            areas.push("Improving communication rate and efficiency".to_string());
        }

        areas
    }

    fn estimate_user_acceptance(&self, amplification: f32, clarity: f32, adaptation: f32) -> f32 {
        // Higher need for technology but lower adaptation ability = lower acceptance
        let technology_need = (amplification + clarity) / 2.0;
        let acceptance = (technology_need * 0.6 + adaptation * 0.4).min(1.0);
        acceptance
    }

    fn recommend_assistive_technologies(
        &self,
        amplification: f32,
        clarity: f32,
        adaptation: f32,
    ) -> Vec<String> {
        let mut technologies = Vec::new();

        if amplification > 0.6 {
            technologies.push("Personal voice amplifier".to_string());
            technologies.push("Portable PA system".to_string());
        }

        if clarity > 0.6 {
            technologies.push("Speech enhancement software".to_string());
            technologies.push("Real-time speech processing app".to_string());
        }

        if adaptation > 0.7 {
            technologies.push("Voice banking software".to_string());
            technologies.push("Communication app with pre-recorded messages".to_string());
        }

        if technologies.is_empty() {
            technologies.push("Communication strategies training".to_string());
        }

        technologies
    }

    fn suggest_technology_configurations(
        &self,
        technologies: &[String],
        amplification: f32,
        clarity: f32,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if technologies.iter().any(|t| t.contains("amplifier")) {
            if amplification > 0.8 {
                suggestions.push("Set amplification to maximum comfortable level".to_string());
            } else {
                suggestions
                    .push("Start with moderate amplification and adjust gradually".to_string());
            }
        }

        if technologies.iter().any(|t| t.contains("enhancement")) {
            if clarity > 0.8 {
                suggestions.push("Enable maximum clarity enhancement features".to_string());
            } else {
                suggestions.push("Use mild to moderate enhancement settings".to_string());
            }
        }

        suggestions.push("Provide comprehensive user training and support".to_string());
        suggestions.push("Regular follow-up to adjust settings as needed".to_string());

        suggestions
    }

    fn generate_clinical_recommendations(
        &self,
        voice_quality: &VoiceQualityMetrics,
        dysarthria_severity: Option<SeverityLevel>,
        respiratory_support: f32,
        motor_speech_control: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if voice_quality.breathiness > 0.6 {
            recommendations.push(
                "Voice therapy focusing on breath support and vocal fold adduction".to_string(),
            );
        }

        if voice_quality.roughness > 0.6 {
            recommendations
                .push("Medical evaluation for possible vocal fold pathology".to_string());
        }

        if let Some(severity) = dysarthria_severity {
            match severity {
                SeverityLevel::Moderate | SeverityLevel::Severe | SeverityLevel::Profound => {
                    recommendations
                        .push("Comprehensive speech therapy evaluation and treatment".to_string());
                }
                _ => {}
            }
        }

        if respiratory_support < 0.6 {
            recommendations
                .push("Pulmonary function evaluation and respiratory therapy".to_string());
        }

        if motor_speech_control < 0.6 {
            recommendations
                .push("Motor speech therapy with focus on articulation precision".to_string());
        }

        recommendations
    }

    fn generate_therapy_goals(
        &self,
        voice_quality: &VoiceQualityMetrics,
        dysarthria_severity: Option<SeverityLevel>,
        respiratory_support: f32,
    ) -> Vec<String> {
        let mut goals = Vec::new();

        if voice_quality.f0_stability < 0.7 {
            goals.push("Improve vocal stability and reduce tremor".to_string());
        }

        if voice_quality.harmonic_noise_ratio < 15.0 {
            goals.push("Enhance voice quality and reduce breathiness".to_string());
        }

        if respiratory_support < 0.7 {
            goals.push("Strengthen respiratory support for speech".to_string());
        }

        if let Some(severity) = dysarthria_severity {
            if matches!(severity, SeverityLevel::Moderate | SeverityLevel::Severe) {
                goals.push(
                    "Improve speech intelligibility for functional communication".to_string(),
                );
            }
        }

        goals.push("Maintain current level of communication function".to_string());

        goals
    }

    async fn calculate_overall_score(
        &self,
        communication_effectiveness: &CommunicationEffectivenessResult,
        pathological_features: &HashMap<PathologicalCondition, f32>,
        severity_level: SeverityLevel,
        context_adjusted_scores: &HashMap<String, f32>,
    ) -> Result<f32, EvaluationError> {
        // Weight communication effectiveness heavily
        let mut score = communication_effectiveness.overall_effectiveness * 0.6;

        // Adjust for pathological features
        let pathological_impact = pathological_features
            .values()
            .map(|&s| s * 0.1)
            .sum::<f32>()
            .min(0.3);
        score -= pathological_impact;

        // Adjust for severity
        let severity_adjustment = match severity_level {
            SeverityLevel::Normal => 0.0,
            SeverityLevel::Mild => -0.05,
            SeverityLevel::Moderate => -0.15,
            SeverityLevel::Severe => -0.3,
            SeverityLevel::Profound => -0.5,
        };
        score += severity_adjustment;

        // Apply context adjustments
        if let Some(&context_score) =
            context_adjusted_scores.get("context_adjusted_intelligibility")
        {
            score = (score + context_score * 0.2) / 1.2; // Weight context adjustment
        }

        Ok(score.max(0.0).min(1.0))
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
            confidence += 0.15;
        }

        // Target text increases confidence
        if target_text.is_some() {
            confidence += 0.1;
        }

        // Known conditions increase confidence
        if !self.config.pathological_conditions.is_empty() {
            confidence += 0.1;
        }

        // Age group specified increases confidence
        if self.config.target_age_group.is_some() {
            confidence += 0.05;
        }

        confidence.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_elderly_pathological_evaluator_creation() {
        let evaluator = ElderlyPathologicalEvaluator::new().await.unwrap();
        assert_eq!(
            evaluator.config.target_age_group,
            Some(ElderlyAgeGroup::YoungElderly)
        );
    }

    #[tokio::test]
    async fn test_elderly_pathological_evaluation() {
        let evaluator = ElderlyPathologicalEvaluator::new().await.unwrap();

        // Create test audio
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = evaluator
            .evaluate_elderly_pathological_speech(&audio, None, Some("Hello world"))
            .await
            .unwrap();

        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(result.communication_effectiveness.overall_effectiveness >= 0.0);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_age_group_characteristics() {
        let young_elderly = ElderlyAgeGroup::YoungElderly;
        let changes = young_elderly.expected_changes();

        assert_eq!(young_elderly.age_range(), (65, 74));
        assert!(changes.fundamental_frequency_change < 0.2);
        assert!(changes.voice_tremor_likelihood < 0.3);
    }

    #[test]
    fn test_pathological_condition_features() {
        let parkinsons = PathologicalCondition::Parkinsons;
        let features = parkinsons.characteristic_features();

        assert!(features.reduced_loudness > 0.5);
        assert!(features.monotone_speech > 0.5);
    }

    #[test]
    fn test_severity_level_conversion() {
        assert_eq!(SeverityLevel::from_score(0.95), SeverityLevel::Normal);
        assert_eq!(SeverityLevel::from_score(0.75), SeverityLevel::Mild);
        assert_eq!(SeverityLevel::from_score(0.55), SeverityLevel::Moderate);
        assert_eq!(SeverityLevel::from_score(0.35), SeverityLevel::Severe);
        assert_eq!(SeverityLevel::from_score(0.15), SeverityLevel::Profound);
    }

    #[test]
    fn test_listener_familiarity() {
        assert!(ListenerFamiliarity::VeryFamiliar.intelligibility_adjustment() > 1.0);
        assert!(ListenerFamiliarity::Unfamiliar.intelligibility_adjustment() < 1.0);
    }

    #[test]
    fn test_communication_context() {
        let clinical = CommunicationContext::Clinical;
        let expectations = clinical.expectations();

        assert!(expectations.intelligibility_requirement > 0.8);
        assert!(expectations.listener_patience > 0.8);
        assert!(expectations.background_noise < 0.2);
    }

    #[test]
    fn test_config_default() {
        let config = ElderlyPathologicalConfig::default();
        assert_eq!(config.target_age_group, Some(ElderlyAgeGroup::YoungElderly));
        assert!(config.perform_clinical_assessment);
        assert!(config.evaluate_assistive_technology);
    }

    #[tokio::test]
    async fn test_voice_quality_metrics() {
        let evaluator = ElderlyPathologicalEvaluator::new().await.unwrap();

        // Create test audio with some characteristics
        let samples: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 150.0 * t).sin() * 0.3
                    + (2.0 * std::f32::consts::PI * 75.0 * t).sin() * 0.1 // Add some noise
            })
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let voice_quality = evaluator
            .assess_voice_quality_metrics(&audio)
            .await
            .unwrap();

        assert!(voice_quality.f0_stability >= 0.0);
        assert!(voice_quality.f0_stability <= 1.0);
        assert!(voice_quality.jitter >= 0.0);
        assert!(voice_quality.shimmer >= 0.0);
        assert!(voice_quality.harmonic_noise_ratio >= 0.0);
    }

    #[tokio::test]
    async fn test_pathological_condition_evaluation() {
        let mut config = ElderlyPathologicalConfig::default();
        config.pathological_conditions = vec![PathologicalCondition::Parkinsons];
        let evaluator = ElderlyPathologicalEvaluator::with_config(config)
            .await
            .unwrap();

        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 120.0 * i as f32 / 16000.0).sin() * 0.2) // Quiet voice
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = evaluator
            .evaluate_elderly_pathological_speech(&audio, None, None)
            .await
            .unwrap();

        assert!(result
            .pathological_features
            .contains_key(&PathologicalCondition::Parkinsons));
        assert!(result.pathological_features[&PathologicalCondition::Parkinsons] >= 0.0);
    }
}
