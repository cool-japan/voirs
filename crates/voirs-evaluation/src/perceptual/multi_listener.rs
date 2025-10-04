//! Enhanced multi-listener simulation for comprehensive perceptual evaluation
//!
//! This module provides advanced virtual listener modeling with demographic,
//! cultural, and contextual variations for more realistic listening test simulation.

use crate::audio::streaming::{AudioChunk, StreamingQualityMetrics};
use crate::perceptual::cross_cultural::{
    CrossCulturalAdaptation, CrossCulturalConfig, CrossCulturalPerceptualModel,
};
use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use std::collections::HashMap;
use std::time::Instant;
use voirs_sdk::AudioBuffer;

/// Enhanced configuration for multi-listener simulation
#[derive(Debug, Clone)]
pub struct MultiListenerConfig {
    /// Number of virtual listeners to simulate
    pub num_listeners: usize,
    /// Enable demographic diversity
    pub enable_demographic_diversity: bool,
    /// Enable cultural adaptation
    pub enable_cultural_adaptation: bool,
    /// Enable hearing impairment simulation
    pub enable_hearing_impairment: bool,
    /// Enable environmental condition simulation
    pub enable_environmental_conditions: bool,
    /// Enable cross-cultural perceptual modeling
    pub enable_cross_cultural_modeling: bool,
    /// Target language for evaluation (e.g., "en", "zh", "es")
    pub target_language: String,
    /// Cross-cultural model configuration
    pub cross_cultural_config: CrossCulturalConfig,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
}

impl Default for MultiListenerConfig {
    fn default() -> Self {
        Self {
            num_listeners: 20,
            enable_demographic_diversity: true,
            enable_cultural_adaptation: true,
            enable_hearing_impairment: false,
            enable_environmental_conditions: true,
            enable_cross_cultural_modeling: true,
            target_language: "en".to_string(),
            cross_cultural_config: CrossCulturalConfig::default(),
            random_seed: None,
        }
    }
}

/// Demographic profile for virtual listeners
#[derive(Debug, Clone)]
pub struct DemographicProfile {
    /// Age group
    pub age_group: AgeGroup,
    /// Gender
    pub gender: Gender,
    /// Educational background
    pub education_level: EducationLevel,
    /// Native language
    pub native_language: String,
    /// Audio experience level
    pub audio_experience: ExperienceLevel,
}

/// Age groups for listener simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum AgeGroup {
    Young,      // 18-30
    MiddleAged, // 31-55
    Older,      // 55+
}

/// Gender categories
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub enum Gender {
    Male,
    Female,
    Other,
}

/// Education levels
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub enum EducationLevel {
    HighSchool,
    Bachelor,
    Master,
    PhD,
}

/// Experience levels with audio evaluation
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub enum ExperienceLevel {
    Novice,
    Intermediate,
    Advanced,
    Expert,
}

/// Cultural background information
#[derive(Debug, Clone)]
pub struct CulturalProfile {
    /// Cultural region
    pub region: CulturalRegion,
    /// Language family familiarity
    pub language_familiarity: Vec<String>,
    /// Musical training background
    pub musical_training: bool,
    /// Accent tolerance level
    pub accent_tolerance: f32,
}

/// Cultural regions for listener simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum CulturalRegion {
    NorthAmerica,
    Europe,
    EastAsia,
    SouthAsia,
    MiddleEast,
    Africa,
    SouthAmerica,
    Oceania,
}

/// Hearing profile for impairment simulation
#[derive(Debug, Clone)]
pub struct HearingProfile {
    /// Overall hearing acuity (0.0 = severe loss, 1.0 = perfect)
    pub hearing_acuity: f32,
    /// Frequency-specific hearing loss (by frequency band)
    pub frequency_loss: Vec<f32>,
    /// Noise tolerance
    pub noise_tolerance: f32,
    /// Temporal processing ability
    pub temporal_processing: f32,
}

impl Default for HearingProfile {
    fn default() -> Self {
        Self {
            hearing_acuity: 1.0,
            frequency_loss: vec![0.0; 8], // 8 frequency bands
            noise_tolerance: 1.0,
            temporal_processing: 1.0,
        }
    }
}

/// Environmental listening conditions
#[derive(Debug, Clone)]
pub struct EnvironmentalConditions {
    /// Background noise level (dB)
    pub noise_level: f32,
    /// Reverberation characteristics
    pub reverberation: f32,
    /// Listening device quality
    pub device_quality: DeviceQuality,
    /// Listening attention level
    pub attention_level: f32,
}

/// Audio device quality categories
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub enum DeviceQuality {
    HighEnd,    // Professional monitoring
    Standard,   // Consumer headphones
    LowEnd,     // Basic earbuds
    Smartphone, // Phone speaker
}

impl Default for EnvironmentalConditions {
    fn default() -> Self {
        Self {
            noise_level: -40.0, // Quiet environment
            reverberation: 0.1,
            device_quality: DeviceQuality::Standard,
            attention_level: 0.8,
        }
    }
}

/// Enhanced virtual listener with comprehensive characteristics
#[derive(Debug, Clone)]
pub struct EnhancedVirtualListener {
    /// Unique identifier
    pub id: String,
    /// Demographic information
    pub demographic: DemographicProfile,
    /// Cultural background
    pub cultural: CulturalProfile,
    /// Hearing characteristics
    pub hearing: HearingProfile,
    /// Current listening environment
    pub environment: EnvironmentalConditions,
    /// Response bias parameters
    pub bias_parameters: BiasParameters,
    /// Fatigue state
    pub fatigue_level: f32,
}

/// Response bias parameters
#[derive(Debug, Clone)]
pub struct BiasParameters {
    /// Tendency to rate higher or lower
    pub rating_bias: f32,
    /// Response consistency
    pub consistency: f32,
    /// Extreme response tendency
    pub extreme_tendency: f32,
    /// Acquiescence bias (tendency to agree)
    pub acquiescence: f32,
}

impl Default for BiasParameters {
    fn default() -> Self {
        Self {
            rating_bias: 0.0,
            consistency: 0.8,
            extreme_tendency: 0.0,
            acquiescence: 0.0,
        }
    }
}

/// Results from multi-listener evaluation
#[derive(Debug, Clone)]
pub struct MultiListenerResults {
    /// Individual listener scores
    pub individual_scores: Vec<(String, f32)>,
    /// Aggregate statistics
    pub aggregate_stats: AggregateStatistics,
    /// Demographic breakdowns
    pub demographic_analysis: HashMap<String, f32>,
    /// Cultural variation analysis
    pub cultural_analysis: HashMap<String, f32>,
    /// Reliability metrics
    pub reliability_metrics: ReliabilityMetrics,
    /// Evaluation timestamp
    pub timestamp: Instant,
}

/// Aggregate statistics across all listeners
#[derive(Debug, Clone)]
pub struct AggregateStatistics {
    /// Mean score
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Median score
    pub median: f32,
    /// Confidence interval (95%)
    pub confidence_interval: (f32, f32),
    /// Inter-quartile range
    pub iqr: (f32, f32),
}

/// Reliability metrics for multi-listener evaluation
#[derive(Debug, Clone)]
pub struct ReliabilityMetrics {
    /// Cronbach's alpha
    pub cronbach_alpha: f32,
    /// Inter-rater reliability
    pub inter_rater_reliability: f32,
    /// Agreement percentage
    pub agreement_percentage: f32,
    /// Consistency index
    pub consistency_index: f32,
}

/// Enhanced multi-listener simulator
pub struct EnhancedMultiListenerSimulator {
    /// Configuration
    config: MultiListenerConfig,
    /// Virtual listeners
    listeners: Vec<EnhancedVirtualListener>,
    /// Random number generator
    rng: Box<dyn RngCore + Send>,
    /// Cross-cultural perceptual model
    cross_cultural_model: Option<CrossCulturalPerceptualModel>,
    /// Evaluation history
    evaluation_history: Vec<MultiListenerResults>,
}

impl EnhancedMultiListenerSimulator {
    /// Create a new enhanced multi-listener simulator
    pub fn new(config: MultiListenerConfig) -> Self {
        let mut rng: Box<dyn RngCore + Send> = if let Some(seed) = config.random_seed {
            Box::new(Random::seed(seed))
        } else {
            Box::new(Random::seed(0))
        };

        // Initialize cross-cultural model if enabled
        let cross_cultural_model = if config.enable_cross_cultural_modeling {
            Some(CrossCulturalPerceptualModel::new(
                config.cross_cultural_config.clone(),
            ))
        } else {
            None
        };

        let mut simulator = Self {
            config: config.clone(),
            listeners: Vec::new(),
            rng,
            cross_cultural_model,
            evaluation_history: Vec::new(),
        };

        // Generate virtual listeners
        simulator.generate_virtual_listeners();

        simulator
    }

    /// Generate diverse virtual listeners based on configuration
    fn generate_virtual_listeners(&mut self) {
        for i in 0..self.config.num_listeners {
            let listener = self.create_virtual_listener(i);
            self.listeners.push(listener);
        }
    }

    /// Create a single virtual listener with specific characteristics
    fn create_virtual_listener(&mut self, index: usize) -> EnhancedVirtualListener {
        let id = format!("listener_{:03}", index);

        // Generate demographic profile
        let demographic = if self.config.enable_demographic_diversity {
            self.generate_demographic_profile()
        } else {
            self.default_demographic_profile()
        };

        // Generate cultural profile
        let cultural = if self.config.enable_cultural_adaptation {
            self.generate_cultural_profile(&demographic)
        } else {
            self.default_cultural_profile()
        };

        // Generate hearing profile
        let hearing = if self.config.enable_hearing_impairment {
            self.generate_hearing_profile(&demographic)
        } else {
            HearingProfile::default()
        };

        // Generate environmental conditions
        let environment = if self.config.enable_environmental_conditions {
            self.generate_environmental_conditions()
        } else {
            EnvironmentalConditions::default()
        };

        // Generate bias parameters based on profiles
        let bias_parameters = self.generate_bias_parameters(&demographic, &cultural);

        EnhancedVirtualListener {
            id,
            demographic,
            cultural,
            hearing,
            environment,
            bias_parameters,
            fatigue_level: 0.0,
        }
    }

    /// Generate diverse demographic profile
    fn generate_demographic_profile(&mut self) -> DemographicProfile {
        let age_group = match self.rng.gen_range(0..3) {
            0 => AgeGroup::Young,
            1 => AgeGroup::MiddleAged,
            _ => AgeGroup::Older,
        };

        let gender = match self.rng.gen_range(0..3) {
            0 => Gender::Male,
            1 => Gender::Female,
            _ => Gender::Other,
        };

        let education_level = match self.rng.gen_range(0..4) {
            0 => EducationLevel::HighSchool,
            1 => EducationLevel::Bachelor,
            2 => EducationLevel::Master,
            _ => EducationLevel::PhD,
        };

        let audio_experience = match self.rng.gen_range(0..4) {
            0 => ExperienceLevel::Novice,
            1 => ExperienceLevel::Intermediate,
            2 => ExperienceLevel::Advanced,
            _ => ExperienceLevel::Expert,
        };

        let native_languages = vec!["en", "es", "zh", "hi", "ar", "pt", "ru", "ja", "de", "fr"];
        let native_language =
            native_languages[self.rng.gen_range(0..native_languages.len())].to_string();

        DemographicProfile {
            age_group,
            gender,
            education_level,
            native_language,
            audio_experience,
        }
    }

    /// Generate default demographic profile
    fn default_demographic_profile(&self) -> DemographicProfile {
        DemographicProfile {
            age_group: AgeGroup::MiddleAged,
            gender: Gender::Other,
            education_level: EducationLevel::Bachelor,
            native_language: "en".to_string(),
            audio_experience: ExperienceLevel::Intermediate,
        }
    }

    /// Generate cultural profile based on demographic
    fn generate_cultural_profile(&mut self, demographic: &DemographicProfile) -> CulturalProfile {
        let region = match demographic.native_language.as_str() {
            "en" => CulturalRegion::NorthAmerica,
            "es" | "pt" => CulturalRegion::SouthAmerica,
            "zh" | "ja" => CulturalRegion::EastAsia,
            "hi" => CulturalRegion::SouthAsia,
            "ar" => CulturalRegion::MiddleEast,
            "de" | "fr" | "ru" => CulturalRegion::Europe,
            _ => CulturalRegion::NorthAmerica,
        };

        let language_familiarity = vec![demographic.native_language.clone()];
        let musical_training = self.rng.gen_bool(0.3); // 30% have musical training
        let accent_tolerance = self.rng.gen_range(0.3..1.0);

        CulturalProfile {
            region,
            language_familiarity,
            musical_training,
            accent_tolerance,
        }
    }

    /// Generate default cultural profile
    fn default_cultural_profile(&self) -> CulturalProfile {
        CulturalProfile {
            region: CulturalRegion::NorthAmerica,
            language_familiarity: vec!["en".to_string()],
            musical_training: false,
            accent_tolerance: 0.7,
        }
    }

    /// Generate hearing profile with potential impairments
    fn generate_hearing_profile(&mut self, demographic: &DemographicProfile) -> HearingProfile {
        // Age-related hearing loss
        let age_factor = match demographic.age_group {
            AgeGroup::Young => 1.0,
            AgeGroup::MiddleAged => self.rng.gen_range(0.8..1.0),
            AgeGroup::Older => self.rng.gen_range(0.6..0.9),
        };

        let hearing_acuity = age_factor * self.rng.gen_range(0.8..1.0);

        // Frequency-specific loss (higher frequencies more affected with age)
        let mut frequency_loss = Vec::new();
        for i in 0..8 {
            let freq_factor = i as f32 / 8.0; // Higher frequencies
            let loss = (1.0 - age_factor) * freq_factor * self.rng.gen_range(0.5..1.5);
            frequency_loss.push(loss.min(0.8));
        }

        let noise_tolerance = hearing_acuity * self.rng.gen_range(0.7..1.0);
        let temporal_processing = hearing_acuity * self.rng.gen_range(0.8..1.0);

        HearingProfile {
            hearing_acuity,
            frequency_loss,
            noise_tolerance,
            temporal_processing,
        }
    }

    /// Generate environmental conditions
    fn generate_environmental_conditions(&mut self) -> EnvironmentalConditions {
        let noise_level = self.rng.gen_range(-50.0..-20.0); // Quiet to moderate noise
        let reverberation = self.rng.gen_range(0.0..0.4);

        let device_quality = match self.rng.gen_range(0..4) {
            0 => DeviceQuality::HighEnd,
            1 => DeviceQuality::Standard,
            2 => DeviceQuality::LowEnd,
            _ => DeviceQuality::Smartphone,
        };

        let attention_level = self.rng.gen_range(0.6..1.0);

        EnvironmentalConditions {
            noise_level,
            reverberation,
            device_quality,
            attention_level,
        }
    }

    /// Generate bias parameters based on profiles
    fn generate_bias_parameters(
        &mut self,
        demographic: &DemographicProfile,
        cultural: &CulturalProfile,
    ) -> BiasParameters {
        // Experience affects consistency
        let consistency = match demographic.audio_experience {
            ExperienceLevel::Novice => self.rng.gen_range(0.5..0.7),
            ExperienceLevel::Intermediate => self.rng.gen_range(0.7..0.8),
            ExperienceLevel::Advanced => self.rng.gen_range(0.8..0.9),
            ExperienceLevel::Expert => self.rng.gen_range(0.9..0.95),
        };

        // Cultural factors affect rating bias
        let rating_bias = match cultural.region {
            CulturalRegion::EastAsia => self.rng.gen_range(-0.1..0.0), // Slightly conservative
            CulturalRegion::NorthAmerica => self.rng.gen_range(-0.05..0.05), // Neutral
            _ => self.rng.gen_range(-0.1..0.1),
        };

        let extreme_tendency = self.rng.gen_range(0.0..0.3);
        let acquiescence = self.rng.gen_range(0.0..0.2);

        BiasParameters {
            rating_bias,
            consistency,
            extreme_tendency,
            acquiescence,
        }
    }

    /// Simulate listening test with all virtual listeners
    pub async fn simulate_listening_test(
        &mut self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> EvaluationResult<MultiListenerResults> {
        let mut individual_scores = Vec::new();
        let mut demographic_scores: HashMap<String, Vec<f32>> = HashMap::new();
        let mut cultural_scores: HashMap<String, Vec<f32>> = HashMap::new();

        // Collect scores from all listeners
        for i in 0..self.listeners.len() {
            let listener_id = self.listeners[i].id.clone();
            let age_group = self.listeners[i].demographic.age_group;
            let audio_experience = self.listeners[i].demographic.audio_experience;
            let cultural_region = self.listeners[i].cultural.region;

            let score = self
                .simulate_individual_response(&self.listeners[i], audio, reference)
                .await?;
            individual_scores.push((listener_id, score));

            // Group by demographics
            let age_key = format!("{:?}", age_group);
            demographic_scores
                .entry(age_key)
                .or_insert_with(Vec::new)
                .push(score);

            let exp_key = format!("{:?}", audio_experience);
            demographic_scores
                .entry(exp_key)
                .or_insert_with(Vec::new)
                .push(score);

            // Group by culture
            let region_key = format!("{:?}", cultural_region);
            cultural_scores
                .entry(region_key)
                .or_insert_with(Vec::new)
                .push(score);

            // Update fatigue
            self.listeners[i].fatigue_level = (self.listeners[i].fatigue_level + 0.01).min(0.5);
        }

        // Calculate aggregate statistics
        let scores: Vec<f32> = individual_scores.iter().map(|(_, score)| *score).collect();
        let aggregate_stats = self.calculate_aggregate_statistics(&scores);

        // Calculate demographic analysis
        let demographic_analysis = self.calculate_group_analysis(&demographic_scores);

        // Calculate cultural analysis
        let cultural_analysis = self.calculate_group_analysis(&cultural_scores);

        // Calculate reliability metrics
        let reliability_metrics = self.calculate_reliability_metrics(&scores);

        let results = MultiListenerResults {
            individual_scores,
            aggregate_stats,
            demographic_analysis,
            cultural_analysis,
            reliability_metrics,
            timestamp: Instant::now(),
        };

        // Store in history
        self.evaluation_history.push(results.clone());

        // Keep only recent history
        if self.evaluation_history.len() > 100 {
            self.evaluation_history.remove(0);
        }

        Ok(results)
    }

    /// Simulate individual listener response
    async fn simulate_individual_response(
        &self,
        listener: &EnhancedVirtualListener,
        audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
    ) -> EvaluationResult<f32> {
        // Calculate base quality score
        let base_score = self.calculate_base_quality_score(audio).await?;

        // Apply demographic effects
        let demo_adjusted = self.apply_demographic_effects(base_score, &listener.demographic);

        // Apply cultural effects
        let cultural_adjusted = self.apply_cultural_effects(demo_adjusted, &listener.cultural);

        // Apply cross-cultural perceptual modeling if enabled
        let cross_cultural_adjusted = if let Some(ref model) = self.cross_cultural_model {
            let adaptation = model.calculate_adaptation_factors(
                &listener.cultural,
                &listener.demographic,
                audio,
                &self.config.target_language,
            )?;
            model.apply_cultural_adaptation(cultural_adjusted, &adaptation)
        } else {
            cultural_adjusted
        };

        // Apply hearing effects
        let hearing_adjusted =
            self.apply_hearing_effects(cross_cultural_adjusted, &listener.hearing);

        // Apply environmental effects
        let env_adjusted =
            self.apply_environmental_effects(hearing_adjusted, &listener.environment);

        // Apply bias parameters
        let bias_adjusted = self.apply_bias_effects(
            env_adjusted,
            &listener.bias_parameters,
            listener.fatigue_level,
        );

        Ok(bias_adjusted.clamp(0.0, 1.0))
    }

    /// Calculate base quality score from audio
    async fn calculate_base_quality_score(&self, audio: &AudioBuffer) -> EvaluationResult<f32> {
        let samples = audio.samples();

        if samples.is_empty() {
            return Ok(0.5);
        }

        // Basic quality indicators
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        let dynamic_range = if rms > 0.0 {
            (peak / rms).min(100.0) / 100.0
        } else {
            0.0
        };

        let signal_quality = (rms * 10.0).min(1.0);
        let base_score = (signal_quality * 0.6 + dynamic_range * 0.4).clamp(0.1, 0.9);

        Ok(base_score)
    }

    /// Apply demographic effects to score
    fn apply_demographic_effects(&self, score: f32, demographic: &DemographicProfile) -> f32 {
        let mut adjusted = score;

        // Age effects
        match demographic.age_group {
            AgeGroup::Young => adjusted *= 1.02, // Slightly higher ratings
            AgeGroup::MiddleAged => {}           // No change
            AgeGroup::Older => adjusted *= 0.98, // Slightly lower ratings
        }

        // Experience effects
        match demographic.audio_experience {
            ExperienceLevel::Novice => adjusted *= 1.05, // More generous
            ExperienceLevel::Intermediate => {}          // No change
            ExperienceLevel::Advanced => adjusted *= 0.95, // More critical
            ExperienceLevel::Expert => adjusted *= 0.90, // Most critical
        }

        adjusted
    }

    /// Apply cultural effects to score
    fn apply_cultural_effects(&self, score: f32, cultural: &CulturalProfile) -> f32 {
        let mut adjusted = score;

        // Regional rating tendencies
        match cultural.region {
            CulturalRegion::EastAsia => adjusted *= 0.95, // More conservative
            CulturalRegion::NorthAmerica => {}            // Neutral
            CulturalRegion::Europe => adjusted *= 1.02,   // Slightly higher
            _ => adjusted *= 0.98,                        // Slightly conservative
        }

        // Musical training effect
        if cultural.musical_training {
            adjusted *= 0.93; // More critical with musical training
        }

        adjusted
    }

    /// Apply hearing effects to score
    fn apply_hearing_effects(&self, score: f32, hearing: &HearingProfile) -> f32 {
        let mut adjusted = score;

        // Overall hearing acuity effect
        adjusted *= 0.5 + 0.5 * hearing.hearing_acuity;

        // Noise tolerance effect (simulated)
        adjusted *= 0.8 + 0.2 * hearing.noise_tolerance;

        adjusted
    }

    /// Apply environmental effects to score
    fn apply_environmental_effects(
        &self,
        score: f32,
        environment: &EnvironmentalConditions,
    ) -> f32 {
        let mut adjusted = score;

        // Background noise effect
        let noise_factor = if environment.noise_level > -30.0 {
            0.8 // Noisy environment reduces perceived quality
        } else {
            1.0
        };
        adjusted *= noise_factor;

        // Device quality effect
        let device_factor = match environment.device_quality {
            DeviceQuality::HighEnd => 1.0,
            DeviceQuality::Standard => 0.95,
            DeviceQuality::LowEnd => 0.85,
            DeviceQuality::Smartphone => 0.75,
        };
        adjusted *= device_factor;

        // Attention level effect
        adjusted *= 0.7 + 0.3 * environment.attention_level;

        adjusted
    }

    /// Apply bias effects to score
    fn apply_bias_effects(&self, score: f32, bias: &BiasParameters, fatigue: f32) -> f32 {
        let mut adjusted = score;

        // Rating bias
        adjusted += bias.rating_bias;

        // Consistency (add random variation)
        let variation = (1.0 - bias.consistency) * 0.2;
        let random_factor = thread_rng().gen_range(-variation..variation);
        adjusted += random_factor;

        // Extreme tendency
        if bias.extreme_tendency > 0.5 {
            if adjusted > 0.5 {
                adjusted = 0.5 + (adjusted - 0.5) * (1.0 + bias.extreme_tendency);
            } else {
                adjusted = 0.5 - (0.5 - adjusted) * (1.0 + bias.extreme_tendency);
            }
        }

        // Fatigue effect
        let fatigue_factor = 1.0 - fatigue * 0.3;
        adjusted *= fatigue_factor;

        adjusted
    }

    /// Calculate aggregate statistics
    fn calculate_aggregate_statistics(&self, scores: &[f32]) -> AggregateStatistics {
        if scores.is_empty() {
            return AggregateStatistics {
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
                confidence_interval: (0.0, 0.0),
                iqr: (0.0, 0.0),
            };
        }

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_scores.len() % 2 == 0 {
            (sorted_scores[sorted_scores.len() / 2 - 1] + sorted_scores[sorted_scores.len() / 2])
                / 2.0
        } else {
            sorted_scores[sorted_scores.len() / 2]
        };

        // 95% confidence interval
        let se = std_dev / (scores.len() as f32).sqrt();
        let confidence_interval = (mean - 1.96 * se, mean + 1.96 * se);

        // Interquartile range
        let q1_idx = sorted_scores.len() / 4;
        let q3_idx = 3 * sorted_scores.len() / 4;
        let iqr = (sorted_scores[q1_idx], sorted_scores[q3_idx]);

        AggregateStatistics {
            mean,
            std_dev,
            median,
            confidence_interval,
            iqr,
        }
    }

    /// Calculate group analysis for demographic/cultural groups
    fn calculate_group_analysis(
        &self,
        group_scores: &HashMap<String, Vec<f32>>,
    ) -> HashMap<String, f32> {
        group_scores
            .iter()
            .map(|(group, scores)| {
                let mean = if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f32>() / scores.len() as f32
                };
                (group.clone(), mean)
            })
            .collect()
    }

    /// Calculate reliability metrics
    fn calculate_reliability_metrics(&self, scores: &[f32]) -> ReliabilityMetrics {
        if scores.len() < 2 {
            return ReliabilityMetrics {
                cronbach_alpha: 0.0,
                inter_rater_reliability: 0.0,
                agreement_percentage: 0.0,
                consistency_index: 0.0,
            };
        }

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;

        // Simplified Cronbach's alpha
        let cronbach_alpha = if variance > 0.0 {
            let item_variance =
                scores.iter().map(|&x| x.powi(2)).sum::<f32>() / scores.len() as f32;
            let k = scores.len() as f32;
            ((k / (k - 1.0)) * (1.0 - item_variance / variance))
                .max(0.0)
                .min(1.0)
        } else {
            1.0
        };

        // Inter-rater reliability (simplified as 1 - coefficient of variation)
        let inter_rater_reliability = if mean > 0.0 {
            (1.0 - variance.sqrt() / mean).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Agreement percentage (within 0.2 of mean)
        let agreement_count = scores.iter().filter(|&&x| (x - mean).abs() < 0.2).count();
        let agreement_percentage = agreement_count as f32 / scores.len() as f32;

        // Consistency index
        let consistency_index = (cronbach_alpha * 0.5 + inter_rater_reliability * 0.5).min(1.0);

        ReliabilityMetrics {
            cronbach_alpha,
            inter_rater_reliability,
            agreement_percentage,
            consistency_index,
        }
    }

    /// Get current listener population statistics
    pub fn get_population_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        // Age group distribution
        for listener in &self.listeners {
            let age_key = format!("age_{:?}", listener.demographic.age_group);
            *stats.entry(age_key).or_insert(0) += 1;

            let exp_key = format!("experience_{:?}", listener.demographic.audio_experience);
            *stats.entry(exp_key).or_insert(0) += 1;

            let region_key = format!("region_{:?}", listener.cultural.region);
            *stats.entry(region_key).or_insert(0) += 1;
        }

        stats
    }

    /// Reset all listeners' fatigue levels
    pub fn reset_fatigue(&mut self) {
        for listener in &mut self.listeners {
            listener.fatigue_level = 0.0;
        }
    }

    /// Get evaluation history
    pub fn get_evaluation_history(&self) -> &[MultiListenerResults] {
        &self.evaluation_history
    }

    /// Get cross-cultural model information
    pub fn get_cross_cultural_info(&self) -> Option<(Vec<String>, Vec<CulturalRegion>)> {
        self.cross_cultural_model.as_ref().map(|model| {
            (
                model.get_supported_languages(),
                model.get_cultural_regions(),
            )
        })
    }

    /// Check if cross-cultural modeling is enabled
    pub fn is_cross_cultural_enabled(&self) -> bool {
        self.cross_cultural_model.is_some()
    }

    /// Update target language for cross-cultural evaluation
    pub fn set_target_language(&mut self, language: String) {
        self.config.target_language = language;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_multi_listener_creation() {
        let config = MultiListenerConfig {
            num_listeners: 5,
            ..Default::default()
        };
        let simulator = EnhancedMultiListenerSimulator::new(config);

        assert_eq!(simulator.listeners.len(), 5);
    }

    #[tokio::test]
    async fn test_listening_test_simulation() {
        let config = MultiListenerConfig {
            num_listeners: 3,
            random_seed: Some(42), // For reproducible tests
            ..Default::default()
        };
        let mut simulator = EnhancedMultiListenerSimulator::new(config);

        let samples = (0..1000)
            .map(|i| [0.1, 0.2, -0.1, -0.2][i % 4])
            .collect::<Vec<f32>>();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let results = simulator
            .simulate_listening_test(&audio, None)
            .await
            .unwrap();

        assert_eq!(results.individual_scores.len(), 3);
        assert!(results.aggregate_stats.mean >= 0.0 && results.aggregate_stats.mean <= 1.0);
        assert!(results.reliability_metrics.consistency_index >= 0.0);
    }

    #[test]
    fn test_demographic_diversity() {
        let config = MultiListenerConfig {
            num_listeners: 10,
            enable_demographic_diversity: true,
            random_seed: Some(123),
            ..Default::default()
        };
        let simulator = EnhancedMultiListenerSimulator::new(config);

        // Check that we have diversity in age groups
        let age_groups: std::collections::HashSet<_> = simulator
            .listeners
            .iter()
            .map(|l| l.demographic.age_group)
            .collect();

        assert!(age_groups.len() > 1); // Should have different age groups
    }

    #[test]
    fn test_population_statistics() {
        let config = MultiListenerConfig {
            num_listeners: 8,
            ..Default::default()
        };
        let simulator = EnhancedMultiListenerSimulator::new(config);

        let stats = simulator.get_population_statistics();

        // Should have age distribution
        let age_counts: usize = stats
            .iter()
            .filter(|(k, _)| k.starts_with("age_"))
            .map(|(_, &v)| v)
            .sum();

        assert_eq!(age_counts, 8); // All listeners accounted for
    }

    #[tokio::test]
    async fn test_cross_cultural_modeling() {
        let config = MultiListenerConfig {
            num_listeners: 5,
            enable_cross_cultural_modeling: true,
            target_language: "zh".to_string(), // Chinese target
            random_seed: Some(42),
            ..Default::default()
        };
        let mut simulator = EnhancedMultiListenerSimulator::new(config);

        // Verify cross-cultural model is enabled
        assert!(simulator.is_cross_cultural_enabled());

        // Get supported languages and regions
        let (languages, regions) = simulator.get_cross_cultural_info().unwrap();
        assert!(!languages.is_empty());
        assert!(!regions.is_empty());

        let samples = (0..1000)
            .map(|i| [0.1, 0.2, -0.1, -0.2][i % 4])
            .collect::<Vec<f32>>();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let results = simulator
            .simulate_listening_test(&audio, None)
            .await
            .unwrap();

        // Results should be valid
        assert_eq!(results.individual_scores.len(), 5);
        assert!(results.aggregate_stats.mean >= 0.0 && results.aggregate_stats.mean <= 1.0);

        // Test language switching
        simulator.set_target_language("en".to_string());
        let results_en = simulator
            .simulate_listening_test(&audio, None)
            .await
            .unwrap();

        // Results might be different due to cross-cultural adaptation
        assert_eq!(results_en.individual_scores.len(), 5);
    }

    #[test]
    fn test_cross_cultural_disabled() {
        let config = MultiListenerConfig {
            num_listeners: 3,
            enable_cross_cultural_modeling: false,
            ..Default::default()
        };
        let simulator = EnhancedMultiListenerSimulator::new(config);

        assert!(!simulator.is_cross_cultural_enabled());
        assert!(simulator.get_cross_cultural_info().is_none());
    }
}
