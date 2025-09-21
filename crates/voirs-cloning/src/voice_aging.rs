//! Voice aging system for modeling temporal voice changes
//!
//! This module provides functionality for modeling how voices change over time
//! due to natural aging processes, enabling voice aging progression and regression.

use crate::{Result, SpeakerData, VoiceSample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Age-related voice characteristic changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAgingModel {
    /// Current age estimation
    pub current_age: f32,
    /// Target age for progression/regression
    pub target_age: Option<f32>,
    /// Age-related characteristic changes
    pub aging_characteristics: AgingCharacteristics,
    /// Temporal progression model
    pub progression_model: TemporalModel,
    /// Voice stability factors
    pub stability_factors: StabilityFactors,
    /// Aging configuration
    pub config: VoiceAgingConfig,
}

/// Age-related voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingCharacteristics {
    /// Fundamental frequency changes (Hz change per year)
    pub f0_change_rate: f32,
    /// Formant frequency shifts
    pub formant_shifts: FormantAging,
    /// Voice quality degradation
    pub quality_changes: VoiceQualityAging,
    /// Prosodic changes
    pub prosodic_changes: ProsodicAging,
    /// Articulatory changes
    pub articulatory_changes: ArticulatoryAging,
    /// Respiratory changes
    pub respiratory_changes: RespiratoryAging,
}

/// Formant frequency aging patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantAging {
    /// F1 frequency change rate (Hz/year)
    pub f1_change_rate: f32,
    /// F2 frequency change rate (Hz/year)
    pub f2_change_rate: f32,
    /// F3 frequency change rate (Hz/year)
    pub f3_change_rate: f32,
    /// Formant bandwidth changes
    pub bandwidth_changes: Vec<f32>,
    /// Vocal tract length changes
    pub tract_length_factor: f32,
}

/// Voice quality aging factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityAging {
    /// Roughness increase rate
    pub roughness_rate: f32,
    /// Breathiness increase rate
    pub breathiness_rate: f32,
    /// Tremor development rate
    pub tremor_rate: f32,
    /// Hoarseness development
    pub hoarseness_rate: f32,
    /// Harmonics-to-noise ratio degradation
    pub hnr_degradation_rate: f32,
    /// Jitter increase rate
    pub jitter_rate: f32,
    /// Shimmer increase rate
    pub shimmer_rate: f32,
}

/// Prosodic aging patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicAging {
    /// Speech rate changes (syllables/second change per year)
    pub speech_rate_change: f32,
    /// Pause duration increase rate
    pub pause_duration_rate: f32,
    /// Intonation range reduction rate
    pub intonation_range_rate: f32,
    /// Stress pattern changes
    pub stress_pattern_changes: f32,
    /// Rhythm regularity changes
    pub rhythm_changes: f32,
}

/// Articulatory aging effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulatoryAging {
    /// Consonant precision degradation rate
    pub consonant_precision_rate: f32,
    /// Vowel space compression rate
    pub vowel_space_rate: f32,
    /// Articulation speed changes
    pub articulation_speed_rate: f32,
    /// Tongue movement limitation
    pub tongue_mobility_rate: f32,
    /// Lip movement changes
    pub lip_mobility_rate: f32,
}

/// Respiratory aging effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryAging {
    /// Breath support degradation rate
    pub breath_support_rate: f32,
    /// Vocal intensity reduction rate
    pub intensity_reduction_rate: f32,
    /// Breath group length changes
    pub breath_group_rate: f32,
    /// Subglottal pressure changes
    pub subglottal_pressure_rate: f32,
}

/// Temporal voice aging model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalModel {
    /// Aging curve type
    pub curve_type: AgingCurveType,
    /// Age-specific transition points
    pub transition_points: Vec<AgeTransition>,
    /// Aging rate modifiers
    pub rate_modifiers: HashMap<String, f32>,
    /// Individual variation factors
    pub variation_factors: VariationFactors,
}

/// Types of aging curves
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgingCurveType {
    /// Linear aging progression
    Linear,
    /// Exponential aging (accelerating)
    Exponential,
    /// Logarithmic aging (decelerating)
    Logarithmic,
    /// Sigmoid aging (S-curve)
    Sigmoid,
    /// Custom curve with control points
    Custom(Vec<(f32, f32)>), // (age, aging_factor) pairs
}

/// Age transition points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeTransition {
    /// Age at transition
    pub age: f32,
    /// Transition type
    pub transition_type: TransitionType,
    /// Magnitude of change
    pub magnitude: f32,
    /// Duration of transition (years)
    pub duration: f32,
}

/// Types of age transitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionType {
    /// Puberty voice change
    Puberty,
    /// Adult voice stabilization
    AdultStabilization,
    /// Middle age changes
    MiddleAge,
    /// Senior voice changes
    Senior,
    /// Menopause (for female voices)
    Menopause,
    /// Custom transition
    Custom(String),
}

/// Individual variation factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationFactors {
    /// Genetic aging factors
    pub genetic_factors: f32,
    /// Lifestyle impact on aging
    pub lifestyle_factors: f32,
    /// Health status impact
    pub health_factors: f32,
    /// Vocal usage patterns
    pub usage_factors: f32,
    /// Environmental factors
    pub environmental_factors: f32,
}

/// Voice stability factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityFactors {
    /// Natural voice stability (resistance to aging)
    pub natural_stability: f32,
    /// Training/conditioning effects
    pub conditioning_stability: f32,
    /// Health-based stability
    pub health_stability: f32,
    /// Usage-based wear patterns
    pub usage_patterns: HashMap<String, f32>,
}

/// Voice aging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAgingConfig {
    /// Enable realistic aging constraints
    pub realistic_constraints: bool,
    /// Maximum aging rate per year
    pub max_aging_rate: f32,
    /// Minimum aging rate per year
    pub min_aging_rate: f32,
    /// Enable gender-specific aging patterns
    pub gender_specific: bool,
    /// Enable individual variation
    pub individual_variation: bool,
    /// Aging calculation precision
    pub precision: f32,
    /// Enable temporal smoothing
    pub temporal_smoothing: bool,
}

/// Voice aging result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAgingResult {
    /// Aged speaker data
    pub aged_speaker: SpeakerData,
    /// Applied aging factors
    pub aging_factors: AgingFactors,
    /// Quality assessment of aging
    pub aging_quality: AgingQuality,
    /// Processing statistics
    pub statistics: AgingStatistics,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Applied aging factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingFactors {
    /// Age progression amount (in years)
    pub age_progression: f32,
    /// F0 change amount (Hz)
    pub f0_change: f32,
    /// Formant shifts applied
    pub formant_changes: Vec<f32>,
    /// Voice quality changes
    pub quality_changes: HashMap<String, f32>,
    /// Prosodic modifications
    pub prosodic_changes: HashMap<String, f32>,
    /// Overall aging factor (0.0 to 1.0)
    pub overall_factor: f32,
}

/// Aging quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingQuality {
    /// Realism score (0.0 to 1.0)
    pub realism_score: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness_score: f32,
    /// Age accuracy score (0.0 to 1.0)
    pub age_accuracy: f32,
    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f32,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
}

/// Aging processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingStatistics {
    /// Processing time
    pub processing_time: Duration,
    /// Number of characteristics modified
    pub characteristics_modified: usize,
    /// Aging complexity score
    pub complexity_score: f32,
    /// Memory usage (MB)
    pub memory_usage: f32,
    /// Success rate
    pub success_rate: f32,
}

/// Voice aging engine
pub struct VoiceAgingEngine {
    /// Aging models by speaker
    aging_models: HashMap<String, VoiceAgingModel>,
    /// Default aging configuration
    default_config: VoiceAgingConfig,
    /// Aging presets
    aging_presets: HashMap<String, AgingCharacteristics>,
}

impl VoiceAgingEngine {
    /// Create new voice aging engine
    pub fn new() -> Self {
        let mut engine = Self {
            aging_models: HashMap::new(),
            default_config: VoiceAgingConfig::default(),
            aging_presets: HashMap::new(),
        };

        engine.initialize_presets();
        engine
    }

    /// Initialize aging presets
    fn initialize_presets(&mut self) {
        // Male aging preset
        self.aging_presets.insert(
            "male_standard".to_string(),
            AgingCharacteristics {
                f0_change_rate: -0.8, // F0 decreases with age for males
                formant_shifts: FormantAging {
                    f1_change_rate: -2.0,
                    f2_change_rate: -3.0,
                    f3_change_rate: -1.5,
                    bandwidth_changes: vec![1.2, 1.3, 1.1],
                    tract_length_factor: 1.01,
                },
                quality_changes: VoiceQualityAging {
                    roughness_rate: 0.02,
                    breathiness_rate: 0.015,
                    tremor_rate: 0.008,
                    hoarseness_rate: 0.012,
                    hnr_degradation_rate: 0.3,
                    jitter_rate: 0.001,
                    shimmer_rate: 0.002,
                },
                prosodic_changes: ProsodicAging {
                    speech_rate_change: -0.1,
                    pause_duration_rate: 0.05,
                    intonation_range_rate: -0.03,
                    stress_pattern_changes: 0.02,
                    rhythm_changes: 0.015,
                },
                articulatory_changes: ArticulatoryAging {
                    consonant_precision_rate: -0.02,
                    vowel_space_rate: -0.015,
                    articulation_speed_rate: -0.02,
                    tongue_mobility_rate: -0.01,
                    lip_mobility_rate: -0.008,
                },
                respiratory_changes: RespiratoryAging {
                    breath_support_rate: -0.025,
                    intensity_reduction_rate: -0.02,
                    breath_group_rate: -0.03,
                    subglottal_pressure_rate: -0.015,
                },
            },
        );

        // Female aging preset
        self.aging_presets.insert(
            "female_standard".to_string(),
            AgingCharacteristics {
                f0_change_rate: -1.2, // F0 decreases more for females
                formant_shifts: FormantAging {
                    f1_change_rate: -1.8,
                    f2_change_rate: -4.0,
                    f3_change_rate: -2.0,
                    bandwidth_changes: vec![1.3, 1.4, 1.2],
                    tract_length_factor: 1.008,
                },
                quality_changes: VoiceQualityAging {
                    roughness_rate: 0.025,
                    breathiness_rate: 0.02,
                    tremor_rate: 0.01,
                    hoarseness_rate: 0.015,
                    hnr_degradation_rate: 0.4,
                    jitter_rate: 0.0012,
                    shimmer_rate: 0.0025,
                },
                prosodic_changes: ProsodicAging {
                    speech_rate_change: -0.08,
                    pause_duration_rate: 0.04,
                    intonation_range_rate: -0.025,
                    stress_pattern_changes: 0.018,
                    rhythm_changes: 0.012,
                },
                articulatory_changes: ArticulatoryAging {
                    consonant_precision_rate: -0.018,
                    vowel_space_rate: -0.02,
                    articulation_speed_rate: -0.018,
                    tongue_mobility_rate: -0.012,
                    lip_mobility_rate: -0.01,
                },
                respiratory_changes: RespiratoryAging {
                    breath_support_rate: -0.03,
                    intensity_reduction_rate: -0.025,
                    breath_group_rate: -0.028,
                    subglottal_pressure_rate: -0.02,
                },
            },
        );
    }

    /// Age a voice by a specific number of years
    pub fn age_voice(
        &mut self,
        speaker_data: &SpeakerData,
        age_years: f32,
    ) -> Result<VoiceAgingResult> {
        let start_time = SystemTime::now();
        let speaker_id = &speaker_data.profile.id;

        // Get or create aging model for this speaker
        let aging_model = self
            .get_or_create_aging_model(speaker_id, speaker_data)?
            .clone();

        // Calculate new age
        let new_age = aging_model.current_age + age_years;

        // Apply aging transformations
        let mut aged_speaker = speaker_data.clone();
        let aging_factors = self.calculate_aging_factors(&aging_model, age_years)?;

        // Apply aging factors to speaker characteristics
        self.apply_aging_to_speaker(&mut aged_speaker, &aging_factors)?;

        // Update aging model
        if let Some(model) = self.aging_models.get_mut(speaker_id) {
            model.current_age = new_age;
        }

        // Assess aging quality
        let aging_quality = self.assess_aging_quality(&aged_speaker, &aging_factors)?;

        // Calculate statistics
        let processing_time = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));
        let statistics = AgingStatistics {
            processing_time,
            characteristics_modified: self.count_modified_characteristics(&aging_factors),
            complexity_score: self.calculate_complexity_score(&aging_factors),
            memory_usage: 15.0, // Mock value
            success_rate: 0.95, // Mock value
        };

        Ok(VoiceAgingResult {
            aged_speaker,
            aging_factors,
            aging_quality,
            statistics,
            timestamp: SystemTime::now(),
        })
    }

    /// Age voice to a specific target age
    pub fn age_voice_to_target(
        &mut self,
        speaker_data: &SpeakerData,
        target_age: f32,
    ) -> Result<VoiceAgingResult> {
        let speaker_id = &speaker_data.profile.id;
        let aging_model = self.get_or_create_aging_model(speaker_id, speaker_data)?;
        let age_difference = target_age - aging_model.current_age;

        self.age_voice(speaker_data, age_difference)
    }

    /// Get or create aging model for speaker
    fn get_or_create_aging_model(
        &mut self,
        speaker_id: &str,
        speaker_data: &SpeakerData,
    ) -> Result<&VoiceAgingModel> {
        if !self.aging_models.contains_key(speaker_id) {
            let aging_model = self.create_aging_model_for_speaker(speaker_data)?;
            self.aging_models
                .insert(speaker_id.to_string(), aging_model);
        }

        self.aging_models
            .get(speaker_id)
            .ok_or_else(|| crate::Error::Processing("Failed to get aging model".to_string()))
    }

    /// Create aging model for a speaker
    fn create_aging_model_for_speaker(
        &self,
        speaker_data: &SpeakerData,
    ) -> Result<VoiceAgingModel> {
        // Estimate current age from voice characteristics
        let estimated_age = self.estimate_age_from_voice(speaker_data)?;

        // Select aging characteristics based on gender/voice type
        let aging_characteristics = self.select_aging_characteristics(speaker_data)?;

        // Create temporal model
        let progression_model = self.create_temporal_model(estimated_age);

        // Initialize stability factors
        let stability_factors = self.calculate_stability_factors(speaker_data);

        Ok(VoiceAgingModel {
            current_age: estimated_age,
            target_age: None,
            aging_characteristics,
            progression_model,
            stability_factors,
            config: self.default_config.clone(),
        })
    }

    /// Estimate age from voice characteristics
    fn estimate_age_from_voice(&self, speaker_data: &SpeakerData) -> Result<f32> {
        // Use acoustic features to estimate age
        let characteristics = &speaker_data.profile.characteristics;

        // Basic age estimation based on F0 and voice quality
        let f0_estimate = characteristics
            .adaptive_features
            .get("fundamental_frequency")
            .unwrap_or(&200.0);

        let roughness = characteristics
            .adaptive_features
            .get("roughness")
            .unwrap_or(&0.1);

        let breathiness = characteristics
            .adaptive_features
            .get("breathiness")
            .unwrap_or(&0.1);

        // Simple age estimation model
        let mut estimated_age = 30.0; // Default young adult

        // F0-based age estimation (rough approximation)
        if *f0_estimate < 120.0 {
            estimated_age += 10.0; // Older male
        } else if *f0_estimate > 250.0 {
            estimated_age -= 5.0; // Younger voice
        }

        // Voice quality-based adjustments
        estimated_age += roughness * 50.0; // Roughness indicates aging
        estimated_age += breathiness * 30.0; // Breathiness indicates aging

        // Clamp to reasonable range
        Ok(estimated_age.clamp(18.0, 80.0))
    }

    /// Select aging characteristics based on speaker
    fn select_aging_characteristics(
        &self,
        speaker_data: &SpeakerData,
    ) -> Result<AgingCharacteristics> {
        // Determine gender/voice type
        let voice_type = speaker_data
            .profile
            .characteristics
            .adaptive_features
            .get("voice_type")
            .unwrap_or(&0.5); // 0.0 = male, 1.0 = female, 0.5 = neutral

        let preset_name = if *voice_type < 0.3 {
            "male_standard"
        } else if *voice_type > 0.7 {
            "female_standard"
        } else {
            "male_standard" // Default to male for neutral
        };

        self.aging_presets.get(preset_name).cloned().ok_or_else(|| {
            crate::Error::Processing(format!("Aging preset not found: {preset_name}"))
        })
    }

    /// Create temporal model for aging
    fn create_temporal_model(&self, current_age: f32) -> TemporalModel {
        let mut transition_points = Vec::new();

        // Define typical voice transition points
        if current_age < 25.0 {
            transition_points.push(AgeTransition {
                age: 25.0,
                transition_type: TransitionType::AdultStabilization,
                magnitude: 0.3,
                duration: 2.0,
            });
        }

        if current_age < 45.0 {
            transition_points.push(AgeTransition {
                age: 45.0,
                transition_type: TransitionType::MiddleAge,
                magnitude: 0.5,
                duration: 5.0,
            });
        }

        if current_age < 65.0 {
            transition_points.push(AgeTransition {
                age: 65.0,
                transition_type: TransitionType::Senior,
                magnitude: 0.8,
                duration: 10.0,
            });
        }

        TemporalModel {
            curve_type: AgingCurveType::Sigmoid,
            transition_points,
            rate_modifiers: HashMap::new(),
            variation_factors: VariationFactors {
                genetic_factors: 1.0,
                lifestyle_factors: 1.0,
                health_factors: 1.0,
                usage_factors: 1.0,
                environmental_factors: 1.0,
            },
        }
    }

    /// Calculate stability factors
    fn calculate_stability_factors(&self, speaker_data: &SpeakerData) -> StabilityFactors {
        StabilityFactors {
            natural_stability: 0.8,      // Default stability
            conditioning_stability: 0.9, // Well-trained voice
            health_stability: 0.85,      // Good health
            usage_patterns: HashMap::new(),
        }
    }

    /// Calculate aging factors
    fn calculate_aging_factors(
        &self,
        aging_model: &VoiceAgingModel,
        age_years: f32,
    ) -> Result<AgingFactors> {
        let aging_curve = self.calculate_aging_curve_factor(aging_model, age_years);
        let characteristics = &aging_model.aging_characteristics;

        // Calculate F0 change
        let f0_change = characteristics.f0_change_rate * age_years * aging_curve;

        // Calculate formant changes
        let formant_changes = vec![
            characteristics.formant_shifts.f1_change_rate * age_years * aging_curve,
            characteristics.formant_shifts.f2_change_rate * age_years * aging_curve,
            characteristics.formant_shifts.f3_change_rate * age_years * aging_curve,
        ];

        // Calculate voice quality changes
        let mut quality_changes = HashMap::new();
        quality_changes.insert(
            "roughness".to_string(),
            characteristics.quality_changes.roughness_rate * age_years * aging_curve,
        );
        quality_changes.insert(
            "breathiness".to_string(),
            characteristics.quality_changes.breathiness_rate * age_years * aging_curve,
        );
        quality_changes.insert(
            "tremor".to_string(),
            characteristics.quality_changes.tremor_rate * age_years * aging_curve,
        );

        // Calculate prosodic changes
        let mut prosodic_changes = HashMap::new();
        prosodic_changes.insert(
            "speech_rate".to_string(),
            characteristics.prosodic_changes.speech_rate_change * age_years * aging_curve,
        );
        prosodic_changes.insert(
            "pause_duration".to_string(),
            characteristics.prosodic_changes.pause_duration_rate * age_years * aging_curve,
        );

        Ok(AgingFactors {
            age_progression: age_years,
            f0_change,
            formant_changes,
            quality_changes,
            prosodic_changes,
            overall_factor: aging_curve,
        })
    }

    /// Calculate aging curve factor
    fn calculate_aging_curve_factor(&self, aging_model: &VoiceAgingModel, age_years: f32) -> f32 {
        match &aging_model.progression_model.curve_type {
            AgingCurveType::Linear => age_years.abs(),
            AgingCurveType::Exponential => 1.0 - (-age_years.abs() * 0.1).exp(),
            AgingCurveType::Logarithmic => (1.0 + age_years.abs()).ln(),
            AgingCurveType::Sigmoid => 1.0 / (1.0 + (-age_years * 0.2).exp()),
            AgingCurveType::Custom(_) => age_years.abs(), // Simplified for now
        }
    }

    /// Apply aging factors to speaker
    fn apply_aging_to_speaker(
        &self,
        speaker: &mut SpeakerData,
        factors: &AgingFactors,
    ) -> Result<()> {
        let characteristics = &mut speaker.profile.characteristics.adaptive_features;

        // Apply F0 changes
        if let Some(f0) = characteristics.get_mut("fundamental_frequency") {
            *f0 += factors.f0_change;
            *f0 = f0.clamp(50.0, 500.0); // Reasonable F0 range
        }

        // Apply formant changes
        for (i, formant_change) in factors.formant_changes.iter().enumerate() {
            let formant_key = format!("formant_f{num}", num = i + 1);
            if let Some(formant) = characteristics.get_mut(&formant_key) {
                *formant += formant_change;
                *formant = formant.clamp(200.0, 4000.0); // Reasonable formant range
            }
        }

        // Apply voice quality changes
        for (quality, change) in &factors.quality_changes {
            if let Some(current_value) = characteristics.get_mut(quality) {
                *current_value += change;
                *current_value = current_value.clamp(0.0, 1.0);
            } else {
                characteristics.insert(quality.clone(), change.clamp(0.0, 1.0));
            }
        }

        // Apply prosodic changes
        for (prosody, change) in &factors.prosodic_changes {
            let prosody_key = format!("prosody_{prosody}");
            if let Some(current_value) = characteristics.get_mut(&prosody_key) {
                *current_value += change;
                *current_value = current_value.max(0.1); // Minimum positive values
            } else {
                characteristics.insert(prosody_key, change.max(0.1));
            }
        }

        // Update metadata
        speaker.profile.metadata.insert(
            "aged_years".to_string(),
            factors.age_progression.to_string(),
        );
        speaker.profile.metadata.insert(
            "aging_factor".to_string(),
            factors.overall_factor.to_string(),
        );
        speaker.profile.metadata.insert(
            "aging_timestamp".to_string(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
                .to_string(),
        );

        Ok(())
    }

    /// Assess aging quality
    fn assess_aging_quality(
        &self,
        aged_speaker: &SpeakerData,
        factors: &AgingFactors,
    ) -> Result<AgingQuality> {
        // Calculate realism score based on aging factors
        let realism_score = self.calculate_realism_score(factors);

        // Calculate naturalness score
        let naturalness_score = self.calculate_naturalness_score(aged_speaker);

        // Calculate age accuracy
        let age_accuracy = self.calculate_age_accuracy(factors);

        // Calculate consistency score
        let consistency_score = self.calculate_consistency_score(factors);

        // Overall quality score
        let overall_quality =
            (realism_score + naturalness_score + age_accuracy + consistency_score) / 4.0;

        Ok(AgingQuality {
            realism_score,
            naturalness_score,
            age_accuracy,
            consistency_score,
            overall_quality,
        })
    }

    /// Calculate realism score
    fn calculate_realism_score(&self, factors: &AgingFactors) -> f32 {
        // Check if aging factors are within realistic ranges
        let mut score: f32 = 1.0;

        // Check F0 change realism
        if factors.f0_change.abs() > 50.0 {
            // Very large F0 change
            score -= 0.2;
        }

        // Check formant change realism
        for formant_change in &factors.formant_changes {
            if formant_change.abs() > 200.0 {
                // Very large formant change
                score -= 0.1;
            }
        }

        // Check quality change realism
        for (_, change) in &factors.quality_changes {
            if change.abs() > 0.5 {
                // Very large quality change
                score -= 0.1;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate naturalness score
    fn calculate_naturalness_score(&self, aged_speaker: &SpeakerData) -> f32 {
        // Assess if the aged voice sounds natural
        let characteristics = &aged_speaker.profile.characteristics.adaptive_features;

        // Check for reasonable voice characteristics
        let f0 = characteristics
            .get("fundamental_frequency")
            .unwrap_or(&150.0);
        let roughness = characteristics.get("roughness").unwrap_or(&0.1);
        let breathiness = characteristics.get("breathiness").unwrap_or(&0.1);

        let mut score: f32 = 1.0;

        // Check F0 naturalness
        if *f0 < 80.0 || *f0 > 350.0 {
            score -= 0.3;
        }

        // Check voice quality naturalness
        if *roughness > 0.8 || *breathiness > 0.8 {
            score -= 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate age accuracy
    fn calculate_age_accuracy(&self, factors: &AgingFactors) -> f32 {
        // Assess if the aging progression is accurate for the age change
        let expected_factor = factors.age_progression.abs() / 10.0; // Normalize by decade
        let actual_factor = factors.overall_factor;

        let accuracy = 1.0 - (expected_factor - actual_factor).abs();
        accuracy.clamp(0.0, 1.0)
    }

    /// Calculate consistency score
    fn calculate_consistency_score(&self, factors: &AgingFactors) -> f32 {
        // Check if all aging factors are consistent with each other
        let mut score: f32 = 1.0;

        // Check if quality changes are consistent with age progression
        let expected_quality_change = factors.age_progression.abs() * 0.02; // 2% per year
        let actual_quality_change: f32 =
            factors.quality_changes.values().sum::<f32>() / factors.quality_changes.len() as f32;

        if (expected_quality_change - actual_quality_change).abs() > 0.1 {
            score -= 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    /// Count modified characteristics
    fn count_modified_characteristics(&self, factors: &AgingFactors) -> usize {
        1 + // F0 change
        factors.formant_changes.len() +
        factors.quality_changes.len() +
        factors.prosodic_changes.len()
    }

    /// Calculate complexity score
    fn calculate_complexity_score(&self, factors: &AgingFactors) -> f32 {
        let num_changes = self.count_modified_characteristics(factors);
        let magnitude = factors.overall_factor;

        (num_changes as f32 * magnitude).min(10.0) / 10.0
    }

    /// Get aging model for speaker
    pub fn get_aging_model(&self, speaker_id: &str) -> Option<&VoiceAgingModel> {
        self.aging_models.get(speaker_id)
    }

    /// Update aging model configuration
    pub fn update_aging_config(
        &mut self,
        speaker_id: &str,
        config: VoiceAgingConfig,
    ) -> Result<()> {
        if let Some(model) = self.aging_models.get_mut(speaker_id) {
            model.config = config;
            Ok(())
        } else {
            Err(crate::Error::InvalidInput(format!(
                "Speaker not found: {speaker_id}"
            )))
        }
    }

    /// Get all aged speakers
    pub fn get_aged_speakers(&self) -> Vec<String> {
        self.aging_models.keys().cloned().collect()
    }
}

impl Default for VoiceAgingConfig {
    fn default() -> Self {
        Self {
            realistic_constraints: true,
            max_aging_rate: 2.0,
            min_aging_rate: 0.1,
            gender_specific: true,
            individual_variation: true,
            precision: 0.1,
            temporal_smoothing: true,
        }
    }
}

impl Default for VariationFactors {
    fn default() -> Self {
        Self {
            genetic_factors: 1.0,
            lifestyle_factors: 1.0,
            health_factors: 1.0,
            usage_factors: 1.0,
            environmental_factors: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SpeakerCharacteristics, SpeakerProfile};
    use std::collections::HashMap;

    fn create_test_speaker() -> SpeakerData {
        use crate::types::{AgeGroup, Gender, VoiceQuality};

        let mut adaptive_features = HashMap::new();
        adaptive_features.insert("fundamental_frequency".to_string(), 200.0);
        adaptive_features.insert("roughness".to_string(), 0.1);
        adaptive_features.insert("breathiness".to_string(), 0.1);
        adaptive_features.insert("voice_type".to_string(), 0.5); // Neutral

        SpeakerData {
            profile: SpeakerProfile {
                id: "test_speaker".to_string(),
                name: "Test Speaker".to_string(),
                characteristics: SpeakerCharacteristics {
                    average_pitch: 200.0,
                    pitch_range: 12.0,
                    average_energy: 0.5,
                    speaking_rate: 150.0,
                    voice_quality: VoiceQuality {
                        roughness: 0.1,
                        breathiness: 0.1,
                        brightness: 0.5,
                        warmth: 0.5,
                        resonance: crate::types::ResonanceProfile::default(),
                    },
                    accent: Some("neutral".to_string()),
                    gender: Some(Gender::Other),
                    age_group: Some(AgeGroup::YoungAdult),
                    adaptive_features,
                },
                samples: Vec::new(),
                embedding: Some(vec![0.1; 512]),
                languages: vec!["en".to_string()],
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            },
            reference_samples: Vec::new(),
            target_text: None,
            target_language: None,
            context: HashMap::new(),
        }
    }

    #[test]
    fn test_voice_aging_engine_creation() {
        let engine = VoiceAgingEngine::new();
        assert!(engine.aging_presets.contains_key("male_standard"));
        assert!(engine.aging_presets.contains_key("female_standard"));
    }

    #[test]
    fn test_age_estimation() {
        let engine = VoiceAgingEngine::new();
        let speaker = create_test_speaker();

        let estimated_age = engine.estimate_age_from_voice(&speaker).unwrap();
        assert!(estimated_age >= 18.0 && estimated_age <= 80.0);
    }

    #[test]
    fn test_aging_characteristics_selection() {
        let engine = VoiceAgingEngine::new();
        let speaker = create_test_speaker();

        let characteristics = engine.select_aging_characteristics(&speaker).unwrap();
        assert!(characteristics.f0_change_rate != 0.0);
        assert!(characteristics.quality_changes.roughness_rate > 0.0);
    }

    #[test]
    fn test_voice_aging() {
        let mut engine = VoiceAgingEngine::new();
        let speaker = create_test_speaker();

        let result = engine.age_voice(&speaker, 10.0).unwrap();
        assert_eq!(result.aging_factors.age_progression, 10.0);
        assert!(result.aging_quality.overall_quality > 0.0);
        assert!(result.statistics.characteristics_modified > 0);
    }

    #[test]
    fn test_voice_aging_to_target() {
        let mut engine = VoiceAgingEngine::new();
        let speaker = create_test_speaker();

        let result = engine.age_voice_to_target(&speaker, 50.0).unwrap();
        assert!(result.aging_factors.age_progression != 0.0);
        assert!(result.aging_quality.overall_quality > 0.0);
    }

    #[test]
    fn test_aging_curve_types() {
        let engine = VoiceAgingEngine::new();
        let model = VoiceAgingModel {
            current_age: 30.0,
            target_age: None,
            aging_characteristics: engine.aging_presets["male_standard"].clone(),
            progression_model: TemporalModel {
                curve_type: AgingCurveType::Linear,
                transition_points: Vec::new(),
                rate_modifiers: HashMap::new(),
                variation_factors: VariationFactors::default(),
            },
            stability_factors: StabilityFactors {
                natural_stability: 0.8,
                conditioning_stability: 0.9,
                health_stability: 0.85,
                usage_patterns: HashMap::new(),
            },
            config: VoiceAgingConfig::default(),
        };

        let linear_factor = engine.calculate_aging_curve_factor(&model, 5.0);
        assert_eq!(linear_factor, 5.0);
    }

    #[test]
    fn test_aging_factors_calculation() {
        let engine = VoiceAgingEngine::new();
        let model = VoiceAgingModel {
            current_age: 30.0,
            target_age: None,
            aging_characteristics: engine.aging_presets["male_standard"].clone(),
            progression_model: TemporalModel {
                curve_type: AgingCurveType::Linear,
                transition_points: Vec::new(),
                rate_modifiers: HashMap::new(),
                variation_factors: VariationFactors::default(),
            },
            stability_factors: StabilityFactors {
                natural_stability: 0.8,
                conditioning_stability: 0.9,
                health_stability: 0.85,
                usage_patterns: HashMap::new(),
            },
            config: VoiceAgingConfig::default(),
        };

        let factors = engine.calculate_aging_factors(&model, 10.0).unwrap();
        assert_eq!(factors.age_progression, 10.0);
        assert!(factors.f0_change != 0.0);
        assert_eq!(factors.formant_changes.len(), 3);
        assert!(!factors.quality_changes.is_empty());
    }

    #[test]
    fn test_aging_quality_assessment() {
        let engine = VoiceAgingEngine::new();
        let speaker = create_test_speaker();
        let factors = AgingFactors {
            age_progression: 10.0,
            f0_change: -8.0,
            formant_changes: vec![-20.0, -30.0, -15.0],
            quality_changes: {
                let mut map = HashMap::new();
                map.insert("roughness".to_string(), 0.2);
                map
            },
            prosodic_changes: HashMap::new(),
            overall_factor: 1.0,
        };

        let quality = engine.assess_aging_quality(&speaker, &factors).unwrap();
        assert!(quality.realism_score >= 0.0 && quality.realism_score <= 1.0);
        assert!(quality.naturalness_score >= 0.0 && quality.naturalness_score <= 1.0);
        assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
    }

    #[test]
    fn test_aging_configuration() {
        let config = VoiceAgingConfig::default();
        assert!(config.realistic_constraints);
        assert!(config.gender_specific);
        assert!(config.individual_variation);
        assert!(config.temporal_smoothing);
    }

    #[test]
    fn test_transition_types() {
        let transition = AgeTransition {
            age: 65.0,
            transition_type: TransitionType::Senior,
            magnitude: 0.8,
            duration: 10.0,
        };

        assert_eq!(transition.age, 65.0);
        assert_eq!(transition.transition_type, TransitionType::Senior);
        assert_eq!(transition.magnitude, 0.8);
        assert_eq!(transition.duration, 10.0);
    }
}
