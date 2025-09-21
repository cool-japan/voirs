//! Natural Variation System for Realistic Emotional Micro-variations
//!
//! This module provides tools for adding realistic micro-variations to emotional
//! expression, making synthesized speech sound more natural and less robotic.

use crate::types::{Emotion, EmotionIntensity, EmotionParameters};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Configuration for natural variation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalVariationConfig {
    /// Base variation intensity (0.0 to 1.0)
    pub base_variation_intensity: f32,
    /// Temporal variation frequency (Hz)
    pub temporal_frequency: f32,
    /// Enable prosodic micro-variations
    pub enable_prosodic_variation: bool,
    /// Enable voice quality variations
    pub enable_voice_quality_variation: bool,
    /// Enable breathing pattern simulation
    pub enable_breathing_patterns: bool,
    /// Emotion-specific variation scaling
    pub emotion_scaling: HashMap<String, f32>,
    /// Speaker characteristics influence
    pub speaker_characteristics_influence: f32,
    /// Random seed for reproducible variations
    pub random_seed: Option<u64>,
    /// Variation smoothing factor
    pub smoothing_factor: f32,
}

impl Default for NaturalVariationConfig {
    fn default() -> Self {
        let mut emotion_scaling = HashMap::new();
        emotion_scaling.insert("Happy".to_string(), 1.2);
        emotion_scaling.insert("Sad".to_string(), 0.8);
        emotion_scaling.insert("Angry".to_string(), 1.5);
        emotion_scaling.insert("Fear".to_string(), 1.3);
        emotion_scaling.insert("Surprise".to_string(), 1.4);
        emotion_scaling.insert("Neutral".to_string(), 0.6);

        Self {
            base_variation_intensity: 0.15, // 15% base variation
            temporal_frequency: 0.5,        // 0.5 Hz variation frequency
            enable_prosodic_variation: true,
            enable_voice_quality_variation: true,
            enable_breathing_patterns: true,
            emotion_scaling,
            speaker_characteristics_influence: 0.3,
            random_seed: None,
            smoothing_factor: 0.7,
        }
    }
}

/// Types of natural variations that can be applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariationType {
    /// Prosodic variations (pitch, timing, stress)
    Prosodic,
    /// Voice quality variations (breathiness, roughness)
    VoiceQuality,
    /// Breathing pattern variations
    Breathing,
    /// Micro-emotional fluctuations
    MicroEmotional,
    /// Speaker idiosyncrasies
    SpeakerIdiosyncratic,
}

/// Individual variation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationPattern {
    /// Pattern name
    pub name: String,
    /// Variation type
    pub variation_type: VariationType,
    /// Amplitude of variation
    pub amplitude: f32,
    /// Frequency of variation (Hz)
    pub frequency: f32,
    /// Phase offset (radians)
    pub phase_offset: f32,
    /// Duration of pattern
    pub duration: Duration,
    /// Pattern envelope shape
    pub envelope: VariationEnvelope,
}

/// Envelope shapes for variations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariationEnvelope {
    /// Linear envelope
    Linear { attack: f32, decay: f32 },
    /// Exponential envelope
    Exponential { attack: f32, decay: f32, curve: f32 },
    /// Sinusoidal envelope
    Sinusoidal { frequency: f32, amplitude: f32 },
    /// Random walk envelope
    RandomWalk { step_size: f32, bounds: (f32, f32) },
}

/// Applied variation instance
#[derive(Debug, Clone)]
pub struct AppliedVariation {
    /// Base parameters before variation
    pub base_parameters: EmotionParameters,
    /// Varied parameters after variation
    pub varied_parameters: EmotionParameters,
    /// Active variation patterns
    pub active_patterns: Vec<VariationPattern>,
    /// Variation timestamp
    pub timestamp: SystemTime,
    /// Variation strength used
    pub strength: f32,
}

/// Speaker characteristics for personalized variations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    /// Speaker age group
    pub age_group: AgeGroup,
    /// Speaker gender
    pub gender: Gender,
    /// Emotional expressiveness level (0.0 to 2.0)
    pub expressiveness: f32,
    /// Voice stability (0.0 = very unstable, 1.0 = very stable)
    pub stability: f32,
    /// Breathing pattern characteristics
    pub breathing_pattern: BreathingPattern,
    /// Personal speaking quirks
    pub quirks: Vec<SpeakingQuirk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgeGroup {
    Child,
    YoungAdult,
    MiddleAged,
    Senior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathingPattern {
    /// Breathing rate (breaths per minute)
    pub rate: f32,
    /// Breathing depth variation
    pub depth_variation: f32,
    /// Emotional breathing influence
    pub emotion_influence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakingQuirk {
    /// Quirk name
    pub name: String,
    /// Occurrence probability (0.0 to 1.0)
    pub probability: f32,
    /// Effect on parameters
    pub parameter_effect: HashMap<String, f32>,
}

impl Default for SpeakerCharacteristics {
    fn default() -> Self {
        Self {
            age_group: AgeGroup::MiddleAged,
            gender: Gender::NonBinary,
            expressiveness: 1.0,
            stability: 0.8,
            breathing_pattern: BreathingPattern {
                rate: 15.0, // 15 breaths per minute
                depth_variation: 0.2,
                emotion_influence: 0.3,
            },
            quirks: Vec::new(),
        }
    }
}

/// Natural variation generator
pub struct NaturalVariationGenerator {
    /// Configuration
    config: NaturalVariationConfig,
    /// Speaker characteristics
    speaker_characteristics: SpeakerCharacteristics,
    /// Random number generator
    rng: fastrand::Rng,
    /// Variation history for smoothing
    variation_history: std::collections::VecDeque<AppliedVariation>,
    /// Current time offset for temporal patterns
    time_offset: f32,
    /// Pre-generated variation patterns
    cached_patterns: HashMap<String, Vec<VariationPattern>>,
}

impl NaturalVariationGenerator {
    /// Create new variation generator
    pub fn new(
        config: NaturalVariationConfig,
        speaker_characteristics: SpeakerCharacteristics,
    ) -> Self {
        let rng = if let Some(seed) = config.random_seed {
            fastrand::Rng::with_seed(seed)
        } else {
            fastrand::Rng::new()
        };

        Self {
            config,
            speaker_characteristics,
            rng,
            variation_history: std::collections::VecDeque::new(),
            time_offset: 0.0,
            cached_patterns: HashMap::new(),
        }
    }

    /// Apply natural variations to emotion parameters
    pub fn apply_variations(
        &mut self,
        base_parameters: &EmotionParameters,
        emotion: &Emotion,
        intensity: EmotionIntensity,
        duration: Duration,
    ) -> crate::Result<AppliedVariation> {
        let mut varied_parameters = base_parameters.clone();
        let mut active_patterns = Vec::new();

        // Get emotion-specific scaling factor
        let emotion_name = format!("{:?}", emotion);
        let emotion_scale = self
            .config
            .emotion_scaling
            .get(&emotion_name)
            .unwrap_or(&1.0);

        let base_intensity = intensity.value();
        let variation_strength =
            self.config.base_variation_intensity * emotion_scale * base_intensity;

        // Apply different types of variations
        if self.config.enable_prosodic_variation {
            let prosodic_patterns =
                self.generate_prosodic_variations(variation_strength, duration)?;
            for pattern in &prosodic_patterns {
                self.apply_variation_pattern(&mut varied_parameters, pattern, &base_parameters)?;
            }
            active_patterns.extend(prosodic_patterns);
        }

        if self.config.enable_voice_quality_variation {
            let voice_patterns =
                self.generate_voice_quality_variations(variation_strength, duration)?;
            for pattern in &voice_patterns {
                self.apply_variation_pattern(&mut varied_parameters, pattern, &base_parameters)?;
            }
            active_patterns.extend(voice_patterns);
        }

        if self.config.enable_breathing_patterns {
            let breathing_patterns =
                self.generate_breathing_variations(variation_strength, duration)?;
            for pattern in &breathing_patterns {
                self.apply_variation_pattern(&mut varied_parameters, pattern, &base_parameters)?;
            }
            active_patterns.extend(breathing_patterns);
        }

        // Apply micro-emotional fluctuations
        self.apply_micro_emotional_variations(
            &mut varied_parameters,
            base_parameters,
            variation_strength,
        )?;

        // Apply speaker-specific characteristics
        self.apply_speaker_characteristics(
            &mut varied_parameters,
            base_parameters,
            variation_strength,
        )?;

        // Apply smoothing based on history
        self.apply_variation_smoothing(&mut varied_parameters, base_parameters)?;

        let applied_variation = AppliedVariation {
            base_parameters: base_parameters.clone(),
            varied_parameters,
            active_patterns,
            timestamp: SystemTime::now(),
            strength: variation_strength,
        };

        // Update history
        self.variation_history.push_back(applied_variation.clone());
        if self.variation_history.len() > 10 {
            self.variation_history.pop_front();
        }

        // Update time offset
        self.time_offset += duration.as_secs_f32();

        Ok(applied_variation)
    }

    /// Generate prosodic variations
    fn generate_prosodic_variations(
        &mut self,
        base_strength: f32,
        duration: Duration,
    ) -> crate::Result<Vec<VariationPattern>> {
        let mut patterns = Vec::new();

        // Pitch micro-variations
        patterns.push(VariationPattern {
            name: "pitch_tremolo".to_string(),
            variation_type: VariationType::Prosodic,
            amplitude: base_strength * 0.1, // 10% pitch variation
            frequency: 4.0 + self.rng.f32() * 2.0, // 4-6 Hz tremolo
            phase_offset: self.rng.f32() * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::Sinusoidal {
                frequency: 0.5,
                amplitude: 1.0,
            },
        });

        // Timing micro-variations
        patterns.push(VariationPattern {
            name: "timing_jitter".to_string(),
            variation_type: VariationType::Prosodic,
            amplitude: base_strength * 0.05, // 5% timing variation
            frequency: 1.0 + self.rng.f32() * 1.0, // 1-2 Hz
            phase_offset: self.rng.f32() * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::RandomWalk {
                step_size: 0.02,
                bounds: (-0.1, 0.1),
            },
        });

        // Energy micro-variations
        patterns.push(VariationPattern {
            name: "energy_fluctuation".to_string(),
            variation_type: VariationType::Prosodic,
            amplitude: base_strength * 0.15, // 15% energy variation
            frequency: 0.3 + self.rng.f32() * 0.4, // 0.3-0.7 Hz
            phase_offset: self.rng.f32() * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::Exponential {
                attack: 0.1,
                decay: 0.3,
                curve: 2.0,
            },
        });

        Ok(patterns)
    }

    /// Generate voice quality variations
    fn generate_voice_quality_variations(
        &mut self,
        base_strength: f32,
        duration: Duration,
    ) -> crate::Result<Vec<VariationPattern>> {
        let mut patterns = Vec::new();

        // Breathiness variations
        patterns.push(VariationPattern {
            name: "breathiness_variation".to_string(),
            variation_type: VariationType::VoiceQuality,
            amplitude: base_strength * 0.2, // 20% breathiness variation
            frequency: 0.1 + self.rng.f32() * 0.2, // 0.1-0.3 Hz slow variation
            phase_offset: self.rng.f32() * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::Linear {
                attack: 0.3,
                decay: 0.7,
            },
        });

        // Roughness micro-variations
        patterns.push(VariationPattern {
            name: "roughness_texture".to_string(),
            variation_type: VariationType::VoiceQuality,
            amplitude: base_strength * 0.1, // 10% roughness variation
            frequency: 8.0 + self.rng.f32() * 4.0, // 8-12 Hz fast texture
            phase_offset: self.rng.f32() * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::RandomWalk {
                step_size: 0.01,
                bounds: (-0.05, 0.05),
            },
        });

        Ok(patterns)
    }

    /// Generate breathing pattern variations
    fn generate_breathing_variations(
        &mut self,
        base_strength: f32,
        duration: Duration,
    ) -> crate::Result<Vec<VariationPattern>> {
        let mut patterns = Vec::new();
        let breathing = &self.speaker_characteristics.breathing_pattern;

        // Breathing cycle influence
        let breath_freq = breathing.rate / 60.0; // Convert to Hz
        patterns.push(VariationPattern {
            name: "breathing_cycle".to_string(),
            variation_type: VariationType::Breathing,
            amplitude: base_strength * breathing.depth_variation,
            frequency: breath_freq,
            phase_offset: self.time_offset * breath_freq * 2.0 * std::f32::consts::PI,
            duration,
            envelope: VariationEnvelope::Sinusoidal {
                frequency: breath_freq,
                amplitude: breathing.emotion_influence,
            },
        });

        Ok(patterns)
    }

    /// Apply micro-emotional fluctuations
    fn apply_micro_emotional_variations(
        &mut self,
        parameters: &mut EmotionParameters,
        base_parameters: &EmotionParameters,
        strength: f32,
    ) -> crate::Result<()> {
        // Small random fluctuations in emotional dimensions
        let valence_variation = (self.rng.f32() - 0.5) * strength * 0.1;
        let arousal_variation = (self.rng.f32() - 0.5) * strength * 0.15;
        let dominance_variation = (self.rng.f32() - 0.5) * strength * 0.05;

        parameters.emotion_vector.dimensions.valence =
            (base_parameters.emotion_vector.dimensions.valence + valence_variation)
                .clamp(-1.0, 1.0);
        parameters.emotion_vector.dimensions.arousal =
            (base_parameters.emotion_vector.dimensions.arousal + arousal_variation)
                .clamp(-1.0, 1.0);
        parameters.emotion_vector.dimensions.dominance =
            (base_parameters.emotion_vector.dimensions.dominance + dominance_variation)
                .clamp(-1.0, 1.0);

        Ok(())
    }

    /// Apply speaker-specific characteristic variations
    fn apply_speaker_characteristics(
        &mut self,
        parameters: &mut EmotionParameters,
        _base_parameters: &EmotionParameters,
        strength: f32,
    ) -> crate::Result<()> {
        let characteristics = &self.speaker_characteristics;
        let influence = self.config.speaker_characteristics_influence;

        // Apply expressiveness influence
        let expressiveness_factor = characteristics.expressiveness * influence;
        parameters.energy_scale *= 1.0 + (expressiveness_factor - 1.0) * strength;

        // Apply stability influence (inverse relationship with variation)
        let stability_factor = 1.0 - characteristics.stability;
        let stability_variation = (self.rng.f32() - 0.5) * stability_factor * strength * 0.2;
        parameters.tempo_scale += stability_variation;

        // Apply age-group specific variations
        match characteristics.age_group {
            AgeGroup::Child => {
                // Children have more variable pitch and energy
                parameters.pitch_shift *= 1.0 + strength * 0.1;
                parameters.breathiness += strength * 0.05;
            }
            AgeGroup::Senior => {
                // Seniors may have more voice quality variations
                parameters.roughness += strength * 0.15;
                parameters.pitch_shift *= 1.0 - strength * 0.05;
            }
            _ => {
                // Default adult characteristics
            }
        }

        // Apply speaking quirks
        for quirk in &characteristics.quirks {
            if self.rng.f32() < quirk.probability * strength {
                for (param_name, effect) in &quirk.parameter_effect {
                    match param_name.as_str() {
                        "tempo_scale" => parameters.tempo_scale *= 1.0 + effect * strength,
                        "breathiness" => parameters.breathiness += effect * strength,
                        "energy_scale" => parameters.energy_scale *= 1.0 + effect * strength,
                        "pitch_shift" => parameters.pitch_shift *= 1.0 + effect * strength,
                        "roughness" => parameters.roughness += effect * strength,
                        _ => {} // Unknown parameter
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply variation pattern to parameters
    fn apply_variation_pattern(
        &mut self,
        parameters: &mut EmotionParameters,
        pattern: &VariationPattern,
        base_parameters: &EmotionParameters,
    ) -> crate::Result<()> {
        // Calculate time-based modulation
        let time_phase = self.time_offset * pattern.frequency * 2.0 * std::f32::consts::PI
            + pattern.phase_offset;
        let base_modulation = time_phase.sin();

        // Apply envelope
        let envelope_value = self.calculate_envelope_value(&pattern.envelope, self.time_offset)?;
        let modulation = base_modulation * envelope_value * pattern.amplitude;

        // Apply modulation based on variation type
        match pattern.variation_type {
            VariationType::Prosodic => {
                match pattern.name.as_str() {
                    "pitch_tremolo" => {
                        parameters.pitch_shift *= 1.0 + modulation;
                    }
                    "timing_jitter" => {
                        // Timing variations are applied through tempo scaling
                        parameters.tempo_scale *= 1.0 + modulation * 0.5;
                    }
                    "energy_fluctuation" => {
                        parameters.energy_scale *= 1.0 + modulation;
                    }
                    _ => {}
                }
            }
            VariationType::VoiceQuality => match pattern.name.as_str() {
                "breathiness_variation" => {
                    parameters.breathiness += modulation;
                }
                "roughness_texture" => {
                    parameters.roughness += modulation;
                }
                _ => {
                    parameters.breathiness += modulation * 0.5;
                }
            },
            VariationType::Breathing => {
                // Breathing affects both energy and voice quality
                parameters.energy_scale *= 1.0 + modulation * 0.3;
                parameters.breathiness += modulation * 0.2;
            }
            VariationType::MicroEmotional => {
                // Micro-emotional variations affect dimensions slightly
                parameters.emotion_vector.dimensions.arousal += modulation * 0.1;
                parameters.emotion_vector.dimensions.arousal = parameters
                    .emotion_vector
                    .dimensions
                    .arousal
                    .clamp(-1.0, 1.0);
            }
            VariationType::SpeakerIdiosyncratic => {
                // Speaker-specific variations can affect any parameter
                parameters.tempo_scale *= 1.0 + modulation * 0.1;
                parameters.breathiness += modulation * 0.05;
            }
        }

        Ok(())
    }

    /// Calculate envelope value at given time
    fn calculate_envelope_value(
        &mut self,
        envelope: &VariationEnvelope,
        time: f32,
    ) -> crate::Result<f32> {
        let value = match envelope {
            VariationEnvelope::Linear { attack, decay } => {
                if time < *attack {
                    time / attack
                } else {
                    (1.0 - ((time - attack) / decay)).max(0.0)
                }
            }
            VariationEnvelope::Exponential {
                attack,
                decay,
                curve,
            } => {
                if time < *attack {
                    (time / attack).powf(*curve)
                } else {
                    (1.0 - ((time - attack) / decay)).max(0.0).powf(1.0 / curve)
                }
            }
            VariationEnvelope::Sinusoidal {
                frequency,
                amplitude,
            } => (time * frequency * 2.0 * std::f32::consts::PI).sin() * amplitude,
            VariationEnvelope::RandomWalk {
                step_size: _,
                bounds,
            } => {
                // Simplified random walk - would need state tracking for full implementation
                let random_val = self.rng.f32();
                bounds.0 + (bounds.1 - bounds.0) * random_val
            }
        };

        Ok(value.clamp(-1.0, 1.0))
    }

    /// Apply smoothing based on variation history
    fn apply_variation_smoothing(
        &mut self,
        parameters: &mut EmotionParameters,
        base_parameters: &EmotionParameters,
    ) -> crate::Result<()> {
        if let Some(last_variation) = self.variation_history.back() {
            let smoothing = self.config.smoothing_factor;

            // Smooth transitions
            parameters.tempo_scale = parameters.tempo_scale * (1.0 - smoothing)
                + last_variation.varied_parameters.tempo_scale * smoothing;

            parameters.breathiness = parameters.breathiness * (1.0 - smoothing)
                + last_variation.varied_parameters.breathiness * smoothing;

            parameters.energy_scale = parameters.energy_scale * (1.0 - smoothing)
                + last_variation.varied_parameters.energy_scale * smoothing;
        }

        Ok(())
    }

    /// Update speaker characteristics
    pub fn update_speaker_characteristics(&mut self, characteristics: SpeakerCharacteristics) {
        self.speaker_characteristics = characteristics;
    }

    /// Get variation statistics
    pub fn get_variation_statistics(&self) -> VariationStatistics {
        if self.variation_history.is_empty() {
            return VariationStatistics::default();
        }

        let count = self.variation_history.len();

        let avg_strength = self
            .variation_history
            .iter()
            .map(|v| v.strength)
            .sum::<f32>()
            / count as f32;

        let avg_patterns = self
            .variation_history
            .iter()
            .map(|v| v.active_patterns.len())
            .sum::<usize>() as f32
            / count as f32;

        VariationStatistics {
            total_variations: count,
            average_strength: avg_strength,
            average_patterns_per_variation: avg_patterns,
            pattern_type_distribution: self.calculate_pattern_distribution(),
        }
    }

    /// Calculate pattern type distribution
    fn calculate_pattern_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for variation in &self.variation_history {
            for pattern in &variation.active_patterns {
                let type_name = format!("{:?}", pattern.variation_type);
                *distribution.entry(type_name).or_insert(0) += 1;
            }
        }

        distribution
    }
}

/// Variation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationStatistics {
    /// Total number of variations applied
    pub total_variations: usize,
    /// Average variation strength
    pub average_strength: f32,
    /// Average number of patterns per variation
    pub average_patterns_per_variation: f32,
    /// Distribution of pattern types
    pub pattern_type_distribution: HashMap<String, usize>,
}

impl Default for VariationStatistics {
    fn default() -> Self {
        Self {
            total_variations: 0,
            average_strength: 0.0,
            average_patterns_per_variation: 0.0,
            pattern_type_distribution: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector};

    #[test]
    fn test_natural_variation_config() {
        let config = NaturalVariationConfig::default();
        assert!(config.base_variation_intensity > 0.0);
        assert!(config.temporal_frequency > 0.0);
        assert!(config.emotion_scaling.contains_key("Happy"));
    }

    #[test]
    fn test_speaker_characteristics() {
        let characteristics = SpeakerCharacteristics::default();
        assert_eq!(characteristics.expressiveness, 1.0);
        assert!(characteristics.stability > 0.0);
        assert!(characteristics.breathing_pattern.rate > 0.0);
    }

    #[test]
    fn test_variation_generator() {
        let config = NaturalVariationConfig {
            random_seed: Some(42), // For reproducible testing
            ..Default::default()
        };

        let characteristics = SpeakerCharacteristics::default();
        let mut generator = NaturalVariationGenerator::new(config, characteristics);

        let base_params = EmotionParameters::new(EmotionVector::new());

        let variation = generator
            .apply_variations(
                &base_params,
                &Emotion::Happy,
                EmotionIntensity::new(0.8),
                Duration::from_secs(2),
            )
            .unwrap();

        // Verify that variations were applied
        assert_ne!(
            variation.base_parameters.tempo_scale,
            variation.varied_parameters.tempo_scale
        );
        assert!(!variation.active_patterns.is_empty());
        assert!(variation.strength > 0.0);
    }

    #[test]
    fn test_prosodic_variations() {
        let config = NaturalVariationConfig {
            enable_voice_quality_variation: false,
            enable_breathing_patterns: false,
            random_seed: Some(123),
            ..Default::default()
        };

        let characteristics = SpeakerCharacteristics::default();
        let mut generator = NaturalVariationGenerator::new(config, characteristics);

        let base_params = EmotionParameters::default();

        let variation = generator
            .apply_variations(
                &base_params,
                &Emotion::Happy,
                EmotionIntensity::new(0.8),
                Duration::from_secs(1),
            )
            .unwrap();

        // Should have prosodic patterns only
        let prosodic_patterns = variation
            .active_patterns
            .iter()
            .filter(|p| matches!(p.variation_type, VariationType::Prosodic))
            .count();

        assert!(prosodic_patterns > 0);
    }

    #[test]
    fn test_speaker_influence() {
        let config = NaturalVariationConfig {
            speaker_characteristics_influence: 0.5,
            random_seed: Some(456),
            ..Default::default()
        };

        let mut highly_expressive = SpeakerCharacteristics::default();
        highly_expressive.expressiveness = 2.0;

        let mut low_expressive = SpeakerCharacteristics::default();
        low_expressive.expressiveness = 0.5;

        let base_params = EmotionParameters::default();

        let mut generator1 = NaturalVariationGenerator::new(config.clone(), highly_expressive);
        let variation1 = generator1
            .apply_variations(
                &base_params,
                &Emotion::Happy,
                EmotionIntensity::new(0.8),
                Duration::from_secs(1),
            )
            .unwrap();

        let mut generator2 = NaturalVariationGenerator::new(config, low_expressive);
        let variation2 = generator2
            .apply_variations(
                &base_params,
                &Emotion::Happy,
                EmotionIntensity::new(0.8),
                Duration::from_secs(1),
            )
            .unwrap();

        // Highly expressive speaker should have higher energy scaling
        assert!(
            variation1.varied_parameters.energy_scale > variation2.varied_parameters.energy_scale
        );
    }

    #[test]
    fn test_variation_statistics() {
        let config = NaturalVariationConfig {
            random_seed: Some(789),
            ..Default::default()
        };

        let characteristics = SpeakerCharacteristics::default();
        let mut generator = NaturalVariationGenerator::new(config, characteristics);

        let base_params = EmotionParameters::default();

        // Apply multiple variations
        for _ in 0..5 {
            generator
                .apply_variations(
                    &base_params,
                    &Emotion::Happy,
                    EmotionIntensity::new(0.8),
                    Duration::from_secs(1),
                )
                .unwrap();
        }

        let stats = generator.get_variation_statistics();
        assert_eq!(stats.total_variations, 5);
        assert!(stats.average_strength > 0.0);
        assert!(stats.average_patterns_per_variation > 0.0);
        assert!(!stats.pattern_type_distribution.is_empty());
    }
}
