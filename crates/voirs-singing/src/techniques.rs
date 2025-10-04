//! Singing techniques and vocal processing

#![allow(dead_code, clippy::derivable_impls)]

use crate::types::Expression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Singing technique parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingTechnique {
    /// Breath control settings
    pub breath_control: BreathControl,
    /// Vibrato settings
    pub vibrato: VibratoSettings,
    /// Vocal fry settings
    pub vocal_fry: VocalFry,
    /// Legato settings
    pub legato: LegatoSettings,
    /// Portamento settings
    pub portamento: PortamentoSettings,
    /// Dynamics settings
    pub dynamics: DynamicsSettings,
    /// Articulation settings
    pub articulation: ArticulationSettings,
    /// Expression settings
    pub expression: ExpressionSettings,
    /// Formant settings
    pub formant: FormantSettings,
    /// Resonance settings
    pub resonance: ResonanceSettings,
}

/// Breath control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathControl {
    /// Breath support strength (0.0-1.0)
    pub support: f32,
    /// Breath flow rate (0.0-1.0)
    pub flow_rate: f32,
    /// Breath capacity (0.0-1.0)
    pub capacity: f32,
    /// Breath noise level (0.0-1.0)
    pub noise_level: f32,
    /// Breath pattern type
    pub pattern: BreathPattern,
    /// Automatic breath insertion
    pub auto_breath: bool,
    /// Breath timing adjustment
    pub timing_adjustment: f32,
    /// Breath depth variation
    pub depth_variation: f32,
}

/// Vibrato settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoSettings {
    /// Vibrato frequency in Hz
    pub frequency: f32,
    /// Vibrato depth (0.0-1.0)
    pub depth: f32,
    /// Vibrato onset time in seconds
    pub onset: f32,
    /// Vibrato rate variation
    pub rate_variation: f32,
    /// Vibrato depth variation
    pub depth_variation: f32,
    /// Vibrato shape
    pub shape: VibratoShape,
    /// Enable automatic vibrato
    pub auto_vibrato: bool,
    /// Vibrato modulation
    pub modulation: VibratoModulation,
}

/// Vocal fry parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalFry {
    /// Vocal fry amount (0.0-1.0)
    pub amount: f32,
    /// Vocal fry frequency in Hz
    pub frequency: f32,
    /// Vocal fry roughness (0.0-1.0)
    pub roughness: f32,
    /// Vocal fry onset (0.0-1.0)
    pub onset: f32,
    /// Vocal fry irregularity
    pub irregularity: f32,
    /// Enable automatic vocal fry
    pub auto_fry: bool,
}

/// Legato settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegatoSettings {
    /// Legato strength (0.0-1.0)
    pub strength: f32,
    /// Legato smoothness (0.0-1.0)
    pub smoothness: f32,
    /// Portamento time in seconds
    pub portamento_time: f32,
    /// Glissando speed
    pub glissando_speed: f32,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Enable automatic legato
    pub auto_legato: bool,
}

/// Portamento settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortamentoSettings {
    /// Portamento time in seconds
    pub time: f32,
    /// Portamento curve
    pub curve: PortamentoCurve,
    /// Portamento strength (0.0-1.0)
    pub strength: f32,
    /// Enable automatic portamento
    pub auto_portamento: bool,
    /// Portamento threshold in semitones
    pub threshold: f32,
}

/// Dynamics settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsSettings {
    /// Base volume (0.0-1.0)
    pub base_volume: f32,
    /// Dynamic range (0.0-1.0)
    pub range: f32,
    /// Compression ratio
    pub compression: f32,
    /// Attack time in seconds
    pub attack: f32,
    /// Release time in seconds
    pub release: f32,
    /// Enable automatic dynamics
    pub auto_dynamics: bool,
}

/// Articulation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationSettings {
    /// Staccato amount (0.0-1.0)
    pub staccato: f32,
    /// Accent strength (0.0-1.0)
    pub accent: f32,
    /// Tenuto amount (0.0-1.0)
    pub tenuto: f32,
    /// Marcato strength (0.0-1.0)
    pub marcato: f32,
    /// Slur smoothness (0.0-1.0)
    pub slur: f32,
    /// Default articulation
    pub default_articulation: crate::types::Articulation,
}

/// Expression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionSettings {
    /// Expression intensity (0.0-1.0)
    pub intensity: f32,
    /// Expression variation (0.0-1.0)
    pub variation: f32,
    /// Expression transition time in seconds
    pub transition_time: f32,
    /// Expression mapping
    pub mapping: HashMap<Expression, ExpressionParams>,
    /// Enable automatic expression
    pub auto_expression: bool,
}

/// Formant settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantSettings {
    /// Formant frequencies in Hz
    pub frequencies: Vec<f32>,
    /// Formant bandwidths in Hz
    pub bandwidths: Vec<f32>,
    /// Formant amplitudes (0.0-1.0)
    pub amplitudes: Vec<f32>,
    /// Formant shift (0.0-2.0, 1.0 = no shift)
    pub shift: f32,
    /// Enable formant modeling
    pub enabled: bool,
}

/// Resonance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceSettings {
    /// Chest resonance (0.0-1.0)
    pub chest: f32,
    /// Head resonance (0.0-1.0)
    pub head: f32,
    /// Nasal resonance (0.0-1.0)
    pub nasal: f32,
    /// Throat resonance (0.0-1.0)
    pub throat: f32,
    /// Resonance mixing
    pub mixing: f32,
    /// Enable resonance modeling
    pub enabled: bool,
}

/// Breath pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreathPattern {
    /// Natural breathing
    Natural,
    /// Controlled breathing
    Controlled,
    /// Rapid breathing
    Rapid,
    /// Deep breathing
    Deep,
    /// Shallow breathing
    Shallow,
}

/// Vibrato shape types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VibratoShape {
    /// Sine wave vibrato
    Sine,
    /// Triangle wave vibrato
    Triangle,
    /// Sawtooth wave vibrato
    Sawtooth,
    /// Square wave vibrato
    Square,
    /// Random vibrato
    Random,
}

/// Vibrato modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoModulation {
    /// Amplitude modulation depth (0.0-1.0)
    pub amplitude: f32,
    /// Frequency modulation depth (0.0-1.0)
    pub frequency: f32,
    /// Modulation rate in Hz
    pub rate: f32,
    /// Modulation phase
    pub phase: f32,
}

/// Connection types for legato
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Smooth connection
    Smooth,
    /// Glissando connection
    Glissando,
    /// Portamento connection
    Portamento,
    /// Slur connection
    Slur,
}

/// Portamento curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortamentoCurve {
    /// Linear curve
    Linear,
    /// Exponential curve
    Exponential,
    /// Logarithmic curve
    Logarithmic,
    /// Sigmoid curve
    Sigmoid,
}

/// Expression parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionParams {
    /// Pitch modifier
    pub pitch_modifier: f32,
    /// Volume modifier
    pub volume_modifier: f32,
    /// Tempo modifier
    pub tempo_modifier: f32,
    /// Vibrato modifier
    pub vibrato_modifier: f32,
    /// Breath modifier
    pub breath_modifier: f32,
    /// Timbre modifier
    pub timbre_modifier: f32,
}

/// Vibrato processor for real-time vibrato effect application
///
/// Processes audio samples to apply vibrato modulation based on configured settings.
/// Maintains internal state for phase and time tracking to produce smooth, continuous
/// vibrato effects across sample buffers.
pub struct VibratoProcessor {
    /// Vibrato configuration settings
    settings: VibratoSettings,
    /// Current phase of the vibrato oscillator (0.0-1.0)
    phase: f32,
    /// Elapsed time in seconds since processor creation
    time: f32,
}

/// Breath control processor for realistic breathing simulation
///
/// Manages breath sound insertion, breath noise modeling, and timing of breath events
/// in singing synthesis. Tracks breath state and timing to create natural breathing
/// patterns between phrases and during sustained notes.
pub struct BreathProcessor {
    /// Breath control configuration settings
    settings: BreathControl,
    /// Current breath state (0.0 = empty, 1.0 = full capacity)
    breath_state: f32,
    /// Time in seconds since last breath event
    last_breath_time: f32,
}

/// Legato processor for smooth note transitions
///
/// Handles smooth connections between consecutive notes using portamento, glissando,
/// and other legato techniques. Maintains transition state and frequency tracking to
/// create seamless pitch changes without audible gaps or artifacts.
pub struct LegatoProcessor {
    /// Legato configuration settings
    settings: LegatoSettings,
    /// Current transition progress (0.0 = start, 1.0 = complete)
    transition_state: f32,
    /// Previous note frequency in Hz for smooth interpolation
    last_frequency: f32,
}

/// Vocal fry processor for creaky voice effect
///
/// Simulates vocal fry (also known as glottal fry or creaky voice), a low-frequency
/// vocal effect characterized by irregular glottal pulses. Commonly used in contemporary
/// vocal styles and for expressive emphasis in singing and speech.
pub struct VocalFryProcessor {
    /// Vocal fry configuration settings
    settings: VocalFry,
    /// Current fry state intensity (0.0-1.0)
    fry_state: f32,
    /// Phase accumulator for irregularity modulation
    irregularity_phase: f32,
}

impl SingingTechnique {
    /// Create technique for specific singing style
    pub fn for_style(style: &str) -> Self {
        match style {
            "classical" => Self::classical(),
            "pop" => Self::pop(),
            "jazz" => Self::jazz(),
            "opera" => Self::opera(),
            "folk" => Self::folk(),
            "gospel" => Self::gospel(),
            "rock" => Self::rock(),
            "country" => Self::country(),
            _ => Self::default(),
        }
    }

    /// Classical singing technique
    pub fn classical() -> Self {
        Self {
            breath_control: BreathControl {
                support: 0.9,
                flow_rate: 0.8,
                capacity: 0.9,
                noise_level: 0.1,
                pattern: BreathPattern::Controlled,
                auto_breath: true,
                timing_adjustment: 0.0,
                depth_variation: 0.2,
            },
            vibrato: VibratoSettings {
                frequency: 6.0,
                depth: 0.3,
                onset: 0.5,
                rate_variation: 0.1,
                depth_variation: 0.1,
                shape: VibratoShape::Sine,
                auto_vibrato: true,
                modulation: VibratoModulation {
                    amplitude: 0.1,
                    frequency: 0.05,
                    rate: 0.5,
                    phase: 0.0,
                },
            },
            vocal_fry: VocalFry {
                amount: 0.1,
                frequency: 80.0,
                roughness: 0.2,
                onset: 0.9,
                irregularity: 0.1,
                auto_fry: false,
            },
            legato: LegatoSettings {
                strength: 0.8,
                smoothness: 0.9,
                portamento_time: 0.1,
                glissando_speed: 0.5,
                connection_type: ConnectionType::Smooth,
                auto_legato: true,
            },
            portamento: PortamentoSettings {
                time: 0.15,
                curve: PortamentoCurve::Sigmoid,
                strength: 0.6,
                auto_portamento: true,
                threshold: 2.0,
            },
            dynamics: DynamicsSettings {
                base_volume: 0.8,
                range: 0.7,
                compression: 0.3,
                attack: 0.1,
                release: 0.2,
                auto_dynamics: true,
            },
            articulation: ArticulationSettings {
                staccato: 0.2,
                accent: 0.3,
                tenuto: 0.8,
                marcato: 0.4,
                slur: 0.9,
                default_articulation: crate::types::Articulation::Legato,
            },
            expression: ExpressionSettings {
                intensity: 0.7,
                variation: 0.3,
                transition_time: 0.5,
                mapping: HashMap::new(),
                auto_expression: true,
            },
            formant: FormantSettings {
                frequencies: vec![800.0, 1200.0, 2600.0, 3600.0],
                bandwidths: vec![80.0, 100.0, 120.0, 150.0],
                amplitudes: vec![1.0, 0.8, 0.6, 0.4],
                shift: 1.0,
                enabled: true,
            },
            resonance: ResonanceSettings {
                chest: 0.6,
                head: 0.8,
                nasal: 0.2,
                throat: 0.5,
                mixing: 0.7,
                enabled: true,
            },
        }
    }

    /// Pop singing technique
    pub fn pop() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.6;
        technique.breath_control.flow_rate = 0.7;
        technique.vibrato.frequency = 5.5;
        technique.vibrato.depth = 0.4;
        technique.vocal_fry.amount = 0.3;
        technique.legato.strength = 0.6;
        technique.dynamics.compression = 0.6;
        technique.articulation.default_articulation = crate::types::Articulation::Normal;
        technique.resonance.chest = 0.7;
        technique.resonance.head = 0.5;
        technique
    }

    /// Jazz singing technique
    pub fn jazz() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.7;
        technique.vibrato.frequency = 6.5;
        technique.vibrato.depth = 0.5;
        technique.vocal_fry.amount = 0.2;
        technique.legato.strength = 0.9;
        technique.portamento.time = 0.2;
        technique.dynamics.range = 0.8;
        technique.articulation.slur = 0.8;
        technique.expression.intensity = 0.8;
        technique.resonance.chest = 0.8;
        technique.resonance.throat = 0.6;
        technique
    }

    /// Opera singing technique
    pub fn opera() -> Self {
        let mut technique = Self::classical();
        technique.breath_control.support = 1.0;
        technique.breath_control.capacity = 1.0;
        technique.vibrato.frequency = 7.0;
        technique.vibrato.depth = 0.4;
        technique.dynamics.base_volume = 0.9;
        technique.dynamics.range = 0.9;
        technique.formant.shift = 1.1;
        technique.resonance.chest = 0.5;
        technique.resonance.head = 0.9;
        technique
    }

    /// Folk singing technique
    pub fn folk() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.5;
        technique.breath_control.pattern = BreathPattern::Natural;
        technique.vibrato.frequency = 4.0;
        technique.vibrato.depth = 0.2;
        technique.vocal_fry.amount = 0.4;
        technique.legato.strength = 0.5;
        technique.dynamics.compression = 0.2;
        technique.articulation.default_articulation = crate::types::Articulation::Normal;
        technique.resonance.chest = 0.8;
        technique.resonance.nasal = 0.3;
        technique
    }

    /// Gospel singing technique
    pub fn gospel() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.8;
        technique.vibrato.frequency = 5.0;
        technique.vibrato.depth = 0.6;
        technique.vocal_fry.amount = 0.1;
        technique.legato.strength = 0.7;
        technique.dynamics.base_volume = 0.8;
        technique.dynamics.range = 0.8;
        technique.articulation.accent = 0.5;
        technique.expression.intensity = 0.9;
        technique.resonance.chest = 0.9;
        technique.resonance.head = 0.7;
        technique
    }

    /// Rock singing technique
    pub fn rock() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.7;
        technique.vibrato.frequency = 4.5;
        technique.vibrato.depth = 0.3;
        technique.vocal_fry.amount = 0.5;
        technique.legato.strength = 0.4;
        technique.dynamics.base_volume = 0.9;
        technique.dynamics.compression = 0.8;
        technique.articulation.accent = 0.7;
        technique.articulation.marcato = 0.6;
        technique.resonance.chest = 0.9;
        technique.resonance.throat = 0.8;
        technique
    }

    /// Country singing technique
    pub fn country() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.6;
        technique.vibrato.frequency = 5.5;
        technique.vibrato.depth = 0.4;
        technique.vocal_fry.amount = 0.3;
        technique.legato.strength = 0.5;
        technique.dynamics.compression = 0.4;
        technique.articulation.default_articulation = crate::types::Articulation::Normal;
        technique.resonance.chest = 0.7;
        technique.resonance.nasal = 0.4;
        technique
    }

    /// Apply technique to audio parameters
    pub fn apply_to_note(&self, note: &mut crate::types::NoteEvent) {
        // Apply vibrato
        note.vibrato = (note.vibrato + self.vibrato.depth) / 2.0;

        // Apply expression
        let expr_params = self.expression.mapping.get(&note.expression);
        if let Some(params) = expr_params {
            note.frequency *= params.pitch_modifier;
            note.velocity *= params.volume_modifier;
            note.vibrato *= params.vibrato_modifier;
        }

        // Apply dynamics
        note.velocity *= self.dynamics.base_volume;

        // Apply breath control
        note.breath_before = self.breath_control.noise_level;
    }
}

impl VibratoProcessor {
    /// Create new vibrato processor
    pub fn new(settings: VibratoSettings) -> Self {
        Self {
            settings,
            phase: 0.0,
            time: 0.0,
        }
    }

    /// Process vibrato for a sample
    pub fn process_sample(&mut self, sample: f32, sample_rate: f32) -> f32 {
        if self.time < self.settings.onset {
            self.time += 1.0 / sample_rate;
            return sample;
        }

        let dt = 1.0 / sample_rate;
        let frequency = self.settings.frequency
            * (1.0 + self.settings.rate_variation * (self.phase * 0.1).sin());

        let vibrato_value = match self.settings.shape {
            VibratoShape::Sine => (self.phase * 2.0 * std::f32::consts::PI).sin(),
            VibratoShape::Triangle => {
                let normalized = (self.phase % 1.0) * 2.0 - 1.0;
                if normalized < 0.0 {
                    -1.0 - 2.0 * normalized
                } else {
                    1.0 - 2.0 * normalized
                }
            }
            VibratoShape::Sawtooth => (self.phase % 1.0) * 2.0 - 1.0,
            VibratoShape::Square => {
                if (self.phase % 1.0) < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            VibratoShape::Random => (scirs2_core::random::random::<f32>() - 0.5) * 2.0,
        };

        let depth =
            self.settings.depth * (1.0 + self.settings.depth_variation * (self.phase * 0.05).sin());
        let modulated_sample = sample * (1.0 + depth * vibrato_value);

        self.phase += frequency * dt;
        if self.phase > 1.0 {
            self.phase -= 1.0;
        }

        modulated_sample
    }
}

impl Default for SingingTechnique {
    fn default() -> Self {
        Self {
            breath_control: BreathControl::default(),
            vibrato: VibratoSettings::default(),
            vocal_fry: VocalFry::default(),
            legato: LegatoSettings::default(),
            portamento: PortamentoSettings::default(),
            dynamics: DynamicsSettings::default(),
            articulation: ArticulationSettings::default(),
            expression: ExpressionSettings::default(),
            formant: FormantSettings::default(),
            resonance: ResonanceSettings::default(),
        }
    }
}

impl Default for BreathControl {
    fn default() -> Self {
        Self {
            support: 0.8,
            flow_rate: 0.7,
            capacity: 0.8,
            noise_level: 0.2,
            pattern: BreathPattern::Natural,
            auto_breath: true,
            timing_adjustment: 0.0,
            depth_variation: 0.3,
        }
    }
}

impl Default for VibratoSettings {
    fn default() -> Self {
        Self {
            frequency: 5.0,
            depth: 0.4,
            onset: 0.3,
            rate_variation: 0.1,
            depth_variation: 0.1,
            shape: VibratoShape::Sine,
            auto_vibrato: true,
            modulation: VibratoModulation::default(),
        }
    }
}

impl Default for VibratoModulation {
    fn default() -> Self {
        Self {
            amplitude: 0.1,
            frequency: 0.05,
            rate: 0.5,
            phase: 0.0,
        }
    }
}

impl Default for VocalFry {
    fn default() -> Self {
        Self {
            amount: 0.2,
            frequency: 80.0,
            roughness: 0.3,
            onset: 0.8,
            irregularity: 0.2,
            auto_fry: false,
        }
    }
}

impl Default for LegatoSettings {
    fn default() -> Self {
        Self {
            strength: 0.7,
            smoothness: 0.8,
            portamento_time: 0.1,
            glissando_speed: 0.5,
            connection_type: ConnectionType::Smooth,
            auto_legato: true,
        }
    }
}

impl Default for PortamentoSettings {
    fn default() -> Self {
        Self {
            time: 0.1,
            curve: PortamentoCurve::Sigmoid,
            strength: 0.5,
            auto_portamento: false,
            threshold: 3.0,
        }
    }
}

impl Default for DynamicsSettings {
    fn default() -> Self {
        Self {
            base_volume: 0.8,
            range: 0.6,
            compression: 0.4,
            attack: 0.05,
            release: 0.1,
            auto_dynamics: true,
        }
    }
}

impl Default for ArticulationSettings {
    fn default() -> Self {
        Self {
            staccato: 0.3,
            accent: 0.4,
            tenuto: 0.6,
            marcato: 0.5,
            slur: 0.7,
            default_articulation: crate::types::Articulation::Normal,
        }
    }
}

impl Default for ExpressionSettings {
    fn default() -> Self {
        Self {
            intensity: 0.6,
            variation: 0.4,
            transition_time: 0.3,
            mapping: HashMap::new(),
            auto_expression: true,
        }
    }
}

impl Default for FormantSettings {
    fn default() -> Self {
        Self {
            frequencies: vec![800.0, 1200.0, 2600.0, 3600.0],
            bandwidths: vec![80.0, 100.0, 120.0, 150.0],
            amplitudes: vec![1.0, 0.8, 0.6, 0.4],
            shift: 1.0,
            enabled: true,
        }
    }
}

impl Default for ResonanceSettings {
    fn default() -> Self {
        Self {
            chest: 0.7,
            head: 0.6,
            nasal: 0.3,
            throat: 0.5,
            mixing: 0.6,
            enabled: true,
        }
    }
}

impl Default for BeltingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FalsettoProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MixedVoiceProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WhistleRegisterProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// Add new technique variants to existing SingingTechnique
impl SingingTechnique {
    /// Belting singing technique (powerful chest voice)
    pub fn belting() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 1.0;
        technique.breath_control.flow_rate = 0.9;
        technique.vibrato.frequency = 4.0;
        technique.vibrato.depth = 0.2; // Less vibrato for power
        technique.dynamics.base_volume = 0.9;
        technique.dynamics.compression = 0.8;
        technique.resonance.chest = 1.0;
        technique.resonance.head = 0.3;
        technique
    }

    /// Falsetto singing technique (light head voice)
    pub fn falsetto() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.4;
        technique.breath_control.flow_rate = 0.3;
        technique.breath_control.noise_level = 0.4; // Breathiness
        technique.vibrato.frequency = 5.5;
        technique.vibrato.depth = 0.5;
        technique.dynamics.base_volume = 0.4;
        technique.dynamics.compression = 0.2;
        technique.resonance.chest = 0.1;
        technique.resonance.head = 0.9;
        technique
    }

    /// Mixed voice singing technique
    pub fn mixed_voice() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.8;
        technique.breath_control.flow_rate = 0.7;
        technique.vibrato.frequency = 5.0;
        technique.vibrato.depth = 0.4;
        technique.dynamics.base_volume = 0.7;
        technique.dynamics.compression = 0.5;
        technique.resonance.chest = 0.6;
        technique.resonance.head = 0.7;
        technique.resonance.mixing = 0.9; // Key for mixed voice
        technique
    }

    /// Whistle register technique (ultra-high frequencies)
    pub fn whistle_register() -> Self {
        let mut technique = Self::default();
        technique.breath_control.support = 0.3;
        technique.breath_control.flow_rate = 0.2;
        technique.breath_control.capacity = 0.4;
        technique.vibrato.frequency = 3.0;
        technique.vibrato.depth = 0.1; // Minimal vibrato
        technique.dynamics.base_volume = 0.3;
        technique.dynamics.compression = 0.1;
        technique.resonance.chest = 0.0;
        technique.resonance.head = 0.4;
        technique.resonance.throat = 0.1; // Very focused
        technique
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technique_styles() {
        let classical = SingingTechnique::classical();
        let pop = SingingTechnique::pop();
        let jazz = SingingTechnique::jazz();

        assert!(classical.breath_control.support > pop.breath_control.support);
        assert!(jazz.legato.strength > pop.legato.strength);
        assert!(pop.vocal_fry.amount > classical.vocal_fry.amount);
    }

    #[test]
    fn test_vibrato_processor() {
        let settings = VibratoSettings::default();
        let mut processor = VibratoProcessor::new(settings);

        let sample = 0.5;
        let processed = processor.process_sample(sample, 44100.0);

        // Should be close to original during onset
        assert!((processed - sample).abs() < 0.1);
    }

    #[test]
    fn test_technique_application() {
        let technique = SingingTechnique::pop();
        let mut note = crate::types::NoteEvent::new("C".to_string(), 4, 1.0, 0.8);

        let _original_frequency = note.frequency;
        technique.apply_to_note(&mut note);

        // Note should be modified
        assert!(note.velocity <= 1.0);
        assert!(note.breath_before >= 0.0);
    }
}

// ===== ADVANCED VOCAL TECHNIQUES =====

/// Belting vocal technique processor
pub struct BeltingProcessor {
    /// Chest voice strength (0.0-1.0)
    pub chest_strength: f32,
    /// Power level (0.0-1.0)
    pub power_level: f32,
    /// Formant boost in Hz
    pub formant_boost: f32,
    /// Compression ratio
    pub compression: f32,
    /// Breath support requirement
    pub breath_support: f32,
}

/// Falsetto vocal technique processor  
pub struct FalsettoProcessor {
    /// Head voice ratio (0.0-1.0)
    pub head_voice_ratio: f32,
    /// Breathiness level (0.0-1.0)
    pub breathiness: f32,
    /// Lightness factor (0.0-1.0)  
    pub lightness: f32,
    /// High frequency emphasis
    pub high_freq_emphasis: f32,
    /// Reduced power factor
    pub power_reduction: f32,
}

/// Mixed voice processor
pub struct MixedVoiceProcessor {
    /// Chest-head blend ratio (0.0 = full chest, 1.0 = full head)
    pub blend_ratio: f32,
    /// Smooth transition range in semitones
    pub transition_range: f32,
    /// Passaggio frequency in Hz
    pub passaggio_freq: f32,
    /// Resonance balance
    pub resonance_balance: f32,
}

/// Whistle register processor
pub struct WhistleRegisterProcessor {
    /// Ultra-high frequency mode
    pub ultra_high_mode: bool,
    /// Harmonic compression
    pub harmonic_compression: f32,
    /// Focused resonance
    pub focused_resonance: f32,
    /// Minimal breath requirement
    pub breath_efficiency: f32,
}

impl BeltingProcessor {
    /// Create new belting processor
    pub fn new() -> Self {
        Self {
            chest_strength: 0.9,
            power_level: 0.8,
            formant_boost: 500.0,
            compression: 0.7,
            breath_support: 0.9,
        }
    }

    /// Apply belting technique to note
    pub fn apply_belting(&self, note: &mut crate::types::NoteEvent) {
        // Increase power and chest resonance
        note.velocity *= 1.0 + (self.power_level * 0.5);
        note.velocity = note.velocity.clamp(0.0, 1.0);

        // Boost formants for belting brightness
        if note.frequency > 200.0 {
            note.frequency *= 1.0 + (self.formant_boost / 10000.0);
        }

        // Reduce vibrato for powerful directness
        note.vibrato *= 0.5;

        // Increase breath support requirement
        note.breath_before = self.breath_support;
    }
}

impl FalsettoProcessor {
    /// Create new falsetto processor
    pub fn new() -> Self {
        Self {
            head_voice_ratio: 0.9,
            breathiness: 0.4,
            lightness: 0.8,
            high_freq_emphasis: 0.6,
            power_reduction: 0.6,
        }
    }

    /// Apply falsetto technique to note
    pub fn apply_falsetto(&self, note: &mut crate::types::NoteEvent) {
        // Reduce power for lightness
        note.velocity *= self.power_reduction;

        // Add breathiness
        note.breath_before = self.breathiness;

        // Increase vibrato for expressiveness
        note.vibrato *= 1.0 + (self.lightness * 0.3);

        // Boost high frequencies slightly
        if note.frequency > 400.0 {
            note.frequency *= 1.0 + (self.high_freq_emphasis * 0.05);
        }
    }
}

impl MixedVoiceProcessor {
    /// Create new mixed voice processor
    pub fn new() -> Self {
        Self {
            blend_ratio: 0.5,
            transition_range: 6.0, // 6 semitones
            passaggio_freq: 500.0, // Around F5 for typical voice
            resonance_balance: 0.5,
        }
    }

    /// Apply mixed voice technique to note
    pub fn apply_mixed_voice(&self, note: &mut crate::types::NoteEvent) {
        // Calculate blend based on frequency relative to passaggio
        let freq_ratio = note.frequency / self.passaggio_freq;
        let semitones_from_passaggio = 12.0 * freq_ratio.log2();

        // Adjust blend ratio based on pitch
        let dynamic_blend = if semitones_from_passaggio.abs() < self.transition_range {
            // In transition zone - use smooth blending
            self.blend_ratio + (semitones_from_passaggio / self.transition_range * 0.3)
        } else if semitones_from_passaggio > 0.0 {
            // Above passaggio - more head voice
            (self.blend_ratio + 0.3).clamp(0.0, 1.0)
        } else {
            // Below passaggio - more chest voice
            (self.blend_ratio - 0.3).clamp(0.0, 1.0)
        };

        // Apply mixed voice characteristics
        note.velocity *= 0.7 + (dynamic_blend * 0.3);
        note.vibrato *= 0.8 + (dynamic_blend * 0.4);
    }
}

impl WhistleRegisterProcessor {
    /// Create new whistle register processor
    pub fn new() -> Self {
        Self {
            ultra_high_mode: true,
            harmonic_compression: 0.8,
            focused_resonance: 0.9,
            breath_efficiency: 0.95,
        }
    }

    /// Apply whistle register technique to note
    pub fn apply_whistle_register(&self, note: &mut crate::types::NoteEvent) {
        // Only apply to very high frequencies (above 1000Hz)
        if note.frequency > 1000.0 {
            // Reduce power but maintain clarity
            note.velocity *= 0.4;

            // Minimal vibrato for purity
            note.vibrato *= 0.1;

            // Efficient breath usage
            note.breath_before = 1.0 - self.breath_efficiency;

            // Boost frequency slightly for whistle clarity
            if self.ultra_high_mode {
                note.frequency *= 1.02;
            }
        }
    }
}
