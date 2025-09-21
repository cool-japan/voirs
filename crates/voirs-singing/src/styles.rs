//! Musical style implementations for singing synthesis
//!
//! This module provides comprehensive musical style frameworks that go beyond
//! basic technique parameters to include genre-specific characteristics,
//! performance practices, and cultural variations.

#![allow(dead_code)]

use crate::techniques::SingingTechnique;
use crate::types::{NoteEvent, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Musical style processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalStyle {
    /// Style name
    pub name: String,
    /// Base singing technique
    pub technique: SingingTechnique,
    /// Style-specific characteristics
    pub characteristics: StyleCharacteristics,
    /// Phrase shaping rules
    pub phrase_shaping: PhraseShaping,
    /// Ornamentation patterns
    pub ornamentation: Ornamentation,
    /// Cultural variations
    pub cultural_variants: HashMap<String, CulturalVariant>,
    /// Performance guidelines
    pub performance: PerformanceGuidelines,
}

/// Style-specific characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleCharacteristics {
    /// Preferred voice types
    pub voice_types: Vec<VoiceType>,
    /// Typical pitch range in MIDI notes
    pub pitch_range: (u8, u8),
    /// Characteristic intervals
    pub intervals: Vec<f32>,
    /// Preferred scales/modes
    pub scales: Vec<Scale>,
    /// Rhythmic patterns
    pub rhythmic_patterns: Vec<RhythmicPattern>,
    /// Dynamic preferences
    pub dynamics: DynamicPreferences,
    /// Timbral qualities
    pub timbre: TimbreQualities,
}

/// Phrase shaping parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseShaping {
    /// Phrase arc type
    pub arc_type: PhraseArc,
    /// Crescendo/diminuendo patterns
    pub dynamic_shaping: Vec<DynamicShape>,
    /// Rubato characteristics
    pub rubato: RubatoStyle,
    /// Breath management
    pub breath_placement: BreathPlacement,
    /// Line connection preferences
    pub line_connection: LineConnection,
}

/// Ornamentation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ornamentation {
    /// Grace notes usage
    pub grace_notes: GraceNoteStyle,
    /// Trill patterns
    pub trills: TrillStyle,
    /// Mordent usage
    pub mordents: MordentStyle,
    /// Turn patterns
    pub turns: TurnStyle,
    /// Appoggiatura usage
    pub appoggiaturas: AppoggiaturaStyle,
    /// Glissando usage
    pub glissandos: GlissandoStyle,
}

/// Cultural variant parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalVariant {
    /// Variant name (e.g., "Italian", "German", "American")
    pub name: String,
    /// Pronunciation adjustments
    pub pronunciation: HashMap<String, String>,
    /// Vowel modifications
    pub vowel_colors: HashMap<char, VowelColor>,
    /// Regional ornamentation
    pub regional_ornaments: Vec<RegionalOrnament>,
    /// Tempo preferences
    pub tempo_tendencies: TempoTendencies,
}

/// Performance guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGuidelines {
    /// Ensemble considerations
    pub ensemble_role: EnsembleRole,
    /// Microphone technique
    pub microphone_technique: MicrophoneTechnique,
    /// Stage presence requirements
    pub stage_presence: StagePresence,
    /// Costume/visual considerations
    pub visual_elements: VisualElements,
    /// Audience interaction level
    pub audience_interaction: f32,
}

// === Enums and Supporting Types ===

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceType {
    Soprano,
    MezzoSoprano,
    Alto,
    Tenor,
    Baritone,
    Bass,
    Countertenor,
    Contralto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scale {
    pub name: String,
    pub intervals: Vec<f32>, // In semitones
    pub characteristic_notes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmicPattern {
    pub name: String,
    pub pattern: Vec<f32>, // Note durations
    pub accent_pattern: Vec<bool>,
    pub swing_feel: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPreferences {
    pub typical_range: (f32, f32), // Min/max dynamics
    pub contrast_level: f32,
    pub sudden_changes: bool,
    pub gradual_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimbreQualities {
    pub brightness: f32,
    pub warmth: f32,
    pub edge: f32,
    pub breathiness: f32,
    pub richness: f32,
    pub focus: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhraseArc {
    Linear,
    Arch,
    InvertedArch,
    Stepped,
    Undulating,
    Asymmetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicShape {
    pub start_dynamic: f32,
    pub peak_position: f32, // 0.0-1.0 within phrase
    pub peak_dynamic: f32,
    pub end_dynamic: f32,
    pub curve_type: CurveType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveType {
    Linear,
    Exponential,
    Logarithmic,
    Sinusoidal,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubatoStyle {
    pub amount: f32,
    pub phrase_initial: bool,
    pub phrase_final: bool,
    pub high_notes: bool,
    pub emotional_peaks: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreathPlacement {
    Grammatical,
    Musical,
    Dramatic,
    Technical,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineConnection {
    Legato,
    SemiLegato,
    Articulated,
    Mixed,
}

// Ornamentation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraceNoteStyle {
    pub frequency: f32,
    pub preferred_types: Vec<GraceNoteType>,
    pub placement_rules: Vec<PlacementRule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraceNoteType {
    Acciaccatura,
    Appoggiatura,
    Mordent,
    Turn,
    Trill,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementRule {
    pub context: String,
    pub probability: f32,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrillStyle {
    pub frequency: f32,
    pub speed_range: (f32, f32),
    pub interval: f32, // Usually 1.0 (semitone) or 2.0 (whole tone)
    pub termination: TrillTermination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrillTermination {
    Natural,
    WithTurn,
    WithMordent,
    Abrupt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MordentStyle {
    pub frequency: f32,
    pub upper_lower_ratio: f32,
    pub speed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStyle {
    pub frequency: f32,
    pub direction_preference: TurnDirection,
    pub speed_variation: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnDirection {
    Upper,
    Lower,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppoggiaturaStyle {
    pub frequency: f32,
    pub resolution_timing: f32,
    pub emphasis_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlissandoStyle {
    pub frequency: f32,
    pub interval_threshold: f32, // Minimum interval for glissando
    pub speed_variation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelColor {
    pub formant_shift: f32,
    pub brightness_adjustment: f32,
    pub openness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalOrnament {
    pub name: String,
    pub frequency: f32,
    pub execution_style: String,
    pub cultural_context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoTendencies {
    pub base_tempo_preference: f32,
    pub rubato_amount: f32,
    pub ritardando_tendency: f32,
    pub accelerando_tendency: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleRole {
    Soloist,
    Principal,
    Chorus,
    Background,
    Duet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrophoneTechnique {
    pub distance_preference: f32,
    pub dynamic_compensation: bool,
    pub breath_control_emphasis: f32,
    pub proximity_effect_usage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePresence {
    pub movement_level: f32,
    pub gesture_integration: f32,
    pub eye_contact_importance: f32,
    pub dramatic_intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualElements {
    pub costume_importance: f32,
    pub lighting_considerations: String,
    pub makeup_style: String,
    pub prop_usage: f32,
}

// === Implementation ===

impl MusicalStyle {
    /// Create a classical singing style
    pub fn classical() -> Self {
        Self {
            name: "Classical".to_string(),
            technique: SingingTechnique::classical(),
            characteristics: StyleCharacteristics {
                voice_types: vec![VoiceType::Soprano, VoiceType::Tenor, VoiceType::Bass],
                pitch_range: (48, 84),                               // C3 to C6
                intervals: vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 12.0], // Common intervals
                scales: vec![
                    Scale {
                        name: "Major".to_string(),
                        intervals: vec![0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
                        characteristic_notes: vec![0, 4, 7],
                    },
                    Scale {
                        name: "Natural Minor".to_string(),
                        intervals: vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0],
                        characteristic_notes: vec![0, 3, 7],
                    },
                ],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Even quarters".to_string(),
                    pattern: vec![1.0, 1.0, 1.0, 1.0],
                    accent_pattern: vec![true, false, false, false],
                    swing_feel: 0.0,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.3, 0.9),
                    contrast_level: 0.8,
                    sudden_changes: false,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.7,
                    warmth: 0.8,
                    edge: 0.3,
                    breathiness: 0.2,
                    richness: 0.9,
                    focus: 0.8,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Arch,
                dynamic_shaping: vec![DynamicShape {
                    start_dynamic: 0.5,
                    peak_position: 0.618, // Golden ratio
                    peak_dynamic: 0.8,
                    end_dynamic: 0.4,
                    curve_type: CurveType::Sinusoidal,
                }],
                rubato: RubatoStyle {
                    amount: 0.4,
                    phrase_initial: false,
                    phrase_final: true,
                    high_notes: true,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Musical,
                line_connection: LineConnection::Legato,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.3,
                    preferred_types: vec![GraceNoteType::Appoggiatura, GraceNoteType::Turn],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.2,
                    speed_range: (4.0, 8.0),
                    interval: 1.0,
                    termination: TrillTermination::WithTurn,
                },
                mordents: MordentStyle {
                    frequency: 0.1,
                    upper_lower_ratio: 0.7,
                    speed: 6.0,
                },
                turns: TurnStyle {
                    frequency: 0.15,
                    direction_preference: TurnDirection::Upper,
                    speed_variation: 0.2,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.25,
                    resolution_timing: 0.5,
                    emphasis_level: 0.7,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.1,
                    interval_threshold: 4.0,
                    speed_variation: 0.3,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.8,
                    dynamic_compensation: true,
                    breath_control_emphasis: 0.9,
                    proximity_effect_usage: 0.2,
                },
                stage_presence: StagePresence {
                    movement_level: 0.4,
                    gesture_integration: 0.6,
                    eye_contact_importance: 0.8,
                    dramatic_intensity: 0.7,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.8,
                    lighting_considerations: "Elegant".to_string(),
                    makeup_style: "Natural".to_string(),
                    prop_usage: 0.1,
                },
                audience_interaction: 0.3,
            },
        }
    }

    /// Create a pop singing style
    pub fn pop() -> Self {
        Self {
            name: "Pop".to_string(),
            technique: SingingTechnique::pop(),
            characteristics: StyleCharacteristics {
                voice_types: vec![VoiceType::Soprano, VoiceType::Tenor, VoiceType::Baritone],
                pitch_range: (55, 76), // G3 to E5
                intervals: vec![1.0, 2.0, 3.0, 5.0, 7.0, 12.0],
                scales: vec![Scale {
                    name: "Pop Major".to_string(),
                    intervals: vec![0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
                    characteristic_notes: vec![0, 4, 7, 9],
                }],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Pop groove".to_string(),
                    pattern: vec![1.0, 0.5, 1.0, 0.5],
                    accent_pattern: vec![true, false, true, false],
                    swing_feel: 0.1,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.4, 0.9),
                    contrast_level: 0.6,
                    sudden_changes: true,
                    gradual_changes: false,
                },
                timbre: TimbreQualities {
                    brightness: 0.8,
                    warmth: 0.6,
                    edge: 0.5,
                    breathiness: 0.3,
                    richness: 0.6,
                    focus: 0.7,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Linear,
                dynamic_shaping: vec![DynamicShape {
                    start_dynamic: 0.6,
                    peak_position: 0.8,
                    peak_dynamic: 0.9,
                    end_dynamic: 0.7,
                    curve_type: CurveType::Linear,
                }],
                rubato: RubatoStyle {
                    amount: 0.2,
                    phrase_initial: false,
                    phrase_final: false,
                    high_notes: false,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Grammatical,
                line_connection: LineConnection::SemiLegato,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.4,
                    preferred_types: vec![GraceNoteType::Acciaccatura],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.05,
                    speed_range: (6.0, 10.0),
                    interval: 0.5,
                    termination: TrillTermination::Abrupt,
                },
                mordents: MordentStyle {
                    frequency: 0.05,
                    upper_lower_ratio: 0.8,
                    speed: 8.0,
                },
                turns: TurnStyle {
                    frequency: 0.05,
                    direction_preference: TurnDirection::Both,
                    speed_variation: 0.4,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.1,
                    resolution_timing: 0.3,
                    emphasis_level: 0.5,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.3,
                    interval_threshold: 2.0,
                    speed_variation: 0.5,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.3,
                    dynamic_compensation: false,
                    breath_control_emphasis: 0.5,
                    proximity_effect_usage: 0.7,
                },
                stage_presence: StagePresence {
                    movement_level: 0.8,
                    gesture_integration: 0.9,
                    eye_contact_importance: 0.9,
                    dramatic_intensity: 0.8,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.9,
                    lighting_considerations: "Dynamic".to_string(),
                    makeup_style: "Performance".to_string(),
                    prop_usage: 0.5,
                },
                audience_interaction: 0.9,
            },
        }
    }

    /// Create a jazz singing style
    pub fn jazz() -> Self {
        Self {
            name: "Jazz".to_string(),
            technique: SingingTechnique::jazz(),
            characteristics: StyleCharacteristics {
                voice_types: vec![
                    VoiceType::Alto,
                    VoiceType::Baritone,
                    VoiceType::MezzoSoprano,
                ],
                pitch_range: (48, 81), // C3 to A5
                intervals: vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 9.0, 11.0], // Jazz intervals
                scales: vec![
                    Scale {
                        name: "Blues".to_string(),
                        intervals: vec![0.0, 3.0, 5.0, 6.0, 7.0, 10.0],
                        characteristic_notes: vec![0, 3, 7, 10],
                    },
                    Scale {
                        name: "Dorian".to_string(),
                        intervals: vec![0.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0],
                        characteristic_notes: vec![0, 3, 7, 9],
                    },
                ],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Swing".to_string(),
                    pattern: vec![1.0, 0.67, 0.33, 1.0],
                    accent_pattern: vec![true, false, true, false],
                    swing_feel: 0.67,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.2, 0.95),
                    contrast_level: 0.9,
                    sudden_changes: true,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.6,
                    warmth: 0.9,
                    edge: 0.4,
                    breathiness: 0.4,
                    richness: 0.8,
                    focus: 0.6,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Asymmetric,
                dynamic_shaping: vec![DynamicShape {
                    start_dynamic: 0.4,
                    peak_position: 0.7,
                    peak_dynamic: 0.85,
                    end_dynamic: 0.3,
                    curve_type: CurveType::Exponential,
                }],
                rubato: RubatoStyle {
                    amount: 0.7,
                    phrase_initial: true,
                    phrase_final: true,
                    high_notes: true,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Dramatic,
                line_connection: LineConnection::Mixed,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.6,
                    preferred_types: vec![GraceNoteType::Acciaccatura, GraceNoteType::Mordent],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.15,
                    speed_range: (3.0, 8.0),
                    interval: 1.0,
                    termination: TrillTermination::Natural,
                },
                mordents: MordentStyle {
                    frequency: 0.3,
                    upper_lower_ratio: 0.6,
                    speed: 5.0,
                },
                turns: TurnStyle {
                    frequency: 0.2,
                    direction_preference: TurnDirection::Both,
                    speed_variation: 0.6,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.4,
                    resolution_timing: 0.4,
                    emphasis_level: 0.8,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.5,
                    interval_threshold: 2.0,
                    speed_variation: 0.7,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.4,
                    dynamic_compensation: true,
                    breath_control_emphasis: 0.6,
                    proximity_effect_usage: 0.6,
                },
                stage_presence: StagePresence {
                    movement_level: 0.6,
                    gesture_integration: 0.8,
                    eye_contact_importance: 0.7,
                    dramatic_intensity: 0.9,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.6,
                    lighting_considerations: "Intimate".to_string(),
                    makeup_style: "Sophisticated".to_string(),
                    prop_usage: 0.2,
                },
                audience_interaction: 0.7,
            },
        }
    }

    /// Create a musical theater singing style
    pub fn musical_theater() -> Self {
        Self {
            name: "Musical Theater".to_string(),
            technique: SingingTechnique::default(), // Will customize below
            characteristics: StyleCharacteristics {
                voice_types: vec![
                    VoiceType::Soprano,
                    VoiceType::MezzoSoprano,
                    VoiceType::Tenor,
                    VoiceType::Baritone,
                ],
                pitch_range: (48, 86), // C3 to D6
                intervals: vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0],
                scales: vec![Scale {
                    name: "Broadway Major".to_string(),
                    intervals: vec![0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
                    characteristic_notes: vec![0, 4, 7],
                }],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Show tune".to_string(),
                    pattern: vec![1.0, 1.0, 0.5, 0.5, 1.0],
                    accent_pattern: vec![true, false, true, false, true],
                    swing_feel: 0.2,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.3, 1.0),
                    contrast_level: 0.95,
                    sudden_changes: true,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.9,
                    warmth: 0.7,
                    edge: 0.6,
                    breathiness: 0.1,
                    richness: 0.7,
                    focus: 0.9,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Arch,
                dynamic_shaping: vec![DynamicShape {
                    start_dynamic: 0.6,
                    peak_position: 0.75,
                    peak_dynamic: 0.95,
                    end_dynamic: 0.5,
                    curve_type: CurveType::Exponential,
                }],
                rubato: RubatoStyle {
                    amount: 0.5,
                    phrase_initial: true,
                    phrase_final: true,
                    high_notes: true,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Dramatic,
                line_connection: LineConnection::Articulated,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.5,
                    preferred_types: vec![GraceNoteType::Appoggiatura, GraceNoteType::Acciaccatura],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.1,
                    speed_range: (6.0, 12.0),
                    interval: 1.0,
                    termination: TrillTermination::WithTurn,
                },
                mordents: MordentStyle {
                    frequency: 0.1,
                    upper_lower_ratio: 0.8,
                    speed: 8.0,
                },
                turns: TurnStyle {
                    frequency: 0.15,
                    direction_preference: TurnDirection::Upper,
                    speed_variation: 0.3,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.3,
                    resolution_timing: 0.4,
                    emphasis_level: 0.9,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.4,
                    interval_threshold: 3.0,
                    speed_variation: 0.4,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Principal,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.5,
                    dynamic_compensation: true,
                    breath_control_emphasis: 0.8,
                    proximity_effect_usage: 0.4,
                },
                stage_presence: StagePresence {
                    movement_level: 0.9,
                    gesture_integration: 0.95,
                    eye_contact_importance: 0.9,
                    dramatic_intensity: 0.95,
                },
                visual_elements: VisualElements {
                    costume_importance: 1.0,
                    lighting_considerations: "Theatrical".to_string(),
                    makeup_style: "Stage".to_string(),
                    prop_usage: 0.8,
                },
                audience_interaction: 0.8,
            },
        }
    }

    /// Create a folk singing style
    pub fn folk() -> Self {
        Self {
            name: "Folk".to_string(),
            technique: SingingTechnique::folk(),
            characteristics: StyleCharacteristics {
                voice_types: vec![
                    VoiceType::Alto,
                    VoiceType::Baritone,
                    VoiceType::MezzoSoprano,
                    VoiceType::Tenor,
                ],
                pitch_range: (52, 76),                          // E3 to E5
                intervals: vec![1.0, 2.0, 4.0, 5.0, 7.0, 12.0], // Pentatonic emphasis
                scales: vec![
                    Scale {
                        name: "Pentatonic Major".to_string(),
                        intervals: vec![0.0, 2.0, 4.0, 7.0, 9.0],
                        characteristic_notes: vec![0, 4, 7],
                    },
                    Scale {
                        name: "Dorian".to_string(),
                        intervals: vec![0.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0],
                        characteristic_notes: vec![0, 3, 7],
                    },
                ],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Folk ballad".to_string(),
                    pattern: vec![2.0, 1.0, 1.0, 2.0],
                    accent_pattern: vec![true, false, false, true],
                    swing_feel: 0.0,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.4, 0.7),
                    contrast_level: 0.4,
                    sudden_changes: false,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.5,
                    warmth: 0.9,
                    edge: 0.2,
                    breathiness: 0.5,
                    richness: 0.6,
                    focus: 0.5,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Linear,
                dynamic_shaping: vec![DynamicShape {
                    start_dynamic: 0.5,
                    peak_position: 0.5,
                    peak_dynamic: 0.6,
                    end_dynamic: 0.5,
                    curve_type: CurveType::Linear,
                }],
                rubato: RubatoStyle {
                    amount: 0.3,
                    phrase_initial: false,
                    phrase_final: true,
                    high_notes: false,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Grammatical,
                line_connection: LineConnection::SemiLegato,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.2,
                    preferred_types: vec![GraceNoteType::Acciaccatura],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.05,
                    speed_range: (3.0, 5.0),
                    interval: 1.0,
                    termination: TrillTermination::Natural,
                },
                mordents: MordentStyle {
                    frequency: 0.1,
                    upper_lower_ratio: 0.5,
                    speed: 4.0,
                },
                turns: TurnStyle {
                    frequency: 0.05,
                    direction_preference: TurnDirection::Both,
                    speed_variation: 0.3,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.15,
                    resolution_timing: 0.6,
                    emphasis_level: 0.6,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.2,
                    interval_threshold: 3.0,
                    speed_variation: 0.3,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.6,
                    dynamic_compensation: false,
                    breath_control_emphasis: 0.4,
                    proximity_effect_usage: 0.3,
                },
                stage_presence: StagePresence {
                    movement_level: 0.3,
                    gesture_integration: 0.4,
                    eye_contact_importance: 0.8,
                    dramatic_intensity: 0.6,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.4,
                    lighting_considerations: "Natural".to_string(),
                    makeup_style: "Minimal".to_string(),
                    prop_usage: 0.6,
                },
                audience_interaction: 0.6,
            },
        }
    }

    /// Create a world music singing style template
    pub fn world_music(tradition: &str) -> Self {
        match tradition.to_lowercase().as_str() {
            "indian_classical" => Self::indian_classical(),
            "flamenco" => Self::flamenco(),
            "irish" => Self::irish_traditional(),
            "african" => Self::african_traditional(),
            "middle_eastern" => Self::middle_eastern(),
            _ => Self::world_music_generic(),
        }
    }

    /// Indian classical singing style
    fn indian_classical() -> Self {
        Self {
            name: "Indian Classical".to_string(),
            technique: SingingTechnique::default(),
            characteristics: StyleCharacteristics {
                voice_types: vec![VoiceType::Soprano, VoiceType::Tenor, VoiceType::Alto],
                pitch_range: (48, 96), // Very wide range
                intervals: vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], // Microtonal
                scales: vec![Scale {
                    name: "Raga Yaman".to_string(),
                    intervals: vec![0.0, 2.0, 4.5, 6.0, 7.0, 9.0, 11.5],
                    characteristic_notes: vec![0, 4, 7, 11],
                }],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "Teentaal".to_string(),
                    pattern: vec![
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0,
                    ],
                    accent_pattern: vec![
                        true, false, false, false, true, false, false, false, true, false, false,
                        false, true, false, false, false,
                    ],
                    swing_feel: 0.0,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.3, 0.9),
                    contrast_level: 0.8,
                    sudden_changes: false,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.7,
                    warmth: 0.8,
                    edge: 0.3,
                    breathiness: 0.1,
                    richness: 0.9,
                    focus: 0.9,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Undulating,
                dynamic_shaping: vec![],
                rubato: RubatoStyle {
                    amount: 0.9,
                    phrase_initial: true,
                    phrase_final: true,
                    high_notes: true,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Technical,
                line_connection: LineConnection::Legato,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.8,
                    preferred_types: vec![GraceNoteType::Acciaccatura, GraceNoteType::Trill],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.6,
                    speed_range: (2.0, 12.0),
                    interval: 0.5,
                    termination: TrillTermination::Natural,
                },
                mordents: MordentStyle {
                    frequency: 0.4,
                    upper_lower_ratio: 0.5,
                    speed: 6.0,
                },
                turns: TurnStyle {
                    frequency: 0.3,
                    direction_preference: TurnDirection::Both,
                    speed_variation: 0.8,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.5,
                    resolution_timing: 0.7,
                    emphasis_level: 0.8,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.9,
                    interval_threshold: 0.5,
                    speed_variation: 0.9,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.7,
                    dynamic_compensation: true,
                    breath_control_emphasis: 1.0,
                    proximity_effect_usage: 0.1,
                },
                stage_presence: StagePresence {
                    movement_level: 0.2,
                    gesture_integration: 0.3,
                    eye_contact_importance: 0.5,
                    dramatic_intensity: 0.8,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.8,
                    lighting_considerations: "Traditional".to_string(),
                    makeup_style: "Cultural".to_string(),
                    prop_usage: 0.3,
                },
                audience_interaction: 0.4,
            },
        }
    }

    /// Flamenco singing style
    fn flamenco() -> Self {
        let mut style = Self::world_music_generic();
        style.name = "Flamenco".to_string();
        style.characteristics.timbre.edge = 0.9;
        style.characteristics.timbre.brightness = 0.8;
        style.ornamentation.glissandos.frequency = 0.8;
        style.phrase_shaping.rubato.amount = 0.8;
        style.performance.stage_presence.dramatic_intensity = 1.0;
        style
    }

    /// Irish traditional singing style
    fn irish_traditional() -> Self {
        let mut style = Self::folk();
        style.name = "Irish Traditional".to_string();
        style.ornamentation.grace_notes.frequency = 0.6;
        style.characteristics.timbre.warmth = 1.0;
        style.phrase_shaping.line_connection = LineConnection::Legato;
        style
    }

    /// African traditional singing style
    fn african_traditional() -> Self {
        let mut style = Self::world_music_generic();
        style.name = "African Traditional".to_string();
        style.characteristics.rhythmic_patterns[0].accent_pattern =
            vec![true, false, true, false, false, true, false, true];
        style.characteristics.timbre.richness = 1.0;
        style.performance.ensemble_role = EnsembleRole::Chorus;
        style
    }

    /// Middle Eastern singing style
    fn middle_eastern() -> Self {
        let mut style = Self::world_music_generic();
        style.name = "Middle Eastern".to_string();
        style.characteristics.intervals = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; // Maqam intervals
        style.ornamentation.glissandos.frequency = 0.7;
        style.characteristics.timbre.edge = 0.6;
        style
    }

    /// Generic world music template
    fn world_music_generic() -> Self {
        Self {
            name: "World Music".to_string(),
            technique: SingingTechnique::default(),
            characteristics: StyleCharacteristics {
                voice_types: vec![
                    VoiceType::Soprano,
                    VoiceType::Alto,
                    VoiceType::Tenor,
                    VoiceType::Baritone,
                ],
                pitch_range: (48, 84),
                intervals: vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
                scales: vec![],
                rhythmic_patterns: vec![RhythmicPattern {
                    name: "World rhythm".to_string(),
                    pattern: vec![1.0, 0.5, 1.0, 0.5, 1.0],
                    accent_pattern: vec![true, false, true, false, true],
                    swing_feel: 0.3,
                }],
                dynamics: DynamicPreferences {
                    typical_range: (0.4, 0.8),
                    contrast_level: 0.7,
                    sudden_changes: false,
                    gradual_changes: true,
                },
                timbre: TimbreQualities {
                    brightness: 0.6,
                    warmth: 0.8,
                    edge: 0.4,
                    breathiness: 0.3,
                    richness: 0.8,
                    focus: 0.6,
                },
            },
            phrase_shaping: PhraseShaping {
                arc_type: PhraseArc::Undulating,
                dynamic_shaping: vec![],
                rubato: RubatoStyle {
                    amount: 0.6,
                    phrase_initial: true,
                    phrase_final: true,
                    high_notes: false,
                    emotional_peaks: true,
                },
                breath_placement: BreathPlacement::Mixed,
                line_connection: LineConnection::Mixed,
            },
            ornamentation: Ornamentation {
                grace_notes: GraceNoteStyle {
                    frequency: 0.5,
                    preferred_types: vec![GraceNoteType::Acciaccatura, GraceNoteType::Appoggiatura],
                    placement_rules: vec![],
                },
                trills: TrillStyle {
                    frequency: 0.3,
                    speed_range: (3.0, 8.0),
                    interval: 1.0,
                    termination: TrillTermination::Natural,
                },
                mordents: MordentStyle {
                    frequency: 0.2,
                    upper_lower_ratio: 0.6,
                    speed: 5.0,
                },
                turns: TurnStyle {
                    frequency: 0.2,
                    direction_preference: TurnDirection::Both,
                    speed_variation: 0.5,
                },
                appoggiaturas: AppoggiaturaStyle {
                    frequency: 0.3,
                    resolution_timing: 0.5,
                    emphasis_level: 0.7,
                },
                glissandos: GlissandoStyle {
                    frequency: 0.4,
                    interval_threshold: 2.0,
                    speed_variation: 0.6,
                },
            },
            cultural_variants: HashMap::new(),
            performance: PerformanceGuidelines {
                ensemble_role: EnsembleRole::Soloist,
                microphone_technique: MicrophoneTechnique {
                    distance_preference: 0.6,
                    dynamic_compensation: true,
                    breath_control_emphasis: 0.7,
                    proximity_effect_usage: 0.4,
                },
                stage_presence: StagePresence {
                    movement_level: 0.6,
                    gesture_integration: 0.7,
                    eye_contact_importance: 0.7,
                    dramatic_intensity: 0.7,
                },
                visual_elements: VisualElements {
                    costume_importance: 0.8,
                    lighting_considerations: "Cultural".to_string(),
                    makeup_style: "Traditional".to_string(),
                    prop_usage: 0.5,
                },
                audience_interaction: 0.6,
            },
        }
    }

    /// Apply the musical style to a note event
    pub fn apply_to_note(&self, note: &mut NoteEvent) {
        // Apply base technique
        self.technique.apply_to_note(note);

        // Apply style-specific characteristics
        self.apply_timbre_qualities(note);
        self.apply_ornamentation_probability(note);
        self.apply_phrase_shaping(note);
        self.apply_cultural_variations(note);
    }

    /// Apply timbre qualities from the style
    fn apply_timbre_qualities(&self, note: &mut NoteEvent) {
        let timbre = &self.characteristics.timbre;

        // Modify velocity based on dynamic preferences
        note.velocity *= timbre.focus * timbre.richness;
        note.velocity = note.velocity.clamp(0.0, 1.0);

        // Apply breathiness
        note.breath_before += timbre.breathiness * 0.3;
        note.breath_before = note.breath_before.clamp(0.0, 1.0);

        // Modify vibrato based on warmth and edge
        note.vibrato *= timbre.warmth;
        if timbre.edge > 0.5 {
            note.vibrato *= 0.7; // Reduce vibrato for edgy sounds
        }
    }

    /// Apply ornamentation probability
    fn apply_ornamentation_probability(&self, note: &mut NoteEvent) {
        // This is a simplified version - in practice, you'd use random number generation
        // and more sophisticated rules based on musical context
        let grace_note_prob = self.ornamentation.grace_notes.frequency;

        // Simple heuristic: apply ornamentation to longer notes
        if note.duration > 1.0 && grace_note_prob > 0.5 {
            // Add slight pitch bend to simulate grace note effect
            note.frequency *= 1.02;
        }
    }

    /// Apply phrase shaping characteristics
    fn apply_phrase_shaping(&self, note: &mut NoteEvent) {
        let rubato = &self.phrase_shaping.rubato;

        // Apply rubato timing adjustments
        if rubato.amount > 0.5 {
            // Slight timing adjustment for expressive phrasing
            // This would need more context in a real implementation
            note.duration *= 1.0 + (rubato.amount - 0.5) * 0.1;
        }
    }

    /// Apply cultural variations
    fn apply_cultural_variations(&self, _note: &mut NoteEvent) {
        // This would apply cultural-specific modifications
        // Implementation would depend on the specific cultural variant
    }

    /// Get recommended voice characteristics for this style
    pub fn get_voice_characteristics(&self) -> VoiceCharacteristics {
        let mut resonance = HashMap::new();
        resonance.insert("chest".to_string(), self.technique.resonance.chest);
        resonance.insert("head".to_string(), self.technique.resonance.head);

        let mut timbre = HashMap::new();
        timbre.insert(
            "brightness".to_string(),
            self.characteristics.timbre.brightness,
        );
        timbre.insert("warmth".to_string(), self.characteristics.timbre.warmth);

        VoiceCharacteristics {
            voice_type: self.characteristics.voice_types[0].clone().into(),
            range: (
                self.characteristics.pitch_range.0 as f32,
                self.characteristics.pitch_range.1 as f32,
            ),
            f0_mean: (self.characteristics.pitch_range.0 + self.characteristics.pitch_range.1)
                as f32
                / 2.0,
            f0_std: 20.0, // Default value
            vibrato_frequency: self.technique.vibrato.frequency,
            vibrato_depth: self.technique.vibrato.depth,
            breath_capacity: self.technique.breath_control.capacity,
            vocal_power: self.technique.dynamics.base_volume,
            resonance,
            timbre,
        }
    }
}

// Conversion helpers
impl From<VoiceType> for crate::types::VoiceType {
    fn from(voice_type: VoiceType) -> Self {
        match voice_type {
            VoiceType::Soprano => crate::types::VoiceType::Soprano,
            VoiceType::MezzoSoprano => crate::types::VoiceType::MezzoSoprano,
            VoiceType::Alto => crate::types::VoiceType::Alto,
            VoiceType::Tenor => crate::types::VoiceType::Tenor,
            VoiceType::Baritone => crate::types::VoiceType::Baritone,
            VoiceType::Bass => crate::types::VoiceType::Bass,
            VoiceType::Countertenor => crate::types::VoiceType::Soprano, // Approximate mapping
            VoiceType::Contralto => crate::types::VoiceType::Alto,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_style_creation() {
        let style = MusicalStyle::classical();
        assert_eq!(style.name, "Classical");
        assert!(style.characteristics.timbre.richness > 0.8);
        assert_eq!(style.phrase_shaping.arc_type, PhraseArc::Arch);
    }

    #[test]
    fn test_pop_style_creation() {
        let style = MusicalStyle::pop();
        assert_eq!(style.name, "Pop");
        assert!(style.characteristics.timbre.brightness > 0.7);
        assert!(style.performance.audience_interaction > 0.8);
    }

    #[test]
    fn test_jazz_style_creation() {
        let style = MusicalStyle::jazz();
        assert_eq!(style.name, "Jazz");
        assert!(style.phrase_shaping.rubato.amount > 0.6);
        assert!(style.ornamentation.appoggiaturas.frequency > 0.3);
    }

    #[test]
    fn test_musical_theater_style() {
        let style = MusicalStyle::musical_theater();
        assert_eq!(style.name, "Musical Theater");
        assert!(style.characteristics.dynamics.contrast_level > 0.9);
        assert!(style.performance.stage_presence.dramatic_intensity > 0.9);
    }

    #[test]
    fn test_folk_style_creation() {
        let style = MusicalStyle::folk();
        assert_eq!(style.name, "Folk");
        assert!(style.characteristics.timbre.warmth > 0.8);
        assert!(style.characteristics.dynamics.contrast_level < 0.5);
    }

    #[test]
    fn test_world_music_variations() {
        let indian = MusicalStyle::world_music("indian_classical");
        let flamenco = MusicalStyle::world_music("flamenco");
        let irish = MusicalStyle::world_music("irish");

        assert_eq!(indian.name, "Indian Classical");
        assert_eq!(flamenco.name, "Flamenco");
        assert_eq!(irish.name, "Irish Traditional");

        assert!(indian.ornamentation.trills.frequency > flamenco.ornamentation.trills.frequency);
        assert!(flamenco.characteristics.timbre.edge > irish.characteristics.timbre.edge);
    }

    #[test]
    fn test_style_note_application() {
        let style = MusicalStyle::jazz();
        let mut note = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);

        let original_velocity = note.velocity;
        style.apply_to_note(&mut note);

        // Note should be modified by the style
        assert!(
            note.velocity != original_velocity || note.vibrato != 0.0 || note.breath_before != 0.0
        );
    }

    #[test]
    fn test_voice_characteristics_extraction() {
        let style = MusicalStyle::classical();
        let characteristics = style.get_voice_characteristics();

        assert!(characteristics.range.0 < characteristics.range.1);
        assert!(characteristics.vibrato_depth >= 0.0 && characteristics.vibrato_depth <= 1.0);
        assert!(characteristics.vibrato_frequency > 0.0);
    }
}
