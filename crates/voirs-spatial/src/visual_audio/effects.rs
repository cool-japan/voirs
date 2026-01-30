//! Visual effects library and management

use super::types::{
    AnimationParams, AnimationType, ColorRGBA, EasingFunction, FrequencyBand, ShapeType,
    SpatialVisualEvent, VisualEffect, VisualElement, VisualElementType, VisualEventType,
};
use crate::{Position3D, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Visual effect library management
pub(crate) struct VisualEffectLibrary {
    /// Built-in effects
    builtin_effects: HashMap<String, VisualEffect>,

    /// User-created effects
    user_effects: HashMap<String, VisualEffect>,

    /// Effect templates
    templates: HashMap<String, VisualEffectTemplate>,

    /// Usage statistics
    usage_stats: HashMap<String, EffectUsageStats>,
}

/// Visual effect template for generating effects
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisualEffectTemplate {
    /// Template name
    name: String,

    /// Template description
    description: String,

    /// Parameter definitions
    parameters: Vec<TemplateParameter>,

    /// Effect generation function
    generator: String, // Function name for effect generation
}

/// Template parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemplateParameter {
    /// Parameter name
    name: String,

    /// Parameter type
    param_type: ParameterType,

    /// Default value
    default_value: ParameterValue,

    /// Value range/constraints
    constraints: ParameterConstraints,
}

/// Parameter types for templates
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ParameterType {
    /// Floating point number
    Float,

    /// Integer number
    Integer,

    /// Boolean value
    Boolean,

    /// String value
    String,

    /// Color value
    Color,

    /// Position value
    Position,

    /// Duration value
    Duration,
}

/// Parameter value union
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ParameterValue {
    /// Float value
    Float(f32),

    /// Integer value
    Integer(i32),

    /// Boolean value
    Boolean(bool),

    /// String value
    String(String),

    /// Color value
    Color(ColorRGBA),

    /// Position value
    Position(Position3D),

    /// Duration value
    Duration(Duration),
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ParameterConstraints {
    /// No constraints
    None,

    /// Numeric range
    Range { min: f64, max: f64 },

    /// Enumerated values
    Enum(Vec<String>),

    /// String pattern
    Pattern(String),
}

/// Effect usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectUsageStats {
    /// Usage count
    usage_count: u32,

    /// Average user rating
    average_rating: f32,

    /// Performance metrics
    performance_stats: EffectPerformanceStats,

    /// Last used timestamp
    #[serde(skip)]
    last_used: Option<Instant>,
}

/// Performance statistics for effects
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectPerformanceStats {
    /// Average render time
    avg_render_time: f32,

    /// Memory usage
    memory_usage: usize,

    /// GPU utilization
    gpu_utilization: f32,

    /// Frame rate impact
    fps_impact: f32,
}

impl VisualEffectLibrary {
    pub(crate) fn new() -> Self {
        let mut builtin_effects = HashMap::new();

        // Add some built-in effects
        builtin_effects.insert("bass_glow".to_string(), Self::create_bass_glow_effect());
        builtin_effects.insert(
            "treble_sparkle".to_string(),
            Self::create_treble_sparkle_effect(),
        );
        builtin_effects.insert("beat_pulse".to_string(), Self::create_beat_pulse_effect());

        Self {
            builtin_effects,
            user_effects: HashMap::new(),
            templates: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }

    pub(crate) fn get_effect(&self, effect_id: &str) -> Option<&VisualEffect> {
        self.builtin_effects
            .get(effect_id)
            .or_else(|| self.user_effects.get(effect_id))
    }

    pub(crate) fn select_effect_for_event(
        &self,
        event: &SpatialVisualEvent,
    ) -> Result<VisualEffect> {
        // Simple effect selection logic
        let effect_id = match &event.base_event.event_type {
            VisualEventType::FrequencyBand(FrequencyBand::Low) => "bass_glow",
            VisualEventType::FrequencyBand(FrequencyBand::High) => "treble_sparkle",
            VisualEventType::Beat | VisualEventType::Downbeat => "beat_pulse",
            _ => "bass_glow",
        };

        self.get_effect(effect_id)
            .cloned()
            .ok_or_else(|| crate::Error::LegacyProcessing("Effect not found".to_string()))
    }

    pub(crate) fn size(&self) -> usize {
        self.builtin_effects.len() + self.user_effects.len()
    }

    fn create_bass_glow_effect() -> VisualEffect {
        VisualEffect {
            id: "bass_glow".to_string(),
            name: "Bass Glow".to_string(),
            elements: vec![VisualElement {
                start_time: Duration::from_millis(0),
                duration: Duration::from_millis(500),
                element_type: VisualElementType::PointLight,
                color: ColorRGBA {
                    r: 1.0,
                    g: 0.3,
                    b: 0.1,
                    a: 0.8,
                },
                intensity: 0.8,
                size: 2.0,
                animation: Some(AnimationParams {
                    animation_type: AnimationType::Pulse,
                    speed: 1.0,
                    amplitude: 0.5,
                    phase: 0.0,
                    easing: EasingFunction::EaseInOut,
                }),
                distance_attenuation: 1.0,
            }],
            duration: Duration::from_millis(500),
            looping: false,
            priority: 5,
            position: Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            audio_source_id: None,
        }
    }

    fn create_treble_sparkle_effect() -> VisualEffect {
        VisualEffect {
            id: "treble_sparkle".to_string(),
            name: "Treble Sparkle".to_string(),
            elements: vec![VisualElement {
                start_time: Duration::from_millis(0),
                duration: Duration::from_millis(200),
                element_type: VisualElementType::ParticleEffect,
                color: ColorRGBA {
                    r: 0.8,
                    g: 0.8,
                    b: 1.0,
                    a: 0.9,
                },
                intensity: 1.0,
                size: 0.5,
                animation: Some(AnimationParams {
                    animation_type: AnimationType::Fade,
                    speed: 2.0,
                    amplitude: 1.0,
                    phase: 0.0,
                    easing: EasingFunction::EaseOut,
                }),
                distance_attenuation: 1.0,
            }],
            duration: Duration::from_millis(200),
            looping: false,
            priority: 8,
            position: Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            audio_source_id: None,
        }
    }

    fn create_beat_pulse_effect() -> VisualEffect {
        VisualEffect {
            id: "beat_pulse".to_string(),
            name: "Beat Pulse".to_string(),
            elements: vec![VisualElement {
                start_time: Duration::from_millis(0),
                duration: Duration::from_millis(300),
                element_type: VisualElementType::Shape(ShapeType::Ring),
                color: ColorRGBA {
                    r: 0.2,
                    g: 1.0,
                    b: 0.2,
                    a: 0.7,
                },
                intensity: 0.9,
                size: 1.5,
                animation: Some(AnimationParams {
                    animation_type: AnimationType::Scale,
                    speed: 1.5,
                    amplitude: 2.0,
                    phase: 0.0,
                    easing: EasingFunction::EaseOut,
                }),
                distance_attenuation: 1.0,
            }],
            duration: Duration::from_millis(300),
            looping: false,
            priority: 7,
            position: Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            audio_source_id: None,
        }
    }
}
