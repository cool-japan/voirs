//! Audio-to-visual mapping configuration

use super::types::{ColorRGBA, DirectionZone, ShapeType, VisualEffect, VisualElementType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Audio-to-visual mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioVisualMapping {
    /// Low frequency visual mapping
    pub low_freq_mapping: FrequencyVisualMapping,

    /// Mid frequency visual mapping
    pub mid_freq_mapping: FrequencyVisualMapping,

    /// High frequency visual mapping
    pub high_freq_mapping: FrequencyVisualMapping,

    /// Amplitude-based visual scaling
    pub amplitude_scaling: AmplitudeVisualMapping,

    /// Directional visual cues
    pub directional_cues: DirectionalCueMapping,

    /// Event-based visual triggers
    pub event_triggers: EventTriggerMapping,
}

/// Frequency band to visual mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyVisualMapping {
    /// Frequency range (Hz)
    pub frequency_range: (f32, f32),

    /// Visual element type for this frequency
    pub element_type: VisualElementType,

    /// Base color for this frequency band
    pub base_color: ColorRGBA,

    /// Intensity scaling factor
    pub intensity_scale: f32,

    /// Size scaling factor
    pub size_scale: f32,

    /// Animation responsiveness
    pub animation_responsiveness: f32,
}

/// Amplitude-based visual scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplitudeVisualMapping {
    /// Minimum amplitude for visual activation
    pub threshold: f32,

    /// Amplitude to intensity scaling
    pub intensity_curve: ScalingCurve,

    /// Amplitude to size scaling
    pub size_curve: ScalingCurve,

    /// Dynamic range compression
    pub compression_ratio: f32,
}

/// Scaling curve types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScalingCurve {
    /// Linear scaling
    Linear,

    /// Logarithmic scaling
    Logarithmic,

    /// Exponential scaling
    Exponential,

    /// Power law scaling
    Power(f32),

    /// Custom curve
    Custom,
}

/// Directional visual cue mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalCueMapping {
    /// Enable directional indicators
    pub enabled: bool,

    /// Arrow/pointer visual element
    pub directional_element: VisualElementType,

    /// Color coding for direction zones
    pub direction_colors: HashMap<DirectionZone, ColorRGBA>,

    /// Distance-based directional scaling
    pub distance_scaling: bool,

    /// Peripheral vision enhancement
    pub peripheral_enhancement: bool,
}

/// Event-based visual trigger mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTriggerMapping {
    /// Audio onset triggers
    pub onset_triggers: Vec<OnsetTrigger>,

    /// Beat/rhythm triggers
    pub rhythm_triggers: Vec<RhythmTrigger>,

    /// Spectral event triggers
    pub spectral_triggers: Vec<SpectralTrigger>,

    /// Silence/quiet triggers
    pub silence_triggers: Vec<SilenceTrigger>,
}

/// Audio onset trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsetTrigger {
    /// Trigger sensitivity
    pub sensitivity: f32,

    /// Visual effect to trigger
    pub effect: VisualEffect,

    /// Minimum time between triggers
    pub cooldown: Duration,
}

/// Rhythm-based trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmTrigger {
    /// Target tempo range (BPM)
    pub tempo_range: (f32, f32),

    /// Beat emphasis visual effect
    pub beat_effect: VisualEffect,

    /// Downbeat special effect
    pub downbeat_effect: Option<VisualEffect>,

    /// Rhythm confidence threshold
    pub confidence_threshold: f32,
}

/// Spectral event trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralTrigger {
    /// Frequency range to monitor
    pub frequency_range: (f32, f32),

    /// Energy change threshold
    pub energy_threshold: f32,

    /// Visual effect for spectral events
    pub effect: VisualEffect,

    /// Trigger duration
    pub duration: Duration,
}

/// Silence detection trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceTrigger {
    /// Silence threshold (dB)
    pub threshold_db: f32,

    /// Minimum silence duration
    pub min_duration: Duration,

    /// Visual effect during silence
    pub silence_effect: VisualEffect,

    /// Visual effect for silence end
    pub end_effect: Option<VisualEffect>,
}

/// Visual distance attenuation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualDistanceAttenuation {
    /// Minimum distance for full intensity
    pub min_distance: f32,

    /// Maximum distance for zero intensity
    pub max_distance: f32,

    /// Attenuation curve type
    pub curve_type: ScalingCurve,

    /// Size scaling with distance
    pub size_scaling: bool,

    /// Perspective correction
    pub perspective_correction: bool,
}

// Default implementations

impl Default for AudioVisualMapping {
    fn default() -> Self {
        Self {
            low_freq_mapping: FrequencyVisualMapping {
                frequency_range: (20.0, 250.0),
                element_type: VisualElementType::PointLight,
                base_color: ColorRGBA {
                    r: 1.0,
                    g: 0.2,
                    b: 0.2,
                    a: 0.8,
                },
                intensity_scale: 1.5,
                size_scale: 1.2,
                animation_responsiveness: 0.8,
            },
            mid_freq_mapping: FrequencyVisualMapping {
                frequency_range: (250.0, 4000.0),
                element_type: VisualElementType::Shape(ShapeType::Sphere),
                base_color: ColorRGBA {
                    r: 0.2,
                    g: 1.0,
                    b: 0.2,
                    a: 0.8,
                },
                intensity_scale: 1.0,
                size_scale: 1.0,
                animation_responsiveness: 1.0,
            },
            high_freq_mapping: FrequencyVisualMapping {
                frequency_range: (4000.0, 20000.0),
                element_type: VisualElementType::ParticleEffect,
                base_color: ColorRGBA {
                    r: 0.2,
                    g: 0.2,
                    b: 1.0,
                    a: 0.8,
                },
                intensity_scale: 0.8,
                size_scale: 0.6,
                animation_responsiveness: 1.5,
            },
            amplitude_scaling: AmplitudeVisualMapping {
                threshold: 0.1,
                intensity_curve: ScalingCurve::Logarithmic,
                size_curve: ScalingCurve::Linear,
                compression_ratio: 3.0,
            },
            directional_cues: DirectionalCueMapping::default(),
            event_triggers: EventTriggerMapping::default(),
        }
    }
}

impl Default for DirectionalCueMapping {
    fn default() -> Self {
        let mut direction_colors = HashMap::new();
        direction_colors.insert(
            DirectionZone::Front,
            ColorRGBA {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        );
        direction_colors.insert(
            DirectionZone::Left,
            ColorRGBA {
                r: 1.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        );
        direction_colors.insert(
            DirectionZone::Back,
            ColorRGBA {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );
        direction_colors.insert(
            DirectionZone::Right,
            ColorRGBA {
                r: 0.0,
                g: 0.0,
                b: 1.0,
                a: 1.0,
            },
        );
        direction_colors.insert(
            DirectionZone::Above,
            ColorRGBA {
                r: 1.0,
                g: 0.0,
                b: 1.0,
                a: 1.0,
            },
        );
        direction_colors.insert(
            DirectionZone::Below,
            ColorRGBA {
                r: 0.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        );

        Self {
            enabled: true,
            directional_element: VisualElementType::Shape(ShapeType::Arrow),
            direction_colors,
            distance_scaling: true,
            peripheral_enhancement: true,
        }
    }
}

impl Default for EventTriggerMapping {
    fn default() -> Self {
        Self {
            onset_triggers: vec![],
            rhythm_triggers: vec![],
            spectral_triggers: vec![],
            silence_triggers: vec![],
        }
    }
}

impl Default for VisualDistanceAttenuation {
    fn default() -> Self {
        Self {
            min_distance: 0.5,
            max_distance: 20.0,
            curve_type: ScalingCurve::Linear,
            size_scaling: true,
            perspective_correction: true,
        }
    }
}
