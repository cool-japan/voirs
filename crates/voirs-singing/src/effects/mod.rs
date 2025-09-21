//! Audio effects for singing synthesis
//!
//! This module provides a comprehensive set of audio effects specifically designed
//! for singing voice synthesis, including time-based effects, dynamic processing,
//! spectral manipulation, and synthesis-based enhancements.

pub mod core;
pub mod dynamic_effects;
pub mod filters;
pub mod helpers;
pub mod spectral_effects;
pub mod synthesis_effects;
pub mod time_effects;

// Re-export main types and traits
pub use core::{EffectChain, EffectProcessor, SingingEffect};

// Re-export filter types
pub use filters::{
    AllPassFilter, BandPassFilter, DelayLine, HighPassFilter, HighShelfFilter, LowPassFilter,
    LowShelfFilter, PeakingFilter,
};

// Re-export helper types
pub use helpers::{
    AntiFormantFilter, EnvelopeFollower, FormantFilter, InterpolationMode, InterpolationType,
    LFOWaveform, MorphType, NoiseGenerator, NoiseType, PhaseAlignment, LFO,
};

// Re-export time-based effects
pub use time_effects::{ChorusEffect, ReverbEffect, VibratoEffect};

// Re-export dynamic effects
pub use dynamic_effects::{CompressorEffect, GateEffect, LimiterEffect};

// Re-export synthesis effects
pub use synthesis_effects::{BreathNoiseEffect, HarmonicsEffect, VocalFryEffect};

// Re-export spectral effects
pub use spectral_effects::{EQEffect, FormantControlEffect, SpectralMorphingEffect};

/// Create a default effect chain with commonly used singing effects
pub fn create_default_chain() -> EffectChain {
    EffectChain::with_defaults()
}

/// Effect categories for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectCategory {
    /// Time-based effects (reverb, chorus, delay)
    Time,
    /// Dynamic range effects (compressor, limiter, gate)
    Dynamics,
    /// Spectral effects (EQ, formant control)
    Spectral,
    /// Synthesis-based effects (breath noise, vocal fry)
    Synthesis,
    /// Utility effects (gain, pan)
    Utility,
}

/// Effect information structure
#[derive(Debug, Clone)]
pub struct EffectInfo {
    /// Effect name
    pub name: String,
    /// Effect category
    pub category: EffectCategory,
    /// Available parameters
    pub parameters: Vec<String>,
    /// Parameter descriptions
    pub parameter_descriptions: HashMap<String, String>,
}

/// Get information about all available effects
pub fn get_available_effects() -> Vec<EffectInfo> {
    vec![
        EffectInfo {
            name: "reverb".to_string(),
            category: EffectCategory::Time,
            parameters: vec![
                "room_size".to_string(),
                "damping".to_string(),
                "wet_level".to_string(),
                "dry_level".to_string(),
                "width".to_string(),
                "pre_delay".to_string(),
            ],
            parameter_descriptions: {
                let mut map = HashMap::new();
                map.insert(
                    "room_size".to_string(),
                    "Size of the reverb room (0.0-1.0)".to_string(),
                );
                map.insert(
                    "damping".to_string(),
                    "High frequency damping (0.0-1.0)".to_string(),
                );
                map.insert(
                    "wet_level".to_string(),
                    "Wet signal level (0.0-1.0)".to_string(),
                );
                map.insert(
                    "dry_level".to_string(),
                    "Dry signal level (0.0-1.0)".to_string(),
                );
                map.insert("width".to_string(), "Stereo width (0.0-1.0)".to_string());
                map.insert(
                    "pre_delay".to_string(),
                    "Pre-delay in milliseconds".to_string(),
                );
                map
            },
        },
        EffectInfo {
            name: "chorus".to_string(),
            category: EffectCategory::Time,
            parameters: vec![
                "rate".to_string(),
                "depth".to_string(),
                "feedback".to_string(),
                "mix".to_string(),
            ],
            parameter_descriptions: {
                let mut map = HashMap::new();
                map.insert("rate".to_string(), "LFO rate in Hz".to_string());
                map.insert(
                    "depth".to_string(),
                    "Modulation depth (0.0-1.0)".to_string(),
                );
                map.insert(
                    "feedback".to_string(),
                    "Feedback amount (0.0-0.99)".to_string(),
                );
                map.insert("mix".to_string(), "Dry/wet mix (0.0-1.0)".to_string());
                map
            },
        },
        EffectInfo {
            name: "compressor".to_string(),
            category: EffectCategory::Dynamics,
            parameters: vec![
                "threshold".to_string(),
                "ratio".to_string(),
                "attack".to_string(),
                "release".to_string(),
                "knee".to_string(),
                "makeup_gain".to_string(),
            ],
            parameter_descriptions: {
                let mut map = HashMap::new();
                map.insert(
                    "threshold".to_string(),
                    "Compression threshold in dB".to_string(),
                );
                map.insert(
                    "ratio".to_string(),
                    "Compression ratio (1.0-20.0)".to_string(),
                );
                map.insert("attack".to_string(), "Attack time in seconds".to_string());
                map.insert("release".to_string(), "Release time in seconds".to_string());
                map.insert("knee".to_string(), "Soft knee amount (0.0-1.0)".to_string());
                map.insert("makeup_gain".to_string(), "Makeup gain in dB".to_string());
                map
            },
        },
        EffectInfo {
            name: "breath_noise".to_string(),
            category: EffectCategory::Synthesis,
            parameters: vec![
                "level".to_string(),
                "frequency".to_string(),
                "bandwidth".to_string(),
                "sensitivity".to_string(),
            ],
            parameter_descriptions: {
                let mut map = HashMap::new();
                map.insert(
                    "level".to_string(),
                    "Breath noise level (0.0-1.0)".to_string(),
                );
                map.insert(
                    "frequency".to_string(),
                    "Filter center frequency in Hz".to_string(),
                );
                map.insert(
                    "bandwidth".to_string(),
                    "Filter bandwidth in Hz".to_string(),
                );
                map.insert(
                    "sensitivity".to_string(),
                    "Envelope sensitivity (0.0-1.0)".to_string(),
                );
                map
            },
        },
    ]
}

use std::collections::HashMap;
