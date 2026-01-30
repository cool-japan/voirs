//! # ControlParameter - Trait Implementations
//!
//! This module contains trait implementations for `ControlParameter`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ControlParameter;

impl std::fmt::Display for ControlParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlParameter::Volume => write!(f, "volume"),
            ControlParameter::PitchBend => write!(f, "pitch_bend"),
            ControlParameter::VibratoRate => write!(f, "vibrato_rate"),
            ControlParameter::VibratoDepth => write!(f, "vibrato_depth"),
            ControlParameter::BreathIntensity => write!(f, "breath_intensity"),
            ControlParameter::VoiceCharacteristic(name) => write!(f, "voice_{}", name),
            ControlParameter::EffectParameter(effect, param) => {
                write!(f, "effect_{}_{}", effect, param)
            }
        }
    }
}
