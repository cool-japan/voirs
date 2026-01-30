//! # Visual Audio Integration for Spatial Audio
//!
//! This module provides visual spatial cues integration with spatial audio systems,
//! enabling multi-sensory experiences that combine audio positioning with visual feedback.
//! This includes visual indicators, lighting effects, AR overlays, and accessibility features.

mod analyzer;
mod config;
mod effects;
mod mapping;
mod processor;
mod sync;
mod types;

#[cfg(test)]
mod tests;

// Public re-exports from types
pub use types::{
    AnimationParams, AnimationType, ColorRGBA, DirectionZone, EasingFunction, ShapeType,
    VisualAudioMetrics, VisualDisplay, VisualDisplayCapabilities, VisualEffect, VisualElement,
    VisualElementType, VisualResourceUsage,
};

// Public re-exports from config
pub use config::{
    ColorScheme, ColorSchemeType, VisualAccessibilitySettings, VisualAudioConfig,
    VisualPerformanceSettings,
};

// Public re-exports from mapping
pub use mapping::{
    AmplitudeVisualMapping, AudioVisualMapping, DirectionalCueMapping, EventTriggerMapping,
    FrequencyVisualMapping, OnsetTrigger, RhythmTrigger, ScalingCurve, SilenceTrigger,
    SpectralTrigger, VisualDistanceAttenuation,
};

// Public re-exports from sync
pub use sync::VisualSyncSettings;

// Public re-exports from processor
pub use processor::VisualAudioProcessor;
