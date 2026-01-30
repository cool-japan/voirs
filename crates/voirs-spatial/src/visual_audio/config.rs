//! Configuration types for visual audio integration

use super::mapping::{AudioVisualMapping, VisualDistanceAttenuation};
use super::sync::VisualSyncSettings;
use super::types::ColorRGBA;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for visual-audio integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAudioConfig {
    /// Enable visual feedback
    pub enabled: bool,

    /// Master visual intensity multiplier
    pub master_intensity: f32,

    /// Audio-to-visual mapping settings
    pub audio_mapping: AudioVisualMapping,

    /// Synchronization settings
    pub sync_settings: VisualSyncSettings,

    /// Distance-based visual attenuation
    pub distance_attenuation: VisualDistanceAttenuation,

    /// Color scheme preferences
    pub color_scheme: ColorScheme,

    /// Accessibility settings
    pub accessibility: VisualAccessibilitySettings,

    /// Performance optimization settings
    pub performance: VisualPerformanceSettings,
}

/// Color scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Color scheme type
    pub scheme_type: ColorSchemeType,

    /// Base colors for different audio sources
    pub source_colors: HashMap<String, ColorRGBA>,

    /// Ambient lighting color
    pub ambient_color: ColorRGBA,

    /// Color temperature (Kelvin)
    pub color_temperature: f32,

    /// Saturation level
    pub saturation: f32,

    /// Brightness level
    pub brightness: f32,
}

/// Color scheme types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSchemeType {
    /// Default balanced colors
    Default,

    /// High contrast for accessibility
    HighContrast,

    /// Warm color palette
    Warm,

    /// Cool color palette
    Cool,

    /// Monochromatic
    Monochromatic,

    /// Custom user-defined scheme
    Custom,
}

/// Visual accessibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAccessibilitySettings {
    /// High contrast mode
    pub high_contrast: bool,

    /// Color blind friendly palette
    pub color_blind_friendly: bool,

    /// Motion sensitivity reduction
    pub reduced_motion: bool,

    /// Visual indicator scaling
    pub indicator_scaling: f32,

    /// Text size scaling
    pub text_scaling: f32,

    /// Audio description visual cues
    pub audio_description_mode: bool,

    /// Screen reader compatibility
    pub screen_reader_compatible: bool,
}

/// Visual performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPerformanceSettings {
    /// Target frame rate
    pub target_fps: f32,

    /// Quality level (0.0-1.0)
    pub quality_level: f32,

    /// Level of detail (LOD) enable
    pub lod_enabled: bool,

    /// Culling distance
    pub culling_distance: f32,

    /// Maximum particle count
    pub max_particles: usize,

    /// Anti-aliasing level
    pub anti_aliasing: u8,

    /// Adaptive quality
    pub adaptive_quality: bool,
}

// Default implementations

impl Default for VisualAudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            master_intensity: 0.8,
            audio_mapping: AudioVisualMapping::default(),
            sync_settings: VisualSyncSettings::default(),
            distance_attenuation: VisualDistanceAttenuation::default(),
            color_scheme: ColorScheme::default(),
            accessibility: VisualAccessibilitySettings::default(),
            performance: VisualPerformanceSettings::default(),
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            scheme_type: ColorSchemeType::Default,
            source_colors: HashMap::new(),
            ambient_color: ColorRGBA {
                r: 0.1,
                g: 0.1,
                b: 0.2,
                a: 1.0,
            },
            color_temperature: 6500.0, // Daylight white
            saturation: 0.8,
            brightness: 0.7,
        }
    }
}

impl Default for VisualAccessibilitySettings {
    fn default() -> Self {
        Self {
            high_contrast: false,
            color_blind_friendly: false,
            reduced_motion: false,
            indicator_scaling: 1.0,
            text_scaling: 1.0,
            audio_description_mode: false,
            screen_reader_compatible: false,
        }
    }
}

impl Default for VisualPerformanceSettings {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            quality_level: 0.8,
            lod_enabled: true,
            culling_distance: 50.0,
            max_particles: 1000,
            anti_aliasing: 4,
            adaptive_quality: true,
        }
    }
}
