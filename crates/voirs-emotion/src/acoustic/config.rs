//! Configuration types for acoustic model integration
//!
//! This module provides configuration structures for controlling
//! emotion-aware acoustic synthesis processing.

/// Configuration for acoustic integration
#[derive(Debug, Clone)]
pub struct AcousticIntegrationConfig {
    /// Enable direct acoustic model conditioning
    pub enable_acoustic_conditioning: bool,
    /// Enable speaker-specific emotion adaptation
    pub enable_speaker_adaptation: bool,
    /// Enable advanced emotion-to-prosody mapping
    pub enable_advanced_prosody: bool,
    /// Fallback to basic processing when acoustic models unavailable
    pub enable_fallback_processing: bool,
    /// Quality preset for acoustic processing
    pub quality_preset: AcousticQualityPreset,
}

/// Quality presets for acoustic processing
#[derive(Debug, Clone, PartialEq)]
pub enum AcousticQualityPreset {
    /// High quality with full processing
    High,
    /// Balanced quality and performance
    Balanced,
    /// Fast processing with basic quality
    Fast,
    /// Minimal processing for maximum speed
    Minimal,
}

impl Default for AcousticIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_acoustic_conditioning: true,
            enable_speaker_adaptation: true,
            enable_advanced_prosody: true,
            enable_fallback_processing: true,
            quality_preset: AcousticQualityPreset::Balanced,
        }
    }
}
