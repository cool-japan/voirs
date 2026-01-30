//! Acoustic emotion adapter implementation
//!
//! This module provides the main `AcousticEmotionAdapter` for integrating
//! emotion control with acoustic synthesis models.
//!
//! The implementation is organized into several submodules for maintainability:
//! - `core`: Core adapter structure and operations
//! - `mappings`: Emotion-to-acoustic parameter mappings
//! - `synthesis`: Synthesis and configuration methods
//! - `effects`: Audio effect processing
//! - `features`: Feature extraction from audio

#[cfg(feature = "acoustic-integration")]
pub mod core;
#[cfg(feature = "acoustic-integration")]
pub mod effects;
#[cfg(feature = "acoustic-integration")]
pub mod features;
#[cfg(feature = "acoustic-integration")]
pub mod mappings;
#[cfg(feature = "acoustic-integration")]
pub mod synthesis;

#[cfg(test)]
mod tests;

#[cfg(feature = "acoustic-integration")]
pub use core::AcousticEmotionAdapter;

// Stub implementation when acoustic integration is disabled
#[cfg(not(feature = "acoustic-integration"))]
mod stub {
    use crate::{types::EmotionParameters, types::EmotionVector, Error, Result};

    /// Stub acoustic emotion adapter (acoustic integration disabled)
    #[derive(Debug, Clone)]
    pub struct AcousticEmotionAdapter;

    impl AcousticEmotionAdapter {
        /// Create new stub adapter
        pub fn new() -> Self {
            Self
        }

        /// Stub synthesis method
        pub fn synthesize_with_emotion(
            &self,
            _text: &str,
            _emotion_params: &EmotionParameters,
        ) -> Result<Vec<f32>> {
            Err(Error::Config(
                "Acoustic integration not enabled".to_string(),
            ))
        }

        /// Stub feature extraction
        pub fn extract_emotion_features(
            &self,
            _audio: &[f32],
            _sample_rate: u32,
        ) -> Result<EmotionVector> {
            Err(Error::Config(
                "Acoustic integration not enabled".to_string(),
            ))
        }
    }

    impl Default for AcousticEmotionAdapter {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(not(feature = "acoustic-integration"))]
pub use stub::AcousticEmotionAdapter;
