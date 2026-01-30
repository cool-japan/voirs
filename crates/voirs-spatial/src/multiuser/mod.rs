//! # Multi-user Spatial Audio System
//!
//! This module provides shared spatial audio environments for multiple users,
//! enabling collaborative and social spatial audio experiences.

// Module declarations
mod impls;
mod types;

// Re-export all public types
pub use types::*;

// Re-export builder and other items from impls
pub use impls::MultiUserConfigBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiuser_config_creation() {
        let config = MultiUserConfig::default();
        assert_eq!(config.max_users_per_room, 50);
        assert_eq!(config.max_sources_per_user, 5);
        assert!(config.privacy_settings.encryption_enabled);
    }

    #[test]
    fn test_multiuser_config_builder() {
        let config = MultiUserConfigBuilder::new()
            .max_users(25)
            .audio_quality(0.9)
            .max_latency_ms(50.0)
            .build();

        assert_eq!(config.max_users_per_room, 25);
        assert_eq!(config.audio_quality, 0.9);
    }

    #[test]
    fn test_position_interpolator_default() {
        let _interpolator = PositionInterpolator::default();
        // Just verify it compiles and can be created
    }

    #[test]
    fn test_synchronization_manager_default() {
        let _manager = SynchronizationManager::default();
        // Just verify it compiles and can be created
    }

    #[test]
    fn test_voice_activity_detector_default() {
        let _vad = VoiceActivityDetector::default();
        // Just verify it compiles and can be created
    }
}
