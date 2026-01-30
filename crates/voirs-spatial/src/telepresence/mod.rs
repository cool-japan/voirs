//! # High-Fidelity Spatial Telepresence System
//!
//! This module provides advanced spatial audio telepresence capabilities,
//! enabling immersive remote communication experiences with realistic 3D positioning,
//! room simulation, and high-quality voice processing.

pub mod audio;
pub mod network;
pub mod processor;
pub mod quality;
pub mod room;
pub mod security;
pub mod session;
pub mod spatial;
pub mod types;

// Re-export core types
pub use types::*;

// Re-export audio types
pub use audio::{
    AudioDeviceConfig, AudioEnhancementSettings, AudioFormat, AudioMetadata,
    AudioQualityPreferences, AudioQualitySettings, AudioQualityStats, BandwidthConstraints,
    BandwidthExtensionSettings, CodecPreferences, CodecSettings, ComfortNoiseSettings,
    CompressionSettings, EchoCancellationSettings, EqBand, EqualizationSettings,
    FrequencyFilterSettings, FrequencyResponseAdjustment, HrtfPersonalizationSettings,
    NoiseSuppressionSettings, NotchFilter, ReceivedAudio, TelepresenceAudioSettings,
    UserMeasurements, VadSettings, VoiceProcessingSettings, VoiceSpatializationSettings,
};

// Re-export network types
pub use network::{
    CongestionControlSettings, ConnectionPreferences, FirewallSettings, IceSettings,
    JitterBufferSettings, NetworkConditions, NetworkSettings, NetworkThresholds, QosSettings,
    RedundancySettings, StunServerConfig, TrafficShapingSettings, TurnServerConfig,
};

// Re-export spatial types
pub use spatial::{
    AcousticEchoSettings, AmbientSharingSettings, AudioPresenceSettings, BackgroundNoiseSettings,
    BreathingRoomSettings, EnvironmentalAwarenessSettings, HeadTrackingSettings,
    NoiseProfilingSettings, PresenceIndicatorSettings, ProximityAdjustments, SpatialAudioInfo,
    SpatialAudioStats, SpatialQualitySettings, SpatialTelepresenceSettings,
    TrackingPredictionSettings, TrackingSmoothingSettings, UserPosition, VisualPresenceSettings,
};

// Re-export room types
pub use room::{
    AcousticMatchingSettings, AcousticProperties, AirAbsorptionSettings, CrossRoomSettings,
    DistanceModelingSettings, DopplerEffectsSettings, FurnitureAcoustics, FurnitureGeometry,
    FurnitureItem, MaterialProperties, ObjectMaterial, Opening, OpeningAcoustics, OpeningGeometry,
    RoomLayout, RoomMaterials, RoomSimulationSettings, SharedSpace, VirtualRoomParameters,
};

// Re-export quality types
pub use quality::{
    AdaptiveQualitySettings, PerformanceMonitoringSettings, PerformanceStats, QualityBounds,
    QualityInfo, QualitySettings,
};

// Re-export security types
pub use security::{
    AccessControlSettings, AnonymizationSettings, AuthenticationSettings,
    ConsentManagementSettings, DataCollectionSettings, EncryptionSettings, PrivacySettings,
    RateLimitingSettings, RecordingSettings, SecuritySettings,
};

// Re-export session types
pub use session::{
    SessionCapabilities, SessionConfig, SessionJoinResult, SessionRecordingSettings, SessionState,
    SessionStatistics, SessionUser, TelepresenceSession, UserCapabilities, UserConfig,
};

// Re-export processor types
pub use processor::{
    AudioProcessingPipeline, GlobalAudioSettings, NetworkManager, QualityManager, ResourceLimits,
    SessionLimits, SpatialProcessingPipeline, StatisticsCollector, TelepresenceConfig,
    TelepresenceProcessor,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telepresence_config_creation() {
        let config = TelepresenceConfig::default();
        assert_eq!(
            config
                .default_user_config
                .audio_settings
                .input_device
                .sample_rate,
            48000
        );
        assert!(config.security_settings.encryption.required);
    }

    #[test]
    fn test_user_config_creation() {
        let user_config = UserConfig::default();
        assert_eq!(user_config.user_id, "default_user");
        assert!(user_config.audio_settings.voice_processing.agc_enabled);
    }

    #[test]
    fn test_audio_device_config() {
        let device_config = AudioDeviceConfig::default();
        assert_eq!(device_config.sample_rate, 48000);
        assert_eq!(device_config.channels, 2);
        assert_eq!(device_config.buffer_size, 1024);
    }

    #[test]
    fn test_voice_processing_settings() {
        let voice_settings = VoiceProcessingSettings::default();
        assert!(voice_settings.agc_enabled);
        assert!(voice_settings.noise_suppression.enabled);
        assert!(voice_settings.echo_cancellation.enabled);
    }

    #[test]
    fn test_spatial_telepresence_settings() {
        let spatial_settings = SpatialTelepresenceSettings::default();
        assert!(spatial_settings.spatial_enabled);
        assert_eq!(
            spatial_settings.spatial_quality,
            SpatialQualityLevel::Full3D
        );
    }

    #[test]
    fn test_room_simulation_settings() {
        let room_settings = RoomSimulationSettings::default();
        assert!(room_settings.enabled);
        assert_eq!(room_settings.virtual_room.dimensions, (8.0, 3.0, 6.0));
    }

    #[test]
    fn test_telepresence_processor_creation() {
        let config = TelepresenceConfig::default();
        let _processor = TelepresenceProcessor::new(config);
        // Basic creation test - more functionality would be tested with actual implementation
    }

    #[test]
    fn test_session_join_result() {
        use crate::Position3D;

        let join_result = SessionJoinResult {
            success: true,
            session_id: "test_session".to_string(),
            user_session_id: "user_123".to_string(),
            assigned_position: Some(Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }),
            capabilities: SessionCapabilities {
                max_users: 32,
                supported_codecs: vec![AudioCodec::Opus],
                quality_levels: vec![QualityLevel::High],
                spatial_audio: true,
                recording_capable: false,
                screen_sharing: false,
            },
            other_users: vec![],
            error_message: None,
        };

        assert!(join_result.success);
        assert_eq!(join_result.session_id, "test_session");
    }

    #[test]
    fn test_codec_preferences() {
        let codec_prefs = CodecPreferences::default();
        assert_eq!(codec_prefs.preferred_codecs[0], AudioCodec::Opus);
        assert_eq!(
            codec_prefs.fallback_behavior,
            CodecFallbackBehavior::NextPreferred
        );
    }

    #[test]
    fn test_network_settings() {
        let network_settings = NetworkSettings::default();
        assert_eq!(
            network_settings.connection_preferences.preferred_type,
            ConnectionType::Auto
        );
        assert!(network_settings.qos_settings.jitter_buffer.adaptive);
    }

    #[test]
    fn test_quality_adaptation() {
        let adaptive_settings = AdaptiveQualitySettings::default();
        assert!(adaptive_settings.enabled);
        assert_eq!(
            adaptive_settings.algorithm,
            QualityAdaptationAlgorithm::Hybrid
        );
        assert_eq!(adaptive_settings.adaptation_speed, AdaptationSpeed::Medium);
    }
}
