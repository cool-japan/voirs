//! TelepresenceSession trait and related types

use super::audio::{AudioMetadata, ReceivedAudio, TelepresenceAudioSettings};
use super::network::{NetworkConditions, NetworkSettings};
use super::quality::{PerformanceStats, QualitySettings};
use super::security::PrivacySettings;
use super::spatial::{SpatialAudioStats, SpatialTelepresenceSettings};
use super::types::*;
use crate::{Position3D, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Telepresence session interface
pub trait TelepresenceSession: Send + Sync {
    /// Join the telepresence session
    fn join(&mut self, user_config: &UserConfig) -> Result<SessionJoinResult>;

    /// Leave the telepresence session
    fn leave(&mut self) -> Result<()>;

    /// Send audio data to the session
    fn send_audio(&mut self, audio_data: &[f32], metadata: &AudioMetadata) -> Result<()>;

    /// Receive audio data from the session
    fn receive_audio(&mut self) -> Result<Vec<ReceivedAudio>>;

    /// Update user position
    fn update_position(&mut self, position: Position3D, orientation: Orientation) -> Result<()>;

    /// Get session state
    fn session_state(&self) -> SessionState;

    /// Get session statistics
    fn statistics(&self) -> SessionStatistics;
}

/// User configuration for telepresence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    /// User identifier
    pub user_id: String,

    /// Display name
    pub display_name: String,

    /// Audio settings
    pub audio_settings: TelepresenceAudioSettings,

    /// Spatial settings
    pub spatial_settings: SpatialTelepresenceSettings,

    /// Network preferences
    pub network_settings: NetworkSettings,

    /// Quality preferences
    pub quality_settings: QualitySettings,

    /// Privacy settings
    pub privacy_settings: PrivacySettings,
}

/// Session join result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionJoinResult {
    /// Success indicator
    pub success: bool,

    /// Session identifier
    pub session_id: String,

    /// User identifier in session
    pub user_session_id: String,

    /// Assigned position
    pub assigned_position: Option<Position3D>,

    /// Session capabilities
    pub capabilities: SessionCapabilities,

    /// Other users in session
    pub other_users: Vec<SessionUser>,

    /// Error message if failed
    pub error_message: Option<String>,
}

/// Session capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCapabilities {
    /// Maximum users
    pub max_users: usize,

    /// Supported codecs
    pub supported_codecs: Vec<AudioCodec>,

    /// Supported quality levels
    pub quality_levels: Vec<QualityLevel>,

    /// Spatial audio support
    pub spatial_audio: bool,

    /// Recording capability
    pub recording_capable: bool,

    /// Screen sharing support
    pub screen_sharing: bool,
}

/// Session user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionUser {
    /// User identifier
    pub user_id: String,

    /// Display name
    pub display_name: String,

    /// Current position
    pub position: Position3D,

    /// Current orientation
    pub orientation: Orientation,

    /// User state
    pub state: UserState,

    /// Capabilities
    pub capabilities: UserCapabilities,
}

/// User capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCapabilities {
    /// Can speak
    pub can_speak: bool,

    /// Can move
    pub can_move: bool,

    /// Has spatial audio
    pub spatial_audio: bool,

    /// Can record
    pub can_record: bool,

    /// Quality level
    pub quality_level: QualityLevel,
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Session identifier
    pub session_id: String,

    /// Session status
    pub status: SessionStatus,

    /// Connected users
    pub connected_users: Vec<String>,

    /// Session start time
    pub start_time: SystemTime,

    /// Session duration
    pub duration: Duration,

    /// Current audio quality
    pub current_quality: QualityLevel,

    /// Network conditions
    pub network_conditions: NetworkConditions,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total data sent (bytes)
    pub data_sent: u64,

    /// Total data received (bytes)
    pub data_received: u64,

    /// Packets sent
    pub packets_sent: u64,

    /// Packets received
    pub packets_received: u64,

    /// Packets lost
    pub packets_lost: u64,

    /// Average latency (ms)
    pub avg_latency: f32,

    /// Peak latency (ms)
    pub peak_latency: f32,

    /// Audio quality statistics
    pub audio_quality_stats: super::audio::AudioQualityStats,

    /// Spatial audio statistics
    pub spatial_stats: SpatialAudioStats,

    /// Performance statistics
    pub performance_stats: PerformanceStats,
}

/// Configuration for telepresence sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session name
    pub name: String,

    /// Session description
    pub description: Option<String>,

    /// Maximum users
    pub max_users: usize,

    /// Session privacy
    pub privacy: SessionPrivacy,

    /// Recording settings
    pub recording: SessionRecordingSettings,

    /// Virtual room settings
    pub virtual_room: Option<super::room::VirtualRoomParameters>,
}

/// Session recording settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecordingSettings {
    /// Allow recording
    pub allow_recording: bool,

    /// Auto-record session
    pub auto_record: bool,

    /// Recording quality
    pub quality: QualityLevel,

    /// Include video
    pub include_video: bool,
}

// Default implementations

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            display_name: "User".to_string(),
            audio_settings: TelepresenceAudioSettings::default(),
            spatial_settings: SpatialTelepresenceSettings::default(),
            network_settings: NetworkSettings::default(),
            quality_settings: QualitySettings::default(),
            privacy_settings: PrivacySettings::default(),
        }
    }
}
