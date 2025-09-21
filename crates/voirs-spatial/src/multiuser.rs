//! # Multi-user Spatial Audio System
//!
//! This module provides shared spatial audio environments for multiple users,
//! enabling collaborative and social spatial audio experiences.

use crate::types::{Position3D, SpatialResult};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Unique identifier for users in the multi-user environment
pub type UserId = String;
/// Unique identifier for audio sources
pub type SourceId = String;
/// Unique identifier for rooms/environments
pub type RoomId = String;

/// Configuration for multi-user spatial audio environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiUserConfig {
    /// Maximum number of users per room
    pub max_users_per_room: usize,
    /// Maximum number of audio sources per user
    pub max_sources_per_user: usize,
    /// Network synchronization interval in milliseconds
    pub sync_interval_ms: u64,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: f64,
    /// Voice activity detection threshold
    pub voice_activity_threshold: f32,
    /// Spatial audio processing quality (0.0-1.0)
    pub audio_quality: f32,
    /// Enable position interpolation for smooth movement
    pub position_interpolation: bool,
    /// Maximum distance for audio interaction
    pub max_audio_distance: f32,
    /// Distance-based volume attenuation curve
    pub attenuation_curve: MultiUserAttenuationCurve,
    /// Privacy and security settings
    pub privacy_settings: PrivacySettings,
    /// Bandwidth optimization settings
    pub bandwidth_settings: BandwidthSettings,
}

/// Distance-based audio attenuation curves
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MultiUserAttenuationCurve {
    /// Linear attenuation (simple distance falloff)
    Linear,
    /// Inverse distance attenuation (1/distance)
    InverseDistance,
    /// Inverse square law (1/distance^2)
    InverseSquare,
    /// Logarithmic attenuation for natural feeling
    Logarithmic,
    /// Custom curve with configurable parameters
    Custom {
        /// Rate of volume decrease with distance
        rolloff: f32,
        /// Distance at which attenuation begins
        reference_distance: f32,
    },
}

/// Privacy and security settings for multi-user environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    /// Enable end-to-end encryption for audio streams
    pub encryption_enabled: bool,
    /// Allow recording of multi-user sessions
    pub recording_allowed: bool,
    /// Enable mute controls for users
    pub mute_controls_enabled: bool,
    /// Enable spatial zones with restricted access
    pub spatial_zones_enabled: bool,
    /// User permission levels
    pub permission_system: PermissionSystem,
    /// Anonymization settings
    pub anonymization: AnonymizationSettings,
}

/// Permission system for user roles and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSystem {
    /// Enable role-based access control
    pub rbac_enabled: bool,
    /// Default role for new users
    pub default_role: UserRole,
    /// Role-specific permissions
    pub role_permissions: HashMap<UserRole, Vec<Permission>>,
}

/// User roles in the multi-user environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UserRole {
    /// Regular participant with basic permissions
    Participant,
    /// Moderator with additional controls
    Moderator,
    /// Administrator with full permissions
    Administrator,
    /// Observer with read-only access
    Observer,
    /// Presenter with enhanced audio capabilities
    Presenter,
    /// Guest with limited permissions
    Guest,
}

/// Specific permissions that can be granted to users
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Can speak and be heard by others
    Speak,
    /// Can move freely in the spatial environment
    Move,
    /// Can mute/unmute other users
    MuteOthers,
    /// Can kick users from the room
    KickUsers,
    /// Can modify room settings
    ModifyRoom,
    /// Can create new audio sources
    CreateSources,
    /// Can record the session
    Record,
    /// Can access private spatial zones
    AccessPrivateZones,
    /// Can broadcast to all users
    Broadcast,
    /// Can moderate content
    Moderate,
}

/// Anonymization settings for user privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationSettings {
    /// Replace user identifiers with anonymous IDs
    pub anonymous_ids: bool,
    /// Obfuscate precise positioning data
    pub position_obfuscation: bool,
    /// Remove temporal patterns from audio
    pub temporal_obfuscation: bool,
    /// Use voice modulation for identity protection
    pub voice_modulation: bool,
}

/// Bandwidth optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthSettings {
    /// Adaptive bitrate based on network conditions
    pub adaptive_bitrate: bool,
    /// Maximum bandwidth per user in kbps
    pub max_bandwidth_kbps: u32,
    /// Compression level (0-10, higher = more compression)
    pub compression_level: u8,
    /// Enable proximity-based quality adjustment
    pub proximity_quality_scaling: bool,
    /// Low bandwidth mode settings
    pub low_bandwidth_mode: LowBandwidthMode,
}

/// Low bandwidth mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowBandwidthMode {
    /// Enable low bandwidth mode
    pub enabled: bool,
    /// Reduced sample rate for audio
    pub sample_rate: u32,
    /// Reduced bit depth
    pub bit_depth: u16,
    /// Maximum number of simultaneous audio streams
    pub max_streams: usize,
    /// Disable spatial effects to save bandwidth
    pub disable_spatial_effects: bool,
}

/// User in the multi-user spatial audio environment
#[derive(Debug, Clone)]
pub struct MultiUserUser {
    /// Unique user identifier
    pub id: UserId,
    /// Display name for the user
    pub display_name: String,
    /// Current position in 3D space
    pub position: Position3D,
    /// Orientation as quaternion (w, x, y, z)
    pub orientation: [f32; 4],
    /// Velocity for movement prediction
    pub velocity: Position3D,
    /// User role and permissions
    pub role: UserRole,
    /// Whether user is currently speaking
    pub is_speaking: bool,
    /// Audio sources owned by this user
    pub audio_sources: HashMap<SourceId, MultiUserAudioSource>,
    /// User's connection status
    pub connection_status: ConnectionStatus,
    /// Last update timestamp
    pub last_update: Instant,
    /// User-specific audio settings
    pub audio_settings: UserAudioSettings,
    /// Network statistics for this user
    pub network_stats: NetworkStats,
    /// Spatial zones the user has access to
    pub accessible_zones: Vec<SpatialZone>,
    /// List of friends (user IDs)
    pub friends: HashSet<UserId>,
}

/// Connection status for users
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Connected and active
    Connected,
    /// Connected but experiencing latency issues
    Unstable,
    /// Temporarily disconnected, attempting reconnection
    Reconnecting,
    /// Disconnected
    Disconnected,
    /// Connection timed out
    TimedOut,
}

/// Audio source in multi-user environment
#[derive(Debug, Clone)]
pub struct MultiUserAudioSource {
    /// Unique source identifier
    pub id: SourceId,
    /// Owner of this audio source
    pub owner_id: UserId,
    /// Source position in 3D space
    pub position: Position3D,
    /// Source type and characteristics
    pub source_type: AudioSourceType,
    /// Volume level (0.0-1.0)
    pub volume: f32,
    /// Whether the source is currently active
    pub is_active: bool,
    /// Spatial audio properties
    pub spatial_properties: SpatialProperties,
    /// Access control for this source
    pub access_control: SourceAccessControl,
    /// Quality settings for this source
    pub quality_settings: SourceQualitySettings,
}

/// Types of audio sources in multi-user environment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSourceType {
    /// User's voice audio
    Voice,
    /// Music or sound effects
    Media,
    /// Environmental/ambient audio
    Ambient,
    /// Screen sharing audio
    ScreenShare,
    /// Notification sounds
    Notification,
    /// Interactive audio objects
    Interactive,
}

/// Spatial properties for audio sources
#[derive(Debug, Clone)]
pub struct SpatialProperties {
    /// Directional audio pattern
    pub directivity: DirectionalPattern,
    /// Distance at which audio starts attenuating
    pub reference_distance: f32,
    /// Rate of attenuation with distance
    pub rolloff_factor: f32,
    /// Maximum distance for audio audibility
    pub max_distance: f32,
    /// Doppler effect strength
    pub doppler_factor: f32,
    /// Room acoustics interaction
    pub room_interaction: bool,
}

/// Audio directional patterns
#[derive(Debug, Clone, Copy)]
pub enum DirectionalPattern {
    /// Omnidirectional (equal in all directions)
    Omnidirectional,
    /// Cardioid (heart-shaped, directional)
    Cardioid,
    /// Bidirectional (figure-8 pattern)
    Bidirectional,
    /// Cone-shaped directional pattern
    Cone {
        /// Inner cone angle in radians
        inner_angle: f32,
        /// Outer cone angle in radians
        outer_angle: f32,
        /// Volume multiplier outside the outer cone
        outer_gain: f32,
    },
}

/// Access control for audio sources
#[derive(Debug, Clone)]
pub struct SourceAccessControl {
    /// Who can hear this source
    pub visibility: SourceVisibility,
    /// Specific users who can access (if visibility is Whitelist)
    pub allowed_users: Vec<UserId>,
    /// Users who are blocked from accessing
    pub blocked_users: Vec<UserId>,
    /// Spatial zones where this source is audible
    pub spatial_zones: Vec<SpatialZone>,
}

/// Visibility settings for audio sources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceVisibility {
    /// Everyone can hear
    Public,
    /// Only friends/contacts can hear
    Friends,
    /// Only specific users can hear (whitelist)
    Whitelist,
    /// Private to owner only
    Private,
    /// Moderators and above can hear
    Moderators,
}

/// Quality settings for individual audio sources
#[derive(Debug, Clone)]
pub struct SourceQualitySettings {
    /// Bitrate for this source
    pub bitrate_kbps: u32,
    /// Spatial audio quality level
    pub spatial_quality: f32,
    /// Noise reduction level
    pub noise_reduction: f32,
    /// Echo cancellation strength
    pub echo_cancellation: f32,
    /// Dynamic range compression
    pub compression_ratio: f32,
}

/// User-specific audio settings
#[derive(Debug, Clone)]
pub struct UserAudioSettings {
    /// Overall volume multiplier
    pub master_volume: f32,
    /// Voice chat volume
    pub voice_volume: f32,
    /// Media/effects volume
    pub media_volume: f32,
    /// Spatial audio intensity
    pub spatial_intensity: f32,
    /// Personal HRTF profile identifier
    pub hrtf_profile: Option<String>,
    /// Microphone settings
    pub microphone_settings: MicrophoneSettings,
    /// Hearing accessibility options
    pub accessibility: AccessibilitySettings,
}

/// Microphone settings for users
#[derive(Debug, Clone)]
pub struct MicrophoneSettings {
    /// Input gain level
    pub gain: f32,
    /// Noise gate threshold
    pub noise_gate_threshold: f32,
    /// Automatic gain control
    pub auto_gain_control: bool,
    /// Push-to-talk mode
    pub push_to_talk: bool,
    /// Voice activation detection sensitivity
    pub vad_sensitivity: f32,
}

/// Accessibility settings for hearing-impaired users
#[derive(Debug, Clone)]
pub struct AccessibilitySettings {
    /// Enable visual audio indicators
    pub visual_indicators: bool,
    /// Enhance high-frequency audio
    pub high_frequency_boost: f32,
    /// Reduce background noise
    pub background_noise_reduction: f32,
    /// Convert speech to text
    pub speech_to_text: bool,
    /// Haptic feedback for audio events
    pub haptic_feedback: bool,
}

/// Network performance statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Round-trip time in milliseconds
    pub rtt_ms: f64,
    /// Packet loss percentage
    pub packet_loss_percent: f32,
    /// Jitter in milliseconds
    pub jitter_ms: f64,
    /// Bandwidth utilization in kbps
    pub bandwidth_usage_kbps: u32,
    /// Number of reconnection attempts
    pub reconnect_count: u32,
    /// Audio dropouts in the last minute
    pub audio_dropouts: u32,
}

/// Spatial zone for access control and audio routing
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialZone {
    /// Zone identifier
    pub id: String,
    /// Zone name
    pub name: String,
    /// Zone type
    pub zone_type: ZoneType,
    /// Geometric bounds of the zone
    pub bounds: ZoneBounds,
    /// Required permissions to enter
    pub required_permissions: Vec<Permission>,
    /// Audio properties within this zone
    pub audio_properties: ZoneAudioProperties,
}

/// Types of spatial zones
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZoneType {
    /// General meeting/conversation area
    Meeting,
    /// Private conversation zone
    Private,
    /// Presentation/stage area
    Presentation,
    /// Quiet zone with reduced audio
    Quiet,
    /// High-priority broadcast zone
    Broadcast,
    /// Social gathering area
    Social,
}

/// Geometric bounds for spatial zones
#[derive(Debug, Clone, PartialEq)]
pub enum ZoneBounds {
    /// Spherical zone with center and radius
    Sphere {
        /// Center point of the sphere
        center: Position3D,
        /// Radius of the sphere
        radius: f32,
    },
    /// Cubic zone with min and max coordinates
    Box {
        /// Minimum corner coordinates
        min: Position3D,
        /// Maximum corner coordinates
        max: Position3D,
    },
    /// Cylindrical zone with center, radius, and height
    Cylinder {
        /// Center point of the cylinder base
        center: Position3D,
        /// Radius of the cylinder
        radius: f32,
        /// Height of the cylinder
        height: f32,
    },
    /// Polygonal zone defined by vertices
    Polygon {
        /// List of vertices defining the polygon boundary
        vertices: Vec<Position3D>,
    },
}

/// Audio properties specific to spatial zones
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneAudioProperties {
    /// Volume multiplier for all audio in this zone
    pub volume_multiplier: f32,
    /// Acoustic properties (reverb, echo, etc.)
    pub acoustic_settings: AcousticSettings,
    /// Whether audio from this zone can be heard outside
    pub audio_isolation: bool,
    /// Maximum number of simultaneous speakers
    pub max_speakers: Option<usize>,
    /// Priority level for audio processing
    pub priority: u8,
}

/// Acoustic settings for zones
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticSettings {
    /// Reverb amount (0.0-1.0)
    pub reverb: f32,
    /// Echo delay in milliseconds
    pub echo_delay_ms: f32,
    /// Echo feedback amount
    pub echo_feedback: f32,
    /// Room size simulation
    pub room_size: f32,
    /// Damping factor for reflections
    pub damping: f32,
}

/// Multi-user spatial audio environment manager
pub struct MultiUserEnvironment {
    /// Configuration for this environment
    config: MultiUserConfig,
    /// All users in the environment
    users: Arc<RwLock<HashMap<UserId, MultiUserUser>>>,
    /// All audio sources in the environment
    sources: Arc<RwLock<HashMap<SourceId, MultiUserAudioSource>>>,
    /// Spatial zones in the environment
    zones: Arc<RwLock<HashMap<String, SpatialZone>>>,
    /// Network synchronization manager
    sync_manager: SynchronizationManager,
    /// Audio processing pipeline
    audio_processor: MultiUserAudioProcessor,
    /// Performance metrics
    metrics: Arc<RwLock<MultiUserMetrics>>,
    /// Event history for debugging and analysis
    event_history: Arc<RwLock<VecDeque<MultiUserEvent>>>,
}

/// Synchronization manager for network coordination
pub struct SynchronizationManager {
    /// Synchronization clock
    clock: Arc<RwLock<SynchronizedClock>>,
    /// User position interpolation
    position_interpolator: PositionInterpolator,
    /// Network latency compensation
    latency_compensator: LatencyCompensator,
}

/// Synchronized clock for multi-user coordination
#[derive(Debug)]
pub struct SynchronizedClock {
    /// Local time reference
    local_time: Instant,
    /// Network-synchronized time offset
    time_offset_ms: i64,
    /// Clock synchronization accuracy
    sync_accuracy_ms: f64,
    /// Last synchronization timestamp
    last_sync: Instant,
}

/// Position interpolation for smooth user movement
pub struct PositionInterpolator {
    /// User position histories
    position_histories: HashMap<UserId, VecDeque<PositionSnapshot>>,
    /// Interpolation method
    interpolation_method: InterpolationMethod,
    /// Prediction horizon in milliseconds
    prediction_horizon_ms: f64,
}

/// Position snapshot with timestamp
#[derive(Debug, Clone)]
pub struct PositionSnapshot {
    /// Position in 3D space
    pub position: Position3D,
    /// Orientation quaternion
    pub orientation: [f32; 4],
    /// Velocity vector
    pub velocity: Position3D,
    /// Timestamp of this snapshot
    pub timestamp: Instant,
    /// Network latency when this was received
    pub latency_ms: f64,
}

/// Interpolation methods for position smoothing
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// Kalman filter-based prediction
    Kalman,
    /// Physics-based prediction
    Physics,
}

/// Network latency compensation system
pub struct LatencyCompensator {
    /// Per-user latency estimates
    user_latencies: HashMap<UserId, f64>,
    /// Compensation strategies
    compensation_method: CompensationMethod,
    /// Maximum compensation amount
    max_compensation_ms: f64,
}

/// Latency compensation methods
#[derive(Debug, Clone, Copy)]
pub enum CompensationMethod {
    /// Simple time-shift compensation
    TimeShift,
    /// Predictive compensation with motion models
    Predictive,
    /// Adaptive compensation based on network conditions
    Adaptive,
    /// No compensation (for low-latency networks)
    None,
}

/// Multi-user audio processing pipeline
pub struct MultiUserAudioProcessor {
    /// Audio mixing engine
    mixer: SpatialAudioMixer,
    /// Voice activity detection
    vad: VoiceActivityDetector,
    /// Audio effects processor
    effects: AudioEffectsProcessor,
    /// Compression and encoding
    codec: AudioCodec,
}

/// Spatial audio mixing for multiple users and sources
pub struct SpatialAudioMixer {
    /// Current listener position (for each user)
    listener_positions: HashMap<UserId, Position3D>,
    /// Audio source management
    source_manager: Arc<RwLock<HashMap<SourceId, MixerAudioSource>>>,
    /// Mixing configuration
    mixer_config: MixerConfig,
}

/// Audio source representation for the mixer
#[derive(Debug, Clone)]
pub struct MixerAudioSource {
    /// Source identifier
    pub id: SourceId,
    /// Current audio buffer
    pub buffer: Vec<f32>,
    /// Spatial position
    pub position: Position3D,
    /// Audio properties
    pub properties: SpatialProperties,
    /// Processing state
    pub processing_state: SourceProcessingState,
}

/// Processing state for audio sources
#[derive(Debug, Clone)]
pub struct SourceProcessingState {
    /// Current playback position
    pub playback_position: usize,
    /// Volume envelope state
    pub envelope_state: f32,
    /// Spatial processing parameters
    pub spatial_params: SpatialProcessingParams,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Spatial processing parameters
#[derive(Debug, Clone)]
pub struct SpatialProcessingParams {
    /// Distance-based attenuation
    pub distance_attenuation: f32,
    /// Doppler shift factor
    pub doppler_shift: f32,
    /// Spatial position in listener coordinates
    pub listener_relative_position: Position3D,
    /// Occlusion factors
    pub occlusion: f32,
}

/// Mixer configuration
#[derive(Debug, Clone)]
pub struct MixerConfig {
    /// Maximum number of simultaneous sources
    pub max_sources: usize,
    /// Audio buffer size
    pub buffer_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Spatial processing quality
    pub spatial_quality: f32,
    /// CPU optimization level
    pub optimization_level: OptimizationLevel,
}

/// CPU optimization levels for audio processing
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    /// Highest quality, highest CPU usage
    Quality,
    /// Balanced quality and performance
    Balanced,
    /// Performance optimized, reduced quality
    Performance,
    /// Lowest CPU usage, basic quality
    MinimalCpu,
}

/// Voice activity detection system
pub struct VoiceActivityDetector {
    /// Detection algorithm
    algorithm: VadAlgorithm,
    /// Per-user VAD state
    user_states: HashMap<UserId, VadState>,
    /// Detection thresholds
    thresholds: VadThresholds,
}

/// Voice activity detection algorithms
#[derive(Debug, Clone, Copy)]
pub enum VadAlgorithm {
    /// Energy-based detection
    Energy,
    /// Spectral analysis based
    Spectral,
    /// Machine learning based
    MachineLearning,
    /// Combination of multiple methods
    Hybrid,
}

/// VAD state for individual users
#[derive(Debug, Clone)]
pub struct VadState {
    /// Current speaking state
    pub is_speaking: bool,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
    /// Recent audio energy levels
    pub energy_history: VecDeque<f32>,
    /// Speaking duration
    pub speaking_duration: Duration,
    /// Silence duration
    pub silence_duration: Duration,
}

/// VAD detection thresholds
#[derive(Debug, Clone)]
pub struct VadThresholds {
    /// Energy threshold for voice detection
    pub energy_threshold: f32,
    /// Minimum speaking duration
    pub min_speaking_duration_ms: u64,
    /// Minimum silence duration
    pub min_silence_duration_ms: u64,
    /// Confidence threshold
    pub confidence_threshold: f32,
}

/// Audio effects processing for multi-user environments
pub struct AudioEffectsProcessor {
    /// Available audio effects
    effects: HashMap<String, Box<dyn AudioEffect>>,
    /// Per-user effect chains
    user_effects: HashMap<UserId, Vec<String>>,
    /// Zone-based effects
    zone_effects: HashMap<String, Vec<String>>,
}

/// Trait for audio effects
pub trait AudioEffect {
    /// Process audio samples
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>;

    /// Get effect parameters
    fn parameters(&self) -> HashMap<String, f32>;

    /// Set effect parameters
    fn set_parameters(&mut self, params: HashMap<String, f32>) -> Result<()>;

    /// Reset effect state
    fn reset(&mut self);
}

/// Audio codec for compression and encoding
pub struct AudioCodec {
    /// Encoding format
    format: AudioFormat,
    /// Compression settings
    compression: CompressionSettings,
    /// Per-user codec state
    codec_states: HashMap<UserId, CodecState>,
}

/// Supported audio formats
#[derive(Debug, Clone, Copy)]
pub enum AudioFormat {
    /// Opus codec (recommended for real-time)
    Opus,
    /// AAC codec
    Aac,
    /// MP3 codec
    Mp3,
    /// Raw PCM (uncompressed)
    Pcm,
}

/// Compression settings for audio codec
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Bitrate in kbps
    pub bitrate_kbps: u32,
    /// Complexity level (0-10)
    pub complexity: u8,
    /// Variable bitrate mode
    pub variable_bitrate: bool,
    /// Low latency mode
    pub low_latency: bool,
}

/// Per-user codec state
#[derive(Debug)]
pub struct CodecState {
    /// Encoder state
    pub encoder_state: Vec<u8>,
    /// Decoder state
    pub decoder_state: Vec<u8>,
    /// Frame buffer
    pub frame_buffer: VecDeque<Vec<u8>>,
    /// Timing information
    pub timing: CodecTiming,
}

/// Codec timing information
#[derive(Debug, Clone)]
pub struct CodecTiming {
    /// Encoding latency
    pub encode_latency_ms: f64,
    /// Decoding latency
    pub decode_latency_ms: f64,
    /// Buffer latency
    pub buffer_latency_ms: f64,
    /// Total latency
    pub total_latency_ms: f64,
}

/// Performance metrics for multi-user environment
#[derive(Debug, Clone, Default)]
pub struct MultiUserMetrics {
    /// Number of active users
    pub active_users: usize,
    /// Number of active audio sources
    pub active_sources: usize,
    /// Average network latency
    pub avg_latency_ms: f64,
    /// Audio processing CPU usage
    pub audio_cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// Network bandwidth usage
    pub bandwidth_usage_kbps: u32,
    /// Audio quality metrics
    pub audio_quality: AudioQualityMetrics,
    /// Synchronization accuracy
    pub sync_accuracy_ms: f64,
    /// Number of reconnections in last hour
    pub reconnections_per_hour: u32,
}

/// Audio quality metrics
#[derive(Debug, Clone, Default)]
pub struct AudioQualityMetrics {
    /// Signal-to-noise ratio
    pub snr_db: f32,
    /// Total harmonic distortion
    pub thd_percent: f32,
    /// Audio dropouts per minute
    pub dropouts_per_minute: f32,
    /// Perceived audio quality (MOS scale 1-5)
    pub perceived_quality: f32,
}

/// Events in the multi-user environment
#[derive(Debug, Clone)]
pub enum MultiUserEvent {
    /// User joined the environment
    UserJoined {
        /// ID of the user who joined
        user_id: UserId,
        /// When the user joined
        timestamp: SystemTime,
        /// Initial position of the user
        position: Position3D,
    },
    /// User left the environment
    UserLeft {
        /// ID of the user who left
        user_id: UserId,
        /// When the user left
        timestamp: SystemTime,
        /// Reason for leaving
        reason: DisconnectReason,
    },
    /// User moved in space
    UserMoved {
        /// ID of the user who moved
        user_id: UserId,
        /// When the movement occurred
        timestamp: SystemTime,
        /// Previous position
        old_position: Position3D,
        /// New position
        new_position: Position3D,
    },
    /// User started speaking
    UserStartedSpeaking {
        /// ID of the user who started speaking
        user_id: UserId,
        /// When speaking started
        timestamp: SystemTime,
        /// Voice activity detection confidence level
        confidence: f32,
    },
    /// User stopped speaking
    UserStoppedSpeaking {
        /// ID of the user who stopped speaking
        user_id: UserId,
        /// When speaking stopped
        timestamp: SystemTime,
        /// Duration of the speaking session
        duration: Duration,
    },
    /// Audio source created
    SourceCreated {
        /// ID of the created source
        source_id: SourceId,
        /// ID of the user who created the source
        user_id: UserId,
        /// When the source was created
        timestamp: SystemTime,
        /// Type of the audio source
        source_type: AudioSourceType,
    },
    /// Audio source removed
    SourceRemoved {
        /// ID of the removed source
        source_id: SourceId,
        /// When the source was removed
        timestamp: SystemTime,
        /// Reason for removal
        reason: String,
    },
    /// Network event (latency, packet loss, etc.)
    NetworkEvent {
        /// ID of the affected user
        user_id: UserId,
        /// When the event occurred
        timestamp: SystemTime,
        /// Type of network event
        event_type: NetworkEventType,
        /// Measured value for the event
        value: f64,
    },
}

/// Reasons for user disconnection
#[derive(Debug, Clone, Copy)]
pub enum DisconnectReason {
    /// User voluntarily left
    UserLeft,
    /// Network connection lost
    NetworkTimeout,
    /// Kicked by moderator
    Kicked,
    /// Technical error
    Error,
    /// Server shutdown
    ServerShutdown,
}

/// Types of network events
#[derive(Debug, Clone, Copy)]
pub enum NetworkEventType {
    /// Latency measurement
    Latency,
    /// Packet loss percentage
    PacketLoss,
    /// Jitter measurement
    Jitter,
    /// Bandwidth measurement
    Bandwidth,
    /// Connection established
    ConnectionEstablished,
    /// Connection lost
    ConnectionLost,
}

impl Default for MultiUserConfig {
    fn default() -> Self {
        Self {
            max_users_per_room: 50,
            max_sources_per_user: 5,
            sync_interval_ms: 50,
            max_latency_ms: 100.0,
            voice_activity_threshold: 0.3,
            audio_quality: 0.8,
            position_interpolation: true,
            max_audio_distance: 100.0,
            attenuation_curve: MultiUserAttenuationCurve::InverseDistance,
            privacy_settings: PrivacySettings::default(),
            bandwidth_settings: BandwidthSettings::default(),
        }
    }
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            encryption_enabled: true,
            recording_allowed: false,
            mute_controls_enabled: true,
            spatial_zones_enabled: true,
            permission_system: PermissionSystem::default(),
            anonymization: AnonymizationSettings::default(),
        }
    }
}

impl Default for PermissionSystem {
    fn default() -> Self {
        let mut role_permissions = HashMap::new();

        role_permissions.insert(UserRole::Guest, vec![Permission::Speak, Permission::Move]);
        role_permissions.insert(
            UserRole::Participant,
            vec![
                Permission::Speak,
                Permission::Move,
                Permission::CreateSources,
            ],
        );
        role_permissions.insert(
            UserRole::Presenter,
            vec![
                Permission::Speak,
                Permission::Move,
                Permission::CreateSources,
                Permission::Broadcast,
            ],
        );
        role_permissions.insert(
            UserRole::Moderator,
            vec![
                Permission::Speak,
                Permission::Move,
                Permission::CreateSources,
                Permission::MuteOthers,
                Permission::ModifyRoom,
                Permission::Moderate,
            ],
        );
        role_permissions.insert(
            UserRole::Administrator,
            vec![
                Permission::Speak,
                Permission::Move,
                Permission::CreateSources,
                Permission::MuteOthers,
                Permission::KickUsers,
                Permission::ModifyRoom,
                Permission::Record,
                Permission::AccessPrivateZones,
                Permission::Broadcast,
                Permission::Moderate,
            ],
        );
        role_permissions.insert(UserRole::Observer, vec![Permission::Move]);

        Self {
            rbac_enabled: true,
            default_role: UserRole::Participant,
            role_permissions,
        }
    }
}

impl Default for AnonymizationSettings {
    fn default() -> Self {
        Self {
            anonymous_ids: false,
            position_obfuscation: false,
            temporal_obfuscation: false,
            voice_modulation: false,
        }
    }
}

impl Default for BandwidthSettings {
    fn default() -> Self {
        Self {
            adaptive_bitrate: true,
            max_bandwidth_kbps: 128,
            compression_level: 5,
            proximity_quality_scaling: true,
            low_bandwidth_mode: LowBandwidthMode {
                enabled: false,
                sample_rate: 16000,
                bit_depth: 16,
                max_streams: 5,
                disable_spatial_effects: true,
            },
        }
    }
}

impl MultiUserEnvironment {
    /// Create a new multi-user environment
    pub fn new(config: MultiUserConfig) -> Result<Self> {
        let sync_manager = SynchronizationManager::new();
        let audio_processor = MultiUserAudioProcessor::new(&config)?;

        Ok(Self {
            config,
            users: Arc::new(RwLock::new(HashMap::new())),
            sources: Arc::new(RwLock::new(HashMap::new())),
            zones: Arc::new(RwLock::new(HashMap::new())),
            sync_manager,
            audio_processor,
            metrics: Arc::new(RwLock::new(MultiUserMetrics::default())),
            event_history: Arc::new(RwLock::new(VecDeque::new())),
        })
    }

    /// Add a user to the environment
    pub fn add_user(&self, user: MultiUserUser) -> Result<()> {
        let user_id = user.id.clone();
        let position = user.position;

        // Check user limit
        {
            let users = self.users.read().unwrap();
            if users.len() >= self.config.max_users_per_room {
                return Err(Error::LegacyProcessing(
                    "Maximum users per room exceeded".to_string(),
                ));
            }
        }

        // Add user
        {
            let mut users = self.users.write().unwrap();
            users.insert(user_id.clone(), user);
        }

        // Record event
        self.record_event(MultiUserEvent::UserJoined {
            user_id,
            timestamp: SystemTime::now(),
            position,
        });

        Ok(())
    }

    /// Remove a user from the environment
    pub fn remove_user(&self, user_id: &UserId, reason: DisconnectReason) -> Result<()> {
        {
            let mut users = self.users.write().unwrap();
            users.remove(user_id);
        }

        // Record event
        self.record_event(MultiUserEvent::UserLeft {
            user_id: user_id.clone(),
            timestamp: SystemTime::now(),
            reason,
        });

        Ok(())
    }

    /// Update user position
    pub fn update_user_position(
        &mut self,
        user_id: &UserId,
        position: Position3D,
        orientation: [f32; 4],
    ) -> Result<()> {
        let old_position;
        let old_timestamp;
        let calculated_velocity;

        {
            let mut users = self.users.write().unwrap();
            if let Some(user) = users.get_mut(user_id) {
                old_position = user.position;
                old_timestamp = user.last_update;

                // Calculate velocity based on position change over time
                let current_time = Instant::now();
                let time_delta = current_time.duration_since(old_timestamp).as_secs_f32();

                if time_delta > 0.0 {
                    let position_delta = Position3D::new(
                        position.x - old_position.x,
                        position.y - old_position.y,
                        position.z - old_position.z,
                    );
                    calculated_velocity = Position3D::new(
                        position_delta.x / time_delta,
                        position_delta.y / time_delta,
                        position_delta.z / time_delta,
                    );
                    user.velocity = calculated_velocity;
                } else {
                    // If no time has passed, maintain previous velocity
                    calculated_velocity = user.velocity;
                }

                user.position = position;
                user.orientation = orientation;
                user.last_update = current_time;
            } else {
                return Err(Error::LegacyPosition(format!("User {user_id} not found")));
            }
        }

        // Estimate latency based on position interpolator history
        let estimated_latency = self
            .sync_manager
            .position_interpolator
            .estimate_latency(user_id)
            .unwrap_or(0.0);

        // Update position interpolator
        self.sync_manager.position_interpolator.add_position_sample(
            user_id,
            PositionSnapshot {
                position,
                orientation,
                velocity: calculated_velocity,
                timestamp: Instant::now(),
                latency_ms: estimated_latency,
            },
        );

        // Record event
        self.record_event(MultiUserEvent::UserMoved {
            user_id: user_id.clone(),
            timestamp: SystemTime::now(),
            old_position,
            new_position: position,
        });

        Ok(())
    }

    /// Add a friend relationship between two users
    pub fn add_friend(&self, user_id: &UserId, friend_id: &UserId) -> Result<()> {
        if user_id == friend_id {
            return Err(Error::LegacyPosition(
                "Cannot add self as friend".to_string(),
            ));
        }

        {
            let mut users = self.users.write().unwrap();

            // Verify both users exist
            if !users.contains_key(user_id) {
                return Err(Error::LegacyPosition(format!("User {user_id} not found")));
            }
            if !users.contains_key(friend_id) {
                return Err(Error::LegacyPosition(format!(
                    "Friend {friend_id} not found"
                )));
            }

            // Add bidirectional friendship
            if let Some(user) = users.get_mut(user_id) {
                user.friends.insert(friend_id.clone());
            }
            if let Some(friend) = users.get_mut(friend_id) {
                friend.friends.insert(user_id.clone());
            }
        }

        // Record event
        self.record_event(MultiUserEvent::UserMoved {
            user_id: user_id.clone(),
            timestamp: SystemTime::now(),
            old_position: Position3D::new(0.0, 0.0, 0.0), // Placeholder
            new_position: Position3D::new(0.0, 0.0, 0.0), // Placeholder
        });

        Ok(())
    }

    /// Remove a friend relationship between two users
    pub fn remove_friend(&self, user_id: &UserId, friend_id: &UserId) -> Result<()> {
        {
            let mut users = self.users.write().unwrap();

            // Remove bidirectional friendship
            if let Some(user) = users.get_mut(user_id) {
                user.friends.remove(friend_id);
            }
            if let Some(friend) = users.get_mut(friend_id) {
                friend.friends.remove(user_id);
            }
        }

        Ok(())
    }

    /// Check if two users are friends
    pub fn are_friends(&self, user_id: &UserId, friend_id: &UserId) -> Result<bool> {
        let users = self.users.read().unwrap();

        if let Some(user) = users.get(user_id) {
            Ok(user.friends.contains(friend_id))
        } else {
            Err(Error::LegacyPosition(format!("User {user_id} not found")))
        }
    }

    /// Get list of friends for a user
    pub fn get_friends(&self, user_id: &UserId) -> Result<Vec<UserId>> {
        let users = self.users.read().unwrap();

        if let Some(user) = users.get(user_id) {
            Ok(user.friends.iter().cloned().collect())
        } else {
            Err(Error::LegacyPosition(format!("User {user_id} not found")))
        }
    }

    /// Add an audio source
    pub fn add_audio_source(&self, source: MultiUserAudioSource) -> Result<()> {
        let source_id = source.id.clone();
        let user_id = source.owner_id.clone();
        let source_type = source.source_type;

        {
            let mut sources = self.sources.write().unwrap();
            sources.insert(source_id.clone(), source);
        }

        // Record event
        self.record_event(MultiUserEvent::SourceCreated {
            source_id,
            user_id,
            timestamp: SystemTime::now(),
            source_type,
        });

        Ok(())
    }

    /// Remove an audio source
    pub fn remove_audio_source(&self, source_id: &SourceId, reason: &str) -> Result<()> {
        {
            let mut sources = self.sources.write().unwrap();
            sources.remove(source_id);
        }

        // Record event
        self.record_event(MultiUserEvent::SourceRemoved {
            source_id: source_id.clone(),
            timestamp: SystemTime::now(),
            reason: reason.to_string(),
        });

        Ok(())
    }

    /// Process spatial audio for all users
    pub fn process_audio(&mut self) -> Result<HashMap<UserId, Vec<f32>>> {
        let users = self.users.read().unwrap();
        let mut output_buffers = HashMap::new();

        for (user_id, user) in users.iter() {
            // Create personalized audio mix for this user
            let audio_buffer =
                self.audio_processor
                    .process_for_user(user, &users, &self.sources)?;
            output_buffers.insert(user_id.clone(), audio_buffer);
        }

        Ok(output_buffers)
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> MultiUserMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Create a spatial zone
    pub fn create_zone(&self, zone: SpatialZone) -> Result<()> {
        let mut zones = self.zones.write().unwrap();
        zones.insert(zone.id.clone(), zone);
        Ok(())
    }

    /// Check if user has permission for an action
    pub fn check_permission(&self, user_id: &UserId, permission: Permission) -> Result<bool> {
        let users = self.users.read().unwrap();
        if let Some(user) = users.get(user_id) {
            if let Some(permissions) = self
                .config
                .privacy_settings
                .permission_system
                .role_permissions
                .get(&user.role)
            {
                Ok(permissions.contains(&permission))
            } else {
                Ok(false)
            }
        } else {
            Err(Error::LegacyPosition(format!("User {user_id} not found")))
        }
    }

    /// Record an event in the history
    fn record_event(&self, event: MultiUserEvent) {
        let mut history = self.event_history.write().unwrap();
        history.push_back(event);

        // Keep only recent events (last 1000)
        if history.len() > 1000 {
            history.pop_front();
        }
    }
}

impl SynchronizationManager {
    /// Create a new synchronization manager
    pub fn new() -> Self {
        Self {
            clock: Arc::new(RwLock::new(SynchronizedClock::new())),
            position_interpolator: PositionInterpolator::new(),
            latency_compensator: LatencyCompensator::new(),
        }
    }
}

impl Default for SynchronizationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SynchronizedClock {
    /// Create a new synchronized clock
    pub fn new() -> Self {
        Self {
            local_time: Instant::now(),
            time_offset_ms: 0,
            sync_accuracy_ms: 0.0,
            last_sync: Instant::now(),
        }
    }

    /// Get the current synchronized time
    pub fn current_time(&self) -> Instant {
        self.local_time
    }
}

impl Default for SynchronizedClock {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionInterpolator {
    /// Create a new position interpolator
    pub fn new() -> Self {
        Self {
            position_histories: HashMap::new(),
            interpolation_method: InterpolationMethod::Linear,
            prediction_horizon_ms: 50.0,
        }
    }

    /// Add a position sample for a user
    pub fn add_position_sample(&mut self, user_id: &UserId, snapshot: PositionSnapshot) {
        let history = self.position_histories.entry(user_id.clone()).or_default();
        history.push_back(snapshot);

        // Keep only recent samples (last 10)
        if history.len() > 10 {
            history.pop_front();
        }
    }

    /// Interpolate user position at a specific time
    pub fn interpolate_position(
        &self,
        user_id: &UserId,
        target_time: Instant,
    ) -> Option<Position3D> {
        let history = self.position_histories.get(user_id)?;
        if history.len() < 1 {
            return None;
        }

        if history.len() == 1 {
            return history.back().map(|s| s.position);
        }

        match self.interpolation_method {
            InterpolationMethod::Linear => self.linear_interpolation(history, target_time),
            InterpolationMethod::CubicSpline => {
                self.cubic_spline_interpolation(history, target_time)
            }
            InterpolationMethod::Kalman => self.kalman_interpolation(history, target_time),
            InterpolationMethod::Physics => self.physics_interpolation(history, target_time),
        }
    }

    /// Linear interpolation between two closest points
    fn linear_interpolation(
        &self,
        history: &VecDeque<PositionSnapshot>,
        target_time: Instant,
    ) -> Option<Position3D> {
        // Find the two samples that bracket the target time
        let mut before_sample = None;
        let mut after_sample = None;

        for (i, sample) in history.iter().enumerate() {
            if sample.timestamp <= target_time {
                before_sample = Some(sample);
            }
            if sample.timestamp >= target_time && after_sample.is_none() {
                after_sample = Some(sample);
                break;
            }
        }

        match (before_sample, after_sample) {
            (Some(before), Some(after)) if before.timestamp != after.timestamp => {
                let total_duration = after
                    .timestamp
                    .duration_since(before.timestamp)
                    .as_secs_f32();
                let elapsed_duration = target_time.duration_since(before.timestamp).as_secs_f32();
                let t = elapsed_duration / total_duration;

                Some(Position3D::new(
                    before.position.x + t * (after.position.x - before.position.x),
                    before.position.y + t * (after.position.y - before.position.y),
                    before.position.z + t * (after.position.z - before.position.z),
                ))
            }
            (Some(sample), _) => Some(sample.position),
            (_, Some(sample)) => Some(sample.position),
            _ => history.back().map(|s| s.position),
        }
    }

    /// Cubic spline interpolation for smoother movement
    fn cubic_spline_interpolation(
        &self,
        history: &VecDeque<PositionSnapshot>,
        target_time: Instant,
    ) -> Option<Position3D> {
        if history.len() < 4 {
            return self.linear_interpolation(history, target_time);
        }

        // For simplicity, use a cubic Hermite spline with velocity information
        let samples: Vec<_> = history.iter().collect();
        let n = samples.len();

        // Find the segment containing target_time
        for i in 1..n {
            if samples[i].timestamp >= target_time {
                let p0 = if i >= 2 { samples[i - 2] } else { samples[0] };
                let p1 = samples[i - 1];
                let p2 = samples[i];
                let p3 = if i + 1 < n {
                    samples[i + 1]
                } else {
                    samples[n - 1]
                };

                let t1 = p1
                    .timestamp
                    .duration_since(p0.timestamp)
                    .as_secs_f32()
                    .max(0.001);
                let t2 = p2
                    .timestamp
                    .duration_since(p1.timestamp)
                    .as_secs_f32()
                    .max(0.001);
                let t3 = p3
                    .timestamp
                    .duration_since(p2.timestamp)
                    .as_secs_f32()
                    .max(0.001);

                let target_offset = target_time.duration_since(p1.timestamp).as_secs_f32();
                let t = target_offset / t2;

                // Calculate tangents (velocities)
                let m1 = Position3D::new(
                    (p2.position.x - p0.position.x) / (t1 + t2),
                    (p2.position.y - p0.position.y) / (t1 + t2),
                    (p2.position.z - p0.position.z) / (t1 + t2),
                );
                let m2 = Position3D::new(
                    (p3.position.x - p1.position.x) / (t2 + t3),
                    (p3.position.y - p1.position.y) / (t2 + t3),
                    (p3.position.z - p1.position.z) / (t2 + t3),
                );

                // Hermite basis functions
                let t2 = t * t;
                let t3 = t2 * t;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;

                return Some(Position3D::new(
                    h00 * p1.position.x + h10 * m1.x * t2 + h01 * p2.position.x + h11 * m2.x * t2,
                    h00 * p1.position.y + h10 * m1.y * t2 + h01 * p2.position.y + h11 * m2.y * t2,
                    h00 * p1.position.z + h10 * m1.z * t2 + h01 * p2.position.z + h11 * m2.z * t2,
                ));
            }
        }

        // Fallback to linear interpolation
        self.linear_interpolation(history, target_time)
    }

    /// Kalman filter-based prediction with velocity estimation
    fn kalman_interpolation(
        &self,
        history: &VecDeque<PositionSnapshot>,
        target_time: Instant,
    ) -> Option<Position3D> {
        if history.len() < 2 {
            return history.back().map(|s| s.position);
        }

        let latest = history.back()?;
        let time_delta = target_time.duration_since(latest.timestamp).as_secs_f32();

        // Simple Kalman-like prediction using stored velocity
        Some(Position3D::new(
            latest.position.x + latest.velocity.x * time_delta,
            latest.position.y + latest.velocity.y * time_delta,
            latest.position.z + latest.velocity.z * time_delta,
        ))
    }

    /// Physics-based prediction considering acceleration
    fn physics_interpolation(
        &self,
        history: &VecDeque<PositionSnapshot>,
        target_time: Instant,
    ) -> Option<Position3D> {
        if history.len() < 3 {
            return self.kalman_interpolation(history, target_time);
        }

        let samples: Vec<_> = history.iter().rev().take(3).collect();
        let latest = samples[0];
        let prev1 = samples[1];
        let prev2 = samples[2];

        let dt1 = latest
            .timestamp
            .duration_since(prev1.timestamp)
            .as_secs_f32()
            .max(0.001);
        let dt2 = prev1
            .timestamp
            .duration_since(prev2.timestamp)
            .as_secs_f32()
            .max(0.001);

        // Calculate acceleration from velocity changes
        let accel = Position3D::new(
            (latest.velocity.x - prev1.velocity.x) / dt1,
            (latest.velocity.y - prev1.velocity.y) / dt1,
            (latest.velocity.z - prev1.velocity.z) / dt1,
        );

        let time_delta = target_time.duration_since(latest.timestamp).as_secs_f32();

        // Physics equation: s = s0 + v0*t + 0.5*a*t^2
        Some(Position3D::new(
            latest.position.x
                + latest.velocity.x * time_delta
                + 0.5 * accel.x * time_delta * time_delta,
            latest.position.y
                + latest.velocity.y * time_delta
                + 0.5 * accel.y * time_delta * time_delta,
            latest.position.z
                + latest.velocity.z * time_delta
                + 0.5 * accel.z * time_delta * time_delta,
        ))
    }

    /// Estimate network latency based on position update patterns
    pub fn estimate_latency(&self, user_id: &UserId) -> Option<f64> {
        let history = self.position_histories.get(user_id)?;

        if history.len() < 2 {
            return Some(0.0);
        }

        // Calculate average time between position updates
        let mut total_intervals = 0.0;
        let mut interval_count = 0;

        for i in 1..history.len() {
            let current_sample = &history[i];
            let previous_sample = &history[i - 1];

            let interval = current_sample
                .timestamp
                .duration_since(previous_sample.timestamp)
                .as_secs_f64()
                * 1000.0;
            total_intervals += interval;
            interval_count += 1;
        }

        if interval_count > 0 {
            let avg_interval = total_intervals / interval_count as f64;
            // Estimate latency as half the average update interval
            // This assumes network latency is roughly proportional to update frequency
            Some((avg_interval * 0.5).min(100.0)) // Cap at 100ms
        } else {
            Some(0.0)
        }
    }
}

impl Default for PositionInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyCompensator {
    /// Create a new latency compensator
    pub fn new() -> Self {
        Self {
            user_latencies: HashMap::new(),
            compensation_method: CompensationMethod::Adaptive,
            max_compensation_ms: 100.0,
        }
    }

    /// Update the measured latency for a user
    pub fn update_user_latency(&mut self, user_id: &UserId, latency_ms: f64) {
        self.user_latencies.insert(user_id.clone(), latency_ms);
    }

    /// Get the compensation delay for a user
    pub fn get_compensation_delay(&self, user_id: &UserId) -> f64 {
        if let Some(&latency) = self.user_latencies.get(user_id) {
            (latency * 0.5).min(self.max_compensation_ms)
        } else {
            0.0
        }
    }
}

impl Default for LatencyCompensator {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiUserAudioProcessor {
    /// Create a new multi-user audio processor
    pub fn new(config: &MultiUserConfig) -> Result<Self> {
        let mixer_config = MixerConfig {
            max_sources: config.max_sources_per_user * config.max_users_per_room,
            buffer_size: 1024,
            sample_rate: 48000,
            spatial_quality: config.audio_quality,
            optimization_level: OptimizationLevel::Balanced,
        };

        Ok(Self {
            mixer: SpatialAudioMixer::new(mixer_config),
            vad: VoiceActivityDetector::new(),
            effects: AudioEffectsProcessor::new(),
            codec: AudioCodec::new(AudioFormat::Opus),
        })
    }

    /// Process spatial audio for a specific user
    pub fn process_for_user(
        &mut self,
        listener: &MultiUserUser,
        all_users: &HashMap<UserId, MultiUserUser>,
        sources: &Arc<RwLock<HashMap<SourceId, MultiUserAudioSource>>>,
    ) -> Result<Vec<f32>> {
        // Set listener position for spatial processing
        self.mixer
            .set_listener_position(&listener.id, listener.position);

        // Process all audible sources for this listener
        let sources_guard = sources.read().unwrap();
        let mut mixed_audio = vec![0.0f32; 1024]; // Buffer size

        for (source_id, source) in sources_guard.iter() {
            // Check if this source is audible to the listener
            if self.is_source_audible(listener, source)? {
                // Calculate spatial audio for this source
                let spatial_audio = self.mixer.process_source(source, listener.position)?;

                // Mix into output buffer
                for (i, sample) in spatial_audio.iter().enumerate() {
                    if i < mixed_audio.len() {
                        mixed_audio[i] += sample;
                    }
                }
            }
        }

        // Apply listener-specific effects
        self.effects
            .process_user_effects(&listener.id, &mut mixed_audio)?;

        Ok(mixed_audio)
    }

    /// Check if an audio source is audible to a listener
    fn is_source_audible(
        &self,
        listener: &MultiUserUser,
        source: &MultiUserAudioSource,
    ) -> Result<bool> {
        // Check distance
        let distance = listener.position.distance_to(&source.position);
        if distance > source.spatial_properties.max_distance {
            return Ok(false);
        }

        // Check access control
        match source.access_control.visibility {
            SourceVisibility::Public => Ok(true),
            SourceVisibility::Private => Ok(source.owner_id == listener.id),
            SourceVisibility::Whitelist => {
                Ok(source.access_control.allowed_users.contains(&listener.id))
            }
            SourceVisibility::Friends => {
                // Check if listener is friends with the source owner
                Ok(listener.friends.contains(&source.owner_id) || source.owner_id == listener.id)
            }
            SourceVisibility::Moderators => Ok(matches!(
                listener.role,
                UserRole::Moderator | UserRole::Administrator
            )),
        }
    }
}

impl SpatialAudioMixer {
    /// Create a new spatial audio mixer
    pub fn new(config: MixerConfig) -> Self {
        Self {
            listener_positions: HashMap::new(),
            source_manager: Arc::new(RwLock::new(HashMap::new())),
            mixer_config: config,
        }
    }

    /// Set the position of a listener for spatial audio processing
    pub fn set_listener_position(&mut self, user_id: &UserId, position: Position3D) {
        self.listener_positions.insert(user_id.clone(), position);
    }

    /// Process an audio source for spatial audio rendering
    pub fn process_source(
        &self,
        source: &MultiUserAudioSource,
        listener_position: Position3D,
    ) -> Result<Vec<f32>> {
        // Calculate spatial parameters
        let distance = listener_position.distance_to(&source.position);
        let attenuation = self.calculate_distance_attenuation(distance, &source.spatial_properties);

        // Generate dummy spatial audio (in real implementation, this would be proper HRTF processing)
        let buffer_size = self.mixer_config.buffer_size;
        let mut output = vec![0.0f32; buffer_size];

        // Simple distance-based attenuation
        let volume = source.volume * attenuation;
        for sample in output.iter_mut() {
            *sample = volume * 0.1; // Dummy audio signal
        }

        Ok(output)
    }

    /// Calculate volume attenuation based on distance
    fn calculate_distance_attenuation(&self, distance: f32, properties: &SpatialProperties) -> f32 {
        if distance <= properties.reference_distance {
            return 1.0;
        }

        let ratio = properties.reference_distance / distance;
        ratio.powf(properties.rolloff_factor).min(1.0)
    }
}

impl VoiceActivityDetector {
    /// Create a new voice activity detector
    pub fn new() -> Self {
        Self {
            algorithm: VadAlgorithm::Energy,
            user_states: HashMap::new(),
            thresholds: VadThresholds {
                energy_threshold: 0.01,
                min_speaking_duration_ms: 100,
                min_silence_duration_ms: 200,
                confidence_threshold: 0.7,
            },
        }
    }

    /// Process audio buffer to detect voice activity
    pub fn process_user_audio(&mut self, user_id: &UserId, audio_buffer: &[f32]) -> bool {
        // Calculate energy level
        let energy = audio_buffer.iter().map(|&x| x * x).sum::<f32>() / audio_buffer.len() as f32;

        // Get or create user state
        let state = self
            .user_states
            .entry(user_id.clone())
            .or_insert_with(|| VadState {
                is_speaking: false,
                confidence: 0.0,
                energy_history: VecDeque::new(),
                speaking_duration: Duration::new(0, 0),
                silence_duration: Duration::new(0, 0),
            });

        // Update energy history
        state.energy_history.push_back(energy);
        if state.energy_history.len() > 10 {
            state.energy_history.pop_front();
        }

        // Simple energy-based detection
        state.is_speaking = energy > self.thresholds.energy_threshold;
        state.confidence = if state.is_speaking { 0.9 } else { 0.1 };

        state.is_speaking
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEffectsProcessor {
    /// Create a new audio effects processor
    pub fn new() -> Self {
        Self {
            effects: HashMap::new(),
            user_effects: HashMap::new(),
            zone_effects: HashMap::new(),
        }
    }

    /// Apply user-specific audio effects to an audio buffer
    pub fn process_user_effects(
        &mut self,
        user_id: &UserId,
        audio_buffer: &mut [f32],
    ) -> Result<()> {
        // Apply user-specific effects
        if let Some(effect_chain) = self.user_effects.get(user_id) {
            let mut temp_buffer = audio_buffer.to_vec();
            for effect_name in effect_chain {
                if let Some(effect) = self.effects.get_mut(effect_name) {
                    effect.process(&temp_buffer, audio_buffer)?;
                    temp_buffer.copy_from_slice(audio_buffer);
                }
            }
        }
        Ok(())
    }
}

impl Default for AudioEffectsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioCodec {
    /// Create a new audio codec with the specified format
    pub fn new(format: AudioFormat) -> Self {
        Self {
            format,
            compression: CompressionSettings {
                bitrate_kbps: 64,
                complexity: 5,
                variable_bitrate: true,
                low_latency: true,
            },
            codec_states: HashMap::new(),
        }
    }
}

/// Builder for multi-user configuration
pub struct MultiUserConfigBuilder {
    config: MultiUserConfig,
}

impl MultiUserConfigBuilder {
    /// Create a new multi-user configuration builder
    pub fn new() -> Self {
        Self {
            config: MultiUserConfig::default(),
        }
    }

    /// Set maximum number of users per room
    pub fn max_users(mut self, max_users: usize) -> Self {
        self.config.max_users_per_room = max_users;
        self
    }

    /// Set audio quality level (0.0-1.0)
    pub fn audio_quality(mut self, quality: f32) -> Self {
        self.config.audio_quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Set maximum latency in milliseconds
    pub fn max_latency_ms(mut self, latency: f64) -> Self {
        self.config.max_latency_ms = latency;
        self
    }

    /// Enable or disable encryption
    pub fn enable_encryption(mut self, enabled: bool) -> Self {
        self.config.privacy_settings.encryption_enabled = enabled;
        self
    }

    /// Set bandwidth limit in kbps
    pub fn bandwidth_limit_kbps(mut self, limit: u32) -> Self {
        self.config.bandwidth_settings.max_bandwidth_kbps = limit;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MultiUserConfig {
        self.config
    }
}

impl Default for MultiUserConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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
            .enable_encryption(false)
            .bandwidth_limit_kbps(256)
            .build();

        assert_eq!(config.max_users_per_room, 25);
        assert_eq!(config.audio_quality, 0.9);
        assert_eq!(config.max_latency_ms, 50.0);
        assert!(!config.privacy_settings.encryption_enabled);
        assert_eq!(config.bandwidth_settings.max_bandwidth_kbps, 256);
    }

    #[test]
    fn test_multiuser_environment_creation() {
        let config = MultiUserConfig::default();
        let environment = MultiUserEnvironment::new(config);
        assert!(environment.is_ok());
    }

    #[test]
    fn test_user_role_permissions() {
        let permission_system = PermissionSystem::default();
        let participant_permissions = &permission_system.role_permissions[&UserRole::Participant];

        assert!(participant_permissions.contains(&Permission::Speak));
        assert!(participant_permissions.contains(&Permission::Move));
        assert!(participant_permissions.contains(&Permission::CreateSources));
        assert!(!participant_permissions.contains(&Permission::KickUsers));
    }

    #[test]
    fn test_spatial_zone_bounds() {
        let sphere_zone = ZoneBounds::Sphere {
            center: Position3D::new(0.0, 0.0, 0.0),
            radius: 5.0,
        };

        match sphere_zone {
            ZoneBounds::Sphere { center, radius } => {
                assert_eq!(center.x, 0.0);
                assert_eq!(radius, 5.0);
            }
            _ => panic!("Expected sphere zone"),
        }
    }

    #[test]
    fn test_position_interpolator() {
        let mut interpolator = PositionInterpolator::new();
        let user_id = "test_user".to_string();

        let snapshot = PositionSnapshot {
            position: Position3D::new(1.0, 2.0, 3.0),
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: Position3D::new(0.0, 0.0, 0.0),
            timestamp: Instant::now(),
            latency_ms: 10.0,
        };

        interpolator.add_position_sample(&user_id, snapshot);
        let interpolated = interpolator.interpolate_position(&user_id, Instant::now());

        assert!(interpolated.is_some());
        let pos = interpolated.unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
        assert_eq!(pos.z, 3.0);
    }

    #[test]
    fn test_voice_activity_detector() {
        let mut vad = VoiceActivityDetector::new();
        let user_id = "test_user".to_string();

        // Test with silent audio
        let silent_audio = vec![0.0f32; 1024];
        let is_speaking = vad.process_user_audio(&user_id, &silent_audio);
        assert!(!is_speaking);

        // Test with loud audio
        let loud_audio = vec![0.5f32; 1024];
        let is_speaking = vad.process_user_audio(&user_id, &loud_audio);
        assert!(is_speaking);
    }

    #[test]
    fn test_distance_attenuation() {
        let config = MixerConfig {
            max_sources: 10,
            buffer_size: 1024,
            sample_rate: 48000,
            spatial_quality: 0.8,
            optimization_level: OptimizationLevel::Balanced,
        };

        let mixer = SpatialAudioMixer::new(config);
        let properties = SpatialProperties {
            directivity: DirectionalPattern::Omnidirectional,
            reference_distance: 1.0,
            rolloff_factor: 1.0,
            max_distance: 100.0,
            doppler_factor: 1.0,
            room_interaction: true,
        };

        // Test reference distance
        let attenuation = mixer.calculate_distance_attenuation(1.0, &properties);
        assert_eq!(attenuation, 1.0);

        // Test double distance
        let attenuation = mixer.calculate_distance_attenuation(2.0, &properties);
        assert_eq!(attenuation, 0.5);

        // Test quadruple distance
        let attenuation = mixer.calculate_distance_attenuation(4.0, &properties);
        assert_eq!(attenuation, 0.25);
    }

    #[test]
    fn test_audio_source_visibility() {
        use crate::multiuser::*;

        let source = MultiUserAudioSource {
            id: "test_source".to_string(),
            owner_id: "owner".to_string(),
            position: Position3D::new(0.0, 0.0, 0.0),
            source_type: AudioSourceType::Voice,
            volume: 1.0,
            is_active: true,
            spatial_properties: SpatialProperties {
                directivity: DirectionalPattern::Omnidirectional,
                reference_distance: 1.0,
                rolloff_factor: 1.0,
                max_distance: 10.0,
                doppler_factor: 1.0,
                room_interaction: true,
            },
            access_control: SourceAccessControl {
                visibility: SourceVisibility::Public,
                allowed_users: vec![],
                blocked_users: vec![],
                spatial_zones: vec![],
            },
            quality_settings: SourceQualitySettings {
                bitrate_kbps: 64,
                spatial_quality: 0.8,
                noise_reduction: 0.5,
                echo_cancellation: 0.7,
                compression_ratio: 2.0,
            },
        };

        assert_eq!(source.source_type, AudioSourceType::Voice);
        assert_eq!(source.access_control.visibility, SourceVisibility::Public);
    }
}
