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
    pub(crate) config: MultiUserConfig,
    /// All users in the environment
    pub(crate) users: Arc<RwLock<HashMap<UserId, MultiUserUser>>>,
    /// All audio sources in the environment
    pub(crate) sources: Arc<RwLock<HashMap<SourceId, MultiUserAudioSource>>>,
    /// Spatial zones in the environment
    pub(crate) zones: Arc<RwLock<HashMap<String, SpatialZone>>>,
    /// Network synchronization manager
    pub(crate) sync_manager: SynchronizationManager,
    /// Audio processing pipeline
    pub(crate) audio_processor: MultiUserAudioProcessor,
    /// Performance metrics
    pub(crate) metrics: Arc<RwLock<MultiUserMetrics>>,
    /// Event history for debugging and analysis
    pub(crate) event_history: Arc<RwLock<VecDeque<MultiUserEvent>>>,
}

/// Synchronization manager for network coordination
pub struct SynchronizationManager {
    /// Synchronization clock
    pub(crate) clock: Arc<RwLock<SynchronizedClock>>,
    /// User position interpolation
    pub(crate) position_interpolator: PositionInterpolator,
    /// Network latency compensation
    pub(crate) latency_compensator: LatencyCompensator,
}

/// Synchronized clock for multi-user coordination
#[derive(Debug)]
pub struct SynchronizedClock {
    /// Local time reference
    pub(crate) local_time: Instant,
    /// Network-synchronized time offset
    pub(crate) time_offset_ms: i64,
    /// Clock synchronization accuracy
    pub(crate) sync_accuracy_ms: f64,
    /// Last synchronization timestamp
    pub(crate) last_sync: Instant,
}

/// Position interpolation for smooth user movement
pub struct PositionInterpolator {
    /// User position histories
    pub(crate) position_histories: HashMap<UserId, VecDeque<PositionSnapshot>>,
    /// Interpolation method
    pub(crate) interpolation_method: InterpolationMethod,
    /// Prediction horizon in milliseconds
    pub(crate) prediction_horizon_ms: f64,
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub(crate) user_latencies: HashMap<UserId, f64>,
    /// Compensation strategies
    pub(crate) compensation_method: CompensationMethod,
    /// Maximum compensation amount
    pub(crate) max_compensation_ms: f64,
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
    pub(crate) mixer: SpatialAudioMixer,
    /// Voice activity detection
    pub(crate) vad: VoiceActivityDetector,
    /// Audio effects processor
    pub(crate) effects: AudioEffectsProcessor,
    /// Compression and encoding
    pub(crate) codec: AudioCodec,
}

/// Spatial audio mixing for multiple users and sources
pub struct SpatialAudioMixer {
    /// Current listener position (for each user)
    pub(crate) listener_positions: HashMap<UserId, Position3D>,
    /// Audio source management
    pub(crate) source_manager: Arc<RwLock<HashMap<SourceId, MixerAudioSource>>>,
    /// Mixing configuration
    pub(crate) mixer_config: MixerConfig,
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
    pub(crate) algorithm: VadAlgorithm,
    /// Per-user VAD state
    pub(crate) user_states: HashMap<UserId, VadState>,
    /// Detection thresholds
    pub(crate) thresholds: VadThresholds,
}

/// Voice activity detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub(crate) effects: HashMap<String, Box<dyn AudioEffect>>,
    /// Per-user effect chains
    pub(crate) user_effects: HashMap<UserId, Vec<String>>,
    /// Zone-based effects
    pub(crate) zone_effects: HashMap<String, Vec<String>>,
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
    pub(crate) format: AudioFormat,
    /// Compression settings
    pub(crate) compression: CompressionSettings,
    /// Per-user codec state
    pub(crate) codec_states: HashMap<UserId, CodecState>,
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
