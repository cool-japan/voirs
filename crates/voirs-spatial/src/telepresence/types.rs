//! Core types and enums for telepresence system

use crate::Position3D;
use serde::{Deserialize, Serialize};

/// Noise suppression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseSuppressionAlgorithm {
    /// Spectral subtraction
    SpectralSubtraction,

    /// Wiener filtering
    WienerFilter,

    /// Neural network based
    NeuralNetwork,

    /// Minimum mean square error
    MMSE,

    /// Hybrid approach
    Hybrid,
}

/// Echo cancellation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EchoCancellationAlgorithm {
    /// Normalized Least Mean Squares
    NLMS,

    /// Recursive Least Squares
    RLS,

    /// Proportionate NLMS
    PNLMS,

    /// Kalman filter based
    Kalman,

    /// Frequency domain adaptive filter
    FrequencyDomain,
}

/// VAD algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VadAlgorithm {
    /// Energy-based VAD
    Energy,

    /// Spectral-based VAD
    Spectral,

    /// Model-based VAD
    Model,

    /// Neural network VAD
    Neural,

    /// Hybrid VAD
    Hybrid,
}

/// EQ band types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqBandType {
    /// Low shelf
    LowShelf,

    /// High shelf
    HighShelf,

    /// Peaking
    Peaking,

    /// Low pass
    LowPass,

    /// High pass
    HighPass,

    /// Notch
    Notch,
}

/// Equalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqualizationType {
    /// Parametric EQ
    Parametric,

    /// Graphic EQ
    Graphic,

    /// Shelving EQ
    Shelving,

    /// Custom filter
    Custom,
}

/// Bandwidth extension algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandwidthExtensionAlgorithm {
    /// Spectral replication
    SpectralReplication,

    /// Harmonic extension
    HarmonicExtension,

    /// Neural network extension
    NeuralExtension,

    /// Model-based extension
    ModelBased,
}

/// Noise color types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseColor {
    /// White noise
    White,

    /// Pink noise
    Pink,

    /// Brown noise
    Brown,

    /// Blue noise
    Blue,

    /// Custom spectrum
    Custom,
}

/// HRTF personalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalizationMethod {
    /// Anthropometric scaling
    Anthropometric,

    /// Machine learning adaptation
    MachineLearning,

    /// User feedback adaptation
    UserFeedback,

    /// Hybrid approach
    Hybrid,
}

/// Room shape types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomShape {
    /// Rectangular room
    Rectangular,

    /// L-shaped room
    LShaped,

    /// Circular room
    Circular,

    /// Irregular shape
    Irregular,

    /// Custom shape
    Custom,
}

/// Opening types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpeningType {
    /// Door
    Door,

    /// Window
    Window,

    /// Archway
    Archway,

    /// Vent
    Vent,

    /// Custom opening
    Custom,
}

/// Furniture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FurnitureType {
    /// Table
    Table,

    /// Chair
    Chair,

    /// Sofa
    Sofa,

    /// Bookshelf
    Bookshelf,

    /// Desk
    Desk,

    /// Bed
    Bed,

    /// Custom furniture
    Custom,
}

/// Movement types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementType {
    /// Free movement
    Free,

    /// Walking only
    Walking,

    /// Seated position
    Seated,

    /// Standing only
    Standing,

    /// Teleport movement
    Teleport,
}

/// Acoustic matching algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcousticMatchingAlgorithm {
    /// Direct parameter matching
    Direct,

    /// Convolution-based matching
    Convolution,

    /// ML-based matching
    MachineLearning,

    /// Hybrid matching
    Hybrid,
}

/// Shared space types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharedSpaceType {
    /// Virtual lobby
    Lobby,

    /// Meeting room
    MeetingRoom,

    /// Breakout room
    BreakoutRoom,

    /// Social space
    SocialSpace,

    /// Custom space
    Custom,
}

/// Attenuation models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttenuationModel {
    /// Inverse distance law
    InverseDistance,

    /// Inverse square law
    InverseSquare,

    /// Linear attenuation
    Linear,

    /// Exponential attenuation
    Exponential,

    /// Custom model
    Custom,
}

/// Quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Low quality (optimized for bandwidth)
    Low,

    /// Medium quality (balanced)
    Medium,

    /// High quality (optimized for quality)
    High,

    /// Ultra quality (maximum quality)
    Ultra,

    /// Custom quality settings
    Custom,
}

/// Latency priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyPriority {
    /// Minimize latency
    Low,

    /// Balance latency and quality
    Medium,

    /// Accept higher latency for quality
    High,
}

/// Audio codecs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodec {
    /// Opus codec
    Opus,

    /// AAC codec
    AAC,

    /// MP3 codec
    MP3,

    /// PCM (uncompressed)
    PCM,

    /// G.722 codec
    G722,

    /// G.711 codec
    G711,

    /// Custom codec
    Custom(String),
}

/// Codec fallback behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecFallbackBehavior {
    /// Use next preferred codec
    NextPreferred,

    /// Use most compatible codec
    MostCompatible,

    /// Use lowest latency codec
    LowestLatency,

    /// Fail if preferred not available
    Fail,
}

/// Spatial quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialQualityLevel {
    /// Basic stereo positioning
    Basic,

    /// Enhanced spatial processing
    Enhanced,

    /// Full 3D spatial audio
    Full3D,

    /// Ultra-high fidelity spatial
    UltraHiFi,
}

/// Head tracking sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingSource {
    /// VR headset tracking
    VRHeadset,

    /// Webcam-based tracking
    Webcam,

    /// IMU-based tracking
    IMU,

    /// Phone/tablet gyroscope
    MobileGyroscope,

    /// External tracking system
    External,

    /// No tracking (static)
    None,
}

/// Prediction algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionAlgorithm {
    /// Linear extrapolation
    Linear,

    /// Kalman filter
    Kalman,

    /// Neural network
    Neural,

    /// Adaptive filter
    Adaptive,
}

/// Indicator styles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorStyle {
    /// Minimal indicators
    Minimal,

    /// Standard indicators
    Standard,

    /// Rich indicators
    Rich,

    /// Custom style
    Custom,
}

/// Connection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct peer-to-peer
    P2P,

    /// Server-mediated
    ServerMediated,

    /// TURN relay
    TurnRelay,

    /// STUN-assisted
    StunAssisted,

    /// Automatic selection
    Auto,
}

/// DSCP marking for QoS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DscpMarking {
    /// Best effort
    BestEffort,

    /// Expedited forwarding
    ExpeditedForwarding,

    /// Assured forwarding
    AssuredForwarding,

    /// Voice
    Voice,

    /// Custom DSCP value
    Custom(u8),
}

/// Congestion control algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CongestionControlAlgorithm {
    /// TCP-friendly
    TcpFriendly,

    /// Google Congestion Control
    GCC,

    /// WebRTC congestion control
    WebRTC,

    /// Custom algorithm
    Custom,
}

/// STUN protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StunProtocol {
    /// UDP
    UDP,

    /// TCP
    TCP,

    /// TLS
    TLS,
}

/// TURN protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnProtocol {
    /// UDP
    UDP,

    /// TCP
    TCP,

    /// TLS
    TLS,

    /// DTLS
    DTLS,
}

/// ICE gathering policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IceGatheringPolicy {
    /// Gather all candidates
    All,

    /// Only relay candidates
    Relay,

    /// No host candidates
    NoHost,
}

/// ICE transport policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IceTransportPolicy {
    /// All transports
    All,

    /// Only relay
    Relay,

    /// No UDP
    NoUDP,
}

/// Redundancy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedundancyType {
    /// Active-passive
    ActivePassive,

    /// Active-active
    ActiveActive,

    /// Load balancing
    LoadBalancing,

    /// Path diversity
    PathDiversity,
}

/// Channel configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelConfiguration {
    /// Mono
    Mono,

    /// Stereo
    Stereo,

    /// Binaural
    Binaural,

    /// Multi-channel
    MultiChannel(u8),
}

/// HRTF quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HrtfQualityLevel {
    /// Basic HRTF
    Basic,

    /// Standard HRTF
    Standard,

    /// High-quality HRTF
    High,

    /// Ultra HRTF
    Ultra,
}

/// Room quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomQualityLevel {
    /// Simple room model
    Simple,

    /// Standard room model
    Standard,

    /// Advanced room model
    Advanced,

    /// Ultra-realistic room model
    UltraRealistic,
}

/// Distance precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistancePrecisionLevel {
    /// Low precision
    Low,

    /// Medium precision
    Medium,

    /// High precision
    High,

    /// Ultra precision
    Ultra,
}

/// Quality adaptation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityAdaptationAlgorithm {
    /// Bandwidth-based adaptation
    BandwidthBased,

    /// Latency-based adaptation
    LatencyBased,

    /// Machine learning adaptation
    MachineLearning,

    /// Hybrid adaptation
    Hybrid,
}

/// Adaptation speeds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationSpeed {
    /// Slow adaptation
    Slow,

    /// Medium adaptation
    Medium,

    /// Fast adaptation
    Fast,

    /// Instant adaptation
    Instant,
}

/// Performance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Audio latency
    AudioLatency,

    /// Processing latency
    ProcessingLatency,

    /// Network latency
    NetworkLatency,

    /// Packet loss
    PacketLoss,

    /// Jitter
    Jitter,

    /// CPU usage
    CpuUsage,

    /// Memory usage
    MemoryUsage,

    /// Audio quality score
    AudioQuality,

    /// Spatial accuracy
    SpatialAccuracy,
}

/// Anonymization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnonymizationMethod {
    /// Voice conversion
    VoiceConversion,

    /// Pitch shifting
    PitchShifting,

    /// Spectral masking
    SpectralMasking,

    /// Statistical anonymization
    Statistical,
}

/// Anonymization strengths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnonymizationStrength {
    /// Light anonymization
    Light,

    /// Medium anonymization
    Medium,

    /// Strong anonymization
    Strong,

    /// Maximum anonymization
    Maximum,
}

/// Consent withdrawal mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentWithdrawalMechanism {
    /// Immediate withdrawal
    Immediate,

    /// End of session withdrawal
    EndOfSession,

    /// Manual request
    ManualRequest,

    /// Automatic expiry
    AutomaticExpiry,
}

/// User state in session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserState {
    /// Active and speaking
    Active,

    /// Connected but muted
    Muted,

    /// Away from keyboard
    Away,

    /// Busy/do not disturb
    Busy,

    /// Disconnected
    Disconnected,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Initializing
    Initializing,

    /// Active
    Active,

    /// Paused
    Paused,

    /// Reconnecting
    Reconnecting,

    /// Terminated
    Terminated,

    /// Error state
    Error,
}

/// Audio processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingStatus {
    /// Raw received data
    Raw,

    /// Decoded but not processed
    Decoded,

    /// Spatially processed
    SpatiallyProcessed,

    /// Ready for playback
    ReadyForPlayback,

    /// Processing error
    Error,
}

/// Encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256
    AES256,

    /// ChaCha20-Poly1305
    ChaCha20Poly1305,

    /// DTLS-SRTP
    DTLSSRTP,

    /// Custom algorithm
    Custom,
}

/// Key exchange methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyExchangeMethod {
    /// Diffie-Hellman
    DiffieHellman,

    /// ECDH
    ECDH,

    /// RSA
    RSA,

    /// Pre-shared key
    PreSharedKey,
}

/// Authentication methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// Username/password
    UsernamePassword,

    /// Token-based
    Token,

    /// Certificate-based
    Certificate,

    /// OAuth
    OAuth,

    /// SAML
    SAML,
}

/// Access control modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessControlMode {
    /// Allow all (default)
    AllowAll,

    /// Whitelist only
    Whitelist,

    /// Blacklist
    Blacklist,

    /// Geographic restrictions
    Geographic,
}

/// Session privacy levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionPrivacy {
    /// Public session
    Public,

    /// Private session (invite only)
    Private,

    /// Password protected
    PasswordProtected,

    /// Authenticated users only
    AuthenticatedOnly,
}

/// 3D orientation representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Orientation {
    /// Yaw (rotation around Y axis, degrees)
    pub yaw: f32,

    /// Pitch (rotation around X axis, degrees)
    pub pitch: f32,

    /// Roll (rotation around Z axis, degrees)
    pub roll: f32,
}

/// Velocity vector
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Velocity {
    /// X component (m/s)
    pub x: f32,

    /// Y component (m/s)
    pub y: f32,

    /// Z component (m/s)
    pub z: f32,
}

/// Bounding box for movement
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum corner
    pub min: Position3D,

    /// Maximum corner
    pub max: Position3D,
}

/// Movement constraints for users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementConstraints {
    /// Allowed area bounds
    pub bounds: Option<BoundingBox>,

    /// Movement speed limit (m/s)
    pub max_speed: f32,

    /// Allowed movement types
    pub allowed_movements: Vec<MovementType>,
}
