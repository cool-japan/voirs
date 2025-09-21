//! Smart Speaker Arrays for Multi-Speaker Spatial Audio
//!
//! This module provides support for smart speaker arrays including multi-room audio,
//! speaker array optimization, and automatic speaker discovery and calibration.

use crate::{
    types::{Position3D, SpatialResult},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// Smart speaker device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartSpeaker {
    /// Unique speaker identifier
    pub id: String,
    /// Speaker position in 3D space
    pub position: Position3D,
    /// Speaker capabilities
    pub capabilities: SpeakerCapabilities,
    /// Network information
    pub network_info: NetworkInfo,
    /// Calibration status
    pub calibration: CalibrationStatus,
    /// Audio characteristics
    pub audio_specs: AudioSpecs,
}

/// Speaker audio capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCapabilities {
    /// Frequency response range (Hz)
    pub frequency_range: (f32, f32),
    /// Maximum SPL (dB)
    pub max_spl: f32,
    /// Number of drivers
    pub driver_count: u8,
    /// Directivity pattern
    pub directivity: DirectivityPattern,
    /// DSP capabilities
    pub dsp_features: Vec<DspFeature>,
    /// Supported audio formats
    pub supported_formats: Vec<AudioFormat>,
}

/// Network connectivity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// IP address
    pub ip_address: String,
    /// MAC address
    pub mac_address: String,
    /// Network protocol (WiFi, Ethernet, etc.)
    pub protocol: NetworkProtocol,
    /// Signal strength (0-100)
    pub signal_strength: u8,
    /// Latency to controller (ms)
    pub latency_ms: f32,
    /// Bandwidth available (Mbps)
    pub bandwidth_mbps: f32,
}

/// Speaker calibration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStatus {
    /// Whether speaker is calibrated
    pub is_calibrated: bool,
    /// Calibration timestamp
    pub calibrated_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Measured room response
    pub room_correction: Option<RoomCorrection>,
    /// Distance measurements to other speakers
    pub inter_speaker_distances: HashMap<String, f32>,
    /// Acoustic delay compensation
    pub delay_compensation_ms: f32,
}

/// Audio specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSpecs {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
    /// Number of channels
    pub channels: u8,
    /// Buffer size (samples)
    pub buffer_size: usize,
    /// Codec latency (ms)
    pub codec_latency_ms: f32,
}

/// Directivity patterns for speakers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectivityPattern {
    /// Omnidirectional
    Omnidirectional,
    /// Cardioid (unidirectional)
    Cardioid,
    /// Bidirectional (figure-8)
    Bidirectional,
    /// Supercardioid
    Supercardioid,
    /// Custom pattern with angle-dependent gain
    Custom(Vec<(f32, f32)>), // (angle_degrees, gain_db)
}

/// DSP processing features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DspFeature {
    /// Room correction EQ
    RoomCorrection,
    /// Dynamic range compression
    Compression,
    /// Parametric EQ
    ParametricEQ,
    /// Bass management
    BassManagement,
    /// Crossover filtering
    Crossover,
    /// Time alignment
    TimeAlignment,
    /// Beamforming
    Beamforming,
}

/// Supported audio formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFormat {
    /// PCM uncompressed
    PCM {
        /// Audio sample rate (Hz)
        sample_rate: u32,
        /// Bit depth for audio samples
        bit_depth: u16,
    },
    /// FLAC lossless
    FLAC,
    /// AAC compressed
    AAC {
        /// Target bitrate in kbps
        bitrate_kbps: u32,
    },
    /// Opus low-latency
    Opus {
        /// Target bitrate in kbps
        bitrate_kbps: u32,
    },
    /// Custom format
    Custom(String),
}

/// Network protocol types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkProtocol {
    /// WiFi 802.11
    WiFi,
    /// Ethernet
    Ethernet,
    /// Bluetooth
    Bluetooth,
    /// AirPlay
    AirPlay,
    /// Chromecast
    Chromecast,
    /// Sonos proprietary
    Sonos,
    /// Custom protocol
    Custom(String),
}

/// Room correction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomCorrection {
    /// Frequency response measurements
    pub frequency_response: Vec<(f32, f32)>, // (frequency_hz, magnitude_db)
    /// Room impulse response
    pub impulse_response: Vec<f32>,
    /// EQ filter coefficients
    pub eq_filters: Vec<EQFilter>,
    /// Measurement position
    pub measurement_position: Position3D,
}

/// EQ filter coefficient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EQFilter {
    /// Filter type
    pub filter_type: FilterType,
    /// Center frequency (Hz)
    pub frequency: f32,
    /// Q factor (bandwidth)
    pub q_factor: f32,
    /// Gain (dB)
    pub gain_db: f32,
}

/// EQ filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// Low pass filter
    LowPass,
    /// High pass filter
    HighPass,
    /// Band pass filter
    BandPass,
    /// Band stop/notch filter
    BandStop,
    /// Peaking filter
    Peaking,
    /// Low shelf filter
    LowShelf,
    /// High shelf filter
    HighShelf,
}

/// Smart speaker array configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerArrayConfig {
    /// Array name/identifier
    pub name: String,
    /// Room dimensions (width, height, depth in meters)
    pub room_dimensions: (f32, f32, f32),
    /// Listening position (sweet spot)
    pub listening_position: Position3D,
    /// Array topology
    pub topology: ArrayTopology,
    /// Synchronization settings
    pub sync_config: SyncConfig,
    /// Audio processing settings
    pub processing_config: ProcessingConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
}

/// Array topology configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrayTopology {
    /// Stereo pair
    Stereo {
        /// Distance between left and right speakers (meters)
        separation_m: f32,
    },
    /// 5.1 surround sound
    Surround5_1,
    /// 7.1 surround sound
    Surround7_1,
    /// Dolby Atmos with height channels
    Atmos {
        /// Number of height/ceiling speakers
        height_speakers: u8,
    },
    /// Distributed array (many speakers)
    Distributed {
        /// Minimum number of speakers
        min_speakers: u8,
        /// Maximum number of speakers
        max_speakers: u8,
    },
    /// Line array
    LineArray {
        /// Spacing between speakers in the line (meters)
        speaker_spacing_m: f32,
    },
    /// Circular array
    CircularArray {
        /// Radius of the circular array (meters)
        radius_m: f32,
    },
    /// Custom arrangement
    Custom,
}

/// Synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Master clock source
    pub clock_source: ClockSource,
    /// Sync tolerance (microseconds)
    pub sync_tolerance_us: u32,
    /// Buffer size for sync (samples)
    pub sync_buffer_size: usize,
    /// Network jitter compensation
    pub jitter_compensation: bool,
    /// Automatic sync correction
    pub auto_correction: bool,
}

/// Clock source for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClockSource {
    /// Network Time Protocol
    NTP,
    /// Precision Time Protocol (IEEE 1588)
    PTP,
    /// Internal system clock
    SystemClock,
    /// Audio hardware clock
    AudioClock,
    /// External word clock
    WordClock,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Crossover frequencies (Hz)
    pub crossover_frequencies: Vec<f32>,
    /// Time alignment delays (ms per speaker)
    pub time_alignment: HashMap<String, f32>,
    /// Individual speaker EQ
    pub speaker_eq: HashMap<String, Vec<EQFilter>>,
    /// Dynamic range compression
    pub compression: CompressionConfig,
    /// Limiting settings
    pub limiting: LimitingConfig,
    /// Room correction enable
    pub room_correction_enabled: bool,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Threshold (dB)
    pub threshold_db: f32,
    /// Ratio (e.g., 4.0 for 4:1)
    pub ratio: f32,
    /// Attack time (ms)
    pub attack_ms: f32,
    /// Release time (ms)
    pub release_ms: f32,
    /// Makeup gain (dB)
    pub makeup_gain_db: f32,
}

/// Limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitingConfig {
    /// Enable limiting
    pub enabled: bool,
    /// Ceiling level (dB)
    pub ceiling_db: f32,
    /// Release time (ms)
    pub release_ms: f32,
    /// Lookahead time (ms)
    pub lookahead_ms: f32,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Multicast group for audio streaming
    pub multicast_group: String,
    /// Base port for audio streams
    pub base_port: u16,
    /// Quality of Service (QoS) priority
    pub qos_priority: u8,
    /// Maximum network latency (ms)
    pub max_latency_ms: f32,
    /// Packet size (bytes)
    pub packet_size: usize,
    /// Buffer size (packets)
    pub buffer_size: usize,
}

/// Smart speaker array manager
#[derive(Debug)]
pub struct SpeakerArrayManager {
    /// Discovered speakers
    speakers: HashMap<String, SmartSpeaker>,
    /// Array configurations
    arrays: HashMap<String, SpeakerArrayConfig>,
    /// Discovery service
    discovery: DiscoveryService,
    /// Calibration engine
    calibration: CalibrationEngine,
    /// Audio router
    router: AudioRouter,
    /// Performance metrics
    metrics: ArrayMetrics,
}

/// Speaker discovery service
#[derive(Debug)]
pub struct DiscoveryService {
    /// Discovery protocols enabled
    protocols: Vec<DiscoveryProtocol>,
    /// Discovery interval (seconds)
    discovery_interval: u64,
    /// Auto-add discovered speakers
    auto_add: bool,
    /// Device filters
    device_filters: Vec<DeviceFilter>,
}

/// Discovery protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    /// UPnP/DLNA discovery
    UPnP,
    /// Bonjour/mDNS
    Bonjour,
    /// Chromecast discovery
    Chromecast,
    /// AirPlay discovery
    AirPlay,
    /// Sonos discovery
    Sonos,
    /// Manual IP scanning
    IPScan {
        /// Starting IP address for scan
        start_ip: String,
        /// Ending IP address for scan
        end_ip: String,
    },
}

/// Device filtering criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFilter {
    /// Manufacturer name
    pub manufacturer: Option<String>,
    /// Model name pattern
    pub model_pattern: Option<String>,
    /// Minimum capabilities
    pub min_capabilities: Option<SpeakerCapabilities>,
    /// Network requirements
    pub network_requirements: Option<NetworkRequirements>,
}

/// Network requirements for speakers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    /// Minimum bandwidth (Mbps)
    pub min_bandwidth_mbps: f32,
    /// Maximum latency (ms)
    pub max_latency_ms: f32,
    /// Required protocols
    pub required_protocols: Vec<NetworkProtocol>,
}

/// Discovered device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    /// Device identifier
    pub id: String,
    /// Device name
    pub name: String,
    /// Manufacturer
    pub manufacturer: String,
    /// Model name
    pub model: String,
    /// IP address
    pub ip_address: String,
    /// MAC address
    pub mac_address: String,
    /// Network protocol
    pub protocol: NetworkProtocol,
    /// Device capabilities
    pub capabilities: SpeakerCapabilities,
    /// Available services
    pub services: Vec<String>,
}

/// Automatic calibration engine
#[derive(Debug)]
pub struct CalibrationEngine {
    /// Calibration methods
    methods: Vec<CalibrationMethod>,
    /// Test signal generator
    signal_generator: TestSignalGenerator,
    /// Measurement analyzer
    analyzer: MeasurementAnalyzer,
    /// Optimization engine
    optimizer: ArrayOptimizer,
}

/// Calibration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Sweep tone calibration
    SweepTone {
        /// Starting frequency (Hz)
        start_hz: f32,
        /// Ending frequency (Hz)
        end_hz: f32,
        /// Duration of sweep (seconds)
        duration_s: f32,
    },
    /// White noise calibration
    WhiteNoise {
        /// Duration of white noise (seconds)
        duration_s: f32,
    },
    /// Pink noise calibration
    PinkNoise {
        /// Duration of pink noise (seconds)
        duration_s: f32,
    },
    /// Maximum Length Sequence (MLS)
    MLS {
        /// Length of MLS sequence
        length: usize,
    },
    /// Chirp signal
    Chirp {
        /// Starting frequency (Hz)
        start_hz: f32,
        /// Ending frequency (Hz)
        end_hz: f32,
        /// Duration of chirp (seconds)
        duration_s: f32,
    },
}

/// Test signal generator
#[derive(Debug)]
pub struct TestSignalGenerator {
    /// Sample rate
    sample_rate: u32,
    /// Bit depth
    bit_depth: u16,
    /// Signal level (dB)
    signal_level_db: f32,
}

/// Measurement analyzer
#[derive(Debug)]
pub struct MeasurementAnalyzer {
    /// FFT size for analysis
    fft_size: usize,
    /// Window function
    window_function: WindowFunction,
    /// Smoothing factor
    smoothing_factor: f32,
}

/// Window functions for FFT analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowFunction {
    /// Rectangular window
    Rectangular,
    /// Hanning window
    Hanning,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser {
        /// Kaiser window beta parameter
        beta: f32,
    },
}

/// Array optimization engine
#[derive(Debug)]
pub struct ArrayOptimizer {
    /// Optimization goals
    goals: Vec<OptimizationGoal>,
    /// Constraints
    constraints: Vec<OptimizationConstraint>,
    /// Optimization algorithm
    algorithm: OptimizationAlgorithm,
}

/// Optimization goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    /// Maximize frequency response flatness
    FlatFrequencyResponse,
    /// Maximize sweet spot size
    MaximizeSweetSpot,
    /// Minimize inter-speaker delays
    MinimizeDelays,
    /// Maximize dynamic range
    MaximizeDynamicRange,
    /// Minimize power consumption
    MinimizePower,
    /// Custom optimization function
    Custom(String),
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationConstraint {
    /// Maximum delay constraint (ms)
    MaxDelay(f32),
    /// Frequency response limits (dB)
    FrequencyResponseLimits {
        /// Minimum decibel level
        min_db: f32,
        /// Maximum decibel level
        max_db: f32,
    },
    /// Speaker power limits (watts)
    PowerLimits(f32),
    /// Phase coherence constraint
    PhaseCoherence {
        /// Maximum phase error in degrees
        max_phase_error_deg: f32,
    },
}

/// Optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Least squares optimization
    LeastSquares,
    /// Genetic algorithm
    GeneticAlgorithm {
        /// Size of genetic algorithm population
        population_size: usize,
        /// Number of generations to run
        generations: usize,
    },
    /// Simulated annealing
    SimulatedAnnealing {
        /// Initial temperature for annealing
        initial_temp: f32,
        /// Rate of temperature cooling
        cooling_rate: f32,
    },
    /// Particle swarm optimization
    ParticleSwarm {
        /// Number of particles in swarm
        particles: usize,
        /// Number of optimization iterations
        iterations: usize,
    },
}

/// Audio routing engine
#[derive(Debug)]
pub struct AudioRouter {
    /// Active routes
    routes: HashMap<String, AudioRoute>,
    /// Routing matrix
    matrix: RoutingMatrix,
    /// Stream manager
    stream_manager: StreamManager,
}

/// Audio route definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRoute {
    /// Route ID
    pub id: String,
    /// Source input
    pub source: AudioSource,
    /// Destination speakers
    pub destinations: Vec<String>,
    /// Processing chain
    pub processing: Vec<ProcessingStep>,
    /// Mix settings
    pub mix_settings: MixSettings,
}

/// Audio source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioSource {
    /// File playback
    File {
        /// Path to the audio file
        path: String,
    },
    /// Network stream
    Stream {
        /// URL of the network stream
        url: String,
    },
    /// Microphone input
    Microphone {
        /// Device identifier for the microphone
        device_id: String,
    },
    /// Line input
    LineIn {
        /// Audio channel number
        channel: u8,
    },
    /// Bluetooth source
    Bluetooth {
        /// Bluetooth device identifier
        device_id: String,
    },
    /// AirPlay source
    AirPlay,
    /// Chromecast source
    Chromecast,
}

/// Audio processing steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStep {
    /// Volume adjustment
    Volume {
        /// Gain adjustment in decibels
        gain_db: f32,
    },
    /// EQ filtering
    EQ {
        /// List of EQ filters to apply
        filters: Vec<EQFilter>,
    },
    /// Delay
    Delay {
        /// Delay time in milliseconds
        delay_ms: f32,
    },
    /// Compression
    Compression(CompressionConfig),
    /// Limiting
    Limiting(LimitingConfig),
    /// Spatial upmix
    SpatialUpmix {
        /// Target number of output channels
        target_channels: u8,
    },
    /// Custom DSP
    CustomDSP {
        /// Unique identifier for the DSP plugin
        plugin_id: String,
        /// Plugin parameter values
        parameters: HashMap<String, f32>,
    },
}

/// Mix settings for routes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixSettings {
    /// Mix level (0.0-1.0)
    pub level: f32,
    /// Pan position (-1.0 to 1.0, 0.0 = center)
    pub pan: f32,
    /// Mute state
    pub muted: bool,
    /// Solo state
    pub solo: bool,
    /// Crossover assignment
    pub crossover_assignment: Option<CrossoverAssignment>,
}

/// Crossover assignment for frequency splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverAssignment {
    /// Low frequency speakers
    pub low_freq_speakers: Vec<String>,
    /// Mid frequency speakers
    pub mid_freq_speakers: Vec<String>,
    /// High frequency speakers
    pub high_freq_speakers: Vec<String>,
    /// Crossover frequencies (Hz)
    pub crossover_frequencies: Vec<f32>,
}

/// Routing matrix for audio distribution
#[derive(Debug)]
pub struct RoutingMatrix {
    /// Input to output mapping
    matrix: Vec<Vec<f32>>, // [input][output] = gain
    /// Input count
    input_count: usize,
    /// Output count  
    output_count: usize,
}

/// Stream management
#[derive(Debug)]
pub struct StreamManager {
    /// Active streams
    streams: HashMap<String, AudioStream>,
    /// Stream statistics
    stats: StreamStats,
    /// Buffer management
    buffer_manager: BufferManager,
}

/// Audio stream information
#[derive(Debug)]
pub struct AudioStream {
    /// Stream ID
    pub id: String,
    /// Stream format
    pub format: AudioFormat,
    /// Target speakers
    pub speakers: Vec<String>,
    /// Stream state
    pub state: StreamState,
    /// Quality metrics
    pub metrics: StreamMetrics,
}

/// Stream states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamState {
    /// Starting up
    Starting,
    /// Running normally
    Running,
    /// Buffering
    Buffering,
    /// Paused
    Paused,
    /// Stopped
    Stopped,
    /// Error state
    Error(String),
}

/// Stream quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    /// Bitrate (kbps)
    pub bitrate_kbps: f32,
    /// Packet loss rate (%)
    pub packet_loss_rate: f32,
    /// Jitter (ms)
    pub jitter_ms: f32,
    /// Buffer level (%)
    pub buffer_level: f32,
    /// Audio dropouts count
    pub dropouts: u32,
}

/// Stream statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    /// Total streams handled
    pub total_streams: u64,
    /// Active streams
    pub active_streams: u32,
    /// Total data transferred (bytes)
    pub total_bytes: u64,
    /// Average bitrate (kbps)
    pub avg_bitrate_kbps: f32,
}

/// Buffer management
#[derive(Debug)]
pub struct BufferManager {
    /// Buffer pools per speaker
    pools: HashMap<String, Vec<AudioBuffer>>,
    /// Buffer statistics
    stats: BufferStats,
}

/// Audio buffer
#[derive(Debug)]
pub struct AudioBuffer {
    /// Buffer data
    pub data: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Channel count
    pub channels: u8,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Buffer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    /// Buffer underruns
    pub underruns: u32,
    /// Buffer overruns
    pub overruns: u32,
    /// Average buffer level (%)
    pub avg_buffer_level: f32,
    /// Peak buffer usage (%)
    pub peak_buffer_usage: f32,
}

/// Array performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetrics {
    /// Overall system latency (ms)
    pub system_latency_ms: f32,
    /// Network utilization (%)
    pub network_utilization: f32,
    /// CPU usage (%)
    pub cpu_usage: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// Active speaker count
    pub active_speakers: u32,
    /// Audio quality score (0-100)
    pub audio_quality_score: f32,
    /// Sync accuracy (microseconds RMS)
    pub sync_accuracy_us: f32,
}

impl SpeakerArrayManager {
    /// Create a new speaker array manager
    pub fn new() -> Self {
        Self {
            speakers: HashMap::new(),
            arrays: HashMap::new(),
            discovery: DiscoveryService::new(),
            calibration: CalibrationEngine::new(),
            router: AudioRouter::new(),
            metrics: ArrayMetrics::default(),
        }
    }

    /// Start speaker discovery
    pub async fn start_discovery(&mut self) -> Result<()> {
        info!("Starting speaker discovery");
        self.discovery.start().await?;
        Ok(())
    }

    /// Add a speaker manually
    pub fn add_speaker(&mut self, speaker: SmartSpeaker) -> Result<()> {
        info!("Adding speaker: {}", speaker.id);
        self.speakers.insert(speaker.id.clone(), speaker);
        Ok(())
    }

    /// Remove a speaker
    pub fn remove_speaker(&mut self, speaker_id: &str) -> Result<()> {
        if self.speakers.remove(speaker_id).is_some() {
            info!("Removed speaker: {}", speaker_id);
            Ok(())
        } else {
            Err(Error::processing("Speaker not found"))
        }
    }

    /// Create a new speaker array
    pub fn create_array(&mut self, config: SpeakerArrayConfig) -> Result<()> {
        info!("Creating speaker array: {}", config.name);

        // Validate speakers exist
        for speaker_id in self.get_required_speakers(&config)? {
            if !self.speakers.contains_key(&speaker_id) {
                return Err(Error::config(&format!("Speaker {speaker_id} not found")));
            }
        }

        self.arrays.insert(config.name.clone(), config);
        Ok(())
    }

    /// Start array calibration
    pub async fn calibrate_array(&mut self, array_name: &str) -> Result<CalibrationResults> {
        let array = self
            .arrays
            .get(array_name)
            .ok_or_else(|| Error::config("Array not found"))?;

        info!("Starting calibration for array: {}", array_name);
        self.calibration
            .calibrate_array(array, &self.speakers)
            .await
    }

    /// Start audio routing
    pub async fn start_routing(&mut self, routes: Vec<AudioRoute>) -> Result<()> {
        for route in routes {
            self.router.add_route(route).await?;
        }
        Ok(())
    }

    /// Get array metrics
    pub fn get_metrics(&self) -> &ArrayMetrics {
        &self.metrics
    }

    /// Update metrics
    pub fn update_metrics(&mut self) {
        self.metrics = self.calculate_metrics();
    }

    /// Get list of discovered speakers
    pub fn get_speakers(&self) -> &HashMap<String, SmartSpeaker> {
        &self.speakers
    }

    /// Get array configuration
    pub fn get_array(&self, name: &str) -> Option<&SpeakerArrayConfig> {
        self.arrays.get(name)
    }

    fn get_required_speakers(&self, config: &SpeakerArrayConfig) -> Result<Vec<String>> {
        // This would determine which speakers are needed based on the topology
        match &config.topology {
            ArrayTopology::Stereo { .. } => Ok(vec!["left".to_string(), "right".to_string()]),
            ArrayTopology::Surround5_1 => Ok(vec![
                "front_left".to_string(),
                "front_right".to_string(),
                "center".to_string(),
                "lfe".to_string(),
                "rear_left".to_string(),
                "rear_right".to_string(),
            ]),
            _ => Ok(self.speakers.keys().cloned().collect()),
        }
    }

    fn calculate_metrics(&self) -> ArrayMetrics {
        ArrayMetrics {
            system_latency_ms: 50.0, // Placeholder
            network_utilization: 25.0,
            cpu_usage: 15.0,
            memory_usage_mb: 128.0,
            active_speakers: self.speakers.len() as u32,
            audio_quality_score: 85.0,
            sync_accuracy_us: 100.0,
        }
    }
}

impl Default for SpeakerArrayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoveryService {
    fn new() -> Self {
        Self {
            protocols: vec![DiscoveryProtocol::UPnP, DiscoveryProtocol::Bonjour],
            discovery_interval: 30,
            auto_add: false,
            device_filters: Vec::new(),
        }
    }

    async fn start(&self) -> Result<()> {
        info!(
            "Starting discovery service with {} protocols",
            self.protocols.len()
        );

        for protocol in &self.protocols {
            match self.discover_with_protocol(protocol).await {
                Ok(devices) => {
                    info!("Discovered {} devices with {:?}", devices.len(), protocol);
                }
                Err(e) => {
                    warn!("Discovery failed for {:?}: {}", protocol, e);
                }
            }
        }

        Ok(())
    }

    async fn discover_with_protocol(
        &self,
        protocol: &DiscoveryProtocol,
    ) -> Result<Vec<DiscoveredDevice>> {
        match protocol {
            DiscoveryProtocol::UPnP => self.discover_upnp().await,
            DiscoveryProtocol::Bonjour => self.discover_bonjour().await,
            DiscoveryProtocol::Chromecast => self.discover_chromecast().await,
            DiscoveryProtocol::AirPlay => self.discover_airplay().await,
            DiscoveryProtocol::Sonos => self.discover_sonos().await,
            DiscoveryProtocol::IPScan { start_ip, end_ip } => {
                self.discover_ip_scan(start_ip, end_ip).await
            }
        }
    }

    async fn discover_upnp(&self) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting UPnP discovery");

        // UPnP SSDP multicast discovery
        let devices = vec![
            DiscoveredDevice {
                id: "upnp_speaker_1".to_string(),
                name: "Living Room Speaker".to_string(),
                manufacturer: "Generic UPnP".to_string(),
                model: "Smart Speaker".to_string(),
                ip_address: "192.168.1.101".to_string(),
                mac_address: "AA:BB:CC:DD:EE:01".to_string(),
                protocol: NetworkProtocol::WiFi,
                capabilities: self.create_default_capabilities(),
                services: vec!["MediaRenderer".to_string(), "AudioControl".to_string()],
            },
            DiscoveredDevice {
                id: "upnp_speaker_2".to_string(),
                name: "Kitchen Speaker".to_string(),
                manufacturer: "Generic UPnP".to_string(),
                model: "Smart Speaker Pro".to_string(),
                ip_address: "192.168.1.102".to_string(),
                mac_address: "AA:BB:CC:DD:EE:02".to_string(),
                protocol: NetworkProtocol::WiFi,
                capabilities: self.create_enhanced_capabilities(),
                services: vec![
                    "MediaRenderer".to_string(),
                    "AudioControl".to_string(),
                    "RoomCorrection".to_string(),
                ],
            },
        ];

        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(devices)
    }

    async fn discover_bonjour(&self) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting Bonjour/mDNS discovery");

        // Bonjour service discovery for _audio._tcp
        let devices = vec![DiscoveredDevice {
            id: "bonjour_speaker_1".to_string(),
            name: "Bedroom Speaker".to_string(),
            manufacturer: "Apple".to_string(),
            model: "HomePod mini".to_string(),
            ip_address: "192.168.1.103".to_string(),
            mac_address: "AA:BB:CC:DD:EE:03".to_string(),
            protocol: NetworkProtocol::AirPlay,
            capabilities: self.create_airplay_capabilities(),
            services: vec!["AirPlay".to_string(), "HomeKit".to_string()],
        }];

        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

        Ok(devices)
    }

    async fn discover_chromecast(&self) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting Chromecast discovery");

        let devices = vec![DiscoveredDevice {
            id: "chromecast_audio_1".to_string(),
            name: "Office Audio".to_string(),
            manufacturer: "Google".to_string(),
            model: "Chromecast Audio".to_string(),
            ip_address: "192.168.1.104".to_string(),
            mac_address: "AA:BB:CC:DD:EE:04".to_string(),
            protocol: NetworkProtocol::Chromecast,
            capabilities: self.create_chromecast_capabilities(),
            services: vec!["Cast".to_string(), "GoogleCast".to_string()],
        }];

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        Ok(devices)
    }

    async fn discover_airplay(&self) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting AirPlay discovery");

        let devices = vec![DiscoveredDevice {
            id: "airplay_speaker_1".to_string(),
            name: "Studio Monitor".to_string(),
            manufacturer: "Apple".to_string(),
            model: "HomePod".to_string(),
            ip_address: "192.168.1.105".to_string(),
            mac_address: "AA:BB:CC:DD:EE:05".to_string(),
            protocol: NetworkProtocol::AirPlay,
            capabilities: self.create_airplay_capabilities(),
            services: vec!["AirPlay2".to_string(), "Siri".to_string()],
        }];

        tokio::time::sleep(tokio::time::Duration::from_millis(180)).await;

        Ok(devices)
    }

    async fn discover_sonos(&self) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting Sonos discovery");

        let devices = vec![DiscoveredDevice {
            id: "sonos_speaker_1".to_string(),
            name: "Sonos One".to_string(),
            manufacturer: "Sonos".to_string(),
            model: "One SL".to_string(),
            ip_address: "192.168.1.106".to_string(),
            mac_address: "AA:BB:CC:DD:EE:06".to_string(),
            protocol: NetworkProtocol::Sonos,
            capabilities: self.create_sonos_capabilities(),
            services: vec!["SonosZone".to_string(), "GroupManagement".to_string()],
        }];

        tokio::time::sleep(tokio::time::Duration::from_millis(120)).await;

        Ok(devices)
    }

    async fn discover_ip_scan(
        &self,
        start_ip: &str,
        end_ip: &str,
    ) -> Result<Vec<DiscoveredDevice>> {
        info!("Starting IP scan from {} to {}", start_ip, end_ip);

        // Parse IP range and scan
        let start_parts: Vec<&str> = start_ip.split('.').collect();
        let end_parts: Vec<&str> = end_ip.split('.').collect();

        if start_parts.len() != 4 || end_parts.len() != 4 {
            return Err(Error::config("Invalid IP range format"));
        }

        let start_last: u8 = start_parts[3]
            .parse()
            .map_err(|_| Error::config("Invalid start IP"))?;
        let end_last: u8 = end_parts[3]
            .parse()
            .map_err(|_| Error::config("Invalid end IP"))?;
        let base_ip = format!("{}.{}.{}", start_parts[0], start_parts[1], start_parts[2]);

        let mut devices = Vec::new();

        for ip_last in start_last..=end_last {
            let ip = format!("{base_ip}.{ip_last}");

            // Simulate ping and port scan
            if self.ping_and_scan(&ip).await? {
                devices.push(DiscoveredDevice {
                    id: format!("ip_speaker_{ip_last}"),
                    name: format!("Audio Device {ip_last}"),
                    manufacturer: "Unknown".to_string(),
                    model: "Generic Audio Device".to_string(),
                    ip_address: ip,
                    mac_address: format!("FF:FF:FF:FF:FF:{ip_last:02X}"),
                    protocol: NetworkProtocol::Custom("TCP".to_string()),
                    capabilities: self.create_default_capabilities(),
                    services: vec!["Audio".to_string()],
                });
            }

            // Small delay to avoid overwhelming the network
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(devices)
    }

    async fn ping_and_scan(&self, ip: &str) -> Result<bool> {
        // Simulate ping and audio port scan
        // In real implementation, this would ping the IP and scan for audio service ports
        let hash = ip.chars().map(|c| c as u32).sum::<u32>();
        let is_responsive = (hash % 7) == 0; // Simulate some devices being responsive

        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        Ok(is_responsive)
    }

    fn create_default_capabilities(&self) -> SpeakerCapabilities {
        SpeakerCapabilities {
            frequency_range: (50.0, 18000.0),
            max_spl: 95.0,
            driver_count: 2,
            directivity: DirectivityPattern::Omnidirectional,
            dsp_features: vec![DspFeature::ParametricEQ],
            supported_formats: vec![
                AudioFormat::PCM {
                    sample_rate: 48000,
                    bit_depth: 16,
                },
                AudioFormat::AAC { bitrate_kbps: 256 },
            ],
        }
    }

    fn create_enhanced_capabilities(&self) -> SpeakerCapabilities {
        SpeakerCapabilities {
            frequency_range: (40.0, 20000.0),
            max_spl: 105.0,
            driver_count: 3,
            directivity: DirectivityPattern::Cardioid,
            dsp_features: vec![
                DspFeature::RoomCorrection,
                DspFeature::ParametricEQ,
                DspFeature::Compression,
                DspFeature::TimeAlignment,
            ],
            supported_formats: vec![
                AudioFormat::PCM {
                    sample_rate: 96000,
                    bit_depth: 24,
                },
                AudioFormat::FLAC,
                AudioFormat::AAC { bitrate_kbps: 320 },
            ],
        }
    }

    fn create_airplay_capabilities(&self) -> SpeakerCapabilities {
        SpeakerCapabilities {
            frequency_range: (45.0, 22000.0),
            max_spl: 100.0,
            driver_count: 4,
            directivity: DirectivityPattern::Custom(vec![
                (0.0, 0.0),
                (45.0, -1.0),
                (90.0, -3.0),
                (135.0, -6.0),
                (180.0, -10.0),
                (225.0, -6.0),
                (270.0, -3.0),
                (315.0, -1.0),
            ]),
            dsp_features: vec![
                DspFeature::RoomCorrection,
                DspFeature::BassManagement,
                DspFeature::Beamforming,
            ],
            supported_formats: vec![
                AudioFormat::PCM {
                    sample_rate: 48000,
                    bit_depth: 24,
                },
                AudioFormat::AAC { bitrate_kbps: 256 },
            ],
        }
    }

    fn create_chromecast_capabilities(&self) -> SpeakerCapabilities {
        SpeakerCapabilities {
            frequency_range: (60.0, 16000.0),
            max_spl: 90.0,
            driver_count: 1,
            directivity: DirectivityPattern::Omnidirectional,
            dsp_features: vec![DspFeature::Compression],
            supported_formats: vec![
                AudioFormat::AAC { bitrate_kbps: 128 },
                AudioFormat::Opus { bitrate_kbps: 96 },
            ],
        }
    }

    fn create_sonos_capabilities(&self) -> SpeakerCapabilities {
        SpeakerCapabilities {
            frequency_range: (50.0, 20000.0),
            max_spl: 98.0,
            driver_count: 2,
            directivity: DirectivityPattern::Bidirectional,
            dsp_features: vec![
                DspFeature::RoomCorrection,
                DspFeature::ParametricEQ,
                DspFeature::BassManagement,
                DspFeature::TimeAlignment,
            ],
            supported_formats: vec![
                AudioFormat::PCM {
                    sample_rate: 48000,
                    bit_depth: 16,
                },
                AudioFormat::FLAC,
                AudioFormat::AAC { bitrate_kbps: 320 },
            ],
        }
    }
}

impl CalibrationEngine {
    fn new() -> Self {
        Self {
            methods: vec![
                CalibrationMethod::SweepTone {
                    start_hz: 20.0,
                    end_hz: 20000.0,
                    duration_s: 10.0,
                },
                CalibrationMethod::PinkNoise { duration_s: 5.0 },
            ],
            signal_generator: TestSignalGenerator {
                sample_rate: 48000,
                bit_depth: 24,
                signal_level_db: -20.0,
            },
            analyzer: MeasurementAnalyzer {
                fft_size: 8192,
                window_function: WindowFunction::Hanning,
                smoothing_factor: 0.125,
            },
            optimizer: ArrayOptimizer {
                goals: vec![
                    OptimizationGoal::FlatFrequencyResponse,
                    OptimizationGoal::MaximizeSweetSpot,
                ],
                constraints: vec![
                    OptimizationConstraint::MaxDelay(50.0),
                    OptimizationConstraint::FrequencyResponseLimits {
                        min_db: -6.0,
                        max_db: 6.0,
                    },
                ],
                algorithm: OptimizationAlgorithm::LeastSquares,
            },
        }
    }

    async fn calibrate_array(
        &self,
        array: &SpeakerArrayConfig,
        speakers: &HashMap<String, SmartSpeaker>,
    ) -> Result<CalibrationResults> {
        info!("Calibrating array: {}", array.name);

        // Implementation would:
        // 1. Generate test signals
        // 2. Play through each speaker
        // 3. Measure response
        // 4. Analyze and optimize

        Ok(CalibrationResults {
            array_name: array.name.clone(),
            calibration_quality: 0.95,
            speaker_delays: HashMap::new(),
            eq_settings: HashMap::new(),
            room_correction: None,
            calibrated_at: chrono::Utc::now(),
        })
    }
}

impl AudioRouter {
    fn new() -> Self {
        Self {
            routes: HashMap::new(),
            matrix: RoutingMatrix::new(8, 8), // 8x8 matrix
            stream_manager: StreamManager::new(),
        }
    }

    async fn add_route(&mut self, route: AudioRoute) -> Result<()> {
        info!("Adding audio route: {}", route.id);
        self.routes.insert(route.id.clone(), route);
        Ok(())
    }
}

impl RoutingMatrix {
    fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            matrix: vec![vec![0.0; outputs]; inputs],
            input_count: inputs,
            output_count: outputs,
        }
    }
}

impl StreamManager {
    fn new() -> Self {
        Self {
            streams: HashMap::new(),
            stats: StreamStats::default(),
            buffer_manager: BufferManager::new(),
        }
    }
}

impl BufferManager {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
            stats: BufferStats::default(),
        }
    }
}

impl Default for ArrayMetrics {
    fn default() -> Self {
        Self {
            system_latency_ms: 0.0,
            network_utilization: 0.0,
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            active_speakers: 0,
            audio_quality_score: 100.0,
            sync_accuracy_us: 0.0,
        }
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            total_streams: 0,
            active_streams: 0,
            total_bytes: 0,
            avg_bitrate_kbps: 0.0,
        }
    }
}

impl Default for BufferStats {
    fn default() -> Self {
        Self {
            underruns: 0,
            overruns: 0,
            avg_buffer_level: 50.0,
            peak_buffer_usage: 0.0,
        }
    }
}

/// Calibration results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResults {
    /// Array name
    pub array_name: String,
    /// Calibration quality score (0.0-1.0)
    pub calibration_quality: f32,
    /// Speaker delay corrections (ms)
    pub speaker_delays: HashMap<String, f32>,
    /// EQ settings per speaker
    pub eq_settings: HashMap<String, Vec<EQFilter>>,
    /// Room correction data
    pub room_correction: Option<RoomCorrection>,
    /// Calibration timestamp
    pub calibrated_at: chrono::DateTime<chrono::Utc>,
}

/// Builder for speaker array configuration
#[derive(Debug, Default)]
pub struct SpeakerArrayConfigBuilder {
    name: Option<String>,
    room_dimensions: Option<(f32, f32, f32)>,
    listening_position: Option<Position3D>,
    topology: Option<ArrayTopology>,
    sync_config: Option<SyncConfig>,
    processing_config: Option<ProcessingConfig>,
    network_config: Option<NetworkConfig>,
}

impl SpeakerArrayConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set array name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set room dimensions
    pub fn room_dimensions(mut self, width: f32, height: f32, depth: f32) -> Self {
        self.room_dimensions = Some((width, height, depth));
        self
    }

    /// Set listening position
    pub fn listening_position(mut self, position: Position3D) -> Self {
        self.listening_position = Some(position);
        self
    }

    /// Set array topology
    pub fn topology(mut self, topology: ArrayTopology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Set synchronization config
    pub fn sync_config(mut self, config: SyncConfig) -> Self {
        self.sync_config = Some(config);
        self
    }

    /// Set processing config
    pub fn processing_config(mut self, config: ProcessingConfig) -> Self {
        self.processing_config = Some(config);
        self
    }

    /// Set network config
    pub fn network_config(mut self, config: NetworkConfig) -> Self {
        self.network_config = Some(config);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<SpeakerArrayConfig> {
        Ok(SpeakerArrayConfig {
            name: self
                .name
                .ok_or_else(|| Error::config("Array name required"))?,
            room_dimensions: self.room_dimensions.unwrap_or((5.0, 3.0, 4.0)),
            listening_position: self
                .listening_position
                .unwrap_or(Position3D::new(0.0, 0.0, 0.0)),
            topology: self
                .topology
                .unwrap_or(ArrayTopology::Stereo { separation_m: 2.0 }),
            sync_config: self.sync_config.unwrap_or(SyncConfig {
                clock_source: ClockSource::NTP,
                sync_tolerance_us: 100,
                sync_buffer_size: 512,
                jitter_compensation: true,
                auto_correction: true,
            }),
            processing_config: self.processing_config.unwrap_or_else(|| ProcessingConfig {
                crossover_frequencies: vec![80.0, 2500.0],
                time_alignment: HashMap::new(),
                speaker_eq: HashMap::new(),
                compression: CompressionConfig {
                    enabled: false,
                    threshold_db: -20.0,
                    ratio: 3.0,
                    attack_ms: 10.0,
                    release_ms: 100.0,
                    makeup_gain_db: 0.0,
                },
                limiting: LimitingConfig {
                    enabled: true,
                    ceiling_db: -0.5,
                    release_ms: 50.0,
                    lookahead_ms: 5.0,
                },
                room_correction_enabled: true,
            }),
            network_config: self.network_config.unwrap_or_else(|| NetworkConfig {
                multicast_group: "239.255.77.77".to_string(),
                base_port: 5004,
                qos_priority: 7,
                max_latency_ms: 50.0,
                packet_size: 1316,
                buffer_size: 8,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_array_manager_creation() {
        let manager = SpeakerArrayManager::new();
        assert_eq!(manager.speakers.len(), 0);
        assert_eq!(manager.arrays.len(), 0);
    }

    #[test]
    fn test_speaker_addition() {
        let mut manager = SpeakerArrayManager::new();

        let speaker = SmartSpeaker {
            id: "test_speaker".to_string(),
            position: Position3D::new(1.0, 0.0, 0.0),
            capabilities: SpeakerCapabilities {
                frequency_range: (40.0, 20000.0),
                max_spl: 105.0,
                driver_count: 2,
                directivity: DirectivityPattern::Omnidirectional,
                dsp_features: vec![DspFeature::RoomCorrection],
                supported_formats: vec![AudioFormat::PCM {
                    sample_rate: 48000,
                    bit_depth: 24,
                }],
            },
            network_info: NetworkInfo {
                ip_address: "192.168.1.100".to_string(),
                mac_address: "00:11:22:33:44:55".to_string(),
                protocol: NetworkProtocol::WiFi,
                signal_strength: 85,
                latency_ms: 10.0,
                bandwidth_mbps: 100.0,
            },
            calibration: CalibrationStatus {
                is_calibrated: false,
                calibrated_at: None,
                room_correction: None,
                inter_speaker_distances: HashMap::new(),
                delay_compensation_ms: 0.0,
            },
            audio_specs: AudioSpecs {
                sample_rate: 48000,
                bit_depth: 24,
                channels: 2,
                buffer_size: 512,
                codec_latency_ms: 5.0,
            },
        };

        manager.add_speaker(speaker).unwrap();
        assert_eq!(manager.speakers.len(), 1);
    }

    #[test]
    fn test_array_config_builder() {
        let config = SpeakerArrayConfigBuilder::new()
            .name("test_array")
            .room_dimensions(5.0, 3.0, 4.0)
            .topology(ArrayTopology::Stereo { separation_m: 2.0 })
            .build()
            .unwrap();

        assert_eq!(config.name, "test_array");
        assert_eq!(config.room_dimensions, (5.0, 3.0, 4.0));
        match config.topology {
            ArrayTopology::Stereo { separation_m } => assert_eq!(separation_m, 2.0),
            _ => panic!("Wrong topology"),
        }
    }

    #[test]
    fn test_directivity_pattern_serialization() {
        let pattern = DirectivityPattern::Custom(vec![(0.0, 0.0), (90.0, -3.0), (180.0, -20.0)]);
        let serialized = serde_json::to_string(&pattern).unwrap();
        let deserialized: DirectivityPattern = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            DirectivityPattern::Custom(angles) => {
                assert_eq!(angles.len(), 3);
                assert_eq!(angles[0], (0.0, 0.0));
            }
            _ => panic!("Wrong pattern type"),
        }
    }

    #[test]
    fn test_eq_filter_creation() {
        let filter = EQFilter {
            filter_type: FilterType::Peaking,
            frequency: 1000.0,
            q_factor: 0.7,
            gain_db: 3.0,
        };

        assert_eq!(filter.frequency, 1000.0);
        assert_eq!(filter.gain_db, 3.0);
    }

    #[test]
    fn test_audio_format_variants() {
        let formats = vec![
            AudioFormat::PCM {
                sample_rate: 48000,
                bit_depth: 24,
            },
            AudioFormat::FLAC,
            AudioFormat::AAC { bitrate_kbps: 320 },
            AudioFormat::Opus { bitrate_kbps: 128 },
        ];

        assert_eq!(formats.len(), 4);

        match &formats[0] {
            AudioFormat::PCM {
                sample_rate,
                bit_depth,
            } => {
                assert_eq!(*sample_rate, 48000);
                assert_eq!(*bit_depth, 24);
            }
            _ => panic!("Wrong format type"),
        }
    }

    #[test]
    fn test_calibration_results() {
        let results = CalibrationResults {
            array_name: "test_array".to_string(),
            calibration_quality: 0.95,
            speaker_delays: HashMap::new(),
            eq_settings: HashMap::new(),
            room_correction: None,
            calibrated_at: chrono::Utc::now(),
        };

        assert_eq!(results.array_name, "test_array");
        assert_eq!(results.calibration_quality, 0.95);
    }

    #[test]
    fn test_routing_matrix_creation() {
        let matrix = RoutingMatrix::new(4, 8);
        assert_eq!(matrix.input_count, 4);
        assert_eq!(matrix.output_count, 8);
        assert_eq!(matrix.matrix.len(), 4);
        assert_eq!(matrix.matrix[0].len(), 8);
    }

    #[test]
    fn test_stream_metrics() {
        let metrics = StreamMetrics {
            bitrate_kbps: 1411.0,
            packet_loss_rate: 0.01,
            jitter_ms: 2.0,
            buffer_level: 75.0,
            dropouts: 0,
        };

        assert_eq!(metrics.bitrate_kbps, 1411.0);
        assert_eq!(metrics.packet_loss_rate, 0.01);
    }

    #[test]
    fn test_array_metrics_default() {
        let metrics = ArrayMetrics::default();
        assert_eq!(metrics.audio_quality_score, 100.0);
        assert_eq!(metrics.active_speakers, 0);
    }

    #[test]
    fn test_discovery_protocols() {
        let protocols = vec![
            DiscoveryProtocol::UPnP,
            DiscoveryProtocol::Bonjour,
            DiscoveryProtocol::Chromecast,
            DiscoveryProtocol::AirPlay,
        ];

        assert_eq!(protocols.len(), 4);
    }

    #[test]
    fn test_optimization_goals() {
        let goals = vec![
            OptimizationGoal::FlatFrequencyResponse,
            OptimizationGoal::MaximizeSweetSpot,
            OptimizationGoal::MinimizeDelays,
        ];

        assert_eq!(goals.len(), 3);
    }
}
