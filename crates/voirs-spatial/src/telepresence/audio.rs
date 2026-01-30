//! Audio device configuration and voice processing types

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime};

/// Audio settings for telepresence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelepresenceAudioSettings {
    /// Input device configuration
    pub input_device: AudioDeviceConfig,

    /// Output device configuration
    pub output_device: AudioDeviceConfig,

    /// Voice processing settings
    pub voice_processing: VoiceProcessingSettings,

    /// Audio quality preferences
    pub quality_preferences: AudioQualityPreferences,

    /// Codec preferences
    pub codec_preferences: CodecPreferences,
}

/// Audio device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDeviceConfig {
    /// Device identifier
    pub device_id: Option<String>,

    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Buffer size (samples)
    pub buffer_size: usize,

    /// Channel count
    pub channels: u8,

    /// Bit depth
    pub bit_depth: u8,

    /// Device-specific settings
    pub device_settings: HashMap<String, String>,
}

/// Voice processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProcessingSettings {
    /// Automatic gain control
    pub agc_enabled: bool,

    /// Noise suppression
    pub noise_suppression: NoiseSuppressionSettings,

    /// Echo cancellation
    pub echo_cancellation: EchoCancellationSettings,

    /// Voice activity detection
    pub vad_settings: VadSettings,

    /// Audio enhancement
    pub enhancement: AudioEnhancementSettings,

    /// Spatialization settings
    pub spatialization: VoiceSpatializationSettings,
}

/// Noise suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSuppressionSettings {
    /// Enable noise suppression
    pub enabled: bool,

    /// Suppression strength (0.0-1.0)
    pub strength: f32,

    /// Suppression algorithm
    pub algorithm: NoiseSuppressionAlgorithm,

    /// Adaptive learning
    pub adaptive: bool,

    /// Stationary noise suppression
    pub stationary_suppression: f32,

    /// Non-stationary noise suppression
    pub non_stationary_suppression: f32,
}

/// Echo cancellation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoCancellationSettings {
    /// Enable echo cancellation
    pub enabled: bool,

    /// Cancellation strength (0.0-1.0)
    pub strength: f32,

    /// Echo cancellation algorithm
    pub algorithm: EchoCancellationAlgorithm,

    /// Tail length (samples)
    pub tail_length: usize,

    /// Adaptation rate
    pub adaptation_rate: f32,

    /// Non-linear processing
    pub non_linear_processing: bool,
}

/// Voice Activity Detection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadSettings {
    /// Enable VAD
    pub enabled: bool,

    /// Detection sensitivity (0.0-1.0)
    pub sensitivity: f32,

    /// VAD algorithm
    pub algorithm: VadAlgorithm,

    /// Minimum voice duration (ms)
    pub min_voice_duration: f32,

    /// Minimum silence duration (ms)
    pub min_silence_duration: f32,

    /// Hangover time (ms)
    pub hangover_time: f32,
}

/// Audio enhancement settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEnhancementSettings {
    /// Enable enhancement
    pub enabled: bool,

    /// Dynamic range compression
    pub dynamic_range_compression: CompressionSettings,

    /// Equalization
    pub equalization: EqualizationSettings,

    /// Bandwidth extension
    pub bandwidth_extension: BandwidthExtensionSettings,

    /// Comfort noise generation
    pub comfort_noise: ComfortNoiseSettings,
}

/// Dynamic range compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,

    /// Compression ratio
    pub ratio: f32,

    /// Threshold (dB)
    pub threshold: f32,

    /// Attack time (ms)
    pub attack_time: f32,

    /// Release time (ms)
    pub release_time: f32,

    /// Makeup gain (dB)
    pub makeup_gain: f32,
}

/// Equalization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualizationSettings {
    /// Enable EQ
    pub enabled: bool,

    /// EQ bands
    pub bands: Vec<EqBand>,

    /// EQ type
    pub eq_type: EqualizationType,

    /// Adaptive EQ
    pub adaptive: bool,
}

/// EQ band configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqBand {
    /// Center frequency (Hz)
    pub frequency: f32,

    /// Gain (dB)
    pub gain: f32,

    /// Q factor
    pub q_factor: f32,

    /// Band type
    pub band_type: EqBandType,
}

/// Bandwidth extension settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthExtensionSettings {
    /// Enable bandwidth extension
    pub enabled: bool,

    /// Target bandwidth (Hz)
    pub target_bandwidth: f32,

    /// Extension algorithm
    pub algorithm: BandwidthExtensionAlgorithm,

    /// Extension strength
    pub strength: f32,
}

/// Comfort noise settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortNoiseSettings {
    /// Enable comfort noise
    pub enabled: bool,

    /// Noise level (dB)
    pub level: f32,

    /// Noise color
    pub color: NoiseColor,

    /// Adaptive level
    pub adaptive_level: bool,
}

/// Voice spatialization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSpatializationSettings {
    /// Enable spatialization
    pub enabled: bool,

    /// HRTF personalization
    pub hrtf_personalization: HrtfPersonalizationSettings,

    /// Room simulation
    pub room_simulation: super::room::RoomSimulationSettings,

    /// Distance modeling
    pub distance_modeling: super::room::DistanceModelingSettings,

    /// Doppler effects
    pub doppler_effects: super::room::DopplerEffectsSettings,
}

/// HRTF personalization for voice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfPersonalizationSettings {
    /// Enable personalization
    pub enabled: bool,

    /// User measurements
    pub measurements: Option<UserMeasurements>,

    /// Personalization method
    pub method: PersonalizationMethod,

    /// Adaptation strength
    pub adaptation_strength: f32,
}

/// User physical measurements for HRTF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMeasurements {
    /// Head circumference (cm)
    pub head_circumference: f32,

    /// Pinna length (cm)
    pub pinna_length: f32,

    /// Pinna width (cm)
    pub pinna_width: f32,

    /// Torso width (cm)
    pub torso_width: f32,

    /// Custom measurements
    pub custom_measurements: HashMap<String, f32>,
}

/// Audio quality preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityPreferences {
    /// Preferred quality level
    pub quality_level: QualityLevel,

    /// Adaptive quality
    pub adaptive_quality: bool,

    /// Latency priority
    pub latency_priority: LatencyPriority,

    /// Bandwidth constraints
    pub bandwidth_constraints: BandwidthConstraints,
}

/// Bandwidth constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConstraints {
    /// Maximum bandwidth (kbps)
    pub max_bandwidth: u32,

    /// Minimum bandwidth (kbps)
    pub min_bandwidth: u32,

    /// Adaptive bandwidth
    pub adaptive: bool,

    /// Bandwidth measurement interval (ms)
    pub measurement_interval: u32,
}

/// Codec preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecPreferences {
    /// Preferred codecs in priority order
    pub preferred_codecs: Vec<AudioCodec>,

    /// Codec-specific settings
    pub codec_settings: HashMap<AudioCodec, CodecSettings>,

    /// Fallback behavior
    pub fallback_behavior: CodecFallbackBehavior,
}

/// Codec-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecSettings {
    /// Bitrate (kbps)
    pub bitrate: u32,

    /// Complexity level
    pub complexity: u8,

    /// Variable bitrate
    pub variable_bitrate: bool,

    /// Forward error correction
    pub fec: bool,

    /// Codec-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Frequency filter settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyFilterSettings {
    /// High-pass cutoff (Hz)
    pub highpass_cutoff: f32,

    /// Low-pass cutoff (Hz)
    pub lowpass_cutoff: f32,

    /// Notch filters
    pub notch_filters: Vec<NotchFilter>,
}

/// Notch filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotchFilter {
    /// Center frequency (Hz)
    pub frequency: f32,

    /// Q factor
    pub q_factor: f32,

    /// Attenuation (dB)
    pub attenuation: f32,
}

/// Frequency response adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyResponseAdjustment {
    /// Low frequency boost/cut (dB)
    pub low_freq_adjustment: f32,

    /// Mid frequency boost/cut (dB)
    pub mid_freq_adjustment: f32,

    /// High frequency boost/cut (dB)
    pub high_freq_adjustment: f32,
}

/// Audio format information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Codec used
    pub codec: AudioCodec,

    /// Sample rate
    pub sample_rate: u32,

    /// Channels
    pub channels: u8,

    /// Bitrate
    pub bitrate: u32,

    /// Frame size
    pub frame_size: usize,
}

/// Audio metadata for transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Sequence number
    pub sequence: u64,

    /// Audio format
    pub format: AudioFormat,

    /// Spatial information
    pub spatial_info: super::spatial::SpatialAudioInfo,

    /// Quality information
    pub quality_info: super::quality::QualityInfo,
}

/// Received audio data
#[derive(Debug, Clone)]
pub struct ReceivedAudio {
    /// User identifier
    pub user_id: String,

    /// Audio samples
    pub samples: Vec<f32>,

    /// Metadata
    pub metadata: AudioMetadata,

    /// Reception timestamp
    pub received_at: Instant,

    /// Processing status
    pub processing_status: ProcessingStatus,
}

/// Audio quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualitySettings {
    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Bit depth
    pub bit_depth: u8,

    /// Channel configuration
    pub channels: ChannelConfiguration,

    /// Dynamic range (dB)
    pub dynamic_range: f32,

    /// THD+N specification
    pub thd_n: f32,
}

/// Audio quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityStats {
    /// Average quality score
    pub avg_quality: f32,

    /// Minimum quality
    pub min_quality: f32,

    /// Maximum quality
    pub max_quality: f32,

    /// Quality adaptations count
    pub adaptations: u32,

    /// Audio dropouts
    pub dropouts: u32,

    /// Compression efficiency
    pub compression_ratio: f32,
}

// Default implementations

impl Default for TelepresenceAudioSettings {
    fn default() -> Self {
        Self {
            input_device: AudioDeviceConfig::default(),
            output_device: AudioDeviceConfig::default(),
            voice_processing: VoiceProcessingSettings::default(),
            quality_preferences: AudioQualityPreferences::default(),
            codec_preferences: CodecPreferences::default(),
        }
    }
}

impl Default for AudioDeviceConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            sample_rate: 48000,
            buffer_size: 1024,
            channels: 2,
            bit_depth: 16,
            device_settings: HashMap::new(),
        }
    }
}

impl Default for VoiceProcessingSettings {
    fn default() -> Self {
        Self {
            agc_enabled: true,
            noise_suppression: NoiseSuppressionSettings::default(),
            echo_cancellation: EchoCancellationSettings::default(),
            vad_settings: VadSettings::default(),
            enhancement: AudioEnhancementSettings::default(),
            spatialization: VoiceSpatializationSettings::default(),
        }
    }
}

impl Default for NoiseSuppressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 0.7,
            algorithm: NoiseSuppressionAlgorithm::Hybrid,
            adaptive: true,
            stationary_suppression: 0.8,
            non_stationary_suppression: 0.6,
        }
    }
}

impl Default for EchoCancellationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 0.8,
            algorithm: EchoCancellationAlgorithm::NLMS,
            tail_length: 1024,
            adaptation_rate: 0.01,
            non_linear_processing: true,
        }
    }
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.7,
            algorithm: VadAlgorithm::Hybrid,
            min_voice_duration: 100.0,
            min_silence_duration: 200.0,
            hangover_time: 150.0,
        }
    }
}

impl Default for AudioEnhancementSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            dynamic_range_compression: CompressionSettings::default(),
            equalization: EqualizationSettings::default(),
            bandwidth_extension: BandwidthExtensionSettings::default(),
            comfort_noise: ComfortNoiseSettings::default(),
        }
    }
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            ratio: 3.0,
            threshold: -18.0,
            attack_time: 5.0,
            release_time: 50.0,
            makeup_gain: 2.0,
        }
    }
}

impl Default for EqualizationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            bands: vec![],
            eq_type: EqualizationType::Parametric,
            adaptive: false,
        }
    }
}

impl Default for BandwidthExtensionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            target_bandwidth: 20000.0,
            algorithm: BandwidthExtensionAlgorithm::SpectralReplication,
            strength: 0.5,
        }
    }
}

impl Default for ComfortNoiseSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            level: -40.0,
            color: NoiseColor::Pink,
            adaptive_level: true,
        }
    }
}

impl Default for VoiceSpatializationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            hrtf_personalization: HrtfPersonalizationSettings::default(),
            room_simulation: super::room::RoomSimulationSettings::default(),
            distance_modeling: super::room::DistanceModelingSettings::default(),
            doppler_effects: super::room::DopplerEffectsSettings::default(),
        }
    }
}

impl Default for HrtfPersonalizationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            measurements: None,
            method: PersonalizationMethod::Anthropometric,
            adaptation_strength: 0.5,
        }
    }
}

impl Default for AudioQualityPreferences {
    fn default() -> Self {
        Self {
            quality_level: QualityLevel::High,
            adaptive_quality: true,
            latency_priority: LatencyPriority::Medium,
            bandwidth_constraints: BandwidthConstraints::default(),
        }
    }
}

impl Default for BandwidthConstraints {
    fn default() -> Self {
        Self {
            max_bandwidth: 320, // 320 kbps
            min_bandwidth: 32,  // 32 kbps
            adaptive: true,
            measurement_interval: 1000, // 1 second
        }
    }
}

impl Default for CodecPreferences {
    fn default() -> Self {
        let mut codec_settings = HashMap::new();
        codec_settings.insert(
            AudioCodec::Opus,
            CodecSettings {
                bitrate: 128,
                complexity: 8,
                variable_bitrate: true,
                fec: true,
                parameters: HashMap::new(),
            },
        );

        Self {
            preferred_codecs: vec![AudioCodec::Opus, AudioCodec::AAC, AudioCodec::G722],
            codec_settings,
            fallback_behavior: CodecFallbackBehavior::NextPreferred,
        }
    }
}

impl Default for FrequencyFilterSettings {
    fn default() -> Self {
        Self {
            highpass_cutoff: 100.0, // 100 Hz
            lowpass_cutoff: 8000.0, // 8 kHz
            notch_filters: vec![],
        }
    }
}

impl Default for FrequencyResponseAdjustment {
    fn default() -> Self {
        Self {
            low_freq_adjustment: 0.0,
            mid_freq_adjustment: 1.0,   // Slight mid boost for intimacy
            high_freq_adjustment: -1.0, // Slight high cut for warmth
        }
    }
}

impl Default for AudioQualitySettings {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            bit_depth: 16,
            channels: ChannelConfiguration::Binaural,
            dynamic_range: 96.0, // 96 dB
            thd_n: 0.01,         // 0.01% THD+N
        }
    }
}
