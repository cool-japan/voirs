//! Processing configuration for real-time audio processing

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::error::ErrorHandlingConfig;
use super::quality::QualityControlConfig;
use super::streaming::StreamingConfig;

/// Processing configuration for real-time operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Processing chain
    pub processing_chain: Vec<ProcessingStage>,
    /// Parallel processing configuration
    pub parallel_config: ParallelProcessingConfig,
    /// Quality control settings
    pub quality_control: QualityControlConfig,
    /// Error handling configuration
    pub error_handling: ErrorHandlingConfig,
    /// Streaming configuration
    pub streaming_config: StreamingConfig,
}

/// Processing stages in the real-time pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    /// Pre-processing stage
    PreProcessing {
        /// Noise reduction
        noise_reduction: bool,
        /// Gain control
        gain_control: bool,
        /// High-pass filtering
        high_pass_filter: Option<f32>,
        /// Low-pass filtering
        low_pass_filter: Option<f32>,
    },
    /// Analysis stage
    Analysis {
        /// Spectral analysis
        spectral_analysis: bool,
        /// Pitch detection
        pitch_detection: bool,
        /// Voice activity detection
        voice_activity_detection: bool,
        /// Quality metrics computation
        quality_metrics: bool,
    },
    /// Enhancement stage
    Enhancement {
        /// Noise suppression
        noise_suppression: bool,
        /// Echo cancellation
        echo_cancellation: bool,
        /// Dynamic range compression
        dynamic_range_compression: bool,
        /// Equalization
        equalization: Option<EqualizationConfig>,
    },
    /// Post-processing stage
    PostProcessing {
        /// Normalization
        normalization: bool,
        /// Fade in/out
        fade_effects: bool,
        /// Output formatting
        output_formatting: bool,
    },
}

/// Equalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualizationConfig {
    /// Equalizer type
    pub eq_type: EqualizerType,
    /// Frequency bands
    pub frequency_bands: Vec<FrequencyBand>,
    /// Automatic gain control
    pub automatic_gain_control: bool,
}

/// Equalizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EqualizerType {
    /// Parametric equalizer
    Parametric,
    /// Graphic equalizer
    Graphic,
    /// Shelving equalizer
    Shelving,
    /// Notch filter
    Notch,
}

/// Frequency band configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBand {
    /// Center frequency (Hz)
    pub center_frequency: f32,
    /// Bandwidth (Hz)
    pub bandwidth: f32,
    /// Gain (dB)
    pub gain: f32,
    /// Q factor
    pub q_factor: f32,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessingConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Work distribution strategy
    pub distribution_strategy: WorkDistributionStrategy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Thread priority
    pub thread_priority: ThreadPriority,
}

/// Work distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkDistributionStrategy {
    /// Round robin distribution
    RoundRobin,
    /// Load-based distribution
    LoadBased,
    /// Frequency-based distribution
    FrequencyBased,
    /// Channel-based distribution
    ChannelBased,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub dynamic_balancing: bool,
    /// Load monitoring interval
    pub monitoring_interval: Duration,
    /// Load threshold for rebalancing
    pub rebalancing_threshold: f32,
    /// Migration strategy
    pub migration_strategy: MigrationStrategy,
}

/// Migration strategies for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Immediate migration
    Immediate,
    /// Gradual migration
    Gradual,
    /// Batch migration
    Batch,
    /// No migration
    None,
}

/// Thread priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    RealTime,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            processing_chain: vec![
                ProcessingStage::PreProcessing {
                    noise_reduction: true,
                    gain_control: true,
                    high_pass_filter: Some(80.0),
                    low_pass_filter: Some(8000.0),
                },
                ProcessingStage::Analysis {
                    spectral_analysis: true,
                    pitch_detection: true,
                    voice_activity_detection: true,
                    quality_metrics: true,
                },
            ],
            parallel_config: ParallelProcessingConfig::default(),
            quality_control: QualityControlConfig::default(),
            error_handling: ErrorHandlingConfig::default(),
            streaming_config: StreamingConfig::default(),
        }
    }
}

impl Default for ParallelProcessingConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            distribution_strategy: WorkDistributionStrategy::LoadBased,
            load_balancing: LoadBalancingConfig::default(),
            thread_priority: ThreadPriority::High,
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            dynamic_balancing: true,
            monitoring_interval: Duration::from_millis(100),
            rebalancing_threshold: 0.8,
            migration_strategy: MigrationStrategy::Gradual,
        }
    }
}
