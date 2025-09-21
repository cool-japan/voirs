//! Streaming configuration for real-time audio processing

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Stream protocol
    pub protocol: StreamProtocol,
    /// Stream format
    pub format: StreamFormat,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
}

/// Stream protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamProtocol {
    /// TCP stream
    TCP,
    /// UDP stream
    UDP,
    /// WebSocket stream
    WebSocket,
    /// HTTP stream
    HTTP,
    /// Custom protocol
    Custom(String),
}

/// Stream formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamFormat {
    /// Raw PCM
    RawPCM,
    /// WAV format
    WAV,
    /// MP3 format
    MP3,
    /// Opus format
    Opus,
    /// Custom format
    Custom(String),
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
    /// Bitrate (for lossy compression)
    pub bitrate: Option<u32>,
    /// Quality setting
    pub quality: Option<f32>,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// FLAC compression
    FLAC,
    /// MP3 compression
    MP3,
    /// Opus compression
    Opus,
    /// Custom compression
    Custom(String),
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Buffer size for network operations
    pub buffer_size: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Exponential backoff
    pub exponential_backoff: bool,
    /// Jitter
    pub jitter: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            protocol: StreamProtocol::TCP,
            format: StreamFormat::RawPCM,
            compression: CompressionConfig::default(),
            network_config: NetworkConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::None,
            level: 5,
            bitrate: None,
            quality: None,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            connection_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(5),
            write_timeout: Duration::from_secs(5),
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            retry_delay: Duration::from_millis(100),
            exponential_backoff: true,
            jitter: true,
        }
    }
}
