//! REST API types and request/response structures.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Standard API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub request_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            request_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            request_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Speech recognition request
#[derive(Debug, Clone, Deserialize)]
pub struct RecognitionRequest {
    /// Audio data (base64 encoded)
    pub audio_data: Option<String>,
    /// Audio file URL (alternative to audio_data)
    pub audio_url: Option<String>,
    /// Audio format specification
    pub audio_format: Option<AudioFormatRequest>,
    /// Recognition configuration
    pub config: Option<RecognitionConfigRequest>,
    /// Whether to return detailed segments
    pub include_segments: Option<bool>,
    /// Whether to include confidence scores
    pub include_confidence: Option<bool>,
    /// Whether to include timestamps
    pub include_timestamps: Option<bool>,
}

/// Audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormatRequest {
    /// Sample rate (Hz)
    pub sample_rate: Option<u32>,
    /// Number of channels
    pub channels: Option<u16>,
    /// Bits per sample
    pub bits_per_sample: Option<u16>,
    /// Audio format type ("wav", "mp3", "flac", "ogg", "m4a")
    pub format: Option<String>,
}

/// Recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfigRequest {
    /// Model name to use
    pub model: Option<String>,
    /// Language code ("en", "es", "fr", etc.)
    pub language: Option<String>,
    /// Enable voice activity detection
    pub enable_vad: Option<bool>,
    /// Confidence threshold (0.0 to 1.0)
    pub confidence_threshold: Option<f32>,
    /// Beam size for decoding
    pub beam_size: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f32>,
    /// Whether to suppress blank tokens
    pub suppress_blank: Option<bool>,
    /// List of token IDs to suppress
    pub suppress_tokens: Option<Vec<u32>>,
}

impl Default for RecognitionConfigRequest {
    fn default() -> Self {
        Self {
            model: None,
            language: None,
            enable_vad: None,
            confidence_threshold: None,
            beam_size: None,
            temperature: None,
            suppress_blank: None,
            suppress_tokens: None,
        }
    }
}

/// Speech recognition response
#[derive(Debug, Serialize)]
pub struct RecognitionResponse {
    /// Recognized text
    pub text: String,
    /// Overall confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Detected language (if auto-detection was used)
    pub detected_language: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Audio duration in seconds
    pub audio_duration_s: f64,
    /// Number of segments
    pub segment_count: usize,
    /// Detailed segments (if requested)
    pub segments: Option<Vec<SegmentResponse>>,
    /// Audio metadata
    pub audio_metadata: AudioMetadataResponse,
    /// Recognition metadata
    pub metadata: RecognitionMetadataResponse,
}

/// Individual speech segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentResponse {
    /// Segment start time in seconds
    pub start_time: f64,
    /// Segment end time in seconds
    pub end_time: f64,
    /// Segment text
    pub text: String,
    /// Segment confidence score
    pub confidence: f32,
    /// No speech probability
    pub no_speech_prob: f32,
    /// Token-level information (if available)
    pub tokens: Option<Vec<TokenResponse>>,
}

/// Token-level information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: String,
    /// Token probability
    pub probability: f32,
    /// Token start time
    pub start_time: Option<f64>,
    /// Token end time
    pub end_time: Option<f64>,
}

/// Audio metadata
#[derive(Debug, Serialize)]
pub struct AudioMetadataResponse {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f64,
    /// Audio format
    pub format: String,
    /// File size in bytes
    pub size_bytes: usize,
    /// Bit rate (if applicable)
    pub bit_rate: Option<u32>,
}

/// Recognition metadata
#[derive(Debug, Serialize)]
pub struct RecognitionMetadataResponse {
    /// Model used for recognition
    pub model: String,
    /// Language code used
    pub language: String,
    /// VAD enabled
    pub vad_enabled: bool,
    /// Beam size used
    pub beam_size: usize,
    /// Temperature used
    pub temperature: f32,
    /// Processing statistics
    pub processing_stats: ProcessingStatsResponse,
}

/// Processing statistics
#[derive(Debug, Serialize)]
pub struct ProcessingStatsResponse {
    /// Real-time factor
    pub real_time_factor: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: Option<f32>,
    /// GPU usage percentage (if applicable)
    pub gpu_usage_percent: Option<f32>,
}

/// Streaming recognition request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingRecognitionRequest {
    /// Streaming configuration
    pub config: Option<StreamingConfigRequest>,
    /// Recognition configuration
    pub recognition_config: Option<RecognitionConfigRequest>,
    /// Audio format
    pub audio_format: Option<AudioFormatRequest>,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfigRequest {
    /// Chunk duration in seconds
    pub chunk_duration: Option<f32>,
    /// Overlap duration in seconds
    pub overlap_duration: Option<f32>,
    /// VAD threshold
    pub vad_threshold: Option<f32>,
    /// Silence duration threshold
    pub silence_duration: Option<f32>,
    /// Maximum chunk size
    pub max_chunk_size: Option<usize>,
    /// Enable interim results
    pub enable_interim_results: Option<bool>,
}

/// Streaming recognition response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingRecognitionResponse {
    /// Session ID
    pub session_id: String,
    /// Whether this is an interim result
    pub is_interim: bool,
    /// Whether this is the final result
    pub is_final: bool,
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Segment information
    pub segment: Option<SegmentResponse>,
    /// Processing time for this chunk
    pub processing_time_ms: f64,
    /// Chunk sequence number
    pub sequence_number: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
    /// Uptime in seconds
    pub uptime_seconds: f64,
    /// Memory usage information
    pub memory_usage: MemoryUsageResponse,
    /// Model status
    pub model_status: ModelStatusResponse,
    /// Performance metrics
    pub performance_metrics: PerformanceMetricsResponse,
}

/// Memory usage information
#[derive(Debug, Serialize)]
pub struct MemoryUsageResponse {
    /// Used memory in MB
    pub used_mb: f64,
    /// Available memory in MB
    pub available_mb: f64,
    /// Cache size in MB
    pub cache_size_mb: f64,
    /// Memory usage percentage
    pub usage_percent: f32,
}

/// Model status information
#[derive(Debug, Serialize)]
pub struct ModelStatusResponse {
    /// Currently loaded models
    pub loaded_models: Vec<ModelInfoResponse>,
    /// Model loading status
    pub loading_status: String,
    /// Current default model
    pub default_model: Option<String>,
    /// Supported models
    pub supported_models: Vec<String>,
    /// Supported languages
    pub supported_languages: Vec<String>,
}

/// Individual model information
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: String,
    /// Model size in MB
    pub size_mb: f64,
    /// Whether the model is currently loaded
    pub is_loaded: bool,
    /// Supported languages for this model
    pub languages: Vec<String>,
    /// Model version
    pub version: Option<String>,
}

/// Performance metrics
#[derive(Debug, Serialize)]
pub struct PerformanceMetricsResponse {
    /// Total recognitions performed
    pub total_recognitions: u64,
    /// Total audio duration processed (seconds)
    pub total_audio_duration: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Real-time factor
    pub real_time_factor: f32,
    /// Active streaming sessions
    pub active_sessions: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Error rate
    pub error_rate: f32,
}

/// Model management request
#[derive(Debug, Deserialize)]
pub struct ModelManagementRequest {
    /// Action to perform ("load", "unload", "switch")
    pub action: String,
    /// Model name
    pub model_name: String,
    /// Additional parameters
    pub parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Model management response
#[derive(Debug, Serialize)]
pub struct ModelManagementResponse {
    /// Action performed
    pub action: String,
    /// Model name
    pub model_name: String,
    /// Whether the action was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Time taken for the action (ms)
    pub time_taken_ms: f64,
}

/// Batch recognition request
#[derive(Debug, Deserialize)]
pub struct BatchRecognitionRequest {
    /// List of audio files or data
    pub inputs: Vec<RecognitionRequest>,
    /// Batch configuration
    pub config: Option<BatchConfigRequest>,
    /// Whether to process in parallel
    pub parallel: Option<bool>,
    /// Maximum number of concurrent jobs
    pub max_concurrency: Option<usize>,
}

/// Batch configuration
#[derive(Debug, Deserialize)]
pub struct BatchConfigRequest {
    /// Default recognition config for all inputs
    pub default_config: Option<RecognitionConfigRequest>,
    /// Whether to continue on errors
    pub continue_on_error: Option<bool>,
    /// Timeout per job (seconds)
    pub timeout_per_job: Option<u64>,
    /// Priority level
    pub priority: Option<String>,
}

/// Batch recognition response
#[derive(Debug, Serialize)]
pub struct BatchRecognitionResponse {
    /// Batch job ID
    pub batch_id: String,
    /// Individual results
    pub results: Vec<BatchResultResponse>,
    /// Batch statistics
    pub statistics: BatchStatisticsResponse,
    /// Overall status
    pub status: String,
}

/// Individual batch result
#[derive(Debug, Serialize)]
pub struct BatchResultResponse {
    /// Input index
    pub index: usize,
    /// Whether this result is successful
    pub success: bool,
    /// Recognition result (if successful)
    pub result: Option<RecognitionResponse>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Processing time for this input
    pub processing_time_ms: f64,
}

/// Batch processing statistics
#[derive(Debug, Serialize)]
pub struct BatchStatisticsResponse {
    /// Total inputs
    pub total_inputs: usize,
    /// Successful recognitions
    pub successful: usize,
    /// Failed recognitions
    pub failed: usize,
    /// Total processing time
    pub total_processing_time_ms: f64,
    /// Average processing time per input
    pub avg_processing_time_ms: f64,
    /// Batch start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Batch end time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Additional details
    pub details: Option<HashMap<String, serde_json::Value>>,
    /// Request ID for tracking
    pub request_id: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// WebSocket message types
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Start streaming session
    StartStreaming {
        session_id: String,
        config: StreamingRecognitionRequest,
    },
    /// Audio chunk data
    AudioChunk {
        session_id: String,
        chunk_data: String, // base64 encoded
        sequence_number: u64,
    },
    /// Stop streaming session
    StopStreaming { session_id: String },
    /// Recognition result
    RecognitionResult {
        session_id: String,
        result: StreamingRecognitionResponse,
    },
    /// Error message
    Error {
        session_id: Option<String>,
        error: ErrorResponse,
    },
    /// Session status
    SessionStatus {
        session_id: String,
        status: String,
        message: Option<String>,
    },
}
