//! C API types and structures for VoiRS speech recognition.

use std::ffi::c_char;

/// Opaque handle to a VoiRS recognizer instance
#[repr(C)]
pub struct VoirsRecognizer {
    _private: [u8; 0],
}

/// Recognition result structure for C API
#[repr(C)]
pub struct VoirsRecognitionResult {
    /// Recognized text (null-terminated C string)
    pub text: *const c_char,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Detected language code (null-terminated C string, may be null)
    pub language: *const c_char,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Audio duration in seconds
    pub audio_duration_s: f64,
    /// Number of segments
    pub segment_count: usize,
    /// Array of segments
    pub segments: *const VoirsSegment,
}

/// Speech segment structure for C API
#[repr(C)]
#[derive(Clone)]
pub struct VoirsSegment {
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Segment text (null-terminated C string)
    pub text: *const c_char,
    /// Segment confidence (0.0 to 1.0)
    pub confidence: f32,
    /// No speech probability (0.0 to 1.0)
    pub no_speech_prob: f32,
}

/// Configuration structure for recognition
#[repr(C)]
pub struct VoirsRecognitionConfig {
    /// Model name (null-terminated C string)
    pub model_name: *const c_char,
    /// Language code (null-terminated C string, may be null for auto-detection)
    pub language: *const c_char,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Enable voice activity detection
    pub enable_vad: bool,
    /// Confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f32,
    /// Beam size for decoding
    pub beam_size: usize,
    /// Temperature for sampling (0.0 to 1.0)
    pub temperature: f32,
    /// Suppress blank tokens
    pub suppress_blank: bool,
}

/// Streaming configuration structure
#[repr(C)]
#[derive(Clone)]
pub struct VoirsStreamingConfig {
    /// Chunk duration in seconds
    pub chunk_duration: f32,
    /// Overlap duration in seconds
    pub overlap_duration: f32,
    /// VAD threshold
    pub vad_threshold: f32,
    /// Silence duration threshold
    pub silence_duration: f32,
    /// Maximum chunk size in samples
    pub max_chunk_size: usize,
}

/// Audio format information
#[repr(C)]
pub struct VoirsAudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Audio format type
    pub format: VoirsAudioFormatType,
}

/// Audio format types
#[repr(C)]
pub enum VoirsAudioFormatType {
    /// 16-bit signed PCM
    PCM16 = 0,
    /// 32-bit signed PCM
    PCM32 = 1,
    /// 32-bit floating point
    Float32 = 2,
    /// WAV format
    WAV = 3,
    /// MP3 format
    MP3 = 4,
    /// FLAC format
    FLAC = 5,
    /// OGG format
    OGG = 6,
    /// M4A format
    M4A = 7,
}

/// Error codes for C API
#[repr(C)]
pub enum VoirsError {
    /// Success (no error)
    Success = 0,
    /// Invalid argument
    InvalidArgument = 1,
    /// Null pointer
    NullPointer = 2,
    /// Initialization failed
    InitializationFailed = 3,
    /// Model loading failed
    ModelLoadFailed = 4,
    /// Recognition failed
    RecognitionFailed = 5,
    /// Audio format not supported
    UnsupportedFormat = 6,
    /// Out of memory
    OutOfMemory = 7,
    /// Internal error
    InternalError = 8,
    /// Streaming not started
    StreamingNotStarted = 9,
    /// Invalid configuration
    InvalidConfiguration = 10,
}

/// Callback function type for streaming recognition results
pub type VoirsStreamingCallback =
    extern "C" fn(result: *const VoirsRecognitionResult, user_data: *mut std::ffi::c_void);

/// Callback function type for progress updates
pub type VoirsProgressCallback =
    extern "C" fn(progress: f32, message: *const c_char, user_data: *mut std::ffi::c_void);

/// Version information structure
#[repr(C)]
pub struct VoirsVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Version string (null-terminated)
    pub version_string: *const c_char,
    /// Build timestamp (null-terminated)
    pub build_timestamp: *const c_char,
}

// SAFETY: VoirsVersion contains only static string pointers that are valid for the program's lifetime
unsafe impl Send for VoirsVersion {}
unsafe impl Sync for VoirsVersion {}

/// Capability flags
#[repr(C)]
pub struct VoirsCapabilities {
    /// Support for streaming recognition
    pub streaming: bool,
    /// Support for multiple languages
    pub multilingual: bool,
    /// Support for voice activity detection
    pub vad: bool,
    /// Support for confidence scoring
    pub confidence_scoring: bool,
    /// Support for segment timestamps
    pub segment_timestamps: bool,
    /// Support for language detection
    pub language_detection: bool,
    /// Number of supported models
    pub supported_models_count: usize,
    /// Array of supported model names
    pub supported_models: *const *const c_char,
    /// Number of supported languages
    pub supported_languages_count: usize,
    /// Array of supported language codes
    pub supported_languages: *const *const c_char,
}

/// Performance metrics structure
#[repr(C)]
pub struct VoirsPerformanceMetrics {
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,
    /// Average processing time per second of audio (ms)
    pub avg_processing_time_ms: f32,
    /// Peak processing time (ms)
    pub peak_processing_time_ms: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of processed chunks
    pub processed_chunks: usize,
    /// Number of failed recognitions
    pub failed_recognitions: usize,
}

/// Default values for configuration
impl Default for VoirsRecognitionConfig {
    fn default() -> Self {
        Self {
            model_name: std::ptr::null(),
            language: std::ptr::null(),
            sample_rate: 16000,
            enable_vad: true,
            confidence_threshold: 0.5,
            beam_size: 5,
            temperature: 0.0,
            suppress_blank: true,
        }
    }
}

impl Default for VoirsStreamingConfig {
    fn default() -> Self {
        Self {
            chunk_duration: 1.0,
            overlap_duration: 0.1,
            vad_threshold: 0.5,
            silence_duration: 2.0,
            max_chunk_size: 16000,
        }
    }
}

impl Default for VoirsAudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
            format: VoirsAudioFormatType::PCM16,
        }
    }
}
