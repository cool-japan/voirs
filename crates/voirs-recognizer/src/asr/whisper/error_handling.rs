//! Comprehensive error handling and recovery for Whisper components
//!
//! This module provides detailed error classification, recovery strategies,
//! and error context management for robust production deployments.

use crate::RecognitionError;
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Detailed Whisper-specific error types with recovery suggestions
#[derive(Debug, Clone)]
/// Whisper Error
pub enum WhisperError {
    /// Model loading errors with specific component information
    ModelLoad {
        /// Which model component failed to load
        component: ModelComponent,
        /// Detailed error message
        details: String,
        /// Whether the error can be recovered from
        recoverable: bool,
        /// Suggested action to resolve the error
        suggested_action: String,
    },
    /// Audio processing errors with format details
    AudioProcessing {
        /// Which processing stage failed
        stage: AudioStage,
        /// Input audio format that caused the error
        input_format: AudioFormat,
        /// Detailed error message
        details: String,
        /// Whether the error can be recovered from
        recoverable: bool,
    },
    /// Memory allocation and management errors
    Memory {
        /// Memory operation that failed
        operation: MemoryOperation,
        /// Size of memory requested in bytes
        requested_size: usize,
        /// Available memory size if known
        available_size: Option<usize>,
        /// Device where memory allocation failed
        device: String,
        /// Whether the error can be recovered from
        recoverable: bool,
    },
    /// Attention computation errors with context
    Attention {
        /// Which transformer layer the error occurred in
        layer: usize,
        /// Which attention head failed
        head: usize,
        /// Length of the sequence being processed
        sequence_length: usize,
        /// Detailed error message
        details: String,
        /// Whether a fallback computation method is available
        fallback_available: bool,
    },
    /// Tokenization errors with language context
    Tokenization {
        /// Language being processed when error occurred
        language: String,
        /// Sample of text that caused the tokenization error
        text_sample: String,
        /// Position in token sequence where error occurred
        token_position: Option<usize>,
        /// Detailed error message
        details: String,
        /// Whether the error can be recovered from
        recoverable: bool,
    },
    /// Streaming processing errors
    Streaming {
        /// Identifier of the audio chunk that failed
        chunk_id: u64,
        /// Current state of the audio buffer
        buffer_state: BufferState,
        /// Detailed error message
        details: String,
        /// Whether streaming can continue after this error
        can_continue: bool,
    },
    /// Device-specific errors (CUDA, Metal, etc.)
    Device {
        /// Type of device that failed (e.g., "CUDA", "Metal", "CPU")
        device_type: String,
        /// Device identifier if multiple devices of same type
        device_id: Option<usize>,
        /// Operation that was being performed when error occurred
        operation: String,
        /// Detailed error message
        details: String,
        /// Whether a fallback device is available
        fallback_available: bool,
    },
}

/// Model component identifier
#[derive(Debug, Clone)]
/// Model Component
pub enum ModelComponent {
    /// Encoder component of the model
    Encoder,
    /// Decoder component of the model
    Decoder,
    /// Tokenizer component of the model
    Tokenizer,
    /// Audio preprocessing component
    AudioProcessor,
    /// Attention layer at specified index
    Attention(usize), // layer index
    /// MLP layer at specified index
    MLP(usize), // layer index
    /// Embedding layer component
    Embedding,
    /// Layer normalization at specified index
    LayerNorm(usize),
}

/// Audio processing stage
#[derive(Debug, Clone)]
/// Audio Stage
pub enum AudioStage {
    /// Audio preprocessing stage
    Preprocessing,
    /// Short-Time Fourier Transform stage
    STFT,
    /// Mel-frequency filter bank stage
    MelFilterBank,
    /// Audio normalization stage
    Normalization,
    /// Audio resampling stage
    Resampling,
    /// Voice Activity Detection stage
    VAD,
}

/// Audio format specification for audio processing
#[derive(Debug, Clone)]
/// Audio Format
pub struct AudioFormat {
    /// Audio sample rate in Hz (e.g., 16000, 44100)
    pub sample_rate: u32,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u32,
    /// Bit depth of audio samples (e.g., 16, 24, 32)
    pub bit_depth: u32,
    /// Audio format string (e.g., "wav", "mp3", "flac")
    pub format: String,
    /// Duration of audio in seconds
    pub duration_seconds: f32,
}

/// Memory operation type for error context
#[derive(Debug, Clone, Copy)]
/// Memory Operation
pub enum MemoryOperation {
    /// Memory allocation operation
    Allocation,
    /// Memory deallocation operation
    Deallocation,
    /// Memory transfer between devices
    Transfer,
    /// Memory copy operation
    Copy,
    /// Memory resize operation
    Resize,
    /// Memory pool operation
    Pool,
}

/// Buffer state for streaming errors
#[derive(Debug, Clone)]
/// Buffer State
pub struct BufferState {
    /// Current buffer fill percentage (0.0 to 100.0)
    pub fill_percentage: f32,
    /// Buffer capacity in megabytes
    pub capacity_mb: f32,
    /// Number of pending audio chunks in buffer
    pub pending_chunks: usize,
    /// Current processing latency in milliseconds
    pub processing_latency_ms: u32,
}

impl fmt::Display for WhisperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WhisperError::ModelLoad {
                component,
                details,
                suggested_action,
                ..
            } => {
                write!(
                    f,
                    "Model loading failed for {component:?}: {details}. Suggestion: {suggested_action}"
                )
            }
            WhisperError::AudioProcessing {
                stage,
                input_format,
                details,
                ..
            } => {
                write!(
                    f,
                    "Audio processing failed at {:?} stage with format {}kHz/{}ch: {}",
                    stage, input_format.sample_rate, input_format.channels, details
                )
            }
            WhisperError::Memory {
                operation,
                requested_size,
                device,
                ..
            } => {
                write!(
                    f,
                    "Memory {operation:?} failed on {device}: requested {requested_size} bytes"
                )
            }
            WhisperError::Attention {
                layer,
                head,
                sequence_length,
                details,
                ..
            } => {
                write!(
                    f,
                    "Attention computation failed at layer {layer} head {head} (seq_len={sequence_length}): {details}"
                )
            }
            WhisperError::Tokenization {
                language,
                text_sample,
                details,
                ..
            } => {
                write!(
                    f,
                    "Tokenization failed for {} language with text '{}...': {}",
                    language,
                    text_sample.chars().take(50).collect::<String>(),
                    details
                )
            }
            WhisperError::Streaming {
                chunk_id,
                buffer_state,
                details,
                ..
            } => {
                write!(
                    f,
                    "Streaming failed at chunk {} (buffer {}% full): {}",
                    chunk_id, buffer_state.fill_percentage, details
                )
            }
            WhisperError::Device {
                device_type,
                device_id,
                operation,
                details,
                ..
            } => {
                write!(
                    f,
                    "Device {device_type} operation '{operation}' failed on {device_id:?}: {details}"
                )
            }
        }
    }
}

impl std::error::Error for WhisperError {}

/// Error recovery manager with retry strategies
pub struct ErrorRecoveryManager {
    retry_counts: Arc<RwLock<std::collections::HashMap<String, u32>>>,
    max_retries: u32,
    fallback_enabled: bool,
    memory_monitor: Arc<RwLock<MemoryMonitor>>,
}

/// Memory usage monitor with cleanup strategies
#[derive(Debug)]
/// Memory Monitor
pub struct MemoryMonitor {
    peak_usage_mb: f32,
    current_usage_mb: f32,
    allocation_count: u64,
    deallocation_count: u64,
    memory_leaks_detected: u32,
    last_cleanup_time: std::time::SystemTime,
    cleanup_threshold_mb: f32,
    emergency_cleanup_enabled: bool,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager with specified configuration
    ///
    /// # Arguments
    /// * `max_retries` - Maximum number of retry attempts for recoverable errors
    /// * `fallback_enabled` - Whether fallback mechanisms are enabled
    /// * `memory_threshold_mb` - Memory threshold in MB for triggering cleanup
    #[must_use]
    /// new
    pub fn new(max_retries: u32, fallback_enabled: bool, memory_threshold_mb: f32) -> Self {
        Self {
            retry_counts: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_retries,
            fallback_enabled,
            memory_monitor: Arc::new(RwLock::new(MemoryMonitor::new(memory_threshold_mb))),
        }
    }

    /// Attempt to recover from an error with appropriate strategy
    pub async fn handle_error(&self, error: &WhisperError) -> RecoveryAction {
        match error {
            WhisperError::Memory {
                operation,
                requested_size,
                device,
                recoverable,
                ..
            } => {
                if *recoverable {
                    self.handle_memory_error(*operation, *requested_size, device)
                        .await
                } else {
                    RecoveryAction::Fail("Non-recoverable memory error".to_string())
                }
            }
            WhisperError::Attention {
                fallback_available, ..
            } => {
                if *fallback_available && self.fallback_enabled {
                    RecoveryAction::UseFallback("Standard attention computation".to_string())
                } else {
                    RecoveryAction::Retry
                }
            }
            WhisperError::Device {
                fallback_available, ..
            } => {
                if *fallback_available && self.fallback_enabled {
                    RecoveryAction::UseFallback("CPU computation".to_string())
                } else {
                    RecoveryAction::Fail("No device fallback available".to_string())
                }
            }
            WhisperError::Streaming { can_continue, .. } => {
                if *can_continue {
                    RecoveryAction::SkipAndContinue
                } else {
                    RecoveryAction::Reset
                }
            }
            _ => {
                if self.should_retry(&error.error_key()).await {
                    RecoveryAction::Retry
                } else {
                    RecoveryAction::Fail("Max retries exceeded".to_string())
                }
            }
        }
    }

    async fn handle_memory_error(
        &self,
        _operation: MemoryOperation,
        requested_size: usize,
        device: &str,
    ) -> RecoveryAction {
        let mut monitor = self.memory_monitor.write().await;

        // Trigger emergency cleanup if needed
        if monitor.should_emergency_cleanup(requested_size) {
            monitor.emergency_cleanup().await;
            RecoveryAction::RetryAfterCleanup
        } else if monitor.can_satisfy_request(requested_size) {
            RecoveryAction::Retry
        } else {
            RecoveryAction::Fail(format!(
                "Insufficient memory: requested {} MB on {}",
                requested_size / 1024 / 1024,
                device
            ))
        }
    }

    async fn should_retry(&self, error_key: &str) -> bool {
        let mut counts = self.retry_counts.write().await;
        let count = counts.entry(error_key.to_string()).or_insert(0);
        *count += 1;
        *count <= self.max_retries
    }

    /// reset retry count
    pub async fn reset_retry_count(&self, error_key: &str) {
        let mut counts = self.retry_counts.write().await;
        counts.remove(error_key);
    }

    /// get memory stats
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let monitor = self.memory_monitor.read().await;
        MemoryStats {
            peak_usage_mb: monitor.peak_usage_mb,
            current_usage_mb: monitor.current_usage_mb,
            allocation_count: monitor.allocation_count,
            deallocation_count: monitor.deallocation_count,
            memory_leaks_detected: monitor.memory_leaks_detected,
        }
    }
}

/// Recovery action to take after error analysis
#[derive(Debug, Clone)]
/// Recovery Action
pub enum RecoveryAction {
    /// Retry
    Retry,
    /// Retry after cleanup
    RetryAfterCleanup,
    /// Use fallback
    UseFallback(String),
    /// Skip and continue
    SkipAndContinue,
    /// Reset
    Reset,
    /// Fail
    Fail(String),
}

/// Memory usage statistics
#[derive(Debug, Clone)]
/// Memory Stats
pub struct MemoryStats {
    /// peak usage mb
    pub peak_usage_mb: f32,
    /// current usage mb
    pub current_usage_mb: f32,
    /// allocation count
    pub allocation_count: u64,
    /// deallocation count
    pub deallocation_count: u64,
    /// memory leaks detected
    pub memory_leaks_detected: u32,
}

impl MemoryMonitor {
    fn new(cleanup_threshold_mb: f32) -> Self {
        Self {
            peak_usage_mb: 0.0,
            current_usage_mb: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
            memory_leaks_detected: 0,
            last_cleanup_time: std::time::SystemTime::now(),
            cleanup_threshold_mb,
            emergency_cleanup_enabled: true,
        }
    }

    /// record allocation
    pub fn record_allocation(&mut self, size_mb: f32) {
        self.current_usage_mb += size_mb;
        self.allocation_count += 1;
        if self.current_usage_mb > self.peak_usage_mb {
            self.peak_usage_mb = self.current_usage_mb;
        }
    }

    /// record deallocation
    pub fn record_deallocation(&mut self, size_mb: f32) {
        self.current_usage_mb = (self.current_usage_mb - size_mb).max(0.0);
        self.deallocation_count += 1;
    }

    fn should_emergency_cleanup(&self, requested_size: usize) -> bool {
        let requested_mb = requested_size as f32 / 1024.0 / 1024.0;
        self.emergency_cleanup_enabled
            && (self.current_usage_mb + requested_mb) > self.cleanup_threshold_mb
    }

    async fn emergency_cleanup(&mut self) {
        // Trigger garbage collection and memory cleanup
        // This would integrate with the actual memory management system
        let cleanup_amount = self.current_usage_mb * 0.3; // Clean up 30%
        self.current_usage_mb -= cleanup_amount;
        self.last_cleanup_time = std::time::SystemTime::now();

        // In a real implementation, this would:
        // 1. Force garbage collection
        // 2. Clear tensor caches
        // 3. Defragment memory pools
        // 4. Release unused GPU memory
    }

    fn can_satisfy_request(&self, requested_size: usize) -> bool {
        let requested_mb = requested_size as f32 / 1024.0 / 1024.0;
        (self.current_usage_mb + requested_mb) < self.cleanup_threshold_mb
    }
}

impl WhisperError {
    fn error_key(&self) -> String {
        match self {
            WhisperError::ModelLoad { component, .. } => format!("model_load_{component:?}"),
            WhisperError::AudioProcessing { stage, .. } => format!("audio_{stage:?}"),
            WhisperError::Memory {
                operation, device, ..
            } => format!("memory_{operation:?}_{device}"),
            WhisperError::Attention { layer, .. } => format!("attention_{layer}"),
            WhisperError::Tokenization { language, .. } => format!("tokenization_{language}"),
            WhisperError::Streaming { .. } => "streaming".to_string(),
            WhisperError::Device { device_type, .. } => format!("device_{device_type}"),
        }
    }

    #[must_use]
    /// is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            WhisperError::ModelLoad { recoverable, .. } => *recoverable,
            WhisperError::AudioProcessing { recoverable, .. } => *recoverable,
            WhisperError::Memory { recoverable, .. } => *recoverable,
            WhisperError::Attention {
                fallback_available, ..
            } => *fallback_available,
            WhisperError::Tokenization { recoverable, .. } => *recoverable,
            WhisperError::Streaming { can_continue, .. } => *can_continue,
            WhisperError::Device {
                fallback_available, ..
            } => *fallback_available,
        }
    }

    #[must_use]
    /// to recognition error
    pub fn to_recognition_error(self) -> RecognitionError {
        match self {
            WhisperError::ModelLoad { details, .. } => RecognitionError::ModelLoadError {
                message: details,
                source: None,
            },
            WhisperError::AudioProcessing { details, .. } => {
                RecognitionError::AudioProcessingError {
                    message: details,
                    source: None,
                }
            }
            WhisperError::Memory {
                operation,
                requested_size,
                device,
                ..
            } => RecognitionError::ModelError {
                message: format!(
                    "Memory error: {operation:?} operation failed on {device} (requested {requested_size} bytes)"
                ),
                source: None,
            },
            _ => RecognitionError::ModelError {
                message: self.to_string(),
                source: None,
            },
        }
    }
}
