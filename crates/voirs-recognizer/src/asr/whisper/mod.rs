//! Pure Rust implementation of `OpenAI` Whisper
//!
//! This module provides a complete Rust port of the `OpenAI` Whisper model,
//! eliminating Python dependencies while maintaining full compatibility
//! with the original model architecture and trained weights.

pub mod attention;
pub mod audio_processor;
pub mod batch_processing;
pub mod benchmarking;
pub mod decoder;
pub mod encoder;
pub mod error_handling;
pub mod memory_manager;
pub mod quantization;
pub mod streaming;
pub mod tokenizer;

pub use attention::{KVCache, MultiHeadAttention};
pub use audio_processor::WhisperAudioProcessor;
pub use batch_processing::{
    BatchConfig, BatchInput, BatchOutput, BatchStats, WhisperBatchProcessor,
};
pub use benchmarking::{
    BenchmarkConfig, BenchmarkResults, OptimizationSuggestion, OverallPerformance, WhisperBenchmark,
};
pub use decoder::{DecoderBlock, SamplingConfig, SamplingStrategy, WhisperDecoder};
pub use encoder::{QuantizationMode, TransformerBlock, WhisperConfig, WhisperEncoder, MLP};
pub use error_handling::{
    ErrorRecoveryManager, MemoryStats as ErrorMemoryStats, RecoveryAction, WhisperError,
};
pub use memory_manager::{CleanupStats, MemoryConfig, MemoryStats, WhisperMemoryManager};
pub use quantization::{
    ModelQuantizer, QuantizationConfig, QuantizationSavings, QuantizationStats,
};
pub use streaming::{
    ProcessingStats, StreamingConfig, StreamingWhisperProcessor, TranscriptSegment,
};
pub use tokenizer::{BytePairEncoding, SpecialTokens, WhisperTask, WhisperTokenizer};
