//! Pure Rust implementation of `OpenAI` Whisper
//!
//! This module provides a Rust port of the `OpenAI` Whisper architecture that runs on
//! the real trained weights published for that architecture.
//!
//! # Weights are mandatory
//!
//! Every constructor requires real pretrained parameters, supplied through
//! [`WhisperConfig::with_assets`](encoder::WhisperConfig::with_assets) or discovered from
//! the environment with [`assets::assets_from_env`]. Without them construction fails
//! closed with [`crate::RecognitionError::ModelLoadError`]: an untrained network emits
//! text that looks like a transcript but carries no information, so VoiRS refuses to
//! produce one.

pub mod assets;
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

pub use assets::{assets_from_env, WhisperAssets, ASSETS_ENV_VAR};
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
