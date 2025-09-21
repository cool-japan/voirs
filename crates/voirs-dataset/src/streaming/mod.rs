//! Streaming support for memory-efficient dataset processing
//!
//! This module provides streaming capabilities for datasets, including:
//! - Memory-efficient iteration with configurable buffers
//! - Chunk processing for fixed and variable-size chunks
//! - Network streaming with HTTP support
//! - Bandwidth throttling and connection pooling

pub mod chunks;
pub mod dataset;
pub mod network;

pub use chunks::{
    BoundaryStrategy, ChunkConfig, ChunkMetadata, ChunkProcessor, ChunkSizeStrategy,
    ChunkStatistics, DatasetChunk,
};
pub use dataset::{
    BufferStatistics, PrefetchStrategy, StreamingConfig, StreamingDataset, StreamingIterator,
};
pub use network::{
    AuthConfig, BandwidthLimit, ConnectionPoolConfig, NetworkStatistics, NetworkStreamingConfig,
    NetworkStreamingDataset, RetryConfig,
};
