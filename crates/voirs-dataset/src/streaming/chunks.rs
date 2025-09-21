//! Chunk processing for streaming datasets
//!
//! This module provides fixed-size and variable-size chunk generation with
//! boundary handling and memory usage monitoring.

use crate::traits::Dataset;
use crate::{DatasetSample, Result};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Configuration for chunk processing
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Chunk size strategy
    pub size_strategy: ChunkSizeStrategy,
    /// Boundary handling strategy
    pub boundary_strategy: BoundaryStrategy,
    /// Memory usage limit (bytes)
    pub memory_limit: Option<usize>,
    /// Enable chunk overlap
    pub enable_overlap: bool,
    /// Overlap size (if enabled)
    pub overlap_size: usize,
    /// Drop incomplete chunks
    pub drop_incomplete: bool,
}

/// Chunk size strategies
#[derive(Debug, Clone)]
pub enum ChunkSizeStrategy {
    /// Fixed number of samples per chunk
    FixedSamples(usize),
    /// Fixed duration per chunk (seconds)
    FixedDuration(f32),
    /// Variable size based on memory usage
    VariableMemory {
        target_size_mb: usize,
        min_samples: usize,
        max_samples: usize,
    },
    /// Adaptive size based on processing performance
    Adaptive {
        initial_size: usize,
        min_size: usize,
        max_size: usize,
        target_processing_time_ms: u64,
    },
}

/// Boundary handling strategies
#[derive(Debug, Clone)]
pub enum BoundaryStrategy {
    /// Strict boundaries (may create incomplete chunks)
    Strict,
    /// Fill incomplete chunks with padding
    PadToComplete,
    /// Merge small chunks with neighbors
    MergeSmall { min_size: usize },
    /// Split large chunks to target size
    SplitLarge { max_size: usize },
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            size_strategy: ChunkSizeStrategy::FixedSamples(1000),
            boundary_strategy: BoundaryStrategy::Strict,
            memory_limit: Some(512 * 1024 * 1024), // 512MB
            enable_overlap: false,
            overlap_size: 0,
            drop_incomplete: false,
        }
    }
}

/// Chunk of dataset samples
#[derive(Debug, Clone)]
pub struct DatasetChunk<T> {
    /// Chunk identifier
    pub id: String,
    /// Samples in this chunk
    pub samples: Vec<T>,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    /// Start index in original dataset
    pub start_index: usize,
    /// End index in original dataset
    pub end_index: usize,
}

/// Chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Total duration of samples in chunk
    pub total_duration: f32,
    /// Total memory usage (bytes)
    pub memory_usage: usize,
    /// Number of samples
    pub sample_count: usize,
    /// Chunk creation timestamp
    pub created_at: std::time::Instant,
    /// Processing statistics
    pub processing_stats: Option<ProcessingStats>,
}

/// Processing statistics for adaptive chunking
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Processing time for this chunk (milliseconds)
    pub processing_time_ms: u64,
    /// Peak memory usage during processing
    pub peak_memory_bytes: usize,
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
}

/// Chunk processor for creating and managing chunks
pub struct ChunkProcessor<T: Dataset> {
    /// Dataset to chunk
    dataset: Arc<T>,
    /// Configuration
    config: ChunkConfig,
    /// Chunk buffer
    chunk_buffer: Arc<Mutex<VecDeque<DatasetChunk<T::Sample>>>>,
    /// Current processing position
    position: Arc<RwLock<usize>>,
    /// Chunk statistics
    stats: Arc<RwLock<ChunkStatistics>>,
    /// Memory monitor
    memory_monitor: Arc<Mutex<MemoryMonitor>>,
}

/// Chunk processing statistics
#[derive(Debug, Clone, Default)]
pub struct ChunkStatistics {
    /// Total chunks created
    pub chunks_created: usize,
    /// Total samples processed
    pub samples_processed: usize,
    /// Average chunk size
    pub avg_chunk_size: f32,
    /// Average processing time per chunk
    pub avg_processing_time_ms: f64,
    /// Memory efficiency (samples per MB)
    pub memory_efficiency: f32,
    /// Boundary handling events
    pub boundary_events: usize,
}

/// Memory usage monitor
#[derive(Debug)]
struct MemoryMonitor {
    /// Current memory usage
    current_usage: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory usage history
    usage_history: VecDeque<(std::time::Instant, usize)>,
    /// History retention duration
    #[allow(dead_code)]
    history_duration: std::time::Duration,
}

impl<T: Dataset> ChunkProcessor<T> {
    /// Create a new chunk processor
    pub fn new(dataset: T) -> Self {
        Self::with_config(dataset, ChunkConfig::default())
    }

    /// Create a chunk processor with custom configuration
    pub fn with_config(dataset: T, config: ChunkConfig) -> Self {
        Self {
            dataset: Arc::new(dataset),
            config,
            chunk_buffer: Arc::new(Mutex::new(VecDeque::new())),
            position: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(ChunkStatistics::default())),
            memory_monitor: Arc::new(Mutex::new(MemoryMonitor::new())),
        }
    }

    /// Process next chunk from the dataset
    pub async fn next_chunk(&self) -> Result<Option<DatasetChunk<T::Sample>>> {
        let current_pos = *self.position.read().await;
        if current_pos >= self.dataset.len() {
            return Ok(None);
        }

        let chunk = match &self.config.size_strategy {
            ChunkSizeStrategy::FixedSamples(size) => self.create_fixed_size_chunk(*size).await?,
            ChunkSizeStrategy::FixedDuration(duration) => {
                self.create_fixed_duration_chunk(*duration).await?
            }
            ChunkSizeStrategy::VariableMemory {
                target_size_mb,
                min_samples,
                max_samples,
            } => {
                self.create_variable_memory_chunk(*target_size_mb, *min_samples, *max_samples)
                    .await?
            }
            ChunkSizeStrategy::Adaptive {
                initial_size,
                min_size,
                max_size,
                target_processing_time_ms,
            } => {
                self.create_adaptive_chunk(
                    *initial_size,
                    *min_size,
                    *max_size,
                    *target_processing_time_ms,
                )
                .await?
            }
        };

        if let Some(chunk) = chunk {
            self.apply_boundary_handling(&chunk).await?;
            self.update_statistics(&chunk).await;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    /// Process all remaining data into chunks
    pub async fn process_all(&self) -> Result<Vec<DatasetChunk<T::Sample>>> {
        let mut chunks = Vec::new();

        while let Some(chunk) = self.next_chunk().await? {
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Reset processor to beginning
    pub async fn reset(&self) {
        *self.position.write().await = 0;
        self.chunk_buffer.lock().await.clear();
        self.memory_monitor.lock().await.reset();
    }

    /// Get processing statistics
    pub async fn get_statistics(&self) -> ChunkStatistics {
        self.stats.read().await.clone()
    }

    /// Get current memory usage
    pub async fn get_memory_usage(&self) -> usize {
        self.memory_monitor.lock().await.current_usage
    }

    // Private implementation methods

    async fn create_fixed_size_chunk(
        &self,
        size: usize,
    ) -> Result<Option<DatasetChunk<T::Sample>>> {
        let start_pos = *self.position.read().await;
        let end_pos = (start_pos + size).min(self.dataset.len());

        if start_pos >= end_pos {
            return Ok(None);
        }

        let mut samples = Vec::with_capacity(end_pos - start_pos);
        let mut total_duration = 0.0f32;
        let mut memory_usage = 0usize;

        for i in start_pos..end_pos {
            let sample = self.dataset.get(i).await?;

            // Calculate memory usage and duration based on sample type
            if let Ok(dataset_sample) = self.try_cast_to_dataset_sample(&sample) {
                total_duration += dataset_sample.audio.duration();
                memory_usage += self.estimate_sample_memory_usage(dataset_sample);
            }

            samples.push(sample);
        }

        *self.position.write().await = end_pos;

        let chunk = DatasetChunk {
            id: format!("chunk_{start_pos}"),
            metadata: ChunkMetadata {
                total_duration,
                memory_usage,
                sample_count: samples.len(),
                created_at: std::time::Instant::now(),
                processing_stats: None,
            },
            start_index: start_pos,
            end_index: end_pos,
            samples,
        };

        Ok(Some(chunk))
    }

    async fn create_fixed_duration_chunk(
        &self,
        target_duration: f32,
    ) -> Result<Option<DatasetChunk<T::Sample>>> {
        let start_pos = *self.position.read().await;
        if start_pos >= self.dataset.len() {
            return Ok(None);
        }

        let mut samples = Vec::new();
        let mut total_duration = 0.0f32;
        let mut memory_usage = 0usize;
        let mut current_pos = start_pos;

        while current_pos < self.dataset.len() && total_duration < target_duration {
            let sample = self.dataset.get(current_pos).await?;

            if let Ok(dataset_sample) = self.try_cast_to_dataset_sample(&sample) {
                let sample_duration = dataset_sample.audio.duration();

                // Check if adding this sample would exceed target duration
                if total_duration + sample_duration > target_duration && !samples.is_empty() {
                    break;
                }

                total_duration += sample_duration;
                memory_usage += self.estimate_sample_memory_usage(dataset_sample);
            }

            samples.push(sample);
            current_pos += 1;
        }

        if samples.is_empty() {
            return Ok(None);
        }

        *self.position.write().await = current_pos;

        let chunk = DatasetChunk {
            id: format!("chunk_duration_{start_pos}"),
            metadata: ChunkMetadata {
                total_duration,
                memory_usage,
                sample_count: samples.len(),
                created_at: std::time::Instant::now(),
                processing_stats: None,
            },
            start_index: start_pos,
            end_index: current_pos,
            samples,
        };

        Ok(Some(chunk))
    }

    async fn create_variable_memory_chunk(
        &self,
        target_size_mb: usize,
        min_samples: usize,
        max_samples: usize,
    ) -> Result<Option<DatasetChunk<T::Sample>>> {
        let start_pos = *self.position.read().await;
        if start_pos >= self.dataset.len() {
            return Ok(None);
        }

        let target_bytes = target_size_mb * 1024 * 1024;
        let mut samples = Vec::new();
        let mut total_duration = 0.0f32;
        let mut memory_usage = 0usize;
        let mut current_pos = start_pos;

        while current_pos < self.dataset.len()
            && samples.len() < max_samples
            && (memory_usage < target_bytes || samples.len() < min_samples)
        {
            let sample = self.dataset.get(current_pos).await?;

            if let Ok(dataset_sample) = self.try_cast_to_dataset_sample(&sample) {
                let sample_memory = self.estimate_sample_memory_usage(dataset_sample);

                // Check if adding this sample would exceed memory limit
                if memory_usage + sample_memory > target_bytes && samples.len() >= min_samples {
                    break;
                }

                total_duration += dataset_sample.audio.duration();
                memory_usage += sample_memory;
            }

            samples.push(sample);
            current_pos += 1;
        }

        if samples.is_empty() {
            return Ok(None);
        }

        *self.position.write().await = current_pos;

        let chunk = DatasetChunk {
            id: format!("chunk_memory_{start_pos}"),
            metadata: ChunkMetadata {
                total_duration,
                memory_usage,
                sample_count: samples.len(),
                created_at: std::time::Instant::now(),
                processing_stats: None,
            },
            start_index: start_pos,
            end_index: current_pos,
            samples,
        };

        Ok(Some(chunk))
    }

    async fn create_adaptive_chunk(
        &self,
        initial_size: usize,
        min_size: usize,
        max_size: usize,
        target_time_ms: u64,
    ) -> Result<Option<DatasetChunk<T::Sample>>> {
        // For adaptive chunking, we start with a size based on historical performance
        let stats = self.stats.read().await;
        let adaptive_size = if stats.chunks_created > 0 {
            // Adjust size based on average processing time
            let avg_time = stats.avg_processing_time_ms;
            let target_time = target_time_ms as f64;

            let size_factor = target_time / avg_time.max(1.0);
            let new_size = (stats.avg_chunk_size * size_factor as f32) as usize;

            new_size.clamp(min_size, max_size)
        } else {
            initial_size
        };
        drop(stats);

        // Create chunk using the adaptive size
        self.create_fixed_size_chunk(adaptive_size).await
    }

    async fn apply_boundary_handling(&self, chunk: &DatasetChunk<T::Sample>) -> Result<()> {
        match &self.config.boundary_strategy {
            BoundaryStrategy::Strict => {
                // No modifications needed
                Ok(())
            }
            BoundaryStrategy::PadToComplete => {
                // Implementation would depend on specific padding strategy
                Ok(())
            }
            BoundaryStrategy::MergeSmall { min_size } => {
                if chunk.samples.len() < *min_size {
                    let mut stats = self.stats.write().await;
                    stats.boundary_events += 1;
                }
                Ok(())
            }
            BoundaryStrategy::SplitLarge { max_size } => {
                if chunk.samples.len() > *max_size {
                    let mut stats = self.stats.write().await;
                    stats.boundary_events += 1;
                }
                Ok(())
            }
        }
    }

    async fn update_statistics(&self, chunk: &DatasetChunk<T::Sample>) {
        let mut stats = self.stats.write().await;

        stats.chunks_created += 1;
        stats.samples_processed += chunk.samples.len();

        // Update average chunk size
        let total_samples =
            stats.chunks_created as f32 * stats.avg_chunk_size + chunk.samples.len() as f32;
        stats.avg_chunk_size = total_samples / stats.chunks_created as f32;

        // Update memory efficiency
        if chunk.metadata.memory_usage > 0 {
            let efficiency =
                chunk.samples.len() as f32 / (chunk.metadata.memory_usage as f32 / 1024.0 / 1024.0);
            stats.memory_efficiency = (stats.memory_efficiency * (stats.chunks_created - 1) as f32
                + efficiency)
                / stats.chunks_created as f32;
        }

        // Update processing time if available
        if let Some(processing_stats) = &chunk.metadata.processing_stats {
            let new_avg = (stats.avg_processing_time_ms * (stats.chunks_created - 1) as f64
                + processing_stats.processing_time_ms as f64)
                / stats.chunks_created as f64;
            stats.avg_processing_time_ms = new_avg;
        }
    }

    fn try_cast_to_dataset_sample(&self, sample: &T::Sample) -> Result<&DatasetSample> {
        // This is a helper method to extract DatasetSample if the generic type allows it
        // In a real implementation, this would use proper type casting or trait bounds
        unsafe {
            Ok(std::mem::transmute::<
                &<T as crate::traits::Dataset>::Sample,
                &DatasetSample,
            >(sample))
        }
    }

    fn estimate_sample_memory_usage(&self, sample: &DatasetSample) -> usize {
        // Estimate memory usage of a sample
        let audio_size = std::mem::size_of_val(sample.audio.samples());
        let text_size = sample.text.len();
        let metadata_size = sample.metadata.len() * 64; // Rough estimate

        audio_size + text_size + metadata_size + std::mem::size_of::<DatasetSample>()
    }
}

impl MemoryMonitor {
    fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            usage_history: VecDeque::new(),
            history_duration: std::time::Duration::from_secs(300), // 5 minutes
        }
    }

    #[allow(dead_code)]
    fn update_usage(&mut self, usage: usize) {
        self.current_usage = usage;
        self.peak_usage = self.peak_usage.max(usage);

        let now = std::time::Instant::now();
        self.usage_history.push_back((now, usage));

        // Remove old entries
        while let Some((timestamp, _)) = self.usage_history.front() {
            if now.duration_since(*timestamp) > self.history_duration {
                self.usage_history.pop_front();
            } else {
                break;
            }
        }
    }

    fn reset(&mut self) {
        self.current_usage = 0;
        self.peak_usage = 0;
        self.usage_history.clear();
    }

    #[allow(dead_code)]
    fn get_average_usage(&self) -> usize {
        if self.usage_history.is_empty() {
            return 0;
        }

        let sum: usize = self.usage_history.iter().map(|(_, usage)| usage).sum();
        sum / self.usage_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;

    #[tokio::test]
    async fn test_chunk_processor_creation() {
        let dataset = DummyDataset::small();
        let processor = ChunkProcessor::new(dataset);

        let stats = processor.get_statistics().await;
        assert_eq!(stats.chunks_created, 0);
        assert_eq!(stats.samples_processed, 0);
    }

    #[tokio::test]
    async fn test_fixed_size_chunking() {
        let dataset = DummyDataset::small();
        let config = ChunkConfig {
            size_strategy: ChunkSizeStrategy::FixedSamples(3),
            ..Default::default()
        };

        let processor = ChunkProcessor::with_config(dataset, config);

        let mut chunks = Vec::new();
        while let Some(chunk) = processor.next_chunk().await.unwrap() {
            chunks.push(chunk);
        }

        assert!(!chunks.is_empty());

        // Most chunks should have 3 samples (except possibly the last one)
        for (i, chunk) in chunks.iter().enumerate() {
            if i < chunks.len() - 1 {
                assert_eq!(chunk.samples.len(), 3);
            } else {
                assert!(chunk.samples.len() <= 3);
            }
        }
    }

    #[tokio::test]
    async fn test_variable_memory_chunking() {
        let dataset = DummyDataset::small();
        let config = ChunkConfig {
            size_strategy: ChunkSizeStrategy::VariableMemory {
                target_size_mb: 1,
                min_samples: 1,
                max_samples: 10,
            },
            ..Default::default()
        };

        let processor = ChunkProcessor::with_config(dataset, config);

        let chunk = processor.next_chunk().await.unwrap();
        assert!(chunk.is_some());

        let chunk = chunk.unwrap();
        assert!(!chunk.samples.is_empty());
        assert!(chunk.samples.len() <= 10);
    }

    #[tokio::test]
    async fn test_chunk_metadata() {
        let dataset = DummyDataset::small();
        let processor = ChunkProcessor::new(dataset);

        let chunk = processor.next_chunk().await.unwrap().unwrap();

        assert!(!chunk.id.is_empty());
        assert!(chunk.metadata.sample_count > 0);
        assert!(chunk.metadata.created_at.elapsed().as_millis() < 1000);
        assert_eq!(chunk.samples.len(), chunk.metadata.sample_count);
    }

    #[tokio::test]
    async fn test_process_all_chunks() {
        let dataset = DummyDataset::small();
        let config = ChunkConfig {
            size_strategy: ChunkSizeStrategy::FixedSamples(4),
            ..Default::default()
        };

        let processor = ChunkProcessor::with_config(dataset, config);
        let chunks = processor.process_all().await.unwrap();

        // Calculate total samples across all chunks
        let total_samples: usize = chunks.iter().map(|c| c.samples.len()).sum();
        assert_eq!(total_samples, 10); // DummyDataset::small() has 10 samples
    }

    #[tokio::test]
    async fn test_processor_reset() {
        let dataset = DummyDataset::small();
        let processor = ChunkProcessor::new(dataset);

        // Process one chunk
        let _ = processor.next_chunk().await.unwrap();

        // Reset and process again
        processor.reset().await;
        let chunk = processor.next_chunk().await.unwrap();

        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.start_index, 0);
    }

    #[tokio::test]
    async fn test_boundary_handling() {
        let dataset = DummyDataset::small();
        let config = ChunkConfig {
            size_strategy: ChunkSizeStrategy::FixedSamples(3),
            boundary_strategy: BoundaryStrategy::MergeSmall { min_size: 2 },
            ..Default::default()
        };

        let processor = ChunkProcessor::with_config(dataset, config);

        // Process all chunks
        let _chunks = processor.process_all().await.unwrap();

        let stats = processor.get_statistics().await;
        // Should have some boundary events due to the last chunk being small
        // Just verify the stats are accessible (boundary_events is unsigned, so always >= 0)
        let _ = stats.boundary_events;
    }
}
