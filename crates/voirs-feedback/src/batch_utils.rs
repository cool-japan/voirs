//! Batch processing utilities for efficient operations.
//!
//! This module provides utilities for processing large datasets in batches
//! to improve memory efficiency and performance.

use std::time::{Duration, Instant};

/// Configuration for batch processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of items to process in each batch
    pub batch_size: usize,
    /// Maximum time to spend processing a single batch
    pub timeout_per_batch: Option<Duration>,
    /// Maximum number of retries for failed batches
    pub max_retries: usize,
    /// Enable parallel processing of batches
    pub parallel: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            timeout_per_batch: Some(Duration::from_secs(30)),
            max_retries: 3,
            parallel: false,
        }
    }
}

/// Result of batch processing.
#[derive(Debug)]
pub struct BatchResult<T> {
    /// Successfully processed items
    pub successes: Vec<T>,
    /// Failed items with error messages
    pub failures: Vec<(usize, String)>,
    /// Total processing time
    pub duration: Duration,
    /// Number of batches processed
    pub batch_count: usize,
}

/// Process items in batches with a given function.
///
/// # Arguments
///
/// * `items` - Items to process
/// * `config` - Batch configuration
/// * `processor` - Function to process each batch
///
/// # Examples
///
/// ```
/// use voirs_feedback::batch_utils::{process_batches, BatchConfig};
///
/// let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let config = BatchConfig {
///     batch_size: 3,
///     ..Default::default()
/// };
///
/// let result = process_batches(
///     items,
///     config,
///     |batch| {
///         // Process each batch
///         Ok(batch.iter().map(|&x| x * 2).collect())
///     }
/// );
///
/// assert_eq!(result.successes.len(), 10);
/// ```
pub fn process_batches<T, R, F>(
    items: Vec<T>,
    config: BatchConfig,
    mut processor: F,
) -> BatchResult<R>
where
    T: Clone,
    F: FnMut(&[T]) -> Result<Vec<R>, String>,
{
    let start = Instant::now();
    let mut successes = Vec::new();
    let mut failures = Vec::new();
    let mut batch_count = 0;

    for (batch_idx, chunk) in items.chunks(config.batch_size).enumerate() {
        batch_count += 1;
        let mut attempts = 0;
        let mut success = false;

        while attempts < config.max_retries && !success {
            attempts += 1;

            match processor(chunk) {
                Ok(results) => {
                    successes.extend(results);
                    success = true;
                }
                Err(e) => {
                    if attempts >= config.max_retries {
                        for (item_idx, _) in chunk.iter().enumerate() {
                            let global_idx = batch_idx * config.batch_size + item_idx;
                            failures.push((global_idx, e.clone()));
                        }
                    }
                }
            }
        }
    }

    BatchResult {
        successes,
        failures,
        duration: start.elapsed(),
        batch_count,
    }
}

/// Chunk iterator that yields batches of a specified size.
///
/// # Examples
///
/// ```
/// use voirs_feedback::batch_utils::ChunkIterator;
///
/// let data = vec![1, 2, 3, 4, 5, 6, 7];
/// let mut chunks = ChunkIterator::new(data, 3);
///
/// assert_eq!(chunks.next(), Some(vec![1, 2, 3]));
/// assert_eq!(chunks.next(), Some(vec![4, 5, 6]));
/// assert_eq!(chunks.next(), Some(vec![7]));
/// assert_eq!(chunks.next(), None);
/// ```
pub struct ChunkIterator<T> {
    data: Vec<T>,
    chunk_size: usize,
    current_index: usize,
}

impl<T: Clone> ChunkIterator<T> {
    /// Create a new chunk iterator.
    #[must_use]
    pub fn new(data: Vec<T>, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        Self {
            data,
            chunk_size,
            current_index: 0,
        }
    }

    /// Get the number of remaining items.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.current_index)
    }
}

impl<T: Clone> Iterator for ChunkIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.data.len() {
            return None;
        }

        let end = (self.current_index + self.chunk_size).min(self.data.len());
        let chunk = self.data[self.current_index..end].to_vec();
        self.current_index = end;

        Some(chunk)
    }
}

/// Batch processor with progress tracking.
pub struct BatchProcessor<T, R> {
    config: BatchConfig,
    processed_count: usize,
    total_count: usize,
    successes: Vec<R>,
    failures: Vec<(usize, String)>,
    start_time: Instant,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, R> BatchProcessor<T, R>
where
    T: Clone,
{
    /// Create a new batch processor.
    ///
    /// # Arguments
    ///
    /// * `total_count` - Total number of items to process
    /// * `config` - Batch configuration
    #[must_use]
    pub fn new(total_count: usize, config: BatchConfig) -> Self {
        Self {
            config,
            processed_count: 0,
            total_count,
            successes: Vec::new(),
            failures: Vec::new(),
            start_time: Instant::now(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process a batch of items.
    ///
    /// # Arguments
    ///
    /// * `batch_items` - Items in the current batch
    /// * `processor` - Function to process the batch
    pub fn process_batch<F>(&mut self, batch_items: &[T], mut processor: F) -> Result<(), String>
    where
        F: FnMut(&[T]) -> Result<Vec<R>, String>,
    {
        let batch_start_idx = self.processed_count;

        match processor(batch_items) {
            Ok(results) => {
                self.successes.extend(results);
                self.processed_count += batch_items.len();
                Ok(())
            }
            Err(e) => {
                for (idx, _) in batch_items.iter().enumerate() {
                    self.failures.push((batch_start_idx + idx, e.clone()));
                }
                self.processed_count += batch_items.len();
                Err(e)
            }
        }
    }

    /// Get current progress as a percentage (0.0 to 100.0).
    #[must_use]
    pub fn progress_percent(&self) -> f32 {
        if self.total_count == 0 {
            100.0
        } else {
            (self.processed_count as f32 / self.total_count as f32) * 100.0
        }
    }

    /// Get estimated time remaining.
    #[must_use]
    pub fn estimated_time_remaining(&self) -> Option<Duration> {
        if self.processed_count == 0 {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let rate = self.processed_count as f64 / elapsed.as_secs_f64();

        if rate <= 0.0 {
            return None;
        }

        let remaining_items = self.total_count.saturating_sub(self.processed_count);
        let estimated_secs = remaining_items as f64 / rate;

        Some(Duration::from_secs_f64(estimated_secs))
    }

    /// Get the current processing rate (items per second).
    #[must_use]
    pub fn processing_rate(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.processed_count as f32 / elapsed
        } else {
            0.0
        }
    }

    /// Finalize processing and return results.
    #[must_use]
    pub fn finalize(self) -> BatchResult<R> {
        BatchResult {
            successes: self.successes,
            failures: self.failures,
            duration: self.start_time.elapsed(),
            batch_count: self.total_count.div_ceil(self.config.batch_size),
        }
    }
}

/// Adaptive batch size calculator based on processing performance.
pub struct AdaptiveBatchSize {
    min_size: usize,
    max_size: usize,
    current_size: usize,
    target_duration: Duration,
    recent_durations: Vec<Duration>,
    max_history: usize,
}

impl AdaptiveBatchSize {
    /// Create a new adaptive batch size calculator.
    ///
    /// # Arguments
    ///
    /// * `min_size` - Minimum batch size
    /// * `max_size` - Maximum batch size
    /// * `initial_size` - Initial batch size
    /// * `target_duration` - Target duration per batch
    #[must_use]
    pub fn new(
        min_size: usize,
        max_size: usize,
        initial_size: usize,
        target_duration: Duration,
    ) -> Self {
        assert!(min_size > 0, "Minimum batch size must be greater than 0");
        assert!(max_size >= min_size, "Maximum size must be >= minimum size");
        assert!(
            initial_size >= min_size && initial_size <= max_size,
            "Initial size must be within min and max bounds"
        );

        Self {
            min_size,
            max_size,
            current_size: initial_size,
            target_duration,
            recent_durations: Vec::new(),
            max_history: 10,
        }
    }

    /// Update with the duration of the last batch processing.
    ///
    /// Returns the recommended batch size for the next batch.
    pub fn update(&mut self, duration: Duration) -> usize {
        self.recent_durations.push(duration);

        // Keep only recent history
        if self.recent_durations.len() > self.max_history {
            self.recent_durations.remove(0);
        }

        // Calculate average duration
        let avg_duration = if self.recent_durations.is_empty() {
            duration
        } else {
            let total: Duration = self.recent_durations.iter().sum();
            total / self.recent_durations.len() as u32
        };

        // Adjust batch size based on performance
        if avg_duration < self.target_duration {
            // Processing is fast, increase batch size
            self.current_size = ((self.current_size as f32 * 1.2) as usize).min(self.max_size);
        } else if avg_duration > self.target_duration * 2 {
            // Processing is slow, decrease batch size
            self.current_size = ((self.current_size as f32 * 0.8) as usize).max(self.min_size);
        }

        self.current_size
    }

    /// Get the current recommended batch size.
    #[must_use]
    pub fn current_size(&self) -> usize {
        self.current_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_retries, 3);
        assert!(!config.parallel);
    }

    #[test]
    fn test_process_batches_success() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let config = BatchConfig {
            batch_size: 3,
            ..Default::default()
        };

        let result = process_batches(items, config, |batch| {
            Ok(batch.iter().map(|&x| x * 2).collect())
        });

        assert_eq!(result.successes.len(), 10);
        assert_eq!(result.failures.len(), 0);
        assert_eq!(result.batch_count, 4); // 3 + 3 + 3 + 1
    }

    #[test]
    fn test_process_batches_with_failures() {
        let items = vec![1, 2, 3, 4, 5];
        let config = BatchConfig {
            batch_size: 2,
            max_retries: 1,
            ..Default::default()
        };

        let result = process_batches(items, config, |batch| {
            if batch[0] == 3 {
                Err("Error processing batch".to_string())
            } else {
                Ok(batch.iter().map(|&x| x * 2).collect())
            }
        });

        // Batches: [1,2], [3,4], [5]
        // [1,2] succeeds (2 items), [3,4] fails (0 items), [5] succeeds (1 item)
        assert_eq!(result.successes.len(), 3); // Items 1,2,5 processed successfully
        assert_eq!(result.failures.len(), 2); // Items 3,4 failed
    }

    #[test]
    fn test_chunk_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let mut chunks = ChunkIterator::new(data, 3);

        assert_eq!(chunks.remaining(), 7);
        assert_eq!(chunks.next(), Some(vec![1, 2, 3]));
        assert_eq!(chunks.remaining(), 4);
        assert_eq!(chunks.next(), Some(vec![4, 5, 6]));
        assert_eq!(chunks.remaining(), 1);
        assert_eq!(chunks.next(), Some(vec![7]));
        assert_eq!(chunks.remaining(), 0);
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::<i32, i32>::new(10, BatchConfig::default());

        let batch1 = vec![1, 2, 3, 4, 5];
        processor
            .process_batch(&batch1, |batch| Ok(batch.iter().map(|&x| x * 2).collect()))
            .unwrap();

        assert_eq!(processor.progress_percent(), 50.0);
        assert!(processor.processing_rate() > 0.0);

        let batch2 = vec![6, 7, 8, 9, 10];
        processor
            .process_batch(&batch2, |batch| Ok(batch.iter().map(|&x| x * 2).collect()))
            .unwrap();

        assert_eq!(processor.progress_percent(), 100.0);

        let result = processor.finalize();
        assert_eq!(result.successes.len(), 10);
        assert_eq!(result.failures.len(), 0);
    }

    #[test]
    fn test_adaptive_batch_size() {
        let mut adaptive = AdaptiveBatchSize::new(10, 1_000, 100, Duration::from_millis(100));

        assert_eq!(adaptive.current_size(), 100);

        // Fast processing - should increase
        let new_size = adaptive.update(Duration::from_millis(50));
        assert!(new_size > 100);

        let size_after_increase = adaptive.current_size();

        // Slow processing - should decrease significantly
        // Need to add multiple slow samples to trigger reduction
        adaptive.update(Duration::from_millis(500));
        adaptive.update(Duration::from_millis(500));
        let new_size = adaptive.update(Duration::from_millis(500));
        assert!(
            new_size < size_after_increase,
            "Expected size {} to be less than {}",
            new_size,
            size_after_increase
        );
    }

    #[test]
    #[should_panic(expected = "Chunk size must be greater than 0")]
    fn test_chunk_iterator_zero_size() {
        ChunkIterator::new(vec![1, 2, 3], 0);
    }

    #[test]
    #[should_panic(expected = "Minimum batch size must be greater than 0")]
    fn test_adaptive_batch_size_zero_min() {
        AdaptiveBatchSize::new(0, 100, 50, Duration::from_millis(100));
    }
}
