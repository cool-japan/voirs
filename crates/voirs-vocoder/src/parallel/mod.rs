//! Parallel processing utilities for voirs-vocoder
//!
//! Provides Rayon-based parallelization, work-stealing queues,
//! load balancing strategies, and NUMA-aware processing.

use crate::{Result, VocoderError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub mod load_balancer;
pub mod numa_aware;
pub mod thread_pool;
pub mod work_stealing;

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_threads: Option<usize>,
    pub enable_work_stealing: bool,
    pub enable_numa_aware: bool,
    pub chunk_size: usize,
    pub max_queue_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use default thread count
            enable_work_stealing: true,
            enable_numa_aware: cfg!(target_os = "linux"),
            chunk_size: 1024,
            max_queue_size: 10000,
        }
    }
}

/// Parallel processing statistics
#[derive(Debug)]
pub struct ParallelStats {
    pub tasks_submitted: AtomicUsize,
    pub tasks_completed: AtomicUsize,
    pub tasks_failed: AtomicUsize,
    pub work_stolen: AtomicUsize,
    pub load_balancing_events: AtomicUsize,
    pub numa_migrations: AtomicUsize,
}

impl Default for ParallelStats {
    fn default() -> Self {
        Self {
            tasks_submitted: AtomicUsize::new(0),
            tasks_completed: AtomicUsize::new(0),
            tasks_failed: AtomicUsize::new(0),
            work_stolen: AtomicUsize::new(0),
            load_balancing_events: AtomicUsize::new(0),
            numa_migrations: AtomicUsize::new(0),
        }
    }
}

impl Clone for ParallelStats {
    fn clone(&self) -> Self {
        Self {
            tasks_submitted: AtomicUsize::new(
                self.tasks_submitted
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            tasks_completed: AtomicUsize::new(
                self.tasks_completed
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            tasks_failed: AtomicUsize::new(
                self.tasks_failed.load(std::sync::atomic::Ordering::Relaxed),
            ),
            work_stolen: AtomicUsize::new(
                self.work_stolen.load(std::sync::atomic::Ordering::Relaxed),
            ),
            load_balancing_events: AtomicUsize::new(
                self.load_balancing_events
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            numa_migrations: AtomicUsize::new(
                self.numa_migrations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// Main parallel processor
pub struct ParallelProcessor {
    config: ParallelConfig,
    stats: Arc<ParallelStats>,
    thread_pool: Arc<thread_pool::ThreadPool>,
}

impl ParallelProcessor {
    /// Create new parallel processor
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let stats = Arc::new(ParallelStats::default());
        let thread_pool = Arc::new(thread_pool::ThreadPool::new(config.clone())?);

        Ok(Self {
            config,
            stats,
            thread_pool,
        })
    }

    /// Process audio samples in parallel
    pub fn process_audio_parallel<F>(&self, samples: &mut [f32], processor: F) -> Result<()>
    where
        F: Fn(&mut [f32]) + Send + Sync + Clone + 'static,
    {
        if samples.len() < self.config.chunk_size {
            // Too small for parallel processing
            self.stats.tasks_submitted.fetch_add(1, Ordering::SeqCst);
            processor(samples);
            self.stats.tasks_completed.fetch_add(1, Ordering::SeqCst);
            return Ok(());
        }

        self.stats.tasks_submitted.fetch_add(1, Ordering::SeqCst);

        let chunks: Vec<_> = samples.chunks_mut(self.config.chunk_size).collect();
        let _num_chunks = chunks.len();

        // Use rayon for parallel processing
        use rayon::prelude::*;

        let result = chunks.into_par_iter().try_for_each(|chunk| {
            processor(chunk);
            Ok::<(), VocoderError>(())
        });

        match result {
            Ok(_) => {
                self.stats.tasks_completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                self.stats.tasks_failed.fetch_add(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    /// Process mel spectrograms in parallel
    pub fn process_mel_parallel<F>(&self, mels: &mut [Vec<f32>], processor: F) -> Result<()>
    where
        F: Fn(&mut [f32]) + Send + Sync + Clone + 'static,
    {
        self.stats.tasks_submitted.fetch_add(1, Ordering::SeqCst);

        use rayon::prelude::*;

        let result = mels.par_iter_mut().try_for_each(|mel| {
            processor(mel);
            Ok::<(), VocoderError>(())
        });

        match result {
            Ok(_) => {
                self.stats.tasks_completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                self.stats.tasks_failed.fetch_add(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    /// Process batch of items with load balancing
    pub fn process_batch<T, F>(&self, items: Vec<T>, processor: F) -> Result<Vec<T>>
    where
        T: Send + 'static,
        F: Fn(T) -> Result<T> + Send + Sync + Clone + 'static,
    {
        self.stats.tasks_submitted.fetch_add(1, Ordering::SeqCst);

        use rayon::prelude::*;

        let results: Result<Vec<T>> = items.into_par_iter().map(processor).collect();

        match results {
            Ok(processed_items) => {
                self.stats.tasks_completed.fetch_add(1, Ordering::SeqCst);
                Ok(processed_items)
            }
            Err(e) => {
                self.stats.tasks_failed.fetch_add(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    /// Get processing statistics
    pub fn stats(&self) -> ParallelStats {
        self.stats.as_ref().clone()
    }

    /// Get current thread pool utilization
    pub fn thread_utilization(&self) -> f32 {
        self.thread_pool.utilization()
    }

    /// Adjust thread pool size dynamically
    pub fn adjust_thread_count(&self, new_count: usize) -> Result<()> {
        self.thread_pool.resize(new_count)
    }
}

/// Parallel map operation with optimized chunking
pub fn parallel_map<T, U, F>(input: &[T], chunk_size: usize, mapper: F) -> Vec<U>
where
    T: Send + Sync,
    U: Send,
    F: Fn(&T) -> U + Send + Sync,
{
    use rayon::prelude::*;

    let len = input.len();

    // Use sequential processing for small inputs
    if len < chunk_size.max(100) {
        return input.iter().map(mapper).collect();
    }

    // Use adaptive chunking for better load balancing
    let num_cpus = rayon::current_num_threads();
    let optimal_chunk_size = (len / (num_cpus * 4)).max(chunk_size).min(len / 2);

    if optimal_chunk_size > chunk_size {
        // Use chunked parallel processing for better load distribution
        input
            .par_chunks(optimal_chunk_size)
            .flat_map(|chunk| chunk.par_iter().map(&mapper))
            .collect()
    } else {
        // Use standard parallel iterator for smaller chunks
        input.par_iter().map(mapper).collect()
    }
}

/// Parallel reduce operation
pub fn parallel_reduce<T, F>(input: &[T], chunk_size: usize, reducer: F) -> Option<T>
where
    T: Send + Sync + Clone,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    use rayon::prelude::*;

    if input.is_empty() {
        return None;
    }

    if input.len() < chunk_size {
        return input.iter().cloned().reduce(|a, b| reducer(&a, &b));
    }

    input
        .par_iter()
        .cloned()
        .reduce_with(|a, b| reducer(&a, &b))
}

/// Parallel filter operation
pub fn parallel_filter<T, F>(input: &[T], chunk_size: usize, predicate: F) -> Vec<T>
where
    T: Send + Sync + Clone,
    F: Fn(&T) -> bool + Send + Sync,
{
    use rayon::prelude::*;

    if input.len() < chunk_size {
        return input
            .iter()
            .filter(|item| predicate(item))
            .cloned()
            .collect();
    }

    input
        .par_iter()
        .filter(|item| predicate(item))
        .cloned()
        .collect()
}

/// Parallel convolution operation
pub fn parallel_convolve(signal: &[f32], kernel: &[f32], output: &mut [f32], chunk_size: usize) {
    use rayon::prelude::*;

    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let output_len = output.len();

    if output_len < chunk_size {
        // Fall back to sequential convolution
        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..kernel_len {
                if i + j < signal_len {
                    sum += signal[i + j] * kernel[j];
                }
            }
            output[i] = sum;
        }
        return;
    }

    // Parallel convolution by chunks
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start_idx = chunk_idx * chunk_size;

            for (i, output_sample) in chunk.iter_mut().enumerate() {
                let signal_idx = start_idx + i;
                let mut sum = 0.0;

                for j in 0..kernel_len {
                    if signal_idx + j < signal_len {
                        sum += signal[signal_idx + j] * kernel[j];
                    }
                }

                *output_sample = sum;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_processor_creation() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let stats = processor.stats();
        assert_eq!(stats.tasks_submitted.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_parallel_audio_processing() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        processor
            .process_audio_parallel(&mut samples, |chunk| {
                for sample in chunk {
                    *sample *= 2.0;
                }
            })
            .unwrap();

        assert_eq!(samples, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_parallel_mel_processing() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let mut mels = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        processor
            .process_mel_parallel(&mut mels, |mel| {
                for value in mel {
                    *value += 1.0;
                }
            })
            .unwrap();

        assert_eq!(
            mels,
            vec![
                vec![2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![8.0, 9.0, 10.0],
            ]
        );
    }

    #[test]
    fn test_parallel_batch_processing() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let items = vec![1, 2, 3, 4, 5];

        let results = processor.process_batch(items, |item| Ok(item * 2)).unwrap();

        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_map() {
        let input = vec![1, 2, 3, 4, 5];
        let output = parallel_map(&input, 2, |x| x * 2);

        assert_eq!(output, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_reduce() {
        let input = vec![1, 2, 3, 4, 5];
        let sum = parallel_reduce(&input, 2, |a, b| a + b);

        assert_eq!(sum, Some(15));
    }

    #[test]
    fn test_parallel_filter() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let evens = parallel_filter(&input, 3, |x| *x % 2 == 0);

        assert_eq!(evens, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_convolve() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; 5];

        parallel_convolve(&signal, &kernel, &mut output, 2);

        // Check that output is not all zeros (convolution worked)
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_processor_stats() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let mut samples = vec![1.0; 100];

        processor
            .process_audio_parallel(&mut samples, |chunk| {
                for sample in chunk {
                    *sample *= 2.0;
                }
            })
            .unwrap();

        let stats = processor.stats();
        assert_eq!(stats.tasks_submitted.load(Ordering::SeqCst), 1);
        assert_eq!(stats.tasks_completed.load(Ordering::SeqCst), 1);
        assert_eq!(stats.tasks_failed.load(Ordering::SeqCst), 0);
    }
}
