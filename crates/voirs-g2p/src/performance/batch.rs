//! Batch processing utilities for high-throughput G2P conversion

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{G2p, LanguageCode, Phoneme, Result};
use serde::Serialize;
use tracing::{debug, info};

/// Batch processing utilities for high-throughput G2P conversion
pub struct BatchProcessor;

impl BatchProcessor {
    /// Process multiple texts in parallel
    pub async fn process_batch<T: crate::G2p + Send + Sync + 'static>(
        backend: Arc<T>,
        texts: Vec<String>,
        language: Option<LanguageCode>,
        max_concurrent: usize,
    ) -> Result<Vec<Result<Vec<Phoneme>>>> {
        use tokio::task;

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::new();

        for text in texts {
            let backend = backend.clone();
            let semaphore = semaphore.clone();

            let handle = task::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                backend.to_phonemes(&text, language).await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        Ok(results)
    }

    /// Process texts with different languages in parallel
    pub async fn process_multilingual_batch<T: crate::G2p + Send + Sync + 'static>(
        backend: Arc<T>,
        texts_with_langs: Vec<(String, Option<LanguageCode>)>,
        max_concurrent: usize,
    ) -> Result<Vec<Result<Vec<Phoneme>>>> {
        use tokio::task;

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::new();

        for (text, lang) in texts_with_langs {
            let backend = backend.clone();
            let semaphore = semaphore.clone();

            let handle = task::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                backend.to_phonemes(&text, lang).await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        Ok(results)
    }
}

/// Memory-efficient batch phoneme processor
pub struct BatchPhonemeProcessor {
    /// Maximum batch size to prevent memory spikes
    max_batch_size: usize,
    /// Memory pool for reusable allocations  
    memory_pool: Arc<Mutex<Vec<Vec<Vec<Phoneme>>>>>,
    /// Processing statistics
    stats: Arc<Mutex<BatchProcessingStats>>,
}

/// Statistics for batch processing operations
#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchProcessingStats {
    /// Total batches processed
    pub batches_processed: u64,
    /// Total phonemes processed
    pub phonemes_processed: u64,
    /// Average processing time per batch
    pub avg_batch_time_ms: f32,
    /// Memory efficiency ratio
    pub memory_efficiency: f32,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
}

impl BatchPhonemeProcessor {
    /// Create new batch processor with specified configuration
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            memory_pool: Arc::new(Mutex::new(Vec::<Vec<Vec<Phoneme>>>::with_capacity(10))),
            stats: Arc::new(Mutex::new(BatchProcessingStats::default())),
        }
    }

    /// Process multiple phoneme sequences with memory optimization
    pub async fn process_batch<G: G2p>(
        &self,
        g2p: &G,
        texts: &[String],
        language: Option<LanguageCode>,
    ) -> Result<Vec<Vec<Phoneme>>> {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(texts.len());
        let mut total_phonemes = 0;

        // Process in chunks to avoid memory spikes
        for chunk in texts.chunks(self.max_batch_size) {
            let mut chunk_results = self.get_reusable_vector();
            chunk_results.clear();
            chunk_results.reserve(chunk.len());

            for text in chunk {
                let phonemes = g2p.to_phonemes(text, language).await?;
                total_phonemes += phonemes.len();
                chunk_results.push(phonemes);
            }

            results.append(&mut chunk_results);
            self.return_reusable_vector(chunk_results);
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        self.update_stats(1, total_phonemes, processing_time);

        Ok(results)
    }

    /// Get a reusable vector from the memory pool
    fn get_reusable_vector(&self) -> Vec<Vec<Phoneme>> {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.pop()
            .unwrap_or_else(|| Vec::<Vec<Phoneme>>::with_capacity(self.max_batch_size))
    }

    /// Return a vector to the memory pool for reuse
    fn return_reusable_vector(&self, mut vec: Vec<Vec<Phoneme>>) {
        vec.clear();
        if vec.capacity() > 0 {
            let mut pool = self.memory_pool.lock().unwrap();
            if pool.len() < 10 {
                // Limit pool size to prevent unbounded growth
                pool.push(vec);
            }
        }
    }

    /// Update processing statistics
    fn update_stats(&self, batches: u64, phonemes: usize, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.batches_processed += batches;
        stats.phonemes_processed += phonemes as u64;

        let batch_time_ms = duration.as_millis() as f32;
        if stats.batches_processed == 1 {
            stats.avg_batch_time_ms = batch_time_ms;
        } else {
            stats.avg_batch_time_ms =
                (stats.avg_batch_time_ms * (stats.batches_processed - 1) as f32 + batch_time_ms)
                    / stats.batches_processed as f32;
        }

        // Calculate memory efficiency (phonemes per MB)
        let estimated_memory_mb = (phonemes * std::mem::size_of::<Phoneme>()) as f32 / 1_048_576.0;
        if estimated_memory_mb > 0.0 {
            stats.memory_efficiency = phonemes as f32 / estimated_memory_mb;
        }
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> BatchProcessingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset processing statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = BatchProcessingStats::default();
    }

    /// Clear memory pool to free unused memory
    pub fn clear_memory_pool(&self) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.clear();
    }
}

impl BatchProcessingStats {
    /// Get average phonemes processed per second
    pub fn phonemes_per_second(&self) -> f32 {
        if self.avg_batch_time_ms > 0.0 {
            (self.phonemes_processed as f32) / (self.avg_batch_time_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Get memory efficiency description
    pub fn memory_efficiency_description(&self) -> String {
        format!("{:.2} phonemes per MB", self.memory_efficiency)
    }
}

/// Dynamic batch processor for variable-length sequences
pub struct DynamicBatchProcessor {
    /// Target memory usage per batch (in bytes)
    target_batch_memory: usize,
    /// Minimum batch size to ensure efficiency
    min_batch_size: usize,
    /// Maximum batch size to prevent memory spikes
    max_batch_size: usize,
    /// Length-based grouping threshold
    length_grouping_threshold: usize,
    /// Processing statistics
    stats: Arc<Mutex<DynamicBatchStats>>,
}

/// Statistics for dynamic batch processing
#[derive(Debug, Clone, Default, Serialize)]
pub struct DynamicBatchStats {
    /// Total adaptive batches processed
    pub adaptive_batches_processed: u64,
    /// Average batch efficiency score
    pub avg_batch_efficiency: f32,
    /// Length distribution histogram
    pub length_distribution: HashMap<usize, u64>,
    /// Memory utilization efficiency
    pub memory_utilization: f32,
    /// Average sequence length variance within batches
    pub avg_length_variance: f32,
    /// Total processing time across all batches
    pub total_processing_time: Duration,
}

/// Batch with sequences grouped by similar lengths
#[derive(Debug)]
pub struct LengthGroupedBatch {
    /// Texts grouped by approximate length
    pub texts: Vec<String>,
    /// Original indices in the input
    pub original_indices: Vec<usize>,
    /// Average length of sequences in this batch
    pub avg_length: f32,
    /// Estimated memory requirement
    pub estimated_memory: usize,
}

impl DynamicBatchProcessor {
    /// Create new dynamic batch processor
    pub fn new(target_batch_memory: usize, min_batch_size: usize, max_batch_size: usize) -> Self {
        Self {
            target_batch_memory,
            min_batch_size,
            max_batch_size,
            length_grouping_threshold: 10, // Group sequences within 10 chars of each other
            stats: Arc::new(Mutex::new(DynamicBatchStats::default())),
        }
    }

    /// Process texts with dynamic batching based on sequence lengths
    pub async fn process_adaptive_batch<G: G2p>(
        &self,
        g2p: &G,
        texts: &[String],
        language: Option<LanguageCode>,
    ) -> Result<Vec<Vec<Phoneme>>> {
        info!(
            "Starting adaptive batch processing for {} texts",
            texts.len()
        );
        let start_time = Instant::now();

        // Group texts by similar lengths for optimal batching
        let length_groups = self.group_by_length(texts);
        info!(
            "Created {} length groups for adaptive processing",
            length_groups.len()
        );

        // Create adaptive batches
        let batches = self.create_adaptive_batches(length_groups);
        debug!("Generated {} adaptive batches", batches.len());
        let batches_count = batches.len();

        // Process each batch
        let mut all_results = vec![Vec::new(); texts.len()];
        let mut total_efficiency = 0.0;

        for batch in batches {
            let batch_start = Instant::now();
            let batch_results = self
                .process_length_grouped_batch(g2p, &batch, language)
                .await?;
            let batch_time = batch_start.elapsed();

            // Map results back to original indices
            for (result, &original_idx) in batch_results.iter().zip(&batch.original_indices) {
                all_results[original_idx] = result.clone();
            }

            // Calculate batch efficiency
            let efficiency = self.calculate_batch_efficiency(&batch, batch_time);
            total_efficiency += efficiency;

            debug!(
                "Processed batch with {} texts, avg_length={:.1}, efficiency={:.2}",
                batch.texts.len(),
                batch.avg_length,
                efficiency
            );
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        self.update_dynamic_stats(batches_count, texts, total_efficiency, processing_time);

        info!(
            "Adaptive batch processing completed in {:.2}ms for {} texts",
            processing_time.as_millis(),
            texts.len()
        );

        Ok(all_results)
    }

    /// Group texts by similar lengths for optimal batching
    fn group_by_length<'a>(&self, texts: &'a [String]) -> Vec<Vec<(usize, &'a String)>> {
        let mut length_map: HashMap<usize, Vec<(usize, &String)>> = HashMap::new();

        // Group by length ranges
        for (idx, text) in texts.iter().enumerate() {
            let length_bucket =
                (text.len() / self.length_grouping_threshold) * self.length_grouping_threshold;
            length_map
                .entry(length_bucket)
                .or_default()
                .push((idx, text));
        }

        // Sort groups by length for better processing order
        let mut groups: Vec<_> = length_map.into_values().collect();
        groups.sort_by_key(|group| group.first().map(|(_, text)| text.len()).unwrap_or(0));

        groups
    }

    /// Create adaptive batches from length groups
    fn create_adaptive_batches(
        &self,
        length_groups: Vec<Vec<(usize, &String)>>,
    ) -> Vec<LengthGroupedBatch> {
        let mut batches = Vec::new();

        for group in length_groups {
            let mut current_batch_texts = Vec::new();
            let mut current_batch_indices = Vec::new();
            let mut current_memory = 0;

            for (idx, text) in group {
                let text_memory = self.estimate_text_memory(text);

                // Check if adding this text would exceed memory or size limits
                if (current_memory + text_memory > self.target_batch_memory
                    && current_batch_texts.len() >= self.min_batch_size)
                    || current_batch_texts.len() >= self.max_batch_size
                {
                    // Create batch from current texts
                    if !current_batch_texts.is_empty() {
                        batches.push(self.create_length_grouped_batch(
                            current_batch_texts,
                            current_batch_indices,
                            current_memory,
                        ));
                        current_batch_texts = Vec::new();
                        current_batch_indices = Vec::new();
                        current_memory = 0;
                    }
                }

                current_batch_texts.push(text.clone());
                current_batch_indices.push(idx);
                current_memory += text_memory;
            }

            // Add remaining texts as a batch
            if !current_batch_texts.is_empty() {
                batches.push(self.create_length_grouped_batch(
                    current_batch_texts,
                    current_batch_indices,
                    current_memory,
                ));
            }
        }

        batches
    }

    /// Create a length-grouped batch from texts and indices
    fn create_length_grouped_batch(
        &self,
        texts: Vec<String>,
        original_indices: Vec<usize>,
        estimated_memory: usize,
    ) -> LengthGroupedBatch {
        let avg_length = if texts.is_empty() {
            0.0
        } else {
            texts.iter().map(|t| t.len()).sum::<usize>() as f32 / texts.len() as f32
        };

        LengthGroupedBatch {
            texts,
            original_indices,
            avg_length,
            estimated_memory,
        }
    }

    /// Process a single length-grouped batch
    async fn process_length_grouped_batch<G: G2p>(
        &self,
        g2p: &G,
        batch: &LengthGroupedBatch,
        language: Option<LanguageCode>,
    ) -> Result<Vec<Vec<Phoneme>>> {
        let mut results = Vec::with_capacity(batch.texts.len());

        for text in &batch.texts {
            let phonemes = g2p.to_phonemes(text, language).await?;
            results.push(phonemes);
        }

        Ok(results)
    }

    /// Estimate memory requirement for a text
    fn estimate_text_memory(&self, text: &str) -> usize {
        // Rough estimate: text length * average phonemes per character * phoneme size
        let avg_phonemes_per_char = 1.2; // Empirical average
        let phoneme_size = std::mem::size_of::<Phoneme>();
        (text.len() as f32 * avg_phonemes_per_char * phoneme_size as f32) as usize
    }

    /// Calculate efficiency score for a batch
    fn calculate_batch_efficiency(
        &self,
        batch: &LengthGroupedBatch,
        processing_time: Duration,
    ) -> f32 {
        let time_efficiency = 1.0 / (processing_time.as_millis() as f32 + 1.0);
        let memory_efficiency = batch.texts.len() as f32 / (batch.estimated_memory as f32 / 1024.0);
        let length_variance = self.calculate_length_variance(&batch.texts);
        let variance_efficiency = 1.0 / (length_variance + 1.0);

        (time_efficiency + memory_efficiency + variance_efficiency) / 3.0
    }

    /// Calculate length variance within a batch
    fn calculate_length_variance(&self, texts: &[String]) -> f32 {
        if texts.len() <= 1 {
            return 0.0;
        }

        let mean = texts.iter().map(|t| t.len()).sum::<usize>() as f32 / texts.len() as f32;
        let variance = texts
            .iter()
            .map(|t| (t.len() as f32 - mean).powi(2))
            .sum::<f32>()
            / texts.len() as f32;

        variance.sqrt()
    }

    /// Update dynamic processing statistics
    fn update_dynamic_stats(
        &self,
        batches_count: usize,
        texts: &[String],
        total_efficiency: f32,
        processing_time: Duration,
    ) {
        let mut stats = self.stats.lock().unwrap();

        stats.adaptive_batches_processed += batches_count as u64;

        // Update average efficiency
        let new_efficiency = total_efficiency / batches_count as f32;
        if stats.adaptive_batches_processed == batches_count as u64 {
            stats.avg_batch_efficiency = new_efficiency;
        } else {
            let prev_total = stats.adaptive_batches_processed - batches_count as u64;
            stats.avg_batch_efficiency = (stats.avg_batch_efficiency * prev_total as f32
                + total_efficiency)
                / stats.adaptive_batches_processed as f32;
        }

        // Update length distribution
        for text in texts {
            let length_bucket = (text.len() / 10) * 10; // Group in buckets of 10
            *stats.length_distribution.entry(length_bucket).or_insert(0) += 1;
        }

        // Calculate memory utilization
        let total_memory_used = texts
            .iter()
            .map(|t| self.estimate_text_memory(t))
            .sum::<usize>();
        let target_memory = self.target_batch_memory * batches_count;
        stats.memory_utilization = if target_memory > 0 {
            total_memory_used as f32 / target_memory as f32
        } else {
            0.0
        };

        // Calculate average length variance
        stats.avg_length_variance = self.calculate_length_variance(texts);

        // Update total processing time
        stats.total_processing_time += processing_time;
    }

    /// Get current dynamic processing statistics
    pub fn get_dynamic_stats(&self) -> DynamicBatchStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset dynamic processing statistics
    pub fn reset_dynamic_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = DynamicBatchStats::default();
    }

    /// Configure length grouping threshold
    pub fn set_length_grouping_threshold(&mut self, threshold: usize) {
        self.length_grouping_threshold = threshold;
        info!("Updated length grouping threshold to {}", threshold);
    }

    /// Configure target batch memory
    pub fn set_target_batch_memory(&mut self, memory_bytes: usize) {
        self.target_batch_memory = memory_bytes;
        info!("Updated target batch memory to {} bytes", memory_bytes);
    }
}

impl DynamicBatchStats {
    /// Get processing efficiency description
    pub fn efficiency_description(&self) -> String {
        format!(
            "Avg efficiency: {:.2}, Memory utilization: {:.2}%",
            self.avg_batch_efficiency,
            self.memory_utilization * 100.0
        )
    }

    /// Get length distribution summary
    pub fn length_distribution_summary(&self) -> String {
        let mut summary = String::new();
        let mut sorted_lengths: Vec<_> = self.length_distribution.iter().collect();
        sorted_lengths.sort_by_key(|(length, _)| *length);

        for (length, count) in sorted_lengths.iter().take(5) {
            summary.push_str(&format!("{length}chars: {count} texts, "));
        }

        if sorted_lengths.len() > 5 {
            summary.push_str("...");
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DummyG2p;

    #[tokio::test]
    async fn test_batch_processing() {
        let backend = Arc::new(DummyG2p::new());
        let texts = vec!["hello".to_string(), "world".to_string()];

        let results = BatchProcessor::process_batch(backend, texts, Some(LanguageCode::EnUs), 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[tokio::test]
    async fn test_multilingual_batch_processing() {
        let backend = Arc::new(DummyG2p::new());
        let texts = vec![
            ("hello".to_string(), Some(LanguageCode::EnUs)),
            ("world".to_string(), Some(LanguageCode::EnUs)),
        ];

        let results = BatchProcessor::process_multilingual_batch(backend, texts, 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[tokio::test]
    async fn test_batch_phoneme_processor() {
        let processor = BatchPhonemeProcessor::new(100);
        let backend = DummyG2p::new();
        let texts = vec!["hello".to_string(), "world".to_string()];

        let results = processor
            .process_batch(&backend, &texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);

        let stats = processor.get_stats();
        assert_eq!(stats.batches_processed, 1);
        assert!(stats.phonemes_processed > 0);
    }

    #[tokio::test]
    async fn test_batch_processor_memory_pool() {
        let processor = BatchPhonemeProcessor::new(2);
        let backend = DummyG2p::new();
        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        // Process batch to populate memory pool
        let _ = processor
            .process_batch(&backend, &texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        // Process again to test memory pool reuse
        let _ = processor
            .process_batch(&backend, &texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        let stats = processor.get_stats();
        assert_eq!(stats.batches_processed, 2);
    }

    #[test]
    fn test_batch_processing_stats() {
        let stats = BatchProcessingStats {
            batches_processed: 10,
            phonemes_processed: 1000,
            avg_batch_time_ms: 100000.0, // 100 seconds = 100,000 ms
            memory_efficiency: 500.0,
            peak_memory_bytes: 1024,
        };

        assert_eq!(stats.phonemes_per_second(), 10.0);
        assert_eq!(
            stats.memory_efficiency_description(),
            "500.00 phonemes per MB"
        );
    }

    #[tokio::test]
    async fn test_dynamic_batch_processor() {
        let processor = DynamicBatchProcessor::new(1024, 2, 10);
        let backend = DummyG2p::new();

        // Test with texts of varying lengths
        let texts = vec![
            "hi".to_string(),                                                  // 2 chars
            "hello".to_string(),                                               // 5 chars
            "hello world".to_string(),                                         // 11 chars
            "a".to_string(),                                                   // 1 char
            "testing dynamic batching with longer text sequences".to_string(), // 52 chars
        ];

        let results = processor
            .process_adaptive_batch(&backend, &texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        assert_eq!(results.len(), 5);

        // Check that results are in correct order
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty(), "Result {i} should not be empty");
        }

        let stats = processor.get_dynamic_stats();
        assert!(stats.adaptive_batches_processed > 0);
        assert!(stats.avg_batch_efficiency > 0.0);
        assert!(!stats.length_distribution.is_empty());
    }

    #[tokio::test]
    async fn test_dynamic_batch_length_grouping() {
        let processor = DynamicBatchProcessor::new(2048, 1, 5);
        let backend = DummyG2p::new();

        // Create texts with similar lengths that should be grouped together
        let texts = vec![
            "abc".to_string(),                 // ~3 chars
            "def".to_string(),                 // ~3 chars
            "ghi".to_string(),                 // ~3 chars
            "hello world test".to_string(),    // ~17 chars
            "another longer text".to_string(), // ~20 chars
            "xyz".to_string(),                 // ~3 chars
        ];

        let results = processor
            .process_adaptive_batch(&backend, &texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        assert_eq!(results.len(), 6);

        let stats = processor.get_dynamic_stats();

        // Should have processed multiple batches due to length grouping
        assert!(stats.adaptive_batches_processed >= 1);

        // Should have length distribution data
        assert!(!stats.length_distribution.is_empty());

        // Memory utilization should be reasonable
        assert!(stats.memory_utilization >= 0.0);
        assert!(stats.memory_utilization <= 2.0); // Allow for some overhead
    }

    #[test]
    fn test_dynamic_batch_stats() {
        let stats = DynamicBatchStats {
            adaptive_batches_processed: 5,
            avg_batch_efficiency: 0.75,
            length_distribution: {
                let mut map = HashMap::new();
                map.insert(0, 3); // 3 texts of 0-9 chars
                map.insert(10, 2); // 2 texts of 10-19 chars
                map
            },
            memory_utilization: 0.85,
            avg_length_variance: 2.5,
            total_processing_time: Duration::from_millis(150),
        };

        let efficiency_desc = stats.efficiency_description();
        assert!(efficiency_desc.contains("0.75"));
        assert!(efficiency_desc.contains("85.00%"));

        let length_summary = stats.length_distribution_summary();
        assert!(length_summary.contains("0chars: 3 texts"));
        assert!(length_summary.contains("10chars: 2 texts"));
    }

    #[test]
    fn test_dynamic_batch_processor_configuration() {
        let mut processor = DynamicBatchProcessor::new(1024, 2, 10);

        // Test configuration updates
        processor.set_length_grouping_threshold(15);
        processor.set_target_batch_memory(2048);

        // Verify the processor still works after configuration changes
        assert_eq!(processor.length_grouping_threshold, 15);
        assert_eq!(processor.target_batch_memory, 2048);
    }
}
