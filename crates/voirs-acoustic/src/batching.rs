//! Dynamic batching system for variable-length sequences
//!
//! This module provides efficient batching strategies for variable-length input sequences,
//! optimizing memory usage and computational efficiency for real-time synthesis.

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::time::{Duration, Instant};

use crate::{AcousticError, Phoneme, Result};

/// Configuration for dynamic batching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchConfig {
    /// Maximum batch size (number of sequences)
    pub max_batch_size: usize,
    /// Maximum sequence length in a batch
    pub max_sequence_length: usize,
    /// Maximum waiting time before forcing a batch
    pub max_wait_time_ms: u64,
    /// Target batch utilization (0.0 to 1.0)
    pub target_utilization: f32,
    /// Whether to sort sequences by length
    pub sort_by_length: bool,
    /// Padding strategy for variable-length sequences
    pub padding_strategy: PaddingStrategy,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 1000,
            max_wait_time_ms: 50,
            target_utilization: 0.8,
            sort_by_length: true,
            padding_strategy: PaddingStrategy::ZeroPadding,
            memory_optimization: MemoryOptimization::Balanced,
        }
    }
}

/// Padding strategies for variable-length sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Zero padding (default)
    ZeroPadding,
    /// Repeat last token
    RepeatLast,
    /// Use special padding token
    PaddingToken(u32),
    /// Minimum necessary padding
    MinimalPadding,
}

/// Memory optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimization {
    /// Maximize speed, may use more memory
    Speed,
    /// Balance speed and memory usage
    Balanced,
    /// Minimize memory usage
    Memory,
}

/// Statistics for batch processing
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total number of batches processed
    pub total_batches: usize,
    /// Total number of sequences processed
    pub total_sequences: usize,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Average sequence length
    pub avg_sequence_length: f32,
    /// Average processing time per batch
    pub avg_processing_time: Duration,
    /// Memory utilization efficiency
    pub memory_efficiency: f32,
    /// Padding overhead percentage
    pub padding_overhead: f32,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            total_batches: 0,
            total_sequences: 0,
            avg_batch_size: 0.0,
            avg_sequence_length: 0.0,
            avg_processing_time: Duration::default(),
            memory_efficiency: 0.0,
            padding_overhead: 0.0,
        }
    }
}

/// A sequence waiting to be batched
#[derive(Debug, Clone)]
pub struct PendingSequence {
    /// Unique identifier for the sequence
    pub id: u64,
    /// Phoneme sequence data
    pub phonemes: Vec<Phoneme>,
    /// Timestamp when the sequence was added
    pub timestamp: Instant,
    /// Priority level (higher = more important)
    pub priority: u8,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl PendingSequence {
    /// Create a new pending sequence
    pub fn new(id: u64, phonemes: Vec<Phoneme>) -> Self {
        Self {
            id,
            phonemes,
            timestamp: Instant::now(),
            priority: 0,
            metadata: serde_json::Value::Null,
        }
    }

    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.phonemes.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.phonemes.is_empty()
    }

    /// Get the age of the sequence
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

// Implement ordering for priority queue
impl PartialEq for PendingSequence {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.len() == other.len()
    }
}

impl Eq for PendingSequence {}

impl PartialOrd for PendingSequence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingSequence {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher priority first)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Then by length (for batching efficiency)
                self.len().cmp(&other.len())
            }
            other => other,
        }
    }
}

/// Batch of sequences ready for processing
#[derive(Debug, Clone)]
pub struct ProcessingBatch {
    /// Batch identifier
    pub id: u64,
    /// Sequences in the batch
    pub sequences: Vec<PendingSequence>,
    /// Padded tensor data ready for model inference
    pub tensor_data: Option<Tensor>,
    /// Attention mask for variable-length sequences
    pub attention_mask: Option<Tensor>,
    /// Sequence lengths for each item in the batch
    pub sequence_lengths: Vec<usize>,
    /// Maximum sequence length in this batch
    pub max_length: usize,
    /// Batch creation timestamp
    pub created_at: Instant,
}

impl ProcessingBatch {
    /// Create a new processing batch
    pub fn new(id: u64, sequences: Vec<PendingSequence>) -> Self {
        let max_length = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let sequence_lengths = sequences.iter().map(|s| s.len()).collect();

        Self {
            id,
            sequences,
            tensor_data: None,
            attention_mask: None,
            sequence_lengths,
            max_length,
            created_at: Instant::now(),
        }
    }

    /// Get the batch size
    pub fn size(&self) -> usize {
        self.sequences.len()
    }

    /// Get the total number of tokens in the batch
    pub fn total_tokens(&self) -> usize {
        self.sequence_lengths.iter().sum()
    }

    /// Get the padding overhead as a percentage
    pub fn padding_overhead(&self) -> f32 {
        if self.max_length == 0 {
            return 0.0;
        }

        let total_padded = self.size() * self.max_length;
        let total_actual = self.total_tokens();

        if total_padded == 0 {
            0.0
        } else {
            ((total_padded - total_actual) as f32 / total_padded as f32) * 100.0
        }
    }

    /// Convert phoneme sequences to tensor format
    pub fn create_tensors(&mut self, device: &Device, config: &DynamicBatchConfig) -> Result<()> {
        if self.sequences.is_empty() {
            return Err(AcousticError::InputError("Empty batch".to_string()));
        }

        let batch_size = self.size();
        let seq_len = self.max_length;

        // Create a simple phoneme symbol to ID mapping
        let mut symbol_to_id = std::collections::HashMap::new();
        let mut next_id = 0u32;

        // Collect all unique symbols
        for sequence in &self.sequences {
            for phoneme in &sequence.phonemes {
                if !symbol_to_id.contains_key(&phoneme.symbol) {
                    symbol_to_id.insert(phoneme.symbol.clone(), next_id);
                    next_id += 1;
                }
            }
        }

        // Create padded phoneme tensor
        let mut phoneme_data = vec![0u32; batch_size * seq_len];
        let mut attention_data = vec![0f32; batch_size * seq_len];

        for (batch_idx, sequence) in self.sequences.iter().enumerate() {
            let seq_start = batch_idx * seq_len;

            // Copy phoneme data
            for (token_idx, phoneme) in sequence.phonemes.iter().enumerate() {
                if token_idx < seq_len {
                    let phoneme_id = symbol_to_id.get(&phoneme.symbol).copied().unwrap_or(0);
                    phoneme_data[seq_start + token_idx] = phoneme_id;
                    attention_data[seq_start + token_idx] = 1.0; // Valid token
                }
            }

            // Apply padding strategy for remaining tokens
            let actual_len = sequence.len().min(seq_len);
            for token_idx in actual_len..seq_len {
                let padding_value = match config.padding_strategy {
                    PaddingStrategy::ZeroPadding => 0,
                    PaddingStrategy::RepeatLast => {
                        if actual_len > 0 {
                            let last_phoneme = &sequence.phonemes[actual_len - 1];
                            symbol_to_id.get(&last_phoneme.symbol).copied().unwrap_or(0)
                        } else {
                            0
                        }
                    }
                    PaddingStrategy::PaddingToken(token) => token,
                    PaddingStrategy::MinimalPadding => 0,
                };

                phoneme_data[seq_start + token_idx] = padding_value;
                attention_data[seq_start + token_idx] = 0.0; // Padding token
            }
        }

        // Create tensors
        self.tensor_data = Some(
            Tensor::from_vec(phoneme_data, (batch_size, seq_len), device)?.to_dtype(DType::U32)?,
        );

        self.attention_mask = Some(
            Tensor::from_vec(attention_data, (batch_size, seq_len), device)?
                .to_dtype(DType::F32)?,
        );

        Ok(())
    }
}

/// Dynamic batch processor for variable-length sequences
pub struct DynamicBatcher {
    /// Configuration
    config: DynamicBatchConfig,
    /// Queue of pending sequences waiting to be batched
    pending_sequences: BinaryHeap<Reverse<PendingSequence>>,
    /// Queue organized by sequence length for efficient batching
    length_queues: std::collections::HashMap<usize, VecDeque<PendingSequence>>,
    /// Next batch ID
    next_batch_id: u64,
    /// Processing statistics
    stats: BatchStats,
    /// Device for tensor operations
    device: Device,
}

impl DynamicBatcher {
    /// Create a new dynamic batcher
    pub fn new(config: DynamicBatchConfig, device: Device) -> Self {
        Self {
            config,
            pending_sequences: BinaryHeap::new(),
            length_queues: std::collections::HashMap::new(),
            next_batch_id: 0,
            stats: BatchStats::default(),
            device,
        }
    }

    /// Add a sequence to the batch queue
    pub fn add_sequence(&mut self, sequence: PendingSequence) -> Result<()> {
        // Validate sequence length
        if sequence.len() > self.config.max_sequence_length {
            return Err(AcousticError::InputError(format!(
                "Sequence length {} exceeds maximum {}",
                sequence.len(),
                self.config.max_sequence_length
            )));
        }

        // Add to length-based queue if enabled
        if self.config.sort_by_length {
            let length = sequence.len();
            self.length_queues
                .entry(length)
                .or_default()
                .push_back(sequence.clone());
        }

        // Add to priority queue
        self.pending_sequences.push(Reverse(sequence));

        Ok(())
    }

    /// Try to create a batch from pending sequences
    pub fn try_create_batch(&mut self) -> Option<ProcessingBatch> {
        if self.pending_sequences.is_empty() {
            return None;
        }

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        // Determine batching strategy
        let batch_sequences = if self.config.sort_by_length {
            self.create_length_sorted_batch()
        } else {
            self.create_priority_batch()
        };

        if batch_sequences.is_empty() {
            return None;
        }

        // Check if we should wait for more sequences
        if !self.should_process_batch(&batch_sequences) {
            // Put sequences back
            for seq in batch_sequences {
                self.pending_sequences.push(Reverse(seq));
            }
            return None;
        }

        // Create and return the batch
        let mut batch = ProcessingBatch::new(batch_id, batch_sequences);

        // Create tensors
        if batch.create_tensors(&self.device, &self.config).is_ok() {
            self.update_stats(&batch);
            Some(batch)
        } else {
            None
        }
    }

    /// Create a batch sorted by sequence length
    fn create_length_sorted_batch(&mut self) -> Vec<PendingSequence> {
        let mut batch_sequences = Vec::new();
        let mut target_length = None;

        // Find the most suitable length group
        for (&length, queue) in &mut self.length_queues {
            if queue.is_empty() {
                continue;
            }

            // Start with the first available length
            if target_length.is_none() {
                target_length = Some(length);
            }

            // Prefer length groups that can fill a batch
            if queue.len() >= self.config.max_batch_size {
                target_length = Some(length);
                break;
            }
        }

        // Extract sequences from the chosen length group
        if let Some(length) = target_length {
            if let Some(queue) = self.length_queues.get_mut(&length) {
                while batch_sequences.len() < self.config.max_batch_size && !queue.is_empty() {
                    if let Some(seq) = queue.pop_front() {
                        batch_sequences.push(seq);
                    }
                }
            }
        }

        // Remove sequences from priority queue
        self.remove_sequences_from_priority_queue(&batch_sequences);

        batch_sequences
    }

    /// Create a batch based on priority
    fn create_priority_batch(&mut self) -> Vec<PendingSequence> {
        let mut batch_sequences = Vec::new();
        let mut temp_sequences = Vec::new();

        // Extract sequences considering both priority and batch constraints
        while batch_sequences.len() < self.config.max_batch_size
            && !self.pending_sequences.is_empty()
        {
            if let Some(Reverse(seq)) = self.pending_sequences.pop() {
                // Check if this sequence fits well with the current batch
                if self.sequence_fits_batch(&seq, &batch_sequences) {
                    batch_sequences.push(seq);
                } else {
                    temp_sequences.push(seq);
                }
            }
        }

        // Put back sequences that didn't fit
        for seq in temp_sequences {
            self.pending_sequences.push(Reverse(seq));
        }

        batch_sequences
    }

    /// Check if a sequence fits well in the current batch
    fn sequence_fits_batch(&self, seq: &PendingSequence, batch: &[PendingSequence]) -> bool {
        if batch.is_empty() {
            return true;
        }

        let max_length_in_batch = batch.iter().map(|s| s.len()).max().unwrap_or(0);
        let new_max_length = max_length_in_batch.max(seq.len());

        // Check memory efficiency
        if new_max_length > max_length_in_batch {
            let current_efficiency = self.calculate_memory_efficiency(batch, max_length_in_batch);
            let new_efficiency =
                self.calculate_memory_efficiency_with_seq(batch, seq, new_max_length);

            // Only accept if efficiency doesn't drop too much
            new_efficiency >= current_efficiency * 0.8
        } else {
            true
        }
    }

    /// Calculate memory efficiency for a batch
    fn calculate_memory_efficiency(&self, batch: &[PendingSequence], max_length: usize) -> f32 {
        if max_length == 0 || batch.is_empty() {
            return 1.0;
        }

        let total_tokens: usize = batch.iter().map(|s| s.len()).sum();
        let total_memory = batch.len() * max_length;

        total_tokens as f32 / total_memory as f32
    }

    /// Calculate memory efficiency if a sequence is added
    fn calculate_memory_efficiency_with_seq(
        &self,
        batch: &[PendingSequence],
        seq: &PendingSequence,
        max_length: usize,
    ) -> f32 {
        if max_length == 0 {
            return 1.0;
        }

        let total_tokens: usize = batch.iter().map(|s| s.len()).sum::<usize>() + seq.len();
        let total_memory = (batch.len() + 1) * max_length;

        total_tokens as f32 / total_memory as f32
    }

    /// Remove sequences from the priority queue
    fn remove_sequences_from_priority_queue(&mut self, sequences: &[PendingSequence]) {
        let seq_ids: std::collections::HashSet<u64> = sequences.iter().map(|s| s.id).collect();

        // Rebuild priority queue without the removed sequences
        let mut new_queue = BinaryHeap::new();
        while let Some(Reverse(seq)) = self.pending_sequences.pop() {
            if !seq_ids.contains(&seq.id) {
                new_queue.push(Reverse(seq));
            }
        }
        self.pending_sequences = new_queue;
    }

    /// Check if a batch should be processed now
    fn should_process_batch(&self, batch: &[PendingSequence]) -> bool {
        if batch.is_empty() {
            return false;
        }

        // Always process if batch is full
        if batch.len() >= self.config.max_batch_size {
            return true;
        }

        // Check timeout
        let oldest_sequence = batch.iter().min_by_key(|s| s.timestamp);
        if let Some(seq) = oldest_sequence {
            if seq.age() >= Duration::from_millis(self.config.max_wait_time_ms) {
                return true;
            }
        }

        // Check utilization target
        let current_utilization = batch.len() as f32 / self.config.max_batch_size as f32;
        if current_utilization >= self.config.target_utilization {
            return true;
        }

        false
    }

    /// Update processing statistics
    fn update_stats(&mut self, batch: &ProcessingBatch) {
        self.stats.total_batches += 1;
        self.stats.total_sequences += batch.size();

        // Update averages using exponential moving average
        let alpha = 0.1; // Smoothing factor

        self.stats.avg_batch_size =
            (1.0 - alpha) * self.stats.avg_batch_size + alpha * batch.size() as f32;

        let avg_seq_len = batch.total_tokens() as f32 / batch.size() as f32;
        self.stats.avg_sequence_length =
            (1.0 - alpha) * self.stats.avg_sequence_length + alpha * avg_seq_len;

        self.stats.memory_efficiency = (1.0 - alpha) * self.stats.memory_efficiency
            + alpha * self.calculate_memory_efficiency(&batch.sequences, batch.max_length);

        self.stats.padding_overhead =
            (1.0 - alpha) * self.stats.padding_overhead + alpha * batch.padding_overhead();
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Get number of pending sequences
    pub fn pending_count(&self) -> usize {
        self.pending_sequences.len()
    }

    /// Force process all pending sequences
    pub fn flush(&mut self) -> Vec<ProcessingBatch> {
        let mut batches = Vec::new();

        while !self.pending_sequences.is_empty() {
            let mut batch_sequences = Vec::new();

            // Take up to max_batch_size sequences
            while batch_sequences.len() < self.config.max_batch_size
                && !self.pending_sequences.is_empty()
            {
                if let Some(Reverse(seq)) = self.pending_sequences.pop() {
                    batch_sequences.push(seq);
                }
            }

            if !batch_sequences.is_empty() {
                let batch_id = self.next_batch_id;
                self.next_batch_id += 1;

                let mut batch = ProcessingBatch::new(batch_id, batch_sequences);
                if batch.create_tensors(&self.device, &self.config).is_ok() {
                    self.update_stats(&batch);
                    batches.push(batch);
                }
            }
        }

        // Clear length queues
        self.length_queues.clear();

        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phoneme;
    use candle_core::Device;

    fn create_test_phoneme(id: u32) -> Phoneme {
        Phoneme {
            symbol: format!("ph_{}", id),
            features: Some(std::collections::HashMap::new()),
            duration: Some(0.1), // 100ms in seconds
        }
    }

    fn create_test_sequence(id: u64, length: usize) -> PendingSequence {
        let phonemes = (0..length).map(|i| create_test_phoneme(i as u32)).collect();
        PendingSequence::new(id, phonemes)
    }

    #[test]
    fn test_dynamic_batch_config_default() {
        let config = DynamicBatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_sequence_length, 1000);
        assert!(config.sort_by_length);
    }

    #[test]
    fn test_pending_sequence_creation() {
        let seq = create_test_sequence(1, 10);
        assert_eq!(seq.id, 1);
        assert_eq!(seq.len(), 10);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_processing_batch_creation() {
        let sequences = vec![
            create_test_sequence(1, 5),
            create_test_sequence(2, 8),
            create_test_sequence(3, 3),
        ];

        let batch = ProcessingBatch::new(1, sequences);
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size(), 3);
        assert_eq!(batch.max_length, 8);
        assert_eq!(batch.total_tokens(), 16);
    }

    #[test]
    fn test_padding_overhead_calculation() {
        let sequences = vec![create_test_sequence(1, 5), create_test_sequence(2, 8)];

        let batch = ProcessingBatch::new(1, sequences);
        let overhead = batch.padding_overhead();

        // Batch has 2 sequences, max length 8
        // Total padded: 2 * 8 = 16
        // Total actual: 5 + 8 = 13
        // Overhead: (16 - 13) / 16 = 3/16 = 18.75%
        assert!((overhead - 18.75).abs() < 0.01);
    }

    #[test]
    fn test_dynamic_batcher_creation() {
        let config = DynamicBatchConfig::default();
        let device = Device::Cpu;
        let batcher = DynamicBatcher::new(config, device);

        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_add_sequence_to_batcher() {
        let config = DynamicBatchConfig::default();
        let device = Device::Cpu;
        let mut batcher = DynamicBatcher::new(config, device);

        let seq = create_test_sequence(1, 10);
        assert!(batcher.add_sequence(seq).is_ok());
        assert_eq!(batcher.pending_count(), 1);
    }

    #[test]
    fn test_sequence_length_validation() {
        let mut config = DynamicBatchConfig::default();
        config.max_sequence_length = 5;
        let device = Device::Cpu;
        let mut batcher = DynamicBatcher::new(config, device);

        let seq = create_test_sequence(1, 10); // Too long
        assert!(batcher.add_sequence(seq).is_err());
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        let config = DynamicBatchConfig::default();
        let device = Device::Cpu;
        let batcher = DynamicBatcher::new(config, device);

        let sequences = vec![create_test_sequence(1, 5), create_test_sequence(2, 5)];

        let efficiency = batcher.calculate_memory_efficiency(&sequences, 5);
        assert!((efficiency - 1.0).abs() < 0.01); // Perfect efficiency

        let efficiency = batcher.calculate_memory_efficiency(&sequences, 10);
        assert!((efficiency - 0.5).abs() < 0.01); // 50% efficiency
    }

    #[test]
    fn test_batch_stats_initialization() {
        let stats = BatchStats::default();
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_sequences, 0);
        assert_eq!(stats.avg_batch_size, 0.0);
    }

    #[test]
    fn test_sequence_ordering() {
        let seq1 = PendingSequence {
            priority: 1,
            ..create_test_sequence(1, 5)
        };
        let seq2 = PendingSequence {
            priority: 2,
            ..create_test_sequence(2, 5)
        };

        assert!(seq2 > seq1); // Higher priority first
    }
}
