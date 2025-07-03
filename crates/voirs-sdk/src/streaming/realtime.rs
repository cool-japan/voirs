//! Real-time synthesis processing and low-latency operations.

use crate::{
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::SynthesisConfig,
    VoirsError,
};
use super::{
    management::{StreamingConfig, AudioChunk, LatencyStats},
    pipeline::StreamingPipeline,
};
use futures::{Stream, StreamExt};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, RwLock, Semaphore},
    time::{interval, sleep, timeout},
};

/// Real-time processor for streaming synthesis with minimal latency
pub struct RealtimeProcessor {
    /// G2P component
    g2p: Arc<dyn G2p>,
    
    /// Acoustic model component
    acoustic: Arc<dyn AcousticModel>,
    
    /// Vocoder component
    vocoder: Arc<dyn Vocoder>,
    
    /// Streaming configuration
    config: StreamingConfig,
    
    /// Latency tracking
    latency_stats: Arc<RwLock<LatencyStats>>,
    
    /// Processing semaphore for rate limiting
    semaphore: Arc<Semaphore>,
}

impl RealtimeProcessor {
    /// Create new real-time processor
    pub fn new(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: StreamingConfig,
    ) -> Self {
        let max_concurrent = config.max_concurrent_chunks;
        
        Self {
            g2p,
            acoustic,
            vocoder,
            config,
            latency_stats: Arc::new(RwLock::new(LatencyStats::default())),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Process streaming text input with real-time constraints
    pub async fn process_stream(
        &self,
        mut text_stream: impl Stream<Item = String> + Send + Unpin,
        sender: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let mut chunk_id = 0;
        let mut text_buffer = TextBuffer::new(self.config.clone());
        let mut last_synthesis = Instant::now();
        let mut synthesis_scheduler = SynthesisScheduler::new(self.config.clone());

        // Start background latency monitor
        let latency_monitor = self.start_latency_monitor();

        while let Some(text_fragment) = text_stream.next().await {
            text_buffer.add_text(text_fragment);
            
            // Check if we should synthesize current buffer
            let should_synthesize = self.should_synthesize_buffer(&text_buffer, last_synthesis);
            
            if should_synthesize {
                let text_to_synthesize = text_buffer.extract_ready_text();
                
                if !text_to_synthesize.trim().is_empty() {
                    // Schedule synthesis with priority based on urgency
                    let urgency = self.calculate_urgency(last_synthesis);
                    synthesis_scheduler.schedule_synthesis(
                        chunk_id,
                        text_to_synthesize,
                        urgency,
                    );
                    
                    // Process scheduled syntheses
                    let processed_chunks = synthesis_scheduler
                        .process_scheduled(&self, &sender)
                        .await?;
                    
                    if processed_chunks > 0 {
                        chunk_id += processed_chunks;
                        last_synthesis = Instant::now();
                    }
                }
            }
            
            // Prevent buffer overflow
            if text_buffer.len() > self.config.max_buffer_size {
                tracing::warn!("Text buffer overflow, forcing synthesis");
                let text_to_synthesize = text_buffer.force_extract();
                if !text_to_synthesize.trim().is_empty() {
                    if let Ok(chunk) = self.synthesize_fragment_urgent(chunk_id, text_to_synthesize).await {
                        if sender.send(chunk).await.is_err() {
                            break; // Receiver dropped
                        }
                        chunk_id += 1;
                        last_synthesis = Instant::now();
                    }
                }
            }
        }

        // Process remaining buffer and scheduled syntheses
        let remaining_text = text_buffer.extract_all();
        if !remaining_text.trim().is_empty() {
            synthesis_scheduler.schedule_synthesis(chunk_id, remaining_text, UrgencyLevel::High);
        }
        
        // Finalize all scheduled syntheses
        synthesis_scheduler.finalize(&self, &sender).await?;
        
        latency_monitor.abort();
        Ok(())
    }

    /// Determine if buffer should be synthesized now
    fn should_synthesize_buffer(&self, buffer: &TextBuffer, last_synthesis: Instant) -> bool {
        let buffer_size = buffer.len();
        let time_since_last = last_synthesis.elapsed();
        
        // Multiple conditions for triggering synthesis
        buffer_size >= self.config.min_chunk_chars ||
        time_since_last >= self.config.max_latency ||
        buffer.has_sentence_boundary() ||
        buffer.is_near_capacity()
    }

    /// Calculate urgency level based on timing
    fn calculate_urgency(&self, last_synthesis: Instant) -> UrgencyLevel {
        let elapsed = last_synthesis.elapsed();
        let max_latency = self.config.max_latency;
        
        if elapsed >= max_latency {
            UrgencyLevel::Critical
        } else if elapsed >= max_latency / 2 {
            UrgencyLevel::High
        } else {
            UrgencyLevel::Normal
        }
    }

    /// Synthesize fragment with normal priority
    pub(super) async fn synthesize_fragment(&self, chunk_id: usize, text: String) -> Result<AudioChunk> {
        let _permit = self.semaphore.acquire().await.unwrap();
        
        let start_time = Instant::now();
        let result = StreamingPipeline::process_text_chunk(
            chunk_id,
            text,
            Arc::clone(&self.g2p),
            Arc::clone(&self.acoustic),
            Arc::clone(&self.vocoder),
            &SynthesisConfig::default(),
            &self.config,
        ).await;
        
        // Update latency stats
        let processing_time = start_time.elapsed();
        {
            let mut stats = self.latency_stats.write().await;
            stats.update(processing_time);
        }
        
        result
    }

    /// Synthesize fragment with urgent priority (bypasses rate limiting)
    async fn synthesize_fragment_urgent(&self, chunk_id: usize, text: String) -> Result<AudioChunk> {
        tracing::debug!("Urgent synthesis for chunk {}", chunk_id);
        
        let start_time = Instant::now();
        
        // Use timeout for urgent synthesis to prevent blocking
        let fast_config = self.create_fast_synthesis_config();
        let synthesis_future = StreamingPipeline::process_text_chunk(
            chunk_id,
            text,
            Arc::clone(&self.g2p),
            Arc::clone(&self.acoustic),
            Arc::clone(&self.vocoder),
            &fast_config,
            &self.config,
        );
        
        let result = timeout(self.config.urgent_timeout, synthesis_future).await
            .map_err(|_| VoirsError::timeout("Urgent synthesis timeout"))?;
        
        // Update latency stats
        let processing_time = start_time.elapsed();
        {
            let mut stats = self.latency_stats.write().await;
            stats.update_urgent(processing_time);
        }
        
        result
    }

    /// Create synthesis config optimized for speed
    fn create_fast_synthesis_config(&self) -> SynthesisConfig {
        let mut config = SynthesisConfig::default();
        
        // Optimize for speed over quality in urgent cases
        config.quality = crate::types::QualityLevel::Medium;
        config.enable_enhancement = false;
        
        config
    }

    /// Start background latency monitoring task
    fn start_latency_monitor(&self) -> tokio::task::JoinHandle<()> {
        let latency_stats = Arc::clone(&self.latency_stats);
        let monitor_interval = Duration::from_secs(1);
        
        tokio::spawn(async move {
            let mut interval = interval(monitor_interval);
            
            loop {
                interval.tick().await;
                
                let stats = latency_stats.read().await;
                if stats.sample_count > 0 {
                    tracing::debug!(
                        "Latency stats - avg: {:.2}ms, p95: {:.2}ms, urgent: {}",
                        stats.average_latency.as_millis(),
                        stats.p95_latency.as_millis(),
                        stats.urgent_count
                    );
                }
            }
        })
    }

    /// Get current latency statistics
    pub async fn get_latency_stats(&self) -> LatencyStats {
        self.latency_stats.read().await.clone()
    }

    /// Reset latency statistics
    pub async fn reset_latency_stats(&self) {
        let mut stats = self.latency_stats.write().await;
        *stats = LatencyStats::default();
    }

    /// Check if processor is meeting real-time requirements
    pub async fn is_meeting_realtime_requirements(&self) -> bool {
        let stats = self.latency_stats.read().await;
        let target_latency = self.config.max_latency;
        
        stats.average_latency <= target_latency && 
        stats.p95_latency <= target_latency * 2
    }

    /// Estimate current processing capacity
    pub fn estimate_processing_capacity(&self) -> f32 {
        let available_permits = self.semaphore.available_permits();
        let total_permits = self.config.max_concurrent_chunks;
        
        available_permits as f32 / total_permits as f32
    }

    /// Get configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Text buffer for accumulating input text
struct TextBuffer {
    buffer: String,
    config: StreamingConfig,
    sentence_boundaries: Vec<usize>,
}

impl TextBuffer {
    fn new(config: StreamingConfig) -> Self {
        Self {
            buffer: String::new(),
            config,
            sentence_boundaries: Vec::new(),
        }
    }
    
    fn add_text(&mut self, text: String) {
        let start_pos = self.buffer.len();
        self.buffer.push_str(&text);
        
        // Track sentence boundaries
        for (i, ch) in text.char_indices() {
            if matches!(ch, '.' | '!' | '?') {
                self.sentence_boundaries.push(start_pos + i + 1);
            }
        }
    }
    
    fn len(&self) -> usize {
        self.buffer.len()
    }
    
    fn has_sentence_boundary(&self) -> bool {
        !self.sentence_boundaries.is_empty()
    }
    
    fn is_near_capacity(&self) -> bool {
        self.buffer.len() > (self.config.max_buffer_size * 3) / 4
    }
    
    fn extract_ready_text(&mut self) -> String {
        if self.sentence_boundaries.is_empty() {
            // No sentence boundary, extract by size
            if self.buffer.len() >= self.config.min_chunk_chars {
                let extract_pos = self.find_good_break_point();
                self.extract_up_to(extract_pos)
            } else {
                String::new()
            }
        } else {
            // Extract up to last sentence boundary
            let boundary = self.sentence_boundaries.remove(0);
            self.extract_up_to(boundary)
        }
    }
    
    fn find_good_break_point(&self) -> usize {
        let target_size = self.config.min_chunk_chars;
        let max_search = target_size + 50; // Look a bit further for good break
        
        if self.buffer.len() <= target_size {
            return self.buffer.len();
        }
        
        // Look for word boundaries near target size
        let search_range = target_size..max_search.min(self.buffer.len());
        
        for pos in search_range.rev() {
            if self.buffer.chars().nth(pos).map_or(false, |c| c.is_whitespace()) {
                return pos;
            }
        }
        
        // Fallback to target size
        target_size
    }
    
    fn extract_up_to(&mut self, pos: usize) -> String {
        if pos >= self.buffer.len() {
            return self.extract_all();
        }
        
        let extracted = self.buffer[..pos].to_string();
        self.buffer = self.buffer[pos..].to_string();
        
        // Update sentence boundaries
        self.sentence_boundaries.retain(|&boundary| boundary > pos);
        for boundary in &mut self.sentence_boundaries {
            *boundary -= pos;
        }
        
        extracted
    }
    
    fn extract_all(&mut self) -> String {
        let extracted = self.buffer.clone();
        self.buffer.clear();
        self.sentence_boundaries.clear();
        extracted
    }
    
    fn force_extract(&mut self) -> String {
        // Extract everything when buffer overflows
        self.extract_all()
    }
}

/// Synthesis scheduler for managing processing priorities
struct SynthesisScheduler {
    queue: VecDeque<SynthesisTask>,
    config: StreamingConfig,
}

impl SynthesisScheduler {
    fn new(config: StreamingConfig) -> Self {
        Self {
            queue: VecDeque::new(),
            config,
        }
    }
    
    fn schedule_synthesis(&mut self, chunk_id: usize, text: String, urgency: UrgencyLevel) {
        let task = SynthesisTask {
            chunk_id,
            text,
            urgency,
            scheduled_at: Instant::now(),
        };
        
        // Insert based on urgency (higher urgency at front)
        let insert_pos = self.queue
            .iter()
            .position(|existing| existing.urgency < urgency)
            .unwrap_or(self.queue.len());
        
        self.queue.insert(insert_pos, task);
    }
    
    async fn process_scheduled(
        &mut self,
        processor: &RealtimeProcessor,
        sender: &mpsc::Sender<AudioChunk>,
    ) -> Result<usize> {
        let mut processed = 0;
        
        while let Some(task) = self.queue.pop_front() {
            // Check if task is still relevant (not too old)
            if task.scheduled_at.elapsed() > self.config.task_timeout {
                tracing::warn!("Dropping stale synthesis task for chunk {}", task.chunk_id);
                continue;
            }
            
            let chunk = match task.urgency {
                UrgencyLevel::Critical => {
                    processor.synthesize_fragment_urgent(task.chunk_id, task.text).await?
                }
                _ => {
                    processor.synthesize_fragment(task.chunk_id, task.text).await?
                }
            };
            
            if sender.send(chunk).await.is_err() {
                return Ok(processed); // Receiver dropped
            }
            
            processed += 1;
            
            // Rate limit processing to prevent overwhelming
            if processed >= self.config.max_concurrent_chunks {
                break;
            }
        }
        
        Ok(processed)
    }
    
    async fn finalize(
        &mut self,
        processor: &RealtimeProcessor,
        sender: &mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        // Process all remaining tasks
        while !self.queue.is_empty() {
            self.process_scheduled(processor, sender).await?;
        }
        Ok(())
    }
}

/// Urgency level for synthesis tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UrgencyLevel {
    Normal = 0,
    High = 1,
    Critical = 2,
}

/// Synthesis task in scheduler queue
#[derive(Debug)]
struct SynthesisTask {
    chunk_id: usize,
    text: String,
    urgency: UrgencyLevel,
    scheduled_at: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};
    use tokio::time::sleep;

    fn create_test_processor() -> RealtimeProcessor {
        let config = StreamingConfig {
            min_chunk_chars: 20,
            max_latency: Duration::from_millis(100),
            max_buffer_size: 1000,
            urgent_timeout: Duration::from_millis(50),
            task_timeout: Duration::from_millis(200),
            ..Default::default()
        };
        
        RealtimeProcessor::new(
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
            config,
        )
    }

    #[tokio::test]
    async fn test_text_buffer() {
        let config = StreamingConfig::default();
        let mut buffer = TextBuffer::new(config);
        
        // Add some text
        buffer.add_text("Hello world.".to_string());
        assert!(buffer.has_sentence_boundary());
        
        // Extract ready text
        let extracted = buffer.extract_ready_text();
        assert_eq!(extracted, "Hello world.");
        assert_eq!(buffer.len(), 0);
    }

    #[tokio::test]
    async fn test_text_buffer_word_boundary() {
        let config = StreamingConfig {
            min_chunk_chars: 10,
            ..Default::default()
        };
        let mut buffer = TextBuffer::new(config);
        
        buffer.add_text("This is a longer text without sentence ending".to_string());
        
        let extracted = buffer.extract_ready_text();
        assert!(!extracted.is_empty());
        assert!(extracted.len() >= 10);
        // Should break at word boundary
        assert!(extracted.ends_with(' ') || buffer.buffer.starts_with(' '));
    }

    #[tokio::test]
    async fn test_synthesis_scheduler() {
        let config = StreamingConfig::default();
        let mut scheduler = SynthesisScheduler::new(config);
        
        // Schedule tasks with different urgencies
        scheduler.schedule_synthesis(0, "Normal task".to_string(), UrgencyLevel::Normal);
        scheduler.schedule_synthesis(1, "Critical task".to_string(), UrgencyLevel::Critical);
        scheduler.schedule_synthesis(2, "High task".to_string(), UrgencyLevel::High);
        
        // Critical task should be first
        assert_eq!(scheduler.queue[0].urgency, UrgencyLevel::Critical);
        assert_eq!(scheduler.queue[1].urgency, UrgencyLevel::High);
        assert_eq!(scheduler.queue[2].urgency, UrgencyLevel::Normal);
    }

    #[tokio::test]
    async fn test_urgency_calculation() {
        let processor = create_test_processor();
        
        let recent = Instant::now();
        let urgency = processor.calculate_urgency(recent);
        assert_eq!(urgency, UrgencyLevel::Normal);
        
        // Simulate old timestamp
        let old_time = Instant::now() - Duration::from_millis(150);
        let urgency = processor.calculate_urgency(old_time);
        assert_eq!(urgency, UrgencyLevel::Critical);
    }

    #[tokio::test]
    async fn test_buffer_conditions() {
        let config = StreamingConfig {
            min_chunk_chars: 20,
            max_latency: Duration::from_millis(100),
            ..Default::default()
        };
        let processor = create_test_processor();
        let buffer = TextBuffer::new(config);
        
        // Should not synthesize empty buffer
        let should_synth = processor.should_synthesize_buffer(&buffer, Instant::now());
        assert!(!should_synth);
        
        // Should synthesize after timeout
        let old_time = Instant::now() - Duration::from_millis(150);
        let should_synth = processor.should_synthesize_buffer(&buffer, old_time);
        assert!(should_synth);
    }

    #[tokio::test]
    async fn test_latency_stats() {
        let processor = create_test_processor();
        
        // Initial stats should be empty
        let stats = processor.get_latency_stats().await;
        assert_eq!(stats.sample_count, 0);
        
        // Process a fragment to generate stats
        let result = processor.synthesize_fragment(0, "Test text".to_string()).await;
        assert!(result.is_ok());
        
        let stats = processor.get_latency_stats().await;
        assert_eq!(stats.sample_count, 1);
        assert!(stats.average_latency > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_processing_capacity() {
        let processor = create_test_processor();
        
        let capacity = processor.estimate_processing_capacity();
        assert_eq!(capacity, 1.0); // Should be at full capacity initially
        
        // Acquire a permit and check capacity
        let _permit = processor.semaphore.acquire().await.unwrap();
        let capacity = processor.estimate_processing_capacity();
        assert!(capacity < 1.0);
    }

    #[tokio::test]
    async fn test_urgent_synthesis() {
        let processor = create_test_processor();
        
        let start = Instant::now();
        let result = processor.synthesize_fragment_urgent(0, "Urgent test".to_string()).await;
        let elapsed = start.elapsed();
        
        assert!(result.is_ok());
        // Urgent synthesis should complete within timeout
        assert!(elapsed < processor.config.urgent_timeout);
    }

    #[tokio::test]
    async fn test_real_time_processing() {
        let processor = create_test_processor();
        let (tx, mut rx) = mpsc::channel(10);
        
        // Create a simple text stream
        let text_stream = futures::stream::iter(vec![
            "Hello ".to_string(),
            "world. ".to_string(),
            "This is a test.".to_string(),
        ]);
        
        // Process stream in background
        let processor_handle = tokio::spawn(async move {
            processor.process_stream(text_stream, tx).await
        });
        
        // Collect results
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk);
        }
        
        // Wait for processor to complete
        let result = processor_handle.await.unwrap();
        assert!(result.is_ok());
        assert!(!chunks.is_empty());
    }
}