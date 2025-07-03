//! Streaming synthesis module with modular architecture.
//!
//! This module provides comprehensive streaming synthesis capabilities organized into
//! modular components:
//!
//! - [`pipeline`] - Core streaming pipeline functionality and chunk processing
//! - [`realtime`] - Real-time synthesis processing and low-latency operations
//! - [`management`] - Stream management, ordering, and state tracking
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::streaming::{StreamingPipeline, StreamingConfig};
//! use voirs_sdk::pipeline::VoirsPipeline;
//! use futures::StreamExt;
//!
//! # async fn example() -> voirs_sdk::Result<()> {
//! // Create a pipeline
//! let pipeline = VoirsPipeline::builder()
//!     .with_voice("en-US-female-calm")
//!     .build()
//!     .await?;
//!
//! // Get components for streaming
//! let (g2p, acoustic, vocoder) = pipeline.components();
//!
//! // Create streaming pipeline
//! let streaming = StreamingPipeline::new(
//!     g2p.clone(),
//!     acoustic.clone(), 
//!     vocoder.clone(),
//!     StreamingConfig::low_latency(),
//! );
//!
//! // Stream synthesis
//! let text = "This is a long text that will be streamed in chunks.";
//! let mut stream = streaming.synthesize_stream(text).await?;
//!
//! while let Some(chunk) = stream.next().await {
//!     let chunk = chunk?;
//!     println!("Generated chunk {}: {:.2}s audio", chunk.chunk_id, chunk.audio.duration());
//! }
//! # Ok(())
//! # }
//! ```

pub mod pipeline;
pub mod realtime;
pub mod management;

// Re-export the main types for convenience
pub use pipeline::StreamingPipeline;
pub use realtime::RealtimeProcessor;
pub use management::{
    StreamingConfig, StreamingState, QualityMetrics, LatencyStats,
    AudioChunk, ChunkMetadata, OrderedChunkStream, StreamCombiner,
    CombinationStrategy, StreamStats, ThroughputMetrics,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};
    use futures::StreamExt;
    use std::{sync::Arc, time::Duration};
    use tokio::time::sleep;

    fn create_test_pipeline() -> StreamingPipeline {
        StreamingPipeline::new(
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
            StreamingConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_end_to_end_streaming() {
        let pipeline = create_test_pipeline();
        
        let text = "This is a comprehensive test of the streaming synthesis system. \
                   It should split this text into multiple chunks and process them efficiently. \
                   Each chunk should be processed with high quality and low latency.";
        
        let mut stream = pipeline.synthesize_stream(text).await.unwrap();
        let mut chunks = Vec::new();
        let mut total_duration = 0.0;
        
        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            
            // Verify chunk properties
            assert!(chunk.audio.duration() > 0.0);
            assert!(!chunk.text.trim().is_empty());
            assert!(chunk.metadata.phoneme_count > 0);
            assert!(chunk.metadata.mel_frames > 0);
            
            total_duration += chunk.audio.duration();
            chunks.push(chunk);
        }
        
        // Should have multiple chunks
        assert!(chunks.len() > 1);
        assert!(total_duration > 0.0);
        
        // Verify chunks are ordered correctly
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
        }
    }

    #[tokio::test]
    async fn test_real_time_synthesis() {
        let pipeline = create_test_pipeline();
        
        // Create a text stream that arrives over time
        let text_parts = vec![
            "Hello there, ".to_string(),
            "this is a real-time ".to_string(),
            "synthesis test. ".to_string(),
            "Each part arrives separately.".to_string(),
        ];
        
        let text_stream = futures::stream::iter(text_parts)
            .then(|text| async move {
                sleep(Duration::from_millis(50)).await; // Simulate real-time arrival
                text
            })
            .boxed();
        
        let mut audio_stream = pipeline.synthesize_realtime(text_stream).await.unwrap();
        let mut chunk_count = 0;
        
        while let Some(result) = audio_stream.next().await {
            let chunk = result.unwrap();
            assert!(chunk.audio.duration() > 0.0);
            chunk_count += 1;
        }
        
        assert!(chunk_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_configuration_impact() {
        let g2p = Arc::new(DummyG2p::new());
        let acoustic = Arc::new(DummyAcoustic::new());
        let vocoder = Arc::new(DummyVocoder::new());
        
        // Test low latency config
        let low_latency_config = StreamingConfig::low_latency();
        let low_latency_pipeline = StreamingPipeline::new(
            g2p.clone(),
            acoustic.clone(),
            vocoder.clone(),
            low_latency_config.clone(),
        );
        
        // Test high quality config
        let high_quality_config = StreamingConfig::high_quality();
        let high_quality_pipeline = StreamingPipeline::new(
            g2p.clone(),
            acoustic.clone(),
            vocoder.clone(),
            high_quality_config.clone(),
        );
        
        let test_text = "This is a test text for comparing different configurations.";
        
        // Compare chunk sizes
        let low_latency_chunks = low_latency_pipeline.split_text_for_streaming(test_text);
        let high_quality_chunks = high_quality_pipeline.split_text_for_streaming(test_text);
        
        // Low latency should create smaller chunks
        assert!(low_latency_chunks.len() >= high_quality_chunks.len());
        
        for chunk in &low_latency_chunks {
            assert!(chunk.len() <= low_latency_config.max_chunk_chars);
        }
        
        for chunk in &high_quality_chunks {
            assert!(chunk.len() <= high_quality_config.max_chunk_chars);
        }
    }

    #[tokio::test]
    async fn test_ordered_chunk_stream() {
        // Create chunks in random order
        let chunks = vec![
            Ok(AudioChunk::new(
                2,
                crate::audio::AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5),
                "Third chunk".to_string(),
                Duration::from_millis(10),
                5,
                20,
            )),
            Ok(AudioChunk::new(
                0,
                crate::audio::AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5),
                "First chunk".to_string(),
                Duration::from_millis(10),
                5,
                20,
            )),
            Ok(AudioChunk::new(
                1,
                crate::audio::AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5),
                "Second chunk".to_string(),
                Duration::from_millis(10),
                5,
                20,
            )),
        ];
        
        let stream = futures::stream::iter(chunks);
        let mut ordered_stream = OrderedChunkStream::new(Box::pin(stream), 10);
        
        let mut received_chunks = Vec::new();
        while let Some(result) = ordered_stream.next().await {
            received_chunks.push(result.unwrap());
        }
        
        // Should receive chunks in correct order
        assert_eq!(received_chunks.len(), 3);
        assert_eq!(received_chunks[0].chunk_id, 0);
        assert_eq!(received_chunks[1].chunk_id, 1);
        assert_eq!(received_chunks[2].chunk_id, 2);
        
        // Check stream statistics
        let stats = ordered_stream.stats();
        assert_eq!(stats.chunks_received, 3);
        assert_eq!(stats.chunks_delivered, 3);
        assert!(stats.is_healthy());
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let pipeline = create_test_pipeline();
        
        let text = "This is a performance monitoring test.";
        let mut stream = pipeline.synthesize_stream(text).await.unwrap();
        
        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            
            // Check real-time factor
            let rtf = chunk.real_time_factor();
            assert!(rtf > 0.0);
            
            // Check efficiency score
            let efficiency = chunk.efficiency_score();
            assert!(efficiency > 0.0);
            assert!(efficiency <= 1.0);
            
            // Metadata should be complete
            assert!(chunk.metadata.confidence_score >= 0.0);
            assert!(chunk.metadata.confidence_score <= 1.0);
        }
        
        // Check pipeline state
        let state = pipeline.get_state().await;
        assert!(state.chunks_processed > 0);
        assert!(state.total_duration > 0.0);
    }

    #[tokio::test]
    async fn test_streaming_state_management() {
        let pipeline = create_test_pipeline();
        
        // Initial state
        let initial_state = pipeline.get_state().await;
        assert_eq!(initial_state.chunks_processed, 0);
        
        // Process some text
        let text = "Test text for state management.";
        let mut stream = pipeline.synthesize_stream(text).await.unwrap();
        
        // Consume stream
        while let Some(_) = stream.next().await {}
        
        // Check updated state
        let final_state = pipeline.get_state().await;
        assert!(final_state.chunks_processed > 0);
        assert!(final_state.total_duration > 0.0);
        assert!(final_state.total_chars_processed > 0);
        
        // Reset state
        pipeline.reset_state().await;
        let reset_state = pipeline.get_state().await;
        assert_eq!(reset_state.chunks_processed, 0);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let pipeline = create_test_pipeline();
        
        // Test with empty text
        let empty_stream = pipeline.synthesize_stream("").await.unwrap();
        let chunks: Vec<_> = empty_stream.collect().await;
        assert!(chunks.is_empty());
        
        // Test with whitespace only
        let whitespace_stream = pipeline.synthesize_stream("   \n\t  ").await.unwrap();
        let chunks: Vec<_> = whitespace_stream.collect().await;
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn test_chunk_metadata_completeness() {
        let pipeline = create_test_pipeline();
        
        let text = "This is a sentence. This is another sentence!";
        let mut stream = pipeline.synthesize_stream(text).await.unwrap();
        
        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            
            // Verify metadata completeness
            assert!(chunk.metadata.phoneme_count > 0);
            assert!(chunk.metadata.mel_frames > 0);
            assert!(chunk.metadata.confidence_score >= 0.0);
            assert!(chunk.metadata.confidence_score <= 1.0);
            
            // Check boundary detection
            if chunk.text.trim_end().ends_with('.') || chunk.text.trim_end().ends_with('!') {
                assert!(chunk.metadata.is_sentence_boundary);
            }
            
            // Test metadata export
            let json = chunk.export_metadata().unwrap();
            assert!(json.contains("phoneme_count"));
            assert!(json.contains("mel_frames"));
        }
    }

    #[tokio::test]
    async fn test_adaptive_configuration() {
        let mut config = StreamingConfig::default();
        config.adaptive_chunking = true;
        
        let original_max_chars = config.max_chunk_chars;
        
        // Simulate poor performance (high RTF, high latency)
        config.adapt_for_performance(2.0, Duration::from_millis(800));
        assert!(config.max_chunk_chars < original_max_chars);
        
        // Reset and simulate good performance
        config.max_chunk_chars = original_max_chars;
        config.adapt_for_performance(0.1, Duration::from_millis(50));
        assert!(config.max_chunk_chars >= original_max_chars);
    }

    #[tokio::test]
    async fn test_streaming_with_different_text_types() {
        let pipeline = create_test_pipeline();
        
        let test_cases = vec![
            ("Short.", 1),
            ("This is a medium length sentence with multiple words.", 1),
            ("This is the first sentence. This is the second sentence. This is the third sentence.", 2),
            ("Very long text that should definitely be split into multiple chunks because it exceeds the default chunk size limits and contains multiple sentences.", 2),
        ];
        
        for (text, expected_min_chunks) in test_cases {
            let mut stream = pipeline.synthesize_stream(text).await.unwrap();
            let chunks: Vec<_> = stream.collect().await;
            
            assert!(chunks.len() >= expected_min_chunks, 
                "Text '{}' should produce at least {} chunks, got {}", 
                text, expected_min_chunks, chunks.len());
            
            for chunk in chunks {
                let chunk = chunk.unwrap();
                assert!(chunk.audio.duration() > 0.0);
                assert!(!chunk.text.trim().is_empty());
            }
        }
    }

    #[test]
    fn test_configuration_presets() {
        let low_latency = StreamingConfig::low_latency();
        let high_quality = StreamingConfig::high_quality();
        let batch = StreamingConfig::batch_processing();
        
        // Low latency should have smaller chunks and lower latency limits
        assert!(low_latency.max_chunk_chars < high_quality.max_chunk_chars);
        assert!(low_latency.max_latency < high_quality.max_latency);
        assert!(low_latency.quality_vs_latency < high_quality.quality_vs_latency);
        
        // Batch processing should allow larger chunks and more concurrency
        assert!(batch.max_chunk_chars >= high_quality.max_chunk_chars);
        assert!(batch.max_concurrent_chunks >= high_quality.max_concurrent_chunks);
        
        // All configurations should be valid
        assert!(low_latency.validate().is_ok());
        assert!(high_quality.validate().is_ok());
        assert!(batch.validate().is_ok());
    }

    #[test]
    fn test_quality_metrics() {
        let mut metrics = QualityMetrics::default();
        
        // Create test chunk with good performance
        let audio = crate::audio::AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let good_chunk = AudioChunk::new(
            0,
            audio.clone(),
            "Good performance".to_string(),
            Duration::from_millis(100), // 0.1 RTF for 1s audio
            10,
            100,
        );
        
        metrics.update_with_chunk(&good_chunk);
        assert!(metrics.real_time_factor < 1.0);
        assert!(!metrics.is_quality_degrading());
        
        // Create chunk with poor performance
        let poor_chunk = AudioChunk::new(
            1,
            audio,
            "Poor performance".to_string(),
            Duration::from_millis(2000), // 2.0 RTF for 1s audio
            10,
            100,
        );
        
        metrics.update_with_chunk(&poor_chunk);
        assert!(metrics.peak_rtf > 1.0);
        assert!(metrics.is_quality_degrading());
    }
}