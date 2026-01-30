//! Batch processor implementation with parallel execution.

use super::*;
use crate::config::PipelineConfig;
use crate::types::SynthesisConfig;
use futures::future::join_all;
use std::collections::VecDeque;

/// High-performance batch processor for parallel synthesis operations.
pub struct BatchProcessor {
    context: BatchContext,
    scheduler: Arc<RwLock<BatchScheduler>>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(pipeline: Arc<VoirsPipeline>, config: BatchConfig) -> Self {
        let context = BatchContext::new(pipeline.clone(), config.clone());
        let scheduler = Arc::new(RwLock::new(BatchScheduler::new(config.scheduling_strategy)));

        Self { context, scheduler }
    }

    /// Create a new batch processor with progress callback.
    pub fn with_progress<F>(pipeline: Arc<VoirsPipeline>, config: BatchConfig, callback: F) -> Self
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        let progress_callback = Arc::new(callback);
        let context = BatchContext::new(pipeline.clone(), config.clone())
            .with_progress_callback(progress_callback);
        let scheduler = Arc::new(RwLock::new(BatchScheduler::new(config.scheduling_strategy)));

        Self { context, scheduler }
    }

    /// Process a batch of synthesis requests in parallel.
    ///
    /// # Arguments
    ///
    /// * `requests` - Vector of batch requests to process
    ///
    /// # Returns
    ///
    /// Vector of batch results in the same order as requests
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    /// use voirs_sdk::batch::{BatchProcessor, BatchConfig, BatchRequest};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());
    ///
    ///     let requests = vec![
    ///         BatchRequest::new("Hello", None),
    ///         BatchRequest::new("World", None),
    ///     ];
    ///
    ///     let results = processor.process(requests).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn process(&self, requests: Vec<BatchRequest>) -> VoirsResult<Vec<BatchResult>> {
        let start_time = Instant::now();
        let total_requests = requests.len();

        info!("Starting batch processing of {} requests", total_requests);

        // Update statistics
        {
            let mut stats = self.context.statistics.write().await;
            stats.total_requests += total_requests;
            stats.start_time = Some(start_time);
        }

        // Schedule requests
        let scheduled_batches = {
            let mut scheduler = self.scheduler.write().await;
            scheduler.schedule(requests, &self.context.config)?
        };

        debug!(
            "Scheduled {} requests into {} batches",
            total_requests,
            scheduled_batches.len()
        );

        // Process batches in parallel
        let mut all_results = Vec::new();
        for (batch_idx, batch) in scheduled_batches.into_iter().enumerate() {
            debug!("Processing batch {}: {} requests", batch_idx, batch.len());

            let batch_results = self.process_single_batch(batch, batch_idx).await?;
            all_results.extend(batch_results);

            // Report progress
            if let Some(callback) = &self.context.progress_callback {
                callback(all_results.len(), total_requests);
            }
        }

        // Update final statistics
        {
            let mut stats = self.context.statistics.write().await;
            stats.successful_requests = all_results.iter().filter(|r| r.is_success()).count();
            stats.failed_requests = all_results.iter().filter(|r| !r.is_success()).count();
            stats.total_time = start_time.elapsed();
        }

        info!(
            "Batch processing complete: {}/{} successful in {:?}",
            all_results.iter().filter(|r| r.is_success()).count(),
            total_requests,
            start_time.elapsed()
        );

        Ok(all_results)
    }

    /// Process a single batch of requests concurrently.
    async fn process_single_batch(
        &self,
        requests: Vec<BatchRequest>,
        batch_id: usize,
    ) -> VoirsResult<Vec<BatchResult>> {
        let batch_size = requests.len();
        let semaphore = self.context.semaphore.clone();
        let pipeline = self.context.pipeline.clone();
        let config = self.context.config.clone();

        // Process requests concurrently with semaphore for concurrency control
        let tasks: Vec<_> = requests
            .into_iter()
            .enumerate()
            .map(|(idx, request)| {
                let semaphore = semaphore.clone();
                let pipeline = pipeline.clone();
                let config = config.clone();
                let worker_id = idx % config.max_concurrency;

                tokio::spawn(async move {
                    // Acquire semaphore permit
                    let _permit = semaphore.acquire().await.map_err(|e| {
                        VoirsError::audio_error(format!("Failed to acquire semaphore: {}", e))
                    })?;

                    debug!("Worker {} processing request: {}", worker_id, request.id);

                    Self::process_single_request(pipeline, request, Some(worker_id), &config).await
                })
            })
            .collect();

        // Wait for all tasks to complete
        let results = join_all(tasks).await;

        // Collect results
        let mut batch_results = Vec::with_capacity(batch_size);
        for (idx, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(batch_result)) => {
                    debug!("Request {} completed successfully", idx);
                    batch_results.push(batch_result);
                }
                Ok(Err(e)) => {
                    warn!("Request {} failed: {}", idx, e);
                    batch_results.push(BatchResult {
                        request_id: format!("request-{}-{}", batch_id, idx),
                        result: Err(e),
                        processing_time: Duration::from_secs(0),
                        retry_count: 0,
                        worker_id: None,
                    });
                }
                Err(e) => {
                    warn!("Request {} task failed: {}", idx, e);
                    batch_results.push(BatchResult {
                        request_id: format!("request-{}-{}", batch_id, idx),
                        result: Err(VoirsError::audio_error(format!("Task failed: {}", e))),
                        processing_time: Duration::from_secs(0),
                        retry_count: 0,
                        worker_id: None,
                    });
                }
            }
        }

        Ok(batch_results)
    }

    /// Process a single synthesis request with retry logic.
    async fn process_single_request(
        pipeline: Arc<VoirsPipeline>,
        request: BatchRequest,
        worker_id: Option<usize>,
        config: &BatchConfig,
    ) -> VoirsResult<BatchResult> {
        let request_id = request.id.clone();
        let mut retry_count = 0;
        let start_time = Instant::now();

        loop {
            let result = tokio::time::timeout(
                config.synthesis_timeout,
                Self::synthesize_request(&pipeline, &request),
            )
            .await;

            match result {
                Ok(Ok(audio)) => {
                    return Ok(BatchResult {
                        request_id,
                        result: Ok(audio),
                        processing_time: start_time.elapsed(),
                        retry_count,
                        worker_id,
                    });
                }
                Ok(Err(e)) if config.retry_failed && retry_count < config.max_retries => {
                    retry_count += 1;
                    warn!(
                        "Request {} failed (attempt {}/{}): {}",
                        request_id, retry_count, config.max_retries, e
                    );
                    tokio::time::sleep(Duration::from_millis(100 * retry_count as u64)).await;
                    continue;
                }
                Ok(Err(e)) => {
                    return Ok(BatchResult {
                        request_id,
                        result: Err(e),
                        processing_time: start_time.elapsed(),
                        retry_count,
                        worker_id,
                    });
                }
                Err(_) => {
                    return Ok(BatchResult {
                        request_id,
                        result: Err(VoirsError::audio_error("Request timeout")),
                        processing_time: start_time.elapsed(),
                        retry_count,
                        worker_id,
                    });
                }
            }
        }
    }

    /// Synthesize audio for a single request.
    async fn synthesize_request(
        pipeline: &VoirsPipeline,
        request: &BatchRequest,
    ) -> VoirsResult<AudioBuffer> {
        // Support voice switching if specified in request
        if let Some(voice) = &request.voice {
            debug!("Switching to voice: {}", voice);
            pipeline.set_voice(voice).await?;
        }

        // Support speed and pitch configuration
        // If speed or pitch is specified, create custom synthesis config
        let needs_custom_config = request.speed.is_some() || request.pitch.is_some();

        if needs_custom_config {
            // Get default synthesis config and override with request parameters
            let mut config = SynthesisConfig::default();

            if let Some(speed) = request.speed {
                // Clamp speed to valid range (0.5 - 2.0)
                config.speaking_rate = speed.clamp(0.5, 2.0);
                debug!("Using custom speaking rate: {}", config.speaking_rate);
            }

            if let Some(pitch) = request.pitch {
                // Clamp pitch to valid range (-12.0 - 12.0 semitones)
                config.pitch_shift = pitch.clamp(-12.0, 12.0);
                debug!("Using custom pitch shift: {}", config.pitch_shift);
            }

            // Perform synthesis with custom configuration
            pipeline
                .synthesize_with_config(&request.text, &config)
                .await
        } else {
            // Perform synthesis with default pipeline configuration
            pipeline.synthesize(&request.text).await
        }
    }

    /// Get current batch statistics.
    pub async fn statistics(&self) -> BatchStatistics {
        self.context.statistics.read().await.clone()
    }

    /// Reset batch statistics.
    pub async fn reset_statistics(&self) {
        let mut stats = self.context.statistics.write().await;
        *stats = BatchStatistics::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoirsPipelineBuilder;

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        let stats = processor.statistics().await;
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        let requests = vec![
            BatchRequest::new("Hello", None),
            BatchRequest::new("World", None),
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 2);

        let stats = processor.statistics().await;
        assert_eq!(stats.total_requests, 2);
    }

    #[tokio::test]
    async fn test_batch_processing_with_priority() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        let requests = vec![
            BatchRequest::new("Low priority", None).with_priority(1),
            BatchRequest::new("High priority", None).with_priority(10),
            BatchRequest::new("Medium priority", None).with_priority(5),
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_batch_statistics() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        let requests = vec![BatchRequest::new("Test", None)];
        let _ = processor.process(requests).await.unwrap();

        let stats = processor.statistics().await;
        assert!(stats.total_requests > 0);
        assert!(stats.total_time > Duration::from_secs(0));
    }

    #[tokio::test]
    async fn test_batch_voice_switching() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with voice switching
        let requests = vec![
            BatchRequest::new("Hello", Some("voice-1")),
            BatchRequest::new("World", Some("voice-2")),
            BatchRequest::new("Test", None), // No voice switch
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 3);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_custom_speed() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with custom speed
        let requests = vec![
            BatchRequest::new("Fast", None).with_speed(1.5),
            BatchRequest::new("Slow", None).with_speed(0.8),
            BatchRequest::new("Normal", None), // Default speed
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 3);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_custom_pitch() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with custom pitch
        let requests = vec![
            BatchRequest::new("High pitch", None).with_pitch(5.0),
            BatchRequest::new("Low pitch", None).with_pitch(-5.0),
            BatchRequest::new("Normal pitch", None), // Default pitch
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 3);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_speed_and_pitch() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with both speed and pitch
        let requests = vec![
            BatchRequest::new("Fast and high", None)
                .with_speed(1.5)
                .with_pitch(3.0),
            BatchRequest::new("Slow and low", None)
                .with_speed(0.7)
                .with_pitch(-3.0),
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 2);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_parameter_clamping() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with out-of-range parameters (should be clamped)
        let requests = vec![
            BatchRequest::new("Too fast", None).with_speed(5.0), // Should clamp to 2.0
            BatchRequest::new("Too slow", None).with_speed(0.1), // Should clamp to 0.5
            BatchRequest::new("Too high pitch", None).with_pitch(50.0), // Should clamp to 12.0
            BatchRequest::new("Too low pitch", None).with_pitch(-50.0), // Should clamp to -12.0
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 4);

        // All should succeed even with out-of-range values
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_mixed_configuration() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with mixed configurations
        let requests = vec![
            BatchRequest::new("Voice only", Some("voice-1")),
            BatchRequest::new("Speed only", None).with_speed(1.2),
            BatchRequest::new("Pitch only", None).with_pitch(2.0),
            BatchRequest::new("Voice and speed", Some("voice-2")).with_speed(0.9),
            BatchRequest::new("All params", Some("voice-3"))
                .with_speed(1.1)
                .with_pitch(-1.0),
            BatchRequest::new("Default", None), // No custom params
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 6);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }

        let stats = processor.statistics().await;
        assert_eq!(stats.total_requests, 6);
        assert_eq!(stats.successful_requests, 6);
        assert_eq!(stats.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_batch_with_priority_and_params() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test combining priority with custom parameters
        let requests = vec![
            BatchRequest::new("High priority fast", None)
                .with_priority(10)
                .with_speed(1.5),
            BatchRequest::new("Low priority slow", None)
                .with_priority(1)
                .with_speed(0.8),
            BatchRequest::new("Medium priority custom voice", Some("voice-1")).with_priority(5),
        ];

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 3);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }

    #[tokio::test]
    async fn test_batch_variable_parameters() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();
        let processor = BatchProcessor::new(Arc::new(pipeline), BatchConfig::default());

        // Test with gradually varying parameters
        let requests: Vec<_> = (0..10)
            .map(|i| {
                let speed = 0.8 + (i as f32 * 0.1);
                let pitch = -5.0 + (i as f32);
                BatchRequest::new(format!("Sentence {}", i), None)
                    .with_speed(speed)
                    .with_pitch(pitch)
            })
            .collect();

        let results = processor.process(requests).await.unwrap();
        assert_eq!(results.len(), 10);

        // All should succeed
        for result in &results {
            assert!(result.is_success());
        }
    }
}
