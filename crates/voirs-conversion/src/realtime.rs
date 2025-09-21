//! Real-time voice conversion

use crate::{
    config::ConversionConfig,
    core::VoiceConverter,
    processing::{AudioBuffer, ProcessingPipeline, ProcessingStage, StageType},
    transforms::{AgeTransform, GenderTransform, PitchTransform, SpeedTransform, Transform},
    types::{ConversionRequest, ConversionTarget, ConversionType, VoiceCharacteristics},
    Error, Result,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Real-time converter configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Buffer size for processing (smaller = lower latency, higher CPU)
    pub buffer_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Latency target in milliseconds
    pub target_latency_ms: f32,
    /// Overlap factor for processing windows (0.0-1.0)
    pub overlap_factor: f32,
    /// Enable adaptive buffer sizing
    pub adaptive_buffering: bool,
    /// Maximum processing threads
    pub max_threads: usize,
    /// Enable lookahead processing
    pub enable_lookahead: bool,
    /// Lookahead buffer size
    pub lookahead_size: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            buffer_size: 256, // Smaller for lower latency
            sample_rate: 22050,
            target_latency_ms: 20.0,
            overlap_factor: 0.25,
            adaptive_buffering: true,
            max_threads: 2,
            enable_lookahead: true,
            lookahead_size: 128,
        }
    }
}

impl RealtimeConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.buffer_size == 0 {
            return Err(Error::config(
                "Buffer size must be greater than 0".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(Error::config(
                "Sample rate must be greater than 0".to_string(),
            ));
        }
        if self.overlap_factor < 0.0 || self.overlap_factor >= 1.0 {
            return Err(Error::config(
                "Overlap factor must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.max_threads == 0 {
            return Err(Error::config(
                "Max threads must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate theoretical latency
    pub fn calculate_latency_ms(&self) -> f32 {
        (self.buffer_size as f32 / self.sample_rate as f32) * 1000.0
    }

    /// Optimize for lowest latency
    pub fn optimize_for_latency(mut self) -> Self {
        self.buffer_size = 128;
        self.overlap_factor = 0.125;
        self.adaptive_buffering = true;
        self.enable_lookahead = false;
        self
    }

    /// Optimize for quality
    pub fn optimize_for_quality(mut self) -> Self {
        self.buffer_size = 512;
        self.overlap_factor = 0.5;
        self.enable_lookahead = true;
        self.lookahead_size = 256;
        self
    }
}

/// Real-time processing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Pass-through mode (no conversion)
    PassThrough,
    /// Low-latency mode with reduced quality
    LowLatency,
    /// Balanced mode
    Balanced,
    /// High-quality mode with higher latency
    HighQuality,
}

/// Real-time voice converter
#[derive(Debug)]
pub struct RealtimeConverter {
    /// Configuration
    config: RealtimeConfig,
    /// Conversion configuration
    conversion_config: ConversionConfig,
    /// Processing mode
    processing_mode: ProcessingMode,
    /// Input ring buffer
    input_buffer: Arc<RwLock<AudioBuffer>>,
    /// Output ring buffer
    output_buffer: Arc<RwLock<AudioBuffer>>,
    /// Lookahead buffer
    lookahead_buffer: VecDeque<f32>,
    /// Processing pipeline
    processing_pipeline: ProcessingPipeline,
    /// Voice converter for complex transformations
    voice_converter: Option<Arc<VoiceConverter>>,
    /// Current conversion target
    conversion_target: Option<ConversionTarget>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Processing thread pool
    thread_pool: rayon::ThreadPool,
}

impl RealtimeConverter {
    /// Create new real-time converter
    pub fn new(config: RealtimeConfig) -> Result<Self> {
        config.validate()?;

        let conversion_config = ConversionConfig {
            output_sample_rate: config.sample_rate,
            buffer_size: config.buffer_size,
            quality_level: 0.7,
            use_gpu: false, // Start with CPU for real-time stability
            ..Default::default()
        };

        let input_buffer = Arc::new(RwLock::new(AudioBuffer::new_ring_buffer(
            config.buffer_size * 4, // Larger ring buffer
            config.sample_rate,
        )));

        let output_buffer = Arc::new(RwLock::new(AudioBuffer::new_ring_buffer(
            config.buffer_size * 4,
            config.sample_rate,
        )));

        let lookahead_buffer = VecDeque::with_capacity(config.lookahead_size);

        // Create optimized processing pipeline
        let processing_pipeline = Self::create_realtime_pipeline(&config)?;

        // Create thread pool for parallel processing
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.max_threads)
            .build()
            .map_err(|e| Error::realtime(format!("Failed to create thread pool: {e}")))?;

        Ok(Self {
            config,
            conversion_config,
            processing_mode: ProcessingMode::Balanced,
            input_buffer,
            output_buffer,
            lookahead_buffer,
            processing_pipeline,
            voice_converter: None,
            conversion_target: None,
            metrics: PerformanceMetrics::default(),
            thread_pool,
        })
    }

    /// Create with voice converter for advanced transformations
    pub fn with_voice_converter(mut self, converter: Arc<VoiceConverter>) -> Self {
        self.voice_converter = Some(converter);
        self
    }

    /// Set processing mode
    pub fn set_processing_mode(&mut self, mode: ProcessingMode) {
        self.processing_mode = mode;
        debug!("Set processing mode to: {:?}", mode);
    }

    /// Set conversion target for voice transformation
    pub fn set_conversion_target(&mut self, target: ConversionTarget) {
        self.conversion_target = Some(target);
        info!("Set conversion target for real-time processing");
    }

    /// Process audio chunk in real-time
    pub async fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        if input.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Processing real-time chunk: {} samples", input.len());

        // Add input to ring buffer
        {
            let mut input_buf = self.input_buffer.write().await;
            input_buf.push_samples(input)?;
        }

        // Process based on mode
        let processed_audio = match self.processing_mode {
            ProcessingMode::PassThrough => self.process_passthrough(input).await?,
            ProcessingMode::LowLatency => self.process_low_latency(input).await?,
            ProcessingMode::Balanced => self.process_balanced(input).await?,
            ProcessingMode::HighQuality => self.process_high_quality(input).await?,
        };

        // Add processed audio to output buffer
        {
            let mut output_buf = self.output_buffer.write().await;
            output_buf.push_samples(&processed_audio)?;
        }

        // Update metrics
        let processing_time = start_time.elapsed();
        self.metrics.update(processing_time, input.len());

        // Check if we're meeting latency targets
        if processing_time.as_millis() as f32 > self.config.target_latency_ms {
            warn!(
                "Processing time ({:.2}ms) exceeds target latency ({:.2}ms)",
                processing_time.as_millis() as f32,
                self.config.target_latency_ms
            );
        }

        // Return processed samples
        let mut output_buf = self.output_buffer.write().await;
        Ok(output_buf.drain())
    }

    /// Process multiple chunks sequentially (simplified implementation)
    pub async fn process_chunks_parallel(&mut self, chunks: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(chunks.len());

        // Process chunks sequentially for now (can be optimized later)
        for chunk in chunks {
            match self.process_chunk(chunk).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Chunk processing failed: {}", e);
                    results.push(Vec::new()); // Return empty chunk on error
                }
            }
        }

        Ok(results)
    }

    /// Get current latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        let processing_latency = self.metrics.average_processing_time_ms();
        let buffer_latency = self.config.calculate_latency_ms();
        processing_latency + buffer_latency
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = PerformanceMetrics::default();
    }

    /// Check if real-time constraints are being met
    pub fn is_realtime_stable(&self) -> bool {
        let current_latency = self.get_latency_ms();
        current_latency <= self.config.target_latency_ms * 1.2 // 20% tolerance
    }

    // Private processing methods

    async fn process_passthrough(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        Ok(input.to_vec())
    }

    async fn process_low_latency(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Minimal processing for lowest latency
        let mut processed = input.to_vec();

        // Apply only essential processing
        if let Some(ref target) = self.conversion_target {
            // Quick pitch adjustment
            if target.characteristics.pitch.mean_f0 != 150.0 {
                let pitch_factor = target.characteristics.pitch.mean_f0 / 150.0;
                let pitch_transform = PitchTransform::new(pitch_factor);
                processed = pitch_transform.apply(&processed)?;
            }
        }

        Ok(processed)
    }

    async fn process_balanced(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Balanced processing with moderate quality
        let mut processed = self.processing_pipeline.process(input).await?;

        if let Some(ref target) = self.conversion_target {
            // Apply basic transformations
            processed = self.apply_basic_transforms(&processed, target).await?;
        }

        Ok(processed)
    }

    async fn process_high_quality(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // High-quality processing with neural models if available
        let mut processed = self.processing_pipeline.process(input).await?;

        if let Some(ref converter) = self.voice_converter {
            if let Some(ref target) = self.conversion_target {
                // Use full voice converter for best quality
                let request = ConversionRequest {
                    id: format!("rt_{}", Instant::now().elapsed().as_nanos()),
                    source_audio: processed.clone(),
                    source_sample_rate: self.config.sample_rate,
                    target: target.clone(),
                    conversion_type: ConversionType::SpeakerConversion,
                    realtime: true,
                    quality_level: 0.8,
                    parameters: std::collections::HashMap::new(),
                    timestamp: std::time::SystemTime::now(),
                };

                match converter.convert(request).await {
                    Ok(result) => processed = result.converted_audio,
                    Err(e) => {
                        warn!("Neural conversion failed, falling back to basic: {}", e);
                        processed = self.apply_basic_transforms(&processed, target).await?;
                    }
                }
            }
        } else if let Some(ref target) = self.conversion_target {
            processed = self.apply_basic_transforms(&processed, target).await?;
        }

        Ok(processed)
    }

    async fn apply_basic_transforms(
        &self,
        audio: &[f32],
        target: &ConversionTarget,
    ) -> Result<Vec<f32>> {
        let mut processed = audio.to_vec();

        // Apply transformations based on target characteristics
        let chars = &target.characteristics;

        // Pitch transformation
        if chars.pitch.mean_f0 != 150.0 {
            let pitch_factor = chars.pitch.mean_f0 / 150.0;
            let transform = PitchTransform::new(pitch_factor);
            processed = transform.apply(&processed)?;
        }

        // Speed transformation
        if chars.timing.speaking_rate != 1.0 {
            let transform = SpeedTransform::new(chars.timing.speaking_rate);
            processed = transform.apply(&processed)?;
        }

        // Gender transformation
        if let Some(gender) = chars.gender {
            let gender_value = match gender {
                crate::types::Gender::Male => -1.0,
                crate::types::Gender::Female => 1.0,
                crate::types::Gender::NonBinary => 0.0,
                crate::types::Gender::Other => 0.0,
                crate::types::Gender::Unknown => 0.0,
            };
            let transform = GenderTransform::new(gender_value);
            processed = transform.apply(&processed)?;
        }

        // Age transformation
        if let Some(age_group) = chars.age_group {
            let source_age = 30.0; // Default
            let target_age = match age_group {
                crate::types::AgeGroup::Child => 8.0,
                crate::types::AgeGroup::Teen => 16.0,
                crate::types::AgeGroup::YoungAdult => 25.0,
                crate::types::AgeGroup::Adult => 35.0,
                crate::types::AgeGroup::MiddleAged => 45.0,
                crate::types::AgeGroup::Senior => 65.0,
                crate::types::AgeGroup::Unknown => 30.0,
            };
            let transform = AgeTransform::new(source_age, target_age);
            processed = transform.apply(&processed)?;
        }

        Ok(processed)
    }

    fn create_realtime_pipeline(config: &RealtimeConfig) -> Result<ProcessingPipeline> {
        let mut pipeline = ProcessingPipeline::new();

        // Add real-time optimized stages
        pipeline.add_stage(
            ProcessingStage::new("normalize".to_string(), StageType::Normalize)
                .with_parameter("target_level".to_string(), 0.95)
                .with_parallel(true),
        );

        // Add noise reduction for quality above 0.5
        pipeline.add_stage(
            ProcessingStage::new("denoise".to_string(), StageType::NoiseReduction)
                .with_parameter("noise_threshold".to_string(), 0.02)
                .with_parallel(true),
        );

        // Add light compression for dynamic range control
        pipeline.add_stage(
            ProcessingStage::new("compress".to_string(), StageType::Compression)
                .with_parameter("ratio".to_string(), 3.0)
                .with_parameter("threshold".to_string(), 0.8)
                .with_parallel(true),
        );

        Ok(pipeline)
    }
}

impl Default for RealtimeConverter {
    fn default() -> Self {
        Self::new(RealtimeConfig::default()).expect("Failed to create default RealtimeConverter")
    }
}

/// Performance metrics for real-time processing
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total samples processed
    pub total_samples: u64,
    /// Total processing time
    pub total_processing_time_ms: f64,
    /// Number of processing operations
    pub processing_count: u64,
    /// Maximum processing time for a single chunk
    pub max_processing_time_ms: f32,
    /// Minimum processing time for a single chunk
    pub min_processing_time_ms: f32,
    /// Number of latency violations
    pub latency_violations: u64,
}

impl PerformanceMetrics {
    /// Update metrics with new processing information
    pub fn update(&mut self, processing_time: std::time::Duration, sample_count: usize) {
        let time_ms = processing_time.as_millis() as f64;

        self.total_samples += sample_count as u64;
        self.total_processing_time_ms += time_ms;
        self.processing_count += 1;

        let time_ms_f32 = time_ms as f32;
        if self.processing_count == 1 {
            self.max_processing_time_ms = time_ms_f32;
            self.min_processing_time_ms = time_ms_f32;
        } else {
            self.max_processing_time_ms = self.max_processing_time_ms.max(time_ms_f32);
            self.min_processing_time_ms = self.min_processing_time_ms.min(time_ms_f32);
        }
    }

    /// Get average processing time per chunk
    pub fn average_processing_time_ms(&self) -> f32 {
        if self.processing_count == 0 {
            0.0
        } else {
            (self.total_processing_time_ms / self.processing_count as f64) as f32
        }
    }

    /// Get processing efficiency (samples per millisecond)
    pub fn processing_efficiency(&self) -> f32 {
        if self.total_processing_time_ms == 0.0 {
            0.0
        } else {
            self.total_samples as f32 / self.total_processing_time_ms as f32
        }
    }

    /// Get real-time factor (how much faster than real-time)
    pub fn realtime_factor(&self, sample_rate: u32) -> f32 {
        if self.total_processing_time_ms == 0.0 {
            0.0
        } else {
            let audio_duration_ms = (self.total_samples as f32 / sample_rate as f32) * 1000.0;
            audio_duration_ms / self.total_processing_time_ms as f32
        }
    }
}
