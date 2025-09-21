# VoiRS Emotion Integration Guide

This guide provides comprehensive examples for integrating the VoiRS emotion control system into various applications and workflows.

## Table of Contents

1. [Real-time Audio Processing](#real-time-audio-processing)
2. [Batch Audio Processing](#batch-audio-processing)
3. [SSML Integration](#ssml-integration)
4. [Multi-speaker Systems](#multi-speaker-systems)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Testing and Validation](#testing-and-validation)

## Real-time Audio Processing

### Streaming Audio with Emotion Control

```rust
use voirs_emotion::prelude::*;
use tokio::time::{interval, Duration};
use tokio::sync::mpsc;

pub struct RealtimeEmotionProcessor {
    processor: EmotionProcessor,
    config: RealtimeEmotionConfig,
}

impl RealtimeEmotionProcessor {
    pub async fn new() -> Result<Self> {
        let config = RealtimeEmotionConfig {
            buffer_size: 1024,
            sample_rate: 44100,
            update_interval: Duration::from_millis(50),
            enable_adaptation: true,
            smoothing_factor: 0.8,
        };
        
        let processor = EmotionProcessor::builder()
            .config(
                EmotionConfig::builder()
                    .transition_smoothing(0.7)
                    .use_gpu(true) // Enable GPU for real-time performance
                    .build()?
            )
            .build()?;
        
        Ok(Self { processor, config })
    }
    
    pub async fn process_audio_stream(
        &self,
        mut audio_receiver: mpsc::Receiver<Vec<f32>>,
        emotion_receiver: mpsc::Receiver<(Emotion, f32)>,
    ) -> Result<mpsc::Receiver<Vec<f32>>> {
        let (output_sender, output_receiver) = mpsc::channel(10);
        let processor = self.processor.clone();
        
        tokio::spawn(async move {
            let mut emotion_rx = emotion_receiver;
            
            while let Some(audio_chunk) = audio_receiver.recv().await {
                // Check for emotion updates
                if let Ok((emotion, intensity)) = emotion_rx.try_recv() {
                    if let Err(e) = processor.set_emotion(emotion, Some(intensity)).await {
                        eprintln!("Failed to set emotion: {}", e);
                    }
                }
                
                // Process audio with current emotion
                match processor.process_audio(&audio_chunk).await {
                    Ok(processed_audio) => {
                        if output_sender.send(processed_audio).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        eprintln!("Audio processing error: {}", e);
                        // Send original audio as fallback
                        if output_sender.send(audio_chunk).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
        
        Ok(output_receiver)
    }
}

// Usage example
#[tokio::main]
async fn realtime_example() -> Result<()> {
    let emotion_processor = RealtimeEmotionProcessor::new().await?;
    
    let (audio_sender, audio_receiver) = mpsc::channel(10);
    let (emotion_sender, emotion_receiver) = mpsc::channel(10);
    
    let output_receiver = emotion_processor
        .process_audio_stream(audio_receiver, emotion_receiver)
        .await?;
    
    // Simulate real-time audio input
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_millis(20)); // 50Hz update rate
        
        for i in 0..1000 {
            interval.tick().await;
            
            // Generate test audio chunk
            let audio_chunk: Vec<f32> = (0..1024)
                .map(|j| 0.1 * ((i * 1024 + j) as f32 * 0.01).sin())
                .collect();
            
            if audio_sender.send(audio_chunk).await.is_err() {
                break;
            }
        }
    });
    
    // Simulate emotion changes
    tokio::spawn(async move {
        let emotions = [
            (Emotion::Happy, 0.8),
            (Emotion::Excited, 0.9),
            (Emotion::Calm, 0.6),
            (Emotion::Thoughtful, 0.7),
        ];
        
        for (emotion, intensity) in emotions.iter().cycle().take(20) {
            tokio::time::sleep(Duration::from_secs(2)).await;
            if emotion_sender.send((emotion.clone(), *intensity)).await.is_err() {
                break;
            }
        }
    });
    
    // Process output (in a real application, this would go to audio output)
    let mut processed_chunks = 0;
    let mut output_rx = output_receiver;
    
    while let Some(_processed_audio) = output_rx.recv().await {
        processed_chunks += 1;
        if processed_chunks >= 100 {
            break;
        }
    }
    
    println!("Processed {} audio chunks in real-time", processed_chunks);
    Ok(())
}
```

## Batch Audio Processing

### Processing Multiple Audio Files with Different Emotions

```rust
use voirs_emotion::prelude::*;
use std::path::Path;
use tokio::fs;

pub struct BatchEmotionProcessor {
    processor: EmotionProcessor,
    history: EmotionHistory,
}

impl BatchEmotionProcessor {
    pub async fn new() -> Result<Self> {
        // Configure for high-quality batch processing
        let config = EmotionConfig::builder()
            .enabled(true)
            .transition_smoothing(0.95) // Very smooth for high quality
            .prosody_strength(0.9)
            .voice_quality_strength(0.8)
            .use_gpu(true) // Utilize GPU for faster batch processing
            .build()?;
        
        let processor = EmotionProcessor::with_config(config)?;
        let history = EmotionHistory::new();
        
        Ok(Self { processor, history })
    }
    
    pub async fn process_batch(
        &mut self,
        jobs: Vec<BatchJob>,
    ) -> Result<Vec<BatchResult>> {
        let mut results = Vec::new();
        
        for job in jobs {
            println!("Processing: {} with emotion {:?}", job.input_file, job.emotion);
            
            // Load audio file (placeholder - replace with actual audio loading)
            let audio_data = self.load_audio_file(&job.input_file).await?;
            
            // Set emotion for this job
            self.processor.set_emotion(job.emotion.clone(), Some(job.intensity)).await?;
            
            // Wait for emotion transition to complete
            self.wait_for_transition_completion().await?;
            
            // Process the audio
            let start_time = std::time::Instant::now();
            let processed_audio = self.processor.process_audio(&audio_data).await?;
            let processing_time = start_time.elapsed();
            
            // Save processed audio (placeholder)
            self.save_audio_file(&job.output_file, &processed_audio).await?;
            
            // Record processing statistics
            let result = BatchResult {
                input_file: job.input_file.clone(),
                output_file: job.output_file.clone(),
                emotion: job.emotion.clone(),
                intensity: job.intensity,
                processing_time,
                input_duration: Duration::from_secs_f32(audio_data.len() as f32 / 44100.0),
                success: true,
                error_message: None,
            };
            
            results.push(result);
            
            // Add to emotion history for analysis
            self.history.add_entry(EmotionHistoryEntry {
                emotion: job.emotion,
                intensity: EmotionIntensity::new(job.intensity)?,
                timestamp: std::time::Instant::now(),
                duration: Some(processing_time),
                confidence: 1.0,
                context: Some(job.input_file),
                metadata: None,
            });
        }
        
        Ok(results)
    }
    
    async fn load_audio_file(&self, _file_path: &str) -> Result<Vec<f32>> {
        // Placeholder: In a real implementation, use a library like `hound` or `symphonia`
        // to load various audio formats
        Ok(vec![0.1; 44100]) // 1 second of placeholder audio
    }
    
    async fn save_audio_file(&self, _file_path: &str, _audio_data: &[f32]) -> Result<()> {
        // Placeholder: In a real implementation, save using appropriate audio library
        Ok(())
    }
    
    async fn wait_for_transition_completion(&self) -> Result<()> {
        for _ in 0..100 { // Max 10 seconds
            let state = self.processor.get_current_state().await;
            if !state.is_transitioning() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(()) // Continue even if transition didn't complete
    }
    
    pub fn get_processing_statistics(&self) -> BatchStatistics {
        let entries = self.history.get_all_entries();
        
        let total_jobs = entries.len();
        let total_duration: Duration = entries.iter()
            .filter_map(|e| e.duration)
            .sum();
        
        let emotion_counts: std::collections::HashMap<Emotion, usize> = 
            entries.iter()
            .fold(std::collections::HashMap::new(), |mut acc, entry| {
                *acc.entry(entry.emotion.clone()).or_insert(0) += 1;
                acc
            });
        
        BatchStatistics {
            total_jobs,
            total_processing_time: total_duration,
            average_processing_time: if total_jobs > 0 {
                total_duration / total_jobs as u32
            } else {
                Duration::ZERO
            },
            emotion_distribution: emotion_counts,
        }
    }
}

pub struct BatchJob {
    pub input_file: String,
    pub output_file: String,
    pub emotion: Emotion,
    pub intensity: f32,
}

pub struct BatchResult {
    pub input_file: String,
    pub output_file: String,
    pub emotion: Emotion,
    pub intensity: f32,
    pub processing_time: Duration,
    pub input_duration: Duration,
    pub success: bool,
    pub error_message: Option<String>,
}

pub struct BatchStatistics {
    pub total_jobs: usize,
    pub total_processing_time: Duration,
    pub average_processing_time: Duration,
    pub emotion_distribution: std::collections::HashMap<Emotion, usize>,
}

// Usage example
#[tokio::main]
async fn batch_processing_example() -> Result<()> {
    let mut batch_processor = BatchEmotionProcessor::new().await?;
    
    let jobs = vec![
        BatchJob {
            input_file: "input1.wav".to_string(),
            output_file: "output1_happy.wav".to_string(),
            emotion: Emotion::Happy,
            intensity: 0.8,
        },
        BatchJob {
            input_file: "input2.wav".to_string(),
            output_file: "output2_sad.wav".to_string(),
            emotion: Emotion::Sad,
            intensity: 0.7,
        },
        BatchJob {
            input_file: "input3.wav".to_string(),
            output_file: "output3_excited.wav".to_string(),
            emotion: Emotion::Excited,
            intensity: 0.9,
        },
    ];
    
    let results = batch_processor.process_batch(jobs).await?;
    
    // Print results
    for result in &results {
        println!("Processed {} -> {} ({:?}) in {:.2}ms",
                 result.input_file,
                 result.output_file,
                 result.emotion,
                 result.processing_time.as_secs_f64() * 1000.0);
    }
    
    // Get processing statistics
    let stats = batch_processor.get_processing_statistics();
    println!("\nBatch Statistics:");
    println!("  Total jobs: {}", stats.total_jobs);
    println!("  Total time: {:.2}s", stats.total_processing_time.as_secs_f64());
    println!("  Average time per job: {:.2}ms", 
             stats.average_processing_time.as_secs_f64() * 1000.0);
    
    Ok(())
}
```

## SSML Integration

### Processing SSML with Emotion Markup

```rust
use voirs_emotion::prelude::*;

pub struct SSMLEmotionProcessor {
    processor: EmotionProcessor,
    ssml_processor: EmotionSSMLProcessor,
}

impl SSMLEmotionProcessor {
    pub async fn new() -> Result<Self> {
        let processor = EmotionProcessor::new()?;
        let ssml_processor = EmotionSSMLProcessor::new();
        
        Ok(Self { processor, ssml_processor })
    }
    
    pub async fn process_ssml_document(&self, ssml_content: &str) -> Result<ProcessedSSML> {
        // Parse SSML content
        let segments = self.ssml_processor.parse_emotion_segments(ssml_content)?;
        
        let mut processed_segments = Vec::new();
        
        for segment in segments {
            // Apply emotion from SSML markup
            if let Some(emotion_attrs) = &segment.emotion_attributes {
                let emotion_params = emotion_attrs.to_emotion_parameters()?;
                self.processor.apply_emotion_parameters(emotion_params).await?;
            }
            
            // In a real implementation, this would process the text content
            // with the current emotion settings
            let audio_data = self.synthesize_text_with_emotion(&segment.text).await?;
            
            processed_segments.push(ProcessedSegment {
                text: segment.text.clone(),
                emotion: segment.emotion_attributes.clone(),
                audio_data,
                duration: Duration::from_secs_f32(audio_data.len() as f32 / 44100.0),
            });
        }
        
        Ok(ProcessedSSML {
            segments: processed_segments,
            total_duration: processed_segments.iter()
                .map(|s| s.duration)
                .sum(),
        })
    }
    
    async fn synthesize_text_with_emotion(&self, _text: &str) -> Result<Vec<f32>> {
        // Placeholder: In a real implementation, this would integrate with
        // a text-to-speech system that uses the current emotion state
        let current_state = self.processor.get_current_state().await;
        let energy_scale = current_state.current.emotion_parameters.energy_scale;
        
        // Generate audio based on emotion parameters
        let base_audio: Vec<f32> = (0..44100) // 1 second
            .map(|i| 0.1 * energy_scale * (i as f32 * 0.01).sin())
            .collect();
        
        self.processor.process_audio(&base_audio).await
    }
}

pub struct ProcessedSSML {
    pub segments: Vec<ProcessedSegment>,
    pub total_duration: Duration,
}

pub struct ProcessedSegment {
    pub text: String,
    pub emotion: Option<EmotionAttributes>,
    pub audio_data: Vec<f32>,
    pub duration: Duration,
}

// Example SSML processing
#[tokio::main]
async fn ssml_example() -> Result<()> {
    let processor = SSMLEmotionProcessor::new().await?;
    
    let ssml_content = r#"
    <speak>
        <emotion name="happy" intensity="high">
            Welcome to our voice synthesis system!
        </emotion>
        
        <emotion name="calm" intensity="medium">
            Today I'll be demonstrating the emotion control features.
        </emotion>
        
        <emotion name="excited" intensity="high" rate="fast">
            This is really exciting technology!
        </emotion>
        
        <emotion name="thoughtful" intensity="medium" rate="slow">
            Let me explain how this works in detail...
        </emotion>
    </speak>
    "#;
    
    let result = processor.process_ssml_document(ssml_content).await?;
    
    println!("Processed SSML document:");
    println!("  Total segments: {}", result.segments.len());
    println!("  Total duration: {:.2}s", result.total_duration.as_secs_f64());
    
    for (i, segment) in result.segments.iter().enumerate() {
        println!("  Segment {}: \"{}\" ({:.2}s)",
                 i + 1,
                 &segment.text[..std::cmp::min(50, segment.text.len())],
                 segment.duration.as_secs_f64());
        
        if let Some(emotion) = &segment.emotion {
            println!("    Emotion: {} (intensity: {})",
                     emotion.name.as_ref().unwrap_or(&"none".to_string()),
                     emotion.intensity.as_ref().unwrap_or(&"medium".to_string()));
        }
    }
    
    Ok(())
}
```

## Performance Optimization

### Optimizing for Different Use Cases

```rust
use voirs_emotion::prelude::*;

// Low-latency configuration for real-time applications
pub fn create_realtime_processor() -> Result<EmotionProcessor> {
    let config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.3)    // Fast transitions
        .prosody_strength(0.7)        // Moderate effects for speed
        .voice_quality_strength(0.5)  // Lighter processing
        .max_emotions(2)              // Limit complexity
        .use_gpu(true)                // Use GPU if available
        .build()?;
    
    EmotionProcessor::with_config(config)
}

// High-quality configuration for content creation
pub fn create_production_processor() -> Result<EmotionProcessor> {
    let config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.95)   // Very smooth transitions
        .prosody_strength(1.0)        // Full prosodic effects
        .voice_quality_strength(0.9)  // Rich voice processing
        .max_emotions(5)              // Complex emotion mixing
        .use_gpu(true)                // Use all available performance
        .build()?;
    
    EmotionProcessor::with_config(config)
}

// Memory-efficient configuration for embedded systems
pub fn create_embedded_processor() -> Result<EmotionProcessor> {
    let config = EmotionConfig::builder()
        .enabled(true)
        .transition_smoothing(0.6)    // Moderate smoothing
        .prosody_strength(0.6)        // Basic effects only
        .voice_quality_strength(0.3)  // Minimal voice processing
        .max_emotions(1)              // Single emotion only
        .use_gpu(false)               // CPU only for compatibility
        .build()?;
    
    EmotionProcessor::with_config(config)
}

// Performance benchmarking
#[tokio::main]
async fn performance_comparison() -> Result<()> {
    let processors = [
        ("Real-time", create_realtime_processor()?),
        ("Production", create_production_processor()?),
        ("Embedded", create_embedded_processor()?),
    ];
    
    let test_audio = vec![0.1; 44100]; // 1 second of audio
    
    for (name, processor) in &processors {
        // Set a test emotion
        processor.set_emotion(Emotion::Happy, Some(0.8)).await?;
        
        // Benchmark processing time
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _processed = processor.process_audio(&test_audio).await?;
        }
        let total_time = start.elapsed();
        let avg_time = total_time / 100;
        
        println!("{} processor:", name);
        println!("  Average processing time: {:.2}ms", avg_time.as_secs_f64() * 1000.0);
        println!("  Real-time factor: {:.2}x", 
                 1000.0 / (avg_time.as_secs_f64() * 1000.0)); // How many times faster than real-time
        
        // Memory usage would require additional tooling in a real scenario
        if processor.is_gpu_enabled() {
            println!("  GPU acceleration: enabled");
        } else {
            println!("  GPU acceleration: disabled");
        }
        println!();
    }
    
    Ok(())
}
```

## Production Deployment

### Production-Ready Emotion Service

```rust
use voirs_emotion::prelude::*;
use tokio::sync::{RwLock, Semaphore};
use std::sync::Arc;
use std::collections::HashMap;

pub struct EmotionService {
    processors: Arc<RwLock<HashMap<String, EmotionProcessor>>>,
    semaphore: Arc<Semaphore>,
    config: ServiceConfig,
}

pub struct ServiceConfig {
    pub max_concurrent_requests: usize,
    pub default_timeout: Duration,
    pub enable_metrics: bool,
    pub enable_gpu: bool,
}

impl EmotionService {
    pub async fn new(config: ServiceConfig) -> Result<Self> {
        let processors = Arc::new(RwLock::new(HashMap::new()));
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        
        Ok(Self {
            processors,
            semaphore,
            config,
        })
    }
    
    pub async fn process_request(&self, request: EmotionRequest) -> Result<EmotionResponse> {
        // Acquire semaphore permit for rate limiting
        let _permit = self.semaphore.acquire().await
            .map_err(|_| Error::Processing("Service overloaded".to_string()))?;
        
        // Get or create processor for this session
        let processor = self.get_or_create_processor(&request.session_id).await?;
        
        let start_time = std::time::Instant::now();
        
        // Set emotion
        processor.set_emotion(request.emotion, Some(request.intensity)).await?;
        
        // Process audio
        let processed_audio = processor.process_audio(&request.audio_data).await?;
        
        let processing_time = start_time.elapsed();
        
        // Return response with metrics
        Ok(EmotionResponse {
            session_id: request.session_id,
            processed_audio,
            processing_time_ms: processing_time.as_millis() as u32,
            emotion_applied: request.emotion,
            intensity_applied: request.intensity,
            success: true,
            error_message: None,
        })
    }
    
    async fn get_or_create_processor(&self, session_id: &str) -> Result<EmotionProcessor> {
        {
            let processors = self.processors.read().await;
            if let Some(processor) = processors.get(session_id) {
                return Ok(processor.clone());
            }
        }
        
        // Create new processor
        let processor_config = EmotionConfig::builder()
            .enabled(true)
            .transition_smoothing(0.8)
            .use_gpu(self.config.enable_gpu)
            .build()?;
        
        let processor = EmotionProcessor::with_config(processor_config)?;
        
        // Store in cache
        {
            let mut processors = self.processors.write().await;
            processors.insert(session_id.to_string(), processor.clone());
        }
        
        Ok(processor)
    }
    
    pub async fn cleanup_sessions(&self, max_age: Duration) {
        // In a real implementation, you'd track session last access times
        // and remove old sessions to prevent memory leaks
        let mut processors = self.processors.write().await;
        
        // Simple cleanup: remove all sessions older than max_age
        // This is a placeholder - real implementation would track timestamps
        if processors.len() > 1000 { // Arbitrary limit
            processors.clear();
        }
    }
    
    pub async fn get_service_statistics(&self) -> ServiceStatistics {
        let processors = self.processors.read().await;
        
        ServiceStatistics {
            active_sessions: processors.len(),
            available_permits: self.semaphore.available_permits(),
            gpu_enabled: self.config.enable_gpu,
        }
    }
}

pub struct EmotionRequest {
    pub session_id: String,
    pub emotion: Emotion,
    pub intensity: f32,
    pub audio_data: Vec<f32>,
}

pub struct EmotionResponse {
    pub session_id: String,
    pub processed_audio: Vec<f32>,
    pub processing_time_ms: u32,
    pub emotion_applied: Emotion,
    pub intensity_applied: f32,
    pub success: bool,
    pub error_message: Option<String>,
}

pub struct ServiceStatistics {
    pub active_sessions: usize,
    pub available_permits: usize,
    pub gpu_enabled: bool,
}

// Example production service usage
#[tokio::main]
async fn production_service_example() -> Result<()> {
    let config = ServiceConfig {
        max_concurrent_requests: 10,
        default_timeout: Duration::from_secs(30),
        enable_metrics: true,
        enable_gpu: true,
    };
    
    let service = EmotionService::new(config).await?;
    
    // Simulate multiple concurrent requests
    let mut handles = Vec::new();
    
    for i in 0..20 {
        let service = service.clone();
        let handle = tokio::spawn(async move {
            let request = EmotionRequest {
                session_id: format!("session_{}", i % 5), // 5 different sessions
                emotion: match i % 4 {
                    0 => Emotion::Happy,
                    1 => Emotion::Sad,
                    2 => Emotion::Excited,
                    _ => Emotion::Calm,
                },
                intensity: 0.7 + (i as f32 * 0.02),
                audio_data: vec![0.1; 1024], // Small audio chunk
            };
            
            match service.process_request(request).await {
                Ok(response) => {
                    println!("Request {} processed in {}ms", 
                             i, response.processing_time_ms);
                }
                Err(e) => {
                    println!("Request {} failed: {}", i, e);
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Get service statistics
    let stats = service.get_service_statistics().await;
    println!("\nService Statistics:");
    println!("  Active sessions: {}", stats.active_sessions);
    println!("  Available permits: {}", stats.available_permits);
    println!("  GPU enabled: {}", stats.gpu_enabled);
    
    Ok(())
}
```

## Conclusion

This integration guide provides practical examples for incorporating the VoiRS emotion control system into various types of applications:

- **Real-time systems** benefit from streaming processing and low-latency configurations
- **Batch processing** systems can utilize high-quality settings and parallel processing
- **SSML integration** enables rich emotional markup in text-to-speech applications
- **Multi-speaker systems** can maintain separate emotion states per speaker
- **Production services** require proper resource management, error handling, and monitoring

Each pattern can be adapted and combined based on your specific requirements. The emotion system is designed to be flexible and performant across a wide range of use cases.

For additional examples and advanced patterns, see:
- `examples_basic.rs` - Fundamental usage patterns
- `examples_advanced.rs` - Advanced features and machine learning
- Test files in `tests/` - Integration and stress testing examples
- Benchmark files in `benches/` - Performance optimization examples