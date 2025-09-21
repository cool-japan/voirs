# Tutorial 6: Real-time Processing

**Duration**: 40-45 minutes  
**Level**: Intermediate  
**Prerequisites**: Tutorials 1-5 completed

## Overview

Real-time processing is essential for interactive applications like voice assistants, live streaming, and conversational AI. This tutorial covers streaming synthesis, latency optimization, and building responsive voice systems.

## What You'll Learn

- Understanding real-time constraints and latency requirements
- Streaming synthesis and buffering strategies
- Latency optimization techniques
- Building interactive voice applications
- Performance monitoring and optimization
- Handling network issues and buffering

## Real-time Fundamentals

### Understanding Latency Requirements

```rust
use voirs_sdk::{StreamingSynthesizer, LatencyTarget, BufferingStrategy};
use std::time::{Duration, Instant};

fn analyze_latency_requirements() {
    println!("🎯 Real-time Application Latency Requirements:");
    println!();
    
    let requirements = vec![
        ("Voice Assistant", "< 200ms", "Interactive conversation"),
        ("Live Broadcasting", "< 100ms", "Professional broadcasting"),
        ("Gaming", "< 50ms", "Real-time character dialogue"),
        ("Phone Calls", "< 150ms", "Natural conversation flow"),
        ("Live Streaming", "< 500ms", "Audience interaction"),
        ("Accessibility", "< 300ms", "Screen reader applications"),
    ];
    
    println!("{:<20} {:<12} {:<30}", "Application", "Target", "Use Case");
    println!("{}", "-".repeat(65));
    
    for (app, target, use_case) in requirements {
        println!("{:<20} {:<12} {:<30}", app, target, use_case);
    }
    
    println!("\n💡 Key Factors:");
    println!("  • End-to-end latency includes processing + network + buffering");
    println!("  • Interactive applications need <200ms for natural feel");
    println!("  • Quality vs. latency trade-offs are essential");
    println!("  • Predictive processing can reduce perceived latency");
}
```

### Basic Streaming Setup

```rust
use voirs_sdk::{VoirsSdk, StreamingConfig, StreamingSynthesizer};
use tokio::sync::mpsc;

#[tokio::main]
async fn basic_streaming_setup() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    
    // Configure streaming parameters
    let streaming_config = StreamingConfig::builder()
        .chunk_size(512)                    // Audio chunk size in samples
        .buffer_size(2048)                  // Internal buffer size
        .latency_target(Duration::from_millis(100))  // Target latency
        .quality_vs_latency_balance(0.7)    // Favor latency slightly
        .enable_predictive_synthesis(true)   // Start synthesis early
        .build()?;
    
    // Create streaming synthesizer
    let mut synthesizer = sdk.create_streaming_synthesizer(&streaming_config).await?;
    
    println!("🎙️ Streaming synthesizer ready");
    println!("Configuration:");
    println!("  Chunk size: {} samples", streaming_config.chunk_size);
    println!("  Target latency: {}ms", streaming_config.latency_target.as_millis());
    println!("  Buffer size: {} samples", streaming_config.buffer_size);
    
    // Test with simple text
    let text = "This is a test of real-time streaming synthesis.";
    
    println!("\n🔄 Starting streaming synthesis...");
    let start_time = Instant::now();
    
    let mut stream = synthesizer.synthesize_streaming(text).await?;
    let mut first_chunk_time = None;
    let mut chunk_count = 0;
    
    while let Some(chunk) = stream.next().await {
        chunk_count += 1;
        
        if first_chunk_time.is_none() {
            first_chunk_time = Some(start_time.elapsed());
            println!("⚡ First chunk received in: {:?}", first_chunk_time.unwrap());
        }
        
        // In a real application, you would send this chunk to audio output
        println!("📦 Received chunk {} ({} samples)", chunk_count, chunk.len());
        
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    let total_time = start_time.elapsed();
    println!("\n✅ Streaming complete!");
    println!("  Total time: {:?}", total_time);
    println!("  Chunks received: {}", chunk_count);
    println!("  Average chunk interval: {:?}", total_time / chunk_count as u32);
    
    Ok(())
}
```

## Advanced Streaming Techniques

### Predictive Synthesis

```rust
use voirs_sdk::{PredictiveSynthesizer, TextPredictor, SynthesisPredictor};

struct PredictiveStreamingSynthesizer {
    synthesizer: StreamingSynthesizer,
    text_predictor: TextPredictor,
    synthesis_predictor: SynthesisPredictor,
    prediction_buffer: VecDeque<PredictionEntry>,
}

impl PredictiveStreamingSynthesizer {
    async fn new(config: &StreamingConfig) -> anyhow::Result<Self> {
        let sdk = VoirsSdk::new().await?;
        
        Ok(Self {
            synthesizer: sdk.create_streaming_synthesizer(config).await?,
            text_predictor: TextPredictor::new()?,
            synthesis_predictor: SynthesisPredictor::new()?,
            prediction_buffer: VecDeque::new(),
        })
    }
    
    async fn process_text_incrementally(&mut self, partial_text: &str) -> anyhow::Result<()> {
        // Predict what might come next
        let predictions = self.text_predictor.predict_continuation(partial_text, 3)?;
        
        println!("🔮 Text predictions for '{}':", partial_text);
        for (i, prediction) in predictions.iter().enumerate() {
            println!("  {}: {} (confidence: {:.2})", 
                i + 1, prediction.text, prediction.confidence);
        }
        
        // Pre-synthesize likely continuations
        for prediction in predictions {
            if prediction.confidence > 0.7 {  // Only pre-synthesize high-confidence predictions
                let full_text = format!("{} {}", partial_text, prediction.text);
                
                // Start background synthesis
                let prediction_entry = PredictionEntry {
                    text: full_text.clone(),
                    confidence: prediction.confidence,
                    synthesis_future: self.synthesizer.synthesize_background(&full_text),
                    created_at: Instant::now(),
                };
                
                self.prediction_buffer.push_back(prediction_entry);
                
                println!("⚡ Pre-synthesizing: {} (confidence: {:.2})", 
                    full_text, prediction.confidence);
            }
        }
        
        // Clean up old predictions
        self.cleanup_old_predictions();
        
        Ok(())
    }
    
    async fn synthesize_with_prediction(&mut self, final_text: &str) -> anyhow::Result<AudioStream> {
        // Check if we have a pre-synthesized version
        for (i, prediction) in self.prediction_buffer.iter().enumerate() {
            if prediction.text == final_text {
                println!("🎯 Using pre-synthesized audio for: {}", final_text);
                
                let audio_stream = prediction.synthesis_future.await?;
                self.prediction_buffer.remove(i);
                
                return Ok(audio_stream);
            }
        }
        
        // No prediction available, synthesize normally
        println!("🔄 Synthesizing live: {}", final_text);
        self.synthesizer.synthesize_streaming(final_text).await
    }
    
    fn cleanup_old_predictions(&mut self) {
        let cutoff_time = Instant::now() - Duration::from_secs(5);  // 5-second timeout
        
        self.prediction_buffer.retain(|prediction| {
            prediction.created_at > cutoff_time
        });
    }
}

struct PredictionEntry {
    text: String,
    confidence: f64,
    synthesis_future: SynthesisFuture,
    created_at: Instant,
}

async fn predictive_synthesis_demo() -> anyhow::Result<()> {
    let streaming_config = StreamingConfig::builder()
        .latency_target(Duration::from_millis(50))
        .enable_predictive_synthesis(true)
        .build()?;
    
    let mut predictive_synthesizer = PredictiveStreamingSynthesizer::new(&streaming_config).await?;
    
    // Simulate incremental text input (like typing)
    let incremental_inputs = vec![
        "Hello",
        "Hello there",
        "Hello there, how",
        "Hello there, how are",
        "Hello there, how are you",
        "Hello there, how are you today?",
    ];
    
    for input in incremental_inputs {
        println!("\n📝 Processing incremental input: '{}'", input);
        
        // Process and predict
        predictive_synthesizer.process_text_incrementally(input).await?;
        
        // Small delay to simulate typing
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    
    // Final synthesis
    let final_text = "Hello there, how are you today?";
    println!("\n🎬 Final synthesis for: '{}'", final_text);
    
    let start_time = Instant::now();
    let stream = predictive_synthesizer.synthesize_with_prediction(final_text).await?;
    let synthesis_time = start_time.elapsed();
    
    println!("⚡ Synthesis started in: {:?}", synthesis_time);
    
    // Process the stream
    let mut chunk_count = 0;
    let mut stream = stream;
    
    while let Some(chunk) = stream.next().await {
        chunk_count += 1;
        println!("📦 Chunk {} received", chunk_count);
    }
    
    Ok(())
}
```

### Dynamic Buffer Management

```rust
use voirs_sdk::{DynamicBuffer, BufferStrategy, NetworkMonitor};

struct AdaptiveStreamingBuffer {
    buffer: DynamicBuffer,
    network_monitor: NetworkMonitor,
    strategy: BufferStrategy,
    target_latency: Duration,
    current_network_quality: f64,
}

impl AdaptiveStreamingBuffer {
    fn new(target_latency: Duration) -> anyhow::Result<Self> {
        Ok(Self {
            buffer: DynamicBuffer::new()?,
            network_monitor: NetworkMonitor::new()?,
            strategy: BufferStrategy::Adaptive,
            target_latency,
            current_network_quality: 1.0,
        })
    }
    
    async fn process_audio_chunk(&mut self, chunk: AudioChunk) -> anyhow::Result<()> {
        // Monitor network conditions
        self.update_network_quality().await?;
        
        // Adapt buffer strategy based on conditions
        self.adapt_buffer_strategy();
        
        // Add chunk to buffer with current strategy
        self.buffer.add_chunk(chunk, &self.strategy)?;
        
        // Check if we should emit buffered audio
        if self.should_emit_audio() {
            self.emit_buffered_audio().await?;
        }
        
        Ok(())
    }
    
    async fn update_network_quality(&mut self) -> anyhow::Result<()> {
        let network_stats = self.network_monitor.get_current_stats().await?;
        
        // Calculate quality score based on latency, jitter, and packet loss
        self.current_network_quality = self.calculate_quality_score(
            network_stats.latency,
            network_stats.jitter,
            network_stats.packet_loss_rate,
        );
        
        println!("📊 Network quality: {:.2} (latency: {}ms, jitter: {}ms, loss: {:.1}%)",
            self.current_network_quality,
            network_stats.latency.as_millis(),
            network_stats.jitter.as_millis(),
            network_stats.packet_loss_rate * 100.0
        );
        
        Ok(())
    }
    
    fn adapt_buffer_strategy(&mut self) {
        self.strategy = if self.current_network_quality > 0.8 {
            // Good network: minimize buffering for low latency
            BufferStrategy::MinimalBuffering {
                min_buffer_size: 256,
                target_buffer_size: 512,
            }
        } else if self.current_network_quality > 0.5 {
            // Medium network: balanced approach
            BufferStrategy::Adaptive {
                min_buffer_size: 512,
                max_buffer_size: 2048,
                target_latency: self.target_latency,
            }
        } else {
            // Poor network: prioritize stability
            BufferStrategy::Conservative {
                buffer_size: 4096,
                safety_margin: 1024,
            }
        };
        
        println!("🔧 Buffer strategy adapted to: {:?}", self.strategy);
    }
    
    fn should_emit_audio(&self) -> bool {
        match &self.strategy {
            BufferStrategy::MinimalBuffering { target_buffer_size, .. } => {
                self.buffer.size() >= *target_buffer_size
            }
            BufferStrategy::Adaptive { min_buffer_size, .. } => {
                self.buffer.size() >= *min_buffer_size && 
                self.buffer.duration() >= self.target_latency / 2
            }
            BufferStrategy::Conservative { buffer_size, .. } => {
                self.buffer.size() >= *buffer_size
            }
        }
    }
    
    async fn emit_buffered_audio(&mut self) -> anyhow::Result<()> {
        let chunk = self.buffer.get_optimal_chunk(&self.strategy)?;
        
        // In a real application, this would go to audio output
        println!("🔊 Emitting audio chunk: {} samples, {:.2}ms duration",
            chunk.len(), chunk.duration_ms());
        
        Ok(())
    }
    
    fn calculate_quality_score(&self, latency: Duration, jitter: Duration, packet_loss: f64) -> f64 {
        // Simple quality scoring algorithm
        let latency_score = (200.0 - latency.as_millis() as f64).max(0.0) / 200.0;
        let jitter_score = (50.0 - jitter.as_millis() as f64).max(0.0) / 50.0;
        let loss_score = (1.0 - packet_loss).max(0.0);
        
        (latency_score * 0.5 + jitter_score * 0.3 + loss_score * 0.2).clamp(0.0, 1.0)
    }
}
```

## Low-Latency Optimization

### Ultra-Low Latency Techniques

```rust
use voirs_sdk::{LowLatencyConfig, OptimizationLevel, ThreadingStrategy};

async fn ultra_low_latency_setup() -> anyhow::Result<()> {
    println!("⚡ Configuring ultra-low latency synthesis...");
    
    // Configure for minimal latency
    let low_latency_config = LowLatencyConfig::builder()
        .target_latency(Duration::from_millis(20))  // Ultra-low target
        .optimization_level(OptimizationLevel::Maximum)
        .threading_strategy(ThreadingStrategy::Dedicated)
        .enable_simd_acceleration(true)
        .enable_gpu_acceleration(true)
        .disable_quality_enhancements(true)         // Trade quality for speed
        .chunk_size(256)                            // Smaller chunks
        .lookahead_samples(128)                     // Minimal lookahead
        .enable_zero_copy_operations(true)          // Avoid memory copies
        .use_real_time_scheduler(true)              // Real-time OS scheduling
        .build()?;
    
    let sdk = VoirsSdk::new().await?;
    let synthesizer = sdk.create_low_latency_synthesizer(&low_latency_config).await?;
    
    println!("✅ Low-latency synthesizer configured:");
    println!("  Target latency: {}ms", low_latency_config.target_latency.as_millis());
    println!("  Chunk size: {} samples", low_latency_config.chunk_size);
    println!("  SIMD acceleration: {}", low_latency_config.enable_simd_acceleration);
    println!("  GPU acceleration: {}", low_latency_config.enable_gpu_acceleration);
    
    // Benchmark different latency targets
    let test_latencies = vec![
        Duration::from_millis(5),
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(50),
    ];
    
    println!("\n🏁 Latency benchmarking:");
    println!("{:<12} {:<15} {:<15} {:<10}", "Target", "Actual", "Quality", "Success");
    println!("{}", "-".repeat(55));
    
    for target_latency in test_latencies {
        let config = LowLatencyConfig::builder()
            .target_latency(target_latency)
            .optimization_level(OptimizationLevel::Maximum)
            .build()?;
        
        let synthesizer = sdk.create_low_latency_synthesizer(&config).await?;
        
        // Benchmark
        let test_text = "Quick test.";
        let start_time = Instant::now();
        
        let result = synthesizer.synthesize_immediate(test_text).await;
        let actual_latency = start_time.elapsed();
        
        match result {
            Ok(audio) => {
                let quality_score = audio.assess_quality().unwrap_or(0.0);
                println!("{:<12} {:<15} {:<15.3} {:<10}", 
                    format!("{}ms", target_latency.as_millis()),
                    format!("{}ms", actual_latency.as_millis()),
                    quality_score,
                    "✅"
                );
            }
            Err(_) => {
                println!("{:<12} {:<15} {:<15} {:<10}", 
                    format!("{}ms", target_latency.as_millis()),
                    format!("{}ms", actual_latency.as_millis()),
                    "N/A",
                    "❌"
                );
            }
        }
    }
    
    Ok(())
}
```

### Real-time Performance Monitoring

```rust
use voirs_sdk::{PerformanceMonitor, LatencyMetrics, ThroughputMetrics};

struct RealTimePerformanceMonitor {
    monitor: PerformanceMonitor,
    latency_history: VecDeque<Duration>,
    throughput_history: VecDeque<f64>,
    quality_history: VecDeque<f64>,
    alert_thresholds: AlertThresholds,
}

impl RealTimePerformanceMonitor {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            monitor: PerformanceMonitor::new()?,
            latency_history: VecDeque::with_capacity(1000),
            throughput_history: VecDeque::with_capacity(1000),
            quality_history: VecDeque::with_capacity(1000),
            alert_thresholds: AlertThresholds::default(),
        })
    }
    
    async fn monitor_synthesis_operation(&mut self, operation: SynthesisOperation) -> anyhow::Result<()> {
        let start_time = Instant::now();
        
        // Execute the operation
        let result = operation.execute().await?;
        
        let latency = start_time.elapsed();
        let throughput = result.samples_generated as f64 / latency.as_secs_f64();
        let quality = result.quality_score;
        
        // Update history
        self.update_metrics(latency, throughput, quality);
        
        // Check for alerts
        self.check_alerts(latency, throughput, quality)?;
        
        // Log metrics periodically
        if self.latency_history.len() % 100 == 0 {
            self.log_performance_summary();
        }
        
        Ok(())
    }
    
    fn update_metrics(&mut self, latency: Duration, throughput: f64, quality: f64) {
        // Keep only recent history
        if self.latency_history.len() >= 1000 {
            self.latency_history.pop_front();
            self.throughput_history.pop_front();
            self.quality_history.pop_front();
        }
        
        self.latency_history.push_back(latency);
        self.throughput_history.push_back(throughput);
        self.quality_history.push_back(quality);
    }
    
    fn check_alerts(&self, latency: Duration, throughput: f64, quality: f64) -> anyhow::Result<()> {
        // Latency alert
        if latency > self.alert_thresholds.max_latency {
            println!("🚨 LATENCY ALERT: {}ms (threshold: {}ms)", 
                latency.as_millis(), 
                self.alert_thresholds.max_latency.as_millis());
        }
        
        // Throughput alert
        if throughput < self.alert_thresholds.min_throughput {
            println!("🚨 THROUGHPUT ALERT: {:.1} samples/s (threshold: {:.1})", 
                throughput, 
                self.alert_thresholds.min_throughput);
        }
        
        // Quality alert
        if quality < self.alert_thresholds.min_quality {
            println!("🚨 QUALITY ALERT: {:.3} (threshold: {:.3})", 
                quality, 
                self.alert_thresholds.min_quality);
        }
        
        // Trend alerts
        self.check_trend_alerts()?;
        
        Ok(())
    }
    
    fn check_trend_alerts(&self) -> anyhow::Result<()> {
        if self.latency_history.len() < 50 {
            return Ok(());  // Not enough data
        }
        
        // Check for degrading latency trend
        let recent_latencies: Vec<_> = self.latency_history.iter().rev().take(20).collect();
        let older_latencies: Vec<_> = self.latency_history.iter().rev().skip(20).take(20).collect();
        
        let recent_avg: Duration = recent_latencies.iter().copied().sum::<Duration>() / recent_latencies.len() as u32;
        let older_avg: Duration = older_latencies.iter().copied().sum::<Duration>() / older_latencies.len() as u32;
        
        if recent_avg > older_avg * 2 {  // 100% increase
            println!("📈 TREND ALERT: Latency degrading (recent: {}ms, older: {}ms)", 
                recent_avg.as_millis(), 
                older_avg.as_millis());
        }
        
        Ok(())
    }
    
    fn log_performance_summary(&self) {
        if self.latency_history.is_empty() {
            return;
        }
        
        // Calculate statistics
        let latencies: Vec<_> = self.latency_history.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        let throughputs: Vec<_> = self.throughput_history.iter().copied().collect();
        let qualities: Vec<_> = self.quality_history.iter().copied().collect();
        
        let latency_stats = calculate_statistics(&latencies);
        let throughput_stats = calculate_statistics(&throughputs);
        let quality_stats = calculate_statistics(&qualities);
        
        println!("\n📊 PERFORMANCE SUMMARY (last {} operations):", self.latency_history.len());
        println!("  Latency    - avg: {:.1}ms, p95: {:.1}ms, p99: {:.1}ms", 
            latency_stats.mean, latency_stats.p95, latency_stats.p99);
        println!("  Throughput - avg: {:.1} samples/s, min: {:.1}, max: {:.1}", 
            throughput_stats.mean, throughput_stats.min, throughput_stats.max);
        println!("  Quality    - avg: {:.3}, min: {:.3}, max: {:.3}", 
            quality_stats.mean, quality_stats.min, quality_stats.max);
        println!();
    }
}

struct AlertThresholds {
    max_latency: Duration,
    min_throughput: f64,
    min_quality: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(200),
            min_throughput: 10000.0,  // samples/second
            min_quality: 0.7,
        }
    }
}

fn calculate_statistics(values: &[f64]) -> Statistics {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    
    Statistics { mean, min, max, p95, p99 }
}

struct Statistics {
    mean: f64,
    min: f64,
    max: f64,
    p95: f64,
    p99: f64,
}
```

## Interactive Voice Applications

### Real-time Voice Assistant

```rust
use voirs_sdk::{VoirsSdk, InteractiveSession, VoiceActivityDetector};
use tokio::sync::{mpsc, broadcast};

struct RealTimeVoiceAssistant {
    synthesizer: StreamingSynthesizer,
    session: InteractiveSession,
    vad: VoiceActivityDetector,
    response_sender: broadcast::Sender<AudioChunk>,
}

impl RealTimeVoiceAssistant {
    async fn new() -> anyhow::Result<(Self, broadcast::Receiver<AudioChunk>)> {
        let sdk = VoirsSdk::new().await?;
        
        let streaming_config = StreamingConfig::builder()
            .latency_target(Duration::from_millis(100))
            .enable_interrupt_handling(true)  // Allow interruptions
            .enable_barge_in(true)            // User can interrupt bot
            .build()?;
        
        let (response_sender, response_receiver) = broadcast::channel(1000);
        
        let assistant = Self {
            synthesizer: sdk.create_streaming_synthesizer(&streaming_config).await?,
            session: InteractiveSession::new()?,
            vad: VoiceActivityDetector::new()?,
            response_sender,
        };
        
        Ok((assistant, response_receiver))
    }
    
    async fn handle_user_input(&mut self, audio_input: &[f32]) -> anyhow::Result<()> {
        // Detect voice activity
        let voice_activity = self.vad.process_audio(audio_input)?;
        
        if voice_activity.speech_detected {
            println!("👤 User speaking...");
            
            // If we're currently speaking, handle interruption
            if self.session.is_bot_speaking() {
                self.handle_interruption().await?;
            }
            
            // Update session state
            self.session.set_user_speaking(true);
            
        } else if voice_activity.speech_ended {
            println!("👤 User finished speaking");
            
            // Process the complete user input
            self.process_complete_user_input().await?;
            self.session.set_user_speaking(false);
        }
        
        Ok(())
    }
    
    async fn handle_interruption(&mut self) -> anyhow::Result<()> {
        println!("🛑 User interruption detected - stopping current synthesis");
        
        // Stop current synthesis immediately
        self.synthesizer.stop_current_synthesis().await?;
        
        // Update session state
        self.session.handle_interruption();
        
        // Send silence to clear audio buffer
        let silence = AudioChunk::silence(1024);
        let _ = self.response_sender.send(silence);
        
        Ok(())
    }
    
    async fn process_complete_user_input(&mut self) -> anyhow::Result<()> {
        // In a real application, this would involve speech recognition
        // For this example, we'll simulate with predefined responses
        
        let response_text = self.generate_response().await?;
        
        println!("🤖 Bot responding: '{}'", response_text);
        
        // Start streaming synthesis
        self.session.set_bot_speaking(true);
        let mut stream = self.synthesizer.synthesize_streaming(&response_text).await?;
        
        // Stream audio chunks to output
        while let Some(chunk) = stream.next().await {
            // Check if we were interrupted
            if self.session.was_interrupted() {
                break;
            }
            
            // Send chunk to audio output
            if self.response_sender.send(chunk).is_err() {
                // No receivers, stop synthesis
                break;
            }
        }
        
        self.session.set_bot_speaking(false);
        println!("🤖 Bot finished speaking");
        
        Ok(())
    }
    
    async fn generate_response(&self) -> anyhow::Result<String> {
        // Simplified response generation
        let responses = vec![
            "I understand your question. Let me help you with that.",
            "That's an interesting point. Here's what I think about it.",
            "I'm here to assist you. What else would you like to know?",
            "Thank you for that input. Let me provide some information.",
        ];
        
        let index = self.session.turn_count() % responses.len();
        Ok(responses[index].to_string())
    }
    
    fn get_session_metrics(&self) -> SessionMetrics {
        SessionMetrics {
            total_turns: self.session.turn_count(),
            average_response_time: self.session.average_response_time(),
            interruption_count: self.session.interruption_count(),
            total_session_duration: self.session.duration(),
        }
    }
}

async fn voice_assistant_demo() -> anyhow::Result<()> {
    let (mut assistant, mut audio_receiver) = RealTimeVoiceAssistant::new().await?;
    
    println!("🎤 Voice assistant started. Simulating conversation...");
    
    // Spawn audio output handler
    let audio_output_task = tokio::spawn(async move {
        while let Ok(chunk) = audio_receiver.recv().await {
            // In a real application, this would go to speakers
            println!("🔊 Playing audio chunk: {} samples", chunk.len());
            tokio::time::sleep(Duration::from_millis(20)).await;  // Simulate playback time
        }
    });
    
    // Simulate conversation
    let conversation_events = vec![
        (1000, "user_start_speaking"),
        (3000, "user_stop_speaking"),
        (3100, "bot_start_response"),
        (5000, "user_interrupt"),
        (5100, "user_start_speaking"),
        (7000, "user_stop_speaking"),
        (7100, "bot_start_response"),
        (9000, "bot_finish_response"),
    ];
    
    let start_time = Instant::now();
    
    for (delay_ms, event) in conversation_events {
        // Wait until the scheduled time
        let target_time = Duration::from_millis(delay_ms);
        let elapsed = start_time.elapsed();
        
        if target_time > elapsed {
            tokio::time::sleep(target_time - elapsed).await;
        }
        
        println!("\n⏰ {}ms: {}", delay_ms, event);
        
        match event {
            "user_start_speaking" => {
                // Simulate user audio input
                let fake_audio = vec![0.1; 1024];  // Fake audio data
                assistant.handle_user_input(&fake_audio).await?;
            }
            "user_stop_speaking" => {
                let silence = vec![0.0; 1024];  // Silence indicates end of speech
                assistant.handle_user_input(&silence).await?;
            }
            "user_interrupt" => {
                let interrupt_audio = vec![0.2; 512];  // Louder audio indicates interruption
                assistant.handle_user_input(&interrupt_audio).await?;
            }
            _ => {}
        }
    }
    
    // Show session metrics
    let metrics = assistant.get_session_metrics();
    println!("\n📊 Session Summary:");
    println!("  Total turns: {}", metrics.total_turns);
    println!("  Average response time: {:?}", metrics.average_response_time);
    println!("  Interruptions: {}", metrics.interruption_count);
    println!("  Session duration: {:?}", metrics.total_session_duration);
    
    // Clean up
    audio_output_task.abort();
    
    Ok(())
}

struct SessionMetrics {
    total_turns: u32,
    average_response_time: Duration,
    interruption_count: u32,
    total_session_duration: Duration,
}
```

## Network-Aware Streaming

### Adaptive Quality Control

```rust
use voirs_sdk::{AdaptiveQualityController, NetworkConditions, QualityLevel};

struct NetworkAwareStreaming {
    synthesizer: StreamingSynthesizer,
    quality_controller: AdaptiveQualityController,
    network_monitor: NetworkMonitor,
    current_quality: QualityLevel,
    quality_history: VecDeque<QualityAdjustment>,
}

impl NetworkAwareStreaming {
    async fn new() -> anyhow::Result<Self> {
        let sdk = VoirsSdk::new().await?;
        
        let streaming_config = StreamingConfig::builder()
            .enable_adaptive_quality(true)
            .quality_adjustment_interval(Duration::from_secs(2))
            .build()?;
        
        Ok(Self {
            synthesizer: sdk.create_streaming_synthesizer(&streaming_config).await?,
            quality_controller: AdaptiveQualityController::new()?,
            network_monitor: NetworkMonitor::new()?,
            current_quality: QualityLevel::Medium,
            quality_history: VecDeque::new(),
        })
    }
    
    async fn stream_with_adaptation(&mut self, text: &str) -> anyhow::Result<()> {
        println!("🌐 Starting network-aware streaming for: '{}'", text);
        
        // Monitor network conditions
        let network_conditions = self.network_monitor.assess_conditions().await?;
        self.log_network_conditions(&network_conditions);
        
        // Determine initial quality level
        let initial_quality = self.quality_controller.recommend_quality(&network_conditions)?;
        self.adjust_quality(initial_quality).await?;
        
        // Start streaming
        let mut stream = self.synthesizer.synthesize_streaming(text).await?;
        let mut chunk_count = 0;
        let start_time = Instant::now();
        
        while let Some(chunk) = stream.next().await {
            chunk_count += 1;
            
            // Periodically check network conditions and adjust quality
            if chunk_count % 20 == 0 {  // Check every 20 chunks
                let current_conditions = self.network_monitor.assess_conditions().await?;
                
                if self.should_adjust_quality(&current_conditions) {
                    let new_quality = self.quality_controller.recommend_quality(&current_conditions)?;
                    
                    if new_quality != self.current_quality {
                        println!("🔧 Adjusting quality: {:?} -> {:?}", self.current_quality, new_quality);
                        self.adjust_quality(new_quality).await?;
                        
                        // Update the stream with new quality
                        stream = self.synthesizer.update_stream_quality(new_quality).await?;
                    }
                }
            }
            
            // In a real application, send chunk to audio output
            println!("📦 Streaming chunk {} (quality: {:?})", chunk_count, self.current_quality);
            
            // Simulate network delay
            let delay = self.calculate_network_delay(&network_conditions);
            tokio::time::sleep(delay).await;
        }
        
        let total_time = start_time.elapsed();
        println!("✅ Streaming completed in {:?} with {} quality adjustments", 
            total_time, self.quality_history.len());
        
        Ok(())
    }
    
    fn log_network_conditions(&self, conditions: &NetworkConditions) {
        println!("📊 Network conditions:");
        println!("  Bandwidth: {:.1} Mbps", conditions.bandwidth_mbps);
        println!("  Latency: {}ms", conditions.latency.as_millis());
        println!("  Jitter: {}ms", conditions.jitter.as_millis());
        println!("  Packet loss: {:.2}%", conditions.packet_loss_rate * 100.0);
        println!("  Stability: {:.2}/1.0", conditions.stability_score);
    }
    
    fn should_adjust_quality(&self, conditions: &NetworkConditions) -> bool {
        // Check if conditions have changed significantly
        let recent_bandwidth = self.network_monitor.get_recent_bandwidth_trend();
        let bandwidth_change = (conditions.bandwidth_mbps - recent_bandwidth).abs() / recent_bandwidth;
        
        bandwidth_change > 0.3 ||  // 30% bandwidth change
        conditions.packet_loss_rate > 0.05 ||  // >5% packet loss
        conditions.latency > Duration::from_millis(300)  // High latency
    }
    
    async fn adjust_quality(&mut self, new_quality: QualityLevel) -> anyhow::Result<()> {
        let adjustment = QualityAdjustment {
            from: self.current_quality,
            to: new_quality,
            timestamp: Instant::now(),
            reason: self.determine_adjustment_reason(new_quality),
        };
        
        self.current_quality = new_quality;
        self.quality_history.push_back(adjustment);
        
        // Update synthesizer quality
        self.synthesizer.set_quality_level(new_quality).await?;
        
        println!("🎛️  Quality adjusted to {:?} (reason: {})", 
            new_quality, adjustment.reason);
        
        Ok(())
    }
    
    fn determine_adjustment_reason(&self, new_quality: QualityLevel) -> String {
        match new_quality {
            QualityLevel::Low => "Poor network conditions".to_string(),
            QualityLevel::Medium => "Balanced quality/bandwidth".to_string(),
            QualityLevel::High => "Excellent network conditions".to_string(),
            QualityLevel::Ultra => "Premium network quality".to_string(),
        }
    }
    
    fn calculate_network_delay(&self, conditions: &NetworkConditions) -> Duration {
        // Simulate realistic network delays
        let base_delay = Duration::from_millis(10);
        let latency_factor = conditions.latency.as_millis() as f64 / 100.0;
        let jitter_factor = conditions.jitter.as_millis() as f64 / 50.0;
        
        let total_delay_ms = base_delay.as_millis() as f64 * (1.0 + latency_factor + jitter_factor);
        Duration::from_millis(total_delay_ms as u64)
    }
}

struct QualityAdjustment {
    from: QualityLevel,
    to: QualityLevel,
    timestamp: Instant,
    reason: String,
}

async fn network_aware_streaming_demo() -> anyhow::Result<()> {
    let mut streaming = NetworkAwareStreaming::new().await?;
    
    // Simulate different network conditions over time
    let test_scenarios = vec![
        "This is the first message with good network conditions.",
        "Now we're simulating moderate network conditions with some latency.",
        "This message tests poor network conditions with high packet loss.",
        "Finally, we return to excellent network conditions for optimal quality.",
    ];
    
    for (i, text) in test_scenarios.iter().enumerate() {
        println!("\n🎬 Scenario {}: Simulating network conditions...", i + 1);
        
        // Simulate changing network conditions
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        streaming.stream_with_adaptation(text).await?;
        
        // Pause between scenarios
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    
    Ok(())
}
```

## Best Practices for Real-time Processing

1. **Target Appropriate Latency**: Match latency targets to application requirements
2. **Monitor Performance**: Continuously track latency, throughput, and quality
3. **Handle Interruptions**: Support natural conversation flow with barge-in
4. **Adapt to Conditions**: Adjust quality based on network and system conditions
5. **Use Predictive Processing**: Pre-synthesize likely content to reduce latency
6. **Optimize Buffer Management**: Balance latency and stability
7. **Test Under Load**: Validate performance under realistic conditions

## Common Issues and Solutions

### Issue: "High latency spikes"
**Solution**: Implement adaptive buffering and monitor system resources:
```rust
.enable_adaptive_buffering(true)
.buffer_size_range(512, 4096)
.latency_spike_detection(true)
```

### Issue: "Audio dropouts during poor network"
**Solution**: Use conservative buffering and quality adaptation:
```rust
.fallback_strategy(FallbackStrategy::Conservative)
.enable_quality_adaptation(true)
.min_buffer_duration(Duration::from_millis(500))
```

### Issue: "Interruptions not handled smoothly"
**Solution**: Enable proper interruption handling:
```rust
.enable_barge_in(true)
.interruption_fade_time(Duration::from_millis(50))
.resume_after_interruption(true)
```

## Next Steps

In the next tutorial, we'll explore quality optimization techniques to achieve the best possible audio output while maintaining performance.

Continue to [Tutorial 7: Quality Optimization](./07-quality-optimization.md) →

## Additional Resources

- [Streaming Synthesis Examples](../streaming_synthesis.rs)
- [Real-time Applications](../realtime_voice_coach.rs)
- [Performance Optimization](../performance_optimization_techniques.rs)

---

**Estimated completion time**: 40-45 minutes  
**Difficulty**: ⭐⭐⭐☆☆  
**Next tutorial**: [Quality Optimization](./07-quality-optimization.md)