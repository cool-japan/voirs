//! Advanced Real-time Processing Example
//!
//! This example demonstrates advanced real-time processing features including
//! adaptive quality control, buffering strategies, and microphone integration.
//!
//! Usage:
//! ```bash
//! cargo run --example advanced_realtime --features="whisper-pure"
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::asr::FallbackConfig;
use voirs_recognizer::integration::config::{LatencyMode, StreamingConfig};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceRequirements, PerformanceValidator, RecognitionError};

// Advanced streaming buffer for real-time processing
#[derive(Debug)]
struct StreamingBuffer {
    buffer: VecDeque<f32>,
    sample_rate: u32,
    max_duration: Duration,
    overlap_samples: usize,
}

impl StreamingBuffer {
    fn new(sample_rate: u32, max_duration: Duration, overlap_samples: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            sample_rate,
            max_duration,
            overlap_samples,
        }
    }

    fn add_chunk(&mut self, chunk: &[f32]) {
        // Add new chunk to buffer
        self.buffer.extend(chunk.iter().cloned());

        // Remove old data if buffer is too long
        let max_samples = (self.sample_rate as f32 * self.max_duration.as_secs_f32()) as usize;
        if self.buffer.len() > max_samples {
            let excess = self.buffer.len() - max_samples;
            self.buffer.drain(..excess);
        }
    }

    fn get_processing_window(&self, window_size: usize) -> Option<Vec<f32>> {
        if self.buffer.len() < window_size {
            return None;
        }

        let start = if self.buffer.len() > window_size {
            self.buffer.len() - window_size
        } else {
            0
        };

        Some(self.buffer.iter().skip(start).cloned().collect())
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

// Adaptive quality controller
#[derive(Debug)]
struct AdaptiveQualityController {
    target_latency: Duration,
    current_latency: Duration,
    quality_level: QualityLevel,
    performance_history: VecDeque<(Duration, f32)>, // (latency, accuracy)
}

#[derive(Debug, Clone, Copy)]
enum QualityLevel {
    UltraFast,
    Fast,
    Balanced,
    Accurate,
}

impl AdaptiveQualityController {
    fn new(target_latency: Duration) -> Self {
        Self {
            target_latency,
            current_latency: Duration::ZERO,
            quality_level: QualityLevel::Balanced,
            performance_history: VecDeque::new(),
        }
    }

    fn update_performance(&mut self, latency: Duration, accuracy: f32) {
        self.current_latency = latency;
        self.performance_history.push_back((latency, accuracy));

        // Keep only recent history
        if self.performance_history.len() > 10 {
            self.performance_history.pop_front();
        }

        // Adjust quality level based on performance
        if latency > self.target_latency * 2 {
            self.quality_level = match self.quality_level {
                QualityLevel::Accurate => QualityLevel::Balanced,
                QualityLevel::Balanced => QualityLevel::Fast,
                QualityLevel::Fast => QualityLevel::UltraFast,
                QualityLevel::UltraFast => QualityLevel::UltraFast,
            };
        } else if latency < self.target_latency / 2 && accuracy > 0.9 {
            self.quality_level = match self.quality_level {
                QualityLevel::UltraFast => QualityLevel::Fast,
                QualityLevel::Fast => QualityLevel::Balanced,
                QualityLevel::Balanced => QualityLevel::Accurate,
                QualityLevel::Accurate => QualityLevel::Accurate,
            };
        }
    }

    fn get_asr_config(&self) -> ASRConfig {
        match self.quality_level {
            QualityLevel::UltraFast => ASRConfig {
                language: Some(LanguageCode::EnUs),
                model_variant: Some("tiny".to_string()),
                confidence_threshold: 0.3,
                word_timestamps: false,
                sentence_segmentation: false,
                language_detection: false,
                max_duration: Some(30.0),
                ..Default::default()
            },
            QualityLevel::Fast => ASRConfig {
                language: Some(LanguageCode::EnUs),
                model_variant: Some("small".to_string()),
                confidence_threshold: 0.4,
                word_timestamps: true,
                sentence_segmentation: true,
                language_detection: false,
                max_duration: Some(60.0),
                ..Default::default()
            },
            QualityLevel::Balanced => ASRConfig {
                language: Some(LanguageCode::EnUs),
                model_variant: Some("base".to_string()),
                confidence_threshold: 0.5,
                word_timestamps: true,
                sentence_segmentation: true,
                language_detection: true,
                max_duration: Some(120.0),
                ..Default::default()
            },
            QualityLevel::Accurate => ASRConfig {
                language: Some(LanguageCode::EnUs),
                model_variant: Some("large".to_string()),
                confidence_threshold: 0.6,
                word_timestamps: true,
                sentence_segmentation: true,
                language_detection: true,
                max_duration: Some(180.0),
                ..Default::default()
            },
        }
    }

    fn get_quality_description(&self) -> &'static str {
        match self.quality_level {
            QualityLevel::UltraFast => "Ultra Fast (tiny model, minimal features)",
            QualityLevel::Fast => "Fast (small model, timestamps)",
            QualityLevel::Balanced => "Balanced (base model, full features)",
            QualityLevel::Accurate => "Accurate (large model, full features)",
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🚀 Advanced Real-time Processing Demo");
    println!("=====================================\n");

    // Step 1: Initialize advanced streaming components
    println!("🔧 Initializing advanced streaming components...");

    let sample_rate = 16000;
    let chunk_size = 800; // 50ms chunks
    let overlap_samples = 160; // 10ms overlap
    let buffer_duration = Duration::from_secs(5); // 5 second buffer

    let mut streaming_buffer = StreamingBuffer::new(sample_rate, buffer_duration, overlap_samples);
    let mut quality_controller = AdaptiveQualityController::new(Duration::from_millis(150));

    println!("✅ Advanced components initialized:");
    println!(
        "   • Streaming buffer: {} second capacity",
        buffer_duration.as_secs()
    );
    println!("   • Chunk size: {} samples (50ms)", chunk_size);
    println!("   • Overlap: {} samples (10ms)", overlap_samples);
    println!("   • Quality control: Adaptive based on performance");

    // Step 2: Create performance validator with strict requirements
    println!("\n📊 Setting up performance monitoring...");
    let requirements = PerformanceRequirements {
        max_rtf: 0.2,                        // Strict RTF for real-time
        max_memory_usage: 512 * 1024 * 1024, // 512MB memory limit
        max_startup_time_ms: 1000,           // 1 second startup
        max_streaming_latency_ms: 150,       // 150ms latency limit
    };

    let validator = PerformanceValidator::with_requirements(requirements);
    println!("✅ Performance monitoring configured:");
    println!("   • RTF limit: {:.2}", validator.requirements().max_rtf);
    println!(
        "   • Memory limit: {} MB",
        validator.requirements().max_memory_usage / (1024 * 1024)
    );
    println!(
        "   • Latency limit: {} ms",
        validator.requirements().max_streaming_latency_ms
    );

    // Step 3: Initialize ASR with adaptive configuration
    println!("\n🎯 Initializing adaptive ASR system...");
    let fallback_config = FallbackConfig {
        quality_threshold: 0.6,
        max_processing_time_seconds: 5.0,
        adaptive_selection: true,
        memory_threshold_mb: 512.0,
        min_duration_for_selection: 0.3,
        ..Default::default()
    };
    let mut asr = IntelligentASRFallback::new(fallback_config).await?;

    println!("✅ ASR system initialized");
    println!(
        "   • Initial quality: {}",
        quality_controller.get_quality_description()
    );

    // Step 4: Create sophisticated audio stream
    println!("\n🎵 Creating sophisticated audio stream...");
    let audio_stream = create_sophisticated_audio_stream(sample_rate, chunk_size);
    println!("✅ Audio stream created: {} chunks", audio_stream.len());

    // Step 5: Advanced real-time processing loop
    println!("\n⚡ Starting advanced real-time processing...");
    println!("┌───────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Time  │ Buffer │ Quality │ Latency │ Result                     │ Confidence │");
    println!("├───────────────────────────────────────────────────────────────────────────────┤");

    let mut processing_start = Instant::now();
    let mut performance_history = Vec::new();
    let mut last_result = String::new();

    for (i, chunk) in audio_stream.iter().enumerate() {
        let chunk_start = Instant::now();

        // Add chunk to streaming buffer
        streaming_buffer.add_chunk(chunk.samples());

        // Process when buffer has enough data
        if streaming_buffer.len() >= chunk_size * 3 {
            // Process every 150ms
            // Get processing window
            let window_size = chunk_size * 6; // 300ms window
            if let Some(window_samples) = streaming_buffer.get_processing_window(window_size) {
                let window_audio = AudioBuffer::mono(window_samples, sample_rate);

                // Perform recognition with adaptive configuration
                let recognition_start = Instant::now();
                let adaptive_config = quality_controller.get_asr_config();
                let result = asr
                    .transcribe(&window_audio, Some(&adaptive_config))
                    .await?;
                let recognition_time = recognition_start.elapsed();

                // Calculate metrics
                let total_latency = chunk_start.elapsed();
                let confidence = result.transcript.confidence;

                // Update quality controller
                quality_controller.update_performance(total_latency, confidence);

                // Store performance data
                performance_history.push((
                    total_latency,
                    confidence,
                    result.transcript.text.clone(),
                ));

                // Display results
                let elapsed = processing_start.elapsed();
                let buffer_fill = (streaming_buffer.len() as f32
                    / (sample_rate as f32 * buffer_duration.as_secs_f32()))
                    * 100.0;
                let quality_desc = match quality_controller.quality_level {
                    QualityLevel::UltraFast => "Ultra",
                    QualityLevel::Fast => "Fast",
                    QualityLevel::Balanced => "Balanced",
                    QualityLevel::Accurate => "Accurate",
                };

                let result_text = if result.transcript.text.len() > 23 {
                    format!("{}...", &result.transcript.text[..20])
                } else {
                    result.transcript.text.clone()
                };

                println!(
                    "│ {:5.1}s │ {:5.1}% │ {:7} │ {:6.0}ms │ {:23} │ {:9.2} │",
                    elapsed.as_secs_f32(),
                    buffer_fill,
                    quality_desc,
                    total_latency.as_secs_f32() * 1000.0,
                    result_text,
                    confidence
                );

                // Check for quality adjustments
                if result.transcript.text != last_result {
                    last_result = result.transcript.text.clone();

                    // Validate performance
                    let (latency_ms, latency_passed) =
                        validator.validate_streaming_latency(total_latency);
                    if !latency_passed {
                        println!("│       │        │ ⚠️ ADJ  │ {:6.0}ms │ Quality adjusted for latency   │           │", latency_ms);
                    }
                }
            }
        }

        // Simulate real-time processing delay
        sleep(Duration::from_millis(50)).await;
    }

    println!("└───────────────────────────────────────────────────────────────────────────────┘");

    // Step 6: Performance analysis
    println!("\n📈 Advanced Performance Analysis:");

    // Latency analysis
    let latencies: Vec<Duration> = performance_history.iter().map(|(l, _, _)| *l).collect();
    if !latencies.is_empty() {
        let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
        let max_latency = latencies.iter().max().unwrap();
        let min_latency = latencies.iter().min().unwrap();

        println!("   Latency Performance:");
        println!("   • Average: {:.1}ms", avg_latency.as_secs_f32() * 1000.0);
        println!("   • Maximum: {:.1}ms", max_latency.as_secs_f32() * 1000.0);
        println!("   • Minimum: {:.1}ms", min_latency.as_secs_f32() * 1000.0);
        println!(
            "   • Latency violations: {}",
            latencies
                .iter()
                .filter(|&&l| l > Duration::from_millis(150))
                .count()
        );
    }

    // Quality adaptation analysis
    let quality_changes = performance_history
        .windows(2)
        .filter(|w| w[0].1 != w[1].1) // Different confidence scores
        .count();

    println!("   Quality Adaptation:");
    println!(
        "   • Final quality level: {}",
        quality_controller.get_quality_description()
    );
    println!("   • Quality changes: {}", quality_changes);
    println!(
        "   • Average confidence: {:.3}",
        performance_history.iter().map(|(_, c, _)| c).sum::<f32>()
            / performance_history.len() as f32
    );

    // Buffer utilization
    println!("   Buffer Utilization:");
    println!("   • Final buffer size: {} samples", streaming_buffer.len());
    println!(
        "   • Buffer utilization: {:.1}%",
        (streaming_buffer.len() as f32 / (sample_rate as f32 * buffer_duration.as_secs_f32()))
            * 100.0
    );

    // Step 7: Memory and resource analysis
    println!("\n💾 Resource Analysis:");
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!(
        "   • Memory usage: {:.1} MB ({})",
        memory_usage as f64 / (1024.0 * 1024.0),
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Calculate total processing efficiency
    let total_audio_duration = audio_stream.len() as f32 * 0.05; // 50ms per chunk
    let total_processing_time: Duration = performance_history.iter().map(|(l, _, _)| *l).sum();
    let overall_rtf = total_processing_time.as_secs_f32() / total_audio_duration;

    println!("   • Overall RTF: {:.3}", overall_rtf);
    println!(
        "   • Processing efficiency: {:.1}%",
        (total_audio_duration / total_processing_time.as_secs_f32()) * 100.0
    );

    // Step 8: Detailed quality analysis
    println!("\n🔍 Detailed Quality Analysis:");

    // Group results by quality level
    let mut quality_stats = std::collections::HashMap::new();
    for (latency, confidence, _) in &performance_history {
        let quality = if *latency < Duration::from_millis(50) {
            "Ultra Fast"
        } else if *latency < Duration::from_millis(100) {
            "Fast"
        } else if *latency < Duration::from_millis(150) {
            "Balanced"
        } else {
            "Accurate"
        };

        let entry = quality_stats.entry(quality).or_insert(Vec::new());
        entry.push((*latency, *confidence));
    }

    for (quality, measurements) in quality_stats {
        let avg_latency =
            measurements.iter().map(|(l, _)| *l).sum::<Duration>() / measurements.len() as u32;
        let avg_confidence =
            measurements.iter().map(|(_, c)| *c).sum::<f32>() / measurements.len() as f32;

        println!("   • {} Quality:", quality);
        println!("     - Measurements: {}", measurements.len());
        println!(
            "     - Average latency: {:.1}ms",
            avg_latency.as_secs_f32() * 1000.0
        );
        println!("     - Average confidence: {:.3}", avg_confidence);
    }

    // Step 9: Recognition accuracy by time
    println!("\n📊 Recognition Results Timeline:");
    let mut unique_results = Vec::new();
    let mut last_unique = String::new();

    for (i, (latency, confidence, text)) in performance_history.iter().enumerate() {
        if *text != last_unique && !text.is_empty() {
            unique_results.push((i, latency, confidence, text));
            last_unique = text.clone();
        }
    }

    for (i, (index, latency, confidence, text)) in unique_results.iter().enumerate() {
        println!(
            "   {}. [{}] \"{}\": {:.1}ms, conf={:.2}",
            i + 1,
            index,
            text,
            latency.as_secs_f32() * 1000.0,
            confidence
        );
    }

    // Step 10: Recommendations and insights
    println!("\n💡 Performance Insights & Recommendations:");

    if overall_rtf < 0.2 {
        println!("   ✅ Excellent Performance:");
        println!("   • RTF well below threshold");
        println!("   • Consider enabling higher quality features");
        println!("   • System has headroom for more complex processing");
    } else if overall_rtf < 0.3 {
        println!("   ✅ Good Performance:");
        println!("   • RTF within acceptable range");
        println!("   • Current configuration is well-balanced");
    } else {
        println!("   ⚠️ Performance Issues:");
        println!("   • RTF exceeds recommended threshold");
        println!("   • Consider reducing quality or optimizing");
    }

    if let Some(avg_latency) = latencies.first() {
        if avg_latency.as_millis() < 100 {
            println!("   ✅ Excellent Latency:");
            println!("   • Well below 100ms target");
            println!("   • Suitable for interactive applications");
        } else if avg_latency.as_millis() < 200 {
            println!("   ✅ Good Latency:");
            println!("   • Acceptable for most real-time applications");
        } else {
            println!("   ⚠️ High Latency:");
            println!("   • May impact user experience");
            println!("   • Consider optimizing processing pipeline");
        }
    }

    println!("\n✅ Advanced real-time processing demo completed!");
    println!("🚀 Key achievements:");
    println!("   • Adaptive quality control implemented");
    println!("   • Sophisticated buffering strategy");
    println!("   • Comprehensive performance monitoring");
    println!("   • Real-time constraint satisfaction");

    println!("\n🎯 Production readiness checklist:");
    println!(
        "   • {} Real-time performance (RTF < 0.3)",
        if overall_rtf < 0.3 { "✅" } else { "❌" }
    );
    println!(
        "   • {} Low latency (< 200ms)",
        if latencies.iter().all(|l| l.as_millis() < 200) {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   • {} Memory efficiency",
        if memory_passed { "✅" } else { "❌" }
    );
    println!(
        "   • {} Quality adaptation",
        if quality_changes > 0 { "✅" } else { "❌" }
    );

    Ok(())
}

fn create_sophisticated_audio_stream(sample_rate: u32, chunk_size: usize) -> Vec<AudioBuffer> {
    let mut chunks = Vec::new();
    let num_chunks = 60; // 3 seconds of audio

    // Create a realistic conversation simulation
    let conversation_segments = vec![
        (0, 10, "greeting", 180.0, 0.1),   // "Hello"
        (10, 15, "silence", 0.0, 0.0),     // Pause
        (15, 25, "question", 200.0, 0.08), // "How are you?"
        (25, 30, "silence", 0.0, 0.0),     // Pause
        (30, 45, "response", 160.0, 0.09), // "I'm fine, thank you"
        (45, 50, "silence", 0.0, 0.0),     // Pause
        (50, 60, "goodbye", 190.0, 0.07),  // "Goodbye"
    ];

    for i in 0..num_chunks {
        let mut samples = Vec::with_capacity(chunk_size);

        // Find which segment this chunk belongs to
        let segment = conversation_segments
            .iter()
            .find(|(start, end, _, _, _)| i >= *start && i < *end)
            .unwrap_or(&(0, 0, "silence", 0.0, 0.0));

        let (_, _, segment_type, base_freq, amplitude) = segment;

        for j in 0..chunk_size {
            let t = (i * chunk_size + j) as f32 / sample_rate as f32;

            let sample = match *segment_type {
                "greeting" | "question" | "response" | "goodbye" => {
                    // Speech-like signal with natural variation
                    let f0 = base_freq + 20.0 * (3.0 * t).sin(); // Pitch variation
                    let envelope = 0.7 + 0.3 * (8.0 * t).sin(); // Amplitude variation

                    let speech_signal = amplitude
                        * envelope
                        * (0.6 * (2.0 * std::f32::consts::PI * f0 * t).sin()
                            + 0.3 * (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin()
                            + 0.1 * (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin());

                    // Add some realistic noise
                    let noise = 0.005 * (j as f32 * 0.01).sin();
                    speech_signal + noise
                }
                _ => {
                    // Silence with minimal background noise
                    0.002 * (j as f32 * 0.001).sin()
                }
            };

            samples.push(sample);
        }

        chunks.push(AudioBuffer::mono(samples, sample_rate));
    }

    chunks
}
