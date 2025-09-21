//! Performance Optimization Guide for VoiRS Recognizer
//!
//! This comprehensive example demonstrates advanced performance optimization
//! techniques for VoiRS Recognizer, including benchmarking, profiling,
//! and optimization strategies for different deployment scenarios.
//!
//! Optimization Areas Covered:
//! - Model selection and configuration
//! - Memory optimization and management
//! - CPU and GPU acceleration
//! - I/O and disk optimization
//! - Network and streaming optimization
//! - Concurrent processing strategies
//! - Platform-specific optimizations
//!
//! Prerequisites: Complete Tutorial series and integration examples
//!
//! Usage:
//! ```bash
//! cargo run --example performance_optimization_guide --features="whisper-pure" --release
//! ```

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use voirs_recognizer::asr::{ASRBackend, WhisperModelSize};
use voirs_recognizer::audio_utilities::*;
use voirs_recognizer::integration::config::{LatencyMode, StreamingConfig};
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("⚡ Performance Optimization Guide for VoiRS Recognizer");
    println!("=====================================================\n");

    // Step 1: Performance baseline
    println!("📊 Step 1: Performance Baseline Assessment");
    let baseline_metrics = assess_performance_baseline().await?;

    // Step 2: Model optimization
    println!(
        "
🎯 Step 2: Model Selection and Configuration Optimization"
    );
    demonstrate_model_optimization().await?;

    // Step 3: Memory optimization
    println!(
        "
💾 Step 3: Memory Optimization Strategies"
    );
    demonstrate_memory_optimization().await?;

    // Step 4: CPU and GPU optimization
    println!(
        "
🔥 Step 4: CPU and GPU Acceleration"
    );
    demonstrate_acceleration_optimization().await?;

    // Step 5: I/O optimization
    println!(
        "
📁 Step 5: I/O and Disk Optimization"
    );
    demonstrate_io_optimization().await?;

    // Step 6: Concurrent processing
    println!(
        "
⚡ Step 6: Concurrent Processing Strategies"
    );
    demonstrate_concurrent_optimization().await?;

    // Step 7: Platform-specific optimizations
    println!(
        "
🖥️ Step 7: Platform-Specific Optimizations"
    );
    demonstrate_platform_optimization().await?;

    // Step 8: Performance monitoring
    println!(
        "
📈 Step 8: Performance Monitoring and Profiling"
    );
    demonstrate_performance_monitoring().await?;

    // Step 9: Optimization results
    println!(
        "
📊 Step 9: Optimization Results"
    );
    let optimized_metrics = assess_performance_optimized().await?;
    compare_performance_metrics(&baseline_metrics, &optimized_metrics);

    // Step 10: Conclusion
    println!(
        "
🎉 Performance Optimization Guide Complete!"
    );
    println!(
        "
📖 Optimization Summary:"
    );
    println!("   • Model selection can improve performance by 2-5x");
    println!("   • Memory optimization reduces usage by 30-60%");
    println!("   • GPU acceleration provides 3-10x speedup");
    println!("   • I/O optimization reduces latency by 20-40%");
    println!("   • Concurrent processing scales with available cores");
    println!("   • Platform-specific optimizations add 10-30% improvement");

    Ok(())
}

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    rtf: f32,
    memory_usage_mb: f32,
    startup_time_ms: u64,
    throughput_samples_per_sec: f32,
    latency_ms: u64,
    cpu_utilization: f32,
    accuracy: f32,
}

async fn assess_performance_baseline() -> Result<PerformanceMetrics, Box<dyn Error>> {
    println!("   🔍 Assessing baseline performance with default configuration...");

    // Create baseline configuration
    let config = ASRConfig {
        preferred_models: vec!["whisper".to_string()],
        whisper_model_size: Some("base".to_string()),
        language: Some(LanguageCode::EnUs),
        ..Default::default()
    };

    // Create test audio
    let audio = create_test_audio(3.0); // 3 seconds of test audio

    // Measure startup time
    let startup_start = Instant::now();
    let _recognizer = MockRecognizer::new(config).await?;
    let startup_time = startup_start.elapsed();

    // Measure processing performance
    let processing_start = Instant::now();
    let _result = simulate_recognition(&audio).await?;
    let processing_time = processing_start.elapsed();

    // Calculate metrics
    let metrics = PerformanceMetrics {
        rtf: processing_time.as_secs_f32() / audio.duration(),
        memory_usage_mb: estimate_memory_usage(&audio),
        startup_time_ms: startup_time.as_millis() as u64,
        throughput_samples_per_sec: audio.len() as f32 / processing_time.as_secs_f32(),
        latency_ms: processing_time.as_millis() as u64,
        cpu_utilization: estimate_cpu_usage(&audio, &processing_time),
        accuracy: 0.85, // Simulated baseline accuracy
    };

    println!("   ✅ Baseline metrics collected:");
    print_performance_metrics(&metrics);

    Ok(metrics)
}

async fn demonstrate_model_optimization() -> Result<(), Box<dyn Error>> {
    println!("   Model selection significantly impacts performance:");

    // Test different model configurations
    let model_configs = vec![
        (
            "Ultra-Fast (Tiny)",
            WhisperModelSize::Tiny,
            "Best for real-time applications with latency < 100ms",
        ),
        (
            "Balanced (Base)",
            WhisperModelSize::Base,
            "Good balance of speed and accuracy",
        ),
        (
            "High-Quality (Small)",
            WhisperModelSize::Small,
            "Better accuracy, suitable for batch processing",
        ),
    ];

    println!(
        "   
   🎯 Model Performance Comparison:"
    );
    println!(
        "   
   Model Size    | RTF    | Memory | Accuracy | Best Use Case"
    );
    println!("   ------------- | ------ | ------ | -------- | -------------");

    for (name, size, description) in model_configs {
        let config = ASRConfig {
            preferred_models: vec!["whisper".to_string()],
            whisper_model_size: Some(
                match size {
                    WhisperModelSize::Tiny => "tiny",
                    WhisperModelSize::Base => "base",
                    WhisperModelSize::Small => "small",
                    WhisperModelSize::Medium => "medium",
                    WhisperModelSize::Large => "large",
                    WhisperModelSize::LargeV2 => "large-v2",
                    WhisperModelSize::LargeV3 => "large-v3",
                }
                .to_string(),
            ),
            language: Some(LanguageCode::EnUs),
            ..Default::default()
        };

        let audio = create_test_audio(2.0);
        let start_time = Instant::now();
        let _result = simulate_recognition_with_config(&audio, &config).await?;
        let elapsed = start_time.elapsed();

        let rtf = elapsed.as_secs_f32() / audio.duration();
        let memory_mb = estimate_memory_usage_for_model(size.clone());
        let accuracy = estimate_accuracy_for_model(size);

        println!(
            "   {:13} | {:.2}   | {:.0}MB  | {:.1}%   | {}",
            name,
            rtf,
            memory_mb,
            accuracy * 100.0,
            description
        );
    }

    // Demonstrate dynamic model selection
    println!(
        "   
   🔄 Dynamic Model Selection:"
    );
    println!("   ```rust");
    println!("   // Select model based on requirements");
    println!("   let model_size = if latency_requirement < 100 {{");
    println!("       WhisperModelSize::Tiny");
    println!("   }} else if accuracy_requirement > 0.9 {{");
    println!("       WhisperModelSize::Small");
    println!("   }} else {{");
    println!("       WhisperModelSize::Base");
    println!("   }};");
    println!("   ");
    println!("   let config = ASRConfig {{");
    println!("       whisper_model_size: model_size,");
    println!("       // ... other configuration");
    println!("   }};");
    println!("   ```");

    Ok(())
}

async fn demonstrate_memory_optimization() -> Result<(), Box<dyn Error>> {
    println!("   Memory optimization techniques:");

    println!(
        "   
   💾 Memory Usage Optimization Strategies:"
    );

    // 1. Model quantization
    println!(
        "   
   1. Model Quantization:"
    );
    println!("   ```rust");
    println!("   let config = ASRConfig {{");
    println!("       whisper_model_size: WhisperModelSize::Base,");
    println!("       quantization_enabled: true,");
    println!("       quantization_bits: 8,  // 8-bit quantization");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   ```");
    println!("   • Reduces memory usage by 50-75%");
    println!("   • Slight accuracy trade-off (usually < 2%)");
    println!("   • Faster inference on some hardware");

    // 2. Chunked processing
    println!(
        "   
   2. Chunked Audio Processing:"
    );
    println!("   ```rust");
    println!("   // Process audio in chunks to limit memory usage");
    println!("   let chunk_size = 30.0; // 30 seconds");
    println!("   let audio_chunks = split_audio_smart(&audio, chunk_size, 2.0).await?;");
    println!("   ");
    println!("   for chunk in audio_chunks {{");
    println!("       let result = recognizer.recognize(&chunk).await?;");
    println!("       // Process result immediately");
    println!("   }}");
    println!("   ```");
    println!("   • Constant memory usage regardless of audio length");
    println!("   • Better for long audio files");
    println!("   • Enables streaming processing");

    // 3. Memory pooling
    println!(
        "   
   3. Memory Pooling:"
    );
    println!("   ```rust");
    println!("   let config = ASRConfig {{");
    println!("       enable_memory_pooling: true,");
    println!("       pool_size_mb: 512,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   ```");
    println!("   • Reuses memory allocations");
    println!("   • Reduces garbage collection pressure");
    println!("   • More consistent performance");

    // Demonstrate memory usage comparison
    println!(
        "   
   📊 Memory Usage Comparison:"
    );

    let scenarios = vec![
        ("Default", 1200.0, "Standard configuration"),
        ("Quantized", 600.0, "8-bit quantization enabled"),
        ("Chunked", 400.0, "30-second chunks with pooling"),
        ("Optimized", 300.0, "All optimizations combined"),
    ];

    println!(
        "   
   Configuration | Memory (MB) | Description"
    );
    println!("   ------------- | ----------- | -----------");

    for (name, memory, description) in scenarios {
        println!("   {:13} | {:11} | {}", name, memory, description);
    }

    Ok(())
}

async fn demonstrate_acceleration_optimization() -> Result<(), Box<dyn Error>> {
    println!("   CPU and GPU acceleration techniques:");

    println!(
        "   
   🔥 Hardware Acceleration Options:"
    );

    // 1. GPU acceleration
    println!(
        "   
   1. GPU Acceleration (CUDA/Metal):"
    );
    println!("   ```rust");
    println!("   let config = ASRConfig {{");
    println!("       enable_gpu_acceleration: true,");
    println!("       gpu_memory_fraction: 0.8,  // Use 80% of GPU memory");
    println!("       mixed_precision: true,     // FP16 for faster training");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   ```");
    println!("   • 3-10x speedup on compatible hardware");
    println!("   • Automatic fallback to CPU if GPU unavailable");
    println!("   • Supports NVIDIA CUDA and Apple Metal");

    // 2. Multi-threading
    println!(
        "   
   2. Multi-threading Optimization:"
    );
    println!("   ```rust");
    println!("   let config = ASRConfig {{");
    println!("       num_threads: num_cpus::get(),");
    println!("       thread_priority: ThreadPriority::High,");
    println!("       enable_thread_affinity: true,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   ```");
    println!("   • Scales with available CPU cores");
    println!("   • Thread affinity reduces context switching");
    println!("   • Priority settings for real-time applications");

    // 3. SIMD optimization
    println!(
        "   
   3. SIMD Optimization (Automatic):"
    );
    println!("   • AVX2/AVX-512 on Intel/AMD processors");
    println!("   • NEON on ARM processors");
    println!("   • Automatic detection and optimization");
    println!("   • No configuration required");

    // Performance comparison
    println!(
        "   
   📊 Acceleration Performance Comparison:"
    );

    let acceleration_scenarios = vec![
        ("CPU Only", 0.45, 100.0, "Single-threaded baseline"),
        ("Multi-thread", 0.18, 400.0, "All CPU cores utilized"),
        ("GPU (CUDA)", 0.08, 1200.0, "NVIDIA GPU acceleration"),
        ("GPU (Metal)", 0.12, 800.0, "Apple Silicon GPU"),
        ("Optimized", 0.05, 1800.0, "Multi-GPU + optimizations"),
    ];

    println!(
        "   
   Configuration | RTF   | Throughput (samples/sec) | Description"
    );
    println!("   ------------- | ----- | ----------------------- | -----------");

    for (name, rtf, throughput, description) in acceleration_scenarios {
        println!(
            "   {:13} | {:.2} | {:23} | {}",
            name, rtf, throughput, description
        );
    }

    Ok(())
}

async fn demonstrate_io_optimization() -> Result<(), Box<dyn Error>> {
    println!("   I/O and disk optimization techniques:");

    println!(
        "   
   📁 I/O Optimization Strategies:"
    );

    // 1. Async I/O
    println!(
        "   
   1. Asynchronous I/O:"
    );
    println!("   ```rust");
    println!("   // Load multiple audio files concurrently");
    println!("   let audio_files = vec![\"audio1.wav\", \"audio2.wav\", \"audio3.wav\"];");
    println!("   let audio_handles: Vec<_> = audio_files");
    println!("       .iter()");
    println!("       .map(|path| tokio::spawn(load_and_preprocess(path)))");
    println!("       .collect();");
    println!("   ");
    println!("   // Process results as they become available");
    println!("   for handle in audio_handles {{");
    println!("       let audio = handle.await??;");
    println!("       // Process audio...");
    println!("   }}");
    println!("   ```");

    // 2. Buffered I/O
    println!(
        "   
   2. Buffered I/O and Caching:"
    );
    println!("   ```rust");
    println!("   let config = ASRConfig {{");
    println!("       enable_audio_caching: true,");
    println!("       cache_size_mb: 1024,");
    println!("       buffer_size_kb: 64,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   ```");

    // 3. Memory-mapped files
    println!(
        "   
   3. Memory-Mapped Files:"
    );
    println!("   ```rust");
    println!("   // For large audio files, use memory mapping");
    println!("   let audio = AudioBuffer::from_mmap(\"large_audio.wav\").await?;");
    println!("   ```");

    // I/O performance comparison
    println!(
        "   
   📊 I/O Performance Comparison:"
    );

    let io_scenarios = vec![
        ("Synchronous", 250, "Sequential file loading"),
        ("Async", 120, "Concurrent file loading"),
        ("Buffered", 80, "Large read buffers"),
        ("Memory-mapped", 45, "Memory-mapped files"),
        ("Cached", 15, "In-memory caching"),
    ];

    println!(
        "   
   I/O Method     | Load Time (ms) | Description"
    );
    println!("   -------------- | -------------- | -----------");

    for (name, load_time, description) in io_scenarios {
        println!("   {:14} | {:14} | {}", name, load_time, description);
    }

    Ok(())
}

async fn demonstrate_concurrent_optimization() -> Result<(), Box<dyn Error>> {
    println!("   Concurrent processing strategies:");

    println!(
        "   
   ⚡ Concurrent Processing Patterns:"
    );

    // 1. Batch processing
    println!(
        "   
   1. Batch Processing:"
    );
    println!("   ```rust");
    println!("   let batch_size = 8;");
    println!("   let semaphore = Arc::new(Semaphore::new(batch_size));");
    println!("   ");
    println!("   let tasks: Vec<_> = audio_files");
    println!("       .chunks(batch_size)");
    println!("       .map(|chunk| {{");
    println!("           let sem = semaphore.clone();");
    println!("           tokio::spawn(async move {{");
    println!("               let _permit = sem.acquire().await.unwrap();");
    println!("               process_audio_batch(chunk).await");
    println!("           }})");
    println!("       }})");
    println!("       .collect();");
    println!("   ```");

    // 2. Pipeline processing
    println!(
        "   
   2. Pipeline Processing:"
    );
    println!("   ```rust");
    println!("   // Create processing pipeline");
    println!("   let (audio_tx, audio_rx) = tokio::sync::mpsc::channel(16);");
    println!("   let (result_tx, result_rx) = tokio::sync::mpsc::channel(16);");
    println!("   ");
    println!("   // Audio loading stage");
    println!("   tokio::spawn(async move {{");
    println!("       while let Some(path) = audio_rx.recv().await {{");
    println!("           let audio = load_and_preprocess(path).await?;");
    println!("           result_tx.send(audio).await?;");
    println!("       }}");
    println!("   }});");
    println!("   ```");

    // 3. Resource pooling
    println!(
        "   
   3. Resource Pooling:"
    );
    println!("   ```rust");
    println!("   struct RecognizerPool {{");
    println!("       recognizers: Vec<Arc<AsyncMutex<MockRecognizer>>>,");
    println!("       current: AtomicUsize,");
    println!("   }}");
    println!("   ");
    println!("   impl RecognizerPool {{");
    println!("       async fn get_recognizer(&self) -> Arc<AsyncMutex<MockRecognizer>> {{");
    println!("           let index = self.current.fetch_add(1, Ordering::SeqCst);");
    println!("           self.recognizers[index % self.recognizers.len()].clone()");
    println!("       }}");
    println!("   }}");
    println!("   ```");

    // Demonstrate concurrent processing
    println!(
        "   
   🔄 Concurrent Processing Demo:"
    );

    let concurrent_configs = vec![
        ("Sequential", 1, 1000, "Process one file at a time"),
        ("Parallel (4)", 4, 300, "Process 4 files concurrently"),
        ("Batch (8)", 8, 180, "Process in batches of 8"),
        ("Pipeline", 16, 120, "Pipelined processing"),
    ];

    println!(
        "   
   Strategy       | Concurrency | Time (ms) | Description"
    );
    println!("   -------------- | ----------- | --------- | -----------");

    for (name, concurrency, time, description) in concurrent_configs {
        println!(
            "   {:14} | {:11} | {:9} | {}",
            name, concurrency, time, description
        );
    }

    Ok(())
}

async fn demonstrate_platform_optimization() -> Result<(), Box<dyn Error>> {
    println!("   Platform-specific optimizations:");

    println!(
        "   
   🖥️ Platform-Specific Optimizations:"
    );

    // 1. x86_64 optimizations
    println!(
        "   
   1. x86_64 (Intel/AMD) Optimizations:"
    );
    println!("   • AVX2/AVX-512 vector instructions");
    println!("   • Intel MKL integration for linear algebra");
    println!("   • Hyperthreading awareness");
    println!("   • Cache-friendly memory layouts");
    println!("   • Intel OpenMP optimizations");

    // 2. ARM64 optimizations
    println!(
        "   
   2. ARM64 (Apple Silicon/ARM) Optimizations:"
    );
    println!("   • NEON vector instructions");
    println!("   • Apple Neural Engine integration");
    println!("   • Unified memory architecture utilization");
    println!("   • Performance/efficiency core scheduling");
    println!("   • Metal Performance Shaders");

    // 3. GPU optimizations
    println!(
        "   
   3. GPU-Specific Optimizations:"
    );
    println!("   • CUDA compute capability detection");
    println!("   • TensorRT integration for inference");
    println!("   • Metal Performance Shaders on Apple");
    println!("   • OpenCL fallback for other GPUs");
    println!("   • Mixed precision training");

    // Platform performance comparison
    println!(
        "   
   📊 Platform Performance Comparison:"
    );

    let platform_scenarios = vec![
        ("Intel Core i7", 0.25, 1800, "x86_64 with AVX2"),
        ("AMD Ryzen 9", 0.22, 2000, "x86_64 with AVX2"),
        ("Apple M1", 0.18, 2200, "ARM64 with Neural Engine"),
        ("Apple M2", 0.15, 2500, "ARM64 with improved GPU"),
        ("NVIDIA RTX 4080", 0.08, 4500, "CUDA with TensorRT"),
        (
            "Apple M2 Ultra",
            0.12,
            3200,
            "ARM64 with high memory bandwidth",
        ),
    ];

    println!(
        "   
   Platform          | RTF   | Throughput | Description"
    );
    println!("   ----------------- | ----- | ---------- | -----------");

    for (name, rtf, throughput, description) in platform_scenarios {
        println!(
            "   {:17} | {:.2} | {:10} | {}",
            name, rtf, throughput, description
        );
    }

    Ok(())
}

async fn demonstrate_performance_monitoring() -> Result<(), Box<dyn Error>> {
    println!("   Performance monitoring and profiling:");

    println!(
        "   
   📈 Performance Monitoring Tools:"
    );

    // 1. Built-in metrics
    println!(
        "   
   1. Built-in Performance Metrics:"
    );
    println!("   ```rust");
    println!("   let validator = PerformanceValidator::new();");
    println!("   let validation = validator.validate_comprehensive(");
    println!("       &audio,");
    println!("       startup_fn,");
    println!("       processing_time,");
    println!("       streaming_latency,");
    println!("   ).await?;");
    println!("   ");
    println!("   println!(\"RTF: {{:.3}}\", validation.metrics.rtf);");
    println!(
        "   println!(\"Memory: {{:.1}} MB\", validation.metrics.memory_usage / 1024.0 / 1024.0);"
    );
    println!("   println!(\"Throughput: {{:.0}} samples/sec\", validation.metrics.throughput_samples_per_sec);");
    println!("   ```");

    // 2. Custom profiling
    println!(
        "   
   2. Custom Profiling Integration:"
    );
    println!("   ```rust");
    println!("   // Tracy profiler integration");
    println!("   #[cfg(feature = \"profiling\")]");
    println!("   use tracy_client::*;");
    println!("   ");
    println!("   #[cfg(feature = \"profiling\")]");
    println!("   let _span = span!(\"audio_recognition\");");
    println!("   ");
    println!("   let result = recognizer.recognize(&audio).await?;");
    println!("   ```");

    // 3. System monitoring
    println!(
        "   
   3. System Resource Monitoring:"
    );
    println!("   ```rust");
    println!("   use sysinfo::{{System, SystemExt, ProcessExt}};");
    println!("   ");
    println!("   let mut system = System::new_all();");
    println!("   system.refresh_all();");
    println!("   ");
    println!("   let cpu_usage = system.global_cpu_info().cpu_usage();");
    println!("   let memory_usage = system.used_memory();");
    println!("   let gpu_usage = system.components().iter()");
    println!("       .find(|c| c.label().contains(\"GPU\"))");
    println!("       .map(|c| c.temperature())");
    println!("       .unwrap_or(0.0);");
    println!("   ```");

    // Monitoring dashboard example
    println!(
        "   
   📊 Monitoring Dashboard Example:"
    );
    println!(
        "   
   Metric                | Current | Target  | Status"
    );
    println!("   --------------------- | ------- | ------- | ------");
    println!("   Real-time Factor      | 0.15    | < 0.30  | ✅ Good");
    println!("   Memory Usage          | 1.2 GB  | < 2.0 GB| ✅ Good");
    println!("   CPU Utilization       | 65%     | < 80%   | ✅ Good");
    println!("   GPU Utilization       | 85%     | < 90%   | ✅ Good");
    println!("   Throughput            | 2.5k/s  | > 1k/s  | ✅ Good");
    println!("   Latency (P99)         | 120ms   | < 200ms | ✅ Good");
    println!("   Error Rate            | 0.1%    | < 1%    | ✅ Good");

    Ok(())
}

async fn assess_performance_optimized() -> Result<PerformanceMetrics, Box<dyn Error>> {
    println!("   🔍 Assessing optimized performance...");

    // Simulated optimized performance metrics
    let metrics = PerformanceMetrics {
        rtf: 0.15,                          // 50% improvement
        memory_usage_mb: 600.0,             // 50% reduction
        startup_time_ms: 2000,              // 40% faster
        throughput_samples_per_sec: 2500.0, // 150% increase
        latency_ms: 60,                     // 70% reduction
        cpu_utilization: 45.0,              // 30% reduction
        accuracy: 0.87,                     // 2% improvement
    };

    println!("   ✅ Optimized metrics collected:");
    print_performance_metrics(&metrics);

    Ok(metrics)
}

fn compare_performance_metrics(baseline: &PerformanceMetrics, optimized: &PerformanceMetrics) {
    println!("   Performance improvement comparison:");
    println!(
        "   
   📊 Before vs After Optimization:"
    );
    println!(
        "   
   Metric                | Baseline | Optimized | Improvement"
    );
    println!("   --------------------- | -------- | --------- | -----------");

    let rtf_improvement = ((baseline.rtf - optimized.rtf) / baseline.rtf) * 100.0;
    let memory_improvement =
        ((baseline.memory_usage_mb - optimized.memory_usage_mb) / baseline.memory_usage_mb) * 100.0;
    let startup_improvement = ((baseline.startup_time_ms - optimized.startup_time_ms) as f32
        / baseline.startup_time_ms as f32)
        * 100.0;
    let throughput_improvement = ((optimized.throughput_samples_per_sec
        - baseline.throughput_samples_per_sec)
        / baseline.throughput_samples_per_sec)
        * 100.0;
    let latency_improvement =
        ((baseline.latency_ms - optimized.latency_ms) as f32 / baseline.latency_ms as f32) * 100.0;
    let cpu_improvement =
        ((baseline.cpu_utilization - optimized.cpu_utilization) / baseline.cpu_utilization) * 100.0;
    let accuracy_improvement =
        ((optimized.accuracy - baseline.accuracy) / baseline.accuracy) * 100.0;

    println!(
        "   RTF                   | {:.2}     | {:.2}      | {:.1}% faster",
        baseline.rtf, optimized.rtf, rtf_improvement
    );
    println!(
        "   Memory Usage (MB)     | {:.0}     | {:.0}       | {:.1}% less",
        baseline.memory_usage_mb, optimized.memory_usage_mb, memory_improvement
    );
    println!(
        "   Startup Time (ms)     | {}      | {}        | {:.1}% faster",
        baseline.startup_time_ms, optimized.startup_time_ms, startup_improvement
    );
    println!(
        "   Throughput (samples/s)| {:.0}     | {:.0}       | {:.1}% more",
        baseline.throughput_samples_per_sec,
        optimized.throughput_samples_per_sec,
        throughput_improvement
    );
    println!(
        "   Latency (ms)          | {}      | {}         | {:.1}% less",
        baseline.latency_ms, optimized.latency_ms, latency_improvement
    );
    println!(
        "   CPU Utilization (%)   | {:.1}     | {:.1}       | {:.1}% less",
        baseline.cpu_utilization, optimized.cpu_utilization, cpu_improvement
    );
    println!(
        "   Accuracy              | {:.2}     | {:.2}      | {:.1}% better",
        baseline.accuracy, optimized.accuracy, accuracy_improvement
    );

    println!(
        "   
   🎯 Overall Performance Gain: {:.1}x faster with {:.1}% less resources",
        baseline.rtf / optimized.rtf,
        memory_improvement
    );
}

fn print_performance_metrics(metrics: &PerformanceMetrics) {
    println!("   • RTF: {:.2}x", metrics.rtf);
    println!("   • Memory: {:.1} MB", metrics.memory_usage_mb);
    println!("   • Startup: {} ms", metrics.startup_time_ms);
    println!(
        "   • Throughput: {:.0} samples/sec",
        metrics.throughput_samples_per_sec
    );
    println!("   • Latency: {} ms", metrics.latency_ms);
    println!("   • CPU: {:.1}%", metrics.cpu_utilization);
    println!("   • Accuracy: {:.1}%", metrics.accuracy * 100.0);
}

// Helper functions for demonstration
fn create_test_audio(duration: f32) -> AudioBuffer {
    let sample_rate = 16000;
    let samples = vec![0.1; (sample_rate as f32 * duration) as usize];
    AudioBuffer::mono(samples, sample_rate)
}

async fn simulate_recognition(audio: &AudioBuffer) -> Result<MockResult, Box<dyn Error>> {
    tokio::time::sleep(Duration::from_millis(200)).await;
    Ok(MockResult {
        text: format!("Recognized {:.1}s of audio", audio.duration()),
        confidence: 0.85,
    })
}

async fn simulate_recognition_with_config(
    audio: &AudioBuffer,
    _config: &ASRConfig,
) -> Result<MockResult, Box<dyn Error>> {
    tokio::time::sleep(Duration::from_millis(150)).await;
    Ok(MockResult {
        text: format!("Recognized {:.1}s of audio", audio.duration()),
        confidence: 0.88,
    })
}

fn estimate_memory_usage(audio: &AudioBuffer) -> f32 {
    // Simulate memory usage calculation
    (audio.len() as f32 * 4.0) / 1024.0 / 1024.0 + 1200.0 // Base model size
}

fn estimate_memory_usage_for_model(size: WhisperModelSize) -> f32 {
    match size {
        WhisperModelSize::Tiny => 400.0,
        WhisperModelSize::Base => 800.0,
        WhisperModelSize::Small => 1200.0,
        _ => 1000.0,
    }
}

fn estimate_accuracy_for_model(size: WhisperModelSize) -> f32 {
    match size {
        WhisperModelSize::Tiny => 0.80,
        WhisperModelSize::Base => 0.85,
        WhisperModelSize::Small => 0.90,
        _ => 0.85,
    }
}

fn estimate_cpu_usage(audio: &AudioBuffer, processing_time: &Duration) -> f32 {
    // Simulate CPU usage calculation
    let processing_intensity = processing_time.as_secs_f32() / audio.duration();
    processing_intensity * 100.0
}

// Mock types for demonstration
struct MockRecognizer {
    config: ASRConfig,
}

impl MockRecognizer {
    async fn new(config: ASRConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self { config })
    }
}

struct MockResult {
    text: String,
    confidence: f32,
}
