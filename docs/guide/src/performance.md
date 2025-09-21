# Performance Tuning

This guide provides comprehensive strategies for optimizing VoiRS Recognizer performance across different use cases and hardware configurations.

## Performance Metrics

### Key Performance Indicators

VoiRS Recognizer tracks several important metrics:

- **Real-time Factor (RTF)**: Processing time / audio duration (lower is better)
- **Latency**: Time from audio input to result output
- **Throughput**: Audio processed per unit time
- **Memory Usage**: Peak and average memory consumption
- **Accuracy**: Word Error Rate (WER) and confidence scores

### Benchmarking

Use the built-in performance validator:

```rust
use voirs_recognizer::prelude::*;

// Set performance requirements
let requirements = PerformanceRequirements {
    max_rtf: 0.3,                    // Process faster than 0.3x real-time
    max_memory_usage: 2_000_000_000, // 2GB memory limit
    max_startup_time_ms: 3000,       // 3 second startup
    max_streaming_latency_ms: 200,   // 200ms streaming latency
};

let validator = PerformanceValidator::with_requirements(requirements);

// Validate during processing
let metrics = validator.measure_inference(&recognizer, &audio).await?;
println!("RTF: {:.3}, Memory: {:.1}MB", 
         metrics.rtf, metrics.peak_memory_mb);
```

## Hardware Optimization

### CPU Optimization

#### Thread Configuration

```rust
let config = RecognitionConfig::default()
    .with_cpu_threads(num_cpus::get())  // Use all available cores
    .with_thread_affinity(true)         // Pin threads to cores
    .with_numa_awareness(true);         // NUMA-aware allocation
```

#### SIMD Acceleration

Enable SIMD optimizations:

```rust
// Compile with target-specific optimizations
// RUSTFLAGS="-C target-cpu=native" cargo build --release

let config = RecognitionConfig::default()
    .with_simd_acceleration(true)       // Enable SIMD operations
    .with_cpu_optimization_level(3);    // Maximum CPU optimization
```

#### Memory Configuration

```rust
let config = RecognitionConfig::default()
    .with_memory_pool_size(1_000_000_000)  // 1GB memory pool
    .with_cache_size(500_000_000)          // 500MB model cache
    .with_garbage_collection(false);       // Disable GC for real-time
```

### GPU Acceleration

#### CUDA Configuration (NVIDIA)

```rust
let config = RecognitionConfig::default()
    .with_gpu_acceleration(true)
    .with_gpu_device_id(0)              // Select GPU device
    .with_gpu_memory_fraction(0.8)      // Use 80% of GPU memory
    .with_mixed_precision(true);        // Enable FP16/INT8
```

#### Metal Configuration (Apple Silicon)

```rust
let config = RecognitionConfig::default()
    .with_gpu_acceleration(true)
    .with_metal_performance_shaders(true)  // Use MPS on Apple Silicon
    .with_unified_memory(true);            // Leverage unified memory
```

## Model Optimization

### Model Size Selection

Choose models based on your performance requirements:

| Model Size | Parameters | Memory | RTF | Accuracy | Use Case |
|------------|------------|--------|-----|----------|----------|
| Tiny       | 39M        | ~150MB | 0.1 | Good     | Edge devices |
| Base       | 74M        | ~300MB | 0.2 | Better   | General use |
| Small      | 244M       | ~1GB   | 0.4 | Great    | High accuracy |
| Medium     | 769M       | ~2.5GB | 0.7 | Excellent| Research |
| Large      | 1550M      | ~5GB   | 1.2 | Best     | Production |

```rust
// For real-time applications
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Tiny)
    .with_precision(Precision::Float16);

// For batch processing with high accuracy
let config = RecognitionConfig::default()
    .with_model_size(ModelSize::Large)
    .with_precision(Precision::Float32);
```

### Quantization

Reduce model size and improve speed:

```rust
let config = RecognitionConfig::default()
    .with_quantization(Quantization::Int8)    // 8-bit quantization
    .with_dynamic_quantization(true)          // Dynamic quantization
    .with_pruning_ratio(0.1);                 // 10% weight pruning
```

### Model Caching

Cache models for faster subsequent loads:

```rust
let config = RecognitionConfig::default()
    .with_model_cache_dir("/path/to/cache")
    .with_persistent_cache(true)
    .with_cache_compression(true);
```

## Audio Processing Optimization

### Sample Rate Optimization

Choose optimal sample rates:

```rust
// For speech recognition (recommended)
let config = RecognitionConfig::default()
    .with_sample_rate(16000);

// For high-quality audio analysis
let config = RecognitionConfig::default()
    .with_sample_rate(48000);
```

### Chunk Size Tuning

Optimize chunk sizes for your use case:

```rust
// Low latency (real-time)
let config = RecognitionConfig::default()
    .with_chunk_duration_ms(100)     // 100ms chunks
    .with_overlap_duration_ms(25);   // 25ms overlap

// High throughput (batch)
let config = RecognitionConfig::default()
    .with_chunk_duration_ms(1000)    // 1 second chunks
    .with_overlap_duration_ms(100);  // 100ms overlap
```

### Preprocessing Pipeline

Optimize preprocessing for performance:

```rust
// Minimal preprocessing for speed
let config = RecognitionConfig::default()
    .with_noise_suppression(false)
    .with_auto_gain_control(true)     // Keep AGC for stability
    .with_echo_cancellation(false)
    .with_bandwidth_extension(false);

// Full preprocessing for accuracy
let config = RecognitionConfig::default()
    .with_noise_suppression(true)
    .with_auto_gain_control(true)
    .with_echo_cancellation(true)
    .with_bandwidth_extension(true)
    .with_quality_enhancement(true);
```

## Memory Management

### Memory Pool Configuration

```rust
let config = RecognitionConfig::default()
    .with_memory_pool_size(2_000_000_000)     // 2GB pool
    .with_memory_alignment(64)                // 64-byte alignment
    .with_zero_copy_optimization(true)        // Avoid unnecessary copies
    .with_memory_mapped_models(true);         // Memory-map large models
```

### Garbage Collection

For real-time applications:

```rust
let config = RecognitionConfig::default()
    .with_manual_memory_management(true)      // Manual GC control
    .with_memory_pressure_threshold(0.8)      // GC at 80% usage
    .with_incremental_gc(true);               // Incremental collection
```

## Streaming Optimization

### Buffer Management

```rust
let config = RecognitionConfig::default()
    .with_input_buffer_size(4096)             // 4KB input buffer
    .with_output_buffer_size(8192)            // 8KB output buffer
    .with_circular_buffer(true)               // Use circular buffers
    .with_zero_copy_streaming(true);          // Minimize copies
```

### Latency Reduction

```rust
let config = RecognitionConfig::default()
    .with_streaming_mode(StreamingMode::LowLatency)
    .with_vad_aggressiveness(VadLevel::Moderate)    // Balance detection
    .with_partial_results(true)                     // Immediate feedback
    .with_result_timeout_ms(50);                   // 50ms timeout
```

## Batch Processing Optimization

### Parallel Processing

```rust
use rayon::prelude::*;

// Process multiple files in parallel
let files = vec!["audio1.wav", "audio2.wav", "audio3.wav"];
let results: Vec<_> = files.par_iter()
    .map(|file| {
        let recognizer = Recognizer::new(config.clone()).await?;
        let audio = recognizer.load_audio(file).await?;
        recognizer.recognize(&audio).await
    })
    .collect();
```

### Batch Size Tuning

```rust
let config = RecognitionConfig::default()
    .with_batch_size(16)                      // Process 16 files at once
    .with_dynamic_batching(true)              // Adjust batch size dynamically
    .with_batch_timeout_ms(1000);             // 1 second batch timeout
```

## Platform-Specific Optimizations

### Linux Optimizations

```bash
# Enable huge pages
echo 'vm.nr_hugepages=1024' | sudo tee -a /etc/sysctl.conf

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

```rust
let config = RecognitionConfig::default()
    .with_huge_pages(true)                    // Use huge pages on Linux
    .with_cpu_affinity(vec![0, 1, 2, 3])     // Pin to specific cores
    .with_process_priority(ProcessPriority::High);
```

### macOS Optimizations

```rust
let config = RecognitionConfig::default()
    .with_grand_central_dispatch(true)        // Use GCD on macOS
    .with_metal_acceleration(true)            // Metal on Apple Silicon
    .with_core_ml_backend(true);              // CoreML integration
```

### Windows Optimizations

```rust
let config = RecognitionConfig::default()
    .with_windows_ml_backend(true)            // Use Windows ML
    .with_directx_acceleration(true)          // DirectX 12 acceleration
    .with_numa_aware_allocation(true);        // NUMA awareness
```

## Profiling and Monitoring

### Built-in Profiling

```rust
let config = RecognitionConfig::default()
    .with_profiling(true)                     // Enable profiling
    .with_metrics_collection(true)            // Collect metrics
    .with_performance_logging(true);          // Log performance data

// Access profiling data
let profiler = recognizer.profiler();
let report = profiler.generate_report();
println!("Processing breakdown:\n{}", report);
```

### External Profiling Tools

#### CPU Profiling with `perf`

```bash
# Profile your application
perf record --call-graph=dwarf ./your_app
perf report

# Or use flamegraph
cargo flamegraph --example basic_speech_recognition
```

#### Memory Profiling with `valgrind`

```bash
# Check for memory leaks
valgrind --leak-check=full ./your_app

# Profile memory usage
valgrind --tool=massif ./your_app
ms_print massif.out.* > memory_profile.txt
```

## Performance Best Practices

### 1. Choose the Right Model

```rust
// For edge devices
ModelSize::Tiny + Precision::Int8

// For mobile devices  
ModelSize::Base + Precision::Float16

// For servers
ModelSize::Large + Precision::Float32
```

### 2. Optimize Audio Pipeline

```rust
// Minimize allocations
let config = RecognitionConfig::default()
    .with_zero_copy_optimization(true)
    .with_preallocated_buffers(true)
    .with_memory_pool(true);
```

### 3. Cache Everything Possible

```rust
let config = RecognitionConfig::default()
    .with_model_cache(true)
    .with_feature_cache(true)
    .with_result_cache(true);
```

### 4. Use Appropriate Precision

```rust
// High precision for research
.with_precision(Precision::Float32)

// Balanced precision for production
.with_precision(Precision::Float16)

// Maximum speed for edge
.with_precision(Precision::Int8)
```

### 5. Monitor Performance

```rust
// Set up monitoring
let monitor = PerformanceMonitor::new()
    .with_rtf_threshold(0.5)
    .with_memory_threshold(2_000_000_000)
    .with_latency_threshold(200);

monitor.start_monitoring(&recognizer);
```

## Troubleshooting Performance Issues

### High Memory Usage

1. **Reduce model size**: Use smaller models
2. **Enable quantization**: Reduce precision
3. **Limit batch size**: Process fewer files simultaneously
4. **Enable memory compression**: Compress cached data

### High Latency

1. **Reduce chunk size**: Process smaller audio chunks
2. **Disable heavy preprocessing**: Turn off non-essential features
3. **Use GPU acceleration**: Offload computation to GPU
4. **Optimize thread count**: Match CPU cores

### Low Throughput

1. **Increase batch size**: Process more files at once
2. **Enable parallel processing**: Use multiple threads
3. **Use faster models**: Trade accuracy for speed
4. **Optimize I/O**: Use faster storage and network

### Memory Leaks

1. **Check model lifecycle**: Properly dispose of models
2. **Monitor buffer usage**: Ensure buffers are released
3. **Use profiling tools**: Identify leak sources
4. **Enable garbage collection**: For long-running applications

## Performance Validation

Create performance tests for your specific use case:

```rust
#[tokio::test]
async fn test_performance_requirements() {
    let config = RecognitionConfig::default()
        .with_model_size(ModelSize::Base);
    
    let recognizer = Recognizer::new(config).await.unwrap();
    let audio = create_test_audio();
    
    let start = std::time::Instant::now();
    let result = recognizer.recognize(&audio).await.unwrap();
    let duration = start.elapsed();
    
    let rtf = duration.as_secs_f64() / (audio.len() as f64 / audio.sample_rate() as f64);
    
    assert!(rtf < 0.3, "RTF too high: {:.3}", rtf);
    assert!(result.confidence > 0.8, "Confidence too low: {:.3}", result.confidence);
}
```

By following these performance tuning guidelines, you can optimize VoiRS Recognizer for your specific hardware, use case, and performance requirements. Regular monitoring and profiling will help you maintain optimal performance as your application evolves.