# VoiRS Feedback Performance Optimization Guide

This guide provides best practices and optimization strategies for achieving optimal performance with the VoiRS Feedback system.

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Real-Time Feedback Optimization](#real-time-feedback-optimization)
3. [Memory Management](#memory-management)
4. [Database and Persistence](#database-and-persistence)
5. [Concurrent Operations](#concurrent-operations)
6. [Feature Flag Optimization](#feature-flag-optimization)
7. [Profiling and Monitoring](#profiling-and-monitoring)
8. [Common Performance Pitfalls](#common-performance-pitfalls)

## Performance Targets

The VoiRS Feedback system is designed to meet the following performance targets:

- **Real-Time Feedback Latency**: < 100ms end-to-end
- **Real-Time Factor (RTF)**: < 0.1× (process audio faster than real-time)
- **Memory Usage**: < 2GB for typical workloads
- **Concurrent Sessions**: Support 20+ simultaneous active sessions
- **Throughput**: > 1000 feedback operations per second

## Real-Time Feedback Optimization

### 1. Configure Appropriate Buffer Sizes

```rust
use voirs_feedback::realtime::{RealtimeConfig, RealtimeFeedbackSystem};

let config = RealtimeConfig {
    // Smaller buffers = lower latency, but higher CPU usage
    audio_buffer_size: 1024,  // Recommended: 1024-2048

    // Maximum acceptable latency
    max_latency_ms: 50,  // Target sub-100ms

    // Reasonable timeout for streaming
    stream_timeout: std::time::Duration::from_secs(120),

    // Limit concurrent streams based on available resources
    max_concurrent_streams: 10,

    ..Default::default()
};

let system = RealtimeFeedbackSystem::with_config(config).await?;
```

### 2. Use Streaming for Long Audio

For audio longer than a few seconds, use streaming mode to process chunks incrementally:

```rust
// Process in chunks for lower memory usage and faster initial feedback
for chunk in audio_chunks {
    let partial_feedback = system.process_audio_chunk(&chunk).await?;
    // Handle partial results immediately
}
```

### 3. Enable Confidence Filtering

Filter low-confidence feedback to reduce noise and improve processing speed:

```rust
let config = RealtimeConfig {
    enable_confidence_filtering: true,
    quality_threshold: 0.7,           // Only show quality feedback > 0.7
    pronunciation_threshold: 0.8,     // Only show pronunciation feedback > 0.8
    ..Default::default()
};
```

### 4. Optimize Feature Extraction

The feature extraction is one of the most compute-intensive operations. Optimize by:

- Using appropriate audio preprocessing (resampling, normalization)
- Enabling SIMD operations (enabled by default on supported platforms)
- Caching extracted features when processing the same audio multiple times

```rust
// Cache features for repeated processing
let features = system.extract_features(&audio).await?;
// Reuse features for multiple analyses
```

## Memory Management

### 1. Monitor Memory Usage

The feedback system includes built-in memory monitoring:

```rust
use voirs_feedback::performance::MemoryMonitor;

let monitor = MemoryMonitor::new();

// Check current memory usage
let usage = monitor.get_memory_usage().await?;
println!("Memory: {:.2} MB", usage.current_usage_mb);
println!("Peak: {:.2} MB", usage.peak_usage_mb);

// Set memory limits
monitor.set_memory_limit(1024 * 1024 * 1500).await?; // 1.5 GB limit
```

### 2. Cleanup Inactive Sessions

Regularly cleanup inactive sessions to free memory:

```rust
// Configure automatic cleanup
let config = FeedbackConfig {
    session_timeout: std::time::Duration::from_secs(1800), // 30 minutes
    auto_cleanup_interval: std::time::Duration::from_secs(300), // 5 minutes
    ..Default::default()
};
```

### 3. Use Connection Pooling

Reuse database connections instead of creating new ones:

```rust
use voirs_feedback::persistence::PersistenceConfig;

let config = PersistenceConfig {
    max_connections: 10,
    min_connections: 2,
    connection_timeout: std::time::Duration::from_secs(30),
    ..Default::default()
};
```

## Database and Persistence

### 1. Enable Query Caching

The persistence layer includes intelligent query caching:

```rust
let config = PersistenceConfig {
    enable_cache: true,
    cache_size_mb: 256,
    cache_ttl: std::time::Duration::from_secs(300),
    ..Default::default()
};
```

### 2. Use Batch Operations

When saving multiple records, use batch operations:

```rust
// Batch save multiple user progress records
let mut batch = Vec::new();
for user in users {
    batch.push(user_progress);
}
manager.save_batch(&batch).await?;
```

### 3. Optimize Index Usage

The persistence layer automatically creates optimal indexes, but you can customize:

```rust
// Enable index recommendations
manager.enable_index_recommendations().await?;

// Get recommendations
let recommendations = manager.get_index_recommendations().await?;
for rec in recommendations {
    println!("Consider adding index: {}", rec);
}
```

### 4. Configure Prepared Statements

Use prepared statements for frequently executed queries:

```rust
// Prepared statements are cached automatically
// Reuse the same query pattern for best performance
let progress = manager.load_user_progress(&user_id).await?;
```

## Concurrent Operations

### 1. Use Tokio Runtime Efficiently

Configure Tokio for your workload:

```rust
use tokio::runtime::Builder;

let runtime = Builder::new_multi_thread()
    .worker_threads(num_cpus::get())
    .thread_name("voirs-feedback")
    .thread_stack_size(3 * 1024 * 1024)
    .enable_all()
    .build()?;
```

### 2. Parallel Processing

Process multiple sessions concurrently:

```rust
use futures::stream::{self, StreamExt};

// Process up to 10 sessions concurrently
let results = stream::iter(sessions)
    .map(|session| async move {
        session.process_synthesis(&audio, text).await
    })
    .buffer_unordered(10)
    .collect::<Vec<_>>()
    .await;
```

### 3. Load Balancing

The system includes automatic load balancing:

```rust
use voirs_feedback::load_balancer::LoadBalancer;

let balancer = LoadBalancer::new(config).await?;

// Automatically distributes requests across workers
let result = balancer.process_request(request).await?;
```

## Feature Flag Optimization

### 1. Minimal Feature Set for Production

Enable only the features you need:

```toml
[dependencies]
voirs-feedback = { version = "0.1.0-alpha.2", features = [
    "realtime",
    "adaptive",
    "progress-tracking"
] }
# Don't enable "ui", "gamification", etc. if not needed
```

### 2. Disable Debugging Features

In production, disable expensive debugging features:

```rust
let config = RealtimeConfig {
    enable_metrics: false,        // Disable detailed metrics in production
    enable_debugging: false,       // Disable debug logging
    ..Default::default()
};
```

## Profiling and Monitoring

### 1. Built-in Performance Metrics

The system includes comprehensive metrics:

```rust
// Get real-time statistics
let stats = system.get_statistics().await?;
println!("Average latency: {:?}", stats.avg_latency);
println!("Throughput: {:.0} ops/sec", stats.throughput);
println!("Active sessions: {}", stats.active_sessions);
println!("Memory usage: {:.2} MB", stats.memory_usage_mb);
```

### 2. Health Monitoring

Monitor system health:

```rust
use voirs_feedback::health::HealthMonitor;

let monitor = HealthMonitor::new(config).await?;

// Quick health check
let health = monitor.quick_health_check().await?;
println!("Status: {:?}", health.status); // Healthy, Warning, Degraded, Critical

// Comprehensive health check
let detailed = monitor.comprehensive_health_check().await?;
for component in detailed.components {
    println!("{}: {:?} ({:?})",
        component.name,
        component.status,
        component.response_time
    );
}
```

### 3. Performance Profiling

Use the built-in profiler:

```rust
use voirs_feedback::enhanced_performance::SystemProfiler;

let profiler = SystemProfiler::new();

// Start profiling
let _guard = profiler.start_profiling("feedback_generation").await;

// ... perform operations ...

// Get profile data
let profile = profiler.get_profile("feedback_generation").await?;
println!("Average time: {:?}", profile.avg_duration);
println!("P95 latency: {:?}", profile.p95_latency);
println!("P99 latency: {:?}", profile.p99_latency);
```

## Common Performance Pitfalls

### 1. Creating Too Many Sessions

**Problem**: Creating a new session for every request

```rust
// ❌ BAD: Creates new session each time
for request in requests {
    let session = system.create_session(&user_id).await?;
    session.process_synthesis(&audio, text).await?;
}
```

**Solution**: Reuse sessions within their lifetime

```rust
// ✓ GOOD: Reuse session
let session = system.create_session(&user_id).await?;
for request in requests {
    session.process_synthesis(&audio, text).await?;
}
session.save_progress().await?;
```

### 2. Not Using Connection Pooling

**Problem**: Opening a new database connection for each operation

**Solution**: Use the persistence manager's built-in connection pooling

```rust
// Connection pooling is automatic with PersistenceManager
let manager = PersistenceManager::new(config).await?;
// Connections are reused automatically
```

### 3. Processing Entire Audio Files in Memory

**Problem**: Loading large audio files entirely into memory

```rust
// ❌ BAD: Load entire 10-minute audio file
let large_audio = load_entire_file("recording.wav")?; // Could be > 100MB
```

**Solution**: Use streaming for large files

```rust
// ✓ GOOD: Stream audio in chunks
let chunk_size = 16000; // 1 second at 16kHz
for chunk in audio_file.chunks(chunk_size) {
    let feedback = system.process_audio_chunk(chunk).await?;
    // Process feedback immediately
}
```

### 4. Ignoring Memory Limits

**Problem**: Not monitoring or limiting memory usage

**Solution**: Set memory limits and monitor usage

```rust
use voirs_feedback::performance::MemoryMonitor;

let monitor = MemoryMonitor::new();
monitor.set_memory_limit(1024 * 1024 * 2000).await?; // 2GB limit

// Enable automatic cleanup when approaching limit
monitor.enable_auto_cleanup(0.85).await?; // Cleanup at 85% usage
```

### 5. Synchronous Processing in Async Context

**Problem**: Using blocking operations in async code

```rust
// ❌ BAD: Blocking the async runtime
async fn bad_example() {
    std::thread::sleep(std::time::Duration::from_secs(1)); // Blocks!
}
```

**Solution**: Use async-friendly alternatives

```rust
// ✓ GOOD: Non-blocking sleep
async fn good_example() {
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
}
```

## Production Deployment Checklist

- [ ] Set appropriate buffer sizes for your latency requirements
- [ ] Configure connection pooling with optimal pool size
- [ ] Enable query caching with appropriate TTL
- [ ] Set memory limits and enable monitoring
- [ ] Configure automatic session cleanup
- [ ] Disable debug features in production
- [ ] Enable only required feature flags
- [ ] Set up health check endpoints
- [ ] Configure logging levels appropriately
- [ ] Test under expected load with realistic data
- [ ] Monitor performance metrics continuously
- [ ] Set up alerts for degraded performance

## Performance Tuning for Different Scenarios

### Low-Latency Real-Time Feedback

```rust
let config = RealtimeConfig {
    max_latency_ms: 30,
    audio_buffer_size: 512,
    enable_confidence_filtering: true,
    max_concurrent_streams: 5,
    ..Default::default()
};
```

### High-Throughput Batch Processing

```rust
let config = RealtimeConfig {
    max_latency_ms: 200,
    audio_buffer_size: 4096,
    enable_confidence_filtering: false,
    max_concurrent_streams: 50,
    ..Default::default()
};
```

### Memory-Constrained Environments

```rust
let config = FeedbackConfig {
    session_timeout: std::time::Duration::from_secs(600), // 10 minutes
    auto_cleanup_interval: std::time::Duration::from_secs(120), // 2 minutes
    max_cached_models: 5,
    ..Default::default()
};
```

## Conclusion

By following these optimization strategies, you can achieve:

- **Sub-100ms latency** for real-time feedback
- **Efficient memory usage** with automatic cleanup
- **High throughput** with concurrent processing
- **Reliable performance** under load with monitoring

For additional help, see the examples in the `examples/` directory or consult the API documentation at https://docs.rs/voirs-feedback.
