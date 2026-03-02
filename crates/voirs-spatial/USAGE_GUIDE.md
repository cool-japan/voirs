# VoiRS Spatial Audio - Usage Guide

> **Comprehensive guide to using the VoiRS Spatial Audio system for 3D spatial audio processing**

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [Usage Patterns](#usage-patterns)
- [Best Practices](#best-practices)
- [Performance Optimization](#performance-optimization)
- [Platform-Specific Guides](#platform-specific-guides)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
voirs-spatial = "0.1.0-alpha.2"

# Optional platform features
voirs-spatial = { version = "0.1.0-alpha.2", features = ["steamvr", "webxr"] }
```

### Basic Usage

```rust
use voirs_spatial::{Position3D, SpatialConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create spatial configuration
    let config = SpatialConfig::default();

    // Position a sound source
    let source_pos = Position3D::new(1.0, 0.0, 0.5); // 1m right, 0.5m up
    let listener_pos = Position3D::new(0.0, 0.0, 0.0); // Origin

    let distance = source_pos.distance_to(&listener_pos);
    println!("Source distance: {:.2}m", distance);

    Ok(())
}
```

---

## Architecture Overview

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    VoiRS Spatial System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Position  │  │     HRTF     │  │    Binaural      │   │
│  │   Tracking  │  │   Database   │  │    Rendering     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    Room     │  │  Ambisonics  │  │     Neural       │   │
│  │  Acoustics  │  │   Encoding   │  │   Processing     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Platform  │  │   Multi-user │  │   Performance    │   │
│  │ Integration │  │ Environments │  │   Monitoring     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Audio → Position Tracking → HRTF Processing → Room Simulation
    ↓                                                        ↓
    └────────────→ Binaural Rendering ←───────────────────┘
                        ↓
                  Output (Stereo)
```

---

## Core Concepts

### 1. 3D Positioning

**Position3D** represents a point in 3D space using Cartesian coordinates:

```rust
use voirs_spatial::Position3D;

// Create positions
let origin = Position3D::new(0.0, 0.0, 0.0);
let front_right = Position3D::new(1.0, 0.0, 1.0); // +X = right, +Z = forward

// Vector operations
let distance = origin.distance_to(&front_right);
let normalized = front_right.normalized();
let direction = front_right - origin;
```

**Coordinate System:**
- **X-axis**: Left (-) to Right (+)
- **Y-axis**: Down (-) to Up (+)
- **Z-axis**: Back (-) to Forward (+)

### 2. HRTF (Head-Related Transfer Function)

HRTFs encode how sound from different directions reaches each ear:

```rust
use std::sync::Arc;
use voirs_spatial::HrtfDatabase;

// Load default HRTF database
let hrtf_db = HrtfDatabase::load_default().await?;

// Or load custom database
let custom_hrtf = HrtfDatabase::load_from_file("path/to/hrtf.sofa").await?;

// Personalize HRTF
let measurements = HeadMeasurements {
    head_circumference: 57.0, // cm
    pinna_height: 6.5,        // cm
    pinna_width: 3.5,         // cm
    ..Default::default()
};
let personalized = hrtf_db.personalize(&measurements)?;
```

### 3. Binaural Rendering

Convert mono sources to stereo with spatial cues:

```rust
use voirs_spatial::{BinauralRenderer, BinauralConfig};

let config = BinauralConfig {
    sample_rate: 48000,
    buffer_size: 512,
    max_sources: 16,
    quality_level: 0.9,
    enable_distance_modeling: true,
    ..Default::default()
};

let mut renderer = BinauralRenderer::new(config, Arc::new(hrtf_db))?;

// Add sources
let source_id = renderer.add_source(
    Position3D::new(1.0, 0.0, 0.0),
    SourceType::Static
)?;

// Process audio
let output = renderer.process_frame()?;
```

### 4. Room Acoustics

Simulate realistic room reverberation and reflections:

```rust
use voirs_spatial::room::{RoomConfig, RoomSimulator, WallMaterial};

let room_config = RoomConfig {
    dimensions: Position3D::new(10.0, 5.0, 8.0), // 10m × 5m × 8m
    wall_material: WallMaterial::Concrete,
    enable_early_reflections: true,
    enable_late_reverb: true,
    max_reflection_order: 3,
    reverb_time: 1.2,
    absorption_coefficient: 0.3,
};

let mut room = RoomSimulator::new(room_config)?;

// Process audio with room acoustics
let processed = room.process(
    &input_audio,
    &source_position,
    &listener_position
)?;
```

---

## Usage Patterns

### Pattern 1: Real-time VR Audio

**Goal:** Achieve <20ms latency for VR applications

```rust
use voirs_spatial::{
    Position3D, BinauralRenderer, BinauralConfig,
    position::HeadTracker, HrtfDatabase,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optimize for low latency
    let config = BinauralConfig {
        sample_rate: 48000,
        buffer_size: 256,  // Small buffer for low latency
        max_sources: 8,    // Limit sources
        optimize_for_latency: true,
        quality_level: 0.8, // Slightly reduce quality for speed
        ..Default::default()
    };

    let hrtf_db = HrtfDatabase::load_default().await?;
    let mut renderer = BinauralRenderer::new(config, Arc::new(hrtf_db))?;

    // Setup head tracking with prediction
    let head_tracker = HeadTracker::new(HeadTrackerConfig {
        prediction_time_ms: 15.0, // Compensate for VR latency
        ..Default::default()
    })?;

    // Real-time processing loop
    loop {
        // Get predicted head position
        let head_pos = head_tracker.get_predicted_position()?;
        renderer.update_listener_position(head_pos)?;

        // Process audio
        let output = renderer.process_frame()?;

        // Check latency
        let metrics = renderer.get_performance_metrics();
        if metrics.avg_processing_time_ms > 20.0 {
            eprintln!("WARNING: Latency exceeded VR target!");
        }
    }
}
```

### Pattern 2: Multi-source Gaming Environment

**Goal:** Handle many simultaneous sources efficiently

```rust
use voirs_spatial::{Position3D, BinauralRenderer, SourceType};

async fn setup_game_audio() -> Result<BinauralRenderer, Box<dyn std::error::Error>> {
    let config = BinauralConfig {
        max_sources: 32,  // Many sources
        use_gpu: true,    // Enable GPU acceleration
        quality_level: 0.7, // Medium quality for performance
        ..Default::default()
    };

    let hrtf_db = HrtfDatabase::load_default().await?;
    let mut renderer = BinauralRenderer::new(config, Arc::new(hrtf_db))?;

    // Add game audio sources
    let sources = vec![
        ("footsteps", Position3D::new(3.0, 0.0, 2.0), SourceType::Moving),
        ("ambient", Position3D::new(0.0, 5.0, 0.0), SourceType::Streaming),
        ("gunfire", Position3D::new(-2.0, 0.0, 5.0), SourceType::OneShot),
    ];

    for (name, pos, stype) in sources {
        renderer.add_source(pos, stype)?;
    }

    Ok(renderer)
}
```

### Pattern 3: Multi-user Virtual Environment

**Goal:** Synchronized spatial audio for multiple users

```rust
use voirs_spatial::multiuser::{MultiuserEnvironment, MultiuserConfig, UserRole};

async fn create_virtual_world() -> Result<MultiuserEnvironment, Box<dyn std::error::Error>> {
    let config = MultiuserConfig {
        max_users: 50,
        enable_voice_activity_detection: true,
        network_buffer_ms: 100.0,
        enable_spatial_zones: true,
        ..Default::default()
    };

    let mut env = MultiuserEnvironment::new(config)?;

    // Create spatial zones
    env.add_zone(SpatialZone {
        id: "lobby".to_string(),
        zone_type: ZoneType::Public,
        center: Position3D::new(0.0, 0.0, 0.0),
        radius: 10.0,
        ..Default::default()
    })?;

    // Add users
    let alice = env.add_user(
        "alice".to_string(),
        UserRole::Speaker,
        Position3D::new(0.0, 0.0, 0.0)
    )?;

    Ok(env)
}
```

### Pattern 4: Neural Spatial Audio

**Goal:** Use AI-powered spatial synthesis

```rust
use voirs_spatial::neural::{NeuralProcessor, NeuralSpatialConfig, NeuralModelType};

async fn neural_spatial_processing() -> Result<(), Box<dyn std::error::Error>> {
    let config = NeuralSpatialConfig {
        model_type: NeuralModelType::Feedforward,
        hidden_layers: vec![256, 512, 256],
        use_gpu: true,
        enable_adaptive_quality: true,
        max_latency_ms: 20.0,
        ..Default::default()
    };

    let processor = NeuralProcessor::new(config)?;

    // Process with neural network
    let output = processor.process_source(
        &input_audio,
        &source_position,
        &listener_position,
        &listener_orientation
    )?;

    println!("Neural confidence: {:.1}%", output.confidence * 100.0);
    println!("Quality score: {:.3}", output.quality_score);

    Ok(())
}
```

---

## Best Practices

### 1. Configuration Selection

**Choose sample rate wisely:**
- **48kHz**: Standard for VR/AR and gaming (recommended)
- **44.1kHz**: Standard for music applications
- **96kHz**: High-quality applications (higher CPU cost)

**Buffer size trade-offs:**
- **Small (128-256)**: Low latency, higher CPU usage, risk of underruns
- **Medium (512)**: Balanced (recommended for most applications)
- **Large (1024-2048)**: High latency, lower CPU usage, stable

### 2. Memory Management

**Enable memory pools for better performance:**

```rust
use voirs_spatial::memory::{MemoryManager, MemoryConfig};

let mem_config = MemoryConfig {
    enable_buffer_pools: true,
    max_pool_size_mb: 64,
    enable_caching: true,
    cache_size_mb: 32,
    enable_prefetching: true,
};

let mem_manager = MemoryManager::new(mem_config)?;
```

**Monitor memory usage:**

```rust
let stats = mem_manager.get_statistics();
println!("Memory usage: {:.1}MB", stats.current_usage_mb);
println!("Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
```

### 3. Error Handling

**Always handle errors properly:**

```rust
use voirs_spatial::SpatialError;

match renderer.add_source(position, source_type) {
    Ok(source_id) => {
        println!("Source added: {}", source_id);
    }
    Err(SpatialError::MaxSourcesReached { max, .. }) => {
        eprintln!("Cannot add source: max {} sources reached", max);
        // Handle gracefully - maybe remove old sources
    }
    Err(e) => {
        eprintln!("Error adding source: {}", e);
        return Err(e.into());
    }
}
```

### 4. Performance Monitoring

**Track performance metrics:**

```rust
use voirs_spatial::performance::PerformanceMonitor;

let monitor = PerformanceMonitor::new();
monitor.start();

// ... process audio ...

let metrics = monitor.get_metrics();
if metrics.avg_latency_ms > target_latency {
    eprintln!("Performance degraded: {:.2}ms", metrics.avg_latency_ms);
    // Reduce quality or source count
}
```

### 5. Thread Safety

**VoiRS Spatial is designed for multi-threaded use:**

```rust
use std::sync::Arc;
use tokio::task;

let renderer = Arc::new(Mutex::new(renderer));

// Spawn audio processing thread
let renderer_clone = Arc::clone(&renderer);
task::spawn(async move {
    loop {
        let mut r = renderer_clone.lock().await;
        r.process_frame()?;
    }
});
```

---

## Performance Optimization

### CPU Optimization

1. **Reduce source count** when CPU usage is high
2. **Lower quality_level** (0.6-0.8 for non-critical applications)
3. **Disable expensive features**:
   - `enable_air_absorption: false`
   - `max_reflection_order: 1` (reduce reflections)
4. **Use SIMD operations** (automatically enabled on supported platforms)

### GPU Optimization

```rust
let config = BinauralConfig {
    use_gpu: true,  // Enable GPU acceleration
    ..Default::default()
};
```

**GPU is beneficial when:**
- Processing 8+ sources simultaneously
- Using high-quality settings (quality_level > 0.8)
- Neural processing enabled
- Complex room acoustics with many reflections

### Latency Optimization

**For VR/AR (<20ms target):**

```rust
BinauralConfig {
    buffer_size: 256,
    optimize_for_latency: true,
    quality_level: 0.75,
    max_sources: 8,
    enable_early_reflections: true,
    enable_late_reverb: false,  // Disable late reverb for lower latency
    ..Default::default()
}
```

**For Gaming (<30ms target):**

```rust
BinauralConfig {
    buffer_size: 512,
    quality_level: 0.8,
    max_sources: 16,
    enable_distance_modeling: true,
    ..Default::default()
}
```

---

## Platform-Specific Guides

### VR Platforms

#### SteamVR

```toml
[dependencies]
voirs-spatial = { version = "0.1.0-alpha.2", features = ["steamvr"] }
```

```rust
use voirs_spatial::platforms::steamvr::SteamVRPlatform;

let platform = SteamVRPlatform::new()?;

// Get HMD tracking
let (position, orientation) = platform.get_hmd_tracking()?;

// Update renderer
renderer.update_listener_position(position)?;
renderer.update_listener_orientation(orientation)?;
```

#### Oculus/Meta

```rust
use voirs_spatial::platforms::oculus::OculusPlatform;

let platform = OculusPlatform::new()?;
let tracking_data = platform.get_tracking_data()?;
```

### Mobile Platforms

#### iOS

```toml
[target.'cfg(target_os = "ios")'.dependencies]
voirs-spatial = { version = "0.1.0-alpha.2", features = ["arkit"] }
```

```rust
use voirs_spatial::mobile::IOSOptimizer;

let optimizer = IOSOptimizer::new()?;
optimizer.enable_low_power_mode()?;

// Optimize for battery life
let config = optimizer.get_optimized_config(
    BatteryLevel::Low,    // Battery state
    ThermalState::Normal  // Thermal state
)?;
```

#### Android

```toml
[target.'cfg(target_os = "android")'.dependencies]
voirs-spatial = { version = "0.1.0-alpha.2", features = ["arcore"] }
```

```rust
use voirs_spatial::mobile::AndroidOptimizer;

let optimizer = AndroidOptimizer::new()?;
let config = optimizer.optimize_for_device()?;
```

### Web (WebXR)

```toml
[dependencies]
voirs-spatial = { version = "0.1.0-alpha.2", features = ["webxr"] }
```

```rust
use voirs_spatial::webxr::WebXRProcessor;

let processor = WebXRProcessor::new(BrowserType::Chrome)?;

// Process spatial audio in browser
let output = processor.process_with_pose(
    &input_audio,
    &webxr_pose
)?;
```

---

## Troubleshooting

### Common Issues

#### Issue: High Latency

**Symptoms:** Audio lags behind visual movement (>50ms)

**Solutions:**
1. Reduce `buffer_size` to 256 or 512
2. Set `optimize_for_latency: true`
3. Reduce `max_sources`
4. Lower `quality_level` to 0.7-0.8
5. Disable `enable_late_reverb`

#### Issue: Audio Artifacts/Clicks

**Symptoms:** Popping or clicking sounds in output

**Solutions:**
1. Increase `buffer_size` to 512 or 1024
2. Check for underruns: `metrics.underruns > 0`
3. Enable `enable_caching` in memory configuration
4. Ensure processing time < buffer duration

#### Issue: Poor Localization

**Symptoms:** Cannot distinguish front/back or elevation

**Solutions:**
1. Use personalized HRTF
2. Increase `quality_level` to 0.9+
3. Enable `enable_distance_modeling`
4. Check head tracking accuracy
5. Verify correct coordinate system

#### Issue: High CPU Usage

**Symptoms:** CPU usage >50%

**Solutions:**
1. Enable GPU: `use_gpu: true`
2. Reduce `max_sources`
3. Lower `quality_level`
4. Reduce `max_reflection_order` in room simulation
5. Disable `enable_air_absorption` if not critical

### Debug Mode

Enable detailed logging:

```rust
env_logger::init();

// Set log level
std::env::set_var("RUST_LOG", "voirs_spatial=debug");
```

### Performance Profiling

```rust
use voirs_spatial::performance::PerformanceProfiler;

let profiler = PerformanceProfiler::new();
profiler.start_recording();

// ... run audio processing ...

let report = profiler.generate_report();
println!("{}", report);
```

---

## Additional Resources

- **API Documentation**: <https://docs.rs/voirs-spatial>
- **Examples**: See `examples/` directory
- **Source Code**: <https://github.com/cool-japan/voirs>
- **Issue Tracker**: <https://github.com/cool-japan/voirs/issues>

---

**Version:** 0.1.0-alpha.2
**Last Updated:** 2025-12-09
**License:** Apache-2.0
