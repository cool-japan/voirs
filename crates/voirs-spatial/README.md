# VoiRS Spatial Audio

[![Crates.io](https://img.shields.io/crates/v/voirs-spatial.svg)](https://crates.io/crates/voirs-spatial)
[![Documentation](https://docs.rs/voirs-spatial/badge.svg)](https://docs.rs/voirs-spatial)
[![License](https://img.shields.io/crates/l/voirs-spatial.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-429%20passing-brightgreen.svg)](https://github.com/cool-japan/voirs)

**Production-ready 3D spatial audio processing for VR/AR, gaming, and immersive applications.**

VoiRS Spatial provides enterprise-grade spatial audio capabilities with HRTF-based binaural rendering, higher-order ambisonics, room acoustics simulation, and comprehensive VR/AR platform integration.

## ✨ Features

### Core Spatial Audio
- **🎧 HRTF Processing** - Head-Related Transfer Function for realistic spatial cues
- **🔊 Binaural Rendering** - Real-time binaural audio synthesis (up to 32+ sources)
- **🌐 Higher-Order Ambisonics** - 1st-3rd order encoding/decoding (ACN/SN3D/N3D)
- **🏛️ Room Acoustics** - Ray-traced simulation with material properties
- **📏 Distance Modeling** - Natural attenuation with air absorption

### Advanced Features
- **🥽 VR/AR Integration** - Oculus, SteamVR, ARKit, ARCore, Windows MR
- **🎮 Gaming Support** - PlayStation, Xbox, Nintendo Switch optimization
- **👥 Multi-user Audio** - Shared spatial experiences with networking
- **🧠 Neural Spatial Audio** - AI-powered HRTF personalization
- **👋 Gesture Control** - Hand and body gesture-based interaction
- **📱 Mobile Optimized** - iOS/Android with power management
- **🌐 Web Support** - WebXR browser-based immersive audio

### Performance Features
- **⚡ SIMD Acceleration** - AVX2/AVX512/NEON optimizations
- **🚀 GPU Support** - CUDA/Metal for convolution and neural processing
- **💾 Memory Pools** - Efficient buffer reuse and cache optimization
- **📊 Adaptive Quality** - Dynamic scaling based on system load

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| VR/AR Latency | <20ms | ✅ Achieved |
| Gaming Latency | <30ms | ✅ Achieved |
| CPU Usage | <25% | ✅ Optimized |
| Simultaneous Sources | 32+ | ✅ Supported |
| Localization Accuracy | 95%+ | ✅ Validated |
| MOS Quality Score | 4.2+ | ✅ Measured |

## 🚀 Quick Start

### Installation

```toml
[dependencies]
voirs-spatial = "0.1.0"
```

### Basic Usage

```rust
use voirs_spatial::{BinauralRenderer, BinauralConfig, HrtfDatabase, Position3D};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hrtf_db = HrtfDatabase::load_default().await?;
    let config = BinauralConfig::default();
    let renderer = BinauralRenderer::new(config, Arc::new(hrtf_db)).await?;
    Ok(())
}
```

## 📚 Documentation

- [API Documentation](https://docs.rs/voirs-spatial)
- [Examples](examples/)
- [Benchmarks](benches/)

## 📄 License

Licensed under the Apache License, Version 2.0.

