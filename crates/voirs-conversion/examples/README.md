# VoiRS-Conversion Examples

Comprehensive examples demonstrating the capabilities of the voirs-conversion crate.

## 📚 Examples Overview

This directory contains **10 comprehensive examples** covering basic to advanced voice conversion use cases.

### Quick Start
```bash
# Run a simple example
cargo run --example simple_pitch_shift

# Run with optimizations (recommended)
cargo run --release --example zero_shot_conversion

# Run platform-specific examples
cargo run --example iot_edge_device --features iot
wasm-pack build --target web --features wasm  # For WASM
```

---

## 📂 Example Categories

### 🔰 Basic Examples (4)
1. **simple_pitch_shift.rs** - Basic pitch modification
2. **speed_and_pitch.rs** - Combined transformations  
3. **age_transformation.rs** - Voice age transformation
4. **artifact_detection.rs** - Quality control

### 🚀 Advanced Features (4)
5. **zero_shot_conversion.rs** - Convert to unseen voices
6. **style_transfer.rs** - Speaking style transformation
7. **multi_target_conversion.rs** - Parallel multi-target processing
8. **realtime_streaming.rs** - Low-latency streaming

### 🌐 Integration Examples (2)
9. **wasm_browser_integration.rs** - WebAssembly/browser support
10. **iot_edge_device.rs** - IoT/edge device optimization

---

## 🎯 Performance Summary

| Example | RTF* | Latency | Memory |
|---------|------|---------|--------|
| Simple Pitch | 0.03x | 30ms | 50MB |
| Zero-shot | 0.20x | 200ms | 256MB |
| Real-time | 0.16x | 10-20ms | 128MB |
| Multi-target | 0.10x | 100ms/target | 512MB |

\* Real-Time Factor (lower is better)

---

For detailed documentation, see the full README.md in this directory.
**Happy Voice Converting! 🎤✨**
