# VoiRS Troubleshooting Guide

> **Common Issues and Solutions for VoiRS Voice Synthesis**

This guide helps you resolve common issues when working with VoiRS voice synthesis examples and applications.

## 🚨 Quick Diagnostics

### Is your issue here?
- [Installation Problems](#installation-problems) - Can't get VoiRS working
- [Build Errors](#build-errors) - Compilation failures
- [Runtime Errors](#runtime-errors) - Crashes and exceptions
- [Audio Issues](#audio-issues) - No sound or poor quality
- [Performance Problems](#performance-problems) - Slow or high memory usage
- [Platform-Specific Issues](#platform-specific-issues) - OS-related problems

---

## 🔧 Installation Problems

### Issue: Cargo build fails with dependency errors
**Symptoms:**
```
error: failed to resolve dependencies
```

**Solutions:**
1. **Update Rust toolchain:**
   ```bash
   rustup update stable
   rustup default stable
   ```

2. **Clean and rebuild:**
   ```bash
   cargo clean
   cargo build --release
   ```

3. **Check Rust version:**
   ```bash
   rustc --version  # Should be 1.70 or later
   ```

### Issue: CUDA/GPU compilation errors on macOS
**Symptoms:**
```
Failed to execute `nvcc`: No such file or directory
```

**Solutions:**
1. **Disable GPU features:**
   ```bash
   cargo build --no-default-features
   ```

2. **Use CPU-only build:**
   ```bash
   cargo build --features="cpu-only"
   ```

3. **Skip CLI build:**
   ```bash
   cargo build --workspace --exclude voirs-cli
   ```

### Issue: Missing system dependencies
**Symptoms:**
```
error: linking with `cc` failed
```

**Solutions:**

**macOS:**
```bash
xcode-select --install
brew install cmake portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libasound2-dev portaudio19-dev
```

**Windows:**
```powershell
# Install Visual Studio Build Tools
# Install cmake through Visual Studio Installer
```

---

## 🔨 Build Errors

### Issue: Out of memory during compilation
**Symptoms:**
```
error: could not compile `voirs` due to 1 previous error
signal: 9, SIGKILL: kill
```

**Solutions:**
1. **Reduce parallel jobs:**
   ```bash
   cargo build -j 1
   ```

2. **Use debug build for testing:**
   ```bash
   cargo build  # Instead of --release
   ```

3. **Build specific crates:**
   ```bash
   cargo build -p voirs-emotion
   ```

### Issue: Feature conflicts
**Symptoms:**
```
error: multiple versions of crate found
```

**Solutions:**
1. **Check feature flags:**
   ```bash
   cargo tree --features=default --duplicates
   ```

2. **Use specific feature sets:**
   ```bash
   cargo build --features="emotion,synthesis" --no-default-features
   ```

### Issue: Outdated dependencies
**Symptoms:**
```
error: package requires Rust version X but Y is installed
```

**Solutions:**
1. **Update workspace dependencies:**
   ```bash
   cargo update
   ```

2. **Check for security advisories:**
   ```bash
   cargo audit
   ```

---

## ⚠️ Runtime Errors

### Issue: "No audio device found"
**Symptoms:**
```
Error: NoDevice
thread 'main' panicked at 'Failed to create audio output'
```

**Solutions:**
1. **Check audio permissions (macOS):**
   - System Preferences → Security & Privacy → Microphone
   - Grant permission to Terminal/IDE

2. **Test audio device:**
   ```bash
   # macOS
   system_profiler SPAudioDataType
   
   # Linux
   aplay -l
   
   # Windows
   # Use Device Manager → Sound devices
   ```

3. **Use default audio config:**
   ```rust
   let config = AudioConfig::default();
   // Instead of specific device selection
   ```

### Issue: Memory allocation failures
**Symptoms:**
```
thread 'main' panicked at 'memory allocation failed'
```

**Solutions:**
1. **Reduce buffer sizes:**
   ```rust
   let config = Config {
       buffer_size: 1024,  // Reduce from 4096
       // ...
   };
   ```

2. **Enable memory optimization:**
   ```rust
   let config = Config {
       memory_optimization: true,
       cache_size: 100,  // Reduce cache
       // ...
   };
   ```

3. **Monitor memory usage:**
   ```rust
   use voirs::monitoring::MemoryMonitor;
   let monitor = MemoryMonitor::new();
   monitor.track_usage();
   ```

### Issue: Model loading failures
**Symptoms:**
```
Error: Failed to load model: No such file or directory
```

**Solutions:**
1. **Check model paths:**
   ```rust
   let model_path = std::path::Path::new("models/acoustic_model.bin");
   if !model_path.exists() {
       eprintln!("Model file not found: {:?}", model_path);
   }
   ```

2. **Use relative paths:**
   ```rust
   let config = ModelConfig {
       model_path: "./models/".to_string(),
       // ...
   };
   ```

3. **Download models if missing:**
   ```bash
   # Check the models/ directory in repository root
   git submodule update --init --recursive
   ```

---

## 🔊 Audio Issues

### Issue: No audio output
**Symptoms:**
- Code runs without errors but no sound

**Solutions:**
1. **Check audio format:**
   ```rust
   let audio_config = AudioConfig {
       sample_rate: 22050,  // Try standard rates
       channels: 1,         // Use mono first
       bit_depth: 16,       // Use 16-bit
   };
   ```

2. **Test with simple synthesis:**
   ```rust
   use voirs::examples::hello_world;
   hello_world::run()?;  // Should produce audio
   ```

3. **Check system volume and audio routing**

### Issue: Poor audio quality
**Symptoms:**
- Robotic or distorted voice
- Artifacts and noise

**Solutions:**
1. **Increase quality settings:**
   ```rust
   let config = SynthesisConfig {
       quality: QualityLevel::High,
       sample_rate: 44100,
       // ...
   };
   ```

2. **Enable quality optimizations:**
   ```rust
   let config = Config {
       enable_quality_enhancement: true,
       noise_reduction: true,
       // ...
   };
   ```

3. **Check input text formatting:**
   ```rust
   let text = "Hello, world!";  // Use proper punctuation
   // Avoid ALL CAPS or excessive special characters
   ```

### Issue: Slow audio processing
**Symptoms:**
- Long delays before audio output
- Real-time processing too slow

**Solutions:**
1. **Enable GPU acceleration:**
   ```rust
   let config = Config {
       use_gpu: true,
       gpu_device_id: 0,
       // ...
   };
   ```

2. **Optimize for speed:**
   ```rust
   let config = Config {
       quality: QualityLevel::Fast,
       parallel_processing: true,
       // ...
   };
   ```

3. **Use streaming mode:**
   ```rust
   use voirs::streaming::StreamingSynthesizer;
   let synthesizer = StreamingSynthesizer::new(config)?;
   ```

---

## 🚀 Performance Problems

### Issue: High memory usage
**Symptoms:**
- System becomes slow
- Out of memory errors

**Solutions:**
1. **Enable memory optimization:**
   ```rust
   let config = Config {
       memory_optimization_level: OptimizationLevel::Aggressive,
       max_cache_size: 50,  // MB
       // ...
   };
   ```

2. **Use smaller models:**
   ```rust
   let config = ModelConfig {
       model_size: ModelSize::Small,
       quantization: true,
       // ...
   };
   ```

3. **Monitor memory usage:**
   ```rust
   use voirs::profiling::MemoryProfiler;
   let profiler = MemoryProfiler::new();
   profiler.start_monitoring();
   ```

### Issue: CPU usage too high
**Symptoms:**
- System fan running constantly
- Other applications slow

**Solutions:**
1. **Limit CPU threads:**
   ```rust
   let config = Config {
       max_threads: 2,  // Reduce from automatic detection
       // ...
   };
   ```

2. **Use power-efficient mode:**
   ```rust
   let config = Config {
       power_mode: PowerMode::Efficient,
       // ...
   };
   ```

### Issue: Real-time processing drops
**Symptoms:**
- Audio stuttering
- Processing cannot keep up

**Solutions:**
1. **Increase buffer size:**
   ```rust
   let config = Config {
       buffer_size: 2048,  // Increase from 1024
       latency_target: LatencyTarget::Balanced,
       // ...
   };
   ```

2. **Enable priority mode:**
   ```rust
   let config = Config {
       realtime_priority: true,
       dedicated_thread: true,
       // ...
   };
   ```

---

## 💻 Platform-Specific Issues

### macOS Issues

**Audio permissions:**
```bash
# Reset audio permissions
tccutil reset Microphone
# Then restart your application
```

**Code signing for distribution:**
```bash
# For development
codesign --force --deep --sign - target/release/your_app
```

**Metal GPU issues:**
```rust
let config = GpuConfig {
    prefer_metal: true,
    fallback_to_cpu: true,
    // ...
};
```

### Linux Issues

**ALSA configuration:**
```bash
# Install ALSA development headers
sudo apt-get install libasound2-dev

# Test audio
speaker-test -t wav
```

**PulseAudio issues:**
```bash
# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

**Permission issues:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Then log out and back in
```

### Windows Issues

**WASAPI configuration:**
```rust
let config = AudioConfig {
    api: AudioApi::Wasapi,
    exclusive_mode: false,
    // ...
};
```

**Windows Defender:**
- Add exception for VoiRS build directory
- Disable real-time protection during development

**Visual C++ Redistributables:**
```powershell
# Install latest Visual C++ Redistributable
# Download from Microsoft website
```

---

## 🔍 Debugging Tools

### Enable debug logging
```rust
env_logger::init();
log::set_max_level(log::LevelFilter::Debug);
```

### Performance profiling
```rust
use voirs::profiling::Profiler;
let profiler = Profiler::new();
profiler.start();
// ... your code ...
let report = profiler.stop();
println!("{}", report);
```

### Memory debugging
```rust
use voirs::debug::MemoryDebugger;
let debugger = MemoryDebugger::new();
debugger.enable_leak_detection();
```

### Quality analysis
```rust
use voirs::quality::QualityAnalyzer;
let analyzer = QualityAnalyzer::new();
let metrics = analyzer.analyze_synthesis(&audio_data)?;
println!("Quality score: {:.2}", metrics.overall_score);
```

---

## 📞 Getting Help

### Before asking for help:
1. Check this troubleshooting guide
2. Search through [FAQ Examples](faq_examples.rs)
3. Review [Best Practices](best_practices_guide.rs)
4. Test with minimal examples ([Hello World](hello_world.rs))

### When reporting issues:
1. **Include system information:**
   ```bash
   rustc --version
   cargo --version
   uname -a  # Linux/macOS
   # or systeminfo  # Windows
   ```

2. **Provide error logs:**
   ```bash
   RUST_LOG=debug cargo run 2>&1 | tee error.log
   ```

3. **Include minimal reproduction code:**
   ```rust
   // Smallest possible code that reproduces the issue
   ```

4. **Specify VoiRS version and features used**

### Community Resources
- [Community Contributions](community_contributions_gallery.rs)
- [Use Case Gallery](use_case_gallery.rs)
- [Example Testing Framework](examples_testing_framework.rs)

---

## 📊 System Requirements

### Minimum Requirements
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Rust**: 1.70+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Audio**: Any audio output device

### Recommended Requirements
- **RAM**: 16GB+ for large models
- **GPU**: CUDA-compatible for acceleration
- **SSD**: For faster model loading
- **CPU**: Multi-core for real-time processing

### Performance Expectations
| Configuration | Quality | Speed | Memory |
|---------------|---------|-------|---------|
| CPU Only | Good | 0.1-0.5x RTF | 2-4GB |
| GPU Accelerated | High | 1-5x RTF | 4-8GB |
| Optimized | Medium | 1-2x RTF | 1-2GB |

*RTF = Real-Time Factor (1.0 = real-time)*

---

*Need more help? Check our [FAQ Examples](faq_examples.rs) or review the [Learning Paths](LEARNING_PATHS.md) for structured guidance.*