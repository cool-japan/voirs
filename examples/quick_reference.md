# VoiRS Quick Reference Guide

**Fast lookup for common VoiRS operations and configurations**

## 🚀 Quick Start

```rust
use voirs_sdk::VoirsSdk;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk = VoirsSdk::new().await?;
    let audio = sdk.synthesize("Hello, world!", &Default::default()).await?;
    std::fs::write("output.wav", audio.data)?;
    Ok(())
}
```

## 📦 Dependencies

```toml
[dependencies]
voirs-sdk = "0.1"
voirs-cloning = "0.1"
voirs-emotion = "0.1"
voirs-spatial = "0.1"
tokio = { version = "1.0", features = ["full"] }
```

## 🎛️ Configuration Cheat Sheet

### Basic Synthesis
```rust
use voirs_sdk::{SynthesisConfig, AudioFormat, VoiceId};

let config = SynthesisConfig::builder()
    .voice_id(VoiceId::Default)
    .sample_rate(22050)
    .format(AudioFormat::Wav)
    .quality(0.8)
    .build()?;
```

### Voice Cloning
```rust
use voirs_cloning::{VoiceCloner, CloningMethod};

let cloner = VoiceCloner::builder()
    .method(CloningMethod::FewShot)
    .reference_samples(reference_audio)
    .similarity_threshold(0.85)
    .build()?;
```

### Emotion Control
```rust
use voirs_emotion::{EmotionProcessor, Emotion, EmotionIntensity};

let processor = EmotionProcessor::builder()
    .emotion(Emotion::Happy)
    .intensity(EmotionIntensity::MEDIUM)
    .enable_natural_variation(true)
    .build()?;
```

### Streaming Synthesis
```rust
use voirs_sdk::{StreamingConfig, StreamingSynthesizer};

let synthesizer = StreamingSynthesizer::builder()
    .chunk_size(1024)
    .buffer_size(4096)
    .latency_target_ms(50)
    .build()?;
```

## 🎭 Emotion Values

| Emotion | Description | Typical Use |
|---------|-------------|-------------|
| `Emotion::Neutral` | Default, balanced | General synthesis |
| `Emotion::Happy` | Upbeat, positive | Announcements, greetings |
| `Emotion::Sad` | Melancholic, low energy | Somber content |
| `Emotion::Angry` | Intense, aggressive | Dramatic readings |
| `Emotion::Excited` | High energy, enthusiastic | Sports, celebrations |
| `Emotion::Calm` | Peaceful, relaxed | Meditation, instructions |
| `Emotion::Confident` | Assertive, strong | Presentations, commands |

## 🔊 Audio Formats

| Format | Quality | Size | Use Case |
|--------|---------|------|----------|
| `AudioFormat::Wav` | Highest | Large | Development, high quality |
| `AudioFormat::Mp3` | Good | Medium | General distribution |
| `AudioFormat::Opus` | Excellent | Small | Streaming, real-time |
| `AudioFormat::Flac` | Lossless | Large | Archival, professional |

## 🎯 Quality vs Performance

### High Quality (Slow)
```rust
SynthesisConfig::builder()
    .quality(1.0)
    .sample_rate(44100)
    .use_gpu(true)
    .enable_post_processing(true)
```

### Balanced (Recommended)
```rust
SynthesisConfig::builder()
    .quality(0.8)
    .sample_rate(22050)
    .use_gpu(true)
```

### Fast (Real-time)
```rust
SynthesisConfig::builder()
    .quality(0.6)
    .sample_rate(16000)
    .use_gpu(false)
    .enable_streaming(true)
```

## 🛠️ Common Patterns

### Error Handling
```rust
match sdk.synthesize(text, &config).await {
    Ok(audio) => println!("Success: {} seconds", audio.duration_seconds()),
    Err(e) => eprintln!("Synthesis failed: {}", e),
}
```

### Batch Processing
```rust
let texts = vec!["First sentence", "Second sentence"];
let mut outputs = Vec::new();

for text in texts {
    let audio = sdk.synthesize(text, &config).await?;
    outputs.push(audio);
}
```

### Custom Voice with Emotion
```rust
let cloned_voice = cloner.clone_voice(&reference_samples).await?;
let emotional_config = SynthesisConfig::builder()
    .voice(cloned_voice)
    .emotion(Emotion::Happy)
    .emotion_intensity(0.7)
    .build()?;
```

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `VoiceNotFound` | Check voice ID, ensure models are downloaded |
| `AudioError` | Verify sample rate, format compatibility |
| `OutOfMemory` | Reduce quality, enable streaming, use smaller batches |
| `GpuError` | Disable GPU acceleration or update drivers |
| `TimeoutError` | Increase timeout, reduce text length |

### Performance Optimization
```rust
// Enable GPU acceleration
.use_gpu(true)

// Use appropriate chunk sizes
.chunk_size(1024)  // For real-time
.chunk_size(4096)  // For quality

// Optimize sample rate
.sample_rate(16000)  // Fast
.sample_rate(22050)  // Balanced
.sample_rate(44100)  // High quality
```

## 📊 Monitoring & Metrics

### Performance Tracking
```rust
use std::time::Instant;

let start = Instant::now();
let audio = sdk.synthesize(text, &config).await?;
let duration = start.elapsed();

println!("Synthesis took: {:?}", duration);
println!("Real-time factor: {:.2}x", 
    audio.duration_seconds() / duration.as_secs_f64());
```

### Quality Assessment
```rust
use voirs_evaluation::QualityAssessor;

let assessor = QualityAssessor::new();
let metrics = assessor.assess(&audio, &reference).await?;

println!("MOS Score: {:.2}", metrics.mos_score);
println!("Similarity: {:.2}%", metrics.similarity * 100.0);
```

## 🌐 Platform-Specific Notes

### Windows
- Use `AudioFormat::Wav` for best compatibility
- GPU acceleration requires CUDA drivers
- Use `\\` path separators or raw strings

### macOS  
- Core Audio integration available
- Metal GPU acceleration supported
- Use `.app` bundles for distribution

### Linux
- ALSA/PulseAudio supported
- CUDA and OpenCL available
- Consider system audio permissions

### Web/WASM
- Limited to certain audio formats
- No GPU acceleration
- Use streaming for better performance

## 🔗 Useful Examples

| Example | Description | File |
|---------|-------------|------|
| Hello World | Basic synthesis | `hello_world.rs` |
| Voice Cloning | Clone a voice | `voice_cloning_example.rs` |
| Emotions | Add emotions | `emotion_control_example.rs` |
| Streaming | Real-time synthesis | `streaming_synthesis.rs` |
| Batch Processing | Multiple files | `batch_synthesis.rs` |
| Quality Testing | A/B testing | `ab_testing_quality_comparison.rs` |

## 📚 Further Reading

- [Complete Tutorial Series](./tutorials/README.md)
- [API Documentation](https://docs.rs/voirs-sdk)
- [Example Gallery](./README.md)
- [Performance Guide](./performance_optimization_techniques.rs)
- [Troubleshooting Guide](./debug_troubleshooting_example.rs)

---

**💡 Tip**: Start with `hello_world.rs` and work through the examples directory for hands-on learning!

**📞 Support**: Check the FAQ examples or community contributions for additional help.