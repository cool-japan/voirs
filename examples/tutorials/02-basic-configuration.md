# Tutorial 2: Basic Configuration

**Duration**: 20-25 minutes  
**Level**: Beginner  
**Prerequisites**: Tutorial 1 completed

## Overview

In this tutorial, you'll learn how to configure VoiRS for different use cases. Configuration is crucial for getting the best performance and quality from VoiRS in your specific applications.

## What You'll Learn

- Understanding VoiRS configuration options
- Optimizing for different use cases (quality vs speed)
- Voice selection and customization
- Audio format selection
- Error handling best practices

## Configuration Fundamentals

### Basic Configuration Structure

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig, AudioFormat, VoiceId};

let config = SynthesisConfig::builder()
    .voice_id(VoiceId::Default)
    .sample_rate(22050)
    .format(AudioFormat::Wav)
    .quality(0.8)
    .build()?;
```

### Key Configuration Parameters

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `voice_id` | Voice model to use | `Default`, `Female`, `Male` |
| `sample_rate` | Audio quality in Hz | 16000, 22050, 44100 |
| `format` | Output audio format | `Wav`, `Mp3`, `Opus` |
| `quality` | Synthesis quality (0.0-1.0) | 0.6 (fast), 0.8 (balanced), 1.0 (best) |

## Use Case Configurations

### High Quality Configuration
For applications where audio quality is paramount:

```rust
use voirs_sdk::{SynthesisConfig, AudioFormat, QualityLevel};

let high_quality_config = SynthesisConfig::builder()
    .voice_id("premium-voice")
    .sample_rate(44100)
    .format(AudioFormat::Wav)
    .quality(1.0)
    .enable_post_processing(true)
    .use_gpu(true)
    .build()?;

// Example usage
let text = "This synthesis uses the highest quality settings.";
let audio = sdk.synthesize(text, &high_quality_config).await?;
```

### Real-time Configuration
For real-time applications like chatbots or voice assistants:

```rust
let realtime_config = SynthesisConfig::builder()
    .voice_id("fast-voice")
    .sample_rate(16000)
    .format(AudioFormat::Opus)
    .quality(0.6)
    .enable_streaming(true)
    .latency_target_ms(100)
    .build()?;

// Enable streaming for real-time processing
let streaming_synthesizer = sdk.create_streaming_synthesizer(&realtime_config).await?;
```

### Balanced Configuration
Good balance between quality and performance:

```rust
let balanced_config = SynthesisConfig::builder()
    .voice_id("default")
    .sample_rate(22050)
    .format(AudioFormat::Mp3)
    .quality(0.8)
    .compression_level(5)
    .build()?;
```

## Voice Selection

### Available Voice Types

```rust
use voirs_sdk::VoiceId;

// Built-in voices
let voices = vec![
    VoiceId::Default,           // Generic, balanced voice
    VoiceId::Female,            // Female voice model
    VoiceId::Male,              // Male voice model
    VoiceId::Child,             // Child-like voice
    VoiceId::Elderly,           // Elderly voice characteristics
    VoiceId::Custom("my-voice") // Custom trained voice
];
```

### Voice Characteristics

```rust
use voirs_sdk::{VoiceCharacteristics, Age, Gender, Accent};

let voice_config = VoiceCharacteristics::builder()
    .age(Age::Adult)
    .gender(Gender::Female)
    .accent(Accent::American)
    .pitch_range(0.8, 1.2)
    .speaking_rate(1.0)
    .build()?;

let config = SynthesisConfig::builder()
    .voice_characteristics(voice_config)
    .build()?;
```

## Audio Format Deep Dive

### Format Comparison

```rust
use voirs_sdk::AudioFormat;

// Different formats for different needs
let formats = vec![
    // Highest quality, largest files
    AudioFormat::Wav,
    
    // Good compression, wide compatibility  
    AudioFormat::Mp3 { bitrate: 128 },
    
    // Best for streaming, small files
    AudioFormat::Opus { bitrate: 64 },
    
    // Lossless compression
    AudioFormat::Flac
];
```

### Format Selection Guide

```rust
fn choose_format(use_case: &str) -> AudioFormat {
    match use_case {
        "development" | "testing" => AudioFormat::Wav,
        "mobile_app" => AudioFormat::Opus { bitrate: 64 },
        "web_streaming" => AudioFormat::Mp3 { bitrate: 128 },
        "archival" => AudioFormat::Flac,
        _ => AudioFormat::Mp3 { bitrate: 128 }
    }
}

let config = SynthesisConfig::builder()
    .format(choose_format("mobile_app"))
    .build()?;
```

## Performance Configuration

### GPU Acceleration

```rust
// Check GPU availability
if sdk.is_gpu_available() {
    let gpu_config = SynthesisConfig::builder()
        .use_gpu(true)
        .gpu_memory_limit_mb(2048)
        .build()?;
} else {
    println!("GPU not available, using CPU");
}
```

### Memory Management

```rust
let memory_optimized_config = SynthesisConfig::builder()
    .max_memory_usage_mb(512)
    .enable_memory_pooling(true)
    .chunk_size(1024)
    .build()?;
```

## Error Handling

### Robust Configuration with Validation

```rust
use voirs_sdk::{SynthesisConfig, VoirsError};

fn create_validated_config() -> Result<SynthesisConfig, VoirsError> {
    let config = SynthesisConfig::builder()
        .voice_id("default")
        .sample_rate(22050)
        .format(AudioFormat::Wav)
        .quality(0.8);
    
    // Validate configuration before building
    config.validate()?;
    config.build()
}

// Usage with proper error handling
match create_validated_config() {
    Ok(config) => {
        let audio = sdk.synthesize("Hello world", &config).await?;
        println!("Synthesis successful!");
    }
    Err(VoirsError::InvalidVoice(voice)) => {
        eprintln!("Voice '{}' not available", voice);
    }
    Err(VoirsError::InvalidSampleRate(rate)) => {
        eprintln!("Sample rate {} not supported", rate);
    }
    Err(e) => {
        eprintln!("Configuration error: {}", e);
    }
}
```

## Dynamic Configuration

### Adaptive Configuration

```rust
use voirs_sdk::SystemInfo;

async fn create_adaptive_config(sdk: &VoirsSdk) -> SynthesisConfig {
    let system_info = sdk.get_system_info().await;
    
    let quality = if system_info.available_memory_gb > 4.0 {
        1.0  // High quality for systems with plenty of RAM
    } else if system_info.available_memory_gb > 2.0 {
        0.8  // Balanced quality
    } else {
        0.6  // Lower quality for constrained systems
    };
    
    let sample_rate = if system_info.cpu_cores >= 8 {
        44100  // High sample rate for powerful CPUs
    } else {
        22050  // Standard rate for typical systems
    };
    
    SynthesisConfig::builder()
        .quality(quality)
        .sample_rate(sample_rate)
        .use_gpu(system_info.has_gpu)
        .build()
        .expect("Failed to create adaptive config")
}
```

## Configuration Presets

### Creating Reusable Presets

```rust
use voirs_sdk::ConfigPreset;

// Define presets for common scenarios
struct VoirsPresets;

impl VoirsPresets {
    pub fn podcast() -> SynthesisConfig {
        SynthesisConfig::builder()
            .voice_id("narrative-voice")
            .sample_rate(44100)
            .format(AudioFormat::Mp3 { bitrate: 192 })
            .quality(0.9)
            .enable_normalization(true)
            .build()
            .unwrap()
    }
    
    pub fn chatbot() -> SynthesisConfig {
        SynthesisConfig::builder()
            .voice_id("conversational")
            .sample_rate(16000)
            .format(AudioFormat::Opus { bitrate: 32 })
            .quality(0.7)
            .enable_streaming(true)
            .latency_target_ms(50)
            .build()
            .unwrap()
    }
    
    pub fn audiobook() -> SynthesisConfig {
        SynthesisConfig::builder()
            .voice_id("reader-voice")
            .sample_rate(22050)
            .format(AudioFormat::Mp3 { bitrate: 128 })
            .quality(0.85)
            .enable_chapter_markers(true)
            .build()
            .unwrap()
    }
}

// Usage
let config = VoirsPresets::podcast();
let audio = sdk.synthesize("Welcome to my podcast", &config).await?;
```

## Complete Example

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig, AudioFormat, VoiceId};
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    
    // Create different configurations for different content types
    let configs = vec![
        ("announcement", SynthesisConfig::builder()
            .voice_id(VoiceId::Female)
            .sample_rate(22050)
            .format(AudioFormat::Mp3 { bitrate: 128 })
            .quality(0.8)
            .speaking_rate(1.1)
            .build()?),
            
        ("narration", SynthesisConfig::builder()
            .voice_id(VoiceId::Male)
            .sample_rate(44100)
            .format(AudioFormat::Wav)
            .quality(1.0)
            .speaking_rate(0.9)
            .build()?),
    ];
    
    // Synthesize with different configurations
    for (name, config) in configs {
        let text = format!("This is a {} example using VoiRS.", name);
        
        match sdk.synthesize(&text, &config).await {
            Ok(audio) => {
                let filename = format!("{}_example.{}", name, 
                    config.format.extension());
                
                let mut file = File::create(&filename)?;
                file.write_all(&audio.data)?;
                
                println!("✅ Created {} ({:.2}s, {} bytes)", 
                    filename, audio.duration_seconds(), audio.data.len());
            }
            Err(e) => {
                eprintln!("❌ Failed to synthesize {}: {}", name, e);
            }
        }
    }
    
    Ok(())
}
```

## Best Practices

1. **Start with presets**: Use configuration presets for common scenarios
2. **Test configurations**: Always test with your specific content and hardware
3. **Monitor performance**: Use profiling to optimize for your use case
4. **Validate early**: Check configuration validity before synthesis
5. **Consider constraints**: Balance quality with available resources

## Common Pitfalls

- **Over-optimization**: Don't sacrifice quality unless necessary
- **Ignoring format**: Choose the right format for your distribution method
- **Fixed configuration**: Consider adaptive configuration for varying conditions
- **Missing validation**: Always validate configuration parameters

## Next Steps

In the next tutorial, we'll dive into actual voice synthesis and explore different techniques for generating natural-sounding speech.

Continue to [Tutorial 3: Simple Voice Synthesis](./03-simple-synthesis.md) →

## Additional Resources

- [Configuration Reference](../configuration_example.rs)
- [Performance Guide](../performance_optimization_techniques.rs)
- [Audio Format Documentation](../audio_quality_assessment.rs)

---

**Estimated completion time**: 20-25 minutes  
**Difficulty**: ⭐⭐☆☆☆  
**Next tutorial**: [Simple Voice Synthesis](./03-simple-synthesis.md)