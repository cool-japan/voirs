# voirs-sdk

[![Crates.io](https://img.shields.io/crates/v/voirs.svg)](https://crates.io/crates/voirs)
[![Documentation](https://docs.rs/voirs/badge.svg)](https://docs.rs/voirs)

**Unified SDK and public API for VoiRS speech synthesis framework.**

The VoiRS SDK provides a comprehensive, easy-to-use interface for integrating high-quality text-to-speech synthesis into your Rust applications. It unifies all VoiRS components under a single, well-designed API with powerful builder patterns, async support, and extensive customization options.

## Features

- **🎯 Unified API**: Single entry point for all VoiRS functionality
- **🔧 Builder Pattern**: Fluent, discoverable API for configuration
- **⚡ Async/Await**: Non-blocking operations with full async support
- **🎭 Voice Control**: Easy voice management and customization
- **📱 Streaming**: Real-time synthesis with low latency
- **🔌 Extensible**: Plugin system for custom components
- **💾 Caching**: Intelligent model and result caching
- **🛡️ Error Handling**: Comprehensive error types with context

## Quick Start

Add VoiRS to your `Cargo.toml`:

```toml
[dependencies]
voirs = "0.1"

# Or with specific features
voirs = { version = "0.1", features = ["gpu", "streaming", "ssml"] }
```

### Basic Usage

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    // Create a simple pipeline
    let pipeline = VoirsPipeline::builder()
        .build()
        .await?;
    
    // Synthesize text to audio
    let audio = pipeline
        .synthesize("Hello, world! Welcome to VoiRS.")
        .await?;
    
    // Save to file
    audio.save_wav("hello.wav")?;
    
    println!("Synthesis complete! Duration: {:.2}s", audio.duration());
    Ok(())
}
```

### Advanced Configuration

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .with_quality(Quality::Ultra)
        .with_gpu_acceleration(true)
        .with_streaming(StreamConfig {
            chunk_size: 256,
            overlap: 64,
            max_latency_ms: 50,
        })
        .with_enhancement(Enhancement {
            noise_reduction: true,
            dynamic_range_compression: true,
            high_frequency_boost: 0.1,
        })
        .build()
        .await?;
    
    let audio = pipeline
        .synthesize("This is high-quality synthesis with GPU acceleration.")
        .await?;
    
    audio.save("output.flac", AudioFormat::Flac)?;
    Ok(())
}
```

## Core API

### VoirsPipeline

The main entry point for speech synthesis.

```rust
impl VoirsPipeline {
    /// Create a new pipeline builder
    pub fn builder() -> VoirsPipelineBuilder;
    
    /// Synthesize text to audio
    pub async fn synthesize(&self, text: &str) -> Result<AudioBuffer>;
    
    /// Synthesize with custom configuration
    pub async fn synthesize_with_config(
        &self, 
        text: &str, 
        config: &SynthesisConfig
    ) -> Result<AudioBuffer>;
    
    /// Synthesize SSML markup
    pub async fn synthesize_ssml(&self, ssml: &str) -> Result<AudioBuffer>;
    
    /// Stream synthesis for long texts
    pub async fn synthesize_stream(
        &self,
        text: &str
    ) -> Result<impl Stream<Item = Result<AudioBuffer>>>;
    
    /// Change voice during runtime
    pub async fn set_voice(&mut self, voice: &str) -> Result<()>;
    
    /// Get current voice information
    pub fn current_voice(&self) -> &VoiceInfo;
    
    /// List available voices
    pub async fn list_voices(&self) -> Result<Vec<VoiceInfo>>;
}
```

### VoirsPipelineBuilder

Fluent builder for configuring the synthesis pipeline.

```rust
impl VoirsPipelineBuilder {
    /// Set the voice to use
    pub fn with_voice(self, voice: impl Into<String>) -> Self;
    
    /// Set synthesis quality
    pub fn with_quality(self, quality: Quality) -> Self;
    
    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(self, enabled: bool) -> Self;
    
    /// Configure streaming synthesis
    pub fn with_streaming(self, config: StreamConfig) -> Self;
    
    /// Enable audio enhancement
    pub fn with_enhancement(self, enhancement: Enhancement) -> Self;
    
    /// Set device for computation
    pub fn with_device(self, device: Device) -> Self;
    
    /// Set custom cache directory
    pub fn with_cache_dir(self, path: impl AsRef<Path>) -> Self;
    
    /// Add custom plugin
    pub fn with_plugin<P: Plugin>(self, plugin: P) -> Self;
    
    /// Build the pipeline
    pub async fn build(self) -> Result<VoirsPipeline>;
}
```

### AudioBuffer

Represents synthesized audio with rich metadata and processing capabilities.

```rust
impl AudioBuffer {
    /// Get audio samples as slice
    pub fn samples(&self) -> &[f32];
    
    /// Get sample rate in Hz
    pub fn sample_rate(&self) -> u32;
    
    /// Get number of channels
    pub fn channels(&self) -> u32;
    
    /// Get duration in seconds
    pub fn duration(&self) -> f32;
    
    /// Convert to different sample rate
    pub fn resample(&self, target_rate: u32) -> Result<AudioBuffer>;
    
    /// Apply audio effects
    pub fn with_effects(&self, effects: &[AudioEffect]) -> Result<AudioBuffer>;
    
    /// Save as WAV file
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<()>;
    
    /// Save in specified format
    pub fn save(&self, path: impl AsRef<Path>, format: AudioFormat) -> Result<()>;
    
    /// Play audio through system speakers
    pub fn play(&self) -> Result<()>;
    
    /// Get audio metadata
    pub fn metadata(&self) -> &AudioMetadata;
}
```

## Configuration Types

### Quality Levels

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Fast synthesis with acceptable quality
    Low,
    /// Balanced speed and quality (default)
    Medium,
    /// High quality synthesis
    High,
    /// Maximum quality (slowest)
    Ultra,
}
```

### Device Selection

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// Automatically select best device
    Auto,
    /// Use CPU only
    Cpu,
    /// Use specific GPU
    Gpu(u32),
    /// Use Metal (macOS)
    Metal,
}
```

### Streaming Configuration

```rust
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Size of each audio chunk in frames
    pub chunk_size: usize,
    /// Overlap between chunks for smooth concatenation
    pub overlap: usize,
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: u32,
    /// Buffer size for audio output
    pub buffer_size: usize,
}
```

### Audio Enhancement

```rust
#[derive(Debug, Clone)]
pub struct Enhancement {
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// Enable dynamic range compression
    pub dynamic_range_compression: bool,
    /// High frequency boost (0.0 - 1.0)
    pub high_frequency_boost: f32,
    /// Warmth enhancement for natural sound
    pub warmth: f32,
    /// Stereo widening effect
    pub stereo_width: f32,
}
```

## Advanced Usage Examples

### Voice Management

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let mut pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .build()
        .await?;
    
    // List available voices
    let voices = pipeline.list_voices().await?;
    for voice in voices {
        println!("Voice: {} ({})", voice.name, voice.language);
        println!("  Quality: {:?}", voice.quality);
        println!("  Features: {:?}", voice.features);
    }
    
    // Switch voice dynamically
    pipeline.set_voice("en-GB-male-formal").await?;
    
    let audio = pipeline
        .synthesize("Hello from Britain!")
        .await?;
    
    audio.save_wav("british.wav")?;
    Ok(())
}
```

### Streaming Synthesis

```rust
use voirs::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_streaming(StreamConfig {
            chunk_size: 256,
            overlap: 64,
            max_latency_ms: 50,
            buffer_size: 1024,
        })
        .build()
        .await?;
    
    let long_text = "This is a very long text that will be synthesized in real-time chunks...";
    
    let mut stream = pipeline.synthesize_stream(long_text).await?;
    let mut chunk_count = 0;
    
    while let Some(audio_chunk) = stream.next().await {
        let chunk = audio_chunk?;
        chunk_count += 1;
        
        println!("Chunk {}: {:.3}s", chunk_count, chunk.duration());
        
        // Play each chunk immediately for real-time audio
        chunk.play()?;
    }
    
    println!("Streaming complete! {} chunks processed", chunk_count);
    Ok(())
}
```

### SSML Processing

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .with_quality(Quality::High)
        .build()
        .await?;
    
    let ssml = r#"
        <speak>
            <p>Welcome to <emphasis level="strong">VoiRS</emphasis>!</p>
            
            <break time="1s"/>
            
            <p>This framework supports:</p>
            <ul>
                <li><prosody rate="slow">Slow speech</prosody></li>
                <li><prosody rate="fast">Fast speech</prosody></li>
                <li><prosody pitch="high">High pitch</prosody></li>
                <li><prosody pitch="low">Low pitch</prosody></li>
            </ul>
            
            <p>You can even specify pronunciations:
            <phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>
            or <phoneme alphabet="ipa" ph="təˈmɑːtoʊ">tomato</phoneme>.</p>
        </speak>
    "#;
    
    let audio = pipeline.synthesize_ssml(ssml).await?;
    audio.save_wav("ssml_demo.wav")?;
    
    Ok(())
}
```

### Batch Processing

```rust
use voirs::prelude::*;
use futures::future::join_all;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .with_quality(Quality::High)
        .build()
        .await?;
    
    let texts = vec![
        "This is the first sentence.",
        "Here's the second sentence.",
        "And finally, the third sentence.",
    ];
    
    // Process all texts concurrently
    let synthesis_tasks: Vec<_> = texts
        .iter()
        .map(|text| pipeline.synthesize(text))
        .collect();
    
    let audio_results = join_all(synthesis_tasks).await;
    
    // Save all results
    for (i, result) in audio_results.into_iter().enumerate() {
        let audio = result?;
        audio.save_wav(&format!("batch_{:02}.wav", i + 1))?;
    }
    
    println!("Batch processing complete!");
    Ok(())
}
```

### Custom Synthesis Configuration

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .build()
        .await?;
    
    let config = SynthesisConfig {
        speaking_rate: 1.2,          // 20% faster
        pitch_shift: 0.5,            // Half semitone higher
        volume_gain: 2.0,            // 2dB louder
        emphasis: Some(Emphasis {
            words: vec!["important".to_string(), "critical".to_string()],
            level: EmphasisLevel::Strong,
        }),
        prosody: Some(Prosody {
            rhythm: Rhythm::Natural,
            intonation: Intonation::Expressive,
        }),
        effects: vec![
            AudioEffect::Reverb {
                room_size: 0.3,
                damping: 0.5,
                wet_level: 0.1,
            },
            AudioEffect::Eq {
                low_gain: 0.0,
                mid_gain: 1.0,
                high_gain: 0.5,
            },
        ],
    };
    
    let audio = pipeline
        .synthesize_with_config(
            "This is an important and critical announcement!",
            &config
        )
        .await?;
    
    audio.save_wav("enhanced.wav")?;
    Ok(())
}
```

### Error Handling

```rust
use voirs::prelude::*;

#[tokio::main]
async fn main() {
    match create_and_synthesize().await {
        Ok(()) => println!("Synthesis completed successfully!"),
        Err(e) => handle_voirs_error(e),
    }
}

async fn create_and_synthesize() -> Result<(), VoirsError> {
    let pipeline = VoirsPipeline::builder()
        .with_voice("nonexistent-voice")  // This will cause an error
        .build()
        .await?;
    
    let audio = pipeline.synthesize("Hello").await?;
    audio.save_wav("output.wav")?;
    
    Ok(())
}

fn handle_voirs_error(error: VoirsError) {
    match error {
        VoirsError::VoiceNotFound { voice, available } => {
            eprintln!("Voice '{}' not found.", voice);
            eprintln!("Available voices:");
            for v in available {
                eprintln!("  - {}", v);
            }
        }
        VoirsError::SynthesisFailed { text, cause } => {
            eprintln!("Failed to synthesize: '{}'", text);
            eprintln!("Cause: {}", cause);
        }
        VoirsError::IoError { path, source } => {
            eprintln!("File I/O error at '{}': {}", path.display(), source);
        }
        VoirsError::DeviceError { device, message } => {
            eprintln!("Device error ({:?}): {}", device, message);
        }
        _ => {
            eprintln!("Other VoiRS error: {}", error);
        }
    }
}
```

### Plugin System

```rust
use voirs::prelude::*;

// Custom plugin for post-processing
#[derive(Debug)]
struct CustomEnhancer {
    gain: f32,
    filter_cutoff: f32,
}

impl Plugin for CustomEnhancer {
    fn name(&self) -> &str {
        "CustomEnhancer"
    }
    
    fn process(&self, audio: &mut AudioBuffer) -> Result<(), PluginError> {
        // Apply custom enhancement
        let samples = audio.samples_mut();
        for sample in samples {
            *sample *= self.gain;
            // Apply custom filtering...
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let enhancer = CustomEnhancer {
        gain: 1.2,
        filter_cutoff: 8000.0,
    };
    
    let pipeline = VoirsPipeline::builder()
        .with_voice("en-US-female-calm")
        .with_plugin(enhancer)
        .build()
        .await?;
    
    let audio = pipeline.synthesize("Hello with custom enhancement!").await?;
    audio.save_wav("enhanced.wav")?;
    
    Ok(())
}
```

## Feature Flags

The VoiRS SDK supports various feature flags for customizing functionality:

```toml
[dependencies]
voirs = { version = "0.1", features = ["full"] }

# Or select specific features:
voirs = { 
    version = "0.1", 
    features = [
        "gpu",           # GPU acceleration
        "streaming",     # Real-time streaming
        "ssml",          # SSML markup support
        "effects",       # Audio effects processing
        "ml",            # Advanced ML features
        "networking",    # Network-based features
    ] 
}
```

### Available Features

- **`gpu`**: Enable GPU acceleration (CUDA, Metal, OpenCL)
- **`streaming`**: Real-time streaming synthesis
- **`ssml`**: SSML markup processing
- **`effects`**: Audio effects and post-processing
- **`ml`**: Advanced ML features (voice cloning, etc.)
- **`networking`**: Network-based model loading and APIs
- **`python`**: Python integration (re-exports from voirs-ffi)
- **`full`**: Enable all features

## Performance

### Benchmarks

| Configuration | RTF (CPU) | RTF (GPU) | Memory | Quality (MOS) |
|---------------|-----------|-----------|--------|---------------|
| Quality::Low | 0.15× | 0.02× | 256MB | 3.8 |
| Quality::Medium | 0.28× | 0.04× | 384MB | 4.2 |
| Quality::High | 0.42× | 0.06× | 512MB | 4.4 |
| Quality::Ultra | 0.68× | 0.12× | 768MB | 4.6 |

*Benchmarks on Intel i7-12700K + RTX 4080, 22kHz synthesis*

### Performance Tips

```rust
// For maximum performance
let pipeline = VoirsPipeline::builder()
    .with_quality(Quality::Medium)    // Balance quality vs speed
    .with_gpu_acceleration(true)      // Use GPU if available
    .with_streaming(StreamConfig::fast()) // Optimize for speed
    .build()
    .await?;

// For maximum quality
let pipeline = VoirsPipeline::builder()
    .with_quality(Quality::Ultra)
    .with_enhancement(Enhancement::studio()) // High-quality enhancement
    .build()
    .await?;

// For batch processing
let pipeline = VoirsPipeline::builder()
    .with_quality(Quality::High)
    .with_cache_dir("/fast/ssd/cache")   // Use fast storage
    .build()
    .await?;

// Reuse pipeline instances to avoid initialization overhead
```

## Integration Examples

### Web Framework Integration (Axum)

```rust
use axum::{extract::Query, http::StatusCode, response::Json, routing::post, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use voirs::prelude::*;

#[derive(Deserialize)]
struct SynthesisRequest {
    text: String,
    voice: Option<String>,
    quality: Option<Quality>,
}

#[derive(Serialize)]
struct SynthesisResponse {
    audio_url: String,
    duration: f32,
    sample_rate: u32,
}

async fn synthesize_handler(
    Query(req): Query<SynthesisRequest>,
    pipeline: Arc<VoirsPipeline>,
) -> Result<Json<SynthesisResponse>, StatusCode> {
    // Configure synthesis
    let mut synthesis_config = SynthesisConfig::default();
    if let Some(quality) = req.quality {
        synthesis_config.quality = quality;
    }
    
    // Synthesize audio
    let audio = pipeline
        .synthesize_with_config(&req.text, &synthesis_config)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Save to temporary file
    let filename = format!("synthesis_{}.wav", uuid::Uuid::new_v4());
    let path = format!("/tmp/{}", filename);
    audio.save_wav(&path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(SynthesisResponse {
        audio_url: format!("/audio/{}", filename),
        duration: audio.duration(),
        sample_rate: audio.sample_rate(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = Arc::new(
        VoirsPipeline::builder()
            .with_voice("en-US-female-calm")
            .with_quality(Quality::High)
            .build()
            .await?
    );
    
    let app = Router::new()
        .route("/synthesize", post(synthesize_handler))
        .with_state(pipeline);
    
    println!("Server running on http://localhost:3000");
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}
```

### Game Engine Integration

```rust
use voirs::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct GameNarrator {
    pipeline: Arc<VoirsPipeline>,
    current_voice: String,
}

impl GameNarrator {
    pub async fn new() -> Result<Self, VoirsError> {
        let pipeline = VoirsPipeline::builder()
            .with_voice("en-US-male-narrator")
            .with_quality(Quality::High)
            .with_streaming(StreamConfig::gaming())
            .build()
            .await?;
        
        Ok(Self {
            pipeline: Arc::new(pipeline),
            current_voice: "en-US-male-narrator".to_string(),
        })
    }
    
    pub async fn narrate(&self, text: &str) -> Result<AudioBuffer, VoirsError> {
        self.pipeline.synthesize(text).await
    }
    
    pub async fn character_speak(
        &mut self,
        character: &str,
        text: &str
    ) -> Result<AudioBuffer, VoirsError> {
        let voice = self.get_character_voice(character);
        
        if voice != self.current_voice {
            self.pipeline.set_voice(&voice).await?;
            self.current_voice = voice;
        }
        
        self.pipeline.synthesize(text).await
    }
    
    fn get_character_voice(&self, character: &str) -> String {
        match character {
            "wizard" => "en-US-male-wise",
            "princess" => "en-US-female-elegant",
            "warrior" => "en-US-male-strong",
            "narrator" => "en-US-male-narrator",
            _ => "en-US-neutral",
        }.to_string()
    }
}

// Usage in game loop
async fn game_dialogue_example() -> Result<(), VoirsError> {
    let mut narrator = GameNarrator::new().await?;
    
    // Narration
    let narration = narrator
        .narrate("The hero enters the ancient castle...")
        .await?;
    narration.play()?;
    
    // Character dialogue
    let wizard_speech = narrator
        .character_speak("wizard", "Welcome, brave adventurer!")
        .await?;
    wizard_speech.play()?;
    
    let princess_speech = narrator
        .character_speak("princess", "Please save our kingdom!")
        .await?;
    princess_speech.play()?;
    
    Ok(())
}
```

## Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum VoirsError {
    #[error("Voice '{voice}' not found")]
    VoiceNotFound {
        voice: String,
        available: Vec<String>,
    },
    
    #[error("Synthesis failed for text: '{text}'")]
    SynthesisFailed {
        text: String,
        #[source]
        cause: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Device error ({device:?}): {message}")]
    DeviceError {
        device: Device,
        message: String,
    },
    
    #[error("I/O error at path '{path}'")]
    IoError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    
    #[error("Configuration error: {message}")]
    ConfigError {
        message: String,
    },
    
    #[error("Plugin error in '{plugin}': {message}")]
    PluginError {
        plugin: String,
        message: String,
    },
    
    #[error("Network error: {message}")]
    NetworkError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
```

## Contributing

We welcome contributions! Please see the [main repository](https://github.com/cool-japan/voirs) for contribution guidelines.

### Development Setup

```bash
git clone https://github.com/cool-japan/voirs.git
cd voirs

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run examples
cargo run --example simple_synthesis
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.