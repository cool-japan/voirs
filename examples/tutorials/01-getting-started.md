# Tutorial 1: Getting Started with VoiRS

**Duration**: 15-20 minutes  
**Level**: Beginner  
**Prerequisites**: Basic Rust knowledge

## Overview

In this tutorial, you'll set up VoiRS and create your first voice synthesis program. By the end, you'll have a working text-to-speech system that can convert text into natural-sounding speech.

## What You'll Learn

- How to set up VoiRS in your Rust project
- Basic VoiRS concepts and architecture
- Creating your first speech synthesis program
- Understanding audio output and file formats

## Setup

### 1. Create a New Rust Project

```bash
cargo new my_first_voirs --bin
cd my_first_voirs
```

### 2. Add VoiRS Dependencies

Add these dependencies to your `Cargo.toml`:

```toml
[dependencies]
voirs-sdk = { version = "0.1", features = ["default"] }
voirs-cloning = "0.1"
voirs-emotion = "0.1"
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

### 3. Verify Installation

Create a simple test to verify everything works:

```rust
// src/main.rs
use voirs_sdk::VoirsSdk;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("VoiRS is ready!");
    Ok(())
}
```

Run it:
```bash
cargo run
```

You should see "VoiRS is ready!" printed.

## Your First Synthesis

Now let's create a simple text-to-speech program:

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig, AudioFormat};
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the VoiRS SDK
    let sdk = VoirsSdk::new().await?;
    
    // Configure synthesis settings
    let config = SynthesisConfig::builder()
        .voice_id("default")
        .sample_rate(22050)
        .format(AudioFormat::Wav)
        .build()?;
    
    // Text to synthesize
    let text = "Hello! This is my first VoiRS synthesis.";
    
    // Perform synthesis
    println!("Synthesizing: {}", text);
    let audio = sdk.synthesize(text, &config).await?;
    
    // Save to file
    let mut file = File::create("first_synthesis.wav")?;
    file.write_all(&audio.data)?;
    
    println!("Audio saved to 'first_synthesis.wav'");
    println!("Duration: {:.2} seconds", audio.duration_seconds());
    
    Ok(())
}
```

## Understanding the Code

Let's break down what this program does:

### 1. SDK Initialization
```rust
let sdk = VoirsSdk::new().await?;
```
This creates a new VoiRS SDK instance with default settings.

### 2. Configuration
```rust
let config = SynthesisConfig::builder()
    .voice_id("default")
    .sample_rate(22050)
    .format(AudioFormat::Wav)
    .build()?;
```
This configures:
- **voice_id**: Which voice model to use
- **sample_rate**: Audio quality (22050 Hz is CD quality)
- **format**: Output format (WAV is uncompressed, high quality)

### 3. Synthesis
```rust
let audio = sdk.synthesize(text, &config).await?;
```
This converts text to audio using the specified configuration.

### 4. File Output
```rust
let mut file = File::create("first_synthesis.wav")?;
file.write_all(&audio.data)?;
```
This saves the synthesized audio to a WAV file.

## Running Your Program

```bash
cargo run
```

You should see output like:
```
Synthesizing: Hello! This is my first VoiRS synthesis.
Audio saved to 'first_synthesis.wav'
Duration: 2.34 seconds
```

## Testing Your Audio

Play the generated audio file using your system's audio player:

**macOS**: `open first_synthesis.wav`  
**Linux**: `play first_synthesis.wav` or `aplay first_synthesis.wav`  
**Windows**: `start first_synthesis.wav`

## Common Issues and Solutions

### Issue: "Voice model not found"
**Solution**: Make sure you have the default voice models installed:
```bash
cargo run --features "download-models"
```

### Issue: "Permission denied" when saving files
**Solution**: Make sure you have write permissions in the current directory, or specify a different path:
```rust
let mut file = File::create("/tmp/first_synthesis.wav")?;
```

### Issue: Low audio quality
**Solution**: Increase the sample rate:
```rust
.sample_rate(44100)  // Higher quality
```

## Key Concepts

- **SDK**: The main interface for VoiRS functionality
- **Configuration**: Settings that control synthesis behavior
- **Async/Await**: VoiRS operations are asynchronous for better performance
- **Audio Format**: Different ways to encode audio (WAV, MP3, etc.)

## What's Next?

Congratulations! You've successfully created your first VoiRS program. In the next tutorial, we'll explore:

- Different configuration options
- Voice selection and customization
- Error handling best practices
- Performance considerations

Continue to [Tutorial 2: Basic Configuration](./02-basic-configuration.md) →

## Additional Resources

- [VoiRS API Documentation](https://docs.rs/voirs-sdk)
- [Audio Format Guide](../audio_quality_assessment.rs)
- [Basic Configuration Example](../basic_configuration.rs)
- [Hello World Example](../hello_world.rs)

## Practice Exercises

1. **Modify the text**: Change the text being synthesized to something personal
2. **Try different sample rates**: Experiment with 16000, 22050, and 44100 Hz
3. **Generate multiple files**: Create a loop that synthesizes different phrases
4. **Add error messages**: Improve error handling with descriptive messages

## Troubleshooting Checklist

- [ ] Rust and Cargo are properly installed
- [ ] All dependencies are added to Cargo.toml
- [ ] Project compiles without errors (`cargo check`)
- [ ] Audio file is created in the project directory
- [ ] Audio file can be played and sounds natural

---

**Estimated completion time**: 15-20 minutes  
**Difficulty**: ⭐☆☆☆☆  
**Next tutorial**: [Basic Configuration](./02-basic-configuration.md)