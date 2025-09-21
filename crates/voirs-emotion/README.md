# voirs-emotion

> **Emotion Expression Control System for Neural Speech Synthesis**

This crate provides comprehensive emotion expression control for voice synthesis, enabling dynamic emotional expression through prosody modification, acoustic parameter adjustment, and emotion interpolation.

## 🎭 Features

### Core Emotion Processing
- **Multi-dimensional Emotion Model** - Support for arousal, valence, and dominance dimensions
- **Emotion Interpolation** - Smooth blending between different emotional states
- **Real-time Processing** - Low-latency emotion modulation for live applications
- **Preset Library** - Pre-configured emotion presets (happy, sad, angry, calm, excited)

### Prosody Control
- **Pitch Modulation** - F0 contour modification based on emotion intensity
- **Timing Control** - Speaking rate adjustment for emotional expression
- **Energy Dynamics** - Volume and emphasis control
- **Stress Patterns** - Emotional stress placement and emphasis

### Advanced Features
- **SSML Integration** - Emotion markup support in Speech Synthesis Markup Language
- **Acoustic Integration** - Direct integration with acoustic models for emotion conditioning
- **Custom Emotion Vectors** - Support for user-defined emotion characteristics
- **Emotion Validation** - Automatic validation of emotion parameters

## 🚀 Quick Start

### Basic Emotion Processing

```rust
use voirs_emotion::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create emotion processor with default configuration
    let processor = EmotionProcessor::builder()
        .with_default_config()
        .build()?;

    // Create emotion state
    let emotion = EmotionState::new(
        Emotion::Happy,
        EmotionIntensity::Medium,
    );

    // Apply emotion to prosody parameters
    let prosody_params = ProsodyParameters::default();
    let emotional_prosody = processor
        .apply_emotion(&prosody_params, &emotion)
        .await?;

    println!("Emotional prosody: {:?}", emotional_prosody);
    Ok(())
}
```

### Emotion Interpolation

```rust
use voirs_emotion::prelude::*;

// Create two emotion states
let happy = EmotionState::new(Emotion::Happy, EmotionIntensity::High);
let calm = EmotionState::new(Emotion::Calm, EmotionIntensity::Low);

// Create interpolator
let interpolator = EmotionInterpolator::new(InterpolationMethod::Linear);

// Interpolate between emotions (0.0 = happy, 1.0 = calm)
let blended_emotion = interpolator.interpolate(&happy, &calm, 0.3)?;
```

### Custom Emotion Configuration

```rust
use voirs_emotion::prelude::*;

// Create custom emotion configuration
let config = EmotionConfig::builder()
    .with_pitch_range(0.8, 1.4)  // 80% to 140% of base pitch
    .with_speed_range(0.7, 1.2)  // 70% to 120% of base speed
    .with_energy_range(0.6, 1.5) // 60% to 150% of base energy
    .with_emotion_sensitivity(0.8)
    .build()?;

let processor = EmotionProcessor::builder()
    .with_config(config)
    .build()?;
```

## 🎯 Emotion Types

### Primary Emotions
- **Happy** - Increased pitch, faster tempo, higher energy
- **Sad** - Decreased pitch, slower tempo, lower energy
- **Angry** - Sharp pitch changes, irregular timing, high energy
- **Calm** - Stable pitch, steady tempo, moderate energy
- **Excited** - High pitch variance, fast tempo, very high energy
- **Fear** - Trembling pitch, irregular timing, moderate energy

### Emotion Dimensions
- **Arousal** - Energy level and activation (0.0 = calm, 1.0 = excited)
- **Valence** - Emotional positivity (0.0 = negative, 1.0 = positive)
- **Dominance** - Control and confidence (0.0 = submissive, 1.0 = dominant)

## 🔧 Configuration

### Emotion Parameters

```rust
use voirs_emotion::types::*;

// Create emotion with specific parameters
let emotion_params = EmotionParameters {
    arousal: 0.8,      // High energy
    valence: 0.7,      // Positive emotion
    dominance: 0.6,    // Moderate confidence
    intensity: EmotionIntensity::High,
};

let custom_emotion = EmotionState::from_parameters(emotion_params);
```

### Prosody Modification

```rust
use voirs_emotion::prosody::*;

// Configure prosody modifier
let modifier = ProsodyModifier::builder()
    .with_pitch_scale(1.2)     // 20% pitch increase
    .with_tempo_scale(0.9)     // 10% slower
    .with_energy_scale(1.1)    // 10% louder
    .with_stress_emphasis(1.3) // 30% more stress
    .build()?;
```

## 🎪 Advanced Usage

### Real-time Emotion Control

```rust
use voirs_emotion::prelude::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let processor = EmotionProcessor::builder()
        .with_real_time_config()
        .build()?;

    let mut timer = interval(Duration::from_millis(100));
    let mut current_emotion = EmotionState::neutral();

    loop {
        timer.tick().await;
        
        // Update emotion based on external input
        let new_emotion = get_emotion_from_input().await?;
        
        // Smooth transition between emotions
        current_emotion = processor
            .transition_emotion(&current_emotion, &new_emotion, 0.1)
            .await?;
        
        // Apply to synthesis pipeline
        apply_emotion_to_synthesis(&current_emotion).await?;
    }
}
```

### SSML Integration

```rust
use voirs_emotion::ssml::*;

// Parse SSML with emotion markup
let ssml_text = r#"
<speak>
    <emotion name="happy" intensity="medium">
        Hello there!
    </emotion>
    <emotion name="calm" intensity="low">
        How are you doing today?
    </emotion>
</speak>
"#;

let emotion_elements = EmotionSSMLParser::parse(ssml_text)?;
for element in emotion_elements {
    let emotion_state = EmotionState::from_ssml(&element)?;
    // Process with emotion
}
```

## 🔍 Performance

- **Latency**: <5ms emotion processing overhead
- **Memory**: <50MB emotion model footprint
- **CPU Usage**: <2% additional CPU overhead
- **Real-time**: Supports 100+ concurrent emotion streams

## 🧪 Testing

```bash
# Run emotion processing tests
cargo test --package voirs-emotion

# Run emotion interpolation tests
cargo test --package voirs-emotion interpolation

# Run SSML parsing tests
cargo test --package voirs-emotion ssml

# Run performance benchmarks
cargo bench --package voirs-emotion
```

## 🔗 Integration

### With Acoustic Models

```rust
use voirs_emotion::acoustic::*;

// Enable acoustic integration feature
let emotion_conditioner = AcousticEmotionConditioner::new();
let conditioned_features = emotion_conditioner
    .condition_features(&acoustic_features, &emotion_state)
    .await?;
```

### With Other VoiRS Crates

- **voirs-acoustic** - Emotion conditioning for neural acoustic models
- **voirs-prosody** - Advanced prosody control integration
- **voirs-ssml** - SSML parsing and emotion markup support
- **voirs-sdk** - High-level emotion control API

## 🛡️ Safety & Ethics

- **Emotion Validation** - Automatic validation prevents harmful emotion manipulation
- **Intensity Limits** - Built-in limits prevent extreme emotional distortion
- **Audit Logging** - Optional logging of emotion modifications for transparency
- **User Consent** - Clear indication when emotion modification is applied

## 📊 Benchmarks

| Operation | Time | Memory | Notes |
|-----------|------|--------|---------|
| Emotion Processing | 2.3ms | 12MB | Single emotion state |
| Interpolation | 1.1ms | 8MB | Linear interpolation |
| SSML Parsing | 5.7ms | 20MB | Complex markup |
| Real-time Stream | 0.8ms | 15MB | Per 100ms chunk |

## 🎓 Examples

See the [`examples/`](../../examples/) directory for comprehensive usage examples:

- [`emotion_control_example.rs`](../../examples/emotion_control_example.rs) - Basic emotion control
- [`emotion_interpolation.rs`](../../examples/emotion_interpolation.rs) - Emotion blending
- [`realtime_emotion.rs`](../../examples/realtime_emotion.rs) - Real-time processing

## 📝 License

Licensed under either of Apache License 2.0 or MIT License at your option.

---

*Part of the [VoiRS](../../README.md) neural speech synthesis ecosystem.*