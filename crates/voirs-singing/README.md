# voirs-singing

> **Comprehensive Singing Voice Synthesis System**

This crate provides comprehensive singing voice synthesis capabilities including musical note processing, pitch contour generation, rhythm control, vibrato modeling, and musical format support for creating high-quality singing voices.

## 🎭 Features

### Core Singing Synthesis
- **Musical Note Processing** - Precise control over pitch, duration, and timing
- **Pitch Contour Generation** - Natural pitch transitions and vibrato
- **Rhythm Control** - Accurate timing and beat alignment
- **Breath Modeling** - Realistic breath patterns and phrasing

### Voice Techniques
- **Vibrato Control** - Customizable vibrato rate, depth, and onset
- **Legato Processing** - Smooth note transitions and portamento
- **Vocal Fry** - Natural vocal fry effects at phrase boundaries
- **Breath Control** - Intelligent breath placement and intensity

### Musical Format Support
- **MIDI Integration** - Direct MIDI file processing and real-time input
- **MusicXML Support** - Industry-standard score format parsing
- **Custom Score Format** - Optimized internal score representation
- **Real-time Performance** - Live singing synthesis for performances

### Voice Management
- **Voice Banks** - Multiple singing voice libraries
- **Voice Characteristics** - Configurable vocal range, timbre, and style
- **Multi-voice Harmony** - Simultaneous multi-part singing
- **Voice Blending** - Smooth transitions between different voices

## 🚀 Quick Start

### Basic Singing Synthesis

```rust
use voirs_singing::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create singing engine
    let engine = SingingEngine::builder()
        .with_voice_bank("soprano_voice")
        .with_sample_rate(44100)
        .build().await?;

    // Create a simple musical score
    let score = MusicalScore::builder()
        .add_note(MusicalNote {
            pitch: Pitch::from_midi(60), // C4
            duration: Duration::quarter_note(),
            lyrics: "Hello".to_string(),
            start_time: Time::zero(),
        })
        .add_note(MusicalNote {
            pitch: Pitch::from_midi(64), // E4
            duration: Duration::quarter_note(),
            lyrics: "world".to_string(),
            start_time: Time::from_beats(1.0),
        })
        .with_tempo(120) // BPM
        .with_key_signature(KeySignature::C_major())
        .build()?;

    // Synthesize singing
    let request = SingingRequest::new(score)
        .with_vibrato(VibratoParams::natural())
        .with_breath_control(true)
        .with_expression(Expression::Gentle);

    let result = engine.synthesize(request).await?;
    
    // Save synthesized audio
    result.save_wav("singing_output.wav").await?;
    
    println!("Synthesis completed in {:.2}s", result.processing_time);
    Ok(())
}
```

### MIDI File Processing

```rust
use voirs_singing::prelude::*;

// Load and process MIDI file
let midi_parser = MidiParser::new();
let musical_score = midi_parser
    .parse_file("song.mid")
    .await?
    .with_lyrics_from_file("lyrics.txt")
    .await?;

// Create singing engine with appropriate voice
let engine = SingingEngine::builder()
    .with_voice_type(VoiceType::Soprano)
    .with_language("en-US")
    .build().await?;

// Synthesize the entire song
let singing_audio = engine
    .synthesize_score(&musical_score)
    .await?;
```

### Real-time Singing Performance

```rust
use voirs_singing::prelude::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    // Create real-time singing engine
    let engine = SingingEngine::builder()
        .with_real_time_mode(true)
        .with_buffer_size(256)  // Low latency
        .build().await?;

    // Setup MIDI input
    let mut midi_input = MidiInput::new().await?;
    let mut timer = interval(Duration::from_millis(10));
    
    loop {
        timer.tick().await;
        
        // Process MIDI events
        if let Some(midi_event) = midi_input.next_event().await? {
            match midi_event {
                MidiEvent::NoteOn { note, velocity, .. } => {
                    let note_event = NoteEvent {
                        pitch: Pitch::from_midi(note),
                        velocity: velocity as f32 / 127.0,
                        lyrics: get_current_lyrics(),
                        start_time: engine.current_time(),
                    };
                    
                    engine.start_note(note_event).await?;
                }
                MidiEvent::NoteOff { note, .. } => {
                    engine.stop_note(Pitch::from_midi(note)).await?;
                }
                _ => {}
            }
        }
        
        // Generate audio chunk
        let audio_chunk = engine.process_audio_chunk().await?;
        output_audio(audio_chunk).await?;
    }
}
```

## 🎵 Musical Elements

### Pitch Control

```rust
use voirs_singing::pitch::*;

// Create pitch contour with vibrato
let pitch_generator = PitchGenerator::builder()
    .with_base_frequency(440.0) // A4
    .with_vibrato(VibratoParams {
        rate: 6.0,           // 6 Hz vibrato
        depth: 0.1,          // 10% pitch variation
        onset_delay: 0.2,    // Start vibrato after 200ms
        fade_in_time: 0.1,   // 100ms fade-in
    })
    .build()?;

// Generate pitch contour for a note
let pitch_contour = pitch_generator
    .generate_contour(
        Duration::from_seconds(2.0),
        PitchStyle::Lyrical
    )
    .await?;
```

### Rhythm and Timing

```rust
use voirs_singing::rhythm::*;

// Create rhythm processor
let rhythm_processor = RhythmProcessor::builder()
    .with_tempo(120)              // 120 BPM
    .with_time_signature(4, 4)    // 4/4 time
    .with_swing_ratio(0.1)        // Slight swing
    .build()?;

// Process note timing
let timing_controller = TimingController::new()
    .with_humanization(0.05)      // 5% timing variation
    .with_breath_pause_detection(true)
    .build()?;

let adjusted_timing = timing_controller
    .adjust_note_timing(&musical_notes)
    .await?;
```

### Voice Techniques

```rust
use voirs_singing::techniques::*;

// Configure vibrato
let vibrato = VibratoProcessor::builder()
    .with_natural_variation(true)
    .with_rate_range(5.0..8.0)    // Variable vibrato rate
    .with_depth_modulation(true)
    .build()?;

// Configure legato
let legato = LegatoProcessor::builder()
    .with_portamento_time(0.1)    // 100ms portamento
    .with_phrase_connection(true)
    .build()?;

// Configure breath control
let breath_control = BreathControl::builder()
    .with_automatic_phrases(true)
    .with_breath_noise(true)
    .with_phrase_length_limit(8.0) // Max 8 seconds per phrase
    .build()?;
```

## 🔧 Configuration

### Voice Configuration

```rust
use voirs_singing::voice::*;

// Configure voice characteristics
let voice_config = VoiceCharacteristics {
    voice_type: VoiceType::Soprano,
    vocal_range: VocalRange {
        lowest_note: Pitch::from_midi(48), // C3
        highest_note: Pitch::from_midi(84), // C6
        comfortable_low: Pitch::from_midi(60), // C4
        comfortable_high: Pitch::from_midi(72), // C5
    },
    timbre: TimbreConfig {
        brightness: 0.7,
        warmth: 0.6,
        breathiness: 0.3,
        nasality: 0.2,
    },
    vibrato_tendency: 0.8,
    language_accent: "en-US".to_string(),
};

// Create voice bank
let voice_bank = VoiceBank::builder()
    .with_voice("soprano_1", voice_config)
    .with_voice("alto_1", alto_config)
    .with_voice("tenor_1", tenor_config)
    .with_voice("bass_1", bass_config)
    .build()?;
```

### Synthesis Configuration

```rust
use voirs_singing::config::*;

// High-quality synthesis configuration
let config = SingingConfig::builder()
    .with_sample_rate(48000)
    .with_bit_depth(24)
    .with_synthesis_quality(QualityLevel::Studio)
    .with_pitch_accuracy(0.99)      // 99% pitch accuracy
    .with_timing_precision(0.98)    // 98% timing precision
    .with_natural_variations(true)
    .build()?;

// Real-time performance configuration
let realtime_config = SingingConfig::builder()
    .with_sample_rate(44100)
    .with_buffer_size(256)
    .with_synthesis_quality(QualityLevel::Realtime)
    .with_latency_target(10)        // 10ms target latency
    .build()?;
```

## 🎪 Advanced Features

### Multi-voice Harmony

```rust
use voirs_singing::harmony::*;

// Create harmony arrangement
let harmony_arranger = HarmonyArranger::new();

// Define voice parts
let soprano_part = VoicePart {
    voice_name: "soprano".to_string(),
    melody: soprano_melody,
    harmony_role: HarmonyRole::Melody,
};

let alto_part = VoicePart {
    voice_name: "alto".to_string(),
    melody: alto_harmony,
    harmony_role: HarmonyRole::Harmony,
};

// Synthesize multi-voice arrangement
let harmony_parts = vec![soprano_part, alto_part, tenor_part, bass_part];
let harmony_audio = harmony_arranger
    .synthesize_harmony(harmony_parts)
    .await?;
```

### Vocal Effects

```rust
use voirs_singing::effects::*;

// Create effect chain
let effect_chain = EffectChain::builder()
    .add_effect(SingingEffect::Reverb {
        room_size: 0.6,
        damping: 0.4,
        wet_level: 0.3,
    })
    .add_effect(SingingEffect::Chorus {
        rate: 0.8,
        depth: 0.2,
        feedback: 0.1,
    })
    .add_effect(SingingEffect::EQ {
        low_gain: 1.0,
        mid_gain: 1.1,
        high_gain: 1.05,
    })
    .build()?;

// Apply effects to synthesized audio
let processed_audio = effect_chain
    .process(&singing_audio)
    .await?;
```

### Custom Vocal Styles

```rust
use voirs_singing::styles::*;

// Define custom singing style
let jazz_style = SingingStyle::builder()
    .with_name("Jazz")
    .with_vibrato_characteristics(VibratoStyle {
        onset_timing: OnsetTiming::Late,
        rate_variation: 0.3,
        depth_expression: 0.8,
    })
    .with_phrasing(PhrasingStyle {
        breath_frequency: BreathFrequency::Moderate,
        legato_preference: 0.7,
        staccato_preference: 0.3,
    })
    .with_articulation(ArticulationStyle {
        consonant_emphasis: 0.8,
        vowel_modification: true,
        swing_timing: 0.15,
    })
    .build()?;

// Apply style to synthesis
let styled_request = SingingRequest::new(score)
    .with_style(jazz_style)
    .with_expression_intensity(0.8);
```

## 🔍 Performance

### Synthesis Performance

| Quality Level | RTF | Latency | Memory | CPU Usage |
|---------------|-----|---------|--------|-----------|
| Draft | 0.1× | 5ms | 200MB | 15% |
| Standard | 0.3× | 15ms | 400MB | 25% |
| High | 0.6× | 30ms | 600MB | 40% |
| Studio | 1.2× | 100ms | 1GB | 60% |

### Real-time Performance

```rust
use voirs_singing::performance::*;

// Performance monitoring
let monitor = SynthesisPerformanceMonitor::new();

// Optimize for different scenarios
let performance_config = match use_case {
    UseCase::RealtimePerformance => PerformanceConfig {
        quality_level: QualityLevel::Realtime,
        buffer_size: 128,
        max_latency_ms: 10,
        cpu_usage_limit: 0.3,
    },
    UseCase::StudioRecording => PerformanceConfig {
        quality_level: QualityLevel::Studio,
        buffer_size: 2048,
        max_latency_ms: 200,
        cpu_usage_limit: 0.8,
    },
};
```

## 🎹 Musical Format Support

### MIDI Integration

```rust
use voirs_singing::formats::*;

// Enhanced MIDI processing
let midi_processor = MidiProcessor::builder()
    .with_velocity_mapping(VelocityMapping::Exponential)
    .with_pedal_interpretation(true)
    .with_expression_controllers(vec![1, 7, 11]) // Modulation, Volume, Expression
    .build()?;

// Process MIDI with advanced features
let enhanced_score = midi_processor
    .process_midi_file("complex_song.mid")
    .await?
    .with_dynamics_interpretation(true)
    .with_articulation_marks(true)
    .with_tempo_variations(true);
```

### MusicXML Support

```rust
use voirs_singing::formats::*;

// Parse MusicXML with full feature support
let musicxml_parser = MusicXmlParser::new()
    .with_lyrics_extraction(true)
    .with_chord_symbol_support(true)
    .with_expression_marks(true)
    .build()?;

let complete_score = musicxml_parser
    .parse_file("score.musicxml")
    .await?;

// Access rich musical information
for measure in complete_score.measures() {
    for note in measure.notes() {
        println!("Note: {} {}, Lyrics: {}", 
                 note.pitch, note.duration, note.lyrics);
        
        if let Some(dynamics) = note.dynamics {
            println!("Dynamics: {:?}", dynamics);
        }
    }
}
```

## 🧪 Testing

```bash
# Run singing synthesis tests
cargo test --package voirs-singing

# Run musical format tests
cargo test --package voirs-singing formats

# Run voice technique tests
cargo test --package voirs-singing techniques

# Run real-time performance tests
cargo test --package voirs-singing realtime

# Run performance benchmarks
cargo bench --package voirs-singing
```

## 🔗 Integration

### With Emotion Control

```rust
use voirs_singing::emotion::*;

// Integrate emotional expression in singing
let emotional_singing = EmotionalSingingProcessor::new()
    .with_emotion_mapping(EmotionMapping::Musical)
    .build()?;

let expressive_audio = emotional_singing
    .apply_emotion(&singing_audio, &emotion_state)
    .await?;
```

### With Other VoiRS Crates

- **voirs-emotion** - Emotional expression in singing
- **voirs-acoustic** - Advanced acoustic modeling for singing
- **voirs-vocoder** - High-quality singing voice synthesis
- **voirs-cloning** - Custom singing voice creation
- **voirs-sdk** - High-level singing API

## 🎓 Examples

See the [`examples/`](../../examples/) directory for comprehensive usage examples:

- [`singing_synthesis_example.rs`](../../examples/singing_synthesis_example.rs) - Basic singing
- [`midi_to_singing.rs`](../../examples/midi_to_singing.rs) - MIDI processing
- [`harmony_synthesis.rs`](../../examples/harmony_synthesis.rs) - Multi-voice harmony
- [`realtime_singing.rs`](../../examples/realtime_singing.rs) - Live performance

## 📝 License

Licensed under either of Apache License 2.0 or MIT License at your option.

---

*Part of the [VoiRS](../../README.md) neural speech synthesis ecosystem.*