# VoiRS Ecosystem Integration Guide

> **Guide for integrating voirs-spatial with other VoiRS crates and external systems**

## Table of Contents

- [VoiRS Ecosystem Overview](#voirs-ecosystem-overview)
- [Core Integration Points](#core-integration-points)
- [Integration Patterns](#integration-patterns)
- [External System Integration](#external-system-integration)
- [Advanced Integration](#advanced-integration)

---

## VoiRS Ecosystem Overview

The VoiRS ecosystem consists of modular crates that work together to provide comprehensive speech synthesis and audio processing capabilities:

```
┌────────────────────────────────────────────────────────────┐
│                      VoiRS Ecosystem                        │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  voirs-g2p   │   │ voirs-acoustic│   │ voirs-vocoder│   │
│  │  (Phonemes)  │──→│ (Mel Spectro) │──→│  (Waveform)  │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│                              │                               │
│                              ↓                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         voirs-spatial (3D Spatial Audio)             │   │
│  │  • HRTF Processing    • Room Acoustics               │   │
│  │  • Binaural Rendering • Multi-user Environments      │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ↓                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ voirs-emotion│   │voirs-cloning │   │ voirs-singing│   │
│  │ (Expressive) │   │(Voice Clone) │   │  (Singing)   │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              voirs-sdk (Unified API)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

---

## Core Integration Points

### 1. Integration with voirs-acoustic

**Use Case:** Spatialize synthesized speech from acoustic models

```rust
use voirs_acoustic::{AcousticModel, VitsModel};
use voirs_spatial::{BinauralRenderer, Position3D};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Generate mel spectrogram from acoustic model
    let acoustic_model = VitsModel::load("path/to/model").await?;
    let mel_spectrogram = acoustic_model.synthesize(phonemes).await?;

    // Step 2: Convert mel to waveform using vocoder
    let vocoder = HiFiGAN::load("path/to/vocoder").await?;
    let mono_audio = vocoder.generate(&mel_spectrogram)?;

    // Step 3: Spatialize the audio
    let hrtf_db = HrtfDatabase::load_default().await?;
    let mut spatial_renderer = BinauralRenderer::new(
        BinauralConfig::default(),
        Arc::new(hrtf_db)
    )?;

    // Add the synthesized speech as a spatial source
    let source_id = spatial_renderer.add_source(
        Position3D::new(1.0, 0.0, 2.0), // Position in 3D space
        SourceType::Static
    )?;

    // Process the mono audio through spatial rendering
    spatial_renderer.set_source_audio(source_id, &mono_audio)?;
    let binaural_output = spatial_renderer.process_frame()?;

    println!("✓ Synthesized speech spatialized successfully");
    Ok(())
}
```

### 2. Integration with voirs-emotion

**Use Case:** Add emotional spatial characteristics to synthesized speech

```rust
use voirs_emotion::{EmotionController, EmotionType};
use voirs_spatial::room::{RoomSimulator, WallMaterial};

async fn emotional_spatial_speech() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Apply emotion to speech
    let mut emotion_ctrl = EmotionController::new()?;
    emotion_ctrl.set_emotion(EmotionType::Joy, 0.8)?;

    let emotional_audio = emotion_ctrl.process(&base_audio)?;

    // Step 2: Match room acoustics to emotion
    let room_config = match emotion_ctrl.current_emotion() {
        EmotionType::Joy => RoomConfig {
            dimensions: Position3D::new(15.0, 8.0, 12.0), // Large, open space
            wall_material: WallMaterial::Wood,             // Warm acoustics
            reverb_time: 1.5,
            ..Default::default()
        },
        EmotionType::Sadness => RoomConfig {
            dimensions: Position3D::new(6.0, 3.0, 5.0),   // Small, intimate
            wall_material: WallMaterial::Carpet,          // Damped
            reverb_time: 0.5,
            ..Default::default()
        },
        _ => RoomConfig::default(),
    };

    let mut room = RoomSimulator::new(room_config)?;
    let spatialized = room.process(
        &emotional_audio,
        &source_position,
        &listener_position
    )?;

    Ok(())
}
```

### 3. Integration with voirs-cloning

**Use Case:** Create spatial scenes with cloned voices

```rust
use voirs_cloning::{VoiceCloner, CloningConfig};
use voirs_spatial::multiuser::{MultiuserEnvironment, UserRole};

async fn cloned_voice_multiuser() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Clone target voice
    let cloning_config = CloningConfig::default();
    let cloner = VoiceCloner::new(cloning_config)?;

    let reference_audio = load_audio("reference.wav")?;
    let voice_embedding = cloner.create_embedding(&reference_audio).await?;

    // Step 2: Create multi-user environment
    let mut multi_env = MultiuserEnvironment::new(MultiuserConfig::default())?;

    // Step 3: Add user with cloned voice
    let user_id = multi_env.add_user(
        "cloned_speaker".to_string(),
        UserRole::Speaker,
        Position3D::new(2.0, 0.0, 1.0)
    )?;

    // Step 4: Synthesize and spatialize speech
    let synthesized_speech = cloner.synthesize_with_embedding(
        &voice_embedding,
        "Hello from my cloned voice!"
    ).await?;

    multi_env.set_user_audio(user_id, &synthesized_speech)?;

    Ok(())
}
```

### 4. Integration with voirs-recognizer

**Use Case:** Spatial audio feedback for speech recognition

```rust
use voirs_recognizer::{Recognizer, RecognitionConfig};
use voirs_spatial::Position3D;

async fn spatial_recognition_feedback() -> Result<(), Box<dyn std::error::Error>> {
    // Setup recognizer
    let recognizer = Recognizer::new(RecognitionConfig::default()).await?;

    // Setup spatial renderer
    let mut spatial = setup_spatial_renderer().await?;

    // Process audio input
    let input_audio = capture_microphone()?;

    // Recognize speech
    let recognition_result = recognizer.recognize(&input_audio).await?;

    // Provide spatial feedback based on recognition confidence
    let feedback_position = match recognition_result.confidence {
        c if c > 0.8 => Position3D::new(0.0, 1.0, 1.0),  // Above and forward (success)
        c if c > 0.5 => Position3D::new(1.0, 0.0, 1.0),  // To the right (uncertain)
        _ => Position3D::new(-1.0, -0.5, 1.0),           // Left and down (error)
    };

    // Play confirmation sound at appropriate position
    let confirmation_sound = generate_confirmation_tone(recognition_result.confidence);
    spatial.add_source(feedback_position, SourceType::OneShot)?;
    spatial.set_source_audio(0, &confirmation_sound)?;

    println!("Recognized: {}", recognition_result.text);
    println!("Confidence: {:.1}%", recognition_result.confidence * 100.0);

    Ok(())
}
```

---

## Integration Patterns

### Pattern 1: TTS Pipeline with Spatial Output

```rust
use voirs_g2p::G2p;
use voirs_acoustic::AcousticModel;
use voirs_vocoder::Vocoder;
use voirs_spatial::{BinauralRenderer, Position3D};

async fn complete_tts_spatial_pipeline(
    text: &str,
    voice_position: Position3D
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Step 1: Text → Phonemes
    let g2p = G2p::new()?;
    let phonemes = g2p.convert(text)?;

    // Step 2: Phonemes → Mel Spectrogram
    let acoustic = VitsModel::load_default().await?;
    let mel = acoustic.synthesize(&phonemes).await?;

    // Step 3: Mel → Waveform
    let vocoder = HiFiGAN::load_default().await?;
    let mono_audio = vocoder.generate(&mel)?;

    // Step 4: Mono → Binaural Spatial Audio
    let hrtf_db = HrtfDatabase::load_default().await?;
    let mut spatial = BinauralRenderer::new(
        BinauralConfig::default(),
        Arc::new(hrtf_db)
    )?;

    let source_id = spatial.add_source(voice_position, SourceType::Static)?;
    spatial.set_source_audio(source_id, &mono_audio)?;

    let binaural_output = spatial.process_frame()?;

    // Return stereo output
    Ok(binaural_output.interleaved())
}
```

### Pattern 2: Dynamic Scene Management

```rust
use voirs_spatial::{BinauralRenderer, Position3D, SourceType};
use std::collections::HashMap;

struct SpatialScene {
    renderer: BinauralRenderer,
    sources: HashMap<String, (usize, Position3D)>,
}

impl SpatialScene {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let hrtf_db = HrtfDatabase::load_default().await?;
        let renderer = BinauralRenderer::new(
            BinauralConfig::default(),
            Arc::new(hrtf_db)
        )?;

        Ok(Self {
            renderer,
            sources: HashMap::new(),
        })
    }

    pub fn add_tts_source(
        &mut self,
        name: String,
        text: &str,
        position: Position3D
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Generate speech from text
        let audio = generate_tts_audio(text).await?;

        // Add to spatial scene
        let source_id = self.renderer.add_source(position, SourceType::Static)?;
        self.renderer.set_source_audio(source_id, &audio)?;

        self.sources.insert(name, (source_id, position));
        Ok(())
    }

    pub fn move_source(
        &mut self,
        name: &str,
        new_position: Position3D
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((source_id, _)) = self.sources.get_mut(name) {
            self.renderer.update_source_position(*source_id, new_position)?;
            Ok(())
        } else {
            Err("Source not found".into())
        }
    }

    pub fn process(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let output = self.renderer.process_frame()?;
        Ok(output.interleaved())
    }
}
```

### Pattern 3: Adaptive Quality Based on System Load

```rust
use voirs_spatial::{BinauralRenderer, BinauralConfig};
use voirs_acoustic::AcousticModel;

struct AdaptiveRenderer {
    spatial: BinauralRenderer,
    acoustic: AcousticModel,
    current_quality: f32,
}

impl AdaptiveRenderer {
    pub fn update_quality_based_on_load(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get system metrics
        let cpu_usage = get_cpu_usage();
        let spatial_latency = self.spatial.get_performance_metrics().avg_processing_time_ms;

        // Adjust quality dynamically
        let target_quality = if cpu_usage > 80.0 || spatial_latency > 20.0 {
            0.6  // Reduce quality under high load
        } else if cpu_usage < 50.0 && spatial_latency < 10.0 {
            0.9  // Increase quality when resources available
        } else {
            self.current_quality
        };

        if (target_quality - self.current_quality).abs() > 0.1 {
            self.current_quality = target_quality;

            // Update both acoustic and spatial quality
            self.acoustic.set_quality_level(target_quality)?;
            self.spatial.set_quality_level(target_quality)?;

            println!("Adapted quality to {:.1}%", target_quality * 100.0);
        }

        Ok(())
    }
}
```

---

## External System Integration

### Unity Game Engine

```csharp
// Unity C# script
using System.Runtime.InteropServices;
using UnityEngine;

public class VoirsSpatialAudio : MonoBehaviour
{
    // Import C API functions
    [DllImport("voirs_spatial")]
    private static extern IntPtr voirs_spatial_create(SpatialConfig config);

    [DllImport("voirs_spatial")]
    private static extern int voirs_spatial_add_source(
        IntPtr renderer,
        Vector3 position,
        int sourceType
    );

    [DllImport("voirs_spatial")]
    private static extern void voirs_spatial_process(
        IntPtr renderer,
        float[] output,
        int outputLength
    );

    private IntPtr spatialRenderer;

    void Start()
    {
        // Initialize VoiRS Spatial from Unity
        SpatialConfig config = new SpatialConfig {
            sampleRate = 48000,
            bufferSize = 512,
            maxSources = 16
        };

        spatialRenderer = voirs_spatial_create(config);
    }

    public int AddAudioSource(Vector3 position)
    {
        return voirs_spatial_add_source(
            spatialRenderer,
            position,
            0  // SourceType::Static
        );
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        // Process spatial audio in Unity audio thread
        voirs_spatial_process(spatialRenderer, data, data.Length);
    }

    void OnDestroy()
    {
        // Cleanup
        voirs_spatial_destroy(spatialRenderer);
    }
}
```

### Unreal Engine

```cpp
// Unreal Engine C++ integration
#include "VoirsSpatialAudio.h"

class FVoirsSpatialAudioModule : public IModuleInterface
{
public:
    virtual void StartupModule() override
    {
        // Initialize VoiRS Spatial
        SpatialConfig Config;
        Config.SampleRate = 48000;
        Config.BufferSize = 512;
        Config.MaxSources = 16;

        SpatialRenderer = voirs_spatial_create(Config);
    }

    virtual void ShutdownModule() override
    {
        voirs_spatial_destroy(SpatialRenderer);
    }

    int32 AddSpatialSource(FVector Position, uint8 SourceType)
    {
        return voirs_spatial_add_source(
            SpatialRenderer,
            Position,
            SourceType
        );
    }

    void ProcessAudio(TArray<float>& OutputBuffer)
    {
        voirs_spatial_process(
            SpatialRenderer,
            OutputBuffer.GetData(),
            OutputBuffer.Num()
        );
    }

private:
    void* SpatialRenderer;
};
```

### Web Integration (WebAssembly)

```javascript
// JavaScript/WebAssembly integration
import init, { BinauralRenderer } from './voirs_spatial.js';

async function setupSpatialAudio() {
    // Initialize WASM module
    await init();

    // Create spatial renderer
    const renderer = new BinauralRenderer({
        sampleRate: 48000,
        bufferSize: 512,
        maxSources: 8
    });

    // Add TTS source at specific position
    const sourceId = renderer.addSource(
        { x: 1.0, y: 0.0, z: 2.0 },
        'Static'
    );

    // Connect to Web Audio API
    const audioContext = new AudioContext();
    const scriptNode = audioContext.createScriptProcessor(512, 0, 2);

    scriptNode.onaudioprocess = (event) => {
        const output = renderer.processFrame();
        const left = event.outputBuffer.getChannelData(0);
        const right = event.outputBuffer.getChannelData(1);

        for (let i = 0; i < output.left.length; i++) {
            left[i] = output.left[i];
            right[i] = output.right[i];
        }
    };

    scriptNode.connect(audioContext.destination);
}
```

---

## Advanced Integration

### Custom Acoustic Model Integration

```rust
use voirs_spatial::{BinauralRenderer, Position3D};

// Custom acoustic model trait
trait CustomAcousticModel {
    async fn synthesize(&self, input: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
}

// Integration function
async fn integrate_custom_model<M: CustomAcousticModel>(
    model: &M,
    text: &str,
    position: Position3D
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Generate audio from custom model
    let mono_audio = model.synthesize(text).await?;

    // Spatialize using VoiRS Spatial
    let hrtf_db = HrtfDatabase::load_default().await?;
    let mut renderer = BinauralRenderer::new(
        BinauralConfig::default(),
        Arc::new(hrtf_db)
    )?;

    let source_id = renderer.add_source(position, SourceType::Static)?;
    renderer.set_source_audio(source_id, &mono_audio)?;

    let output = renderer.process_frame()?;
    Ok(output.interleaved())
}
```

### Real-time Streaming Integration

```rust
use tokio::sync::mpsc;
use voirs_spatial::BinauralRenderer;

async fn streaming_spatial_tts() -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel::<Vec<f32>>(32);

    // Spawn TTS generation task
    tokio::spawn(async move {
        let mut tts_engine = VitsModel::load_default().await.unwrap();

        while let Some(text) = get_next_text_chunk() {
            let audio = tts_engine.synthesize_streaming(&text).await.unwrap();
            tx.send(audio).await.unwrap();
        }
    });

    // Spatial processing task
    let mut spatial = setup_spatial_renderer().await?;

    while let Some(audio_chunk) = rx.recv().await {
        spatial.add_streaming_audio(0, &audio_chunk)?;
        let output = spatial.process_frame()?;

        // Send to audio output
        output_to_speakers(&output)?;
    }

    Ok(())
}
```

---

## Performance Considerations

### Memory Sharing

When integrating multiple VoiRS crates, share resources to reduce memory usage:

```rust
use std::sync::Arc;
use voirs_spatial::HrtfDatabase;
use voirs_acoustic::AcousticModel;

struct SharedResources {
    hrtf_db: Arc<HrtfDatabase>,
    acoustic_model: Arc<AcousticModel>,
}

impl SharedResources {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            hrtf_db: Arc::new(HrtfDatabase::load_default().await?),
            acoustic_model: Arc::new(AcousticModel::load_default().await?),
        })
    }

    fn create_spatial_renderer(&self) -> Result<BinauralRenderer, Box<dyn std::error::Error>> {
        BinauralRenderer::new(
            BinauralConfig::default(),
            Arc::clone(&self.hrtf_db)
        )
    }
}
```

### Thread Pool Management

```rust
use tokio::runtime::Runtime;

fn setup_optimized_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)  // Dedicated threads for audio
        .thread_name("voirs-audio")
        .enable_all()
        .build()
        .unwrap()
}
```

---

## Summary

VoiRS Spatial integrates seamlessly with:
- ✅ **voirs-acoustic**: Spatialize synthesized speech
- ✅ **voirs-emotion**: Emotional spatial characteristics
- ✅ **voirs-cloning**: Multi-user cloned voices
- ✅ **voirs-recognizer**: Spatial feedback systems
- ✅ **Unity/Unreal**: Game engine integration
- ✅ **WebAssembly**: Browser-based applications

**Key Integration Principles:**
1. Use `Arc<T>` for shared resources
2. Process pipeline: G2P → Acoustic → Vocoder → Spatial
3. Maintain consistent sample rates across crates
4. Monitor performance at integration boundaries
5. Leverage async/await for I/O operations

---

**Version:** 0.1.0
**Last Updated:** 2025-12-09
