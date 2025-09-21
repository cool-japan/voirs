# voirs-spatial

> **3D Spatial Audio Processing and Binaural Rendering System**

This crate provides 3D spatial audio processing capabilities including HRTF (Head-Related Transfer Function) processing, binaural audio rendering, 3D position tracking, room acoustics simulation, and AR/VR integration for immersive audio experiences.

## 🎭 Features

### Core Spatial Processing
- **3D Audio Positioning** - Accurate 3D sound source placement and tracking
- **HRTF Processing** - Head-Related Transfer Function for realistic spatial cues
- **Binaural Rendering** - High-quality binaural audio synthesis
- **Real-time Processing** - Low-latency spatial audio for interactive applications

### Environmental Simulation
- **Room Acoustics** - Realistic room reverb and acoustic modeling
- **Distance Modeling** - Natural distance attenuation and air absorption
- **Occlusion/Obstruction** - Audio blocking and filtering effects
- **Multi-room Environments** - Complex architectural acoustic simulation

### Interactive Features
- **Head Tracking** - Real-time head orientation tracking integration
- **Dynamic Sources** - Moving sound sources with Doppler effects
- **Listener Movement** - First-person perspective audio navigation
- **Gesture Control** - Hand and body gesture-based audio interaction

### Platform Integration
- **VR/AR Support** - Integration with VR/AR platforms and headsets
- **Gaming Engines** - Unity, Unreal Engine, and custom engine support
- **Mobile Platforms** - iOS and Android spatial audio capabilities
- **WebXR** - Browser-based immersive audio experiences

## 🚀 Quick Start

### Basic 3D Audio Setup

```rust
use voirs_spatial::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create spatial audio processor
    let processor = SpatialProcessor::builder()
        .with_sample_rate(48000)
        .with_hrtf_database("mit_kemar")
        .with_room_simulation(true)
        .build().await?;

    // Create listener (user's head position)
    let listener = Listener::new()
        .at_position(Position3D::origin())
        .facing_direction(Vector3D::forward())
        .with_head_radius(0.0875); // 8.75cm average head radius

    // Create sound source
    let sound_source = SoundSource::builder()
        .at_position(Position3D::new(2.0, 1.0, 0.0)) // 2m right, 1m up
        .with_audio_file("voice_sample.wav")
        .with_directivity(Directivity::Omnidirectional)
        .build().await?;

    // Process spatial audio
    let spatial_request = SpatialRequest {
        listener,
        sources: vec![sound_source],
        room_config: Some(RoomConfig::living_room()),
    };

    let result = processor.process(spatial_request).await?;
    
    // Output binaural audio (stereo with 3D cues)
    result.save_binaural_audio("spatial_output.wav").await?;
    
    println!("Spatial processing latency: {:.1}ms", result.processing_latency_ms);
    Ok(())
}
```

### Real-time VR Audio

```rust
use voirs_spatial::prelude::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    // Create VR-optimized spatial processor
    let processor = SpatialProcessor::builder()
        .with_vr_optimization(true)
        .with_buffer_size(256)    // Low latency for VR
        .with_update_rate(90)     // 90Hz for VR headsets
        .build().await?;

    // Setup head tracking
    let mut head_tracker = HeadTracker::new("oculus_rift").await?;
    let mut timer = interval(Duration::from_millis(11)); // 90Hz
    
    // Create virtual environment
    let mut environment = VirtualEnvironment::builder()
        .with_room_size(10.0, 3.0, 8.0) // 10x3x8 meter room
        .with_material(Material::Concrete, Surface::Floor)
        .with_material(Material::Drywall, Surface::Walls)
        .with_material(Material::Acoustic_Tile, Surface::Ceiling)
        .build()?;

    // Add multiple sound sources
    environment.add_source(SoundSource::new()
        .at_position(Position3D::new(-2.0, 1.5, 3.0))
        .with_audio_stream(create_voice_stream().await?)
        .with_name("Speaker 1"));
    
    environment.add_source(SoundSource::new()
        .at_position(Position3D::new(2.0, 1.5, 3.0))
        .with_audio_stream(create_music_stream().await?)
        .with_name("Speaker 2"));

    loop {
        timer.tick().await;
        
        // Update head position from VR headset
        let head_pose = head_tracker.get_current_pose().await?;
        let listener = Listener::from_head_pose(head_pose);
        
        // Process real-time spatial audio
        let audio_frame = processor
            .process_realtime(&listener, &environment)
            .await?;
        
        // Output to VR headset audio
        vr_audio_output(audio_frame).await?;
    }
}
```

### Room Acoustics Simulation

```rust
use voirs_spatial::room::*;

// Create detailed room acoustic model
let room = RoomAcoustics::builder()
    .with_dimensions(8.0, 6.0, 3.0) // Length, width, height in meters
    .with_wall_material(Material::Brick, Absorption::Low)
    .with_floor_material(Material::Hardwood, Absorption::Medium)
    .with_ceiling_material(Material::Plaster, Absorption::Medium)
    .with_furniture(vec![
        Furniture::Sofa.at_position(Position3D::new(2.0, 1.0, 0.5)),
        Furniture::BookShelf.at_position(Position3D::new(-3.0, 0.0, 1.0)),
        Furniture::Table.at_position(Position3D::new(0.0, 2.0, 0.8)),
    ])
    .with_openings(vec![
        Opening::Door.at_position(Position3D::new(4.0, -3.0, 1.0)),
        Opening::Window.at_position(Position3D::new(4.0, 0.0, 1.5)),
    ])
    .build()?;

// Simulate room acoustics
let room_simulator = RoomSimulator::new()
    .with_reflection_order(3)     // Up to 3rd order reflections
    .with_diffraction_modeling(true)
    .with_air_absorption(true)
    .build()?;

let acoustic_response = room_simulator
    .simulate_room_response(&room, &source_position, &listener_position)
    .await?;

println!("RT60: {:.2}s", acoustic_response.rt60);
println!("Early reflections: {} ms", acoustic_response.early_reflection_time);
println!("Clarity (C50): {:.1} dB", acoustic_response.clarity_c50);
```

## 🔧 Configuration

### HRTF Configuration

```rust
use voirs_spatial::hrtf::*;

// Load and configure HRTF database
let hrtf_config = HrtfConfig::builder()
    .with_database(HrtfDatabase::MIT_KEMAR)  // High-quality research database
    .with_interpolation(InterpolationMethod::Bilinear)
    .with_personalization(PersonalizationLevel::Basic)
    .with_head_size_adjustment(true)
    .build()?;

// Create personalized HRTF
let personal_hrtf = HrtfPersonalizer::new()
    .with_head_measurements(HeadMeasurements {
        head_width: 15.2,      // cm
        head_depth: 19.1,      // cm
        ear_height: 6.5,       // cm from head center
        pinna_height: 6.2,     // cm
        pinna_width: 3.8,      // cm
    })
    .personalize_hrtf(&standard_hrtf)
    .await?;
```

### Environmental Configuration

```rust
use voirs_spatial::environment::*;

// Configure acoustic environment
let environment_config = EnvironmentConfig::builder()
    .with_air_temperature(20.0)      // Celsius
    .with_humidity(50.0)             // Percent
    .with_air_pressure(101325.0)     // Pascal (sea level)
    .with_wind_velocity(Vector3D::zero())
    .with_atmospheric_absorption(true)
    .build()?;

// Configure processing quality
let quality_config = ProcessingQuality {
    spatial_resolution: SpatialResolution::High,  // 1-degree resolution
    frequency_resolution: 512,                    // FFT size
    reflection_order: 4,                          // Up to 4th order reflections
    diffraction_accuracy: DiffractionAccuracy::Medium,
    update_rate: 60,                              // 60Hz spatial updates
};
```

### Audio Source Configuration

```rust
use voirs_spatial::sources::*;

// Configure sound source properties
let source_config = SoundSourceConfig {
    directivity: Directivity::Cardioid {
        main_lobe_direction: Vector3D::forward(),
        directivity_factor: 0.5,
    },
    frequency_response: FrequencyResponse::Flat,
    max_distance: 50.0,                // Meters
    reference_distance: 1.0,           // Meters
    rolloff_factor: 1.0,               // Natural distance rolloff
    doppler_enabled: true,
    air_absorption_enabled: true,
};

// Create moving sound source
let moving_source = MovingSoundSource::builder()
    .with_initial_position(Position3D::new(0.0, 2.0, 10.0))
    .with_velocity(Vector3D::new(0.0, 0.0, -5.0)) // Moving towards listener
    .with_trajectory(Trajectory::Linear)
    .with_doppler_scaling(1.0)
    .build()?;
```

## 🎪 Advanced Features

### Multi-room Audio Navigation

```rust
use voirs_spatial::navigation::*;

// Create multi-room environment
let building = BuildingLayout::builder()
    .add_room("living_room", Room::new()
        .with_dimensions(8.0, 6.0, 3.0)
        .with_acoustic_properties(AcousticProperties::living_room()))
    .add_room("kitchen", Room::new()
        .with_dimensions(4.0, 4.0, 3.0)
        .with_acoustic_properties(AcousticProperties::kitchen()))
    .add_room("hallway", Room::new()
        .with_dimensions(2.0, 8.0, 3.0)
        .with_acoustic_properties(AcousticProperties::hallway()))
    .add_connection("living_room", "hallway", Opening::Doorway)
    .add_connection("hallway", "kitchen", Opening::Archway)
    .build()?;

// Simulate audio propagation between rooms
let propagation_simulator = AudioPropagationSimulator::new()
    .with_inter_room_attenuation(true)
    .with_sound_transmission_loss(true)
    .build()?;

let audio_scene = propagation_simulator
    .simulate_building_acoustics(&building, &sound_sources)
    .await?;
```

### Gesture-based Audio Control

```rust
use voirs_spatial::gesture::*;

// Setup gesture recognition for audio control
let gesture_controller = GestureController::new()
    .with_hand_tracking(true)
    .with_gesture_library(GestureLibrary::Standard)
    .build()?;

// Define gesture mappings
let gesture_mappings = vec![
    GestureMapping {
        gesture: Gesture::Point,
        action: AudioAction::SelectSource,
        parameters: vec![("ray_casting", true.into())],
    },
    GestureMapping {
        gesture: Gesture::Grab,
        action: AudioAction::MoveSource,
        parameters: vec![("follow_hand", true.into())],
    },
    GestureMapping {
        gesture: Gesture::VolumeKnob,
        action: AudioAction::AdjustVolume,
        parameters: vec![("sensitivity", 0.5.into())],
    },
];

// Process gestures in real-time
loop {
    let hand_data = hand_tracker.get_current_pose().await?;
    
    if let Some(gesture) = gesture_controller.recognize(&hand_data).await? {
        if let Some(mapping) = gesture_mappings.iter()
            .find(|m| m.gesture == gesture) {
            spatial_processor.apply_gesture_action(&mapping.action, &mapping.parameters).await?;
        }
    }
}
```

### Occlusion and Obstruction

```rust
use voirs_spatial::occlusion::*;

// Create occlusion calculator
let occlusion_calculator = OcclusionCalculator::new()
    .with_ray_tracing(true)
    .with_diffraction_modeling(true)
    .with_material_properties(true)
    .build()?;

// Add obstacles to the environment
let obstacles = vec![
    Obstacle::Wall {
        start: Position3D::new(-2.0, -3.0, 0.0),
        end: Position3D::new(2.0, -3.0, 0.0),
        height: 3.0,
        material: Material::Concrete,
        thickness: 0.2,
    },
    Obstacle::Furniture {
        position: Position3D::new(1.0, 0.0, 0.5),
        size: Size3D::new(2.0, 1.0, 1.5),
        material: Material::Wood,
        shape: Shape::Box,
    },
];

// Calculate occlusion effects
let occlusion_result = occlusion_calculator
    .calculate_occlusion(&source_position, &listener_position, &obstacles)
    .await?;

println!("Direct path blocked: {}", occlusion_result.direct_path_blocked);
println!("Attenuation: {:.1} dB", occlusion_result.attenuation_db);
println!("Diffraction paths: {}", occlusion_result.diffraction_paths.len());
```

## 🔍 Performance

### Real-time Performance

| Configuration | Latency | CPU Usage | Memory | Max Sources |
|---------------|---------|-----------|--------|--------------|
| Mobile/VR | 5-10ms | 15% | 150MB | 8 sources |
| Desktop | 10-15ms | 25% | 300MB | 16 sources |
| High Quality | 20-30ms | 40% | 500MB | 32 sources |
| Studio | 50-100ms | 60% | 1GB | 64+ sources |

### Optimization Strategies

```rust
use voirs_spatial::optimization::*;

// Performance optimization for different platforms
let mobile_config = OptimizationConfig {
    spatial_resolution: SpatialResolution::Medium,
    hrtf_interpolation: InterpolationMethod::Linear,
    reflection_order: 2,
    update_rate: 30, // 30Hz for mobile
    use_gpu_acceleration: false,
};

let desktop_config = OptimizationConfig {
    spatial_resolution: SpatialResolution::High,
    hrtf_interpolation: InterpolationMethod::Cubic,
    reflection_order: 4,
    update_rate: 60, // 60Hz for desktop
    use_gpu_acceleration: true,
};

// Adaptive quality based on performance
let adaptive_processor = AdaptiveSpatialProcessor::new()
    .with_performance_monitoring(true)
    .with_quality_scaling(true)
    .with_cpu_budget(0.3) // 30% CPU budget
    .build()?;
```

## 🧪 Testing

```bash
# Run spatial audio tests
cargo test --package voirs-spatial

# Run HRTF processing tests
cargo test --package voirs-spatial hrtf

# Run room acoustics tests
cargo test --package voirs-spatial room

# Run real-time performance tests
cargo test --package voirs-spatial realtime

# Run VR integration tests
cargo test --package voirs-spatial vr

# Run performance benchmarks
cargo bench --package voirs-spatial
```

## 🔗 Integration

### VR/AR Platform Integration

```rust
use voirs_spatial::platforms::*;

// Oculus Integration
let oculus_adapter = OculusAdapter::new()
    .with_audio_sdk_integration(true)
    .with_hand_tracking(true)
    .build().await?;

// Unity Integration
let unity_plugin = UnitySpatialPlugin::new()
    .with_audio_source_component(true)
    .with_listener_component(true)
    .build()?;

// WebXR Integration
let webxr_processor = WebXRSpatialProcessor::new()
    .with_web_audio_api(true)
    .with_web_assembly_optimization(true)
    .build().await?;
```

### With Other VoiRS Crates

- **voirs-synthesis** - Spatial synthesis of synthesized voices
- **voirs-emotion** - Spatially-aware emotional audio
- **voirs-cloning** - 3D positioning of cloned voices
- **voirs-conversion** - Spatial voice conversion effects
- **voirs-sdk** - High-level spatial audio API

## 🎓 Examples

See the [`examples/`](../../examples/) directory for comprehensive usage examples:

- [`spatial_audio_example.rs`](../../examples/spatial_audio_example.rs) - Basic 3D audio
- [`vr_audio_demo.rs`](../../examples/vr_audio_demo.rs) - VR spatial audio
- [`room_simulation.rs`](../../examples/room_simulation.rs) - Acoustic simulation
- [`multi_room_navigation.rs`](../../examples/multi_room_navigation.rs) - Complex environments

## 📝 License

Licensed under either of Apache License 2.0 or MIT License at your option.

---

*Part of the [VoiRS](../../README.md) neural speech synthesis ecosystem.*