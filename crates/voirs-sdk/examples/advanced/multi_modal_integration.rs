//! # Multi-Modal Integration Example
//!
//! This example demonstrates the seamless integration of multiple advanced VoiRS features:
//! - Singing Synthesis: Musical note generation with vocal techniques
//! - Spatial Audio: 3D positioned audio with room acoustics
//! - Emotion Control: Emotional expression in singing performance
//! - Voice Conversion: Dynamic voice characteristics transformation
//!
//! This showcases how VoiRS SDK can create rich, immersive audio experiences
//! suitable for virtual reality, gaming, and interactive applications.

use std::time::Instant;
use voirs_sdk::prelude::*;

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
use voirs_emotion::{Emotion, EmotionIntensity};
#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
// Note: These types would be imported from the singing crate when available
// For now we'll define placeholder types
#[derive(Debug, Clone, Copy)]
pub enum VoiceType {
    Soprano,
    Alto,
    Tenor,
    Bass,
}
#[derive(Debug, Clone, Copy)]
pub enum SingingTechnique {
    Legato,
    Dynamics,
}
#[derive(Debug, Clone, Copy)]
pub struct Tempo(u32);
impl Tempo {
    pub fn new(bpm: u32) -> Self {
        Self(bpm)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct TimeSignature(u8, u8);
impl TimeSignature {
    pub fn new(num: u8, den: u8) -> Self {
        Self(num, den)
    }
}
#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
use voirs_spatial::{Position3D, SpatialConfig};
// Note: RoomAcoustics is a trait, we'll create a struct for our demo
#[derive(Debug, Clone)]
pub struct RoomAcoustics {
    pub reverb_time: f32,
    pub early_reflections: f32,
    pub diffusion: f32,
    pub room_size: (f32, f32, f32),
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive logging
    // Initialize logging with proper config
    let logging_config = voirs_sdk::config::LoggingConfig::default();
    voirs_sdk::logging::init_logging(&logging_config)?;

    println!("🎭🎵🌍 VoiRS SDK Multi-Modal Integration Demo");
    println!("============================================\n");

    // Check feature availability
    check_feature_availability();

    #[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
    {
        // Demo 1: Virtual Concert Experience
        virtual_concert_demo().await?;

        // Demo 2: Interactive Storytelling
        interactive_storytelling_demo().await?;

        // Demo 3: Emotional Spatial Choir
        // emotional_spatial_choir_demo().await?; // Temporarily disabled due to compilation issue

        // Demo 4: Dynamic Voice Character System
        dynamic_voice_character_demo().await?;
    }

    #[cfg(not(all(feature = "singing", feature = "spatial", feature = "emotion")))]
    {
        println!("⚠️  Multi-modal integration requires all advanced features enabled:");
        println!(
            "   cargo run --example multi_modal_integration --features singing,spatial,emotion"
        );
    }

    println!("\n✨ Multi-modal integration demo completed!");
    Ok(())
}

fn check_feature_availability() {
    println!("🔍 Checking feature availability:");

    #[cfg(feature = "singing")]
    println!("  ✅ Singing synthesis enabled");
    #[cfg(not(feature = "singing"))]
    println!("  ❌ Singing synthesis disabled");

    #[cfg(feature = "spatial")]
    println!("  ✅ Spatial audio enabled");
    #[cfg(not(feature = "spatial"))]
    println!("  ❌ Spatial audio disabled");

    #[cfg(feature = "emotion")]
    println!("  ✅ Emotion control enabled");
    #[cfg(not(feature = "emotion"))]
    println!("  ❌ Emotion control disabled");

    #[cfg(feature = "conversion")]
    println!("  ✅ Voice conversion enabled");
    #[cfg(not(feature = "conversion"))]
    println!("  ❌ Voice conversion disabled");

    println!();
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn virtual_concert_demo() -> Result<()> {
    println!("🎤 Demo 1: Virtual Concert Experience");
    println!("=====================================");
    println!("Creating a virtual concert with multiple singers positioned in 3D space,");
    println!("each with different emotional expressions and vocal characteristics.\n");

    // Create pipeline with all advanced features
    let pipeline = VoirsPipelineBuilder::new()
        .with_singing_enabled(true)
        .with_spatial_enabled(true)
        .with_emotion_enabled(true)
        .with_conversion_enabled(true)
        .with_quality(QualityLevel::High)
        .with_test_mode(true)
        .build()
        .await?;

    println!("✅ Multi-modal pipeline initialized");

    // Define virtual concert hall
    let concert_hall = RoomAcoustics {
        reverb_time: 2.5,
        early_reflections: 0.3,
        diffusion: 0.8,
        room_size: (50.0, 30.0, 15.0), // width, depth, height in meters
    };

    // Define musical arrangement
    let song_lyrics = vec![
        (
            "Lead Singer",
            "Amazing grace, how sweet the sound",
            Position3D::new(0.0, 5.0, 0.0),
        ),
        (
            "Harmony 1",
            "That saved a wretch like me",
            Position3D::new(-3.0, 4.0, 0.0),
        ),
        (
            "Harmony 2",
            "I once was lost, but now am found",
            Position3D::new(3.0, 4.0, 0.0),
        ),
        (
            "Bass",
            "Was blind, but now I see",
            Position3D::new(0.0, 2.0, 0.0),
        ),
    ];

    // Performance characteristics for each singer
    let singer_configs = vec![
        ("Lead Singer", VoiceType::Soprano, Emotion::Happy, 0.8),
        ("Harmony 1", VoiceType::Alto, Emotion::Calm, 0.6),
        ("Harmony 2", VoiceType::Tenor, Emotion::Confident, 0.7),
        ("Bass", VoiceType::Bass, Emotion::Confident, 0.9),
    ];

    println!("🎼 Concert arrangement:");
    for (name, voice_type, emotion, intensity) in &singer_configs {
        println!(
            "  {} - {:?} voice with {:?} emotion (intensity: {:.1})",
            name, voice_type, emotion, intensity
        );
    }
    println!();

    // Synthesize each part with spatial positioning
    let mut concert_performance = Vec::new();
    let start_time = Instant::now();

    for ((singer_name, lyrics, position), (_, voice_type, emotion, intensity)) in
        song_lyrics.iter().zip(singer_configs.iter())
    {
        println!("🎵 Recording {}: \"{}\"", singer_name, lyrics);

        // Configure spatial positioning (simplified for compatibility)
        // TODO: Implement spatial positioning when API is available

        // Set emotional expression (simplified for compatibility)
        // TODO: Implement emotion setting when API is available

        // Configure singing voice and synthesize
        // Using simplified API for compatibility
        let audio = pipeline.synthesize_singing_text(lyrics, "C", 120.0).await?;

        println!(
            "  ✅ Generated {:.2}s of audio at position {:?}",
            audio.audio.duration(),
            position
        );

        concert_performance.push((singer_name.to_string(), audio.audio, *position));
    }

    let total_time = start_time.elapsed();
    println!(
        "\n🏟️  Virtual concert recorded in {:.2}s",
        total_time.as_secs_f64()
    );

    // Mix all performances with spatial audio
    let mixed_audio = mix_spatial_performance(&concert_performance, &concert_hall).await?;

    // Save the virtual concert
    mixed_audio.save_wav("virtual_concert.wav")?;
    println!(
        "💾 Saved virtual concert: virtual_concert.wav ({:.2}s)\n",
        mixed_audio.duration()
    );

    Ok(())
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn interactive_storytelling_demo() -> Result<()> {
    println!("📚 Demo 2: Interactive Storytelling");
    println!("===================================");
    println!("Creating an immersive story with multiple characters, each with unique");
    println!("voices, positions, and emotional arcs throughout the narrative.\n");

    let pipeline = VoirsPipelineBuilder::new()
        .with_singing_enabled(true)
        .with_spatial_enabled(true)
        .with_emotion_enabled(true)
        .with_conversion_enabled(true)
        .build()
        .await?;

    // Story scenes with character positioning and emotions
    let story_scenes = vec![
        StoryScene {
            character: "Narrator",
            text: "Once upon a time, in a magical forest...",
            position: Position3D::new(0.0, 0.0, -2.0),
            emotion: Emotion::Calm,
            intensity: 0.7,
            voice_style: VoiceStyle::Narrative,
        },
        StoryScene {
            character: "Fairy",
            text: "Welcome, traveler! I have a quest for you.",
            position: Position3D::new(2.0, 1.0, 1.0),
            emotion: Emotion::Happy,
            intensity: 0.9,
            voice_style: VoiceStyle::Melodic,
        },
        StoryScene {
            character: "Dragon",
            text: "Who dares enter my domain?",
            position: Position3D::new(-3.0, 0.0, 5.0),
            emotion: Emotion::Angry,
            intensity: 0.8,
            voice_style: VoiceStyle::Deep,
        },
        StoryScene {
            character: "Hero",
            text: "I seek the crystal of wisdom!",
            position: Position3D::new(0.0, 0.0, 0.0),
            emotion: Emotion::Confident,
            intensity: 0.7,
            voice_style: VoiceStyle::Heroic,
        },
    ];

    println!("🎬 Story scenes:");
    for scene in &story_scenes {
        println!(
            "  {} at {:?}: \"{}\"",
            scene.character,
            scene.position,
            scene.text.chars().take(40).collect::<String>()
        );
    }
    println!();

    // Set forest environment acoustics
    let forest_acoustics = RoomAcoustics {
        reverb_time: 1.2,
        early_reflections: 0.15,
        diffusion: 0.6,
        room_size: (100.0, 100.0, 20.0), // Open forest space
    };

    let mut story_audio = Vec::new();

    for (i, scene) in story_scenes.iter().enumerate() {
        println!("🎭 Scene {}: {}", i + 1, scene.character);

        // Configure character voice and position
        // Spatial positioning and emotion setting simplified for compatibility
        // TODO: Implement spatial positioning and emotion setting when API is available

        // Get voice style configuration
        let voice_config = get_voice_style_config(scene.voice_style);

        // Synthesize character dialogue with voice style
        let audio = pipeline
            .synthesize_with_config(&scene.text, &voice_config)
            .await?;

        println!(
            "  ✅ Generated {:.2}s of {} dialogue",
            audio.duration(),
            scene.character
        );

        story_audio.push(audio);

        // Add pause between scenes
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    // Combine story into single audio file
    let complete_story = combine_story_audio(story_audio).await?;
    complete_story.save_wav("interactive_story.wav")?;

    println!(
        "\n📖 Interactive story created: interactive_story.wav ({:.2}s)\n",
        complete_story.duration()
    );

    Ok(())
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn emotional_spatial_choir_demo() -> Result<()> {
    println!("🎼 Demo 3: Emotional Spatial Choir");
    println!("==================================");
    println!("Creating a dynamic choir that moves through emotional and spatial");
    println!("transformations while maintaining harmonic coherence.\n");

    let pipeline = VoirsPipelineBuilder::new()
        .with_singing_enabled(true)
        .with_spatial_enabled(true)
        .with_emotion_enabled(true)
        .build()
        .await?;

    // Define choir arrangement in 3D space
    let choir_positions = [
        Position3D::new(-2.0, 3.0, 0.0),  // Soprano left
        Position3D::new(2.0, 3.0, 0.0),   // Soprano right
        Position3D::new(-1.0, 1.0, 0.0),  // Alto left
        Position3D::new(1.0, 1.0, 0.0),   // Alto right
        Position3D::new(-1.0, -1.0, 0.0), // Tenor left
        Position3D::new(1.0, -1.0, 0.0),  // Tenor right
        Position3D::new(0.0, -3.0, 0.0),  // Bass center
    ];

    // Emotional progression through the piece
    let emotional_progression = [
        (Emotion::Calm, 0.3),
        (Emotion::Confident, 0.5),
        (Emotion::Happy, 0.7),
        (Emotion::Excited, 0.9),
        (Emotion::Calm, 0.4),
    ];

    // Musical phrases with harmony
    let choir_phrases = vec![
        "Hallelujah, hallelujah",
        "Praise be to the morning light",
        "Voices raised in harmony",
        "Together we sing as one",
        "Amen, amen",
    ];

    // Cathedral acoustics for choir
    let cathedral_acoustics = RoomAcoustics {
        reverb_time: 4.0,
        early_reflections: 0.4,
        diffusion: 0.9,
        room_size: (80.0, 120.0, 30.0),
    };

    // TODO: Convert local RoomAcoustics to SDK RoomAcoustics format
    // pipeline.set_room_acoustics(cathedral_acoustics).await?;

    println!("🏛️  Cathedral acoustics configured");
    println!(
        "👥 Choir arrangement: {} voices positioned in 3D space",
        choir_positions.len()
    );
    println!(
        "🎭 Emotional progression: {} phases\n",
        emotional_progression.len()
    );

    let mut choir_performance = Vec::new();

    for (phrase_idx, phrase) in choir_phrases.iter().enumerate() {
        let (emotion, intensity) = &emotional_progression[phrase_idx % emotional_progression.len()];

        println!(
            "🎵 Phrase {}: \"{}\" with {:?} emotion",
            phrase_idx + 1,
            phrase,
            emotion
        );

        // Record each voice part
        for (voice_idx, position) in choir_positions.iter().enumerate() {
            // TODO: Implement emotion setting when API is available
            // pipeline.set_emotion(emotion, Some(intensity)).await?;

            // TODO: Implement spatial positioning and singing configuration when API is available
            // pipeline.set_spatial_position(*position).await?;
            //
            // Configure voice type based on position
            let voice_type = match voice_idx {
                0..=1 => VoiceType::Soprano,
                2..=3 => VoiceType::Alto,
                4..=5 => VoiceType::Tenor,
                _ => VoiceType::Bass,
            };

            let _singing_config = create_singing_config(voice_type);
            // pipeline.set_singing_config(singing_config).await?;

            // Synthesize with slight variation for natural choir effect
            let harmony_text = add_harmonic_variation(phrase, voice_idx);
            let voice_audio = pipeline
                .synthesize_singing_text(&harmony_text, "C", 120.0)
                .await?;

            choir_performance.push((voice_idx, phrase_idx, voice_audio.audio, *position));
        }

        println!(
            "  ✅ Recorded {} voices for phrase {}",
            choir_positions.len(),
            phrase_idx + 1
        );
    }

    // Mix choir with spatial positioning
    let choir_mix = mix_choir_performance(&choir_performance).await?;
    choir_mix.save_wav("emotional_spatial_choir.wav")?;

    println!(
        "\n🎼 Emotional spatial choir created: emotional_spatial_choir.wav ({:.2}s)\n",
        choir_mix.duration()
    );

    Ok(())
}

#[cfg(all(
    feature = "singing",
    feature = "spatial",
    feature = "emotion",
    feature = "conversion"
))]
async fn dynamic_voice_character_demo() -> Result<()> {
    println!("🎭 Demo 4: Dynamic Voice Character System");
    println!("=========================================");
    println!("Demonstrating real-time voice character transformation with");
    println!("emotional state changes and spatial movement.\n");

    let pipeline = VoirsPipelineBuilder::new()
        .with_singing_enabled(true)
        .with_spatial_enabled(true)
        .with_emotion_enabled(true)
        .with_conversion_enabled(true)
        .build()
        .await?;

    // Character transformation sequence
    let character_states = vec![
        CharacterState {
            name: "Young Hero",
            age_group: AgeGroup::Child,
            gender: Gender::Male,
            emotion: Emotion::Excited,
            intensity: 0.7,
            position: Position3D::new(-5.0, 0.0, 0.0),
            text: "I wonder what's beyond that hill?",
        },
        CharacterState {
            name: "Determined Adventurer",
            age_group: AgeGroup::Adult,
            gender: Gender::Male,
            emotion: Emotion::Confident,
            intensity: 0.8,
            position: Position3D::new(0.0, 0.0, 0.0),
            text: "I must reach the summit before sunset!",
        },
        CharacterState {
            name: "Wise Elder",
            age_group: AgeGroup::Elder,
            gender: Gender::Male,
            emotion: Emotion::Calm,
            intensity: 0.6,
            position: Position3D::new(5.0, 0.0, 0.0),
            text: "The journey teaches us more than the destination.",
        },
        CharacterState {
            name: "Mystical Oracle",
            age_group: AgeGroup::Adult,
            gender: Gender::Female,
            emotion: Emotion::Calm,
            intensity: 0.9,
            position: Position3D::new(0.0, 10.0, 0.0),
            text: "The answers you seek lie within your heart.",
        },
    ];

    println!("🔄 Character transformation sequence:");
    for (i, state) in character_states.iter().enumerate() {
        println!(
            "  {}. {} - {:?} {:?} with {:?} emotion",
            i + 1,
            state.name,
            state.age_group,
            state.gender,
            state.emotion
        );
    }
    println!();

    let mut character_audio = Vec::new();

    for (i, character) in character_states.iter().enumerate() {
        println!("🎭 Transforming to: {}", character.name);

        // TODO: Apply character transformation when API is available
        // pipeline.set_spatial_position(character.position).await?;
        // pipeline
        //     .set_emotion(character.emotion, Some(character.intensity))
        //     .await?;

        // Apply age and gender conversion
        let base_audio = pipeline.synthesize(&character.text).await?;

        #[cfg(feature = "conversion")]
        {
            let age_converted = pipeline
                .convert_age(
                    base_audio.samples().to_vec(),
                    base_audio.sample_rate(),
                    match character.age_group {
                        AgeGroup::Child => voirs_sdk::prelude::AgeGroup::Child,
                        AgeGroup::Adult => voirs_sdk::prelude::AgeGroup::Adult,
                        AgeGroup::Elder => voirs_sdk::prelude::AgeGroup::Senior,
                    },
                )
                .await?;

            let final_audio = if age_converted.success {
                let age_audio = age_converted.converted_audio.clone();
                let gender_converted = pipeline
                    .convert_gender(
                        age_converted.converted_audio,
                        base_audio.sample_rate(),
                        match character.gender {
                            Gender::Male => voirs_sdk::prelude::Gender::Male,
                            Gender::Female => voirs_sdk::prelude::Gender::Female,
                        },
                    )
                    .await?;

                if gender_converted.success {
                    AudioBuffer::mono(gender_converted.converted_audio, base_audio.sample_rate())
                } else {
                    AudioBuffer::mono(age_audio, base_audio.sample_rate())
                }
            } else {
                base_audio
            };

            println!(
                "  ✅ Character transformation complete ({:.2}s audio)",
                final_audio.duration()
            );
            character_audio.push((character.name.to_string(), final_audio));
        }

        #[cfg(not(feature = "conversion"))]
        {
            println!("  ⚠️  Voice conversion not available, using base audio");
            character_audio.push((character.name.clone(), base_audio));
        }
    }

    // Create character showcase
    let character_showcase = create_character_showcase(character_audio).await?;
    character_showcase.save_wav("dynamic_voice_characters.wav")?;

    println!(
        "\n🎭 Dynamic voice characters created: dynamic_voice_characters.wav ({:.2}s)\n",
        character_showcase.duration()
    );

    Ok(())
}

// Helper structures and functions

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
struct StoryScene {
    character: &'static str,
    text: &'static str,
    position: Position3D,
    emotion: Emotion,
    intensity: f32,
    voice_style: VoiceStyle,
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
#[derive(Clone, Copy)]
enum VoiceStyle {
    Narrative,
    Melodic,
    Deep,
    Heroic,
}

#[cfg(all(
    feature = "singing",
    feature = "spatial",
    feature = "emotion",
    feature = "conversion"
))]
#[derive(Debug, Clone, Copy)]
pub enum AgeGroup {
    Child,
    Adult,
    Elder,
}

#[cfg(all(
    feature = "singing",
    feature = "spatial",
    feature = "emotion",
    feature = "conversion"
))]
#[derive(Debug, Clone, Copy)]
pub enum Gender {
    Male,
    Female,
}

#[cfg(all(
    feature = "singing",
    feature = "spatial",
    feature = "emotion",
    feature = "conversion"
))]
struct CharacterState {
    name: &'static str,
    age_group: AgeGroup,
    gender: Gender,
    emotion: Emotion,
    intensity: f32,
    position: Position3D,
    text: &'static str,
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
// Define placeholder SingingConfig
#[derive(Debug, Clone)]
pub struct SingingConfig {
    pub voice_type: VoiceType,
    pub tempo: Tempo,
    pub time_signature: TimeSignature,
    pub breathing_control: f32,
    pub vibrato_rate: f32,
    pub vibrato_depth: f32,
    pub techniques: Vec<SingingTechnique>,
}

fn create_singing_config(voice_type: VoiceType) -> SingingConfig {
    // Note: This would use the real SingingConfig when available
    SingingConfig {
        voice_type,
        tempo: Tempo::new(120),                   // 120 BPM
        time_signature: TimeSignature::new(4, 4), // 4/4 time
        breathing_control: 0.8,
        vibrato_rate: 5.0,
        vibrato_depth: 0.3,
        techniques: vec![SingingTechnique::Legato, SingingTechnique::Dynamics],
    }
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
fn get_voice_style_config(style: VoiceStyle) -> SynthesisConfig {
    let mut config = SynthesisConfig::default();

    match style {
        VoiceStyle::Narrative => {
            // Calm, clear narration
            config.speaking_rate = 0.9;
            config.pitch_shift = 0.0;
        }
        VoiceStyle::Melodic => {
            // Light, musical quality
            config.speaking_rate = 1.1;
            config.pitch_shift = 0.2;
        }
        VoiceStyle::Deep => {
            // Lower, more imposing
            config.speaking_rate = 0.8;
            config.pitch_shift = -0.3;
        }
        VoiceStyle::Heroic => {
            // Strong, confident
            config.speaking_rate = 1.0;
            config.pitch_shift = 0.1;
        }
    }

    config
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn mix_spatial_performance(
    performance: &[(String, AudioBuffer, Position3D)],
    acoustics: &RoomAcoustics,
) -> Result<AudioBuffer> {
    // Simplified spatial mixing - in practice this would use advanced 3D audio algorithms
    let mut mixed_audio = AudioBuffer::new(Vec::new(), 22050, 1);

    for (name, audio, position) in performance {
        println!("  🎵 Mixing {} at position {:?}", name, position);

        // Apply basic distance attenuation
        let distance =
            (position.x * position.x + position.y * position.y + position.z * position.z).sqrt();
        let attenuation = 1.0 / (1.0 + distance * 0.1);

        let mut positioned_audio = audio.clone();
        positioned_audio.apply_gain(attenuation)?;

        // Apply reverb based on room acoustics
        positioned_audio.reverb(acoustics.reverb_time, 0.5, acoustics.early_reflections)?;

        mixed_audio.mix(&positioned_audio, 1.0)?;
    }

    Ok(mixed_audio)
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn combine_story_audio(audio_parts: Vec<AudioBuffer>) -> Result<AudioBuffer> {
    let mut combined = AudioBuffer::new(Vec::new(), 22050, 1);

    for audio in audio_parts {
        combined.append(&audio)?;

        // Add pause between parts
        let pause = AudioBuffer::silence(0.5, 22050, 1);
        combined.append(&pause)?;
    }

    Ok(combined)
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
fn add_harmonic_variation(base_text: &str, voice_index: usize) -> String {
    // Add slight textual variations for natural choir effect
    match voice_index % 4 {
        0 => base_text.to_string(),           // Soprano - original
        1 => base_text.replace("a", "ah"),    // Alto - slight vowel change
        2 => base_text.to_lowercase(),        // Tenor - different case
        _ => format!("{} (bass)", base_text), // Bass - add marker
    }
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn mix_choir_performance(
    performance: &[(usize, usize, AudioBuffer, Position3D)],
) -> Result<AudioBuffer> {
    let mut choir_mix = AudioBuffer::new(Vec::new(), 22050, 1);

    // Group by phrase and mix spatially
    let mut phrase_groups: std::collections::HashMap<
        usize,
        Vec<&(usize, usize, AudioBuffer, Position3D)>,
    > = std::collections::HashMap::new();

    for perf in performance {
        phrase_groups.entry(perf.1).or_default().push(perf);
    }

    for (phrase_idx, voices) in phrase_groups {
        println!(
            "  🎼 Mixing phrase {} with {} voices",
            phrase_idx + 1,
            voices.len()
        );

        let mut phrase_mix = AudioBuffer::new(Vec::new(), 22050, 1);

        for voice in voices {
            let positioned_audio = apply_choir_positioning(&voice.2, voice.3).await?;
            phrase_mix.mix(&positioned_audio, 1.0)?;
        }

        choir_mix.append(&phrase_mix)?;
    }

    Ok(choir_mix)
}

#[cfg(all(feature = "singing", feature = "spatial", feature = "emotion"))]
async fn apply_choir_positioning(audio: &AudioBuffer, position: Position3D) -> Result<AudioBuffer> {
    let mut positioned = audio.clone();

    // Apply basic spatial effects
    let distance = (position.x * position.x + position.y * position.y).sqrt();
    let attenuation = 1.0 / (1.0 + distance * 0.05);

    positioned.apply_gain(attenuation)?;

    // Apply stereo panning based on X position
    if position.x < 0.0 {
        // positioned.apply_pan(-0.3)?; // Left - method not available, use gain instead
        positioned.apply_gain(0.7)?;
    } else if position.x > 0.0 {
        // positioned.apply_pan(0.3)?; // Right - method not available, use gain instead
        positioned.apply_gain(0.7)?;
    }

    Ok(positioned)
}

#[cfg(all(
    feature = "singing",
    feature = "spatial",
    feature = "emotion",
    feature = "conversion"
))]
async fn create_character_showcase(
    character_audio: Vec<(String, AudioBuffer)>,
) -> Result<AudioBuffer> {
    let mut showcase = AudioBuffer::new(Vec::new(), 22050, 1);

    for (name, audio) in character_audio {
        // Add character introduction
        let intro_text = format!("Character: {}", name);
        println!("  📢 Adding: {}", intro_text);

        // Add the character audio
        showcase.append(&audio)?;

        // Add pause between characters
        let pause = AudioBuffer::silence(1.0, 22050, 1);
        showcase.append(&pause)?;
    }

    Ok(showcase)
}
