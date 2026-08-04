//! Singing Synthesis Example (API Compatible)
//!
//! This example demonstrates the singing synthesis capabilities of VoiRS SDK.

use voirs_singing::types::Articulation;
use voirs_singing::{
    Expression, MusicalNote, NoteEvent, SingingConfigBuilder, SynthesisEngine, VoiceType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 VoiRS Singing Synthesis Example");
    println!("==================================");

    // Create singing config with proper API
    let singing_config = SingingConfigBuilder::new().enabled(true).build();

    println!("✅ Created singing configuration");

    // Create a simple note event
    let note_event = NoteEvent {
        note: "C".to_string(),
        octave: 4,
        frequency: 261.63, // C4 frequency
        duration: 1.0,
        velocity: 0.8,
        vibrato: 0.1,
        lyric: Some("La".to_string()),
        phonemes: vec!["l".to_string(), "a".to_string()],
        expression: Expression::Happy,
        timing_offset: 0.0,
        breath_before: 0.0,
        legato: false,
        articulation: Articulation::Normal,
    };

    // Create a musical note with proper API
    let musical_note = MusicalNote::new(note_event, 0.0, 1.0);

    println!(
        "✅ Created musical note: {} at {:.2}Hz",
        musical_note.event.note, musical_note.event.frequency
    );

    // Create synthesis engine (construction is the point of this demo; a full
    // synthesis run requires a musical score and is shown in other examples)
    let _synthesis_engine = SynthesisEngine::new(singing_config);

    println!("✅ Created synthesis engine");

    // Demonstrate voice characteristics
    demonstrate_voice_characteristics();

    println!("🎵 Singing synthesis example completed successfully!");
    Ok(())
}

fn demonstrate_voice_characteristics() {
    println!("\n🎤 Voice Type Demonstrations:");

    for voice_type in [VoiceType::Soprano, VoiceType::Tenor, VoiceType::Bass] {
        // Each voice type gets its own enabled singing configuration; this demo
        // only shows that construction succeeds, so the built config is unused.
        let _config = SingingConfigBuilder::new().enabled(true).build();

        println!("  {:?} voice configuration created", voice_type);
    }

    println!("\n🎭 Expression Demonstrations:");
    for expression in [Expression::Happy, Expression::Sad, Expression::Passionate] {
        println!("  {:?} expression available", expression);
    }
}
