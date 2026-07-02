/*!
 * Educational Tools Example - Language Learning with VoiRS
 *
 * This example demonstrates how VoiRS can be integrated into educational applications:
 * - Interactive language learning systems
 * - Pronunciation training and feedback
 * - Adaptive learning with voice guidance
 * - Multi-language voice synthesis for education
 * - Speech assessment and correction
 * - Interactive vocabulary training
 * - Gamified learning experiences with voice
 *
 * Features demonstrated:
 * - Pronunciation analysis and feedback
 * - Interactive conversation practice
 * - Vocabulary building with audio reinforcement
 * - Grammar lessons with voice explanation
 * - Cultural context learning
 * - Accessibility features for diverse learners
 * - Progress tracking and adaptive difficulty
 * - Multi-modal learning reinforcement
 *
 * Run with: cargo run --example educational_tools_example
 *
 * This example is organized into focused sibling modules (see the
 * `educational_tools_example/` directory) grouped by learning concern:
 * - `types`: shared data structures (lessons, profiles, preferences, ...)
 * - `errors`: the `EducationalVoiceError` type
 * - `system`: the `EducationalVoiceSystem` facade and its subsystems
 * - `lesson`: lesson creation and session management
 * - `pronunciation`: pronunciation assessment and feedback
 * - `adaptive`: adaptive/difficulty-aware content generation
 * - `vocabulary`: interactive vocabulary building
 * - `cultural`: cultural-context learning scenarios
 * - `gamification` / `gamification_types`: challenges, achievements, and
 *   motivation systems
 */

// NOTE: this example binary's crate root is this file itself (see
// `examples/Cargo.toml`'s `[[example]] path = "educational_tools_example.rs"`
// entry). Crate-root files resolve bare `mod foo;` declarations to a
// *sibling* `foo.rs` next to the crate root (the same rule `main.rs`/`lib.rs`
// follow), not into a subdirectory named after the crate root itself. The
// explicit `#[path = ...]` attributes below keep the implementation split
// across the tidy `educational_tools_example/` subdirectory regardless.
// These are `pub mod` (rather than plain `mod`) so that every item's
// effective visibility chain reaches all the way back to the crate root,
// exactly as it did when everything lived directly in this one file. This
// matters for more than style: rustc's `dead_code` lint only exempts an
// item from "never constructed/read" analysis when it is part of the
// crate's publicly-reachable surface, which requires every module on its
// path (not just the item itself) to be `pub`. With plain (private) `mod`
// declarations here, the many `pub struct`/`pub enum` data types nested one
// level down would lose that exemption and spuriously trigger `dead_code`
// warnings for fields that are only ever read from the `#[cfg(test)]`
// module (which isn't compiled under a plain `cargo check`/`clippy` run).
#[path = "educational_tools_example/adaptive.rs"]
pub mod adaptive;
#[path = "educational_tools_example/cultural.rs"]
pub mod cultural;
#[path = "educational_tools_example/errors.rs"]
pub mod errors;
#[path = "educational_tools_example/gamification.rs"]
pub mod gamification;
#[path = "educational_tools_example/gamification_types.rs"]
pub mod gamification_types;
#[path = "educational_tools_example/lesson.rs"]
pub mod lesson;
#[path = "educational_tools_example/pronunciation.rs"]
pub mod pronunciation;
#[path = "educational_tools_example/system.rs"]
pub mod system;
#[path = "educational_tools_example/types.rs"]
pub mod types;
#[path = "educational_tools_example/vocabulary.rs"]
pub mod vocabulary;

#[cfg(test)]
#[path = "educational_tools_example/tests.rs"]
mod tests;

use system::EducationalVoiceSystem;
use types::*;

/// Main demonstration function
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 VoiRS Educational Tools Example - Language Learning");
    println!("====================================================\n");

    // Create the educational voice system
    let edu_system = EducationalVoiceSystem::new();

    // Demonstrate pronunciation training
    edu_system.demonstrate_pronunciation_training().await?;

    // Demonstrate vocabulary learning
    edu_system.demonstrate_vocabulary_learning().await?;

    // Demonstrate cultural context learning
    edu_system.demonstrate_cultural_learning().await?;

    // Demonstrate gamified learning experience
    edu_system.demonstrate_gamified_learning().await?;

    // Create and run a sample lesson
    println!("\n📖 === Sample Interactive Lesson ===\n");
    let sample_lesson = edu_system
        .create_lesson("Spanish", "Daily Conversations", SkillLevel::Intermediate)
        .await?;
    println!("✅ Created lesson: '{}'", sample_lesson.title);
    println!("   • Level: {:?}", sample_lesson.level.level_name);
    println!("   • Duration: {:?}", sample_lesson.estimated_duration);
    println!("   • Activities: {}", sample_lesson.activities.len());
    println!(
        "   • Skills Targeted: {} skills",
        sample_lesson.skills_targeted.len()
    );

    // Start a learning session
    let learning_session = edu_system
        .start_learning_session("demo_student", sample_lesson)
        .await?;
    println!("\n🎓 Learning session started:");
    println!("   • Session ID: {}", learning_session.session_id);
    println!("   • State: {:?}", learning_session.session_state);
    println!(
        "   • Current Activity: {}",
        learning_session
            .current_activity
            .map(|a| a.title)
            .unwrap_or("None".to_string())
    );

    // Demonstrate pronunciation assessment
    println!("\n🎤 === Pronunciation Assessment Demo ===\n");
    let sample_texts = vec![
        ("English", "The quick brown fox jumps over the lazy dog"),
        ("Spanish", "Me gusta mucho estudiar español todos los días"),
        ("French", "Bonjour, comment allez-vous aujourd'hui?"),
    ];

    for (language, text) in sample_texts {
        let mock_audio = AudioData; // Mock audio data
        let assessment = edu_system
            .assess_pronunciation("demo_student", text, mock_audio, language)
            .await?;

        println!("🌍 {} Assessment for: '{}'", language, text);
        println!("📊 Scores:");
        println!(
            "   • Overall: {:.1}/10",
            assessment.pronunciation_score.overall_score * 10.0
        );
        println!(
            "   • Phonemes: {:.1}/10",
            assessment.pronunciation_score.phoneme_accuracy * 10.0
        );
        println!(
            "   • Rhythm: {:.1}/10",
            assessment.pronunciation_score.rhythm_score * 10.0
        );
        println!(
            "   • Fluency: {:.1}/10",
            assessment.pronunciation_score.fluency_score * 10.0
        );
        println!("💬 Feedback: {}", assessment.feedback.overall_feedback);
        println!(
            "🎯 Suggestions: {}",
            assessment.improvement_suggestions.len()
        );
        println!();
    }

    // System capabilities summary
    println!("📊 === Educational System Capabilities ===\n");
    println!("🎓 Learning Features:");
    println!("   • Multi-language support with cultural adaptation");
    println!("   • Adaptive difficulty based on learner progress");
    println!("   • Comprehensive pronunciation assessment");
    println!("   • Interactive vocabulary building");
    println!("   • Cultural context integration");
    println!("   • Gamified learning experience");

    println!("\n🤖 AI-Powered Features:");
    println!("   • Intelligent content personalization");
    println!("   • Real-time pronunciation feedback");
    println!("   • Adaptive learning path optimization");
    println!("   • Performance prediction and intervention");

    println!("\n♿ Accessibility Features:");
    println!("   • Multiple learning modalities (visual, auditory, kinesthetic)");
    println!("   • Adjustable speech rate and pitch");
    println!("   • Text-to-speech for all content");
    println!("   • Visual and audio feedback options");

    println!("\n📱 Platform Integration:");
    println!("   • Cross-platform learning sessions");
    println!("   • Progress synchronization");
    println!("   • Offline learning capabilities");
    println!("   • Social learning features");

    println!("\n✨ Educational tools example completed successfully!");
    println!("🎯 This example demonstrates:");
    println!("   • Comprehensive language learning system");
    println!("   • AI-powered pronunciation assessment");
    println!("   • Adaptive and personalized learning");
    println!("   • Multi-modal educational content");
    println!("   • Gamification and motivation systems");
    println!("   • Cultural context integration");
    println!("   • Accessibility and inclusion features");
    println!("   • Progress tracking and analytics");

    Ok(())
}
