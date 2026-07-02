/*!
 * AI Integration Example
 *
 * This example demonstrates how VoiRS integrates with AI systems:
 * - Large Language Model (LLM) integration
 * - Conversational AI with voice synthesis
 * - Real-time voice chat systems
 * - AI-powered content generation with speech
 * - Multi-turn dialogue with emotion and personality
 * - Voice-based AI assistants
 *
 * Features demonstrated:
 * - LLM text generation with voice synthesis
 * - Conversational AI with context awareness
 * - Emotion-aware AI responses
 * - Real-time voice chat with AI
 * - Multi-modal AI interactions
 * - AI personality simulation through voice
 * - Content generation and narration
 *
 * Run with: cargo run --example ai_integration_example
 *
 * The implementation is split by logical concern under `ai_integration_example/`:
 * - `error`: shared error type
 * - `llm`: LLM client/config, usage tracking, response caching
 * - `conversation`: conversation/dialogue state management
 * - `personality`: personality profiling and emotion analysis
 * - `voice_synthesis`: AI-enhanced voice synthesis
 * - `voice_chat`: real-time voice chat/streaming session handling
 * - `content_generation`: AI content generation and narration
 * - `system`: top-level orchestration (`AIVoiceSystem`)
 */

// NOTE: this file is registered as the `path` of a `[[example]]` target, which
// makes it a crate root in its own right. Crate roots resolve bare `mod foo;`
// declarations directly inside their *containing* directory (`examples/foo.rs`),
// not inside a same-named sibling directory (`examples/ai_integration_example/foo.rs`)
// -- that convention only applies to non-root modules. `#[path = "..."]` is the
// standard, documented way to keep this file at its original top-level path while
// still storing its submodules under `ai_integration_example/`.
//
// The submodules are declared `pub mod` (rather than private `mod`) so that the
// `pub` types/fns they contain stay reachable from the crate root exactly as they
// were in the original single-file layout -- a private `mod` here would make
// rustc's dead-code reachability analysis treat everything inside as effectively
// private, spuriously flagging never-externally-read `pub` fields (on the plain
// structs that lack `Debug`/`Clone` derives) as dead code.
#[path = "ai_integration_example/content_generation.rs"]
pub mod content_generation;
#[path = "ai_integration_example/conversation.rs"]
pub mod conversation;
#[path = "ai_integration_example/error.rs"]
pub mod error;
#[path = "ai_integration_example/llm.rs"]
pub mod llm;
#[path = "ai_integration_example/personality.rs"]
pub mod personality;
#[path = "ai_integration_example/system.rs"]
pub mod system;
#[path = "ai_integration_example/voice_chat.rs"]
pub mod voice_chat;
#[path = "ai_integration_example/voice_synthesis.rs"]
pub mod voice_synthesis;

// These are only exercised by the test suite below; gating them behind
// `#[cfg(test)]` keeps the non-test build free of unused-import warnings.
#[cfg(test)]
use content_generation::{ContentRequest, ContentType};
#[cfg(test)]
use personality::{EmotionAnalysis, EmotionScore, PersonalityProfile, SentimentScore};
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::time::Duration;
use system::AIVoiceSystem;
use uuid::Uuid;

/// Main demonstration function
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 VoiRS AI Integration Example");
    println!("===============================\n");

    // Create the AI voice system
    let ai_system = AIVoiceSystem::new();

    // Demonstrate AI personalities with different voices
    ai_system.demonstrate_ai_personalities().await?;

    // Demonstrate conversational AI scenarios
    ai_system.demonstrate_conversation_scenarios().await?;

    // Demonstrate content generation
    ai_system.demonstrate_content_generation().await?;

    // Demonstrate real-time voice chat setup
    println!("🎙️ === Real-Time Voice Chat Demo ===\n");
    let chat_session_id = ai_system.start_voice_chat("demo_user").await?;
    println!("✅ Voice chat session initiated: {}", chat_session_id);

    // Simulate some chat interactions
    let conversation_id = Uuid::new_v4(); // Mock conversation ID
    let chat_messages = vec![
        "Hello, can you help me with a quick question?",
        "I'm trying to understand how AI voice synthesis works.",
        "This is really fascinating technology!",
    ];

    for message in chat_messages {
        println!("👤 User: {}", message);
        if let Ok(response) = ai_system
            .process_user_message(conversation_id, message)
            .await
        {
            println!("🤖 AI: {}", response.content);
            if let Some(voice_meta) = response.voice_metadata {
                println!(
                    "   🎵 Synthesized in {:?} (quality: {:.2})",
                    voice_meta.synthesis_time, voice_meta.quality_metrics.naturalness_score
                );
            }
        }
        println!();
    }

    // Performance summary
    println!("📊 === AI System Performance Summary ===\n");
    println!("🧠 LLM Engine:");
    println!(
        "   • Average Response Time: {:?}",
        ai_system.llm_engine.performance_metrics.average_latency
    );
    println!(
        "   • Success Rate: {:.1}%",
        ai_system.llm_engine.performance_metrics.success_rate * 100.0
    );
    println!(
        "   • Requests per Second: {:.1}",
        ai_system.llm_engine.performance_metrics.requests_per_second
    );

    println!("\n🎵 Voice Synthesis:");
    println!(
        "   • Target Latency: {:?}",
        ai_system
            .voice_synthesizer
            .synthesis_optimizer
            .performance_targets
            .max_latency
    );
    println!(
        "   • Quality Target: {:.1}",
        ai_system
            .voice_synthesizer
            .synthesis_optimizer
            .performance_targets
            .min_quality_score
    );
    println!(
        "   • Audio Format: {} Hz, {} channels",
        ai_system
            .voice_synthesizer
            .synthesis_optimizer
            .quality_settings
            .sample_rate,
        ai_system
            .voice_synthesizer
            .synthesis_optimizer
            .quality_settings
            .channels
    );

    println!("\n🎭 Personality Engine:");
    println!(
        "   • Analysis Confidence: {:.1}%",
        ai_system
            .personality_engine
            .trait_analyzer
            .analysis_confidence
            * 100.0
    );
    println!(
        "   • Dynamic Adaptation: {}",
        ai_system
            .personality_engine
            .personality_voice_mapper
            .dynamic_adjustment
    );

    println!("\n🔄 Real-Time Chat:");
    println!(
        "   • Buffer Size: {} bytes",
        ai_system
            .chat_manager
            .voice_streamer
            .streaming_config
            .buffer_size
    );
    println!(
        "   • Latency Target: {:?}",
        ai_system
            .chat_manager
            .voice_streamer
            .streaming_config
            .latency_target
    );
    println!(
        "   • Quality Level: {}/10",
        ai_system
            .chat_manager
            .voice_streamer
            .streaming_config
            .quality_level
    );

    println!("\n✨ AI Integration example completed successfully!");
    println!("🎯 This example demonstrates:");
    println!("   • LLM integration with voice synthesis");
    println!("   • Personality-aware AI conversations");
    println!("   • Emotion detection and voice adaptation");
    println!("   • Real-time voice chat capabilities");
    println!("   • AI content generation with narration");
    println!("   • Multi-modal AI interactions");
    println!("   • Performance optimization for AI systems");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ai_voice_system_creation() {
        let system = AIVoiceSystem::new();
        // Verify system components are initialized
        assert_eq!(system.llm_engine.model_config.model_name, "gpt-4");
        assert!(
            system
                .voice_synthesizer
                .emotion_predictor
                .confidence_threshold
                > 0.0
        );
    }

    #[tokio::test]
    async fn test_conversation_start() {
        let system = AIVoiceSystem::new();
        let result = system.start_conversation("test_user", "Hello").await;
        assert!(result.is_ok());

        let conversation = result.unwrap();
        assert_eq!(conversation.user_id, "test_user");
        assert!(!conversation.messages.is_empty()); // Should have initial message
    }

    #[tokio::test]
    async fn test_emotion_analysis() {
        let system = AIVoiceSystem::new();

        // Test positive emotion
        let happy_analysis = system
            .personality_engine
            .analyze_message_emotion("I'm so happy and excited!")
            .await
            .unwrap();
        assert!(happy_analysis.sentiment.polarity > 0.0);
        assert!(happy_analysis
            .detected_emotions
            .iter()
            .any(|e| e.emotion == "joy" || e.emotion == "excitement"));

        // Test negative emotion
        let sad_analysis = system
            .personality_engine
            .analyze_message_emotion("I'm feeling really sad and upset.")
            .await
            .unwrap();
        assert!(sad_analysis.sentiment.polarity < 0.0);
        assert!(sad_analysis
            .detected_emotions
            .iter()
            .any(|e| e.emotion == "sadness"));

        // Test neutral emotion
        let neutral_analysis = system
            .personality_engine
            .analyze_message_emotion("This is a normal sentence.")
            .await
            .unwrap();
        assert!(neutral_analysis
            .detected_emotions
            .iter()
            .any(|e| e.emotion == "neutral"));
    }

    #[tokio::test]
    async fn test_personality_profiles() {
        let system = AIVoiceSystem::new();

        let professor = system
            .create_personality_profile("Professor")
            .await
            .unwrap();
        assert_eq!(professor.voice_style, "authoritative");
        assert!(professor.traits.get("openness").unwrap() > &0.8);

        let friend = system.create_personality_profile("Friend").await.unwrap();
        assert_eq!(friend.voice_style, "friendly");
        assert!(friend.traits.get("extraversion").unwrap() > &0.8);
    }

    #[tokio::test]
    async fn test_content_generation() {
        let system = AIVoiceSystem::new();

        let story_request = ContentRequest {
            title: "Test Story".to_string(),
            content_type: ContentType::Story,
            target_audience: "children".to_string(),
            length_target: 100,
            style_preferences: vec!["whimsical".to_string()],
            voice_style: "storyteller".to_string(),
        };

        let result = system.generate_narrated_content(story_request).await;
        assert!(result.is_ok());

        let content = result.unwrap();
        assert!(!content.content.is_empty());
        assert!(!content.voice_segments.is_empty());
        assert!(content.total_duration > Duration::from_secs(0));
    }

    #[tokio::test]
    async fn test_voice_synthesis() {
        let system = AIVoiceSystem::new();

        let emotion_analysis = EmotionAnalysis {
            detected_emotions: vec![EmotionScore {
                emotion: "joy".to_string(),
                score: 0.8,
                confidence: 0.9,
            }],
            sentiment: SentimentScore {
                polarity: 0.7,
                magnitude: 0.6,
                confidence: 0.8,
            },
            emotion_progression: vec![],
        };

        let result = system
            .voice_synthesizer
            .synthesize_response(
                "Hello, this is a test message!",
                &emotion_analysis,
                Uuid::new_v4(),
            )
            .await;

        assert!(result.is_ok());
        let voice_meta = result.unwrap();
        assert!(voice_meta.synthesis_time > Duration::from_millis(0));
        assert!(voice_meta.audio_duration > Duration::from_millis(0));
        assert!(voice_meta.quality_metrics.naturalness_score > 0.0);
    }

    #[tokio::test]
    async fn test_llm_response_generation() {
        let system = AIVoiceSystem::new();

        let emotion_analysis = EmotionAnalysis {
            detected_emotions: vec![EmotionScore {
                emotion: "neutral".to_string(),
                score: 0.6,
                confidence: 0.7,
            }],
            sentiment: SentimentScore {
                polarity: 0.0,
                magnitude: 0.3,
                confidence: 0.7,
            },
            emotion_progression: vec![],
        };

        let response = system
            .llm_engine
            .generate_response(
                Uuid::new_v4(),
                "Can you help me with something?",
                &emotion_analysis,
            )
            .await;

        assert!(response.is_ok());
        let response_text = response.unwrap();
        assert!(!response_text.is_empty());
        assert!(response_text.to_lowercase().contains("help"));
    }

    #[tokio::test]
    async fn test_voice_chat_session() {
        let system = AIVoiceSystem::new();

        let session_id = system.start_voice_chat("test_user").await;
        assert!(session_id.is_ok());

        let session_uuid = session_id.unwrap();

        // Verify session was created
        let active_sessions = system.chat_manager.active_sessions.read().await;
        assert!(active_sessions.contains_key(&session_uuid));
    }

    #[tokio::test]
    async fn test_narration_analysis() {
        let system = AIVoiceSystem::new();

        let content =
            "This is an exciting story about amazing adventures and wonderful discoveries.";
        let analysis = system.analyze_content_for_narration(content).await.unwrap();

        assert!(analysis.word_count > 0);
        assert!(analysis.sentence_count > 0);
        assert!(analysis.estimated_duration > Duration::from_secs(0));
        assert!(analysis
            .emotion_profile
            .emotion_distribution
            .contains_key("excitement"));
        assert!(analysis.quality_score > 0.0);
    }

    #[tokio::test]
    async fn test_voice_settings_selection() {
        let system = AIVoiceSystem::new();

        let personality = PersonalityProfile {
            traits: HashMap::from([
                ("extraversion".to_string(), 0.8),
                ("openness".to_string(), 0.7),
                ("conscientiousness".to_string(), 0.9),
            ]),
            voice_style: "energetic".to_string(),
            interaction_style: "motivational".to_string(),
            emotional_expressiveness: 0.9,
        };

        let settings = system
            .select_voice_for_personality(&personality)
            .await
            .unwrap();

        assert!(settings.speed > 1.0); // Should be faster for extraverted personality
        assert!(settings.emotion_intensity > 0.8); // Should match high expressiveness
        assert!(settings.breathing_patterns); // Should be true for high conscientiousness
    }
}
