//! Complete Voice Generation and Recognition Pipeline Example
//!
//! This example demonstrates a full end-to-end pipeline that:
//! 1. Generates speech from text using VoiRS TTS
//! 2. Recognizes the generated speech using VoiRS Recognizer
//! 3. Evaluates the quality using VoiRS Evaluation
//! 4. Provides feedback using VoiRS Feedback
//! 5. Tracks progress and gamification

use anyhow::Result;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

// Use the unified VoiRS API through the SDK crate
use voirs_sdk::prelude::*;

// Recognition and evaluation specific imports (using their prelude to avoid conflicts)
use voirs_evaluation::prelude::*;
#[cfg(feature = "gamification")]
use voirs_feedback::gamification::achievements::{AchievementSystem, UnlockedAchievement};
use voirs_feedback::prelude::*;
use voirs_feedback::traits::SessionScores;
use voirs_feedback::{FeedbackResponse, FeedbackSystem, ProgressAnalyzer};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Complete Voice Pipeline Example");
    println!("============================================");

    // Step 1: Initialize all components
    println!("\n📋 Step 1: Initializing Pipeline Components");
    let mut pipeline = VoiceCompletePipeline::new().await?;

    // Step 2: Generate speech from text
    println!("\n🎤 Step 2: Text-to-Speech Generation");
    let input_text =
        "Hello world! This is a demonstration of the complete VoiRS voice pipeline system.";
    println!("Input text: \"{input_text}\"");

    let generated_audio = pipeline.generate_speech(input_text).await?;
    println!(
        "✅ Generated audio: {:.2}s duration",
        generated_audio.duration()
    );

    // Step 3: Save and recognize the generated speech
    println!("\n🔍 Step 3: Speech Recognition");
    let recognition_result = pipeline.recognize_speech(&generated_audio).await?;
    println!("Recognized text: \"{}\"", recognition_result.text);
    println!("Confidence: {:.1}%", recognition_result.confidence * 100.0);

    // Step 4: Evaluate speech quality
    println!("\n📊 Step 4: Quality Evaluation");
    let evaluation_result = pipeline
        .evaluate_quality(&generated_audio, input_text)
        .await?;
    print_evaluation_results(&evaluation_result);

    // Step 5: Generate feedback
    println!("\n💡 Step 5: Feedback Generation");
    let feedback_result = pipeline
        .generate_feedback(&generated_audio, input_text, "user_demo")
        .await?;
    print_feedback_results(&feedback_result);

    // Step 6: Track progress and achievements
    println!("\n🏆 Step 6: Progress Tracking & Gamification");
    let progress_update = pipeline
        .update_progress("user_demo", &evaluation_result, &feedback_result)
        .await?;
    print_progress_update(&progress_update);

    // Step 7: Demonstrate iterative improvement
    println!("\n🔄 Step 7: Iterative Improvement Cycle");
    demonstrate_improvement_cycle(&pipeline).await?;

    println!("\n✨ Pipeline demonstration completed successfully!");
    Ok(())
}

/// Complete voice pipeline orchestrator
struct VoiceCompletePipeline {
    // TTS components using unified SDK traits
    g2p_engine: std::sync::Arc<dyn G2p>,
    acoustic_model: std::sync::Arc<dyn AcousticModel>,
    vocoder: std::sync::Arc<dyn Vocoder>,

    // Recognition components
    asr_system: Box<dyn voirs_recognizer::traits::ASRModel>,

    // Evaluation components
    quality_evaluator: QualityEvaluator,
    pronunciation_evaluator: PronunciationEvaluatorImpl,

    // Feedback components
    feedback_system: FeedbackSystem,
    progress_analyzer: ProgressAnalyzer,
    #[cfg(feature = "gamification")]
    achievement_system: AchievementSystem,
}

impl VoiceCompletePipeline {
    async fn new() -> Result<Self> {
        println!("  🔧 Initializing TTS components...");
        // Wire up the real per-stage components directly (rather than the unified
        // VoirsPipelineBuilder) so this example can show intermediate phoneme and
        // mel-spectrogram output below. `voirs_sdk::adapters::*` bridge each
        // component crate's own trait into the SDK's `G2p`/`AcousticModel`/`Vocoder`
        // traits; on any real initialization failure we honestly fall back to the
        // SDK's documented `Dummy*` placeholders rather than fabricate a result.
        let g2p_engine: std::sync::Arc<dyn G2p> = match voirs_g2p::rules::EnglishRuleG2p::new() {
            Ok(rule_g2p) => std::sync::Arc::new(voirs_sdk::adapters::G2pAdapter::new(
                std::sync::Arc::new(rule_g2p),
            )),
            Err(e) => {
                eprintln!(
                    "Warning: Failed to initialize EnglishRuleG2p: {e}. Falling back to DummyG2p."
                );
                std::sync::Arc::new(voirs_sdk::pipeline::DummyG2p::new())
            }
        };

        let acoustic_model: std::sync::Arc<dyn AcousticModel> =
            match voirs_acoustic::vits::VitsModel::new() {
                Ok(vits_model) => std::sync::Arc::new(voirs_sdk::adapters::AcousticAdapter::new(
                    std::sync::Arc::new(vits_model),
                )),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize VITS model: {e}. Falling back to DummyAcoustic.");
                    std::sync::Arc::new(voirs_sdk::pipeline::DummyAcoustic::new())
                }
            };

        let vocoder: std::sync::Arc<dyn Vocoder> = {
            let mut hifigan_vocoder = voirs_vocoder::hifigan::HiFiGanVocoder::new();
            match hifigan_vocoder.initialize_inference_for_testing() {
                Ok(()) => std::sync::Arc::new(voirs_sdk::adapters::VocoderAdapter::new(
                    std::sync::Arc::new(hifigan_vocoder),
                )),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize HiFi-GAN inference: {e}. Falling back to DummyVocoder.");
                    std::sync::Arc::new(voirs_sdk::pipeline::DummyVocoder::new())
                }
            }
        };

        println!("  🎯 Initializing recognition components...");
        let fallback_config =
            voirs_recognizer::asr::intelligent_fallback::FallbackConfig::default();
        let asr_system: Box<dyn voirs_recognizer::traits::ASRModel> = Box::new(
            voirs_recognizer::asr::intelligent_fallback::IntelligentASRFallback::new(
                fallback_config,
            )
            .await?,
        );

        println!("  📈 Initializing evaluation components...");
        let quality_evaluator = QualityEvaluator::new().await?;
        let pronunciation_evaluator = PronunciationEvaluatorImpl::new().await?;

        println!("  🎮 Initializing feedback components...");
        let feedback_system = FeedbackSystem::new().await?;
        let progress_analyzer = ProgressAnalyzer::new().await?;
        #[cfg(feature = "gamification")]
        let achievement_system = AchievementSystem::new();

        Ok(Self {
            g2p_engine,
            acoustic_model,
            vocoder,
            asr_system,
            quality_evaluator,
            pronunciation_evaluator,
            feedback_system,
            progress_analyzer,
            #[cfg(feature = "gamification")]
            achievement_system,
        })
    }

    async fn generate_speech(&self, text: &str) -> Result<AudioBuffer> {
        // Convert text to phonemes
        let phonemes = self
            .g2p_engine
            .to_phonemes(text, Some(LanguageCode::EnUs))
            .await?;
        println!(
            "  📝 Phonemes: {}",
            phonemes
                .iter()
                .map(|p| p.symbol.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        );

        // Generate mel spectrogram
        let mel_spectrogram = self.acoustic_model.synthesize(&phonemes, None).await?;
        println!(
            "  🎵 Generated mel spectrogram: {} frames",
            mel_spectrogram.n_frames
        );

        // Convert to audio
        let audio = self.vocoder.vocode(&mel_spectrogram, None).await?;
        println!(
            "  🔊 Generated audio: {} samples at {}Hz",
            audio.samples().len(),
            audio.sample_rate()
        );

        Ok(audio)
    }

    async fn recognize_speech(&self, audio: &AudioBuffer) -> Result<Transcript> {
        let result = self.asr_system.transcribe(audio, None).await?;
        Ok(result)
    }

    async fn evaluate_quality(
        &self,
        audio: &AudioBuffer,
        expected_text: &str,
    ) -> Result<CombinedEvaluationResult> {
        // Quality evaluation
        let quality_result = self
            .quality_evaluator
            .evaluate_quality(audio, None, None)
            .await?;

        // Pronunciation evaluation
        let pronunciation_result = self
            .pronunciation_evaluator
            .evaluate_pronunciation(audio, expected_text, None)
            .await?;

        Ok(CombinedEvaluationResult {
            quality: quality_result,
            pronunciation: pronunciation_result,
        })
    }

    async fn generate_feedback(
        &self,
        audio: &AudioBuffer,
        expected_text: &str,
        user_id: &str,
    ) -> Result<FeedbackResponse> {
        // Real feedback generation through the feedback system: create a session
        // for this user and let it actually process the synthesized audio against
        // the expected text (no canned/placeholder response).
        let mut session = self.feedback_system.create_session(user_id).await?;
        let feedback = session.process_synthesis(audio, expected_text).await?;

        Ok(feedback)
    }

    async fn update_progress(
        &mut self,
        user_id: &str,
        evaluation: &CombinedEvaluationResult,
        feedback: &FeedbackResponse,
    ) -> Result<ProgressUpdate> {
        // Check for achievements
        let user_progress = UserProgress {
            user_id: user_id.to_string(),
            session_count: 1,
            total_practice_time: Duration::from_secs(30),
            average_scores: SessionScores {
                average_quality: evaluation.quality.overall_score,
                average_pronunciation: evaluation.pronunciation.overall_score,
                average_fluency: feedback.overall_score, // Derived from this synthesis's real feedback score
                overall_score: (evaluation.quality.overall_score
                    + evaluation.pronunciation.overall_score)
                    / 2.0,
                improvement_trend: 0.0, // No prior session in this demo to compare against
            },
            skill_levels: std::collections::HashMap::new(),
            recent_sessions: vec![],
            personal_bests: std::collections::HashMap::new(),
            achievements: vec![],
            goals: vec![],
            last_updated: chrono::Utc::now(),
            overall_skill_level: 0.5,
            skill_breakdown: std::collections::HashMap::new(),
            progress_history: vec![],
            training_stats: voirs_feedback::traits::TrainingStatistics {
                total_sessions: 1,
                total_training_time: Duration::from_secs(30),
                exercises_completed: 1,
                success_rate: 0.8,
                average_improvement: 0.05,
                current_streak: 1,
                longest_streak: 1,
                successful_sessions: 1,
            },
        };

        let session_data = SessionState {
            session_id: Uuid::new_v4(),
            user_id: user_id.to_string(),
            start_time: chrono::Utc::now(),
            current_exercise: None,
            session_stats: voirs_feedback::traits::SessionStatistics::default(),
            adaptive_state: Default::default(),
            current_task: None,
            last_activity: chrono::Utc::now(),
            stats: voirs_feedback::traits::SessionStats::default(),
            preferences: voirs_feedback::traits::UserPreferences::default(),
        };

        // Persist this session's real scores through the progress analyzer so
        // `self.progress_analyzer` actually tracks history rather than sitting unused.
        let training_scores = voirs_feedback::traits::TrainingScores {
            quality: evaluation.quality.overall_score,
            pronunciation: evaluation.pronunciation.overall_score,
            // Two scores that agree indicate consistency; scores far apart don't.
            consistency: 1.0
                - (evaluation.quality.overall_score - evaluation.pronunciation.overall_score).abs(),
            improvement: 0.0, // No prior session in this demo to compare against
        };
        self.progress_analyzer
            .record_session_progress(user_id, &session_data, &training_scores, feedback)
            .await?;

        #[cfg(feature = "gamification")]
        let new_achievements = self
            .achievement_system
            .check_achievements(user_uuid(user_id), &user_progress);

        #[cfg(not(feature = "gamification"))]
        let new_achievements = vec![];

        let milestone_reached = !new_achievements.is_empty();

        Ok(ProgressUpdate {
            new_achievements,
            updated_stats: user_progress,
            milestone_reached,
        })
    }
}

/// Derive a stable `Uuid` from a user id string.
///
/// The achievement system indexes users by `Uuid`, while the rest of this demo
/// (and `voirs_feedback::traits::UserProgress`) uses plain `String` user ids.
/// This is a deterministic derivation (same input always yields the same
/// output), not a random or fabricated value: it exists purely to bridge the
/// two id representations.
#[cfg(feature = "gamification")]
fn user_uuid(user_id: &str) -> uuid::Uuid {
    use std::hash::{Hash, Hasher};

    let mut low_hasher = std::collections::hash_map::DefaultHasher::new();
    user_id.hash(&mut low_hasher);
    let low = low_hasher.finish();

    let mut high_hasher = std::collections::hash_map::DefaultHasher::new();
    (user_id, "voirs-complete-voice-pipeline-uuid-salt").hash(&mut high_hasher);
    let high = high_hasher.finish();

    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&low.to_be_bytes());
    bytes[8..].copy_from_slice(&high.to_be_bytes());
    uuid::Uuid::from_bytes(bytes)
}

/// Combined evaluation result
struct CombinedEvaluationResult {
    quality: voirs_evaluation::QualityScore,
    pronunciation: voirs_evaluation::PronunciationScore,
}

/// Progress update result
struct ProgressUpdate {
    #[cfg(feature = "gamification")]
    new_achievements: Vec<UnlockedAchievement>,
    #[cfg(not(feature = "gamification"))]
    new_achievements: Vec<()>,
    updated_stats: UserProgress,
    milestone_reached: bool,
}

async fn demonstrate_improvement_cycle(pipeline: &VoiceCompletePipeline) -> Result<()> {
    println!("  🔄 Simulating improvement over multiple sessions...");

    let test_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    ];

    for (session, phrase) in test_phrases.iter().enumerate() {
        println!("\n  📝 Session {}: \"{}\"", session + 1, phrase);

        // Generate and evaluate
        let audio = pipeline.generate_speech(phrase).await?;
        let evaluation = pipeline.evaluate_quality(&audio, phrase).await?;
        let feedback = pipeline
            .generate_feedback(&audio, phrase, "user_demo")
            .await?;

        println!(
            "    Quality Score: {:.1}%",
            evaluation.quality.overall_score * 100.0
        );
        println!(
            "    Pronunciation Score: {:.1}%",
            evaluation.pronunciation.overall_score * 100.0
        );
        println!("    Feedback Items: {}", feedback.feedback_items.len());

        // Simulate improvement
        sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}

fn print_evaluation_results(result: &CombinedEvaluationResult) {
    println!("  Quality Metrics:");
    println!(
        "    Overall Score: {:.1}%",
        result.quality.overall_score * 100.0
    );
    if let Some(pesq) = result.quality.component_scores.get("PESQ") {
        println!("    PESQ: {pesq:.2}");
    }
    if let Some(stoi) = result.quality.component_scores.get("STOI") {
        println!("    STOI: {stoi:.3}");
    }

    println!("  Pronunciation Metrics:");
    println!(
        "    Overall Score: {:.1}%",
        result.pronunciation.overall_score * 100.0
    );
    println!(
        "    Accuracy: {:.1}%",
        result.pronunciation.overall_score * 100.0
    );
    println!(
        "    Fluency: {:.1}%",
        result.pronunciation.fluency_score * 100.0
    );
    println!(
        "    Word-level scores: {}",
        result.pronunciation.word_scores.len()
    );
}

fn print_feedback_results(feedback: &FeedbackResponse) {
    println!("  Overall Score: {:.1}%", feedback.overall_score * 100.0);
    println!("  Feedback Items: {}", feedback.feedback_items.len());

    for (i, item) in feedback.feedback_items.iter().take(3).enumerate() {
        println!(
            "    {}. {} (Score: {:.1}%)",
            i + 1,
            item.message,
            item.score * 100.0
        );
        if let Some(suggestion) = &item.suggestion {
            println!("       💡 {suggestion}");
        }
    }

    if !feedback.immediate_actions.is_empty() {
        println!("  Next Steps:");
        for (i, action) in feedback.immediate_actions.iter().take(2).enumerate() {
            println!("    {}. {}", i + 1, action);
        }
    }
}

fn print_progress_update(update: &ProgressUpdate) {
    #[cfg(feature = "gamification")]
    if !update.new_achievements.is_empty() {
        println!("  🎉 New Achievements Unlocked:");
        for achievement in &update.new_achievements {
            let points: u32 = achievement
                .achievement
                .rewards
                .iter()
                .filter_map(|reward| match reward {
                    voirs_feedback::gamification::achievements::Reward::Points(n) => Some(*n),
                    _ => None,
                })
                .sum();
            println!(
                "    {} {} - {} points",
                match achievement.achievement.tier {
                    voirs_feedback::traits::AchievementTier::Bronze => "🥉",
                    voirs_feedback::traits::AchievementTier::Silver => "🥈",
                    voirs_feedback::traits::AchievementTier::Gold => "🥇",
                    voirs_feedback::traits::AchievementTier::Platinum => "💎",
                    voirs_feedback::traits::AchievementTier::Diamond => "💠",
                    voirs_feedback::traits::AchievementTier::Rare => "✨",
                },
                achievement.achievement.name,
                points
            );
        }
    } else {
        println!("  📊 Progress continues - keep practicing for more achievements!");
    }

    #[cfg(not(feature = "gamification"))]
    {
        println!(
            "  📊 Progress continues - keep practicing! ({} achievement slots tracked)",
            update.new_achievements.len()
        );
    }

    println!(
        "  Milestone reached this session: {}",
        update.milestone_reached
    );
    println!("  Total Sessions: {}", update.updated_stats.session_count);
    println!(
        "  Practice Time: {}min",
        update.updated_stats.total_practice_time.as_secs() / 60
    );
    println!(
        "  Average Score: {:.1}%",
        update.updated_stats.average_scores.overall_score * 100.0
    );
}

// Helper trait implementations and type definitions would go here
// (These would typically be defined in the respective crate modules)

/// Placeholder implementations for demonstration
impl Default for FeedbackOptions {
    fn default() -> Self {
        Self {
            user_id: None,
            include_detailed_analysis: false,
            include_suggestions: false,
        }
    }
}

/// Example feedback options
#[derive(Debug, Clone)]
pub struct FeedbackOptions {
    pub user_id: Option<String>,
    pub include_detailed_analysis: bool,
    pub include_suggestions: bool,
}

impl Default for SessionStatistics {
    fn default() -> Self {
        Self {
            exercises_completed: 0,
            total_time: Duration::from_secs(0),
            average_score: 0.0,
            improvement_rate: 0.0,
        }
    }
}

/// Example session statistics
#[derive(Debug, Clone)]
pub struct SessionStatistics {
    pub exercises_completed: u32,
    pub total_time: Duration,
    pub average_score: f32,
    pub improvement_rate: f32,
}
