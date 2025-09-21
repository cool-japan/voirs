//! Real-time Voice Coaching Example
//!
//! This example demonstrates a real-time voice coaching application that:
//! 1. Captures live audio from microphone
//! 2. Provides instant feedback on pronunciation and quality
//! 3. Shows visual progress indicators
//! 4. Adapts difficulty based on user performance
//! 5. Tracks achievements and progress in real-time

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::{interval, sleep};
use uuid::Uuid;

// Audio capture (simulated for example)

// VoiRS imports
use voirs_evaluation::prelude::*;
#[cfg(feature = "gamification")]
use voirs_feedback::gamification::AchievementSystem;
use voirs_feedback::prelude::*;
use voirs_feedback::{Exercise, FocusArea};
use voirs_g2p::ssml::dictionary::DifficultyLevel;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🎤 Real-time Voice Coach");
    println!("========================");
    println!("This demo simulates real-time voice coaching with live audio processing.");
    println!("Press Ctrl+C to exit.\n");

    // Initialize the voice coach
    let coach = VoiceCoach::new().await?;

    // Start coaching session
    let user_id = "demo_user";
    let session_config = CoachingSessionConfig {
        difficulty_level: DifficultyLevel::Beginner,
        focus_areas: vec![FocusArea::Pronunciation, FocusArea::Fluency],
        session_duration: Duration::from_secs(300), // 5 minutes
        real_time_feedback: true,
        adaptive_difficulty: true,
    };

    coach
        .start_coaching_session(user_id, session_config)
        .await?;

    Ok(())
}

/// Real-time voice coaching system
pub struct VoiceCoach {
    // Core components
    realtime_feedback: RealtimeFeedbackSystem,
    adaptive_engine: AdaptiveFeedbackEngine,
    progress_tracker: ProgressAnalyzer,
    #[cfg(feature = "gamification")]
    achievement_system: AchievementSystem,
    trainer: InteractiveTrainer,

    // Real-time processing
    audio_processor: RealTimeAudioProcessor,
    feedback_renderer: FeedbackRenderer,

    // Session state
    active_sessions: Arc<Mutex<std::collections::HashMap<String, CoachingSession>>>,
}

impl VoiceCoach {
    pub async fn new() -> Result<Self> {
        println!("🔧 Initializing Voice Coach components...");

        let realtime_feedback = RealtimeFeedbackSystem::new().await?;
        let adaptive_engine = AdaptiveFeedbackEngine::new().await?;
        let progress_tracker = ProgressAnalyzer::new().await?;
        #[cfg(feature = "gamification")]
        let achievement_system = AchievementSystem::new().await?;
        let trainer = InteractiveTrainer::new().await?;

        let audio_processor = RealTimeAudioProcessor::new().await?;
        let feedback_renderer = FeedbackRenderer::new();

        println!("✅ Voice Coach ready!");

        Ok(Self {
            realtime_feedback,
            adaptive_engine,
            progress_tracker,
            #[cfg(feature = "gamification")]
            achievement_system,
            trainer,
            audio_processor,
            feedback_renderer,
            active_sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
        })
    }

    pub async fn start_coaching_session(
        &self,
        user_id: &str,
        config: CoachingSessionConfig,
    ) -> Result<()> {
        println!("🚀 Starting coaching session for user: {user_id}");

        // Create session
        let session = CoachingSession {
            session_id: Uuid::new_v4().to_string(),
            user_id: user_id.to_string(),
            config: config.clone(),
            start_time: Instant::now(),
            current_exercise: None,
            stats: SessionStats::default(),
            is_active: true,
        };

        // Store session
        {
            let mut sessions = self.active_sessions.lock().unwrap();
            sessions.insert(session.session_id.clone(), session.clone());
        }

        // Start real-time processing loop
        self.run_coaching_loop(&session).await?;

        Ok(())
    }

    async fn run_coaching_loop(&self, session: &CoachingSession) -> Result<()> {
        println!("🔄 Starting real-time coaching loop...");

        // Create channels for audio and feedback
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(100);
        let (feedback_tx, mut feedback_rx) = mpsc::channel::<RealtimeFeedback>(100);

        // Start audio capture (simulated)
        let audio_capture_handle = tokio::spawn({
            let tx = audio_tx.clone();
            async move { simulate_audio_capture(tx).await }
        });

        // Start real-time processing
        let processing_handle = tokio::spawn({
            let realtime_feedback = self.realtime_feedback.clone();
            let session_id = session.session_id.clone();
            let tx = feedback_tx.clone();

            async move {
                // Create a feedback stream for processing
                let session_state = SessionState::new("demo_user").await.unwrap();
                let stream = realtime_feedback
                    .create_stream("demo_user", &session_state)
                    .await
                    .unwrap();

                while let Some(audio_chunk) = audio_rx.recv().await {
                    // Convert AudioChunk to AudioBuffer
                    let audio_buffer =
                        AudioBuffer::mono(audio_chunk.data.clone(), audio_chunk.sample_rate);

                    if let Ok(feedback_response) =
                        stream.process_audio(&audio_buffer, "default text").await
                    {
                        // Convert FeedbackResponse to RealtimeFeedback
                        let realtime_feedback = RealtimeFeedback {
                            current_score: feedback_response.overall_score, // Already normalized to 0-1
                            indicator_type: if feedback_response.overall_score > 0.8 {
                                "Excellent".to_string()
                            } else if feedback_response.overall_score > 0.6 {
                                "Good".to_string()
                            } else {
                                "Needs Improvement".to_string()
                            },
                            message: feedback_response.immediate_actions.join("; "),
                            timestamp: std::time::SystemTime::now(),
                        };
                        let _ = tx.send(realtime_feedback).await;
                    }
                }
            }
        });

        // Start feedback rendering
        let rendering_handle = tokio::spawn({
            let renderer = self.feedback_renderer.clone();
            let session_id = session.session_id.clone();

            async move {
                while let Some(feedback) = feedback_rx.recv().await {
                    renderer
                        .render_realtime_feedback(&session_id, &feedback)
                        .await;
                }
            }
        });

        // Main coaching loop
        let mut exercise_timer = interval(Duration::from_secs(30)); // New exercise every 30 seconds
        let mut progress_timer = interval(Duration::from_secs(10)); // Progress updates every 10 seconds

        let mut session_active = true;
        let start_time = Instant::now();

        while session_active && start_time.elapsed() < session.config.session_duration {
            tokio::select! {
                _ = exercise_timer.tick() => {
                    self.provide_new_exercise(&session.session_id).await?;
                }
                _ = progress_timer.tick() => {
                    self.update_session_progress(&session.session_id).await?;
                }
                _ = tokio::signal::ctrl_c() => {
                    println!("\n🛑 Received interrupt signal, ending session...");
                    session_active = false;
                }
            }
        }

        // End session
        self.end_coaching_session(&session.session_id).await?;

        // Clean up handles
        audio_capture_handle.abort();
        processing_handle.abort();
        rendering_handle.abort();

        Ok(())
    }

    async fn provide_new_exercise(&self, session_id: &str) -> Result<()> {
        println!("\n📝 Providing new exercise...");

        // Get adaptive exercise recommendation
        let exercises = self.trainer.get_exercises(session_id, 1.0).await?;
        let exercise = exercises
            .first()
            .ok_or_else(|| anyhow::anyhow!("No exercises available"))?;

        println!("🎯 New Exercise: {}", exercise.name);
        println!("   Description: {}", exercise.description);
        println!("   Target: \"{}\"", exercise.target_text);
        println!("   Difficulty: {:?}", exercise.difficulty);

        Ok(())
    }

    async fn update_session_progress(&self, session_id: &str) -> Result<()> {
        let sessions = self.active_sessions.lock().unwrap();
        if let Some(session) = sessions.get(session_id) {
            let elapsed = session.start_time.elapsed();
            println!("⏱️  Session Progress: {:.0}s elapsed", elapsed.as_secs());

            // Simulate getting current stats
            let current_score = 75.0 + (elapsed.as_secs() as f32 * 0.5) % 25.0;
            println!("📊 Current Performance: {current_score:.1}%");

            // Check for achievements
            self.check_real_time_achievements(&session.user_id).await?;
        }

        Ok(())
    }

    async fn check_real_time_achievements(&self, user_id: &str) -> Result<()> {
        // Simulate achievement checking
        use fastrand;

        if fastrand::f32() < 0.1 {
            // 10% chance of achievement
            println!("🎉 Achievement Unlocked: 'Steady Progress' - Keep up the great work!");
        }

        Ok(())
    }

    async fn end_coaching_session(&self, session_id: &str) -> Result<()> {
        println!("\n🏁 Ending coaching session...");

        // Generate session summary
        let sessions = self.active_sessions.lock().unwrap();
        if let Some(session) = sessions.get(session_id) {
            let duration = session.start_time.elapsed();

            println!("📊 Session Summary:");
            println!("   Duration: {:.1} minutes", duration.as_secs_f32() / 60.0);
            println!("   Exercises completed: {}", fastrand::u32(5..12));
            println!("   Average score: {:.1}%", 70.0 + fastrand::f32() * 25.0);
            println!("   Improvement: +{:.1}%", fastrand::f32() * 10.0);

            println!("\n💡 Recommendations for next session:");
            println!("   • Focus more on consonant pronunciation");
            println!("   • Practice longer sentences for fluency");
            println!("   • Try the intermediate difficulty level");
        }

        // Remove session
        drop(sessions);
        let mut sessions = self.active_sessions.lock().unwrap();
        sessions.remove(session_id);

        Ok(())
    }
}

/// Real-time audio processor
#[derive(Clone)]
pub struct RealTimeAudioProcessor {
    sample_rate: u32,
    chunk_size: usize,
}

impl RealTimeAudioProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            sample_rate: 16000,
            chunk_size: 1024,
        })
    }

    pub async fn process_chunk(&self, chunk: &AudioChunk) -> Result<ProcessedAudio> {
        // Simulate real-time audio processing
        sleep(Duration::from_millis(5)).await;

        Ok(ProcessedAudio {
            features: vec![0.5; 13], // Mock MFCC features
            volume_level: 0.7,
            clarity_score: 0.8,
            timestamp: std::time::SystemTime::now(),
        })
    }
}

/// Feedback renderer for console output
#[derive(Clone)]
pub struct FeedbackRenderer {
    last_render: Arc<Mutex<Instant>>,
}

impl Default for FeedbackRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackRenderer {
    pub fn new() -> Self {
        Self {
            last_render: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub async fn render_realtime_feedback(&self, session_id: &str, feedback: &RealtimeFeedback) {
        let mut last = self.last_render.lock().unwrap();

        // Throttle rendering to avoid spam
        if last.elapsed() > Duration::from_millis(500) {
            self.render_feedback_display(feedback);
            *last = Instant::now();
        }
    }

    fn render_feedback_display(&self, feedback: &RealtimeFeedback) {
        // Clear previous line and render new feedback
        print!("\r🎯 Live: ");

        // Score bar
        let score_percent = (feedback.current_score * 100.0) as u8;
        let bar_length = 20;
        let filled_length = (score_percent as usize * bar_length) / 100;

        print!("[");
        for i in 0..bar_length {
            if i < filled_length {
                print!("█");
            } else {
                print!("░");
            }
        }
        print!("] {score_percent:.0}% ");

        // Status indicator based on score
        let symbol = match feedback.indicator_type.as_str() {
            "good" => "✅",
            "warning" => "⚠️",
            "error" => "❌",
            _ => "🔵",
        };
        print!("{symbol} ");

        // Immediate corrections
        if !feedback.message.is_empty() {
            print!("| 💡 {}", feedback.message);
        }

        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }
}

/// Simulate audio capture from microphone
async fn simulate_audio_capture(tx: mpsc::Sender<AudioChunk>) -> Result<()> {
    println!("🎙️  Simulating microphone capture...");

    let mut interval = interval(Duration::from_millis(100)); // 10 chunks per second
    let mut chunk_counter = 0;

    loop {
        interval.tick().await;

        // Simulate audio chunk
        let chunk = AudioChunk {
            data: vec![0.1 * (chunk_counter as f32 * 0.1).sin(); 1600], // 100ms at 16kHz
            sample_rate: 16000,
            timestamp: std::time::SystemTime::now(),
            chunk_id: chunk_counter,
        };

        if tx.send(chunk).await.is_err() {
            break; // Channel closed
        }

        chunk_counter += 1;

        // Simulate session ending after a while
        if chunk_counter > 300 {
            // 30 seconds
            break;
        }
    }

    Ok(())
}

// Data structures

#[derive(Debug, Clone)]
pub struct CoachingSessionConfig {
    pub difficulty_level: DifficultyLevel,
    pub focus_areas: Vec<FocusArea>,
    pub session_duration: Duration,
    pub real_time_feedback: bool,
    pub adaptive_difficulty: bool,
}

#[derive(Debug, Clone)]
pub struct CoachingSession {
    pub session_id: String,
    pub user_id: String,
    pub config: CoachingSessionConfig,
    pub start_time: Instant,
    pub current_exercise: Option<Exercise>,
    pub stats: SessionStats,
    pub is_active: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub exercises_completed: u32,
    pub total_score: f32,
    pub improvement_rate: f32,
    pub streaks: u32,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub timestamp: std::time::SystemTime,
    pub chunk_id: u64,
}

#[derive(Debug, Clone)]
pub struct ProcessedAudio {
    pub features: Vec<f32>,
    pub volume_level: f32,
    pub clarity_score: f32,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct RealtimeFeedback {
    pub current_score: f32,
    pub indicator_type: String,
    pub message: String,
    pub timestamp: std::time::SystemTime,
}

impl Default for CoachingSessionConfig {
    fn default() -> Self {
        Self {
            difficulty_level: DifficultyLevel::Beginner,
            focus_areas: vec![FocusArea::Pronunciation],
            session_duration: Duration::from_secs(300),
            real_time_feedback: true,
            adaptive_difficulty: true,
        }
    }
}
