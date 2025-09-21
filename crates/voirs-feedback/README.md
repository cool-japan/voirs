# VoiRS Feedback

[![Crates.io](https://img.shields.io/crates/v/voirs-feedback)](https://crates.io/crates/voirs-feedback)
[![Documentation](https://docs.rs/voirs-feedback/badge.svg)](https://docs.rs/voirs-feedback)
[![License](https://img.shields.io/crates/l/voirs-feedback)](LICENSE)

**Interactive feedback, gamification, and training systems for VoiRS**

VoiRS Feedback provides real-time feedback generation, adaptive learning systems, progress tracking, and gamification features to enhance user engagement and accelerate improvement in speech synthesis applications.

## Features

### 🔄 Real-time Feedback
- **Instant Analysis**: Live speech quality assessment
- **Contextual Suggestions**: Targeted improvement recommendations
- **Visual Indicators**: Real-time quality visualizations
- **Audio Highlighting**: Problem area identification

### 🧠 Adaptive Learning
- **Personal Profiles**: Individual learning patterns
- **Difficulty Adjustment**: Dynamic exercise adaptation
- **Weakness Targeting**: Focus on problematic areas
- **Performance Prediction**: ML-based progress forecasting

### 📊 Progress Tracking
- **Detailed Analytics**: Comprehensive performance metrics
- **Historical Trends**: Long-term improvement visualization
- **Skill Breakdown**: Area-specific progress tracking
- **Milestone Recognition**: Achievement celebration

### 🎮 Gamification & Training
- **Achievement System**: Unlock badges and rewards
- **Leaderboards**: Social competition features
- **Interactive Exercises**: Engaging training activities
- **Progress Visualization**: Beautiful charts and graphs

## Quick Start

Add VoiRS Feedback to your `Cargo.toml`:

```toml
[dependencies]
voirs-feedback = "0.1.0"

# Enable specific features
voirs-feedback = { version = "0.1.0", features = ["realtime", "adaptive", "gamification"] }
```

### Basic Feedback System

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize feedback system
    let feedback_system = FeedbackSystem::new().await?;
    
    // Load audio for analysis
    let audio = AudioBuffer::from_file("speech.wav")?;
    let expected_text = "Hello world, this is a test.";
    
    // Generate feedback
    let feedback = feedback_system.generate_feedback(
        &audio,
        expected_text,
        None
    ).await?;
    
    println!("Feedback Summary:");
    println!("  Overall Score: {:.1}%", feedback.overall_score * 100.0);
    println!("  Feedback Items: {}", feedback.feedback_items.len());
    
    // Display feedback items
    for (i, item) in feedback.feedback_items.iter().enumerate() {
        println!("  {}. {} (Score: {:.1}%)", 
                 i + 1, 
                 item.message, 
                 item.score * 100.0);
        
        if let Some(suggestion) = &item.suggestion {
            println!("     💡 {}", suggestion);
        }
    }
    
    // Check for immediate actions
    if !feedback.immediate_actions.is_empty() {
        println!("\nNext Steps:");
        for action in &feedback.immediate_actions {
            println!("  • {}", action);
        }
    }
    
    Ok(())
}
```

### Real-time Feedback

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize real-time feedback system
    let realtime_system = RealtimeFeedbackSystem::new().await?;
    
    // Start real-time session
    let session_id = realtime_system.start_session("user_123").await?;
    
    // Simulate real-time audio streaming
    let audio_chunk = AudioChunk::from_bytes(&audio_data, 16000)?;
    
    // Process audio chunk
    let realtime_feedback = realtime_system.process_audio_chunk(
        &session_id,
        &audio_chunk
    ).await?;
    
    // Display real-time feedback
    println!("Real-time Feedback:");
    println!("  Current Score: {:.1}%", realtime_feedback.current_score * 100.0);
    println!("  Confidence: {:.1}%", realtime_feedback.confidence * 100.0);
    
    // Show visual indicators
    for indicator in &realtime_feedback.visual_indicators {
        println!("  {}: {}", indicator.area, indicator.status);
    }
    
    // Display immediate corrections
    if !realtime_feedback.immediate_corrections.is_empty() {
        println!("\nImmediate Corrections:");
        for correction in &realtime_feedback.immediate_corrections {
            println!("  ⚠️  {}", correction.message);
        }
    }
    
    Ok(())
}
```

### Adaptive Learning Engine

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize adaptive learning engine
    let adaptive_engine = AdaptiveFeedbackEngine::new().await?;
    
    // Create user profile
    let user_profile = UserProfile {
        user_id: "user_123".to_string(),
        skill_level: SkillLevel::Intermediate,
        learning_preferences: LearningPreferences {
            preferred_feedback_style: FeedbackStyle::Detailed,
            difficulty_preference: DifficultyPreference::Adaptive,
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Fluency],
        },
        historical_performance: vec![],
        current_session_state: SessionState::default(),
    };
    
    // Generate adaptive recommendations
    let recommendations = adaptive_engine.generate_recommendations(
        &user_profile,
        &session_data
    ).await?;
    
    println!("Adaptive Recommendations:");
    println!("  Recommended Exercises: {}", recommendations.exercises.len());
    println!("  Difficulty Level: {:?}", recommendations.difficulty_level);
    println!("  Focus Areas: {:?}", recommendations.focus_areas);
    
    // Show personalized exercises
    for exercise in &recommendations.exercises {
        println!("  📝 {}: {} (Difficulty: {:?})", 
                 exercise.name, 
                 exercise.description,
                 exercise.difficulty);
    }
    
    // Update user profile based on performance
    let updated_profile = adaptive_engine.update_user_profile(
        &user_profile,
        &session_results
    ).await?;
    
    println!("\nProfile Updates:");
    println!("  New Skill Level: {:?}", updated_profile.skill_level);
    println!("  Strengths: {:?}", updated_profile.strengths);
    println!("  Areas for Improvement: {:?}", updated_profile.weaknesses);
    
    Ok(())
}
```

### Progress Tracking

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize progress analyzer
    let progress_analyzer = ProgressAnalyzer::new().await?;
    
    // Analyze user progress
    let progress_analysis = progress_analyzer.analyze_progress(
        "user_123",
        &date_range
    ).await?;
    
    println!("Progress Analysis:");
    println!("  Overall Improvement: {:.1}%", progress_analysis.overall_improvement * 100.0);
    println!("  Sessions Completed: {}", progress_analysis.total_sessions);
    println!("  Current Streak: {} days", progress_analysis.current_streak);
    
    // Skill breakdown
    println!("\nSkill Breakdown:");
    for (skill, progress) in &progress_analysis.skill_progress {
        println!("  {:?}: {:.1}% (trend: {:.1}%)", 
                 skill, 
                 progress.current_level * 100.0,
                 progress.trend * 100.0);
    }
    
    // Milestones
    println!("\nRecent Milestones:");
    for milestone in &progress_analysis.recent_milestones {
        println!("  🎯 {} - {}", 
                 milestone.name, 
                 milestone.achieved_at.format("%Y-%m-%d"));
    }
    
    // Generate progress report
    let report = progress_analyzer.generate_report(
        &progress_analysis,
        ReportFormat::Detailed
    ).await?;
    
    println!("\nProgress Report Generated: {} pages", report.pages.len());
    
    Ok(())
}
```

### Gamification System

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize achievement system
    let achievement_system = AchievementSystem::new().await?;
    
    // Check for new achievements
    let new_achievements = achievement_system.check_achievements(
        "user_123",
        &user_progress,
        &session_data
    ).await?;
    
    if !new_achievements.is_empty() {
        println!("🎉 New Achievements Unlocked!");
        for achievement in &new_achievements {
            println!("  {} {} - {} ({} points)", 
                     achievement.achievement.tier_emoji(),
                     achievement.achievement.name,
                     achievement.achievement.description,
                     achievement.achievement.points);
        }
    }
    
    // Get leaderboard
    let leaderboard = achievement_system.get_leaderboard(
        LeaderboardType::Weekly,
        10
    ).await?;
    
    println!("\nWeekly Leaderboard:");
    for (rank, entry) in leaderboard.entries.iter().enumerate() {
        println!("  {}. {} - {} points", 
                 rank + 1, 
                 entry.display_name, 
                 entry.points);
    }
    
    // User's current standing
    let user_stats = achievement_system.get_user_stats("user_123").await?;
    println!("\nYour Stats:");
    println!("  Total Points: {}", user_stats.total_points);
    println!("  Current Level: {}", user_stats.current_level);
    println!("  Achievements: {}/{}", user_stats.unlocked_achievements, user_stats.total_achievements);
    println!("  Weekly Rank: #{}", user_stats.weekly_rank);
    
    Ok(())
}
```

### Interactive Training

```rust
use voirs_feedback::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize interactive trainer
    let trainer = InteractiveTrainer::new().await?;
    
    // Start training session
    let training_session = trainer.start_session(
        "user_123",
        Some(TrainingSessionConfig {
            session_type: SessionType::Guided,
            difficulty_level: DifficultyLevel::Intermediate,
            focus_areas: vec![FocusArea::Pronunciation],
            duration_minutes: 30,
            ..Default::default()
        })
    ).await?;
    
    println!("Training Session Started: {}", training_session.session_id);
    
    // Get next exercise
    let exercise = trainer.get_next_exercise(&training_session.session_id).await?;
    
    println!("Next Exercise: {}", exercise.name);
    println!("  Type: {:?}", exercise.exercise_type);
    println!("  Difficulty: {:?}", exercise.difficulty);
    println!("  Description: {}", exercise.description);
    println!("  Instructions: {}", exercise.instructions);
    
    // Simulate exercise completion
    let exercise_result = ExerciseResult {
        exercise_id: exercise.id,
        user_audio: audio_buffer,
        completion_time: Duration::from_secs(45),
        user_response: Some("Hello world".to_string()),
    };
    
    // Submit exercise result
    let assessment = trainer.submit_exercise_result(
        &training_session.session_id,
        &exercise_result
    ).await?;
    
    println!("\nExercise Assessment:");
    println!("  Score: {:.1}%", assessment.score * 100.0);
    println!("  Feedback: {}", assessment.feedback);
    
    // Get session summary
    let session_summary = trainer.get_session_summary(&training_session.session_id).await?;
    
    println!("\nSession Summary:");
    println!("  Exercises Completed: {}", session_summary.exercises_completed);
    println!("  Average Score: {:.1}%", session_summary.average_score * 100.0);
    println!("  Time Spent: {}min", session_summary.total_time.as_secs() / 60);
    println!("  Achievements: {}", session_summary.achievements_earned.len());
    
    Ok(())
}
```

## Feature Configuration

### Feature Flags

Enable specific functionality through feature flags:

```toml
[dependencies]
voirs-feedback = { 
    version = "0.1.0", 
    features = [
        "realtime",          # Real-time feedback
        "adaptive",          # Adaptive learning
        "progress-tracking", # Progress analytics
        "gamification",      # Achievement system
        "training",          # Interactive training
        "ui",                # Visualization components
        "audio-export",      # Audio export capabilities
        "all-features",      # Enable all features
    ]
}
```

### System Configuration

```rust
use voirs_feedback::prelude::*;

let config = FeedbackSystemConfig {
    // Real-time settings
    realtime_config: RealtimeConfig {
        feedback_frequency_ms: 200,
        confidence_threshold: 0.7,
        enable_visual_indicators: true,
        max_suggestions_per_second: 2,
    },
    
    // Adaptive learning settings
    adaptive_config: AdaptiveConfig {
        learning_rate: 0.1,
        adaptation_frequency: 5, // Every 5 sessions
        min_data_points: 10,
        enable_difficulty_adjustment: true,
    },
    
    // Progress tracking settings
    progress_config: ProgressConfig {
        tracking_granularity: TrackingGranularity::Session,
        retention_period_days: 365,
        enable_trend_analysis: true,
        milestone_frequency: MilestoneFrequency::Weekly,
    },
    
    // Gamification settings
    gamification_config: GamificationConfig {
        enable_achievements: true,
        enable_leaderboards: true,
        achievement_notification_style: NotificationStyle::Enthusiastic,
        leaderboard_update_frequency: Duration::from_secs(3600),
    },
};

let feedback_system = FeedbackSystem::with_config(config).await?;
```

## Visualization Components

### Real-time Dashboard

```rust
use voirs_feedback::prelude::*;

#[cfg(feature = "ui")]
{
    // Create feedback visualizer
    let visualizer = FeedbackVisualizer::new();
    
    // Render in egui application
    egui::CentralPanel::default().show(ctx, |ui| {
        // Real-time metrics
        visualizer.render_realtime_metrics(ui, &current_metrics);
        
        // Progress chart
        visualizer.render_progress_chart(ui, &progress_data);
        
        // Achievement showcase
        visualizer.render_achievement_showcase(ui, &recent_achievements);
        
        // Skill radar
        visualizer.render_skill_radar(ui, &skill_breakdown);
    });
}
```

### Progress Charts

```rust
use voirs_feedback::prelude::*;

#[cfg(feature = "ui")]
{
    // Create progress chart
    let mut chart = ProgressChart::new();
    
    // Add data points
    for session in &session_history {
        chart.add_data_point(
            session.timestamp,
            session.average_score,
            Some(format!("Session {}", session.id))
        );
    }
    
    // Generate SVG
    let svg_content = chart.generate_svg(800, 400)?;
    
    // Save to file
    std::fs::write("progress_chart.svg", svg_content)?;
}
```

## Training Exercises

### Exercise Types

VoiRS Feedback includes various exercise types:

| Exercise Type | Description | Difficulty | Focus Areas |
|---------------|-------------|------------|-------------|
| Pronunciation Drills | Phoneme-specific practice | Beginner-Advanced | Pronunciation, Accuracy |
| Fluency Practice | Sentence reading exercises | Intermediate-Advanced | Fluency, Rhythm |
| Intonation Training | Pitch contour exercises | Beginner-Intermediate | Prosody, Expressiveness |
| Tongue Twisters | Challenging phrase repetition | Intermediate-Advanced | Articulation, Speed |
| Dialogue Practice | Conversational speech | Advanced | Naturalness, Context |
| Accent Reduction | Native-like pronunciation | Intermediate-Advanced | Accent, Clarity |

### Custom Exercise Creation

```rust
use voirs_feedback::prelude::*;

// Create custom exercise
let custom_exercise = Exercise {
    id: Uuid::new_v4().to_string(),
    name: "Custom Pronunciation Drill".to_string(),
    description: "Practice difficult consonant clusters".to_string(),
    exercise_type: ExerciseType::Pronunciation,
    difficulty: DifficultyLevel::Intermediate,
    instructions: "Read each word clearly, focusing on the consonant sounds".to_string(),
    target_text: "The sixth sick sheik's sixth sheep's sick".to_string(),
    expected_duration: Duration::from_secs(30),
    scoring_criteria: ScoringCriteria {
        accuracy_weight: 0.5,
        fluency_weight: 0.2,
        prosody_weight: 0.2,
        timing_weight: 0.1,
    },
    metadata: ExerciseMetadata {
        tags: vec!["consonants".to_string(), "clusters".to_string()],
        difficulty_factors: vec![
            DifficultyFactor::ConsonantClusters,
            DifficultyFactor::LongSentences,
        ],
        focus_phonemes: vec!["s", "ʃ", "θ", "k"].into_iter().map(|s| s.to_string()).collect(),
    },
};

// Add to exercise library
trainer.add_custom_exercise(custom_exercise).await?;
```

## Performance Optimization

### Caching and Memory Management

```rust
use voirs_feedback::prelude::*;

let config = FeedbackSystemConfig {
    // Cache configuration
    cache_config: CacheConfig {
        max_cache_size_mb: 500,
        cache_ttl_seconds: 3600,
        enable_persistent_cache: true,
    },
    
    // Memory optimization
    memory_config: MemoryConfig {
        max_memory_usage_mb: 1000,
        gc_frequency_seconds: 300,
        enable_memory_monitoring: true,
    },
    
    // Performance settings
    performance_config: PerformanceConfig {
        enable_parallel_processing: true,
        max_concurrent_sessions: 10,
        audio_chunk_size_ms: 100,
    },
};
```

### Asynchronous Processing

```rust
use voirs_feedback::prelude::*;
use tokio::time::{timeout, Duration};

// Process multiple feedback requests concurrently
let feedback_tasks: Vec<_> = audio_samples.iter()
    .map(|audio| {
        let feedback_system = feedback_system.clone();
        let audio = audio.clone();
        let text = expected_text.clone();
        
        tokio::spawn(async move {
            timeout(Duration::from_secs(10), 
                feedback_system.generate_feedback(&audio, &text, None)).await
        })
    })
    .collect();

// Wait for all tasks to complete
let results = futures::future::join_all(feedback_tasks).await;
```

## Integration Examples

### Web Service Integration

```rust
use voirs_feedback::prelude::*;
use axum::{Json, extract::State};

#[derive(Clone)]
struct AppState {
    feedback_system: FeedbackSystem,
}

async fn feedback_endpoint(
    State(app_state): State<AppState>,
    Json(request): Json<FeedbackRequest>,
) -> Result<Json<FeedbackResponse>, String> {
    let feedback = app_state.feedback_system
        .generate_feedback(&request.audio, &request.text, Some(request.options))
        .await
        .map_err(|e| e.to_string())?;
    
    Ok(Json(feedback))
}
```

### CLI Application

```rust
use voirs_feedback::prelude::*;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    audio_file: String,
    #[arg(long)]
    text: String,
    #[arg(long)]
    user_id: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let feedback_system = FeedbackSystem::new().await?;
    let audio = AudioBuffer::from_file(&args.audio_file)?;
    
    let feedback = feedback_system.generate_feedback(&audio, &args.text, None).await?;
    
    println!("{}", serde_json::to_string_pretty(&feedback)?);
    
    Ok(())
}
```

## Error Handling

VoiRS Feedback provides comprehensive error handling:

```rust
use voirs_feedback::prelude::*;

match feedback_system.generate_feedback(&audio, &text, None).await {
    Ok(feedback) => {
        println!("Feedback generated successfully");
    }
    Err(FeedbackError::AudioTooShort { duration }) => {
        eprintln!("Audio too short: {:.1}s", duration);
    }
    Err(FeedbackError::TextTooLong { length }) => {
        eprintln!("Text too long: {} characters", length);
    }
    Err(FeedbackError::UserNotFound { user_id }) => {
        eprintln!("User not found: {}", user_id);
    }
    Err(FeedbackError::SessionExpired { session_id }) => {
        eprintln!("Session expired: {}", session_id);
    }
    Err(FeedbackError::ConfigurationError { message }) => {
        eprintln!("Configuration error: {}", message);
    }
    Err(e) => {
        eprintln!("Feedback generation failed: {}", e);
    }
}
```

## Examples

Check out the [examples](examples/) directory for comprehensive usage examples:

- [`realtime_feedback.rs`](examples/realtime_feedback.rs) - Real-time feedback system
- [`adaptive_learning.rs`](examples/adaptive_learning.rs) - Adaptive learning engine
- [`progress_tracking.rs`](examples/progress_tracking.rs) - Progress analytics
- [`gamification.rs`](examples/gamification.rs) - Achievement system
- [`interactive_training.rs`](examples/interactive_training.rs) - Training exercises
- [`visualization.rs`](examples/visualization.rs) - UI components
- [`web_service.rs`](examples/web_service.rs) - Web service integration
- [`cli_app.rs`](examples/cli_app.rs) - Command-line application

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/voirs.git
cd voirs/crates/voirs-feedback

# Install dependencies
cargo build --all-features

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench --all-features
```

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Citation

If you use VoiRS Feedback in your research, please cite:

```bibtex
@software{voirs_feedback,
  title = {VoiRS Feedback: Interactive Learning and Gamification for Speech Synthesis},
  author = {Tetsuya Kitahata},
  organization = {Cool Japan Co., Ltd.},
  year = {2024},
  url = {https://github.com/cool-japan/voirs}
}
```