//! # `VoiRS` Feedback
//!
//! Real-time feedback and interactive training system for `VoiRS` speech synthesis.
//! This crate provides comprehensive feedback mechanisms, adaptive learning,
//! and user progress tracking for continuous improvement.
//!
//! ## Features
//!
//! - **Real-time Feedback**: Immediate response during speech synthesis
//! - **Adaptive Learning**: ML-driven personalized feedback
//! - **Progress Tracking**: Detailed analytics and improvement metrics
//! - **Interactive Training**: Guided exercises and challenges
//! - **Gamification**: Achievement systems and leaderboards
//! - **Multi-modal Output**: Visual, audio, and haptic feedback
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use voirs_feedback::prelude::*;
//! use voirs_feedback::FeedbackSystem;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create feedback system
//!     let feedback_system = FeedbackSystem::new().await?;
//!     
//!     // Create user session
//!     let mut session = feedback_system.create_session("user123").await?;
//!     
//!     // Generate audio and get feedback
//!     let generated_audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//!     let expected_text = "Hello world";
//!     
//!     let feedback = session.process_synthesis(&generated_audio, expected_text).await?;
//!     
//!     // Print feedback items
//!     for item in &feedback.feedback_items {
//!         println!("Feedback: {} (Score: {:.1}%)", item.message, item.score * 100.0);
//!         if let Some(suggestion) = &item.suggestion {
//!             println!("  Suggestion: {}", suggestion);
//!         }
//!     }
//!     
//!     // Save progress
//!     session.save_progress().await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Configuration
//!
//! ```rust
//! use voirs_feedback::prelude::*;
//! use voirs_feedback::realtime::{RealtimeFeedbackSystem, RealtimeConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = RealtimeConfig {
//!     max_latency_ms: 50,
//!     stream_timeout: std::time::Duration::from_secs(180),
//!     audio_buffer_size: 1024,
//!     max_concurrent_streams: 25,
//!     enable_metrics: true,
//!     enable_confidence_filtering: true,
//!     quality_threshold: 0.7,
//!     pronunciation_threshold: 0.8,
//! };
//!
//! let feedback_system = RealtimeFeedbackSystem::with_config(config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Training Exercises
//!
//! ```rust,no_run
//! use voirs_feedback::prelude::*;
//! use voirs_feedback::{FeedbackSystem, FocusArea, ExerciseType, SuccessCriteria};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let feedback_system = FeedbackSystem::new().await?;
//! let mut session = feedback_system.create_session("user123").await?;
//!
//! // Start a training exercise
//! let exercise = TrainingExercise {
//!     exercise_id: "basic_pronunciation".to_string(),
//!     name: "Basic Pronunciation".to_string(),
//!     description: "Practice basic phoneme pronunciation".to_string(),
//!     difficulty: 0.3,
//!     focus_areas: vec![FocusArea::Pronunciation],
//!     exercise_type: ExerciseType::Pronunciation,
//!     target_text: "Hello world".to_string(),
//!     reference_audio: None,
//!     success_criteria: SuccessCriteria {
//!         min_quality_score: 0.7,
//!         min_pronunciation_score: 0.8,
//!         consistency_required: 1,
//!         max_attempts: 3,
//!         time_limit: Some(std::time::Duration::from_secs(300)),
//!     },
//!     estimated_duration: std::time::Duration::from_secs(600),
//! };
//!
//! session.start_exercise(&exercise).await?;
//! let result = session.complete_exercise().await?;
//! println!("Exercise completed: success = {}", result.success);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export core VoiRS types
pub use voirs_evaluation::{ComparisonResult, PronunciationScore, QualityScore};
pub use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

// Import for async trait
use async_trait::async_trait;
use std::time::Duration;

// Core modules
pub mod adaptive;
pub mod ai_coaching;
pub mod analytics;
pub mod cloud_deployment;
pub mod computer_vision;
pub mod data_management;
pub mod data_pipeline;
#[cfg(feature = "adaptive")]
pub mod deep_learning_feedback;
/// Description
pub mod emotional_intelligence;
pub mod enhanced_performance;
/// Description
pub mod enterprise;
pub mod error_context;
#[cfg(feature = "gamification")]
pub mod gamification;
pub mod gdpr;
pub mod group_learning;
pub mod health;
pub mod i18n_support;
pub mod integration;
pub mod load_balancer;
pub mod memory_monitor;
pub mod metrics_dashboard;
#[cfg(feature = "microservices")]
pub mod microservices;
pub mod natural_language_generation;
pub mod oauth2_auth;
pub mod peer_learning;
pub mod persistence;
pub mod platform;
pub mod progress;
pub mod quality_monitor;
pub mod realtime;
pub mod recovery;
pub mod training;
pub mod traits;
pub mod ux_analytics;
pub mod visualization;

// Re-export all public types from traits
pub use traits::*;

// Re-export core types needed by FeedbackSystem
pub use adaptive::core::AdaptiveFeedbackEngine;
pub use progress::core::ProgressAnalyzer;
pub use realtime::stream::FeedbackStream;
pub use realtime::system::RealtimeFeedbackSystem;
pub use training::core::InteractiveTrainer;

// Note: Full feature modules are not glob re-exported to avoid ambiguity.
// Import from specific modules: feedback::adaptive::*, feedback::training::*, etc.
// Or use the prelude: use voirs_feedback::prelude::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convenient prelude for common imports
pub mod prelude {
    //! Prelude module for convenient imports

    pub use crate::traits::{
        AdaptiveConfig, AdaptiveLearner, FeedbackConfig, FeedbackProvider, FeedbackResult,
        FeedbackSession, ProgressConfig, ProgressTracker, SessionState, TrainingExercise,
        TrainingProvider, TrainingResult, UserFeedback,
    };

    pub use crate::adaptive::core::AdaptiveFeedbackEngine;
    pub use crate::adaptive::types::LearningStyle;
    pub use crate::cloud_deployment::{
        CloudOrchestrator, CloudProvider, DeploymentConfig, KubernetesOrchestrator,
    };
    pub use crate::data_management::{DataExportPackage, DataManager, ExportFormat, ImportOptions};
    pub use crate::data_pipeline::{
        DataPipeline, DataProcessor, DataSink, DataType, PipelineConfig,
    };
    pub use crate::error_context::{
        ErrorCategory, ErrorContext, ErrorContextBuilder, ErrorSeverity,
    };
    pub use crate::gdpr::{
        ConsentRecord, DataSubject, GdprCompliance, GdprComplianceManager, ProcessingPurpose,
    };
    pub use crate::health::{
        ComponentHealth, HealthConfig, HealthMonitor, HealthReport, HealthStatus,
    };
    pub use crate::load_balancer::{
        LoadBalancer, LoadBalancerConfig, LoadBalancingAlgorithm, WorkerNode,
    };
    pub use crate::progress::core::ProgressAnalyzer;
    pub use crate::quality_monitor::{
        QualityAlert, QualityMetrics, QualityMonitor, QualityMonitorConfig,
    };
    pub use crate::realtime::stream::FeedbackStream;
    pub use crate::realtime::system::RealtimeFeedbackSystem;
    pub use crate::training::core::InteractiveTrainer;
    pub use crate::training::types::TrainingSession;
    pub use crate::traits::UserProgress;

    #[cfg(feature = "gamification")]
    pub use crate::gamification::achievements::AchievementSystem;
    #[cfg(feature = "gamification")]
    pub use crate::gamification::Leaderboard;

    #[cfg(feature = "ui")]
    pub use crate::visualization::core::FeedbackVisualizer;
    #[cfg(feature = "ui")]
    pub use crate::visualization::charts::ProgressChart;

    // Re-export SDK types
    pub use voirs_evaluation::{ComparisonResult, PronunciationScore, QualityScore};
    pub use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
    pub use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, VoirsError};

    // Re-export async trait
    pub use async_trait::async_trait;
}

// ============================================================================
// Error Types
// ============================================================================

/// Feedback-specific error types
#[derive(Debug, thiserror::Error)]
pub enum FeedbackError {
    /// Feedback generation failed
    #[error("Feedback generation failed: {message}")]
    FeedbackGenerationError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Session management failed
    #[error("Session management failed: {message}")]
    SessionError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Adaptive learning failed
    #[error("Adaptive learning failed: {message}")]
    AdaptiveLearningError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Progress tracking failed
    #[error("Progress tracking failed: {message}")]
    ProgressTrackingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Training system failed
    #[error("Training system failed: {message}")]
    TrainingError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Real-time processing failed
    #[error("Real-time processing failed: {message}")]
    RealtimeError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },

    /// Invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// Feature not supported
    #[error("Feature not supported: {feature}")]
    FeatureNotSupported {
        /// Feature name
        feature: String,
    },

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Lock contention detected
    #[error("Lock contention detected - operation aborted to preserve UI responsiveness")]
    LockContention,

    /// Operation timeout
    #[error("Operation timed out")]
    Timeout,
}

impl From<FeedbackError> for VoirsError {
    fn from(err: FeedbackError) -> Self {
        // Enhanced error logging for better diagnostics
        log::warn!("Converting FeedbackError to VoirsError: {err}");

        match err {
            FeedbackError::FeedbackGenerationError { message, source: _ } => {
                log::error!("Feedback generation failed: {message}");
                VoirsError::AudioError {
                    message,
                    buffer_info: None,
                }
            }
            FeedbackError::SessionError { message, source: _ } => {
                log::error!("Session management error: {message}");
                VoirsError::ConfigError {
                    field: "session".to_string(),
                    message,
                }
            }
            FeedbackError::AdaptiveLearningError { message, source } => {
                log::error!("Adaptive learning error: {message}");
                VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::Vocoder,
                    message,
                    source,
                }
            }
            FeedbackError::ProgressTrackingError { message, source: _ } => {
                log::error!("Progress tracking error: {message}");
                VoirsError::ConfigError {
                    field: "progress_tracking".to_string(),
                    message,
                }
            }
            FeedbackError::TrainingError { message, source: _ } => {
                log::error!("Training system error: {message}");
                VoirsError::ConfigError {
                    field: "training".to_string(),
                    message,
                }
            }
            FeedbackError::RealtimeError { message, source: _ } => {
                log::error!("Real-time processing error: {message}");
                VoirsError::AudioError {
                    message,
                    buffer_info: None,
                }
            }
            FeedbackError::ConfigurationError { message } => {
                log::error!("Configuration error: {message}");
                VoirsError::ConfigError {
                    field: "configuration".to_string(),
                    message,
                }
            }
            FeedbackError::InvalidInput { message } => {
                log::warn!("Invalid input provided: {message}");
                VoirsError::ConfigError {
                    field: "input".to_string(),
                    message: format!("Invalid input: {message}"),
                }
            }
            FeedbackError::FeatureNotSupported { feature } => {
                log::warn!("Feature not supported: {feature}");
                VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::Vocoder,
                    message: format!("Feature not supported: {feature}"),
                    source: None,
                }
            }
            FeedbackError::ProcessingError(message) => {
                log::error!("Processing error: {message}");
                VoirsError::AudioError {
                    message,
                    buffer_info: None,
                }
            }
            FeedbackError::LockContention => {
                log::warn!("Lock contention detected - preserving UI responsiveness");
                VoirsError::AudioError {
                    message:
                        "Lock contention detected - operation aborted to preserve UI responsiveness"
                            .to_string(),
                    buffer_info: None,
                }
            }
            FeedbackError::Timeout => {
                log::error!("Operation timed out");
                VoirsError::AudioError {
                    message: "Operation timed out".to_string(),
                    buffer_info: None,
                }
            }
        }
    }
}

impl From<VoirsError> for FeedbackError {
    fn from(err: VoirsError) -> Self {
        FeedbackError::FeedbackGenerationError {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}

// ============================================================================
// Main Feedback System
// ============================================================================

/// Main feedback system that coordinates all feedback components
pub struct FeedbackSystem {
    /// Real-time feedback system
    realtime: RealtimeFeedbackSystem,
    /// Adaptive learning engine
    adaptive: AdaptiveFeedbackEngine,
    /// Progress tracking
    progress: ProgressAnalyzer,
    /// Training system
    trainer: InteractiveTrainer,
    /// Configuration
    config: FeedbackSystemConfig,
    /// Persistence manager
    #[cfg(feature = "persistence")]
    persistence_manager: std::sync::Arc<dyn crate::persistence::PersistenceManager>,
}

impl FeedbackSystem {
    /// Create a new feedback system
    pub async fn new() -> Result<Self, FeedbackError> {
        Self::with_config(FeedbackSystemConfig::default()).await
    }

    /// Create feedback system with custom configuration
    pub async fn with_config(config: FeedbackSystemConfig) -> Result<Self, FeedbackError> {
        let realtime =
            RealtimeFeedbackSystem::new()
                .await
                .map_err(|e| FeedbackError::RealtimeError {
                    message: format!("Failed to initialize real-time system: {e}"),
                    source: Some(Box::new(e)),
                })?;

        let adaptive = AdaptiveFeedbackEngine::new().await.map_err(|e| {
            FeedbackError::AdaptiveLearningError {
                message: format!("Failed to initialize adaptive engine: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let progress =
            ProgressAnalyzer::new()
                .await
                .map_err(|e| FeedbackError::ProgressTrackingError {
                    message: format!("Failed to initialize progress analyzer: {e}"),
                    source: Some(Box::new(e)),
                })?;

        let trainer =
            InteractiveTrainer::new()
                .await
                .map_err(|e| FeedbackError::TrainingError {
                    message: format!("Failed to initialize trainer: {e}"),
                    source: Some(Box::new(e)),
                })?;

        #[cfg(feature = "persistence")]
        let persistence_manager = {
            use crate::persistence::backends::sqlite::SQLitePersistenceManager;
            use crate::persistence::{PersistenceBackend, PersistenceConfig, PersistenceManager};

            let persistence_config = PersistenceConfig {
                backend: PersistenceBackend::SQLite,
                connection_string: "feedback.db".to_string(),
                enable_encryption: false,
                max_cache_size: 1000,
                cache_ttl_seconds: 3600,
                auto_cleanup_interval_hours: 24,
                data_retention_days: 365,
                enable_compression: true,
                connection_pool_size: 10,
            };

            let mut manager = SQLitePersistenceManager::new(persistence_config)
                .await
                .map_err(|e| FeedbackError::SessionError {
                    message: format!("Failed to initialize persistence manager: {e}"),
                    source: Some(Box::new(e)),
                })?;

            manager
                .initialize()
                .await
                .map_err(|e| FeedbackError::SessionError {
                    message: format!("Failed to initialize persistence backend: {e}"),
                    source: Some(Box::new(e)),
                })?;

            std::sync::Arc::new(manager)
                as std::sync::Arc<dyn crate::persistence::PersistenceManager>
        };

        Ok(Self {
            realtime,
            adaptive,
            progress,
            trainer,
            config,
            #[cfg(feature = "persistence")]
            persistence_manager,
        })
    }

    /// Create a new user session
    pub async fn create_session(
        &self,
        user_id: &str,
    ) -> Result<Box<dyn FeedbackSession>, FeedbackError> {
        let session = FeedbackSessionImpl::new(
            user_id.to_string(),
            &self.realtime,
            &self.adaptive,
            &self.progress,
            &self.trainer,
            &self.config,
            #[cfg(feature = "persistence")]
            self.persistence_manager.clone(),
        )
        .await?;

        Ok(Box::new(session))
    }

    /// Get system configuration
    #[must_use]
    pub fn config(&self) -> &FeedbackSystemConfig {
        &self.config
    }

    /// Get system statistics
    pub async fn get_statistics(&self) -> Result<FeedbackSystemStats, FeedbackError> {
        let realtime_stats = self.realtime.get_statistics().await?;
        let adaptive_stats = self.adaptive.get_statistics().await?;
        let progress_stats = self.progress.get_statistics().await?;

        // Enhanced logging for system monitoring
        log::info!(
            "System statistics: {} sessions, {} active streams, {:.2}ms avg response time",
            progress_stats.total_users,
            realtime_stats.active_streams,
            realtime_stats.average_latency_ms
        );

        Ok(FeedbackSystemStats {
            total_sessions: progress_stats.total_users,
            active_sessions: realtime_stats.active_streams,
            total_feedback_generated: adaptive_stats.total_feedback_count,
            average_response_time_ms: realtime_stats.average_latency_ms,
            system_uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default(),
        })
    }

    /// Enhanced system health monitoring
    pub async fn get_system_health(&self) -> Result<SystemHealthReport, FeedbackError> {
        let stats = self.get_statistics().await?;

        let mut health_score = 1.0;
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check response time performance
        if stats.average_response_time_ms > 500.0 {
            health_score -= 0.3;
            issues.push("High response time detected".to_string());
            recommendations.push("Consider optimizing audio processing pipeline".to_string());
        } else if stats.average_response_time_ms > 200.0 {
            health_score -= 0.1;
            issues.push("Elevated response time".to_string());
            recommendations.push("Monitor system load and resource usage".to_string());
        }

        // Check active session load
        let max_sessions = self.config.max_concurrent_sessions;
        let session_load = stats.active_sessions as f32 / max_sessions as f32;

        if session_load > 0.9 {
            health_score -= 0.2;
            issues.push("Near maximum session capacity".to_string());
            recommendations.push("Consider scaling up resources or load balancing".to_string());
        } else if session_load > 0.7 {
            health_score -= 0.1;
            issues.push("High session load".to_string());
            recommendations.push("Monitor for potential capacity issues".to_string());
        }

        // Check system uptime
        let uptime_hours = stats.system_uptime.as_secs() as f32 / 3600.0;
        if uptime_hours < 1.0 {
            health_score -= 0.1;
            issues.push("Recent system restart detected".to_string());
            recommendations.push("Monitor for startup issues or crashes".to_string());
        }

        let health_status = if health_score >= 0.9 {
            HealthStatus::Excellent
        } else if health_score >= 0.7 {
            HealthStatus::Good
        } else if health_score >= 0.5 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };

        log::info!(
            "System health: {:?} (score: {:.2}) - {} issues, {} recommendations",
            health_status,
            health_score,
            issues.len(),
            recommendations.len()
        );

        Ok(SystemHealthReport {
            health_score,
            health_status,
            issues,
            recommendations,
            timestamp: chrono::Utc::now(),
            statistics: stats,
        })
    }
}

/// Feedback system configuration
#[derive(Debug, Clone)]
pub struct FeedbackSystemConfig {
    /// Enable real-time feedback
    pub enable_realtime: bool,
    /// Enable adaptive learning
    pub enable_adaptive: bool,
    /// Enable progress tracking
    pub enable_progress_tracking: bool,
    /// Enable gamification
    pub enable_gamification: bool,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Feedback response timeout
    pub response_timeout_ms: u64,
    /// Auto-save interval for progress
    pub auto_save_interval_sec: u64,
}

impl Default for FeedbackSystemConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            enable_adaptive: true,
            enable_progress_tracking: true,
            enable_gamification: false,
            max_concurrent_sessions: 100,
            response_timeout_ms: 500,
            auto_save_interval_sec: 30,
        }
    }
}

/// Feedback system statistics
#[derive(Debug, Clone)]
pub struct FeedbackSystemStats {
    /// Total number of sessions created
    pub total_sessions: usize,
    /// Currently active sessions
    pub active_sessions: usize,
    /// Total feedback messages generated
    pub total_feedback_generated: usize,
    /// Average response time in milliseconds
    pub average_response_time_ms: f32,
    /// System uptime
    pub system_uptime: std::time::Duration,
}

/// System health report for monitoring and diagnostics
#[derive(Debug, Clone)]
pub struct SystemHealthReport {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f32,
    /// Current health status
    pub health_status: HealthStatus,
    /// List of detected issues
    pub issues: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Timestamp of health check
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Current system statistics
    pub statistics: FeedbackSystemStats,
}

/// System health status categories
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// System operating optimally
    Excellent,
    /// System operating well with minor issues
    Good,
    /// System operating with noticeable issues requiring attention
    Warning,
    /// System experiencing critical issues requiring immediate attention
    Critical,
}

/// Internal session implementation
struct FeedbackSessionImpl {
    user_id: String,
    state: SessionState,
    realtime: RealtimeFeedbackSystem,
    adaptive: AdaptiveFeedbackEngine,
    progress: ProgressAnalyzer,
    trainer: InteractiveTrainer,
    config: FeedbackSystemConfig,
    feedback_stream: Option<FeedbackStream>,
    #[cfg(feature = "persistence")]
    persistence_manager: std::sync::Arc<dyn crate::persistence::PersistenceManager>,
}

impl FeedbackSessionImpl {
    async fn new(
        user_id: String,
        realtime: &RealtimeFeedbackSystem,
        adaptive: &AdaptiveFeedbackEngine,
        progress: &ProgressAnalyzer,
        trainer: &InteractiveTrainer,
        config: &FeedbackSystemConfig,
        #[cfg(feature = "persistence")] persistence_manager: std::sync::Arc<
            dyn crate::persistence::PersistenceManager,
        >,
    ) -> Result<Self, FeedbackError> {
        let state = SessionState::new(&user_id).await?;

        Ok(Self {
            user_id,
            state,
            realtime: realtime.clone(),
            adaptive: adaptive.clone(),
            progress: progress.clone(),
            trainer: trainer.clone(),
            config: config.clone(),
            feedback_stream: None,
            #[cfg(feature = "persistence")]
            persistence_manager,
        })
    }
}

#[async_trait]
impl FeedbackSession for FeedbackSessionImpl {
    async fn process_synthesis(
        &mut self,
        audio: &AudioBuffer,
        text: &str,
    ) -> FeedbackResult<FeedbackResponse> {
        // Create feedback stream if it doesn't exist
        if self.feedback_stream.is_none() {
            let stream = self
                .realtime
                .create_stream(&self.user_id, &self.state)
                .await?;
            self.feedback_stream = Some(stream);
        }

        // Generate feedback using realtime system
        let feedback = if let Some(stream) = &self.feedback_stream {
            stream.process_audio(audio, text).await?
        } else {
            return Err(FeedbackError::RealtimeError {
                message: "Failed to create feedback stream".to_string(),
                source: None,
            }
            .into());
        };

        // Learn from this interaction
        let user_feedback = UserFeedback {
            message: text.to_string(),
            suggestion: Some("Continue practicing".to_string()),
            confidence: 0.8,
            score: 0.8,
            priority: 0.5,
            metadata: std::collections::HashMap::new(),
        };
        let interaction = UserInteraction {
            user_id: self.user_id.clone(),
            timestamp: chrono::Utc::now(),
            interaction_type: InteractionType::Practice,
            audio: audio.clone(),
            text: text.to_string(),
            feedback: feedback.clone(),
            user_response: None,
        };
        self.adaptive.learn_from_interaction(&interaction).await?;

        Ok(feedback)
    }

    async fn start_exercise(&mut self, exercise: &TrainingExercise) -> FeedbackResult<()> {
        // Update session state with current exercise
        let exercise_session = crate::traits::ExerciseSession {
            exercise_id: exercise.exercise_id.clone(),
            exercise_name: exercise.name.clone(),
            start_time: chrono::Utc::now(),
            current_attempt: 0,
            progress: 0.0,
        };

        self.state.current_exercise = Some(exercise_session);
        self.state.stats.exercises_completed += 1;

        Ok(())
    }

    async fn complete_exercise(&mut self) -> FeedbackResult<TrainingResult> {
        let exercise_session =
            self.state
                .current_exercise
                .take()
                .ok_or_else(|| FeedbackError::SessionError {
                    message: "No active exercise to complete".to_string(),
                    source: None,
                })?;

        let completion_time = chrono::Utc::now()
            .signed_duration_since(exercise_session.start_time)
            .to_std()
            .unwrap_or_default();

        // Create a basic exercise for the result
        let exercise = TrainingExercise {
            exercise_id: exercise_session.exercise_id,
            name: exercise_session.exercise_name,
            description: "Completed training exercise".to_string(),
            difficulty: 0.5,
            focus_areas: vec![FocusArea::Pronunciation, FocusArea::Quality],
            exercise_type: ExerciseType::FreeForm,
            target_text: "Practice text".to_string(),
            reference_audio: None,
            success_criteria: SuccessCriteria {
                min_quality_score: 0.7,
                min_pronunciation_score: 0.7,
                consistency_required: 1,
                max_attempts: 3,
                time_limit: Some(Duration::from_secs(300)),
            },
            estimated_duration: Duration::from_secs(300),
        };

        let final_scores = TrainingScores {
            quality: self.state.stats.average_quality,
            pronunciation: self.state.stats.average_pronunciation,
            consistency: 0.8,
            improvement: 0.1,
        };

        let success = final_scores.quality >= 0.7 && final_scores.pronunciation >= 0.7;

        let feedback = FeedbackResponse {
            feedback_items: vec![UserFeedback {
                message: if success {
                    "Great work! Exercise completed successfully.".to_string()
                } else {
                    "Good effort! Keep practicing to improve.".to_string()
                },
                suggestion: Some("Continue with the next exercise".to_string()),
                confidence: 0.8,
                score: (final_scores.quality + final_scores.pronunciation) / 2.0,
                priority: 0.5,
                metadata: std::collections::HashMap::new(),
            }],
            overall_score: (final_scores.quality + final_scores.pronunciation) / 2.0,
            immediate_actions: vec!["Review feedback and continue training".to_string()],
            long_term_goals: vec!["Maintain consistent quality".to_string()],
            progress_indicators: ProgressIndicators {
                improving_areas: vec!["Overall Performance".to_string()],
                attention_areas: Vec::new(),
                stable_areas: vec!["Basic Skills".to_string()],
                overall_trend: 0.1,
                completion_percentage: exercise_session.progress,
            },
            timestamp: chrono::Utc::now(),
            processing_time: Duration::from_millis(100),
            feedback_type: FeedbackType::Quality,
        };

        let improvement_recommendations = if success {
            vec![
                "Maintain current level of performance".to_string(),
                "Try more challenging exercises".to_string(),
            ]
        } else {
            vec![
                "Focus on audio quality improvement".to_string(),
                "Practice pronunciation accuracy".to_string(),
            ]
        };

        Ok(TrainingResult {
            exercise,
            success,
            attempts_made: exercise_session.current_attempt + 1,
            completion_time,
            final_scores,
            feedback,
            improvement_recommendations,
        })
    }

    async fn update_preferences(&mut self, preferences: UserPreferences) -> FeedbackResult<()> {
        // Update session state with new preferences
        self.state.preferences = preferences;

        // Log the preference update
        log::info!("Updated user preferences for user: {}", self.user_id);

        Ok(())
    }

    fn get_state(&self) -> &SessionState {
        &self.state
    }

    async fn save_progress(&self) -> FeedbackResult<()> {
        log::info!(
            "Saving progress for user: {} at session: {}",
            self.user_id,
            self.state.session_id
        );

        #[cfg(feature = "persistence")]
        {
            // Save session state to database
            self.persistence_manager
                .save_session(&self.state)
                .await
                .map_err(|e| FeedbackError::SessionError {
                    message: format!("Failed to save session to database: {e}"),
                    source: Some(Box::new(e)),
                })?;

            // Create UserProgress from session stats for saving
            let user_progress = UserProgress {
                user_id: self.user_id.clone(),
                overall_skill_level: (self.state.stats.average_quality
                    + self.state.stats.average_pronunciation)
                    / 2.0,
                skill_breakdown: std::collections::HashMap::new(),
                progress_history: vec![], // Empty for now
                achievements: vec![],     // Empty achievements for now
                session_count: 1,         // Current session
                total_practice_time: self.state.stats.session_duration,
                training_stats: crate::traits::TrainingStatistics {
                    total_sessions: 1,      // Default to 1 (current session)
                    successful_sessions: 1, // Assume current session is successful
                    total_training_time: self.state.stats.session_duration,
                    exercises_completed: self.state.stats.exercises_completed,
                    success_rate: 1.0,         // Default success rate
                    average_improvement: 0.05, // Default improvement
                    current_streak: 1,         // Default streak
                    longest_streak: 1,         // Default longest streak
                },
                goals: vec![], // Empty goals for now
                last_updated: chrono::Utc::now(),
                average_scores: crate::traits::SessionScores {
                    average_quality: self.state.stats.average_quality,
                    average_pronunciation: self.state.stats.average_pronunciation,
                    average_fluency: 0.0, // Default
                    overall_score: (self.state.stats.average_quality
                        + self.state.stats.average_pronunciation)
                        / 2.0,
                    improvement_trend: 0.05, // Default improvement trend
                },
                skill_levels: std::collections::HashMap::new(),
                recent_sessions: vec![],
                personal_bests: std::collections::HashMap::new(),
            };

            // Save user progress to database
            self.persistence_manager
                .save_user_progress(&self.user_id, &user_progress)
                .await
                .map_err(|e| FeedbackError::SessionError {
                    message: format!("Failed to save user progress to database: {e}"),
                    source: Some(Box::new(e)),
                })?;

            log::info!(
                "Successfully saved progress to database for user: {}",
                self.user_id
            );
        }

        #[cfg(not(feature = "persistence"))]
        {
            // Fallback to JSON serialization for logging when persistence is disabled
            let progress_data = serde_json::to_string(&self.state.stats).map_err(|e| {
                FeedbackError::SessionError {
                    message: format!("Failed to serialize progress data: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

            log::debug!("Progress data (persistence disabled): {}", progress_data);
        }

        Ok(())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create a default feedback configuration
#[must_use]
pub fn default_feedback_config() -> FeedbackConfig {
    FeedbackConfig::default()
}

/// Create a default adaptive configuration
#[must_use]
pub fn default_adaptive_config() -> AdaptiveConfig {
    AdaptiveConfig::default()
}

/// Create a default progress configuration
#[must_use]
pub fn default_progress_config() -> ProgressConfig {
    ProgressConfig::default()
}

/// Validate feedback configuration
pub fn validate_feedback_config(config: &FeedbackConfig) -> Result<(), FeedbackError> {
    if config.response_timeout_ms < 100 {
        return Err(FeedbackError::ConfigurationError {
            message: "Response timeout must be at least 100ms".to_string(),
        });
    }

    if config.feedback_detail_level > 1.0 || config.feedback_detail_level < 0.0 {
        return Err(FeedbackError::ConfigurationError {
            message: "Feedback detail level must be between 0.0 and 1.0".to_string(),
        });
    }

    Ok(())
}

/// Calculate feedback priority based on error severity
#[must_use]
pub fn calculate_feedback_priority(quality_score: f32, pronunciation_score: f32) -> f32 {
    let quality_urgency = 1.0 - quality_score;
    let pronunciation_urgency = 1.0 - pronunciation_score;

    // Weight pronunciation errors as more urgent for learning
    (quality_urgency * 0.4 + pronunciation_urgency * 0.6).min(1.0)
}

/// Format feedback message for different audiences
#[must_use]
pub fn format_feedback_message(feedback: &UserFeedback, audience: FeedbackAudience) -> String {
    match audience {
        FeedbackAudience::Beginner => {
            let suggestion = feedback.suggestion.as_deref().unwrap_or("Keep practicing!");
            format!("💡 {}\n\n📚 Tip: {suggestion}", feedback.message)
        }
        FeedbackAudience::Intermediate => {
            let suggestion = feedback
                .suggestion
                .as_deref()
                .unwrap_or("Continue refining your technique");
            format!("Analysis: {}\nImprovement: {suggestion}", feedback.message)
        }
        FeedbackAudience::Advanced => {
            let confidence = feedback.confidence * 100.0;
            let suggestion = feedback
                .suggestion
                .as_deref()
                .unwrap_or("Focus on advanced techniques");
            format!(
                "Technical feedback: {} (Confidence: {confidence:.1}%)\nOptimization: {suggestion}",
                feedback.message
            )
        }
        FeedbackAudience::Developer => {
            let message = &feedback.message;
            let score = feedback.score;
            let priority = feedback.priority;
            let metadata = &feedback.metadata;
            format!(
                    "Debug: {message} | Score: {score:.3} | Priority: {priority:.3} | Metrics: {metadata:?}"
                )
        }
    }
}

/// Feedback audience types
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackAudience {
    /// Beginner users
    Beginner,
    /// Intermediate users
    Intermediate,
    /// Advanced users
    Advanced,
    /// Developers and researchers
    Developer,
}

// ============================================================================
// Enhanced Performance Monitoring and Benchmarking
// ============================================================================

/// Performance benchmark results for system analysis
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    /// Audio processing throughput (samples per second)
    pub audio_throughput: f32,
    /// Feedback generation rate (requests per second)
    pub feedback_generation_rate: f32,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Real-time factor (processing time / audio duration)
    pub real_time_factor: f32,
    /// Benchmark timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    /// Current memory usage in bytes
    pub current_usage_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Memory allocation rate (allocations per second)
    pub allocation_rate: f32,
    /// Buffer pool efficiency (0.0 to 1.0)
    pub buffer_pool_efficiency: f32,
}

/// Enhanced system performance profiler
pub struct SystemProfiler {
    /// Start time for profiling session
    start_time: std::time::Instant,
    /// Collected performance samples
    samples: Vec<PerformanceSample>,
    /// Profiling configuration
    config: ProfilerConfig,
}

/// Individual performance sample
#[derive(Debug, Clone)]
struct PerformanceSample {
    /// Sample timestamp
    timestamp: std::time::Instant,
    /// Audio processing time
    audio_processing_time: Duration,
    /// Feedback generation time
    feedback_generation_time: Duration,
    /// Memory usage at sample time
    memory_usage: usize,
    /// Active session count
    active_sessions: usize,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Sample collection interval
    pub sample_interval: Duration,
    /// Maximum number of samples to retain
    pub max_samples: usize,
    /// Enable detailed memory tracking
    pub enable_memory_tracking: bool,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_secs(10),
            max_samples: 1000,
            enable_memory_tracking: true,
            enable_cpu_profiling: false, // Disabled by default for performance
        }
    }
}

impl SystemProfiler {
    /// Create new system profiler
    #[must_use]
    pub fn new(config: ProfilerConfig) -> Self {
        log::info!("Initializing system profiler with config: {config:?}");

        Self {
            start_time: std::time::Instant::now(),
            samples: Vec::with_capacity(config.max_samples),
            config,
        }
    }

    /// Record a performance sample
    pub fn record_sample(
        &mut self,
        audio_processing_time: Duration,
        feedback_generation_time: Duration,
        active_sessions: usize,
    ) {
        let memory_usage = if self.config.enable_memory_tracking {
            self.get_current_memory_usage()
        } else {
            0
        };

        let sample = PerformanceSample {
            timestamp: std::time::Instant::now(),
            audio_processing_time,
            feedback_generation_time,
            memory_usage,
            active_sessions,
        };

        self.samples.push(sample);

        // Keep samples within limit
        if self.samples.len() > self.config.max_samples {
            self.samples.remove(0);
        }

        log::debug!(
            "Recorded performance sample: audio_time={:?}, feedback_time={:?}, memory={}KB, sessions={}",
            audio_processing_time,
            feedback_generation_time,
            memory_usage / 1024,
            active_sessions
        );
    }

    /// Generate comprehensive performance benchmark
    pub fn generate_benchmark(&self) -> Result<PerformanceBenchmark, FeedbackError> {
        if self.samples.is_empty() {
            return Err(FeedbackError::ConfigurationError {
                message: "No performance samples available for benchmark".to_string(),
            });
        }

        let total_audio_time: Duration = self.samples.iter().map(|s| s.audio_processing_time).sum();

        let total_feedback_time: Duration = self
            .samples
            .iter()
            .map(|s| s.feedback_generation_time)
            .sum();

        let total_session_time = self.start_time.elapsed();

        // Calculate throughput metrics
        let audio_throughput = if total_session_time.as_secs_f32() > 0.0 {
            // Assume 16kHz sample rate for throughput calculation
            (self.samples.len() * 16000) as f32 / total_session_time.as_secs_f32()
        } else {
            0.0
        };

        let feedback_generation_rate = if total_session_time.as_secs_f32() > 0.0 {
            self.samples.len() as f32 / total_session_time.as_secs_f32()
        } else {
            0.0
        };

        // Calculate real-time factor
        let real_time_factor = if total_audio_time.as_secs_f32() > 0.0 {
            total_feedback_time.as_secs_f32() / total_audio_time.as_secs_f32()
        } else {
            0.0
        };

        // Memory usage statistics
        let memory_usage = self.calculate_memory_stats();

        // CPU utilization (simplified calculation)
        let cpu_utilization = self.estimate_cpu_utilization();

        let benchmark = PerformanceBenchmark {
            audio_throughput,
            feedback_generation_rate,
            memory_usage,
            cpu_utilization,
            real_time_factor,
            timestamp: chrono::Utc::now(),
        };

        log::info!(
            "Generated performance benchmark: throughput={:.0} samples/s, rate={:.2} req/s, RTF={:.3}, CPU={:.1}%",
            benchmark.audio_throughput,
            benchmark.feedback_generation_rate,
            benchmark.real_time_factor,
            benchmark.cpu_utilization
        );

        Ok(benchmark)
    }

    /// Get current memory usage in bytes
    fn get_current_memory_usage(&self) -> usize {
        // Simplified memory usage estimation
        // In a real implementation, this would use platform-specific APIs
        std::mem::size_of::<Self>()
            + (self.samples.len() * std::mem::size_of::<PerformanceSample>())
    }

    /// Calculate memory usage statistics
    fn calculate_memory_stats(&self) -> MemoryUsageStats {
        if self.samples.is_empty() {
            return MemoryUsageStats {
                current_usage_bytes: 0,
                peak_usage_bytes: 0,
                allocation_rate: 0.0,
                buffer_pool_efficiency: 1.0,
            };
        }

        let current_usage = self.samples.last().unwrap().memory_usage;
        let peak_usage = self
            .samples
            .iter()
            .map(|s| s.memory_usage)
            .max()
            .unwrap_or(0);

        let allocation_rate = if self.start_time.elapsed().as_secs_f32() > 0.0 {
            self.samples.len() as f32 / self.start_time.elapsed().as_secs_f32()
        } else {
            0.0
        };

        // Buffer pool efficiency (simplified calculation)
        let buffer_pool_efficiency = if peak_usage > 0 {
            (current_usage as f32 / peak_usage as f32).min(1.0)
        } else {
            1.0
        };

        MemoryUsageStats {
            current_usage_bytes: current_usage,
            peak_usage_bytes: peak_usage,
            allocation_rate,
            buffer_pool_efficiency,
        }
    }

    /// Estimate CPU utilization
    fn estimate_cpu_utilization(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        // Simplified CPU utilization calculation
        // In reality, this would use platform-specific performance counters
        let total_processing_time: Duration = self
            .samples
            .iter()
            .map(|s| s.audio_processing_time + s.feedback_generation_time)
            .sum();

        let total_wall_time = self.start_time.elapsed();

        if total_wall_time.as_secs_f32() > 0.0 {
            (total_processing_time.as_secs_f32() / total_wall_time.as_secs_f32() * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Reset profiler and clear collected samples
    pub fn reset(&mut self) {
        log::info!(
            "Resetting system profiler - clearing {} samples",
            self.samples.len()
        );
        self.start_time = std::time::Instant::now();
        self.samples.clear();
    }

    /// Get current sample count
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get profiling duration
    #[must_use]
    pub fn profiling_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_default_configs() {
        let feedback_config = default_feedback_config();
        assert!(feedback_config.enable_realtime);

        let adaptive_config = default_adaptive_config();
        assert!(adaptive_config.enable_learning);

        let progress_config = default_progress_config();
        assert!(progress_config.track_improvements);
    }

    #[test]
    fn test_config_validation() {
        let mut config = default_feedback_config();
        assert!(validate_feedback_config(&config).is_ok());

        // Test invalid timeout
        config.response_timeout_ms = 50;
        assert!(validate_feedback_config(&config).is_err());

        // Test invalid detail level
        config.response_timeout_ms = 200;
        config.feedback_detail_level = 1.5;
        assert!(validate_feedback_config(&config).is_err());
    }

    #[test]
    fn test_feedback_priority() {
        let priority = calculate_feedback_priority(0.8, 0.6);
        assert!(priority > 0.0);
        assert!(priority <= 1.0);

        // Low quality should increase priority
        let high_priority = calculate_feedback_priority(0.3, 0.3);
        assert!(high_priority > priority);
    }

    #[test]
    fn test_feedback_formatting() {
        let feedback = UserFeedback {
            message: "Test message".to_string(),
            suggestion: Some("Test suggestion".to_string()),
            confidence: 0.85,
            score: 0.7,
            priority: 0.5,
            metadata: std::collections::HashMap::new(),
        };

        let beginner_msg = format_feedback_message(&feedback, FeedbackAudience::Beginner);
        assert!(beginner_msg.contains("💡"));
        assert!(beginner_msg.contains("📚"));

        let developer_msg = format_feedback_message(&feedback, FeedbackAudience::Developer);
        assert!(developer_msg.contains("Debug:"));
        assert!(developer_msg.contains("Score:"));
    }
}
