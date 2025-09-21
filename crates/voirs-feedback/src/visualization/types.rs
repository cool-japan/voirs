//! Basic types and data structures for visualization

#[cfg(feature = "ui")]
use crate::traits::*;
#[cfg(feature = "ui")]
use chrono::{DateTime, Utc};
#[cfg(feature = "ui")]
use egui::{Color32, Vec2};
#[cfg(feature = "ui")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(feature = "ui")]
use std::time::Duration;

// ============================================================================
// Color and Theme Types
// ============================================================================

/// Color schemes for visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub enum ColorScheme {
    /// Dark theme
    Dark,
    /// Light theme
    Light,
    /// High contrast theme
    HighContrast,
    /// Custom color palette
    Custom(CustomColorPalette),
}

/// Custom color palette
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub struct CustomColorPalette {
    /// Background color
    pub background: Color32,
    /// Primary text color
    pub text: Color32,
    /// Accent color
    pub accent: Color32,
    /// Success color
    pub success: Color32,
    /// Warning color
    pub warning: Color32,
    /// Error color
    pub error: Color32,
}

/// Font size configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct FontSizes {
    /// Small text
    pub small: f32,
    /// Normal text
    pub normal: f32,
    /// Large text
    pub large: f32,
    /// Heading text
    pub heading: f32,
}

#[cfg(feature = "ui")]
impl Default for FontSizes {
    fn default() -> Self {
        Self {
            small: 10.0,
            normal: 12.0,
            large: 14.0,
            heading: 18.0,
        }
    }
}

// ============================================================================
// Problem and Quality Types
// ============================================================================

/// Problem area identification for audio highlighting
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ProblemArea {
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Problem description
    pub description: String,
    /// Severity level
    pub severity: ProblemSeverity,
    /// Problem category
    pub category: ProblemCategory,
}

/// Problem severity levels
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProblemSeverity {
    /// Critical issues requiring immediate attention
    Critical,
    /// High priority issues
    High,
    /// Medium priority issues
    Medium,
    /// Low priority issues
    Low,
    /// Informational markers
    Info,
}

/// Problem categories for classification
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemCategory {
    /// Pronunciation issues
    Pronunciation,
    /// Timing and rhythm issues
    Timing,
    /// Audio quality issues
    Quality,
    /// Stress and intonation issues
    Prosody,
    /// Breathing and pacing issues
    Breathing,
}

// ============================================================================
// Alignment and Timing Types
// ============================================================================

/// Temporal alignment data for visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct AlignmentData {
    /// Overall alignment score [0.0, 1.0]
    pub overall_score: f32,
    /// Timing accuracy [0.0, 1.0]
    pub timing_accuracy: f32,
    /// Average timing offset in milliseconds
    pub average_offset_ms: f32,
    /// Maximum timing offset in milliseconds
    pub max_offset_ms: f32,
    /// Alignment markers for visualization
    pub alignment_markers: Vec<AlignmentMarker>,
    /// Timing offset data points
    pub timing_offsets: Vec<TimingOffset>,
}

#[cfg(feature = "ui")]
impl Default for AlignmentData {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            timing_accuracy: 0.0,
            average_offset_ms: 0.0,
            max_offset_ms: 0.0,
            alignment_markers: Vec::new(),
            timing_offsets: Vec::new(),
        }
    }
}

/// Alignment marker for temporal visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct AlignmentMarker {
    /// Timestamp in seconds
    pub timestamp: f32,
    /// Alignment quality at this point
    pub alignment_quality: AlignmentQuality,
    /// Reference timestamp
    pub reference_timestamp: f32,
}

/// Alignment quality levels
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentQuality {
    /// Good alignment
    Good,
    /// Fair alignment
    Fair,
    /// Poor alignment
    Poor,
}

/// Timing offset data point
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct TimingOffset {
    /// Position in audio (seconds)
    pub position: f32,
    /// Offset in milliseconds (positive = late, negative = early)
    pub offset_ms: f32,
    /// Confidence in the offset measurement [0.0, 1.0]
    pub confidence: f32,
}

// ============================================================================
// Playback Types
// ============================================================================

/// Playback state for audio controls
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct PlaybackState {
    /// Whether audio is currently playing
    pub is_playing: bool,
    /// Current playback position in seconds
    pub current_position: f32,
    /// Total duration of audio in seconds
    pub total_duration: f32,
    /// Playback speed multiplier
    pub playback_speed: f32,
    /// Current audio track being played
    pub current_track: PlaybackTrack,
}

#[cfg(feature = "ui")]
impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            is_playing: false,
            current_position: 0.0,
            total_duration: 0.0,
            playback_speed: 1.0,
            current_track: PlaybackTrack::None,
        }
    }
}

/// Audio track types for playback
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub enum PlaybackTrack {
    /// No audio selected
    None,
    /// Reference audio
    Reference,
    /// User audio
    User,
    /// Comparison mode (alternating)
    Comparison,
}

// ============================================================================
// Radar Chart Types
// ============================================================================

/// Radar chart skill data
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RadarSkill {
    /// Skill name
    pub name: String,
    /// Skill value (0.0 to 1.0)
    pub value: f32,
    /// Whether to show the value label
    pub show_value: bool,
    /// Description for tooltips
    pub description: String,
}

/// Radar chart animation state
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RadarAnimationState {
    /// Current animation progress (0.0 to 1.0)
    pub progress: f32,
    /// Whether animation is active
    pub is_animating: bool,
    /// Animation start time
    pub start_time: std::time::Instant,
}

#[cfg(feature = "ui")]
impl Default for RadarAnimationState {
    fn default() -> Self {
        Self {
            progress: 0.0,
            is_animating: false,
            start_time: std::time::Instant::now(),
        }
    }
}

// ============================================================================
// Export and Privacy Types
// ============================================================================

/// Export options for sharing and exporting data
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Export format type
    pub format: ExportFormat,
    /// Whether to include detailed information
    pub include_details: bool,
    /// Privacy level for exported data
    pub privacy_level: PrivacyLevel,
    /// Whether to include personal user data
    pub include_personal_data: bool,
    /// Whether to anonymize sensitive data
    pub anonymize_data: bool,
    /// Whether to include audio sample data
    pub include_audio_samples: bool,
}

#[cfg(feature = "ui")]
impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::JSON,
            include_details: true,
            privacy_level: PrivacyLevel::Minimal,
            include_personal_data: false,
            anonymize_data: true,
            include_audio_samples: false,
        }
    }
}

/// Export format types for data export
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// JSON format for structured data
    JSON,
    /// CSV format for tabular data
    CSV,
    /// PDF format for printable documents
    PDF,
    /// PNG format for images
    PNG,
    /// HTML format for web display
    HTML,
    /// Shareable URL format
    URL,
}

/// Privacy level for exported data
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum PrivacyLevel {
    /// Full data including personal information
    Full,
    /// Anonymized data with personal information removed
    Anonymized,
    /// Public data suitable for sharing
    Public,
    /// Minimal data with only essential information
    Minimal,
}

// ============================================================================
// Feedback and Goal Types
// ============================================================================

/// Instant feedback message
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct InstantFeedback {
    /// Feedback message
    pub message: String,
    /// Message priority (0.0 to 1.0)
    pub priority: f32,
    /// Timestamp in milliseconds
    pub timestamp_ms: f64,
    /// Feedback category
    pub category: FeedbackCategory,
}

/// Feedback category for instant feedback
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum FeedbackCategory {
    /// Quality-related feedback
    Quality,
    /// Pronunciation-related feedback
    Pronunciation,
    /// Fluency-related feedback
    Fluency,
    /// Technical feedback
    Technical,
    /// Encouragement and motivation
    Motivation,
}

/// Active goal for goal tracking
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ActiveGoal {
    /// Goal name
    pub name: String,
    /// Goal description
    pub description: String,
    /// Goal progress (0.0 to 1.0)
    pub progress: f32,
    /// Goal due date
    pub due_date: DateTime<Utc>,
    /// Goal category
    pub category: GoalCategory,
    /// Goal priority
    pub priority: GoalPriority,
}

/// Goal category
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum GoalCategory {
    /// Daily practice goal
    Daily,
    /// Weekly achievement goal
    Weekly,
    /// Monthly milestone goal
    Monthly,
    /// Skill-specific goal
    Skill,
    /// Custom user-defined goal
    Custom,
}

/// Goal priority
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum GoalPriority {
    /// High priority goal
    High,
    /// Medium priority goal
    Medium,
    /// Low priority goal
    Low,
}

// ============================================================================
// Layout and UI Types
// ============================================================================

/// Layout preferences for dashboard customization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct LayoutPreferences {
    /// Layout type
    pub layout_type: LayoutType,
    /// Theme preference
    pub theme: ThemePreference,
    /// Whether to show animations
    pub show_animations: bool,
    /// Whether to show detailed metrics
    pub show_detailed_metrics: bool,
    /// Preferred dashboard sections
    pub enabled_sections: Vec<DashboardSection>,
}

/// Layout type for dashboard
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum LayoutType {
    /// Grid layout
    Grid,
    /// List layout
    List,
    /// Compact layout
    Compact,
    /// Custom layout
    Custom,
}

/// Theme preference
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum ThemePreference {
    /// Dark theme
    Dark,
    /// Light theme
    Light,
    /// Auto theme based on system
    Auto,
    /// Custom theme
    Custom,
}

/// Dashboard section types
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum DashboardSection {
    /// Performance metrics section
    Performance,
    /// Instant feedback section
    Feedback,
    /// Progress indicators section
    Progress,
    /// Goal tracking section
    Goals,
    /// Achievement showcase section
    Achievements,
    /// Settings section
    Settings,
}

// ============================================================================
// Timeline and Progress Types
// ============================================================================

/// Timeline range enumeration for time-based filtering
#[cfg(feature = "ui")]
#[derive(Debug, Clone, PartialEq)]
pub enum TimelineRange {
    /// Single day view
    Day,
    /// Weekly view
    Week,
    /// Monthly view
    Month,
    /// Yearly view
    Year,
    /// Custom range
    Custom,
}

/// Timeline category
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum TimelineCategory {
    /// Session data
    Session,
    /// Achievement data
    Achievement,
    /// Goal milestones
    Goal,
    /// Skill improvements
    Skill,
}

/// Timeline data point
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct TimelineDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f32,
    /// Optional label
    pub label: Option<String>,
    /// Data category
    pub category: TimelineCategory,
}

/// Progress data point
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ProgressDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Progress value (0.0 to 1.0)
    pub progress: f32,
    /// Category
    pub category: ProgressCategory,
    /// Optional metadata
    pub metadata: Option<String>,
}

/// Progress category
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum ProgressCategory {
    /// Overall progress
    Overall,
    /// Skill-specific progress
    Skill,
    /// Goal progress
    Goal,
    /// Session progress
    Session,
}

// ============================================================================
// Animation Types
// ============================================================================

/// Animation state
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct AnimationState {
    /// Current progress (0.0 to 1.0)
    pub progress: f32,
    /// Animation direction
    pub direction: AnimationDirection,
    /// Is animation active
    pub is_active: bool,
}

#[cfg(feature = "ui")]
impl Default for AnimationState {
    fn default() -> Self {
        Self {
            progress: 0.0,
            direction: AnimationDirection::Forward,
            is_active: false,
        }
    }
}

#[cfg(feature = "ui")]
impl AnimationState {
    /// Update animation state
    pub fn update(&mut self, delta_time: f32) {
        if self.is_active {
            match self.direction {
                AnimationDirection::Forward => {
                    self.progress += delta_time;
                    if self.progress >= 1.0 {
                        self.progress = 1.0;
                        self.is_active = false;
                    }
                }
                AnimationDirection::Backward => {
                    self.progress -= delta_time;
                    if self.progress <= 0.0 {
                        self.progress = 0.0;
                        self.is_active = false;
                    }
                }
                AnimationDirection::PingPong => {
                    self.progress += delta_time;
                    if self.progress >= 1.0 {
                        self.progress = 1.0;
                        self.direction = AnimationDirection::Backward;
                    } else if self.progress <= 0.0 {
                        self.progress = 0.0;
                        self.direction = AnimationDirection::Forward;
                    }
                }
            }
        }
    }
}

/// Animation direction
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub enum AnimationDirection {
    /// Forward animation
    Forward,
    /// Backward animation
    Backward,
    /// Ping-pong animation
    PingPong,
}

// ============================================================================
// Real-time Dashboard Types
// ============================================================================

/// Real-time dashboard data structure
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RealtimeDashboardData {
    /// Current CPU usage (0.0 to 1.0)
    pub cpu_usage: f32,
    /// Current memory usage (0.0 to 1.0)
    pub memory_usage: f32,
    /// Current audio processing latency (0.0 to 1.0)
    pub audio_latency: f32,
    /// Current user score (0.0 to 1.0)
    pub current_score: f32,
    /// Queue of instant feedback messages
    pub feedback_queue: Vec<InstantFeedback>,
    /// Current session progress (0.0 to 1.0)
    pub session_progress: f32,
    /// Daily goal progress (0.0 to 1.0)
    pub daily_goal_progress: f32,
    /// Weekly goal progress (0.0 to 1.0)
    pub weekly_goal_progress: f32,
    /// List of active goals
    pub active_goals: Vec<ActiveGoal>,
    /// User layout preferences
    pub layout_preferences: LayoutPreferences,
}

/// Training session result data for dashboard display
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct TrainingSessionResult {
    /// Total number of exercises completed
    pub total_exercises: u32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Duration of the session
    pub session_duration: std::time::Duration,
    /// Average scores by category
    pub average_scores: HashMap<String, f32>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

// ============================================================================
// Performance Types
// ============================================================================

/// Performance optimization settings for UI responsiveness
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct PerformanceOptions {
    /// Maximum frame rate for UI updates
    pub max_fps: f32,
    /// Enable automatic level-of-detail adjustments
    pub enable_lod: bool,
    /// Cache rendered elements for better performance
    pub enable_caching: bool,
    /// Reduce visual quality for better performance
    pub reduce_quality_on_lag: bool,
    /// Maximum processing time per frame (milliseconds)
    pub max_frame_time_ms: f32,
    /// Enable background rendering optimizations
    pub enable_background_optimization: bool,
}

#[cfg(feature = "ui")]
impl Default for PerformanceOptions {
    fn default() -> Self {
        Self {
            max_fps: 60.0,
            enable_lod: true,
            enable_caching: true,
            reduce_quality_on_lag: true,
            max_frame_time_ms: 16.0, // ~60 FPS
            enable_background_optimization: true,
        }
    }
}

/// Frame rate limiter to prevent excessive UI redraws
#[cfg(feature = "ui")]
#[derive(Debug)]
pub struct FrameLimiter {
    target_fps: f32,
    last_frame_time: std::time::Instant,
    frame_time_target: Duration,
    performance_mode: PerformanceMode,
}

#[cfg(feature = "ui")]
#[derive(Debug, Clone, Copy)]
pub enum PerformanceMode {
    /// High quality rendering with potential slower frame rates
    Quality,
    /// Balanced quality and performance
    Balanced,
    /// Performance optimized with reduced quality
    Performance,
}

#[cfg(feature = "ui")]
impl FrameLimiter {
    /// Create a new frame limiter with target FPS
    pub fn new(target_fps: f32) -> Self {
        Self {
            target_fps,
            last_frame_time: std::time::Instant::now(),
            frame_time_target: Duration::from_nanos((1_000_000_000.0 / target_fps) as u64),
            performance_mode: PerformanceMode::Balanced,
        }
    }

    /// Check if a new frame should be rendered
    pub fn should_render(&mut self) -> bool {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_frame_time);

        if elapsed >= self.frame_time_target {
            self.last_frame_time = now;
            true
        } else {
            false
        }
    }

    /// Adjust performance mode based on system load
    pub fn adjust_performance_mode(&mut self, avg_frame_time: Duration) {
        let frame_time_ms = avg_frame_time.as_secs_f32() * 1000.0;

        self.performance_mode = if frame_time_ms > 20.0 {
            PerformanceMode::Performance
        } else if frame_time_ms > 16.0 {
            PerformanceMode::Balanced
        } else {
            PerformanceMode::Quality
        };
    }

    /// Get current performance mode
    pub fn get_performance_mode(&self) -> PerformanceMode {
        self.performance_mode
    }

    /// Update target FPS
    pub fn set_target_fps(&mut self, fps: f32) {
        self.target_fps = fps;
        self.frame_time_target = Duration::from_nanos((1_000_000_000.0 / fps) as u64);
    }
}
