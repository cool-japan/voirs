//! Visualization and UI components for feedback display
//!
//! This module provides visual components for displaying feedback, progress,
//! and analytics in an intuitive and engaging way.

// Module declarations
pub mod charts;
pub mod config;
pub mod core;
pub mod realtime;
pub mod types;

// Re-export main components
pub use core::FeedbackVisualizer;

// Re-export types (feature-gated)
#[cfg(feature = "ui")]
pub use types::{
    ActiveGoal,
    // Alignment and Timing Types
    AlignmentData,
    AlignmentMarker,
    AlignmentQuality,
    AnimationDirection,

    // Animation Types
    AnimationState,
    // Color and Theme Types
    ColorScheme,
    CustomColorPalette,
    DashboardSection,

    ExportFormat,
    // Export and Privacy Types
    ExportOptions,
    FeedbackCategory,
    FontSizes,

    FrameLimiter,
    GoalCategory,
    GoalPriority,

    // Feedback and Goal Types
    InstantFeedback,
    // Layout and UI Types
    LayoutPreferences,
    LayoutType,
    PerformanceMode,

    // Performance Types
    PerformanceOptions,
    // Playback Types
    PlaybackState,
    PlaybackTrack,

    PrivacyLevel,

    // Problem and Quality Types
    ProblemArea,
    ProblemCategory,

    ProblemSeverity,
    ProgressCategory,

    ProgressDataPoint,
    RadarAnimationState,

    // Radar Chart Types
    RadarSkill,
    // Real-time Dashboard Data
    RealtimeDashboardData,
    ThemePreference,
    TimelineCategory,
    TimelineDataPoint,
    // Timeline and Progress Types
    TimelineRange,
    TimingOffset,

    TrainingSessionResult,
};

// Re-export configuration (feature-gated)
#[cfg(feature = "ui")]
pub use config::{
    CachedChart, ChartConfig, ChartDataPoint, ProgressVisualizationConfig, RadarChartConfig,
    RealtimeConfig, TimelineConfig, VisualizationConfig, VisualizationTheme,
};

#[cfg(not(feature = "ui"))]
pub use config::{
    CachedChart, ChartConfig, ChartDataPoint, ProgressVisualizationConfig, RadarChartConfig,
    RealtimeConfig, TimelineConfig, VisualizationConfig, VisualizationTheme,
};

// Re-export chart components (feature-gated)
#[cfg(feature = "ui")]
pub use charts::{
    EnhancedRadarChart, InteractiveTimeline, ProgressChart, RichProgressVisualization,
};

#[cfg(not(feature = "ui"))]
pub use charts::{
    EnhancedRadarChart, InteractiveTimeline, ProgressChart, RichProgressVisualization,
};

// Re-export real-time components (feature-gated)
#[cfg(feature = "ui")]
pub use realtime::{RealtimeDashboard, RealtimeWidget};

#[cfg(not(feature = "ui"))]
pub use realtime::{RealtimeDashboard, RealtimeWidget};

// Re-export everything for backwards compatibility
#[cfg(feature = "ui")]
pub use crate::traits::*;

// Provide convenience imports for common use cases
#[cfg(feature = "ui")]
/// Description
pub mod prelude {
    pub use super::{
        ColorScheme, EnhancedRadarChart, FeedbackVisualizer, PerformanceOptions, ProgressChart,
        RealtimeDashboard, RealtimeWidget, VisualizationConfig, VisualizationTheme,
    };
}
