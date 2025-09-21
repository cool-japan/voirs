//! Configuration structures for visualization components

#[cfg(feature = "ui")]
use crate::visualization::types::{ColorScheme, FontSizes, TimelineRange};
#[cfg(feature = "ui")]
use chrono::{DateTime, Utc};
#[cfg(feature = "ui")]
use egui::{Color32, Vec2};
#[cfg(feature = "ui")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Main Visualization Configuration
// ============================================================================

/// Visualization configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Enable animations
    pub enable_animations: bool,
    /// Chart update frequency
    pub update_frequency_ms: u64,
    /// Maximum data points to display
    pub max_data_points: usize,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Default chart size
    pub default_chart_size: (u32, u32),
}

#[cfg(feature = "ui")]
impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_animations: true,
            update_frequency_ms: 100,
            max_data_points: 100,
            color_scheme: ColorScheme::Dark,
            default_chart_size: (800, 400),
        }
    }
}

/// Visualization theme
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct VisualizationTheme {
    /// Primary colors
    pub primary_color: Color32,
    /// Secondary colors
    pub secondary_color: Color32,
    /// Background color
    pub background_color: Color32,
    /// Text color
    pub text_color: Color32,
    /// Border radius for elements
    pub border_radius: f32,
    /// Font sizes
    pub font_sizes: FontSizes,
}

#[cfg(feature = "ui")]
impl Default for VisualizationTheme {
    fn default() -> Self {
        Self {
            primary_color: Color32::from_rgb(100, 150, 255),
            secondary_color: Color32::from_rgb(150, 100, 255),
            background_color: Color32::from_gray(40),
            text_color: Color32::WHITE,
            border_radius: 5.0,
            font_sizes: FontSizes::default(),
        }
    }
}

// ============================================================================
// Chart Configuration
// ============================================================================

/// Chart configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ChartConfig {
    /// Chart title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Show grid
    pub show_grid: bool,
    /// Line color
    pub line_color: String,
    /// Fill area under line
    pub fill_area: bool,
    /// Show data points
    pub show_points: bool,
}

#[cfg(feature = "ui")]
impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            title: "Progress Chart".to_string(),
            x_label: "Time".to_string(),
            y_label: "Score".to_string(),
            show_grid: true,
            line_color: "#4A90E2".to_string(),
            fill_area: false,
            show_points: true,
        }
    }
}

/// Chart data point
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ChartDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f32,
    /// Optional label
    pub label: Option<String>,
}

/// Cached chart for performance optimization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct CachedChart {
    /// Chart data
    pub data: Vec<ChartDataPoint>,
    /// Generated chart content
    pub content: String,
    /// Last update time
    pub last_updated: DateTime<Utc>,
}

// ============================================================================
// Real-time Configuration
// ============================================================================

/// Real-time configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Maximum history points to keep
    pub max_history_points: usize,
    /// Update interval
    pub update_interval_ms: u64,
    /// Show sparklines
    pub show_sparklines: bool,
}

#[cfg(feature = "ui")]
impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            max_history_points: 50,
            update_interval_ms: 100,
            show_sparklines: true,
        }
    }
}

// ============================================================================
// Radar Chart Configuration
// ============================================================================

/// Radar chart configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RadarChartConfig {
    /// Chart size
    pub chart_size: Vec2,
    /// Primary data color
    pub primary_color: Color32,
    /// Comparison data color
    pub comparison_color: Color32,
    /// Background color
    pub background_color: Color32,
    /// Enable animations
    pub enable_animations: bool,
    /// Animation duration
    pub animation_duration: f32,
}

#[cfg(feature = "ui")]
impl Default for RadarChartConfig {
    fn default() -> Self {
        Self {
            chart_size: Vec2::new(300.0, 300.0),
            primary_color: Color32::from_rgb(100, 150, 255),
            comparison_color: Color32::from_rgb(255, 100, 100),
            background_color: Color32::from_gray(40),
            enable_animations: true,
            animation_duration: 1.0,
        }
    }
}

// ============================================================================
// Timeline Configuration
// ============================================================================

/// Interactive timeline configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct TimelineConfig {
    /// Whether zoom functionality is enabled
    pub enable_zoom: bool,
    /// Default time range for timeline display
    pub default_time_range: TimelineRange,
}

#[cfg(feature = "ui")]
impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            enable_zoom: true,
            default_time_range: TimelineRange::Week,
        }
    }
}

// ============================================================================
// Progress Visualization Configuration
// ============================================================================

/// Progress visualization configuration
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct ProgressVisualizationConfig {
    /// Enable animations
    pub enable_animations: bool,
    /// Animation duration
    pub animation_duration: f32,
    /// Chart colors
    pub colors: Vec<Color32>,
    /// Show data labels
    pub show_labels: bool,
}

#[cfg(feature = "ui")]
impl Default for ProgressVisualizationConfig {
    fn default() -> Self {
        Self {
            enable_animations: true,
            animation_duration: 1.0,
            colors: vec![
                Color32::from_rgb(100, 150, 255),
                Color32::from_rgb(255, 150, 100),
                Color32::from_rgb(150, 255, 100),
                Color32::from_rgb(255, 100, 150),
            ],
            show_labels: true,
        }
    }
}

// ============================================================================
// Stub Types for Non-UI Feature
// ============================================================================

/// Visualization configuration (stub implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, Default)]
pub struct VisualizationConfig;

/// Visualization theme (stub implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, Default)]
pub struct VisualizationTheme;

/// Cached chart (stub implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, Default)]
pub struct CachedChart;

/// Chart configuration (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct ChartConfig {
    /// Chart title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Show grid
    pub show_grid: bool,
    /// Line color
    pub line_color: String,
    /// Fill area under line
    pub fill_area: bool,
    /// Show data points
    pub show_points: bool,
}

#[cfg(not(feature = "ui"))]
impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            title: "Progress Chart".to_string(),
            x_label: "Time".to_string(),
            y_label: "Score".to_string(),
            show_grid: true,
            line_color: "#4A90E2".to_string(),
            fill_area: false,
            show_points: true,
        }
    }
}

/// Chart data point (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChartDataPoint {
    /// Data point timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Data point value
    pub value: f32,
    /// Optional data point label
    pub label: Option<String>,
}

/// Real-time configuration (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Buffer size for real-time data
    pub buffer_size: usize,
    /// Enable auto-scaling
    pub auto_scale: bool,
}

#[cfg(not(feature = "ui"))]
impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 100,
            buffer_size: 50,
            auto_scale: true,
        }
    }
}

/// Radar chart configuration (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct RadarChartConfig {
    /// Chart size as (width, height)
    pub chart_size: (f32, f32),
    /// Primary color (as hex string)
    pub primary_color: String,
    /// Comparison color (as hex string)
    pub comparison_color: String,
    /// Background color (as hex string)
    pub background_color: String,
    /// Enable animations
    pub enable_animations: bool,
    /// Animation duration in seconds
    pub animation_duration: f32,
}

#[cfg(not(feature = "ui"))]
impl Default for RadarChartConfig {
    fn default() -> Self {
        Self {
            chart_size: (300.0, 300.0),
            primary_color: "#6496FF".to_string(),
            comparison_color: "#FF6464".to_string(),
            background_color: "#282828".to_string(),
            enable_animations: true,
            animation_duration: 1.0,
        }
    }
}

/// Timeline configuration (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct TimelineConfig {
    /// Whether zoom functionality is enabled
    pub enable_zoom: bool,
    /// Default time range for timeline display
    pub default_time_range: TimelineRange,
}

#[cfg(not(feature = "ui"))]
impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            enable_zoom: true,
            default_time_range: TimelineRange::Week,
        }
    }
}

/// Timeline range enum for non-UI implementation
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub enum TimelineRange {
    /// Show events from the last day
    Day,
    /// Show events from the last week
    Week,
    /// Show events from the last month
    Month,
    /// Show events from the last year
    Year,
    /// Show all events
    All,
}

#[cfg(not(feature = "ui"))]
impl Default for TimelineRange {
    fn default() -> Self {
        TimelineRange::Week
    }
}

/// Progress visualization configuration (enhanced implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct ProgressVisualizationConfig {
    /// Enable animations
    pub enable_animations: bool,
    /// Animation duration
    pub animation_duration: f32,
    /// Chart colors (as hex strings)
    pub colors: Vec<String>,
    /// Show data labels
    pub show_labels: bool,
}

#[cfg(not(feature = "ui"))]
impl Default for ProgressVisualizationConfig {
    fn default() -> Self {
        Self {
            enable_animations: true,
            animation_duration: 1.0,
            colors: vec![
                "#6496FF".to_string(),
                "#FF9664".to_string(),
                "#96FF64".to_string(),
                "#FF6496".to_string(),
            ],
            show_labels: true,
        }
    }
}
