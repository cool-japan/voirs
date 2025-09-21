//! Chart-related functionality and components

#[cfg(feature = "ui")]
use crate::visualization::config::{
    ChartConfig, ChartDataPoint, ProgressVisualizationConfig, RadarChartConfig, TimelineConfig,
};
#[cfg(feature = "ui")]
use crate::visualization::types::{
    AnimationState, ProgressDataPoint, RadarAnimationState, RadarSkill, TimelineCategory,
    TimelineDataPoint, TimelineRange,
};
#[cfg(feature = "ui")]
use crate::FeedbackError;
#[cfg(feature = "ui")]
use chrono::{DateTime, Utc};
#[cfg(feature = "ui")]
use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use std::collections::HashMap;

#[cfg(feature = "ui")]
use plotters::prelude::*;

// ============================================================================
// Progress Chart Implementation
// ============================================================================

/// Progress chart generator for creating standalone charts
#[cfg(feature = "ui")]
pub struct ProgressChart {
    /// Chart configuration
    config: ChartConfig,
    /// Chart data
    data: Vec<ChartDataPoint>,
}

#[cfg(feature = "ui")]
impl ProgressChart {
    /// Create a new progress chart
    pub fn new() -> Self {
        Self {
            config: ChartConfig::default(),
            data: Vec::new(),
        }
    }

    /// Add data point to the chart
    pub fn add_data_point(&mut self, timestamp: DateTime<Utc>, value: f32, label: Option<String>) {
        self.data.push(ChartDataPoint {
            timestamp,
            value,
            label,
        });
    }

    /// Generate SVG chart
    #[cfg(feature = "ui")]
    pub fn generate_svg(&self, width: u32, height: u32) -> Result<String, FeedbackError> {
        if self.data.is_empty() {
            return Err(FeedbackError::InvalidInput {
                message: String::from("No data points to chart"),
            });
        }

        let mut svg_buffer = String::new();

        {
            let backend = SVGBackend::with_string(&mut svg_buffer, (width, height));
            let root = backend.into_drawing_area();
            root.fill(&WHITE)
                .map_err(|e| FeedbackError::ConfigurationError {
                    message: format!("Failed to create chart background: {}", e),
                })?;

            let min_value = self
                .data
                .iter()
                .map(|p| p.value)
                .fold(f32::INFINITY, f32::min);
            let max_value = self
                .data
                .iter()
                .map(|p| p.value)
                .fold(f32::NEG_INFINITY, f32::max);

            let min_time = self.data.iter().map(|p| p.timestamp).min().unwrap();
            let max_time = self.data.iter().map(|p| p.timestamp).max().unwrap();

            let mut chart = ChartBuilder::on(&root)
                .caption(&self.config.title, ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(min_time..max_time, min_value..max_value)
                .map_err(|e| FeedbackError::ConfigurationError {
                    message: format!("Failed to build chart: {}", e),
                })?;

            chart
                .configure_mesh()
                .draw()
                .map_err(|e| FeedbackError::ConfigurationError {
                    message: format!("Failed to draw chart mesh: {}", e),
                })?;

            chart
                .draw_series(LineSeries::new(
                    self.data.iter().map(|p| (p.timestamp, p.value)),
                    &BLUE,
                ))
                .map_err(|e| FeedbackError::ConfigurationError {
                    message: format!("Failed to draw chart series: {}", e),
                })?;

            root.present()
                .map_err(|e| FeedbackError::ConfigurationError {
                    message: format!("Failed to finalize chart: {}", e),
                })?;
        }

        Ok(svg_buffer)
    }

    /// Set chart configuration
    pub fn set_config(&mut self, config: ChartConfig) {
        self.config = config;
    }
}

#[cfg(feature = "ui")]
impl Default for ProgressChart {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Enhanced Radar Chart Implementation
// ============================================================================

/// Enhanced radar chart for multi-dimensional skill visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct EnhancedRadarChart {
    /// Skills and their values
    pub skills: Vec<RadarSkill>,
    /// Chart configuration
    pub config: RadarChartConfig,
    /// Animation state
    pub animation_state: RadarAnimationState,
    /// Comparison data (for before/after visualization)
    pub comparison_data: Option<Vec<RadarSkill>>,
}

#[cfg(feature = "ui")]
impl EnhancedRadarChart {
    /// Create a new enhanced radar chart
    pub fn new(skills: Vec<RadarSkill>) -> Self {
        Self {
            skills,
            config: RadarChartConfig::default(),
            animation_state: RadarAnimationState::default(),
            comparison_data: None,
        }
    }

    /// Set comparison data for before/after visualization
    pub fn with_comparison(mut self, comparison_data: Vec<RadarSkill>) -> Self {
        self.comparison_data = Some(comparison_data);
        self
    }

    /// Render the enhanced radar chart
    pub fn render(&mut self, ui: &mut Ui) {
        let chart_size = self.config.chart_size;
        let (rect, _response) = ui.allocate_exact_size(chart_size, egui::Sense::hover());

        let painter = ui.painter();
        let center = rect.center();
        let radius = (chart_size.x.min(chart_size.y) / 2.0) * 0.8;

        // Draw background circles
        self.draw_background_circles(painter, center, radius);

        // Draw axes
        self.draw_axes(painter, center, radius);

        // Draw data polygons
        self.draw_data_polygon(
            painter,
            center,
            radius,
            &self.skills,
            self.config.primary_color,
        );

        // Draw comparison data if available
        if let Some(comparison_data) = &self.comparison_data {
            self.draw_data_polygon(
                painter,
                center,
                radius,
                comparison_data,
                self.config.comparison_color,
            );
        }

        // Draw labels
        self.draw_labels(painter, center, radius);

        // Draw legend
        self.draw_legend(ui, rect);
    }

    /// Draw background concentric circles
    fn draw_background_circles(&self, painter: &egui::Painter, center: Pos2, radius: f32) {
        let levels = 5;
        for i in 1..=levels {
            let level_radius = radius * (i as f32 / levels as f32);
            painter.circle_stroke(
                center,
                level_radius,
                Stroke::new(1.0, Color32::from_gray(60)),
            );

            // Add percentage labels
            let percentage = (i as f32 / levels as f32) * 100.0;
            painter.text(
                Pos2::new(center.x + level_radius + 5.0, center.y),
                egui::Align2::LEFT_CENTER,
                format!("{:.0}%", percentage),
                egui::FontId::proportional(10.0),
                Color32::from_gray(120),
            );
        }
    }

    /// Draw axes from center to edge
    fn draw_axes(&self, painter: &egui::Painter, center: Pos2, radius: f32) {
        let num_skills = self.skills.len();
        for i in 0..num_skills {
            let angle = (i as f32 / num_skills as f32) * 2.0 * std::f32::consts::PI
                - std::f32::consts::PI / 2.0;
            let end_point = Pos2::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );

            painter.line_segment(
                [center, end_point],
                Stroke::new(1.0, Color32::from_gray(80)),
            );
        }
    }

    /// Draw data polygon
    fn draw_data_polygon(
        &self,
        painter: &egui::Painter,
        center: Pos2,
        radius: f32,
        skills: &[RadarSkill],
        color: Color32,
    ) {
        if skills.is_empty() {
            return;
        }

        let num_skills = skills.len();
        let mut points = Vec::new();

        for (i, skill) in skills.iter().enumerate() {
            let angle = (i as f32 / num_skills as f32) * 2.0 * std::f32::consts::PI
                - std::f32::consts::PI / 2.0;
            let skill_radius = radius * skill.value;
            let point = Pos2::new(
                center.x + skill_radius * angle.cos(),
                center.y + skill_radius * angle.sin(),
            );
            points.push(point);
        }

        // Draw filled polygon
        painter.add(egui::Shape::convex_polygon(
            points.clone(),
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 80),
            Stroke::new(2.0, color),
        ));

        // Draw points
        for (i, point) in points.iter().enumerate() {
            let skill = &skills[i];
            painter.circle_filled(*point, 4.0, color);

            // Add value tooltips
            if skill.show_value {
                painter.text(
                    Pos2::new(point.x, point.y - 15.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("{:.1}%", skill.value * 100.0),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }
        }
    }

    /// Draw skill labels
    fn draw_labels(&self, painter: &egui::Painter, center: Pos2, radius: f32) {
        let num_skills = self.skills.len();
        for (i, skill) in self.skills.iter().enumerate() {
            let angle = (i as f32 / num_skills as f32) * 2.0 * std::f32::consts::PI
                - std::f32::consts::PI / 2.0;
            let label_radius = radius + 20.0;
            let label_pos = Pos2::new(
                center.x + label_radius * angle.cos(),
                center.y + label_radius * angle.sin(),
            );

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                &skill.name,
                egui::FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
    }

    /// Draw legend
    fn draw_legend(&self, ui: &mut Ui, rect: Rect) {
        ui.allocate_ui_at_rect(
            Rect::from_min_size(
                Pos2::new(rect.max.x - 120.0, rect.min.y),
                Vec2::new(120.0, 60.0),
            ),
            |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.painter().circle_filled(
                            ui.next_widget_position() + Vec2::new(5.0, 5.0),
                            3.0,
                            self.config.primary_color,
                        );
                        ui.add_space(15.0);
                        ui.label("Current");
                    });

                    if self.comparison_data.is_some() {
                        ui.horizontal(|ui| {
                            ui.painter().circle_filled(
                                ui.next_widget_position() + Vec2::new(5.0, 5.0),
                                3.0,
                                self.config.comparison_color,
                            );
                            ui.add_space(15.0);
                            ui.label("Previous");
                        });
                    }
                });
            },
        );
    }
}

// ============================================================================
// Interactive Timeline Implementation
// ============================================================================

/// Interactive timeline for progress visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct InteractiveTimeline {
    /// Timeline configuration
    pub config: TimelineConfig,
    /// Timeline data points
    pub data: Vec<TimelineDataPoint>,
    /// Current zoom level
    pub zoom_level: f32,
    /// Current time range selection
    pub selected_range: TimelineRange,
}

#[cfg(feature = "ui")]
impl InteractiveTimeline {
    /// Create new interactive timeline
    pub fn new() -> Self {
        Self {
            config: TimelineConfig::default(),
            data: Vec::new(),
            zoom_level: 1.0,
            selected_range: TimelineRange::Week,
        }
    }

    /// Add data point to timeline
    pub fn add_data_point(&mut self, point: TimelineDataPoint) {
        self.data.push(point);
    }

    /// Set timeline range
    pub fn set_range(&mut self, range: TimelineRange) {
        self.selected_range = range;
    }
}

#[cfg(feature = "ui")]
impl Default for InteractiveTimeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rich Progress Visualization Implementation
// ============================================================================

/// Rich progress visualization
#[cfg(feature = "ui")]
#[derive(Debug, Clone)]
pub struct RichProgressVisualization {
    /// Configuration
    pub config: ProgressVisualizationConfig,
    /// Progress data
    pub progress_data: Vec<ProgressDataPoint>,
    /// Animation state
    pub animation_state: AnimationState,
}

#[cfg(feature = "ui")]
impl RichProgressVisualization {
    /// Create new rich progress visualization
    pub fn new() -> Self {
        Self {
            config: ProgressVisualizationConfig::default(),
            progress_data: Vec::new(),
            animation_state: AnimationState::default(),
        }
    }

    /// Add progress data point
    pub fn add_data_point(&mut self, point: ProgressDataPoint) {
        self.progress_data.push(point);
    }

    /// Update animation state
    pub fn update_animation(&mut self, delta_time: f32) {
        self.animation_state.update(delta_time);
    }
}

#[cfg(feature = "ui")]
impl Default for RichProgressVisualization {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stub Implementations for Non-UI Feature
// ============================================================================

/// Progress chart component (enhanced non-UI implementation)
#[cfg(not(feature = "ui"))]
pub struct ProgressChart {
    /// Chart configuration
    config: crate::visualization::config::ChartConfig,
    /// Chart data points
    data: Vec<crate::visualization::config::ChartDataPoint>,
    /// Statistical analysis cache
    stats_cache: Option<ProgressChartStats>,
}

/// Statistical analysis for progress charts
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct ProgressChartStats {
    /// Minimum value in dataset
    pub min_value: f32,
    /// Maximum value in dataset
    pub max_value: f32,
    /// Average value
    pub average_value: f32,
    /// Standard deviation
    pub standard_deviation: f32,
    /// Linear trend slope
    pub trend_slope: f32,
    /// Data point count
    pub data_count: usize,
}

#[cfg(not(feature = "ui"))]
impl Default for ProgressChart {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl ProgressChart {
    /// Create a new progress chart
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: crate::visualization::config::ChartConfig::default(),
            data: Vec::new(),
            stats_cache: None,
        }
    }

    /// Add a data point to the chart
    pub fn add_data_point(
        &mut self,
        timestamp: chrono::DateTime<chrono::Utc>,
        value: f32,
        label: Option<String>,
    ) {
        self.data
            .push(crate::visualization::config::ChartDataPoint {
                timestamp,
                value,
                label,
            });
        self.stats_cache = None; // Invalidate cache
    }

    /// Add multiple data points at once
    pub fn add_data_points(&mut self, points: Vec<crate::visualization::config::ChartDataPoint>) {
        self.data.extend(points);
        self.stats_cache = None; // Invalidate cache
    }

    /// Get statistical analysis of the data
    pub fn get_stats(&mut self) -> Option<&ProgressChartStats> {
        if self.data.is_empty() {
            return None;
        }

        if self.stats_cache.is_none() {
            self.stats_cache = Some(self.calculate_stats());
        }

        self.stats_cache.as_ref()
    }

    /// Calculate statistical analysis
    fn calculate_stats(&self) -> ProgressChartStats {
        if self.data.is_empty() {
            return ProgressChartStats {
                min_value: 0.0,
                max_value: 0.0,
                average_value: 0.0,
                standard_deviation: 0.0,
                trend_slope: 0.0,
                data_count: 0,
            };
        }

        let values: Vec<f32> = self.data.iter().map(|p| p.value).collect();
        let min_value = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let average_value = values.iter().sum::<f32>() / values.len() as f32;

        // Calculate standard deviation
        let variance = values
            .iter()
            .map(|v| (v - average_value).powi(2))
            .sum::<f32>()
            / values.len() as f32;
        let standard_deviation = variance.sqrt();

        // Calculate trend slope using linear regression
        let trend_slope = if self.data.len() > 1 {
            self.calculate_trend_slope()
        } else {
            0.0
        };

        ProgressChartStats {
            min_value,
            max_value,
            average_value,
            standard_deviation,
            trend_slope,
            data_count: self.data.len(),
        }
    }

    /// Calculate trend slope using simple linear regression
    fn calculate_trend_slope(&self) -> f32 {
        if self.data.len() < 2 {
            return 0.0;
        }

        let n = self.data.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, point) in self.data.iter().enumerate() {
            let x = i as f32;
            let y = point.value;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Export data as JSON
    pub fn export_json(&self) -> Result<String, crate::FeedbackError> {
        serde_json::to_string_pretty(&self.data).map_err(|e| {
            crate::FeedbackError::ProcessingError(format!("Failed to serialize chart data: {}", e))
        })
    }

    /// Export data as CSV
    pub fn export_csv(&self) -> Result<String, crate::FeedbackError> {
        let mut csv_content = String::from("timestamp,value,label\n");

        for point in &self.data {
            let label = point.label.as_deref().unwrap_or("");
            csv_content.push_str(&format!(
                "{},{},{}\n",
                point.timestamp.format("%Y-%m-%d %H:%M:%S"),
                point.value,
                label
            ));
        }

        Ok(csv_content)
    }

    /// Get data points in a time range
    pub fn get_data_in_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Vec<&crate::visualization::config::ChartDataPoint> {
        self.data
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
        self.stats_cache = None;
    }

    /// Get data count
    #[must_use]
    pub fn data_count(&self) -> usize {
        self.data.len()
    }
}

/// Enhanced radar chart (non-UI implementation with data analysis)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct EnhancedRadarChart {
    /// Chart configuration
    config: crate::visualization::config::RadarChartConfig,
    /// Skill data points
    skills: HashMap<String, f32>,
    /// Maximum value for normalization
    max_value: f32,
    /// Chart metadata
    metadata: RadarChartMetadata,
}

/// Metadata for radar chart analysis
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct RadarChartMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// Total skill count
    pub skill_count: usize,
    /// Average skill level
    pub average_skill_level: f32,
    /// Skill balance score (0.0 = very unbalanced, 1.0 = perfectly balanced)
    pub balance_score: f32,
}

#[cfg(not(feature = "ui"))]
impl Default for EnhancedRadarChart {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl EnhancedRadarChart {
    /// Create a new radar chart
    #[must_use]
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            config: crate::visualization::config::RadarChartConfig::default(),
            skills: HashMap::new(),
            max_value: 1.0,
            metadata: RadarChartMetadata {
                created_at: now,
                last_updated: now,
                skill_count: 0,
                average_skill_level: 0.0,
                balance_score: 1.0,
            },
        }
    }

    /// Add or update a skill value
    pub fn set_skill(&mut self, skill_name: String, value: f32) {
        let normalized_value = value.clamp(0.0, self.max_value);
        self.skills.insert(skill_name, normalized_value);
        self.update_metadata();
    }

    /// Add multiple skills at once
    pub fn set_skills(&mut self, skills: HashMap<String, f32>) {
        for (skill, value) in skills {
            let normalized_value = value.clamp(0.0, self.max_value);
            self.skills.insert(skill, normalized_value);
        }
        self.update_metadata();
    }

    /// Get skill value
    #[must_use]
    pub fn get_skill(&self, skill_name: &str) -> Option<f32> {
        self.skills.get(skill_name).copied()
    }

    /// Get all skills
    #[must_use]
    pub fn get_all_skills(&self) -> &HashMap<String, f32> {
        &self.skills
    }

    /// Update metadata after skill changes
    fn update_metadata(&mut self) {
        self.metadata.last_updated = chrono::Utc::now();
        self.metadata.skill_count = self.skills.len();

        if self.skills.is_empty() {
            self.metadata.average_skill_level = 0.0;
            self.metadata.balance_score = 1.0;
            return;
        }

        // Calculate average skill level
        let total: f32 = self.skills.values().sum();
        self.metadata.average_skill_level = total / self.skills.len() as f32;

        // Calculate balance score (inverse of coefficient of variation)
        let variance = self
            .skills
            .values()
            .map(|&v| (v - self.metadata.average_skill_level).powi(2))
            .sum::<f32>()
            / self.skills.len() as f32;

        let std_dev = variance.sqrt();

        // Balance score: higher when skills are more evenly distributed
        self.metadata.balance_score = if self.metadata.average_skill_level > 0.0 {
            1.0 - (std_dev / self.metadata.average_skill_level).min(1.0)
        } else {
            1.0
        };
    }

    /// Get chart metadata
    #[must_use]
    pub fn get_metadata(&self) -> &RadarChartMetadata {
        &self.metadata
    }

    /// Find strongest skills (top N)
    #[must_use]
    pub fn get_strongest_skills(&self, limit: usize) -> Vec<(String, f32)> {
        let mut skills: Vec<_> = self
            .skills
            .iter()
            .map(|(name, &value)| (name.clone(), value))
            .collect();

        skills.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        skills.truncate(limit);
        skills
    }

    /// Find weakest skills (bottom N)
    #[must_use]
    pub fn get_weakest_skills(&self, limit: usize) -> Vec<(String, f32)> {
        let mut skills: Vec<_> = self
            .skills
            .iter()
            .map(|(name, &value)| (name.clone(), value))
            .collect();

        skills.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        skills.truncate(limit);
        skills
    }

    /// Generate skill improvement recommendations
    #[must_use]
    pub fn get_improvement_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.skills.is_empty() {
            recommendations.push(String::from("Start practicing to develop skills"));
            return recommendations;
        }

        let weakest_skills = self.get_weakest_skills(3);
        let threshold = self.metadata.average_skill_level * 0.8; // Below 80% of average

        for (skill, value) in weakest_skills {
            if value < threshold {
                recommendations.push(format!(
                    "Focus on improving {} (current: {:.1}%, target: {:.1}%)",
                    skill,
                    value * 100.0,
                    self.metadata.average_skill_level * 100.0
                ));
            }
        }

        if self.metadata.balance_score < 0.7 {
            recommendations.push(String::from("Work on balancing skills more evenly"));
        }

        if recommendations.is_empty() {
            recommendations.push(String::from(
                "Maintain current skill levels and explore advanced techniques",
            ));
        }

        recommendations
    }

    /// Export radar chart data as JSON
    pub fn export_json(&self) -> Result<String, crate::FeedbackError> {
        let export_data = serde_json::json!({
            "skills": self.skills,
            "metadata": {
                "created_at": self.metadata.created_at,
                "last_updated": self.metadata.last_updated,
                "skill_count": self.metadata.skill_count,
                "average_skill_level": self.metadata.average_skill_level,
                "balance_score": self.metadata.balance_score
            },
            "recommendations": self.get_improvement_recommendations()
        });

        serde_json::to_string_pretty(&export_data).map_err(|e| {
            crate::FeedbackError::ProcessingError(format!(
                "Failed to serialize radar chart data: {}",
                e
            ))
        })
    }

    /// Clear all skills
    pub fn clear(&mut self) {
        self.skills.clear();
        self.update_metadata();
    }

    /// Set maximum value for normalization
    pub fn set_max_value(&mut self, max_value: f32) {
        self.max_value = max_value.max(0.1); // Ensure positive max value

        // Re-normalize existing skills
        for value in self.skills.values_mut() {
            *value = value.clamp(0.0, self.max_value);
        }

        self.update_metadata();
    }
}

/// Interactive timeline (non-UI implementation with event analysis)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct InteractiveTimeline {
    /// Timeline configuration
    config: crate::visualization::config::TimelineConfig,
    /// Timeline events
    events: Vec<TimelineEvent>,
    /// Timeline range
    time_range: Option<TimelineRange>,
    /// Event categories
    categories: HashMap<String, TimelineCategory>,
}

/// Timeline event structure
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineEvent {
    /// Event ID
    pub id: String,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Event title
    pub title: String,
    /// Event description
    pub description: Option<String>,
    /// Event category
    pub category: String,
    /// Event value/score
    pub value: Option<f32>,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Timeline range for filtering
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineRange {
    /// Start time
    pub start: chrono::DateTime<chrono::Utc>,
    /// End time
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Timeline category definition
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineCategory {
    /// Category name
    pub name: String,
    /// Category color (as hex string)
    pub color: String,
    /// Category description
    pub description: Option<String>,
    /// Event count in this category
    pub event_count: usize,
}

#[cfg(not(feature = "ui"))]
impl Default for InteractiveTimeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl InteractiveTimeline {
    /// Create a new interactive timeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: crate::visualization::config::TimelineConfig::default(),
            events: Vec::new(),
            time_range: None,
            categories: HashMap::new(),
        }
    }

    /// Add an event to the timeline
    pub fn add_event(&mut self, event: TimelineEvent) {
        // Update category count
        if let Some(category) = self.categories.get_mut(&event.category) {
            category.event_count += 1;
        } else {
            // Create new category if it doesn't exist
            self.categories.insert(
                event.category.clone(),
                TimelineCategory {
                    name: event.category.clone(),
                    color: self.generate_category_color(&event.category),
                    description: None,
                    event_count: 1,
                },
            );
        }

        self.events.push(event);

        // Sort events by timestamp
        self.events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Update time range
        self.update_time_range();
    }

    /// Add multiple events at once
    pub fn add_events(&mut self, events: Vec<TimelineEvent>) {
        for event in events {
            self.add_event(event);
        }
    }

    /// Generate a color for a category (simple hash-based approach)
    fn generate_category_color(&self, category: &str) -> String {
        // Simple hash-based color generation
        let hash = category
            .chars()
            .map(|c| c as u32)
            .fold(0u32, |acc, x| acc.wrapping_mul(31).wrapping_add(x));

        let r = ((hash >> 16) & 0xFF) as u8;
        let g = ((hash >> 8) & 0xFF) as u8;
        let b = (hash & 0xFF) as u8;

        format!("#{:02x}{:02x}{:02x}", r, g, b)
    }

    /// Update the time range based on events
    fn update_time_range(&mut self) {
        if self.events.is_empty() {
            self.time_range = None;
            return;
        }

        let min_time = self.events.iter().map(|e| e.timestamp).min().unwrap();
        let max_time = self.events.iter().map(|e| e.timestamp).max().unwrap();

        self.time_range = Some(TimelineRange {
            start: min_time,
            end: max_time,
        });
    }

    /// Get events in a specific time range
    #[must_use]
    pub fn get_events_in_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Vec<&TimelineEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Get events by category
    #[must_use]
    pub fn get_events_by_category(&self, category: &str) -> Vec<&TimelineEvent> {
        self.events
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Get all categories
    #[must_use]
    pub fn get_categories(&self) -> &HashMap<String, TimelineCategory> {
        &self.categories
    }

    /// Get timeline statistics
    #[must_use]
    pub fn get_statistics(&self) -> TimelineStatistics {
        let total_events = self.events.len();
        let category_count = self.categories.len();

        let events_with_values: Vec<f32> = self.events.iter().filter_map(|e| e.value).collect();

        let average_value = if events_with_values.is_empty() {
            None
        } else {
            Some(events_with_values.iter().sum::<f32>() / events_with_values.len() as f32)
        };

        let time_span = self.time_range.as_ref().map(|range| {
            range
                .end
                .signed_duration_since(range.start)
                .to_std()
                .unwrap_or_default()
        });

        // Calculate event frequency (events per day)
        let event_frequency = if let Some(span) = time_span {
            let days = span.as_secs() as f32 / (24.0 * 3600.0);
            if days > 0.0 {
                Some(total_events as f32 / days)
            } else {
                None
            }
        } else {
            None
        };

        TimelineStatistics {
            total_events,
            category_count,
            average_value,
            time_span,
            event_frequency,
            most_active_category: self.get_most_active_category(),
        }
    }

    /// Get the most active category
    fn get_most_active_category(&self) -> Option<String> {
        self.categories
            .iter()
            .max_by_key(|(_, category)| category.event_count)
            .map(|(name, _)| name.clone())
    }

    /// Export timeline data as JSON
    pub fn export_json(&self) -> Result<String, crate::FeedbackError> {
        let export_data = serde_json::json!({
            "events": self.events,
            "categories": self.categories,
            "time_range": self.time_range,
            "statistics": self.get_statistics()
        });

        serde_json::to_string_pretty(&export_data).map_err(|e| {
            crate::FeedbackError::ProcessingError(format!(
                "Failed to serialize timeline data: {}",
                e
            ))
        })
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.categories.clear();
        self.time_range = None;
    }

    /// Get event count
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get events within the last N days
    #[must_use]
    pub fn get_recent_events(&self, days: u32) -> Vec<&TimelineEvent> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days.into());
        self.events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .collect()
    }
}

/// Timeline statistics
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineStatistics {
    /// Total number of events
    pub total_events: usize,
    /// Number of categories
    pub category_count: usize,
    /// Average event value (if events have values)
    pub average_value: Option<f32>,
    /// Time span covered by timeline
    pub time_span: Option<std::time::Duration>,
    /// Event frequency (events per day)
    pub event_frequency: Option<f32>,
    /// Most active category name
    pub most_active_category: Option<String>,
}

/// Rich progress visualization (non-UI implementation with comprehensive analysis)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone)]
pub struct RichProgressVisualization {
    /// Visualization configuration
    config: crate::visualization::config::ProgressVisualizationConfig,
    /// Progress data points over time
    progress_data: Vec<ProgressDataPoint>,
    /// Milestones and achievements
    milestones: Vec<ProgressMilestone>,
    /// Goals and targets
    goals: Vec<ProgressGoal>,
    /// Analysis cache
    analysis_cache: Option<ProgressAnalysis>,
}

/// Progress data point with rich metadata
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressDataPoint {
    /// Timestamp of the measurement
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Overall progress score (0.0 to 1.0)
    pub overall_score: f32,
    /// Individual skill scores
    pub skill_scores: HashMap<String, f32>,
    /// Session details
    pub session_details: Option<SessionDetails>,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Session details for progress tracking
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionDetails {
    /// Session ID
    pub session_id: String,
    /// Session duration
    pub duration: std::time::Duration,
    /// Number of exercises completed
    pub exercises_completed: usize,
    /// Average difficulty level
    pub average_difficulty: f32,
}

/// Progress milestone
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressMilestone {
    /// Milestone ID
    pub id: String,
    /// Milestone name
    pub name: String,
    /// Achievement timestamp
    pub achieved_at: chrono::DateTime<chrono::Utc>,
    /// Milestone description
    pub description: String,
    /// Milestone value/score
    pub value: f32,
    /// Milestone category
    pub category: String,
}

/// Progress goal
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressGoal {
    /// Goal ID
    pub id: String,
    /// Goal name
    pub name: String,
    /// Target value
    pub target_value: f32,
    /// Current value
    pub current_value: f32,
    /// Target date
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Goal status
    pub status: ProgressGoalStatus,
    /// Goal priority
    pub priority: ProgressGoalPriority,
}

/// Progress goal status
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub enum ProgressGoalStatus {
    /// Goal is active and in progress
    Active,
    /// Goal has been achieved
    Achieved,
    /// Goal is paused
    Paused,
    /// Goal has been cancelled
    Cancelled,
}

/// Progress goal priority
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub enum ProgressGoalPriority {
    /// High priority goal
    High,
    /// Medium priority goal
    Medium,
    /// Low priority goal
    Low,
}

/// Comprehensive progress analysis
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressAnalysis {
    /// Overall progress trend
    pub overall_trend: ProgressTrend,
    /// Skill-specific trends
    pub skill_trends: HashMap<String, ProgressTrend>,
    /// Learning velocity (progress per unit time)
    pub learning_velocity: f32,
    /// Consistency score
    pub consistency_score: f32,
    /// Achievement rate
    pub achievement_rate: f32,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub analyzed_at: chrono::DateTime<chrono::Utc>,
}

/// Progress trend analysis
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f32,
    /// Recent change rate
    pub change_rate: f32,
    /// Confidence in trend analysis
    pub confidence: f32,
}

/// Trend direction
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, serde::Serialize)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Declining trend
    Declining,
    /// Insufficient data
    Insufficient,
}

#[cfg(not(feature = "ui"))]
impl Default for RichProgressVisualization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl RichProgressVisualization {
    /// Create a new rich progress visualization
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: crate::visualization::config::ProgressVisualizationConfig::default(),
            progress_data: Vec::new(),
            milestones: Vec::new(),
            goals: Vec::new(),
            analysis_cache: None,
        }
    }

    /// Add a progress data point
    pub fn add_progress_data(&mut self, data_point: ProgressDataPoint) {
        self.progress_data.push(data_point);

        // Sort by timestamp
        self.progress_data
            .sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Invalidate analysis cache
        self.analysis_cache = None;
    }

    /// Add a milestone
    pub fn add_milestone(&mut self, milestone: ProgressMilestone) {
        self.milestones.push(milestone);

        // Sort by achievement time
        self.milestones
            .sort_by(|a, b| a.achieved_at.cmp(&b.achieved_at));
    }

    /// Add a goal
    pub fn add_goal(&mut self, goal: ProgressGoal) {
        self.goals.push(goal);
    }

    /// Update goal progress
    pub fn update_goal_progress(
        &mut self,
        goal_id: &str,
        current_value: f32,
    ) -> Result<(), crate::FeedbackError> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or_else(|| crate::FeedbackError::InvalidInput {
                message: format!("Goal with ID '{}' not found", goal_id),
            })?;

        goal.current_value = current_value;

        // Update status if goal is achieved
        if goal.current_value >= goal.target_value
            && matches!(goal.status, ProgressGoalStatus::Active)
        {
            goal.status = ProgressGoalStatus::Achieved;
        }

        self.analysis_cache = None; // Invalidate cache
        Ok(())
    }

    /// Get comprehensive progress analysis
    pub fn get_analysis(&mut self) -> Option<&ProgressAnalysis> {
        if self.progress_data.is_empty() {
            return None;
        }

        if self.analysis_cache.is_none() {
            self.analysis_cache = Some(self.calculate_analysis());
        }

        self.analysis_cache.as_ref()
    }

    /// Calculate comprehensive progress analysis
    fn calculate_analysis(&self) -> ProgressAnalysis {
        let overall_trend = self.calculate_overall_trend();
        let skill_trends = self.calculate_skill_trends();
        let learning_velocity = self.calculate_learning_velocity();
        let consistency_score = self.calculate_consistency_score();
        let achievement_rate = self.calculate_achievement_rate();
        let recommendations = self.generate_recommendations(&overall_trend, &skill_trends);

        ProgressAnalysis {
            overall_trend,
            skill_trends,
            learning_velocity,
            consistency_score,
            achievement_rate,
            recommendations,
            analyzed_at: chrono::Utc::now(),
        }
    }

    /// Calculate overall progress trend
    fn calculate_overall_trend(&self) -> ProgressTrend {
        if self.progress_data.len() < 2 {
            return ProgressTrend {
                direction: TrendDirection::Insufficient,
                strength: 0.0,
                change_rate: 0.0,
                confidence: 0.0,
            };
        }

        let scores: Vec<f32> = self.progress_data.iter().map(|p| p.overall_score).collect();
        let slope = self.calculate_linear_regression_slope(&scores);

        let direction = if slope > 0.01 {
            TrendDirection::Improving
        } else if slope < -0.01 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        let strength = slope.abs().min(1.0);
        let confidence = (self.progress_data.len() as f32 / 10.0).min(1.0); // More data = higher confidence

        ProgressTrend {
            direction,
            strength,
            change_rate: slope,
            confidence,
        }
    }

    /// Calculate skill-specific trends
    fn calculate_skill_trends(&self) -> HashMap<String, ProgressTrend> {
        let mut skill_trends = HashMap::new();

        // Collect all skill names
        let mut skill_names = std::collections::HashSet::new();
        for data_point in &self.progress_data {
            skill_names.extend(data_point.skill_scores.keys().cloned());
        }

        for skill_name in skill_names {
            let skill_scores: Vec<f32> = self
                .progress_data
                .iter()
                .filter_map(|p| p.skill_scores.get(&skill_name).copied())
                .collect();

            if skill_scores.len() >= 2 {
                let slope = self.calculate_linear_regression_slope(&skill_scores);

                let direction = if slope > 0.01 {
                    TrendDirection::Improving
                } else if slope < -0.01 {
                    TrendDirection::Declining
                } else {
                    TrendDirection::Stable
                };

                skill_trends.insert(
                    skill_name,
                    ProgressTrend {
                        direction,
                        strength: slope.abs().min(1.0),
                        change_rate: slope,
                        confidence: (skill_scores.len() as f32 / 10.0).min(1.0),
                    },
                );
            }
        }

        skill_trends
    }

    /// Calculate linear regression slope for trend analysis
    fn calculate_linear_regression_slope(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f32;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Calculate learning velocity
    fn calculate_learning_velocity(&self) -> f32 {
        if self.progress_data.len() < 2 {
            return 0.0;
        }

        let first = &self.progress_data[0];
        let last = &self.progress_data[self.progress_data.len() - 1];

        let score_improvement = last.overall_score - first.overall_score;
        let time_span = last.timestamp.signed_duration_since(first.timestamp);

        let hours = time_span.num_hours() as f32;
        if hours > 0.0 {
            score_improvement / hours
        } else {
            0.0
        }
    }

    /// Calculate consistency score
    fn calculate_consistency_score(&self) -> f32 {
        if self.progress_data.len() < 3 {
            return 1.0; // Assume perfect consistency with insufficient data
        }

        let scores: Vec<f32> = self.progress_data.iter().map(|p| p.overall_score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;

        let variance =
            scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;

        let std_dev = variance.sqrt();

        // Consistency score: higher when variance is lower
        1.0 - (std_dev / mean).min(1.0)
    }

    /// Calculate achievement rate
    fn calculate_achievement_rate(&self) -> f32 {
        if self.goals.is_empty() {
            return 1.0; // Perfect rate if no goals
        }

        let achieved_goals = self
            .goals
            .iter()
            .filter(|g| matches!(g.status, ProgressGoalStatus::Achieved))
            .count();

        achieved_goals as f32 / self.goals.len() as f32
    }

    /// Generate improvement recommendations
    fn generate_recommendations(
        &self,
        overall_trend: &ProgressTrend,
        skill_trends: &HashMap<String, ProgressTrend>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Overall trend recommendations
        match overall_trend.direction {
            TrendDirection::Declining => {
                recommendations.push(String::from("Consider reviewing your learning strategy and identifying areas for improvement"));
            }
            TrendDirection::Stable => {
                recommendations.push(
                    "Try varying your practice routine to break through the current plateau"
                        .to_string(),
                );
            }
            TrendDirection::Improving => {
                recommendations.push(String::from(
                    "Great progress! Continue with your current approach",
                ));
            }
            TrendDirection::Insufficient => {
                recommendations.push(
                    "Complete more practice sessions to generate meaningful progress analysis"
                        .to_string(),
                );
            }
        }

        // Skill-specific recommendations
        let declining_skills: Vec<_> = skill_trends
            .iter()
            .filter(|(_, trend)| matches!(trend.direction, TrendDirection::Declining))
            .collect();

        if !declining_skills.is_empty() {
            for (skill, _) in declining_skills.iter().take(3) {
                recommendations.push(format!(
                    "Focus additional practice time on improving {}",
                    skill
                ));
            }
        }

        // Goal-based recommendations
        let overdue_goals: Vec<_> = self
            .goals
            .iter()
            .filter(|g| {
                matches!(g.status, ProgressGoalStatus::Active)
                    && g.target_date
                        .map_or(false, |date| date < chrono::Utc::now())
            })
            .collect();

        if !overdue_goals.is_empty() {
            recommendations.push(String::from(
                "Review and update overdue goals to maintain motivation",
            ));
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Continue your current practice routine and consider setting new challenges"
                    .to_string(),
            );
        }

        recommendations
    }

    /// Export comprehensive progress data as JSON
    pub fn export_json(&self) -> Result<String, crate::FeedbackError> {
        let export_data = serde_json::json!({
            "progress_data": self.progress_data,
            "milestones": self.milestones,
            "goals": self.goals,
            "analysis": self.analysis_cache
        });

        serde_json::to_string_pretty(&export_data).map_err(|e| {
            crate::FeedbackError::ProcessingError(format!(
                "Failed to serialize progress visualization data: {}",
                e
            ))
        })
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.progress_data.clear();
        self.milestones.clear();
        self.goals.clear();
        self.analysis_cache = None;
    }

    /// Get active goals
    #[must_use]
    pub fn get_active_goals(&self) -> Vec<&ProgressGoal> {
        self.goals
            .iter()
            .filter(|g| matches!(g.status, ProgressGoalStatus::Active))
            .collect()
    }

    /// Get recent milestones (within last N days)
    #[must_use]
    pub fn get_recent_milestones(&self, days: u32) -> Vec<&ProgressMilestone> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days.into());
        self.milestones
            .iter()
            .filter(|m| m.achieved_at >= cutoff)
            .collect()
    }
}
