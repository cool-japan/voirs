//! Core FeedbackVisualizer implementation

#[cfg(feature = "ui")]
use crate::traits::*;
#[cfg(feature = "ui")]
use crate::visualization::config::{CachedChart, VisualizationConfig, VisualizationTheme};
#[cfg(feature = "ui")]
use crate::visualization::realtime::RealtimeDashboard;
#[cfg(feature = "ui")]
use crate::visualization::types::{
    AlignmentData, FrameLimiter, PerformanceOptions, ProblemArea, RealtimeDashboardData,
    TrainingSessionResult,
};
#[cfg(feature = "ui")]
use crate::FeedbackError;
#[cfg(feature = "ui")]
use egui::{Color32, Context, Pos2, Rect, Stroke, Ui, Vec2};
use std::collections::HashMap;
#[cfg(feature = "ui")]
use std::time::Duration;

// ============================================================================
// Main FeedbackVisualizer Implementation
// ============================================================================

/// Feedback visualizer for creating visual representations of feedback data
#[cfg(feature = "ui")]
#[derive(Clone)]
pub struct FeedbackVisualizer {
    /// Visualization configuration
    config: VisualizationConfig,
    /// Theme settings
    theme: VisualizationTheme,
    /// Chart cache for performance
    chart_cache: std::sync::Arc<std::sync::RwLock<HashMap<String, CachedChart>>>,
    /// Performance optimization settings
    performance_opts: PerformanceOptions,
    /// Frame rate limiter to prevent excessive redraws
    frame_limiter: std::sync::Arc<std::sync::Mutex<FrameLimiter>>,
}

#[cfg(feature = "ui")]
impl FeedbackVisualizer {
    /// Create a new feedback visualizer
    pub fn new() -> Self {
        Self::with_config(VisualizationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self {
            config,
            theme: VisualizationTheme::default(),
            chart_cache: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            performance_opts: PerformanceOptions::default(),
            frame_limiter: std::sync::Arc::new(std::sync::Mutex::new(FrameLimiter::new(60.0))),
        }
    }

    /// Render feedback panel in egui
    pub fn render_feedback_panel(&self, ui: &mut Ui, feedback: &FeedbackResponse) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                // Header
                ui.heading("📊 Feedback Summary");
                ui.separator();

                // Overall score
                self.render_score_gauge(ui, "Overall Score", feedback.overall_score);

                ui.separator();

                // Individual feedback items
                ui.heading("💡 Feedback Items");
                for (i, item) in feedback.feedback_items.iter().enumerate() {
                    self.render_feedback_item(ui, i, item);
                }

                ui.separator();

                // Progress indicators
                if !feedback.progress_indicators.improving_areas.is_empty() {
                    ui.heading("📈 Improving Areas");
                    for area in &feedback.progress_indicators.improving_areas {
                        ui.label(format!("✅ {}", area));
                    }
                }

                if !feedback.progress_indicators.attention_areas.is_empty() {
                    ui.heading("⚠️ Needs Attention");
                    for area in &feedback.progress_indicators.attention_areas {
                        ui.label(format!("🔍 {}", area));
                    }
                }

                // Immediate actions
                if !feedback.immediate_actions.is_empty() {
                    ui.separator();
                    ui.heading("🎯 Next Steps");
                    for (i, action) in feedback.immediate_actions.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}.", i + 1));
                            ui.label(action);
                        });
                    }
                }
            });
        });
    }

    /// Render progress chart
    pub fn render_progress_chart(&self, ui: &mut Ui, progress_data: &[ProgressSnapshot]) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading("📊 Progress Over Time");

                if progress_data.is_empty() {
                    ui.label("No progress data available");
                    return;
                }

                let chart_size = Vec2::new(400.0, 200.0);
                let (rect, _response) = ui.allocate_exact_size(chart_size, egui::Sense::hover());

                self.draw_progress_chart(ui, rect, progress_data);
            });
        });
    }

    /// Render achievement showcase
    pub fn render_achievement_showcase(&self, ui: &mut Ui, achievements: &[Achievement]) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading("🏆 Recent Achievements");

                if achievements.is_empty() {
                    ui.label("No achievements yet - keep practicing!");
                    return;
                }

                for achievement in achievements.iter().take(5) {
                    self.render_achievement_card(ui, achievement);
                }
            });
        });
    }

    /// Render skill radar chart
    pub fn render_skill_radar(&self, ui: &mut Ui, skill_breakdown: &HashMap<FocusArea, f32>) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading("🎯 Skill Assessment");

                let chart_size = Vec2::new(300.0, 300.0);
                let (rect, _response) = ui.allocate_exact_size(chart_size, egui::Sense::hover());

                self.draw_radar_chart(ui, rect, skill_breakdown);
            });
        });
    }

    /// Render training dashboard
    pub fn render_training_dashboard(&self, ui: &mut Ui, session_data: &TrainingSessionResult) {
        ui.vertical(|ui| {
            ui.heading("🎮 Training Dashboard");

            // Session summary
            ui.horizontal(|ui| {
                self.render_metric_card(ui, "Exercises", &session_data.total_exercises.to_string());
                self.render_metric_card(
                    ui,
                    "Success Rate",
                    &format!("{:.1}%", session_data.success_rate * 100.0),
                );
                self.render_metric_card(
                    ui,
                    "Duration",
                    &self.format_duration(session_data.session_duration),
                );
            });

            ui.separator();

            // Score breakdown
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("📊 Scores");
                    self.render_score_breakdown(ui, &session_data.average_scores);
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.heading("🎯 Next Steps");
                    for recommendation in &session_data.recommendations {
                        ui.label(format!("• {}", recommendation));
                    }
                });
            });
        });
    }

    /// Render real-time dashboard
    pub fn render_realtime_dashboard(&self, ui: &mut Ui, dashboard_data: &RealtimeDashboardData) {
        RealtimeDashboard::render(ui, dashboard_data);
    }

    /// Render real-time waveform visualization
    pub fn render_waveform(&self, ui: &mut Ui, audio_data: &[f32], title: &str) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading(format!("🎵 {}", title));

                if audio_data.is_empty() {
                    ui.label("No audio data available");
                    return;
                }

                let waveform_size = Vec2::new(600.0, 150.0);
                let (rect, _response) = ui.allocate_exact_size(waveform_size, egui::Sense::hover());

                self.draw_waveform(ui, rect, audio_data);
            });
        });
    }

    /// Render audio highlighting system with problem area marking
    pub fn render_audio_highlighting(
        &self,
        ui: &mut Ui,
        audio_data: &[f32],
        problem_areas: &[ProblemArea],
        title: &str,
    ) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading(format!("🎧 {}", title));

                if audio_data.is_empty() {
                    ui.label("No audio data available");
                    return;
                }

                let waveform_size = Vec2::new(700.0, 200.0);
                let (rect, response) =
                    ui.allocate_exact_size(waveform_size, egui::Sense::click_and_drag());

                self.draw_highlighted_waveform(ui, rect, audio_data, problem_areas);

                // Interactive scrubbing
                if response.clicked() {
                    let click_pos = response.interact_pointer_pos().unwrap_or_default();
                    let relative_x = (click_pos.x - rect.min.x) / rect.width();
                    let timestamp = relative_x * (audio_data.len() as f32 / 44100.0);

                    // Emit audio playback event (implementation would connect to audio system)
                    ui.ctx().request_repaint();
                }

                // Problem area legend
                if !problem_areas.is_empty() {
                    ui.separator();
                    self.render_problem_area_legend(ui, problem_areas);
                }
            });
        });
    }

    /// Render temporal alignment visualization
    pub fn render_temporal_alignment(
        &self,
        ui: &mut Ui,
        reference_audio: &[f32],
        user_audio: &[f32],
        alignment_data: &AlignmentData,
    ) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading("🔄 Temporal Alignment");

                // Reference audio waveform
                ui.label("Reference Audio:");
                let ref_size = Vec2::new(600.0, 80.0);
                let (ref_rect, _) = ui.allocate_exact_size(ref_size, egui::Sense::hover());
                self.draw_waveform(ui, ref_rect, reference_audio);

                ui.add_space(5.0);

                // User audio waveform with alignment markers
                ui.label("Your Audio:");
                let user_size = Vec2::new(600.0, 80.0);
                let (user_rect, _) = ui.allocate_exact_size(user_size, egui::Sense::hover());
                self.draw_aligned_waveform(ui, user_rect, user_audio, alignment_data);

                ui.separator();

                // Alignment statistics
                self.render_alignment_stats(ui, alignment_data);
            });
        });
    }

    // ========================================================================
    // Helper Methods - Core Rendering
    // ========================================================================

    /// Render score gauge
    fn render_score_gauge(&self, ui: &mut Ui, label: &str, score: f32) {
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.label(label);

                let size = Vec2::new(80.0, 80.0);
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                let painter = ui.painter();
                let center = rect.center();
                let radius = size.x / 2.0 - 10.0;

                // Background circle
                painter.circle_stroke(center, radius, Stroke::new(4.0, Color32::from_gray(80)));

                // Score arc
                let score_angle = score * 2.0 * std::f32::consts::PI;
                let start_angle = -std::f32::consts::PI / 2.0;

                let color = self.score_to_color(score);
                self.draw_arc(painter, center, radius, start_angle, score_angle, color);

                // Score text
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}%", score * 100.0),
                    egui::FontId::proportional(14.0),
                    Color32::WHITE,
                );
            });
        });
    }

    /// Render feedback item
    fn render_feedback_item(&self, ui: &mut Ui, index: usize, item: &UserFeedback) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{}.", index + 1));
                ui.vertical(|ui| {
                    ui.label(&item.message);
                    if let Some(suggestion) = &item.suggestion {
                        ui.small(format!("💡 {}", suggestion));
                    }
                });
            });
        });
    }

    /// Render achievement card
    fn render_achievement_card(&self, ui: &mut Ui, achievement: &Achievement) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("🏆");
                ui.vertical(|ui| {
                    ui.strong(&achievement.name);
                    ui.label(&achievement.description);
                    ui.small(format!(
                        "Earned: {}",
                        achievement.unlocked_at.format("%m/%d/%Y")
                    ));
                });
            });
        });
    }

    /// Render metric card
    fn render_metric_card(&self, ui: &mut Ui, label: &str, value: &str) {
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.heading(value);
                ui.label(label);
            });
        });
    }

    /// Format duration for display
    fn format_duration(&self, duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{}:{:02}", minutes, seconds)
    }

    // ========================================================================
    // Drawing Helper Methods - Placeholders for Extracted Functionality
    // ========================================================================

    /// Draw progress chart
    fn draw_progress_chart(&self, ui: &mut Ui, rect: Rect, progress_data: &[ProgressSnapshot]) {
        // Implementation extracted from original file
        let painter = ui.painter();
        painter.rect_filled(rect, 5.0, Color32::from_gray(40));

        if progress_data.is_empty() {
            return;
        }

        // Simple line chart implementation
        let mut points = Vec::new();
        for (i, snapshot) in progress_data.iter().enumerate() {
            let x = rect.min.x + (i as f32 / (progress_data.len() - 1) as f32) * rect.width();
            let y = rect.max.y - (snapshot.overall_score * rect.height());
            points.push(Pos2::new(x, y));
        }

        // Draw line
        for window in points.windows(2) {
            painter.line_segment(
                [window[0], window[1]],
                Stroke::new(2.0, Color32::from_rgb(100, 150, 255)),
            );
        }

        // Draw points
        for point in points {
            painter.circle_filled(point, 3.0, Color32::from_rgb(100, 150, 255));
        }
    }

    /// Draw radar chart
    fn draw_radar_chart(&self, ui: &mut Ui, rect: Rect, skill_breakdown: &HashMap<FocusArea, f32>) {
        // Placeholder implementation - full implementation would be extracted from original
        let painter = ui.painter();
        let center = rect.center();
        let radius = rect.width().min(rect.height()) / 2.0 - 20.0;

        // Draw background circle
        painter.circle_stroke(center, radius, Stroke::new(1.0, Color32::from_gray(80)));

        // Draw skill areas (simplified)
        let skills: Vec<_> = skill_breakdown.iter().collect();
        for (i, (_area, &value)) in skills.iter().enumerate() {
            let angle = (i as f32 / skills.len() as f32) * 2.0 * std::f32::consts::PI;
            let end_point = Pos2::new(
                center.x + radius * value * angle.cos(),
                center.y + radius * value * angle.sin(),
            );

            painter.line_segment(
                [center, end_point],
                Stroke::new(2.0, Color32::from_rgb(100, 150, 255)),
            );
            painter.circle_filled(end_point, 3.0, Color32::from_rgb(100, 150, 255));
        }
    }

    /// Draw waveform
    fn draw_waveform(&self, ui: &mut Ui, rect: Rect, audio_data: &[f32]) {
        let painter = ui.painter();
        painter.rect_filled(rect, 2.0, Color32::from_gray(30));

        if audio_data.is_empty() {
            return;
        }

        let samples_per_pixel = audio_data.len() / rect.width() as usize;
        let center_y = rect.center().y;
        let amplitude_scale = rect.height() / 2.0;

        for x in 0..rect.width() as usize {
            let sample_start = x * samples_per_pixel;
            let sample_end = (sample_start + samples_per_pixel).min(audio_data.len());

            if sample_start < audio_data.len() && sample_end > sample_start {
                let max_amplitude = audio_data[sample_start..sample_end]
                    .iter()
                    .map(|s| s.abs())
                    .fold(0.0, f32::max);

                let y_offset = max_amplitude * amplitude_scale;
                let x_pos = rect.min.x + x as f32;

                painter.line_segment(
                    [
                        Pos2::new(x_pos, center_y - y_offset),
                        Pos2::new(x_pos, center_y + y_offset),
                    ],
                    Stroke::new(1.0, Color32::from_rgb(100, 200, 255)),
                );
            }
        }
    }

    /// Draw highlighted waveform with problem areas
    fn draw_highlighted_waveform(
        &self,
        ui: &mut Ui,
        rect: Rect,
        audio_data: &[f32],
        problem_areas: &[ProblemArea],
    ) {
        // Draw base waveform
        self.draw_waveform(ui, rect, audio_data);

        // Highlight problem areas
        let painter = ui.painter();
        let total_duration = audio_data.len() as f32 / 44100.0; // Assuming 44.1kHz

        for problem in problem_areas {
            let start_x = rect.min.x + (problem.start_time / total_duration) * rect.width();
            let end_x = rect.min.x + (problem.end_time / total_duration) * rect.width();

            let highlight_rect = Rect::from_x_y_ranges(start_x..=end_x, rect.min.y..=rect.max.y);

            let color = match problem.severity {
                crate::visualization::types::ProblemSeverity::Critical => {
                    Color32::from_rgba_unmultiplied(255, 0, 0, 80)
                }
                crate::visualization::types::ProblemSeverity::High => {
                    Color32::from_rgba_unmultiplied(255, 100, 0, 80)
                }
                crate::visualization::types::ProblemSeverity::Medium => {
                    Color32::from_rgba_unmultiplied(255, 200, 0, 80)
                }
                _ => Color32::from_rgba_unmultiplied(100, 100, 100, 40),
            };

            painter.rect_filled(highlight_rect, 0.0, color);
        }
    }

    /// Draw aligned waveform
    fn draw_aligned_waveform(
        &self,
        ui: &mut Ui,
        rect: Rect,
        audio_data: &[f32],
        alignment_data: &AlignmentData,
    ) {
        // Draw base waveform
        self.draw_waveform(ui, rect, audio_data);

        // Add alignment markers
        let painter = ui.painter();
        for marker in &alignment_data.alignment_markers {
            let x_pos = rect.min.x
                + (marker.timestamp / (audio_data.len() as f32 / 44100.0)) * rect.width();
            let color = match marker.alignment_quality {
                crate::visualization::types::AlignmentQuality::Good => {
                    Color32::from_rgb(100, 255, 100)
                }
                crate::visualization::types::AlignmentQuality::Fair => {
                    Color32::from_rgb(255, 200, 100)
                }
                crate::visualization::types::AlignmentQuality::Poor => {
                    Color32::from_rgb(255, 100, 100)
                }
            };

            painter.line_segment(
                [Pos2::new(x_pos, rect.min.y), Pos2::new(x_pos, rect.max.y)],
                Stroke::new(2.0, color),
            );
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Convert score to color
    fn score_to_color(&self, score: f32) -> Color32 {
        if score > 0.8 {
            Color32::from_rgb(100, 255, 100)
        } else if score > 0.6 {
            Color32::from_rgb(255, 200, 100)
        } else {
            Color32::from_rgb(255, 100, 100)
        }
    }

    /// Draw arc for circular progress meters
    fn draw_arc(
        &self,
        painter: &egui::Painter,
        center: Pos2,
        radius: f32,
        start_angle: f32,
        arc_angle: f32,
        color: Color32,
    ) {
        let segments = 32;
        let angle_step = arc_angle / segments as f32;

        for i in 0..segments {
            let angle1 = start_angle + i as f32 * angle_step;
            let angle2 = start_angle + (i + 1) as f32 * angle_step;

            let point1 = Pos2::new(
                center.x + radius * angle1.cos(),
                center.y + radius * angle1.sin(),
            );
            let point2 = Pos2::new(
                center.x + radius * angle2.cos(),
                center.y + radius * angle2.sin(),
            );

            painter.line_segment([point1, point2], Stroke::new(3.0, color));
        }
    }

    // Placeholder methods for additional functionality
    fn render_score_breakdown(&self, ui: &mut Ui, scores: &HashMap<String, f32>) {
        // Implementation would render score breakdown
        for (category, &score) in scores {
            ui.horizontal(|ui| {
                ui.label(category);
                ui.label(format!("{:.1}%", score * 100.0));
            });
        }
    }

    fn render_problem_area_legend(&self, ui: &mut Ui, problem_areas: &[ProblemArea]) {
        // Implementation would render legend for problem areas
        ui.horizontal(|ui| {
            ui.label("Legend:");
            ui.colored_label(Color32::from_rgb(255, 100, 100), "Critical");
            ui.colored_label(Color32::from_rgb(255, 200, 100), "High");
            ui.colored_label(Color32::from_rgb(255, 255, 100), "Medium");
        });
    }

    fn render_alignment_stats(&self, ui: &mut Ui, alignment_data: &AlignmentData) {
        // Implementation would render alignment statistics
        ui.horizontal(|ui| {
            ui.label(format!(
                "Overall Score: {:.1}%",
                alignment_data.overall_score * 100.0
            ));
            ui.label(format!(
                "Timing Accuracy: {:.1}%",
                alignment_data.timing_accuracy * 100.0
            ));
            ui.label(format!(
                "Avg Offset: {:.1}ms",
                alignment_data.average_offset_ms
            ));
        });
    }
}

#[cfg(feature = "ui")]
impl Default for FeedbackVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stub Implementation for Non-UI Feature
// ============================================================================

/// Feedback visualization component (stub implementation)
#[cfg(not(feature = "ui"))]
#[derive(Clone)]
pub struct FeedbackVisualizer {
    /// Visualization configuration
    config: crate::visualization::config::VisualizationConfig,
    /// Theme settings
    theme: crate::visualization::config::VisualizationTheme,
    /// Chart cache for performance
    chart_cache: std::sync::Arc<
        std::sync::RwLock<HashMap<String, crate::visualization::config::CachedChart>>,
    >,
}

#[cfg(not(feature = "ui"))]
impl Default for FeedbackVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl FeedbackVisualizer {
    /// Create a new feedback visualizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: crate::visualization::config::VisualizationConfig::default(),
            theme: crate::visualization::config::VisualizationTheme::default(),
            chart_cache: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }
}
