//! Real-time visualization components and dashboard functionality

#[cfg(feature = "ui")]
use crate::visualization::config::RealtimeConfig;
#[cfg(feature = "ui")]
use crate::visualization::types::{
    ActiveGoal, InstantFeedback, LayoutPreferences, RealtimeDashboardData,
};
#[cfg(feature = "ui")]
use egui::{Color32, Pos2, Rect, Stroke, StrokeKind, Ui, Vec2};
use std::collections::HashMap;

// ============================================================================
// Real-time Widget Implementation
// ============================================================================

/// Real-time visualization widget
#[cfg(feature = "ui")]
pub struct RealtimeWidget {
    /// Current values to display
    values: HashMap<String, f32>,
    /// Value history for sparklines
    history: HashMap<String, Vec<f32>>,
    /// Configuration
    config: RealtimeConfig,
}

#[cfg(feature = "ui")]
impl RealtimeWidget {
    /// Create a new real-time widget
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            history: HashMap::new(),
            config: RealtimeConfig::default(),
        }
    }

    /// Update a value
    pub fn update_value(&mut self, key: &str, value: f32) {
        self.values.insert(key.to_string(), value);

        let history = self.history.entry(key.to_string()).or_insert_with(Vec::new);
        history.push(value);

        // Limit history size
        if history.len() > self.config.max_history_points {
            history.remove(0);
        }
    }

    /// Render the widget
    pub fn render(&self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.heading("📊 Real-time Metrics");

                for (key, &value) in &self.values {
                    ui.horizontal(|ui| {
                        ui.label(key);
                        ui.add_space(10.0);

                        // Current value
                        ui.strong(format!("{:.2}", value));

                        // Sparkline
                        if let Some(history) = self.history.get(key) {
                            self.render_sparkline(ui, history);
                        }
                    });
                }
            });
        });
    }

    fn render_sparkline(&self, ui: &mut Ui, data: &[f32]) {
        if data.len() < 2 {
            return;
        }

        let size = Vec2::new(60.0, 20.0);
        let (rect, _response) = ui.allocate_exact_size(size, egui::Sense::hover());

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range > 0.0 {
            let points: Vec<Pos2> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let x = rect.min.x + (i as f32 / (data.len() - 1) as f32) * rect.width();
                    let y = rect.max.y - ((val - min_val) / range) * rect.height();
                    Pos2::new(x, y)
                })
                .collect();

            for window in points.windows(2) {
                ui.painter().line_segment(
                    [window[0], window[1]],
                    Stroke::new(1.0, Color32::from_rgb(100, 150, 255)),
                );
            }
        }
    }
}

#[cfg(feature = "ui")]
impl Default for RealtimeWidget {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Real-time Dashboard Implementation
// ============================================================================

/// Real-time dashboard renderer
#[cfg(feature = "ui")]
pub struct RealtimeDashboard;

#[cfg(feature = "ui")]
impl RealtimeDashboard {
    /// Render the complete real-time dashboard
    pub fn render(ui: &mut Ui, dashboard_data: &RealtimeDashboardData) {
        ui.vertical(|ui| {
            ui.heading("📊 Real-Time Dashboard");

            // Live performance meters
            ui.group(|ui| {
                ui.heading("⚡ Live Performance");
                ui.horizontal(|ui| {
                    Self::render_live_meter(ui, "CPU", dashboard_data.cpu_usage, "🖥️");
                    Self::render_live_meter(ui, "Memory", dashboard_data.memory_usage, "🧠");
                    Self::render_live_meter(ui, "Audio", dashboard_data.audio_latency, "🎵");
                });
            });

            ui.separator();

            // Instant feedback display
            ui.group(|ui| {
                ui.heading("💬 Instant Feedback");
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label("Current Score:");
                        Self::render_instant_score_display(ui, dashboard_data.current_score);
                    });
                    ui.separator();
                    ui.vertical(|ui| {
                        ui.label("Feedback Queue:");
                        Self::render_feedback_queue(ui, &dashboard_data.feedback_queue);
                    });
                });
            });

            ui.separator();

            // Progress indicators
            ui.group(|ui| {
                ui.heading("📈 Progress Indicators");
                ui.horizontal(|ui| {
                    Self::render_progress_indicator(
                        ui,
                        "Session Progress",
                        dashboard_data.session_progress,
                    );
                    Self::render_progress_indicator(
                        ui,
                        "Daily Goal",
                        dashboard_data.daily_goal_progress,
                    );
                    Self::render_progress_indicator(
                        ui,
                        "Weekly Goal",
                        dashboard_data.weekly_goal_progress,
                    );
                });
            });

            ui.separator();

            // Goal tracking widgets
            ui.group(|ui| {
                ui.heading("🎯 Goal Tracking");
                Self::render_goal_tracking_widgets(ui, &dashboard_data.active_goals);
            });

            ui.separator();

            // Customizable layout controls
            ui.group(|ui| {
                ui.heading("⚙️ Layout Options");
                Self::render_layout_controls(ui, &dashboard_data.layout_preferences);
            });
        });
    }

    /// Render live performance meter
    fn render_live_meter(ui: &mut Ui, label: &str, value: f32, icon: &str) {
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.label(format!("{} {}", icon, label));

                // Circular progress meter
                let meter_size = 60.0;
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(meter_size, meter_size), egui::Sense::hover());

                let painter = ui.painter();
                let center = rect.center();
                let radius = meter_size / 2.0 - 5.0;

                // Background circle
                painter.circle_stroke(center, radius, Stroke::new(3.0, Color32::from_gray(80)));

                // Progress arc
                let progress_angle = value * 2.0 * std::f32::consts::PI;
                let start_angle = -std::f32::consts::PI / 2.0;

                let color = if value > 0.8 {
                    Color32::from_rgb(255, 100, 100)
                } else if value > 0.6 {
                    Color32::from_rgb(255, 200, 100)
                } else {
                    Color32::from_rgb(100, 255, 100)
                };

                Self::draw_arc(painter, center, radius, start_angle, progress_angle, color);

                // Value text
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}%", value * 100.0),
                    egui::FontId::proportional(12.0),
                    Color32::WHITE,
                );
            });
        });
    }

    /// Render instant score display
    fn render_instant_score_display(ui: &mut Ui, score: f32) {
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                // Large score display
                ui.heading(format!("{:.1}", score * 100.0));
                ui.label("Current Score");

                // Score trend indicator
                let trend_icon = if score > 0.8 {
                    "📈"
                } else if score > 0.6 {
                    "➡️"
                } else {
                    "📉"
                };
                ui.colored_label(
                    Self::score_to_color(score),
                    format!("{} {:.1}%", trend_icon, score * 100.0),
                );
            });
        });
    }

    /// Render feedback queue
    fn render_feedback_queue(ui: &mut Ui, queue: &[InstantFeedback]) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                if queue.is_empty() {
                    ui.label("No pending feedback");
                } else {
                    for (i, feedback) in queue.iter().take(3).enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}.", i + 1));
                            ui.label(&feedback.message);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.colored_label(
                                        Self::priority_to_color(feedback.priority),
                                        format!("{:.0}ms", feedback.timestamp_ms),
                                    );
                                },
                            );
                        });
                    }
                    if queue.len() > 3 {
                        ui.label(format!("... and {} more", queue.len() - 3));
                    }
                }
            });
        });
    }

    /// Render progress indicator
    fn render_progress_indicator(ui: &mut Ui, label: &str, progress: f32) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(label);

                // Progress bar
                let bar_width = 120.0;
                let bar_height = 20.0;
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(bar_width, bar_height), egui::Sense::hover());

                let painter = ui.painter();

                // Background
                painter.rect_filled(rect, 5.0, Color32::from_gray(50));

                // Progress fill
                let fill_width = rect.width() * progress;
                let fill_rect = Rect::from_min_size(rect.min, Vec2::new(fill_width, rect.height()));
                painter.rect_filled(fill_rect, 5.0, Color32::from_rgb(100, 150, 255));

                // Border
                painter.rect_stroke(
                    rect,
                    5.0,
                    Stroke::new(1.0, Color32::WHITE),
                    StrokeKind::Middle,
                );

                // Progress text
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}%", progress * 100.0),
                    egui::FontId::proportional(11.0),
                    Color32::WHITE,
                );
            });
        });
    }

    /// Render goal tracking widgets
    fn render_goal_tracking_widgets(ui: &mut Ui, goals: &[ActiveGoal]) {
        ui.vertical(|ui| {
            if goals.is_empty() {
                ui.label("No active goals");
            } else {
                for goal in goals {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.strong(&goal.name);
                                ui.label(&goal.description);
                                ui.small(format!("Due: {}", goal.due_date.format("%m/%d/%Y")));
                            });

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.vertical(|ui| {
                                        ui.label(format!("{:.0}%", goal.progress * 100.0));
                                        let progress_bar_size = Vec2::new(60.0, 8.0);
                                        let (rect, _) = ui.allocate_exact_size(
                                            progress_bar_size,
                                            egui::Sense::hover(),
                                        );

                                        let painter = ui.painter();
                                        painter.rect_filled(rect, 2.0, Color32::from_gray(50));

                                        let fill_width = rect.width() * goal.progress;
                                        let fill_rect = Rect::from_min_size(
                                            rect.min,
                                            Vec2::new(fill_width, rect.height()),
                                        );
                                        painter.rect_filled(
                                            fill_rect,
                                            2.0,
                                            Color32::from_rgb(100, 255, 100),
                                        );
                                    });
                                },
                            );
                        });
                    });
                }
            }
        });
    }

    /// Render layout controls
    fn render_layout_controls(ui: &mut Ui, preferences: &LayoutPreferences) {
        ui.horizontal(|ui| {
            ui.label("Layout:");
            if ui.button("Grid").clicked() {
                // Switch to grid layout
            }
            if ui.button("List").clicked() {
                // Switch to list layout
            }
            if ui.button("Compact").clicked() {
                // Switch to compact layout
            }

            ui.separator();

            ui.label("Theme:");
            if ui.button("Dark").clicked() {
                // Switch to dark theme
            }
            if ui.button("Light").clicked() {
                // Switch to light theme
            }

            ui.separator();

            ui.checkbox(&mut preferences.show_animations.clone(), "Animations");
            ui.checkbox(
                &mut preferences.show_detailed_metrics.clone(),
                "Detailed Metrics",
            );
        });
    }

    /// Draw arc for circular progress meters
    fn draw_arc(
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

    /// Convert score to color
    fn score_to_color(score: f32) -> Color32 {
        if score > 0.8 {
            Color32::from_rgb(100, 255, 100)
        } else if score > 0.6 {
            Color32::from_rgb(255, 200, 100)
        } else {
            Color32::from_rgb(255, 100, 100)
        }
    }

    /// Convert priority to color
    fn priority_to_color(priority: f32) -> Color32 {
        if priority > 0.8 {
            Color32::from_rgb(255, 100, 100)
        } else if priority > 0.6 {
            Color32::from_rgb(255, 200, 100)
        } else {
            Color32::from_rgb(100, 255, 100)
        }
    }
}

// ============================================================================
// Stub Implementations for Non-UI Feature
// ============================================================================

/// Real-time widget component (stub implementation)
#[cfg(not(feature = "ui"))]
pub struct RealtimeWidget {
    /// Current values to display
    values: HashMap<String, f32>,
    /// Value history for sparklines
    history: HashMap<String, Vec<f32>>,
    /// Configuration
    config: crate::visualization::config::RealtimeConfig,
}

#[cfg(not(feature = "ui"))]
impl Default for RealtimeWidget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "ui"))]
impl RealtimeWidget {
    /// Create a new real-time widget
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            history: HashMap::new(),
            config: crate::visualization::config::RealtimeConfig::default(),
        }
    }
}

/// Real-time dashboard renderer (stub implementation)
#[cfg(not(feature = "ui"))]
#[derive(Debug, Clone, Default)]
pub struct RealtimeDashboard;
