//! Real-time Terminal Dashboard for VoiRS Synthesis Monitoring
//!
//! This module provides a real-time TUI (Terminal User Interface) dashboard
//! for monitoring synthesis operations, resource usage, and system metrics.
//!
//! # Features
//!
//! - Live synthesis queue monitoring
//! - Resource usage graphs (CPU, Memory)
//! - Throughput metrics and statistics
//! - Error rate tracking
//! - Recent synthesis history
//! - Active operations display
//!
//! # Usage
//!
//! ```bash
//! # Start the dashboard
//! voirs dashboard
//!
//! # Dashboard with custom update interval
//! voirs dashboard --interval 1000
//!
//! # Quit: Press 'q' or Ctrl+C
//! ```

use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{BarChart, Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Tabs, Wrap},
    Frame, Terminal,
};
use std::{
    io,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::time;

/// Dashboard state and metrics
#[derive(Debug, Clone)]
pub struct DashboardState {
    /// Total number of synthesis operations
    pub total_syntheses: u64,
    /// Successful synthesis count
    pub successful_syntheses: u64,
    /// Failed synthesis count
    pub failed_syntheses: u64,
    /// Current throughput (operations/second)
    pub throughput: f64,
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// Active operations count
    pub active_operations: usize,
    /// Recent synthesis history (text, duration_ms, success)
    pub recent_history: Vec<(String, u64, bool)>,
    /// CPU usage history for sparkline (last 50 samples)
    pub cpu_history: Vec<u64>,
    /// Memory usage history for sparkline (last 50 samples)
    pub memory_history: Vec<u64>,
    /// Error messages log
    pub error_log: Vec<String>,
    /// Current time
    pub current_time: Instant,
    /// Dashboard start time
    pub start_time: Instant,
}

impl Default for DashboardState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            total_syntheses: 0,
            successful_syntheses: 0,
            failed_syntheses: 0,
            throughput: 0.0,
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            active_operations: 0,
            recent_history: Vec::new(),
            cpu_history: vec![0; 50],
            memory_history: vec![0; 50],
            error_log: Vec::new(),
            current_time: now,
            start_time: now,
        }
    }
}

impl DashboardState {
    /// Update metrics with simulated or real data
    pub fn update_metrics(&mut self) {
        self.current_time = Instant::now();

        // Simulate some metrics for demonstration
        // In production, these would come from actual synthesis operations
        self.cpu_usage = (fastrand::f32() * 30.0 + 10.0).min(100.0);
        self.memory_usage_mb = (fastrand::f32() * 200.0 + 100.0).min(1000.0);

        // Update history buffers
        self.cpu_history.rotate_left(1);
        self.cpu_history[49] = self.cpu_usage as u64;

        self.memory_history.rotate_left(1);
        self.memory_history[49] = self.memory_usage_mb as u64;

        // Calculate throughput
        let elapsed = self
            .current_time
            .duration_since(self.start_time)
            .as_secs_f64();
        if elapsed > 0.0 {
            self.throughput = self.total_syntheses as f64 / elapsed;
        }
    }

    /// Add a synthesis result to history
    pub fn add_synthesis_result(&mut self, text: String, duration_ms: u64, success: bool) {
        self.total_syntheses += 1;
        if success {
            self.successful_syntheses += 1;
        } else {
            self.failed_syntheses += 1;
            self.error_log.push(format!(
                "[{}] Synthesis failed: {}",
                chrono::Local::now().format("%H:%M:%S"),
                text
            ));
            // Keep only last 10 errors
            if self.error_log.len() > 10 {
                self.error_log.remove(0);
            }
        }

        self.recent_history.push((text, duration_ms, success));
        // Keep only last 10 items
        if self.recent_history.len() > 10 {
            self.recent_history.remove(0);
        }
    }

    /// Calculate success rate percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_syntheses == 0 {
            return 0.0;
        }
        (self.successful_syntheses as f64 / self.total_syntheses as f64) * 100.0
    }

    /// Calculate error rate percentage
    pub fn error_rate(&self) -> f64 {
        if self.total_syntheses == 0 {
            return 0.0;
        }
        (self.failed_syntheses as f64 / self.total_syntheses as f64) * 100.0
    }

    /// Get uptime string
    pub fn uptime_string(&self) -> String {
        let uptime = self.current_time.duration_since(self.start_time);
        let secs = uptime.as_secs();
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    }
}

/// Dashboard application
pub struct DashboardApp {
    /// Shared dashboard state
    state: Arc<Mutex<DashboardState>>,
    /// Current selected tab
    current_tab: usize,
    /// Tab titles
    tab_titles: Vec<&'static str>,
    /// Should quit flag
    should_quit: bool,
}

impl Default for DashboardApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardApp {
    /// Create a new dashboard application
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(DashboardState::default())),
            current_tab: 0,
            tab_titles: vec!["Overview", "Resources", "History", "Errors"],
            should_quit: false,
        }
    }

    /// Run the dashboard
    pub async fn run(&mut self, update_interval_ms: u64) -> Result<()> {
        // Setup terminal
        enable_raw_mode().context("Failed to enable raw mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
            .context("Failed to setup terminal")?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("Failed to create terminal instance")?;

        // Run the app
        let result = self.run_app(&mut terminal, update_interval_ms).await;

        // Restore terminal
        disable_raw_mode().context("Failed to disable raw mode")?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .context("Failed to restore terminal")?;
        terminal.show_cursor().context("Failed to show cursor")?;

        result
    }

    /// Main application loop
    async fn run_app<B: Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
        update_interval_ms: u64,
    ) -> Result<()> {
        let update_interval = Duration::from_millis(update_interval_ms);
        let mut last_update = Instant::now();

        loop {
            terminal.draw(|f| self.ui(f))?;

            // Check for events (non-blocking)
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => {
                            self.should_quit = true;
                        }
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.should_quit = true;
                        }
                        KeyCode::Right => {
                            self.current_tab = (self.current_tab + 1) % self.tab_titles.len();
                        }
                        KeyCode::Left => {
                            if self.current_tab > 0 {
                                self.current_tab -= 1;
                            } else {
                                self.current_tab = self.tab_titles.len() - 1;
                            }
                        }
                        _ => {}
                    }
                }
            }

            if self.should_quit {
                break;
            }

            // Update metrics periodically
            if last_update.elapsed() >= update_interval {
                let mut state = self.state.lock().unwrap();
                state.update_metrics();
                last_update = Instant::now();
            }

            time::sleep(Duration::from_millis(50)).await;
        }

        Ok(())
    }

    /// Render the UI
    fn ui(&self, f: &mut Frame) {
        let state = self.state.lock().unwrap();

        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Tabs
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer
            ])
            .split(f.area());

        // Render tabs
        self.render_tabs(f, chunks[0]);

        // Render content based on selected tab
        match self.current_tab {
            0 => self.render_overview(f, chunks[1], &state),
            1 => self.render_resources(f, chunks[1], &state),
            2 => self.render_history(f, chunks[1], &state),
            3 => self.render_errors(f, chunks[1], &state),
            _ => {}
        }

        // Render footer
        self.render_footer(f, chunks[2], &state);
    }

    /// Render tab bar
    fn render_tabs(&self, f: &mut Frame, area: Rect) {
        let titles: Vec<Line> = self
            .tab_titles
            .iter()
            .map(|t| Line::from(Span::styled(*t, Style::default())))
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("VoiRS Dashboard"),
            )
            .select(self.current_tab)
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Yellow),
            );

        f.render_widget(tabs, area);
    }

    /// Render overview tab
    fn render_overview(&self, f: &mut Frame, area: Rect, state: &DashboardState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7), // Statistics
                Constraint::Length(7), // Performance metrics
                Constraint::Min(0),    // Recent activity
            ])
            .split(area);

        // Statistics block
        let stats_text = vec![
            Line::from(vec![
                Span::styled("Total Syntheses: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", state.total_syntheses),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Successful: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", state.successful_syntheses),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Failed: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", state.failed_syntheses),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Success Rate: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{:.1}%", state.success_rate()),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ];

        let stats_widget = Paragraph::new(stats_text)
            .block(Block::default().borders(Borders::ALL).title("Statistics"));

        f.render_widget(stats_widget, chunks[0]);

        // Performance metrics
        let perf_text = vec![
            Line::from(vec![
                Span::styled("Throughput: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{:.2} ops/sec", state.throughput),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("CPU Usage: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{:.1}%", state.cpu_usage),
                    Style::default()
                        .fg(if state.cpu_usage > 80.0 {
                            Color::Red
                        } else if state.cpu_usage > 50.0 {
                            Color::Yellow
                        } else {
                            Color::Green
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Memory: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{:.1} MB", state.memory_usage_mb),
                    Style::default()
                        .fg(if state.memory_usage_mb > 800.0 {
                            Color::Red
                        } else if state.memory_usage_mb > 500.0 {
                            Color::Yellow
                        } else {
                            Color::Green
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Active Operations: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", state.active_operations),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ];

        let perf_widget = Paragraph::new(perf_text)
            .block(Block::default().borders(Borders::ALL).title("Performance"));

        f.render_widget(perf_widget, chunks[1]);

        // Recent activity
        let history_items: Vec<ListItem> = state
            .recent_history
            .iter()
            .rev()
            .map(|(text, duration, success)| {
                let status = if *success { "✓" } else { "✗" };
                let color = if *success { Color::Green } else { Color::Red };
                let truncated_text = if text.len() > 40 {
                    format!("{}...", &text[..37])
                } else {
                    text.clone()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        status,
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(truncated_text, Style::default().fg(Color::White)),
                    Span::raw(" "),
                    Span::styled(
                        format!("({} ms)", duration),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let history_list = List::new(history_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Recent Activity"),
        );

        f.render_widget(history_list, chunks[2]);
    }

    /// Render resources tab
    fn render_resources(&self, f: &mut Frame, area: Rect, state: &DashboardState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // CPU gauge
                Constraint::Length(8), // CPU history
                Constraint::Length(3), // Memory gauge
                Constraint::Min(0),    // Memory history
            ])
            .split(area);

        // CPU gauge
        let cpu_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
            .gauge_style(
                Style::default()
                    .fg(if state.cpu_usage > 80.0 {
                        Color::Red
                    } else if state.cpu_usage > 50.0 {
                        Color::Yellow
                    } else {
                        Color::Green
                    })
                    .bg(Color::Black),
            )
            .percent(state.cpu_usage as u16);

        f.render_widget(cpu_gauge, chunks[0]);

        // CPU history sparkline
        let cpu_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("CPU History"))
            .data(&state.cpu_history)
            .style(Style::default().fg(Color::Cyan));

        f.render_widget(cpu_sparkline, chunks[1]);

        // Memory gauge
        let memory_percentage = ((state.memory_usage_mb / 1000.0) * 100.0).min(100.0) as u16;
        let memory_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
            .gauge_style(
                Style::default()
                    .fg(if state.memory_usage_mb > 800.0 {
                        Color::Red
                    } else if state.memory_usage_mb > 500.0 {
                        Color::Yellow
                    } else {
                        Color::Green
                    })
                    .bg(Color::Black),
            )
            .percent(memory_percentage);

        f.render_widget(memory_gauge, chunks[2]);

        // Memory history sparkline
        let memory_sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Memory History"),
            )
            .data(&state.memory_history)
            .style(Style::default().fg(Color::Magenta));

        f.render_widget(memory_sparkline, chunks[3]);
    }

    /// Render history tab
    fn render_history(&self, f: &mut Frame, area: Rect, state: &DashboardState) {
        let history_items: Vec<ListItem> = state
            .recent_history
            .iter()
            .rev()
            .enumerate()
            .map(|(i, (text, duration, success))| {
                let status = if *success { "SUCCESS" } else { "FAILED " };
                let color = if *success { Color::Green } else { Color::Red };
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("#{} ", state.recent_history.len() - i),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            status,
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(" - "),
                        Span::styled(
                            format!("{} ms", duration),
                            Style::default().fg(Color::Yellow),
                        ),
                    ]),
                    Line::from(vec![Span::styled(
                        format!("  {}", text),
                        Style::default().fg(Color::White),
                    )]),
                ])
            })
            .collect();

        let history_list = List::new(history_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Synthesis History"),
        );

        f.render_widget(history_list, area);
    }

    /// Render errors tab
    fn render_errors(&self, f: &mut Frame, area: Rect, state: &DashboardState) {
        let error_items: Vec<ListItem> = state
            .error_log
            .iter()
            .rev()
            .map(|error| {
                ListItem::new(Line::from(vec![
                    Span::styled(
                        "✗ ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(error, Style::default().fg(Color::White)),
                ]))
            })
            .collect();

        let error_list = if error_items.is_empty() {
            let no_errors = vec![ListItem::new(Line::from(vec![Span::styled(
                "No errors recorded",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::ITALIC),
            )]))];
            List::new(no_errors).block(Block::default().borders(Borders::ALL).title("Error Log"))
        } else {
            List::new(error_items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!("Error Log ({} errors)", state.error_log.len())),
            )
        };

        f.render_widget(error_list, area);
    }

    /// Render footer
    fn render_footer(&self, f: &mut Frame, area: Rect, state: &DashboardState) {
        let footer_text = vec![Line::from(vec![
            Span::styled("Uptime: ", Style::default().fg(Color::White)),
            Span::styled(
                state.uptime_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                "[q]",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Quit | "),
            Span::styled(
                "[←/→]",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Navigate Tabs"),
        ])];

        let footer = Paragraph::new(footer_text)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);

        f.render_widget(footer, area);
    }

    /// Get reference to dashboard state for external updates
    pub fn state(&self) -> Arc<Mutex<DashboardState>> {
        Arc::clone(&self.state)
    }
}

/// Run the dashboard command
pub async fn run_dashboard(update_interval_ms: u64) -> Result<()> {
    let mut app = DashboardApp::new();

    // Spawn a background task to simulate synthesis operations
    let state = app.state();
    tokio::spawn(async move {
        let mut counter = 0;
        loop {
            time::sleep(Duration::from_secs(3)).await;
            counter += 1;

            let mut state = state.lock().unwrap();
            let success = fastrand::f32() > 0.1; // 90% success rate
            state.add_synthesis_result(
                format!("Synthesis operation #{}", counter),
                fastrand::u64(100..500),
                success,
            );
            state.active_operations = fastrand::usize(0..5);
        }
    });

    app.run(update_interval_ms).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_creation() {
        let state = DashboardState::default();
        assert_eq!(state.total_syntheses, 0);
        assert_eq!(state.successful_syntheses, 0);
        assert_eq!(state.failed_syntheses, 0);
        assert_eq!(state.throughput, 0.0);
    }

    #[test]
    fn test_add_synthesis_result() {
        let mut state = DashboardState::default();
        state.add_synthesis_result("Test text".to_string(), 250, true);

        assert_eq!(state.total_syntheses, 1);
        assert_eq!(state.successful_syntheses, 1);
        assert_eq!(state.failed_syntheses, 0);
        assert_eq!(state.recent_history.len(), 1);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut state = DashboardState::default();
        state.add_synthesis_result("Text 1".to_string(), 250, true);
        state.add_synthesis_result("Text 2".to_string(), 300, true);
        state.add_synthesis_result("Text 3".to_string(), 200, false);

        assert!((state.success_rate() - 200.0 / 3.0).abs() < 0.0001);
        assert!((state.error_rate() - 100.0 / 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_history_limit() {
        let mut state = DashboardState::default();
        for i in 0..15 {
            state.add_synthesis_result(format!("Text {}", i), 250, true);
        }

        assert_eq!(state.recent_history.len(), 10);
        assert_eq!(state.total_syntheses, 15);
    }

    #[test]
    fn test_error_log_limit() {
        let mut state = DashboardState::default();
        for i in 0..15 {
            state.add_synthesis_result(format!("Text {}", i), 250, false);
        }

        assert_eq!(state.error_log.len(), 10);
        assert_eq!(state.failed_syntheses, 15);
    }

    #[test]
    fn test_uptime_string() {
        let mut state = DashboardState::default();
        state.current_time = state.start_time + Duration::from_secs(3665); // 1h 1m 5s

        let uptime = state.uptime_string();
        assert_eq!(uptime, "01:01:05");
    }

    #[test]
    fn test_dashboard_app_creation() {
        let app = DashboardApp::new();
        assert_eq!(app.current_tab, 0);
        assert_eq!(app.tab_titles.len(), 4);
        assert!(!app.should_quit);
    }
}
