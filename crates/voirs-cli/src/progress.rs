//! Progress indication utilities.

use indicatif::{ProgressBar, ProgressFinish, ProgressState, ProgressStyle};
use std::fmt::Write;
use std::time::Duration;

/// Progress tracker for synthesis operations
pub struct SynthesisProgress {
    bar: ProgressBar,
}

impl SynthesisProgress {
    /// Create a new synthesis progress bar
    pub fn new(total_items: u64) -> Self {
        let bar = ProgressBar::new(total_items);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {pos}/{len} {msg}",
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
        );
        bar.set_message("Synthesizing...");

        Self { bar }
    }

    /// Update progress with message
    pub fn update(&self, pos: u64, msg: &str) {
        self.bar.set_position(pos);
        self.bar.set_message(msg.to_string());
    }

    /// Increment progress by 1
    pub fn inc(&self, msg: &str) {
        self.bar.inc(1);
        self.bar.set_message(msg.to_string());
    }

    /// Finish progress bar
    pub fn finish(&self, msg: &str) {
        self.bar.finish_with_message(msg.to_string());
    }
}

/// Progress tracker for download operations
pub struct DownloadProgress {
    bar: ProgressBar,
}

impl DownloadProgress {
    /// Create a new download progress bar
    pub fn new(total_bytes: u64, filename: &str) -> Self {
        let bar = ProgressBar::new(total_bytes);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta}) {msg}"
            )
            .unwrap()
            .progress_chars("#>-")
        );
        bar.set_message(format!("Downloading {}", filename));

        Self { bar }
    }

    /// Update download progress
    pub fn update(&self, downloaded_bytes: u64) {
        self.bar.set_position(downloaded_bytes);
    }

    /// Finish download
    pub fn finish(&self, msg: &str) {
        self.bar.finish_with_message(msg.to_string());
    }
}

/// Progress tracker for batch operations
pub struct BatchProgress {
    bar: ProgressBar,
    start_time: std::time::Instant,
}

impl BatchProgress {
    /// Create a new batch progress bar
    pub fn new(total_items: u64, operation: &str) -> Self {
        let bar = ProgressBar::new(total_items);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {pos}/{len} ({per_sec}) {msg}"
            )
            .unwrap()
            .progress_chars("#>-")
        );
        bar.set_message(format!("Processing {} items...", operation));

        Self {
            bar,
            start_time: std::time::Instant::now(),
        }
    }

    /// Update batch progress
    pub fn update(&self, pos: u64, current_item: &str) {
        self.bar.set_position(pos);
        self.bar
            .set_message(format!("Processing: {}", current_item));
    }

    /// Increment batch progress
    pub fn inc(&self, current_item: &str) {
        self.bar.inc(1);
        self.bar
            .set_message(format!("Processing: {}", current_item));
    }

    /// Finish batch processing
    pub fn finish(&self, msg: &str) {
        let elapsed = self.start_time.elapsed();
        self.bar.finish_with_message(format!(
            "{} (completed in {:.2}s)",
            msg,
            elapsed.as_secs_f64()
        ));
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Simple spinner for indeterminate operations
pub struct Spinner {
    bar: ProgressBar,
}

impl Spinner {
    /// Create a new spinner
    pub fn new(msg: &str) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.enable_steady_tick(Duration::from_millis(120));
        bar.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap()
                .tick_strings(&[
                    "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂", "▁",
                ]),
        );
        bar.set_message(msg.to_string());

        Self { bar }
    }

    /// Update spinner message
    pub fn update(&self, msg: &str) {
        self.bar.set_message(msg.to_string());
    }

    /// Finish spinner
    pub fn finish(&self, msg: &str) {
        self.bar.finish_with_message(msg.to_string());
    }
}

/// Create a simple progress bar for file operations
pub fn create_file_progress(total_files: usize, operation: &str) -> ProgressBar {
    let bar = ProgressBar::new(total_files as u64);
    bar.set_style(
        ProgressStyle::with_template(&format!(
            "{{spinner:.green}} {} [{{bar:.cyan/blue}}] {{pos}}/{{len}} {{msg}}",
            operation
        ))
        .unwrap()
        .progress_chars("#>-"),
    );
    bar
}

/// Create a progress bar with custom style
pub fn create_custom_progress(total: u64, template: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    if let Ok(style) = ProgressStyle::with_template(template) {
        bar.set_style(style.progress_chars("#>-"));
    }
    bar
}
